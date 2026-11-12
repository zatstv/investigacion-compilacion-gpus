#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>
#include <SDL.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>

#define ANCHO 1280
#define ALTO 720
#define GRUPO 256

static int n = 32768;
static float dt = 0.004f;
static float zoom = 240.0f;

static cl_context ctx;
static cl_command_queue cola;
static cl_device_id gpu;
static cl_program prog = NULL;
static cl_kernel k_mover, k_dibujar, k_empacar;
static cl_mem d_pos, d_pos2, d_vel, d_luz, d_pix;

static float *pos, *vel, *luz;
static Uint32 *pixeles;

static const char *presets[][2] = {
    { "gravedad",
      "    return 1.0f / (d * d);",
    },
    { "remolino",
      "    return 0.5f / (d * d);",
    },
    { "repulsion",
      "    return -0.3f / (d * d);",
    },
    { "resorte",
      "    return 4.0f * d;",
    },
    { "ondas",
      "    return sin(d * 25.0f - t * 4.0f) / (d + 0.05f);",
    },
    { "latido",
      "    return (1.0f + 1.5f * cos(t * 2.0f)) / (d * d);",
    },
};

static const char *campos[] = {
    "    return (float2)(0.0f, 0.0f);",
    "    return (float2)(-p.y, p.x) * 1.5f - p * 0.6f;",
    "    return -p * 3.0f;",
    "    return -v * 0.2f;",
    "    return -p * 0.5f;",
    "    return (float2)(0.0f, 0.0f);",
};

static double ahora(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec * 1000.0 + t.tv_nsec / 1e6;
}

static char *leer_archivo(const char *nombre)
{
    FILE *f = fopen(nombre, "rb");
    if (!f)
        return NULL;
    fseek(f, 0, SEEK_END);
    long largo = ftell(f);
    rewind(f);
    char *texto = malloc(largo + 1);
    fread(texto, 1, largo, f);
    texto[largo] = '\0';
    fclose(f);
    return texto;
}

static void escribir_preset(int i)
{
    FILE *f = fopen("ley.cl", "w");
    fprintf(f, "// Ley: %s\n", presets[i][0]);
    fprintf(f, "// Guarda el archivo y la galaxia se recompila sola.\n");
    fprintf(f, "// d = distancia entre dos estrellas, t = tiempo\n");
    fprintf(f, "// p = posicion de la estrella, v = su velocidad\n\n");
    fprintf(f, "float fuerza(float d, float t)\n{\n%s\n}\n\n", presets[i][1]);
    fprintf(f, "float2 campo(float2 p, float2 v, float t)\n{\n%s\n}\n", campos[i]);
    fclose(f);
}

static int compilar_ley(double *ms)
{
    char *ley = leer_archivo("ley.cl");
    char *motor = leer_archivo("motor.cl");
    if (!ley || !motor) {
        printf("No se encontro ley.cl o motor.cl\n");
        free(ley);
        free(motor);
        return 0;
    }
    const char *fuentes[2] = { ley, motor };
    cl_int err;
    cl_program nuevo = clCreateProgramWithSource(ctx, 2, fuentes, NULL, &err);
    double t0 = ahora();
    err = clBuildProgram(nuevo, 1, &gpu, "-cl-std=CL1.2 -cl-fast-relaxed-math", NULL, NULL);
    *ms = ahora() - t0;
    free(ley);
    free(motor);

    if (err != CL_SUCCESS) {
        size_t largo;
        clGetProgramBuildInfo(nuevo, gpu, CL_PROGRAM_BUILD_LOG, 0, NULL, &largo);
        char *log = malloc(largo + 1);
        clGetProgramBuildInfo(nuevo, gpu, CL_PROGRAM_BUILD_LOG, largo, log, NULL);
        log[largo] = '\0';
        printf("\n[ERROR] El compilador del driver rechazo la ley (sigue la anterior):\n%s\n", log);
        free(log);
        clReleaseProgram(nuevo);
        return 0;
    }

    if (prog) {
        clReleaseKernel(k_mover);
        clReleaseKernel(k_dibujar);
        clReleaseKernel(k_empacar);
        clReleaseProgram(prog);
    }
    prog = nuevo;
    k_mover = clCreateKernel(prog, "mover", &err);
    k_dibujar = clCreateKernel(prog, "dibujar", &err);
    k_empacar = clCreateKernel(prog, "empacar", &err);
    printf("[OK] Ley compilada por el driver en %.1f ms\n", *ms);
    return 1;
}

static float azar(void)
{
    return (float)rand() / RAND_MAX;
}

static void crear_galaxias(void)
{
    srand(7);
    for (int i = 0; i < n; i++) {
        int g = i < n / 2 ? 0 : 1;
        float cx = g ? 0.9f : -0.9f, cy = g ? -0.35f : 0.35f;
        float vx = g ? -0.22f : 0.22f;
        float radio = 0.55f;
        float u = azar();
        float r = radio * u * u + 0.01f;
        float ang = azar() * 6.2831853f;
        float x = r * cosf(ang), y = r * sinf(ang);
        float masa = 0.5f * sqrtf(r / radio);
        float vc = sqrtf(masa * r / powf(r * r + 0.0004f, 0.75f)) * 0.9f;
        float giro = g ? -1.0f : 1.0f;
        pos[i * 2] = cx + x;
        pos[i * 2 + 1] = cy + y;
        vel[i * 2] = vx - giro * vc * y / r + (azar() - 0.5f) * 0.06f;
        vel[i * 2 + 1] = giro * vc * x / r + (azar() - 0.5f) * 0.06f;
    }
    memset(luz, 0, sizeof(float) * ANCHO * ALTO * 3);
}

static void subir_a_gpu(void)
{
    clEnqueueWriteBuffer(cola, d_pos, CL_TRUE, 0, sizeof(float) * 2 * n, pos, 0, NULL, NULL);
    clEnqueueWriteBuffer(cola, d_vel, CL_TRUE, 0, sizeof(float) * 2 * n, vel, 0, NULL, NULL);
    clEnqueueWriteBuffer(cola, d_luz, CL_TRUE, 0, sizeof(float) * ANCHO * ALTO * 3, luz, 0, NULL, NULL);
}

static void bajar_de_gpu(void)
{
    clEnqueueReadBuffer(cola, d_pos, CL_TRUE, 0, sizeof(float) * 2 * n, pos, 0, NULL, NULL);
    clEnqueueReadBuffer(cola, d_vel, CL_TRUE, 0, sizeof(float) * 2 * n, vel, 0, NULL, NULL);
    clEnqueueReadBuffer(cola, d_luz, CL_TRUE, 0, sizeof(float) * ANCHO * ALTO * 3, luz, 0, NULL, NULL);
}

static void paso_gpu(float t)
{
    size_t global = ((n + GRUPO - 1) / GRUPO) * GRUPO, local = GRUPO;
    clSetKernelArg(k_mover, 0, sizeof(cl_mem), &d_pos);
    clSetKernelArg(k_mover, 1, sizeof(cl_mem), &d_vel);
    clSetKernelArg(k_mover, 2, sizeof(cl_mem), &d_pos2);
    clSetKernelArg(k_mover, 3, sizeof(int), &n);
    clSetKernelArg(k_mover, 4, sizeof(float), &t);
    clSetKernelArg(k_mover, 5, sizeof(float), &dt);
    clSetKernelArg(k_mover, 6, sizeof(float) * 2 * GRUPO, NULL);
    clEnqueueNDRangeKernel(cola, k_mover, 1, NULL, &global, &local, 0, NULL, NULL);
    cl_mem tmp = d_pos;
    d_pos = d_pos2;
    d_pos2 = tmp;

    int ancho = ANCHO, alto = ALTO;
    clSetKernelArg(k_dibujar, 0, sizeof(cl_mem), &d_pos);
    clSetKernelArg(k_dibujar, 1, sizeof(cl_mem), &d_vel);
    clSetKernelArg(k_dibujar, 2, sizeof(cl_mem), &d_luz);
    clSetKernelArg(k_dibujar, 3, sizeof(int), &n);
    clSetKernelArg(k_dibujar, 4, sizeof(int), &ancho);
    clSetKernelArg(k_dibujar, 5, sizeof(int), &alto);
    clSetKernelArg(k_dibujar, 6, sizeof(float), &zoom);
    clEnqueueNDRangeKernel(cola, k_dibujar, 1, NULL, &global, NULL, 0, NULL, NULL);

    float apagado = 0.86f;
    size_t total = ANCHO * ALTO;
    clSetKernelArg(k_empacar, 0, sizeof(cl_mem), &d_luz);
    clSetKernelArg(k_empacar, 1, sizeof(cl_mem), &d_pix);
    clSetKernelArg(k_empacar, 2, sizeof(float), &apagado);
    clEnqueueNDRangeKernel(cola, k_empacar, 1, NULL, &total, NULL, 0, NULL, NULL);
    clEnqueueReadBuffer(cola, d_pix, CL_TRUE, 0, sizeof(Uint32) * total, pixeles, 0, NULL, NULL);
}

static void paso_cpu(void)
{
    float m = 1.0f / n;
    float *nueva = malloc(sizeof(float) * 2 * n);
    for (int i = 0; i < n; i++) {
        float px = pos[i * 2], py = pos[i * 2 + 1];
        float ax = 0.0f, ay = 0.0f;
        for (int j = 0; j < n; j++) {
            float rx = pos[j * 2] - px, ry = pos[j * 2 + 1] - py;
            float d = sqrtf(rx * rx + ry * ry + 0.0004f);
            float f = 1.0f / (d * d) * m / d;
            ax += rx * f;
            ay += ry * f;
        }
        vel[i * 2] += ax * dt;
        vel[i * 2 + 1] += ay * dt;
        nueva[i * 2] = px + vel[i * 2] * dt;
        nueva[i * 2 + 1] = py + vel[i * 2 + 1] * dt;
    }
    memcpy(pos, nueva, sizeof(float) * 2 * n);
    free(nueva);

    for (int i = 0; i < n; i++) {
        int x = (int)(ANCHO * 0.5f + pos[i * 2] * zoom);
        int y = (int)(ALTO * 0.5f - pos[i * 2 + 1] * zoom);
        if (x < 0 || y < 0 || x >= ANCHO || y >= ALTO)
            continue;
        float v = sqrtf(vel[i * 2] * vel[i * 2] + vel[i * 2 + 1] * vel[i * 2 + 1]);
        float c = fminf(v * 0.6f, 1.0f);
        int k = (y * ANCHO + x) * 3;
        luz[k] += 0.22f * (1.00f + (0.45f - 1.00f) * c);
        luz[k + 1] += 0.22f * (0.55f + (0.75f - 0.55f) * c);
        luz[k + 2] += 0.22f * (0.20f + (1.00f - 0.20f) * c);
    }
    for (int i = 0; i < ANCHO * ALTO; i++) {
        float r = 1.0f - expf(-luz[i * 3]);
        float g = 1.0f - expf(-luz[i * 3 + 1]);
        float b = 1.0f - expf(-luz[i * 3 + 2]);
        pixeles[i] = 0xFF000000u | ((Uint32)(r * 255) << 16) | ((Uint32)(g * 255) << 8) | (Uint32)(b * 255);
        luz[i * 3] *= 0.86f;
        luz[i * 3 + 1] *= 0.86f;
        luz[i * 3 + 2] *= 0.86f;
    }
}

static void guardar_ppm(const char *nombre)
{
    FILE *f = fopen(nombre, "wb");
    fprintf(f, "P6\n%d %d\n255\n", ANCHO, ALTO);
    for (int i = 0; i < ANCHO * ALTO; i++) {
        unsigned char rgb[3] = { (pixeles[i] >> 16) & 255, (pixeles[i] >> 8) & 255, pixeles[i] & 255 };
        fwrite(rgb, 1, 3, f);
    }
    fclose(f);
}

static double fecha_ley(void)
{
    struct stat st;
    if (stat("ley.cl", &st) != 0)
        return 0;
    return st.st_mtim.tv_sec + st.st_mtim.tv_nsec / 1e9;
}

int main(int argc, char **argv)
{
    int captura = 0, usar_cpu = 0;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--captura") && i + 1 < argc)
            captura = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--cpu"))
            usar_cpu = 1;
        else if (!strcmp(argv[i], "--estrellas") && i + 1 < argc)
            n = atoi(argv[++i]);
    }

    cl_platform_id plataforma;
    cl_int err;
    clGetPlatformIDs(1, &plataforma, NULL);
    clGetDeviceIDs(plataforma, CL_DEVICE_TYPE_GPU, 1, &gpu, NULL);
    char nombre_gpu[128];
    clGetDeviceInfo(gpu, CL_DEVICE_NAME, sizeof(nombre_gpu), nombre_gpu, NULL);
    ctx = clCreateContext(NULL, 1, &gpu, NULL, NULL, &err);
    cola = clCreateCommandQueue(ctx, gpu, 0, &err);

    pos = malloc(sizeof(float) * 2 * n);
    vel = malloc(sizeof(float) * 2 * n);
    luz = calloc(ANCHO * ALTO * 3, sizeof(float));
    pixeles = malloc(sizeof(Uint32) * ANCHO * ALTO);
    d_pos = clCreateBuffer(ctx, CL_MEM_READ_WRITE, sizeof(float) * 2 * n, NULL, &err);
    d_pos2 = clCreateBuffer(ctx, CL_MEM_READ_WRITE, sizeof(float) * 2 * n, NULL, &err);
    d_vel = clCreateBuffer(ctx, CL_MEM_READ_WRITE, sizeof(float) * 2 * n, NULL, &err);
    d_luz = clCreateBuffer(ctx, CL_MEM_READ_WRITE, sizeof(float) * ANCHO * ALTO * 3, NULL, &err);
    d_pix = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, sizeof(Uint32) * ANCHO * ALTO, NULL, &err);

    if (!fecha_ley())
        escribir_preset(0);
    double ms;
    if (!compilar_ley(&ms)) {
        printf("Arregla ley.cl y vuelve a correr el programa.\n");
        return 1;
    }
    double ultima = fecha_ley();

    crear_galaxias();
    subir_a_gpu();

    if (captura) {
        double t0 = ahora();
        float t = 0.0f;
        for (int f = 0; f < captura; f++) {
            if (usar_cpu)
                paso_cpu();
            else
                paso_gpu(t);
            t += dt;
        }
        clFinish(cola);
        double total = ahora() - t0;
        printf("%s: %d cuadros con %d estrellas en %.0f ms (%.1f FPS)\n", usar_cpu ? "CPU" : "GPU", captura, n, total, captura * 1000.0 / total);
        guardar_ppm("captura.ppm");
        return 0;
    }

    printf("\nGPU: %s | %d estrellas | %d hilos por grupo\n", nombre_gpu, n, GRUPO);
    printf("Edita ley.cl y guarda: el driver la recompila sin cerrar la ventana.\n");
    printf("Teclas: G = GPU, C = CPU, 1-6 = leyes de ejemplo, R = reiniciar, espacio = pausa, Esc = salir\n\n");

    SDL_Init(SDL_INIT_VIDEO);
    SDL_Window *ventana = SDL_CreateWindow("Galaxia", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, ANCHO, ALTO, SDL_WINDOW_RESIZABLE);
    SDL_Renderer *render = SDL_CreateRenderer(ventana, -1, SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC);
    SDL_RenderSetLogicalSize(render, ANCHO, ALTO);
    SDL_Texture *textura = SDL_CreateTexture(render, SDL_PIXELFORMAT_ARGB8888, SDL_TEXTUREACCESS_STREAMING, ANCHO, ALTO);

    int corriendo = 1, pausa = 0, ley_ok = 1, cuadros = 0;
    float t = 0.0f;
    double t_titulo = ahora(), ms_cuadro = 0.0;
    char titulo[256];

    while (corriendo) {
        SDL_Event e;
        while (SDL_PollEvent(&e)) {
            if (e.type == SDL_QUIT)
                corriendo = 0;
            else if (e.type == SDL_KEYDOWN) {
                SDL_Keycode k = e.key.keysym.sym;
                if (k == SDLK_ESCAPE)
                    corriendo = 0;
                else if (k == SDLK_SPACE)
                    pausa = !pausa;
                else if (k == SDLK_r) {
                    crear_galaxias();
                    t = 0.0f;
                    if (!usar_cpu)
                        subir_a_gpu();
                } else if (k == SDLK_c && !usar_cpu) {
                    bajar_de_gpu();
                    usar_cpu = 1;
                    printf("Modo CPU: 1 hilo, ley de gravedad escrita en C\n");
                } else if (k == SDLK_g && usar_cpu) {
                    subir_a_gpu();
                    usar_cpu = 0;
                    printf("Modo GPU\n");
                } else if (k >= SDLK_1 && k <= SDLK_6) {
                    escribir_preset(k - SDLK_1);
                }
            }
        }

        double fecha = fecha_ley();
        if (fecha != ultima) {
            ultima = fecha;
            SDL_Delay(30);
            ley_ok = compilar_ley(&ms);
        }

        double t0 = ahora();
        if (!pausa) {
            if (usar_cpu)
                paso_cpu();
            else
                paso_gpu(t);
            t += dt;
        }
        ms_cuadro = ms_cuadro * 0.9 + (ahora() - t0) * 0.1;

        SDL_UpdateTexture(textura, NULL, pixeles, ANCHO * sizeof(Uint32));
        SDL_RenderClear(render);
        SDL_RenderCopy(render, textura, NULL, NULL);
        SDL_RenderPresent(render);
        cuadros++;

        if (ahora() - t_titulo > 400 || usar_cpu) {
            double fps = cuadros * 1000.0 / (ahora() - t_titulo);
            snprintf(titulo, sizeof(titulo), "%s  |  %d estrellas  |  %.1f FPS  |  %.1f ms por cuadro  |  ley %s en %.1f ms%s",
                     usar_cpu ? "CPU (1 hilo)" : "GPU GTX 1650", n, fps, ms_cuadro,
                     ley_ok ? "compilada" : "CON ERROR", ms, pausa ? "  |  PAUSA" : "");
            SDL_SetWindowTitle(ventana, titulo);
            t_titulo = ahora();
            cuadros = 0;
        }
    }

    SDL_DestroyTexture(textura);
    SDL_DestroyRenderer(render);
    SDL_DestroyWindow(ventana);
    SDL_Quit();
    return 0;
}
