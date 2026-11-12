#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double ahora(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec * 1000.0 + t.tv_nsec / 1e6;
}

static void revisar(cl_int err, const char *donde)
{
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error %d en %s\n", err, donde);
        exit(1);
    }
}

static char *leer_archivo(const char *nombre)
{
    FILE *f = fopen(nombre, "rb");
    if (!f) {
        fprintf(stderr, "No se pudo abrir %s\n", nombre);
        exit(1);
    }
    fseek(f, 0, SEEK_END);
    long largo = ftell(f);
    rewind(f);
    char *texto = malloc(largo + 1);
    fread(texto, 1, largo, f);
    texto[largo] = '\0';
    fclose(f);
    return texto;
}

static void mostrar_log(cl_program prog, cl_device_id gpu)
{
    size_t largo;
    clGetProgramBuildInfo(prog, gpu, CL_PROGRAM_BUILD_LOG, 0, NULL, &largo);
    char *log = malloc(largo + 1);
    clGetProgramBuildInfo(prog, gpu, CL_PROGRAM_BUILD_LOG, largo, log, NULL);
    log[largo] = '\0';
    printf("%s\n", log);
    free(log);
}

static void guardar_ptx(cl_program prog, const char *nombre)
{
    size_t largo;
    clGetProgramInfo(prog, CL_PROGRAM_BINARY_SIZES, sizeof(largo), &largo, NULL);
    unsigned char *bin = malloc(largo);
    clGetProgramInfo(prog, CL_PROGRAM_BINARIES, sizeof(bin), &bin, NULL);
    while (largo > 0 && bin[largo - 1] == '\0')
        largo--;
    FILE *f = fopen(nombre, "wb");
    fwrite(bin, 1, largo, f);
    fclose(f);
    free(bin);
    printf("Codigo PTX guardado en %s (%zu bytes)\n", nombre, largo);
}

static void suma_cpu(const float *a, const float *b, float *c, int n)
{
    for (int i = 0; i < n; i++)
        c[i] = a[i] + b[i];
}

static void matrices_cpu(const float *a, const float *b, float *c, int n)
{
    for (int fila = 0; fila < n; fila++)
        for (int col = 0; col < n; col++) {
            float suma = 0.0f;
            for (int k = 0; k < n; k++)
                suma += a[fila * n + k] * b[k * n + col];
            c[fila * n + col] = suma;
        }
}

static float diferencia(const float *x, const float *y, int n)
{
    float max = 0.0f;
    for (int i = 0; i < n; i++) {
        float d = x[i] - y[i];
        if (d < 0) d = -d;
        if (d > max) max = d;
    }
    return max;
}

int main(int argc, char **argv)
{
    const char *archivo = argc > 1 ? argv[1] : "kernels.cl";
    int n_mat = argc > 2 ? atoi(argv[2]) : 1024;
    int n_vec = 1 << 24;
    cl_int err;

    cl_platform_id plataforma;
    cl_device_id gpu;
    revisar(clGetPlatformIDs(1, &plataforma, NULL), "clGetPlatformIDs");
    revisar(clGetDeviceIDs(plataforma, CL_DEVICE_TYPE_GPU, 1, &gpu, NULL), "clGetDeviceIDs");

    char nombre[128];
    cl_uint unidades;
    clGetDeviceInfo(gpu, CL_DEVICE_NAME, sizeof(nombre), nombre, NULL);
    clGetDeviceInfo(gpu, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(unidades), &unidades, NULL);
    printf("GPU: %s (%u multiprocesadores)\n\n", nombre, unidades);

    cl_context ctx = clCreateContext(NULL, 1, &gpu, NULL, NULL, &err);
    revisar(err, "clCreateContext");
    cl_command_queue cola = clCreateCommandQueue(ctx, gpu, 0, &err);
    revisar(err, "clCreateCommandQueue");

    printf("== 1. Compilar %s en tiempo de ejecucion ==\n", archivo);
    char *fuente = leer_archivo(archivo);
    cl_program prog = clCreateProgramWithSource(ctx, 1, (const char **)&fuente, NULL, &err);
    revisar(err, "clCreateProgramWithSource");
    double t0 = ahora();
    err = clBuildProgram(prog, 1, &gpu, "-cl-std=CL1.2", NULL, NULL);
    double t_build = ahora() - t0;
    if (err != CL_SUCCESS) {
        printf("El compilador del driver encontro errores:\n");
        mostrar_log(prog, gpu);
        return 1;
    }
    printf("Compilado en %.1f ms\n", t_build);
    guardar_ptx(prog, "kernels.ptx");

    cl_kernel k_suma = clCreateKernel(prog, "suma_vectores", &err);
    revisar(err, "clCreateKernel suma_vectores");
    cl_kernel k_mat = clCreateKernel(prog, "multiplicar_matrices", &err);
    revisar(err, "clCreateKernel multiplicar_matrices");

    printf("\n== 2. Suma de vectores (%d elementos) ==\n", n_vec);
    size_t bytes = n_vec * sizeof(float);
    float *a = malloc(bytes), *b = malloc(bytes), *c_cpu = malloc(bytes), *c_gpu = malloc(bytes);
    for (int i = 0; i < n_vec; i++) {
        a[i] = (float)(i % 1000);
        b[i] = (float)(i % 7);
    }

    t0 = ahora();
    suma_cpu(a, b, c_cpu, n_vec);
    double t_cpu = ahora() - t0;

    t0 = ahora();
    cl_mem da = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, bytes, a, &err);
    cl_mem db = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, bytes, b, &err);
    cl_mem dc = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, bytes, NULL, &err);
    clFinish(cola);
    double t_copia = ahora() - t0;

    clSetKernelArg(k_suma, 0, sizeof(cl_mem), &da);
    clSetKernelArg(k_suma, 1, sizeof(cl_mem), &db);
    clSetKernelArg(k_suma, 2, sizeof(cl_mem), &dc);
    clSetKernelArg(k_suma, 3, sizeof(int), &n_vec);
    size_t global1 = n_vec;
    t0 = ahora();
    revisar(clEnqueueNDRangeKernel(cola, k_suma, 1, NULL, &global1, NULL, 0, NULL, NULL), "suma_vectores");
    clFinish(cola);
    double t_gpu = ahora() - t0;

    t0 = ahora();
    clEnqueueReadBuffer(cola, dc, CL_TRUE, 0, bytes, c_gpu, 0, NULL, NULL);
    t_copia += ahora() - t0;

    printf("CPU (1 hilo):      %8.2f ms\n", t_cpu);
    printf("GPU (kernel):      %8.2f ms\n", t_gpu);
    printf("Copiar datos:      %8.2f ms\n", t_copia);
    printf("Diferencia maxima: %g\n", diferencia(c_cpu, c_gpu, n_vec));

    clReleaseMemObject(da);
    clReleaseMemObject(db);
    clReleaseMemObject(dc);
    free(a); free(b); free(c_cpu); free(c_gpu);

    printf("\n== 3. Multiplicacion de matrices (%dx%d) ==\n", n_mat, n_mat);
    int total = n_mat * n_mat;
    bytes = total * sizeof(float);
    a = malloc(bytes); b = malloc(bytes); c_cpu = malloc(bytes); c_gpu = malloc(bytes);
    for (int i = 0; i < total; i++) {
        a[i] = (float)(i % 10) / 10.0f;
        b[i] = (float)(i % 3);
    }

    t0 = ahora();
    matrices_cpu(a, b, c_cpu, n_mat);
    t_cpu = ahora() - t0;

    t0 = ahora();
    da = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, bytes, a, &err);
    db = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, bytes, b, &err);
    dc = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, bytes, NULL, &err);
    clFinish(cola);
    t_copia = ahora() - t0;

    clSetKernelArg(k_mat, 0, sizeof(cl_mem), &da);
    clSetKernelArg(k_mat, 1, sizeof(cl_mem), &db);
    clSetKernelArg(k_mat, 2, sizeof(cl_mem), &dc);
    clSetKernelArg(k_mat, 3, sizeof(int), &n_mat);
    size_t global2[2] = { n_mat, n_mat };
    size_t local2[2] = { 16, 16 };
    t0 = ahora();
    revisar(clEnqueueNDRangeKernel(cola, k_mat, 2, NULL, global2, local2, 0, NULL, NULL), "multiplicar_matrices");
    clFinish(cola);
    t_gpu = ahora() - t0;

    t0 = ahora();
    clEnqueueReadBuffer(cola, dc, CL_TRUE, 0, bytes, c_gpu, 0, NULL, NULL);
    t_copia += ahora() - t0;

    printf("Hilos lanzados en la GPU: %d (grupos de 16x16)\n", total);
    printf("CPU (1 hilo):      %8.2f ms\n", t_cpu);
    printf("GPU (kernel):      %8.2f ms\n", t_gpu);
    printf("Copiar datos:      %8.2f ms\n", t_copia);
    printf("La GPU fue %.1f veces mas rapida (contando la copia)\n", t_cpu / (t_gpu + t_copia));
    printf("Diferencia maxima: %g\n", diferencia(c_cpu, c_gpu, total));

    clReleaseMemObject(da);
    clReleaseMemObject(db);
    clReleaseMemObject(dc);
    free(a); free(b); free(c_cpu); free(c_gpu);
    free(fuente);
    clReleaseKernel(k_suma);
    clReleaseKernel(k_mat);
    clReleaseProgram(prog);
    clReleaseCommandQueue(cola);
    clReleaseContext(ctx);
    return 0;
}
