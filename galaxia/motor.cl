__kernel void mover(__global const float2 *pos,
                    __global float2 *vel,
                    __global float2 *pos_nueva,
                    const int n,
                    const float t,
                    const float dt,
                    __local float2 *bloque)
{
    int i = get_global_id(0);
    int li = get_local_id(0);
    int tam = get_local_size(0);
    float2 p = (i < n) ? pos[i] : (float2)(0.0f, 0.0f);
    float2 a = (float2)(0.0f, 0.0f);
    float m = 1.0f / n;

    for (int base = 0; base < n; base += tam) {
        int j = base + li;
        bloque[li] = (j < n) ? pos[j] : p;
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int k = 0; k < tam; k++) {
            float2 r = bloque[k] - p;
            float d = sqrt(dot(r, r) + 0.0004f);
            a += r * (fuerza(d, t) * m / d);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (i < n) {
        float2 v = vel[i] + (a + campo(p, vel[i], t)) * dt;
        vel[i] = v;
        pos_nueva[i] = p + v * dt;
    }
}

__kernel void dibujar(__global const float2 *pos,
                      __global const float2 *vel,
                      __global float *luz,
                      const int n,
                      const int ancho,
                      const int alto,
                      const float zoom)
{
    int i = get_global_id(0);
    if (i >= n)
        return;
    float2 p = pos[i];
    int x = (int)(ancho * 0.5f + p.x * zoom);
    int y = (int)(alto * 0.5f - p.y * zoom);
    if (x < 0 || y < 0 || x >= ancho || y >= alto)
        return;
    float c = clamp(length(vel[i]) * 0.6f, 0.0f, 1.0f);
    int k = (y * ancho + x) * 3;
    luz[k]     += 0.22f * mix(1.00f, 0.45f, c);
    luz[k + 1] += 0.22f * mix(0.55f, 0.75f, c);
    luz[k + 2] += 0.22f * mix(0.20f, 1.00f, c);
}

__kernel void empacar(__global float *luz,
                      __global uint *pixeles,
                      const float apagado)
{
    int i = get_global_id(0);
    float r = 1.0f - exp(-luz[i * 3]);
    float g = 1.0f - exp(-luz[i * 3 + 1]);
    float b = 1.0f - exp(-luz[i * 3 + 2]);
    pixeles[i] = 0xFF000000u | ((uint)(r * 255.0f) << 16) | ((uint)(g * 255.0f) << 8) | (uint)(b * 255.0f);
    luz[i * 3]     *= apagado;
    luz[i * 3 + 1] *= apagado;
    luz[i * 3 + 2] *= apagado;
}
