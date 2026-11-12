__kernel void suma_vectores(__global const float *a,
                            __global const float *b,
                            __global float *c,
                            const int n)
{
    int i = get_global_id(0);
    if (i < n)
        c[i] = a[i] + b[i];
}

__kernel void multiplicar_matrices(__global const float *a,
                                   __global const float *b,
                                   __global float *c,
                                   const int n)
{
    int fila = get_global_id(1);
    int col = get_global_id(0);
    if (fila < n && col < n) {
        float suma = 0.0f;
        for (int k = 0; k < n; k++)
            suma += a[fila * n + k] * b[k * n + col];
        c[fila * n + col] = suma;
    }
}
