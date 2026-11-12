// Ley: gravedad
// Guarda el archivo y la galaxia se recompila sola.
// d = distancia entre dos estrellas, t = tiempo
// p = posicion de la estrella, v = su velocidad

float fuerza(float d, float t)
{
    return 1.0f / (d * d);
}

float2 campo(float2 p, float2 v, float t)
{
    return (float2)(0.0f, 0.0f);
}
