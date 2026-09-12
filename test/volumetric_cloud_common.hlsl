#ifndef VOLUMETRIC_CLOUD_COMMON_HLSL
#define VOLUMETRIC_CLOUD_COMMON_HLSL

#define VC_PI 3.14159265359
#define VC_TWO_PI 6.28318530718
#define VC_MAX_MS_OCTAVES 6

float Hash12(float2 p)
{
    float3 p3 = frac(float3(p.x, p.y, p.x) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return frac((p3.x + p3.y) * p3.z);
}

float2 Hash22(float2 p)
{
    float3 p3 = frac(float3(p.x, p.y, p.x) * float3(0.1031, 0.1030, 0.0973));
    p3 += dot(p3, p3.yzx + 33.33);
    return frac((p3.xx + p3.yz) * p3.zy);
}

float Hash31(float3 p)
{
    p = frac(p * 0.1031);
    p += dot(p, p.yzx + 33.33);
    return frac((p.x + p.y) * p.z);
}

float3 Hash33(float3 p)
{
    return float3(
        Hash31(p + float3(17.1, 3.7, 11.9)),
        Hash31(p + float3(29.4, 7.2, 19.6)),
        Hash31(p + float3(41.8, 13.5, 5.3)));
}

float ValueNoise3D(float3 p)
{
    const float3 cell = floor(p);
    const float3 f = frac(p);
    const float3 w = f * f * (3.0 - 2.0 * f);

    const float n000 = Hash31(cell + float3(0.0, 0.0, 0.0));
    const float n100 = Hash31(cell + float3(1.0, 0.0, 0.0));
    const float n010 = Hash31(cell + float3(0.0, 1.0, 0.0));
    const float n110 = Hash31(cell + float3(1.0, 1.0, 0.0));
    const float n001 = Hash31(cell + float3(0.0, 0.0, 1.0));
    const float n101 = Hash31(cell + float3(1.0, 0.0, 1.0));
    const float n011 = Hash31(cell + float3(0.0, 1.0, 1.0));
    const float n111 = Hash31(cell + float3(1.0, 1.0, 1.0));

    const float nx00 = lerp(n000, n100, w.x);
    const float nx10 = lerp(n010, n110, w.x);
    const float nx01 = lerp(n001, n101, w.x);
    const float nx11 = lerp(n011, n111, w.x);
    return lerp(lerp(nx00, nx10, w.y), lerp(nx01, nx11, w.y), w.z);
}

float ValueFbm3D(float3 p)
{
    float sum = 0.0;
    float amplitude = 0.5;
    float normalization = 0.0;

    [unroll]
    for (int octave = 0; octave < 4; ++octave)
    {
        sum += ValueNoise3D(p) * amplitude;
        normalization += amplitude;
        p = p * 2.01 + float3(19.1, 7.7, 13.3);
        amplitude *= 0.5;
    }
    return sum / max(normalization, 1.0e-5);
}

float WorleyNoise2D(float2 p)
{
    const float2 cell = floor(p);
    const float2 local = frac(p);
    float minimumDistance = 1.0e6;

    [unroll]
    for (int y = -1; y <= 1; ++y)
    {
        [unroll]
        for (int x = -1; x <= 1; ++x)
        {
            const float2 offset = float2((float)x, (float)y);
            const float2 feature = offset + Hash22(cell + offset);
            const float2 delta = feature - local;
            minimumDistance = min(minimumDistance, dot(delta, delta));
        }
    }
    return sqrt(minimumDistance);
}

// Low-frequency cloud coverage. Inverting the cell distance makes large,
// connected cloud bodies instead of isolated Voronoi cell centers.
float WorleyFbm2D(float2 p)
{
    float sum = 0.0;
    float amplitude = 0.5;
    float normalization = 0.0;

    [unroll]
    for (int octave = 0; octave < 4; ++octave)
    {
        sum += (1.0 - saturate(WorleyNoise2D(p))) * amplitude;
        normalization += amplitude;
        p = p * 2.03 + float2(17.0, 9.0);
        amplitude *= 0.5;
    }
    return sum / max(normalization, 1.0e-5);
}

float ValueNoise2D(float2 p)
{
    const float2 cell = floor(p);
    const float2 f = frac(p);
    const float2 w = f * f * (3.0 - 2.0 * f);
    const float n00 = Hash12(cell + float2(0.0, 0.0));
    const float n10 = Hash12(cell + float2(1.0, 0.0));
    const float n01 = Hash12(cell + float2(0.0, 1.0));
    const float n11 = Hash12(cell + float2(1.0, 1.0));
    return lerp(lerp(n00, n10, w.x), lerp(n01, n11, w.x), w.y);
}

float ValueFbm2D(float2 p)
{
    float sum = 0.0;
    float amplitude = 0.5;
    float normalization = 0.0;

    [unroll]
    for (int octave = 0; octave < 4; ++octave)
    {
        sum += ValueNoise2D(p) * amplitude;
        normalization += amplitude;
        p = p * 2.01 + float2(23.0, 41.0);
        amplitude *= 0.5;
    }
    return sum / max(normalization, 1.0e-5);
}

// A broad weather field keeps the low-frequency Worley FBM as the source of
// the cloud clusters, while a slower value field joins nearby cells into
// larger systems. The latter is only a coverage modulator; all 3D volume
// detail still comes from the Worley/curl volume below.
float CloudCoverage2D(float2 p)
{
    const float worley = WorleyFbm2D(p * 0.88 + float2(7.0, 19.0));
    const float broad = ValueFbm2D(p * 0.31 + float2(13.0, 29.0));
    return saturate(worley * 0.74 + broad * 0.26);
}

float WorleyNoise3D(float3 p)
{
    const float3 cell = floor(p);
    const float3 local = frac(p);
    float minimumDistance = 1.0e6;

    [unroll]
    for (int z = -1; z <= 1; ++z)
    {
        [unroll]
        for (int y = -1; y <= 1; ++y)
        {
            [unroll]
            for (int x = -1; x <= 1; ++x)
            {
                const float3 offset = float3((float)x, (float)y, (float)z);
                const float3 feature = offset + Hash33(cell + offset);
                const float3 delta = feature - local;
                minimumDistance = min(minimumDistance, dot(delta, delta));
            }
        }
    }
    return sqrt(minimumDistance);
}

float WorleyFbm3D(float3 p)
{
    float sum = 0.0;
    float amplitude = 0.5;
    float normalization = 0.0;

    [unroll]
    for (int octave = 0; octave < 5; ++octave)
    {
        sum += (1.0 - saturate(WorleyNoise3D(p))) * amplitude;
        normalization += amplitude;
        p = p * 2.01 + float3(13.0, 7.0, 23.0);
        amplitude *= 0.5;
    }
    return sum / max(normalization, 1.0e-5);
}

float3 VectorNoise3D(float3 p)
{
    return float3(ValueNoise3D(p + float3(11.0, 3.0, 17.0)),
                  ValueNoise3D(p + float3(29.0, 13.0, 5.0)),
                  ValueNoise3D(p + float3(47.0, 19.0, 31.0)));
}

// Finite-difference curl of a decorrelated vector field. The field is
// divergence-free, so using it as a coordinate warp preserves cloud volume
// better than repeatedly displacing with a scalar noise gradient.
float3 CurlNoise3D(float3 p)
{
    const float e = 0.08;
    const float3 dx = float3(e, 0.0, 0.0);
    const float3 dy = float3(0.0, e, 0.0);
    const float3 dz = float3(0.0, 0.0, e);

    const float3 x0 = VectorNoise3D(p - dx);
    const float3 x1 = VectorNoise3D(p + dx);
    const float3 y0 = VectorNoise3D(p - dy);
    const float3 y1 = VectorNoise3D(p + dy);
    const float3 z0 = VectorNoise3D(p - dz);
    const float3 z1 = VectorNoise3D(p + dz);

    const float3 curl = float3(
        (y1.z - y0.z) - (z1.y - z0.y),
        (z1.x - z0.x) - (x1.z - x0.z),
        (x1.y - x0.y) - (y1.x - y0.x));
    return curl / (2.0 * e);
}

// Detail signal requested by the cloud setup: Worley FBM supplies the
// billowed body and high-frequency curl-warped Worley/Value noise erodes it.
float WorleyCurlWarpFbm3D(float3 p)
{
    const float3 curl = CurlNoise3D(p * 2.5 + float3(7.0, 19.0, 31.0));
    const float3 warped = p + curl * 0.22;
    const float body = WorleyFbm3D(warped);
    const float highFrequency = 1.0 - WorleyNoise3D(
        warped * 4.0 + curl * 0.55 + float3(3.0, 17.0, 29.0));
    const float filament = ValueNoise3D(
        warped * 8.0 + curl * 0.28 + float3(23.0, 5.0, 41.0));
    return saturate(body * 0.58 + highFrequency * 0.27 + filament * 0.15);
}

float CloudBodyWorley3D(float3 p)
{
    const float3 curl = CurlNoise3D(p * 1.35 + float3(5.0, 11.0, 17.0));
    const float3 warped = p + curl * float3(0.18, 0.11, 0.18);
    return saturate(WorleyFbm3D(warped * 1.35));
}

float CloudErosionWorley3D(float3 p)
{
    const float3 curl = CurlNoise3D(p * 1.35 + float3(5.0, 11.0, 17.0));
    const float3 warped = p + curl * float3(0.18, 0.11, 0.18);
    return saturate(WorleyFbm3D(warped * 4.4 + curl * float3(0.32, 0.22, 0.32)));
}

float CloudFineNoise3D(float3 p)
{
    const float3 curl = CurlNoise3D(p * 1.35 + float3(5.0, 11.0, 17.0));
    const float3 warped = p + curl * float3(0.18, 0.11, 0.18);
    return saturate(ValueFbm3D(warped * 0.58 + float3(61.0, 71.0, 83.0)) * 0.76 +
                    (curl.y * 0.06 + 0.5));
}

// Highest-frequency erosion band. Packs a coarse Worley body with a fine
// value field so the baked volume carries real sub-cell variation that the
// view shader re-tiles for the micro DensityRemap.
float CloudMicroNoise3D(float3 p)
{
    const float3 curl = CurlNoise3D(p * 1.35 + float3(5.0, 11.0, 17.0));
    const float3 warped = p + curl * float3(0.18, 0.11, 0.18);
    return saturate((1.0 - WorleyNoise3D(warped * 3.4 +
                    float3(31.0, 47.0, 59.0))) * 0.68 +
                    ValueNoise3D(warped * 4.6 + float3(83.0, 61.0, 41.0)) * 0.32);
}

float3 SafeRayDirection(float3 direction)
{
    const float3 signDirection = 2.0 * step(0.0, direction) - 1.0;
    return signDirection * max(abs(direction), 1.0e-5);
}

bool IntersectAabb(float3 origin, float3 direction,
                   float3 boundsMin, float3 boundsMax,
                   out float nearDistance, out float farDistance)
{
    const float3 safeDirection = SafeRayDirection(direction);
    const float3 t0 = (boundsMin - origin) / safeDirection;
    const float3 t1 = (boundsMax - origin) / safeDirection;
    const float3 nearPlane = min(t0, t1);
    const float3 farPlane = max(t0, t1);
    nearDistance = max(max(nearPlane.x, nearPlane.y), nearPlane.z);
    farDistance = min(min(farPlane.x, farPlane.y), farPlane.z);
    return farDistance >= max(nearDistance, 0.0);
}

float HenyeyGreenstein(float cosTheta, float anisotropy)
{
    const float g = clamp(anisotropy, -0.95, 0.95);
    const float g2 = g * g;
    const float denominator = pow(max(1.0 + g2 - 2.0 * g * cosTheta, 1.0e-4), 1.5);
    return (1.0 - g2) / (4.0 * VC_PI * denominator);
}

float IsotropicPhase()
{
    return 1.0 / (4.0 * VC_PI);
}

#endif
