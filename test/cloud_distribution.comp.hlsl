#include "volumetric_cloud_common.hlsl"

struct CloudBakeConstants
{
    uint Width;
    uint Height;
    uint Depth;
    uint Seed;
    float Coverage;
    float Frequency;
    float2 Offset;
    float2 Padding;
};

cbuffer CloudBakeCB : register(b0)
{
    CloudBakeConstants g_Bake;
};

RWTexture2D<float4> g_CloudDistribution : register(u0);

[numthreads(8, 8, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    if (dispatchThreadID.x >= g_Bake.Width || dispatchThreadID.y >= g_Bake.Height)
        return;

    const float2 uv = (float2(dispatchThreadID.xy) + 0.5) /
        float2(g_Bake.Width, g_Bake.Height);
    const float seed = (float)g_Bake.Seed * 17.17;
    const float2 p = uv * g_Bake.Frequency + g_Bake.Offset +
        float2(seed, seed * 1.37);
    const float fbm = CloudCoverage2D(p);

    // Coverage remaps the low-frequency FBM into a controllable cloud mask.
    // R is the continuous FBM, G is the thresholded coverage, and B stores a
    // softer edge useful for temporal filtering or artist preview tools.
    // Worley FBM occupies roughly [0.3, 0.9], so direct 1-coverage would
    // make the default coverage almost completely filled.
    const float threshold = lerp(0.68, 0.54, saturate(g_Bake.Coverage));
    const float mask = smoothstep(threshold - 0.065, threshold + 0.075, fbm);
    const float edge = smoothstep(threshold - 0.22, threshold + 0.04, fbm);
    g_CloudDistribution[dispatchThreadID.xy] = float4(fbm, mask, edge, 1.0);
}
