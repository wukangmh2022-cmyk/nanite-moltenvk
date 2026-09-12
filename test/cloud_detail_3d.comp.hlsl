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

RWTexture3D<float4> g_CloudDetailVolume : register(u0);

[numthreads(4, 4, 4)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    if (dispatchThreadID.x >= g_Bake.Width ||
        dispatchThreadID.y >= g_Bake.Height ||
        dispatchThreadID.z >= g_Bake.Depth)
        return;

    const float3 uvw = (float3(dispatchThreadID) + 0.5) /
        float3(g_Bake.Width, g_Bake.Height, g_Bake.Depth);
    const float seed = (float)g_Bake.Seed * 11.31;
    const float3 p = uvw * g_Bake.Frequency +
        float3(g_Bake.Offset, seed);
    const float body = CloudBodyWorley3D(p);
    const float erosion = CloudErosionWorley3D(p);
    const float fine = CloudFineNoise3D(p);
    const float micro = CloudMicroNoise3D(p);

    // R is the billowy body, G is high-frequency curl-warped erosion, B is a
    // fine signal for soft wisps, and A is the micro band used by the second
    // DensityRemap in the view shader.
    g_CloudDetailVolume[dispatchThreadID] = float4(
        body, erosion, fine, micro);
}
