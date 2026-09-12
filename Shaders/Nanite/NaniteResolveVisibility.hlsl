#include "NaniteCommon.hlsl"

Texture2D<float> g_HardwareDepth : register(t1);
Texture2D<uint> g_HardwareVisibility : register(t2);
StructuredBuffer<NaniteVisibility> g_SoftVisibility : register(t3);
RWTexture2D<float> g_FinalDepth : register(u16);
RWTexture2D<uint> g_FinalVisibility : register(u17);

[numthreads(8, 8, 1)]
void main(uint3 threadID : SV_DispatchThreadID)
{
    if (threadID.x >= ViewportWidth || threadID.y >= ViewportHeight)
        return;

    const uint2 pixel = threadID.xy;
    const uint offset = pixel.y * ViewportWidth + pixel.x;
    // A hybrid-raster fragment can leave a depth value behind while deliberately
    // producing no visibility payload (the small-triangle bin is owned by the
    // compute rasterizer).  Depth alone is therefore not a valid hardware
    // candidate: accepting it here would make that pixel win the resolve, then
    // shade as background because its payload is zero.  Keep depth and payload
    // coupled so that every accepted winner describes the same primitive.
    const float hardwareDepth = g_HardwareDepth.Load(int3(pixel, 0));
    const uint hardwarePayload = hardwareDepth < 1.0 ?
        g_HardwareVisibility.Load(int3(pixel, 0)) : 0u;
    float depth = hardwarePayload != 0u ? hardwareDepth : 1.0;
    uint payload = hardwarePayload;

    const NaniteVisibility soft = g_SoftVisibility[offset];
    if (soft.Key != 0u)
    {
        const float softDepth = asfloat(~soft.Key);
        // Equality stays with hardware, matching the graphics path's stable
        // attachment order. The winning payload is always written with its depth.
        if (softDepth < depth)
        {
            depth = softDepth;
            payload = soft.Payload;
        }
    }

    g_FinalDepth[pixel] = depth;
    g_FinalVisibility[pixel] = payload;
}
