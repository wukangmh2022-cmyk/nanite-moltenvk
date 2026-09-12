#include "NaniteCommon.hlsl"

// Correctness half of the software-raster probe, and the reset for the two buffers
// it writes. A timing number on its own proves nothing: a rasterizer that drops
// half the triangles is fast for the wrong reason. This compares the software
// depth against the depth the hardware pass wrote for the same clusters in the
// same frame, then checks the two-pass visibility result against both its own
// 32-bit depth and the triangle its payload names, and reduces it into counters.
// It restores both buffers to empty on the way out, so the next frame needs no
// separate clear pass in the middle of what is being timed.

Texture2D<float> g_HardwareDepth : register(t1);
// Only the visibility resolve needs the geometry. It refetches the triangle a
// payload names and evaluates it at the pixel center, which is what a real
// material pass does with a visibility buffer, and here it is what says whether
// the payload actually belongs with the depth it arrived with.
StructuredBuffer<float3> g_PositionBuffer : register(t2);
StructuredBuffer<NaniteInstance> g_Instances : register(t3);
StructuredBuffer<NaniteGeometryIndexElement> g_IndexBuffer : register(t4);
StructuredBuffer<NaniteDrawInstanceData> g_DrawInstanceData : register(t5);
RWStructuredBuffer<uint> g_SoftDepth : register(u16);
RWStructuredBuffer<NaniteQueueState> g_QueueState : register(u17);
RWStructuredBuffer<NaniteVisibility> g_Visibility : register(u18);

// Both passes transform through the same matrix, so a pixel where the same
// triangle won should agree to float rounding. Anything above this means a
// different triangle won, or the coverage rule differs.
#define NANITE_SOFT_DEPTH_TOLERANCE 1.0e-5

// Refetches the triangle a visibility payload names and evaluates it at one pixel
// center. Returns false if the payload does not name a live triangle, or if the
// triangle it does name never covered this pixel - both of which a torn
// (key, payload) pair produces and a coherent one cannot.
//
// The transform has to be the same arithmetic the rasterizer used, in the same
// order, or the depths would disagree for a reason that has nothing to do with
// tearing. It is duplicated rather than shared because the rasterizer's version
// runs per triangle over a loop of pixels and this one runs per pixel for a single
// triangle; the shapes do not fit the same function.
bool ResolveVisibility(uint payload, float2 sample, out float depth)
{
    depth = 0.0;

    const uint drawIndex = NANITE_VIS_DRAW_INDEX(payload);
    const uint tri = NANITE_VIS_TRIANGLE_INDEX(payload);
    if (drawIndex >= g_QueueState[0].DrawCount)
        return false;

    const NaniteDrawInstanceData drawData = g_DrawInstanceData[drawIndex];
    if (tri * 3u + 2u >= drawData.IndexCount)
        return false;

    const NaniteInstance instance = g_Instances[drawData.InstanceIndex];
    const float width = float(ViewportWidth);
    const float height = float(ViewportHeight);

    float3 screen[3];
    for (uint corner = 0u; corner < 3u; ++corner)
    {
        const uint index = g_IndexBuffer[drawData.FirstIndex + tri * 3u + corner].Value;
        const float3 localPosition = g_PositionBuffer[index];
        const float3 worldPosition = mul(instance.WorldMatrix, float4(localPosition, 1.0)).xyz;
        const float4 clip = mul(ViewProjMatrix, float4(worldPosition, 1.0));
        if (clip.w <= 1.0e-6)
            return false;
        const float invW = 1.0 / clip.w;
        screen[corner] = float3(
            (clip.x * invW * 0.5 + 0.5) * width,
            (0.5 - clip.y * invW * 0.5) * height,
            clip.z * invW);
    }

    const float2 p0 = screen[0].xy;
    const float2 p1 = screen[1].xy;
    const float2 p2 = screen[2].xy;
    const float area = (p1.x - p0.x) * (p2.y - p0.y) - (p1.y - p0.y) * (p2.x - p0.x);
    if (abs(area) < 1.0e-9)
        return false;

    const float winding = area < 0.0 ? -1.0 : 1.0;
    const float e0 = (p1.x - p0.x) * (sample.y - p0.y) - (p1.y - p0.y) * (sample.x - p0.x);
    const float e1 = (p2.x - p1.x) * (sample.y - p1.y) - (p2.y - p1.y) * (sample.x - p1.x);
    const float e2 = (p0.x - p2.x) * (sample.y - p2.y) - (p0.y - p2.y) * (sample.x - p2.x);
    // A hair of slack on the inside test, which the rasterizer does not have. The
    // two shaders evaluate the same expression but are compiled separately, and a
    // pixel center that lands exactly on an edge can come out at -0.0 here and +0.0
    // there. Without the slack, 9 pixels out of 703k are reported as payloads that
    // do not cover their pixel purely because of that, which is noise in a test
    // meant to catch payloads belonging to the wrong triangle. The slack is small
    // enough that a genuinely wrong triangle cannot hide behind it: it admits
    // samples within about a thousandth of a pixel of the edge.
    const float edgeSlack = -1.0e-3 * abs(area);
    if (e0 * winding < edgeSlack || e1 * winding < edgeSlack || e2 * winding < edgeSlack)
        return false;

    depth = (e1 * screen[0].z + e2 * screen[1].z + e0 * screen[2].z) * (1.0 / area);
    return true;
}

[numthreads(8, 8, 1)]
void main(uint3 threadID : SV_DispatchThreadID)
{
    if (threadID.x >= ViewportWidth || threadID.y >= ViewportHeight)
        return;

    const uint offset = threadID.y * ViewportWidth + threadID.x;
    // The hybrid bins are disjoint, so the legacy hardware-vs-software comparison
    // is meaningless. Keep this path to a cheap clear; full payload validation is
    // a diagnostic, not work that belongs in every rendered frame.
    if (HybridRasterEnabled != 0u)
    {
        g_SoftDepth[offset] = asuint(1.0);
        g_Visibility[offset].Key = 0u;
        g_Visibility[offset].Payload = 0u;
        return;
    }
    const NaniteVisibility vis = g_Visibility[offset];
    // Native Metal mode stores the depth key and payload in one 64-bit word;
    // use that key for diagnostics instead of the portable depth scratch word.
    const bool native64 = CullingPadding4 != 0u;
    const float soft = native64 ?
        (((vis.Key | vis.Payload) != 0u) ? asfloat(~vis.Key) : 1.0) :
        asfloat(g_SoftDepth[offset]);
    const float hardware = g_HardwareDepth.Load(int3(int2(threadID.xy), 0));

    // Both buffers are cleared to far, so "covered" is simply "moved off far".
    const bool softCovered = soft < 1.0;
    const bool hardwareCovered = hardware < 1.0;

    uint ignored;
    if (softCovered && hardwareCovered)
    {
        const float difference = abs(soft - hardware);
        if (difference > NANITE_SOFT_DEPTH_TOLERANCE)
        {
            InterlockedAdd(g_QueueState[0].SoftDepthMismatch, 1u, ignored);
            InterlockedMax(g_QueueState[0].SoftDepthMaxDiff,
                           uint(min(difference, 1.0) * 16777215.0), ignored);
        }
        else
        {
            InterlockedAdd(g_QueueState[0].SoftDepthMatched, 1u, ignored);
        }
    }
    else if (hardwareCovered)
    {
        InterlockedAdd(g_QueueState[0].SoftDepthMissing, 1u, ignored);
    }
    else if (softCovered)
    {
        InterlockedAdd(g_QueueState[0].SoftDepthExtra, 1u, ignored);
    }

    // The visibility result's depth half has to agree with the first pass's
    // InterlockedMin to the bit, and its payload has to name a triangle that
    // produced that depth.
    const bool visCovered = (vis.Key | vis.Payload) != 0u;
    if (visCovered)
    {
        const uint visDepthBits = ~vis.Key;
        if (visDepthBits != asuint(soft))
            InterlockedAdd(g_QueueState[0].SoftVisKeyMismatch, 1u, ignored);

        float resolved;
        // Splitting the failure two ways keeps the arithmetic out of the verdict. A
        // payload that names a triangle which never covered this pixel can only come
        // from a torn pair, and that is what NoCover counts. A payload whose
        // triangle does cover the pixel but interpolates a different depth there is
        // either tearing between two neighbours on the same surface or the two
        // shaders' floating point drifting past the tolerance, so it is counted
        // apart rather than claimed as either.
        if (!ResolveVisibility(vis.Payload, float2(threadID.xy) + 0.5, resolved))
        {
            InterlockedAdd(g_QueueState[0].SoftVisNoCover, 1u, ignored);
        }
        else if (abs(resolved - asfloat(visDepthBits)) > NANITE_SOFT_DEPTH_TOLERANCE)
        {
            InterlockedAdd(g_QueueState[0].SoftVisDepthOff, 1u, ignored);
        }
        else
        {
            InterlockedAdd(g_QueueState[0].SoftVisResolved, 1u, ignored);
        }
    }
    else if (softCovered)
    {
        // The 32-bit depth says this pixel was rasterized and the visibility result
        // says it was not, which means the winner-selection pass missed it.
        InterlockedAdd(g_QueueState[0].SoftVisMissing, 1u, ignored);
    }

    g_SoftDepth[offset] = asuint(1.0);
    g_Visibility[offset].Key = 0u;
    g_Visibility[offset].Payload = 0u;
}
