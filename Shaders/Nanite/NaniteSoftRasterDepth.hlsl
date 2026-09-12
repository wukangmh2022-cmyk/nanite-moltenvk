#include "NaniteCommon.hlsl"

// Depth-only software rasterizer, written as a price tag rather than as a
// rendering path: nothing downstream reads what it produces. Hardware raster is
// 19.9 of the frame's 26.5 ms, and the triangle-size heatmap says most of those
// triangles cover 1-4 pixels, which is the regime where a 2x2 quad granularity
// and fixed-function setup cost more than the pixels are worth. Whether a
// compute loop beats them here is what decides whether the visibility-buffer
// rewrite a real software path needs is worth starting, and that question
// deserves a measurement rather than an argument.
//
// This source is compiled twice. The first pass atomically resolves one 32-bit
// depth per pixel. The second pass repeats coverage and only lets a triangle whose
// depth equals that resolved minimum write the visibility payload. This preserves
// the depth/payload pairing without a native 64-bit atomic or a GPU spin lock.
//
// One workgroup per visible cluster. The workgroup first transforms the compact
// meshlet-local vertex table into shared memory; triangle lanes then reuse those
// screen-space vertices while doing coverage. This keeps the probe representative
// of a meshlet rasterizer instead of charging the same vertex transform once per
// triangle corner.

StructuredBuffer<NaniteInstance> g_Instances : register(t2);
StructuredBuffer<NaniteDrawInstanceData> g_DrawInstanceData : register(t4);
StructuredBuffer<NaniteSoftwareDrawIndexElement> g_SoftwareDrawIndices : register(t5);
StructuredBuffer<float3> g_ClusterPositionBuffer : register(t1);
StructuredBuffer<NaniteClusterIndexElement> g_ClusterIndexBuffer : register(t3);
StructuredBuffer<NaniteCluster> g_Clusters : register(t6);
StructuredBuffer<uint> g_ClusterLocalIndexOffsets : register(t7);

// One 32-bit depth per pixel. asuint of a non-negative float orders the same way
// the float does, so an integer comparison resolves nearest depth exactly.
RWStructuredBuffer<uint> g_SoftDepth : register(u16);
RWStructuredBuffer<NaniteQueueState> g_QueueState : register(u17);

// The visibility buffer proper: 64 bits per pixel, depth key in the high half and
// the winning triangle in the low half. It is only written by the second pass.
RWStructuredBuffer<NaniteVisibility> g_Visibility : register(u18);

#ifndef NANITE_SOFT_VISIBILITY_PASS
#define NANITE_SOFT_VISIBILITY_PASS 0
#endif

// Triangles whose screen bounding box exceeds this on either axis are left alone
// and counted. Set high enough that nothing in practice hits it: a per-thread
// scanline loop over a large triangle serializes work the hardware spreads over
// the whole GPU, so a cap looked prudent, but at 32 px it excluded 100 triangles
// per frame and cost 586 of the 607 pixels that then disagreed with the hardware
// depth - a large triangle that owns a pixel cannot be skipped without a farther
// one winning it. Removing the cap left the timing unchanged and took the
// disagreement to 21 pixels in 392k. It stays as a hang guard only, and the
// counter stays so a scene that does hit it is not silently mispriced.
#define NANITE_SOFT_MAX_EXTENT 4096
#define NANITE_SOFT_MAX_CLUSTER_VERTICES 128

// A meshlet's local vertex table is transformed once by the workgroup and then
// reused by every triangle. This is the locality pattern being measured by the
// probe; the fixed-function path remains on its original global index stream.
groupshared float3 s_ScreenVertices[NANITE_SOFT_MAX_CLUSTER_VERTICES];

[numthreads(NANITE_GROUP_SIZE, 1, 1)]
void main(uint3 groupID : SV_GroupID, uint3 threadID : SV_GroupThreadID)
{
    // Hybrid mode consumes the compacted software-cluster list. The legacy probe
    // keeps the canonical draw-list mapping so it can still price the full scene.
    const uint workIndex = groupID.x;
    const uint workCount = HybridRasterEnabled != 0u ?
        g_QueueState[0].SoftClusterCount : g_QueueState[0].DrawCount;
    if (workIndex >= workCount)
        return;
    const uint drawIndex = HybridRasterEnabled != 0u ?
        g_SoftwareDrawIndices[MaxDrawCommands + workIndex].Value : workIndex;

    const NaniteDrawInstanceData drawData = g_DrawInstanceData[drawIndex];
    const NaniteInstance instance = g_Instances[drawData.InstanceIndex];
    const NaniteCluster cluster = g_Clusters[drawData.ClusterIndex];
    const uint packedVertexOffset = cluster.VertexOffset;
    const uint vertexOffset = packedVertexOffset & 0x00FFFFFFu;
    const uint vertexCount = min(packedVertexOffset >> 24u,
                                 NANITE_SOFT_MAX_CLUSTER_VERTICES);
    const uint localIndexOffset = g_ClusterLocalIndexOffsets[drawData.ClusterIndex];
    const uint triangleCount = drawData.IndexCount / 3u;

#if !NANITE_SOFT_VISIBILITY_PASS
    // Keep the packed visibility UAV in the depth pass interface. The portable
    // SPIR-V fallback does not need it here, but the Apple MSL override uses this
    // same pass to atomically publish depth and payload as one 64-bit word.
    // The sentinel branch is unreachable after the normal visibility clear and
    // has no effect on the fallback result.
    const uint visibilityBindingGuard = g_Visibility[0].Key;
    if (visibilityBindingGuard == 0xFFFFFFFFu)
        g_SoftDepth[0] = visibilityBindingGuard;
#endif

    const float width = float(ViewportWidth);
    const float height = float(ViewportHeight);
    const int maxX = int(ViewportWidth) - 1;
    const int maxY = int(ViewportHeight) - 1;

    for (uint vertex = threadID.x; vertex < vertexCount; vertex += NANITE_GROUP_SIZE)
    {
        const float3 localPosition = g_ClusterPositionBuffer[vertexOffset + vertex];
        const float3 worldPosition = mul(instance.WorldMatrix, float4(localPosition, 1.0)).xyz;
        const float4 clip = mul(ViewProjMatrix, float4(worldPosition, 1.0));
        if (clip.w <= 1.0e-6)
        {
            s_ScreenVertices[vertex] = float3(0.0, 0.0, -1.0);
            continue;
        }
        const float invW = 1.0 / clip.w;
        s_ScreenVertices[vertex] = float3(
            (clip.x * invW * 0.5 + 0.5) * width,
            (0.5 - clip.y * invW * 0.5) * height,
            clip.z * invW);
    }
    GroupMemoryBarrierWithGroupSync();

    uint rasterized = 0u;
    uint pixels = 0u;
    uint skippedLarge = 0u;
    uint skippedClip = 0u;

    for (uint tri = threadID.x; tri < triangleCount; tri += NANITE_GROUP_SIZE)
    {
        float3 screen[3];
        bool clipped = false;
        for (uint corner = 0u; corner < 3u; ++corner)
        {
            const uint localVertex = g_ClusterIndexBuffer[
                localIndexOffset + tri * 3u + corner].Value;
            if (localVertex >= vertexCount)
            {
                clipped = true;
                break;
            }
            screen[corner] = s_ScreenVertices[localVertex];
            if (screen[corner].z < 0.0)
                clipped = true;
        }
        if (clipped)
        {
            ++skippedClip;
            continue;
        }

        const float2 p0 = screen[0].xy;
        const float2 p1 = screen[1].xy;
        const float2 p2 = screen[2].xy;

        // Signed doubled area. Zero means degenerate - including the padding
        // triangles a cluster's index range can end with - and there is nothing
        // to cover, so it is not counted as skipped work.
        const float area = (p1.x - p0.x) * (p2.y - p0.y) - (p1.y - p0.y) * (p2.x - p0.x);
        if (abs(area) < 1.0e-9)
            continue;

        // The hardware PSO rasterizes with CULL_MODE_NONE, so both windings have
        // to produce pixels here too. Folding the winding into a sign factor
        // keeps one inside test instead of two.
        const float winding = area < 0.0 ? -1.0 : 1.0;

        const float2 boundsMin = min(min(p0, p1), p2);
        const float2 boundsMax = max(max(p0, p1), p2);
        const int x0 = max(int(floor(boundsMin.x)), 0);
        const int y0 = max(int(floor(boundsMin.y)), 0);
        const int x1 = min(int(floor(boundsMax.x)), maxX);
        const int y1 = min(int(floor(boundsMax.y)), maxY);
        if (x0 > x1 || y0 > y1)
            continue; // Entirely offscreen; the hardware scissor drops it too.

        if ((x1 - x0) > NANITE_SOFT_MAX_EXTENT || (y1 - y0) > NANITE_SOFT_MAX_EXTENT)
        {
            ++skippedLarge;
            continue;
        }

        if (HybridRasterEnabled != 0u && ClusterRasterMode == 0u &&
            abs(area) * 0.5 > RasterBinAreaCutoff)
            continue;
#if !NANITE_SOFT_VISIBILITY_PASS
        ++rasterized;
#endif
        // Names the winner for the visibility write: this cluster's slot in the
        // frame's draw list plus the triangle within it, which is all the resolve
        // needs to refetch the same three vertices.
        const uint payload = NANITE_VIS_ENCODE(drawIndex, tri);
        const float invArea = 1.0 / area;
        // Walk the three edge functions incrementally across each scanline.
        // Recomputing them from pixel coordinates costs six multiplies and six
        // subtracts per candidate pixel; the deltas below preserve the same
        // half-pixel sample convention with only three adds per pixel.
        const float dx01 = p1.x - p0.x;
        const float dy01 = p1.y - p0.y;
        const float dx12 = p2.x - p1.x;
        const float dy12 = p2.y - p1.y;
        const float dx20 = p0.x - p2.x;
        const float dy20 = p0.y - p2.y;
        const float sampleX0 = float(x0) + 0.5;
        for (int y = y0; y <= y1; ++y)
        {
            const float sampleY = float(y) + 0.5;
            float edge0 = dx01 * (sampleY - p0.y) - dy01 * (sampleX0 - p0.x);
            float edge1 = dx12 * (sampleY - p1.y) - dy12 * (sampleX0 - p1.x);
            float edge2 = dx20 * (sampleY - p2.y) - dy20 * (sampleX0 - p2.x);
            for (int x = x0; x <= x1; ++x)
            {
                const bool inside = !(edge0 * winding < 0.0 ||
                                      edge1 * winding < 0.0 ||
                                      edge2 * winding < 0.0);
                if (inside)
                {
                    // NDC z interpolates linearly in screen space after the
                    // perspective divide, so plain barycentrics are correct.
                    const float depth = (edge1 * screen[0].z +
                                         edge2 * screen[1].z +
                                         edge0 * screen[2].z) * invArea;
                    // Outside the depth range the hardware would have clipped.
                    if (depth >= 0.0 && depth <= 1.0)
                    {
                        const uint offset = uint(y) * ViewportWidth + uint(x);
                        const uint depthBits = asuint(depth);
#if NANITE_SOFT_VISIBILITY_PASS
                        // Every writer that reaches this branch produced the
                        // exact depth selected by pass one. Exact ties may race
                        // on Payload, but every possible payload names a valid
                        // winner at that depth.
                        if (depthBits == g_SoftDepth[offset])
                        {
                            g_Visibility[offset].Payload = payload;
                            g_Visibility[offset].Key = ~depthBits;
                        }
#else
                        ++pixels;
                        uint previousDepth;
                        InterlockedMin(g_SoftDepth[offset], depthBits, previousDepth);
#endif
                    }
                }
                edge0 -= dy01;
                edge1 -= dy12;
                edge2 -= dy20;
            }
        }
    }

    // One atomic per thread per counter, and only when there is something to
    // add. Per-pixel counter traffic would dominate the very cost this pass
    // exists to measure.
    uint ignored;
#if !NANITE_SOFT_VISIBILITY_PASS
    if (rasterized != 0u)
        InterlockedAdd(g_QueueState[0].SoftRasterTriangles, rasterized, ignored);
    if (pixels != 0u)
        InterlockedAdd(g_QueueState[0].SoftRasterPixels, pixels, ignored);
    if (skippedLarge != 0u)
        InterlockedAdd(g_QueueState[0].SoftRasterSkippedLarge, skippedLarge, ignored);
    if (skippedClip != 0u)
        InterlockedAdd(g_QueueState[0].SoftRasterSkippedClip, skippedClip, ignored);
#endif
}
