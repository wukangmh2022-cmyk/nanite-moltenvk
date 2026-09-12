#include "NaniteCommon.hlsl"

StructuredBuffer<float3> g_PositionBuffer : register(t1);
StructuredBuffer<NaniteInstance> g_Instances : register(t2);
StructuredBuffer<NaniteGeometryIndexElement> g_IndexBuffer : register(t3);
StructuredBuffer<NaniteDrawInstanceData> g_DrawInstanceData : register(t4);

globallycoherent RWByteAddressBuffer g_HybridDrawCommand : register(u16);
RWStructuredBuffer<NaniteQueueState> g_QueueState : register(u17);
RWStructuredBuffer<NaniteBinDrawIndexElement> g_BinDrawIndices : register(u18);

#define NANITE_DRAW_INSTANCE_COUNT_OFFSET 4u
#define NANITE_HARDWARE_BIN_MIXED 0x80000000u
#define NANITE_SOFT_MAX_EXTENT 4096

groupshared uint s_BinFlags;
// The experiment classifies one whole cluster from its projected AABB. Keeping
// one min/max pair per lane avoids float atomics and costs only 64 * 16 bytes of
// group shared memory, well below the Metal/Vulkan limit used by this demo.
groupshared float2 s_ThreadBoundsMin[NANITE_GROUP_SIZE];
groupshared float2 s_ThreadBoundsMax[NANITE_GROUP_SIZE];
groupshared uint s_ThreadHasGeometry[NANITE_GROUP_SIZE];
groupshared uint s_ThreadNearClipped[NANITE_GROUP_SIZE];

[numthreads(NANITE_GROUP_SIZE, 1, 1)]
void main(uint3 groupID : SV_GroupID, uint3 threadID : SV_GroupThreadID)
{
    const uint lane = threadID.x;
    if (threadID.x == 0u)
        s_BinFlags = 0u;
    s_ThreadBoundsMin[lane] = float2(1.0e30, 1.0e30);
    s_ThreadBoundsMax[lane] = float2(-1.0e30, -1.0e30);
    s_ThreadHasGeometry[lane] = 0u;
    s_ThreadNearClipped[lane] = 0u;
    GroupMemoryBarrierWithGroupSync();

    const uint drawIndex = groupID.x;
    if (drawIndex >= g_QueueState[0].DrawCount)
        return;

    // A triangle clipped to the viewport cannot exceed half its pixel area. At
    // or above that cutoff every in-bounds triangle belongs to software, so do
    // not pay for a redundant projection/area scan just to discover that fact.
    // The software rasterizer still performs its normal bounds and depth checks.
    const float fullSoftwareCutoff =
        float(ViewportWidth) * float(ViewportHeight) * 0.5;
    if (RasterBinAreaCutoff >= fullSoftwareCutoff)
    {
        if (threadID.x == 0u)
        {
            uint softwareIndex;
            InterlockedAdd(g_QueueState[0].SoftClusterCount, 1u, softwareIndex);
            if (softwareIndex < MaxDrawCommands)
                g_BinDrawIndices[MaxDrawCommands + softwareIndex].Value = drawIndex;
            else
            {
                uint ignored;
                InterlockedOr(g_QueueState[0].OverflowFlags, 0x400u, ignored);
            }
        }
        return;
    }

    const NaniteDrawInstanceData drawData = g_DrawInstanceData[drawIndex];
    const NaniteInstance instance = g_Instances[drawData.InstanceIndex];
    const uint triangleCount = drawData.IndexCount / 3u;
    const float2 viewport = float2(ViewportWidth, ViewportHeight);
    const int maxX = int(ViewportWidth) - 1;
    const int maxY = int(ViewportHeight) - 1;

    for (uint tri = threadID.x; tri < triangleCount; tri += NANITE_GROUP_SIZE)
    {
        float2 screen[3];
        bool nearClipped = false;
        for (uint corner = 0u; corner < 3u; ++corner)
        {
            const uint index = g_IndexBuffer[drawData.FirstIndex + tri * 3u + corner].Value;
            const float3 worldPosition = mul(
                instance.WorldMatrix, float4(g_PositionBuffer[index], 1.0)).xyz;
            const float4 clip = mul(ViewProjMatrix, float4(worldPosition, 1.0));
            if (clip.w <= 1.0e-6)
            {
                nearClipped = true;
                break;
            }
            const float invW = 1.0 / clip.w;
            screen[corner] = float2(
                (clip.x * invW * 0.5 + 0.5) * viewport.x,
                (0.5 - clip.y * invW * 0.5) * viewport.y);
        }

        uint ignored;
        if (nearClipped)
        {
            if (ClusterRasterMode != 0u)
                s_ThreadNearClipped[lane] = 1u;
            else
                InterlockedOr(s_BinFlags, 2u, ignored);
            continue;
        }

        const float2 p0 = screen[0];
        const float2 p1 = screen[1];
        const float2 p2 = screen[2];
        const float area = (p1.x - p0.x) * (p2.y - p0.y) -
                           (p1.y - p0.y) * (p2.x - p0.x);
        if (abs(area) < 1.0e-9)
            continue;

        const float2 boundsMin = min(min(p0, p1), p2);
        const float2 boundsMax = max(max(p0, p1), p2);
        if (ClusterRasterMode != 0u)
        {
            s_ThreadBoundsMin[lane] = min(s_ThreadBoundsMin[lane], boundsMin);
            s_ThreadBoundsMax[lane] = max(s_ThreadBoundsMax[lane], boundsMax);
            s_ThreadHasGeometry[lane] = 1u;
            continue;
        }
        const int x0 = max(int(floor(boundsMin.x)), 0);
        const int y0 = max(int(floor(boundsMin.y)), 0);
        const int x1 = min(int(floor(boundsMax.x)), maxX);
        const int y1 = min(int(floor(boundsMax.y)), maxY);
        if (x0 > x1 || y0 > y1)
            continue;

        const bool software = abs(area) * 0.5 <= RasterBinAreaCutoff &&
            (x1 - x0) <= NANITE_SOFT_MAX_EXTENT &&
            (y1 - y0) <= NANITE_SOFT_MAX_EXTENT;
        InterlockedOr(s_BinFlags, software ? 1u : 2u, ignored);
    }

    GroupMemoryBarrierWithGroupSync();
    if (ClusterRasterMode != 0u && threadID.x == 0u)
    {
        float2 clusterMin = float2(1.0e30, 1.0e30);
        float2 clusterMax = float2(-1.0e30, -1.0e30);
        bool hasGeometry = false;
        bool nearClipped = false;
        for (uint laneIndex = 0u; laneIndex < NANITE_GROUP_SIZE; ++laneIndex)
        {
            hasGeometry = hasGeometry || s_ThreadHasGeometry[laneIndex] != 0u;
            nearClipped = nearClipped || s_ThreadNearClipped[laneIndex] != 0u;
            if (s_ThreadHasGeometry[laneIndex] != 0u)
            {
                clusterMin = min(clusterMin, s_ThreadBoundsMin[laneIndex]);
                clusterMax = max(clusterMax, s_ThreadBoundsMax[laneIndex]);
            }
        }

        // A near-plane crossing needs the fixed-function clipper: the simple
        // compute rasterizer intentionally skips such triangles. Keep the whole
        // cluster on hardware in that case so the experiment cannot introduce a
        // hole by splitting the cluster around the clip plane.
        if (hasGeometry && !nearClipped)
        {
            const int maxX = int(ViewportWidth) - 1;
            const int maxY = int(ViewportHeight) - 1;
            const int x0 = max(int(floor(clusterMin.x)), 0);
            const int y0 = max(int(floor(clusterMin.y)), 0);
            const int x1 = min(int(floor(clusterMax.x)), maxX);
            const int y1 = min(int(floor(clusterMax.y)), maxY);
            if (x0 <= x1 && y0 <= y1 &&
                (x1 - x0) <= NANITE_SOFT_MAX_EXTENT &&
                (y1 - y0) <= NANITE_SOFT_MAX_EXTENT)
            {
                const float aabbArea = float(x1 - x0 + 1) * float(y1 - y0 + 1);
                s_BinFlags = aabbArea <= RasterBinAreaCutoff ? 1u : 2u;
            }
        }
    }

    GroupMemoryBarrierWithGroupSync();
    if (threadID.x != 0u)
        return;

    const bool hasSoftware = (s_BinFlags & 1u) != 0u;
    const bool hasHardware = (s_BinFlags & 2u) != 0u;
    uint ignored;
    if (hasSoftware)
    {
        uint softwareIndex;
        InterlockedAdd(g_QueueState[0].SoftClusterCount, 1u, softwareIndex);
        if (softwareIndex < MaxDrawCommands)
            g_BinDrawIndices[MaxDrawCommands + softwareIndex].Value = drawIndex;
        else
            InterlockedOr(g_QueueState[0].OverflowFlags, 0x400u, ignored);
    }
    if (hasHardware)
    {
        uint hardwareIndex;
        uint hardwareCountIgnored;
        InterlockedAdd(g_QueueState[0].HardwareClusterCount, 1u, hardwareCountIgnored);
        g_HybridDrawCommand.InterlockedAdd(
            NANITE_DRAW_INSTANCE_COUNT_OFFSET, 1u, hardwareIndex);
        if (hardwareIndex < MaxDrawCommands)
        {
            const uint mixedFlag = hasSoftware ? NANITE_HARDWARE_BIN_MIXED : 0u;
            g_BinDrawIndices[hardwareIndex].Value = drawIndex | mixedFlag;
        }
        else
        {
            InterlockedOr(g_QueueState[0].OverflowFlags, 0x800u, ignored);
        }
    }
}
