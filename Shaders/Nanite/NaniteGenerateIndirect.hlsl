#include "NaniteCommon.hlsl"

StructuredBuffer<NaniteInstance> g_Instances : register(t1);
StructuredBuffer<NaniteCluster> g_Clusters : register(t2);
StructuredBuffer<NaniteClusterTask> g_ClusterQueue : register(t3);
Texture2D<float> g_HZB : register(t4);
StructuredBuffer<NaniteClusterGroup> g_ClusterGroups : register(t5);
SamplerState g_HZBSampler : register(s8);

// The whole raster pass is one non-indexed instanced draw, so there is a single
// 16-byte VkDrawIndirectCommand instead of an array of them. Claiming a slot is
// the atomic increment of its InstanceCount field, which is why no indirect
// count buffer is needed: DrawCount is always 1 and only InstanceCount comes
// from GPU memory, which plain vkCmdDrawIndirect already reads.
globallycoherent RWByteAddressBuffer g_DrawCommand : register(u16);
RWStructuredBuffer<NaniteQueueState> g_QueueState : register(u17);
RWStructuredBuffer<NaniteDrawInstanceData> g_DrawInstanceData : register(u19);
globallycoherent RWStructuredBuffer<NaniteClusterTask> g_PostClusterQueue : register(u20);

// Byte offset of InstanceCount inside the command.
#define NANITE_DRAW_INSTANCE_COUNT_OFFSET 4u

[numthreads(NANITE_GROUP_SIZE, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint clusterTaskIndex = dispatchThreadID.x;
    uint clusterCount = g_QueueState[0].VisibleClusterCount;
    if (dispatchThreadID.x == 0u && clusterCount > g_QueueState[0].DispatchedClusterTasks)
    {
        // NanitePrepareDispatch recorded how far the grid it wrote can reach. If
        // the queue holds more clusters than that, the tail is outside the grid
        // and would never become draw commands.
        InterlockedOr(g_QueueState[0].OverflowFlags, NANITE_OVERFLOW_DISPATCH_TOO_SMALL);
    }
    if (CullingPass == NANITE_CULLING_PASS_POST)
    {
        // Only the clusters the recovery traversal appended. Everything below
        // the mark was already turned into draw commands by the main phase.
        clusterTaskIndex += g_QueueState[0].MainClusterCount;
    }
    if (clusterTaskIndex >= clusterCount)
        return;

    NaniteClusterTask task = g_ClusterQueue[clusterTaskIndex];
    NaniteInstance instance = g_Instances[task.InstanceIndex];
    NaniteCluster cluster = g_Clusters[task.ClusterIndex];
    bool debugTarget = (MaxRefinementDepth & 0x80000000u) != 0u &&
        task.InstanceIndex == 10u && task.ClusterIndex == 517u;
    if (debugTarget)
    {
        InterlockedOr(g_QueueState[0].OverflowFlags, 0x00010000u);
        if (EnableHZB == 0u)
            InterlockedOr(g_QueueState[0].OverflowFlags, 0x00020000u);
    }
    bool frustumVisible;
    bool hzbVisible;
    float hzbDepth;
    uint hzbMip;
    uint2 hzbTexelMin;
    uint2 hzbTexelMax;
    bool boundsVisible = EvaluateBoundsVisible(
        cluster.BBoxMin,
        cluster.BBoxMax,
        instance.WorldMatrix,
        g_HZB,
        g_HZBSampler,
        frustumVisible,
        hzbVisible,
        hzbDepth,
        hzbMip,
        hzbTexelMin,
        hzbTexelMax);
    if (!frustumVisible)
    {
        uint ignored;
        InterlockedAdd(g_QueueState[0].ClusterFrustumRejected, 1u, ignored);
        if (debugTarget)
            InterlockedOr(g_QueueState[0].OverflowFlags, 0x00040000u);
        return;
    }

    // LOD selection must happen before the occlusion test. A cluster rejected by
    // the HZB is pushed to the post queue, and the post pass only re-tests
    // frustum and HZB - it has no refinement test. Testing HZB first therefore
    // lets clusters at the wrong LOD reach the post raster pass, drawing parent
    // and child geometry on top of each other and adding work instead of
    // removing it.
    bool refinedGroupReady = cluster.RefinedGroupIndex == NANITE_INVALID_ID ||
        ScreenErrorPixels < 0.0;
    if (!refinedGroupReady)
    {
        NaniteClusterGroup refinedGroup = g_ClusterGroups[cluster.RefinedGroupIndex];
        refinedGroupReady =
            ProjectedErrorPixels(refinedGroup.BoundSphere, refinedGroup.ParentError, instance.WorldMatrix) <=
            ScreenErrorPixels;
    }
    if (!refinedGroupReady)
    {
        uint refinementIgnored;
        InterlockedAdd(g_QueueState[0].ClusterRefinementRejected, 1u, refinementIgnored);
        return;
    }

    float projectedRadius = ProjectedSphereRadiusPixels(cluster.BoundSphere, instance.WorldMatrix);
    if (ScreenErrorPixels >= 0.0 && projectedRadius < 0.25)
    {
        uint tinyIgnored;
        InterlockedAdd(g_QueueState[0].ClusterTinyRejected, 1u, tinyIgnored);
        return;
    }

    uint ignored;
    if (!boundsVisible)
    {
        InterlockedAdd(g_QueueState[0].ClusterHZBRejected, 1u, ignored);
        // In the recovery phase the HZB is the current frame's, so a rejection
        // here is final and there is nowhere left to queue it.
        if (CullingPass == NANITE_CULLING_PASS_MAIN && frustumVisible && !hzbVisible)
        {
            uint postWriteIndex;
            InterlockedAdd(g_QueueState[0].PostClusterWrite, 1u, postWriteIndex);
            if (postWriteIndex < MaxClusterTasks)
            {
                NaniteClusterTask postTask;
                postTask.InstanceIndex = task.InstanceIndex;
                postTask.ClusterIndex = task.ClusterIndex;
                g_PostClusterQueue[postWriteIndex] = postTask;
                DeviceMemoryBarrier();
                InterlockedAdd(g_QueueState[0].PostClusterCount, 1u, ignored);
            }
            else
            {
                InterlockedOr(g_QueueState[0].OverflowFlags, 0x20u);
            }
        }
        if (debugTarget)
            InterlockedOr(g_QueueState[0].OverflowFlags, 0x00080000u);
        return;
    }
    if (debugTarget)
    {
        InterlockedOr(g_QueueState[0].OverflowFlags, 0x00100000u);
        float directLoad = g_HZB.Load(int3(1283, 682, 0));
        float directSample = g_HZB.SampleLevel(
            g_HZBSampler,
            (float2(1283.5, 682.5) / float2(HZBWidth, HZBHeight)),
            0.0);
        if (directLoad >= 0.9999)
            InterlockedOr(g_QueueState[0].OverflowFlags, 0x01000000u);
        else
            InterlockedOr(g_QueueState[0].OverflowFlags, 0x02000000u);
        if (directSample >= 0.9999)
            InterlockedOr(g_QueueState[0].OverflowFlags, 0x04000000u);
        else
            InterlockedOr(g_QueueState[0].OverflowFlags, 0x08000000u);
        if (hzbDepth >= 0.9999)
            InterlockedOr(g_QueueState[0].OverflowFlags, 0x00200000u);
        else
            InterlockedOr(g_QueueState[0].OverflowFlags, 0x00400000u);
        InterlockedOr(g_QueueState[0].OverflowFlags, 0x00800000u << hzbMip);
        uint debugIgnored;
        InterlockedExchange(
            g_QueueState[0].DebugProbeX,
            (hzbTexelMin.x & 0xFFFFu) | (hzbTexelMax.x << 16u),
            debugIgnored);
        InterlockedExchange(
            g_QueueState[0].DebugProbeY,
            (hzbTexelMin.y & 0xFFFFu) | (hzbTexelMax.y << 16u),
            debugIgnored);
    }

    uint drawIndex;
    g_DrawCommand.InterlockedAdd(NANITE_DRAW_INSTANCE_COUNT_OFFSET, 1u, drawIndex);
    if (drawIndex >= MaxDrawCommands)
    {
        // MaxDrawCommands is now maxInstances * clusters, and a cluster reaches
        // this point at most once per instance per frame, so this cannot fire.
        // It stays because InstanceCount has already been incremented past the
        // capacity of g_DrawInstanceData if it ever does.
        InterlockedOr(g_QueueState[0].OverflowFlags, 8u);
        return;
    }

    NaniteDrawInstanceData instanceData;
    instanceData.InstanceIndex = task.InstanceIndex;
    instanceData.ClusterIndex = task.ClusterIndex;
    instanceData.FirstIndex = cluster.FirstIndex;
    instanceData.IndexCount = cluster.IndexCount;
    g_DrawInstanceData[drawIndex] = instanceData;
    InterlockedMax(g_QueueState[0].DrawCount, drawIndex + 1u, ignored);
}
