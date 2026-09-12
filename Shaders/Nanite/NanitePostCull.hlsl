#include "NaniteCommon.hlsl"

StructuredBuffer<NaniteInstance> g_Instances : register(t1);
StructuredBuffer<NaniteCluster> g_Clusters : register(t2);
StructuredBuffer<NaniteClusterTask> g_PostClusterQueue : register(t3);
Texture2D<float> g_HZB : register(t4);
SamplerState g_HZBSampler : register(s8);

globallycoherent RWByteAddressBuffer g_DrawCommand : register(u16);
RWStructuredBuffer<NaniteQueueState> g_QueueState : register(u17);
RWStructuredBuffer<NaniteDrawInstanceData> g_DrawInstanceData : register(u19);

// Byte offsets inside the two-command draw buffer. Command 0 is the main raster
// phase's and command 1 is this pass', so the post fields sit one 16-byte
// command further in.
#define NANITE_DRAW_INSTANCE_COUNT_OFFSET 4u
#define NANITE_POST_DRAW_INSTANCE_COUNT_OFFSET 20u
#define NANITE_POST_DRAW_FIRST_INSTANCE_OFFSET 28u

[numthreads(NANITE_GROUP_SIZE, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint postTaskIndex = dispatchThreadID.x;
    uint postTaskCount = g_QueueState[0].PostClusterCount;
    if (dispatchThreadID.x == 0u && postTaskCount > g_QueueState[0].DispatchedPostClusterTasks)
    {
        // NanitePrepareDispatch recorded how far the grid it wrote can reach, so a
        // larger count means the tail of the recovery queue is outside the grid.
        InterlockedOr(g_QueueState[0].OverflowFlags, NANITE_OVERFLOW_DISPATCH_TOO_SMALL);
    }
    if (postTaskIndex >= postTaskCount)
        return;

    NaniteClusterTask task = g_PostClusterQueue[postTaskIndex];
    NaniteInstance instance = g_Instances[task.InstanceIndex];
    NaniteCluster cluster = g_Clusters[task.ClusterIndex];

    float4 clipMin;
    float4 clipMax;
    bool clipValid;
    bool frustumVisible = BBoxIntersectFrustum(
        cluster.BBoxMin,
        cluster.BBoxMax,
        instance.WorldMatrix,
        clipMin,
        clipMax,
        clipValid);
    if (!frustumVisible)
        return;

    bool visible = true;
    if (EnableHZB != 0u && clipValid)
    {
        float hzbDepth;
        uint hzbMip;
        uint2 hzbTexelMin;
        uint2 hzbTexelMax;
        visible = HZBVisible(
            g_HZB,
            g_HZBSampler,
            clipMin,
            clipMax,
            hzbDepth,
            hzbMip,
            hzbTexelMin,
            hzbTexelMax);
    }
    if (!visible)
    {
        uint ignored;
        InterlockedAdd(g_QueueState[0].PostClusterHZBRejected, 1u, ignored);
        return;
    }

    uint ignored;
    uint drawIndex;
    // Slots still come out of the main command's counter, so g_DrawInstanceData
    // stays one array the vertex shader can index with SV_InstanceID directly.
    g_DrawCommand.InterlockedAdd(NANITE_DRAW_INSTANCE_COUNT_OFFSET, 1u, drawIndex);
    if (drawIndex >= MaxDrawCommands)
    {
        InterlockedOr(g_QueueState[0].OverflowFlags, 0x20u);
        return;
    }

    NaniteDrawInstanceData instanceData;
    instanceData.InstanceIndex = task.InstanceIndex;
    instanceData.ClusterIndex = task.ClusterIndex;
    instanceData.FirstIndex = cluster.FirstIndex;
    instanceData.IndexCount = cluster.IndexCount;
    g_DrawInstanceData[drawIndex] = instanceData;
    InterlockedMax(g_QueueState[0].DrawCount, drawIndex + 1u, ignored);

    // Describe the recovered clusters as their own draw range so the post raster
    // pass rasterizes only them. By now this shader is the counter's only writer,
    // so the slots it claims are contiguous and the lowest one plus how many were
    // claimed pin the range exactly. Vulkan folds FirstInstance into
    // SV_InstanceID, so the range needs no base offset on the shader side.
    g_DrawCommand.InterlockedMin(NANITE_POST_DRAW_FIRST_INSTANCE_OFFSET, drawIndex, ignored);
    g_DrawCommand.InterlockedAdd(NANITE_POST_DRAW_INSTANCE_COUNT_OFFSET, 1u, ignored);
}
