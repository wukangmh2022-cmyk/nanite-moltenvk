#include "NaniteCommon.hlsl"

// Keep the depth image in the first texture slot, matching the slot used by
// the HZB build shader on the MoltenVK path. The C++ side binds by name.
Texture2D<float> g_HZB : register(t1);
StructuredBuffer<NaniteInstance> g_Instances : register(t2);
SamplerState g_HZBSampler : register(s8);

RWStructuredBuffer<NaniteNodeTask> g_NodeQueue : register(u16);
globallycoherent RWStructuredBuffer<NaniteQueueState> g_QueueState : register(u17);
// Instances the main phase rejected on occlusion alone. Declared as a UAV in
// both phases because the main phase writes it and the recovery phase reads it,
// and Diligent tracks one state per buffer, so it cannot also be bound as an SRV.
globallycoherent RWStructuredBuffer<uint> g_PostInstanceQueue : register(u18);

[numthreads(NANITE_GROUP_SIZE, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint instanceIndex = dispatchThreadID.x;
    if (CullingPass == NANITE_CULLING_PASS_POST)
    {
        // The recovery phase walks the queue the main phase filled, not the
        // whole scene. Frustum visibility was already established there.
        if (instanceIndex >= g_QueueState[0].PostInstanceCount)
            return;
        instanceIndex = g_PostInstanceQueue[instanceIndex];
    }
    else if (instanceIndex >= InstanceCount)
    {
        return;
    }

    NaniteInstance instance = g_Instances[instanceIndex];
    bool frustumVisible;
    bool hzbVisible;
    float hzbDepth;
    uint hzbMip;
    uint2 hzbTexelMin;
    uint2 hzbTexelMax;
    bool visible;
    if ((MaxRefinementDepth & 0x10000000u) != 0u)
    {
        float4 clipMin;
        float4 clipMax;
        bool clipValid;
        frustumVisible = BBoxIntersectFrustum(
            instance.BBoxMin,
            instance.BBoxMax,
            instance.WorldMatrix,
            clipMin,
            clipMax,
            clipValid);
        hzbVisible = true;
        hzbDepth = 1.0;
        hzbMip = 0u;
        hzbTexelMin = 0u;
        hzbTexelMax = 0u;
        visible = frustumVisible;
    }
    else
    {
        visible = EvaluateBoundsVisible(
            instance.BBoxMin,
            instance.BBoxMax,
            instance.WorldMatrix,
            g_HZB,
            g_HZBSampler,
            frustumVisible,
            hzbVisible,
            hzbDepth,
            hzbMip,
            hzbTexelMin,
            hzbTexelMax);
    }
    if (instanceIndex == 0u && (MaxRefinementDepth & 0x80000000u) != 0u)
    {
        uint ignored;
        // This coordinate is also read back on the CPU from HZB mip 0. It
        // separates a bad HZB resource binding from a bad bounds loop.
        float directLoad = g_HZB.Load(int3(1283, 682, 0));
        InterlockedExchange(g_QueueState[0].DebugProbeX, asuint(directLoad), ignored);
        float directSample = g_HZB.SampleLevel(
            g_HZBSampler,
            float2(1283.5 / 2560.0, 682.5 / 1440.0),
            0.0);
        InterlockedExchange(
            g_QueueState[0].DebugProbeY,
            asuint(directSample),
            ignored);
    }
    if (!visible)
    {
        // A frustum rejection is final: the bounds are static, so no later pass
        // can bring them back this frame. An occlusion rejection is only a guess
        // made from the previous frame's depth, so it has to be re-testable
        // against this frame's HZB or the whole object pops out for a frame.
        if (CullingPass == NANITE_CULLING_PASS_MAIN && frustumVisible && !hzbVisible)
        {
            uint postWriteIndex;
            InterlockedAdd(g_QueueState[0].PostInstanceWrite, 1u, postWriteIndex);
            if (postWriteIndex < InstanceCount)
            {
                g_PostInstanceQueue[postWriteIndex] = instanceIndex;
                DeviceMemoryBarrier();
                uint postIgnored;
                InterlockedAdd(g_QueueState[0].PostInstanceCount, 1u, postIgnored);
            }
            else
            {
                InterlockedOr(g_QueueState[0].OverflowFlags, 0x40u);
            }
        }
        return;
    }

    if (CullingPass == NANITE_CULLING_PASS_POST)
    {
        uint recoveredIgnored;
        InterlockedAdd(g_QueueState[0].PostInstanceRecovered, 1u, recoveredIgnored);
    }

    uint ignored;
    InterlockedAdd(g_QueueState[0].VisibleInstanceCount, 1u, ignored);

    uint writeIndex;
    InterlockedAdd(g_QueueState[0].NodeWrite, 1u, writeIndex);
    if (writeIndex >= MaxNodeTasks)
    {
        InterlockedOr(g_QueueState[0].OverflowFlags, 1u);
        return;
    }

    InterlockedAdd(g_QueueState[0].NodePending, 1u, ignored);

    NaniteNodeTask task;
    task.InstanceIndex = instanceIndex;
    task.NodeIndex = instance.RootNodeIndex;
    task.Ready = QueueGeneration;
    task.Padding0 = 0u;
    g_NodeQueue[writeIndex] = task;
    DeviceMemoryBarrier();
}
