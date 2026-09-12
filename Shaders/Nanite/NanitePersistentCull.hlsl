#include "NaniteCommon.hlsl"

StructuredBuffer<NaniteDagNode> g_Nodes : register(t1);
StructuredBuffer<NaniteClusterGroup> g_ClusterGroups : register(t2);
StructuredBuffer<NaniteInstance> g_Instances : register(t3);
Texture2D<float> g_HZB : register(t4);
StructuredBuffer<NaniteCluster> g_Clusters : register(t5);
SamplerState g_HZBSampler : register(s8);

globallycoherent RWStructuredBuffer<NaniteNodeTask> g_NodeQueue : register(u16);
globallycoherent RWStructuredBuffer<NaniteGroupTask> g_GroupQueue : register(u17);
globallycoherent RWStructuredBuffer<NaniteClusterTask> g_ClusterQueue : register(u18);
globallycoherent RWStructuredBuffer<NaniteQueueState> g_QueueState : register(u19);
// Occlusion-recovery queues, mirroring g_PostClusterQueue one level up. Bound as
// UAVs in both phases: the main phase appends, the seed pass reads them back.
globallycoherent RWStructuredBuffer<NanitePostNodeTask> g_PostNodeQueue : register(u20);
globallycoherent RWStructuredBuffer<NanitePostGroupTask> g_PostGroupQueue : register(u21);

// Every queue cursor and counter lives in the same g_QueueState[0] cache line, so
// one device atomic per lane means a whole wave of atomics serialized on one
// address. These helpers collapse that to a single atomic per wave.
//
// They are deliberately built from ballot counts only. Metal leaves simd_sum,
// simd_min and simd_prefix_exclusive_sum undefined unless every thread of the
// SIMD group is active, and these are called from divergent control flow (only
// the lanes that claimed a task, only the lanes that rejected a group), so
// WaveActiveSum / WavePrefixSum / WaveActiveMin cannot be used here. The
// ballot-based ones are defined over the active lanes.

// Reserves one slot per active lane in a monotonic cursor.
#define NANITE_WAVE_ALLOCATE_ONE(field, outIndex)                              \
    {                                                                          \
        uint waveCount_ = WaveActiveCountBits(true);                           \
        uint wavePrefix_ = WavePrefixCountBits(true);                          \
        uint waveBase_ = 0u;                                                   \
        if (WaveIsFirstLane())                                                 \
            InterlockedAdd(g_QueueState[0].field, waveCount_, waveBase_);       \
        outIndex = WaveReadLaneFirst(waveBase_) + wavePrefix_;                  \
    }

// Reads a queue counter written by other workgroups. Metal makes no promise that
// a plain load of device memory ever sees another core's write: the line can sit
// in this core's cache indefinitely, so a spin on a plain load never ends. An
// atomic that adds nothing forces the access out to memory. One atomic per wave,
// broadcast to its lanes.
#define NANITE_WAVE_READ(field, outValue)                                      \
    {                                                                          \
        uint waveValue_ = 0u;                                                  \
        if (WaveIsFirstLane())                                                 \
            InterlockedAdd(g_QueueState[0].field, 0u, waveValue_);             \
        outValue = WaveReadLaneFirst(waveValue_);                              \
    }

// Adds the number of active lanes for which the predicate holds.
#define NANITE_WAVE_COUNT(field, predicate)                                    \
    {                                                                          \
        uint waveCount_ = WaveActiveCountBits(predicate);                      \
        if (WaveIsFirstLane() && waveCount_ != 0u)                             \
        {                                                                      \
            uint waveIgnored_;                                                 \
            InterlockedAdd(g_QueueState[0].field, waveCount_, waveIgnored_);    \
        }                                                                      \
    }

// Subtracts the number of lanes that reached this point.
#define NANITE_WAVE_RELEASE(field)                                             \
    {                                                                          \
        uint waveCount_ = WaveActiveCountBits(true);                           \
        if (WaveIsFirstLane())                                                 \
        {                                                                      \
            uint waveIgnored_;                                                 \
            InterlockedAdd(g_QueueState[0].field, ~waveCount_ + 1u, waveIgnored_); \
        }                                                                      \
    }

// A node or group the main phase rejected on occlusion alone. Rejecting a node
// prunes its whole subtree, which is where the traversal cost actually is, but
// it is only safe if the subtree can come back when this frame's HZB disagrees.
void PushPostNodeTask(NaniteNodeTask task)
{
    uint writeIndex;
    NANITE_WAVE_ALLOCATE_ONE(PostNodeWrite, writeIndex);
    if (writeIndex >= MaxNodeTasks)
    {
        InterlockedOr(g_QueueState[0].OverflowFlags, 0x80u);
        return;
    }
    g_PostNodeQueue[writeIndex] = NaniteToPostNodeTask(task);
    DeviceMemoryBarrier();
    NANITE_WAVE_COUNT(PostNodeCount, true);
}

void PushPostGroupTask(NaniteGroupTask task)
{
    uint writeIndex;
    NANITE_WAVE_ALLOCATE_ONE(PostGroupWrite, writeIndex);
    if (writeIndex >= MaxGroupTasks)
    {
        InterlockedOr(g_QueueState[0].OverflowFlags, 0x100u);
        return;
    }
    g_PostGroupQueue[writeIndex] = NaniteToPostGroupTask(task);
    DeviceMemoryBarrier();
    NANITE_WAVE_COUNT(PostGroupCount, true);
}

// One CAS per wave claims a run of tasks for all of its lanes at once. With a CAS
// per lane every lane fought over the same cursor word and all but one attempt
// failed, which is where this pass spent most of its time. The run is trimmed to
// the slots whose payload is already published, because producers bump the write
// cursor before storing the task.
// One CAS per wave claims a run of tasks for all of its lanes at once. With a CAS
// per lane every lane fought over the same cursor word and all but one attempt
// failed, which is where this pass spent most of its time. The run is only taken
// when every slot in it is already published, because producers bump the write
// cursor before storing the task; otherwise the wave retries, which is the same
// conservative rule the per-lane version applied to the head slot.
bool TryClaimNode(out uint taskIndex)
{
    taskIndex = 0u;
    [loop]
    for (uint retry = 0u; retry < 16u; ++retry)
    {
        uint wantCount = WaveActiveCountBits(true);
        uint wantPrefix = WavePrefixCountBits(true);

        uint readIndex = g_QueueState[0].NodeRead;
        uint limit = min(g_QueueState[0].NodeWrite, MaxNodeTasks);
        if (readIndex >= limit)
            return false;

        uint batch = min(wantCount, limit - readIndex);
        bool slotReady = wantPrefix < batch &&
            g_NodeQueue[readIndex + wantPrefix].Ready == QueueGeneration;
        if (WaveActiveCountBits(slotReady) != batch)
            return false;

        uint original = readIndex;
        if (WaveIsFirstLane())
            InterlockedCompareExchange(g_QueueState[0].NodeRead, readIndex, readIndex + batch, original);
        if (WaveReadLaneFirst(original) != readIndex)
            continue;

        if (wantPrefix < batch)
        {
            taskIndex = readIndex + wantPrefix;
            return true;
        }
        // Fewer tasks than the wave wanted; the lanes left empty retry.
    }
    return false;
}

bool TryClaimGroup(out uint taskIndex)
{
    taskIndex = 0u;
    [loop]
    for (uint retry = 0u; retry < 16u; ++retry)
    {
        uint wantCount = WaveActiveCountBits(true);
        uint wantPrefix = WavePrefixCountBits(true);

        uint readIndex = g_QueueState[0].GroupRead;
        uint limit = min(g_QueueState[0].GroupWrite, MaxGroupTasks);
        if (readIndex >= limit)
            return false;

        uint batch = min(wantCount, limit - readIndex);
        bool slotReady = wantPrefix < batch &&
            g_GroupQueue[readIndex + wantPrefix].Ready == QueueGeneration;
        if (WaveActiveCountBits(slotReady) != batch)
            return false;

        uint original = readIndex;
        if (WaveIsFirstLane())
            InterlockedCompareExchange(g_QueueState[0].GroupRead, readIndex, readIndex + batch, original);
        if (WaveReadLaneFirst(original) != readIndex)
            continue;

        if (wantPrefix < batch)
        {
            taskIndex = readIndex + wantPrefix;
            return true;
        }
    }
    return false;
}

void PushGroupTask(uint instanceIndex, uint groupIndex, uint refinementDepth)
{
    uint writeIndex;
    NANITE_WAVE_ALLOCATE_ONE(GroupWrite, writeIndex);
    if (writeIndex >= MaxGroupTasks)
    {
        InterlockedOr(g_QueueState[0].OverflowFlags, 2u);
        return;
    }

    NANITE_WAVE_COUNT(GroupPending, true);

    NaniteGroupTask task;
    task.InstanceIndex = instanceIndex;
    task.GroupIndex = groupIndex;
    task.RefinementDepth = refinementDepth;
    task.Ready = QueueGeneration;
    g_GroupQueue[writeIndex] = task;
    DeviceMemoryBarrier();
}

void PushNodeChildren(NaniteNodeTask task, NaniteDagNode node)
{
    uint childCount = min(NodeChildCount(node.NodeData), 8u);
    if (childCount == 0u)
        return;

    // The per-lane child counts differ, so the wave total is accumulated from
    // ballots over each possible count rather than a simd_sum, which Metal does
    // not define for a partially active SIMD group. Eight ballots are register
    // operations; the device atomic they replace is not.
    uint waveTotal = 0u;
    uint wavePrefix = 0u;
    [unroll]
    for (uint size = 1u; size <= 8u; ++size)
    {
        waveTotal += size * WaveActiveCountBits(childCount == size);
        wavePrefix += size * WavePrefixCountBits(childCount == size);
    }
    uint waveBase = 0u;
    if (WaveIsFirstLane())
        InterlockedAdd(g_QueueState[0].NodeWrite, waveTotal, waveBase);
    uint writeIndex = WaveReadLaneFirst(waveBase) + wavePrefix;

    uint available = writeIndex < MaxNodeTasks ? MaxNodeTasks - writeIndex : 0u;
    uint acceptedCount = min(childCount, available);
    // Same ballot trick for the pending counter.
    uint wavePending = 0u;
    [unroll]
    for (uint accepted = 1u; accepted <= 8u; ++accepted)
        wavePending += accepted * WaveActiveCountBits(acceptedCount == accepted);
    if (WaveIsFirstLane() && wavePending != 0u)
    {
        uint pendingIgnored;
        InterlockedAdd(g_QueueState[0].NodePending, wavePending, pendingIgnored);
    }

    for (uint child = 0u; child < acceptedCount; ++child)
    {
        NaniteNodeTask childTask;
        childTask.InstanceIndex = task.InstanceIndex;
        childTask.NodeIndex = NodeChildStart(node.NodeData) + child;
        childTask.Ready = QueueGeneration;
        childTask.Padding0 = 0u;
        g_NodeQueue[writeIndex + child] = childTask;
    }
    if (acceptedCount != childCount)
        InterlockedOr(g_QueueState[0].OverflowFlags, 1u);

    DeviceMemoryBarrier();
}

// The other half of the LOD cut. A cluster is only drawn once the finer group it
// replaced has itself become unnecessary, and generate-indirect used to be the
// first place that got tested - after a full HZB sample per cluster. Evaluating
// it here keeps those clusters out of the cluster queue entirely. This is a copy
// of the test in NaniteGenerateIndirect.hlsl, not an approximation of it, so the
// set of drawn clusters does not change.
bool ClusterRefinedGroupReady(uint clusterIndex, NaniteInstance instance)
{
    if (ScreenErrorPixels < 0.0)
        return true;
    uint refinedGroupIndex = g_Clusters[clusterIndex].RefinedGroupIndex;
    if (refinedGroupIndex == NANITE_INVALID_ID)
        return true;
    NaniteClusterGroup refinedGroup = g_ClusterGroups[refinedGroupIndex];
    return ProjectedErrorPixels(refinedGroup.BoundSphere, refinedGroup.ParentError, instance.WorldMatrix) <=
        ScreenErrorPixels;
}

void PushClusterGroupClusters(NaniteGroupTask task, NaniteClusterGroup group, NaniteInstance instance)
{
    // Two passes over the group's clusters. The first only counts the survivors,
    // so the queue reservation stays one atomic for the whole group: reserving
    // before testing would leave holes in the queue, and reserving per surviving
    // cluster would be one atomic per cluster. A group that loses all of its
    // clusters does not touch the cursor at all.
    uint acceptedCount = 0u;
    for (uint testOffset = 0u; testOffset < group.ClusterCount; ++testOffset)
    {
        if (ClusterRefinedGroupReady(group.ClusterStart + testOffset, instance))
            ++acceptedCount;
    }
    if (acceptedCount == 0u)
        return;

    // Cluster counts per group are unbounded here, so this allocation keeps its
    // per-lane atomic: there is one per visible group, not one per queue probe.
    uint writeIndex;
    InterlockedAdd(g_QueueState[0].ClusterWrite, acceptedCount, writeIndex);
    uint available = writeIndex < MaxClusterTasks ? MaxClusterTasks - writeIndex : 0u;
    uint writtenCount = 0u;
    for (uint clusterOffset = 0u;
         clusterOffset < group.ClusterCount && writtenCount < available;
         ++clusterOffset)
    {
        uint clusterIndex = group.ClusterStart + clusterOffset;
        if (!ClusterRefinedGroupReady(clusterIndex, instance))
            continue;
        NaniteClusterTask clusterTask;
        clusterTask.InstanceIndex = task.InstanceIndex;
        clusterTask.ClusterIndex = clusterIndex;
        g_ClusterQueue[writeIndex + writtenCount] = clusterTask;
        ++writtenCount;
    }
    if (writtenCount != acceptedCount)
        InterlockedOr(g_QueueState[0].OverflowFlags, 4u);

    DeviceMemoryBarrier();
    uint ignored;
    InterlockedAdd(g_QueueState[0].VisibleClusterCount, writtenCount, ignored);
}

// Visibility test for one node plus the push of whatever it expands into. Split
// out of ProcessNodeTask so the LOD prune in front of it stays readable.
void RefineNode(NaniteNodeTask task, NaniteInstance instance, NaniteDagNode node)
{
    bool visible;
    bool nodeFrustumVisible;
    bool nodeHZBVisible;
    if ((MaxRefinementDepth & 0x20000000u) != 0u)
    {
        float4 clipMin;
        float4 clipMax;
        bool clipValid;
        visible = BBoxIntersectFrustum(
            node.BBoxMin,
            node.BBoxMax,
            instance.WorldMatrix,
            clipMin,
            clipMax,
            clipValid);
        nodeFrustumVisible = visible;
        nodeHZBVisible = true;
    }
    else
    {
        float nodeHZBDepth;
        uint nodeHZBMip;
        uint2 nodeHZBTexelMin;
        uint2 nodeHZBTexelMax;
        visible = EvaluateBoundsVisible(
            node.BBoxMin,
            node.BBoxMax,
            instance.WorldMatrix,
            g_HZB,
            g_HZBSampler,
            nodeFrustumVisible,
            nodeHZBVisible,
            nodeHZBDepth,
            nodeHZBMip,
            nodeHZBTexelMin,
            nodeHZBTexelMax);
    }
    if (visible)
    {
        if (CullingPass == NANITE_CULLING_PASS_POST)
        {
            NANITE_WAVE_COUNT(PostNodeRecovered, true);
        }
        if (NodeIsGroup(node.NodeData))
        {
            PushGroupTask(task.InstanceIndex, NodeGroupIndex(node.NodeData), 0u);
        }
        else
        {
            PushNodeChildren(task, node);
        }
    }
    else if (CullingPass == NANITE_CULLING_PASS_MAIN &&
             nodeFrustumVisible && !nodeHZBVisible)
    {
        PushPostNodeTask(task);
    }
}

void ProcessNodeTask(uint taskIndex)
{
    NaniteNodeTask task = g_NodeQueue[taskIndex];
    NaniteInstance instance = g_Instances[task.InstanceIndex];
    NaniteDagNode node = g_Nodes[task.NodeIndex];

    // LOD prune before any visibility work. The node carries a bound that
    // dominates the group error test of everything below it, so once that bound
    // falls under the screen threshold no group in the subtree can emit clusters
    // and expanding the children is pure overhead. This is what makes node work
    // track resolvable detail instead of instances x nodes: a distant instance
    // stops at its root. The post queue is skipped along with it, because the post
    // pass runs the same test and would prune the node again.
    bool lodPruned = ScreenErrorPixels >= 0.0 &&
        ProjectedErrorPixels(node.LodSphere, node.MaxGroupError, instance.WorldMatrix) <=
            ScreenErrorPixels;
    NANITE_WAVE_COUNT(NodeLodRejected, lodPruned);
    if (!lodPruned)
        RefineNode(task, instance, node);

    DeviceMemoryBarrier();
    NANITE_WAVE_RELEASE(NodePending);
}

void ProcessGroupTask(uint taskIndex)
{
    NaniteGroupTask task = g_GroupQueue[taskIndex];
    NaniteInstance instance = g_Instances[task.InstanceIndex];
    NaniteClusterGroup group = g_ClusterGroups[task.GroupIndex];

    bool groupFrustumVisible;
    bool groupHZBVisible;
    float groupHZBDepth;
    uint groupHZBMip;
    uint2 groupHZBTexelMin;
    uint2 groupHZBTexelMax;
    bool visible;
    if ((MaxRefinementDepth & 0x20000000u) != 0u)
    {
        float4 clipMin;
        float4 clipMax;
        bool clipValid;
        groupFrustumVisible = BBoxIntersectFrustum(
            group.BBoxMin,
            group.BBoxMax,
            instance.WorldMatrix,
            clipMin,
            clipMax,
            clipValid);
        groupHZBVisible = true;
        groupHZBDepth = 1.0;
        groupHZBMip = 0u;
        groupHZBTexelMin = 0u;
        groupHZBTexelMax = 0u;
        visible = groupFrustumVisible;
    }
    else
    {
        visible = EvaluateBoundsVisible(
            group.BBoxMin,
            group.BBoxMax,
            instance.WorldMatrix,
            g_HZB,
            g_HZBSampler,
            groupFrustumVisible,
            groupHZBVisible,
            groupHZBDepth,
            groupHZBMip,
            groupHZBTexelMin,
            groupHZBTexelMax);
    }
    // Both counters are diagnostics, so they are folded into one atomic per wave
    // each rather than one per lane.
    NANITE_WAVE_COUNT(GroupFrustumRejected, !groupFrustumVisible);
    NANITE_WAVE_COUNT(GroupHZBRejected, groupFrustumVisible && !groupHZBVisible);
    if (groupFrustumVisible && !groupHZBVisible && CullingPass == NANITE_CULLING_PASS_MAIN)
    {
        PushPostGroupTask(task);
    }
    if (visible &&
        (ScreenErrorPixels < 0.0 ||
         ProjectedErrorPixels(group.BoundSphere, group.ParentError, instance.WorldMatrix) > ScreenErrorPixels))
    {
        if (CullingPass == NANITE_CULLING_PASS_POST)
        {
            NANITE_WAVE_COUNT(PostGroupRecovered, true);
        }
        PushClusterGroupClusters(task, group, instance);
    }

    DeviceMemoryBarrier();
    NANITE_WAVE_RELEASE(GroupPending);
}

[numthreads(NANITE_GROUP_SIZE, 1, 1)]
void main(uint3 groupID : SV_GroupID, uint groupIndex : SV_GroupIndex)
{
    // All lanes independently claim work. This is intentionally persistent:
    // child tasks can be consumed by the same dispatch without CPU round trips.
    // The budget comes from the CPU, sized from the worst-case task count with
    // slack for the iterations burnt spinning while producers catch up. It is a
    // hang guard, not a work limit: the loop normally exits on the pending test.
    for (uint iteration = 0u; iteration < MaxTraversalIterations; ++iteration)
    {
        uint taskIndex;
        if (TryClaimNode(taskIndex))
        {
            ProcessNodeTask(taskIndex);
            continue;
        }

        if (TryClaimGroup(taskIndex))
        {
            ProcessGroupTask(taskIndex);
            continue;
        }

        uint nodePending;
        uint groupPending;
        NANITE_WAVE_READ(NodePending, nodePending);
        NANITE_WAVE_READ(GroupPending, groupPending);
        if (nodePending == 0u && groupPending == 0u)
            break;
    }

    // Reaching the iteration cap while still busy is the failure that matters:
    // the leftover tasks never emit clusters, so geometry silently disappears and
    // reappears as the cut moves. The old check sat at the bottom of the loop
    // body, where a lane that was doing useful work on the last iteration
    // `continue`d straight past it and the flag was never raised.
    uint exitNodePending;
    uint exitGroupPending;
    NANITE_WAVE_READ(NodePending, exitNodePending);
    NANITE_WAVE_READ(GroupPending, exitGroupPending);
    if (exitNodePending != 0u || exitGroupPending != 0u)
        InterlockedOr(g_QueueState[0].OverflowFlags, 16u);
}
