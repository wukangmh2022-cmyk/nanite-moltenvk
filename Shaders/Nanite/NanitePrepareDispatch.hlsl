#include "NaniteCommon.hlsl"

// Sizes the dispatch grids of every pass whose work count only exists on the
// GPU. Those passes used to be dispatched over the full provisioned capacity -
// MaxClusterTasks is tens of millions once the scene is dense - so almost every
// workgroup did nothing but fail its bounds check. Turning the counters the
// traversal already wrote into vkCmdDispatchIndirect arguments makes each grid
// match the work that actually exists, without capping it by a CPU-side number
// that would silently drop clusters.
//
// One thread, and it recomputes all four slots on every invocation, so a slot
// has exactly one meaning no matter which invocation last touched it. A slot
// whose inputs are not final yet is simply overwritten by the invocation that
// runs before the pass reading it.

globallycoherent RWByteAddressBuffer g_DispatchArgs : register(u16);
globallycoherent RWStructuredBuffer<NaniteQueueState> g_QueueState : register(u17);

// 16 bytes per slot: ThreadGroupCountX/Y/Z plus one word of padding, which keeps
// every indirect offset 16-byte aligned.
#define NANITE_DISPATCH_SLOT_GENERATE_INDIRECT 0u
#define NANITE_DISPATCH_SLOT_POST_SEED 1u
#define NANITE_DISPATCH_SLOT_POST_GENERATE_INDIRECT 2u
#define NANITE_DISPATCH_SLOT_POST_CULL 3u
#define NANITE_DISPATCH_SLOT_RASTER_BIN 4u
#define NANITE_DISPATCH_SLOT_SOFT_RASTER 5u

// Writes one slot and returns how many work items the resulting grid can cover,
// which is what the dispatched shader is told about. Reporting the coverage
// rather than the requested count means the check downstream also catches a
// mistake in this arithmetic, not only a counter that grew after the fact.
uint WriteSlot(uint slot, uint workItems)
{
    // Never zero workgroups. An empty grid launches no threads at all, which
    // would also disarm the undersized-grid check that the dispatched shaders
    // run in their first thread, and one idle workgroup costs nothing.
    uint groupCount = max((workItems + NANITE_GROUP_SIZE - 1u) / NANITE_GROUP_SIZE, 1u);
    g_DispatchArgs.Store3(slot * 16u, uint3(groupCount, 1u, 1u));
    return groupCount * NANITE_GROUP_SIZE;
}

// One workgroup per work item instead of one per NANITE_GROUP_SIZE of them, for
// a pass whose group consumes a single queue entry across all of its threads.
// The software rasterizer is the only such pass - it spreads one cluster's
// triangles over the threads of one group - so WriteSlot would launch it over a
// 64th of the clusters.
void WriteSlotPerItem(uint slot, uint workItems)
{
    g_DispatchArgs.Store3(slot * 16u, uint3(max(workItems, 1u), 1u, 1u));
}

[numthreads(1, 1, 1)]
void main()
{
    uint visibleClusters = g_QueueState[0].VisibleClusterCount;
    uint mainClusters = g_QueueState[0].MainClusterCount;
    uint postClusters = g_QueueState[0].PostClusterCount;
    uint seedTasks = max(g_QueueState[0].PostNodeCount, g_QueueState[0].PostGroupCount);

    uint mainCoverage = WriteSlot(NANITE_DISPATCH_SLOT_GENERATE_INDIRECT, visibleClusters);
    uint seedCoverage = WriteSlot(NANITE_DISPATCH_SLOT_POST_SEED, seedTasks);
    // The recovery traversal appends its cluster tasks after the main phase's
    // mark and post generate-indirect only walks that tail, so its grid covers
    // [MainClusterCount, VisibleClusterCount) rather than the whole queue.
    uint postCoverage = WriteSlot(
        NANITE_DISPATCH_SLOT_POST_GENERATE_INDIRECT,
        visibleClusters > mainClusters ? visibleClusters - mainClusters : 0u);
    uint postCullCoverage = WriteSlot(NANITE_DISPATCH_SLOT_POST_CULL, postClusters);
    // Binning runs after draw generation, but VisibleClusterCount remains its
    // conservative dispatch bound. A second invocation after binning rewrites
    // the software slot to the compacted cluster count. Probe mode deliberately
    // keeps the old full draw-list dispatch.
    WriteSlotPerItem(NANITE_DISPATCH_SLOT_RASTER_BIN, visibleClusters);
    const uint softwareClusters = HybridRasterEnabled != 0u ?
        g_QueueState[0].SoftClusterCount : visibleClusters;
    WriteSlotPerItem(NANITE_DISPATCH_SLOT_SOFT_RASTER, softwareClusters);

    // How many queue entries each grid can reach. The dispatched shader compares
    // the count it reads against this, so a grid that cannot cover its queue
    // raises an overflow flag instead of quietly dropping the tail. Both
    // generate-indirect passes share one field: the main pass walks
    // [0, VisibleClusterCount) and the post pass [MainClusterCount, ...), and
    // nothing after this point moves the mark, so the smaller of the two
    // coverages is the safe bound for either.
    g_QueueState[0].DispatchedClusterTasks = min(mainCoverage, mainClusters + postCoverage);
    g_QueueState[0].DispatchedSeedTasks = seedCoverage;
    g_QueueState[0].DispatchedPostClusterTasks = postCullCoverage;
}
