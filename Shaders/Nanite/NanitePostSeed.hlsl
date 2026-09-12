#include "NaniteCommon.hlsl"

// Seeds the occlusion recovery phase. The main phase left the traversal queues
// drained (read cursor == write cursor) and recorded everything it rejected on
// occlusion alone in the post queues. Appending those tasks back onto the live
// queues lets the unmodified persistent traversal re-run over exactly the
// pruned subtrees, this time against the HZB built from the current frame's
// depth. Appending rather than resetting keeps the main phase's cluster tasks
// and statistics intact.

globallycoherent RWStructuredBuffer<NaniteNodeTask> g_NodeQueue : register(u16);
globallycoherent RWStructuredBuffer<NaniteGroupTask> g_GroupQueue : register(u17);
globallycoherent RWStructuredBuffer<NaniteQueueState> g_QueueState : register(u18);
globallycoherent RWStructuredBuffer<NanitePostNodeTask> g_PostNodeQueue : register(u19);
globallycoherent RWStructuredBuffer<NanitePostGroupTask> g_PostGroupQueue : register(u20);

[numthreads(NANITE_GROUP_SIZE, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint index = dispatchThreadID.x;
    if (index == 0u)
    {
        // Mark where the main phase stopped. The recovery phase appends its
        // cluster tasks after this point and generate-indirect only walks the
        // tail, so already rasterized clusters are not emitted twice.
        uint ignored;
        InterlockedExchange(
            g_QueueState[0].MainClusterCount,
            g_QueueState[0].VisibleClusterCount,
            ignored);

        // How far the grid reaches was recorded by NanitePrepareDispatch. Either
        // recovery counter exceeding it would leave tasks unseeded, and they would
        // vanish for a frame.
        uint seedTasks = max(g_QueueState[0].PostNodeCount, g_QueueState[0].PostGroupCount);
        if (seedTasks > g_QueueState[0].DispatchedSeedTasks)
            InterlockedOr(g_QueueState[0].OverflowFlags, NANITE_OVERFLOW_DISPATCH_TOO_SMALL);
    }

    if (index < g_QueueState[0].PostNodeCount)
    {
        NaniteNodeTask task = NaniteFromPostNodeTask(g_PostNodeQueue[index]);
        task.Ready = QueueGeneration;
        uint writeIndex;
        InterlockedAdd(g_QueueState[0].NodeWrite, 1u, writeIndex);
        if (writeIndex < MaxNodeTasks)
        {
            g_NodeQueue[writeIndex] = task;
            DeviceMemoryBarrier();
            uint ignored;
            InterlockedAdd(g_QueueState[0].NodePending, 1u, ignored);
        }
        else
        {
            InterlockedOr(g_QueueState[0].OverflowFlags, 1u);
        }
    }

    if (index < g_QueueState[0].PostGroupCount)
    {
        NaniteGroupTask task = NaniteFromPostGroupTask(g_PostGroupQueue[index]);
        task.Ready = QueueGeneration;
        uint writeIndex;
        InterlockedAdd(g_QueueState[0].GroupWrite, 1u, writeIndex);
        if (writeIndex < MaxGroupTasks)
        {
            g_GroupQueue[writeIndex] = task;
            DeviceMemoryBarrier();
            uint ignored;
            InterlockedAdd(g_QueueState[0].GroupPending, 1u, ignored);
        }
        else
        {
            InterlockedOr(g_QueueState[0].OverflowFlags, 2u);
        }
    }
}
