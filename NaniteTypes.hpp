#pragma once

#include <cstdint>

namespace Nanite
{

struct alignas(16) Instance
{
    float WorldMatrix[16];
    float LocalBoundSphere[4];
    float BBoxMin[3];
    float BBoxMinPadding;
    float BBoxMax[3];
    float BBoxMaxPadding;
    std::uint32_t RootNodeIndex;
    std::uint32_t Padding0;
    std::uint32_t Padding1;
    std::uint32_t Padding2;
};

// Material records consumed by the post-Visibility-Buffer PBR shader. The
// texture index is part of the record so imported material tables can select
// an albedo layer without changing the draw/visibility payload.
struct alignas(16) Material
{
    float BaseColor[4];
    float Metallic;
    float Roughness;
    float AmbientOcclusion;
    std::uint32_t TextureIndex;
};

struct alignas(16) DagNode
{
    // Conservative LOD bound of the whole subtree: a sphere that encloses every
    // cluster group below this node, and the largest ParentError among them.
    // ProjectedErrorPixels grows with the error and shrinks with the distance to
    // the sphere's surface, so evaluating it on this pair is never smaller than
    // evaluating it on any single group underneath. That makes it safe to skip
    // the entire subtree when it falls under the screen error threshold, which is
    // what keeps traversal cost proportional to resolvable detail instead of to
    // instances x nodes.
    float LodSphere[4];
    float BBoxMin[3];
    std::uint32_t BBoxMinPadding;
    float BBoxMax[3];
    float MaxGroupError;
    std::uint32_t NodeData;
    std::uint32_t Padding[3];
};

struct alignas(16) ClusterGroup
{
    float BoundSphere[4];
    float BBoxMin[3];
    std::uint32_t BBoxMinPadding;
    float BBoxMax[3];
    float ParentError;
    std::uint32_t ClusterStart;
    std::uint32_t ClusterCount;
    std::uint32_t RefineGroupIndex;
    std::uint32_t Padding0;
};

struct alignas(16) Cluster
{
    float BoundSphere[4];
    float BBoxMin[3];
    std::uint32_t BBoxMinPadding;
    float BBoxMax[3];
    float MaxError;
    std::uint32_t IndexCount;
    std::uint32_t FirstIndex;
    // Packed for the software rasterizer: high 8 bits are the cluster-local
    // vertex count, low 24 bits are the offset in CpuScene::ClusterPositions.
    // The field was previously unused by this demo, so the 64-byte cluster
    // layout remains unchanged.
    std::uint32_t VertexOffset;
    // Index of the more detailed group that generated this cluster, or invalid.
    std::uint32_t RefinedGroupIndex;
};

struct NodeTask
{
    std::uint32_t InstanceIndex;
    std::uint32_t NodeIndex;
    std::uint32_t Ready;
    std::uint32_t Padding0;
};

struct GroupTask
{
    std::uint32_t InstanceIndex;
    std::uint32_t GroupIndex;
    std::uint32_t RefinementDepth;
    std::uint32_t Ready;
};

struct ClusterTask
{
    std::uint32_t InstanceIndex;
    std::uint32_t ClusterIndex;
};

struct QueueState
{
    std::uint32_t NodeRead;
    std::uint32_t NodeWrite;
    std::uint32_t NodePending;
    std::uint32_t GroupRead;
    std::uint32_t GroupWrite;
    std::uint32_t GroupPending;
    std::uint32_t ClusterWrite;
    std::uint32_t VisibleClusterCount;
    std::uint32_t DrawCount;
    std::uint32_t OverflowFlags;
    std::uint32_t VisibleInstanceCount;
    std::uint32_t GroupFrustumRejected;
    std::uint32_t GroupHZBRejected;
    std::uint32_t ClusterFrustumRejected;
    std::uint32_t ClusterHZBRejected;
    std::uint32_t ClusterRefinementRejected;
    std::uint32_t ClusterTinyRejected;
    std::uint32_t PostClusterWrite;
    std::uint32_t PostClusterCount;
    std::uint32_t PostClusterHZBRejected;
    // Two-phase occlusion recovery. The main phase tests against the previous
    // frame's HZB, so anything it rejects has to be re-testable against the
    // current frame's HZB or it disappears for a frame. Clusters had a recovery
    // queue already; these mirror it at instance, node and group granularity so
    // the HZB can be applied before and during traversal instead of only at the
    // very end, which is the only place where the traversal cost can be saved.
    std::uint32_t PostInstanceWrite;
    std::uint32_t PostInstanceCount;
    std::uint32_t PostNodeWrite;
    std::uint32_t PostNodeCount;
    std::uint32_t PostGroupWrite;
    std::uint32_t PostGroupCount;
    // VisibleClusterCount at the moment the main phase finished. The post phase
    // appends its cluster tasks after that mark and only processes the tail, so
    // the main phase's tasks are neither re-rasterized nor overwritten.
    std::uint32_t MainClusterCount;
    std::uint32_t PostInstanceRecovered;
    std::uint32_t PostNodeRecovered;
    std::uint32_t PostGroupRecovered;
    std::uint32_t DebugProbeX;
    std::uint32_t DebugProbeY;
    // Nodes whose subtree LOD bound was already fine enough for the screen, so
    // neither they nor anything below them produced work. Printed next to the
    // node task count because the two together are what say whether the LOD cut
    // is doing its job.
    std::uint32_t NodeLodRejected;
    // How many queue entries each of the indirect grids NanitePrepareDispatch
    // wrote can actually reach. The dispatched shaders compare their queue count
    // against these and raise overflow flag 0x200 if the grid cannot cover it,
    // so an undersized grid is loud instead of a silent drop of the queue's tail.
    std::uint32_t DispatchedClusterTasks;
    std::uint32_t DispatchedSeedTasks;
    std::uint32_t DispatchedPostClusterTasks;
    // Software-raster probe. Triangles and pixels the compute rasterizer
    // actually covered, next to the two reasons it declined a triangle, so its
    // pass timing can be read as the price of a known amount of work rather than
    // of an unknown fraction of the frame. The depth counters are the
    // correctness half: matched and mismatched pixels against the hardware
    // depth, plus the pixels only one of the two covered.
    std::uint32_t SoftRasterTriangles;
    std::uint32_t SoftRasterPixels;
    std::uint32_t SoftRasterSkippedLarge;
    std::uint32_t SoftRasterSkippedClip;
    std::uint32_t SoftDepthMatched;
    std::uint32_t SoftDepthMismatch;
    std::uint32_t SoftDepthMissing;
    std::uint32_t SoftDepthExtra;
    // Largest absolute depth disagreement, scaled by 2^24-1.
    std::uint32_t SoftDepthMaxDiff;
    // Visibility-buffer half of the probe. The first two compare the depth half of
    // the 64-bit word against the independent 32-bit InterlockedMin; the rest
    // refetch the triangle its payload names and evaluate that triangle at the
    // pixel, so a payload that does not belong with the depth it arrived with is
    // counted rather than passing unnoticed.
    std::uint32_t SoftVisMissing;
    std::uint32_t SoftVisKeyMismatch;
    std::uint32_t SoftVisResolved;
    // NoCover is unambiguous tearing: the payload names a triangle that never
    // covered the pixel. DepthOff is the named triangle covering the pixel but
    // interpolating a different depth there, which is either tearing between
    // neighbours on one surface or the two shaders' arithmetic drifting apart.
    std::uint32_t SoftVisNoCover;
    std::uint32_t SoftVisDepthOff;
    // Number of compacted cluster indices consumed by the hybrid software
    // raster pass. Zero outside hybrid mode.
    std::uint32_t SoftClusterCount;
    // Number of clusters appended to the hardware bin by the same pass.
    std::uint32_t HardwareClusterCount;
    std::uint32_t SoftPadding2;
};

struct GpuTimings
{
    bool Enabled = false;
    bool Valid = false;
    // True only when software raster was submitted to a genuinely separate
    // compute queue. A second graphics+compute family may still serialize on
    // portability layers and must not be reported as parallel.
    bool AsyncRaster = false;
    // True when the Apple MSL override resolves depth and payload in one native
    // 64-bit atomic word; false means the portable two-pass fallback is active.
    bool Native64BitVisibility = false;
    float FrameMs = 0.0f;
    float InstanceCullMs = 0.0f;
    float PersistentCullMs = 0.0f;
    float GenerateIndirectMs = 0.0f;
    float PostCullMs = 0.0f;
    // Per-frame queue initialization. This is GPU work recorded before the
    // culling passes, so it belongs to FrameMs even though it is not a cull.
    float QueueResetMs = 0.0f;
    // The small compute dispatches that turn queue counters into indirect args.
    // Main/bin prepares are timed individually and accumulated here.
    float PrepareDispatchMs = 0.0f;
    // Recovery work after the current-frame HZB: seed, instance/node traversal,
    // post generate, and the recovery prepare dispatches.
    float PostRecoveryMs = 0.0f;
    float RasterMs = 0.0f;
    float PostRasterMs = 0.0f;
    float HizBuildMs = 0.0f;
    // Hybrid mode builds HZB once for post cull and once again after the final
    // visibility resolve. Keep the second build visible instead of hiding it in
    // the unaccounted remainder.
    float FinalHizBuildMs = 0.0f;
    // Two-pass software visibility raster time. Keep the depth and payload sweeps
    // separate: the latter is an implementation tax of the portable fallback
    // when a native 64-bit visibility atomic is unavailable.
    float SoftRasterMs = 0.0f;
    float SoftDepthMs = 0.0f;
    float SoftVisibilityMs = 0.0f;
    float RasterBinMs = 0.0f;
    float SoftCheckMs = 0.0f;
    // The hardware drawing the same clusters with no pixel shader and no colour
    // targets. This, not RasterMs, is what SoftRasterMs has to be compared with.
    float HardwareDepthOnlyMs = 0.0f;
    // Directional shadow-map clear, depth-only cluster draw, and read transition.
    float ShadowMapMs = 0.0f;
    // Full-screen visibility resolve can run once before post cull and once after
    // post raster. This is the sum of both dispatches, not a queue-work sum.
    float VisibilityResolveMs = 0.0f;
    // The dedicated fullscreen material/visibility shading pass. Hardware raster
    // and software raster do not perform material shading anymore.
    float ShadingMs = 0.0f;
};

struct FrameStats
{
    std::uint64_t LogicalTriangleCount = 0;
    std::uint32_t InstanceCount = 0;
    std::uint32_t VisibleInstanceCount = 0;
    std::uint32_t NodeTaskCount = 0;
    std::uint32_t GroupTaskCount = 0;
    // Write cursors and the outstanding-task counters. Compared against the read
    // cursors these say whether the persistent dispatch drained the queues or
    // gave up with work still in them.
    std::uint32_t NodeWriteCount = 0;
    std::uint32_t GroupWriteCount = 0;
    std::uint32_t NodePendingCount = 0;
    std::uint32_t GroupPendingCount = 0;
    std::uint32_t ClusterCandidateCount = 0;
    std::uint32_t DrawCount = 0;
    std::uint32_t OverflowFlags = 0;
    std::uint32_t GroupFrustumRejected = 0;
    std::uint32_t GroupHZBRejected = 0;
    std::uint32_t NodeLodRejected = 0;
    std::uint32_t ClusterFrustumRejected = 0;
    std::uint32_t ClusterHZBRejected = 0;
    std::uint32_t ClusterRefinementRejected = 0;
    std::uint32_t ClusterTinyRejected = 0;
    std::uint32_t PostClusterCount = 0;
    std::uint32_t PostClusterHZBRejected = 0;
    std::uint32_t PostInstanceCount = 0;
    std::uint32_t PostNodeCount = 0;
    std::uint32_t PostGroupCount = 0;
    std::uint32_t PostInstanceRecovered = 0;
    std::uint32_t PostNodeRecovered = 0;
    std::uint32_t PostGroupRecovered = 0;
    std::uint32_t SoftRasterTriangles = 0;
    std::uint32_t SoftRasterPixels = 0;
    std::uint32_t SoftRasterSkippedLarge = 0;
    std::uint32_t SoftRasterSkippedClip = 0;
    std::uint32_t SoftDepthMatched = 0;
    std::uint32_t SoftDepthMismatch = 0;
    std::uint32_t SoftDepthMissing = 0;
    std::uint32_t SoftDepthExtra = 0;
    std::uint32_t SoftDepthMaxDiff = 0;
    std::uint32_t SoftVisMissing = 0;
    std::uint32_t SoftVisKeyMismatch = 0;
    std::uint32_t SoftVisResolved = 0;
    std::uint32_t SoftVisNoCover = 0;
    std::uint32_t SoftVisDepthOff = 0;
    std::uint32_t SoftClusterCount = 0;
    std::uint32_t HardwareClusterCount = 0;
    bool SoftRasterEnabled = false;
    // Experimental route selection. When enabled, AutoRasterHardware means the
    // current frame bypassed the hybrid bin and submitted the canonical draw list.
    bool AutoRasterEnabled = false;
    bool AutoRasterHardware = false;
    bool HZBEnabled = false;
    GpuTimings Gpu;
};

// Each raster pass is one non-indexed instanced indirect draw: one instance per
// visible cluster, VertexCount fixed at the largest cluster's index count.
// VertexCount and StartVertex are written once by the CPU; InstanceCount is the
// GPU-side cluster counter, which is the reason this replaces a per-cluster
// command array. Only DrawCount would need the indirect count-buffer extension
// MoltenVK is missing, and DrawCount is always 1.
struct DrawCommand
{
    std::uint32_t VertexCount;
    std::uint32_t InstanceCount;
    std::uint32_t StartVertex;
    std::uint32_t FirstInstance;
};

// Slots inside the draw command buffer. Both describe ranges of the same
// g_DrawInstanceData array: Main starts at slot 0 and covers everything the main
// phase and post cull together produced, Post starts at the first slot post cull
// claimed and covers only those. The split exists so the post raster pass draws
// the recovered clusters alone. Without it that pass redraws every cluster the
// main phase already rasterized, all of which then fail the strict depth test -
// on a static frame that is the entire pass wasted.
enum class DrawCommandSlot : std::uint32_t
{
    Main = 0,
    Post = 1,
    Count = 2,
};

// One vkCmdDispatchIndirect argument block, padded to 16 bytes so the four
// slots NanitePrepareDispatch writes stay 16-byte aligned. Vulkan only reads the
// three counts; the padding exists for the alignment alone.
struct DispatchCommand
{
    std::uint32_t ThreadGroupCountX;
    std::uint32_t ThreadGroupCountY;
    std::uint32_t ThreadGroupCountZ;
    std::uint32_t Padding;
};

// Slots inside the dispatch argument buffer. Must match the
// NANITE_DISPATCH_SLOT_* defines in NanitePrepareDispatch.hlsl.
enum class DispatchSlot : std::uint32_t
{
    GenerateIndirect = 0,
    PostSeed = 1,
    PostGenerateIndirect = 2,
    PostCull = 3,
    // Both consume one cluster per workgroup: binning scans canonical visible
    // clusters, then software raster consumes the compacted software list.
    RasterBin = 4,
    SoftRaster = 5,
    Count = 6,
};

// Per-cluster data the vertex shader pulls from, indexed by SV_InstanceID.
// FirstIndex and IndexCount are copied out of the cluster so the vertex shader
// needs one fetch instead of also reading the full cluster record.
struct DrawInstanceData
{
    std::uint32_t InstanceIndex;
    std::uint32_t ClusterIndex;
    std::uint32_t FirstIndex;
    std::uint32_t IndexCount;
};

struct alignas(16) CullingConstants
{
    float ViewProjMatrix[16];
    float PreviousViewProjMatrix[16];
    float CameraWorldPosition[3];
    float ScreenErrorPixels;
    std::uint32_t ViewportWidth;
    std::uint32_t ViewportHeight;
    std::uint32_t HZBWidth;
    std::uint32_t HZBHeight;
    std::uint32_t HZBMipCount;
    std::uint32_t EnableHZB;
    std::uint32_t InstanceCount;
    std::uint32_t MaxNodeTasks;
    std::uint32_t MaxGroupTasks;
    std::uint32_t MaxClusterTasks;
    std::uint32_t MaxDrawCommands;
    std::uint32_t MaxRefinementDepth;
    // 0 during the main phase, 1 during the occlusion recovery phase. The main
    // phase queries the previous frame's HZB and therefore has to reproject the
    // bounds into the previous view; the recovery phase queries the HZB built
    // from this frame's depth and uses the current clip rectangle directly.
    std::uint32_t CullingPass;
    // Iteration budget of the persistent traversal loop. Sized on the CPU from
    // the worst-case task count so the loop only ever exits early because the
    // queues drained; it stays finite purely as a hang guard.
    std::uint32_t MaxTraversalIterations;
    // cot(fovY/2), the projection's [1][1]. Passed rather than read out of
    // ViewProjMatrix, whose [1][1] is this times the view basis' up.y - that is,
    // times cos(pitch). Deriving it from the combined matrix therefore shrank every
    // projected error the moment the camera looked up or down, so the same threshold
    // meant a different number of pixels depending on where you were looking.
    float FocalY;
    // Generation written into NodeTask/GroupTask::Ready. It lets the queue
    // consumers reject records left by the previous frame without uploading a
    // full zeroed copy of both queues every frame.
    std::uint32_t QueueGeneration;
    // The hybrid path sends triangles at or below this screen-space area to the
    // software visibility rasterizer. Both rasterizers read the same value.
    float RasterBinAreaCutoff;
    std::uint32_t HybridRasterEnabled;
    // 1 selects the isolated cluster-level HW/SW experiment. The field stays in
    // the old padding slot so the culling constant-buffer size and offsets do not
    // change for existing shaders.
    std::uint32_t ClusterRasterMode;
    // Native Apple MSL visibility path: depth and payload are resolved as one
    // packed 64-bit atomic word. Kept in the former padding slot.
    std::uint32_t Padding4;
};

struct alignas(16) RasterConstants
{
    float ViewProjMatrix[16];
    std::uint32_t ViewportWidth;
    std::uint32_t ViewportHeight;
    std::uint32_t DebugMode;
    std::uint32_t DebugMip;
    // Blinn-Phong's view vector. The pass has the view-projection matrix but not
    // the eye it was built from, and inverting that matrix per pixel to recover a
    // value that is constant for the whole frame would be wasted work.
    float CameraWorldPosition[3];
    // Screen area in pixels below which a triangle belongs to the software
    // rasterizer, for the raster-bin visualization. Only that mode reads it.
    float RasterBinAreaCutoff;
    std::uint32_t HybridRasterEnabled;
    std::uint32_t ClusterRasterMode;
    // The HLSL cbuffer aligns float4 members to 16 bytes, which places
    // CameraRight at offset 112 in the shader; a C++ float[4] member would sit
    // at 104 here. The two padding words force the CPU-side layout to the same
    // 112-byte offset, otherwise every sky field below is read 8 bytes off and
    // the sampled direction is garbage.
    std::uint32_t LayoutPadding[2];
    // Camera basis and projection needed by the sky shader: it reconstructs a
    // view ray per pixel and samples the equirectangular sky in the direction
    // of that ray, so the sky turns with the camera instead of staying a fixed
    // screen-space image.
    float CameraRight[4];
    float CameraUp[4];
    float CameraForward[4];
    // x = pixel focalX, y = pixel focalY, z = centerX, w = centerY.
    float Focal[4];
    std::uint32_t Padding1;
    std::uint32_t Padding2;
    // Direction from the scene toward the sun. Kept in the raster constants so
    // the future cascade-shadow light and the atmosphere use one direction.
    float SkySunDirection[4];
    // x = planet radius, y = atmosphere height, z = Rayleigh scale, w = Mie scale.
    // These are the scaled world-space inputs for the procedural sky preview.
    float SkyAtmosphere[4];
    // x = sky optical scale, y = terrain optical scale, z = terrain haze
    // distance multiplier, w = terrain haze blend weight.
    float SkyPerspective[4];
    // Light-space matrix for the current directional shadow map. The first
    // implementation is one map; the field is an explicit slot for the future
    // cascade array so material shading does not need another geometry path.
    float ShadowViewProjMatrix[16];
    // x = virtual page count per axis, y = reciprocal page count, z = reciprocal
    // physical page size, w = reserved for page-table flags.
    float VsmPageInfo[4];
};

struct alignas(16) HizConstants
{
    // Number of coarser mips this dispatch writes, 1 to 3.
    std::uint32_t OutputMipCount;
    // Logical extent of the source level. The HZB is padded to a power of two,
    // so this drives the output bounds and the dispatch grid.
    std::uint32_t SourceWidth;
    std::uint32_t SourceHeight;
    // Extent of the texture actually bound as the source. For the first batch
    // this is the viewport-sized depth copy, which is smaller than the padded
    // logical extent. Texels outside it read as far.
    std::uint32_t SourceTexWidth;
    std::uint32_t SourceTexHeight;
    std::uint32_t Padding0;
    std::uint32_t Padding1;
    std::uint32_t Padding2;
};

static_assert(sizeof(Instance) == 128, "NaniteInstance stride must match SPIR-V");
static_assert(sizeof(Material) == 32, "NaniteMaterial stride must match SPIR-V");
static_assert(sizeof(DagNode) == 64, "NaniteDagNode stride must match SPIR-V");
static_assert(sizeof(ClusterGroup) == 64, "NaniteClusterGroup stride must match SPIR-V");
static_assert(sizeof(Cluster) == 64, "NaniteCluster stride must match SPIR-V");
static_assert(sizeof(NodeTask) == 16, "NaniteNodeTask stride must match SPIR-V");
static_assert(sizeof(GroupTask) == 16, "NaniteGroupTask stride must match SPIR-V");
static_assert(sizeof(ClusterTask) == 8, "NaniteClusterTask stride must match SPIR-V");
static_assert(sizeof(QueueState) == 212, "NaniteQueueState stride must match SPIR-V");
static_assert(sizeof(DrawCommand) == 16, "VkDrawIndirectCommand layout must match");
static_assert(sizeof(DispatchCommand) == 16, "Nanite dispatch argument slot stride must match");
static_assert(sizeof(DrawInstanceData) == 16, "NaniteDrawInstanceData stride must match SPIR-V");
static_assert(sizeof(CullingConstants) == 224, "NaniteCullingConstants size must match SPIR-V");
static_assert(sizeof(RasterConstants) == 320, "NaniteRasterConstants size must match SPIR-V");
static_assert(sizeof(HizConstants) == 32, "NaniteHizConstants size must match SPIR-V");

} // namespace Nanite
