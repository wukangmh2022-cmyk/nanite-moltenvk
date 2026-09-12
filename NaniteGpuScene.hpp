#pragma once

#include "NaniteModel.hpp"
#include "NanitePipelines.hpp"

#include <DeviceContext.h>
#include <Query.h>
#include <RefCntAutoPtr.hpp>
#include <RenderDevice.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <vector>

namespace Nanite
{

struct Capacity
{
    std::uint32_t MaxNodeTasks = 4096;
    std::uint32_t MaxGroupTasks = 4096;
    std::uint32_t MaxClusterTasks = 8192;
    std::uint32_t MaxDrawCommands = 8192;
    std::uint32_t PersistentWorkgroups = 1;
    // Iteration budget of one persistent traversal workgroup. Must be large
    // enough that PersistentWorkgroups * NANITE_GROUP_SIZE * this covers every
    // node and group task the scene can produce, or the traversal gives up with
    // work still queued and geometry blinks.
    std::uint32_t TraversalIterations = 1024;
};

struct AtmosphereSettings
{
    float SunAzimuthDeg = 0.0f;
    float SunElevationDeg = 16.26f;
    float AtmosphereRadius = 2400.0f;
    float AtmosphereHeight = 220.0f;
    float RayleighScale = 1.0f;
    float MieScale = 0.65f;
    float SkyDensity = 0.22f;
    float TerrainDensity = 0.08f;
    float HazeDistance = 0.55f;
    float HazeWeight = 0.65f;
};

class GpuScene
{
public:
    // Optional second queue used for the software visibility rasterizer. The
    // graphics queue signals BinReady after RasterBin; the async queue waits for
    // it, submits the two software passes, and signals SoftRasterDone before the
    // graphics queue records visibility resolve. A null pointer keeps the legacy
    // single-queue path.
    struct AsyncRasterContext
    {
        Diligent::IDeviceContext* Context = nullptr;
        Diligent::IFence* BinReady = nullptr;
        Diligent::IFence* SoftRasterDone = nullptr;
        Diligent::Uint64 SignalValue = 0;
    };

    GpuScene(Diligent::IRenderDevice* device,
             Pipelines& pipelines,
             const CpuScene& scene,
             const Capacity& capacity = {},
             Diligent::Uint64 immediateContextMask = 1);

    void Render(Diligent::IDeviceContext* context,
                Diligent::ITextureView* renderTarget,
                std::uint32_t viewportWidth,
                std::uint32_t viewportHeight,
                const float viewProjMatrix[16],
                const float cameraWorldPosition[3],
                float cameraYaw,
                float cameraPitch,
                const AsyncRasterContext* asyncRaster = nullptr);

    void SetHZBEnabled(bool enabled) { m_HZBEnabled = enabled; }
    bool IsHZBEnabled() const { return m_HZBEnabled; }
    void SetGpuTimingsEnabled(bool enabled);
    bool AreGpuTimingsSupported() const { return m_GpuTimingSupported; }

    // Pixels of geometric error a cluster group is allowed before its finer
    // clusters have to be drawn: the whole LOD cut, and therefore how visible a
    // level switch is. Live, because judging popping means sweeping it by eye
    // rather than restarting per value. Negative disables the cluster-level cut.
    void SetScreenErrorPixels(float pixels) { m_ScreenErrorPixels = pixels; }
    float GetScreenErrorPixels() const { return m_ScreenErrorPixels; }

    // 0 for Blinn-Phong, 5 for the per-cluster hash colour, and 11 for the PBR
    // Visibility-Buffer pass. Only consulted when no NANITE_VISUALIZE_* variable
    // has claimed the debug mode for itself.
    void SetShadingMode(std::uint32_t mode) { m_ShadingMode = mode; }
    std::uint32_t GetShadingMode() const { return m_ShadingMode; }

    // Area in pixels squared below which raster-bin visualization marks a triangle
    // as software rasterized. Live: only the raster constant buffer changes.
    void SetRasterBinAreaCutoff(float cutoff) { m_RasterBinAreaCutoff = std::max(cutoff, 0.0f); }
    float GetRasterBinAreaCutoff() const { return m_RasterBinAreaCutoff; }

    void SetAtmosphereSettings(const AtmosphereSettings& settings);
    const AtmosphereSettings& GetAtmosphereSettings() const { return m_AtmosphereSettings; }

    // Changes how many instances the culling passes walk, and re-lays out and
    // re-uploads their transforms, without touching any GPU allocation. The
    // buffers are sized for the scene's full instance count, so the count can be
    // moved anywhere in [1, GetMaxInstanceCount()] between frames.
    void SetActiveInstanceCount(std::uint32_t count);
    std::uint32_t GetActiveInstanceCount() const { return m_ActiveInstanceCount; }
    void SetInstanceGrid(std::uint32_t countX, std::uint32_t countY, std::uint32_t countZ);
    std::uint32_t GetInstanceCountX() const { return m_InstanceCountX; }
    std::uint32_t GetInstanceCountY() const { return m_InstanceCountY; }
    std::uint32_t GetInstanceCountZ() const { return m_InstanceCountZ; }
    std::uint32_t GetMaxInstanceCount() const
    {
        return static_cast<std::uint32_t>(m_CpuScene.Instances.size());
    }

    // Distance between neighbouring instances: horizontal covers both the columns
    // and the depth layers, vertical covers the rows. Costs a re-layout and a
    // transform upload, same as the count, so it is live and needs no rebuild.
    void SetInstanceSpacing(float horizontal, float vertical);
    float GetSpacingHorizontal() const { return m_SpacingHorizontal; }
    float GetSpacingVertical() const { return m_SpacingVertical; }

    const CpuScene& GetCpuScene() const { return m_CpuScene; }
    const FrameStats& GetLastFrameStats() const { return m_LastFrameStats; }

private:
    void BindResources(Pipelines& pipelines);
    void ResetQueues(Diligent::IDeviceContext* context);
    void UpdateConstants(Diligent::IDeviceContext* context,
                         std::uint32_t viewportWidth,
                         std::uint32_t viewportHeight,
                         const float viewProjMatrix[16],
                         const float cameraWorldPosition[3]);
    void EnsureRenderTargets(std::uint32_t viewportWidth,
                             std::uint32_t viewportHeight);
    void RenderShadowMap(Diligent::IDeviceContext* context);
    void UpdateVsmPageRequests(const float viewProjMatrix[16],
                               std::uint32_t viewportWidth,
                               std::uint32_t viewportHeight);
    void RenderVirtualShadowPages(Diligent::IDeviceContext* context,
                                  const float viewProjMatrix[16],
                                  std::uint32_t viewportWidth,
                                  std::uint32_t viewportHeight);
    void RenderVirtualShadowPage(Diligent::IDeviceContext* context,
                                 const std::array<float, 16>& pageViewProj,
                                 std::uint32_t physicalPage);
    void InvalidateVirtualShadowPages();
    void BuildHZB(Diligent::IDeviceContext* context,
                  std::uint32_t viewportWidth,
                  std::uint32_t viewportHeight,
                  Diligent::ITexture* depthSource,
                  bool recordTiming);
    void ResolveVisibility(Diligent::IDeviceContext* context,
                           std::uint32_t viewportWidth,
                           std::uint32_t viewportHeight);
    void ShadeVisibility(Diligent::IDeviceContext* context,
                         Diligent::ITextureView* renderTarget,
                         Diligent::ITextureView* depthSource,
                         Diligent::ITextureView* visibilitySource);
    // Diagnostic: read every HZB level back and print its min/max so a failed
    // mip write can be told apart from a bad sampling coordinate.
    void DumpHzbMips(Diligent::IDeviceContext* context);
    void VerifyClusterOcclusion(const Diligent::MappedTextureSubresource& mapped,
                                std::uint32_t viewportWidth,
                                std::uint32_t viewportHeight);
    bool ReadbackStats(Diligent::IDeviceContext* context, bool dumpGroupTasks);
    void BeginGpuTimingFrame(Diligent::IDeviceContext* context);
    void BeginGpuTimingPass(Diligent::IDeviceContext* context, std::uint32_t pass);
    void EndGpuTimingPass(Diligent::IDeviceContext* context, std::uint32_t pass);
    void EndGpuTimingFrame(Diligent::IDeviceContext* context);
    void PollGpuTimings();
    void UpdateAutoRasterDecision();

    // Resolve is issued once before post cull and once after post raster. Keep
    // separate query pairs so the reported resolve time is the sum of those two
    // dispatches without including the work between them.
    // Pass 15 is the second software visibility sweep. It is intentionally a
    // separate timestamp from pass 8, otherwise the visibility-buffer fallback
    // looks like one opaque "soft raster" cost. Passes 16+ cover GPU work that
    // used to be absent from the HUD: queue reset, prepare dispatches, recovery,
    // and the final hybrid HZB build. Pass 22 is the directional shadow map.
    static constexpr std::uint32_t GpuTimingPassCount = 23;
    static constexpr std::uint32_t GpuTimingQueryCount = GpuTimingPassCount * 2;
    static_assert(GpuTimingPassCount <= 32, "IssuedPassMask holds one bit per pass");
    struct GpuTimingSlot
    {
        std::array<Diligent::RefCntAutoPtr<Diligent::IQuery>, GpuTimingQueryCount> Queries{};
        std::uint32_t FrameIndex = 0;
        // One bit per pass, set when that pass actually issued its query pair this
        // frame. Not every pass runs every frame: the hybrid and probe passes are
        // conditional. Asking a timestamp query for data before it has ever
        // been ended dereferences null inside QueryVkImpl::GetData, whose only guard
        // there is a DEV_CHECK_ERR that release builds compile away.
        std::uint32_t IssuedPassMask = 0;
        // Probe passes can be repeated inside one timestamp bracket. Keep this
        // per-slot so hybrid frames are never normalized by a stale probe value.
        std::uint32_t ProbeRepeat = 1;
        bool InFlight = false;
    };

    Diligent::RefCntAutoPtr<Diligent::IRenderDevice> m_Device;
    Pipelines* m_Pipelines = nullptr;
    Diligent::Uint64 m_ImmediateContextMask = 1;
    Diligent::RefCntAutoPtr<Diligent::IBuffer> m_PositionBuffer;
    Diligent::RefCntAutoPtr<Diligent::IBuffer> m_UvBuffer;
    // Parallel to m_PositionBuffer and indexed the same way, so the vertex shader
    // fetches a normal with the index it already resolved for the position.
    Diligent::RefCntAutoPtr<Diligent::IBuffer> m_NormalBuffer;
    // Meshlet-local geometry for software raster. The fixed-function path remains
    // on the original global buffers; these are bound only to the compute passes.
    Diligent::RefCntAutoPtr<Diligent::IBuffer> m_ClusterPositionBuffer;
    Diligent::RefCntAutoPtr<Diligent::IBuffer> m_ClusterIndexBuffer;
    Diligent::RefCntAutoPtr<Diligent::IBuffer> m_ClusterLocalIndexOffsetBuffer;
    Diligent::RefCntAutoPtr<Diligent::IBuffer> m_IndexBuffer;
    Diligent::RefCntAutoPtr<Diligent::IBuffer> m_InstanceBuffer;
    Diligent::RefCntAutoPtr<Diligent::IBuffer> m_NodeBuffer;
    Diligent::RefCntAutoPtr<Diligent::IBuffer> m_ClusterGroupBuffer;
    Diligent::RefCntAutoPtr<Diligent::IBuffer> m_ClusterBuffer;
    Diligent::RefCntAutoPtr<Diligent::IBuffer> m_MaterialBuffer;
    Diligent::RefCntAutoPtr<Diligent::IBuffer> m_MaterialIndexBuffer;
    Diligent::RefCntAutoPtr<Diligent::IBuffer> m_NodeQueue;
    Diligent::RefCntAutoPtr<Diligent::IBuffer> m_GroupQueue;
    Diligent::RefCntAutoPtr<Diligent::IBuffer> m_ClusterQueue;
    Diligent::RefCntAutoPtr<Diligent::IBuffer> m_PostClusterQueue;
    Diligent::RefCntAutoPtr<Diligent::IBuffer> m_PostInstanceQueue;
    Diligent::RefCntAutoPtr<Diligent::IBuffer> m_PostNodeQueue;
    Diligent::RefCntAutoPtr<Diligent::IBuffer> m_PostGroupQueue;
    Diligent::RefCntAutoPtr<Diligent::IBuffer> m_QueueState;
    Diligent::RefCntAutoPtr<Diligent::IBuffer> m_DrawCommands;
    // Hybrid binning compacts the canonical draw indices into disjoint hardware
    // and software cluster lists. The hardware command consumes only its list.
    Diligent::RefCntAutoPtr<Diligent::IBuffer> m_HybridDrawCommand;
    Diligent::RefCntAutoPtr<Diligent::IBuffer> m_HardwareDrawIndices;
    Diligent::RefCntAutoPtr<Diligent::IBuffer> m_SoftwareDrawIndices;
    // vkCmdDispatchIndirect arguments for the passes whose work count only the
    // GPU knows, one 16-byte slot per DispatchSlot. Written by
    // NanitePrepareDispatch, read by the driver.
    Diligent::RefCntAutoPtr<Diligent::IBuffer> m_DispatchArgs;
    Diligent::RefCntAutoPtr<Diligent::IBuffer> m_DrawInstanceIndices;
    Diligent::RefCntAutoPtr<Diligent::IBuffer> m_DebugQueueReadback;
    Diligent::RefCntAutoPtr<Diligent::IBuffer> m_CullingConstants;
    Diligent::RefCntAutoPtr<Diligent::IBuffer> m_RasterConstants;
    Diligent::RefCntAutoPtr<Diligent::IBuffer> m_HizConstants;
    Diligent::RefCntAutoPtr<Diligent::ITexture> m_DepthTexture;
    Diligent::RefCntAutoPtr<Diligent::ITextureView> m_DepthDSV;
    Diligent::RefCntAutoPtr<Diligent::ITextureView> m_DepthSRV;
    Diligent::RefCntAutoPtr<Diligent::ITexture> m_DepthColorTexture;
    Diligent::RefCntAutoPtr<Diligent::ITextureView> m_DepthColorRTV;
    Diligent::RefCntAutoPtr<Diligent::ITextureView> m_DepthColorSRV;
    // Hardware's R32_UINT identity target. It is separate from the final
    // visibility texture because the compute resolve must read it while writing
    // the soft/hard winner.
    Diligent::RefCntAutoPtr<Diligent::ITexture> m_HardwareVisibilityTexture;
    Diligent::RefCntAutoPtr<Diligent::ITextureView> m_HardwareVisibilityRTV;
    Diligent::RefCntAutoPtr<Diligent::ITextureView> m_HardwareVisibilitySRV;
    Diligent::RefCntAutoPtr<Diligent::ITexture> m_FinalDepthTexture;
    Diligent::RefCntAutoPtr<Diligent::ITextureView> m_FinalDepthSRV;
    Diligent::RefCntAutoPtr<Diligent::ITextureView> m_FinalDepthUAV;
    Diligent::RefCntAutoPtr<Diligent::ITexture> m_FinalVisibilityTexture;
    Diligent::RefCntAutoPtr<Diligent::ITextureView> m_FinalVisibilitySRV;
    Diligent::RefCntAutoPtr<Diligent::ITextureView> m_FinalVisibilityUAV;
    Diligent::RefCntAutoPtr<Diligent::ITexture> m_DepthColorReadback;
    Diligent::RefCntAutoPtr<Diligent::ITexture> m_HzbSourceTexture;
    Diligent::RefCntAutoPtr<Diligent::ITextureView> m_HzbSourceSRV;
    Diligent::RefCntAutoPtr<Diligent::ITexture> m_HzbTexture;
    Diligent::RefCntAutoPtr<Diligent::ITextureView> m_HzbSRV;
    std::vector<Diligent::RefCntAutoPtr<Diligent::ITextureView>> m_HzbMipSRV;
    std::vector<Diligent::RefCntAutoPtr<Diligent::ITextureView>> m_HzbMipUAV;
    // Diligent tracks resource state per texture, not per subresource, so the
    // HZB cannot be bound as an SRV and a UAV in the same dispatch. Each
    // continuation batch reads a dedicated copy of its source level instead.
    std::vector<Diligent::RefCntAutoPtr<Diligent::ITexture>> m_HzbBatchSource;
    std::vector<Diligent::RefCntAutoPtr<Diligent::ITextureView>> m_HzbBatchSourceSRV;
    Diligent::RefCntAutoPtr<Diligent::ITexture> m_DisabledHZB;
    Diligent::RefCntAutoPtr<Diligent::ISampler> m_HZBSampler;
    Diligent::RefCntAutoPtr<Diligent::ITexture> m_AlbedoTexture;
    Diligent::RefCntAutoPtr<Diligent::ITexture> m_NormalTexture;
    Diligent::RefCntAutoPtr<Diligent::ITexture> m_MetallicRoughnessTexture;
    Diligent::RefCntAutoPtr<Diligent::ISampler> m_AlbedoSampler;
    Diligent::RefCntAutoPtr<Diligent::ITexture> m_ShadowMapTexture;
    Diligent::RefCntAutoPtr<Diligent::ITextureView> m_ShadowMapDSV;
    Diligent::RefCntAutoPtr<Diligent::ITextureView> m_ShadowMapSRV;
    Diligent::RefCntAutoPtr<Diligent::ISampler> m_ShadowSampler;
    Diligent::RefCntAutoPtr<Diligent::ITexture> m_VsmPhysicalPoolTexture;
    Diligent::RefCntAutoPtr<Diligent::ITextureView> m_VsmPhysicalPoolSRV;
    std::vector<Diligent::RefCntAutoPtr<Diligent::ITextureView>> m_VsmPhysicalPoolDSV;
    Diligent::RefCntAutoPtr<Diligent::IBuffer> m_VsmPageTableBuffer;
    Diligent::RefCntAutoPtr<Diligent::ISampler> m_VsmSampler;
    static constexpr std::uint32_t VsmPageSize = 128;
    static constexpr std::uint32_t VsmPhysicalPageCount = 128;
    struct VsmPageState
    {
        std::int32_t PhysicalPage = -1;
        std::uint32_t LastUsedFrame = 0;
        bool Valid = false;
    };
    std::vector<VsmPageState> m_VsmPages;
    std::vector<std::uint32_t> m_VsmPageTable;
    std::vector<std::uint32_t> m_VsmDirtyPages;
    std::vector<std::uint8_t> m_VsmPhysicalUsed;
    std::uint32_t m_VsmPageCount = 0;
    std::uint32_t m_VsmPageBudget = 1;
    std::uint32_t m_VsmPagesRenderedLastFrame = 0;
    bool m_VsmEnabled = true;
    std::uint32_t m_ShadowMapSize = 2048;
    // The directional shadow map is independent of the camera. Re-render it
    // only after the sun or scene layout changes.
    bool m_ShadowMapValid = false;
    bool m_HasTextureCoordinates = false;
    // Viewport-sized u32 software depth, one word per pixel, resolved by
    // InterlockedMin. The second software pass reads it to select the payload.
    Diligent::RefCntAutoPtr<Diligent::IBuffer> m_SoftDepthBuffer;
    // Two words per pixel: depth key and the winning payload. Only triangles that
    // exactly match the resolved software depth are allowed to write it.
    Diligent::RefCntAutoPtr<Diligent::IBuffer> m_SoftVisBuffer;
    // Scratch depth target for the hardware depth-only reference draw. Separate
    // from m_DepthTexture so the probe cannot disturb the frame it is measuring,
    // and so it starts cleared rather than already populated, which early-Z would
    // otherwise make the reference look free.
    Diligent::RefCntAutoPtr<Diligent::ITexture> m_SoftProbeDepthTexture;
    Diligent::RefCntAutoPtr<Diligent::ITextureView> m_SoftProbeDepthDSV;
    // How many times each probe pass is issued inside its timing bracket. A single
    // issue cannot be read directly: MoltenVK cannot place a timestamp mid-encoder,
    // so the interval also contains render-pass setup and the barrier wait. Sweeping
    // this over 1/2/4/8 gives a slope that is the real per-pass cost.
    std::uint32_t m_ProbeRepeat = 1;
    // Queue counters are incremented once per repeated depth sweep; normalize
    // them before exposing probe work so repeat values remain comparable.
    std::uint32_t m_ProbeStatsRepeat = 1;
    // The staging copy is mapped on a later frame, so retain the repeat count
    // that belongs to the copy rather than consulting the current frame.
    std::uint32_t m_StatsReadbackProbeRepeat = 1;

    CpuScene m_CpuScene;
    Capacity m_Capacity;
    CullingConstants m_CullingData{};
    RasterConstants m_RasterData{};
    HizConstants m_HizData{};
    std::vector<NodeTask> m_EmptyNodeQueue;
    std::vector<GroupTask> m_EmptyGroupQueue;
    // Only NANITE_CPU_DRAW_DEBUG fills this; the normal path never touches
    // instance data on the CPU.
    std::vector<DrawInstanceData> m_DebugDrawInstanceData;
    // Vertex count of the single indirect command: the largest cluster's index
    // count, which every shorter cluster pads out with degenerate triangles.
    std::uint32_t m_MaxClusterIndexCount = 0;
    bool m_HZBEnabled = false;
    // The two-pass software visibility path is available by default; a positive
    // raster-bin cutoff opts triangles into it. NANITE_DISABLE_SOFT_RASTER is an
    // emergency opt-out, while NANITE_SOFT_RASTER_PROBE controls only the legacy
    // full-scene microbenchmark below.
    bool m_SoftRasterEnabled = false;
    // Apple-only MSL override mode: depth and payload are committed together by
    // one native 64-bit atomic, so the portable second coverage sweep is skipped.
    bool m_Native64BitVisibility = false;
    // Legacy full-scene depth/visibility microbenchmark. Kept separate so the
    // hybrid feature switch cannot silently run the probe at cutoff zero.
    bool m_SoftProbeEnabled = false;
    bool m_HybridRasterAvailable = false;
    // Optional one-frame-lag route selection. It is deliberately opt-in because
    // collecting a decision every frame adds a small readback/flush cost.
    bool m_AutoRasterRouting = false;
    bool m_AutoRasterHardware = false;
    bool m_AutoRasterLastHybrid = false;
    std::uint32_t m_AutoRasterHardwareFrames = 0;
    // Seeded from NANITE_SCREEN_ERROR / NANITE_DISABLE_CLUSTER_CULL at
    // construction, then owned by whoever moves the control panel slider.
    float m_ScreenErrorPixels = 1.0f;
    // Camera orientation used by the direction-based sky sampling.
    float m_CameraYaw = 0.0f;
    float m_CameraPitch = 0.0f;
    std::uint32_t m_ShadingMode = 0u;
    // The default launch configuration routes the mostly sub-pixel 512-instance
    // workload through the native software path. NANITE_RASTER_BIN_AREA=0 or the
    // GUI control can restore the pure hardware baseline.
    float m_RasterBinAreaCutoff = 256.0f;
    AtmosphereSettings m_AtmosphereSettings{};
    // Explicit benchmark-only switch. It is environment seeded and intentionally
    // has no GUI control so the normal per-triangle path remains unchanged.
    bool m_ClusterRasterExperiment = false;
    float m_ClusterRasterAreaCutoff = 0.0f;
    // How many of m_CpuScene.Instances are live this frame, and whether their
    // transforms still have to be pushed to the GPU. Set by
    // SetActiveInstanceCount and consumed at the top of Render, because the
    // upload needs a device context.
    std::uint32_t m_ActiveInstanceCount = 0;
    std::uint32_t m_InstanceCountX = 0;
    std::uint32_t m_InstanceCountY = 0;
    std::uint32_t m_InstanceCountZ = 0;
    bool m_CustomInstanceLayout = false;
    bool m_InstanceUploadPending = false;
    // Instance pitch, matching LayoutInstances' defaults. Owned by the control
    // panel once it moves them.
    float m_SpacingHorizontal = 2.0f;
    float m_SpacingVertical = 2.0f;
    bool m_DebugQueuePrinted = false;
    bool m_HzbMipDumpPrinted = false;
    std::uint32_t m_GpuTimingPrintCounter = 0;
    GpuTimings m_GpuTimingAccum{};
    // One sampled frame hides the frames that overflowed, which is exactly what a
    // flicker is, so these keep the worst case seen since the last print.
    std::uint32_t m_OverflowSticky = 0;
    std::uint32_t m_PeakDrawCount = 0;
    std::uint32_t m_PeakNodeTasks = 0;
    std::uint32_t m_PeakCandidates = 0;
    // The low end matters as much as the peak: a static camera that produces a
    // different cluster count from frame to frame is the flicker itself.
    std::uint32_t m_MinDrawCount = 0xFFFFFFFFu;
    std::uint32_t m_MinCandidates = 0xFFFFFFFFu;
    std::uint32_t m_DebugQueueReadbackFrame = 1;
    std::uint32_t m_StatsReadbackInterval = 60;
    bool m_StatsReadbackPending = false;
    std::uint32_t m_FrameIndex = 0;
    std::uint32_t m_HzbWidth = 1;
    std::uint32_t m_HzbHeight = 1;
    std::uint32_t m_HzbMipCount = 1;
    bool m_HzbValid = false;
    bool m_HasPreviousCamera = false;
    std::array<float, 16> m_PreviousViewProj{};
    bool m_CpuVerifyPrinted = false;
    std::uint32_t m_CpuVerifyPixelX = 0;
    std::uint32_t m_CpuVerifyPixelY = 0;
    bool m_GpuTimingEnabled = false;
    bool m_GpuTimingSupported = false;
    bool m_GpuTimingFrameActive = false;
    bool m_LastAsyncRaster = false;
    // Queue task records carry this generation in their Ready field. Keeping
    // generations distinct across frames removes the per-frame full queue clear;
    // NANITE_DISABLE_QUEUE_EPOCH restores the old clear path for diagnostics.
    bool m_QueueEpochEnabled = true;
    std::uint32_t m_QueueGeneration = 0;
    int m_ActiveGpuTimingSlot = -1;
    std::array<GpuTimingSlot, 4> m_GpuTimingSlots{};
    FrameStats m_LastFrameStats{};
};

} // namespace Nanite
