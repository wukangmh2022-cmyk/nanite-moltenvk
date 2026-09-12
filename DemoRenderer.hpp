#pragma once

#include "NaniteGpuScene.hpp"

#include <NativeWindow.h>
#include <Fence.h>
#include <RefCntAutoPtr.hpp>
#include <SwapChain.h>

#include <cstdint>
#include <memory>
#include <string>

namespace NaniteDemo
{

class Renderer
{
public:
    Renderer(const Diligent::NativeWindow& window,
             std::uint32_t width,
             std::uint32_t height);
    ~Renderer();

    void Draw(std::uint32_t width, std::uint32_t height);
    void SetCamera(float x, float y, float z, float yaw, float pitch);
    void ReloadRabbitCount(std::uint32_t count);
    void SetHZBEnabled(bool enabled);
    void SetGpuTimingsEnabled(bool enabled);
    void SetAsyncRasterEnabled(bool enabled);
    bool HasAsyncRasterQueue() const { return m_AsyncRasterAvailable; }
    bool IsAsyncRasterEnabled() const { return m_AsyncRasterEnabled; }
    bool AreGpuTimingsSupported() const;

    // Live, no rebuild: both are constant-buffer fields the culling and raster
    // passes read every frame.
    void SetScreenErrorPixels(float pixels);
    float GetScreenErrorPixels() const;
    void SetShadingMode(std::uint32_t mode);
    std::uint32_t GetShadingMode() const;
    void SetRasterBinAreaCutoff(float cutoff);
    float GetRasterBinAreaCutoff() const;
    void SetAtmosphereSettings(const Nanite::AtmosphereSettings& settings);
    const Nanite::AtmosphereSettings& GetAtmosphereSettings() const
    {
        return m_AtmosphereSettings;
    }

    // Model path and cluster granularity both feed the LOD build, so changing
    // either means loading the file again and rebuilding the DAG from scratch.
    // Stored rather than applied, and consumed by the next ReloadScene, so the two
    // can be changed together instead of paying for two rebuilds.
    void SetModelPath(std::string path) { m_ModelPath = std::move(path); }
    const std::string& GetModelPath() const { return m_ModelPath; }
    void SetClusterTriangles(std::uint32_t triangles) { m_ClusterTriangles = triangles; }
    std::uint32_t GetClusterTriangles() const { return m_ClusterTriangles; }
    // Rebuilds the scene with the currently stored model and granularity. Blocking:
    // a million-triangle PLY takes several seconds to parse and cluster.
    void ReloadSceneFromSettings();

    // Presentation sync interval. 0 uncaps the frame rate so culling changes
    // show up as a frame time difference instead of being hidden by vsync.
    void SetVSyncEnabled(bool enabled) { m_VSyncEnabled = enabled; }
    bool IsVSyncEnabled() const { return m_VSyncEnabled; }

    // Instance pitch, live like the count. Horizontal moves the columns and the
    // depth layers, vertical moves the rows. A grid request can grow the
    // instance-dependent scene buffers when its total exceeds the current size.
    void SetInstanceSpacing(float horizontal, float vertical);
    float GetSpacingHorizontal() const { return m_SpacingHorizontal; }
    float GetSpacingVertical() const { return m_SpacingVertical; }
    void SetInstanceGrid(std::uint32_t countX, std::uint32_t countY, std::uint32_t countZ);
    std::uint32_t GetInstanceCountX() const { return m_InstanceCountX; }
    std::uint32_t GetInstanceCountY() const { return m_InstanceCountY; }
    std::uint32_t GetInstanceCountZ() const { return m_InstanceCountZ; }

    std::uint32_t GetRabbitCount() const { return m_RabbitCount; }
    // Current allocation size. SetInstanceGrid grows it on demand when needed.
    std::uint32_t GetMaxRabbitCount() const;
    bool IsHZBEnabled() const { return m_HZBEnabled; }
    const Nanite::FrameStats& GetLastFrameStats() const;

private:
    void ReloadScene(std::uint32_t count);

    // Triple buffering: three swap chain images and at most three frames in
    // flight, the frame being recorded plus the two the GPU has not finished. The
    // fence is what enforces the second half - without it the CPU would queue
    // frames without limit, and the latency of a click would grow with however far
    // ahead it had run.
    static constexpr std::uint64_t kFramesInFlight = 3;

    Diligent::RefCntAutoPtr<Diligent::IRenderDevice> m_Device;
    Diligent::RefCntAutoPtr<Diligent::IDeviceContext> m_Context;
    Diligent::RefCntAutoPtr<Diligent::IDeviceContext> m_AsyncComputeContext;
    Diligent::RefCntAutoPtr<Diligent::ISwapChain> m_SwapChain;
    // Signalled with the frame's own number after its Present is submitted, so
    // waiting for value N means frame N has left the GPU.
    Diligent::RefCntAutoPtr<Diligent::IFence> m_FrameFence;
    Diligent::RefCntAutoPtr<Diligent::IFence> m_AsyncBinReadyFence;
    Diligent::RefCntAutoPtr<Diligent::IFence> m_AsyncSoftRasterDoneFence;
    bool m_AsyncRasterAvailable = false;
    bool m_AsyncRasterEnabled = false;
    std::uint64_t m_FrameNumber = 0;
    std::unique_ptr<Nanite::Pipelines> m_NanitePipelines;
    std::unique_ptr<Nanite::GpuScene> m_NaniteScene;
    Diligent::SwapChainDesc m_SwapChainDesc;
    Diligent::NativeWindow m_Window;
    std::uint32_t m_RabbitCount = 512;
    // Empty until the first ReloadScene resolves it, so a run without
    // NANITE_MODEL still shows the default path in the control panel.
    std::string m_ModelPath;
    // 0 means "whatever the model loader's own default is".
    std::uint32_t m_ClusterTriangles = 0;
    // Mirrors of the two live scene settings, so a rebuild does not reset them to
    // the environment's values. The first scene seeds them, since it is the one
    // that reads the environment.
    float m_ScreenErrorPixels = 1.0f;
    std::uint32_t m_ShadingMode = 0;
    // Mirrors of the live raster-bin boundary and instance pitch, so a rebuild does not reset them.
    // The default launch configuration matches NANITE_RASTER_BIN_AREA=256.
    // Setting this control to zero restores the pure hardware baseline.
    float m_RasterBinAreaCutoff = 256.0f;
    Nanite::AtmosphereSettings m_AtmosphereSettings{};
    float m_SpacingHorizontal = 2.0f;
    float m_SpacingVertical = 2.0f;
    std::uint32_t m_InstanceCountX = 8;
    std::uint32_t m_InstanceCountY = 8;
    std::uint32_t m_InstanceCountZ = 8;
    bool m_InstanceGridExplicit = true;
    bool m_SceneSettingsSeeded = false;
    bool m_HZBEnabled = false;
    bool m_GpuTimingsEnabled = false;
    bool m_VSyncEnabled = false;
    float m_CameraX = 0.0f;
    float m_CameraY = 0.0f;
    float m_CameraZ = 6.0f;
    float m_CameraYaw = 0.0f;
    float m_CameraPitch = 0.0f;
};

} // namespace NaniteDemo
