#include "DemoRenderer.hpp"

#include <DeviceContext.h>
#include <EngineFactoryVk.h>
#include <GraphicsTypes.h>
#include <RenderDevice.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

using namespace Diligent;

namespace
{

struct Float3
{
    float x;
    float y;
    float z;
};

float Dot(Float3 a, Float3 b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

Float3 Cross(Float3 a, Float3 b)
{
    return {
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x};
}

std::array<float, 16> BuildViewProjection(
    float cameraX,
    float cameraY,
    float cameraZ,
    float yaw,
    float pitch,
    Uint32 width,
    Uint32 height)
{
    constexpr float nearPlane = 0.1f;
    constexpr float farPlane = 200.0f;
    constexpr float verticalFov = 60.0f * 3.14159265358979323846f / 180.0f;
    const float aspect = static_cast<float>(width) / static_cast<float>(std::max(height, 1u));
    const float focalY = 1.0f / std::tan(verticalFov * 0.5f);
    const float focalX = focalY / aspect;
    const float depthScale = farPlane / (nearPlane - farPlane);
    const float depthOffset = nearPlane * farPlane / (nearPlane - farPlane);

    const float cosYaw = std::cos(yaw);
    const float sinYaw = std::sin(yaw);
    const float cosPitch = std::cos(pitch);
    const float sinPitch = std::sin(pitch);
    const Float3 forward{sinYaw * cosPitch, sinPitch, -cosYaw * cosPitch};
    const Float3 right{cosYaw, 0.0f, sinYaw};
    const Float3 up = Cross(right, forward);
    const Float3 back{-forward.x, -forward.y, -forward.z};

    const float view[16] = {
        right.x, right.y, right.z, -Dot(right, {cameraX, cameraY, cameraZ}),
        up.x, up.y, up.z, -Dot(up, {cameraX, cameraY, cameraZ}),
        back.x, back.y, back.z, -Dot(back, {cameraX, cameraY, cameraZ}),
        0.0f, 0.0f, 0.0f, 1.0f};
    const float projection[16] = {
        focalX, 0.0f, 0.0f, 0.0f,
        0.0f, focalY, 0.0f, 0.0f,
        0.0f, 0.0f, depthScale, depthOffset,
        0.0f, 0.0f, -1.0f, 0.0f};
    float clip[16] = {};
    for (int row = 0; row < 4; ++row)
    {
        for (int column = 0; column < 4; ++column)
        {
            for (int element = 0; element < 4; ++element)
                clip[row * 4 + column] +=
                    projection[row * 4 + element] * view[element * 4 + column];
        }
    }

    // glslang lowers the HLSL path to vector-times-row-major-matrix SPIR-V.
    std::array<float, 16> viewProj{};
    for (int row = 0; row < 4; ++row)
        for (int column = 0; column < 4; ++column)
            viewProj[row * 4 + column] = clip[column * 4 + row];
    return viewProj;
}

Nanite::Capacity BuildCapacity(const Nanite::CpuScene& scene)
{
    Nanite::Capacity capacity{};
    const std::uint64_t instanceCount = scene.Instances.size();

    // Provisioned draw slots and cluster tasks, one per (instance, cluster) pair.
    // The raster pass is a single instanced indirect draw whose InstanceCount the
    // GPU writes, so nothing is cleared or drawn per slot any more and the only
    // cost of the full upper bound is 16 bytes of instance data plus 8 bytes of
    // cluster task each. Sizing both at that bound makes the shader-side overflow
    // structurally unreachable: a cluster is queued at most once per instance per
    // frame, so no cluster can be dropped and no draw slot can be lost.
    const std::uint64_t sceneClusterUpperBound =
        static_cast<std::uint64_t>(scene.Clusters.size()) * instanceCount;

    // A composed scene has one root DAG per prototype, but its worst-case
    // Cartesian product (all instances x all prototype clusters) is not a
    // useful allocation target: it reserved 11+ GB for the 2688-instance
    // coastal layout before the first frame. The queues are transient and the
    // existing overflow counters make a bounded budget observable, so keep a
    // practical default for this authored composition. Stress runs can raise
    // the budget without changing the ordinary rabbit/Buddha sizing policy.
    const bool boundedComposition = scene.CustomInstanceLayout;
    std::uint64_t clusterBudget = sceneClusterUpperBound;
    std::uint64_t nodeBudget = static_cast<std::uint64_t>(scene.Nodes.size()) * instanceCount;
    std::uint64_t groupBudget = static_cast<std::uint64_t>(scene.ClusterGroups.size()) * instanceCount;
    if (boundedComposition)
    {
        constexpr std::uint64_t DefaultCoastalQueueCap = 4'000'000u;
        std::uint64_t requestedCap = DefaultCoastalQueueCap;
        if (const char* cap = std::getenv("NANITE_COASTAL_QUEUE_CAP"))
        {
            requestedCap = std::strtoull(cap, nullptr, 10);
            if (requestedCap == 0u)
                throw std::runtime_error{"NANITE_COASTAL_QUEUE_CAP must be greater than zero"};
        }
        clusterBudget = std::min(clusterBudget, requestedCap);
        nodeBudget = std::min(nodeBudget, std::max<std::uint64_t>(requestedCap / 2u, 65536u));
        groupBudget = std::min(groupBudget, std::max<std::uint64_t>(requestedCap / 2u, 65536u));
        std::cerr << "Nanite bounded composition capacity: queueCap=" << requestedCap
                  << " node=" << nodeBudget
                  << " group=" << groupBudget
                  << " cluster=" << clusterBudget << '\n';
    }
    std::uint64_t drawBudget =
        std::max<std::uint64_t>(capacity.MaxDrawCommands, clusterBudget);
    // Kept as a way to force the overflow path back on for testing.
    if (const char* drawCap = std::getenv("NANITE_STRESS_DRAW_CAP"))
    {
        const std::uint64_t cap = std::strtoull(drawCap, nullptr, 10);
        if (cap == 0u)
            throw std::runtime_error{"NANITE_STRESS_DRAW_CAP must be greater than zero"};
        drawBudget = std::min(drawBudget, cap);
    }
    capacity.MaxNodeTasks = static_cast<std::uint32_t>(std::min<std::uint64_t>(
        std::numeric_limits<std::uint32_t>::max(),
        std::max<std::uint64_t>(capacity.MaxNodeTasks,
                                nodeBudget)));
    capacity.MaxGroupTasks = static_cast<std::uint32_t>(std::min<std::uint64_t>(
        std::numeric_limits<std::uint32_t>::max(),
        std::max<std::uint64_t>(capacity.MaxGroupTasks,
                                groupBudget)));
    capacity.MaxClusterTasks = static_cast<std::uint32_t>(std::min<std::uint64_t>(
        std::numeric_limits<std::uint32_t>::max(),
        std::max<std::uint64_t>(capacity.MaxClusterTasks, clusterBudget)));
    capacity.MaxDrawCommands = static_cast<std::uint32_t>(std::min<std::uint64_t>(
        std::numeric_limits<std::uint32_t>::max(), drawBudget));

    // Every node and group task is consumed exactly once, so this is the total
    // work the persistent traversal has to get through in one dispatch.
    const std::uint64_t traversalTasks =
        static_cast<std::uint64_t>(capacity.MaxNodeTasks) + capacity.MaxGroupTasks;
    // The traversal shares one queue cursor, so extra workgroups buy more cores at
    // the price of more contention on it. Measured on 27 rabbits after the
    // wave-aggregated claim: 1 group ~0.5 ms, 2 ~1.0 ms, 8 ~1.05 ms, 32 ~1.0 ms.
    // So stay at one group until the work no longer fits in one group's iteration
    // budget, then add groups only as fast as the task count grows. 65536 tasks
    // per group is 1024 iterations of a fully fed 64-thread group.
    capacity.PersistentWorkgroups = static_cast<std::uint32_t>(std::clamp<std::uint64_t>(
        (traversalTasks + 65535u) / 65536u, 1u, 64u));
    // Sized so a single workgroup could drain the whole scene by itself: the
    // queue is shared, so nothing stops one group from claiming most of the
    // tasks while the others spin. The extra 1024 covers the iterations burnt
    // spinning while producers publish. Overshooting is free because the loop
    // exits on the pending test as soon as the queues drain; undershooting is
    // what made 512 rabbits blink.
    capacity.TraversalIterations = static_cast<std::uint32_t>(std::clamp<std::uint64_t>(
        traversalTasks / 64u + 1024u, 1024u, 1u << 20u));
    if (const char* groups = std::getenv("NANITE_PERSISTENT_GROUPS"))
    {
        const std::uint64_t requested = std::strtoull(groups, nullptr, 10);
        if (requested == 0u)
            throw std::runtime_error{"NANITE_PERSISTENT_GROUPS must be greater than zero"};
        capacity.PersistentWorkgroups = static_cast<std::uint32_t>(
            std::min<std::uint64_t>(std::numeric_limits<std::uint32_t>::max(), requested));
    }
    if (const char* iterations = std::getenv("NANITE_TRAVERSAL_ITERATIONS"))
    {
        const std::uint64_t requested = std::strtoull(iterations, nullptr, 10);
        if (requested == 0u)
            throw std::runtime_error{"NANITE_TRAVERSAL_ITERATIONS must be greater than zero"};
        capacity.TraversalIterations = static_cast<std::uint32_t>(
            std::min<std::uint64_t>(std::numeric_limits<std::uint32_t>::max(), requested));
    }
    return capacity;
}

void ConfigureMoltenVKQueueFamilies()
{
#if PLATFORM_MACOS || PLATFORM_IOS || PLATFORM_TVOS
    // These are the normal demo defaults. Keep every value overrideable from
    // the shell, but make the verified 512-instance/native-atomic configuration
    // work when the app is launched directly from Finder or the IDE.
    if (std::getenv("MVK_CONFIG_SPECIALIZED_QUEUE_FAMILIES") == nullptr)
        setenv("MVK_CONFIG_SPECIALIZED_QUEUE_FAMILIES", "0", 1);
    if (std::getenv("NANITE_ALLOW_SERIAL_SOFT_RASTER") == nullptr)
        setenv("NANITE_ALLOW_SERIAL_SOFT_RASTER", "1", 1);
    if (std::getenv("NANITE_RASTER_BIN_AREA") == nullptr)
        setenv("NANITE_RASTER_BIN_AREA", "256", 1);
    if (std::getenv("NANITE_GPU_TIMINGS") == nullptr)
        setenv("NANITE_GPU_TIMINGS", "1", 1);
    if (std::getenv("NANITE_NO_VSYNC") == nullptr &&
        std::getenv("NANITE_VSYNC") == nullptr)
        setenv("NANITE_NO_VSYNC", "1", 1);

    // This is an explicit startup choice. Enabling specialized families changes
    // MoltenVK's queue topology and can lower performance on a unified GPU even
    // when a compute-only family is available. The control panel can toggle use of
    // an already-created async context, but it cannot recreate the Vulkan device.
    const char* configured = std::getenv("MVK_CONFIG_SPECIALIZED_QUEUE_FAMILIES");
    std::cerr << "Nanite queue family configuration: "
              << (configured != nullptr ? configured : "0 (MoltenVK default)") << '\n';

    // The hand-written Nanite MSL kernels use explicit [[buffer(n)]] indices.
    // MoltenVK's argument-buffer mode collapses descriptor bindings into one
    // argument buffer, so those indices no longer address the resources reflected
    // by the SPIR-V layout. Make the native 64-bit path self-contained: callers can
    // still override this explicitly, but a native request never silently runs with
    // incompatible binding mode.
    const char* native64 = std::getenv("NANITE_NATIVE_64BIT_VISIBILITY");
    if (native64 == nullptr)
    {
        setenv("NANITE_NATIVE_64BIT_VISIBILITY", "1", 1);
        native64 = std::getenv("NANITE_NATIVE_64BIT_VISIBILITY");
    }
    const bool native64Requested = native64 != nullptr && native64[0] != '\0' &&
        std::strcmp(native64, "0") != 0;
    if (native64Requested)
    {
        if (std::getenv("DILIGENT_MSL_OVERRIDE_DIR") == nullptr)
            setenv("DILIGENT_MSL_OVERRIDE_DIR", DEMO_NANITE_MSL_DIR, 1);
        const char* argumentBuffers = std::getenv("MVK_CONFIG_USE_METAL_ARGUMENT_BUFFERS");
        if (argumentBuffers == nullptr || std::strcmp(argumentBuffers, "0") != 0)
        {
            setenv("MVK_CONFIG_USE_METAL_ARGUMENT_BUFFERS", "0", 1);
            std::cerr << "Nanite native 64-bit visibility: forcing "
                         "MVK_CONFIG_USE_METAL_ARGUMENT_BUFFERS=0 for explicit MSL bindings\n";
        }
    }
    else if (std::getenv("DILIGENT_MSL_OVERRIDE_DIR") != nullptr)
    {
        // This demo directory contains the Nanite depth/visibility overrides, not
        // a general shader replacement set. Leaving it active while native mode is
        // off would mix the Metal depth kernel with the portable second pass and
        // produce an invalid visibility buffer. Fall back atomically to SPIR-V.
        unsetenv("DILIGENT_MSL_OVERRIDE_DIR");
        std::cerr << "Nanite native 64-bit visibility: ignoring DILIGENT_MSL_OVERRIDE_DIR "
                     "because NANITE_NATIVE_64BIT_VISIBILITY is disabled\n";
    }
#endif
}

} // namespace

namespace NaniteDemo
{

Renderer::Renderer(const NativeWindow& window, std::uint32_t width, std::uint32_t height) :
    m_Window{window}
{
    // This must happen before Diligent creates or enumerates the Vulkan instance;
    // MoltenVK reads its queue-family configuration during initialization.
    ConfigureMoltenVKQueueFamilies();

    IEngineFactoryVk* factory = GetEngineFactoryVk();
    if (factory == nullptr)
        throw std::runtime_error{"Diligent Vulkan factory is unavailable"};

    // Prefer a genuinely dedicated compute queue for the software rasterizer,
    // then use a different compute-capable queue family when the driver exposes
    // no pure-compute family. Queue ids are Vulkan queue-family ids in Diligent's
    // Vulkan backend, so adapter enumeration is the authoritative source. Devices
    // with only one queue family stay on the old single-context path.
    Uint32 adapterCount = 0;
    factory->EnumerateAdapters(Version{}, adapterCount, nullptr);
    std::vector<GraphicsAdapterInfo> adapters(adapterCount);
    if (adapterCount != 0u)
        factory->EnumerateAdapters(Version{}, adapterCount, adapters.data());

    Uint8 graphicsQueue = DEFAULT_QUEUE_ID;
    Uint8 dedicatedComputeQueue = DEFAULT_QUEUE_ID;
    Uint8 secondaryComputeQueue = DEFAULT_QUEUE_ID;
    if (!adapters.empty())
    {
        const GraphicsAdapterInfo& adapter = adapters.front();
        for (Uint32 queue = 0; queue < adapter.NumQueues; ++queue)
        {
            const COMMAND_QUEUE_TYPE type = adapter.Queues[queue].QueueType;
            if (graphicsQueue == DEFAULT_QUEUE_ID &&
                (type & COMMAND_QUEUE_TYPE_GRAPHICS) == COMMAND_QUEUE_TYPE_GRAPHICS)
                graphicsQueue = static_cast<Uint8>(queue);
            if (dedicatedComputeQueue == DEFAULT_QUEUE_ID &&
                (type & COMMAND_QUEUE_TYPE_COMPUTE) == COMMAND_QUEUE_TYPE_COMPUTE &&
                (type & COMMAND_QUEUE_TYPE_GRAPHICS) != COMMAND_QUEUE_TYPE_GRAPHICS)
                dedicatedComputeQueue = static_cast<Uint8>(queue);
            if (secondaryComputeQueue == DEFAULT_QUEUE_ID &&
                graphicsQueue != DEFAULT_QUEUE_ID && queue != graphicsQueue &&
                (type & COMMAND_QUEUE_TYPE_COMPUTE) == COMMAND_QUEUE_TYPE_COMPUTE)
                secondaryComputeQueue = static_cast<Uint8>(queue);
        }

        std::cerr << "Nanite queue families: graphics=" << static_cast<unsigned>(graphicsQueue)
                  << " dedicatedCompute=" << static_cast<unsigned>(dedicatedComputeQueue)
                  << " secondaryCompute=" << static_cast<unsigned>(secondaryComputeQueue)
                  << " total=" << adapter.NumQueues << '\n';
    }

    // A second graphics+compute family is not sufficient evidence of async
    // overlap. MoltenVK commonly exposes several such families while mapping
    // them to one underlying Metal scheduler, which only adds fence/submit
    // overhead. Prefer a compute-only family when the caller explicitly enables
    // async raster; otherwise keep the graphics-only path for a clean baseline.
    const bool forceSecondaryAsync = std::getenv("NANITE_FORCE_ASYNC_RASTER") != nullptr;
    const Uint8 computeQueue = dedicatedComputeQueue != DEFAULT_QUEUE_ID ?
        dedicatedComputeQueue : (forceSecondaryAsync ? secondaryComputeQueue : DEFAULT_QUEUE_ID);

    EngineVkCreateInfo engineCI;
    ImmediateContextCreateInfo contextInfo[2];
    IDeviceContext* createdContexts[2] = {};
    if (graphicsQueue != DEFAULT_QUEUE_ID && computeQueue != DEFAULT_QUEUE_ID)
    {
        contextInfo[0] = ImmediateContextCreateInfo{"Nanite graphics context", graphicsQueue};
        contextInfo[1] = ImmediateContextCreateInfo{"Nanite async compute context", computeQueue};
        engineCI.NumImmediateContexts = 2;
        engineCI.pImmediateContextInfo = contextInfo;
    }
    else
    {
        engineCI.NumImmediateContexts = 0;
    }
    engineCI.SetValidationLevel(VALIDATION_LEVEL_DISABLED);
    factory->CreateDeviceAndContextsVk(
        engineCI,
        &m_Device,
        engineCI.NumImmediateContexts > 0 ? createdContexts : &createdContexts[0]);
    if (engineCI.NumImmediateContexts > 0 &&
        (m_Device == nullptr || createdContexts[0] == nullptr || createdContexts[1] == nullptr))
    {
        // A queue family can be advertised by the adapter but still reject a
        // multi-queue logical-device configuration on a portability layer. Retry
        // with the guaranteed graphics-only setup before reporting device failure.
        for (IDeviceContext*& createdContext : createdContexts)
        {
            if (createdContext != nullptr)
            {
                createdContext->Release();
                createdContext = nullptr;
            }
        }
        m_Device.Release();
        engineCI.NumImmediateContexts = 0;
        engineCI.pImmediateContextInfo = nullptr;
        factory->CreateDeviceAndContextsVk(engineCI, &m_Device, createdContexts);
    }
    if (engineCI.NumImmediateContexts > 0)
    {
        m_Context.Attach(createdContexts[0]);
        m_AsyncComputeContext.Attach(createdContexts[1]);
        m_AsyncRasterAvailable = m_Context != nullptr && m_AsyncComputeContext != nullptr;
    }
    else
    {
        m_Context.Attach(createdContexts[0]);
    }
    if (m_Device == nullptr || m_Context == nullptr)
        throw std::runtime_error{"Diligent failed to create the Vulkan device"};

    // Exposing a dedicated family is not enough reason to use it by default:
    // on a unified GPU the extra submit and synchronization can cost more than
    // the overlap saves. NANITE_ASYNC_RASTER=1 is the explicit startup opt-in;
    // the Control panel can toggle it live after the device is created.
    m_AsyncRasterEnabled = m_AsyncRasterAvailable &&
        std::getenv("NANITE_ASYNC_RASTER") != nullptr;

    std::cerr << "Nanite async raster: "
              << (m_AsyncRasterAvailable ?
                      (dedicatedComputeQueue != DEFAULT_QUEUE_ID ?
                           (m_AsyncRasterEnabled ?
                                "dedicated compute queue available (enabled)" :
                                "dedicated compute queue available (disabled)") :
                           (m_AsyncRasterEnabled ?
                                "forced secondary compute-capable queue (enabled)" :
                                "forced secondary compute-capable queue (disabled)")) :
                                             (secondaryComputeQueue != DEFAULT_QUEUE_ID &&
                                                      !forceSecondaryAsync ?
                                                  "no dedicated compute queue; serial fallback" :
                                                  "no separate compute queue; serial fallback"))
              << '\n';

    m_SwapChainDesc.Width = width;
    m_SwapChainDesc.Height = height;
    m_SwapChainDesc.ColorBufferFormat = TEX_FORMAT_RGBA8_UNORM;
    m_SwapChainDesc.DepthBufferFormat = TEX_FORMAT_UNKNOWN;
    // Three images, so the CPU always has one to draw into while the display holds
    // one and the GPU finishes another. With two, the CPU has to wait for the
    // display to release its image before it can start, which is the stall that
    // makes a vsync-locked frame rate feel worse than the frame time says it is.
    m_SwapChainDesc.BufferCount = 3;
    m_SwapChainDesc.PreTransform = SURFACE_TRANSFORM_IDENTITY;
    factory->CreateSwapChainVk(
        m_Device, m_Context, m_SwapChainDesc, m_Window, &m_SwapChain);
    if (m_SwapChain == nullptr)
        throw std::runtime_error{"Diligent failed to create the Vulkan swap chain"};
    m_SwapChainDesc = m_SwapChain->GetDesc();

    // CPU_WAIT_ONLY: the only waiter is Draw, on the main thread. A GPU-waitable
    // fence would need the NativeFence feature for nothing in return.
    FenceDesc fenceDesc;
    fenceDesc.Name = "Nanite frame pacing fence";
    fenceDesc.Type = FENCE_TYPE_CPU_WAIT_ONLY;
    m_Device->CreateFence(fenceDesc, &m_FrameFence);
    if (m_FrameFence == nullptr)
        throw std::runtime_error{"Diligent failed to create the frame pacing fence"};

    if (m_AsyncRasterAvailable)
    {
        FenceDesc asyncFenceDesc;
        asyncFenceDesc.Type = FENCE_TYPE_GENERAL;
        asyncFenceDesc.Name = "Nanite async raster bin ready";
        m_Device->CreateFence(asyncFenceDesc, &m_AsyncBinReadyFence);
        asyncFenceDesc.Name = "Nanite async raster complete";
        m_Device->CreateFence(asyncFenceDesc, &m_AsyncSoftRasterDoneFence);
        if (m_AsyncBinReadyFence == nullptr || m_AsyncSoftRasterDoneFence == nullptr)
        {
            std::cerr << "Nanite async raster: GPU fences unavailable; serial fallback\n";
            m_AsyncBinReadyFence.Release();
            m_AsyncSoftRasterDoneFence.Release();
            m_AsyncComputeContext.Release();
            m_AsyncRasterAvailable = false;
        }
    }

    m_NanitePipelines = std::make_unique<Nanite::Pipelines>(
        m_Device,
        m_SwapChainDesc.ColorBufferFormat,
        TEX_FORMAT_D32_FLOAT,
        DEMO_SHADER_DIR,
        m_AsyncRasterAvailable ? 3u : 1u);
    ReloadScene(0);
}

Renderer::~Renderer()
{
    if (m_Context != nullptr)
        m_Context->WaitForIdle();
    if (m_AsyncComputeContext != nullptr)
        m_AsyncComputeContext->WaitForIdle();
}

void Renderer::ReloadRabbitCount(std::uint32_t count)
{
    // Existing allocations are reused for ordinary count changes. Explicit grid
    // edits use SetInstanceGrid, which can grow the allocation when necessary.
    if (count == 0u || m_NaniteScene == nullptr)
        return;
    count = std::min(count, m_NaniteScene->GetMaxInstanceCount());
    if (count == m_RabbitCount)
        return;
    m_NaniteScene->SetActiveInstanceCount(count);
    m_InstanceGridExplicit = false;
    m_RabbitCount = m_NaniteScene->GetActiveInstanceCount();
}

std::uint32_t Renderer::GetMaxRabbitCount() const
{
    return m_NaniteScene != nullptr ? m_NaniteScene->GetMaxInstanceCount() : 0u;
}

void Renderer::ReloadScene(std::uint32_t count)
{
    if (m_Context != nullptr)
        m_Context->WaitForIdle();
    if (m_AsyncComputeContext != nullptr)
        m_AsyncComputeContext->WaitForIdle();

    // Build at the maximum count the UI can ask for, then activate the count we
    // actually want. Everything sized from the scene - queues, draw commands,
    // instance buffer - is therefore provisioned for the worst case up front,
    // which is what lets the count change without reallocating anything.
    std::uint32_t requested = count == 0u ? m_RabbitCount : count;
    if (count == 0u)
    {
        m_InstanceGridExplicit = std::getenv("NANITE_RABBITS") == nullptr;
        if (const char* startCount = std::getenv("NANITE_RABBITS"))
        {
            const std::uint64_t parsed = std::strtoull(startCount, nullptr, 10);
            if (parsed == 0u)
                throw std::runtime_error{"NANITE_RABBITS must be greater than zero"};
            requested = static_cast<std::uint32_t>(parsed);
        }
    }
    const std::string modelPath = m_ModelPath.empty() ? Nanite::ResolveModelPath() : m_ModelPath;
    const bool coastalScene = modelPath == "coastal_scene";
    if (coastalScene)
    {
        // The coastal composition contains the authored 1024/1280/384 instance
        // layers. Keep the control panel count from applying a rabbit grid.
        requested = 2688u;
        m_RabbitCount = requested;
        m_InstanceGridExplicit = false;
    }
    const std::string defaultBuddhaPath = std::string{DEMO_ASSET_DIR} + "/happy_vrip.ply";
    // A Buddha has roughly sixteen times the source triangles of the old bunny.
    // Reserving every per-instance traversal and draw queue for 10,000 copies can
    // consume several GB before the first frame. Keep the normal 27-copy scene
    // comfortably provisioned; an explicit environment override still permits a
    // larger stress run.
    // Coastal is a multi-prototype scene whose 2688 transforms are authored by
    // the loader, rather than copies of one mesh.
    std::uint32_t maxCount = coastalScene ? 2688u :
        (modelPath == defaultBuddhaPath ? 64u : 10000u);
    if (const char* maxEnv = std::getenv("NANITE_MAX_RABBITS"))
    {
        const std::uint64_t parsed = std::strtoull(maxEnv, nullptr, 10);
        if (parsed == 0u)
            throw std::runtime_error{"NANITE_MAX_RABBITS must be greater than zero"};
        maxCount = static_cast<std::uint32_t>(parsed);
    }
    maxCount = std::max(maxCount, requested);

    m_ModelPath = modelPath;
    const Nanite::CpuScene scene =
        Nanite::LoadModelAndBuildScene(modelPath, maxCount, m_ClusterTriangles);
    if (scene.Instances.empty() || scene.Nodes.empty() || scene.Clusters.empty())
        throw std::runtime_error{"Nanite scene failed to build from " + modelPath};

    const Nanite::Capacity capacity = BuildCapacity(scene);
    std::cerr << "Nanite capacity: node=" << capacity.MaxNodeTasks
              << " group=" << capacity.MaxGroupTasks
              << " cluster=" << capacity.MaxClusterTasks
              << " draw=" << capacity.MaxDrawCommands
              << " persistentWorkgroups=" << capacity.PersistentWorkgroups
              << " traversalIterations=" << capacity.TraversalIterations
              << " maxInstances=" << scene.Instances.size() << '\n';
    m_NaniteScene = std::make_unique<Nanite::GpuScene>(
        m_Device,
        *m_NanitePipelines,
        scene,
        capacity,
        m_AsyncRasterAvailable ? 3u : 1u);
    m_NaniteScene->SetHZBEnabled(m_HZBEnabled);
    m_NaniteScene->SetGpuTimingsEnabled(m_GpuTimingsEnabled);
    if (m_SceneSettingsSeeded)
    {
        m_NaniteScene->SetScreenErrorPixels(m_ScreenErrorPixels);
        m_NaniteScene->SetShadingMode(m_ShadingMode);
        m_NaniteScene->SetRasterBinAreaCutoff(m_RasterBinAreaCutoff);
        m_NaniteScene->SetAtmosphereSettings(m_AtmosphereSettings);
        m_NaniteScene->SetInstanceSpacing(m_SpacingHorizontal, m_SpacingVertical);
        if (m_InstanceGridExplicit)
            m_NaniteScene->SetInstanceGrid(m_InstanceCountX, m_InstanceCountY, m_InstanceCountZ);
    }
    else
    {
        // The scene's constructor is what reads the environment, so take the
        // startup values from it rather than duplicating that parsing here.
        m_ScreenErrorPixels = m_NaniteScene->GetScreenErrorPixels();
        m_ShadingMode = m_NaniteScene->GetShadingMode();
        m_RasterBinAreaCutoff = m_NaniteScene->GetRasterBinAreaCutoff();
        m_AtmosphereSettings = m_NaniteScene->GetAtmosphereSettings();
        m_SpacingHorizontal = m_NaniteScene->GetSpacingHorizontal();
        m_SpacingVertical = m_NaniteScene->GetSpacingVertical();
        m_SceneSettingsSeeded = true;
    }
    m_NaniteScene->SetActiveInstanceCount(requested);
    if (m_InstanceGridExplicit)
    {
        const std::uint64_t gridTotal = static_cast<std::uint64_t>(m_InstanceCountX) *
            m_InstanceCountY * m_InstanceCountZ;
        if (gridTotal > scene.Instances.size())
            throw std::runtime_error{"Default instance grid exceeds the scene instance capacity"};
        m_NaniteScene->SetInstanceGrid(m_InstanceCountX, m_InstanceCountY, m_InstanceCountZ);
    }
    m_RabbitCount = m_NaniteScene->GetActiveInstanceCount();
}

void Renderer::SetScreenErrorPixels(float pixels)
{
    m_ScreenErrorPixels = pixels;
    if (m_NaniteScene != nullptr)
        m_NaniteScene->SetScreenErrorPixels(pixels);
}

float Renderer::GetScreenErrorPixels() const
{
    return m_ScreenErrorPixels;
}

void Renderer::SetShadingMode(std::uint32_t mode)
{
    m_ShadingMode = mode;
    if (m_NaniteScene != nullptr)
        m_NaniteScene->SetShadingMode(mode);
}

std::uint32_t Renderer::GetShadingMode() const
{
    return m_ShadingMode;
}

void Renderer::SetRasterBinAreaCutoff(float cutoff)
{
    m_RasterBinAreaCutoff = std::max(cutoff, 0.0f);
    if (m_NaniteScene != nullptr)
    {
        m_NaniteScene->SetRasterBinAreaCutoff(m_RasterBinAreaCutoff);
        m_RasterBinAreaCutoff = m_NaniteScene->GetRasterBinAreaCutoff();
    }
}

float Renderer::GetRasterBinAreaCutoff() const
{
    return m_RasterBinAreaCutoff;
}

void Renderer::SetAtmosphereSettings(const Nanite::AtmosphereSettings& settings)
{
    m_AtmosphereSettings = settings;
    if (m_NaniteScene != nullptr)
    {
        m_NaniteScene->SetAtmosphereSettings(settings);
        m_AtmosphereSettings = m_NaniteScene->GetAtmosphereSettings();
    }
}

void Renderer::ReloadSceneFromSettings()
{
    // Zero keeps the active count: the rebuild is about the geometry, and losing
    // the instance count on every model change would be its own annoyance.
    ReloadScene(m_RabbitCount);
}

void Renderer::SetCamera(float x, float y, float z, float yaw, float pitch)
{
    m_CameraX = x;
    m_CameraY = y;
    m_CameraZ = z;
    m_CameraYaw = yaw;
    m_CameraPitch = pitch;
}

void Renderer::SetHZBEnabled(bool enabled)
{
    m_HZBEnabled = enabled;
    if (m_NaniteScene != nullptr)
        m_NaniteScene->SetHZBEnabled(enabled);
}

void Renderer::SetGpuTimingsEnabled(bool enabled)
{
    // Remembered so a scene rebuild does not silently drop the setting.
    m_GpuTimingsEnabled = enabled;
    if (m_NaniteScene != nullptr)
        m_NaniteScene->SetGpuTimingsEnabled(enabled);
}

void Renderer::SetAsyncRasterEnabled(bool enabled)
{
    m_AsyncRasterEnabled = enabled && m_AsyncRasterAvailable;
}

bool Renderer::AreGpuTimingsSupported() const
{
    return m_NaniteScene != nullptr && m_NaniteScene->AreGpuTimingsSupported();
}

const Nanite::FrameStats& Renderer::GetLastFrameStats() const
{
    return m_NaniteScene->GetLastFrameStats();
}

void Renderer::Draw(std::uint32_t width, std::uint32_t height)
{
    if (m_SwapChain == nullptr || m_Context == nullptr || width == 0u || height == 0u)
        return;

    // Block until the frame kFramesInFlight back has finished, before touching
    // anything this frame writes. Two frames may still be in flight afterwards, so
    // the GPU keeps working; what this rules out is the CPU running arbitrarily far
    // ahead, which is what turns into input lag and into a growing pile of queued
    // command buffers.
    if (m_FrameNumber >= kFramesInFlight)
        m_FrameFence->Wait(m_FrameNumber - kFramesInFlight + 1u);

    const auto& currentDesc = m_SwapChain->GetDesc();
    if (currentDesc.Width != width || currentDesc.Height != height)
        m_SwapChain->Resize(width, height, SURFACE_TRANSFORM_IDENTITY);

    ITextureView* renderTarget = m_SwapChain->GetCurrentBackBufferRTV();
    m_Context->SetRenderTargets(1, &renderTarget, nullptr, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
    constexpr float clearColor[] = {0.035f, 0.045f, 0.065f, 1.0f};
    m_Context->ClearRenderTarget(
        renderTarget, clearColor, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);

    const auto viewProj = BuildViewProjection(
        m_CameraX, m_CameraY, m_CameraZ, m_CameraYaw, m_CameraPitch, width, height);
    const float cameraWorldPosition[3] = {m_CameraX, m_CameraY, m_CameraZ};
    Nanite::GpuScene::AsyncRasterContext asyncRaster;
    const Nanite::GpuScene::AsyncRasterContext* asyncRasterPtr = nullptr;
    if (m_AsyncRasterAvailable && m_AsyncRasterEnabled)
    {
        asyncRaster.Context = m_AsyncComputeContext;
        asyncRaster.BinReady = m_AsyncBinReadyFence;
        asyncRaster.SoftRasterDone = m_AsyncSoftRasterDoneFence;
        asyncRaster.SignalValue = m_FrameNumber + 1u;
        asyncRasterPtr = &asyncRaster;
    }
    m_NaniteScene->Render(
        m_Context,
        renderTarget,
        width,
        height,
        viewProj.data(),
        cameraWorldPosition,
        m_CameraYaw,
        m_CameraPitch,
        asyncRasterPtr);
    // Enqueued before Present because Present is what flushes the context; the
    // signal therefore lands at the end of this frame's submission.
    ++m_FrameNumber;
    m_Context->EnqueueSignal(m_FrameFence, m_FrameNumber);
    m_SwapChain->Present(m_VSyncEnabled ? 1u : 0u);
}

void Renderer::SetInstanceGrid(std::uint32_t countX,
                                std::uint32_t countY,
                                std::uint32_t countZ)
{
    if (countX == 0u || countY == 0u || countZ == 0u)
        throw std::runtime_error{"Instance grid dimensions must be greater than zero"};
    const std::uint64_t totalXY = static_cast<std::uint64_t>(countX) * countY;
    if (totalXY > std::numeric_limits<std::uint32_t>::max() / countZ)
        throw std::runtime_error{"Instance grid total exceeds the 32-bit instance counter"};
    const std::uint64_t total = totalXY * countZ;

    // The UI is deliberately not allowed to rewrite the requested dimensions to
    // fit the current allocation. Grow the scene to the requested total instead;
    // ReloadScene provisions all instance-dependent queues and buffers for it.
    if (m_NaniteScene != nullptr && total > GetMaxRabbitCount())
    {
        const bool hadExplicitGrid = m_InstanceGridExplicit;
        m_InstanceGridExplicit = false;
        try
        {
            ReloadScene(static_cast<std::uint32_t>(total));
        }
        catch (...)
        {
            m_InstanceGridExplicit = hadExplicitGrid;
            throw;
        }
        m_InstanceGridExplicit = hadExplicitGrid;
    }
    if (m_NaniteScene != nullptr)
    {
        m_NaniteScene->SetInstanceGrid(countX, countY, countZ);
        m_InstanceCountX = m_NaniteScene->GetInstanceCountX();
        m_InstanceCountY = m_NaniteScene->GetInstanceCountY();
        m_InstanceCountZ = m_NaniteScene->GetInstanceCountZ();
        m_RabbitCount = m_NaniteScene->GetActiveInstanceCount();
    }
    else
    {
        m_InstanceCountX = countX;
        m_InstanceCountY = countY;
        m_InstanceCountZ = countZ;
        m_RabbitCount = static_cast<std::uint32_t>(total);
    }
    m_InstanceGridExplicit = true;
}

void Renderer::SetInstanceSpacing(float horizontal, float vertical)
{
    m_SpacingHorizontal = horizontal;
    m_SpacingVertical = vertical;
    if (m_NaniteScene != nullptr)
    {
        m_NaniteScene->SetInstanceSpacing(horizontal, vertical);
        // The scene clamps, so read back rather than trusting what went in.
        m_SpacingHorizontal = m_NaniteScene->GetSpacingHorizontal();
        m_SpacingVertical = m_NaniteScene->GetSpacingVertical();
    }
}

} // namespace NaniteDemo
