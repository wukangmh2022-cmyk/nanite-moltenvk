#import <AppKit/AppKit.h>
#import <QuartzCore/CAMetalLayer.h>
#import <QuartzCore/CADisplayLink.h>
#import <UniformTypeIdentifiers/UniformTypeIdentifiers.h>

#include <algorithm>
#include <DeviceContext.h>
#include <EngineFactoryVk.h>
#include <NativeWindow.h>
#include <PipelineState.h>
#include <RefCntAutoPtr.hpp>
#include <RenderDevice.h>
#include <Shader.h>
#include <SwapChain.h>

#include "NanitePipelines.hpp"
#include "NaniteGpuScene.hpp"
#include "NaniteModel.hpp"
#include "DemoRenderer.hpp"

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <fstream>
#include <memory>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

using namespace Diligent;

namespace
{

// TriangleScreenArea is measured in physical pixels squared. Keep a physical
// ceiling that covers a full-screen triangle on a normal Retina viewport; the
// control panel maps this range nonlinearly so its everyday values stay precise.
constexpr double kRasterCutoffSliderMax = 2097152.0;
// Most useful experiments live in the first few square pixels. Keep that range
// easy to scrub, while reserving the final part of the normalized slider for the
// large values needed to force every triangle through software rasterization.
constexpr double kRasterCutoffComfortMax = 16.0;
constexpr double kRasterCutoffComfortEnd = 0.85;

double RasterCutoffSliderToValue(double position)
{
    position = std::clamp(position, 0.0, 1.0);
    if (position <= kRasterCutoffComfortEnd)
        return position / kRasterCutoffComfortEnd * kRasterCutoffComfortMax;

    const double tail = (position - kRasterCutoffComfortEnd) /
        (1.0 - kRasterCutoffComfortEnd);
    // A fourth-power tail keeps the high range at the extreme right instead of
    // making ordinary mouse motion jump straight to thousands of px2.
    return kRasterCutoffComfortMax * std::pow(
        kRasterCutoffSliderMax / kRasterCutoffComfortMax, std::pow(tail, 4.0));
}

double RasterCutoffValueToSlider(double value)
{
    value = std::max(0.0, value);
    if (value <= kRasterCutoffComfortMax)
        return std::clamp(value / kRasterCutoffComfortMax * kRasterCutoffComfortEnd, 0.0, 1.0);
    if (value >= kRasterCutoffSliderMax)
        return 1.0;

    const double exponent = std::log(value / kRasterCutoffComfortMax) /
        std::log(kRasterCutoffSliderMax / kRasterCutoffComfortMax);
    const double tail = std::pow(std::max(0.0, exponent), 0.25);
    return std::clamp(kRasterCutoffComfortEnd +
                          tail * (1.0 - kRasterCutoffComfortEnd),
                      0.0, 1.0);
}

constexpr std::uint32_t InstanceCountMin = 1u;
constexpr std::uint32_t InstanceCountSliderMax = 50u;
constexpr NSInteger AtmosphereControlCount = 10;
// The actual practical limit is GPU memory. The renderer grows its instance
// buffers on demand, so the controls must not impose a smaller editorial limit.
constexpr std::uint32_t InstanceCountMax = std::numeric_limits<std::uint32_t>::max();

std::uint32_t SliderPositionToInstanceCount(double position,
                                            std::uint32_t minCount = InstanceCountMin,
                                            std::uint32_t maxCount = InstanceCountMax)
{
    if (maxCount <= minCount)
        return minCount;
    if (!std::isfinite(position) || position <= 0.0)
        return minCount;
    if (position >= 1.0)
        return maxCount;

    const double logarithmicCount = std::log(static_cast<double>(minCount)) +
        position * (std::log(static_cast<double>(maxCount)) - std::log(static_cast<double>(minCount)));
    if (!std::isfinite(logarithmicCount))
        return minCount;

    const double count = std::exp(logarithmicCount);
    if (!std::isfinite(count))
        return maxCount;
    return static_cast<std::uint32_t>(std::clamp<long long>(
        std::llround(count), minCount, maxCount));
}

double InstanceCountToSliderPosition(std::uint32_t count,
                                     std::uint32_t minCount = InstanceCountMin,
                                     std::uint32_t maxCount = InstanceCountMax)
{
    if (maxCount <= minCount || count <= minCount)
        return 0.0;
    if (count >= maxCount)
        return 1.0;

    const double denominator = std::log(static_cast<double>(maxCount)) -
        std::log(static_cast<double>(minCount));
    if (!std::isfinite(denominator) || denominator <= 0.0)
        return 0.0;

    const double position = (std::log(static_cast<double>(count)) -
        std::log(static_cast<double>(minCount))) / denominator;
    if (!std::isfinite(position))
        return 0.0;
    return std::clamp(position, 0.0, 1.0);
}

std::vector<Uint8> ReadBinaryFile(const char* Path)
{
    std::ifstream File{Path, std::ios::binary | std::ios::ate};
    if (!File)
        throw std::runtime_error(std::string{"Unable to open shader: "} + Path);

    const auto Size = File.tellg();
    if (Size <= 0)
        throw std::runtime_error(std::string{"Shader is empty: "} + Path);

    std::vector<Uint8> Data(static_cast<size_t>(Size));
    File.seekg(0, std::ios::beg);
    File.read(reinterpret_cast<char*>(Data.data()), Size);
    if (!File)
        throw std::runtime_error(std::string{"Unable to read shader: "} + Path);
    return Data;
}

struct Float3
{
    float x;
    float y;
    float z;
};

float Dot(Float3 A, Float3 B)
{
    return A.x * B.x + A.y * B.y + A.z * B.z;
}

Float3 Cross(Float3 A, Float3 B)
{
    return {
        A.y * B.z - A.z * B.y,
        A.z * B.x - A.x * B.z,
        A.x * B.y - A.y * B.x};
}

std::array<float, 16> BuildViewProjection(
    float CameraX,
    float CameraY,
    float CameraZ,
    float Yaw,
    float Pitch,
    Uint32 Width,
    Uint32 Height)
{
    constexpr float NearPlane = 0.1f;
    constexpr float FarPlane = 200.0f;
    constexpr float VerticalFov = 60.0f * 3.14159265358979323846f / 180.0f;
    const float Aspect = static_cast<float>(Width) / static_cast<float>(std::max(Height, 1u));
    const float FocalY = 1.0f / std::tan(VerticalFov * 0.5f);
    const float FocalX = FocalY / Aspect;
    const float DepthScale = FarPlane / (NearPlane - FarPlane);
    const float DepthOffset = NearPlane * FarPlane / (NearPlane - FarPlane);

    const float CosYaw = std::cos(Yaw);
    const float SinYaw = std::sin(Yaw);
    const float CosPitch = std::cos(Pitch);
    const float SinPitch = std::sin(Pitch);
    const Float3 Forward{SinYaw * CosPitch, SinPitch, -CosYaw * CosPitch};
    const Float3 Right{CosYaw, 0.0f, SinYaw};
    const Float3 Up = Cross(Right, Forward);
    const Float3 Back{-Forward.x, -Forward.y, -Forward.z};

    const float View[16] = {
        Right.x, Right.y, Right.z, -Dot(Right, {CameraX, CameraY, CameraZ}),
        Up.x, Up.y, Up.z, -Dot(Up, {CameraX, CameraY, CameraZ}),
        Back.x, Back.y, Back.z, -Dot(Back, {CameraX, CameraY, CameraZ}),
        0.0f, 0.0f, 0.0f, 1.0f};
    const float Projection[16] = {
        FocalX, 0.0f, 0.0f, 0.0f,
        0.0f, FocalY, 0.0f, 0.0f,
        0.0f, 0.0f, DepthScale, DepthOffset,
        0.0f, 0.0f, -1.0f, 0.0f};
    float Clip[16] = {};
    for (int Row = 0; Row < 4; ++Row)
    {
        for (int Column = 0; Column < 4; ++Column)
        {
            for (int Element = 0; Element < 4; ++Element)
                Clip[Row * 4 + Column] += Projection[Row * 4 + Element] * View[Element * 4 + Column];
        }
    }

    // glslang lowers this HLSL path to vector-times-row-major-matrix SPIR-V,
    // so upload the transpose of the usual column-vector clip matrix.
    std::array<float, 16> ViewProj{};
    for (int Row = 0; Row < 4; ++Row)
        for (int Column = 0; Column < 4; ++Column)
            ViewProj[Row * 4 + Column] = Clip[Column * 4 + Row];
    return ViewProj;
}

class VulkanTriangleRenderer
{
public:
    explicit VulkanTriangleRenderer(NSView* View)
    {
        IEngineFactoryVk* Factory = GetEngineFactoryVk();
        if (Factory == nullptr)
            throw std::runtime_error{"Diligent Vulkan factory is unavailable"};

        EngineVkCreateInfo EngineCI;
        EngineCI.NumImmediateContexts = 0;
        EngineCI.SetValidationLevel(VALIDATION_LEVEL_DISABLED);
        Factory->CreateDeviceAndContextsVk(EngineCI, &m_Device, &m_Context);
        if (m_Device == nullptr || m_Context == nullptr)
            throw std::runtime_error{"Diligent failed to create the Vulkan device"};

        m_SwapChainDesc.Width = 1280;
        m_SwapChainDesc.Height = 720;
        m_SwapChainDesc.ColorBufferFormat = TEX_FORMAT_RGBA8_UNORM;
        m_SwapChainDesc.DepthBufferFormat = TEX_FORMAT_UNKNOWN;
        m_SwapChainDesc.BufferCount = 2;
        m_SwapChainDesc.PreTransform = SURFACE_TRANSFORM_IDENTITY;

        NativeWindow Window{(__bridge void*)View};
        Factory->CreateSwapChainVk(m_Device, m_Context, m_SwapChainDesc, Window, &m_SwapChain);
        if (m_SwapChain == nullptr)
            throw std::runtime_error{"Diligent failed to create the Vulkan swap chain"};
        m_SwapChainDesc = m_SwapChain->GetDesc();

        m_NanitePipelines = std::make_unique<Nanite::Pipelines>(
            m_Device,
            m_SwapChainDesc.ColorBufferFormat,
            TEX_FORMAT_D32_FLOAT,
            DEMO_SHADER_DIR);
        const std::string bunnyPath = Nanite::ResolveModelPath();
        const Nanite::CpuScene demoScene = Nanite::LoadModelAndBuildScene(bunnyPath);
        if (demoScene.Instances.empty() || demoScene.Nodes.empty() || demoScene.Clusters.empty())
            throw std::runtime_error{"Stanford Bunny scene failed to build"};
        Nanite::Capacity capacity{};
        const std::uint64_t instanceCount = demoScene.Instances.size();
        const bool stressRequested = std::getenv("NANITE_STRESS_INSTANCES") != nullptr ||
            std::getenv("NANITE_STRESS_TRIANGLES") != nullptr;
        const std::uint64_t clusterInstanceBudget = stressRequested ?
            std::min<std::uint64_t>(instanceCount, 2048u) :
            std::min<std::uint64_t>(instanceCount, 64u);
        const std::uint64_t clusterBudget = std::min<std::uint64_t>(
            1u << 20u,
            std::max<std::uint64_t>(
                capacity.MaxClusterTasks,
                static_cast<std::uint64_t>(demoScene.Clusters.size()) *
                    clusterInstanceBudget));
        // MoltenVK on this machine has no indirect-count draw support, so the
        // fallback path submits every fixed command slot twice. Keep stress
        // mode bounded by the normal draw budget instead of multiplying it by
        // the logical instance count; NANITE_STRESS_DRAW_CAP can raise this
        // when testing a wider camera view.
        std::uint64_t stressDrawCap = 3072u;
        if (const char* drawCap = std::getenv("NANITE_STRESS_DRAW_CAP"))
        {
            stressDrawCap = std::strtoull(drawCap, nullptr, 10);
            if (stressDrawCap == 0u)
                throw std::runtime_error{"NANITE_STRESS_DRAW_CAP must be greater than zero"};
        }
        const std::uint64_t drawBudget = stressRequested ?
            std::min<std::uint64_t>(clusterBudget, stressDrawCap) :
            std::max<std::uint64_t>(
                capacity.MaxDrawCommands,
                static_cast<std::uint64_t>(demoScene.Clusters.size()) *
                    std::min<std::uint64_t>(instanceCount, 64u));
        capacity.MaxNodeTasks = static_cast<std::uint32_t>(std::min<std::uint64_t>(
            std::numeric_limits<std::uint32_t>::max(),
            std::max<std::uint64_t>(capacity.MaxNodeTasks, instanceCount)));
        capacity.MaxGroupTasks = static_cast<std::uint32_t>(std::min<std::uint64_t>(
            std::numeric_limits<std::uint32_t>::max(),
            std::max<std::uint64_t>(capacity.MaxGroupTasks, instanceCount)));
        capacity.MaxClusterTasks = static_cast<std::uint32_t>(clusterBudget);
        capacity.MaxDrawCommands = static_cast<std::uint32_t>(std::min<std::uint64_t>(
            std::numeric_limits<std::uint32_t>::max(), drawBudget));
        capacity.PersistentWorkgroups = static_cast<std::uint32_t>(std::max<std::uint64_t>(
            1u, (instanceCount + 65535u) / 65536u));
        std::cerr << "Nanite capacity: node=" << capacity.MaxNodeTasks
                  << " group=" << capacity.MaxGroupTasks
                  << " cluster=" << capacity.MaxClusterTasks
                  << " draw=" << capacity.MaxDrawCommands
                  << " persistentWorkgroups=" << capacity.PersistentWorkgroups << '\n';
        m_NaniteScene = std::make_unique<Nanite::GpuScene>(
            m_Device,
            *m_NanitePipelines,
            demoScene,
            capacity);
        CreatePipeline();
    }

    ~VulkanTriangleRenderer()
    {
        if (m_Context != nullptr)
            m_Context->WaitForIdle();
    }

    void SetCamera(float X, float Y, float Z, float Yaw, float Pitch)
    {
        m_CameraX = X;
        m_CameraY = Y;
        m_CameraZ = Z;
        m_CameraYaw = Yaw;
        m_CameraPitch = Pitch;
    }

    const Nanite::FrameStats& GetLastFrameStats() const
    {
        return m_NaniteScene->GetLastFrameStats();
    }

    void Draw(NSView* View)
    {
        if (m_SwapChain == nullptr || m_Context == nullptr)
            return;

        const NSRect Bounds = View.bounds;
        const CGFloat Scale = View.window.backingScaleFactor;
        const Uint32 Width = static_cast<Uint32>(Bounds.size.width * Scale);
        const Uint32 Height = static_cast<Uint32>(Bounds.size.height * Scale);
        if (Width == 0 || Height == 0)
            return;

        const auto& CurrentDesc = m_SwapChain->GetDesc();
        if (CurrentDesc.Width != Width || CurrentDesc.Height != Height)
            m_SwapChain->Resize(Width, Height, SURFACE_TRANSFORM_IDENTITY);

        ITextureView* RTV = m_SwapChain->GetCurrentBackBufferRTV();
        m_Context->SetRenderTargets(1, &RTV, nullptr, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);

        constexpr float ClearColor[] = {0.035f, 0.045f, 0.065f, 1.0f};
        m_Context->ClearRenderTarget(RTV, ClearColor, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
        if (m_NaniteScene != nullptr)
        {
            const auto ViewProj = BuildViewProjection(
                m_CameraX, m_CameraY, m_CameraZ, m_CameraYaw, m_CameraPitch, Width, Height);
            const float CameraWorldPosition[3] = {m_CameraX, m_CameraY, m_CameraZ};
            m_NaniteScene->Render(
                m_Context,
                RTV,
                Width,
                Height,
                ViewProj.data(),
                CameraWorldPosition,
                m_CameraYaw,
                m_CameraPitch);
        }
        else
        {
            m_Context->SetPipelineState(m_PipelineState);
            m_Context->Draw(DrawAttribs{3, DRAW_FLAG_NONE});
        }
        // Presentation is uncapped by default so culling changes remain visible.
        // NANITE_VSYNC=1 opts into display synchronization; NANITE_NO_VSYNC keeps
        // its legacy explicit-off meaning when both variables are present.
        static const Diligent::Uint32 syncInterval =
            std::getenv("NANITE_VSYNC") != nullptr &&
                std::getenv("NANITE_NO_VSYNC") == nullptr ? 1u : 0u;
        m_SwapChain->Present(syncInterval);
    }

private:
    void CreatePipeline()
    {
        const std::string VertexPath = std::string{DEMO_SHADER_DIR} + "/triangle.vert.spv";
        const std::string FragmentPath = std::string{DEMO_SHADER_DIR} + "/triangle.frag.spv";
        const auto VertexSPIRV = ReadBinaryFile(VertexPath.c_str());
        const auto FragmentSPIRV = ReadBinaryFile(FragmentPath.c_str());

        ShaderCreateInfo VertexCI;
        VertexCI.Desc = ShaderDesc{"Triangle VS", SHADER_TYPE_VERTEX, true};
        VertexCI.ByteCode = VertexSPIRV.data();
        VertexCI.ByteCodeSize = VertexSPIRV.size();
        VertexCI.SourceLanguage = SHADER_SOURCE_LANGUAGE_DEFAULT;

        ShaderCreateInfo FragmentCI;
        FragmentCI.Desc = ShaderDesc{"Triangle PS", SHADER_TYPE_PIXEL, true};
        FragmentCI.ByteCode = FragmentSPIRV.data();
        FragmentCI.ByteCodeSize = FragmentSPIRV.size();
        FragmentCI.SourceLanguage = SHADER_SOURCE_LANGUAGE_DEFAULT;

        RefCntAutoPtr<IShader> VertexShader;
        RefCntAutoPtr<IShader> FragmentShader;
        m_Device->CreateShader(VertexCI, &VertexShader);
        m_Device->CreateShader(FragmentCI, &FragmentShader);
        if (VertexShader == nullptr || FragmentShader == nullptr)
            throw std::runtime_error{"Diligent failed to create shaders from SPIR-V"};

        GraphicsPipelineStateCreateInfo PipelineCI{"DiligentCore Vulkan Triangle"};
        PipelineCI.pVS = VertexShader;
        PipelineCI.pPS = FragmentShader;
        PipelineCI.GraphicsPipeline.PrimitiveTopology = PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        PipelineCI.GraphicsPipeline.RasterizerDesc.CullMode = CULL_MODE_NONE;
        PipelineCI.GraphicsPipeline.DepthStencilDesc.DepthEnable = False;
        PipelineCI.GraphicsPipeline.NumRenderTargets = 1;
        PipelineCI.GraphicsPipeline.RTVFormats[0] = m_SwapChainDesc.ColorBufferFormat;
        PipelineCI.GraphicsPipeline.DSVFormat = TEX_FORMAT_UNKNOWN;
        m_Device->CreateGraphicsPipelineState(PipelineCI, &m_PipelineState);
        if (m_PipelineState == nullptr)
            throw std::runtime_error{"Diligent failed to create the graphics pipeline"};
    }

    RefCntAutoPtr<IRenderDevice> m_Device;
    RefCntAutoPtr<IDeviceContext> m_Context;
    RefCntAutoPtr<ISwapChain> m_SwapChain;
    RefCntAutoPtr<IPipelineState> m_PipelineState;
    std::unique_ptr<Nanite::Pipelines> m_NanitePipelines;
    std::unique_ptr<Nanite::GpuScene> m_NaniteScene;
    SwapChainDesc m_SwapChainDesc;
    float m_CameraX = 0.0f;
    float m_CameraY = 0.0f;
    float m_CameraZ = 6.0f;
    float m_CameraYaw = 0.0f;
    float m_CameraPitch = 0.0f;
};

} // namespace

@interface NSObject (NaniteCameraInput)
- (void)handleCameraKey:(NSEvent*)Event;
- (void)handleCameraKeyUp:(NSEvent*)Event;
- (void)beginCameraDrag:(NSEvent*)Event;
- (void)handleCameraDrag:(NSEvent*)Event;
- (void)endCameraDrag:(NSEvent*)Event;
@end

@interface DemoView : NSView
@end

@implementation DemoView

- (BOOL)isFlipped
{
    return YES;
}

- (BOOL)acceptsFirstResponder
{
    return YES;
}

- (void)viewDidMoveToWindow
{
    [super viewDidMoveToWindow];
    self.wantsLayer = YES;
    if (![self.layer isKindOfClass:[CAMetalLayer class]])
        self.layer = [CAMetalLayer layer];
    self.layer.frame = self.bounds;
    self.layer.contentsScale = self.window.backingScaleFactor;
    [self.window makeFirstResponder:self];
}

- (void)setFrameSize:(NSSize)NewSize
{
    [super setFrameSize:NewSize];
    self.layer.frame = self.bounds;
}

- (void)keyDown:(NSEvent*)Event
{
    id Delegate = NSApp.delegate;
    if ([Delegate respondsToSelector:@selector(handleCameraKey:)])
        [Delegate handleCameraKey:Event];
}

- (void)keyUp:(NSEvent*)Event
{
    id Delegate = NSApp.delegate;
    if ([Delegate respondsToSelector:@selector(handleCameraKeyUp:)])
        [Delegate handleCameraKeyUp:Event];
}

- (void)mouseDown:(NSEvent*)Event
{
    id Delegate = NSApp.delegate;
    if ([Delegate respondsToSelector:@selector(beginCameraDrag:)])
        [Delegate beginCameraDrag:Event];
}

- (void)mouseDragged:(NSEvent*)Event
{
    id Delegate = NSApp.delegate;
    if ([Delegate respondsToSelector:@selector(handleCameraDrag:)])
        [Delegate handleCameraDrag:Event];
}

- (void)mouseUp:(NSEvent*)Event
{
    id Delegate = NSApp.delegate;
    if ([Delegate respondsToSelector:@selector(endCameraDrag:)])
        [Delegate endCameraDrag:Event];
}

@end

@interface AppDelegate : NSObject <NSApplicationDelegate, NSWindowDelegate>
{
    std::unique_ptr<NaniteDemo::Renderer> _renderer;
    CFAbsoluteTime _lastFrameTime;
    CFAbsoluteTime _lastFpsTitleUpdateTime;
    double _fps;
    float _cameraX;
    float _cameraY;
    float _cameraZ;
    float _cameraYaw;
    float _cameraPitch;
    BOOL _cameraDragging;
    BOOL _moveForward;
    BOOL _moveBackward;
    BOOL _moveLeft;
    BOOL _moveRight;
    float _forwardVelocity;
    float _strafeVelocity;
    NSPanel* _controlPanel;
    NSTextField* _numXField;
    NSTextField* _numYField;
    NSTextField* _numZField;
    NSSlider* _numXSlider;
    NSSlider* _numYSlider;
    NSSlider* _numZSlider;
    NSButton* _hzbCheckbox;
    NSButton* _vsyncCheckbox;
    NSButton* _asyncRasterCheckbox;
    NSButton* _gpuTimingsCheckbox;
    NSTextField* _modelField;
    NSTextField* _clusterField;
    NSSlider* _errorSlider;
    NSTextField* _errorField;
    NSTextField* _rasterCutoffField;
    NSSlider* _rasterCutoffSlider;
    NSPopUpButton* _shadingPopup;
    NSTextField* _spacingXField;
    NSSlider* _spacingXSlider;
    NSTextField* _spacingYField;
    NSSlider* _spacingYSlider;
    NSTextField* _atmosphereFields[AtmosphereControlCount];
    NSSlider* _atmosphereSliders[AtmosphereControlCount];
    NSView* _controlsPage;
    NSView* _statsPage;
    NSSegmentedControl* _panelPageSelector;
    NSTextView* _statsTextView;
    NSButton* _copyStatsButton;
    NSButton* _smoothStatsCheckbox;
    std::deque<Nanite::FrameStats> _statsSamples;
    // Keep frame durations so the smoothed value is throughput (N / sum(dt)),
    // rather than the biased arithmetic mean of per-frame 1/dt samples.
    std::deque<double> _statsFrameDurations;
}
@property(nonatomic, strong) NSWindow* Window;
@property(nonatomic, strong) DemoView* View;
@property(nonatomic, strong) NSTimer* Timer;
// Non-nil when the display link drove the loop; the timer is then unused. Kept as
// id because CADisplayLink on macOS needs 14.0 and the fallback has to build
// against anything older.
@property(nonatomic, strong) id DisplayLink;
@end

@implementation AppDelegate

- (void)applicationDidFinishLaunching:(NSNotification*)Notification
{
    try
    {
        self.Window = [[NSWindow alloc]
            initWithContentRect:NSMakeRect(0, 0, 1280, 720)
                      styleMask:(NSWindowStyleMaskTitled | NSWindowStyleMaskClosable |
                                 NSWindowStyleMaskMiniaturizable | NSWindowStyleMaskResizable)
                        backing:NSBackingStoreBuffered
                          defer:NO];
        self.Window.title = @"DiligentCore Vulkan Demo";
        self.Window.delegate = self;

        self.View = [[DemoView alloc] initWithFrame:self.Window.contentView.bounds];
        self.View.autoresizingMask = NSViewWidthSizable | NSViewHeightSizable;
        self.Window.contentView = self.View;
        [self.Window center];
        [self.Window makeKeyAndOrderFront:nil];
        [self.Window makeFirstResponder:self.View];

        _cameraZ = 6.0f;
        if (const char* startZ = std::getenv("NANITE_START_Z"))
            _cameraZ = std::strtof(startZ, nullptr);
        else if (const char* scene = std::getenv("NANITE_SCENE");
                 scene != nullptr && std::string{scene} == "coastal")
        {
            // The coastal scene is the snow-mountain terrain: a 120-unit mesh
            // that is ~48 units tall. Start above the surface, looking slightly
            // down across the range.
            _cameraZ = 2.8f;
            _cameraY = 18.0f;
            _cameraPitch = -0.3f;
        }

        NativeWindow nativeWindow{(__bridge void*)self.View};
        _renderer = std::make_unique<NaniteDemo::Renderer>(nativeWindow, 1280, 720);
        [self createControlPanel];
        [self startFrameLoop];
        [NSApp activateIgnoringOtherApps:YES];
    }
    catch (const std::exception& Error)
    {
        NSLog(@"DiligentCore Vulkan Demo failed: %s", Error.what());
        [NSApp terminate:nil];
    }
}

- (void)startFrameLoop
{
    // Idempotent: it is also the switch between the two loops, so whatever is
    // running has to go first.
    if (@available(macOS 14.0, *))
        [(CADisplayLink*)self.DisplayLink invalidate];
    self.DisplayLink = nil;
    [self.Timer invalidate];
    self.Timer = nil;

    // A display link fires once per refresh, on the display the view is actually
    // on, and reschedules itself against the next vblank. An NSTimer knows nothing
    // about the refresh rate: at 1/60 against a 60 Hz panel the two beat against
    // each other, and every time the timer fires just after a vblank the frame it
    // produces waits a whole interval in Present - which is the stutter that gets
    // blamed on vsync.
    //
    // With vsync off the display link is exactly the wrong thing, because it would
    // cap the frame rate at the refresh rate and hide the difference that turning
    // vsync off exists to show. A short-interval timer free-runs instead; the frame
    // fence is what keeps that from queueing frames without bound.
    if (_renderer != nullptr && _renderer->IsVSyncEnabled())
    {
        if (@available(macOS 14.0, *))
        {
            CADisplayLink* link = [self.View displayLinkWithTarget:self
                                                          selector:@selector(drawFrame:)];
            if (link != nil)
            {
                [link addToRunLoop:[NSRunLoop mainRunLoop] forMode:NSRunLoopCommonModes];
                self.DisplayLink = link;
                return;
            }
        }
    }
    // 1 ms rather than 0 so the run loop still gets to service input between
    // frames. Anything the GPU cannot keep up with is absorbed by the fence wait.
    const NSTimeInterval interval =
        _renderer != nullptr && _renderer->IsVSyncEnabled() ? 1.0 / 60.0 : 1.0 / 1000.0;
    self.Timer = [NSTimer timerWithTimeInterval:interval
                                          target:self
                                        selector:@selector(drawFrame:)
                                        userInfo:nil
                                         repeats:YES];
    [[NSRunLoop mainRunLoop] addTimer:self.Timer forMode:NSRunLoopCommonModes];
}

- (void)drawFrame:(id)sender
{
    (void)sender;
    if (_renderer != nullptr)
    {
        const CFAbsoluteTime Now = CFAbsoluteTimeGetCurrent();
        const double FrameDuration = _lastFrameTime == 0.0 ?
            0.0 : std::max(0.0, Now - _lastFrameTime);
        const float DeltaTime = _lastFrameTime == 0.0 ?
            0.0f : static_cast<float>(std::min(Now - _lastFrameTime, 0.1));
        _lastFrameTime = Now;
        [self updateCamera:DeltaTime];

        _renderer->SetCamera(_cameraX, _cameraY, _cameraZ, _cameraYaw, _cameraPitch);
        const NSRect bounds = self.View.bounds;
        const CGFloat scale = self.Window.backingScaleFactor;
        const Uint32 width = static_cast<Uint32>(bounds.size.width * scale);
        const Uint32 height = static_cast<Uint32>(bounds.size.height * scale);
        _renderer->Draw(width, height);

        if (FrameDuration > 0.0)
            _fps = 1.0 / FrameDuration;
        [self recordStatsSample:FrameDuration];
        // The value is instantaneous; only the title mutation is throttled so
        // AppKit string/layout work does not become a per-frame CPU cost.
        if (Now - _lastFpsTitleUpdateTime >= 0.1)
        {
            _lastFpsTitleUpdateTime = Now;
            [self updateWindowTitle];
        }
    }
}

- (void)handleCameraKey:(NSEvent*)Event
{
    switch (Event.keyCode)
    {
        case 123: // Left
            _moveLeft = YES;
            return;
        case 124: // Right
            _moveRight = YES;
            return;
        case 125: // Down
            _moveBackward = YES;
            return;
        case 126: // Up
            _moveForward = YES;
            return;
        default:
            break;
    }

    NSString* Key = Event.charactersIgnoringModifiers.lowercaseString;
    if ([Key isEqualToString:@"a"])
        _moveLeft = YES;
    else if ([Key isEqualToString:@"d"])
        _moveRight = YES;
    else if ([Key isEqualToString:@"w"])
        _moveForward = YES;
    else if ([Key isEqualToString:@"s"])
        _moveBackward = YES;
    else
        return;
}

- (void)handleCameraKeyUp:(NSEvent*)Event
{
    switch (Event.keyCode)
    {
        case 123: // Left
            _moveLeft = NO;
            return;
        case 124: // Right
            _moveRight = NO;
            return;
        case 125: // Down
            _moveBackward = NO;
            return;
        case 126: // Up
            _moveForward = NO;
            return;
        default:
            break;
    }

    NSString* Key = Event.charactersIgnoringModifiers.lowercaseString;
    if ([Key isEqualToString:@"a"])
        _moveLeft = NO;
    else if ([Key isEqualToString:@"d"])
        _moveRight = NO;
    else if ([Key isEqualToString:@"w"])
        _moveForward = NO;
    else if ([Key isEqualToString:@"s"])
        _moveBackward = NO;
}

- (void)updateCamera:(float)DeltaTime
{
    const float ForwardInput = (_moveForward ? 1.0f : 0.0f) -
        (_moveBackward ? 1.0f : 0.0f);
    const float StrafeInput = (_moveRight ? 1.0f : 0.0f) -
        (_moveLeft ? 1.0f : 0.0f);
    const float Response = 1.0f - std::exp(-12.0f * DeltaTime);
    constexpr float MoveSpeed = 4.0f;
    _forwardVelocity += (ForwardInput * MoveSpeed - _forwardVelocity) * Response;
    _strafeVelocity += (StrafeInput * MoveSpeed - _strafeVelocity) * Response;

    // Camera-relative movement: the forward vector follows both the yaw and the
    // pitch, so W/S climb and descend with the view instead of staying level in
    // the world XZ plane. A/D stays a pure horizontal strafe.
    const float ForwardX = std::sin(_cameraYaw) * std::cos(_cameraPitch);
    const float ForwardY = std::sin(_cameraPitch);
    const float ForwardZ = -std::cos(_cameraYaw) * std::cos(_cameraPitch);
    const float RightX = std::cos(_cameraYaw);
    const float RightZ = std::sin(_cameraYaw);
    _cameraX += (ForwardX * _forwardVelocity + RightX * _strafeVelocity) * DeltaTime;
    _cameraY += ForwardY * _forwardVelocity * DeltaTime;
    _cameraZ += (ForwardZ * _forwardVelocity + RightZ * _strafeVelocity) * DeltaTime;
}

- (void)beginCameraDrag:(NSEvent*)Event
{
    (void)Event;
    _cameraDragging = YES;
}

- (void)handleCameraDrag:(NSEvent*)Event
{
    if (!_cameraDragging)
        return;
    constexpr float RotationSpeed = 0.006f;
    _cameraYaw += static_cast<float>(Event.deltaX) * RotationSpeed;
    _cameraPitch -= static_cast<float>(Event.deltaY) * RotationSpeed;
    constexpr float MaxPitch = 1.45f;
    _cameraPitch = std::max(-MaxPitch, std::min(MaxPitch, _cameraPitch));
    [self updateWindowTitle];
}

- (void)endCameraDrag:(NSEvent*)Event
{
    (void)Event;
    _cameraDragging = NO;
}

// Left-aligned static text. Every row in the panel needs one and they differ only
// in position and string.
static NSTextField* MakePanelLabel(NSRect frame, NSString* text)
{
    NSTextField* label = [[NSTextField alloc] initWithFrame:frame];
    label.stringValue = text;
    label.bezeled = NO;
    label.drawsBackground = NO;
    label.editable = NO;
    label.selectable = NO;
    return label;
}

// DebugMode value behind each entry of the Shading popup, in the popup's order.
// 1-4 are the HZB and occlusion diagnostics, which have no entry because they need
// their own environment variables to be useful.
static const std::uint32_t kShadingModes[] = {0u, 5u, 6u, 7u, 8u, 9u, 10u, 11u};

static NSInteger IndexForShadingMode(std::uint32_t mode)
{
    for (NSInteger index = 0; index < static_cast<NSInteger>(std::size(kShadingModes)); ++index)
    {
        if (kShadingModes[index] == mode)
            return index;
    }
    return 0;
}

- (void)createControlPanel
{
    _controlPanel = [[NSPanel alloc]
        initWithContentRect:NSMakeRect(0, 0, 430, 720)
                  styleMask:(NSWindowStyleMaskTitled | NSWindowStyleMaskClosable)
                    backing:NSBackingStoreBuffered
                      defer:NO];
    _controlPanel.title = @"Nanite Controls";
    _controlPanel.floatingPanel = YES;
    _controlPanel.level = NSFloatingWindowLevel;
    _controlPanel.releasedWhenClosed = NO;
    NSView* panelContent = _controlPanel.contentView;

    _panelPageSelector = [[NSSegmentedControl alloc]
        initWithFrame:NSMakeRect(16, 686, 398, 26)];
    _panelPageSelector.segmentCount = 2;
    [_panelPageSelector setLabel:@"Controls" forSegment:0];
    [_panelPageSelector setLabel:@"Stats" forSegment:1];
    _panelPageSelector.selectedSegment = 0;
    _panelPageSelector.trackingMode = NSSegmentSwitchTrackingSelectOne;
    _panelPageSelector.target = self;
    _panelPageSelector.action = @selector(panelPageChanged:);
    [panelContent addSubview:_panelPageSelector];

    _controlsPage = [[NSView alloc] initWithFrame:NSMakeRect(0, 0, 430, 678)];
    _controlsPage.autoresizingMask = NSViewWidthSizable | NSViewHeightSizable;
    [panelContent addSubview:_controlsPage];
    NSView* content = _controlsPage;

    [content addSubview:MakePanelLabel(NSMakeRect(16, 648, 160, 22), @"Atmosphere")];
    // Keep the atmosphere controls compact by pairing related values on each row.
    // The text fields remain editable when a slider's range is too small for an
    // experiment; the renderer clamps invalid values and the UI reads them back.
    struct AtmosphereControlSpec
    {
        NSString* Label;
        double Min;
        double Max;
    };
    const AtmosphereControlSpec atmosphereSpecs[AtmosphereControlCount] = {
        {@"Sun az", -180.0, 180.0},
        {@"Sun elev", -10.0, 80.0},
        {@"Radius", 500.0, 5000.0},
        {@"Height", 20.0, 500.0},
        {@"Rayleigh", 0.0, 3.0},
        {@"Mie", 0.0, 3.0},
        {@"Sky", 0.0, 1.0},
        {@"Terrain", 0.0, 0.5},
        {@"Haze d", 0.1, 4.0},
        {@"Haze w", 0.0, 1.0},
    };
    const Nanite::AtmosphereSettings& atmosphere = _renderer->GetAtmosphereSettings();
    const double atmosphereValues[AtmosphereControlCount] = {
        atmosphere.SunAzimuthDeg,
        atmosphere.SunElevationDeg,
        atmosphere.AtmosphereRadius,
        atmosphere.AtmosphereHeight,
        atmosphere.RayleighScale,
        atmosphere.MieScale,
        atmosphere.SkyDensity,
        atmosphere.TerrainDensity,
        atmosphere.HazeDistance,
        atmosphere.HazeWeight,
    };
    for (NSInteger index = 0; index < AtmosphereControlCount; ++index)
    {
        const NSInteger row = index / 2;
        const NSInteger column = index % 2;
        const CGFloat baseX = column == 0 ? 16.0 : 215.0;
        const CGFloat y = 610.0 - static_cast<CGFloat>(row) * 32.0;
        [content addSubview:MakePanelLabel(
            NSMakeRect(baseX, y + 2.0, 58.0, 22.0), atmosphereSpecs[index].Label)];

        _atmosphereFields[index] = [[NSTextField alloc]
            initWithFrame:NSMakeRect(baseX + 60.0, y, 50.0, 26.0)];
        _atmosphereFields[index].alignment = NSTextAlignmentRight;
        _atmosphereFields[index].doubleValue = atmosphereValues[index];
        _atmosphereFields[index].target = self;
        _atmosphereFields[index].tag = index;
        _atmosphereFields[index].action = @selector(atmosphereFieldChanged:);
        [content addSubview:_atmosphereFields[index]];

        _atmosphereSliders[index] = [[NSSlider alloc]
            initWithFrame:NSMakeRect(baseX + 114.0, y, column == 0 ? 84.0 : 85.0, 24.0)];
        _atmosphereSliders[index].minValue = atmosphereSpecs[index].Min;
        _atmosphereSliders[index].maxValue = atmosphereSpecs[index].Max;
        _atmosphereSliders[index].doubleValue = std::clamp(
            atmosphereValues[index], atmosphereSpecs[index].Min, atmosphereSpecs[index].Max);
        _atmosphereSliders[index].continuous = YES;
        _atmosphereSliders[index].target = self;
        _atmosphereSliders[index].tag = index;
        _atmosphereSliders[index].action = @selector(atmosphereSliderChanged:);
        [content addSubview:_atmosphereSliders[index]];
    }

    // Model and cluster granularity, the two settings that need a scene rebuild.
    [content addSubview:MakePanelLabel(NSMakeRect(16, 424, 48, 22), @"Model")];
    _modelField = [[NSTextField alloc] initWithFrame:NSMakeRect(68, 422, 264, 26)];
    _modelField.stringValue = [NSString stringWithUTF8String:_renderer->GetModelPath().c_str()];
    // Still typeable, because pasting a path is faster than walking a file browser
    // to /tmp. Return rebuilds, so the field, the button and the browser agree.
    _modelField.placeholderString = @"absolute path, or a file name in the asset directory";
    _modelField.target = self;
    _modelField.action = @selector(rebuildSceneFromControls:);
    [content addSubview:_modelField];

    NSButton* browseButton = [[NSButton alloc] initWithFrame:NSMakeRect(336, 420, 78, 30)];
    browseButton.title = @"Browse…";
    browseButton.bezelStyle = NSBezelStyleRounded;
    browseButton.target = self;
    browseButton.action = @selector(browseForModel:);
    [content addSubview:browseButton];

    [content addSubview:MakePanelLabel(NSMakeRect(16, 390, 110, 22), @"Cluster tris")];
    _clusterField = [[NSTextField alloc] initWithFrame:NSMakeRect(122, 388, 56, 26)];
    _clusterField.integerValue = _renderer->GetClusterTriangles() != 0u ?
        static_cast<NSInteger>(_renderer->GetClusterTriangles()) : 128;
    _clusterField.alignment = NSTextAlignmentRight;
    _clusterField.target = self;
    _clusterField.action = @selector(rebuildSceneFromControls:);
    [content addSubview:_clusterField];

    NSButton* rebuildButton = [[NSButton alloc] initWithFrame:NSMakeRect(188, 386, 226, 30)];
    rebuildButton.title = @"Rebuild scene (slow)";
    rebuildButton.bezelStyle = NSBezelStyleRounded;
    rebuildButton.target = self;
    rebuildButton.action = @selector(rebuildSceneFromControls:);
    [content addSubview:rebuildButton];

    // The LOD threshold. Live, and the one control that actually decides how
    // visible a level switch is, so it gets both a slider to sweep and a field to
    // type an exact value into.
    [content addSubview:MakePanelLabel(NSMakeRect(16, 356, 110, 22), @"LOD error px")];
    _errorField = [[NSTextField alloc] initWithFrame:NSMakeRect(122, 354, 56, 26)];
    _errorField.stringValue = [NSString stringWithFormat:@"%.2f", _renderer->GetScreenErrorPixels()];
    _errorField.alignment = NSTextAlignmentRight;
    _errorField.target = self;
    _errorField.action = @selector(errorFieldChanged:);
    [content addSubview:_errorField];

    _errorSlider = [[NSSlider alloc] initWithFrame:NSMakeRect(188, 354, 226, 24)];
    _errorSlider.minValue = 0.02;
    _errorSlider.maxValue = 4.0;
    _errorSlider.doubleValue = std::max(0.02, std::min(4.0,
        static_cast<double>(_renderer->GetScreenErrorPixels())));
    _errorSlider.continuous = YES;
    _errorSlider.target = self;
    _errorSlider.action = @selector(errorSliderChanged:);
    [content addSubview:_errorSlider];

    // The raster-bin boundary is live and compares triangle screen area in px^2.
    [content addSubview:MakePanelLabel(NSMakeRect(16, 322, 110, 22), @"Raster cutoff px2")];
    _rasterCutoffField = [[NSTextField alloc] initWithFrame:NSMakeRect(122, 320, 56, 26)];
    _rasterCutoffField.stringValue =
        [NSString stringWithFormat:@"%.2f", _renderer->GetRasterBinAreaCutoff()];
    _rasterCutoffField.alignment = NSTextAlignmentRight;
    _rasterCutoffField.target = self;
    _rasterCutoffField.action = @selector(rasterCutoffFieldChanged:);
    [content addSubview:_rasterCutoffField];

    _rasterCutoffSlider = [[NSSlider alloc] initWithFrame:NSMakeRect(188, 320, 226, 24)];
    _rasterCutoffSlider.minValue = 0.0;
    _rasterCutoffSlider.maxValue = 1.0;
    _rasterCutoffSlider.doubleValue = RasterCutoffValueToSlider(
        static_cast<double>(_renderer->GetRasterBinAreaCutoff()));
    _rasterCutoffSlider.continuous = YES;
    _rasterCutoffSlider.target = self;
    _rasterCutoffSlider.action = @selector(rasterCutoffSliderChanged:);
    [content addSubview:_rasterCutoffSlider];

    [content addSubview:MakePanelLabel(NSMakeRect(16, 288, 110, 22), @"Shading")];
    _shadingPopup = [[NSPopUpButton alloc] initWithFrame:NSMakeRect(122, 286, 292, 26)];
    // Item order has to match kShadingModes.
    [_shadingPopup addItemWithTitle:@"Blinn-Phong"];
    [_shadingPopup addItemWithTitle:@"Cluster colors"];
    [_shadingPopup addItemWithTitle:@"Triangle colors"];
    [_shadingPopup addItemWithTitle:@"Triangle size (red = sub-pixel)"];
    [_shadingPopup addItemWithTitle:@"Overdraw (no depth test)"];
    [_shadingPopup addItemWithTitle:@"Raster bin (blue = software, red = hardware)"];
    [_shadingPopup addItemWithTitle:@"Final depth"];
    [_shadingPopup addItemWithTitle:@"PBR (Visibility Buffer)"];
    [_shadingPopup selectItemAtIndex:IndexForShadingMode(_renderer->GetShadingMode())];
    _shadingPopup.target = self;
    _shadingPopup.action = @selector(shadingPopupChanged:);
    [content addSubview:_shadingPopup];

    // Instance pitch. Only transforms move, so both are live. Horizontal drives
    // the columns and the depth layers; vertical drives the rows.
    [content addSubview:MakePanelLabel(NSMakeRect(16, 254, 110, 22), @"X spacing")];
    _spacingXField = [[NSTextField alloc] initWithFrame:NSMakeRect(122, 252, 56, 26)];
    _spacingXField.stringValue =
        [NSString stringWithFormat:@"%.2f", _renderer->GetSpacingHorizontal()];
    _spacingXField.alignment = NSTextAlignmentRight;
    _spacingXField.target = self;
    _spacingXField.action = @selector(spacingFieldChanged:);
    [content addSubview:_spacingXField];

    _spacingXSlider = [[NSSlider alloc] initWithFrame:NSMakeRect(188, 252, 226, 24)];
    _spacingXSlider.minValue = 0.0;
    _spacingXSlider.maxValue = 8.0;
    _spacingXSlider.doubleValue = _renderer->GetSpacingHorizontal();
    _spacingXSlider.continuous = YES;
    _spacingXSlider.target = self;
    _spacingXSlider.action = @selector(spacingSliderChanged:);
    [content addSubview:_spacingXSlider];

    [content addSubview:MakePanelLabel(NSMakeRect(16, 220, 110, 22), @"Y spacing")];
    _spacingYField = [[NSTextField alloc] initWithFrame:NSMakeRect(122, 218, 56, 26)];
    _spacingYField.stringValue =
        [NSString stringWithFormat:@"%.2f", _renderer->GetSpacingVertical()];
    _spacingYField.alignment = NSTextAlignmentRight;
    _spacingYField.target = self;
    _spacingYField.action = @selector(spacingFieldChanged:);
    [content addSubview:_spacingYField];

    _spacingYSlider = [[NSSlider alloc] initWithFrame:NSMakeRect(188, 218, 226, 24)];
    _spacingYSlider.minValue = 0.0;
    _spacingYSlider.maxValue = 8.0;
    _spacingYSlider.doubleValue = _renderer->GetSpacingVertical();
    _spacingYSlider.continuous = YES;
    _spacingYSlider.target = self;
    _spacingYSlider.action = @selector(spacingSliderChanged:);
    [content addSubview:_spacingYSlider];

    [content addSubview:MakePanelLabel(NSMakeRect(16, 186, 110, 22), @"Num X")];
    _numXField = [[NSTextField alloc] initWithFrame:NSMakeRect(122, 184, 56, 26)];
    _numXField.integerValue = _renderer->GetInstanceCountX();
    _numXField.alignment = NSTextAlignmentRight;
    _numXField.target = self;
    _numXField.action = @selector(numFieldChanged:);
    [content addSubview:_numXField];
    _numXSlider = [[NSSlider alloc] initWithFrame:NSMakeRect(188, 184, 226, 24)];
    _numXSlider.minValue = 0.0;
    _numXSlider.maxValue = 1.0;
    _numXSlider.numberOfTickMarks = 0;
    _numXSlider.allowsTickMarkValuesOnly = NO;
    _numXSlider.doubleValue = InstanceCountToSliderPosition(
        _numXField.integerValue, InstanceCountMin, InstanceCountSliderMax);
    // A value beyond the current allocation can grow the scene buffers, so only
    // commit once when the drag ends instead of rebuilding on every mouse event.
    _numXSlider.continuous = NO;
    _numXSlider.target = self;
    _numXSlider.action = @selector(numSliderChanged:);
    [content addSubview:_numXSlider];

    [content addSubview:MakePanelLabel(NSMakeRect(16, 152, 110, 22), @"Num Y")];
    _numYField = [[NSTextField alloc] initWithFrame:NSMakeRect(122, 150, 56, 26)];
    _numYField.integerValue = _renderer->GetInstanceCountY();
    _numYField.alignment = NSTextAlignmentRight;
    _numYField.target = self;
    _numYField.action = @selector(numFieldChanged:);
    [content addSubview:_numYField];
    _numYSlider = [[NSSlider alloc] initWithFrame:NSMakeRect(188, 150, 226, 24)];
    _numYSlider.minValue = 0.0;
    _numYSlider.maxValue = 1.0;
    _numYSlider.numberOfTickMarks = 0;
    _numYSlider.allowsTickMarkValuesOnly = NO;
    _numYSlider.doubleValue = InstanceCountToSliderPosition(
        _numYField.integerValue, InstanceCountMin, InstanceCountSliderMax);
    _numYSlider.continuous = NO;
    _numYSlider.target = self;
    _numYSlider.action = @selector(numSliderChanged:);
    [content addSubview:_numYSlider];

    [content addSubview:MakePanelLabel(NSMakeRect(16, 118, 110, 22), @"Num Z")];
    _numZField = [[NSTextField alloc] initWithFrame:NSMakeRect(122, 116, 56, 26)];
    _numZField.integerValue = _renderer->GetInstanceCountZ();
    _numZField.alignment = NSTextAlignmentRight;
    _numZField.target = self;
    _numZField.action = @selector(numFieldChanged:);
    [content addSubview:_numZField];
    _numZSlider = [[NSSlider alloc] initWithFrame:NSMakeRect(188, 116, 226, 24)];
    _numZSlider.minValue = 0.0;
    _numZSlider.maxValue = 1.0;
    _numZSlider.numberOfTickMarks = 0;
    _numZSlider.allowsTickMarkValuesOnly = NO;
    _numZSlider.doubleValue = InstanceCountToSliderPosition(
        _numZField.integerValue, InstanceCountMin, InstanceCountSliderMax);
    _numZSlider.continuous = NO;
    _numZSlider.target = self;
    _numZSlider.action = @selector(numSliderChanged:);
    [content addSubview:_numZSlider];
    [self syncInstanceGridControls];

    _hzbCheckbox = [[NSButton alloc] initWithFrame:NSMakeRect(16, 72, 140, 24)];
    _hzbCheckbox.title = @"HZB enabled";
    _hzbCheckbox.buttonType = NSButtonTypeSwitch;
    // Occlusion culling is part of the normal path, so it starts on and
    // NANITE_HZB_OFF turns it off for an A/B against no occlusion at all.
    const bool hzbOnAtStartup = std::getenv("NANITE_HZB_OFF") == nullptr;
    _hzbCheckbox.state = hzbOnAtStartup ? NSControlStateValueOn : NSControlStateValueOff;
    _hzbCheckbox.target = self;
    _hzbCheckbox.action = @selector(hzbCheckboxChanged:);
    [content addSubview:_hzbCheckbox];
    _renderer->SetHZBEnabled(hzbOnAtStartup);

    // Start uncapped so GPU performance changes are visible. NANITE_VSYNC=1 is
    // the explicit startup opt-in; the checkbox remains live after launch.
    const bool vsyncOnAtStartup = std::getenv("NANITE_VSYNC") != nullptr &&
        std::getenv("NANITE_NO_VSYNC") == nullptr;
    _vsyncCheckbox = [[NSButton alloc] initWithFrame:NSMakeRect(16, 40, 100, 24)];
    _vsyncCheckbox.title = @"VSync";
    _vsyncCheckbox.buttonType = NSButtonTypeSwitch;
    _vsyncCheckbox.state = vsyncOnAtStartup ? NSControlStateValueOn : NSControlStateValueOff;
    _vsyncCheckbox.target = self;
    _vsyncCheckbox.action = @selector(vsyncCheckboxChanged:);
    [content addSubview:_vsyncCheckbox];
    _renderer->SetVSyncEnabled(vsyncOnAtStartup);

    _asyncRasterCheckbox = [[NSButton alloc] initWithFrame:NSMakeRect(122, 40, 184, 24)];
    _asyncRasterCheckbox.title = @"Async raster queue";
    _asyncRasterCheckbox.buttonType = NSButtonTypeSwitch;
    _asyncRasterCheckbox.state = _renderer->IsAsyncRasterEnabled() ?
        NSControlStateValueOn : NSControlStateValueOff;
    _asyncRasterCheckbox.target = self;
    _asyncRasterCheckbox.action = @selector(asyncRasterCheckboxChanged:);
    _asyncRasterCheckbox.enabled = _renderer->HasAsyncRasterQueue();
    [content addSubview:_asyncRasterCheckbox];

    const bool timingsOnAtStartup = std::getenv("NANITE_GPU_TIMINGS") != nullptr;
    _gpuTimingsCheckbox = [[NSButton alloc] initWithFrame:NSMakeRect(122, 72, 184, 24)];
    _gpuTimingsCheckbox.title = @"GPU timings";
    _gpuTimingsCheckbox.buttonType = NSButtonTypeSwitch;
    _gpuTimingsCheckbox.state = timingsOnAtStartup ? NSControlStateValueOn : NSControlStateValueOff;
    _gpuTimingsCheckbox.target = self;
    _gpuTimingsCheckbox.action = @selector(gpuTimingsCheckboxChanged:);
    _gpuTimingsCheckbox.enabled = _renderer->AreGpuTimingsSupported();
    [content addSubview:_gpuTimingsCheckbox];
    _renderer->SetGpuTimingsEnabled(timingsOnAtStartup);

    _statsPage = [[NSView alloc] initWithFrame:NSMakeRect(0, 0, 430, 678)];
    _statsPage.autoresizingMask = NSViewWidthSizable | NSViewHeightSizable;
    _statsPage.hidden = YES;
    [panelContent addSubview:_statsPage];

    _copyStatsButton = [[NSButton alloc] initWithFrame:NSMakeRect(16, 12, 190, 26)];
    _copyStatsButton.title = @"Copy 30-frame stats";
    _copyStatsButton.bezelStyle = NSBezelStyleRounded;
    _copyStatsButton.target = self;
    _copyStatsButton.action = @selector(copyStatsToClipboard:);
    [_statsPage addSubview:_copyStatsButton];

    _smoothStatsCheckbox = [[NSButton alloc] initWithFrame:NSMakeRect(220, 12, 194, 26)];
    _smoothStatsCheckbox.title = @"Smooth display (30 frames)";
    _smoothStatsCheckbox.buttonType = NSButtonTypeSwitch;
    _smoothStatsCheckbox.state = NSControlStateValueOn;
    _smoothStatsCheckbox.target = self;
    _smoothStatsCheckbox.action = @selector(smoothStatsCheckboxChanged:);
    [_statsPage addSubview:_smoothStatsCheckbox];

    NSScrollView* statsScrollView = [[NSScrollView alloc]
        initWithFrame:NSMakeRect(16, 46, 398, 626)];
    statsScrollView.hasVerticalScroller = YES;
    statsScrollView.hasHorizontalScroller = NO;
    statsScrollView.autohidesScrollers = YES;
    statsScrollView.borderType = NSBezelBorder;
    [statsScrollView setDrawsBackground:YES];

    _statsTextView = [[NSTextView alloc]
        initWithFrame:NSMakeRect(0, 0, statsScrollView.contentSize.width, 438)];
    _statsTextView.editable = NO;
    _statsTextView.selectable = YES;
    _statsTextView.richText = NO;
    _statsTextView.usesFontPanel = NO;
    _statsTextView.usesFindPanel = YES;
    _statsTextView.font = [NSFont fontWithName:@"Menlo" size:11.0] ?
        [NSFont fontWithName:@"Menlo" size:11.0] : [NSFont monospacedSystemFontOfSize:11.0 weight:NSFontWeightRegular];
    _statsTextView.textColor = NSColor.labelColor;
    _statsTextView.backgroundColor = NSColor.textBackgroundColor;
    _statsTextView.verticallyResizable = YES;
    _statsTextView.horizontallyResizable = NO;
    _statsTextView.autoresizingMask = NSViewWidthSizable;
    _statsTextView.textContainer.containerSize =
        NSMakeSize(statsScrollView.contentSize.width, CGFLOAT_MAX);
    _statsTextView.textContainer.widthTracksTextView = YES;
    statsScrollView.documentView = _statsTextView;
    [_statsPage addSubview:statsScrollView];
    [self updateStatsPanel];

    NSRect windowFrame = self.Window.frame;
    NSPoint panelOrigin = NSMakePoint(
        NSMaxX(windowFrame) + 12.0, NSMinY(windowFrame) + 40.0);
    // The taller panel can otherwise extend beyond the right edge on the
    // 2048-pixel desktop used by the coastal demo. Keep it fully reachable,
    // even when the main window is already near the right side.
    NSScreen* screen = self.Window.screen != nil ? self.Window.screen : NSScreen.mainScreen;
    if (screen != nil)
    {
        const NSRect visibleFrame = screen.visibleFrame;
        panelOrigin.x = std::max(
            NSMinX(visibleFrame),
            std::min(panelOrigin.x,
                     NSMaxX(visibleFrame) - _controlPanel.frame.size.width));
        panelOrigin.y = std::max(
            NSMinY(visibleFrame),
            std::min(panelOrigin.y,
                     NSMaxY(visibleFrame) - _controlPanel.frame.size.height));
    }
    [_controlPanel setFrameOrigin:panelOrigin];
    [_controlPanel makeKeyAndOrderFront:nil];
}

- (void)recordStatsSample:(double)frameDuration
{
    if (_renderer == nullptr)
        return;
    _statsSamples.push_back(_renderer->GetLastFrameStats());
    if (frameDuration > 0.0)
        _statsFrameDurations.push_back(frameDuration);
    while (_statsSamples.size() > 30u)
        _statsSamples.pop_front();
    while (_statsFrameDurations.size() > 30u)
        _statsFrameDurations.pop_front();
}

- (double)displayFps
{
    if (_smoothStatsCheckbox == nil ||
        _smoothStatsCheckbox.state != NSControlStateValueOn ||
        _statsFrameDurations.empty())
        return _fps;

    double totalSeconds = 0.0;
    for (double duration : _statsFrameDurations)
        totalSeconds += duration;
    return totalSeconds > 0.0 ?
        static_cast<double>(_statsFrameDurations.size()) / totalSeconds : _fps;
}

- (void)smoothStatsCheckboxChanged:(NSButton*)sender
{
    (void)sender;
    [self updateWindowTitle];
}

- (void)copyStatsToClipboard:(id)sender
{
    (void)sender;
    const std::size_t sampleCount = _statsSamples.size();
    if (sampleCount == 0u)
        return;

    auto averageCounter = [&](auto member) {
        long double sum = 0.0L;
        for (const Nanite::FrameStats& sample : _statsSamples)
            sum += static_cast<long double>(sample.*member);
        return static_cast<double>(sum / static_cast<long double>(sampleCount));
    };
    auto averageGpu = [&](auto member) {
        double sum = 0.0;
        std::size_t validCount = 0;
        for (const Nanite::FrameStats& sample : _statsSamples)
        {
            if (sample.Gpu.Enabled && sample.Gpu.Valid)
            {
                sum += static_cast<double>(sample.Gpu.*member);
                ++validCount;
            }
        }
        return validCount != 0 ? sum / static_cast<double>(validCount) : 0.0;
    };

    double totalFrameSeconds = 0.0;
    for (double duration : _statsFrameDurations)
        totalFrameSeconds += duration;
    const double averageFps = totalFrameSeconds > 0.0 ?
        static_cast<double>(_statsFrameDurations.size()) / totalFrameSeconds : _fps;
    const Nanite::FrameStats& latest = _statsSamples.back();
    const double rasterBin = averageGpu(&Nanite::GpuTimings::RasterBinMs);
    const double hardwareRaster = averageGpu(&Nanite::GpuTimings::RasterMs);
    const double softwareRaster = averageGpu(&Nanite::GpuTimings::SoftRasterMs);
    const double softwareDepth = averageGpu(&Nanite::GpuTimings::SoftDepthMs);
    const double softwareVisibility = averageGpu(&Nanite::GpuTimings::SoftVisibilityMs);
    const double queueReset = averageGpu(&Nanite::GpuTimings::QueueResetMs);
    const double prepareDispatch = averageGpu(&Nanite::GpuTimings::PrepareDispatchMs);
    const double postRecovery = averageGpu(&Nanite::GpuTimings::PostRecoveryMs);
    const double postRaster = averageGpu(&Nanite::GpuTimings::PostRasterMs);
    const double hzbBuild = averageGpu(&Nanite::GpuTimings::HizBuildMs);
    const double finalHizBuild = averageGpu(&Nanite::GpuTimings::FinalHizBuildMs);
    const double visibilityResolve = averageGpu(&Nanite::GpuTimings::VisibilityResolveMs);
    const double shading = averageGpu(&Nanite::GpuTimings::ShadingMs);
    const double hardwareDepthOnly = averageGpu(&Nanite::GpuTimings::HardwareDepthOnlyMs);
    const double shadowMap = averageGpu(&Nanite::GpuTimings::ShadowMapMs);
    const double softCheck = averageGpu(&Nanite::GpuTimings::SoftCheckMs);
    // Raster bin runs on graphics before both raster queues can overlap. Only a
    // genuinely dedicated async queue gets the max() bound; the serial fallback
    // must report the actual sum so the HUD cannot hide a queue stall.
    const double rasterOptimisticBound = latest.Gpu.AsyncRaster ?
        rasterBin + std::max(hardwareRaster, softwareRaster) :
        rasterBin + hardwareRaster + softwareRaster;
    const double rasterWorkSum = rasterBin + hardwareRaster + softwareRaster;
    const double rasterSpan = rasterBin + (latest.Gpu.AsyncRaster ?
        std::max(hardwareRaster, softwareRaster) : hardwareRaster + softwareRaster);
    const double frame = averageGpu(&Nanite::GpuTimings::FrameMs);
    const double accountedGpu = queueReset +
        averageGpu(&Nanite::GpuTimings::InstanceCullMs) +
        averageGpu(&Nanite::GpuTimings::PersistentCullMs) +
        averageGpu(&Nanite::GpuTimings::GenerateIndirectMs) +
        prepareDispatch + postRecovery + rasterSpan + postRaster + hzbBuild +
        finalHizBuild + averageGpu(&Nanite::GpuTimings::PostCullMs) +
        visibilityResolve + shading + hardwareDepthOnly + shadowMap + softCheck;
    const double gpuOther = frame - accountedGpu;
    NSMutableString* output = [NSMutableString stringWithFormat:
        @"Nanite 30-frame average (samples=%lu)\n", static_cast<unsigned long>(sampleCount)];
    [output appendFormat:@"CPU/Present FPS              %.2f\n", averageFps];
    [output appendFormat:@"Logical triangles            %.3f B\n",
                         averageCounter(&Nanite::FrameStats::LogicalTriangleCount) / 1000000000.0];
    [output appendFormat:@"Instances / visible          %.1f / %.1f\n",
                         averageCounter(&Nanite::FrameStats::InstanceCount),
                         averageCounter(&Nanite::FrameStats::VisibleInstanceCount)];
    [output appendFormat:@"Cluster candidates / draws   %.1f / %.1f\n",
                         averageCounter(&Nanite::FrameStats::ClusterCandidateCount),
                         averageCounter(&Nanite::FrameStats::DrawCount)];
    [output appendFormat:@"Node tasks processed / written %.1f / %.1f\n",
                         averageCounter(&Nanite::FrameStats::NodeTaskCount),
                         averageCounter(&Nanite::FrameStats::NodeWriteCount)];
    [output appendFormat:@"Group tasks processed / written %.1f / %.1f\n",
                         averageCounter(&Nanite::FrameStats::GroupTaskCount),
                         averageCounter(&Nanite::FrameStats::GroupWriteCount)];
    [output appendFormat:@"Group reject F/H               %.1f / %.1f\n",
                         averageCounter(&Nanite::FrameStats::GroupFrustumRejected),
                         averageCounter(&Nanite::FrameStats::GroupHZBRejected)];
    [output appendFormat:@"Cluster reject F/H             %.1f / %.1f\n",
                         averageCounter(&Nanite::FrameStats::ClusterFrustumRejected),
                         averageCounter(&Nanite::FrameStats::ClusterHZBRejected)];
    [output appendFormat:@"Cluster reject LOD/tiny        %.1f / %.1f\n",
                         averageCounter(&Nanite::FrameStats::ClusterRefinementRejected),
                         averageCounter(&Nanite::FrameStats::ClusterTinyRejected)];
    [output appendFormat:@"Post queue / HZB reject        %.1f / %.1f\n",
                         averageCounter(&Nanite::FrameStats::PostClusterCount),
                         averageCounter(&Nanite::FrameStats::PostClusterHZBRejected)];
    [output appendFormat:@"HW / SW cluster bin entries  %.1f / %.1f\n",
                         averageCounter(&Nanite::FrameStats::HardwareClusterCount),
                         averageCounter(&Nanite::FrameStats::SoftClusterCount)];
    [output appendString:@"(Mixed clusters may appear in both bins)\n"];
    [output appendFormat:@"Auto raster route             %@\n",
                         latest.AutoRasterEnabled ?
                             (latest.AutoRasterHardware ? @"hardware" : @"hybrid") : @"manual"];
    [output appendFormat:@"Software triangles / pixels   %.1f / %.1f\n",
                         averageCounter(&Nanite::FrameStats::SoftRasterTriangles),
                         averageCounter(&Nanite::FrameStats::SoftRasterPixels)];
    [output appendFormat:@"Soft depth match / mismatch   %.1f / %.1f\n",
                         averageCounter(&Nanite::FrameStats::SoftDepthMatched),
                         averageCounter(&Nanite::FrameStats::SoftDepthMismatch)];
    [output appendFormat:@"Soft visibility resolved      %.1f\n\n",
                         averageCounter(&Nanite::FrameStats::SoftVisResolved)];
    [output appendString:@"GPU timings (ms; valid samples only)\n"];
    [output appendFormat:@"Frame                       %.3f\n", frame];
    [output appendFormat:@"GPU-bound FPS (Nanite)        %.2f\n",
                         frame > 0.0 ? 1000.0 / frame : 0.0];
    [output appendFormat:@"Instance cull               %.3f\n", averageGpu(&Nanite::GpuTimings::InstanceCullMs)];
    [output appendFormat:@"Persistent cull             %.3f\n", averageGpu(&Nanite::GpuTimings::PersistentCullMs)];
    [output appendFormat:@"Generate indirect           %.3f\n", averageGpu(&Nanite::GpuTimings::GenerateIndirectMs)];
    [output appendFormat:@"Post cull                   %.3f\n", averageGpu(&Nanite::GpuTimings::PostCullMs)];
    [output appendFormat:@"Queue reset                 %.3f\n", queueReset];
    [output appendFormat:@"Prepare dispatch            %.3f\n", prepareDispatch];
    [output appendFormat:@"Post recovery               %.3f\n", postRecovery];
    [output appendFormat:@"%@             %.3f\n",
                         latest.Gpu.AsyncRaster ? @"Raster optimistic bound" : @"Raster serial time",
                         rasterOptimisticBound];
    [output appendFormat:@"Raster work sum             %.3f\n", rasterWorkSum];
    [output appendFormat:@"  Raster bin (graphics)      %.3f\n", rasterBin];
    [output appendFormat:@"  Hardware raster work       %.3f\n", hardwareRaster];
    [output appendFormat:@"  Software raster work       %.3f\n", softwareRaster];
    [output appendFormat:@"    Software depth sweep       %.3f\n", softwareDepth];
    [output appendFormat:@"    Software payload sweep     %.3f\n", softwareVisibility];
    [output appendFormat:@"  Hardware depth-only        %.3f\n", averageGpu(&Nanite::GpuTimings::HardwareDepthOnlyMs)];
    [output appendFormat:@"  Virtual shadow pages       %.3f\n", shadowMap];
    [output appendFormat:@"  Software resolve check     %.3f\n", averageGpu(&Nanite::GpuTimings::SoftCheckMs)];
    [output appendFormat:@"Visibility resolve           %.3f\n", averageGpu(&Nanite::GpuTimings::VisibilityResolveMs)];
    [output appendFormat:@"Visibility shading           %.3f\n", averageGpu(&Nanite::GpuTimings::ShadingMs)];
    [output appendFormat:@"Post raster                 %.3f\n", postRaster];
    [output appendFormat:@"HZB build                   %.3f\n", hzbBuild];
    [output appendFormat:@"Final HZB build             %.3f\n", finalHizBuild];
    [output appendFormat:@"GPU accounted total         %.3f\n", accountedGpu];
    [output appendFormat:@"GPU other (Frame-total)      %+.3f\n", gpuOther];
    [output appendFormat:@"HZB                         %@\nOverflow                    0x%X\n",
                         latest.HZBEnabled ? @"ON" : @"OFF", latest.OverflowFlags];

    NSPasteboard* pasteboard = [NSPasteboard generalPasteboard];
    [pasteboard clearContents];
    [pasteboard setString:output forType:NSPasteboardTypeString];
}

- (void)panelPageChanged:(NSSegmentedControl*)sender
{
    const BOOL showStats = sender.selectedSegment == 1;
    _controlsPage.hidden = showStats;
    _statsPage.hidden = !showStats;
    if (showStats)
        [self updateStatsPanel];
}

- (void)syncAtmosphereControls
{
    if (_renderer == nullptr)
        return;

    const Nanite::AtmosphereSettings& atmosphere = _renderer->GetAtmosphereSettings();
    const double values[AtmosphereControlCount] = {
        atmosphere.SunAzimuthDeg,
        atmosphere.SunElevationDeg,
        atmosphere.AtmosphereRadius,
        atmosphere.AtmosphereHeight,
        atmosphere.RayleighScale,
        atmosphere.MieScale,
        atmosphere.SkyDensity,
        atmosphere.TerrainDensity,
        atmosphere.HazeDistance,
        atmosphere.HazeWeight,
    };
    for (NSInteger index = 0; index < AtmosphereControlCount; ++index)
    {
        NSString* format = index < 2 ? @"%.2f" :
            (index < 4 ? @"%.1f" : @"%.3f");
        _atmosphereFields[index].stringValue =
            [NSString stringWithFormat:format, values[index]];
        _atmosphereSliders[index].doubleValue = std::max(
            _atmosphereSliders[index].minValue,
            std::min(_atmosphereSliders[index].maxValue, values[index]));
    }
}

- (void)setAtmosphereControlValue:(NSInteger)index value:(double)value
{
    if (_renderer == nullptr)
        return;

    Nanite::AtmosphereSettings atmosphere = _renderer->GetAtmosphereSettings();
    switch (index)
    {
        case 0: atmosphere.SunAzimuthDeg = static_cast<float>(value); break;
        case 1: atmosphere.SunElevationDeg = static_cast<float>(value); break;
        case 2: atmosphere.AtmosphereRadius = static_cast<float>(value); break;
        case 3: atmosphere.AtmosphereHeight = static_cast<float>(value); break;
        case 4: atmosphere.RayleighScale = static_cast<float>(value); break;
        case 5: atmosphere.MieScale = static_cast<float>(value); break;
        case 6: atmosphere.SkyDensity = static_cast<float>(value); break;
        case 7: atmosphere.TerrainDensity = static_cast<float>(value); break;
        case 8: atmosphere.HazeDistance = static_cast<float>(value); break;
        case 9: atmosphere.HazeWeight = static_cast<float>(value); break;
        default: return;
    }
    _renderer->SetAtmosphereSettings(atmosphere);
    [self syncAtmosphereControls];
}

- (void)atmosphereSliderChanged:(NSSlider*)sender
{
    const NSInteger index = sender.tag;
    if (index < 0 || index >= AtmosphereControlCount)
        return;
    [self setAtmosphereControlValue:index value:sender.doubleValue];
}

- (void)atmosphereFieldChanged:(NSTextField*)sender
{
    const NSInteger index = sender.tag;
    if (index < 0 || index >= AtmosphereControlCount)
        return;
    [self setAtmosphereControlValue:index value:sender.doubleValue];
}

- (void)errorSliderChanged:(NSSlider*)sender
{
    _errorField.stringValue = [NSString stringWithFormat:@"%.2f", sender.doubleValue];
    _renderer->SetScreenErrorPixels(static_cast<float>(sender.doubleValue));
}

- (void)errorFieldChanged:(NSTextField*)sender
{
    // Typed values are not clamped to the slider's range: a negative value is the
    // documented way to switch the cluster-level LOD cut off entirely, and a large
    // one is how you get a deliberately coarse scene. The slider just parks at
    // whichever end it can reach.
    const double requested = sender.doubleValue;
    _errorSlider.doubleValue = std::max(_errorSlider.minValue,
                                       std::min(_errorSlider.maxValue, requested));
    _renderer->SetScreenErrorPixels(static_cast<float>(requested));
}

- (void)rasterCutoffSliderChanged:(NSSlider*)sender
{
    const double cutoff = RasterCutoffSliderToValue(sender.doubleValue);
    _rasterCutoffField.stringValue = [NSString stringWithFormat:@"%.2f", cutoff];
    _renderer->SetRasterBinAreaCutoff(static_cast<float>(cutoff));
}

- (void)rasterCutoffFieldChanged:(NSTextField*)sender
{
    const double requested = std::max(0.0, sender.doubleValue);
    _rasterCutoffSlider.doubleValue = RasterCutoffValueToSlider(requested);
    _renderer->SetRasterBinAreaCutoff(static_cast<float>(requested));
    _rasterCutoffField.stringValue =
        [NSString stringWithFormat:@"%.2f", _renderer->GetRasterBinAreaCutoff()];
}

- (void)shadingPopupChanged:(NSPopUpButton*)sender
{
    _renderer->SetShadingMode(sender.indexOfSelectedItem >= 0 &&
                                      sender.indexOfSelectedItem <
                                          static_cast<NSInteger>(std::size(kShadingModes)) ?
                                  kShadingModes[sender.indexOfSelectedItem] :
                                  0u);
}

- (void)spacingSliderChanged:(NSSlider*)sender
{
    NSTextField* field = sender == _spacingXSlider ? _spacingXField : _spacingYField;
    field.stringValue = [NSString stringWithFormat:@"%.2f", sender.doubleValue];
    [self applyInstanceSpacing];
}

- (void)spacingFieldChanged:(NSTextField*)sender
{
    // Same rule as the error field: a typed value past the slider's end still
    // applies, and the slider parks at the end it can reach.
    NSSlider* slider = sender == _spacingXField ? _spacingXSlider : _spacingYSlider;
    slider.doubleValue = std::max(slider.minValue,
                                  std::min(slider.maxValue, sender.doubleValue));
    [self applyInstanceSpacing];
}

- (void)applyInstanceSpacing
{
    // Read the fields rather than the sliders, because a slider drag writes its
    // field first and a typed value may be outside the slider's range.
    _renderer->SetInstanceSpacing(static_cast<float>(_spacingXField.doubleValue),
                                  static_cast<float>(_spacingYField.doubleValue));
    // Echo back what was accepted, which is how a negative value shows up as the 0
    // it was clamped to instead of silently disagreeing with the scene.
    _spacingXField.stringValue =
        [NSString stringWithFormat:@"%.2f", _renderer->GetSpacingHorizontal()];
    _spacingYField.stringValue =
        [NSString stringWithFormat:@"%.2f", _renderer->GetSpacingVertical()];
}

- (void)browseForModel:(id)sender
{
    (void)sender;
    NSOpenPanel* panel = [NSOpenPanel openPanel];
    panel.canChooseFiles = YES;
    panel.canChooseDirectories = NO;
    panel.allowsMultipleSelection = NO;
    panel.message = @"Pick an OBJ or PLY mesh to rebuild the scene from";
    // Filter by extension, since that is exactly what the loader dispatches on.
    // typeWithFilenameExtension can return nil for an extension the system has
    // never heard of, and an array with a nil in it would throw.
    NSMutableArray<UTType*>* types = [NSMutableArray array];
    for (NSString* extension in @[@"obj", @"ply"])
    {
        UTType* type = [UTType typeWithFilenameExtension:extension];
        if (type != nil)
            [types addObject:type];
    }
    if (types.count != 0)
        panel.allowedContentTypes = types;
    // Open where the current model lives, which after a first browse is where the
    // next one probably is too.
    const std::string& current = _renderer->GetModelPath();
    if (!current.empty())
    {
        NSString* path = [NSString stringWithUTF8String:current.c_str()];
        panel.directoryURL = [NSURL fileURLWithPath:path.stringByDeletingLastPathComponent
                                       isDirectory:YES];
    }
    [panel beginSheetModalForWindow:_controlPanel
                  completionHandler:^(NSModalResponse response) {
                      if (response != NSModalResponseOK || panel.URLs.count == 0)
                          return;
                      _modelField.stringValue = panel.URLs.firstObject.path;
                      // Picking a file is an unambiguous request to load it, so
                      // rebuild rather than making the button a second step.
                      [self rebuildSceneFromControls:nil];
                  }];
}

- (void)rebuildSceneFromControls:(id)sender
{
    (void)sender;
    const NSInteger clusterTriangles = _clusterField.integerValue;
    if (clusterTriangles < 4 || clusterTriangles > 256)
    {
        [self showPanelError:@"Cluster triangles must be between 4 and 256."];
        return;
    }
    _renderer->SetModelPath(std::string{_modelField.stringValue.UTF8String});
    _renderer->SetClusterTriangles(static_cast<std::uint32_t>(clusterTriangles));
    try
    {
        // Blocking, seconds for a million-triangle model. Nothing here is worth the
        // complexity of a background load: the panel is a development tool and the
        // alternative is a half-built scene being drawn.
        _renderer->ReloadSceneFromSettings();
    }
    catch (const std::exception& error)
    {
        // The old scene survives a failed rebuild, because the new one is only
        // installed after it is fully built. So this is recoverable: fix the path
        // and press the button again.
        [self showPanelError:[NSString stringWithFormat:@"Scene rebuild failed: %s",
                                                        error.what()]];
        return;
    }
    [self syncInstanceGridControls];
    _rasterCutoffField.stringValue =
        [NSString stringWithFormat:@"%.2f", _renderer->GetRasterBinAreaCutoff()];
    _rasterCutoffSlider.doubleValue = RasterCutoffValueToSlider(
        static_cast<double>(_renderer->GetRasterBinAreaCutoff()));
    _gpuTimingsCheckbox.enabled = _renderer->AreGpuTimingsSupported();
    [self updateWindowTitle];
}

- (void)showPanelError:(NSString*)message
{
    NSLog(@"Nanite controls: %@", message);
    NSAlert* alert = [[NSAlert alloc] init];
    alert.messageText = message;
    alert.alertStyle = NSAlertStyleWarning;
    [alert runModal];
}

- (void)hzbCheckboxChanged:(NSButton*)sender
{
    _renderer->SetHZBEnabled(sender.state == NSControlStateValueOn);
    [self updateWindowTitle];
}

- (void)vsyncCheckboxChanged:(NSButton*)sender
{
    _renderer->SetVSyncEnabled(sender.state == NSControlStateValueOn);
    // The display link only fires once per refresh, so leaving it in charge would
    // cap the frame rate at 60 whatever Present's sync interval is. Turning vsync
    // off is a request to measure, so the loop has to come off the refresh too.
    [self startFrameLoop];
}

- (void)gpuTimingsCheckboxChanged:(NSButton*)sender
{
    _renderer->SetGpuTimingsEnabled(sender.state == NSControlStateValueOn);
    [self updateWindowTitle];
}

- (void)asyncRasterCheckboxChanged:(NSButton*)sender
{
    _renderer->SetAsyncRasterEnabled(sender.state == NSControlStateValueOn);
    [self updateWindowTitle];
}

- (void)numSliderChanged:(NSSlider*)sender
{
    NSTextField* field = sender == _numXSlider ? _numXField :
        (sender == _numYSlider ? _numYField : _numZField);
    field.integerValue = static_cast<NSInteger>(SliderPositionToInstanceCount(
        sender.doubleValue, InstanceCountMin, InstanceCountSliderMax));
    [self applyInstanceGridFromControls:sender];
}

- (void)numFieldChanged:(NSTextField*)sender
{
    NSTextField* field = sender;
    NSSlider* slider = sender == _numXField ? _numXSlider :
        (sender == _numYField ? _numYSlider : _numZSlider);
    field.integerValue = std::clamp<NSInteger>(field.integerValue, InstanceCountMin, InstanceCountMax);
    slider.doubleValue = InstanceCountToSliderPosition(
        field.integerValue, InstanceCountMin, InstanceCountSliderMax);
    [self applyInstanceGridFromControls:sender];
}

- (void)syncInstanceGridControls
{
    const std::uint64_t countX = static_cast<std::uint64_t>(
        std::max<NSInteger>(1, _renderer->GetInstanceCountX()));
    const std::uint64_t countY = static_cast<std::uint64_t>(
        std::max<NSInteger>(1, _renderer->GetInstanceCountY()));
    const std::uint64_t countZ = static_cast<std::uint64_t>(
        std::max<NSInteger>(1, _renderer->GetInstanceCountZ()));
    _numXField.integerValue = static_cast<NSInteger>(countX);
    _numYField.integerValue = static_cast<NSInteger>(countY);
    _numZField.integerValue = static_cast<NSInteger>(countZ);
    _numXSlider.doubleValue = InstanceCountToSliderPosition(
        countX, InstanceCountMin, InstanceCountSliderMax);
    _numYSlider.doubleValue = InstanceCountToSliderPosition(
        countY, InstanceCountMin, InstanceCountSliderMax);
    _numZSlider.doubleValue = InstanceCountToSliderPosition(
        countZ, InstanceCountMin, InstanceCountSliderMax);
}

- (void)applyInstanceGridFromControls:(id)sender
{
    (void)sender;
    std::uint64_t countX = static_cast<std::uint64_t>(
        std::max<NSInteger>(1, _numXField.integerValue));
    std::uint64_t countY = static_cast<std::uint64_t>(
        std::max<NSInteger>(1, _numYField.integerValue));
    std::uint64_t countZ = static_cast<std::uint64_t>(
        std::max<NSInteger>(1, _numZField.integerValue));
    const __uint128_t total = static_cast<__uint128_t>(countX) * countY * countZ;
    if (total > std::numeric_limits<std::uint32_t>::max())
    {
        NSLog(@"Instance grid total exceeds the 32-bit instance counter");
        return;
    }
    try
    {
        _renderer->SetInstanceGrid(static_cast<std::uint32_t>(countX),
                                   static_cast<std::uint32_t>(countY),
                                   static_cast<std::uint32_t>(countZ));
        [self syncInstanceGridControls];
        [self updateWindowTitle];
    }
    catch (const std::exception& error)
    {
        NSLog(@"Instance grid failed: %s", error.what());
    }
}

- (void)updateWindowTitle
{
    self.Window.title = [NSString stringWithFormat:@"Nanite | %.1f FPS", [self displayFps]];
    [self updateStatsPanel];
}

- (void)updateStatsPanel
{
    if (_renderer == nullptr || _statsTextView == nil)
        return;

    const Nanite::FrameStats& Stats = _renderer->GetLastFrameStats();
    NSMutableString* text = [NSMutableString string];
    [text appendFormat:@"FRAME\n  CPU/Present FPS             %7.1f\n", [self displayFps]];
    [text appendFormat:@"  Logical triangles            %.3f B\n",
                       static_cast<double>(Stats.LogicalTriangleCount) / 1000000000.0];
    [text appendFormat:@"  Instances                    %u (visible %u)\n",
                       Stats.InstanceCount, Stats.VisibleInstanceCount];
    [text appendFormat:@"  Cluster candidates           %u\n  Draw count                  %u\n",
                       Stats.ClusterCandidateCount, Stats.DrawCount];
    [text appendFormat:@"  Node tasks processed/written  %u / %u\n",
                       Stats.NodeTaskCount, Stats.NodeWriteCount];
    [text appendFormat:@"  Group tasks processed/written %u / %u\n",
                       Stats.GroupTaskCount, Stats.GroupWriteCount];
    [text appendFormat:@"  HZB                         %@\n  Overflow                    0x%X\n\n",
                       Stats.HZBEnabled ? @"ON" : @"OFF", Stats.OverflowFlags];

    [text appendString:@"CULLING\n"];
    [text appendFormat:@"  Group rejected: frustum %u, HZB %u\n",
                       Stats.GroupFrustumRejected, Stats.GroupHZBRejected];
    [text appendFormat:@"  Node LOD rejected            %u\n",
                       Stats.NodeLodRejected];
    [text appendFormat:@"  Cluster rejected: frustum %u, HZB %u\n",
                       Stats.ClusterFrustumRejected, Stats.ClusterHZBRejected];
    [text appendFormat:@"  Cluster rejected: refinement %u, tiny %u\n",
                       Stats.ClusterRefinementRejected, Stats.ClusterTinyRejected];
    [text appendFormat:@"  Post queue / HZB rejected    %u / %u\n",
                       Stats.PostClusterCount, Stats.PostClusterHZBRejected];
    [text appendFormat:@"  Post recovered: instance %u, node %u, group %u\n\n",
                       Stats.PostInstanceRecovered, Stats.PostNodeRecovered,
                       Stats.PostGroupRecovered];

    [text appendString:@"RASTER\n"];
    [text appendFormat:@"  HW cluster bin entries         %u\n  SW cluster bin entries         %u\n",
                       Stats.HardwareClusterCount, Stats.SoftClusterCount];
    [text appendString:@"  (Mixed clusters may be in both)\n"];
    [text appendFormat:@"  Auto raster route              %@\n",
                       Stats.AutoRasterEnabled ?
                           (Stats.AutoRasterHardware ? @"hardware" : @"hybrid") : @"manual"];
    [text appendFormat:@"  Software triangles / pixels   %u / %u\n",
                       Stats.SoftRasterTriangles, Stats.SoftRasterPixels];
    [text appendFormat:@"  Software skipped: large %u, clip %u\n",
                       Stats.SoftRasterSkippedLarge, Stats.SoftRasterSkippedClip];
    [text appendFormat:@"  Depth matched / mismatch      %u / %u\n",
                       Stats.SoftDepthMatched, Stats.SoftDepthMismatch];
    [text appendFormat:@"  Depth missing / extra         %u / %u\n",
                       Stats.SoftDepthMissing, Stats.SoftDepthExtra];
    [text appendFormat:@"  Visibility resolved/no cover  %u / %u\n",
                       Stats.SoftVisResolved, Stats.SoftVisNoCover];
    [text appendFormat:@"  Visibility depth-off/key bad  %u / %u\n\n",
                       Stats.SoftVisDepthOff, Stats.SoftVisKeyMismatch];

    [text appendString:@"GPU TIMINGS (ms)\n"];
    if (!Stats.Gpu.Enabled)
    {
        [text appendString:@"  Disabled\n"];
    }
    else if (!Stats.Gpu.Valid)
    {
        [text appendString:@"  Pending\n"];
    }
    else
    {
        const float rasterOptimisticBound = Stats.Gpu.AsyncRaster ?
            Stats.Gpu.RasterBinMs + std::max(Stats.Gpu.RasterMs, Stats.Gpu.SoftRasterMs) :
            Stats.Gpu.RasterBinMs + Stats.Gpu.RasterMs + Stats.Gpu.SoftRasterMs;
        const float rasterWorkSum = Stats.Gpu.RasterBinMs + Stats.Gpu.RasterMs +
            Stats.Gpu.SoftRasterMs;
        const float rasterSpan = Stats.Gpu.RasterBinMs + (Stats.Gpu.AsyncRaster ?
            std::max(Stats.Gpu.RasterMs, Stats.Gpu.SoftRasterMs) :
            Stats.Gpu.RasterMs + Stats.Gpu.SoftRasterMs);
        const float accountedGpu = Stats.Gpu.QueueResetMs +
            Stats.Gpu.InstanceCullMs + Stats.Gpu.PersistentCullMs +
            Stats.Gpu.GenerateIndirectMs + Stats.Gpu.PrepareDispatchMs +
            Stats.Gpu.PostRecoveryMs + Stats.Gpu.PostCullMs + rasterSpan +
            Stats.Gpu.PostRasterMs + Stats.Gpu.HizBuildMs +
            Stats.Gpu.FinalHizBuildMs + Stats.Gpu.HardwareDepthOnlyMs +
            Stats.Gpu.ShadowMapMs + Stats.Gpu.SoftCheckMs + Stats.Gpu.VisibilityResolveMs +
            Stats.Gpu.ShadingMs;
        const float gpuOther = Stats.Gpu.FrameMs - accountedGpu;
        [text appendFormat:@"  Frame                       %7.2f\n", Stats.Gpu.FrameMs];
        [text appendFormat:@"  GPU-bound FPS (Nanite)       %7.1f\n",
                           Stats.Gpu.FrameMs > 0.0f ? 1000.0f / Stats.Gpu.FrameMs : 0.0f];
        [text appendFormat:@"  Instance cull               %7.2f\n", Stats.Gpu.InstanceCullMs];
        [text appendFormat:@"  Persistent cull             %7.2f\n", Stats.Gpu.PersistentCullMs];
        [text appendFormat:@"  Generate indirect           %7.2f\n", Stats.Gpu.GenerateIndirectMs];
        [text appendFormat:@"  Post cull                   %7.2f\n", Stats.Gpu.PostCullMs];
        [text appendFormat:@"  Queue reset                 %7.2f\n", Stats.Gpu.QueueResetMs];
        [text appendFormat:@"  Prepare dispatch            %7.2f\n", Stats.Gpu.PrepareDispatchMs];
        [text appendFormat:@"  Post recovery               %7.2f\n", Stats.Gpu.PostRecoveryMs];
        [text appendFormat:@"  %@             %7.2f\n",
                           Stats.Gpu.AsyncRaster ? @"Raster optimistic bound" : @"Raster serial time",
                           rasterOptimisticBound];
        [text appendFormat:@"  Raster work sum             %7.2f\n", rasterWorkSum];
        [text appendFormat:@"    Raster bin (graphics)      %7.2f\n", Stats.Gpu.RasterBinMs];
        [text appendFormat:@"    Hardware raster work       %7.2f\n", Stats.Gpu.RasterMs];
        [text appendFormat:@"    Software raster work       %7.2f\n", Stats.Gpu.SoftRasterMs];
        [text appendFormat:@"      Software depth sweep     %7.2f\n", Stats.Gpu.SoftDepthMs];
        [text appendFormat:@"      Software payload sweep   %7.2f\n", Stats.Gpu.SoftVisibilityMs];
        [text appendFormat:@"    Hardware depth-only        %7.2f\n", Stats.Gpu.HardwareDepthOnlyMs];
        [text appendFormat:@"    Virtual shadow pages       %7.2f\n", Stats.Gpu.ShadowMapMs];
        [text appendFormat:@"    Software resolve check     %7.2f\n", Stats.Gpu.SoftCheckMs];
        [text appendFormat:@"  Visibility resolve           %7.2f\n", Stats.Gpu.VisibilityResolveMs];
        [text appendFormat:@"  Visibility shading           %7.2f\n", Stats.Gpu.ShadingMs];
        [text appendFormat:@"  Post raster                 %7.2f\n", Stats.Gpu.PostRasterMs];
        [text appendFormat:@"  HZB build                   %7.2f\n", Stats.Gpu.HizBuildMs];
        [text appendFormat:@"  Final HZB build             %7.2f\n", Stats.Gpu.FinalHizBuildMs];
        [text appendFormat:@"  GPU accounted total         %7.2f\n", accountedGpu];
        [text appendFormat:@"  GPU other (Frame-total)      %+7.2f\n", gpuOther];
    }
    _statsTextView.string = text;
}

- (BOOL)applicationShouldTerminateAfterLastWindowClosed:(NSApplication*)Application
{
    (void)Application;
    return YES;
}

- (void)applicationWillTerminate:(NSNotification*)Notification
{
    (void)Notification;
    // Stop the loop before the renderer goes away. Both of these are retained by
    // the run loop, so a callback could otherwise land on a destroyed renderer.
    if (@available(macOS 14.0, *))
        [(CADisplayLink*)self.DisplayLink invalidate];
    self.DisplayLink = nil;
    [self.Timer invalidate];
    self.Timer = nil;
    _renderer.reset();
}

@end

int main(int argc, const char* argv[])
{
    @autoreleasepool
    {
        (void)argc;
        (void)argv;
        // Prevent AppKit from showing a hidden restore-state modal after a crash.
        [[NSUserDefaults standardUserDefaults] setBool:YES forKey:@"ApplePersistenceIgnoreState"];
        NSApplication* Application = [NSApplication sharedApplication];
        AppDelegate* Delegate = [[AppDelegate alloc] init];
        Application.delegate = Delegate;
        [Application setActivationPolicy:NSApplicationActivationPolicyRegular];
        [Application run];
    }
    return 0;
}
