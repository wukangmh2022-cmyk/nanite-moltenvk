#include "NaniteGpuScene.hpp"

#include <Buffer.h>
#include <Sampler.h>
#include <ShaderResourceVariable.h>
#include <Texture.h>

#if defined(__APPLE__)
#    include <CoreFoundation/CoreFoundation.h>
#    include <CoreGraphics/CoreGraphics.h>
#    include <ImageIO/ImageIO.h>
#endif

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <string>
#include <stdexcept>

namespace
{

using namespace Diligent;

// Below this value the control displays 0.00 and routing a handful of
// near-degenerate triangles through the hybrid path is not useful. Keep the
// requested value intact for the UI, but use the stable hardware fast path.
constexpr float HybridRasterCutoffEpsilon = 1.0e-3f;

void Require(bool condition, const char* message)
{
    if (!condition)
        throw std::runtime_error{message};
}

float ReadFloatEnv(const char* name, float fallback)
{
    const char* value = std::getenv(name);
    if (value == nullptr || *value == '\0')
        return fallback;
    char* end = nullptr;
    const float parsed = std::strtof(value, &end);
    return end != value && std::isfinite(parsed) ? parsed : fallback;
}

struct ShadowFloat3
{
    float x;
    float y;
    float z;
};

ShadowFloat3 Add(ShadowFloat3 a, ShadowFloat3 b)
{
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}

ShadowFloat3 Subtract(ShadowFloat3 a, ShadowFloat3 b)
{
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}

ShadowFloat3 Scale(ShadowFloat3 value, float scalar)
{
    return {value.x * scalar, value.y * scalar, value.z * scalar};
}

float Dot(ShadowFloat3 a, ShadowFloat3 b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

ShadowFloat3 Cross(ShadowFloat3 a, ShadowFloat3 b)
{
    return {
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x};
}

ShadowFloat3 Normalize(ShadowFloat3 value)
{
    const float length = std::sqrt(std::max(Dot(value, value), 1.0e-12f));
    return Scale(value, 1.0f / length);
}

ShadowFloat3 TransformPoint(ShadowFloat3 point, const float matrix[16])
{
    // CPU scene matrices use row-vector storage: translation is in elements
    // 12..14, matching the CPU bounds/debug transform already used by Nanite.
    return {
        point.x * matrix[0] + point.y * matrix[4] + point.z * matrix[8] + matrix[12],
        point.x * matrix[1] + point.y * matrix[5] + point.z * matrix[9] + matrix[13],
        point.x * matrix[2] + point.y * matrix[6] + point.z * matrix[10] + matrix[14]};
}

std::array<float, 4> TransformHomogeneous(
    float x,
    float y,
    float z,
    float w,
    const float matrix[16])
{
    return {
        x * matrix[0] + y * matrix[4] + z * matrix[8] + w * matrix[12],
        x * matrix[1] + y * matrix[5] + z * matrix[9] + w * matrix[13],
        x * matrix[2] + y * matrix[6] + z * matrix[10] + w * matrix[14],
        x * matrix[3] + y * matrix[7] + z * matrix[11] + w * matrix[15]};
}

bool InvertMatrix4x4(const float input[16], float output[16])
{
    float augmented[4][8]{};
    for (int row = 0; row < 4; ++row)
    {
        for (int column = 0; column < 4; ++column)
            augmented[row][column] = input[row * 4 + column];
        augmented[row][row + 4] = 1.0f;
    }

    for (int column = 0; column < 4; ++column)
    {
        int pivot = column;
        for (int row = column + 1; row < 4; ++row)
        {
            if (std::abs(augmented[row][column]) > std::abs(augmented[pivot][column]))
                pivot = row;
        }
        if (std::abs(augmented[pivot][column]) < 1.0e-8f)
            return false;
        if (pivot != column)
            for (int entry = 0; entry < 8; ++entry)
                std::swap(augmented[pivot][entry], augmented[column][entry]);

        const float inversePivot = 1.0f / augmented[column][column];
        for (int entry = 0; entry < 8; ++entry)
            augmented[column][entry] *= inversePivot;
        for (int row = 0; row < 4; ++row)
        {
            if (row == column)
                continue;
            const float factor = augmented[row][column];
            for (int entry = 0; entry < 8; ++entry)
                augmented[row][entry] -= factor * augmented[column][entry];
        }
    }

    for (int row = 0; row < 4; ++row)
        for (int column = 0; column < 4; ++column)
            output[row * 4 + column] = augmented[row][column + 4];
    return true;
}

std::array<float, 16> MultiplyMatrices(
    const float left[16],
    const float right[16])
{
    std::array<float, 16> result{};
    for (int row = 0; row < 4; ++row)
        for (int column = 0; column < 4; ++column)
            for (int element = 0; element < 4; ++element)
                result[row * 4 + column] +=
                    left[row * 4 + element] * right[element * 4 + column];
    return result;
}

std::array<float, 16> BuildVirtualPageViewProjection(
    const float shadowViewProj[16],
    std::uint32_t pageCount,
    std::uint32_t pageX,
    std::uint32_t pageY)
{
    // v * shadowViewProj * pageTransform maps the requested virtual page to
    // the complete 128x128 physical render target.
    std::array<float, 16> pageTransform{
        static_cast<float>(pageCount), 0.0f, 0.0f, 0.0f,
        0.0f, static_cast<float>(pageCount), 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f, 0.0f,
        static_cast<float>(pageCount) - (2.0f * pageX + 1.0f),
        2.0f * pageY + 1.0f - static_cast<float>(pageCount),
        0.0f, 1.0f};
    return MultiplyMatrices(shadowViewProj, pageTransform.data());
}

std::array<float, 16> BuildShadowViewProjection(
    const Nanite::CpuScene& scene,
    std::uint32_t activeInstanceCount,
    const float sunDirection[3])
{
    ShadowFloat3 boundsMin{
        std::numeric_limits<float>::max(),
        std::numeric_limits<float>::max(),
        std::numeric_limits<float>::max()};
    ShadowFloat3 boundsMax{
        std::numeric_limits<float>::lowest(),
        std::numeric_limits<float>::lowest(),
        std::numeric_limits<float>::lowest()};
    bool hasBounds = false;

    const std::uint32_t instanceCount = std::min<std::uint32_t>(
        activeInstanceCount, static_cast<std::uint32_t>(scene.Instances.size()));
    for (std::uint32_t instanceIndex = 0; instanceIndex < instanceCount; ++instanceIndex)
    {
        const Nanite::Instance& instance = scene.Instances[instanceIndex];
        const Nanite::DagNode* root = instance.RootNodeIndex < scene.Nodes.size() ?
            &scene.Nodes[instance.RootNodeIndex] : nullptr;
        const float* localMin = root != nullptr ? root->BBoxMin : instance.BBoxMin;
        const float* localMax = root != nullptr ? root->BBoxMax : instance.BBoxMax;
        for (std::uint32_t corner = 0; corner < 8u; ++corner)
        {
            const ShadowFloat3 local{
                (corner & 1u) != 0u ? localMax[0] : localMin[0],
                (corner & 2u) != 0u ? localMax[1] : localMin[1],
                (corner & 4u) != 0u ? localMax[2] : localMin[2]};
            const ShadowFloat3 world = TransformPoint(local, instance.WorldMatrix);
            boundsMin.x = std::min(boundsMin.x, world.x);
            boundsMin.y = std::min(boundsMin.y, world.y);
            boundsMin.z = std::min(boundsMin.z, world.z);
            boundsMax.x = std::max(boundsMax.x, world.x);
            boundsMax.y = std::max(boundsMax.y, world.y);
            boundsMax.z = std::max(boundsMax.z, world.z);
            hasBounds = true;
        }
    }

    if (!hasBounds)
    {
        return {1.0f, 0.0f, 0.0f, 0.0f,
                0.0f, 1.0f, 0.0f, 0.0f,
                0.0f, 0.0f, 1.0f, 0.0f,
                0.0f, 0.0f, 0.0f, 1.0f};
    }

    const ShadowFloat3 center = Scale(Add(boundsMin, boundsMax), 0.5f);
    const ShadowFloat3 extent = Subtract(boundsMax, boundsMin);
    const float radius = std::max(0.5f * std::sqrt(Dot(extent, extent)), 1.0f);
    const ShadowFloat3 lightDirection = Normalize({
        sunDirection[0], sunDirection[1], sunDirection[2]});
    const float lightDistance = radius + 50.0f;
    const ShadowFloat3 lightPosition = Add(center, Scale(lightDirection, lightDistance));
    const ShadowFloat3 forward = Scale(lightDirection, -1.0f);
    const ShadowFloat3 worldUp = std::abs(forward.y) > 0.95f ?
        ShadowFloat3{0.0f, 0.0f, 1.0f} : ShadowFloat3{0.0f, 1.0f, 0.0f};
    const ShadowFloat3 right = Normalize(Cross(forward, worldUp));
    const ShadowFloat3 up = Normalize(Cross(right, forward));
    const ShadowFloat3 back = Scale(forward, -1.0f);

    const std::array<float, 16> view = {
        right.x, right.y, right.z, -Dot(right, lightPosition),
        up.x, up.y, up.z, -Dot(up, lightPosition),
        back.x, back.y, back.z, -Dot(back, lightPosition),
        0.0f, 0.0f, 0.0f, 1.0f};

    const float halfExtent = radius * 1.10f + 4.0f;
    const float nearPlane = 0.1f;
    const float farPlane = lightDistance + radius + 20.0f;
    const float inverseDepthRange = 1.0f / (nearPlane - farPlane);
    const std::array<float, 16> projection = {
        1.0f / halfExtent, 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f / halfExtent, 0.0f, 0.0f,
        0.0f, 0.0f, inverseDepthRange, 0.0f,
        0.0f, 0.0f, nearPlane * inverseDepthRange, 1.0f};

    // The shader's HLSL matrix multiply and the CPU debug transform use the
    // transposed form of the row-major P*V product, just like the camera matrix
    // built by DemoRenderer. Keeping that convention is what makes shadow UVs
    // agree with the cluster depth-only draw.
    std::array<float, 16> clip{};
    for (int row = 0; row < 4; ++row)
        for (int column = 0; column < 4; ++column)
            for (int element = 0; element < 4; ++element)
                clip[row * 4 + column] +=
                    projection[row * 4 + element] * view[element * 4 + column];

    std::array<float, 16> result{};
    for (int row = 0; row < 4; ++row)
        for (int column = 0; column < 4; ++column)
            result[row * 4 + column] = clip[column * 4 + row];
    return result;
}

std::uint64_t LogicalTriangleCount(const Nanite::CpuScene& scene,
                                   std::uint32_t activeCount)
{
    if (scene.InstanceTriangleCounts.size() >= activeCount)
        return std::accumulate(scene.InstanceTriangleCounts.begin(),
                               scene.InstanceTriangleCounts.begin() + activeCount,
                               std::uint64_t{0});
    return scene.SourceTriangleCount * activeCount;
}

template <typename T>
RefCntAutoPtr<IBuffer> CreateStructuredBuffer(
    IRenderDevice* device,
    const char* name,
    BIND_FLAGS bindFlags,
    const std::vector<T>& data,
    USAGE usage = USAGE_IMMUTABLE,
    Uint64 immediateContextMask = 1)
{
    BufferDesc desc;
    desc.Name = name;
    desc.Size = static_cast<Uint64>(data.size() * sizeof(T));
    desc.BindFlags = bindFlags;
    desc.Usage = usage;
    desc.ImmediateContextMask = immediateContextMask;
    desc.Mode = BUFFER_MODE_STRUCTURED;
    desc.ElementByteStride = sizeof(T);
    BufferData initialData{data.data(), desc.Size};

    RefCntAutoPtr<IBuffer> buffer;
    device->CreateBuffer(desc, &initialData, &buffer);
    Require(buffer != nullptr, "Failed to create immutable Nanite scene buffer");
    return buffer;
}

RefCntAutoPtr<IBuffer> CreateStructuredBuffer(
    IRenderDevice* device,
    const char* name,
    BIND_FLAGS bindFlags,
    Uint64 size,
    Uint32 stride,
    Uint64 immediateContextMask = 1)
{
    BufferDesc desc;
    desc.Name = name;
    desc.Size = size;
    desc.BindFlags = bindFlags;
    desc.Usage = USAGE_DEFAULT;
    desc.ImmediateContextMask = immediateContextMask;
    desc.Mode = BUFFER_MODE_STRUCTURED;
    desc.ElementByteStride = stride;

    RefCntAutoPtr<IBuffer> buffer;
    device->CreateBuffer(desc, nullptr, &buffer);
    Require(buffer != nullptr, "Failed to create Nanite structured buffer");
    return buffer;
}

RefCntAutoPtr<IBuffer> CreateConstantBuffer(
    IRenderDevice* device,
    const char* name,
    Uint32 size,
    Uint64 immediateContextMask = 1)
{
    BufferDesc desc;
    desc.Name = name;
    desc.Size = size;
    desc.BindFlags = BIND_UNIFORM_BUFFER;
    // These buffers are updated through UpdateBuffer. Dynamic buffers must be
    // updated with MapBuffer on Vulkan and are not valid UpdateBuffer targets.
    desc.Usage = USAGE_DEFAULT;
    desc.CPUAccessFlags = CPU_ACCESS_NONE;
    desc.ImmediateContextMask = immediateContextMask;

    RefCntAutoPtr<IBuffer> buffer;
    device->CreateBuffer(desc, nullptr, &buffer);
    Require(buffer != nullptr, "Failed to create Nanite constant buffer");
    return buffer;
}

RefCntAutoPtr<IBuffer> CreateDrawCommandBuffer(
    IRenderDevice* device,
    const char* name,
    Uint64 size,
    Uint32 stride,
    Uint64 immediateContextMask = 1)
{
    BufferDesc desc;
    desc.Name = name;
    desc.Size = size;
    desc.BindFlags = BIND_UNORDERED_ACCESS | BIND_INDIRECT_DRAW_ARGS;
    desc.Usage = USAGE_DEFAULT;
    // Raw so the shaders can address InstanceCount by byte offset for the atomic.
    desc.Mode = BUFFER_MODE_RAW;
    desc.ElementByteStride = stride;
    desc.ImmediateContextMask = immediateContextMask;

    RefCntAutoPtr<IBuffer> buffer;
    device->CreateBuffer(desc, nullptr, &buffer);
    Require(buffer != nullptr, "Failed to create Nanite draw command buffer");
    return buffer;
}

// ALLOW_OVERWRITE on every rebind, not just the HZB's: the SRBs live in Pipelines
// and outlive any one GpuScene, so a scene rebuild re-binds variables that already
// hold the previous scene's buffers. Without the flag that Set is a silent no-op,
// the SRB keeps the dead scene's buffers alive through its own reference, and the
// new scene's queue resets and constant uploads go to buffers no shader reads -
// which showed up as a black screen with draws=0 after a rebuild.
void SetBuffer(
    IShaderResourceBinding* binding,
    SHADER_TYPE stage,
    const char* name,
    IBuffer* buffer,
    BUFFER_VIEW_TYPE viewType)
{
    IShaderResourceVariable* variable = binding->GetVariableByName(stage, name);
    Require(variable != nullptr, name);
    IBufferView* view = buffer->GetDefaultView(viewType);
    Require(view != nullptr, "Nanite buffer view is unavailable");
    variable->Set(view, SET_SHADER_RESOURCE_FLAG_ALLOW_OVERWRITE);
    Require(variable->Get(0) == view, name);
}

void SetTexture(
    IShaderResourceBinding* binding,
    SHADER_TYPE stage,
    const char* name,
    ITextureView* view,
    SET_SHADER_RESOURCE_FLAGS flags = SET_SHADER_RESOURCE_FLAG_ALLOW_OVERWRITE)
{
    IShaderResourceVariable* variable = binding->GetVariableByName(stage, name);
    Require(variable != nullptr, name);
    variable->Set(view, flags);
    // Rebinding a non-dynamic variable that already holds an object is a silent
    // no-op unless ALLOW_OVERWRITE is passed; the internal check is compiled out
    // in release builds. Verify the descriptor really points at the new view so
    // a missing flag fails loudly instead of leaving a stale binding.
    Require(variable->Get(0) == view, name);
}

void SetSampler(
    IShaderResourceBinding* binding,
    SHADER_TYPE stage,
    const char* name,
    ISampler* sampler)
{
    IShaderResourceVariable* variable = binding->GetVariableByName(stage, name);
    Require(variable != nullptr, name);
    variable->Set(sampler, SET_SHADER_RESOURCE_FLAG_ALLOW_OVERWRITE);
}

void SetConstantBuffer(
    IShaderResourceBinding* binding,
    SHADER_TYPE stage,
    const char* name,
    IBuffer* buffer)
{
    IShaderResourceVariable* variable = binding->GetVariableByName(stage, name);
    Require(variable != nullptr, name);
    variable->Set(buffer, SET_SHADER_RESOURCE_FLAG_ALLOW_OVERWRITE);
    Require(variable->Get(0) == buffer, name);
}

#if defined(__APPLE__)
bool DecodeImageRGBA8(const std::string& path,
                      std::uint32_t width,
                      std::uint32_t height,
                      std::vector<std::uint8_t>& pixels)
{
    pixels.assign(static_cast<std::size_t>(width) * height * 4u, 255u);
    CFStringRef pathString = CFStringCreateWithCString(
        kCFAllocatorDefault, path.c_str(), kCFStringEncodingUTF8);
    if (pathString == nullptr)
        return false;
    CFURLRef url = CFURLCreateFromFileSystemRepresentation(
        kCFAllocatorDefault,
        reinterpret_cast<const UInt8*>(path.c_str()),
        path.size(),
        false);
    CGImageSourceRef source = url != nullptr ?
        CGImageSourceCreateWithURL(url, nullptr) : nullptr;
    CGImageRef image = source != nullptr ?
        CGImageSourceCreateImageAtIndex(source, 0, nullptr) : nullptr;
    CGColorSpaceRef colorSpace = CGColorSpaceCreateDeviceRGB();
    CGContextRef bitmap = image != nullptr && colorSpace != nullptr ?
        CGBitmapContextCreate(pixels.data(), width, height, 8, width * 4u,
                              colorSpace,
                              kCGImageAlphaPremultipliedLast |
                                  kCGBitmapByteOrder32Big) : nullptr;
    bool decoded = bitmap != nullptr;
    if (decoded)
    {
        CGContextSetInterpolationQuality(bitmap, kCGInterpolationHigh);
        // ImageIO's origin is top-left while texture V coordinates are bottom-up.
        CGContextTranslateCTM(bitmap, 0.0, static_cast<CGFloat>(height));
        CGContextScaleCTM(bitmap, 1.0, -1.0);
        CGContextDrawImage(bitmap,
                           CGRectMake(0.0, 0.0, static_cast<CGFloat>(width),
                                      static_cast<CGFloat>(height)),
                           image);
    }
    if (bitmap != nullptr)
        CGContextRelease(bitmap);
    if (colorSpace != nullptr)
        CGColorSpaceRelease(colorSpace);
    if (image != nullptr)
        CGImageRelease(image);
    if (source != nullptr)
        CFRelease(source);
    if (url != nullptr)
        CFRelease(url);
    CFRelease(pathString);
    return decoded;
}
#else
bool DecodeImageRGBA8(const std::string&,
                      std::uint32_t,
                      std::uint32_t,
                      std::vector<std::uint8_t>&)
{
    return false;
}
#endif

RefCntAutoPtr<ITexture> CreateTextureArrayFromPaths(
    IRenderDevice* device,
    const char* name,
    const std::vector<std::string>& paths,
    std::uint32_t width,
    std::uint32_t height,
    const std::array<std::uint8_t, 4>& fallback)
{
    const std::uint32_t layerCount = std::max<std::uint32_t>(
        static_cast<std::uint32_t>(paths.size()), 1u);
    std::vector<std::vector<std::uint8_t>> layerPixels(
        layerCount, std::vector<std::uint8_t>(
                        static_cast<std::size_t>(width) * height * 4u));
    std::vector<TextureSubResData> subresources(layerCount);
    for (std::uint32_t layer = 0; layer < layerCount; ++layer)
    {
        std::vector<std::uint8_t>& pixels = layerPixels[layer];
        std::fill(pixels.begin(), pixels.end(), 0u);
        for (std::size_t texel = 0; texel < pixels.size(); texel += 4u)
        {
            pixels[texel + 0u] = fallback[0];
            pixels[texel + 1u] = fallback[1];
            pixels[texel + 2u] = fallback[2];
            pixels[texel + 3u] = fallback[3];
        }
        if (layer < paths.size() && !paths[layer].empty())
        {
            if (!DecodeImageRGBA8(paths[layer], width, height, pixels))
                std::cerr << "Nanite PBR texture fallback: " << paths[layer] << '\n';
        }
        subresources[layer].pData = pixels.data();
        subresources[layer].Stride = width * 4u;
    }

    TextureData data{subresources.data(), layerCount};
    TextureDesc desc;
    desc.Name = name;
    desc.Type = RESOURCE_DIM_TEX_2D_ARRAY;
    desc.Width = width;
    desc.Height = height;
    desc.ArraySize = layerCount;
    desc.MipLevels = 1;
    desc.SampleCount = 1;
    desc.Format = TEX_FORMAT_RGBA8_UNORM;
    desc.Usage = USAGE_IMMUTABLE;
    desc.BindFlags = BIND_SHADER_RESOURCE;
    RefCntAutoPtr<ITexture> texture;
    device->CreateTexture(desc, &data, &texture);
    Require(texture != nullptr, "Failed to create Nanite PBR texture array");
    return texture;
}

RefCntAutoPtr<ITexture> CreateTexture2DFromPath(
    IRenderDevice* device,
    const char* name,
    const std::string& path,
    std::uint32_t width,
    std::uint32_t height)
{
    std::vector<std::uint8_t> pixels;
    if (!DecodeImageRGBA8(path, width, height, pixels))
    {
        std::cerr << "Nanite sky texture fallback: " << path << '\n';
        pixels.resize(static_cast<std::size_t>(width) * height * 4u);
        for (std::size_t texel = 0; texel < pixels.size(); texel += 4u)
        {
            const float t = static_cast<float>((texel / 4u) % width) /
                static_cast<float>(std::max(width - 1u, 1u));
            pixels[texel + 0u] = static_cast<std::uint8_t>(35.0f + 45.0f * t);
            pixels[texel + 1u] = static_cast<std::uint8_t>(90.0f + 55.0f * t);
            pixels[texel + 2u] = static_cast<std::uint8_t>(150.0f + 55.0f * t);
            pixels[texel + 3u] = 255u;
        }
    }
    TextureSubResData subresource{pixels.data(), width * 4u};
    TextureData data{&subresource, 1};
    TextureDesc desc;
    desc.Name = name;
    desc.Type = RESOURCE_DIM_TEX_2D;
    desc.Width = width;
    desc.Height = height;
    desc.MipLevels = 1;
    desc.SampleCount = 1;
    desc.Format = TEX_FORMAT_RGBA8_UNORM;
    desc.Usage = USAGE_IMMUTABLE;
    desc.BindFlags = BIND_SHADER_RESOURCE;
    RefCntAutoPtr<ITexture> texture;
    device->CreateTexture(desc, &data, &texture);
    Require(texture != nullptr, "Failed to create Nanite sky texture");
    return texture;
}

} // namespace

namespace Nanite
{

GpuScene::GpuScene(Diligent::IRenderDevice* device,
                   Pipelines& pipelines,
                   const CpuScene& scene,
                   const Capacity& capacity,
                   Diligent::Uint64 immediateContextMask) :
    m_Device{device},
    m_Pipelines{&pipelines},
    m_ImmediateContextMask{immediateContextMask},
    m_CpuScene{scene},
    m_Capacity{capacity}
{
    const auto readOnly = Diligent::BIND_SHADER_RESOURCE;
    const auto readWrite = Diligent::BIND_SHADER_RESOURCE | Diligent::BIND_UNORDERED_ACCESS;

    // Seed the two live settings from the environment once, here, instead of
    // re-reading getenv every frame: the control panel has to be able to override
    // them, and a per-frame getenv would silently undo whatever it set.
    if (std::getenv("NANITE_DISABLE_CLUSTER_CULL") != nullptr)
        m_ScreenErrorPixels = -1.0f;
    if (const char* screenError = std::getenv("NANITE_SCREEN_ERROR"))
        m_ScreenErrorPixels = std::strtof(screenError, nullptr);
    if (std::getenv("NANITE_VISUALIZE_CLUSTERS") != nullptr)
        m_ShadingMode = 5u;
    // The raw DebugMode value, for reaching a shading mode without the control
    // panel - which is what a scripted screenshot needs.
    if (const char* shadingMode = std::getenv("NANITE_SHADING_MODE"))
        m_ShadingMode = static_cast<std::uint32_t>(std::strtoul(shadingMode, nullptr, 10));
    if (const char* binArea = std::getenv("NANITE_RASTER_BIN_AREA"))
        m_RasterBinAreaCutoff = std::max(std::strtof(binArea, nullptr), 0.0f);
    AtmosphereSettings startupAtmosphere{};
    startupAtmosphere.SunAzimuthDeg = ReadFloatEnv(
        "NANITE_SUN_AZIMUTH_DEG", startupAtmosphere.SunAzimuthDeg);
    startupAtmosphere.SunElevationDeg = ReadFloatEnv(
        "NANITE_SUN_ELEVATION_DEG", startupAtmosphere.SunElevationDeg);
    startupAtmosphere.AtmosphereRadius = ReadFloatEnv(
        "NANITE_ATMOSPHERE_RADIUS", startupAtmosphere.AtmosphereRadius);
    startupAtmosphere.AtmosphereHeight = ReadFloatEnv(
        "NANITE_ATMOSPHERE_HEIGHT", startupAtmosphere.AtmosphereHeight);
    startupAtmosphere.RayleighScale = ReadFloatEnv(
        "NANITE_RAYLEIGH_SCALE", startupAtmosphere.RayleighScale);
    startupAtmosphere.MieScale = ReadFloatEnv(
        "NANITE_MIE_SCALE", startupAtmosphere.MieScale);
    startupAtmosphere.SkyDensity = ReadFloatEnv(
        "NANITE_ATMOSPHERE_DENSITY", startupAtmosphere.SkyDensity);
    startupAtmosphere.TerrainDensity = ReadFloatEnv(
        "NANITE_TERRAIN_ATMOSPHERE_DENSITY", startupAtmosphere.TerrainDensity);
    startupAtmosphere.HazeDistance = ReadFloatEnv(
        "NANITE_ATMOSPHERE_HAZE_DISTANCE", startupAtmosphere.HazeDistance);
    startupAtmosphere.HazeWeight = ReadFloatEnv(
        "NANITE_ATMOSPHERE_HAZE_WEIGHT", startupAtmosphere.HazeWeight);
    SetAtmosphereSettings(startupAtmosphere);
    m_ClusterRasterExperiment = std::getenv("NANITE_CLUSTER_RASTER_EXPERIMENT") != nullptr;
    if (const char* clusterArea = std::getenv("NANITE_CLUSTER_RASTER_AREA"))
        m_ClusterRasterAreaCutoff = std::max(std::strtof(clusterArea, nullptr), 0.0f);
    m_VsmEnabled = std::getenv("NANITE_DISABLE_VSM") == nullptr;
    if (const char* pageBudget = std::getenv("NANITE_VSM_PAGE_BUDGET"))
    {
        const unsigned long parsed = std::strtoul(pageBudget, nullptr, 10);
        m_VsmPageBudget = std::clamp<std::uint32_t>(
            static_cast<std::uint32_t>(parsed), 1u, VsmPhysicalPageCount);
    }

    m_PositionBuffer = CreateStructuredBuffer(
        device, "Nanite positions", readOnly, scene.Positions, Diligent::USAGE_IMMUTABLE,
        m_ImmediateContextMask);
    m_NormalBuffer = CreateStructuredBuffer(
        device, "Nanite normals", readOnly, scene.Normals, Diligent::USAGE_IMMUTABLE,
        m_ImmediateContextMask);
    m_ClusterPositionBuffer = CreateStructuredBuffer(
        device, "Nanite cluster-local positions", readOnly, scene.ClusterPositions,
        Diligent::USAGE_IMMUTABLE, m_ImmediateContextMask);
    m_ClusterIndexBuffer = CreateStructuredBuffer(
        device, "Nanite cluster-local indices", readOnly, scene.ClusterLocalIndices,
        Diligent::USAGE_IMMUTABLE, m_ImmediateContextMask);
    m_ClusterLocalIndexOffsetBuffer = CreateStructuredBuffer(
        device, "Nanite cluster-local index offsets", readOnly,
        scene.ClusterLocalIndexOffsets, Diligent::USAGE_IMMUTABLE,
        m_ImmediateContextMask);
    const std::vector<TexCoord> fallbackUvs = scene.TexCoords.empty() ?
        std::vector<TexCoord>(scene.Positions.size()) : scene.TexCoords;
    const std::vector<std::uint32_t> fallbackMaterialIndices = scene.MaterialIndices.empty() ?
        std::vector<std::uint32_t>(scene.Indices.size(), 0u) : scene.MaterialIndices;
    m_HasTextureCoordinates = !scene.Materials.empty() &&
        fallbackUvs.size() == scene.Positions.size();
    m_UvBuffer = CreateStructuredBuffer(
        device, "Nanite UV coordinates", readOnly, fallbackUvs,
        Diligent::USAGE_IMMUTABLE, m_ImmediateContextMask);
    m_MaterialIndexBuffer = CreateStructuredBuffer(
        device, "Nanite triangle material indices", readOnly, fallbackMaterialIndices,
        Diligent::USAGE_IMMUTABLE, m_ImmediateContextMask);

    // Imported glTF materials are preserved one-for-one. Position-only legacy
    // assets keep the small generated table that the original PBR experiment
    // used, so changing the scene source cannot make the shader bind an empty
    // structured buffer.
    std::vector<Material> materials = scene.Materials;
    std::vector<std::string> albedoPaths = scene.AlbedoTexturePaths;
    std::vector<std::string> normalPaths = scene.NormalTexturePaths;
    std::vector<std::string> metallicRoughnessPaths = scene.MetallicRoughnessTexturePaths;
    if (materials.empty())
    {
        materials = {
        {{0.72f, 0.68f, 0.62f, 1.0f}, 0.0f, 0.58f, 1.0f, 0u},
        {{0.58f, 0.68f, 0.82f, 1.0f}, 0.05f, 0.48f, 1.0f, 1u},
        {{0.76f, 0.46f, 0.28f, 1.0f}, 0.15f, 0.52f, 1.0f, 2u},
        {{0.42f, 0.72f, 0.50f, 1.0f}, 0.0f, 0.64f, 1.0f, 3u},
        };
        albedoPaths.resize(materials.size());
        normalPaths.resize(materials.size());
        metallicRoughnessPaths.resize(materials.size());
    }
    else
    {
        albedoPaths.resize(materials.size());
        normalPaths.resize(materials.size());
        metallicRoughnessPaths.resize(materials.size());
        for (std::size_t materialIndex = 0; materialIndex < materials.size(); ++materialIndex)
            materials[materialIndex].TextureIndex = static_cast<std::uint32_t>(materialIndex);
    }
    m_MaterialBuffer = CreateStructuredBuffer(
        device, "Nanite PBR materials", readOnly, materials,
        Diligent::USAGE_IMMUTABLE, m_ImmediateContextMask);
    // The instance transforms are the one part of the scene that changes at
    // runtime: the rabbit count slider re-lays out the cube and re-uploads them,
    // so this buffer cannot be immutable like the rest of the geometry.
    m_InstanceBuffer = CreateStructuredBuffer(
        device, "Nanite instances", readOnly, m_CpuScene.Instances, Diligent::USAGE_DEFAULT,
        m_ImmediateContextMask);
    m_NodeBuffer = CreateStructuredBuffer(
        device, "Nanite DAG nodes", readOnly, scene.Nodes, Diligent::USAGE_IMMUTABLE,
        m_ImmediateContextMask);
    m_ClusterGroupBuffer = CreateStructuredBuffer(
        device, "Nanite cluster groups", readOnly, scene.ClusterGroups, Diligent::USAGE_IMMUTABLE,
        m_ImmediateContextMask);
    m_ClusterBuffer = CreateStructuredBuffer(
        device, "Nanite clusters", readOnly, scene.Clusters, Diligent::USAGE_IMMUTABLE,
        m_ImmediateContextMask);

    Diligent::BufferDesc indexDesc;
    indexDesc.Name = "Nanite indices";
    indexDesc.Size = scene.Indices.size() * sizeof(std::uint32_t);
    // Pulled by the raster vertex shader rather than bound as an index buffer:
    // the single indirect command has only one FirstIndex field, so the
    // per-cluster index range has to be read in the shader.
    indexDesc.BindFlags = Diligent::BIND_SHADER_RESOURCE;
    indexDesc.Usage = Diligent::USAGE_IMMUTABLE;
    indexDesc.ImmediateContextMask = m_ImmediateContextMask;
    indexDesc.Mode = Diligent::BUFFER_MODE_STRUCTURED;
    indexDesc.ElementByteStride = sizeof(std::uint32_t);
    Diligent::BufferData indexData{scene.Indices.data(), indexDesc.Size};
    device->CreateBuffer(indexDesc, &indexData, &m_IndexBuffer);
    Require(m_IndexBuffer != nullptr, "Failed to create Nanite index buffer");

    m_NodeQueue = CreateStructuredBuffer(
        device, "Nanite node queue", readWrite,
        static_cast<Uint64>(m_Capacity.MaxNodeTasks) * sizeof(NodeTask), sizeof(NodeTask),
        m_ImmediateContextMask);
    m_GroupQueue = CreateStructuredBuffer(
        device, "Nanite group queue", readWrite,
        static_cast<Uint64>(m_Capacity.MaxGroupTasks) * sizeof(GroupTask), sizeof(GroupTask),
        m_ImmediateContextMask);
    m_ClusterQueue = CreateStructuredBuffer(
        device, "Nanite cluster queue", readWrite,
        static_cast<Uint64>(m_Capacity.MaxClusterTasks) * sizeof(ClusterTask), sizeof(ClusterTask),
        m_ImmediateContextMask);
    m_PostClusterQueue = CreateStructuredBuffer(
        device, "Nanite post cluster queue", readWrite,
        static_cast<Uint64>(m_Capacity.MaxClusterTasks) * sizeof(ClusterTask), sizeof(ClusterTask),
        m_ImmediateContextMask);
    // Occlusion recovery queues. The main phase can reject at most every
    // instance, node and group it looks at, so each queue matches the capacity
    // of the live queue it shadows.
    m_PostInstanceQueue = CreateStructuredBuffer(
        device, "Nanite post instance queue", readWrite,
        static_cast<Uint64>(std::max<std::size_t>(scene.Instances.size(), 1u)) *
            sizeof(std::uint32_t),
        sizeof(std::uint32_t), m_ImmediateContextMask);
    m_PostNodeQueue = CreateStructuredBuffer(
        device, "Nanite post node queue", readWrite,
        static_cast<Uint64>(m_Capacity.MaxNodeTasks) * sizeof(NodeTask), sizeof(NodeTask),
        m_ImmediateContextMask);
    m_PostGroupQueue = CreateStructuredBuffer(
        device, "Nanite post group queue", readWrite,
        static_cast<Uint64>(m_Capacity.MaxGroupTasks) * sizeof(GroupTask), sizeof(GroupTask),
        m_ImmediateContextMask);
    m_QueueState = CreateStructuredBuffer(
        device, "Nanite queue state", readWrite, sizeof(QueueState), sizeof(QueueState),
        m_ImmediateContextMask);
    // Two 16-byte VkDrawIndirectCommands: one for the main raster phase, one for
    // the clusters occlusion recovery adds. It is a raw buffer so the culling
    // shaders can run the InstanceCount atomics directly on it.
    m_DrawCommands = CreateDrawCommandBuffer(
        device,
        "Nanite indirect draw commands",
        static_cast<Uint64>(DrawCommandSlot::Count) * sizeof(DrawCommand),
        sizeof(std::uint32_t),
        m_ImmediateContextMask);
    m_HybridDrawCommand = CreateDrawCommandBuffer(
        device,
        "Nanite hybrid hardware draw command",
        sizeof(DrawCommand),
        sizeof(std::uint32_t),
        m_ImmediateContextMask);
    m_DrawInstanceIndices = CreateStructuredBuffer(
        device,
        "Nanite draw instance data",
        readWrite,
        static_cast<Uint64>(m_Capacity.MaxDrawCommands) * sizeof(DrawInstanceData),
        sizeof(DrawInstanceData),
        m_ImmediateContextMask);
    m_HardwareDrawIndices = CreateStructuredBuffer(
        device,
        "Nanite hybrid hardware cluster indices",
        readWrite,
        static_cast<Uint64>(m_Capacity.MaxDrawCommands) * 2u * sizeof(std::uint32_t),
        sizeof(std::uint32_t),
        m_ImmediateContextMask);
    // One storage buffer is shared by both lists. Keeping the descriptor single
    // avoids a MoltenVK reflection edge case with two identical RWStructuredBuffer
    // resources in one compute shader.
    m_SoftwareDrawIndices = m_HardwareDrawIndices;
    // Indirect dispatch arguments. Raw for the same reason as the draw command:
    // the prepare shader stores the three counts by byte offset. Both bind flags
    // are needed because the same buffer is a UAV for that shader and the
    // indirect argument source for the driver.
    m_DispatchArgs = CreateDrawCommandBuffer(
        device,
        "Nanite indirect dispatch args",
        static_cast<Uint64>(DispatchSlot::Count) * sizeof(DispatchCommand),
        sizeof(std::uint32_t),
        m_ImmediateContextMask);
    {
        constexpr Uint64 groupQueueOffset = 256;
        const bool dumpGroupTasks = std::getenv("NANITE_DUMP_QUEUE_STATE") != nullptr;
        Diligent::BufferDesc debugDesc;
        debugDesc.Name = "Nanite queue state readback";
        // Always the full 256 bytes: the queue state lands at 0 and the draw
        // command's InstanceCount after it, so growing QueueState cannot start
        // overwriting a counter that is read back in the same frame.
        debugDesc.Size = dumpGroupTasks ?
            groupQueueOffset + static_cast<Uint64>(m_Capacity.MaxGroupTasks) * sizeof(GroupTask) :
            groupQueueOffset;
        debugDesc.Usage = Diligent::USAGE_STAGING;
        debugDesc.CPUAccessFlags = Diligent::CPU_ACCESS_READ;
        device->CreateBuffer(debugDesc, nullptr, &m_DebugQueueReadback);
        Require(m_DebugQueueReadback != nullptr, "Failed to create Nanite queue state readback");

        if (const char* frame = std::getenv("NANITE_DUMP_QUEUE_STATE_FRAME"))
        {
            const unsigned long parsedFrame = std::strtoul(frame, nullptr, 10);
            m_DebugQueueReadbackFrame = static_cast<std::uint32_t>(std::max(1ul, parsedFrame));
        }
    }
    // The largest cluster decides the shared vertex count of the one command.
    m_MaxClusterIndexCount = scene.MaxClusterIndexCount;
    m_LastFrameStats.LogicalTriangleCount = LogicalTriangleCount(
        scene, static_cast<std::uint32_t>(scene.Instances.size()));
    m_LastFrameStats.InstanceCount = static_cast<std::uint32_t>(scene.Instances.size());
    m_ActiveInstanceCount = static_cast<std::uint32_t>(scene.Instances.size());
    m_CustomInstanceLayout = scene.CustomInstanceLayout;
    m_GpuTimingEnabled = std::getenv("NANITE_GPU_TIMINGS") != nullptr;
    // Hybrid rasterization is a runtime GUI feature. Keep its resources ready so
    // changing the cutoff immediately routes small triangles to software raster;
    // the environment variable is only an emergency opt-out.
    m_HybridRasterAvailable = std::getenv("NANITE_DISABLE_SOFT_RASTER") == nullptr;
    m_AutoRasterRouting = std::getenv("NANITE_AUTO_RASTER") != nullptr;
    if (m_AutoRasterRouting)
    {
        // The controller needs the previous frame's counters. Keep this opt-in
        // cost out of the normal fixed-cutoff path.
        m_StatsReadbackInterval = 1;
    }
    m_SoftProbeEnabled = std::getenv("NANITE_SOFT_RASTER_PROBE") != nullptr;
    m_QueueEpochEnabled = std::getenv("NANITE_DISABLE_QUEUE_EPOCH") == nullptr;
    const char* mslOverrideDir = std::getenv("DILIGENT_MSL_OVERRIDE_DIR");
    const char* nativeVisibility = std::getenv("NANITE_NATIVE_64BIT_VISIBILITY");
    const bool nativeVisibilityRequested = nativeVisibility != nullptr &&
        nativeVisibility[0] != '\0' && std::strcmp(nativeVisibility, "0") != 0;
    m_Native64BitVisibility = nativeVisibilityRequested && mslOverrideDir != nullptr &&
        std::ifstream{std::string{mslOverrideDir} + "/Nanite_Soft_Raster_Depth_CS.metal"}.good() &&
        std::ifstream{std::string{mslOverrideDir} + "/Nanite_Soft_Raster_Visibility_CS.metal"}.good();
    if (nativeVisibilityRequested && !m_Native64BitVisibility)
        std::cerr << "Nanite visibility: native 64-bit override requested but MSL files are unavailable; using fallback\n";
    if (m_Native64BitVisibility)
        std::cerr << "Nanite visibility: native Metal 64-bit packed atomic override enabled\n";
    m_SoftRasterEnabled = m_HybridRasterAvailable || m_SoftProbeEnabled;
    if (const char* repeat = std::getenv("NANITE_PROBE_REPEAT"))
    {
        const int parsed = std::atoi(repeat);
        m_ProbeRepeat = parsed > 0 ? static_cast<std::uint32_t>(parsed) : 1u;
    }
    m_LastFrameStats.SoftRasterEnabled = m_SoftRasterEnabled;
    m_LastFrameStats.AutoRasterEnabled = m_AutoRasterRouting;
    m_GpuTimingSupported =
        device->GetAdapterInfo().Features.TimestampQueries != Diligent::DEVICE_FEATURE_STATE_DISABLED;
    m_LastFrameStats.Gpu.Enabled = m_GpuTimingEnabled;
    // Create the query objects unconditionally so the UI can turn timings on and
    // off at runtime. Idle timestamp queries cost nothing.
    if (m_GpuTimingSupported)
    {
        constexpr const char* passNames[GpuTimingPassCount] = {
            "Nanite GPU Frame Total",
            "Nanite GPU Instance Cull",
            "Nanite GPU Persistent Cull",
            "Nanite GPU Generate Indirect",
            "Nanite GPU Post Cull",
            "Nanite GPU Raster",
            "Nanite GPU Post Raster",
            "Nanite GPU HZB Build",
            "Nanite GPU Soft Raster",
            "Nanite GPU Soft Depth Check",
            "Nanite GPU Hardware Depth Only",
            "Nanite GPU Raster Bin",
            "Nanite GPU Visibility Resolve",
            "Nanite GPU Final Visibility Resolve",
            "Nanite GPU Visibility Shading",
            "Nanite GPU Soft Visibility Payload",
            "Nanite GPU Queue Reset",
            "Nanite GPU Prepare Dispatch Main",
            "Nanite GPU Prepare Dispatch Post",
            "Nanite GPU Prepare Dispatch Soft",
            "Nanite GPU Post Recovery",
            "Nanite GPU Final HZB Build",
            "Nanite GPU Virtual Shadow Pages"};
        for (auto& slot : m_GpuTimingSlots)
        {
            for (std::uint32_t pass = 0; pass < GpuTimingPassCount; ++pass)
            {
                Diligent::QueryDesc queryDesc{Diligent::QUERY_TYPE_TIMESTAMP};
                queryDesc.Name = passNames[pass];
                device->CreateQuery(queryDesc, &slot.Queries[pass * 2]);
                device->CreateQuery(queryDesc, &slot.Queries[pass * 2 + 1]);
                if (slot.Queries[pass * 2] == nullptr || slot.Queries[pass * 2 + 1] == nullptr)
                {
                    m_GpuTimingSupported = false;
                    break;
                }
            }
            if (!m_GpuTimingSupported)
                break;
        }
    }
    if (m_GpuTimingEnabled && !m_GpuTimingSupported)
    {
        std::cerr << "Nanite GPU timings disabled: timestamp queries are unavailable\n";
        m_LastFrameStats.Gpu.Enabled = false;
    }
    m_CullingConstants = CreateConstantBuffer(
        device, "Nanite culling constants", sizeof(CullingConstants), m_ImmediateContextMask);
    m_RasterConstants = CreateConstantBuffer(
        device, "Nanite raster constants", sizeof(RasterConstants), m_ImmediateContextMask);
    m_HizConstants = CreateConstantBuffer(
        device, "Nanite HZB constants", sizeof(HizConstants), m_ImmediateContextMask);

    float hzbValue = 1.0f;
    Diligent::TextureSubResData hzbSubresource{&hzbValue, sizeof(float)};
    Diligent::TextureData hzbData{&hzbSubresource, 1};
    Diligent::TextureDesc hzbDesc{
        "Nanite disabled HZB",
        Diligent::RESOURCE_DIM_TEX_2D,
        1,
        1,
        1,
        Diligent::TEX_FORMAT_R32_FLOAT,
        1,
        1,
        Diligent::USAGE_IMMUTABLE,
        Diligent::BIND_SHADER_RESOURCE};
    device->CreateTexture(hzbDesc, &hzbData, &m_DisabledHZB);
    Require(m_DisabledHZB != nullptr, "Failed to create Nanite HZB placeholder");

    Diligent::SamplerDesc samplerDesc;
    samplerDesc.Name = "Nanite HZB point sampler";
    samplerDesc.MinFilter = Diligent::FILTER_TYPE_POINT;
    samplerDesc.MagFilter = Diligent::FILTER_TYPE_POINT;
    samplerDesc.MipFilter = Diligent::FILTER_TYPE_POINT;
    samplerDesc.AddressU = Diligent::TEXTURE_ADDRESS_CLAMP;
    samplerDesc.AddressV = Diligent::TEXTURE_ADDRESS_CLAMP;
    samplerDesc.AddressW = Diligent::TEXTURE_ADDRESS_CLAMP;
    device->CreateSampler(samplerDesc, &m_HZBSampler);
    Require(m_HZBSampler != nullptr, "Failed to create Nanite HZB sampler");

    // Decode the authored PolyHaven JPGs once into fixed-size arrays. Keeping a
    // uniform layer size makes one descriptor serve every imported material and
    // leaves the fallback path deterministic when an asset has no texture.
    constexpr Uint32 MaterialTextureSize = 1024;
    m_AlbedoTexture = CreateTextureArrayFromPaths(
        device, "Nanite PBR albedo array", albedoPaths,
        MaterialTextureSize, MaterialTextureSize, {255u, 255u, 255u, 255u});
    m_NormalTexture = CreateTextureArrayFromPaths(
        device, "Nanite PBR normal array", normalPaths,
        MaterialTextureSize, MaterialTextureSize, {128u, 128u, 255u, 255u});
    m_MetallicRoughnessTexture = CreateTextureArrayFromPaths(
        device, "Nanite PBR metallic roughness array", metallicRoughnessPaths,
        MaterialTextureSize, MaterialTextureSize, {255u, 255u, 0u, 255u});
    SamplerDesc albedoSamplerDesc;
    albedoSamplerDesc.Name = "Nanite PBR albedo sampler";
    albedoSamplerDesc.MinFilter = FILTER_TYPE_LINEAR;
    albedoSamplerDesc.MagFilter = FILTER_TYPE_LINEAR;
    albedoSamplerDesc.MipFilter = FILTER_TYPE_LINEAR;
    albedoSamplerDesc.AddressU = TEXTURE_ADDRESS_WRAP;
    albedoSamplerDesc.AddressV = TEXTURE_ADDRESS_WRAP;
    albedoSamplerDesc.AddressW = TEXTURE_ADDRESS_CLAMP;
    device->CreateSampler(albedoSamplerDesc, &m_AlbedoSampler);
    Require(m_AlbedoSampler != nullptr, "Failed to create Nanite PBR albedo sampler");

    if (const char* shadowSize = std::getenv("NANITE_SHADOW_SIZE"))
    {
        const unsigned long parsed = std::strtoul(shadowSize, nullptr, 10);
        if (parsed < 256ul || parsed > 4096ul)
            throw std::runtime_error{"NANITE_SHADOW_SIZE must be in [256, 4096]"};
        m_ShadowMapSize = static_cast<std::uint32_t>((parsed + VsmPageSize - 1u) /
                                                      VsmPageSize * VsmPageSize);
    }
    m_VsmPageCount = m_ShadowMapSize / VsmPageSize;
    const std::size_t virtualPageCount = static_cast<std::size_t>(m_VsmPageCount) *
        m_VsmPageCount;
    m_VsmPages.resize(virtualPageCount);
    m_VsmPageTable.assign(virtualPageCount, 0u);
    m_VsmPhysicalUsed.assign(VsmPhysicalPageCount, 0u);
    m_VsmPageTableBuffer = CreateStructuredBuffer(
        device, "Nanite VSM page table", readOnly, m_VsmPageTable,
        Diligent::USAGE_DEFAULT, m_ImmediateContextMask);

    Diligent::TextureDesc shadowDesc;
    shadowDesc.Name = "Nanite directional shadow map";
    shadowDesc.Type = Diligent::RESOURCE_DIM_TEX_2D;
    shadowDesc.Width = m_ShadowMapSize;
    shadowDesc.Height = m_ShadowMapSize;
    shadowDesc.Format = Diligent::TEX_FORMAT_R32_TYPELESS;
    shadowDesc.MipLevels = 1;
    shadowDesc.SampleCount = 1;
    shadowDesc.Usage = Diligent::USAGE_DEFAULT;
    shadowDesc.BindFlags = Diligent::BIND_DEPTH_STENCIL | Diligent::BIND_SHADER_RESOURCE;
    shadowDesc.ClearValue.Format = Diligent::TEX_FORMAT_D32_FLOAT;
    shadowDesc.ClearValue.DepthStencil.Depth = 1.0f;
    device->CreateTexture(shadowDesc, nullptr, &m_ShadowMapTexture);
    Require(m_ShadowMapTexture != nullptr, "Failed to create Nanite shadow map");
    m_ShadowMapDSV = m_ShadowMapTexture->GetDefaultView(
        Diligent::TEXTURE_VIEW_DEPTH_STENCIL);
    m_ShadowMapSRV = m_ShadowMapTexture->GetDefaultView(
        Diligent::TEXTURE_VIEW_SHADER_RESOURCE);
    Require(m_ShadowMapDSV != nullptr && m_ShadowMapSRV != nullptr,
            "Nanite shadow map views are unavailable");

    Diligent::SamplerDesc shadowSamplerDesc;
    shadowSamplerDesc.Name = "Nanite shadow map point sampler";
    shadowSamplerDesc.MinFilter = Diligent::FILTER_TYPE_POINT;
    shadowSamplerDesc.MagFilter = Diligent::FILTER_TYPE_POINT;
    shadowSamplerDesc.MipFilter = Diligent::FILTER_TYPE_POINT;
    shadowSamplerDesc.AddressU = Diligent::TEXTURE_ADDRESS_CLAMP;
    shadowSamplerDesc.AddressV = Diligent::TEXTURE_ADDRESS_CLAMP;
    shadowSamplerDesc.AddressW = Diligent::TEXTURE_ADDRESS_CLAMP;
    device->CreateSampler(shadowSamplerDesc, &m_ShadowSampler);
    Require(m_ShadowSampler != nullptr, "Failed to create Nanite shadow map sampler");

    Diligent::TextureDesc vsmPoolDesc;
    vsmPoolDesc.Name = "Nanite VSM physical page pool";
    vsmPoolDesc.Type = Diligent::RESOURCE_DIM_TEX_2D_ARRAY;
    vsmPoolDesc.Width = VsmPageSize;
    vsmPoolDesc.Height = VsmPageSize;
    vsmPoolDesc.ArraySize = VsmPhysicalPageCount;
    vsmPoolDesc.Format = Diligent::TEX_FORMAT_R32_TYPELESS;
    vsmPoolDesc.MipLevels = 1;
    vsmPoolDesc.SampleCount = 1;
    vsmPoolDesc.Usage = Diligent::USAGE_DEFAULT;
    vsmPoolDesc.BindFlags = Diligent::BIND_DEPTH_STENCIL | Diligent::BIND_SHADER_RESOURCE;
    vsmPoolDesc.ClearValue.Format = Diligent::TEX_FORMAT_D32_FLOAT;
    vsmPoolDesc.ClearValue.DepthStencil.Depth = 1.0f;
    device->CreateTexture(vsmPoolDesc, nullptr, &m_VsmPhysicalPoolTexture);
    Require(m_VsmPhysicalPoolTexture != nullptr, "Failed to create Nanite VSM page pool");
    m_VsmPhysicalPoolSRV = m_VsmPhysicalPoolTexture->GetDefaultView(
        Diligent::TEXTURE_VIEW_SHADER_RESOURCE);
    Require(m_VsmPhysicalPoolSRV != nullptr, "Nanite VSM page pool SRV is unavailable");
    m_VsmPhysicalPoolDSV.resize(VsmPhysicalPageCount);
    for (std::uint32_t page = 0; page < VsmPhysicalPageCount; ++page)
    {
        Diligent::TextureViewDesc pageViewDesc;
        pageViewDesc.Name = "Nanite VSM physical page DSV";
        pageViewDesc.ViewType = Diligent::TEXTURE_VIEW_DEPTH_STENCIL;
        pageViewDesc.TextureDim = Diligent::RESOURCE_DIM_TEX_2D_ARRAY;
        pageViewDesc.Format = Diligent::TEX_FORMAT_D32_FLOAT;
        pageViewDesc.MostDetailedMip = 0;
        pageViewDesc.NumMipLevels = 1;
        pageViewDesc.FirstArraySlice = page;
        pageViewDesc.NumArraySlices = 1;
        m_VsmPhysicalPoolTexture->CreateView(pageViewDesc, &m_VsmPhysicalPoolDSV[page]);
        Require(m_VsmPhysicalPoolDSV[page] != nullptr,
                "Nanite VSM physical page DSV is unavailable");
    }
    m_VsmSampler = m_ShadowSampler;

    m_EmptyNodeQueue.resize(m_Capacity.MaxNodeTasks);
    m_EmptyGroupQueue.resize(m_Capacity.MaxGroupTasks);
    BindResources(pipelines);
}

void GpuScene::BindResources(Pipelines& pipelines)
{
    using namespace Diligent;
    const bool useHzbSource = std::getenv("NANITE_USE_HZB_SOURCE") != nullptr;
    ITextureView* hzbView = useHzbSource && m_HzbSourceSRV != nullptr ?
        m_HzbSourceSRV :
        (m_HzbSRV != nullptr ?
             m_HzbSRV : m_DisabledHZB->GetDefaultView(TEXTURE_VIEW_SHADER_RESOURCE));
    Require(hzbView != nullptr, "Nanite HZB SRV is unavailable");

    IShaderResourceBinding* instanceCull = pipelines.InstanceCullBinding();
    SetConstantBuffer(instanceCull, SHADER_TYPE_COMPUTE, "NaniteCullingConstants", m_CullingConstants);
    SetBuffer(instanceCull, SHADER_TYPE_COMPUTE, "g_Instances", m_InstanceBuffer, BUFFER_VIEW_SHADER_RESOURCE);
    // The SRBs are created before the first resize and initially receive the
    // 1x1 disabled-HZB view. Allow the real HZB view to replace that mutable
    // binding after the render targets are created.
    SetTexture(instanceCull, SHADER_TYPE_COMPUTE, "g_HZB", hzbView,
               SET_SHADER_RESOURCE_FLAG_ALLOW_OVERWRITE);
    SetSampler(instanceCull, SHADER_TYPE_COMPUTE, "g_HZBSampler", m_HZBSampler);
    SetBuffer(instanceCull, SHADER_TYPE_COMPUTE, "g_NodeQueue", m_NodeQueue, BUFFER_VIEW_UNORDERED_ACCESS);
    SetBuffer(instanceCull, SHADER_TYPE_COMPUTE, "g_QueueState", m_QueueState, BUFFER_VIEW_UNORDERED_ACCESS);
    SetBuffer(instanceCull, SHADER_TYPE_COMPUTE, "g_PostInstanceQueue", m_PostInstanceQueue, BUFFER_VIEW_UNORDERED_ACCESS);

    IShaderResourceBinding* persistentCull = pipelines.PersistentCullBinding();
    SetConstantBuffer(persistentCull, SHADER_TYPE_COMPUTE, "NaniteCullingConstants", m_CullingConstants);
    SetBuffer(persistentCull, SHADER_TYPE_COMPUTE, "g_Nodes", m_NodeBuffer, BUFFER_VIEW_SHADER_RESOURCE);
    SetBuffer(persistentCull, SHADER_TYPE_COMPUTE, "g_ClusterGroups", m_ClusterGroupBuffer, BUFFER_VIEW_SHADER_RESOURCE);
    SetBuffer(persistentCull, SHADER_TYPE_COMPUTE, "g_Instances", m_InstanceBuffer, BUFFER_VIEW_SHADER_RESOURCE);
    // Only for the per-cluster RefinedGroupIndex: the group level now runs the same
    // refinement test generate-indirect does, so clusters whose finer group is
    // still needed never enter the cluster queue.
    SetBuffer(persistentCull, SHADER_TYPE_COMPUTE, "g_Clusters", m_ClusterBuffer, BUFFER_VIEW_SHADER_RESOURCE);
    SetTexture(persistentCull, SHADER_TYPE_COMPUTE, "g_HZB", hzbView,
               SET_SHADER_RESOURCE_FLAG_ALLOW_OVERWRITE);
    SetSampler(persistentCull, SHADER_TYPE_COMPUTE, "g_HZBSampler", m_HZBSampler);
    SetBuffer(persistentCull, SHADER_TYPE_COMPUTE, "g_NodeQueue", m_NodeQueue, BUFFER_VIEW_UNORDERED_ACCESS);
    SetBuffer(persistentCull, SHADER_TYPE_COMPUTE, "g_GroupQueue", m_GroupQueue, BUFFER_VIEW_UNORDERED_ACCESS);
    SetBuffer(persistentCull, SHADER_TYPE_COMPUTE, "g_ClusterQueue", m_ClusterQueue, BUFFER_VIEW_UNORDERED_ACCESS);
    SetBuffer(persistentCull, SHADER_TYPE_COMPUTE, "g_QueueState", m_QueueState, BUFFER_VIEW_UNORDERED_ACCESS);
    SetBuffer(persistentCull, SHADER_TYPE_COMPUTE, "g_PostNodeQueue", m_PostNodeQueue, BUFFER_VIEW_UNORDERED_ACCESS);
    SetBuffer(persistentCull, SHADER_TYPE_COMPUTE, "g_PostGroupQueue", m_PostGroupQueue, BUFFER_VIEW_UNORDERED_ACCESS);

    // Seeds the recovery traversal from the node and group queues above.
    IShaderResourceBinding* postSeed = pipelines.PostSeedBinding();
    SetConstantBuffer(postSeed, SHADER_TYPE_COMPUTE, "NaniteCullingConstants", m_CullingConstants);
    SetBuffer(postSeed, SHADER_TYPE_COMPUTE, "g_NodeQueue", m_NodeQueue, BUFFER_VIEW_UNORDERED_ACCESS);
    SetBuffer(postSeed, SHADER_TYPE_COMPUTE, "g_GroupQueue", m_GroupQueue, BUFFER_VIEW_UNORDERED_ACCESS);
    SetBuffer(postSeed, SHADER_TYPE_COMPUTE, "g_QueueState", m_QueueState, BUFFER_VIEW_UNORDERED_ACCESS);
    SetBuffer(postSeed, SHADER_TYPE_COMPUTE, "g_PostNodeQueue", m_PostNodeQueue, BUFFER_VIEW_UNORDERED_ACCESS);
    SetBuffer(postSeed, SHADER_TYPE_COMPUTE, "g_PostGroupQueue", m_PostGroupQueue, BUFFER_VIEW_UNORDERED_ACCESS);

    // Turns the traversal's counters into the dispatch grids of everything
    // downstream of it, so those passes stop being launched over the full
    // provisioned capacity.
    IShaderResourceBinding* prepareDispatch = pipelines.PrepareDispatchBinding();
    SetConstantBuffer(prepareDispatch, SHADER_TYPE_COMPUTE, "NaniteCullingConstants", m_CullingConstants);
    SetBuffer(prepareDispatch, SHADER_TYPE_COMPUTE, "g_DispatchArgs", m_DispatchArgs, BUFFER_VIEW_UNORDERED_ACCESS);
    SetBuffer(prepareDispatch, SHADER_TYPE_COMPUTE, "g_QueueState", m_QueueState, BUFFER_VIEW_UNORDERED_ACCESS);

    IShaderResourceBinding* rasterBin = pipelines.RasterBinBinding();
    SetConstantBuffer(rasterBin, SHADER_TYPE_COMPUTE, "NaniteCullingConstants", m_CullingConstants);
    SetBuffer(rasterBin, SHADER_TYPE_COMPUTE, "g_PositionBuffer", m_PositionBuffer, BUFFER_VIEW_SHADER_RESOURCE);
    SetBuffer(rasterBin, SHADER_TYPE_COMPUTE, "g_Instances", m_InstanceBuffer, BUFFER_VIEW_SHADER_RESOURCE);
    SetBuffer(rasterBin, SHADER_TYPE_COMPUTE, "g_IndexBuffer", m_IndexBuffer, BUFFER_VIEW_SHADER_RESOURCE);
    SetBuffer(rasterBin, SHADER_TYPE_COMPUTE, "g_DrawInstanceData", m_DrawInstanceIndices, BUFFER_VIEW_SHADER_RESOURCE);
    SetBuffer(rasterBin, SHADER_TYPE_COMPUTE, "g_HybridDrawCommand", m_HybridDrawCommand, BUFFER_VIEW_UNORDERED_ACCESS);
    SetBuffer(rasterBin, SHADER_TYPE_COMPUTE, "g_QueueState", m_QueueState, BUFFER_VIEW_UNORDERED_ACCESS);
    SetBuffer(rasterBin, SHADER_TYPE_COMPUTE, "g_BinDrawIndices", m_HardwareDrawIndices, BUFFER_VIEW_UNORDERED_ACCESS);

    IShaderResourceBinding* generateIndirect = pipelines.GenerateIndirectBinding();
    SetConstantBuffer(generateIndirect, SHADER_TYPE_COMPUTE, "NaniteCullingConstants", m_CullingConstants);
    SetBuffer(generateIndirect, SHADER_TYPE_COMPUTE, "g_Instances", m_InstanceBuffer, BUFFER_VIEW_SHADER_RESOURCE);
    SetBuffer(generateIndirect, SHADER_TYPE_COMPUTE, "g_Clusters", m_ClusterBuffer, BUFFER_VIEW_SHADER_RESOURCE);
    SetBuffer(generateIndirect, SHADER_TYPE_COMPUTE, "g_ClusterQueue", m_ClusterQueue, BUFFER_VIEW_SHADER_RESOURCE);
    SetBuffer(generateIndirect, SHADER_TYPE_COMPUTE, "g_ClusterGroups", m_ClusterGroupBuffer, BUFFER_VIEW_SHADER_RESOURCE);
    SetTexture(generateIndirect, SHADER_TYPE_COMPUTE, "g_HZB", hzbView,
               SET_SHADER_RESOURCE_FLAG_ALLOW_OVERWRITE);
    SetSampler(generateIndirect, SHADER_TYPE_COMPUTE, "g_HZBSampler", m_HZBSampler);
    SetBuffer(generateIndirect, SHADER_TYPE_COMPUTE, "g_DrawCommand", m_DrawCommands, BUFFER_VIEW_UNORDERED_ACCESS);
    SetBuffer(generateIndirect, SHADER_TYPE_COMPUTE, "g_DrawInstanceData", m_DrawInstanceIndices, BUFFER_VIEW_UNORDERED_ACCESS);
    SetBuffer(generateIndirect, SHADER_TYPE_COMPUTE, "g_QueueState", m_QueueState, BUFFER_VIEW_UNORDERED_ACCESS);
    SetBuffer(generateIndirect, SHADER_TYPE_COMPUTE, "g_PostClusterQueue", m_PostClusterQueue, BUFFER_VIEW_UNORDERED_ACCESS);

    IShaderResourceBinding* postCull = pipelines.PostCullBinding();
    SetConstantBuffer(postCull, SHADER_TYPE_COMPUTE, "NaniteCullingConstants", m_CullingConstants);
    SetBuffer(postCull, SHADER_TYPE_COMPUTE, "g_Instances", m_InstanceBuffer, BUFFER_VIEW_SHADER_RESOURCE);
    SetBuffer(postCull, SHADER_TYPE_COMPUTE, "g_Clusters", m_ClusterBuffer, BUFFER_VIEW_SHADER_RESOURCE);
    SetBuffer(postCull, SHADER_TYPE_COMPUTE, "g_PostClusterQueue", m_PostClusterQueue, BUFFER_VIEW_SHADER_RESOURCE);
    SetTexture(postCull, SHADER_TYPE_COMPUTE, "g_HZB", hzbView,
               SET_SHADER_RESOURCE_FLAG_ALLOW_OVERWRITE);
    SetSampler(postCull, SHADER_TYPE_COMPUTE, "g_HZBSampler", m_HZBSampler);
    SetBuffer(postCull, SHADER_TYPE_COMPUTE, "g_DrawCommand", m_DrawCommands, BUFFER_VIEW_UNORDERED_ACCESS);
    SetBuffer(postCull, SHADER_TYPE_COMPUTE, "g_DrawInstanceData", m_DrawInstanceIndices, BUFFER_VIEW_UNORDERED_ACCESS);
    SetBuffer(postCull, SHADER_TYPE_COMPUTE, "g_QueueState", m_QueueState, BUFFER_VIEW_UNORDERED_ACCESS);

    IShaderResourceBinding* raster = pipelines.RasterBinding();
    SetConstantBuffer(raster, SHADER_TYPE_VERTEX, "NaniteRasterCB", m_RasterConstants);
    SetConstantBuffer(raster, SHADER_TYPE_PIXEL, "NaniteRasterCB", m_RasterConstants);
    SetBuffer(raster, SHADER_TYPE_VERTEX, "g_PositionBuffer", m_PositionBuffer, BUFFER_VIEW_SHADER_RESOURCE);
    SetBuffer(raster, SHADER_TYPE_VERTEX, "g_NormalBuffer", m_NormalBuffer, BUFFER_VIEW_SHADER_RESOURCE);
    SetBuffer(raster, SHADER_TYPE_VERTEX, "g_Instances", m_InstanceBuffer, BUFFER_VIEW_SHADER_RESOURCE);
    SetBuffer(raster, SHADER_TYPE_VERTEX, "g_IndexBuffer", m_IndexBuffer, BUFFER_VIEW_SHADER_RESOURCE);
    SetBuffer(raster, SHADER_TYPE_VERTEX, "g_DrawInstanceData", m_DrawInstanceIndices, BUFFER_VIEW_SHADER_RESOURCE);
    SetBuffer(raster, SHADER_TYPE_VERTEX, "g_HardwareDrawIndices", m_HardwareDrawIndices, BUFFER_VIEW_SHADER_RESOURCE);
    ITextureView* debugDepthView =
        (m_DepthColorSRV != nullptr && std::getenv("NANITE_VISUALIZE_DEPTH") != nullptr) ?
        m_DepthColorSRV : hzbView;
    SetTexture(raster, SHADER_TYPE_PIXEL, "g_HZB", debugDepthView,
               SET_SHADER_RESOURCE_FLAG_ALLOW_OVERWRITE);
    SetSampler(raster, SHADER_TYPE_PIXEL, "g_HZBSampler", m_HZBSampler);

    IShaderResourceBinding* postRaster = pipelines.PostRasterBinding();
    SetConstantBuffer(postRaster, SHADER_TYPE_VERTEX, "NaniteRasterCB", m_RasterConstants);
    SetConstantBuffer(postRaster, SHADER_TYPE_PIXEL, "NaniteRasterCB", m_RasterConstants);
    SetBuffer(postRaster, SHADER_TYPE_VERTEX, "g_PositionBuffer", m_PositionBuffer, BUFFER_VIEW_SHADER_RESOURCE);
    SetBuffer(postRaster, SHADER_TYPE_VERTEX, "g_NormalBuffer", m_NormalBuffer, BUFFER_VIEW_SHADER_RESOURCE);
    SetBuffer(postRaster, SHADER_TYPE_VERTEX, "g_Instances", m_InstanceBuffer, BUFFER_VIEW_SHADER_RESOURCE);
    SetBuffer(postRaster, SHADER_TYPE_VERTEX, "g_IndexBuffer", m_IndexBuffer, BUFFER_VIEW_SHADER_RESOURCE);
    SetBuffer(postRaster, SHADER_TYPE_VERTEX, "g_DrawInstanceData", m_DrawInstanceIndices, BUFFER_VIEW_SHADER_RESOURCE);
    SetBuffer(postRaster, SHADER_TYPE_VERTEX, "g_HardwareDrawIndices", m_HardwareDrawIndices, BUFFER_VIEW_SHADER_RESOURCE);
    SetTexture(postRaster, SHADER_TYPE_PIXEL, "g_HZB", debugDepthView,
               SET_SHADER_RESOURCE_FLAG_ALLOW_OVERWRITE);
    SetSampler(postRaster, SHADER_TYPE_PIXEL, "g_HZBSampler", m_HZBSampler);

    // Same resources again for the overdraw pass: it differs from the main raster
    // only in its blend and depth state, so it needs its own binding but not its
    // own buffers.
    IShaderResourceBinding* overdraw = pipelines.OverdrawRasterBinding();
    SetConstantBuffer(overdraw, SHADER_TYPE_VERTEX, "NaniteRasterCB", m_RasterConstants);
    SetConstantBuffer(overdraw, SHADER_TYPE_PIXEL, "NaniteRasterCB", m_RasterConstants);
    SetBuffer(overdraw, SHADER_TYPE_VERTEX, "g_PositionBuffer", m_PositionBuffer, BUFFER_VIEW_SHADER_RESOURCE);
    SetBuffer(overdraw, SHADER_TYPE_VERTEX, "g_NormalBuffer", m_NormalBuffer, BUFFER_VIEW_SHADER_RESOURCE);
    SetBuffer(overdraw, SHADER_TYPE_VERTEX, "g_Instances", m_InstanceBuffer, BUFFER_VIEW_SHADER_RESOURCE);
    SetBuffer(overdraw, SHADER_TYPE_VERTEX, "g_IndexBuffer", m_IndexBuffer, BUFFER_VIEW_SHADER_RESOURCE);
    SetBuffer(overdraw, SHADER_TYPE_VERTEX, "g_DrawInstanceData", m_DrawInstanceIndices, BUFFER_VIEW_SHADER_RESOURCE);
    SetBuffer(overdraw, SHADER_TYPE_VERTEX, "g_HardwareDrawIndices", m_HardwareDrawIndices, BUFFER_VIEW_SHADER_RESOURCE);
    SetTexture(overdraw, SHADER_TYPE_PIXEL, "g_HZB", debugDepthView,
               SET_SHADER_RESOURCE_FLAG_ALLOW_OVERWRITE);
    SetSampler(overdraw, SHADER_TYPE_PIXEL, "g_HZBSampler", m_HZBSampler);

    // The depth buffer only exists from the first EnsureRenderTargets onwards, and
    // this runs once before that, so the probe's bindings wait for it. Both
    // resources are ALLOW_OVERWRITE because the pair is rebound on every resize.
    if (m_SoftRasterEnabled && m_SoftDepthBuffer != nullptr)
    {
        IShaderResourceBinding* softRaster = pipelines.SoftRasterBinding();
        SetConstantBuffer(softRaster, SHADER_TYPE_COMPUTE, "NaniteCullingConstants", m_CullingConstants);
        SetBuffer(softRaster, SHADER_TYPE_COMPUTE, "g_ClusterPositionBuffer",
                  m_ClusterPositionBuffer, BUFFER_VIEW_SHADER_RESOURCE);
        SetBuffer(softRaster, SHADER_TYPE_COMPUTE, "g_Instances", m_InstanceBuffer, BUFFER_VIEW_SHADER_RESOURCE);
        SetBuffer(softRaster, SHADER_TYPE_COMPUTE, "g_ClusterIndexBuffer",
                  m_ClusterIndexBuffer, BUFFER_VIEW_SHADER_RESOURCE);
        SetBuffer(softRaster, SHADER_TYPE_COMPUTE, "g_ClusterLocalIndexOffsets",
                  m_ClusterLocalIndexOffsetBuffer, BUFFER_VIEW_SHADER_RESOURCE);
        SetBuffer(softRaster, SHADER_TYPE_COMPUTE, "g_Clusters", m_ClusterBuffer,
                  BUFFER_VIEW_SHADER_RESOURCE);
        SetBuffer(softRaster, SHADER_TYPE_COMPUTE, "g_DrawInstanceData", m_DrawInstanceIndices, BUFFER_VIEW_SHADER_RESOURCE);
        SetBuffer(softRaster, SHADER_TYPE_COMPUTE, "g_SoftwareDrawIndices", m_SoftwareDrawIndices, BUFFER_VIEW_SHADER_RESOURCE);
        SetBuffer(softRaster, SHADER_TYPE_COMPUTE, "g_SoftDepth", m_SoftDepthBuffer, BUFFER_VIEW_UNORDERED_ACCESS);
        SetBuffer(softRaster, SHADER_TYPE_COMPUTE, "g_Visibility", m_SoftVisBuffer,
                  BUFFER_VIEW_UNORDERED_ACCESS);
        SetBuffer(softRaster, SHADER_TYPE_COMPUTE, "g_QueueState", m_QueueState, BUFFER_VIEW_UNORDERED_ACCESS);

        IShaderResourceBinding* softVisibility = pipelines.SoftVisibilityBinding();
        SetConstantBuffer(softVisibility, SHADER_TYPE_COMPUTE, "NaniteCullingConstants", m_CullingConstants);
        SetBuffer(softVisibility, SHADER_TYPE_COMPUTE, "g_ClusterPositionBuffer",
                  m_ClusterPositionBuffer, BUFFER_VIEW_SHADER_RESOURCE);
        SetBuffer(softVisibility, SHADER_TYPE_COMPUTE, "g_Instances", m_InstanceBuffer, BUFFER_VIEW_SHADER_RESOURCE);
        SetBuffer(softVisibility, SHADER_TYPE_COMPUTE, "g_ClusterIndexBuffer",
                  m_ClusterIndexBuffer, BUFFER_VIEW_SHADER_RESOURCE);
        SetBuffer(softVisibility, SHADER_TYPE_COMPUTE, "g_ClusterLocalIndexOffsets",
                  m_ClusterLocalIndexOffsetBuffer, BUFFER_VIEW_SHADER_RESOURCE);
        SetBuffer(softVisibility, SHADER_TYPE_COMPUTE, "g_Clusters", m_ClusterBuffer,
                  BUFFER_VIEW_SHADER_RESOURCE);
        SetBuffer(softVisibility, SHADER_TYPE_COMPUTE, "g_DrawInstanceData", m_DrawInstanceIndices, BUFFER_VIEW_SHADER_RESOURCE);
        SetBuffer(softVisibility, SHADER_TYPE_COMPUTE, "g_SoftwareDrawIndices", m_SoftwareDrawIndices, BUFFER_VIEW_SHADER_RESOURCE);
        SetBuffer(softVisibility, SHADER_TYPE_COMPUTE, "g_SoftDepth", m_SoftDepthBuffer, BUFFER_VIEW_UNORDERED_ACCESS);
        SetBuffer(softVisibility, SHADER_TYPE_COMPUTE, "g_Visibility", m_SoftVisBuffer, BUFFER_VIEW_UNORDERED_ACCESS);
        SetBuffer(softVisibility, SHADER_TYPE_COMPUTE, "g_QueueState", m_QueueState, BUFFER_VIEW_UNORDERED_ACCESS);

        IShaderResourceBinding* softCheck = pipelines.SoftCheckBinding();
        SetConstantBuffer(softCheck, SHADER_TYPE_COMPUTE, "NaniteCullingConstants", m_CullingConstants);
        SetTexture(softCheck, SHADER_TYPE_COMPUTE, "g_HardwareDepth", m_DepthColorSRV,
                   SET_SHADER_RESOURCE_FLAG_ALLOW_OVERWRITE);
        SetBuffer(softCheck, SHADER_TYPE_COMPUTE, "g_SoftDepth", m_SoftDepthBuffer, BUFFER_VIEW_UNORDERED_ACCESS);
        SetBuffer(softCheck, SHADER_TYPE_COMPUTE, "g_Visibility", m_SoftVisBuffer, BUFFER_VIEW_UNORDERED_ACCESS);
        SetBuffer(softCheck, SHADER_TYPE_COMPUTE, "g_QueueState", m_QueueState, BUFFER_VIEW_UNORDERED_ACCESS);
        // The check pass resolves visibility payloads back to triangles, so it needs
        // the same geometry the rasterizer read.
        SetBuffer(softCheck, SHADER_TYPE_COMPUTE, "g_PositionBuffer", m_PositionBuffer, BUFFER_VIEW_SHADER_RESOURCE);
        SetBuffer(softCheck, SHADER_TYPE_COMPUTE, "g_Instances", m_InstanceBuffer, BUFFER_VIEW_SHADER_RESOURCE);
        SetBuffer(softCheck, SHADER_TYPE_COMPUTE, "g_IndexBuffer", m_IndexBuffer, BUFFER_VIEW_SHADER_RESOURCE);
        SetBuffer(softCheck, SHADER_TYPE_COMPUTE, "g_DrawInstanceData", m_DrawInstanceIndices, BUFFER_VIEW_SHADER_RESOURCE);

        // The depth-only reference runs the raster vertex shader with no pixel
        // shader, so it needs the vertex-stage resources and nothing else.
        IShaderResourceBinding* depthOnly = pipelines.HardwareDepthOnlyBinding();
        SetConstantBuffer(depthOnly, SHADER_TYPE_VERTEX, "NaniteRasterCB", m_RasterConstants);
        SetBuffer(depthOnly, SHADER_TYPE_VERTEX, "g_PositionBuffer", m_PositionBuffer, BUFFER_VIEW_SHADER_RESOURCE);
        SetBuffer(depthOnly, SHADER_TYPE_VERTEX, "g_NormalBuffer", m_NormalBuffer, BUFFER_VIEW_SHADER_RESOURCE);
        SetBuffer(depthOnly, SHADER_TYPE_VERTEX, "g_Instances", m_InstanceBuffer, BUFFER_VIEW_SHADER_RESOURCE);
        SetBuffer(depthOnly, SHADER_TYPE_VERTEX, "g_IndexBuffer", m_IndexBuffer, BUFFER_VIEW_SHADER_RESOURCE);
        SetBuffer(depthOnly, SHADER_TYPE_VERTEX, "g_DrawInstanceData", m_DrawInstanceIndices, BUFFER_VIEW_SHADER_RESOURCE);
        SetBuffer(depthOnly, SHADER_TYPE_VERTEX, "g_HardwareDrawIndices", m_HardwareDrawIndices, BUFFER_VIEW_SHADER_RESOURCE);

        IShaderResourceBinding* visibilityResolve = pipelines.VisibilityResolveBinding();
        SetConstantBuffer(visibilityResolve, SHADER_TYPE_COMPUTE, "NaniteCullingConstants", m_CullingConstants);
        SetTexture(visibilityResolve, SHADER_TYPE_COMPUTE, "g_HardwareDepth", m_DepthColorSRV,
                   SET_SHADER_RESOURCE_FLAG_ALLOW_OVERWRITE);
        SetTexture(visibilityResolve, SHADER_TYPE_COMPUTE, "g_HardwareVisibility", m_HardwareVisibilitySRV,
                   SET_SHADER_RESOURCE_FLAG_ALLOW_OVERWRITE);
        SetBuffer(visibilityResolve, SHADER_TYPE_COMPUTE, "g_SoftVisibility", m_SoftVisBuffer,
                  BUFFER_VIEW_SHADER_RESOURCE);
        SetTexture(visibilityResolve, SHADER_TYPE_COMPUTE, "g_FinalDepth", m_FinalDepthUAV,
                   SET_SHADER_RESOURCE_FLAG_ALLOW_OVERWRITE);
        SetTexture(visibilityResolve, SHADER_TYPE_COMPUTE, "g_FinalVisibility", m_FinalVisibilityUAV,
                   SET_SHADER_RESOURCE_FLAG_ALLOW_OVERWRITE);

        IShaderResourceBinding* visibilityShading = pipelines.VisibilityShadingBinding();
        SetConstantBuffer(visibilityShading, SHADER_TYPE_PIXEL, "NaniteRasterCB", m_RasterConstants);
        SetBuffer(visibilityShading, SHADER_TYPE_PIXEL, "g_PositionBuffer", m_PositionBuffer,
                  BUFFER_VIEW_SHADER_RESOURCE);
        SetBuffer(visibilityShading, SHADER_TYPE_PIXEL, "g_NormalBuffer", m_NormalBuffer,
                  BUFFER_VIEW_SHADER_RESOURCE);
        SetBuffer(visibilityShading, SHADER_TYPE_PIXEL, "g_Instances", m_InstanceBuffer,
                  BUFFER_VIEW_SHADER_RESOURCE);
        SetBuffer(visibilityShading, SHADER_TYPE_PIXEL, "g_IndexBuffer", m_IndexBuffer,
                  BUFFER_VIEW_SHADER_RESOURCE);
        SetBuffer(visibilityShading, SHADER_TYPE_PIXEL, "g_DrawInstanceData", m_DrawInstanceIndices,
                  BUFFER_VIEW_SHADER_RESOURCE);
        SetBuffer(visibilityShading, SHADER_TYPE_PIXEL, "g_UvBuffer", m_UvBuffer,
                  BUFFER_VIEW_SHADER_RESOURCE);
        SetBuffer(visibilityShading, SHADER_TYPE_PIXEL, "g_MaterialIndexBuffer",
                  m_MaterialIndexBuffer, BUFFER_VIEW_SHADER_RESOURCE);
        SetBuffer(visibilityShading, SHADER_TYPE_PIXEL, "g_Materials", m_MaterialBuffer,
                  BUFFER_VIEW_SHADER_RESOURCE);
        SetTexture(visibilityShading, SHADER_TYPE_PIXEL, "g_AlbedoTexture",
                   m_AlbedoTexture->GetDefaultView(TEXTURE_VIEW_SHADER_RESOURCE),
                   SET_SHADER_RESOURCE_FLAG_ALLOW_OVERWRITE);
        SetTexture(visibilityShading, SHADER_TYPE_PIXEL, "g_NormalTexture",
                   m_NormalTexture->GetDefaultView(TEXTURE_VIEW_SHADER_RESOURCE),
                   SET_SHADER_RESOURCE_FLAG_ALLOW_OVERWRITE);
        SetTexture(visibilityShading, SHADER_TYPE_PIXEL, "g_MetallicRoughnessTexture",
                   m_MetallicRoughnessTexture->GetDefaultView(TEXTURE_VIEW_SHADER_RESOURCE),
                   SET_SHADER_RESOURCE_FLAG_ALLOW_OVERWRITE);
        SetTexture(visibilityShading, SHADER_TYPE_PIXEL, "g_ShadowMap", m_ShadowMapSRV,
                   SET_SHADER_RESOURCE_FLAG_ALLOW_OVERWRITE);
        SetBuffer(visibilityShading, SHADER_TYPE_PIXEL, "g_VsmPageTable",
                  m_VsmPageTableBuffer, BUFFER_VIEW_SHADER_RESOURCE);
        SetTexture(visibilityShading, SHADER_TYPE_PIXEL, "g_VsmPhysicalPool",
                   m_VsmPhysicalPoolSRV, SET_SHADER_RESOURCE_FLAG_ALLOW_OVERWRITE);
        SetSampler(visibilityShading, SHADER_TYPE_PIXEL, "g_AlbedoSampler", m_AlbedoSampler);
        SetSampler(visibilityShading, SHADER_TYPE_PIXEL, "g_ShadowSampler", m_ShadowSampler);
        SetSampler(visibilityShading, SHADER_TYPE_PIXEL, "g_VsmShadowSampler", m_VsmSampler);
        SetBuffer(visibilityShading, SHADER_TYPE_PIXEL, "g_QueueState", m_QueueState,
                  BUFFER_VIEW_SHADER_RESOURCE);
        SetTexture(visibilityShading, SHADER_TYPE_PIXEL, "g_FinalVisibility", m_FinalVisibilitySRV,
                   SET_SHADER_RESOURCE_FLAG_ALLOW_OVERWRITE);
        SetTexture(visibilityShading, SHADER_TYPE_PIXEL, "g_FinalDepth", m_FinalDepthSRV,
                   SET_SHADER_RESOURCE_FLAG_ALLOW_OVERWRITE);
    }
    else
    {
        // The final-depth GUI mode also works in the pure hardware path. Its
        // pixel shader does not consume the visibility target, but reflection
        // still requires every declared resource to be bound.
        IShaderResourceBinding* visibilityShading = pipelines.VisibilityShadingBinding();
        SetConstantBuffer(visibilityShading, SHADER_TYPE_PIXEL, "NaniteRasterCB", m_RasterConstants);
        SetBuffer(visibilityShading, SHADER_TYPE_PIXEL, "g_PositionBuffer", m_PositionBuffer,
                  BUFFER_VIEW_SHADER_RESOURCE);
        SetBuffer(visibilityShading, SHADER_TYPE_PIXEL, "g_NormalBuffer", m_NormalBuffer,
                  BUFFER_VIEW_SHADER_RESOURCE);
        SetBuffer(visibilityShading, SHADER_TYPE_PIXEL, "g_Instances", m_InstanceBuffer,
                  BUFFER_VIEW_SHADER_RESOURCE);
        SetBuffer(visibilityShading, SHADER_TYPE_PIXEL, "g_IndexBuffer", m_IndexBuffer,
                  BUFFER_VIEW_SHADER_RESOURCE);
        SetBuffer(visibilityShading, SHADER_TYPE_PIXEL, "g_DrawInstanceData", m_DrawInstanceIndices,
                  BUFFER_VIEW_SHADER_RESOURCE);
        SetBuffer(visibilityShading, SHADER_TYPE_PIXEL, "g_UvBuffer", m_UvBuffer,
                  BUFFER_VIEW_SHADER_RESOURCE);
        SetBuffer(visibilityShading, SHADER_TYPE_PIXEL, "g_MaterialIndexBuffer",
                  m_MaterialIndexBuffer, BUFFER_VIEW_SHADER_RESOURCE);
        SetBuffer(visibilityShading, SHADER_TYPE_PIXEL, "g_Materials", m_MaterialBuffer,
                  BUFFER_VIEW_SHADER_RESOURCE);
        SetTexture(visibilityShading, SHADER_TYPE_PIXEL, "g_AlbedoTexture",
                   m_AlbedoTexture->GetDefaultView(TEXTURE_VIEW_SHADER_RESOURCE),
                   SET_SHADER_RESOURCE_FLAG_ALLOW_OVERWRITE);
        SetTexture(visibilityShading, SHADER_TYPE_PIXEL, "g_NormalTexture",
                   m_NormalTexture->GetDefaultView(TEXTURE_VIEW_SHADER_RESOURCE),
                   SET_SHADER_RESOURCE_FLAG_ALLOW_OVERWRITE);
        SetTexture(visibilityShading, SHADER_TYPE_PIXEL, "g_MetallicRoughnessTexture",
                   m_MetallicRoughnessTexture->GetDefaultView(TEXTURE_VIEW_SHADER_RESOURCE),
                   SET_SHADER_RESOURCE_FLAG_ALLOW_OVERWRITE);
        SetTexture(visibilityShading, SHADER_TYPE_PIXEL, "g_ShadowMap", m_ShadowMapSRV,
                   SET_SHADER_RESOURCE_FLAG_ALLOW_OVERWRITE);
        SetBuffer(visibilityShading, SHADER_TYPE_PIXEL, "g_VsmPageTable",
                  m_VsmPageTableBuffer, BUFFER_VIEW_SHADER_RESOURCE);
        SetTexture(visibilityShading, SHADER_TYPE_PIXEL, "g_VsmPhysicalPool",
                   m_VsmPhysicalPoolSRV, SET_SHADER_RESOURCE_FLAG_ALLOW_OVERWRITE);
        SetSampler(visibilityShading, SHADER_TYPE_PIXEL, "g_AlbedoSampler", m_AlbedoSampler);
        SetSampler(visibilityShading, SHADER_TYPE_PIXEL, "g_ShadowSampler", m_ShadowSampler);
        SetSampler(visibilityShading, SHADER_TYPE_PIXEL, "g_VsmShadowSampler", m_VsmSampler);
        SetBuffer(visibilityShading, SHADER_TYPE_PIXEL, "g_QueueState", m_QueueState,
                  BUFFER_VIEW_SHADER_RESOURCE);
        SetTexture(visibilityShading, SHADER_TYPE_PIXEL, "g_FinalVisibility", m_HardwareVisibilitySRV,
                   SET_SHADER_RESOURCE_FLAG_ALLOW_OVERWRITE);
        SetTexture(visibilityShading, SHADER_TYPE_PIXEL, "g_FinalDepth", m_DepthColorSRV,
                   SET_SHADER_RESOURCE_FLAG_ALLOW_OVERWRITE);
    }

    if (m_HzbMipUAV.size() >= 4 && m_HzbMipUAV[0] != nullptr)
    {
        // Every one of these is rebound per batch in BuildHZB, so the initial
        // bind must already use ALLOW_OVERWRITE semantics for the rebind to be
        // accepted later on.
        IShaderResourceBinding* hizBuild = pipelines.HizBuildBinding();
        SetConstantBuffer(hizBuild, SHADER_TYPE_COMPUTE, "NaniteHizConstants", m_HizConstants);
        SetTexture(hizBuild, SHADER_TYPE_COMPUTE, "g_SourceDepth", m_HzbSourceSRV,
                   SET_SHADER_RESOURCE_FLAG_ALLOW_OVERWRITE);
        SetTexture(hizBuild, SHADER_TYPE_COMPUTE, "g_OutMip1", m_HzbMipUAV[1],
                   SET_SHADER_RESOURCE_FLAG_ALLOW_OVERWRITE);
        SetTexture(hizBuild, SHADER_TYPE_COMPUTE, "g_OutMip2", m_HzbMipUAV[2],
                   SET_SHADER_RESOURCE_FLAG_ALLOW_OVERWRITE);
        SetTexture(hizBuild, SHADER_TYPE_COMPUTE, "g_OutMip3", m_HzbMipUAV[3],
                   SET_SHADER_RESOURCE_FLAG_ALLOW_OVERWRITE);
    }
}

void GpuScene::EnsureRenderTargets(std::uint32_t viewportWidth,
                                   std::uint32_t viewportHeight)
{
    if (m_DepthTexture != nullptr &&
        m_DepthTexture->GetDesc().Width == viewportWidth &&
        m_DepthTexture->GetDesc().Height == viewportHeight)
        return;

    Diligent::TextureDesc depthDesc;
    depthDesc.Name = "Nanite depth buffer";
    depthDesc.Type = Diligent::RESOURCE_DIM_TEX_2D;
    depthDesc.Width = viewportWidth;
    depthDesc.Height = viewportHeight;
    // Keep the image typeless so Vulkan can expose the depth attachment as a
    // D32 view and the compute pass as an R32_FLOAT sampled view.
    depthDesc.Format = Diligent::TEX_FORMAT_R32_TYPELESS;
    depthDesc.MipLevels = 1;
    depthDesc.SampleCount = 1;
    depthDesc.Usage = Diligent::USAGE_DEFAULT;
    depthDesc.BindFlags = Diligent::BIND_DEPTH_STENCIL | Diligent::BIND_SHADER_RESOURCE;
    depthDesc.ClearValue.Format = Diligent::TEX_FORMAT_D32_FLOAT;
    m_Device->CreateTexture(depthDesc, nullptr, &m_DepthTexture);
    Require(m_DepthTexture != nullptr, "Failed to create Nanite depth buffer");
    m_DepthDSV = m_DepthTexture->GetDefaultView(Diligent::TEXTURE_VIEW_DEPTH_STENCIL);
    m_DepthSRV = m_DepthTexture->GetDefaultView(Diligent::TEXTURE_VIEW_SHADER_RESOURCE);
    Require(m_DepthDSV != nullptr && m_DepthSRV != nullptr,
            "Nanite depth buffer views are unavailable");

    Diligent::TextureDesc depthColorDesc;
    depthColorDesc.Name = "Nanite depth source color buffer";
    depthColorDesc.Type = Diligent::RESOURCE_DIM_TEX_2D;
    depthColorDesc.Width = viewportWidth;
    depthColorDesc.Height = viewportHeight;
    depthColorDesc.Format = Diligent::TEX_FORMAT_R32_FLOAT;
    depthColorDesc.MipLevels = 1;
    depthColorDesc.SampleCount = 1;
    depthColorDesc.Usage = Diligent::USAGE_DEFAULT;
    depthColorDesc.BindFlags = Diligent::BIND_RENDER_TARGET | Diligent::BIND_SHADER_RESOURCE;
    m_Device->CreateTexture(depthColorDesc, nullptr, &m_DepthColorTexture);
    Require(m_DepthColorTexture != nullptr, "Failed to create Nanite depth source buffer");
    m_DepthColorRTV = m_DepthColorTexture->GetDefaultView(Diligent::TEXTURE_VIEW_RENDER_TARGET);
    m_DepthColorSRV = m_DepthColorTexture->GetDefaultView(Diligent::TEXTURE_VIEW_SHADER_RESOURCE);
    Require(m_DepthColorRTV != nullptr && m_DepthColorSRV != nullptr,
            "Nanite depth source buffer views are unavailable");

    Diligent::TextureDesc hardwareVisibilityDesc = depthColorDesc;
    hardwareVisibilityDesc.Name = "Nanite hardware visibility";
    hardwareVisibilityDesc.Format = Diligent::TEX_FORMAT_R32_UINT;
    m_Device->CreateTexture(hardwareVisibilityDesc, nullptr, &m_HardwareVisibilityTexture);
    Require(m_HardwareVisibilityTexture != nullptr, "Failed to create Nanite hardware visibility");
    m_HardwareVisibilityRTV =
        m_HardwareVisibilityTexture->GetDefaultView(Diligent::TEXTURE_VIEW_RENDER_TARGET);
    m_HardwareVisibilitySRV =
        m_HardwareVisibilityTexture->GetDefaultView(Diligent::TEXTURE_VIEW_SHADER_RESOURCE);
    Require(m_HardwareVisibilityRTV != nullptr && m_HardwareVisibilitySRV != nullptr,
            "Nanite hardware visibility views are unavailable");

    if (m_SoftRasterEnabled)
    {
        Diligent::TextureDesc finalDepthDesc = depthColorDesc;
        finalDepthDesc.Name = "Nanite final depth";
        finalDepthDesc.BindFlags = Diligent::BIND_SHADER_RESOURCE | Diligent::BIND_UNORDERED_ACCESS;
        m_Device->CreateTexture(finalDepthDesc, nullptr, &m_FinalDepthTexture);
        Require(m_FinalDepthTexture != nullptr, "Failed to create Nanite final depth");
        m_FinalDepthSRV = m_FinalDepthTexture->GetDefaultView(Diligent::TEXTURE_VIEW_SHADER_RESOURCE);
        m_FinalDepthUAV = m_FinalDepthTexture->GetDefaultView(Diligent::TEXTURE_VIEW_UNORDERED_ACCESS);

        Diligent::TextureDesc finalVisibilityDesc = finalDepthDesc;
        finalVisibilityDesc.Name = "Nanite final visibility";
        finalVisibilityDesc.Format = Diligent::TEX_FORMAT_R32_UINT;
        m_Device->CreateTexture(finalVisibilityDesc, nullptr, &m_FinalVisibilityTexture);
        Require(m_FinalVisibilityTexture != nullptr, "Failed to create Nanite final visibility");
        m_FinalVisibilitySRV =
            m_FinalVisibilityTexture->GetDefaultView(Diligent::TEXTURE_VIEW_SHADER_RESOURCE);
        m_FinalVisibilityUAV =
            m_FinalVisibilityTexture->GetDefaultView(Diligent::TEXTURE_VIEW_UNORDERED_ACCESS);
        Require(m_FinalDepthSRV != nullptr && m_FinalDepthUAV != nullptr &&
                    m_FinalVisibilitySRV != nullptr && m_FinalVisibilityUAV != nullptr,
                "Nanite final visibility views are unavailable");
    }

    if (m_SoftRasterEnabled)
    {
        // One u32 per pixel for the software rasterizer's InterlockedMin. Cleared
        // to far exactly once, here: the check pass restores each pixel it reads,
        // which removes a per-frame clear pass from the timing being measured.
        const std::vector<std::uint32_t> farDepth(
            static_cast<std::size_t>(viewportWidth) * viewportHeight, 0x3F800000u);
        Diligent::BufferDesc softDesc;
        softDesc.Name = "Nanite software raster depth";
        softDesc.Size = static_cast<Diligent::Uint64>(farDepth.size() * sizeof(std::uint32_t));
        softDesc.BindFlags = Diligent::BIND_UNORDERED_ACCESS;
        softDesc.ImmediateContextMask = m_ImmediateContextMask;
        softDesc.Usage = Diligent::USAGE_DEFAULT;
        softDesc.Mode = Diligent::BUFFER_MODE_STRUCTURED;
        softDesc.ElementByteStride = sizeof(std::uint32_t);
        const Diligent::BufferData softData{farDepth.data(), softDesc.Size};
        m_SoftDepthBuffer.Release();
        m_Device->CreateBuffer(softDesc, &softData, &m_SoftDepthBuffer);
        Require(m_SoftDepthBuffer != nullptr, "Failed to create Nanite software raster depth");

        // Two words per pixel for the visibility buffer, and cleared to zero rather
        // than to far: the depth key is complemented, so an all-zero word decodes
        // to a depth no real pixel can produce and needs no separate "empty" flag.
        // The check pass restores it the same way it restores the depth buffer.
        const std::vector<std::uint32_t> emptyVis(
            static_cast<std::size_t>(viewportWidth) * viewportHeight * 2u, 0u);
        Diligent::BufferDesc visDesc;
        visDesc.Name = "Nanite software raster visibility";
        visDesc.Size = static_cast<Diligent::Uint64>(emptyVis.size() * sizeof(std::uint32_t));
        visDesc.BindFlags = Diligent::BIND_UNORDERED_ACCESS | Diligent::BIND_SHADER_RESOURCE;
        visDesc.ImmediateContextMask = m_ImmediateContextMask;
        visDesc.Usage = Diligent::USAGE_DEFAULT;
        visDesc.Mode = Diligent::BUFFER_MODE_STRUCTURED;
        // 8 bytes per element: payload followed by its depth key.
        visDesc.ElementByteStride = 2u * sizeof(std::uint32_t);
        const Diligent::BufferData visData{emptyVis.data(), visDesc.Size};
        m_SoftVisBuffer.Release();
        m_Device->CreateBuffer(visDesc, &visData, &m_SoftVisBuffer);
        Require(m_SoftVisBuffer != nullptr, "Failed to create Nanite software raster visibility");

        // A scratch depth target for the hardware depth-only reference draw. It
        // gets its own image rather than reusing m_DepthTexture because that one
        // already holds this frame's depth: redrawing into it would let early-Z
        // reject everything and make the reference look free.
        Diligent::TextureDesc probeDesc;
        probeDesc.Name = "Nanite depth only probe target";
        probeDesc.Type = Diligent::RESOURCE_DIM_TEX_2D;
        probeDesc.Width = viewportWidth;
        probeDesc.Height = viewportHeight;
        probeDesc.Format = Diligent::TEX_FORMAT_D32_FLOAT;
        probeDesc.MipLevels = 1;
        probeDesc.SampleCount = 1;
        probeDesc.Usage = Diligent::USAGE_DEFAULT;
        probeDesc.BindFlags = Diligent::BIND_DEPTH_STENCIL;
        probeDesc.ClearValue.Format = Diligent::TEX_FORMAT_D32_FLOAT;
        probeDesc.ClearValue.DepthStencil.Depth = 1.0f;
        m_SoftProbeDepthTexture.Release();
        m_Device->CreateTexture(probeDesc, nullptr, &m_SoftProbeDepthTexture);
        Require(m_SoftProbeDepthTexture != nullptr, "Failed to create Nanite depth only probe target");
        m_SoftProbeDepthDSV =
            m_SoftProbeDepthTexture->GetDefaultView(Diligent::TEXTURE_VIEW_DEPTH_STENCIL);
        Require(m_SoftProbeDepthDSV != nullptr, "Nanite depth only probe view is unavailable");
    }
    if (std::getenv("NANITE_DUMP_DEPTH_SOURCE") != nullptr ||
        std::getenv("NANITE_VERIFY_OCCLUSION") != nullptr)
    {
        Diligent::TextureDesc readbackDesc = depthColorDesc;
        readbackDesc.Name = "Nanite depth source readback";
        readbackDesc.Usage = Diligent::USAGE_STAGING;
        readbackDesc.BindFlags = Diligent::BIND_NONE;
        readbackDesc.CPUAccessFlags = Diligent::CPU_ACCESS_READ;
        m_Device->CreateTexture(readbackDesc, nullptr, &m_DepthColorReadback);
        Require(m_DepthColorReadback != nullptr, "Failed to create Nanite depth source readback");
    }
    Diligent::TextureDesc hzbSourceDesc = depthColorDesc;
    hzbSourceDesc.Name = "Nanite HZB copied depth source";
    hzbSourceDesc.BindFlags = Diligent::BIND_SHADER_RESOURCE;
    m_Device->CreateTexture(hzbSourceDesc, nullptr, &m_HzbSourceTexture);
    Require(m_HzbSourceTexture != nullptr, "Failed to create Nanite HZB copied source");
    m_HzbSourceSRV = m_HzbSourceTexture->GetDefaultView(Diligent::TEXTURE_VIEW_SHADER_RESOURCE);
    Require(m_HzbSourceSRV != nullptr, "Nanite HZB copied source SRV is unavailable");

    // Keep the complete mip chain. A four-level chain only reaches an 8x8
    // footprint, which is not enough for instance and hierarchy bounds that
    // can cover hundreds of screen pixels.
    //
    // Round both extents up to a power of two, then halve them. With odd extents
    // every halving drops the last row or column, so a coarse texel stops
    // covering the screen area the culling code assumes it covers, and the chain
    // runs out of levels before one texel spans the viewport. Mip 0 holds the
    // 2x2 max reduction of depth rather than a full-resolution copy, which is
    // what UE does: it removes the largest level of the pyramid entirely and
    // costs no accuracy, because a cluster small enough to need exact depth is
    // tested against mip 0 anyway. The visible image occupies the top-left
    // sub-rectangle; culling scales its UVs by viewport / (2 * HZB extent).
    const auto roundUpPow2 = [](std::uint32_t value) {
        std::uint32_t result = 1u;
        while (result < value)
            result <<= 1u;
        return result;
    };
    // Logical extent of the depth level that feeds mip 0. The whole chain is
    // derived from it, so the padding rows and columns halve exactly.
    const std::uint32_t depthPotWidth = roundUpPow2(std::max(viewportWidth, 1u));
    const std::uint32_t depthPotHeight = roundUpPow2(std::max(viewportHeight, 1u));
    m_HzbWidth = std::max(depthPotWidth >> 1u, 1u);
    m_HzbHeight = std::max(depthPotHeight >> 1u, 1u);
    m_HzbMipCount = 1;
    for (std::uint32_t width = m_HzbWidth, height = m_HzbHeight;
         width > 1u || height > 1u;
         ++m_HzbMipCount)
    {
        width = std::max(width >> 1u, 1u);
        height = std::max(height >> 1u, 1u);
    }
    Diligent::TextureDesc hzbDesc;
    hzbDesc.Name = "Nanite hierarchical Z buffer";
    hzbDesc.Type = Diligent::RESOURCE_DIM_TEX_2D;
    hzbDesc.Width = m_HzbWidth;
    hzbDesc.Height = m_HzbHeight;
    hzbDesc.Format = Diligent::TEX_FORMAT_R32_FLOAT;
    hzbDesc.MipLevels = m_HzbMipCount;
    hzbDesc.SampleCount = 1;
    hzbDesc.Usage = Diligent::USAGE_DEFAULT;
    hzbDesc.BindFlags = Diligent::BIND_SHADER_RESOURCE | Diligent::BIND_UNORDERED_ACCESS;
    m_Device->CreateTexture(hzbDesc, nullptr, &m_HzbTexture);
    Require(m_HzbTexture != nullptr, "Failed to create Nanite HZB texture");
    m_HzbSRV = m_HzbTexture->GetDefaultView(Diligent::TEXTURE_VIEW_SHADER_RESOURCE);
    Require(m_HzbSRV != nullptr, "Nanite HZB SRV is unavailable");

    m_HzbMipSRV.clear();
    m_HzbMipSRV.resize(m_HzbMipCount);
    m_HzbMipUAV.clear();
    m_HzbMipUAV.resize(m_HzbMipCount);
    for (std::uint32_t mip = 0; mip < m_HzbMipCount; ++mip)
    {
        Diligent::TextureViewDesc srvDesc;
        srvDesc.ViewType = Diligent::TEXTURE_VIEW_SHADER_RESOURCE;
        srvDesc.TextureDim = Diligent::RESOURCE_DIM_TEX_2D;
        srvDesc.Format = Diligent::TEX_FORMAT_R32_FLOAT;
        srvDesc.MostDetailedMip = mip;
        srvDesc.NumMipLevels = 1;
        const std::string srvName = "Nanite HZB mip " + std::to_string(mip) + " SRV";
        srvDesc.Name = srvName.c_str();
        m_HzbTexture->CreateView(srvDesc, &m_HzbMipSRV[mip]);
        Require(m_HzbMipSRV[mip] != nullptr, "Nanite HZB mip SRV is unavailable");

        Diligent::TextureViewDesc viewDesc;
        viewDesc.ViewType = Diligent::TEXTURE_VIEW_UNORDERED_ACCESS;
        viewDesc.TextureDim = Diligent::RESOURCE_DIM_TEX_2D;
        viewDesc.Format = Diligent::TEX_FORMAT_R32_FLOAT;
        viewDesc.MostDetailedMip = mip;
        viewDesc.NumMipLevels = 1;
        const std::string mipName = "Nanite HZB mip " + std::to_string(mip) + " UAV";
        viewDesc.Name = mipName.c_str();
        m_HzbTexture->CreateView(viewDesc, &m_HzbMipUAV[mip]);
        Require(m_HzbMipUAV[mip] != nullptr, "Nanite HZB UAV is unavailable");
    }

    // One dispatch reduces its source level into the next three levels. The
    // first batch reads the copied depth texture and produces mips 0..2; every
    // later batch reads the level the previous batch produced. Diligent tracks
    // resource state per texture rather than per subresource, so the HZB cannot
    // be bound as an SRV and a UAV in the same dispatch. Give each continuation
    // batch a private copy of its source level instead.
    m_HzbBatchSource.clear();
    m_HzbBatchSourceSRV.clear();
    for (std::uint32_t firstOutputMip = 3u; firstOutputMip < m_HzbMipCount; firstOutputMip += 3u)
    {
        Diligent::TextureDesc batchDesc;
        const std::string batchName =
            "Nanite HZB batch source mip " + std::to_string(firstOutputMip - 1u);
        batchDesc.Name = batchName.c_str();
        batchDesc.Type = Diligent::RESOURCE_DIM_TEX_2D;
        batchDesc.Width = std::max(depthPotWidth >> firstOutputMip, 1u);
        batchDesc.Height = std::max(depthPotHeight >> firstOutputMip, 1u);
        batchDesc.Format = Diligent::TEX_FORMAT_R32_FLOAT;
        batchDesc.MipLevels = 1;
        batchDesc.SampleCount = 1;
        batchDesc.Usage = Diligent::USAGE_DEFAULT;
        batchDesc.BindFlags = Diligent::BIND_SHADER_RESOURCE;

        Diligent::RefCntAutoPtr<Diligent::ITexture> batchTexture;
        m_Device->CreateTexture(batchDesc, nullptr, &batchTexture);
        Require(batchTexture != nullptr, "Failed to create Nanite HZB batch source");
        Diligent::RefCntAutoPtr<Diligent::ITextureView> batchSRV{
            batchTexture->GetDefaultView(Diligent::TEXTURE_VIEW_SHADER_RESOURCE)};
        Require(batchSRV != nullptr, "Nanite HZB batch source SRV is unavailable");
        m_HzbBatchSource.push_back(batchTexture);
        m_HzbBatchSourceSRV.push_back(batchSRV);
    }

    m_HzbValid = false;
    BindResources(*m_Pipelines);
}

void GpuScene::RenderShadowMap(Diligent::IDeviceContext* context)
{
    using namespace Diligent;
    Require(context != nullptr && m_ShadowMapTexture != nullptr &&
                m_ShadowMapDSV != nullptr,
            "Nanite shadow map target is unavailable");
    BeginGpuTimingPass(context, 22u);

    const std::array<float, 16> cameraViewProj = [&]() {
        std::array<float, 16> value{};
        std::memcpy(value.data(), m_RasterData.ViewProjMatrix, sizeof(value));
        return value;
    }();
    const std::uint32_t cameraViewportWidth = m_RasterData.ViewportWidth;
    const std::uint32_t cameraViewportHeight = m_RasterData.ViewportHeight;
    const std::uint32_t cameraDebugMode = m_RasterData.DebugMode;
    const std::uint32_t cameraHybridRasterEnabled = m_RasterData.HybridRasterEnabled;
    const std::uint32_t cameraClusterRasterMode = m_RasterData.ClusterRasterMode;

    // After post recovery, Main contains the complete canonical DrawInstanceData
    // list. The shadow pass consumes that list directly; only the projection and
    // viewport change, so the VS still pulls each cluster's own FirstIndex and
    // IndexCount from the shared entries.
    std::memcpy(m_RasterData.ViewProjMatrix,
                m_RasterData.ShadowViewProjMatrix,
                sizeof(m_RasterData.ViewProjMatrix));
    m_RasterData.ViewportWidth = m_ShadowMapSize;
    m_RasterData.ViewportHeight = m_ShadowMapSize;
    m_RasterData.DebugMode = 0u;
    m_RasterData.HybridRasterEnabled = 0u;
    m_RasterData.ClusterRasterMode = 0u;
    context->UpdateBuffer(m_RasterConstants, 0, sizeof(m_RasterData), &m_RasterData,
                          RESOURCE_STATE_TRANSITION_MODE_TRANSITION);

    context->SetRenderTargets(0, nullptr, m_ShadowMapDSV,
                              RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
    context->ClearDepthStencil(
        m_ShadowMapDSV,
        CLEAR_DEPTH_FLAG,
        1.0f,
        0,
        RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
    context->SetPipelineState(m_Pipelines->HardwareDepthOnly());
    context->CommitShaderResources(
        m_Pipelines->HardwareDepthOnlyBinding(),
        RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
    context->DrawIndirect(DrawIndirectAttribs{
        m_DrawCommands,
        DRAW_FLAG_VERIFY_ALL,
        1,
        static_cast<Uint64>(DrawCommandSlot::Main) * sizeof(DrawCommand),
        sizeof(DrawCommand),
        RESOURCE_STATE_TRANSITION_MODE_TRANSITION});
    context->SetRenderTargets(0, nullptr, nullptr,
                              RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
    // Vulkan exposes a depth SRV in the read-only depth layout. This is also
    // the state expected by Diligent's shader resource cache for a depth view.
    context->TransitionResourceState({
        m_ShadowMapTexture,
        RESOURCE_STATE_UNKNOWN,
        RESOURCE_STATE_DEPTH_READ,
        STATE_TRANSITION_FLAG_UPDATE_STATE});

    std::memcpy(m_RasterData.ViewProjMatrix,
                cameraViewProj.data(),
                sizeof(m_RasterData.ViewProjMatrix));
    m_RasterData.ViewportWidth = cameraViewportWidth;
    m_RasterData.ViewportHeight = cameraViewportHeight;
    m_RasterData.DebugMode = cameraDebugMode;
    m_RasterData.HybridRasterEnabled = cameraHybridRasterEnabled;
    m_RasterData.ClusterRasterMode = cameraClusterRasterMode;
    context->UpdateBuffer(m_RasterConstants, 0, sizeof(m_RasterData), &m_RasterData,
                          RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
    m_ShadowMapValid = true;
    EndGpuTimingPass(context, 22u);
}

void GpuScene::UpdateVsmPageRequests(const float viewProjMatrix[16],
                                     std::uint32_t viewportWidth,
                                     std::uint32_t viewportHeight)
{
    (void)viewportWidth;
    (void)viewportHeight;
    m_VsmDirtyPages.clear();
    if (!m_VsmEnabled || m_VsmPageCount == 0u)
        return;

    float inverseViewProj[16]{};
    if (!InvertMatrix4x4(viewProjMatrix, inverseViewProj))
    {
        for (std::uint32_t page = 0; page < m_VsmPageTable.size(); ++page)
            m_VsmDirtyPages.push_back(page);
        return;
    }

    float minU = std::numeric_limits<float>::max();
    float minV = std::numeric_limits<float>::max();
    float maxU = std::numeric_limits<float>::lowest();
    float maxV = std::numeric_limits<float>::lowest();
    bool hasProjectedCorner = false;
    for (std::uint32_t corner = 0; corner < 8u; ++corner)
    {
        const float ndcX = (corner & 1u) != 0u ? 1.0f : -1.0f;
        const float ndcY = (corner & 2u) != 0u ? 1.0f : -1.0f;
        const float ndcZ = (corner & 4u) != 0u ? 1.0f : 0.0f;
        const auto world = TransformHomogeneous(ndcX, ndcY, ndcZ, 1.0f, inverseViewProj);
        if (std::abs(world[3]) < 1.0e-6f)
            continue;
        const ShadowFloat3 worldPosition{
            world[0] / world[3], world[1] / world[3], world[2] / world[3]};
        const auto lightClip = TransformHomogeneous(
            worldPosition.x,
            worldPosition.y,
            worldPosition.z,
            1.0f,
            m_RasterData.ShadowViewProjMatrix);
        if (std::abs(lightClip[3]) < 1.0e-6f)
            continue;
        const float lightX = lightClip[0] / lightClip[3];
        const float lightY = lightClip[1] / lightClip[3];
        minU = std::min(minU, lightX * 0.5f + 0.5f);
        maxU = std::max(maxU, lightX * 0.5f + 0.5f);
        minV = std::min(minV, 0.5f - lightY * 0.5f);
        maxV = std::max(maxV, 0.5f - lightY * 0.5f);
        hasProjectedCorner = true;
    }

    if (!hasProjectedCorner || maxU < 0.0f || maxV < 0.0f || minU > 1.0f || minV > 1.0f)
        return;

    minU = std::clamp(minU, 0.0f, 1.0f);
    minV = std::clamp(minV, 0.0f, 1.0f);
    maxU = std::clamp(maxU, 0.0f, 1.0f);
    maxV = std::clamp(maxV, 0.0f, 1.0f);
    const std::int32_t minPageX = std::max(
        static_cast<std::int32_t>(std::floor(minU * m_VsmPageCount)) - 1, 0);
    const std::int32_t minPageY = std::max(
        static_cast<std::int32_t>(std::floor(minV * m_VsmPageCount)) - 1, 0);
    const std::int32_t maxPageX = std::min(
        static_cast<std::int32_t>(std::floor(maxU * m_VsmPageCount)),
        static_cast<std::int32_t>(m_VsmPageCount) - 1);
    const std::int32_t maxPageY = std::min(
        static_cast<std::int32_t>(std::floor(maxV * m_VsmPageCount)),
        static_cast<std::int32_t>(m_VsmPageCount) - 1);
    for (std::int32_t pageY = minPageY; pageY <= maxPageY; ++pageY)
        for (std::int32_t pageX = minPageX; pageX <= maxPageX; ++pageX)
            m_VsmDirtyPages.push_back(
                static_cast<std::uint32_t>(pageY) * m_VsmPageCount +
                static_cast<std::uint32_t>(pageX));
}

void GpuScene::RenderVirtualShadowPage(
    Diligent::IDeviceContext* context,
    const std::array<float, 16>& pageViewProj,
    std::uint32_t physicalPage)
{
    using namespace Diligent;
    Require(context != nullptr && physicalPage < m_VsmPhysicalPoolDSV.size(),
            "Nanite VSM page target is unavailable");

    const CullingConstants savedCulling = m_CullingData;
    const RasterConstants savedRaster = m_RasterData;
    ResetQueues(context);
    std::memcpy(m_CullingData.ViewProjMatrix, pageViewProj.data(),
                sizeof(m_CullingData.ViewProjMatrix));
    std::memcpy(m_CullingData.PreviousViewProjMatrix, pageViewProj.data(),
                sizeof(m_CullingData.PreviousViewProjMatrix));
    m_CullingData.CameraWorldPosition[0] = 0.0f;
    m_CullingData.CameraWorldPosition[1] = 0.0f;
    m_CullingData.CameraWorldPosition[2] = 0.0f;
    m_CullingData.ViewportWidth = VsmPageSize;
    m_CullingData.ViewportHeight = VsmPageSize;
    m_CullingData.EnableHZB = 0u;
    m_CullingData.CullingPass = 0u;
    m_CullingData.FocalY = 1.0f;
    m_CullingData.RasterBinAreaCutoff = 0.0f;
    m_CullingData.HybridRasterEnabled = 0u;
    m_CullingData.ClusterRasterMode = 0u;
    m_CullingData.MaxRefinementDepth =
        (savedCulling.MaxRefinementDepth & 0x0FFFFFFFu) |
        0x30000000u | 0x08000000u;
    context->UpdateBuffer(m_CullingConstants, 0, sizeof(m_CullingData), &m_CullingData,
                          RESOURCE_STATE_TRANSITION_MODE_TRANSITION);

    context->SetPipelineState(m_Pipelines->InstanceCull());
    context->CommitShaderResources(
        m_Pipelines->InstanceCullBinding(), RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
    context->DispatchCompute({(m_ActiveInstanceCount + 63u) / 64u, 1, 1});
    context->SetPipelineState(m_Pipelines->PersistentCull());
    context->CommitShaderResources(
        m_Pipelines->PersistentCullBinding(), RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
    context->DispatchCompute({m_Capacity.PersistentWorkgroups, 1, 1});
    context->SetPipelineState(m_Pipelines->PrepareDispatch());
    context->CommitShaderResources(
        m_Pipelines->PrepareDispatchBinding(), RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
    context->DispatchCompute({1, 1, 1});
    context->SetPipelineState(m_Pipelines->GenerateIndirect());
    context->CommitShaderResources(
        m_Pipelines->GenerateIndirectBinding(), RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
    context->DispatchComputeIndirect(DispatchComputeIndirectAttribs{
        m_DispatchArgs,
        RESOURCE_STATE_TRANSITION_MODE_TRANSITION,
        static_cast<Uint64>(DispatchSlot::GenerateIndirect) * sizeof(DispatchCommand)});

    std::memcpy(m_RasterData.ViewProjMatrix, pageViewProj.data(),
                sizeof(m_RasterData.ViewProjMatrix));
    m_RasterData.ViewportWidth = VsmPageSize;
    m_RasterData.ViewportHeight = VsmPageSize;
    m_RasterData.DebugMode = 0u;
    m_RasterData.HybridRasterEnabled = 0u;
    m_RasterData.ClusterRasterMode = 0u;
    context->UpdateBuffer(m_RasterConstants, 0, sizeof(m_RasterData), &m_RasterData,
                          RESOURCE_STATE_TRANSITION_MODE_TRANSITION);

    const Viewport viewport{0.0f, 0.0f,
                            static_cast<float>(VsmPageSize),
                            static_cast<float>(VsmPageSize), 0.0f, 1.0f};
    context->SetViewports(1, &viewport, VsmPageSize, VsmPageSize);
    context->SetRenderTargets(0, nullptr, m_VsmPhysicalPoolDSV[physicalPage],
                              RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
    context->ClearDepthStencil(
        m_VsmPhysicalPoolDSV[physicalPage], CLEAR_DEPTH_FLAG, 1.0f, 0,
        RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
    context->SetPipelineState(m_Pipelines->HardwareDepthOnly());
    context->CommitShaderResources(
        m_Pipelines->HardwareDepthOnlyBinding(), RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
    context->DrawIndirect(DrawIndirectAttribs{
        m_DrawCommands,
        DRAW_FLAG_VERIFY_ALL,
        1,
        static_cast<Uint64>(DrawCommandSlot::Main) * sizeof(DrawCommand),
        sizeof(DrawCommand),
        RESOURCE_STATE_TRANSITION_MODE_TRANSITION});
    context->SetRenderTargets(0, nullptr, nullptr,
                              RESOURCE_STATE_TRANSITION_MODE_TRANSITION);

    m_CullingData = savedCulling;
    m_RasterData = savedRaster;
    context->UpdateBuffer(m_CullingConstants, 0, sizeof(m_CullingData), &m_CullingData,
                          RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
    context->UpdateBuffer(m_RasterConstants, 0, sizeof(m_RasterData), &m_RasterData,
                          RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
}

void GpuScene::RenderVirtualShadowPages(Diligent::IDeviceContext* context,
                                        const float viewProjMatrix[16],
                                        std::uint32_t viewportWidth,
                                        std::uint32_t viewportHeight)
{
    using namespace Diligent;
    UpdateVsmPageRequests(viewProjMatrix, viewportWidth, viewportHeight);
    m_VsmPagesRenderedLastFrame = 0u;
    if (!m_VsmEnabled || m_VsmPageTableBuffer == nullptr)
        return;

    std::vector<std::uint8_t> requested(m_VsmPages.size(), 0u);
    for (const std::uint32_t page : m_VsmDirtyPages)
    {
        if (page >= m_VsmPages.size())
            continue;
        requested[page] = 1u;
        m_VsmPages[page].LastUsedFrame = m_FrameIndex;
    }
    for (const std::uint32_t virtualPage : m_VsmDirtyPages)
    {
        if (virtualPage >= m_VsmPages.size())
            continue;
        VsmPageState& state = m_VsmPages[virtualPage];
        if (state.PhysicalPage >= 0)
            continue;

        std::int32_t physicalPage = -1;
        for (std::uint32_t candidate = 0; candidate < m_VsmPhysicalUsed.size(); ++candidate)
        {
            if (m_VsmPhysicalUsed[candidate] == 0u)
            {
                physicalPage = static_cast<std::int32_t>(candidate);
                break;
            }
        }
        if (physicalPage < 0)
        {
            std::uint32_t oldestVirtualPage = 0u;
            std::uint32_t oldestFrame = std::numeric_limits<std::uint32_t>::max();
            bool foundEviction = false;
            for (std::uint32_t candidate = 0; candidate < m_VsmPages.size(); ++candidate)
            {
                const VsmPageState& candidateState = m_VsmPages[candidate];
                if (candidateState.PhysicalPage < 0 || requested[candidate])
                    continue;
                if (candidateState.LastUsedFrame <= oldestFrame)
                {
                    oldestFrame = candidateState.LastUsedFrame;
                    oldestVirtualPage = candidate;
                    foundEviction = true;
                }
            }
            // If the camera footprint is larger than the physical pool, keep
            // the resident subset stable. Evicting one requested page for
            // another requested page every frame would make the cache thrash
            // forever and never let any page become warm.
            if (!foundEviction)
                continue;
            physicalPage = m_VsmPages[oldestVirtualPage].PhysicalPage;
            if (physicalPage < 0)
                continue;
            m_VsmPageTable[oldestVirtualPage] = 0u;
            m_VsmPages[oldestVirtualPage] = VsmPageState{};
        }
        state.PhysicalPage = physicalPage;
        state.Valid = false;
        m_VsmPhysicalUsed[static_cast<std::size_t>(physicalPage)] = 1u;
        m_VsmPageTable[virtualPage] = 0u;
    }

    const std::uint32_t renderBudget = std::min<std::uint32_t>(
        m_VsmPageBudget, VsmPhysicalPageCount);
    if (!m_VsmDirtyPages.empty())
    {
        context->TransitionResourceState({
            m_VsmPhysicalPoolTexture,
            RESOURCE_STATE_UNKNOWN,
            RESOURCE_STATE_DEPTH_WRITE,
            STATE_TRANSITION_FLAG_UPDATE_STATE});
        BeginGpuTimingPass(context, 22u);
        for (const std::uint32_t virtualPage : m_VsmDirtyPages)
        {
            if (m_VsmPagesRenderedLastFrame >= renderBudget || virtualPage >= m_VsmPages.size())
                break;
            VsmPageState& state = m_VsmPages[virtualPage];
            if (state.PhysicalPage < 0 || state.Valid)
                continue;
            const std::uint32_t pageX = virtualPage % m_VsmPageCount;
            const std::uint32_t pageY = virtualPage / m_VsmPageCount;
            const auto pageViewProj = BuildVirtualPageViewProjection(
                m_RasterData.ShadowViewProjMatrix, m_VsmPageCount, pageX, pageY);
            RenderVirtualShadowPage(
                context, pageViewProj, static_cast<std::uint32_t>(state.PhysicalPage));
            state.Valid = true;
            m_VsmPageTable[virtualPage] = static_cast<std::uint32_t>(state.PhysicalPage) + 1u;
            ++m_VsmPagesRenderedLastFrame;
        }
        EndGpuTimingPass(context, 22u);
        context->TransitionResourceState({
            m_VsmPhysicalPoolTexture,
            RESOURCE_STATE_UNKNOWN,
            RESOURCE_STATE_DEPTH_READ,
            STATE_TRANSITION_FLAG_UPDATE_STATE});
    }

    context->UpdateBuffer(
        m_VsmPageTableBuffer,
        0,
        static_cast<Uint64>(m_VsmPageTable.size() * sizeof(std::uint32_t)),
        m_VsmPageTable.data(),
        RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
}

void GpuScene::InvalidateVirtualShadowPages()
{
    std::fill(m_VsmPageTable.begin(), m_VsmPageTable.end(), 0u);
    std::fill(m_VsmPhysicalUsed.begin(), m_VsmPhysicalUsed.end(), 0u);
    std::fill(m_VsmPages.begin(), m_VsmPages.end(), VsmPageState{});
    m_VsmDirtyPages.clear();
}

void GpuScene::ResolveVisibility(Diligent::IDeviceContext* context,
                                 std::uint32_t viewportWidth,
                                 std::uint32_t viewportHeight)
{
    Require(m_FinalDepthTexture != nullptr && m_FinalVisibilityTexture != nullptr,
            "Nanite final visibility targets are unavailable");
    context->SetPipelineState(m_Pipelines->VisibilityResolve());
    context->CommitShaderResources(
        m_Pipelines->VisibilityResolveBinding(),
        Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
    context->DispatchCompute({(viewportWidth + 7u) / 8u, (viewportHeight + 7u) / 8u, 1u});
    context->TransitionResourceState({
        m_FinalDepthTexture,
        Diligent::RESOURCE_STATE_UNKNOWN,
        Diligent::RESOURCE_STATE_SHADER_RESOURCE,
        Diligent::STATE_TRANSITION_FLAG_UPDATE_STATE});
    context->TransitionResourceState({
        m_FinalVisibilityTexture,
        Diligent::RESOURCE_STATE_UNKNOWN,
        Diligent::RESOURCE_STATE_SHADER_RESOURCE,
        Diligent::STATE_TRANSITION_FLAG_UPDATE_STATE});
}

void GpuScene::ShadeVisibility(Diligent::IDeviceContext* context,
                               Diligent::ITextureView* renderTarget,
                               Diligent::ITextureView* depthSource,
                               Diligent::ITextureView* visibilitySource)
{
    Require(renderTarget != nullptr && depthSource != nullptr && visibilitySource != nullptr,
            "Nanite visibility shading target is unavailable");
    IShaderResourceBinding* shading = m_Pipelines->VisibilityShadingBinding();
    SetTexture(shading, Diligent::SHADER_TYPE_PIXEL, "g_FinalVisibility", visibilitySource,
               Diligent::SET_SHADER_RESOURCE_FLAG_ALLOW_OVERWRITE);
    SetTexture(shading, Diligent::SHADER_TYPE_PIXEL, "g_FinalDepth", depthSource,
               Diligent::SET_SHADER_RESOURCE_FLAG_ALLOW_OVERWRITE);
    context->SetRenderTargets(1, &renderTarget, nullptr,
                              Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
    context->SetPipelineState(m_Pipelines->VisibilityShading());
    context->CommitShaderResources(shading, Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
    context->Draw(Diligent::DrawAttribs{3, Diligent::DRAW_FLAG_VERIFY_ALL});
}

void GpuScene::BuildHZB(Diligent::IDeviceContext* context,
                        std::uint32_t viewportWidth,
                        std::uint32_t viewportHeight,
                        Diligent::ITexture* depthSource,
                        bool recordTiming)
{
    Require(depthSource != nullptr, "Nanite HZB source is unavailable");
    context->TransitionResourceState({
        depthSource,
        Diligent::RESOURCE_STATE_UNKNOWN,
        Diligent::RESOURCE_STATE_COPY_SOURCE,
        Diligent::STATE_TRANSITION_FLAG_UPDATE_STATE});
    context->TransitionResourceState({
        m_HzbSourceTexture,
        Diligent::RESOURCE_STATE_UNKNOWN,
        Diligent::RESOURCE_STATE_COPY_DEST,
        Diligent::STATE_TRANSITION_FLAG_UPDATE_STATE});
    context->CopyTexture({
        depthSource,
        Diligent::RESOURCE_STATE_TRANSITION_MODE_VERIFY,
        m_HzbSourceTexture,
        Diligent::RESOURCE_STATE_TRANSITION_MODE_VERIFY});
    context->TransitionResourceState({
        m_HzbSourceTexture,
        Diligent::RESOURCE_STATE_COPY_DEST,
        Diligent::RESOURCE_STATE_SHADER_RESOURCE,
        Diligent::STATE_TRANSITION_FLAG_UPDATE_STATE});
    const bool renderDocLabels = std::getenv("NANITE_RENDERDOC_LABELS") != nullptr;
    if (renderDocLabels)
        context->BeginDebugGroup("Nanite / HZB Build");
    context->TransitionResourceState({
        m_HzbTexture,
        Diligent::RESOURCE_STATE_UNKNOWN,
        Diligent::RESOURCE_STATE_UNORDERED_ACCESS,
        Diligent::STATE_TRANSITION_FLAG_UPDATE_STATE});

    // The hybrid path builds the pyramid twice: the first build feeds post cull,
    // while the second follows the final visibility resolve. Keep those GPU
    // intervals separate so the HUD accounts for both without double counting.
    const std::uint32_t timingPass = recordTiming ? 7u : 21u;
    if (m_GpuTimingFrameActive)
        BeginGpuTimingPass(context, timingPass);
    context->SetPipelineState(m_Pipelines->HizBuild());
    // Each dispatch reduces one source level into the next three levels. The
    // first batch reduces the depth copy into mips 0..2, so mip 0 is already the
    // 2x2 maximum of depth and no full-resolution level is ever stored. Levels
    // are numbered in the depth-rooted chain: chain level L feeds HZB mip L.
    const std::uint32_t depthPotWidth = m_HzbWidth * 2u;
    const std::uint32_t depthPotHeight = m_HzbHeight * 2u;
    for (std::uint32_t firstOutputMip = 0u; firstOutputMip < m_HzbMipCount; firstOutputMip += 3u)
    {
        const std::uint32_t outputMipCount =
            std::min<std::uint32_t>(3u, m_HzbMipCount - firstOutputMip);
        // Logical extent of the source level in the padded power-of-two chain.
        // This drives the output bounds and the dispatch grid so the padding
        // columns and rows are reduced along with the visible area.
        m_HizData.OutputMipCount = outputMipCount;
        m_HizData.SourceWidth = std::max(depthPotWidth >> firstOutputMip, 1u);
        m_HizData.SourceHeight = std::max(depthPotHeight >> firstOutputMip, 1u);
        m_HizData.Padding0 = 0u;
        m_HizData.Padding1 = 0u;
        m_HizData.Padding2 = 0u;

        IShaderResourceBinding* hizBuild = m_Pipelines->HizBuildBinding();
        if (firstOutputMip == 0u)
        {
            // The first batch reads the viewport-sized depth copy directly, so
            // the bound texture is smaller than the logical extent. The shader
            // treats texels outside it as far.
            m_HizData.SourceTexWidth = std::max(viewportWidth, 1u);
            m_HizData.SourceTexHeight = std::max(viewportHeight, 1u);
            SetTexture(hizBuild, Diligent::SHADER_TYPE_COMPUTE, "g_SourceDepth",
                       m_HzbSourceSRV, Diligent::SET_SHADER_RESOURCE_FLAG_ALLOW_OVERWRITE);
        }
        else
        {
            // Diligent tracks state per texture, so reading HZB mip N while
            // writing mips N+1..N+3 of the same texture is not expressible.
            // Copy the source level out first and read the copy.
            const std::uint32_t sourceMip = firstOutputMip - 1u;
            const std::size_t batchIndex = firstOutputMip / 3u - 1u;
            Require(batchIndex < m_HzbBatchSource.size(), "Nanite HZB batch source is missing");
            m_HizData.SourceTexWidth = m_HizData.SourceWidth;
            m_HizData.SourceTexHeight = m_HizData.SourceHeight;

            context->TransitionResourceState({
                m_HzbTexture,
                Diligent::RESOURCE_STATE_UNKNOWN,
                Diligent::RESOURCE_STATE_COPY_SOURCE,
                Diligent::STATE_TRANSITION_FLAG_UPDATE_STATE});
            context->TransitionResourceState({
                m_HzbBatchSource[batchIndex],
                Diligent::RESOURCE_STATE_UNKNOWN,
                Diligent::RESOURCE_STATE_COPY_DEST,
                Diligent::STATE_TRANSITION_FLAG_UPDATE_STATE});
            Diligent::CopyTextureAttribs batchCopy{
                m_HzbTexture,
                Diligent::RESOURCE_STATE_TRANSITION_MODE_VERIFY,
                m_HzbBatchSource[batchIndex],
                Diligent::RESOURCE_STATE_TRANSITION_MODE_VERIFY};
            batchCopy.SrcMipLevel = sourceMip;
            batchCopy.DstMipLevel = 0;
            context->CopyTexture(batchCopy);
            context->TransitionResourceState({
                m_HzbBatchSource[batchIndex],
                Diligent::RESOURCE_STATE_COPY_DEST,
                Diligent::RESOURCE_STATE_SHADER_RESOURCE,
                Diligent::STATE_TRANSITION_FLAG_UPDATE_STATE});
            context->TransitionResourceState({
                m_HzbTexture,
                Diligent::RESOURCE_STATE_COPY_SOURCE,
                Diligent::RESOURCE_STATE_UNORDERED_ACCESS,
                Diligent::STATE_TRANSITION_FLAG_UPDATE_STATE});

            SetTexture(hizBuild, Diligent::SHADER_TYPE_COMPUTE, "g_SourceDepth",
                       m_HzbBatchSourceSRV[batchIndex],
                       Diligent::SET_SHADER_RESOURCE_FLAG_ALLOW_OVERWRITE);
        }
        context->UpdateBuffer(m_HizConstants, 0, sizeof(m_HizData), &m_HizData,
                              Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION);

        // Rebinding a non-dynamic variable is silently ignored without
        // ALLOW_OVERWRITE, which used to leave every batch writing the mips of
        // the first batch and left mips 4 and coarser uninitialized.
        for (std::uint32_t output = 0; output < 3u; ++output)
        {
            // Levels past the end of the chain are clamped to the last valid
            // view; the shader will not write them because OutputMipCount stops
            // the reduction earlier.
            const std::uint32_t mip =
                std::min(firstOutputMip + output, m_HzbMipCount - 1u);
            static const char* const kNames[3] = {"g_OutMip1", "g_OutMip2", "g_OutMip3"};
            SetTexture(hizBuild, Diligent::SHADER_TYPE_COMPUTE, kNames[output],
                       m_HzbMipUAV[mip], Diligent::SET_SHADER_RESOURCE_FLAG_ALLOW_OVERWRITE);
        }
        context->CommitShaderResources(
            hizBuild,
            Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
        context->DispatchCompute({
            (m_HizData.SourceWidth + 7u) / 8u,
            (m_HizData.SourceHeight + 7u) / 8u,
            1});
    }
    if (m_GpuTimingFrameActive)
        EndGpuTimingPass(context, timingPass);
    if (renderDocLabels)
        context->EndDebugGroup();
    // Make the freshly written HZB visible to the culling shaders on the next
    // frame. MoltenVK can otherwise keep the previous SRV contents visible.
    context->TransitionResourceState({
        m_HzbTexture,
        Diligent::RESOURCE_STATE_UNKNOWN,
        Diligent::RESOURCE_STATE_SHADER_RESOURCE,
        Diligent::STATE_TRANSITION_FLAG_UPDATE_STATE});
    const bool dumpDepth = std::getenv("NANITE_DUMP_DEPTH_SOURCE") != nullptr;
    if (std::getenv("NANITE_DUMP_HZB_MIPS") != nullptr && !m_HzbMipDumpPrinted)
        DumpHzbMips(context);
    const bool verifyDepth = std::getenv("NANITE_VERIFY_OCCLUSION") != nullptr &&
        !m_CpuVerifyPrinted;
    if (m_DepthColorReadback != nullptr && (dumpDepth || verifyDepth))
    {
        context->TransitionResourceState({
            depthSource,
            Diligent::RESOURCE_STATE_UNKNOWN,
            Diligent::RESOURCE_STATE_COPY_SOURCE,
            Diligent::STATE_TRANSITION_FLAG_UPDATE_STATE});
        context->TransitionResourceState({
            m_DepthColorReadback,
            Diligent::RESOURCE_STATE_UNKNOWN,
            Diligent::RESOURCE_STATE_COPY_DEST,
            Diligent::STATE_TRANSITION_FLAG_UPDATE_STATE});
        context->CopyTexture({
            depthSource,
            Diligent::RESOURCE_STATE_TRANSITION_MODE_VERIFY,
            m_DepthColorReadback,
            Diligent::RESOURCE_STATE_TRANSITION_MODE_VERIFY});
        context->Flush();
        context->WaitForIdle();

        Diligent::MappedTextureSubresource mapped;
        context->MapTextureSubresource(
            m_DepthColorReadback,
            0,
            0,
            Diligent::MAP_READ,
            Diligent::MAP_FLAG_DO_NOT_WAIT,
            nullptr,
            mapped);
        if (mapped.pData != nullptr)
        {
            float minDepth = 1.0f;
            float maxDepth = 0.0f;
            for (std::uint32_t y = 0; y < viewportHeight; ++y)
            {
                const auto* row = static_cast<const std::uint8_t*>(mapped.pData) +
                    static_cast<std::size_t>(y) * mapped.Stride;
                for (std::uint32_t x = 0; x < viewportWidth; ++x)
                {
                    float value;
                    std::memcpy(&value, row + static_cast<std::size_t>(x) * sizeof(float), sizeof(value));
                    minDepth = std::min(minDepth, value);
                    maxDepth = std::max(maxDepth, value);
                }
            }
            std::cerr << "Nanite depth source readback: min=" << minDepth
                      << " max=" << maxDepth << " stride=" << mapped.Stride << '\n';
            if (verifyDepth)
            {
                VerifyClusterOcclusion(mapped, viewportWidth, viewportHeight);
            }
            context->UnmapTextureSubresource(m_DepthColorReadback, 0, 0);

            if (verifyDepth)
            {
                context->TransitionResourceState({
                    m_HzbSourceTexture,
                    Diligent::RESOURCE_STATE_SHADER_RESOURCE,
                    Diligent::RESOURCE_STATE_COPY_SOURCE,
                    Diligent::STATE_TRANSITION_FLAG_UPDATE_STATE});
                context->TransitionResourceState({
                    m_DepthColorReadback,
                    Diligent::RESOURCE_STATE_UNKNOWN,
                    Diligent::RESOURCE_STATE_COPY_DEST,
                    Diligent::STATE_TRANSITION_FLAG_UPDATE_STATE});
                Diligent::CopyTextureAttribs sourceCopy{
                    m_HzbSourceTexture,
                    Diligent::RESOURCE_STATE_TRANSITION_MODE_VERIFY,
                    m_DepthColorReadback,
                    Diligent::RESOURCE_STATE_TRANSITION_MODE_VERIFY};
                context->CopyTexture(sourceCopy);
                context->Flush();
                context->WaitForIdle();
                Diligent::MappedTextureSubresource sourceMapped;
                context->MapTextureSubresource(
                    m_DepthColorReadback,
                    0,
                    0,
                    Diligent::MAP_READ,
                    Diligent::MAP_FLAG_DO_NOT_WAIT,
                    nullptr,
                    sourceMapped);
                if (sourceMapped.pData != nullptr)
                {
                    const auto* sourceRow = static_cast<const std::uint8_t*>(sourceMapped.pData) +
                        static_cast<std::size_t>(m_CpuVerifyPixelY) * sourceMapped.Stride;
                    float sourceSample = 1.0f;
                    std::memcpy(&sourceSample,
                                sourceRow + static_cast<std::size_t>(m_CpuVerifyPixelX) * sizeof(float),
                                sizeof(sourceSample));
                    std::cerr << "Nanite HZB source readback: sample["
                              << m_CpuVerifyPixelX << ',' << m_CpuVerifyPixelY
                              << "]=" << sourceSample
                              << " stride=" << sourceMapped.Stride << '\n';
                    context->UnmapTextureSubresource(m_DepthColorReadback, 0, 0);
                }
                context->TransitionResourceState({
                    m_HzbSourceTexture,
                    Diligent::RESOURCE_STATE_COPY_SOURCE,
                    Diligent::RESOURCE_STATE_SHADER_RESOURCE,
                    Diligent::STATE_TRANSITION_FLAG_UPDATE_STATE});

                context->TransitionResourceState({
                    m_HzbTexture,
                    Diligent::RESOURCE_STATE_SHADER_RESOURCE,
                    Diligent::RESOURCE_STATE_COPY_SOURCE,
                    Diligent::STATE_TRANSITION_FLAG_UPDATE_STATE});
                context->TransitionResourceState({
                    m_DepthColorReadback,
                    Diligent::RESOURCE_STATE_UNKNOWN,
                    Diligent::RESOURCE_STATE_COPY_DEST,
                    Diligent::STATE_TRANSITION_FLAG_UPDATE_STATE});
                Diligent::CopyTextureAttribs hzbCopy{
                    m_HzbTexture,
                    Diligent::RESOURCE_STATE_TRANSITION_MODE_VERIFY,
                    m_DepthColorReadback,
                    Diligent::RESOURCE_STATE_TRANSITION_MODE_VERIFY};
                hzbCopy.SrcMipLevel = 0;
                hzbCopy.DstMipLevel = 0;
                // HZB mip 0 is padded to a power of two, so copy only the
                // viewport-sized sub-rectangle that the readback texture holds.
                // Mip 0 is half resolution, so the readback target is larger than
                // the level. Copy only what exists.
                Diligent::Box hzbSrcBox;
                hzbSrcBox.MinX = 0;
                hzbSrcBox.MaxX = std::min(viewportWidth, m_HzbWidth);
                hzbSrcBox.MinY = 0;
                hzbSrcBox.MaxY = std::min(viewportHeight, m_HzbHeight);
                hzbCopy.pSrcBox = &hzbSrcBox;
                context->CopyTexture(hzbCopy);
                context->Flush();
                context->WaitForIdle();

                Diligent::MappedTextureSubresource hzbMapped;
                context->MapTextureSubresource(
                    m_DepthColorReadback,
                    0,
                    0,
                    Diligent::MAP_READ,
                    Diligent::MAP_FLAG_DO_NOT_WAIT,
                    nullptr,
                    hzbMapped);
                if (hzbMapped.pData != nullptr)
                {
                    float hzbMin = 1.0f;
                    float hzbMax = 0.0f;
                    const auto* sampleRow = static_cast<const std::uint8_t*>(hzbMapped.pData) +
                        static_cast<std::size_t>(m_CpuVerifyPixelY) * hzbMapped.Stride;
                    float sample = 1.0f;
                    std::memcpy(&sample,
                                sampleRow + static_cast<std::size_t>(m_CpuVerifyPixelX) * sizeof(float),
                                sizeof(sample));
                    float probeRegionMax = 0.0f;
                    std::cerr << "Nanite HZB mip0 probe region:";
                    for (std::uint32_t y = m_CpuVerifyPixelY;
                         y <= std::min(viewportHeight - 1u, m_CpuVerifyPixelY + 3u);
                         ++y)
                    {
                        std::cerr << " [";
                        const auto* row = static_cast<const std::uint8_t*>(hzbMapped.pData) +
                            static_cast<std::size_t>(y) * hzbMapped.Stride;
                        for (std::uint32_t x = m_CpuVerifyPixelX;
                             x <= std::min(viewportWidth - 1u, m_CpuVerifyPixelX + 1u);
                             ++x)
                        {
                            float value;
                            std::memcpy(&value,
                                        row + static_cast<std::size_t>(x) * sizeof(float),
                                        sizeof(value));
                            probeRegionMax = std::max(probeRegionMax, value);
                            std::cerr << value;
                            if (x < std::min(viewportWidth - 1u, m_CpuVerifyPixelX + 1u))
                                std::cerr << ',';
                        }
                        std::cerr << ']';
                    }
                    std::cerr << " max=" << probeRegionMax << '\n';
                    for (std::uint32_t y = 0; y < viewportHeight; ++y)
                    {
                        const auto* row = static_cast<const std::uint8_t*>(hzbMapped.pData) +
                            static_cast<std::size_t>(y) * hzbMapped.Stride;
                        for (std::uint32_t x = 0; x < viewportWidth; ++x)
                        {
                            float value;
                            std::memcpy(&value,
                                        row + static_cast<std::size_t>(x) * sizeof(float),
                                        sizeof(value));
                            hzbMin = std::min(hzbMin, value);
                            hzbMax = std::max(hzbMax, value);
                        }
                    }
                    std::cerr << "Nanite HZB mip0 readback: min=" << hzbMin
                              << " max=" << hzbMax
                              << " sample[" << m_CpuVerifyPixelX << ',' << m_CpuVerifyPixelY
                              << "]=" << sample
                              << " stride=" << hzbMapped.Stride << '\n';
                    context->UnmapTextureSubresource(m_DepthColorReadback, 0, 0);
                }
                context->TransitionResourceState({
                    m_HzbTexture,
                    Diligent::RESOURCE_STATE_COPY_SOURCE,
                    Diligent::RESOURCE_STATE_SHADER_RESOURCE,
                    Diligent::STATE_TRANSITION_FLAG_UPDATE_STATE});
                m_CpuVerifyPrinted = true;
            }
        }
    }
    if (std::getenv("NANITE_VERIFY_OCCLUSION") != nullptr)
    {
        // The diagnostic readback temporarily moves the HZB through COPY_SOURCE
        // and back to SRV. Submit the final transition before the next frame's
        // culling dispatch so the probe does not observe stale image state.
        context->Flush();
        context->WaitForIdle();
    }
    m_HzbValid = true;
}

void GpuScene::DumpHzbMips(Diligent::IDeviceContext* context)
{
    m_HzbMipDumpPrinted = true;
    for (std::uint32_t mip = 0; mip < m_HzbMipCount; ++mip)
    {
        const std::uint32_t mipWidth = std::max(m_HzbWidth >> mip, 1u);
        const std::uint32_t mipHeight = std::max(m_HzbHeight >> mip, 1u);

        Diligent::TextureDesc stagingDesc;
        stagingDesc.Name = "Nanite HZB mip dump staging";
        stagingDesc.Type = Diligent::RESOURCE_DIM_TEX_2D;
        stagingDesc.Width = mipWidth;
        stagingDesc.Height = mipHeight;
        stagingDesc.Format = Diligent::TEX_FORMAT_R32_FLOAT;
        stagingDesc.MipLevels = 1;
        stagingDesc.Usage = Diligent::USAGE_STAGING;
        stagingDesc.BindFlags = Diligent::BIND_NONE;
        stagingDesc.CPUAccessFlags = Diligent::CPU_ACCESS_READ;
        Diligent::RefCntAutoPtr<Diligent::ITexture> staging;
        m_Device->CreateTexture(stagingDesc, nullptr, &staging);
        if (staging == nullptr)
        {
            std::cerr << "Nanite HZB mip dump: staging allocation failed at mip " << mip << '\n';
            return;
        }

        context->TransitionResourceState({
            m_HzbTexture,
            Diligent::RESOURCE_STATE_UNKNOWN,
            Diligent::RESOURCE_STATE_COPY_SOURCE,
            Diligent::STATE_TRANSITION_FLAG_UPDATE_STATE});
        Diligent::CopyTextureAttribs copy{
            m_HzbTexture,
            Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION,
            staging,
            Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION};
        copy.SrcMipLevel = mip;
        copy.DstMipLevel = 0;
        context->CopyTexture(copy);
        context->Flush();
        context->WaitForIdle();

        Diligent::MappedTextureSubresource mapped;
        context->MapTextureSubresource(
            staging, 0, 0, Diligent::MAP_READ, Diligent::MAP_FLAG_DO_NOT_WAIT, nullptr, mapped);
        if (mapped.pData == nullptr)
        {
            std::cerr << "Nanite HZB mip dump: map failed at mip " << mip << '\n';
            continue;
        }
        float minValue = std::numeric_limits<float>::max();
        float maxValue = -std::numeric_limits<float>::max();
        std::uint32_t zeroCount = 0;
        for (std::uint32_t y = 0; y < mipHeight; ++y)
        {
            const auto* row = static_cast<const std::uint8_t*>(mapped.pData) +
                static_cast<std::size_t>(y) * mapped.Stride;
            for (std::uint32_t x = 0; x < mipWidth; ++x)
            {
                float value = 0.0f;
                std::memcpy(&value, row + static_cast<std::size_t>(x) * sizeof(float), sizeof(value));
                minValue = std::min(minValue, value);
                maxValue = std::max(maxValue, value);
                if (value == 0.0f)
                    ++zeroCount;
            }
        }
        const std::uint32_t texelCount = mipWidth * mipHeight;
        std::cerr << "Nanite HZB mip " << mip << ": " << mipWidth << 'x' << mipHeight
                  << " min=" << minValue << " max=" << maxValue
                  << " zeros=" << zeroCount << '/' << texelCount << '\n';
        context->UnmapTextureSubresource(staging, 0, 0);
    }
    context->TransitionResourceState({
        m_HzbTexture,
        Diligent::RESOURCE_STATE_UNKNOWN,
        Diligent::RESOURCE_STATE_SHADER_RESOURCE,
        Diligent::STATE_TRANSITION_FLAG_UPDATE_STATE});
}

void GpuScene::VerifyClusterOcclusion(
    const Diligent::MappedTextureSubresource& mapped,
    std::uint32_t viewportWidth,
    std::uint32_t viewportHeight)
{
    constexpr std::uint32_t targetInstanceIndex = 10u;
    constexpr float depthEpsilon = 2.0f / 16777216.0f;
    if (targetInstanceIndex >= m_CpuScene.Instances.size())
    {
        std::cerr << "Nanite CPU occlusion verify: instance 10 is unavailable\n";
        return;
    }

    struct ClipPoint
    {
        float x;
        float y;
        float z;
        float w;
    };
    struct Probe
    {
        std::uint32_t clusterIndex = 0;
        ClipPoint corners[8]{};
        float clipMinX = 0.0f;
        float clipMinY = 0.0f;
        float clipMinZ = 0.0f;
        float clipMaxX = 0.0f;
        float clipMaxY = 0.0f;
        float clipMaxZ = 0.0f;
        std::uint32_t pixelMinX = 0;
        std::uint32_t pixelMinY = 0;
        std::uint32_t pixelMaxX = 0;
        std::uint32_t pixelMaxY = 0;
        std::uint64_t pixelArea = 0;
    };

    const Instance& instance = m_CpuScene.Instances[targetInstanceIndex];
    auto transform = [](const float point[4], const float matrix[16]) {
        ClipPoint result{};
        float* output = &result.x;
        for (int column = 0; column < 4; ++column)
        {
            output[column] = 0.0f;
            for (int row = 0; row < 4; ++row)
                output[column] += point[row] * matrix[row * 4 + column];
        }
        return result;
    };

    std::vector<Probe> probes;
    probes.reserve(m_CpuScene.Clusters.size());
    for (std::uint32_t clusterIndex = 0;
         clusterIndex < m_CpuScene.Clusters.size();
         ++clusterIndex)
    {
        const Cluster& cluster = m_CpuScene.Clusters[clusterIndex];
        Probe probe;
        probe.clusterIndex = clusterIndex;
        probe.clipMinX = probe.clipMinY = probe.clipMinZ = std::numeric_limits<float>::max();
        probe.clipMaxX = probe.clipMaxY = probe.clipMaxZ = std::numeric_limits<float>::lowest();
        bool allCornersValid = true;
        for (std::uint32_t corner = 0; corner < 8u; ++corner)
        {
            const float point[4] = {
                (corner & 1u) != 0u ? cluster.BBoxMax[0] : cluster.BBoxMin[0],
                (corner & 2u) != 0u ? cluster.BBoxMax[1] : cluster.BBoxMin[1],
                (corner & 4u) != 0u ? cluster.BBoxMax[2] : cluster.BBoxMin[2],
                1.0f};
            const ClipPoint world = transform(point, instance.WorldMatrix);
            const float worldPoint[4] = {world.x, world.y, world.z, world.w};
            probe.corners[corner] = transform(worldPoint, m_CullingData.ViewProjMatrix);
            const ClipPoint& clip = probe.corners[corner];
            allCornersValid = allCornersValid && clip.w > 1.0e-7f;
            if (clip.w <= 1.0e-7f)
                continue;
            const float invW = 1.0f / clip.w;
            probe.clipMinX = std::min(probe.clipMinX, clip.x * invW);
            probe.clipMinY = std::min(probe.clipMinY, clip.y * invW);
            probe.clipMinZ = std::min(probe.clipMinZ, clip.z * invW);
            probe.clipMaxX = std::max(probe.clipMaxX, clip.x * invW);
            probe.clipMaxY = std::max(probe.clipMaxY, clip.y * invW);
            probe.clipMaxZ = std::max(probe.clipMaxZ, clip.z * invW);
        }
        if (!allCornersValid || probe.clipMaxX < -1.0f || probe.clipMinX > 1.0f ||
            probe.clipMaxY < -1.0f || probe.clipMinY > 1.0f)
            continue;

        const float minU = std::clamp(probe.clipMinX * 0.5f + 0.5f, 0.0f, 1.0f);
        const float maxU = std::clamp(probe.clipMaxX * 0.5f + 0.5f, 0.0f, 1.0f);
        const float minV = std::clamp(0.5f - probe.clipMaxY * 0.5f, 0.0f, 1.0f);
        const float maxV = std::clamp(0.5f - probe.clipMinY * 0.5f, 0.0f, 1.0f);
        probe.pixelMinX = std::min<std::uint32_t>(
            static_cast<std::uint32_t>(minU * viewportWidth), viewportWidth - 1u);
        probe.pixelMinY = std::min<std::uint32_t>(
            static_cast<std::uint32_t>(minV * viewportHeight), viewportHeight - 1u);
        probe.pixelMaxX = std::min<std::uint32_t>(
            static_cast<std::uint32_t>(std::ceil(maxU * viewportWidth)) - 1u,
            viewportWidth - 1u);
        probe.pixelMaxY = std::min<std::uint32_t>(
            static_cast<std::uint32_t>(std::ceil(maxV * viewportHeight)) - 1u,
            viewportHeight - 1u);
        if (probe.pixelMaxX < probe.pixelMinX || probe.pixelMaxY < probe.pixelMinY)
            continue;
        probe.pixelArea = static_cast<std::uint64_t>(probe.pixelMaxX - probe.pixelMinX + 1u) *
            (probe.pixelMaxY - probe.pixelMinY + 1u);
        probes.push_back(probe);
    }

    std::sort(probes.begin(), probes.end(), [](const Probe& lhs, const Probe& rhs) {
        if (lhs.pixelArea != rhs.pixelArea)
            return lhs.pixelArea < rhs.pixelArea;
        return lhs.clusterIndex < rhs.clusterIndex;
    });
    if (!probes.empty())
    {
        m_CpuVerifyPixelX = probes.front().pixelMinX;
        m_CpuVerifyPixelY = probes.front().pixelMinY;
    }

    std::ofstream depthFile{"DebugCaptures/67-depth-cpu.pgm", std::ios::binary};
    if (depthFile)
    {
        depthFile << "P5\n" << viewportWidth << ' ' << viewportHeight << "\n65535\n";
        for (std::uint32_t y = 0; y < viewportHeight; ++y)
        {
            const auto* row = static_cast<const std::uint8_t*>(mapped.pData) +
                static_cast<std::size_t>(y) * mapped.Stride;
            for (std::uint32_t x = 0; x < viewportWidth; ++x)
            {
                float depth = 1.0f;
                std::memcpy(&depth,
                            row + static_cast<std::size_t>(x) * sizeof(float),
                            sizeof(depth));
                const std::uint16_t value = static_cast<std::uint16_t>(
                    std::clamp(depth, 0.0f, 1.0f) * 65535.0f);
                const char bytes[2] = {
                    static_cast<char>((value >> 8u) & 0xFFu),
                    static_cast<char>(value & 0xFFu)};
                depthFile.write(bytes, sizeof(bytes));
            }
        }
    }

    std::cerr << "Nanite CPU occlusion verify: instance=10 clusters=" << probes.size()
              << " depthImage=" << (depthFile ? "DebugCaptures/67-depth-cpu.pgm" : "unavailable")
              << '\n';
    const std::size_t printCount = std::min<std::size_t>(probes.size(), 12u);
    for (std::size_t probeIndex = 0; probeIndex < printCount; ++probeIndex)
    {
        const Probe& probe = probes[probeIndex];
        float minDepth = 1.0f;
        float maxDepth = 0.0f;
        std::uint64_t frontPixels = 0;
        std::uint64_t backgroundPixels = 0;
        std::uint64_t candidatePixels = 0;
        for (std::uint32_t y = probe.pixelMinY; y <= probe.pixelMaxY; ++y)
        {
            const auto* row = static_cast<const std::uint8_t*>(mapped.pData) +
                static_cast<std::size_t>(y) * mapped.Stride;
            for (std::uint32_t x = probe.pixelMinX; x <= probe.pixelMaxX; ++x)
            {
                float depth;
                std::memcpy(&depth,
                            row + static_cast<std::size_t>(x) * sizeof(float),
                            sizeof(depth));
                minDepth = std::min(minDepth, depth);
                maxDepth = std::max(maxDepth, depth);
                if (depth >= 1.0f - depthEpsilon)
                    ++backgroundPixels;
                else if (depth < probe.clipMinZ - depthEpsilon)
                    ++frontPixels;
                else
                    ++candidatePixels;
            }
        }
        const bool fullyOccluded = maxDepth <= probe.clipMinZ + depthEpsilon;
        std::cerr << std::fixed << std::setprecision(6)
                  << "  cluster=" << probe.clusterIndex
                  << " localBBoxMin=(" << m_CpuScene.Clusters[probe.clusterIndex].BBoxMin[0]
                  << ',' << m_CpuScene.Clusters[probe.clusterIndex].BBoxMin[1]
                  << ',' << m_CpuScene.Clusters[probe.clusterIndex].BBoxMin[2] << ')'
                  << " localBBoxMax=(" << m_CpuScene.Clusters[probe.clusterIndex].BBoxMax[0]
                  << ',' << m_CpuScene.Clusters[probe.clusterIndex].BBoxMax[1]
                  << ',' << m_CpuScene.Clusters[probe.clusterIndex].BBoxMax[2] << ')'
                  << " clipXY=(" << probe.clipMinX << ',' << probe.clipMinY
                  << ")->(" << probe.clipMaxX << ',' << probe.clipMaxY << ')'
                  << " clipZ=(" << probe.clipMinZ << ',' << probe.clipMaxZ << ')'
                  << " pixels=(" << probe.pixelMinX << ',' << probe.pixelMinY
                  << ")->(" << probe.pixelMaxX << ',' << probe.pixelMaxY << ')'
                  << " area=" << probe.pixelArea
                  << " depth=(" << minDepth << ',' << maxDepth << ')'
                  << " front=" << frontPixels
                  << " background=" << backgroundPixels
                  << " candidate=" << candidatePixels
                  << " fullyOccluded=" << (fullyOccluded ? "yes" : "no")
                  << '\n';
    }
}

bool GpuScene::ReadbackStats(Diligent::IDeviceContext* context, bool dumpGroupTasks)
{
    using namespace Diligent;
    // The draw count is copied into the same staging buffer as the queue state, so
    // it has to sit past the end of it. It used to overlap PostGroupWrite, which
    // only worked because nothing read that field back.
    constexpr Uint64 drawCountOffset = 224;
    constexpr Uint64 groupQueueOffset = 256;
    static_assert(sizeof(QueueState) <= drawCountOffset,
                  "Queue state readback would overlap the draw count");
    static_assert(drawCountOffset + sizeof(std::uint32_t) <= groupQueueOffset,
                  "Draw count readback would overlap the group queue dump");

    if (!m_StatsReadbackPending)
    {
        context->CopyBuffer(
            m_QueueState,
            0,
            RESOURCE_STATE_TRANSITION_MODE_TRANSITION,
            m_DebugQueueReadback,
            0,
            sizeof(QueueState),
            RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
        // The number of drawn clusters is the command's InstanceCount, which the
        // culling shaders increment in place; there is no separate counter buffer.
        context->CopyBuffer(
            m_DrawCommands,
            offsetof(DrawCommand, InstanceCount),
            RESOURCE_STATE_TRANSITION_MODE_TRANSITION,
            m_DebugQueueReadback,
            drawCountOffset,
            sizeof(std::uint32_t),
            RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
        if (dumpGroupTasks)
        {
            context->CopyBuffer(
                m_GroupQueue,
                0,
                RESOURCE_STATE_TRANSITION_MODE_TRANSITION,
                m_DebugQueueReadback,
                groupQueueOffset,
                static_cast<Uint64>(m_Capacity.MaxGroupTasks) * sizeof(GroupTask),
                RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
        }

        // Submit the copy without waiting. The next call polls the staging
        // buffer and updates the displayed counters only when it is ready.
        m_StatsReadbackProbeRepeat = std::max(m_ProbeStatsRepeat, 1u);
        context->Flush();
        m_StatsReadbackPending = true;
        return false;
    }

    PVoid mappedData = nullptr;
    context->MapBuffer(m_DebugQueueReadback, MAP_READ, MAP_FLAG_DO_NOT_WAIT, mappedData);
    if (mappedData == nullptr)
        return false;

    const auto& state = *static_cast<const QueueState*>(mappedData);
    const auto actualDrawCount = *reinterpret_cast<const std::uint32_t*>(
        static_cast<const std::uint8_t*>(mappedData) + drawCountOffset);
    m_LastFrameStats.InstanceCount = m_ActiveInstanceCount;
    m_LastFrameStats.VisibleInstanceCount = state.VisibleInstanceCount;
    m_LastFrameStats.NodeTaskCount = state.NodeRead;
    m_LastFrameStats.GroupTaskCount = state.GroupRead;
    m_LastFrameStats.NodeWriteCount = state.NodeWrite;
    m_LastFrameStats.GroupWriteCount = state.GroupWrite;
    m_LastFrameStats.NodePendingCount = state.NodePending;
    m_LastFrameStats.GroupPendingCount = state.GroupPending;
    m_LastFrameStats.ClusterCandidateCount = state.VisibleClusterCount;
    m_LastFrameStats.DrawCount = actualDrawCount;
    m_LastFrameStats.OverflowFlags = state.OverflowFlags;
    m_LastFrameStats.GroupFrustumRejected = state.GroupFrustumRejected;
    m_LastFrameStats.GroupHZBRejected = state.GroupHZBRejected;
    m_LastFrameStats.NodeLodRejected = state.NodeLodRejected;
    m_LastFrameStats.ClusterFrustumRejected = state.ClusterFrustumRejected;
    m_LastFrameStats.ClusterHZBRejected = state.ClusterHZBRejected;
    m_LastFrameStats.ClusterRefinementRejected = state.ClusterRefinementRejected;
    m_LastFrameStats.ClusterTinyRejected = state.ClusterTinyRejected;
    m_LastFrameStats.PostClusterCount = state.PostClusterCount;
    m_LastFrameStats.PostClusterHZBRejected = state.PostClusterHZBRejected;
    m_LastFrameStats.PostInstanceCount = state.PostInstanceCount;
    m_LastFrameStats.PostNodeCount = state.PostNodeCount;
    m_LastFrameStats.PostGroupCount = state.PostGroupCount;
    m_LastFrameStats.PostInstanceRecovered = state.PostInstanceRecovered;
    m_LastFrameStats.PostNodeRecovered = state.PostNodeRecovered;
    m_LastFrameStats.PostGroupRecovered = state.PostGroupRecovered;
    const std::uint32_t probeRepeat = std::max(m_StatsReadbackProbeRepeat, 1u);
    const auto normalizeProbeCounter = [probeRepeat](std::uint32_t value) {
        return value / probeRepeat;
    };
    m_LastFrameStats.SoftRasterTriangles = normalizeProbeCounter(state.SoftRasterTriangles);
    m_LastFrameStats.SoftRasterPixels = normalizeProbeCounter(state.SoftRasterPixels);
    m_LastFrameStats.SoftRasterSkippedLarge = normalizeProbeCounter(state.SoftRasterSkippedLarge);
    m_LastFrameStats.SoftRasterSkippedClip = normalizeProbeCounter(state.SoftRasterSkippedClip);
    m_LastFrameStats.SoftDepthMatched = state.SoftDepthMatched;
    m_LastFrameStats.SoftDepthMismatch = state.SoftDepthMismatch;
    m_LastFrameStats.SoftDepthMissing = state.SoftDepthMissing;
    m_LastFrameStats.SoftDepthExtra = state.SoftDepthExtra;
    m_LastFrameStats.SoftDepthMaxDiff = state.SoftDepthMaxDiff;
    m_LastFrameStats.SoftVisMissing = state.SoftVisMissing;
    m_LastFrameStats.SoftVisKeyMismatch = state.SoftVisKeyMismatch;
    m_LastFrameStats.SoftVisResolved = state.SoftVisResolved;
    m_LastFrameStats.SoftVisNoCover = state.SoftVisNoCover;
    m_LastFrameStats.SoftVisDepthOff = state.SoftVisDepthOff;
    m_LastFrameStats.SoftClusterCount = state.SoftClusterCount;
    m_LastFrameStats.HardwareClusterCount = state.HardwareClusterCount;
    m_LastFrameStats.OverflowFlags = state.OverflowFlags;
    m_LastFrameStats.HZBEnabled = m_CullingData.EnableHZB != 0;

    if (dumpGroupTasks)
    {
        std::cerr << "Nanite queue state: enableHZB=" << m_CullingData.EnableHZB
                  << " hzb=" << m_HzbWidth << 'x' << m_HzbHeight
                  << " mips=" << m_HzbMipCount
                  << " visibleInstances=" << state.VisibleInstanceCount
                  << " nodeRead=" << state.NodeRead
                  << " nodeWrite=" << state.NodeWrite
                  << " nodePending=" << state.NodePending
                  << " groupRead=" << state.GroupRead
                  << " groupWrite=" << state.GroupWrite
                  << " groupPending=" << state.GroupPending
                  << " clusterWrite=" << state.ClusterWrite
                  << " visibleClusters=" << state.VisibleClusterCount
                  << " drawCount=" << state.DrawCount
                  << " actualDrawCount=" << actualDrawCount
                  << " groupRejects={frustum=" << state.GroupFrustumRejected
                  << ",hzb=" << state.GroupHZBRejected << "}"
                  << " clusterRejects={frustum=" << state.ClusterFrustumRejected
                  << ",hzb=" << state.ClusterHZBRejected
                  << ",refinement=" << state.ClusterRefinementRejected
                  << ",tiny=" << state.ClusterTinyRejected << "}"
                  << " post={count=" << state.PostClusterCount
                  << ",hzb=" << state.PostClusterHZBRejected << "}"
                  << " recover={instance=" << state.PostInstanceRecovered
                  << '/' << state.PostInstanceCount
                  << ",node=" << state.PostNodeRecovered << '/' << state.PostNodeCount
                  << ",group=" << state.PostGroupRecovered << '/' << state.PostGroupCount
                  << ",mainClusters=" << state.MainClusterCount << "}"
                  << " overflow=0x" << std::hex << state.OverflowFlags << std::dec
                  << " debugProbeTexel=(" << (state.DebugProbeX & 0xFFFFu)
                  << ',' << (state.DebugProbeY & 0xFFFFu) << ")->("
                  << (state.DebugProbeX >> 16u)
                  << ',' << (state.DebugProbeY >> 16u) << ")\n";
        float instanceProbeDepth = 0.0f;
        std::memcpy(&instanceProbeDepth, &state.DebugProbeX, sizeof(instanceProbeDepth));
        float instanceProbeFlipped = 0.0f;
        std::memcpy(&instanceProbeFlipped, &state.DebugProbeY, sizeof(instanceProbeFlipped));
        std::cerr << "  instance0Probe: depth=" << instanceProbeDepth
                  << " flipped=" << instanceProbeFlipped << '\n';
        const auto* groupTasks = reinterpret_cast<const GroupTask*>(
            static_cast<const std::uint8_t*>(mappedData) + groupQueueOffset);
        for (std::uint32_t i = 0; i < std::min<std::uint32_t>(state.GroupWrite, 12u); ++i)
        {
            std::cerr << "  groupTask[" << i << "]: instance=" << groupTasks[i].InstanceIndex
                      << " group=" << groupTasks[i].GroupIndex
                      << " depth=" << groupTasks[i].RefinementDepth
                      << " ready=" << groupTasks[i].Ready << '\n';
        }
    }
    context->UnmapBuffer(m_DebugQueueReadback, MAP_READ);
    m_StatsReadbackPending = false;
    return true;
}

void GpuScene::UpdateAutoRasterDecision()
{
    if (!m_AutoRasterRouting || !m_AutoRasterLastHybrid)
    {
        if (m_AutoRasterRouting && m_AutoRasterHardware)
        {
            ++m_AutoRasterHardwareFrames;
            // Re-enter hybrid periodically so a camera move or a new LOD cut is
            // not hidden by a stale all-hardware decision.
            if (m_AutoRasterHardwareFrames >= 120u)
            {
                m_AutoRasterHardware = false;
                m_AutoRasterHardwareFrames = 0;
                std::cerr << "Nanite auto raster: periodic hybrid calibration\n";
            }
        }
        return;
    }

    // A hybrid frame pays RasterBin even when only a handful of entries go to
    // software. Use the actual software coverage and GPU time rather than a raw
    // triangle count: one large triangle can cost more than hundreds of tiny
    // triangles in the compute sweep. Mixed clusters may be counted in both bins,
    // so use DrawCount as the denominator and keep a conservative margin.
    const float drawCount = static_cast<float>(std::max(m_LastFrameStats.DrawCount, 1u));
    const float softwareFraction =
        static_cast<float>(m_LastFrameStats.SoftClusterCount) / drawCount;
    const float softwareMs = m_LastFrameStats.Gpu.Valid ?
        m_LastFrameStats.Gpu.SoftRasterMs : 0.0f;
    const float binMs = m_LastFrameStats.Gpu.Valid ?
        m_LastFrameStats.Gpu.RasterBinMs : 0.0f;
    const bool smallSoftwareBin = softwareFraction <= 0.12f;
    const bool cheapSoftwareWork = !m_LastFrameStats.Gpu.Valid ||
        softwareMs <= std::max(0.20f, binMs * 1.5f);

    if (smallSoftwareBin && cheapSoftwareWork)
    {
        const bool changed = !m_AutoRasterHardware;
        m_AutoRasterHardware = true;
        m_AutoRasterHardwareFrames = 0;
        if (changed)
        {
            std::cerr << "Nanite auto raster: hardware bypass (swEntries="
                      << m_LastFrameStats.SoftClusterCount
                      << "/draws=" << m_LastFrameStats.DrawCount
                      << ", swPixels=" << m_LastFrameStats.SoftRasterPixels
                      << ", binMs=" << binMs
                      << ", softMs=" << softwareMs << ")\n";
        }
    }
    else
    {
        const bool changed = m_AutoRasterHardware;
        m_AutoRasterHardware = false;
        m_AutoRasterHardwareFrames = 0;
        if (changed)
        {
            std::cerr << "Nanite auto raster: hybrid (swEntries="
                      << m_LastFrameStats.SoftClusterCount
                      << "/draws=" << m_LastFrameStats.DrawCount
                      << ", swPixels=" << m_LastFrameStats.SoftRasterPixels
                      << ", binMs=" << binMs
                      << ", softMs=" << softwareMs << ")\n";
        }
    }
}

void GpuScene::BeginGpuTimingFrame(Diligent::IDeviceContext* context)
{
    m_GpuTimingFrameActive = false;
    m_ActiveGpuTimingSlot = -1;
    if (!m_GpuTimingEnabled || !m_GpuTimingSupported)
        return;

    PollGpuTimings();
    for (std::uint32_t slotIndex = 0; slotIndex < m_GpuTimingSlots.size(); ++slotIndex)
    {
        GpuTimingSlot& slot = m_GpuTimingSlots[slotIndex];
        if (slot.InFlight)
            continue;

        slot.FrameIndex = m_FrameIndex;
        slot.InFlight = true;
        slot.IssuedPassMask = 0;
        slot.ProbeRepeat = 1;
        m_ActiveGpuTimingSlot = static_cast<int>(slotIndex);
        m_GpuTimingFrameActive = true;
        BeginGpuTimingPass(context, 0u);
        return;
    }
    // A slow GPU may still own every slot. Leave this frame uninstrumented
    // instead of waiting for it or reusing a query that is still in flight.
}

void GpuScene::BeginGpuTimingPass(Diligent::IDeviceContext* context, std::uint32_t pass)
{
    if (!m_GpuTimingFrameActive || m_ActiveGpuTimingSlot < 0 || pass >= GpuTimingPassCount)
        return;

    auto& query = m_GpuTimingSlots[static_cast<std::size_t>(m_ActiveGpuTimingSlot)].Queries[pass * 2];
    if (query != nullptr)
        context->EndQuery(query);
}

void GpuScene::EndGpuTimingPass(Diligent::IDeviceContext* context, std::uint32_t pass)
{
    if (!m_GpuTimingFrameActive || m_ActiveGpuTimingSlot < 0 || pass >= GpuTimingPassCount)
        return;

    auto& query = m_GpuTimingSlots[static_cast<std::size_t>(m_ActiveGpuTimingSlot)].Queries[pass * 2 + 1];
    if (query != nullptr)
    {
        context->EndQuery(query);
        // Only now is the pair complete and safe to read back. Both halves are
        // ended in lockstep, so recording it here covers the start query too.
        m_GpuTimingSlots[static_cast<std::size_t>(m_ActiveGpuTimingSlot)].IssuedPassMask |=
            1u << pass;
    }
}

void GpuScene::EndGpuTimingFrame(Diligent::IDeviceContext* context)
{
    if (!m_GpuTimingFrameActive)
        return;
    EndGpuTimingPass(context, 0u);
    m_GpuTimingFrameActive = false;
    m_ActiveGpuTimingSlot = -1;
}

void GpuScene::SetGpuTimingsEnabled(bool enabled)
{
    if (!m_GpuTimingSupported)
        enabled = false;
    m_GpuTimingEnabled = enabled;
    m_LastFrameStats.Gpu.Enabled = enabled;
    if (!enabled)
        m_LastFrameStats.Gpu.Valid = false;
}

void GpuScene::SetActiveInstanceCount(std::uint32_t count)
{
    const std::uint32_t maxCount = GetMaxInstanceCount();
    if (maxCount == 0u)
        return;
    count = std::clamp(count, 1u, maxCount);
    if (count == m_ActiveInstanceCount)
        return;

    m_InstanceCountX = 0;
    m_InstanceCountY = 0;
    m_InstanceCountZ = 0;
    m_ActiveInstanceCount = count;
    // Prototype scenes carry authored transforms and roots; only legacy scenes
    // use the live rabbit cube layout controls.
    if (!m_CustomInstanceLayout)
        LayoutInstances(m_CpuScene.Instances, count, m_SpacingHorizontal, m_SpacingVertical);
    m_InstanceUploadPending = true;
    m_ShadowMapValid = false;
    InvalidateVirtualShadowPages();
    m_LastFrameStats.InstanceCount = count;
    m_LastFrameStats.LogicalTriangleCount = LogicalTriangleCount(m_CpuScene, count);
}

void GpuScene::SetAtmosphereSettings(const AtmosphereSettings& settings)
{
    const float oldAzimuth = m_AtmosphereSettings.SunAzimuthDeg;
    const float oldElevation = m_AtmosphereSettings.SunElevationDeg;
    const auto finiteOr = [](float value, float fallback) {
        return std::isfinite(value) ? value : fallback;
    };
    m_AtmosphereSettings.SunAzimuthDeg = finiteOr(
        settings.SunAzimuthDeg, m_AtmosphereSettings.SunAzimuthDeg);
    m_AtmosphereSettings.SunElevationDeg = std::clamp(
        finiteOr(settings.SunElevationDeg, m_AtmosphereSettings.SunElevationDeg),
        -89.0f, 89.0f);
    m_AtmosphereSettings.AtmosphereRadius = std::max(
        finiteOr(settings.AtmosphereRadius, m_AtmosphereSettings.AtmosphereRadius), 1.0f);
    m_AtmosphereSettings.AtmosphereHeight = std::max(
        finiteOr(settings.AtmosphereHeight, m_AtmosphereSettings.AtmosphereHeight), 1.0f);
    m_AtmosphereSettings.RayleighScale = std::max(
        finiteOr(settings.RayleighScale, m_AtmosphereSettings.RayleighScale), 0.0f);
    m_AtmosphereSettings.MieScale = std::max(
        finiteOr(settings.MieScale, m_AtmosphereSettings.MieScale), 0.0f);
    m_AtmosphereSettings.SkyDensity = std::max(
        finiteOr(settings.SkyDensity, m_AtmosphereSettings.SkyDensity), 0.0f);
    m_AtmosphereSettings.TerrainDensity = std::max(
        finiteOr(settings.TerrainDensity, m_AtmosphereSettings.TerrainDensity), 0.0f);
    m_AtmosphereSettings.HazeDistance = std::max(
        finiteOr(settings.HazeDistance, m_AtmosphereSettings.HazeDistance), 0.01f);
    m_AtmosphereSettings.HazeWeight = std::clamp(
        finiteOr(settings.HazeWeight, m_AtmosphereSettings.HazeWeight), 0.0f, 1.0f);

    if (oldAzimuth != m_AtmosphereSettings.SunAzimuthDeg ||
        oldElevation != m_AtmosphereSettings.SunElevationDeg)
    {
        // The VSM is intentionally cached, but a light-direction edit changes
        // every light-space page and must start a fresh cache population.
        m_ShadowMapValid = false;
        InvalidateVirtualShadowPages();
    }
}

void GpuScene::SetInstanceGrid(std::uint32_t countX,
                               std::uint32_t countY,
                               std::uint32_t countZ)
{
    if (countX == 0u || countY == 0u || countZ == 0u)
        throw std::runtime_error{"Instance grid dimensions must be greater than zero"};
    const std::uint64_t total = static_cast<std::uint64_t>(countX) * countY * countZ;
    if (total > GetMaxInstanceCount())
        throw std::runtime_error{"Instance grid exceeds the scene instance capacity"};
    if (countX == m_InstanceCountX && countY == m_InstanceCountY && countZ == m_InstanceCountZ &&
        total == m_ActiveInstanceCount)
        return;

    m_InstanceCountX = countX;
    m_InstanceCountY = countY;
    m_InstanceCountZ = countZ;
    m_ActiveInstanceCount = static_cast<std::uint32_t>(total);
    LayoutInstanceGrid(
        m_CpuScene.Instances,
        countX,
        countY,
        countZ,
        m_SpacingHorizontal,
        m_SpacingVertical);
    m_InstanceUploadPending = true;
    m_ShadowMapValid = false;
    InvalidateVirtualShadowPages();
    m_LastFrameStats.InstanceCount = m_ActiveInstanceCount;
    m_LastFrameStats.LogicalTriangleCount =
        LogicalTriangleCount(m_CpuScene, m_ActiveInstanceCount);
}

void GpuScene::SetInstanceSpacing(float horizontal, float vertical)
{
    // A spacing of zero stacks every instance on the same spot, which is a valid
    // thing to want to look at: it is the worst case for overdraw. Negative would
    // only mirror the block, so clamp there and nowhere else.
    horizontal = std::max(horizontal, 0.0f);
    vertical = std::max(vertical, 0.0f);
    if (horizontal == m_SpacingHorizontal && vertical == m_SpacingVertical)
        return;

    m_SpacingHorizontal = horizontal;
    m_SpacingVertical = vertical;
    // Same deal as the count: transforms only, no GPU allocation, so the upload
    // is all that is needed and it waits for Render like the count's does.
    if (m_InstanceCountX != 0u)
    {
        LayoutInstanceGrid(
            m_CpuScene.Instances,
            m_InstanceCountX,
            m_InstanceCountY,
            m_InstanceCountZ,
            m_SpacingHorizontal,
            m_SpacingVertical);
    }
    else
    {
        LayoutInstances(
            m_CpuScene.Instances, m_ActiveInstanceCount, m_SpacingHorizontal, m_SpacingVertical);
    }
    m_InstanceUploadPending = true;
    m_ShadowMapValid = false;
    InvalidateVirtualShadowPages();
}

void GpuScene::PollGpuTimings()
{
    if (!m_GpuTimingEnabled || !m_GpuTimingSupported)
        return;

    for (auto& slot : m_GpuTimingSlots)
    {
        if (!slot.InFlight)
            continue;

        std::array<Diligent::QueryDataTimestamp, GpuTimingQueryCount> timestamps{};
        bool ready = true;
        for (std::uint32_t pass = 0; pass < GpuTimingPassCount && ready; ++pass)
        {
            // Skip passes this frame never issued. Their queries exist but were
            // never ended, and reading one of those crashes in the Vulkan backend.
            if ((slot.IssuedPassMask & (1u << pass)) == 0u)
                continue;

            for (std::uint32_t half = 0; half < 2u; ++half)
            {
                const std::uint32_t queryIndex = pass * 2 + half;
                if (slot.Queries[queryIndex] == nullptr ||
                    !slot.Queries[queryIndex]->GetData(
                        &timestamps[queryIndex], sizeof(timestamps[queryIndex]), false))
                {
                    ready = false;
                    break;
                }
            }
        }
        if (!ready)
            continue;

        GpuTimings timings{};
        timings.Enabled = true;
        timings.Valid = true;
        timings.AsyncRaster = m_LastAsyncRaster;
        timings.Native64BitVisibility = m_Native64BitVisibility;
        auto durationMs = [&](std::uint32_t pass) {
            // A pass that did not run this frame is reported as zero and does not
            // invalidate the rest of the frame's numbers.
            if ((slot.IssuedPassMask & (1u << pass)) == 0u)
                return 0.0f;
            const auto& start = timestamps[pass * 2];
            const auto& end = timestamps[pass * 2 + 1];
            if (start.Frequency == 0 || end.Frequency == 0 || end.Counter < start.Counter)
            {
                timings.Valid = false;
                return 0.0f;
            }
            return static_cast<float>(
                static_cast<double>(end.Counter - start.Counter) * 1000.0 /
                static_cast<double>(start.Frequency));
        };
        timings.FrameMs = durationMs(0u);
        timings.InstanceCullMs = durationMs(1u);
        timings.PersistentCullMs = durationMs(2u);
        timings.GenerateIndirectMs = durationMs(3u);
        timings.PostCullMs = durationMs(4u);
        timings.QueueResetMs = durationMs(16u);
        timings.PrepareDispatchMs = durationMs(17u) + durationMs(18u) + durationMs(19u);
        timings.PostRecoveryMs = durationMs(20u);
        timings.RasterMs = durationMs(5u);
        timings.PostRasterMs = durationMs(6u);
        timings.HizBuildMs = durationMs(7u);
        timings.FinalHizBuildMs = durationMs(21u);
        const float probeRepeat = static_cast<float>(std::max(slot.ProbeRepeat, 1u));
        timings.SoftDepthMs = m_SoftRasterEnabled ? durationMs(8u) / probeRepeat : 0.0f;
        timings.SoftVisibilityMs = m_SoftRasterEnabled ? durationMs(15u) / probeRepeat : 0.0f;
        timings.SoftRasterMs = timings.SoftDepthMs + timings.SoftVisibilityMs;
        timings.SoftCheckMs = m_SoftRasterEnabled ? durationMs(9u) / probeRepeat : 0.0f;
        timings.HardwareDepthOnlyMs = m_SoftRasterEnabled ? durationMs(10u) / probeRepeat : 0.0f;
        timings.ShadowMapMs = durationMs(22u);
        timings.RasterBinMs = m_SoftRasterEnabled ? durationMs(11u) : 0.0f;
        timings.VisibilityResolveMs = durationMs(12u) + durationMs(13u);
        timings.ShadingMs = durationMs(14u);

        if (timings.Valid)
            m_LastFrameStats.Gpu = timings;

        // The window title is the only other place these land, which makes an
        // A/B comparison of HZB on/off awkward. Accumulate and print an average
        // so a single noisy frame cannot be mistaken for a trend.
        if (timings.Valid)
        {
            m_OverflowSticky |= m_LastFrameStats.OverflowFlags;
            m_PeakDrawCount = std::max(m_PeakDrawCount, m_LastFrameStats.DrawCount);
            m_PeakNodeTasks = std::max(m_PeakNodeTasks, m_LastFrameStats.NodeTaskCount);
            m_PeakCandidates =
                std::max(m_PeakCandidates, m_LastFrameStats.ClusterCandidateCount);
            m_MinDrawCount = std::min(m_MinDrawCount, m_LastFrameStats.DrawCount);
            m_MinCandidates =
                std::min(m_MinCandidates, m_LastFrameStats.ClusterCandidateCount);
            m_GpuTimingAccum.FrameMs += timings.FrameMs;
            m_GpuTimingAccum.InstanceCullMs += timings.InstanceCullMs;
            m_GpuTimingAccum.PersistentCullMs += timings.PersistentCullMs;
            m_GpuTimingAccum.GenerateIndirectMs += timings.GenerateIndirectMs;
            m_GpuTimingAccum.PostCullMs += timings.PostCullMs;
            m_GpuTimingAccum.QueueResetMs += timings.QueueResetMs;
            m_GpuTimingAccum.PrepareDispatchMs += timings.PrepareDispatchMs;
            m_GpuTimingAccum.PostRecoveryMs += timings.PostRecoveryMs;
            m_GpuTimingAccum.RasterMs += timings.RasterMs;
            m_GpuTimingAccum.PostRasterMs += timings.PostRasterMs;
            m_GpuTimingAccum.HizBuildMs += timings.HizBuildMs;
            m_GpuTimingAccum.FinalHizBuildMs += timings.FinalHizBuildMs;
            m_GpuTimingAccum.SoftRasterMs += timings.SoftRasterMs;
            m_GpuTimingAccum.SoftDepthMs += timings.SoftDepthMs;
            m_GpuTimingAccum.SoftVisibilityMs += timings.SoftVisibilityMs;
            m_GpuTimingAccum.RasterBinMs += timings.RasterBinMs;
            m_GpuTimingAccum.SoftCheckMs += timings.SoftCheckMs;
            m_GpuTimingAccum.HardwareDepthOnlyMs += timings.HardwareDepthOnlyMs;
            m_GpuTimingAccum.ShadowMapMs += timings.ShadowMapMs;
            m_GpuTimingAccum.VisibilityResolveMs += timings.VisibilityResolveMs;
            m_GpuTimingAccum.ShadingMs += timings.ShadingMs;
            if (++m_GpuTimingPrintCounter >= 120u)
            {
                const float inv = 1.0f / static_cast<float>(m_GpuTimingPrintCounter);
                std::cout << "Nanite gpu ms avg" << m_GpuTimingPrintCounter
                          << ": frame=" << m_GpuTimingAccum.FrameMs * inv
                          << " instanceCull=" << m_GpuTimingAccum.InstanceCullMs * inv
                          << " persistentCull=" << m_GpuTimingAccum.PersistentCullMs * inv
                          << " genIndirect=" << m_GpuTimingAccum.GenerateIndirectMs * inv
                          << " queueReset=" << m_GpuTimingAccum.QueueResetMs * inv
                          << " prepare=" << m_GpuTimingAccum.PrepareDispatchMs * inv
                          << " postRecovery=" << m_GpuTimingAccum.PostRecoveryMs * inv
                          << " raster=" << m_GpuTimingAccum.RasterMs * inv
                          << " visibilityResolve=" << m_GpuTimingAccum.VisibilityResolveMs * inv
                          << " visibilityShading=" << m_GpuTimingAccum.ShadingMs * inv
                          << " hizBuild=" << m_GpuTimingAccum.HizBuildMs * inv
                          << " finalHiz=" << m_GpuTimingAccum.FinalHizBuildMs * inv
                          << " postCull=" << m_GpuTimingAccum.PostCullMs * inv
                          << " postRaster=" << m_GpuTimingAccum.PostRasterMs * inv
                          << " shadowMap=" << m_GpuTimingAccum.ShadowMapMs * inv
                          // Printed alongside the timings so a change that is fast
                          // because it dropped geometry is obvious in the same line.
                          << " draws=" << m_LastFrameStats.DrawCount
                          << " candidates=" << m_LastFrameStats.ClusterCandidateCount
                          << " visInst=" << m_LastFrameStats.VisibleInstanceCount
                          << " nodeTasks=" << m_LastFrameStats.NodeTaskCount
                          << '/' << m_LastFrameStats.NodeWriteCount
                          << " groupTasks=" << m_LastFrameStats.GroupTaskCount
                          << '/' << m_LastFrameStats.GroupWriteCount
                          << " pending=" << m_LastFrameStats.NodePendingCount
                          << ',' << m_LastFrameStats.GroupPendingCount
                          << " gRejF=" << m_LastFrameStats.GroupFrustumRejected
                          << " gRejH=" << m_LastFrameStats.GroupHZBRejected
                          // The LOD cut and the occlusion cut are the two things
                          // that decide how much of the scene is even looked at,
                          // so they belong in the same line as the timings.
                          << " nodeLodRej=" << m_LastFrameStats.NodeLodRejected
                          << " cRejF=" << m_LastFrameStats.ClusterFrustumRejected
                          << " cRejH=" << m_LastFrameStats.ClusterHZBRejected
                          << " cRejLod=" << m_LastFrameStats.ClusterRefinementRejected
                          << " cRejTiny=" << m_LastFrameStats.ClusterTinyRejected
                          // postRaster now costs exactly what the recovery queue
                          // yields, so its timing is only readable next to these:
                          // queued is what the stale HZB deferred, drawn is what the
                          // fresh one let back in and all that pass rasterizes.
                          << " postQueued=" << m_LastFrameStats.PostClusterCount
                          << " postDrawn="
                          << (m_LastFrameStats.PostClusterCount -
                              m_LastFrameStats.PostClusterHZBRejected)
                          << " hzb=" << (m_LastFrameStats.HZBEnabled ? "ON" : "OFF")
                          << " overflow=0x" << std::hex << m_LastFrameStats.OverflowFlags
                          // The line above is one sampled frame. A flicker only
                          // shows up in the worst frame of the window, so the
                          // sticky overflow bits and the peaks are what tell
                          // whether a cap was hit at all.
                          << " stickyOverflow=0x" << m_OverflowSticky
                          << std::dec
                          << " peakDraws=" << m_PeakDrawCount
                          << " peakNodeTasks=" << m_PeakNodeTasks
                          << " peakCandidates=" << m_PeakCandidates
                          << " drawSpread=" << m_MinDrawCount << ".." << m_PeakDrawCount
                          << " candSpread=" << m_MinCandidates << ".." << m_PeakCandidates
                          << std::endl;
                // The probe's own line, so the default output stays as it was.
                // softRaster next to raster is the whole comparison; the counters
                // after it say how much work that time bought, and the depth
                // counters say whether the work was correct. missing should be
                // accounted for by skipClip and skipLarge - if it is not, the
                // rasterizer is dropping coverage and the timing is meaningless.
                if (m_SoftRasterEnabled)
                {
                    std::cout << "Nanite soft raster avg" << m_GpuTimingPrintCounter
                              << ": rasterBin=" << m_GpuTimingAccum.RasterBinMs * inv
                              << " softDepth=" << m_GpuTimingAccum.SoftDepthMs * inv
                              << " softVisibility=" << m_GpuTimingAccum.SoftVisibilityMs * inv
                              << " softRaster=" << m_GpuTimingAccum.SoftRasterMs * inv
                              << " atomic64=" << (m_LastFrameStats.Gpu.Native64BitVisibility ? 1 : 0)
                              << " hwDepthOnly=" << m_GpuTimingAccum.HardwareDepthOnlyMs * inv
                              << " repeat=" << m_ProbeRepeat
                              << " clusterMode=" << (m_ClusterRasterExperiment ? 1 : 0)
                              << " swClusters=" << m_LastFrameStats.SoftClusterCount
                              << " hwClusters=" << m_LastFrameStats.HardwareClusterCount
                              << " clusterArea=" << m_ClusterRasterAreaCutoff
                              << " softCheck=" << m_GpuTimingAccum.SoftCheckMs * inv
                              << " vs shadedRaster=" << m_GpuTimingAccum.RasterMs * inv
                              << " tris=" << m_LastFrameStats.SoftRasterTriangles
                              << " pixels=" << m_LastFrameStats.SoftRasterPixels
                              << " skipClip=" << m_LastFrameStats.SoftRasterSkippedClip
                              << " skipLarge=" << m_LastFrameStats.SoftRasterSkippedLarge
                              << " depthMatch=" << m_LastFrameStats.SoftDepthMatched
                              << " depthMismatch=" << m_LastFrameStats.SoftDepthMismatch
                              << " missing=" << m_LastFrameStats.SoftDepthMissing
                              << " extra=" << m_LastFrameStats.SoftDepthExtra
                              << " maxDiff="
                              << static_cast<float>(m_LastFrameStats.SoftDepthMaxDiff) /
                                     16777215.0f
                              << " visResolved=" << m_LastFrameStats.SoftVisResolved
                              << " visNoCover=" << m_LastFrameStats.SoftVisNoCover
                              << " visDepthOff=" << m_LastFrameStats.SoftVisDepthOff
                              << " visKeyMismatch=" << m_LastFrameStats.SoftVisKeyMismatch
                              << " visMissing=" << m_LastFrameStats.SoftVisMissing
                              << std::endl;
                }
                m_GpuTimingPrintCounter = 0u;
                m_GpuTimingAccum = GpuTimings{};
                m_OverflowSticky = 0u;
                m_PeakDrawCount = 0u;
                m_PeakNodeTasks = 0u;
                m_PeakCandidates = 0u;
                m_MinDrawCount = 0xFFFFFFFFu;
                m_MinCandidates = 0xFFFFFFFFu;
            }
        }

        for (auto& query : slot.Queries)
        {
            if (query != nullptr)
                query->Invalidate();
        }
        slot.InFlight = false;
    }
}

void GpuScene::ResetQueues(Diligent::IDeviceContext* context)
{
    QueueState zeroState{};
    context->UpdateBuffer(m_QueueState, 0, sizeof(zeroState), &zeroState,
                          Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
    // Each phase draws one command, so the per-frame clear is 32 bytes instead of
    // one zeroed record per potential cluster. VertexCount is the largest
    // cluster's index count and never changes; InstanceCount starts at zero and
    // the culling shaders atomically raise it as clusters claim their slot.
    //
    // The post command's FirstInstance starts at the largest possible value
    // because post cull lowers it with an atomic minimum to the first slot it
    // claims. When post cull recovers nothing, InstanceCount stays zero, the draw
    // is a no-op and the untouched sentinel is never read.
    const DrawCommand emptyCommands[static_cast<std::size_t>(DrawCommandSlot::Count)] = {
        {m_MaxClusterIndexCount, 0u, 0u, 0u},
        {m_MaxClusterIndexCount, 0u, 0u, 0xFFFFFFFFu},
    };
    context->UpdateBuffer(m_DrawCommands, 0, sizeof(emptyCommands), emptyCommands,
                          Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
    const DrawCommand emptyHybridCommand{m_MaxClusterIndexCount, 0u, 0u, 0u};
    context->UpdateBuffer(m_HybridDrawCommand, 0, sizeof(emptyHybridCommand), &emptyHybridCommand,
                          Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
    // g_DrawInstanceData needs no clear: only the slots below InstanceCount are
    // read, and a slot is written before the atomic that publishes it.
    // Queue generations make the Ready field self-invalidating across frames, so
    // the large Node/Group uploads are needed only for the first frame. Keep the
    // old clear path behind an opt-out for A/B and diagnostics.
    if (!m_QueueEpochEnabled || m_QueueGeneration == 1u)
    {
        const Uint64 nodeClearCount = std::min<Uint64>(
            m_EmptyNodeQueue.size(),
            static_cast<Uint64>(m_ActiveInstanceCount) * m_CpuScene.Nodes.size());
        const Uint64 groupClearCount = std::min<Uint64>(
            m_EmptyGroupQueue.size(),
            static_cast<Uint64>(m_ActiveInstanceCount) * m_CpuScene.ClusterGroups.size());
        context->UpdateBuffer(m_NodeQueue, 0,
                              nodeClearCount * sizeof(NodeTask),
                              m_EmptyNodeQueue.data(),
                              Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
        context->UpdateBuffer(m_GroupQueue, 0,
                              groupClearCount * sizeof(GroupTask),
                              m_EmptyGroupQueue.data(),
                              Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
    }
}

void GpuScene::UpdateConstants(Diligent::IDeviceContext* context,
                               std::uint32_t viewportWidth,
                               std::uint32_t viewportHeight,
                               const float viewProjMatrix[16],
                               const float cameraWorldPosition[3])
{
    if (!m_HasPreviousCamera)
    {
        std::copy(viewProjMatrix,
                  viewProjMatrix + m_PreviousViewProj.size(),
                  m_PreviousViewProj.begin());
    }
    std::memcpy(m_CullingData.ViewProjMatrix, viewProjMatrix, sizeof(m_CullingData.ViewProjMatrix));
    std::memcpy(m_CullingData.PreviousViewProjMatrix,
                m_PreviousViewProj.data(),
                sizeof(m_CullingData.PreviousViewProjMatrix));
    std::memcpy(m_CullingData.CameraWorldPosition,
                cameraWorldPosition,
                sizeof(m_CullingData.CameraWorldPosition));
    m_CullingData.ViewportWidth = viewportWidth;
    m_CullingData.ViewportHeight = viewportHeight;
    // cot(fovY/2), recovered from the matrix we already have rather than threaded
    // down from the camera. The combined matrix' second clip row is the projection's
    // (0, focalY, 0, 0) times the view, which is focalY times the view basis' up
    // vector; up is unit length for any rigid view, so the row's length is focalY
    // exactly, whatever the orientation. Reading the diagonal element on its own -
    // which is what the shader used to do - picks up focalY * up.y, that is
    // focalY * cos(pitch), and so silently shrank every projected error as soon as
    // the camera looked up or down. The stride of four is the transpose that
    // BuildViewProjection applies on the way out.
    m_CullingData.FocalY = std::sqrt(viewProjMatrix[1] * viewProjMatrix[1] +
                                     viewProjMatrix[5] * viewProjMatrix[5] +
                                     viewProjMatrix[9] * viewProjMatrix[9]);
    // Camera basis and focal data for the direction-based sky sampling. The
    // basis matches the one BuildViewProjection uses (see main.mm), and the
    // focal is recovered the same way as CullingData.FocalY above.
    const float cosYaw = std::cos(m_CameraYaw);
    const float sinYaw = std::sin(m_CameraYaw);
    const float cosPitch = std::cos(m_CameraPitch);
    const float sinPitch = std::sin(m_CameraPitch);
    const float forward[3] = {sinYaw * cosPitch, sinPitch, -cosYaw * cosPitch};
    const float right[3] = {cosYaw, 0.0f, sinYaw};
    float up[3] = {
        right[1] * forward[2] - right[2] * forward[1],
        right[2] * forward[0] - right[0] * forward[2],
        right[0] * forward[1] - right[1] * forward[0]};
    const float basisLength = std::sqrt(up[0] * up[0] + up[1] * up[1] + up[2] * up[2]);
    if (basisLength > 0.0f)
    {
        up[0] /= basisLength;
        up[1] /= basisLength;
        up[2] /= basisLength;
    }
    std::memcpy(m_RasterData.CameraRight, right, sizeof(right));
    std::memcpy(m_RasterData.CameraUp, up, sizeof(up));
    std::memcpy(m_RasterData.CameraForward, forward, sizeof(forward));
    // The shader divides pixel-space offsets by these values. Convert the
    // projection's dimensionless focal lengths to pixel focal lengths first;
    // using cot(FOV/2) directly makes a 1000-pixel screen offset hundreds of
    // times too large and collapses the view ray onto the screen plane.
    const float pixelHalfWidth = static_cast<float>(viewportWidth) * 0.5f;
    const float pixelHalfHeight = static_cast<float>(viewportHeight) * 0.5f;
    m_RasterData.Focal[0] = m_CullingData.FocalY * pixelHalfHeight;
    m_RasterData.Focal[1] = m_CullingData.FocalY * pixelHalfHeight;
    m_RasterData.Focal[2] = pixelHalfWidth;
    m_RasterData.Focal[3] = pixelHalfHeight;
    m_CullingData.HZBWidth = m_HzbWidth;
    m_CullingData.HZBHeight = m_HzbHeight;
    m_CullingData.HZBMipCount = m_HzbMipCount;

    m_CullingData.EnableHZB =
        (m_HzbValid && m_HZBEnabled && std::getenv("NANITE_DISABLE_HZB") == nullptr) ? 1u : 0u;
    m_CullingData.InstanceCount = m_ActiveInstanceCount;
    m_CullingData.MaxNodeTasks = m_Capacity.MaxNodeTasks;
    m_CullingData.MaxGroupTasks = m_Capacity.MaxGroupTasks;
    m_CullingData.MaxClusterTasks = m_Capacity.MaxClusterTasks;
    m_CullingData.MaxTraversalIterations = m_Capacity.TraversalIterations;
    m_CullingData.QueueGeneration = m_QueueGeneration;
    // The full provisioned capacity: g_DrawInstanceData is sized for every
    // (instance, cluster) pair, so the shader-side bound is only a guard against
    // a capacity that was clamped below that structural upper bound.
    m_CullingData.MaxDrawCommands = m_Capacity.MaxDrawCommands;
    // The high bits are reserved for HZB diagnostics and feature switches.
    // Instance and node level HZB have the same two-phase recovery the cluster
    // level always had: whatever the main phase rejects on occlusion is queued
    // and re-tested against this frame's HZB before the post raster, which is
    // what makes rejecting before/during traversal safe.
    //
    // Off by default because it does not pay in this scene. The 27 rabbits
    // barely occlude each other at node granularity: only ~32 of ~2840 node
    // tasks get rejected, so the extra HZB test on every node task plus the
    // second traversal dispatch costs about 0.5 ms more than it saves
    // (persistentCull ~5.26 ms on, ~4.68 ms off, 120-frame averages). Enable it
    // with NANITE_HIERARCHY_HZB=1 / NANITE_INSTANCE_HZB=1 in occlusion-heavy
    // scenes, where pruning a subtree removes real traversal work.
    const bool enableHierarchyHzb = std::getenv("NANITE_HIERARCHY_HZB") != nullptr;
    const bool enableInstanceHzb = std::getenv("NANITE_INSTANCE_HZB") != nullptr;
    m_CullingData.MaxRefinementDepth = 8u |
        (std::getenv("NANITE_VERIFY_OCCLUSION") != nullptr ? 0x80000000u : 0u) |
        (std::getenv("NANITE_FORCE_HZB_MIP0") != nullptr ? 0x40000000u : 0u) |
        (enableHierarchyHzb ? 0u : 0x20000000u) |
        (enableInstanceHzb ? 0u : 0x10000000u);
    m_CullingData.CullingPass = 0u;
    m_CullingData.ScreenErrorPixels = m_ScreenErrorPixels;
    m_CullingData.RasterBinAreaCutoff = m_ClusterRasterExperiment ?
        m_ClusterRasterAreaCutoff : m_RasterBinAreaCutoff;
    m_CullingData.HybridRasterEnabled = 0u;
    m_CullingData.ClusterRasterMode = m_ClusterRasterExperiment ? 1u : 0u;
    m_CullingData.Padding4 = m_Native64BitVisibility ? 1u : 0u;
    std::copy(viewProjMatrix,
              viewProjMatrix + m_PreviousViewProj.size(),
              m_PreviousViewProj.begin());
    m_HasPreviousCamera = true;
    context->UpdateBuffer(m_CullingConstants, 0, sizeof(m_CullingData), &m_CullingData,
                          Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION);

    std::memcpy(m_RasterData.ViewProjMatrix, viewProjMatrix, sizeof(m_RasterData.ViewProjMatrix));
    std::memcpy(m_RasterData.CameraWorldPosition, cameraWorldPosition,
                sizeof(m_RasterData.CameraWorldPosition));
    m_RasterData.ViewportWidth = viewportWidth;
    m_RasterData.ViewportHeight = viewportHeight;
    m_RasterData.DebugMode =
        (std::getenv("NANITE_VERIFY_OCCLUSION") != nullptr && !m_CpuVerifyPrinted) ? 4u :
        (std::getenv("NANITE_VISUALIZE_HZB") != nullptr ||
         std::getenv("NANITE_VISUALIZE_DEPTH") != nullptr) ?
        (std::getenv("NANITE_DEBUG_RASTER_DEPTH") != nullptr ? 3u :
         (std::getenv("NANITE_DEBUG_CONTRAST") != nullptr ? 2u : 1u)) :
        // Shading is the default, because a per-cluster colour hides LOD popping:
        // the surface changes colour on every cluster switch whether or not the
        // geometry moved. The cluster view is still the only one that shows where
        // the LOD cut fell, so it stays reachable from the control panel.
        m_ShadingMode;
    m_RasterData.DebugMip = 0u;
    m_RasterData.RasterBinAreaCutoff = m_ClusterRasterExperiment ?
        m_ClusterRasterAreaCutoff : m_RasterBinAreaCutoff;
    m_RasterData.HybridRasterEnabled = 0u;
    m_RasterData.ClusterRasterMode = m_ClusterRasterExperiment ? 1u : 0u;
    // Visibility shading uses this bit to distinguish authored glTF UVs from
    // the generated world-position fallback used by OBJ/PLY scenes.
    m_RasterData.Padding1 = m_HasTextureCoordinates ? 1u : 0u;
    m_RasterData.Padding2 = m_HasTextureCoordinates ? 1u : 0u;
    // One shared scene-to-sun direction drives the sky disc, terrain lighting,
    // and the shadow view. Azimuth zero points along the default camera's -Z
    // direction; positive azimuth turns the sun toward +X.
    constexpr float DegreesToRadians = 3.14159265358979323846f / 180.0f;
    const float sunAzimuth = m_AtmosphereSettings.SunAzimuthDeg *
        DegreesToRadians;
    const float sunElevation = std::clamp(
        m_AtmosphereSettings.SunElevationDeg, -89.0f, 89.0f) *
        DegreesToRadians;
    const float cosElevation = std::cos(sunElevation);
    const float skySunDirection[4] = {
        std::sin(sunAzimuth) * cosElevation,
        std::sin(sunElevation),
        -std::cos(sunAzimuth) * cosElevation,
        1.0f};
    std::memcpy(m_RasterData.SkySunDirection, skySunDirection, sizeof(skySunDirection));
    // Keep the curved horizon visible at this demo's compact world scale. The
    // radius and atmosphere height are explicit so a future cascade-shadow and
    // LUT-based scattering path can replace the preview without changing the
    // view-ray reconstruction.
    const float skyAtmosphere[4] = {
        m_AtmosphereSettings.AtmosphereRadius,
        m_AtmosphereSettings.AtmosphereHeight,
        m_AtmosphereSettings.RayleighScale,
        m_AtmosphereSettings.MieScale};
    std::memcpy(m_RasterData.SkyAtmosphere, skyAtmosphere, sizeof(skyAtmosphere));
    const float skyPerspective[4] = {
        m_AtmosphereSettings.SkyDensity,
        m_AtmosphereSettings.TerrainDensity,
        m_AtmosphereSettings.HazeDistance,
        m_AtmosphereSettings.HazeWeight};
    std::memcpy(m_RasterData.SkyPerspective, skyPerspective, sizeof(skyPerspective));
    const auto shadowViewProj = BuildShadowViewProjection(
        m_CpuScene, m_ActiveInstanceCount, skySunDirection);
    std::memcpy(m_RasterData.ShadowViewProjMatrix,
                shadowViewProj.data(), sizeof(m_RasterData.ShadowViewProjMatrix));
    m_RasterData.VsmPageInfo[0] = static_cast<float>(m_VsmPageCount);
    m_RasterData.VsmPageInfo[1] = 1.0f / static_cast<float>(m_VsmPageCount);
    m_RasterData.VsmPageInfo[2] = 1.0f / static_cast<float>(VsmPageSize);
    m_RasterData.VsmPageInfo[3] = m_VsmEnabled ? 1.0f : 0.0f;
    if (const char* debugMip = std::getenv("NANITE_DEBUG_MIP"))
    {
        const unsigned long parsedMip = std::strtoul(debugMip, nullptr, 10);
        m_RasterData.DebugMip = std::min<std::uint32_t>(
            static_cast<std::uint32_t>(parsedMip), m_HzbMipCount - 1u);
    }
    context->UpdateBuffer(m_RasterConstants, 0, sizeof(m_RasterData), &m_RasterData,
                          Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
}

void GpuScene::Render(Diligent::IDeviceContext* context,
                      Diligent::ITextureView* renderTarget,
                      std::uint32_t viewportWidth,
                      std::uint32_t viewportHeight,
                      const float viewProjMatrix[16],
                      const float cameraWorldPosition[3],
                      float cameraYaw,
                      float cameraPitch,
                      const AsyncRasterContext* asyncRaster)
{
    using namespace Diligent;
    Require(m_Pipelines != nullptr, "Nanite pipelines are not initialized");
    ++m_FrameIndex;
    // Keep zero reserved for the initial/unpublished state. Task records carry
    // this value in Ready, so stale records from an earlier frame fail the
    // consumer's generation check without a bulk queue clear.
    if (++m_QueueGeneration == 0u)
        m_QueueGeneration = 1u;
    m_ProbeStatsRepeat = 1;
    m_CameraYaw = cameraYaw;
    m_CameraPitch = cameraPitch;
    if (m_InstanceUploadPending)
    {
        // Recorded before this frame's culling, so every pass below already sees
        // the new layout. Only the active prefix is written; the tail keeps its
        // stale transforms and is never referenced.
        context->UpdateBuffer(
            m_InstanceBuffer,
            0,
            static_cast<Uint64>(m_ActiveInstanceCount) * sizeof(Instance),
            m_CpuScene.Instances.data(),
            RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
        m_InstanceUploadPending = false;
    }
    EnsureRenderTargets(viewportWidth, viewportHeight);
    BeginGpuTimingFrame(context);
    UpdateAutoRasterDecision();
    BeginGpuTimingPass(context, 16u);
    ResetQueues(context);
    EndGpuTimingPass(context, 16u);
    UpdateConstants(context,
                    viewportWidth,
                    viewportHeight,
                    viewProjMatrix,
                    cameraWorldPosition);
    if (m_HzbValid)
    {
        // The previous frame produced HZB through UAV writes. Re-assert the
        // read state immediately before culling so the Vulkan backend emits a
        // UAV-to-SRV visibility barrier across frame boundaries.
        context->TransitionResourceState({
            m_HzbTexture,
            RESOURCE_STATE_UNKNOWN,
            RESOURCE_STATE_SHADER_RESOURCE,
            STATE_TRANSITION_FLAG_UPDATE_STATE});
        context->TransitionResourceState({
            m_HzbSourceTexture,
            RESOURCE_STATE_UNKNOWN,
            RESOURCE_STATE_SHADER_RESOURCE,
            STATE_TRANSITION_FLAG_UPDATE_STATE});
    }
    const bool debugVisualization =
        std::getenv("NANITE_VISUALIZE_HZB") != nullptr ||
        std::getenv("NANITE_VISUALIZE_DEPTH") != nullptr;
    // Overdraw is a third pass appended after the normal two rather than a
    // replacement for them, because the question it answers - did culling remove
    // the hidden layers - only has a meaningful answer if culling ran exactly as it
    // normally does, which means the depth buffer and the HZB built from it have to
    // be produced first.
    const bool overdrawVisualization = m_RasterData.DebugMode == 8u;
    const bool renderDocLabels = std::getenv("NANITE_RENDERDOC_LABELS") != nullptr;
    auto beginDebugGroup = [&](const char* name) {
        if (renderDocLabels)
            context->BeginDebugGroup(name);
    };
    auto endDebugGroup = [&]() {
        if (renderDocLabels)
            context->EndDebugGroup();
    };
    // Recomputes every indirect dispatch grid from the queue state as it stands
    // now. One workgroup of one thread, so it is cheaper to just rewrite all four
    // slots ahead of each consumer than to track which of them went stale.
    auto prepareDispatchArgs = [&]() {
        context->SetPipelineState(m_Pipelines->PrepareDispatch());
        context->CommitShaderResources(
            m_Pipelines->PrepareDispatchBinding(),
            RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
        context->DispatchCompute({1, 1, 1});
    };
    auto dispatchIndirect = [&](DispatchSlot slot) {
        context->DispatchComputeIndirect(DispatchComputeIndirectAttribs{
            m_DispatchArgs,
            RESOURCE_STATE_TRANSITION_MODE_TRANSITION,
            static_cast<Uint64>(slot) * sizeof(DispatchCommand)});
    };
    auto dispatchSoftIndirect = [&](Diligent::IDeviceContext* rasterContext) {
        rasterContext->DispatchComputeIndirect(DispatchComputeIndirectAttribs{
            m_DispatchArgs,
            RESOURCE_STATE_TRANSITION_MODE_TRANSITION,
            static_cast<Uint64>(DispatchSlot::SoftRaster) * sizeof(DispatchCommand)});
    };
    // Portable mode resolves depth first and repeats coverage for payload. The
    // Apple native override publishes both fields in one 64-bit atomic, so its
    // visibility sweep is intentionally skipped below.
    auto dispatchSoftDepth = [&](Diligent::IDeviceContext* rasterContext) {
        rasterContext->SetPipelineState(m_Pipelines->SoftRaster());
        rasterContext->CommitShaderResources(
            m_Pipelines->SoftRasterBinding(),
            RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
        dispatchSoftIndirect(rasterContext);
        rasterContext->TransitionResourceState({
            m_SoftDepthBuffer,
            RESOURCE_STATE_UNORDERED_ACCESS,
            RESOURCE_STATE_UNORDERED_ACCESS});
        rasterContext->TransitionResourceState({
            m_SoftVisBuffer,
            RESOURCE_STATE_UNORDERED_ACCESS,
            RESOURCE_STATE_UNORDERED_ACCESS});
    };
    auto dispatchSoftVisibility = [&](Diligent::IDeviceContext* rasterContext) {
        if (m_Native64BitVisibility)
            return;
        rasterContext->SetPipelineState(m_Pipelines->SoftVisibility());
        rasterContext->CommitShaderResources(
            m_Pipelines->SoftVisibilityBinding(),
            RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
        dispatchSoftIndirect(rasterContext);
    };
    auto dispatchSoftRaster = [&](Diligent::IDeviceContext* rasterContext) {
        dispatchSoftDepth(rasterContext);
        dispatchSoftVisibility(rasterContext);
    };
    const std::uint32_t debugMode = m_RasterData.DebugMode;
    const bool finalDepthVisualization = debugMode == 10u;
    const bool rasterBinVisualization = debugMode == 9u;
    const bool pbrShading = debugMode == 11u;
    // A zero cutoff is the explicit fast-path setting: do not pay even the
    // per-triangle setup cost of the two software passes when no triangle should
    // be routed to them. A positive cutoff from the control panel is itself an
    // explicit request to run the hybrid path; the environment override remains
    // useful for scripted runs that want serial software raster at any cutoff.
    const float activeRasterCutoff = m_ClusterRasterExperiment ?
        m_ClusterRasterAreaCutoff : m_RasterBinAreaCutoff;
    const bool hybridCandidate = m_HybridRasterAvailable &&
        !m_AutoRasterHardware &&
        activeRasterCutoff > HybridRasterCutoffEpsilon &&
        !debugVisualization &&
        std::getenv("NANITE_CPU_DRAW_DEBUG") == nullptr &&
        (debugMode == 0u || pbrShading || rasterBinVisualization || finalDepthVisualization);
    const bool asyncHybridRasterization = hybridCandidate && asyncRaster != nullptr &&
        asyncRaster->Context != nullptr && asyncRaster->BinReady != nullptr &&
        asyncRaster->SoftRasterDone != nullptr && asyncRaster->SignalValue != 0u;
    const bool allowSerialHybrid =
        std::getenv("NANITE_ALLOW_SERIAL_SOFT_RASTER") != nullptr ||
        activeRasterCutoff > HybridRasterCutoffEpsilon;
    const bool hybridRasterization = hybridCandidate &&
        (asyncHybridRasterization || allowSerialHybrid);
    m_LastAsyncRaster = asyncHybridRasterization;
    m_LastFrameStats.AutoRasterEnabled = m_AutoRasterRouting;
    m_LastFrameStats.AutoRasterHardware = m_AutoRasterRouting &&
        m_AutoRasterHardware && !hybridRasterization;
    m_AutoRasterLastHybrid = hybridRasterization;
    if (hybridRasterization)
    {
        m_CullingData.HybridRasterEnabled = 1u;
        m_CullingData.ClusterRasterMode = m_ClusterRasterExperiment ? 1u : 0u;
        m_RasterData.HybridRasterEnabled = 1u;
        m_RasterData.ClusterRasterMode = m_ClusterRasterExperiment ? 1u : 0u;
        context->UpdateBuffer(m_CullingConstants, 0, sizeof(m_CullingData), &m_CullingData,
                              RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
        context->UpdateBuffer(m_RasterConstants, 0, sizeof(m_RasterData), &m_RasterData,
                              RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
    }
    if (debugVisualization)
    {
        // The first raster pass must write the depth buffer without sampling
        // that same resource. Debug visualization is restored for the second
        // pass after HZB construction.
        m_RasterData.DebugMode = 0u;
        context->UpdateBuffer(m_RasterConstants, 0, sizeof(m_RasterData), &m_RasterData,
                              RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
    }

    beginDebugGroup("Nanite / Instance Cull");
    BeginGpuTimingPass(context, 1u);
    context->SetPipelineState(m_Pipelines->InstanceCull());
    context->CommitShaderResources(
        m_Pipelines->InstanceCullBinding(),
        RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
    context->DispatchCompute({
        (m_ActiveInstanceCount + 64u - 1u) / 64u,
        1,
        1});
    EndGpuTimingPass(context, 1u);
    endDebugGroup();

    beginDebugGroup("Nanite / Persistent Cull");
    BeginGpuTimingPass(context, 2u);
    context->SetPipelineState(m_Pipelines->PersistentCull());
    context->CommitShaderResources(
        m_Pipelines->PersistentCullBinding(),
        RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
    context->DispatchCompute({m_Capacity.PersistentWorkgroups, 1, 1});
    EndGpuTimingPass(context, 2u);
    endDebugGroup();

    // VisibleClusterCount, PostNodeCount and PostGroupCount are final now that
    // the traversal has drained, which is what the grids below are sized from.
    beginDebugGroup("Nanite / Prepare Dispatch");
    BeginGpuTimingPass(context, 17u);
    prepareDispatchArgs();
    EndGpuTimingPass(context, 17u);
    endDebugGroup();

    beginDebugGroup("Nanite / Generate Indirect");
    BeginGpuTimingPass(context, 3u);
    context->SetPipelineState(m_Pipelines->GenerateIndirect());
    context->CommitShaderResources(
        m_Pipelines->GenerateIndirectBinding(),
        RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
    dispatchIndirect(DispatchSlot::GenerateIndirect);
    EndGpuTimingPass(context, 3u);
    endDebugGroup();

    // PostClusterCount is only final after generate-indirect, and the post cull
    // grid depends on it whether or not the recovery traversal below runs.
    beginDebugGroup("Nanite / Prepare Dispatch Post");
    BeginGpuTimingPass(context, 18u);
    prepareDispatchArgs();
    EndGpuTimingPass(context, 18u);
    endDebugGroup();

    if (hybridRasterization)
    {
        beginDebugGroup("Nanite / Raster Bin");
        BeginGpuTimingPass(context, 11u);
        context->SetPipelineState(m_Pipelines->RasterBin());
        context->CommitShaderResources(
            m_Pipelines->RasterBinBinding(),
            RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
        dispatchIndirect(DispatchSlot::RasterBin);
        EndGpuTimingPass(context, 11u);

        // Publish SoftClusterCount before rebuilding the software dispatch grid.
        context->TransitionResourceState({
            m_QueueState,
            RESOURCE_STATE_UNORDERED_ACCESS,
            RESOURCE_STATE_UNORDERED_ACCESS});
        beginDebugGroup("Nanite / Prepare Soft Dispatch");
        BeginGpuTimingPass(context, 19u);
        prepareDispatchArgs();
        EndGpuTimingPass(context, 19u);
        endDebugGroup();
        endDebugGroup();

        if (asyncHybridRasterization)
        {
            // Submit the producer command buffer now. The async queue's wait is
            // therefore a real GPU dependency rather than a CPU-side ordering
            // accident, and the graphics queue can move on to hardware raster.
            context->EnqueueSignal(asyncRaster->BinReady, asyncRaster->SignalValue);
            context->Flush();

            Diligent::IDeviceContext* asyncContext = asyncRaster->Context;
            asyncContext->DeviceWaitForFence(asyncRaster->BinReady, asyncRaster->SignalValue);
            beginDebugGroup("Nanite / Async Soft Raster");
            if (m_GpuTimingFrameActive)
                BeginGpuTimingPass(asyncContext, 8u);
            dispatchSoftDepth(asyncContext);
            if (m_GpuTimingFrameActive)
                EndGpuTimingPass(asyncContext, 8u);
            if (!m_Native64BitVisibility)
            {
                if (m_GpuTimingFrameActive)
                    BeginGpuTimingPass(asyncContext, 15u);
                dispatchSoftVisibility(asyncContext);
                if (m_GpuTimingFrameActive)
                    EndGpuTimingPass(asyncContext, 15u);
            }
            endDebugGroup();
            asyncContext->EnqueueSignal(asyncRaster->SoftRasterDone, asyncRaster->SignalValue);
            asyncContext->Flush();
        }
    }

    const bool dumpGroupTasks = std::getenv("NANITE_DUMP_QUEUE_STATE") != nullptr &&
        !m_DebugQueuePrinted && m_FrameIndex >= m_DebugQueueReadbackFrame;
    // Sampling every 60th frame is enough for a stable HUD but useless for
    // chasing a flicker, which lives in the frames between samples.
    static const std::uint32_t statsInterval = []() -> std::uint32_t {
        if (const char* interval = std::getenv("NANITE_STATS_INTERVAL"))
        {
            const unsigned long value = std::strtoul(interval, nullptr, 10);
            if (value > 0ul)
                return static_cast<std::uint32_t>(value);
        }
        return 0u;
    }();
    if (statsInterval != 0u)
        m_StatsReadbackInterval = statsInterval;
    const bool sampleStats = (m_FrameIndex % m_StatsReadbackInterval) == 0;

    ITextureView* renderTargets[] = {renderTarget, m_DepthColorRTV, m_HardwareVisibilityRTV};
    context->SetRenderTargets(3, renderTargets, m_DepthDSV, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
    context->ClearDepthStencil(
        m_DepthDSV,
        Diligent::CLEAR_DEPTH_FLAG,
        1.0f,
        0,
        Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
    const float clearDepthSource[] = {1.0f, 1.0f, 1.0f, 1.0f};
    context->ClearRenderTarget(
        m_DepthColorRTV,
        clearDepthSource,
        Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
    const float clearVisibility[] = {0.0f, 0.0f, 0.0f, 0.0f};
    context->ClearRenderTarget(
        m_HardwareVisibilityRTV,
        clearVisibility,
        Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
    // The visualization only changes the pixel output; use the regular opaque
    // raster state and bindings so it cannot introduce a second draw-order path.
    context->SetPipelineState(m_Pipelines->Raster());
    context->CommitShaderResources(
        m_Pipelines->RasterBinding(),
        RESOURCE_STATE_TRANSITION_MODE_TRANSITION);

    if (std::getenv("NANITE_CPU_DRAW_DEBUG") != nullptr)
    {
        // Bypass culling by writing the command and its instance data from the
        // CPU: every cluster of instance 0, in cluster order.
        const std::uint32_t debugDrawCount = std::min<std::uint32_t>(
            m_Capacity.MaxDrawCommands,
            static_cast<std::uint32_t>(m_CpuScene.Clusters.size()));
        m_DebugDrawInstanceData.resize(debugDrawCount);
        for (std::uint32_t clusterIndex = 0; clusterIndex < debugDrawCount; ++clusterIndex)
        {
            const Cluster& cluster = m_CpuScene.Clusters[clusterIndex];
            m_DebugDrawInstanceData[clusterIndex] = {
                0u,
                clusterIndex,
                cluster.FirstIndex,
                cluster.IndexCount};
        }
        const DrawCommand debugCommand{m_MaxClusterIndexCount, debugDrawCount, 0u, 0u};
        context->UpdateBuffer(
            m_DrawCommands,
            0,
            sizeof(debugCommand),
            &debugCommand,
            RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
        context->UpdateBuffer(
            m_DrawInstanceIndices,
            0,
            m_DebugDrawInstanceData.size() * sizeof(DrawInstanceData),
            m_DebugDrawInstanceData.data(),
            RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
    }

    beginDebugGroup("Nanite / Raster");
    BeginGpuTimingPass(context, 5u);
    // One command, one instance per hardware cluster. Hybrid mode uses the
    // compacted command written by RasterBin; the baseline uses the canonical
    // visible-cluster command. DrawCount remains one in both cases.
    context->DrawIndirect(DrawIndirectAttribs{
        hybridRasterization ? m_HybridDrawCommand.RawPtr() : m_DrawCommands.RawPtr(),
        DRAW_FLAG_VERIFY_ALL,
        1,
        0,
        sizeof(DrawCommand),
        RESOURCE_STATE_TRANSITION_MODE_TRANSITION});
    EndGpuTimingPass(context, 5u);
    endDebugGroup();

    // End the depth attachment before reading it from the HZB compute pass.
    // Keeping the DSV bound can prevent the backend transition from depth
    // write to shader read from taking effect.
    context->SetRenderTargets(0, nullptr, nullptr, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
    if (asyncHybridRasterization)
    {
        // Flush the graphics raster command buffer before waiting for compute.
        // This is the point at which the two queues are allowed to overlap.
        context->Flush();
    }
    if (hybridRasterization)
    {
        beginDebugGroup("Nanite / Soft Raster Visibility");
        if (!asyncHybridRasterization)
        {
            BeginGpuTimingPass(context, 8u);
            dispatchSoftDepth(context);
            EndGpuTimingPass(context, 8u);
            if (!m_Native64BitVisibility)
            {
                BeginGpuTimingPass(context, 15u);
                dispatchSoftVisibility(context);
                EndGpuTimingPass(context, 15u);
            }
        }
        else
        {
            // Resolve is the first graphics operation that consumes software
            // visibility. The wait is recorded on the graphics queue, so it does
            // not serialize hardware raster with the async compute work.
            context->DeviceWaitForFence(asyncRaster->SoftRasterDone, asyncRaster->SignalValue);
            context->TransitionResourceState({
                m_SoftDepthBuffer,
                RESOURCE_STATE_UNKNOWN,
                RESOURCE_STATE_UNORDERED_ACCESS,
                STATE_TRANSITION_FLAG_UPDATE_STATE});
            context->TransitionResourceState({
                m_SoftVisBuffer,
                RESOURCE_STATE_UNKNOWN,
                RESOURCE_STATE_SHADER_RESOURCE,
                STATE_TRANSITION_FLAG_UPDATE_STATE});
            context->TransitionResourceState({
                m_QueueState,
                RESOURCE_STATE_UNKNOWN,
                RESOURCE_STATE_UNORDERED_ACCESS,
                STATE_TRANSITION_FLAG_UPDATE_STATE});
        }
        endDebugGroup();

        beginDebugGroup("Nanite / Visibility Resolve");
        BeginGpuTimingPass(context, 12u);
        ResolveVisibility(context, viewportWidth, viewportHeight);
        EndGpuTimingPass(context, 12u);
        endDebugGroup();
        BuildHZB(context, viewportWidth, viewportHeight, m_FinalDepthTexture, true);
    }
    else
    {
        BuildHZB(context, viewportWidth, viewportHeight, m_DepthColorTexture, true);
    }

    // The software-raster probe. Placed here for two reasons: the hardware depth
    // for exactly these clusters now exists, so the check pass has something to
    // compare against, and the draw count is still the main phase's, so the two
    // rasterizers are priced on the same work. Nothing downstream reads the
    // result - this pass answers whether a visibility-buffer rewrite is worth
    // starting, and is deleted or promoted once it has.
    if (!hybridRasterization && m_SoftProbeEnabled && m_SoftDepthBuffer != nullptr)
    {
        m_ProbeStatsRepeat = std::max(m_ProbeRepeat, 1u);
        if (m_GpuTimingFrameActive && m_ActiveGpuTimingSlot >= 0)
        {
            m_GpuTimingSlots[static_cast<std::size_t>(m_ActiveGpuTimingSlot)].ProbeRepeat =
                std::max(m_ProbeRepeat, 1u);
        }
        beginDebugGroup("Nanite / Soft Raster Depth");
        BeginGpuTimingPass(context, 8u);
        for (std::uint32_t repeat = 0; repeat < m_ProbeRepeat; ++repeat)
            dispatchSoftDepth(context);
        EndGpuTimingPass(context, 8u);
        endDebugGroup();

        if (!m_Native64BitVisibility)
        {
            beginDebugGroup("Nanite / Soft Raster Visibility Payload");
            BeginGpuTimingPass(context, 15u);
            for (std::uint32_t repeat = 0; repeat < m_ProbeRepeat; ++repeat)
                dispatchSoftVisibility(context);
            EndGpuTimingPass(context, 15u);
            endDebugGroup();
        }

        // Timed separately: the comparison and the buffer reset are the probe's
        // own overhead, not part of what a real software rasterizer would cost.
        beginDebugGroup("Nanite / Soft Depth Check");
        BeginGpuTimingPass(context, 9u);
        context->SetPipelineState(m_Pipelines->SoftCheck());
        context->CommitShaderResources(
            m_Pipelines->SoftCheckBinding(),
            RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
        context->DispatchCompute({
            (viewportWidth + 7u) / 8u,
            (viewportHeight + 7u) / 8u,
            1});
        EndGpuTimingPass(context, 9u);
        endDebugGroup();

        // The fair denominator. Same clusters, same indirect command, hardware
        // rasterizer, depth only: no shading, no colour attachment, no depth copy
        // for the HZB.
        //
        // Both this and the software pass are issued m_ProbeRepeat times inside one
        // timing bracket, because a single bracket is not trustworthy here: MoltenVK
        // cannot place a timestamp mid-encoder, so the interval also swallows the
        // render-pass setup and the wait for the compute passes ahead of it to
        // drain. Measured at repeat 1, 2, 4 and 8, the slope is the per-pass cost
        // and the intercept is that scaffolding. Cleared once outside the bracket:
        // repeats after the first see a populated buffer, which for ~1 px triangles
        // with no pixel shader changes almost nothing.
        context->ClearDepthStencil(
            m_SoftProbeDepthDSV,
            CLEAR_DEPTH_FLAG,
            1.0f,
            0,
            RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
        context->SetRenderTargets(0, nullptr, m_SoftProbeDepthDSV,
                                  RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
        beginDebugGroup("Nanite / Hardware Depth Only");
        BeginGpuTimingPass(context, 10u);
        context->SetPipelineState(m_Pipelines->HardwareDepthOnly());
        context->CommitShaderResources(
            m_Pipelines->HardwareDepthOnlyBinding(),
            RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
        for (std::uint32_t repeat = 0; repeat < m_ProbeRepeat; ++repeat)
        {
            context->DrawIndirect(DrawIndirectAttribs{
                m_DrawCommands,
                DRAW_FLAG_VERIFY_ALL,
                1,
                0,
                sizeof(DrawCommand),
                RESOURCE_STATE_TRANSITION_MODE_TRANSITION});
        }
        EndGpuTimingPass(context, 10u);
        endDebugGroup();
        context->SetRenderTargets(0, nullptr, nullptr, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
    }

    // The HZB now holds current-frame depth, so post cull can test against it
    // even on the first frame, when EnableHZB was still zero during main cull.
    const std::uint32_t postCullEnableHZB =
        (m_HZBEnabled && std::getenv("NANITE_DISABLE_HZB") == nullptr) ? 1u : 0u;
    m_CullingData.EnableHZB = postCullEnableHZB;
    // Everything from here on queries the pyramid built from this frame's depth,
    // so bounds must be tested with the current clip rectangle rather than
    // reprojected into the previous view.
    m_CullingData.CullingPass = 1u;
    context->UpdateBuffer(m_CullingConstants, 0, sizeof(m_CullingData), &m_CullingData,
                          RESOURCE_STATE_TRANSITION_MODE_TRANSITION);

    // Occlusion recovery. The main phase pruned instances, nodes and groups
    // using the previous frame's HZB, which is a guess; re-run the same passes
    // over just the pruned work against the pyramid built from this frame's
    // depth, so a wrong guess costs a little extra traversal instead of a hole
    // in the image. Skipped when the HZB is off, and also when neither instance
    // nor node level HZB is active, because then nothing upstream of the cluster
    // level can have been rejected on occlusion and every recovery queue is
    // empty - the cluster level has its own post pass further down.
    const bool hierarchyOrInstanceHzbActive =
        (m_CullingData.MaxRefinementDepth & 0x30000000u) != 0x30000000u;
    if (postCullEnableHZB != 0u && hierarchyOrInstanceHzbActive)
    {
        BeginGpuTimingPass(context, 20u);
        beginDebugGroup("Nanite / Post Seed");
        context->SetPipelineState(m_Pipelines->PostSeed());
        context->CommitShaderResources(
            m_Pipelines->PostSeedBinding(),
            RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
        dispatchIndirect(DispatchSlot::PostSeed);
        endDebugGroup();

        beginDebugGroup("Nanite / Post Instance Cull");
        context->SetPipelineState(m_Pipelines->InstanceCull());
        context->CommitShaderResources(
            m_Pipelines->InstanceCullBinding(),
            RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
        context->DispatchCompute({
            (m_ActiveInstanceCount + 64u - 1u) / 64u,
            1,
            1});
        endDebugGroup();

        beginDebugGroup("Nanite / Post Persistent Cull");
        context->SetPipelineState(m_Pipelines->PersistentCull());
        context->CommitShaderResources(
            m_Pipelines->PersistentCullBinding(),
            RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
        context->DispatchCompute({m_Capacity.PersistentWorkgroups, 1, 1});
        endDebugGroup();

        // The recovery traversal grew VisibleClusterCount and post seed wrote the
        // mark, so the tail that post generate-indirect has to walk is now known.
        beginDebugGroup("Nanite / Prepare Dispatch Post Tail");
        prepareDispatchArgs();
        endDebugGroup();

        beginDebugGroup("Nanite / Post Generate Indirect");
        context->SetPipelineState(m_Pipelines->GenerateIndirect());
        context->CommitShaderResources(
            m_Pipelines->GenerateIndirectBinding(),
            RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
        dispatchIndirect(DispatchSlot::PostGenerateIndirect);
        endDebugGroup();

        // Post generate-indirect appended to the recovery cluster queue, so the
        // post cull grid computed before the recovery phase is now too small.
        beginDebugGroup("Nanite / Prepare Dispatch Post Cull");
        prepareDispatchArgs();
        endDebugGroup();
        EndGpuTimingPass(context, 20u);
    }

    beginDebugGroup("Nanite / Post Cull");
    BeginGpuTimingPass(context, 4u);
    context->SetPipelineState(m_Pipelines->PostCull());
    context->CommitShaderResources(
        m_Pipelines->PostCullBinding(),
        RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
    dispatchIndirect(DispatchSlot::PostCull);
    EndGpuTimingPass(context, 4u);
    endDebugGroup();

    if (!debugVisualization)
    {
        if (hybridRasterization)
        {
            // Recovery draw lists were produced after the software pass. Let
            // hardware rasterize their complete range; the final resolve still
            // compares them with the main pass's software winners.
            m_RasterData.HybridRasterEnabled = 0u;
            context->UpdateBuffer(m_RasterConstants, 0, sizeof(m_RasterData), &m_RasterData,
                                  RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
        }
        // Fill the holes the stale HZB caused the main phase to skip. The strict
        // LESS depth test keeps this from disturbing anything already rasterized
        // where the two overlap.
        context->SetRenderTargets(
            3,
            renderTargets,
            m_DepthDSV,
            RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
        context->SetPipelineState(m_Pipelines->PostRaster());
        context->CommitShaderResources(
            m_Pipelines->PostRasterBinding(),
            RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
        beginDebugGroup("Nanite / Post Raster");
        BeginGpuTimingPass(context, 6u);
        // The post command covers only the clusters post cull recovered. Drawing
        // the main command here instead would re-transform and re-rasterize every
        // cluster the main phase already drew, all of which then fail the depth
        // test - on a static frame nothing is recovered and that is the whole pass
        // spent for no pixels.
        context->DrawIndirect(DrawIndirectAttribs{
            m_DrawCommands,
            DRAW_FLAG_VERIFY_ALL,
            1,
            static_cast<Uint64>(DrawCommandSlot::Post) * sizeof(DrawCommand),
            sizeof(DrawCommand),
            RESOURCE_STATE_TRANSITION_MODE_TRANSITION});
        EndGpuTimingPass(context, 6u);
        endDebugGroup();
    }
    else
    {
        // Keep the query pair complete while the debug visualization replaces
        // the normal post-raster output below.
        BeginGpuTimingPass(context, 6u);
        EndGpuTimingPass(context, 6u);
    }
    if (hybridRasterization)
    {
        context->SetRenderTargets(0, nullptr, nullptr, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
        beginDebugGroup("Nanite / Final Visibility Resolve");
        BeginGpuTimingPass(context, 13u);
        ResolveVisibility(context, viewportWidth, viewportHeight);
        EndGpuTimingPass(context, 13u);
        BuildHZB(context, viewportWidth, viewportHeight, m_FinalDepthTexture, false);
        endDebugGroup();

        // The old probe checker doubles as the GPU clear. In hybrid mode it exits
        // before its legacy full-frame comparison because the two bins are meant
        // to be disjoint, then restores the buffers for the next frame.
        context->SetPipelineState(m_Pipelines->SoftCheck());
        context->CommitShaderResources(
            m_Pipelines->SoftCheckBinding(),
            RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
        context->DispatchCompute({
            (viewportWidth + 7u) / 8u,
            (viewportHeight + 7u) / 8u,
            1});
    }

    // Snapshot after the final resolve and hybrid cleanup. The software counters
    // are written on the async queue and the cleanup pass clears only the scratch
    // buffers when HybridRasterEnabled is set; taking this copy earlier produces
    // the misleading zeroes seen for soft visibility resolved/mismatch.
    if (dumpGroupTasks || sampleStats)
    {
        const bool statsReady = ReadbackStats(context, dumpGroupTasks);
        if (dumpGroupTasks && statsReady)
            m_DebugQueuePrinted = true;
    }

    if (!m_VsmEnabled && !m_ShadowMapValid)
    {
        beginDebugGroup("Nanite / Shadow Map (cache miss)");
        RenderShadowMap(context);
        endDebugGroup();
    }

    if (debugVisualization)
    {
        // Visualize the HZB produced this frame after its UAV writes are complete.
        m_RasterData.DebugMode = debugMode;
        context->UpdateBuffer(m_RasterConstants, 0, sizeof(m_RasterData), &m_RasterData,
                              RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
        context->SetRenderTargets(3, renderTargets, nullptr, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
        context->SetPipelineState(m_Pipelines->Raster());
        context->CommitShaderResources(
            m_Pipelines->RasterBinding(),
            RESOURCE_STATE_TRANSITION_MODE_TRANSITION);

        beginDebugGroup("Nanite / Debug Raster");
        context->DrawIndirect(DrawIndirectAttribs{
            m_DrawCommands,
            DRAW_FLAG_VERIFY_ALL,
            1,
            0,
            sizeof(DrawCommand),
            RESOURCE_STATE_TRANSITION_MODE_TRANSITION});
        endDebugGroup();
    }

    if (overdrawVisualization)
    {
        // Redraw everything the two passes above submitted, with no depth
        // attachment bound and the colour target accumulating, so each pixel ends
        // up holding a count of the fragments that reached it. The shaded image is
        // discarded first: this mode replaces it rather than overlaying on it.
        //
        // What the count measures is one pass' worth of work. The frame runs the
        // range twice - main raster and post raster - so the fragments the hardware
        // actually shades are about double what is on screen here.
        context->SetRenderTargets(
            3, renderTargets, nullptr, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
        const float clearBlack[] = {0.0f, 0.0f, 0.0f, 1.0f};
        context->ClearRenderTarget(
            renderTarget, clearBlack, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
        context->SetPipelineState(m_Pipelines->OverdrawRaster());
        context->CommitShaderResources(
            m_Pipelines->OverdrawRasterBinding(),
            RESOURCE_STATE_TRANSITION_MODE_TRANSITION);

        beginDebugGroup("Nanite / Overdraw");
        context->DrawIndirect(DrawIndirectAttribs{
            m_DrawCommands,
            DRAW_FLAG_VERIFY_ALL,
            1,
            0,
            sizeof(DrawCommand),
            RESOURCE_STATE_TRANSITION_MODE_TRANSITION});
        endDebugGroup();
    }

    const bool visibilityOnlyShading = !debugVisualization &&
        (debugMode == 0u || pbrShading || hybridRasterization || finalDepthVisualization);
    if (visibilityOnlyShading)
    {
        const bool useResolvedVisibility = hybridRasterization;
        BeginGpuTimingPass(context, 14u);
        ShadeVisibility(
            context,
            renderTarget,
            useResolvedVisibility ? m_FinalDepthSRV.RawPtr() : m_DepthColorSRV.RawPtr(),
            useResolvedVisibility ? m_FinalVisibilitySRV.RawPtr() : m_HardwareVisibilitySRV.RawPtr());
        EndGpuTimingPass(context, 14u);
    }

    EndGpuTimingFrame(context);
}

} // namespace Nanite
