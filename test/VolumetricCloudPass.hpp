#pragma once

#include <DeviceContext.h>
#include <PipelineState.h>
#include <RefCntAutoPtr.hpp>
#include <RenderDevice.h>
#include <Texture.h>

#include <cstdint>
#include <string>

namespace VolumetricCloudTest
{

// This layout mirrors VolumetricCloudConstants in volumetric_cloud.frag.hlsl.
// Keep every member in float4/uint4-sized groups for Vulkan constant-buffer
// packing and use this structure directly with IDeviceContext::UpdateBuffer.
struct alignas(16) VolumetricCloudConstants
{
    float InvViewProj[16]{};
    float CameraWorldPosition[4]{};
    float SunDirection[4]{0.45f, 0.8f, 0.2f, 0.0f};
    float SunRadiance[4]{8.0f, 7.3f, 6.4f, 0.0f};
    float CloudBoundsMin[4]{-500.0f, 120.0f, -500.0f, 0.0f};
    float CloudBoundsMax[4]{500.0f, 360.0f, 500.0f, 0.0f};
    // x coverage, y density, z layer bottom, w layer top.
    float CloudLayer[4]{0.58f, 1.0f, 120.0f, 360.0f};
    // x max distance, y view step, z light step, w Henyey-Greenstein g.
    float MarchParameters[4]{2000.0f, 8.0f, 16.0f, 0.35f};
    // x map wind scale, y detail world scale, z erosion strength, w reserved.
    float NoiseParameters[4]{0.0015f, 95.0f, 0.72f, 0.0f};
    // xyz wind direction in world units per second, w elapsed time.
    float WindTime[4]{8.0f, 0.0f, 3.0f, 0.0f};
    // x extinction, y single-scatter albedo, z D in the f_ms estimate, w MS strength.
    float ScatteringParameters[4]{0.035f, 0.92f, 420.0f, 1.0f};
    std::uint32_t Viewport[4]{1280u, 720u, 4u, 0u};
};

static_assert(sizeof(VolumetricCloudConstants) % 16 == 0,
              "Volumetric cloud constants must be 16-byte aligned");

struct alignas(16) CloudBakeConstants
{
    std::uint32_t Width = 512;
    std::uint32_t Height = 512;
    std::uint32_t Depth = 128;
    std::uint32_t Seed = 1;
    float Coverage = 0.58f;
    float Frequency = 4.5f;
    float Offset[2]{0.0f, 0.0f};
    float Padding[2]{0.0f, 0.0f};
};

static_assert(sizeof(CloudBakeConstants) % 16 == 0,
              "Cloud bake constants must be 16-byte aligned");

struct CloudTextures
{
    Diligent::RefCntAutoPtr<Diligent::ITexture> Distribution;
    Diligent::RefCntAutoPtr<Diligent::ITexture> Detail;
};

class VolumetricCloudPass
{
public:
    VolumetricCloudPass(Diligent::IRenderDevice* device,
                        Diligent::TEXTURE_FORMAT colorFormat,
                        const std::string& shaderDirectory,
                        Diligent::Uint64 immediateContextMask = 1);

    CloudTextures CreateTextures(Diligent::Uint32 distributionWidth = 512,
                                  Diligent::Uint32 distributionHeight = 512,
                                  Diligent::Uint32 detailWidth = 128,
                                  Diligent::Uint32 detailHeight = 128,
                                  Diligent::Uint32 detailDepth = 128) const;

    Diligent::RefCntAutoPtr<Diligent::IBuffer> CreateCloudConstantBuffer() const;

    void BakeDistribution(Diligent::IDeviceContext* context,
                          const CloudTextures& textures,
                          const CloudBakeConstants& constants);

    void BakeDetail(Diligent::IDeviceContext* context,
                    const CloudTextures& textures,
                    const CloudBakeConstants& constants);

    // The target must be the current color target. The depth SRV is the opaque
    // scene depth in Vulkan [0, 1]; use a cleared 1.0 texture when no occluder
    // depth is available. The PSO writes premultiplied cloud radiance.
    void Render(Diligent::IDeviceContext* context,
                const CloudTextures& textures,
                Diligent::ITextureView* renderTarget,
                Diligent::ITextureView* sceneDepth,
                Diligent::IBuffer* cloudConstantBuffer,
                const VolumetricCloudConstants& constants);

    Diligent::IPipelineState* DistributionPipeline() const { return m_DistributionPSO; }
    Diligent::IPipelineState* DetailPipeline() const { return m_DetailPSO; }
    Diligent::IPipelineState* RenderPipeline() const { return m_RenderPSO; }

private:
    Diligent::RefCntAutoPtr<Diligent::IShader> LoadShader(
        Diligent::IRenderDevice* device,
        const std::string& path,
        Diligent::SHADER_TYPE type,
        const char* name,
        const char* entryPoint) const;

    Diligent::RefCntAutoPtr<Diligent::IPipelineState> CreateComputePipeline(
        Diligent::IRenderDevice* device,
        Diligent::IShader* shader,
        const char* name) const;

    static void SetTexture(Diligent::IShaderResourceBinding* binding,
                           Diligent::SHADER_TYPE stage,
                           const char* name,
                           Diligent::ITextureView* view);
    static void SetSampler(Diligent::IShaderResourceBinding* binding,
                           Diligent::SHADER_TYPE stage,
                           const char* name,
                           Diligent::ISampler* sampler);
    static void SetConstantBuffer(Diligent::IShaderResourceBinding* binding,
                                  Diligent::SHADER_TYPE stage,
                                  const char* name,
                                  Diligent::IBuffer* buffer);

    Diligent::Uint64 m_ImmediateContextMask = 1;
    Diligent::RefCntAutoPtr<Diligent::IRenderDevice> m_Device;
    Diligent::RefCntAutoPtr<Diligent::IBuffer> m_BakeCB;
    Diligent::RefCntAutoPtr<Diligent::IShader> m_DistributionShader;
    Diligent::RefCntAutoPtr<Diligent::IShader> m_DetailShader;
    Diligent::RefCntAutoPtr<Diligent::IShader> m_VertexShader;
    Diligent::RefCntAutoPtr<Diligent::IShader> m_PixelShader;
    Diligent::RefCntAutoPtr<Diligent::IPipelineState> m_DistributionPSO;
    Diligent::RefCntAutoPtr<Diligent::IPipelineState> m_DetailPSO;
    Diligent::RefCntAutoPtr<Diligent::IPipelineState> m_RenderPSO;
    Diligent::RefCntAutoPtr<Diligent::IShaderResourceBinding> m_DistributionSRB;
    Diligent::RefCntAutoPtr<Diligent::IShaderResourceBinding> m_DetailSRB;
    Diligent::RefCntAutoPtr<Diligent::IShaderResourceBinding> m_RenderSRB;
    Diligent::RefCntAutoPtr<Diligent::ISampler> m_LinearSampler;
    Diligent::RefCntAutoPtr<Diligent::ISampler> m_PointSampler;
};

} // namespace VolumetricCloudTest
