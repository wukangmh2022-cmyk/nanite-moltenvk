#include "VolumetricCloudPass.hpp"

#include <DeviceObject.h>
#include <Sampler.h>
#include <Shader.h>

#include <algorithm>
#include <fstream>
#include <stdexcept>
#include <vector>

namespace VolumetricCloudTest
{
namespace
{

std::vector<std::uint8_t> ReadBinaryFile(const std::string& path)
{
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input)
        throw std::runtime_error{"Unable to open cloud shader: " + path};

    const std::streamsize size = input.tellg();
    if (size <= 0)
        throw std::runtime_error{"Cloud shader is empty: " + path};

    std::vector<std::uint8_t> data(static_cast<size_t>(size));
    input.seekg(0, std::ios::beg);
    if (!input.read(reinterpret_cast<char*>(data.data()), size))
        throw std::runtime_error{"Unable to read cloud shader: " + path};
    return data;
}

void Require(bool condition, const char* message)
{
    if (!condition)
        throw std::runtime_error{message};
}

Diligent::RefCntAutoPtr<Diligent::ITexture> CreateStorageTexture(
    Diligent::IRenderDevice* device,
    const char* name,
    Diligent::RESOURCE_DIMENSION type,
    Diligent::Uint32 width,
    Diligent::Uint32 height,
    Diligent::Uint32 depth,
    Diligent::Uint64 immediateContextMask)
{
    Diligent::TextureDesc desc;
    desc.Name = name;
    desc.Type = type;
    desc.Width = width;
    desc.Height = height;
    desc.Depth = depth;
    desc.Format = Diligent::TEX_FORMAT_RGBA16_FLOAT;
    desc.MipLevels = 1;
    desc.SampleCount = 1;
    desc.Usage = Diligent::USAGE_DEFAULT;
    desc.BindFlags = Diligent::BIND_SHADER_RESOURCE |
                     Diligent::BIND_UNORDERED_ACCESS;
    desc.ImmediateContextMask = immediateContextMask;

    Diligent::RefCntAutoPtr<Diligent::ITexture> texture;
    device->CreateTexture(desc, nullptr, &texture);
    Require(texture != nullptr, "Failed to create cloud storage texture");
    return texture;
}

} // namespace

VolumetricCloudPass::VolumetricCloudPass(
    Diligent::IRenderDevice* device,
    Diligent::TEXTURE_FORMAT colorFormat,
    const std::string& shaderDirectory,
    Diligent::Uint64 immediateContextMask) :
    m_ImmediateContextMask{immediateContextMask},
    m_Device{device}
{
    m_DistributionShader = LoadShader(
        device, shaderDirectory + "/CloudDistribution.comp.spv",
        Diligent::SHADER_TYPE_COMPUTE, "Cloud distribution CS", "main");
    m_DetailShader = LoadShader(
        device, shaderDirectory + "/CloudDetail3D.comp.spv",
        Diligent::SHADER_TYPE_COMPUTE, "Cloud detail CS", "main");
    m_VertexShader = LoadShader(
        device, shaderDirectory + "/VolumetricCloud.vert.spv",
        Diligent::SHADER_TYPE_VERTEX, "Volumetric cloud VS", "FullscreenVS");
    m_PixelShader = LoadShader(
        device, shaderDirectory + "/VolumetricCloud.frag.spv",
        Diligent::SHADER_TYPE_PIXEL, "Volumetric cloud PS", "CloudPS");

    m_DistributionPSO = CreateComputePipeline(
        device, m_DistributionShader, "Cloud distribution bake");
    m_DetailPSO = CreateComputePipeline(
        device, m_DetailShader, "Cloud detail bake");

    Diligent::GraphicsPipelineStateCreateInfo graphicsCI{"Volumetric cloud render"};
    graphicsCI.pVS = m_VertexShader;
    graphicsCI.pPS = m_PixelShader;
    graphicsCI.PSODesc.ImmediateContextMask = m_ImmediateContextMask;
    graphicsCI.PSODesc.ResourceLayout.DefaultVariableType =
        Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE;
    graphicsCI.GraphicsPipeline.PrimitiveTopology =
        Diligent::PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    graphicsCI.GraphicsPipeline.RasterizerDesc.CullMode = Diligent::CULL_MODE_NONE;
    graphicsCI.GraphicsPipeline.DepthStencilDesc.DepthEnable = Diligent::False;
    graphicsCI.GraphicsPipeline.NumRenderTargets = 1;
    graphicsCI.GraphicsPipeline.RTVFormats[0] = colorFormat;

    // The pixel shader returns premultiplied cloud radiance and alpha equal to
    // one minus view transmittance, so this composites over an existing sky.
    auto& blend = graphicsCI.GraphicsPipeline.BlendDesc;
    blend.RenderTargets[0].BlendEnable = Diligent::True;
    blend.RenderTargets[0].SrcBlend = Diligent::BLEND_FACTOR_ONE;
    blend.RenderTargets[0].DestBlend = Diligent::BLEND_FACTOR_INV_SRC_ALPHA;
    blend.RenderTargets[0].BlendOp = Diligent::BLEND_OPERATION_ADD;
    blend.RenderTargets[0].SrcBlendAlpha = Diligent::BLEND_FACTOR_ONE;
    blend.RenderTargets[0].DestBlendAlpha = Diligent::BLEND_FACTOR_INV_SRC_ALPHA;
    blend.RenderTargets[0].BlendOpAlpha = Diligent::BLEND_OPERATION_ADD;

    device->CreateGraphicsPipelineState(graphicsCI, &m_RenderPSO);
    Require(m_DistributionPSO != nullptr && m_DetailPSO != nullptr &&
                m_RenderPSO != nullptr,
            "Failed to create volumetric cloud PSOs");

    m_DistributionPSO->CreateShaderResourceBinding(&m_DistributionSRB, true);
    m_DetailPSO->CreateShaderResourceBinding(&m_DetailSRB, true);
    m_RenderPSO->CreateShaderResourceBinding(&m_RenderSRB, true);
    Require(m_DistributionSRB != nullptr && m_DetailSRB != nullptr &&
                m_RenderSRB != nullptr,
            "Failed to create volumetric cloud SRBs");

    Diligent::SamplerDesc linearDesc;
    linearDesc.Name = "Cloud linear sampler";
    linearDesc.MinFilter = Diligent::FILTER_TYPE_LINEAR;
    linearDesc.MagFilter = Diligent::FILTER_TYPE_LINEAR;
    linearDesc.MipFilter = Diligent::FILTER_TYPE_LINEAR;
    linearDesc.AddressU = Diligent::TEXTURE_ADDRESS_CLAMP;
    linearDesc.AddressV = Diligent::TEXTURE_ADDRESS_CLAMP;
    linearDesc.AddressW = Diligent::TEXTURE_ADDRESS_CLAMP;
    device->CreateSampler(linearDesc, &m_LinearSampler);

    Diligent::SamplerDesc pointDesc = linearDesc;
    pointDesc.Name = "Cloud point sampler";
    pointDesc.MinFilter = Diligent::FILTER_TYPE_POINT;
    pointDesc.MagFilter = Diligent::FILTER_TYPE_POINT;
    pointDesc.MipFilter = Diligent::FILTER_TYPE_POINT;
    device->CreateSampler(pointDesc, &m_PointSampler);
    Require(m_LinearSampler != nullptr && m_PointSampler != nullptr,
            "Failed to create volumetric cloud samplers");

    m_BakeCB = CreateCloudConstantBuffer();
    SetConstantBuffer(m_DistributionSRB, Diligent::SHADER_TYPE_COMPUTE,
                      "CloudBakeCB", m_BakeCB);
    SetConstantBuffer(m_DetailSRB, Diligent::SHADER_TYPE_COMPUTE,
                      "CloudBakeCB", m_BakeCB);
}

Diligent::RefCntAutoPtr<Diligent::IShader> VolumetricCloudPass::LoadShader(
    Diligent::IRenderDevice* device,
    const std::string& path,
    Diligent::SHADER_TYPE type,
    const char* name,
    const char* entryPoint) const
{
    const auto byteCode = ReadBinaryFile(path);
    Diligent::ShaderCreateInfo shaderCI;
    shaderCI.Desc = Diligent::ShaderDesc{name, type, false};
    shaderCI.ByteCode = byteCode.data();
    shaderCI.ByteCodeSize = byteCode.size();
    shaderCI.EntryPoint = entryPoint;
    shaderCI.SourceLanguage = Diligent::SHADER_SOURCE_LANGUAGE_DEFAULT;

    Diligent::RefCntAutoPtr<Diligent::IShader> shader;
    device->CreateShader(shaderCI, &shader);
    Require(shader != nullptr, "Failed to create volumetric cloud shader");
    return shader;
}

Diligent::RefCntAutoPtr<Diligent::IPipelineState>
VolumetricCloudPass::CreateComputePipeline(
    Diligent::IRenderDevice* device,
    Diligent::IShader* shader,
    const char* name) const
{
    Diligent::ComputePipelineStateCreateInfo computeCI{name};
    computeCI.pCS = shader;
    computeCI.PSODesc.ImmediateContextMask = m_ImmediateContextMask;
    computeCI.PSODesc.ResourceLayout.DefaultVariableType =
        Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE;

    Diligent::RefCntAutoPtr<Diligent::IPipelineState> pipeline;
    device->CreateComputePipelineState(computeCI, &pipeline);
    return pipeline;
}

CloudTextures VolumetricCloudPass::CreateTextures(
    Diligent::Uint32 distributionWidth,
    Diligent::Uint32 distributionHeight,
    Diligent::Uint32 detailWidth,
    Diligent::Uint32 detailHeight,
    Diligent::Uint32 detailDepth) const
{
    CloudTextures textures;
    textures.Distribution = CreateStorageTexture(
        m_Device, "Cloud distribution Worley FBM", Diligent::RESOURCE_DIM_TEX_2D,
        distributionWidth, distributionHeight, 1, m_ImmediateContextMask);
    textures.Detail = CreateStorageTexture(
        m_Device, "Cloud detail Worley curl volume", Diligent::RESOURCE_DIM_TEX_3D,
        detailWidth, detailHeight, detailDepth, m_ImmediateContextMask);
    return textures;
}

Diligent::RefCntAutoPtr<Diligent::IBuffer>
VolumetricCloudPass::CreateCloudConstantBuffer() const
{
    Diligent::BufferDesc desc;
    desc.Name = "Volumetric cloud constants";
    desc.Size = sizeof(VolumetricCloudConstants);
    desc.BindFlags = Diligent::BIND_UNIFORM_BUFFER;
    desc.Usage = Diligent::USAGE_DEFAULT;
    desc.ImmediateContextMask = m_ImmediateContextMask;

    Diligent::RefCntAutoPtr<Diligent::IBuffer> buffer;
    m_Device->CreateBuffer(desc, nullptr, &buffer);
    Require(buffer != nullptr, "Failed to create volumetric cloud constant buffer");
    return buffer;
}

void VolumetricCloudPass::SetTexture(
    Diligent::IShaderResourceBinding* binding,
    Diligent::SHADER_TYPE stage,
    const char* name,
    Diligent::ITextureView* view)
{
    Require(binding != nullptr && view != nullptr, "Cloud texture binding is unavailable");
    auto* variable = binding->GetVariableByName(stage, name);
    Require(variable != nullptr, name);
    variable->Set(view, Diligent::SET_SHADER_RESOURCE_FLAG_ALLOW_OVERWRITE);
}

void VolumetricCloudPass::SetSampler(
    Diligent::IShaderResourceBinding* binding,
    Diligent::SHADER_TYPE stage,
    const char* name,
    Diligent::ISampler* sampler)
{
    Require(binding != nullptr && sampler != nullptr, "Cloud sampler binding is unavailable");
    auto* variable = binding->GetVariableByName(stage, name);
    Require(variable != nullptr, name);
    variable->Set(sampler, Diligent::SET_SHADER_RESOURCE_FLAG_ALLOW_OVERWRITE);
}

void VolumetricCloudPass::SetConstantBuffer(
    Diligent::IShaderResourceBinding* binding,
    Diligent::SHADER_TYPE stage,
    const char* name,
    Diligent::IBuffer* buffer)
{
    Require(binding != nullptr && buffer != nullptr,
            "Cloud constant buffer binding is unavailable");
    auto* variable = binding->GetVariableByName(stage, name);
    Require(variable != nullptr, name);
    variable->Set(buffer, Diligent::SET_SHADER_RESOURCE_FLAG_ALLOW_OVERWRITE);
}

void VolumetricCloudPass::BakeDistribution(
    Diligent::IDeviceContext* context,
    const CloudTextures& textures,
    const CloudBakeConstants& constants)
{
    auto* output = textures.Distribution->GetDefaultView(
        Diligent::TEXTURE_VIEW_UNORDERED_ACCESS);
    Require(output != nullptr, "Cloud distribution UAV is unavailable");
    SetTexture(m_DistributionSRB, Diligent::SHADER_TYPE_COMPUTE,
               "g_CloudDistribution", output);
    context->UpdateBuffer(m_BakeCB, 0, sizeof(constants), &constants,
                          Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
    context->SetPipelineState(m_DistributionPSO);
    context->CommitShaderResources(
        m_DistributionSRB, Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
    context->DispatchCompute({
        (constants.Width + 7u) / 8u,
        (constants.Height + 7u) / 8u,
        1u});
}

void VolumetricCloudPass::BakeDetail(
    Diligent::IDeviceContext* context,
    const CloudTextures& textures,
    const CloudBakeConstants& constants)
{
    auto* output = textures.Detail->GetDefaultView(
        Diligent::TEXTURE_VIEW_UNORDERED_ACCESS);
    Require(output != nullptr, "Cloud detail UAV is unavailable");
    SetTexture(m_DetailSRB, Diligent::SHADER_TYPE_COMPUTE,
               "g_CloudDetailVolume", output);
    context->UpdateBuffer(m_BakeCB, 0, sizeof(constants), &constants,
                          Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
    context->SetPipelineState(m_DetailPSO);
    context->CommitShaderResources(
        m_DetailSRB, Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
    context->DispatchCompute({
        (constants.Width + 3u) / 4u,
        (constants.Height + 3u) / 4u,
        (constants.Depth + 3u) / 4u});
}

void VolumetricCloudPass::Render(
    Diligent::IDeviceContext* context,
    const CloudTextures& textures,
    Diligent::ITextureView* renderTarget,
    Diligent::ITextureView* sceneDepth,
    Diligent::IBuffer* cloudConstantBuffer,
    const VolumetricCloudConstants& constants)
{
    Require(renderTarget != nullptr && sceneDepth != nullptr &&
                cloudConstantBuffer != nullptr,
            "Cloud render inputs are unavailable");
    context->UpdateBuffer(cloudConstantBuffer, 0, sizeof(constants), &constants,
                          Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION);

    SetConstantBuffer(m_RenderSRB, Diligent::SHADER_TYPE_PIXEL,
                      "VolumetricCloudCB", cloudConstantBuffer);
    SetTexture(m_RenderSRB, Diligent::SHADER_TYPE_PIXEL, "g_CloudDistribution",
               textures.Distribution->GetDefaultView(
                   Diligent::TEXTURE_VIEW_SHADER_RESOURCE));
    SetTexture(m_RenderSRB, Diligent::SHADER_TYPE_PIXEL, "g_CloudDetailNoise",
               textures.Detail->GetDefaultView(
                   Diligent::TEXTURE_VIEW_SHADER_RESOURCE));
    SetTexture(m_RenderSRB, Diligent::SHADER_TYPE_PIXEL, "g_SceneDepth", sceneDepth);
    SetSampler(m_RenderSRB, Diligent::SHADER_TYPE_PIXEL,
               "g_LinearSampler", m_LinearSampler);
    SetSampler(m_RenderSRB, Diligent::SHADER_TYPE_PIXEL,
               "g_PointSampler", m_PointSampler);

    context->SetRenderTargets(1, &renderTarget, nullptr,
                              Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
    context->SetPipelineState(m_RenderPSO);
    context->CommitShaderResources(
        m_RenderSRB, Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
    context->Draw(Diligent::DrawAttribs{3, Diligent::DRAW_FLAG_VERIFY_ALL});
}

} // namespace VolumetricCloudTest
