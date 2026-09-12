#include "NanitePipelines.hpp"

#include <RenderDevice.h>
#include <ShaderResourceBinding.h>

#include <fstream>
#include <iterator>
#include <stdexcept>
#include <vector>

namespace
{

std::vector<Diligent::Uint8> ReadBinaryFile(const std::string& path)
{
    std::ifstream file{path, std::ios::binary | std::ios::ate};
    if (!file)
        throw std::runtime_error{"Unable to open Nanite shader: " + path};

    const auto size = file.tellg();
    if (size <= 0)
        throw std::runtime_error{"Nanite shader is empty: " + path};

    std::vector<Diligent::Uint8> data(static_cast<size_t>(size));
    file.seekg(0, std::ios::beg);
    file.read(reinterpret_cast<char*>(data.data()), size);
    if (!file)
        throw std::runtime_error{"Unable to read Nanite shader: " + path};
    return data;
}

void Require(bool condition, const char* message)
{
    if (!condition)
        throw std::runtime_error{message};
}

} // namespace

namespace Nanite
{

Pipelines::Pipelines(Diligent::IRenderDevice* device,
                     Diligent::TEXTURE_FORMAT colorFormat,
                     Diligent::TEXTURE_FORMAT depthFormat,
                     const std::string& shaderDirectory,
                     Diligent::Uint64 immediateContextMask) :
    m_ImmediateContextMask{immediateContextMask}
{
    auto instanceCullShader = LoadShader(
        device,
        shaderDirectory + "/Nanite/NaniteInstanceCull.comp.spv",
        Diligent::SHADER_TYPE_COMPUTE,
        "Nanite Instance Cull CS",
        "main");
    auto persistentCullShader = LoadShader(
        device,
        shaderDirectory + "/Nanite/NanitePersistentCull.comp.spv",
        Diligent::SHADER_TYPE_COMPUTE,
        "Nanite Persistent Cull CS",
        "main");
    auto generateIndirectShader = LoadShader(
        device,
        shaderDirectory + "/Nanite/NaniteGenerateIndirect.comp.spv",
        Diligent::SHADER_TYPE_COMPUTE,
        "Nanite Generate Indirect CS",
        "main");
    auto postCullShader = LoadShader(
        device,
        shaderDirectory + "/Nanite/NanitePostCull.comp.spv",
        Diligent::SHADER_TYPE_COMPUTE,
        "Nanite Post Cull CS",
        "main");
    auto hizBuildShader = LoadShader(
        device,
        shaderDirectory + "/Nanite/NaniteHizBuild.comp.spv",
        Diligent::SHADER_TYPE_COMPUTE,
        "Nanite HZB Build CS",
        "main");
    auto postSeedShader = LoadShader(
        device,
        shaderDirectory + "/Nanite/NanitePostSeed.comp.spv",
        Diligent::SHADER_TYPE_COMPUTE,
        "Nanite Post Seed CS",
        "main");
    auto prepareDispatchShader = LoadShader(
        device,
        shaderDirectory + "/Nanite/NanitePrepareDispatch.comp.spv",
        Diligent::SHADER_TYPE_COMPUTE,
        "Nanite Prepare Dispatch CS",
        "main");
    auto rasterBinShader = LoadShader(
        device,
        shaderDirectory + "/Nanite/NaniteRasterBin.comp.spv",
        Diligent::SHADER_TYPE_COMPUTE,
        "Nanite Raster Bin CS",
        "main");
    auto rasterVertexShader = LoadShader(
        device,
        shaderDirectory + "/Nanite/NaniteRaster.vert.spv",
        Diligent::SHADER_TYPE_VERTEX,
        "Nanite Raster VS",
        "NaniteVS");
    auto rasterPixelShader = LoadShader(
        device,
        shaderDirectory + "/Nanite/NaniteRaster.frag.spv",
        Diligent::SHADER_TYPE_PIXEL,
        "Nanite Raster PS",
        "NanitePS");
    auto softRasterShader = LoadShader(
        device,
        shaderDirectory + "/Nanite/NaniteSoftRasterDepth.comp.spv",
        Diligent::SHADER_TYPE_COMPUTE,
        "Nanite Soft Raster Depth CS",
        "main");
    auto softVisibilityShader = LoadShader(
        device,
        shaderDirectory + "/Nanite/NaniteSoftRasterVisibility.comp.spv",
        Diligent::SHADER_TYPE_COMPUTE,
        "Nanite Soft Raster Visibility CS",
        "main");
    auto softCheckShader = LoadShader(
        device,
        shaderDirectory + "/Nanite/NaniteSoftDepthCheck.comp.spv",
        Diligent::SHADER_TYPE_COMPUTE,
        "Nanite Soft Depth Check CS",
        "main");
    auto visibilityResolveShader = LoadShader(
        device,
        shaderDirectory + "/Nanite/NaniteResolveVisibility.comp.spv",
        Diligent::SHADER_TYPE_COMPUTE,
        "Nanite Visibility Resolve CS",
        "main");
    auto visibilityShadeVertexShader = LoadShader(
        device,
        shaderDirectory + "/Nanite/NaniteVisibilityShade.vert.spv",
        Diligent::SHADER_TYPE_VERTEX,
        "Nanite Visibility Shade VS",
        "NaniteVisibilityVS");
    auto visibilityShadePixelShader = LoadShader(
        device,
        shaderDirectory + "/Nanite/NaniteVisibilityShade.frag.spv",
        Diligent::SHADER_TYPE_PIXEL,
        "Nanite Visibility Shade PS",
        "NaniteVisibilityPS");

    m_InstanceCull = CreateComputePipeline(device, instanceCullShader, "Nanite Instance Cull");
    m_PersistentCull = CreateComputePipeline(device, persistentCullShader, "Nanite Persistent Cull");
    m_GenerateIndirect = CreateComputePipeline(device, generateIndirectShader, "Nanite Generate Indirect");
    m_PostCull = CreateComputePipeline(device, postCullShader, "Nanite Post Cull");
    m_PostSeed = CreateComputePipeline(device, postSeedShader, "Nanite Post Seed");
    m_PrepareDispatch = CreateComputePipeline(device, prepareDispatchShader, "Nanite Prepare Dispatch");
    m_RasterBin = CreateComputePipeline(device, rasterBinShader, "Nanite Raster Bin");
    m_SoftRaster = CreateComputePipeline(device, softRasterShader, "Nanite Soft Raster Depth");
    m_SoftVisibility = CreateComputePipeline(device, softVisibilityShader, "Nanite Soft Raster Visibility");
    m_SoftCheck = CreateComputePipeline(device, softCheckShader, "Nanite Soft Depth Check");
    m_VisibilityResolve = CreateComputePipeline(device, visibilityResolveShader, "Nanite Visibility Resolve");
    {
        Diligent::GraphicsPipelineStateCreateInfo shadeCI{"Nanite Visibility Shading"};
        shadeCI.pVS = visibilityShadeVertexShader;
        shadeCI.pPS = visibilityShadePixelShader;
        shadeCI.GraphicsPipeline.PrimitiveTopology = Diligent::PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        shadeCI.GraphicsPipeline.RasterizerDesc.CullMode = Diligent::CULL_MODE_NONE;
        shadeCI.GraphicsPipeline.DepthStencilDesc.DepthEnable = Diligent::False;
        shadeCI.GraphicsPipeline.NumRenderTargets = 1;
        shadeCI.GraphicsPipeline.RTVFormats[0] = colorFormat;
        shadeCI.GraphicsPipeline.DSVFormat = Diligent::TEX_FORMAT_UNKNOWN;
        shadeCI.PSODesc.ImmediateContextMask = m_ImmediateContextMask;
        shadeCI.PSODesc.ResourceLayout.DefaultVariableType =
            Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE;
        device->CreateGraphicsPipelineState(shadeCI, &m_VisibilityShading);
    }

    // The HZB is built with several dispatches per frame, each one reading a
    // different source level and writing a different set of destination mips.
    // MUTABLE variables live in the static/mutable descriptor set, which the
    // Vulkan backend mutates in place and does not version per dispatch, so all
    // dispatches in one submission would observe the bindings written last.
    // DYNAMIC variables get a freshly allocated descriptor set on every
    // CommitShaderResources, which is what makes the per-batch rebinds stick.
    const Diligent::ShaderResourceVariableDesc hizVariables[] = {
        {Diligent::SHADER_TYPE_COMPUTE, "g_SourceDepth", Diligent::SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
        {Diligent::SHADER_TYPE_COMPUTE, "g_OutMip1", Diligent::SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
        {Diligent::SHADER_TYPE_COMPUTE, "g_OutMip2", Diligent::SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
        {Diligent::SHADER_TYPE_COMPUTE, "g_OutMip3", Diligent::SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
    };
    m_HizBuild = CreateComputePipeline(
        device,
        hizBuildShader,
        "Nanite HZB Build",
        hizVariables,
        static_cast<Diligent::Uint32>(std::size(hizVariables)));
    m_Raster = CreateRasterPipeline(
        device, rasterVertexShader, rasterPixelShader, colorFormat, depthFormat,
        Diligent::COMPARISON_FUNC_LESS_EQUAL);
    m_PostRaster = CreateRasterPipeline(
        device, rasterVertexShader, rasterPixelShader, colorFormat, depthFormat,
        Diligent::COMPARISON_FUNC_LESS);
    m_OverdrawRaster = CreateRasterPipeline(
        device, rasterVertexShader, rasterPixelShader, colorFormat, depthFormat,
        Diligent::COMPARISON_FUNC_ALWAYS, true);

    // Same vertex shader, same clusters, no pixel shader and no colour targets:
    // the hardware's depth-only cost, which is what the compute rasterizer has to
    // be measured against. The main raster pass shades and writes two targets, so
    // beating it says little about the rasterization itself.
    {
        Diligent::GraphicsPipelineStateCreateInfo depthOnlyCI{"Nanite Hardware Depth Only"};
        depthOnlyCI.pVS = rasterVertexShader;
        depthOnlyCI.GraphicsPipeline.PrimitiveTopology = Diligent::PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        depthOnlyCI.GraphicsPipeline.RasterizerDesc.CullMode = Diligent::CULL_MODE_NONE;
        depthOnlyCI.GraphicsPipeline.DepthStencilDesc.DepthEnable = Diligent::True;
        depthOnlyCI.GraphicsPipeline.DepthStencilDesc.DepthWriteEnable = Diligent::True;
        depthOnlyCI.GraphicsPipeline.DepthStencilDesc.DepthFunc =
            Diligent::COMPARISON_FUNC_LESS_EQUAL;
        depthOnlyCI.GraphicsPipeline.NumRenderTargets = 0;
        depthOnlyCI.GraphicsPipeline.DSVFormat = depthFormat;
        depthOnlyCI.PSODesc.ImmediateContextMask = m_ImmediateContextMask;
        depthOnlyCI.PSODesc.ResourceLayout.DefaultVariableType =
            Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE;
        device->CreateGraphicsPipelineState(depthOnlyCI, &m_HardwareDepthOnly);
    }

    Require(m_InstanceCull != nullptr, "Failed to create Nanite instance cull PSO");
    Require(m_PersistentCull != nullptr, "Failed to create Nanite persistent cull PSO");
    Require(m_GenerateIndirect != nullptr, "Failed to create Nanite indirect PSO");
    Require(m_PostCull != nullptr, "Failed to create Nanite post cull PSO");
    Require(m_PostSeed != nullptr, "Failed to create Nanite post seed PSO");
    Require(m_PrepareDispatch != nullptr, "Failed to create Nanite prepare dispatch PSO");
    Require(m_RasterBin != nullptr, "Failed to create Nanite raster bin PSO");
    Require(m_HizBuild != nullptr, "Failed to create Nanite HZB PSO");
    Require(m_Raster != nullptr, "Failed to create Nanite raster PSO");
    Require(m_PostRaster != nullptr, "Failed to create Nanite post raster PSO");
    Require(m_OverdrawRaster != nullptr, "Failed to create Nanite overdraw raster PSO");
    Require(m_SoftRaster != nullptr, "Failed to create Nanite soft raster PSO");
    Require(m_SoftVisibility != nullptr, "Failed to create Nanite soft visibility PSO");
    Require(m_SoftCheck != nullptr, "Failed to create Nanite soft depth check PSO");
    Require(m_VisibilityResolve != nullptr, "Failed to create Nanite visibility resolve PSO");
    Require(m_VisibilityShading != nullptr, "Failed to create Nanite visibility shading PSO");
    Require(m_HardwareDepthOnly != nullptr, "Failed to create Nanite depth only PSO");

    m_InstanceCull->CreateShaderResourceBinding(&m_InstanceCullBinding, true);
    m_PersistentCull->CreateShaderResourceBinding(&m_PersistentCullBinding, true);
    m_GenerateIndirect->CreateShaderResourceBinding(&m_GenerateIndirectBinding, true);
    m_PostCull->CreateShaderResourceBinding(&m_PostCullBinding, true);
    m_PostSeed->CreateShaderResourceBinding(&m_PostSeedBinding, true);
    m_PrepareDispatch->CreateShaderResourceBinding(&m_PrepareDispatchBinding, true);
    m_RasterBin->CreateShaderResourceBinding(&m_RasterBinBinding, true);
    m_HizBuild->CreateShaderResourceBinding(&m_HizBuildBinding, true);
    m_Raster->CreateShaderResourceBinding(&m_RasterBinding, true);
    m_PostRaster->CreateShaderResourceBinding(&m_PostRasterBinding, true);
    m_OverdrawRaster->CreateShaderResourceBinding(&m_OverdrawRasterBinding, true);
    m_SoftRaster->CreateShaderResourceBinding(&m_SoftRasterBinding, true);
    m_SoftVisibility->CreateShaderResourceBinding(&m_SoftVisibilityBinding, true);
    m_SoftCheck->CreateShaderResourceBinding(&m_SoftCheckBinding, true);
    m_VisibilityResolve->CreateShaderResourceBinding(&m_VisibilityResolveBinding, true);
    m_VisibilityShading->CreateShaderResourceBinding(&m_VisibilityShadingBinding, true);
    m_HardwareDepthOnly->CreateShaderResourceBinding(&m_HardwareDepthOnlyBinding, true);
    Require(m_InstanceCullBinding != nullptr, "Failed to create Nanite instance cull SRB");
    Require(m_PersistentCullBinding != nullptr, "Failed to create Nanite persistent cull SRB");
    Require(m_GenerateIndirectBinding != nullptr, "Failed to create Nanite indirect SRB");
    Require(m_PostCullBinding != nullptr, "Failed to create Nanite post cull SRB");
    Require(m_PostSeedBinding != nullptr, "Failed to create Nanite post seed SRB");
    Require(m_PrepareDispatchBinding != nullptr, "Failed to create Nanite prepare dispatch SRB");
    Require(m_RasterBinBinding != nullptr, "Failed to create Nanite raster bin SRB");
    Require(m_HizBuildBinding != nullptr, "Failed to create Nanite HZB SRB");
    Require(m_RasterBinding != nullptr, "Failed to create Nanite raster SRB");
    Require(m_PostRasterBinding != nullptr, "Failed to create Nanite post raster SRB");
    Require(m_OverdrawRasterBinding != nullptr, "Failed to create Nanite overdraw raster SRB");
    Require(m_SoftRasterBinding != nullptr, "Failed to create Nanite soft raster SRB");
    Require(m_SoftVisibilityBinding != nullptr, "Failed to create Nanite soft visibility SRB");
    Require(m_SoftCheckBinding != nullptr, "Failed to create Nanite soft depth check SRB");
    Require(m_VisibilityResolveBinding != nullptr, "Failed to create Nanite visibility resolve SRB");
    Require(m_VisibilityShadingBinding != nullptr, "Failed to create Nanite visibility shading SRB");
    Require(m_HardwareDepthOnlyBinding != nullptr, "Failed to create Nanite depth only SRB");
}

Diligent::RefCntAutoPtr<Diligent::IShader> Pipelines::LoadShader(
    Diligent::IRenderDevice* device,
    const std::string& path,
    Diligent::SHADER_TYPE type,
    const char* name,
    const char* entryPoint)
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
    if (shader == nullptr)
        throw std::runtime_error{"Diligent failed to create Nanite shader: " + path};
    return shader;
}

Diligent::RefCntAutoPtr<Diligent::IPipelineState> Pipelines::CreateComputePipeline(
    Diligent::IRenderDevice* device,
    Diligent::IShader* shader,
    const char* name,
    const Diligent::ShaderResourceVariableDesc* variables,
    Diligent::Uint32 variableCount)
{
    Diligent::ComputePipelineStateCreateInfo pipelineCI{name};
    pipelineCI.pCS = shader;
    pipelineCI.PSODesc.ImmediateContextMask = m_ImmediateContextMask;
    pipelineCI.PSODesc.ResourceLayout.DefaultVariableType =
        Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE;
    pipelineCI.PSODesc.ResourceLayout.Variables = variables;
    pipelineCI.PSODesc.ResourceLayout.NumVariables = variableCount;

    Diligent::RefCntAutoPtr<Diligent::IPipelineState> pipeline;
    device->CreateComputePipelineState(pipelineCI, &pipeline);
    return pipeline;
}

Diligent::RefCntAutoPtr<Diligent::IPipelineState> Pipelines::CreateRasterPipeline(
    Diligent::IRenderDevice* device,
    Diligent::IShader* vertexShader,
    Diligent::IShader* pixelShader,
    Diligent::TEXTURE_FORMAT colorFormat,
    Diligent::TEXTURE_FORMAT depthFormat,
    Diligent::COMPARISON_FUNCTION depthFunction,
    bool accumulateWithoutDepth)
{
    Diligent::GraphicsPipelineStateCreateInfo pipelineCI{
        accumulateWithoutDepth ? "Nanite Overdraw Raster" : "Nanite Hardware Raster"};
    pipelineCI.pVS = vertexShader;
    pipelineCI.pPS = pixelShader;
    pipelineCI.PSODesc.ImmediateContextMask = m_ImmediateContextMask;
    pipelineCI.GraphicsPipeline.PrimitiveTopology = Diligent::PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    pipelineCI.GraphicsPipeline.RasterizerDesc.CullMode = Diligent::CULL_MODE_NONE;
    pipelineCI.GraphicsPipeline.DepthStencilDesc.DepthEnable =
        accumulateWithoutDepth ? Diligent::False : Diligent::True;
    pipelineCI.GraphicsPipeline.DepthStencilDesc.DepthWriteEnable =
        accumulateWithoutDepth ? Diligent::False : Diligent::True;
    pipelineCI.GraphicsPipeline.DepthStencilDesc.DepthFunc = depthFunction;
    if (accumulateWithoutDepth)
    {
        // One add per fragment that reaches the pixel shader, which is what makes
        // the colour target a count of how many times each pixel was shaded.
        auto& blend = pipelineCI.GraphicsPipeline.BlendDesc;
        blend.IndependentBlendEnable = Diligent::True;
        blend.RenderTargets[0].BlendEnable = Diligent::True;
        blend.RenderTargets[0].SrcBlend = Diligent::BLEND_FACTOR_ONE;
        blend.RenderTargets[0].DestBlend = Diligent::BLEND_FACTOR_ONE;
        blend.RenderTargets[0].BlendOp = Diligent::BLEND_OPERATION_ADD;
        blend.RenderTargets[0].SrcBlendAlpha = Diligent::BLEND_FACTOR_ONE;
        blend.RenderTargets[0].DestBlendAlpha = Diligent::BLEND_FACTOR_ONE;
        blend.RenderTargets[0].BlendOpAlpha = Diligent::BLEND_OPERATION_ADD;
        // The second target is the depth copy the HZB is built from. This pass runs
        // after that build and must not disturb it, and adding depths together
        // would be meaningless anyway, so drop the writes.
        blend.RenderTargets[1].RenderTargetWriteMask = Diligent::COLOR_MASK_NONE;
        blend.RenderTargets[2].RenderTargetWriteMask = Diligent::COLOR_MASK_NONE;
    }
    pipelineCI.GraphicsPipeline.NumRenderTargets = 3;
    pipelineCI.GraphicsPipeline.RTVFormats[0] = colorFormat;
    pipelineCI.GraphicsPipeline.RTVFormats[1] = Diligent::TEX_FORMAT_R32_FLOAT;
    pipelineCI.GraphicsPipeline.RTVFormats[2] = Diligent::TEX_FORMAT_R32_UINT;
    pipelineCI.GraphicsPipeline.DSVFormat = depthFormat;
    pipelineCI.PSODesc.ResourceLayout.DefaultVariableType =
        Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE;

    Diligent::RefCntAutoPtr<Diligent::IPipelineState> pipeline;
    device->CreateGraphicsPipelineState(pipelineCI, &pipeline);
    return pipeline;
}

} // namespace Nanite
