#pragma once

#include <PipelineState.h>
#include <RefCntAutoPtr.hpp>
#include <Shader.h>

#include <string>

namespace Diligent
{
class IRenderDevice;
class IShaderResourceBinding;
}

namespace Nanite
{

class Pipelines
{
public:
    Pipelines(Diligent::IRenderDevice* device,
              Diligent::TEXTURE_FORMAT colorFormat,
              Diligent::TEXTURE_FORMAT depthFormat,
              const std::string& shaderDirectory,
              Diligent::Uint64 immediateContextMask = 1);

    Diligent::IPipelineState* InstanceCull() const { return m_InstanceCull; }
    Diligent::IPipelineState* PersistentCull() const { return m_PersistentCull; }
    Diligent::IPipelineState* GenerateIndirect() const { return m_GenerateIndirect; }
    Diligent::IPipelineState* PostCull() const { return m_PostCull; }
    Diligent::IPipelineState* PostSeed() const { return m_PostSeed; }
    Diligent::IPipelineState* PrepareDispatch() const { return m_PrepareDispatch; }
    Diligent::IPipelineState* RasterBin() const { return m_RasterBin; }
    Diligent::IPipelineState* HizBuild() const { return m_HizBuild; }
    Diligent::IPipelineState* Raster() const { return m_Raster; }
    Diligent::IPipelineState* PostRaster() const { return m_PostRaster; }
    Diligent::IPipelineState* OverdrawRaster() const { return m_OverdrawRaster; }
    // Two-pass compute rasterizer: depth min first, then payload selection from
    // triangles matching that depth. SoftCheck validates and clears the results.
    Diligent::IPipelineState* SoftRaster() const { return m_SoftRaster; }
    Diligent::IPipelineState* SoftVisibility() const { return m_SoftVisibility; }
    Diligent::IPipelineState* SoftCheck() const { return m_SoftCheck; }
    Diligent::IPipelineState* VisibilityResolve() const { return m_VisibilityResolve; }
    Diligent::IPipelineState* VisibilityShading() const { return m_VisibilityShading; }
    // Vertex-only pipeline over the same clusters, writing depth and nothing
    // else. It exists so the compute rasterizer is priced against comparable
    // work: the main raster pass also shades Blinn-Phong and fills the depth
    // copy the HZB reads, so timing against it would flatter the compute path.
    Diligent::IPipelineState* HardwareDepthOnly() const { return m_HardwareDepthOnly; }

    Diligent::IShaderResourceBinding* InstanceCullBinding() const { return m_InstanceCullBinding; }
    Diligent::IShaderResourceBinding* PersistentCullBinding() const { return m_PersistentCullBinding; }
    Diligent::IShaderResourceBinding* GenerateIndirectBinding() const { return m_GenerateIndirectBinding; }
    Diligent::IShaderResourceBinding* PostCullBinding() const { return m_PostCullBinding; }
    Diligent::IShaderResourceBinding* PostSeedBinding() const { return m_PostSeedBinding; }
    Diligent::IShaderResourceBinding* PrepareDispatchBinding() const { return m_PrepareDispatchBinding; }
    Diligent::IShaderResourceBinding* RasterBinBinding() const { return m_RasterBinBinding; }
    Diligent::IShaderResourceBinding* HizBuildBinding() const { return m_HizBuildBinding; }
    Diligent::IShaderResourceBinding* RasterBinding() const { return m_RasterBinding; }
    Diligent::IShaderResourceBinding* PostRasterBinding() const { return m_PostRasterBinding; }
    Diligent::IShaderResourceBinding* OverdrawRasterBinding() const { return m_OverdrawRasterBinding; }
    Diligent::IShaderResourceBinding* SoftRasterBinding() const { return m_SoftRasterBinding; }
    Diligent::IShaderResourceBinding* SoftVisibilityBinding() const { return m_SoftVisibilityBinding; }
    Diligent::IShaderResourceBinding* SoftCheckBinding() const { return m_SoftCheckBinding; }
    Diligent::IShaderResourceBinding* VisibilityResolveBinding() const { return m_VisibilityResolveBinding; }
    Diligent::IShaderResourceBinding* VisibilityShadingBinding() const { return m_VisibilityShadingBinding; }
    Diligent::IShaderResourceBinding* HardwareDepthOnlyBinding() const { return m_HardwareDepthOnlyBinding; }

private:
    Diligent::RefCntAutoPtr<Diligent::IShader> LoadShader(
        Diligent::IRenderDevice* device,
        const std::string& path,
        Diligent::SHADER_TYPE type,
        const char* name,
        const char* entryPoint);

    Diligent::RefCntAutoPtr<Diligent::IPipelineState> CreateComputePipeline(
        Diligent::IRenderDevice* device,
        Diligent::IShader* shader,
        const char* name,
        const Diligent::ShaderResourceVariableDesc* variables = nullptr,
        Diligent::Uint32 variableCount = 0);

    Diligent::RefCntAutoPtr<Diligent::IPipelineState> CreateRasterPipeline(
        Diligent::IRenderDevice* device,
        Diligent::IShader* vertexShader,
        Diligent::IShader* pixelShader,
        Diligent::TEXTURE_FORMAT colorFormat,
        Diligent::TEXTURE_FORMAT depthFormat,
        Diligent::COMPARISON_FUNCTION depthFunction,
        // The overdraw view needs the same geometry drawn with the depth test off
        // and the colour target accumulating instead of replacing, which is a
        // different PSO but the same shaders.
        bool accumulateWithoutDepth = false);

    Diligent::Uint64 m_ImmediateContextMask = 1;

    Diligent::RefCntAutoPtr<Diligent::IPipelineState> m_InstanceCull;
    Diligent::RefCntAutoPtr<Diligent::IPipelineState> m_PersistentCull;
    Diligent::RefCntAutoPtr<Diligent::IPipelineState> m_GenerateIndirect;
    Diligent::RefCntAutoPtr<Diligent::IPipelineState> m_PostCull;
    Diligent::RefCntAutoPtr<Diligent::IPipelineState> m_PostSeed;
    Diligent::RefCntAutoPtr<Diligent::IPipelineState> m_PrepareDispatch;
    Diligent::RefCntAutoPtr<Diligent::IPipelineState> m_RasterBin;
    Diligent::RefCntAutoPtr<Diligent::IPipelineState> m_HizBuild;
    Diligent::RefCntAutoPtr<Diligent::IPipelineState> m_Raster;
    Diligent::RefCntAutoPtr<Diligent::IPipelineState> m_PostRaster;
    Diligent::RefCntAutoPtr<Diligent::IPipelineState> m_OverdrawRaster;
    Diligent::RefCntAutoPtr<Diligent::IPipelineState> m_SoftRaster;
    Diligent::RefCntAutoPtr<Diligent::IPipelineState> m_SoftVisibility;
    Diligent::RefCntAutoPtr<Diligent::IPipelineState> m_SoftCheck;
    Diligent::RefCntAutoPtr<Diligent::IPipelineState> m_VisibilityResolve;
    Diligent::RefCntAutoPtr<Diligent::IPipelineState> m_VisibilityShading;
    Diligent::RefCntAutoPtr<Diligent::IPipelineState> m_HardwareDepthOnly;

    Diligent::RefCntAutoPtr<Diligent::IShaderResourceBinding> m_InstanceCullBinding;
    Diligent::RefCntAutoPtr<Diligent::IShaderResourceBinding> m_PersistentCullBinding;
    Diligent::RefCntAutoPtr<Diligent::IShaderResourceBinding> m_GenerateIndirectBinding;
    Diligent::RefCntAutoPtr<Diligent::IShaderResourceBinding> m_PostCullBinding;
    Diligent::RefCntAutoPtr<Diligent::IShaderResourceBinding> m_PostSeedBinding;
    Diligent::RefCntAutoPtr<Diligent::IShaderResourceBinding> m_PrepareDispatchBinding;
    Diligent::RefCntAutoPtr<Diligent::IShaderResourceBinding> m_RasterBinBinding;
    Diligent::RefCntAutoPtr<Diligent::IShaderResourceBinding> m_HizBuildBinding;
    Diligent::RefCntAutoPtr<Diligent::IShaderResourceBinding> m_RasterBinding;
    Diligent::RefCntAutoPtr<Diligent::IShaderResourceBinding> m_PostRasterBinding;
    Diligent::RefCntAutoPtr<Diligent::IShaderResourceBinding> m_OverdrawRasterBinding;
    Diligent::RefCntAutoPtr<Diligent::IShaderResourceBinding> m_SoftRasterBinding;
    Diligent::RefCntAutoPtr<Diligent::IShaderResourceBinding> m_SoftVisibilityBinding;
    Diligent::RefCntAutoPtr<Diligent::IShaderResourceBinding> m_SoftCheckBinding;
    Diligent::RefCntAutoPtr<Diligent::IShaderResourceBinding> m_VisibilityResolveBinding;
    Diligent::RefCntAutoPtr<Diligent::IShaderResourceBinding> m_VisibilityShadingBinding;
    Diligent::RefCntAutoPtr<Diligent::IShaderResourceBinding> m_HardwareDepthOnlyBinding;
};

} // namespace Nanite
