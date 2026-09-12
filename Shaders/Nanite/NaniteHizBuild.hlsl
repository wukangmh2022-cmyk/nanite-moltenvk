Texture2D<float> g_SourceDepth : register(t1);
RWTexture2D<float> g_OutMip1 : register(u16);
RWTexture2D<float> g_OutMip2 : register(u17);
RWTexture2D<float> g_OutMip3 : register(u18);

cbuffer NaniteHizConstants : register(b0)
{
    uint OutputMipCount;
    uint SourceWidth;
    uint SourceHeight;
    uint SourceTexWidth;
    uint SourceTexHeight;
    uint Padding0;
    uint Padding1;
    uint Padding2;
};

groupshared float g_Tile[64];

[numthreads(8, 8, 1)]
void main(uint groupIndex : SV_GroupIndex, uint3 dispatchThreadID : SV_DispatchThreadID)
{
    // Each dispatch consumes one single-mip source view and produces up to three
    // consecutive coarser mips through a group-shared max reduction. The source
    // is never a subresource of the destination texture: the C++ side copies the
    // previous batch result into a dedicated texture first, because Diligent
    // tracks resource state per texture rather than per subresource.
    //
    // SourceWidth/Height are the logical extent of the source level. The HZB is
    // padded to a power of two so every level halves exactly and no row or
    // column is ever dropped. SourceTexWidth/Height are the extent of the bound
    // texture, which for the first batch is the viewport-sized depth copy and
    // therefore smaller than the logical extent.
    uint2 baseOutput = dispatchThreadID.xy;
    bool inSourceTexture = baseOutput.x < SourceTexWidth && baseOutput.y < SourceTexHeight;
    uint2 sourceMax = uint2(SourceTexWidth - 1u, SourceTexHeight - 1u);
    // The source is a regular R32_FLOAT texture, so an integer fetch avoids
    // sampler/image-layout differences between Vulkan implementations.
    //
    // Standard depth uses 0 for near and 1 for far. An occlusion test needs the
    // farthest depth in a region: only when that farthest depth is still nearer
    // than the candidate can the whole region be occluded. Padding outside the
    // viewport reads as 1.0 so it can never prove occlusion.
    float maxDepth = inSourceTexture ?
        g_SourceDepth.Load(int3(min(baseOutput, sourceMax), 0)) :
        1.0;

    uint2 sourceExtent = uint2(SourceWidth, SourceHeight);

    g_Tile[groupIndex] = maxDepth;
    GroupMemoryBarrierWithGroupSync();
    if ((groupIndex & 0x09u) == 0u)
    {
        maxDepth = max(max(g_Tile[groupIndex], g_Tile[groupIndex + 1u]),
                       max(g_Tile[groupIndex + 8u], g_Tile[groupIndex + 9u]));
        uint2 mipOutput = baseOutput / 2u;
        uint2 mipExtent = max(sourceExtent >> 1u, uint2(1u, 1u));
        if (all(mipOutput < mipExtent))
            g_OutMip1[mipOutput] = maxDepth;
        g_Tile[groupIndex] = maxDepth;
    }
    if (OutputMipCount == 1u)
        return;

    GroupMemoryBarrierWithGroupSync();
    if ((groupIndex & 0x1Bu) == 0u)
    {
        maxDepth = max(max(g_Tile[groupIndex], g_Tile[groupIndex + 2u]),
                       max(g_Tile[groupIndex + 16u], g_Tile[groupIndex + 18u]));
        uint2 mipOutput = baseOutput / 4u;
        uint2 mipExtent = max(sourceExtent >> 2u, uint2(1u, 1u));
        if (all(mipOutput < mipExtent))
            g_OutMip2[mipOutput] = maxDepth;
        g_Tile[groupIndex] = maxDepth;
    }
    if (OutputMipCount == 2u)
        return;

    GroupMemoryBarrierWithGroupSync();
    if (groupIndex == 0u)
    {
        maxDepth = max(max(g_Tile[0], g_Tile[4]), max(g_Tile[32], g_Tile[36]));
        uint2 mipOutput = baseOutput / 8u;
        uint2 mipExtent = max(sourceExtent >> 3u, uint2(1u, 1u));
        if (all(mipOutput < mipExtent))
            g_OutMip3[mipOutput] = maxDepth;
    }
}
