#ifndef NANITE_COMMON_HLSL
#define NANITE_COMMON_HLSL

#define NANITE_INVALID_ID 0xFFFFFFFFu
#define NANITE_GROUP_SIZE 64
#define NANITE_DEPTH_EPSILON (2.0 / 16777216.0)
// High bits of MaxRefinementDepth are feature flags. This selects the
// orthographic light projection used by Virtual Shadow Map pages.
#define NANITE_SHADOW_PROJECTION_FLAG 0x08000000u

// Keep scalar uint buffer elements in named, role-specific structures. MoltenVK's
// SPIRV-Cross backend otherwise merges same-layout storage blocks and derives the
// Metal struct name from the first resource (for example g_IndexBuffer), which
// makes a second resource of that type hide the generated type. Each structure
// is still one uint, so the existing 4-byte structured-buffer stride is unchanged.
struct NaniteGeometryIndexElement
{
    uint Value;
};

struct NaniteClusterIndexElement
{
    uint Value;
};

struct NaniteSoftwareDrawIndexElement
{
    uint Value;
};

struct NaniteBinDrawIndexElement
{
    uint Value;
};

// These layouts are intentionally close to Nyx's GPU meshlet layouts, but do
// not depend on Nyx's bindless descriptor heap or Slang runtime.
struct NaniteInstance
{
    float4x4 WorldMatrix;
    float4 LocalBoundSphere;
    float3 BBoxMin;
    uint BBoxMinPadding;
    float3 BBoxMax;
    uint BBoxMaxPadding;
    uint RootNodeIndex;
    uint Padding0;
    uint Padding1;
    uint Padding2;
};

struct NaniteMaterial
{
    float4 BaseColor;
    float Metallic;
    float Roughness;
    float AmbientOcclusion;
    uint TextureIndex;
};

struct NaniteDagNode
{
    // Sphere enclosing every cluster group in the subtree, paired with the
    // largest group ParentError below this node. See DagNode in NaniteTypes.hpp:
    // the projected error of this pair dominates every group underneath, so a
    // node under the threshold prunes its whole subtree.
    float4 LodSphere;
    float3 BBoxMin;
    uint BBoxMinPadding;
    float3 BBoxMax;
    float MaxGroupError;
    uint NodeData;
};

struct NaniteClusterGroup
{
    float4 BoundSphere;
    float3 BBoxMin;
    uint BBoxMinPadding;
    float3 BBoxMax;
    float ParentError;
    uint ClusterStart;
    uint ClusterCount;
    uint RefineGroupIndex;
    uint Padding0;
};

struct NaniteCluster
{
    float4 BoundSphere;
    float3 BBoxMin;
    uint BBoxMinPadding;
    float3 BBoxMax;
    float MaxError;
    uint IndexCount;
    uint FirstIndex;
    uint VertexOffset;
    uint RefinedGroupIndex;
};

struct NaniteNodeTask
{
    uint InstanceIndex;
    uint NodeIndex;
    uint Ready;
    uint Padding0;
};

struct NaniteGroupTask
{
    uint InstanceIndex;
    uint GroupIndex;
    uint RefinementDepth;
    uint Ready;
};

struct NaniteClusterTask
{
    uint InstanceIndex;
    uint ClusterIndex;
};

// Layout-identical duplicates of the task structs above, used only by the
// occlusion recovery queues. They exist because SPIRV-Cross names the generated
// MSL struct after one of the variables sharing a block type, so a shader that
// binds both a live queue and its recovery queue ends up with that struct name
// shadowed by the other variable of the same name, which Metal rejects. Giving
// the recovery queues their own element type gives them their own block type.
struct NanitePostNodeTask
{
    uint InstanceIndex;
    uint NodeIndex;
    uint Ready;
    uint Padding0;
};

struct NanitePostGroupTask
{
    uint InstanceIndex;
    uint GroupIndex;
    uint RefinementDepth;
    uint Ready;
};

NanitePostNodeTask NaniteToPostNodeTask(NaniteNodeTask task)
{
    NanitePostNodeTask result;
    result.InstanceIndex = task.InstanceIndex;
    result.NodeIndex     = task.NodeIndex;
    result.Ready         = task.Ready;
    result.Padding0      = task.Padding0;
    return result;
}

NaniteNodeTask NaniteFromPostNodeTask(NanitePostNodeTask task)
{
    NaniteNodeTask result;
    result.InstanceIndex = task.InstanceIndex;
    result.NodeIndex     = task.NodeIndex;
    result.Ready         = task.Ready;
    result.Padding0      = task.Padding0;
    return result;
}

NanitePostGroupTask NaniteToPostGroupTask(NaniteGroupTask task)
{
    NanitePostGroupTask result;
    result.InstanceIndex    = task.InstanceIndex;
    result.GroupIndex       = task.GroupIndex;
    result.RefinementDepth  = task.RefinementDepth;
    result.Ready            = task.Ready;
    return result;
}

NaniteGroupTask NaniteFromPostGroupTask(NanitePostGroupTask task)
{
    NaniteGroupTask result;
    result.InstanceIndex    = task.InstanceIndex;
    result.GroupIndex       = task.GroupIndex;
    result.RefinementDepth  = task.RefinementDepth;
    result.Ready            = task.Ready;
    return result;
}

struct NaniteQueueState
{
    uint NodeRead;
    uint NodeWrite;
    uint NodePending;
    uint GroupRead;
    uint GroupWrite;
    uint GroupPending;
    uint ClusterWrite;
    uint VisibleClusterCount;
    uint DrawCount;
    uint OverflowFlags;
    uint VisibleInstanceCount;
    uint GroupFrustumRejected;
    uint GroupHZBRejected;
    uint ClusterFrustumRejected;
    uint ClusterHZBRejected;
    uint ClusterRefinementRejected;
    uint ClusterTinyRejected;
    uint PostClusterWrite;
    uint PostClusterCount;
    uint PostClusterHZBRejected;
    uint PostInstanceWrite;
    uint PostInstanceCount;
    uint PostNodeWrite;
    uint PostNodeCount;
    uint PostGroupWrite;
    uint PostGroupCount;
    uint MainClusterCount;
    uint PostInstanceRecovered;
    uint PostNodeRecovered;
    uint PostGroupRecovered;
    uint DebugProbeX;
    uint DebugProbeY;
    uint NodeLodRejected;
    // How many queue entries each of the indirect grids NanitePrepareDispatch
    // wrote can actually reach. The dispatched shaders compare their queue count
    // against these to detect a grid too small to cover the queue.
    uint DispatchedClusterTasks;
    uint DispatchedSeedTasks;
    uint DispatchedPostClusterTasks;
    // Software-raster probe. Triangles and pixels the compute rasterizer
    // actually covered, next to the two reasons it declined a triangle, so the
    // pass timing can be read as a price for a known amount of work rather than
    // for an unknown fraction of the frame. The depth counters are the
    // correctness half: matched and mismatched pixels against the hardware
    // depth, plus the pixels only one of the two covered.
    uint SoftRasterTriangles;
    uint SoftRasterPixels;
    uint SoftRasterSkippedLarge;
    uint SoftRasterSkippedClip;
    uint SoftDepthMatched;
    uint SoftDepthMismatch;
    uint SoftDepthMissing;
    uint SoftDepthExtra;
    // Largest absolute depth disagreement, scaled by 2^24-1.
    uint SoftDepthMaxDiff;
    // Visibility-buffer half of the probe: one depth key and triangle payload per
    // pixel, selected in a second pass from triangles matching the first pass's
    // atomic depth minimum. These counters verify that the selected payload really
    // belongs to the stored depth.
    uint SoftVisMissing;
    uint SoftVisKeyMismatch;
    uint SoftVisResolved;
    // The two ways a resolve can fail, kept apart because they do not mean the same
    // thing. NoCover is unambiguous: the payload names a triangle that never
    // touched this pixel, which only a torn pair can produce. DepthOff means the
    // named triangle does cover the pixel but interpolates a different depth there,
    // which is tearing between neighbours on the same surface - or just the two
    // shaders' arithmetic drifting past the tolerance, which is why it is not
    // counted as tearing outright.
    uint SoftVisNoCover;
    uint SoftVisDepthOff;
    // Compacted cluster count consumed by the hybrid software rasterizer.
    uint SoftClusterCount;
    uint HardwareClusterCount;
    uint SoftPadding2;
};

// One pixel of software visibility. Key is ~asuint(depth), so a cleared zero key
// cannot describe a real non-negative D32 value and doubles as the empty marker.
struct NaniteVisibility
{
    uint Payload;
    uint Key;
};

// Payload packing: the cluster's slot in this frame's draw list, and the triangle
// within it. The control panel permits 256 triangles per cluster, so the triangle
// needs eight bits. The draw slot is one-based to reserve zero for background;
// that leaves 24 bits, enough for 16,777,214 visible cluster slots per frame.
#define NANITE_VIS_TRIANGLE_BITS 8
#define NANITE_VIS_TRIANGLE_MASK ((1u << NANITE_VIS_TRIANGLE_BITS) - 1u)
// Zero stays reserved for background in the hardware R32_UINT target. Cluster
// draw slots start at zero, so their high half carries a one-based value.
#define NANITE_VIS_ENCODE(drawIndex, triangleIndex) \
    (((drawIndex) + 1u) << NANITE_VIS_TRIANGLE_BITS | ((triangleIndex) & NANITE_VIS_TRIANGLE_MASK))
#define NANITE_VIS_DRAW_INDEX(payload) (((payload) >> NANITE_VIS_TRIANGLE_BITS) - 1u)
#define NANITE_VIS_TRIANGLE_INDEX(payload) ((payload) & NANITE_VIS_TRIANGLE_MASK)

// Raised when a shader finds more work in its queue than the indirect grid it
// was launched with can cover, which would mean silently dropped clusters.
#define NANITE_OVERFLOW_DISPATCH_TOO_SMALL 0x200u

// Per-cluster data for the single instanced indirect draw. One entry per visible
// cluster, indexed by SV_InstanceID in the raster vertex shader.
struct NaniteDrawInstanceData
{
    uint InstanceIndex;
    uint ClusterIndex;
    uint FirstIndex;
    uint IndexCount;
};

cbuffer NaniteCullingConstants : register(b0)
{
    float4x4 ViewProjMatrix;
    float4x4 PreviousViewProjMatrix;
    float3 CameraWorldPosition;
    float ScreenErrorPixels;

    uint ViewportWidth;
    uint ViewportHeight;
    uint HZBWidth;
    uint HZBHeight;
    uint HZBMipCount;
    uint EnableHZB;
    uint InstanceCount;
    uint MaxNodeTasks;
    uint MaxGroupTasks;
    uint MaxClusterTasks;
    uint MaxDrawCommands;
    uint MaxRefinementDepth;
    uint CullingPass;
    uint MaxTraversalIterations;
    float FocalY;
    uint QueueGeneration;
    float RasterBinAreaCutoff;
    uint HybridRasterEnabled;
    // 1 selects the benchmark-only cluster-level HW/SW split. The value occupies
    // the former padding word to preserve the constant-buffer layout.
    uint ClusterRasterMode;
    uint CullingPadding4;
};

// NANITE_CULLING_PASS_MAIN queries the HZB built from the previous frame's
// depth; NANITE_CULLING_PASS_POST queries the one built from this frame's.
#define NANITE_CULLING_PASS_MAIN 0u
#define NANITE_CULLING_PASS_POST 1u

bool NodeIsGroup(uint nodeData)
{
    return (nodeData & 1u) != 0u;
}

uint NodeChildStart(uint nodeData)
{
    return (nodeData >> 1u) & 0x07FFFFFFu;
}

uint NodeChildCount(uint nodeData)
{
    return (nodeData >> 28u) & 0xFu;
}

uint NodeGroupIndex(uint nodeData)
{
    return (nodeData >> 1u) & 0x00FFFFFFu;
}

float4 BuildAabbCorner(float3 bboxMin, float3 bboxMax, uint cornerIndex)
{
    bool3 useMax = bool3(
        (cornerIndex & 1u) != 0u,
        (cornerIndex & 2u) != 0u,
        (cornerIndex & 4u) != 0u);
    return float4(
        useMax ? bboxMax : bboxMin,
        1.0);
}

uint ComputeHomogeneousClipMask(float4 homogeneousPos)
{
    uint mask = 0u;
    mask |= homogeneousPos.x < -homogeneousPos.w ? 1u : 0u;
    mask |= homogeneousPos.x >  homogeneousPos.w ? 2u : 0u;
    mask |= homogeneousPos.y < -homogeneousPos.w ? 4u : 0u;
    mask |= homogeneousPos.y >  homogeneousPos.w ? 8u : 0u;
    mask |= homogeneousPos.z < 0.0 ? 16u : 0u;
    mask |= homogeneousPos.z > homogeneousPos.w ? 32u : 0u;
    mask |= homogeneousPos.w <= 0.0 ? 64u : 0u;
    return mask;
}

bool ProjectBBoxToClip(
    float3 bboxMin,
    float3 bboxMax,
    float4x4 worldMatrix,
    float4x4 viewProjMatrix,
    out float4 clipMinOut,
    out float4 clipMaxOut,
    out bool clipValidOut)
{
    float4x4 worldToClip = mul(viewProjMatrix, worldMatrix);
    float4 homogeneousPos = mul(worldToClip, BuildAabbCorner(bboxMin, bboxMax, 0u));
    bool validClip = homogeneousPos.w > 1.0e-7;
    float4 clipPos = validClip ? homogeneousPos / homogeneousPos.w : homogeneousPos;
    uint rejectMask = ComputeHomogeneousClipMask(homogeneousPos);
    float4 clipMin = clipPos;
    float4 clipMax = clipPos;
    bool allCornersValid = validClip;

    [unroll]
    for (uint corner = 1u; corner < 8u; ++corner)
    {
        homogeneousPos = mul(worldToClip, BuildAabbCorner(bboxMin, bboxMax, corner));
        validClip = homogeneousPos.w > 1.0e-7;
        clipPos = validClip ? homogeneousPos / homogeneousPos.w : homogeneousPos;
        rejectMask &= ComputeHomogeneousClipMask(homogeneousPos);
        clipMin = min(clipMin, clipPos);
        clipMax = max(clipMax, clipPos);
        allCornersValid = allCornersValid && validClip;
    }

    clipValidOut = allCornersValid;
    clipMinOut = float4(clamp(clipMin.xy, -1.0, 1.0), clipMin.zw);
    clipMaxOut = float4(clamp(clipMax.xy, -1.0, 1.0), clipMax.zw);
    return rejectMask == 0u;
}

bool BBoxIntersectFrustum(
    float3 bboxMin,
    float3 bboxMax,
    float4x4 worldMatrix,
    out float4 clipMinOut,
    out float4 clipMaxOut,
    out bool clipValidOut)
{
    return ProjectBBoxToClip(
        bboxMin,
        bboxMax,
        worldMatrix,
        ViewProjMatrix,
        clipMinOut,
        clipMaxOut,
        clipValidOut);
}

int HZBMipLevel(float2 uvExtent, out bool clampedOut)
{
    float2 pixelExtent = uvExtent * float2(HZBWidth, HZBHeight);
    float largestExtent = max(pixelExtent.x, pixelExtent.y);
    clampedOut = false;
    // Diagnostic switch: force the authoritative full-resolution path to
    // separate mip-chain generation errors from the culling comparison.
    if ((MaxRefinementDepth & 0x40000000u) != 0u)
        return 0;
    // Small projected clusters are cheap to test exactly. A coarse mip texel
    // can include neighboring background pixels and turn a fully occluded
    // 2x4-pixel cluster into a false "visible" result.
    if (largestExtent <= 4.0)
        return 0;
    // Sample a mip whose texel footprint covers the whole projected bounds.
    // Finer mips can contain a nearer depth value outside the actual bounds,
    // which would incorrectly reject a visible cluster.
    float mip = ceil(log2(max(largestExtent, 1.0)));
    int maxMip = max(int(HZBMipCount) - 1, 0);
    // The HZB is padded to a power of two and keeps a full chain down to 1x1,
    // so a covering level always exists. If that ever stops holding, the
    // four-corner load below is not conservative and the caller must bail out.
    clampedOut = int(mip) > maxMip;
    return min(int(mip), maxMip);
}

uint2 HZBMipDimensions(uint mip)
{
    uint2 dimensions = uint2(HZBWidth, HZBHeight);
    [loop]
    for (uint level = 0u; level < mip; ++level)
        dimensions = max(dimensions >> 1u, 1u);
    return dimensions;
}

bool HZBVisible(
    Texture2D<float> hzb,
    SamplerState hzbSampler,
    float4 clipMin,
    float4 clipMax,
    out float farthestOccluderOut,
    out uint mipOut,
    out uint2 texelMinOut,
    out uint2 texelMaxOut)
{
    // Mip 0 is the 2x2 maximum of depth, so one mip 0 texel covers two screen
    // pixels per axis. The chain is also padded up to a power of two, so the
    // rendered image only covers the top-left sub-rectangle. Scale screen UVs by
    // viewport / (2 * HZB extent); the padding is never sampled.
    float2 hzbUvScale =
        float2(ViewportWidth, ViewportHeight) / (2.0 * float2(HZBWidth, HZBHeight));
    float2 minUV = saturate(float2(clipMin.x * 0.5 + 0.5, 0.5 - clipMax.y * 0.5)) * hzbUvScale;
    float2 maxUV = saturate(float2(clipMax.x * 0.5 + 0.5, 0.5 - clipMin.y * 0.5)) * hzbUvScale;
    float2 uvExtent = max(maxUV - minUV, 1.0 / float2(HZBWidth, HZBHeight));
    bool mipClamped;
    int mip = HZBMipLevel(uvExtent, mipClamped);
    mipOut = uint(mip);
    uint2 mipSize = HZBMipDimensions(uint(mip));
    if (mipClamped)
    {
        // No level is coarse enough to cover the bounds, so sampling only the
        // corners could miss a nearer surface inside the rectangle and reject a
        // visible object. Treat it as visible.
        farthestOccluderOut = 1.0;
        texelMinOut = 0u;
        texelMaxOut = 0u;
        return true;
    }

    // The selected mip is at least as coarse as the projected bounds, so the
    // bounds can intersect at most two texels per axis. Load every intersected
    // texel instead of relying on four arbitrary points inside the rectangle.
    uint2 texelMin = min(uint2(floor(minUV * mipSize)), mipSize - 1u);
    uint2 texelMax = min(uint2(floor(maxUV * mipSize)), mipSize - 1u);
    texelMinOut = texelMin;
    texelMaxOut = texelMax;
    float farthestOccluder = 0.0;
    if (mip == 0)
    {
        // At mip 0 the projected bounds are at most 4x4 pixels. Inspect every
        // intersected texel so holes inside the rectangle remain conservative.
        for (uint y = texelMin.y; y <= texelMax.y; ++y)
        {
            for (uint x = texelMin.x; x <= texelMax.x; ++x)
                farthestOccluder = max(farthestOccluder,
                                       hzb.Load(int3(uint2(x, y), mip)));
        }
    }
    else
    {
        [unroll]
        for (uint y = 0u; y < 2u; ++y)
        {
            [unroll]
            for (uint x = 0u; x < 2u; ++x)
            {
                uint2 texel = uint2(
                    x == 0u ? texelMin.x : texelMax.x,
                    y == 0u ? texelMin.y : texelMax.y);
                farthestOccluder = max(farthestOccluder,
                                       hzb.Load(int3(texel, mip)));
            }
        }
    }
    // Diagnostic-only sampler probe. Integer loads remain authoritative for
    // the visibility decision because point sampling can resolve a neighbor
    // differently on some MoltenVK paths.
    if ((MaxRefinementDepth & 0x80000000u) != 0u)
    {
        float2 centerUV = (float2(texelMin) + 0.5) / float2(mipSize);
        float samplerProbe = hzb.SampleLevel(hzbSampler, centerUV, mip);
        if (samplerProbe < 0.0)
            farthestOccluder = min(farthestOccluder, samplerProbe);
    }
    farthestOccluderOut = farthestOccluder;
    return clipMin.z <= farthestOccluder + NANITE_DEPTH_EPSILON;
}

bool EvaluateBoundsVisible(
    float3 bboxMin,
    float3 bboxMax,
    float4x4 worldMatrix,
    Texture2D<float> hzb,
    SamplerState hzbSampler,
    out bool frustumVisibleOut,
    out bool hzbVisibleOut,
    out float hzbDepthOut,
    out uint hzbMipOut,
    out uint2 hzbTexelMinOut,
    out uint2 hzbTexelMaxOut)
{
    float4 clipMin;
    float4 clipMax;
    bool clipValid;
    frustumVisibleOut = BBoxIntersectFrustum(
        bboxMin,
        bboxMax,
        worldMatrix,
        clipMin,
        clipMax,
        clipValid);
    if (!frustumVisibleOut)
    {
        hzbVisibleOut = false;
        hzbDepthOut = 1.0;
        hzbMipOut = 0u;
        hzbTexelMinOut = 0u;
        hzbTexelMaxOut = 0u;
        return false;
    }
    if (EnableHZB == 0u || !clipValid)
    {
        hzbVisibleOut = true;
        hzbDepthOut = 1.0;
        hzbMipOut = 0u;
        hzbTexelMinOut = 0u;
        hzbTexelMaxOut = 0u;
        return true;
    }
    // The recovery phase reads the HZB built from this frame's depth, so the
    // current clip rectangle is the correct query area and no reprojection is
    // needed. Reprojecting here would query the wrong region of a same-frame
    // pyramid and reject visible bounds.
    if (CullingPass == NANITE_CULLING_PASS_POST)
    {
        hzbVisibleOut = HZBVisible(
            hzb, hzbSampler, clipMin, clipMax, hzbDepthOut, hzbMipOut,
            hzbTexelMinOut, hzbTexelMaxOut);
        return hzbVisibleOut;
    }
    // The HZB was rendered from the previous view. Project the same static
    // bounds into that view and query the union of both rectangles. If the
    // object was outside the previous frustum or crossed the near plane, the
    // old HZB cannot prove occlusion, so keep it visible.
    float4 previousClipMin;
    float4 previousClipMax;
    bool previousClipValid;
    if (!ProjectBBoxToClip(
            bboxMin,
            bboxMax,
            worldMatrix,
            PreviousViewProjMatrix,
            previousClipMin,
            previousClipMax,
            previousClipValid) || !previousClipValid)
    {
        hzbVisibleOut = true;
        hzbDepthOut = 1.0;
        hzbMipOut = 0u;
        hzbTexelMinOut = 0u;
        hzbTexelMaxOut = 0u;
        return true;
    }
    // Query the previous-frame rectangle only. Unioning it with the current
    // rectangle inflates the query area under camera rotation, often to most of
    // the screen, which pulls in unrelated near geometry and produces
    // frame-to-frame flicker. The near-plane depth still takes the minimum of
    // both frames so a bound that moved closer is not wrongly rejected.
    float4 hzbClipMin = previousClipMin;
    float4 hzbClipMax = previousClipMax;
    hzbClipMin.z = min(clipMin.z, previousClipMin.z);
    hzbVisibleOut = HZBVisible(
        hzb, hzbSampler, hzbClipMin, hzbClipMax, hzbDepthOut, hzbMipOut,
        hzbTexelMinOut, hzbTexelMaxOut);
    return hzbVisibleOut;
}

bool IsBoundsVisible(
    float3 bboxMin,
    float3 bboxMax,
    float4x4 worldMatrix,
    Texture2D<float> hzb,
    SamplerState hzbSampler)
{
    bool frustumVisible;
    bool hzbVisible;
    float hzbDepth;
    uint hzbMip;
    uint2 hzbTexelMin;
    uint2 hzbTexelMax;
    return EvaluateBoundsVisible(
        bboxMin,
        bboxMax,
        worldMatrix,
        hzb,
        hzbSampler,
        frustumVisible,
        hzbVisible,
        hzbDepth,
        hzbMip,
        hzbTexelMin,
        hzbTexelMax);
}

float ProjectedSphereRadiusPixels(float4 localSphere, float4x4 worldMatrix)
{
    float3 axisX = float3(worldMatrix._m00, worldMatrix._m10, worldMatrix._m20);
    float3 axisY = float3(worldMatrix._m01, worldMatrix._m11, worldMatrix._m21);
    float3 axisZ = float3(worldMatrix._m02, worldMatrix._m12, worldMatrix._m22);
    float worldRadius = localSphere.w * max(length(axisX), max(length(axisY), length(axisZ)));
    if ((MaxRefinementDepth & NANITE_SHADOW_PROJECTION_FLAG) != 0u)
    {
        const float pixelsPerNdcX = abs(ViewProjMatrix._m00) * float(ViewportWidth) * 0.5;
        const float pixelsPerNdcY = abs(ViewProjMatrix._m11) * float(ViewportHeight) * 0.5;
        return worldRadius * max(pixelsPerNdcX, pixelsPerNdcY);
    }
    float3 worldCenter = mul(worldMatrix, float4(localSphere.xyz, 1.0)).xyz;
    float distance = max(length(worldCenter - CameraWorldPosition) - worldRadius, 1.0e-5);
    return worldRadius * FocalY * 0.5 * float(ViewportHeight) / distance;
}

float ProjectedErrorPixels(float4 boundSphere, float error, float4x4 worldMatrix)
{
    float3 axisX = float3(worldMatrix._m00, worldMatrix._m10, worldMatrix._m20);
    float3 axisY = float3(worldMatrix._m01, worldMatrix._m11, worldMatrix._m21);
    float3 axisZ = float3(worldMatrix._m02, worldMatrix._m12, worldMatrix._m22);
    float worldScale = max(length(axisX), max(length(axisY), length(axisZ)));
    float worldRadius = boundSphere.w * worldScale;
    if ((MaxRefinementDepth & NANITE_SHADOW_PROJECTION_FLAG) != 0u)
    {
        const float pixelsPerNdcX = abs(ViewProjMatrix._m00) * float(ViewportWidth) * 0.5;
        const float pixelsPerNdcY = abs(ViewProjMatrix._m11) * float(ViewportHeight) * 0.5;
        return error * worldScale * max(pixelsPerNdcX, pixelsPerNdcY);
    }
    float3 worldCenter = mul(worldMatrix, float4(boundSphere.xyz, 1.0)).xyz;
    float distance = max(length(worldCenter - CameraWorldPosition) - worldRadius, 1.0e-5);
    return error * worldScale * FocalY * 0.5 * float(ViewportHeight) / distance;
}

#endif
