// Keep the scalar index/list element named so SPIRV-Cross does not derive a
// Metal struct type named g_IndexBuffer and then hide it with the resource.
struct NaniteRasterIndexElement
{
    uint Value;
};

struct NaniteRasterDrawIndexElement
{
    uint Value;
};

struct NaniteRasterConstants
{
    float4x4 ViewProjMatrix;
    uint ViewportWidth;
    uint ViewportHeight;
    uint DebugMode;
    uint DebugMip;
    float3 CameraWorldPosition;
    float RasterBinAreaCutoff;
    uint HybridRasterEnabled;
    uint ClusterRasterMode;
    uint Padding1;
    uint Padding2;
};

cbuffer NaniteRasterCB : register(b0)
{
    NaniteRasterConstants g_Raster;
};

struct NaniteInstance
{
    float4x4 WorldMatrix;
    float4 LocalBoundSphere;
    float3 BBoxMin;
    float3 BBoxMax;
    uint RootNodeIndex;
    uint Padding0;
    uint Padding1;
    uint Padding2;
};

StructuredBuffer<float3> g_PositionBuffer : register(t1);
// Smooth normals generated on the CPU from the source triangles, indexed exactly
// like the positions. Screen-space derivatives would give a flat normal for free,
// but the faceting would then change with the LOD level and make a level switch
// look worse than it is.
//
// float4 rather than float3 for a MoltenVK reason, not a layout one: the generated
// MSL names the buffer's struct type after the first variable that uses it, so a
// second StructuredBuffer<float3> comes out as `g_PositionBuffer* g_NormalBuffer`
// with the type name already shadowed by the position variable, and the Metal
// compiler rejects it. A distinct element type gets a distinct struct name. The
// stride is 16 bytes either way, which is what Nanite::Position already is.
StructuredBuffer<float4> g_NormalBuffer : register(t5);
StructuredBuffer<NaniteInstance> g_Instances : register(t2);
// The draw is non-indexed, so the index buffer is pulled here instead of being
// bound as an index buffer. That is what lets every cluster share one command:
// the per-cluster index range comes from g_DrawInstanceData rather than from the
// command's FirstIndex field, which only exists once.
StructuredBuffer<NaniteRasterIndexElement> g_IndexBuffer : register(t3);
struct NaniteDrawInstanceData
{
    uint InstanceIndex;
    uint ClusterIndex;
    uint FirstIndex;
    uint IndexCount;
};

StructuredBuffer<NaniteDrawInstanceData> g_DrawInstanceData : register(t4);
Texture2D<float> g_HZB : register(t6);
StructuredBuffer<NaniteRasterDrawIndexElement> g_HardwareDrawIndices : register(t7);
SamplerState g_HZBSampler : register(s8);

struct NaniteRasterVSOutput
{
    float4 Position : SV_Position;
    nointerpolation uint DrawIndex : TEXCOORD0;
    nointerpolation uint ClusterIndex : TEXCOORD1;
    nointerpolation uint InstanceIndex : TEXCOORD2;
    float3 WorldPosition : TEXCOORD3;
    float3 WorldNormal : TEXCOORD4;
    // Index of this triangle within its cluster, and its screen area in pixels.
    // Area is evaluated for the two triangle debug views and for hybrid binning.
    // The hybrid path consumes it in the VS so software triangles are rejected
    // before fixed-function rasterization rather than discarded by the PS.
    nointerpolation uint TriangleIndex : TEXCOORD5;
    nointerpolation float TriangleAreaPixels : TEXCOORD6;
};

struct NaniteRasterPSOutput
{
    float4 Color : SV_Target0;
    float Depth : SV_Target1;
    uint Visibility : SV_Target2;
};

#define NANITE_DEBUG_TRIANGLE_COLORS 6u
#define NANITE_DEBUG_TRIANGLE_SIZE 7u
#define NANITE_DEBUG_OVERDRAW 8u
#define NANITE_DEBUG_RASTER_BIN 9u
#define NANITE_SOFT_MAX_EXTENT 4096
#define NANITE_VIS_TRIANGLE_BITS 8u
#define NANITE_VIS_ENCODE(drawIndex, triangleIndex) \
    (((drawIndex) + 1u) << NANITE_VIS_TRIANGLE_BITS | ((triangleIndex) & 255u))

// Screen area in pixels of the triangle vertexID belongs to, or 0 when any corner
// is behind the eye and the projected area would be meaningless.
float TriangleScreenArea(NaniteDrawInstanceData drawData,
                         float4x4 worldMatrix,
                         uint vertexID,
                         out bool softwareRasterEligible)
{
    softwareRasterEligible = true;
    uint triangleStart = (vertexID / 3u) * 3u;
    float2 corners[3];
    for (uint corner = 0u; corner < 3u; ++corner)
    {
        uint localIndex = triangleStart + corner;
        localIndex = localIndex < drawData.IndexCount ? localIndex : 0u;
        float3 local = g_PositionBuffer[g_IndexBuffer[drawData.FirstIndex + localIndex].Value];
        float4 clip = mul(g_Raster.ViewProjMatrix,
                          float4(mul(worldMatrix, float4(local, 1.0)).xyz, 1.0));
        if (clip.w <= 1.0e-6)
        {
            softwareRasterEligible = false;
            return 0.0;
        }
        // To pixels, so the number is directly comparable to one.
        corners[corner] = clip.xy / clip.w *
            float2(float(g_Raster.ViewportWidth), float(g_Raster.ViewportHeight)) * 0.5;
    }
    float2 edge1 = corners[1] - corners[0];
    float2 edge2 = corners[2] - corners[0];
    const float2 boundsMin = min(min(corners[0], corners[1]), corners[2]);
    const float2 boundsMax = max(max(corners[0], corners[1]), corners[2]);
    const int x0 = max(int(floor(boundsMin.x)), 0);
    const int y0 = max(int(floor(boundsMin.y)), 0);
    const int x1 = min(int(floor(boundsMax.x)), int(g_Raster.ViewportWidth) - 1);
    const int y1 = min(int(floor(boundsMax.y)), int(g_Raster.ViewportHeight) - 1);
    softwareRasterEligible = x0 <= x1 && y0 <= y1 &&
        (x1 - x0) <= NANITE_SOFT_MAX_EXTENT &&
        (y1 - y0) <= NANITE_SOFT_MAX_EXTENT;
    return abs(edge1.x * edge2.y - edge1.y * edge2.x) * 0.5;
}

NaniteRasterVSOutput NaniteVS(uint vertexID : SV_VertexID, uint instanceID : SV_InstanceID)
{
    uint drawIndex = instanceID;
    bool mixedHardwareCluster = false;
    if (g_Raster.HybridRasterEnabled != 0u)
    {
        const uint packedIndex = g_HardwareDrawIndices[instanceID].Value;
        mixedHardwareCluster = (packedIndex & 0x80000000u) != 0u;
        drawIndex = packedIndex & 0x7FFFFFFFu;
    }
    NaniteDrawInstanceData drawData = g_DrawInstanceData[drawIndex];
    uint instanceIndex = drawData.InstanceIndex;
    NaniteInstance instance = g_Instances[instanceIndex];
    // Every instance of the command runs the same fixed vertex count, sized for
    // the largest cluster. A cluster with fewer triangles folds its whole tail
    // onto its first index, so all three corners of each padded triangle land on
    // the same vertex and the rasterizer drops the zero-area result.
    uint localIndex = vertexID < drawData.IndexCount ? vertexID : 0u;
    uint index = g_IndexBuffer[drawData.FirstIndex + localIndex].Value;
    float3 localPosition = g_PositionBuffer[index];
    float3 worldPosition = mul(instance.WorldMatrix, float4(localPosition, 1.0)).xyz;
    // The w of 0 drops the translation. Every instance transform here is a pure
    // translation, so the upper 3x3 is identity and this is exact; a non-uniform
    // scale would need the inverse transpose instead.
    float3 worldNormal = mul(instance.WorldMatrix, float4(g_NormalBuffer[index].xyz, 0.0)).xyz;

    NaniteRasterVSOutput output;
    const float4 clipPosition = mul(g_Raster.ViewProjMatrix, float4(worldPosition, 1.0));
    output.Position = clipPosition;
    output.DrawIndex = drawIndex;
    output.ClusterIndex = drawData.ClusterIndex;
    output.InstanceIndex = instanceIndex;
    output.WorldPosition = worldPosition;
    output.WorldNormal = worldNormal;
    output.TriangleIndex = vertexID / 3u;
    bool softwareRasterEligible = false;
    const bool needsTriangleArea =
        (g_Raster.HybridRasterEnabled != 0u && mixedHardwareCluster) ||
        g_Raster.DebugMode == NANITE_DEBUG_TRIANGLE_SIZE ||
        (g_Raster.DebugMode == NANITE_DEBUG_RASTER_BIN &&
         g_Raster.HybridRasterEnabled == 0u);
    output.TriangleAreaPixels = 0.0;
    if (needsTriangleArea)
    {
        output.TriangleAreaPixels = TriangleScreenArea(
            drawData, instance.WorldMatrix, vertexID, softwareRasterEligible);
    }

    if (g_Raster.HybridRasterEnabled != 0u && mixedHardwareCluster &&
        softwareRasterEligible &&
        output.TriangleAreaPixels <= g_Raster.RasterBinAreaCutoff)
    {
        // Every vertex of this triangle evaluates the same area and therefore
        // lands on this same point beyond the far clip plane. Primitive assembly
        // still sees the submitted triangle, but clipping removes the degenerate
        // primitive before raster setup and no Fragment/PS invocation is created.
        output.Position = float4(0.0, 0.0, 2.0, 1.0);
    }
    return output;
}

NaniteRasterPSOutput NanitePS(NaniteRasterVSOutput input)
{
    // The CPU occlusion diagnostic needs a depth buffer containing only the
    // potential occluders, so omit the tenth bunny for its first frame.
    if (g_Raster.DebugMode == 4u && input.InstanceIndex == 10u)
        discard;

    NaniteRasterPSOutput output;
    output.Depth = input.Position.z;
    output.Visibility = NANITE_VIS_ENCODE(input.DrawIndex, input.TriangleIndex);
    if (g_Raster.DebugMode == NANITE_DEBUG_OVERDRAW)
    {
        // One fixed increment per shaded fragment. The pass that uses this adds
        // instead of replacing and has the depth test off, so the colour target ends
        // up holding a count: how many times each pixel was shaded by the geometry
        // culling let through, occluded layers included. Anything bright is
        // geometry that was submitted and then thrown away by the depth test.
        //
        // The channel weights are unequal so that the sum runs through a ramp
        // rather than just getting brighter: red saturates at 10 layers, green at
        // 18, blue only past 50, so 1 layer is dark brown, 5 is orange, 10 is
        // bright orange, 18 is yellow and white means far more than the scene
        // should ever need. Counts below 10 are the interesting range and that is
        // where the ramp has its resolution.
        output.Color = float4(0.100, 0.055, 0.020, 1.0);
        return output;
    }

    if (g_Raster.DebugMode >= 1u && g_Raster.DebugMode <= 3u)
    {
        // The HZB is padded to a power of two, so the rendered area only covers
        // its top-left corner. Scale screen UVs into that sub-rectangle so the
        // visualization lines up with the image.
        uint hzbWidth;
        uint hzbHeight;
        uint hzbLevels;
        g_HZB.GetDimensions(0u, hzbWidth, hzbHeight, hzbLevels);
        float2 uv = input.Position.xy / float2(hzbWidth, hzbHeight);
        float depth = g_HZB.SampleLevel(g_HZBSampler, uv, g_Raster.DebugMip);
        float value = g_Raster.DebugMode == 3u ?
            saturate((input.Position.z - 0.90) * 10.0) :
            (g_Raster.DebugMode == 2u ? saturate((depth - 0.90) * 10.0) : depth);
        output.Color = float4(value, value, value, 1.0);
        return output;
    }

    // Normal rendering is visibility-only. Material evaluation happens in the
    // fullscreen Visibility Shading pass after hardware/software winners have
    // been resolved, so this PS must not duplicate that work. Hybrid raster-bin
    // and final-depth views are also shaded from the resolved buffers below.
    if (g_Raster.DebugMode == 0u || g_Raster.DebugMode == 10u ||
        g_Raster.DebugMode == 11u ||
        (g_Raster.HybridRasterEnabled != 0u && g_Raster.DebugMode == NANITE_DEBUG_RASTER_BIN))
    {
        output.Color = float4(0.0, 0.0, 0.0, 1.0);
        return output;
    }

    uint value = input.ClusterIndex * 1664525u + 1013904223u;
    float3 clusterColor = float3(
        float((value >> 0u) & 255u),
        float((value >> 8u) & 255u),
        float((value >> 16u) & 255u)) / 255.0;
    if (g_Raster.DebugMode == 5u)
    {
        // The old per-cluster hash colour, kept because it is the only view that
        // shows the cluster layout and the LOD cut directly.
        output.Color = float4(0.25 + clusterColor * 0.75, 1.0);
        return output;
    }

    if (g_Raster.DebugMode == NANITE_DEBUG_TRIANGLE_COLORS)
    {
        // A flat colour per triangle. The point is not the colours but their size:
        // once triangles are pixel-sized the image reads as uniform noise, and the
        // areas that still show recognisable facets are the ones with pixels to
        // spare. The cluster index is mixed in so neighbouring clusters do not
        // repeat the same sequence of colours.
        uint hash = (input.ClusterIndex * 9781u + input.TriangleIndex) * 1664525u + 1013904223u;
        hash ^= hash >> 15u;
        hash *= 2246822519u;
        hash ^= hash >> 13u;
        output.Color = float4(float3(
            float((hash >> 0u) & 255u),
            float((hash >> 8u) & 255u),
            float((hash >> 16u) & 255u)) / 255.0 * 0.85 + 0.15, 1.0);
        return output;
    }

    if (g_Raster.DebugMode == NANITE_DEBUG_RASTER_BIN)
    {
        // Which of the two rasterizers each triangle would be binned to, the way a
        // hybrid path decides it: blue below the area cutoff goes to the software
        // rasterizer, red above it stays with the hardware. When hybrid routing is
        // active, the fullscreen visibility shader redraws these same colours from
        // the merged winners so both bins remain visible.
        //
        // Read next to Triangle size, which uses the same area on a continuous
        // ramp. This one is deliberately two flat colours: the question here is how
        // much of the screen falls on each side of one threshold, and a gradient
        // makes that harder to see, not easier.
        const bool software = input.TriangleAreaPixels <= g_Raster.RasterBinAreaCutoff;
        // Give every triangle its own colour, then bias the palette toward the
        // rasterizer it belongs to. The output stays opaque so HZB visibility changes
        // cannot turn draw order into dark temporal flicker.
        uint hash = (input.ClusterIndex * 9781u + input.TriangleIndex) * 1664525u + 1013904223u;
        hash ^= hash >> 15u;
        hash *= 2246822519u;
        hash ^= hash >> 13u;
        const float3 triangleColor = 0.25 + 0.55 * float3(
            float((hash >> 0u) & 255u),
            float((hash >> 8u) & 255u),
            float((hash >> 16u) & 255u)) / 255.0;
        const float3 offset = software ? float3(0.00, 0.12, 0.38) :
                                         float3(0.38, 0.06, 0.00);
        output.Color = float4(saturate(triangleColor * 0.72 + offset), 1.0);
        return output;
    }

    if (g_Raster.DebugMode == NANITE_DEBUG_TRIANGLE_SIZE)
    {
        // Screen area per triangle, on a log2 ramp centred so that one pixel is the
        // red/yellow boundary: red is a triangle smaller than a pixel, which is work
        // the hardware rasterizer largely throws away, and blue is one big enough
        // that the pixel shader dominates its cost. Anything red is what a coarser
        // LOD or a software rasterizer would win back.
        float area = max(input.TriangleAreaPixels, 0.015625);
        float t = saturate((log2(area) + 6.0) / 14.0);  // 1/64 px^2 .. 256 px^2
        float3 ramp;
        if (t < 0.4286)  // below 1 px^2: black -> red -> yellow
            ramp = lerp(float3(0.35, 0.0, 0.0), float3(1.0, 0.85, 0.0), t / 0.4286);
        else if (t < 0.7143)  // 1 .. 16 px^2: yellow -> green
            ramp = lerp(float3(1.0, 0.85, 0.0), float3(0.1, 0.9, 0.2), (t - 0.4286) / 0.2857);
        else  // above 16 px^2: green -> blue
            ramp = lerp(float3(0.1, 0.9, 0.2), float3(0.1, 0.3, 1.0), (t - 0.7143) / 0.2857);
        output.Color = float4(ramp, 1.0);
        return output;
    }

    // Blinn-Phong against one fixed world-space directional light. The point is to
    // judge LOD popping, and shading is what makes a silhouette or normal change
    // visible at all - a flat cluster colour hides it, because a cluster switch
    // recolours the surface whether or not its geometry moved.
    float3 normal = normalize(input.WorldNormal);
    float3 view = normalize(g_Raster.CameraWorldPosition - input.WorldPosition);
    // Winding is not guaranteed consistent across a scanned mesh, and a back-facing
    // normal would read as an unlit hole rather than as a surface. Fold it forward
    // instead: for judging popping, a wrong-signed normal is noise.
    if (dot(normal, view) < 0.0)
        normal = -normal;
    float3 light = normalize(float3(0.35, 0.65, 0.67));
    float3 halfway = normalize(light + view);
    float diffuse = saturate(dot(normal, light));
    // Specular is gated on the diffuse term so a highlight cannot appear on a face
    // the light does not reach.
    float specular = diffuse > 0.0 ? pow(saturate(dot(normal, halfway)), 48.0) : 0.0;
    float3 albedo = float3(0.72, 0.68, 0.62);
    float3 shaded = albedo * (0.18 + 0.82 * diffuse) + float3(0.3, 0.3, 0.3) * specular;
    output.Color = float4(shaded, 1.0);
    return output;
}
