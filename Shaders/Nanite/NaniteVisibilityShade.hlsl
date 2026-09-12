#include "NaniteCommon.hlsl"

#define NANITE_SOFT_MAX_EXTENT 4096
#define NANITE_PI 3.14159265359

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
    float4 CameraRight;
    float4 CameraUp;
    float4 CameraForward;
    float4 Focal;
    uint Padding1;
    uint Padding2;
    float4 SkySunDirection;
    float4 SkyAtmosphere;
    float4 SkyPerspective;
    float4x4 ShadowViewProjMatrix;
    float4 VsmPageInfo;
};

cbuffer NaniteRasterCB : register(b0)
{
    NaniteRasterConstants g_Raster;
};

StructuredBuffer<float3> g_PositionBuffer : register(t1);
StructuredBuffer<float4> g_NormalBuffer : register(t2);
StructuredBuffer<NaniteInstance> g_Instances : register(t3);
StructuredBuffer<NaniteGeometryIndexElement> g_IndexBuffer : register(t4);
StructuredBuffer<NaniteDrawInstanceData> g_DrawInstanceData : register(t5);
Texture2D<uint> g_FinalVisibility : register(t6);
Texture2D<float> g_FinalDepth : register(t7);
StructuredBuffer<NaniteQueueState> g_QueueState : register(t8);
StructuredBuffer<NaniteMaterial> g_Materials : register(t9);
Texture2DArray<float4> g_AlbedoTexture : register(t10);
StructuredBuffer<float2> g_UvBuffer : register(t11);
struct NaniteMaterialIndexElement
{
    uint Value;
};
StructuredBuffer<NaniteMaterialIndexElement> g_MaterialIndexBuffer : register(t12);
Texture2DArray<float4> g_NormalTexture : register(t13);
Texture2DArray<float4> g_MetallicRoughnessTexture : register(t14);
Texture2D<float> g_ShadowMap : register(t15);
StructuredBuffer<uint> g_VsmPageTable : register(t16);
Texture2DArray<float> g_VsmPhysicalPool : register(t17);
SamplerState g_AlbedoSampler : register(s9);
SamplerState g_ShadowSampler : register(s10);
SamplerState g_VsmShadowSampler : register(s11);

struct FullscreenOutput
{
    float4 Position : SV_Position;
};

FullscreenOutput NaniteVisibilityVS(uint vertexID : SV_VertexID)
{
    const float2 positions[3] = {
        float2(-1.0, -1.0),
        float2(-1.0,  3.0),
        float2( 3.0, -1.0),
    };
    FullscreenOutput output;
    output.Position = float4(positions[vertexID], 0.0, 1.0);
    return output;
}

float DepthVisualization(float depth)
{
    if (depth >= 1.0)
        return 0.0;
    const float nearPlane = 0.1;
    const float farPlane = 100.0;
    const float viewDepth = nearPlane * farPlane /
        max(farPlane - depth * (farPlane - nearPlane), 1.0e-6);
    return 1.0 - saturate(log2(max(viewDepth / nearPlane, 1.0)) /
                           log2(farPlane / nearPlane));
}

float3 FresnelSchlick(float cosTheta, float3 f0)
{
    return f0 + (1.0 - f0) * pow(1.0 - saturate(cosTheta), 5.0);
}

float DistributionGGX(float3 normal, float3 halfway, float roughness)
{
    const float alpha = roughness * roughness;
    const float alphaSquared = alpha * alpha;
    const float nDotH = saturate(dot(normal, halfway));
    const float denominator = nDotH * nDotH * (alphaSquared - 1.0) + 1.0;
    return alphaSquared / max(NANITE_PI * denominator * denominator, 1.0e-5);
}

float GeometrySchlickGGX(float nDotDirection, float roughness)
{
    const float k = (roughness + 1.0) * (roughness + 1.0) / 8.0;
    return nDotDirection / max(nDotDirection * (1.0 - k) + k, 1.0e-5);
}

float GeometrySmith(float3 normal, float3 view, float3 light, float roughness)
{
    return GeometrySchlickGGX(saturate(dot(normal, view)), roughness) *
           GeometrySchlickGGX(saturate(dot(normal, light)), roughness);
}

float ShadowVisibility(float3 worldPosition, float3 normal, float3 lightDirection)
{
    const float4 lightClip = mul(
        g_Raster.ShadowViewProjMatrix, float4(worldPosition, 1.0));
    if (lightClip.w <= 1.0e-6)
        return 1.0;

    const float3 lightNdc = lightClip.xyz / lightClip.w;
    if (lightNdc.x < -1.0 || lightNdc.x > 1.0 ||
        lightNdc.y < -1.0 || lightNdc.y > 1.0 ||
        lightNdc.z < 0.0 || lightNdc.z > 1.0)
        return 1.0;

    uint shadowWidth;
    uint shadowHeight;
    g_ShadowMap.GetDimensions(shadowWidth, shadowHeight);
    const float2 texelSize = 1.0 / float2(shadowWidth, shadowHeight);
    const float slope = 1.0 - saturate(dot(normal, lightDirection));
    const float bias = 0.0025 + 0.0060 * slope;
    const float2 uv = float2(
        lightNdc.x * 0.5 + 0.5,
        0.5 - lightNdc.y * 0.5);

    if (g_Raster.VsmPageInfo.w > 0.5)
    {
        const uint pageCount = max((uint)(g_Raster.VsmPageInfo.x + 0.5), 1u);
        const uint2 page = min((uint2)(uv * float(pageCount)), pageCount - 1u);
        const uint tableIndex = page.y * pageCount + page.x;
        const uint physicalEntry = g_VsmPageTable[tableIndex];
        // A page that has not reached the small update budget is deliberately
        // treated as lit. This avoids stale/black holes while the cache warms.
        if (physicalEntry == 0u)
            return 1.0;

        const uint physicalPage = physicalEntry - 1u;
        const float2 localUv = frac(uv * float(pageCount));
        const float2 pageTexel = g_Raster.VsmPageInfo.z.xx;
        float visible = 0.0;
        [unroll]
        for (int y = -1; y <= 1; ++y)
        {
            [unroll]
            for (int x = -1; x <= 1; ++x)
            {
                const float2 sampleUv = clamp(
                    localUv + float2(x, y) * pageTexel,
                    0.5 * pageTexel,
                    1.0 - 0.5 * pageTexel);
                const float mapDepth = g_VsmPhysicalPool.SampleLevel(
                    g_VsmShadowSampler,
                    float3(sampleUv, float(physicalPage)),
                    0.0);
                visible += mapDepth + bias >= lightNdc.z ? 1.0 : 0.0;
            }
        }
        return visible / 9.0;
    }

    // A small manual PCF kernel keeps the shadow map a normal sampled depth
    // resource, which is portable across the Vulkan and MoltenVK backends and
    // leaves the comparison operation visible for a future cascade sampler.
    float visible = 0.0;
    [unroll]
    for (int y = -1; y <= 1; ++y)
    {
        [unroll]
        for (int x = -1; x <= 1; ++x)
        {
            const float2 sampleUv = uv + float2(x, y) * texelSize;
            const float mapDepth = g_ShadowMap.SampleLevel(
                g_ShadowSampler, sampleUv, 0.0);
            visible += mapDepth + bias >= lightNdc.z ? 1.0 : 0.0;
        }
    }
    return visible / 9.0;
}

float2 RaySphereIntersection(float3 origin, float3 direction, float3 center, float radius)
{
    const float3 localOrigin = origin - center;
    const float b = dot(localOrigin, direction);
    const float c = dot(localOrigin, localOrigin) - radius * radius;
    const float discriminant = b * b - c;
    if (discriminant < 0.0)
        return float2(-1.0, -1.0);
    const float root = sqrt(discriminant);
    return float2(-b - root, -b + root);
}

float3 ProceduralAtmosphere(
    float3 direction,
    float3 cameraPosition,
    float3 sunDirection,
    float planetRadius,
    float atmosphereHeight,
    float rayleighScale,
    float mieScale)
{
    const float3 planetCenter = float3(0.0, -planetRadius, 0.0);
    const float3 cameraFromCenter = cameraPosition - planetCenter;
    const float cameraRadius = max(length(cameraFromCenter), planetRadius + 1.0e-3);
    const float3 planetUp = cameraFromCenter / cameraRadius;
    const float cameraAltitude = max(cameraRadius - planetRadius, 0.0);

    // The tangent direction of a spherical planet gives a curved horizon. The
    // radius is deliberately expressed in scene units so it can later be
    // replaced by the same planet/terrain scale used by cascade shadows.
    const float horizonCosine = -sqrt(saturate(
        1.0 - (planetRadius * planetRadius) / (cameraRadius * cameraRadius)));
    const float viewUp = dot(direction, planetUp);
    const float horizonBlend = smoothstep(
        horizonCosine - 0.045, horizonCosine + 0.075, viewUp);

    // Start with the broad sky gradient used by the preview. The integration
    // below follows the Atmosphere sample's Rayleigh/Mie model, but keeps the
    // work inside this existing background pass: no extra render target, LUT,
    // or epipolar post-process is needed for this first scattering version.
    const float zenith = saturate(viewUp * 0.5 + 0.5);
    const float altitudeFade = exp(-cameraAltitude / max(atmosphereHeight, 1.0));
    const float3 horizonColor = float3(0.58, 0.73, 0.82);
    const float3 zenithColor = float3(0.12, 0.31, 0.70);
    const float3 baseSky = lerp(horizonColor, zenithColor, smoothstep(0.10, 0.92, zenith));

    const float sunCosine = dot(direction, sunDirection);
    const float rayleighPhase = 3.0 / (16.0 * NANITE_PI) *
        (1.0 + sunCosine * sunCosine);
    const float mieG = 0.76;
    const float mieGFactor = 3.0 * (1.0 - mieG * mieG) /
        (2.0 * (2.0 + mieG * mieG));
    const float mieDenominator = max(
        1.0 + mieG * mieG - 2.0 * mieG * sunCosine, 1.0e-3);
    const float miePhase = mieGFactor * (1.0 + sunCosine * sunCosine) /
        (4.0 * NANITE_PI * pow(mieDenominator, 1.5));

    const float rayleighScaleHeight = max(atmosphereHeight * 0.10, 1.0);
    const float mieScaleHeight = max(atmosphereHeight * 0.015, 0.5);
    // The coefficients are normalized to this compact demo world. Their ratios
    // match the sample's physical values; scaling by atmosphereHeight keeps the
    // optical depth stable if the scene is resized later.
    const float compactWorldScale = g_Raster.SkyPerspective.x;
    const float3 rayleighBeta = float3(0.11, 0.30, 0.88) /
        max(atmosphereHeight, 1.0) * rayleighScale * compactWorldScale;
    const float3 mieBeta = float3(0.045, 0.045, 0.045) /
        max(atmosphereHeight, 1.0) * mieScale * compactWorldScale;

    float3 sky = baseSky * (0.82 + 0.18 * altitudeFade);
    const float topAtmosphereRadius = planetRadius + atmosphereHeight;
    const float2 topIntersection = RaySphereIntersection(
        cameraPosition, direction, planetCenter, topAtmosphereRadius);
    float rayEnd = topIntersection.y;
    const float2 groundIntersection = RaySphereIntersection(
        cameraPosition, direction, planetCenter, planetRadius);
    if (groundIntersection.x > 1.0e-3)
        rayEnd = min(rayEnd, groundIntersection.x);

    if (rayEnd > 1.0e-3 && viewUp > horizonCosine - 0.02)
    {
        const uint scatteringSteps = 8u;
        const float stepLength = rayEnd / float(scatteringSteps);
        float2 viewOpticalDepth = float2(0.0, 0.0);
        float3 inscattering = float3(0.0, 0.0, 0.0);
        // The compact demo atmosphere uses scene-unit coefficients, so the
        // source radiance is raised enough for the single-scattering term to
        // remain visible beside the display-referred terrain.
        const float3 sunRadiance = float3(22.0, 20.0, 18.0);
        [unroll]
        for (uint sampleIndex = 0u; sampleIndex < scatteringSteps; ++sampleIndex)
        {
            const float sampleDistance = (float(sampleIndex) + 0.5) * stepLength;
            const float3 samplePosition = cameraPosition + direction * sampleDistance;
            const float3 sampleUp = normalize(samplePosition - planetCenter);
            const float sampleAltitude = max(
                length(samplePosition - planetCenter) - planetRadius, 0.0);
            const float2 density = float2(
                exp(-sampleAltitude / rayleighScaleHeight),
                exp(-sampleAltitude / mieScaleHeight));

            // The sun-side integral uses the same exponential atmosphere model,
            // with the spherical exit distance limiting the flat-atmosphere
            // approximation near the horizon.
            const float sunCosine = dot(sampleUp, sunDirection);
            const float2 sunIntersection = RaySphereIntersection(
                samplePosition, sunDirection, planetCenter, topAtmosphereRadius);
            const float sunExit = max(sunIntersection.y, 0.0);
            const float sunVisibility = smoothstep(-0.10, 0.12, sunCosine);
            const float2 sunOpticalDepth = density * float2(
                min(sunExit, rayleighScaleHeight / max(sunCosine, 0.08)),
                min(sunExit, mieScaleHeight / max(sunCosine, 0.08))) * sunVisibility;
            const float3 totalOpticalDepth =
                rayleighBeta * (viewOpticalDepth.x + sunOpticalDepth.x) +
                mieBeta * (viewOpticalDepth.y + sunOpticalDepth.y);
            const float3 transmittance = exp(-totalOpticalDepth);
            const float3 differentialScattering =
                density.x * rayleighBeta * rayleighPhase +
                density.y * mieBeta * miePhase;
            inscattering += differentialScattering * transmittance *
                sunRadiance * stepLength;
            viewOpticalDepth += density * stepLength;
        }

        const float3 viewTransmittance = exp(-rayleighBeta * viewOpticalDepth.x -
                                             mieBeta * viewOpticalDepth.y);
        sky = baseSky * viewTransmittance + inscattering;
    }

    // Keep the sun as a small directional-light cue. A broad, bright disc reads
    // like a billboard in this wide mountain view and overwhelms the haze.
    const float sunDisc = pow(saturate(sunCosine), 32768.0) * 1.25;
    const float sunGlow = pow(saturate(sunCosine), 96.0) * 0.10;
    sky += float3(1.0, 0.78, 0.52) * (sunGlow + sunDisc);

    // Rays below the tangent are filled with a desaturated atmospheric floor so
    // gaps between distant peaks reveal the same spherical horizon instead of a
    // hard rectangular sky cutoff.
    const float3 belowHorizon = float3(0.22, 0.31, 0.36) *
        (0.55 + 0.45 * saturate((viewUp - horizonCosine) * 4.0));
    return lerp(belowHorizon, sky, horizonBlend);
}

float3 ApplyAtmosphericPerspective(
    float3 sceneColor,
    float3 cameraPosition,
    float3 worldPosition,
    float3 viewDirection,
    float3 sunDirection,
    float planetRadius,
    float atmosphereHeight,
    float rayleighScale,
    float mieScale)
{
    const float viewDistance = length(worldPosition - cameraPosition);
    if (viewDistance <= 1.0e-3)
        return sceneColor;

    const float3 planetCenter = float3(0.0, -planetRadius, 0.0);
    const float rayleighScaleHeight = max(atmosphereHeight * 0.10, 1.0);
    const float mieScaleHeight = max(atmosphereHeight * 0.015, 0.5);
    const float compactWorldScale = g_Raster.SkyPerspective.y;
    const float3 rayleighBeta = float3(0.11, 0.30, 0.88) /
        max(atmosphereHeight, 1.0) * rayleighScale * compactWorldScale;
    const float3 mieBeta = float3(0.045, 0.045, 0.045) /
        max(atmosphereHeight, 1.0) * mieScale * compactWorldScale;

    const float sunCosine = dot(viewDirection, sunDirection);
    const float rayleighPhase = 3.0 / (16.0 * NANITE_PI) *
        (1.0 + sunCosine * sunCosine);
    const float mieG = 0.76;
    const float mieGFactor = 3.0 * (1.0 - mieG * mieG) /
        (2.0 * (2.0 + mieG * mieG));
    const float mieDenominator = max(
        1.0 + mieG * mieG - 2.0 * mieG * sunCosine, 1.0e-3);
    const float miePhase = mieGFactor * (1.0 + sunCosine * sunCosine) /
        (4.0 * NANITE_PI * pow(mieDenominator, 1.5));

    // Four fixed samples are enough for the short compact-world segment used
    // by this demo. The full Atmosphere sample refines hundreds of epipolar
    // samples because it operates on a separate full-screen post-process.
    const uint perspectiveSteps = 4u;
    const float stepLength = viewDistance / float(perspectiveSteps);
    float2 viewOpticalDepth = float2(0.0, 0.0);
    float3 inscattering = float3(0.0, 0.0, 0.0);
    const float3 sunRadiance = float3(22.0, 20.0, 18.0);
    const float topAtmosphereRadius = planetRadius + atmosphereHeight;
    [unroll]
    for (uint sampleIndex = 0u; sampleIndex < perspectiveSteps; ++sampleIndex)
    {
        const float sampleDistance = (float(sampleIndex) + 0.5) * stepLength;
        const float3 samplePosition = cameraPosition + viewDirection * sampleDistance;
        const float3 sampleUp = normalize(samplePosition - planetCenter);
        const float sampleAltitude = max(
            length(samplePosition - planetCenter) - planetRadius, 0.0);
        const float2 density = float2(
            exp(-sampleAltitude / rayleighScaleHeight),
            exp(-sampleAltitude / mieScaleHeight));
        const float sunCosineAtSample = dot(sampleUp, sunDirection);
        const float2 sunIntersection = RaySphereIntersection(
            samplePosition, sunDirection, planetCenter, topAtmosphereRadius);
        const float sunExit = max(sunIntersection.y, 0.0);
        const float sunVisibility = smoothstep(-0.10, 0.12, sunCosineAtSample);
        const float2 sunOpticalDepth = density * float2(
            min(sunExit, rayleighScaleHeight / max(sunCosineAtSample, 0.08)),
            min(sunExit, mieScaleHeight / max(sunCosineAtSample, 0.08))) * sunVisibility;
        const float3 totalOpticalDepth =
            rayleighBeta * (viewOpticalDepth.x + sunOpticalDepth.x) +
            mieBeta * (viewOpticalDepth.y + sunOpticalDepth.y);
        const float3 transmittance = exp(-totalOpticalDepth);
        const float3 differentialScattering =
            density.x * rayleighBeta * rayleighPhase +
            density.y * mieBeta * miePhase;
        inscattering += differentialScattering * transmittance *
            sunRadiance * stepLength;
        viewOpticalDepth += density * stepLength;
    }

    const float3 viewTransmittance = exp(-rayleighBeta * viewOpticalDepth.x -
                                         mieBeta * viewOpticalDepth.y);
    const float sunHaze = pow(saturate(sunCosine), 4.0);
    const float3 horizonHaze = float3(0.42, 0.56, 0.68);
    const float3 sunlitHaze = float3(0.95, 0.72, 0.46);
    const float3 hazeColor = lerp(horizonHaze, sunlitHaze, sunHaze * 0.72);
    // This compact scene has a much shorter camera-to-mountain distance than
    // the physical Earth scale used by the sample. Add a scene-scale distance
    // term so the effect remains visible before a full LUT provides the exact
    // optical depth. Light removed from the mountain is replaced by air light;
    // it must not be allowed to collapse the distant terrain to black.
    const float opticalHaze = dot(
        1.0 - viewTransmittance, float3(0.2126, 0.7152, 0.0722));
    // The authored terrain is compact compared with the physical atmosphere.
    // Feed the terrain optical scale into the distance term as well, otherwise
    // changing Terrain density only affects the very small integrated optical
    // depth and appears inert on this scene.
    const float terrainDensity = max(g_Raster.SkyPerspective.y, 0.0);
    const float distanceHaze = 1.0 - exp(-viewDistance * terrainDensity /
        max(atmosphereHeight * g_Raster.SkyPerspective.z * 0.12, 1.0));
    const float hazeAmount = saturate(max(
        opticalHaze, distanceHaze * g_Raster.SkyPerspective.w));
    return lerp(sceneColor, hazeColor, hazeAmount) + inscattering * 0.18;
}

float4 NaniteVisibilityPS(float4 position : SV_Position) : SV_Target0
{
    const uint2 pixel = uint2(position.xy);
    const float depth = g_FinalDepth.Load(int3(pixel, 0));
    const float4 background = float4(0.035, 0.045, 0.065, 1.0);
    if (g_Raster.DebugMode == 10u)
    {
        const float value = DepthVisualization(depth);
        return float4(value, value, value, 1.0);
    }

    const uint payload = g_FinalVisibility.Load(int3(pixel, 0));
    if (payload == 0u)
    {
        // Reconstruct the view ray from pixel-space focal lengths. The sky is
        // procedural so its horizon can follow the spherical world model.
        const float2 centered = float2(pixel) + 0.5 -
            float2(g_Raster.Focal.z, g_Raster.Focal.w);
        const float3 direction = normalize(
            g_Raster.CameraForward.xyz +
            g_Raster.CameraRight.xyz * (centered.x / g_Raster.Focal.x) -
            g_Raster.CameraUp.xyz * (centered.y / g_Raster.Focal.y));
        const float3 sky = ProceduralAtmosphere(
            direction,
            g_Raster.CameraWorldPosition,
            normalize(g_Raster.SkySunDirection.xyz),
            g_Raster.SkyAtmosphere.x,
            g_Raster.SkyAtmosphere.y,
            g_Raster.SkyAtmosphere.z,
            g_Raster.SkyAtmosphere.w);
        return float4(saturate(sky), 1.0);
    }

    const uint drawIndex = NANITE_VIS_DRAW_INDEX(payload);
    const uint triangleIndex = NANITE_VIS_TRIANGLE_INDEX(payload);
    // The resolve payload is produced by two GPU queues. Treat a stale or torn
    // payload as background before indexing the draw table; an unchecked index
    // can fetch arbitrary geometry and was the source of the transient magenta
    // debug output when the cutoff crossed zero.
    if (drawIndex >= g_QueueState[0].DrawCount)
        return background;
    const NaniteDrawInstanceData drawData = g_DrawInstanceData[drawIndex];
    const NaniteInstance instance = g_Instances[drawData.InstanceIndex];
    const uint firstCorner = triangleIndex * 3u;
    if (firstCorner + 2u >= drawData.IndexCount)
        return background;

    float4 clip[3];
    float3 world[3];
    float3 normal[3];
    float2 uv[3];
    [unroll]
    for (uint corner = 0u; corner < 3u; ++corner)
    {
        const uint index = g_IndexBuffer[drawData.FirstIndex + firstCorner + corner].Value;
        world[corner] = mul(instance.WorldMatrix, float4(g_PositionBuffer[index], 1.0)).xyz;
        normal[corner] = mul(instance.WorldMatrix, float4(g_NormalBuffer[index].xyz, 0.0)).xyz;
        uv[corner] = g_UvBuffer[index];
        clip[corner] = mul(g_Raster.ViewProjMatrix, float4(world[corner], 1.0));
    }

    const float2 widthHeight = float2(g_Raster.ViewportWidth, g_Raster.ViewportHeight);
    const float2 p0 = float2(clip[0].x / clip[0].w * 0.5 + 0.5,
                             0.5 - clip[0].y / clip[0].w * 0.5) * widthHeight;
    const float2 p1 = float2(clip[1].x / clip[1].w * 0.5 + 0.5,
                             0.5 - clip[1].y / clip[1].w * 0.5) * widthHeight;
    const float2 p2 = float2(clip[2].x / clip[2].w * 0.5 + 0.5,
                             0.5 - clip[2].y / clip[2].w * 0.5) * widthHeight;
    const float area = (p1.x - p0.x) * (p2.y - p0.y) - (p1.y - p0.y) * (p2.x - p0.x);
    if (abs(area) < 1.0e-9)
        return background;

    if (g_Raster.DebugMode == 9u)
    {
        // Mirror the actual RasterBin eligibility, not just the area threshold.
        // Near-plane triangles, triangles outside the viewport, and triangles
        // larger than the compute rasterizer's scan cap stay on hardware even
        // when their projected area is below the cutoff.
        bool softwareEligible = clip[0].w > 1.0e-6 &&
                                clip[1].w > 1.0e-6 &&
                                clip[2].w > 1.0e-6;
        const float2 boundsMin = min(min(p0, p1), p2);
        const float2 boundsMax = max(max(p0, p1), p2);
        const int x0 = max(int(floor(boundsMin.x)), 0);
        const int y0 = max(int(floor(boundsMin.y)), 0);
        const int x1 = min(int(floor(boundsMax.x)), int(g_Raster.ViewportWidth) - 1);
        const int y1 = min(int(floor(boundsMax.y)), int(g_Raster.ViewportHeight) - 1);
        softwareEligible = softwareEligible && x0 <= x1 && y0 <= y1 &&
            (x1 - x0) <= NANITE_SOFT_MAX_EXTENT &&
            (y1 - y0) <= NANITE_SOFT_MAX_EXTENT;
        const bool software = softwareEligible &&
            abs(area) * 0.5 <= g_Raster.RasterBinAreaCutoff;
        uint hash = (drawData.ClusterIndex * 9781u + triangleIndex) * 1664525u + 1013904223u;
        hash ^= hash >> 15u;
        hash *= 2246822519u;
        hash ^= hash >> 13u;
        const float3 triangleColor = 0.25 + 0.55 * float3(
            float((hash >> 0u) & 255u),
            float((hash >> 8u) & 255u),
            float((hash >> 16u) & 255u)) / 255.0;
        const float3 offset = software ? float3(0.00, 0.12, 0.38) :
                                         float3(0.38, 0.06, 0.00);
        return float4(saturate(triangleColor * 0.72 + offset), 1.0);
    }

    const float2 sample = float2(pixel) + 0.5;
    const float e0 = (p1.x - p0.x) * (sample.y - p0.y) - (p1.y - p0.y) * (sample.x - p0.x);
    const float e1 = (p2.x - p1.x) * (sample.y - p1.y) - (p2.y - p1.y) * (sample.x - p1.x);
    const float e2 = (p0.x - p2.x) * (sample.y - p2.y) - (p0.y - p2.y) * (sample.x - p2.x);
    float3 weights = float3(e1, e2, e0) / area;
    weights /= float3(clip[0].w, clip[1].w, clip[2].w);
    weights /= max(weights.x + weights.y + weights.z, 1.0e-8);

    float3 worldPosition = world[0] * weights.x + world[1] * weights.y + world[2] * weights.z;
    float3 worldNormal = normalize(normal[0] * weights.x + normal[1] * weights.y + normal[2] * weights.z);
    float3 view = normalize(g_Raster.CameraWorldPosition - worldPosition);
    if (dot(worldNormal, view) < 0.0)
        worldNormal = -worldNormal;

    if (g_Raster.DebugMode == 11u)
    {
        // Resolve the material through a side-band index stream. The visibility
        // payload remains only draw/triangle identity; the winning triangle's
        // first corner carries the imported glTF material id in parallel.
        const uint materialIndex = g_MaterialIndexBuffer[
            drawData.FirstIndex + firstCorner].Value;
        const NaniteMaterial material = g_Materials[materialIndex];
        const float2 importedUv = uv[0] * weights.x + uv[1] * weights.y + uv[2] * weights.z;
        const float2 sampleUv = g_Raster.Padding1 != 0u ? importedUv :
            frac(worldPosition.xz * 0.08 + worldNormal.xy * 0.17);
        const float3 sampledAlbedo = g_AlbedoTexture.SampleLevel(
            g_AlbedoSampler, float3(sampleUv, material.TextureIndex), 0.0).rgb;
        const float3 albedo = max(material.BaseColor.rgb * sampledAlbedo, 0.001);
        const float4 sampledNormal = g_NormalTexture.SampleLevel(
            g_AlbedoSampler, float3(sampleUv, material.TextureIndex), 0.0);
        const float4 sampledMetallicRoughness = g_MetallicRoughnessTexture.SampleLevel(
            g_AlbedoSampler, float3(sampleUv, material.TextureIndex), 0.0);
        float3 shadingNormal = worldNormal;
        if (g_Raster.Padding1 != 0u)
        {
            const float3 edge1 = world[1] - world[0];
            const float3 edge2 = world[2] - world[0];
            const float2 deltaUv1 = uv[1] - uv[0];
            const float2 deltaUv2 = uv[2] - uv[0];
            const float determinant = deltaUv1.x * deltaUv2.y -
                                      deltaUv1.y * deltaUv2.x;
            if (abs(determinant) > 1.0e-6)
            {
                const float inverseDeterminant = 1.0 / determinant;
                const float3 tangent = normalize(
                    (edge1 * deltaUv2.y - edge2 * deltaUv1.y) * inverseDeterminant);
                const float3 bitangent = normalize(cross(worldNormal, tangent));
                // The texture decode flips V so the JPG's top-left origin lands
                // at UV (0,1), which albedo and metallic-roughness expect. A
                // normal map cannot be flipped like that: the flip reverses the
                // tangent-space Y channel, pointing every surface normal away
                // from the light and leaving only the tiny ambient term. That
                // is what made the dark rock pieces render pure black, so undo
                // the Y reversal here before constructing the tangent frame.
                const float3 tangentNormal = normalize(
                    float3(sampledNormal.x, 1.0 - sampledNormal.y, sampledNormal.z) * 2.0 - 1.0);
                shadingNormal = normalize(tangent * tangentNormal.x +
                                           bitangent * tangentNormal.y +
                                           worldNormal * tangentNormal.z);
            }
        }
        const float metallic = saturate(material.Metallic * sampledMetallicRoughness.b);
        const float roughness = clamp(material.Roughness * sampledMetallicRoughness.g,
                                      0.045, 1.0);
        const float ao = saturate(material.AmbientOcclusion * sampledMetallicRoughness.r);
        const float3 light = normalize(g_Raster.SkySunDirection.xyz);
        const float3 halfway = normalize(view + light);
        const float nDotL = saturate(dot(shadingNormal, light));
        const float nDotV = saturate(dot(shadingNormal, view));
        const float3 f0 = lerp(float3(0.04, 0.04, 0.04), albedo, metallic);
        const float3 fresnel = FresnelSchlick(saturate(dot(halfway, view)), f0);
        const float distribution = DistributionGGX(shadingNormal, halfway, roughness);
        const float geometry = GeometrySmith(shadingNormal, view, light, roughness);
        const float3 specular = distribution * geometry * fresnel /
                                max(4.0 * nDotV * nDotL, 1.0e-4);
        const float3 diffuseWeight = (1.0 - fresnel) * (1.0 - metallic);
        const float3 radiance = float3(3.2, 3.0, 2.8);
        const float shadow = ShadowVisibility(worldPosition, shadingNormal, light);
        const float3 direct = (diffuseWeight * albedo / NANITE_PI + specular) *
            radiance * nDotL * shadow;
        const float3 ambient = albedo * (0.025 * ao);
        const float3 color = ambient + direct;
        const float3 atmosphereColor = ApplyAtmosphericPerspective(
            color,
            g_Raster.CameraWorldPosition,
            worldPosition,
            normalize(worldPosition - g_Raster.CameraWorldPosition),
            light,
            g_Raster.SkyAtmosphere.x,
            g_Raster.SkyAtmosphere.y,
            g_Raster.SkyAtmosphere.z,
            g_Raster.SkyAtmosphere.w);
        // Keep the result in the same display-referred range as the existing
        // debug shading while retaining the full PBR lighting cost above.
        return float4(pow(max(atmosphereColor, 0.0), 1.0 / 2.2), 1.0);
    }

    const float3 light = normalize(g_Raster.SkySunDirection.xyz);
    const float3 halfway = normalize(light + view);
    const float diffuse = saturate(dot(worldNormal, light));
    const float specular = diffuse > 0.0 ? pow(saturate(dot(worldNormal, halfway)), 48.0) : 0.0;
    const float3 albedo = float3(0.72, 0.68, 0.62);
    const float shadow = ShadowVisibility(worldPosition, worldNormal, light);
    const float3 shaded = albedo * (0.18 + 0.82 * diffuse * shadow) +
        float3(0.3, 0.3, 0.3) * specular * shadow;
    const float3 atmosphereColor = ApplyAtmosphericPerspective(
        shaded,
        g_Raster.CameraWorldPosition,
        worldPosition,
        normalize(worldPosition - g_Raster.CameraWorldPosition),
        light,
        g_Raster.SkyAtmosphere.x,
        g_Raster.SkyAtmosphere.y,
        g_Raster.SkyAtmosphere.z,
        g_Raster.SkyAtmosphere.w);
    return float4(atmosphereColor, 1.0);
}
