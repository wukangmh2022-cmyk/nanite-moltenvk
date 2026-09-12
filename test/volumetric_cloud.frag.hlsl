#include "volumetric_cloud_common.hlsl"

struct VolumetricCloudConstants
{
    float4x4 InvViewProj;
    float4 CameraWorldPosition;
    float4 SunDirection;
    float4 SunRadiance;
    float4 CloudBoundsMin;
    float4 CloudBoundsMax;
    float4 CloudLayer;
    float4 MarchParameters;
    float4 NoiseParameters;
    float4 WindTime;
    float4 ScatteringParameters;
    uint4 Viewport;
};

cbuffer VolumetricCloudCB : register(b0)
{
    VolumetricCloudConstants g_Cloud;
};

Texture2D<float4> g_CloudDistribution : register(t0);
Texture3D<float4> g_CloudDetailNoise : register(t1);
Texture2D<float> g_SceneDepth : register(t2);
SamplerState g_LinearSampler : register(s0);
SamplerState g_PointSampler : register(s1);

float2 CloudMapUv(float3 worldPosition)
{
    const float2 boundsMin = g_Cloud.CloudBoundsMin.xz;
    const float2 boundsSize = max(g_Cloud.CloudBoundsMax.xz - boundsMin, 1.0e-4);
    const float2 wind = g_Cloud.WindTime.xz * g_Cloud.WindTime.w;
    return (worldPosition.xz - boundsMin) / boundsSize +
        wind * g_Cloud.NoiseParameters.x;
}

float HeightProfile(float worldY)
{
    const float bottom = g_Cloud.CloudLayer.z;
    const float top = max(g_Cloud.CloudLayer.w, bottom + 1.0e-3);
    const float height = saturate((worldY - bottom) / (top - bottom));

    const float base = smoothstep(0.0, 0.12, height);
    const float anvil = 1.0 - smoothstep(0.88, 1.0, height);
    return base * anvil;
}

float CloudDensityAt(float3 worldPosition, float detailFrequency)
{
    const float3 boundsMin = g_Cloud.CloudBoundsMin.xyz;
    const float3 boundsMax = g_Cloud.CloudBoundsMax.xyz;
    // Only the height band bounds the density. The XZ coordinates wrap with
    // frac() in the map/detail UVs, so the field tiles infinitely and the
    // host can pass large XZ bounds to extend clouds to the horizon instead
    // of stopping at a box wall.
    const float insideY =
        step(boundsMin.y, worldPosition.y) * step(worldPosition.y, boundsMax.y);
    if (insideY <= 0.0)
        return 0.0;

    const float2 mapUv = frac(CloudMapUv(worldPosition) * 1.55);
    const float4 distribution = g_CloudDistribution.SampleLevel(
        g_LinearSampler, saturate(mapUv), 0.0);
    const float macroFbm = distribution.r;
    const float paintedMask = distribution.g;
    const float coverage = saturate(g_Cloud.CloudLayer.x);
    const float threshold = lerp(0.70, 0.54, coverage);
    const float macroMask = smoothstep(threshold - 0.10,
                                       threshold + 0.08,
                                       macroFbm) *
                            lerp(0.78, 1.0,
                                 smoothstep(0.15, 0.78, paintedMask));
    if (macroMask < 0.008)
        return 0.0;

    const float3 windOffset = g_Cloud.WindTime.xyz * g_Cloud.WindTime.w;
    // Shear the uniform advection with a divergence-free low-frequency flow.
    // The 2D curl of a scalar noise field (finite difference through slightly
    // offset samples) advects without stretching the density - a divergent
    // flow swells and compresses the field as it animates, which reads as
    // watercolor smear on the moving clouds.
    const float3 flowP = float3(worldPosition.xz * 0.003 +
        float2(7.0, 3.0) + g_Cloud.WindTime.w * 0.004, 0.6);
    const float s0 = ValueNoise3D(flowP);
    const float sx = ValueNoise3D(flowP + float3(1.0, 0.0, 0.0));
    const float sy = ValueNoise3D(flowP + float3(0.0, 1.0, 0.0));
    // Scale by the wind speed (normalized to the ~8.5 u/s host default) so a
    // zero wind fully freezes the field instead of leaving the shear running.
    const float2 flow = float2(s0 - sy, sx - s0) * 60.0 *
        saturate(length(g_Cloud.WindTime.xyz) * 0.125);
    const float3 advected = worldPosition + windOffset +
        float3(flow * 22.0, 0.0);
    const float detailScale = max(g_Cloud.NoiseParameters.y, 1.0e-4);
    // The 1.25 y-frequency stacks vertical cells into clumps instead of
    // flattening the density into horizontal bands.
    const float3 detailUv = frac(advected /
                                 detailScale * float3(1.0, 1.25, 1.0) *
                                 max(detailFrequency, 0.25));
    const float4 detail = g_CloudDetailNoise.SampleLevel(
        g_LinearSampler, detailUv, 0.0);
    // Micro band re-tiles the same volume at a higher frequency with a fresh
    // offset so the second DensityRemap has genuine high-frequency content.
    // The re-tile is applied to the world UV (advected/detailScale), not the
    // tiled detailUv: a non-integer re-tile of the tiled UV jumps at every
    // volume tile boundary, which shows as a hard vertical seam line.
    const float4 micro = g_CloudDetailNoise.SampleLevel(
        g_LinearSampler,
        frac(advected / detailScale * float3(1.7, 1.25, 1.7) *
             max(detailFrequency, 0.25) + float3(0.23, 0.41, 0.83)), 0.0);
    const float height = saturate((worldPosition.y - g_Cloud.CloudLayer.z) /
                                  max(g_Cloud.CloudLayer.w - g_Cloud.CloudLayer.z, 1.0e-3));
    // Cauliflower crown: a mid-frequency cap band modulates the local top so
    // the silhouette reads as stacked puffs instead of one smooth dome.
    const float crown = g_CloudDetailNoise.SampleLevel(
        g_LinearSampler,
        float3(frac(advected.x / detailScale * 1.35 + 0.19), 0.82,
               frac(advected.z / detailScale * 1.35 + 0.53)), 0.0).r;
    // The local top and base converge toward the coverage edges so the field
    // ends in rounded puffs instead of a uniform vertical slab. A fixed-height
    // slab profile is what reads as "slices" from the side.
    const float coverage01 = smoothstep(threshold - 0.14,
                                        threshold + 0.08, macroFbm);
    const float localTop = lerp(0.30, 0.88, coverage01) +
                           (crown - 0.5) * (0.34 + 0.34 * coverage01);
    const float localBase = lerp(0.09, 0.065, coverage01);
    const float profile = smoothstep(localBase * 0.5, localBase, height) *
        (1.0 - smoothstep(localTop - 0.22, localTop + 0.08, height));
    // Keep the Worley FBM body broad. The curl-warped channel only erodes its
    // boundary, preserving a continuous volume instead of noise islands. The
    // 0.35 floor fills the thin cell-boundary "cracks" of raw 1-worley, which
    // is what reads as honeycomb when carved.
    const float baseShape = pow(max(detail.r, 0.0), 0.60);
    const float billowyErosion = 0.35 + 0.65 * detail.g;
    const float fine = detail.b;
    const float microErosion = micro.a * 0.55 + micro.g * 0.45;
    const float erosionWeight = lerp(0.16, 0.40,
                                     saturate(g_Cloud.NoiseParameters.z));
    // Erosion fades toward the base but never fully switches off, so the
    // underside is shredded and light leaks toward the sunlit interior.
    // Zero base-erosion floor: the very bottom of the cloud layer is not
    // carved, so the condensation base reads as one clean flat line.
    const float detailFade = smoothstep(0.04, 0.14, height);
    const float billowyAmount = billowyErosion * erosionWeight * detailFade;
    const float remapDenominator = max(1.0 - billowyAmount, 0.08);
    const float erodedBody = max((baseShape - billowyAmount) /
                                 remapDenominator, 0.0);
    const float core = smoothstep(0.30, 0.70, erodedBody);
    const float wisps = smoothstep(0.22, 0.62, fine) * (1.0 - core) * 0.18;
    // Second DensityRemap on the micro band. Re-applying the remap sharpens
    // the density distribution so the surface is turbulent, not a smooth
    // puff, which is what gives real clouds their high-frequency look.
    const float microAmount = microErosion *
        lerp(0.08, 0.24, saturate(g_Cloud.NoiseParameters.z)) * detailFade;
    const float microDenominator = max(1.0 - microAmount, 0.08);
    const float erodedMicro = max((max(core, wisps) - microAmount) /
                                  microDenominator, 0.0);
    const float shape = lerp(max(core, wisps), erodedMicro, 0.5);
    // Soft interior variation so the mass has layered detail instead of a
    // uniform 0/1 threshold.
    const float interiorTurb = 1.0 + 0.16 *
        (smoothstep(0.20, 0.80, detail.b * 0.55 + micro.a * 0.45) - 0.5);
    return macroMask * profile * shape * interiorTurb *
        saturate(g_Cloud.CloudLayer.y);
}

float LightOpticalDepth(float3 worldPosition, float3 lightDirection,
                        float detailFrequency)
{
    // Reach ~3x the configured step so the sun-facing side and the shadowed
    // back actually diverge (short marches produce flat lighting).
    const float stepLength = max(g_Cloud.MarchParameters.z, 1.0e-3) * 3.0;
    const int sampleCount = 12;
    float opticalDepth = 0.0;

    [loop]
    for (int i = 0; i < sampleCount; ++i)
    {
        const float t = (float(i) + 0.5) * stepLength;
        const float density = CloudDensityAt(
            worldPosition + lightDirection * t, detailFrequency);
        opticalDepth += density * g_Cloud.ScatteringParameters.x * stepLength * 1.10;
        if (opticalDepth > 12.0)
            break;
    }
    return opticalDepth;
}

float MultiOctaveScatteringFactor(float lightOpticalDepth)
{
    const uint requestedOctaves = min(g_Cloud.Viewport.z, (uint)VC_MAX_MS_OCTAVES);
    const float albedo = saturate(g_Cloud.ScatteringParameters.y);
    float weightedFactor = 0.0;
    float weightSum = 0.0;

    [loop]
    for (uint octave = 0; octave < VC_MAX_MS_OCTAVES; ++octave)
    {
        if (octave >= requestedOctaves)
            break;

        const float attenuation = exp2(-(float)octave * 0.58);
        const float contribution = pow(albedo, (float)octave + 1.0);
        const float weight = exp(-(float)octave * 0.55);
        weightedFactor += exp(-lightOpticalDepth * attenuation) * contribution * weight;
        weightSum += weight;
    }

    return weightedFactor / max(weightSum, 1.0e-4) *
        g_Cloud.ScatteringParameters.w;
}

float OpaqueDistance(float2 pixelPosition, float3 rayDirection)
{
    const float depth = g_SceneDepth.SampleLevel(
        g_PointSampler,
        (pixelPosition + 0.5) / float2(g_Cloud.Viewport.xy),
        0.0);
    if (depth >= 0.999999)
        return 1.0e30;

    const float2 uv = (pixelPosition + 0.5) /
        float2(g_Cloud.Viewport.xy);
    const float2 ndc = float2(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
    const float4 clipPosition = float4(ndc, depth, 1.0);
    const float4 worldPosition = mul(g_Cloud.InvViewProj, clipPosition);
    const float3 world = worldPosition.xyz / max(worldPosition.w, 1.0e-6);
    return max(dot(world - g_Cloud.CameraWorldPosition.xyz, rayDirection), 0.0);
}

float3 ReconstructRay(float2 pixelPosition)
{
    const float2 uv = (pixelPosition + 0.5) /
        float2(g_Cloud.Viewport.xy);
    const float2 ndc = float2(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
    const float4 nearPosition = mul(g_Cloud.InvViewProj, float4(ndc, 0.0, 1.0));
    const float4 farPosition = mul(g_Cloud.InvViewProj, float4(ndc, 1.0, 1.0));
    const float3 nearWorld = nearPosition.xyz / max(nearPosition.w, 1.0e-6);
    const float3 farWorld = farPosition.xyz / max(farPosition.w, 1.0e-6);
    return normalize(farWorld - nearWorld);
}

float4 CloudPS(float4 position : SV_Position) : SV_Target0
{
    const float3 camera = g_Cloud.CameraWorldPosition.xyz;
    const float3 rayDirection = ReconstructRay(position.xy);
    const float3 sunDirection = normalize(g_Cloud.SunDirection.xyz);

    float cloudNear;
    float cloudFar;
    if (!IntersectAabb(camera, rayDirection,
                       g_Cloud.CloudBoundsMin.xyz,
                       g_Cloud.CloudBoundsMax.xyz,
                       cloudNear, cloudFar))
        return float4(0.0, 0.0, 0.0, 0.0);

    const float opaqueDistance = OpaqueDistance(position.xy, rayDirection);
    const float startDistance = max(cloudNear, 0.0);
    const float endDistance = min(
        min(cloudFar, opaqueDistance),
        startDistance + max(g_Cloud.MarchParameters.x, 0.0));
    if (endDistance <= startDistance)
        return float4(0.0, 0.0, 0.0, 0.0);

    const float viewStep = max(g_Cloud.MarchParameters.y, 1.0e-3);
    const uint sampleCount = min(
        (uint)ceil((endDistance - startDistance) / viewStep), 256u);
    const float stepLength = (endDistance - startDistance) /
        max((float)sampleCount, 1.0);

    float transmittance = 1.0;
    float3 radiance = 0.0;
    const float3 viewDirection = -rayDirection;
    const float viewSunDot = dot(viewDirection, sunDirection);
    const float phase = HenyeyGreenstein(
        viewSunDot, g_Cloud.MarchParameters.w);
    const float backwardPhase = HenyeyGreenstein(
        -viewSunDot, g_Cloud.MarchParameters.w);
    const float phaseIso = IsotropicPhase();

    [loop]
    for (uint i = 0; i < 256u; ++i)
    {
        if (i >= sampleCount || transmittance < 0.01)
            break;

        // World-anchored sample jitter: the offset is a hash of the current
        // world position (plus the sample index), not the screen position.
        // A screen-space jitter stays fixed on the screen while the clouds
        // move beneath it, so rotating the camera shimmers the cloud edges.
        const float anchorDistance = startDistance +
            (float(i) + 0.5) * stepLength;
        const float3 anchorPosition = camera + rayDirection * anchorDistance;
        const float sampleJitter =
            Hash31(anchorPosition * 0.9 + float3((float)i * 37.3, 3.1, 11.9)) - 0.5;
        // Low amplitude so the world-anchored hash does not read as per-pixel
        // noise; enough offset remains to break marching-band alignment.
        const float distance = anchorDistance + sampleJitter * stepLength * 0.15;
        const float3 samplePosition = camera + rayDirection * distance;
        const float density = CloudDensityAt(samplePosition, 1.0);
        const float sigmaT = density * g_Cloud.ScatteringParameters.x;
        const float sigmaS = sigmaT * saturate(g_Cloud.ScatteringParameters.y);
        const float alpha = 1.0 - exp(-sigmaT * stepLength);

        if (sigmaT > 1.0e-5)
        {
            const float lightOpticalDepth = LightOpticalDepth(
                samplePosition, sunDirection, 1.0);
            const float lightTransmittance = exp(-lightOpticalDepth * 1.25);
            const float msFactor = MultiOctaveScatteringFactor(
                lightOpticalDepth);
            const float powder = lerp(1.0,
                1.0 - exp(-density * 3.5),
                smoothstep(0.55, -0.15, viewSunDot) * 0.56);
            const float heightFactor = saturate(
                (samplePosition.y - g_Cloud.CloudBoundsMin.y) /
                max(g_Cloud.CloudBoundsMax.y - g_Cloud.CloudBoundsMin.y,
                    1.0e-3));
            const float topLight = smoothstep(0.18, 0.78, heightFactor);
            const float3 singleScattering = g_Cloud.SunRadiance.rgb *
                lightTransmittance * phase * sigmaS * powder *
                lerp(0.45, 1.0, topLight);
            const float3 multipleScattering = g_Cloud.SunRadiance.rgb *
                phaseIso * sigmaS * msFactor;
            // Closed-form geometric-series multiple scattering from the
            // referenced article (p/457997155). It is local - no extra light
            // march - and adds energy where the density is high, so dense
            // cores stay bright. Scale is modest and shadow-aware.
            const float localFms = 0.92 * (1.0 - exp(-2.0 * sigmaT));
            const float localMs = localFms /
                max(1.0 - localFms, 0.05) * 0.08;
            const float3 msArticle = g_Cloud.SunRadiance.rgb *
                phaseIso * sigmaS * localMs * (0.35 + 0.65 * topLight);
            // The underside stays cool and shadowed, but the blue-gray
            // ambient keeps it readable as cloud instead of void.
            const float3 ambientBottom = float3(0.075, 0.115, 0.180);
            const float3 ambientTop = float3(0.32, 0.45, 0.62);
            const float3 ambient = lerp(ambientBottom, ambientTop, topLight) *
                sigmaS * (0.90 + 0.45 * topLight);
            // Silver lining: thin, sunlit cloud at the silhouette scatters
            // toward the viewer. edgeThin is ~1 at the surface and decays in
            // the core; transmittance keeps it strongest at the ray's entry
            // shell. From below this is the light leaking around the crown.
            const float edgeThin = exp(-density * 2.6);
            const float rimLit = pow(max(lightTransmittance, 1.0e-4), 0.16);
            const float rimShell = edgeThin * rimLit *
                pow(max(transmittance, 0.0), 0.5) *
                lerp(0.12, 1.0, topLight);
            const float3 silver = float3(1.30, 1.18, 1.00) * sigmaS *
                rimShell * (phase * 1.4 + backwardPhase * 1.2);
            radiance += transmittance *
                (singleScattering + multipleScattering + msArticle +
                 ambient + silver) * stepLength;
            transmittance *= 1.0 - alpha;
        }
    }

    return float4(radiance, 1.0 - transmittance);
}
