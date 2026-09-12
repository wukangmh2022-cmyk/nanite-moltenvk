# Volumetric cloud test pass

This directory is intentionally independent from the main demo. It contains
the source shaders, a DiligentCore host-side pass helper, and a small CMake
project that compiles only these shaders. No parent CMake file or main-process
source is changed.

## Files

- `cloud_distribution.comp.hlsl` bakes a 2D RGBA16F map. R is continuous
  Worley FBM, G is the coverage mask, and B is a soft edge mask.
- `cloud_detail_3d.comp.hlsl` bakes a 3D RGBA16F volume. R is the billowy
  Worley-FBM body, G is high-frequency curl-warped Worley erosion, B is a fine
  curl/value signal for soft wisps, and A is the micro band used by the second
  DensityRemap in the view shader.
- `generate_cloud_distribution.py` writes a viewable PNG preview using the
  same 2D Worley FBM/hash path as the distribution compute shader. The noise
  is built on a torus (cell indices wrap modulo the period and every octave
  transform uses an integer scale), so the generated PNG is tileable and the
  preview can repeat it with `fract()` without seam lines. The map combines
  low-frequency Worley FBM with a slower value field so neighboring cells form
  connected cloud systems instead of isolated cell centers.
- `cloud_weather3.png` is a reference weather photo kept for comparison. The
  preview intentionally does not sample it: its non-periodic edges produced
  vertical seam lines on the clouds when tiled. The macro coverage comes only
  from the tileable generated Worley FBM map.
- `cloud_preview.html` is a standalone Three.js/WebGL2 page for quickly
  checking the distribution map, separate 3D Worley body and erosion volumes,
  ray marching, single scattering, and multi-scattering controls before
  embedding the pass.
- `lighting_pattern.png` and `tiling_noise_05.png` are UE-style noise assets
  (exported from the reference project's `.uasset` files). The browser preview
  samples `tiling_noise_05` through three planar projections as a high-
  frequency edge-erosion field, and uses the three Worley octaves packed in
  `lighting_pattern` to modulate the scattered energy per-region. They are
  detail textures only; the Vulkan path stays procedural and does not bind
  them.
- `volumetric_cloud.frag.hlsl` performs AABB/height-band ray marching with
  per-sample jitter, height shaping with a cauliflower crown and
  coverage-tapered top/base (rounded puff edges instead of a vertical slab),
  three-band erosion (billowy/fine/micro) with two DensityRemaps, view-to-sun
  single scattering, multi-octave multiple scattering plus the closed-form
  geometric-series approximation from the referenced zhihu article, and a
  silver-lining term that keeps the crown rim visible when the cloud is viewed
  from below. Only the height band bounds the density; XZ wraps periodically,
  so the host can pass large XZ bounds to extend the field to the horizon.
- `VolumetricCloudPass.hpp/.cpp` shows how to create the UAV/SRV textures,
  bake both noise fields, and composite the cloud pass over an existing target.

## Multiple scattering approximation

The implementation uses a cheap multi-octave isotropic approximation:

`T_ms(o) = exp(-lightOD * 2^(-0.8 o))`

`MS = sum(T_ms(o) * albedo^(o+1) * weight(o)) / sum(weight(o)) * strength`

`MultiOctaveScatteringFactor` evaluates progressively deeper light
transmittance octaves and averages them with decreasing weights. This is an
approximation, not a full light transport solve. `Viewport.z` selects 1 to 6
octaves; the view shader also performs a separate local optical-depth march
for single scattering and self-shadowing.

## Compile the shader test

From the repository root:

```sh
cmake -S test -B test/build -G Ninja
cmake --build test/build --parallel
```

Or use the direct script:

```sh
test/compile_shaders.sh test/bin
```

Generate a preview image:

```sh
python3 test/generate_cloud_distribution.py \
    --output test/cloud_distribution_preview.png \
    --raw-output test/cloud_distribution_fbm.png
```

The PNG is an inspection/bake preview. Runtime rendering still uses the GPU
compute path so the same distribution can be regenerated at any resolution.

Run the standalone browser preview from the repository root:

```sh
cd test
python3 -m http.server 8123
```

Open `http://127.0.0.1:8123/cloud_preview.html`. The page loads both the raw
FBM PNG and the thresholded coverage PNG, then generates independent 64^3 body
and 80^3 erosion textures in the browser. No cloud photograph is used as a
surface material. The body volume carries the connected cloud mass; its RGBA
channels are separate base-shape bands. The erosion volume carries independent
wispy, billowy, high-frequency curl-warped Worley-FBM, and fine micro channels.
The preview uses a
full-resolution optical-depth march
for self-shadowing and multi-octave isotropic scattering. Its default
quick-check profile is a camera above the cloud layer, looking toward the
horizon, with 96 view steps, 8 light steps, and 3 multiple-scattering
octaves. The camera deck also provides a below-cloud horizon view and an
external side-volume profile view. The browser body is a continuous 3D field:
low-frequency Worley/value FBM and randomized 3D puffs establish the connected
mass, independent high-frequency curl-warped channels add billows, wisps, and
boundary cavities,
and the 2D tileable Worley FBM map only modulates coverage. The density uses a
cauliflower crown, coverage-tapered top and base so the field ends in rounded
puffs instead of a vertical slab, three erosion bands at increasing tiling
(billowy, fine, micro), a second DensityRemap that sharpens the distribution
so the surface is turbulent instead of a smooth puff, a soft interior
variation for layered detail, and a cheap one-octave
analytic micro detail evaluated only in the view march so the bake resolution
does not bound the high-frequency look. The field is periodic in XZ (3D
volumes tile at ~280x340 world units, the 2D coverage map at ~800), and only
the height band bounds it, so the clouds extend to the horizon instead of
ending at a box wall. Near clouds use the 3D volumes plus an analytic
aperiodic micro detail; past ~340 units an analytic FBM far field takes over
(no visible repetition) and everything fades into haze by ~1400 units. The 3D
body and erosion volumes are baked from per-axis periodic noise (the lattice
indices wrap modulo the axis period, and every band scale is an integer), so
the texture is seamless at every repeat boundary - a hard value jump at the
wrap was the source of the persistent vertical thin lines every ~280 units.
The wind shear flow is the 2D curl of a scalar noise field, which is
divergence-free: it advects the density without stretching it, so moving
clouds no longer smear like watercolor (a divergent flow swells and compresses
the field as it animates). The density thresholds use wider smoothstep bands
and the cauliflower crown gain is reduced, so cloud-top gaps stay soft instead
of geometric. Seam verification: the 3D volume bake is numerically verified
seamless in all three axes (the wrapped function value differs by < 1e-6
across every repeat boundary), and the 2D coverage/mask maps measure only
natural per-texel gradients at their wraps (the thresholded mask's "seam" is
the same magnitude as its own cell edges). The browser preview HUD shows a
`build rN` tag so a screenshot can confirm whether a fresh page (and therefore
the periodic bake) is actually running.
Vertical-line audit (round 10): every candidate source was measured
numerically. The 3D volume bake is seamless in all three axes (wrap diff
< 1e-6). All four 2D textures (coverage map, thresholded mask, TilingNoise05,
lighting pattern) have wrap seams within their own natural per-texel gradients
(the lighting pattern's combined RGB value is exactly continuous at its wrap).
An offline port of the density field, ocean shadow, and lighting-pattern
energy fields finds no thin full-height vertical line at either camera X=0 or
X=+50; the only center-screen feature is a broad cloud silhouette that moves
with the camera (world-space). The sole screen-position-dependent terms are
the per-pixel jitter hashes (per-pixel grain, not a single line), and
`uResolution` is declared but never used. The only world-space plane that can
project as a straight through-camera line is X=0 - exactly the volume-tile
boundary that the periodic bake made continuous - so a still-visible line on
build r10 would point at a stale page running the pre-periodic bake.
The
erosion bands fill the low end of the inverted-Worley signal and use modest
strengths so thin cell boundaries do not carve into a honeycomb, and the two
UE-style assets are sampled at large scales as gentle modulators (soft fluff
multiplier, narrow lighting range) instead of hard carvers. The
view march keeps near-camera steps below
1.2 world units and grows them with distance (up to 256 samples per ray), with
a small per-sample jitter, so the cloud
sides do not show marching bands ("slices") and the residual noise is not
grainy. Two UE-style noise assets add a soft edge texture and a per-region
lighting variation on top of the procedural field. The browser preview uses
a low, warm sun with a lighter self-shadow reach (the
light march steps ~1.1 world units instead of 0.34 so the sun-facing side and
the shadowed back actually diverge), a vertical sky gradient with a warm
horizon, top-biased direct sunlight,
blue-gray ambient light strong enough that the underside reads as cloud, a
silver-lining term that keeps the sunlit crown rim visible when viewed from
below, and an 8-step local optical-depth march so the
underside stays shadowed while the upper cloud remains bright. The `3D body
slice` view shows the final density field before ray marching; `3D erosion
slice` shows the generated high-frequency volume directly. The Vulkan path
remains the source of truth for the final GPU implementation.

Image sharpness audit (round 11): the browser preview renders into a canvas
backing store that is a fraction of the CSS display size and lets the browser
upscale it with bilinear filtering - at `renderScale 0.55` that is a 1.8x
upscale and smears the whole frame (the "watercolor" look). The default is now
0.8 (1.25x upscale); `?scale=1` forces native resolution for a still check,
`?scale=0.55` restores the old speed. There is no bloom/DOF or other
post-processing pass (only ACES tone mapping), so no post blur can be the
cause. The body 48^3 volume spans 280x340 world units in XZ, giving ~6-7-unit
voxel spacing (erosion 64^3 ~4-5 units, re-tiled to ~2.6 in the shader); the
analytic micro detail (2-4 m) and the re-tiled erosion bands compensate the
near-field interpolation softness, but raising the bake resolution is the
remaining lever if the base shape still reads as too soft up close. Round 12
bumps the browser bakes to 64^3 body + 80^3 erosion (~3.5 s bake, shown as
"Baking noise volumes...") - the Vulkan path already bakes 128^3 on the GPU.
Screen-space jitter audit (round 13): the march jitter hashes use only
`vUv` (screen position), no time or world position, so the grain is welded to
the screen when the camera orbits - the hash field is isotropic white noise
(no ordered-dither diagonal structure). The preview has no TAA, so a temporal
offset would only add flicker; the fix is amplitude reduction (rayJitter
0.75->0.25, sampleJitter 0.15->0.10 in GLSL, 0.25->0.15 in HLSL). A diagonal
light band in the wide view is not from the jitter pattern (isotropy measured
at ~1.0) and is most plausibly marching Moire (step size vs density frequency
beating) or the near/far crossfade projection; it needs a rendered frame to
pin down.
Vertical-line root cause (round 14): the periodic bake made the volume
textures seamless, but the shader re-tiled the *tiled* UV by non-integer
factors (crown 0.64x/1.35x/1.90x, erosion bands 0.63x/1.31x/1.79x, light
detail 1.70x). `frac(tiledUv * m)` with non-integer m jumps at every volume
tile boundary (X=0/280, Z=0/340) - measured 0.09-0.24 channel jumps at X=0
(the camera plane) and X=280. The camera deck sits at X=0, so this was the
persistent through-camera vertical line, and the static field stripes at
wind=0. Fixed by computing every re-tile UV from the world position at its own
frequency, so all wraps land on the texture's own period; an offline render
confirms the center line is gone (the old 4.2x column jump drops to normal
cloud-edge variation). This round also rebalances cost: light steps default
8->5 with an adaptive step length (constant ~9-unit shadow reach), the march
step-growth rate scales with the view-steps slider (so lowering steps actually
reduces samples), and renderScale defaults to 0.65.
Round 15 feedback pass: raising renderScale did not change the perceived blur
(the blur comes from the far-field analytic FBM being ~140-unit features with
a flat threshold, not from the upscale), so renderScale is back at 0.55; the
far field is instead sharpened directly (frequency 0.007->0.011, threshold
0.46/0.78->0.43/0.70). The cloud underside was reading as gouged/scraped: the
base-erosion floor drops 0.26->0.12 and the wispy carving weight is reduced
(mix 0.07/0.26 -> 0.05/0.18), keeping a clean flat condensation base. The
reduced jitter from the grain pass re-exposed marching bands on flat cloud
bottoms, so the jitter is rebalanced to rayJitter 0.30 / sampleJitter 0.16 -
between the grainy 0.75 and the banded 0.10.
Round 16: cloud edges shimmered while rotating the camera because the march
jitter was a screen-UV hash - fixed on the screen while the clouds moved under
it (the classic without-TAA temporal shimmer). The jitter is now anchored to
the current world position (hash of the sample anchor plus the sample index),
so the pattern moves with the clouds and stays stable under rotation. The
cloud underside still read as gouged, so the base-erosion floor is now zero
(smoothstep 0.04->0.14) - the very bottom of the cloud layer is not carved at
all and reads as one clean flat condensation line.
Round 17: the world-anchored jitter at 0.5x step was too strong - the hash
jumps every ~1 world unit, so it read as heavy per-pixel noise/shaking on the
cloud surfaces. The amplitude is cut to 0.15x step (matching the spatial level
of the earlier screen-space version) while keeping the world anchoring that
removed the camera-rotation shimmer.

The latest verification captures are written to
`cloud_preview_reference_style_final.png`,
`cloud_preview_corrected_erosion_below.png`, and
`cloud_preview_erosion_raw_v4.png`.

The host helper expects these four SPIR-V files in the shader directory passed
to `VolumetricCloudPass`:

```text
CloudDistribution.comp.spv
CloudDetail3D.comp.spv
VolumetricCloud.vert.spv
VolumetricCloud.frag.spv
```

The render shader expects three SRVs (`g_CloudDistribution`,
`g_CloudDetailNoise`, `g_SceneDepth`) and two samplers. The depth SRV must use
Vulkan's normalized depth range; use a cleared depth texture with value 1 when
there is no opaque scene to clip against. The graphics PSO uses premultiplied
alpha blending, so it can be placed after opaque rendering and before
transparent rendering.
