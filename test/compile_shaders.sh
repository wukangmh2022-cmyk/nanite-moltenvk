#!/bin/sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
VALIDATOR=${GLSLANG_VALIDATOR:-glslangValidator}
OUTPUT_DIR=${1:-"$SCRIPT_DIR/bin"}
mkdir -p "$OUTPUT_DIR"

"$VALIDATOR" -V -D --target-env vulkan1.1 \
    -I"$SCRIPT_DIR" -S comp -e main \
    -o "$OUTPUT_DIR/CloudDistribution.comp.spv" \
    "$SCRIPT_DIR/cloud_distribution.comp.hlsl"

"$VALIDATOR" -V -D --target-env vulkan1.1 \
    -I"$SCRIPT_DIR" -S comp -e main \
    -o "$OUTPUT_DIR/CloudDetail3D.comp.spv" \
    "$SCRIPT_DIR/cloud_detail_3d.comp.hlsl"

"$VALIDATOR" -V -D --target-env vulkan1.1 \
    -S vert -e FullscreenVS \
    -o "$OUTPUT_DIR/VolumetricCloud.vert.spv" \
    "$SCRIPT_DIR/volumetric_cloud.vert.hlsl"

"$VALIDATOR" -V -D --target-env vulkan1.1 \
    -I"$SCRIPT_DIR" -S frag -e CloudPS \
    -o "$OUTPUT_DIR/VolumetricCloud.frag.spv" \
    "$SCRIPT_DIR/volumetric_cloud.frag.hlsl"

echo "Cloud SPIR-V written to $OUTPUT_DIR"
