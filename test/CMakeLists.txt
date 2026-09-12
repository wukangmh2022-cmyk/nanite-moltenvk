cmake_minimum_required(VERSION 3.24)

project(VolumetricCloudShaderTest LANGUAGES NONE)

find_program(GLSLANG_VALIDATOR glslangValidator REQUIRED)

set(CLOUD_SHADER_DIR "${CMAKE_CURRENT_LIST_DIR}")
set(CLOUD_SHADER_BIN_DIR "${CMAKE_CURRENT_BINARY_DIR}/bin")
file(MAKE_DIRECTORY "${CLOUD_SHADER_BIN_DIR}")

function(add_cloud_shader output stage entry source)
    add_custom_command(
        OUTPUT "${CLOUD_SHADER_BIN_DIR}/${output}"
        COMMAND "${GLSLANG_VALIDATOR}" -V -D --target-env vulkan1.1
                -I"${CLOUD_SHADER_DIR}" -S "${stage}" -e "${entry}"
                -o "${CLOUD_SHADER_BIN_DIR}/${output}"
                "${CLOUD_SHADER_DIR}/${source}"
        DEPENDS
            "${CLOUD_SHADER_DIR}/${source}"
            "${CLOUD_SHADER_DIR}/volumetric_cloud_common.hlsl"
        VERBATIM)
endfunction()

add_cloud_shader(CloudDistribution.comp.spv comp main cloud_distribution.comp.hlsl)
add_cloud_shader(CloudDetail3D.comp.spv comp main cloud_detail_3d.comp.hlsl)
add_cloud_shader(VolumetricCloud.vert.spv vert FullscreenVS volumetric_cloud.vert.hlsl)
add_cloud_shader(VolumetricCloud.frag.spv frag CloudPS volumetric_cloud.frag.hlsl)

add_custom_target(VolumetricCloudShaders ALL
    DEPENDS
        "${CLOUD_SHADER_BIN_DIR}/CloudDistribution.comp.spv"
        "${CLOUD_SHADER_BIN_DIR}/CloudDetail3D.comp.spv"
        "${CLOUD_SHADER_BIN_DIR}/VolumetricCloud.vert.spv"
        "${CLOUD_SHADER_BIN_DIR}/VolumetricCloud.frag.spv")

