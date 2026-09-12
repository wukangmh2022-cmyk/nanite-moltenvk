cmake_minimum_required(VERSION 3.24)

project(DiligentCoreVulkanDemo LANGUAGES CXX)

if(APPLE)
    enable_language(OBJCXX)
endif()

set(DILIGENT_CORE_DIR "${CMAKE_CURRENT_LIST_DIR}/../DiligentCore" CACHE PATH
    "Path to the DiligentCore checkout")
set(DILIGENT_CORE_BUILD_DIR "${DILIGENT_CORE_DIR}/build-vulkan-github" CACHE PATH
    "Path to the configured DiligentCore build")
option(DILIGENT_CORE_USE_SOURCE
    "Build and link DiligentCore from DILIGENT_CORE_DIR (recommended on Windows)"
    OFF)
set(MESHOPTIMIZER_DIR "${CMAKE_CURRENT_LIST_DIR}/../References/meshoptimizer" CACHE PATH
    "Path to the meshoptimizer checkout")

if(NOT EXISTS "${MESHOPTIMIZER_DIR}/CMakeLists.txt")
    message(FATAL_ERROR "meshoptimizer checkout not found: ${MESHOPTIMIZER_DIR}")
endif()

set(MESHOPT_BUILD_DEMO OFF CACHE BOOL "" FORCE)
set(MESHOPT_BUILD_GLTFPACK OFF CACHE BOOL "" FORCE)
set(MESHOPT_BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
add_subdirectory("${MESHOPTIMIZER_DIR}" "${CMAKE_CURRENT_BINARY_DIR}/meshoptimizer" EXCLUDE_FROM_ALL)

if(WIN32 AND DILIGENT_CORE_USE_SOURCE)
    set(DILIGENT_NO_DIRECT3D11 ON CACHE BOOL "" FORCE)
    set(DILIGENT_NO_DIRECT3D12 ON CACHE BOOL "" FORCE)
    set(DILIGENT_NO_OPENGL ON CACHE BOOL "" FORCE)
    set(DILIGENT_NO_METAL ON CACHE BOOL "" FORCE)
    set(DILIGENT_NO_WEBGPU ON CACHE BOOL "" FORCE)
    add_subdirectory("${DILIGENT_CORE_DIR}"
                     "${CMAKE_CURRENT_BINARY_DIR}/DiligentCore"
                     EXCLUDE_FROM_ALL)
    find_package(Vulkan REQUIRED)
endif()

find_program(GLSLANG_VALIDATOR glslangValidator REQUIRED)

set(SHADER_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/Shaders")
set(SHADER_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/shaders")
file(MAKE_DIRECTORY "${SHADER_BINARY_DIR}")

set(VERTEX_SHADER "${SHADER_BINARY_DIR}/triangle.vert.spv")
set(FRAGMENT_SHADER "${SHADER_BINARY_DIR}/triangle.frag.spv")

set(NANITE_SHADER_DIR "${SHADER_SOURCE_DIR}/Nanite")
set(NANITE_SHADER_BINARY_DIR "${SHADER_BINARY_DIR}/Nanite")
file(MAKE_DIRECTORY "${NANITE_SHADER_BINARY_DIR}")

set(NANITE_INSTANCE_CULL_SHADER "${NANITE_SHADER_BINARY_DIR}/NaniteInstanceCull.comp.spv")
set(NANITE_PERSISTENT_CULL_SHADER "${NANITE_SHADER_BINARY_DIR}/NanitePersistentCull.comp.spv")
set(NANITE_GENERATE_INDIRECT_SHADER "${NANITE_SHADER_BINARY_DIR}/NaniteGenerateIndirect.comp.spv")
set(NANITE_POST_CULL_SHADER "${NANITE_SHADER_BINARY_DIR}/NanitePostCull.comp.spv")
set(NANITE_POST_SEED_SHADER "${NANITE_SHADER_BINARY_DIR}/NanitePostSeed.comp.spv")
set(NANITE_PREPARE_DISPATCH_SHADER "${NANITE_SHADER_BINARY_DIR}/NanitePrepareDispatch.comp.spv")
set(NANITE_RASTER_BIN_SHADER "${NANITE_SHADER_BINARY_DIR}/NaniteRasterBin.comp.spv")
set(NANITE_HIZ_BUILD_SHADER "${NANITE_SHADER_BINARY_DIR}/NaniteHizBuild.comp.spv")
set(NANITE_RASTER_VERTEX_SHADER "${NANITE_SHADER_BINARY_DIR}/NaniteRaster.vert.spv")
set(NANITE_RASTER_PIXEL_SHADER "${NANITE_SHADER_BINARY_DIR}/NaniteRaster.frag.spv")
set(NANITE_SOFT_RASTER_SHADER "${NANITE_SHADER_BINARY_DIR}/NaniteSoftRasterDepth.comp.spv")
set(NANITE_SOFT_VISIBILITY_SHADER "${NANITE_SHADER_BINARY_DIR}/NaniteSoftRasterVisibility.comp.spv")
set(NANITE_SOFT_CHECK_SHADER "${NANITE_SHADER_BINARY_DIR}/NaniteSoftDepthCheck.comp.spv")
set(NANITE_VISIBILITY_RESOLVE_SHADER "${NANITE_SHADER_BINARY_DIR}/NaniteResolveVisibility.comp.spv")
set(NANITE_VISIBILITY_SHADE_VERTEX_SHADER "${NANITE_SHADER_BINARY_DIR}/NaniteVisibilityShade.vert.spv")
set(NANITE_VISIBILITY_SHADE_PIXEL_SHADER "${NANITE_SHADER_BINARY_DIR}/NaniteVisibilityShade.frag.spv")

add_custom_command(
    OUTPUT "${NANITE_INSTANCE_CULL_SHADER}"
    COMMAND "${GLSLANG_VALIDATOR}" -V -D --target-env vulkan1.1
            -I"${NANITE_SHADER_DIR}" -S comp -e main
            -o "${NANITE_INSTANCE_CULL_SHADER}"
            "${NANITE_SHADER_DIR}/NaniteInstanceCull.hlsl"
    DEPENDS
        "${NANITE_SHADER_DIR}/NaniteCommon.hlsl"
        "${NANITE_SHADER_DIR}/NaniteInstanceCull.hlsl"
    VERBATIM
)

add_custom_command(
    OUTPUT "${NANITE_PERSISTENT_CULL_SHADER}"
    COMMAND "${GLSLANG_VALIDATOR}" -V -D --target-env vulkan1.1
            -I"${NANITE_SHADER_DIR}" -S comp -e main
            -o "${NANITE_PERSISTENT_CULL_SHADER}"
            "${NANITE_SHADER_DIR}/NanitePersistentCull.hlsl"
    DEPENDS
        "${NANITE_SHADER_DIR}/NaniteCommon.hlsl"
        "${NANITE_SHADER_DIR}/NanitePersistentCull.hlsl"
    VERBATIM
)

add_custom_command(
    OUTPUT "${NANITE_GENERATE_INDIRECT_SHADER}"
    COMMAND "${GLSLANG_VALIDATOR}" -V -D --target-env vulkan1.1
            -I"${NANITE_SHADER_DIR}" -S comp -e main
            -o "${NANITE_GENERATE_INDIRECT_SHADER}"
            "${NANITE_SHADER_DIR}/NaniteGenerateIndirect.hlsl"
    DEPENDS
        "${NANITE_SHADER_DIR}/NaniteCommon.hlsl"
        "${NANITE_SHADER_DIR}/NaniteGenerateIndirect.hlsl"
    VERBATIM
)

add_custom_command(
    OUTPUT "${NANITE_HIZ_BUILD_SHADER}"
    COMMAND "${GLSLANG_VALIDATOR}" -V -D --target-env vulkan1.1
            -I"${NANITE_SHADER_DIR}" -S comp -e main
            -o "${NANITE_HIZ_BUILD_SHADER}"
            "${NANITE_SHADER_DIR}/NaniteHizBuild.hlsl"
    DEPENDS "${NANITE_SHADER_DIR}/NaniteHizBuild.hlsl"
    VERBATIM
)

add_custom_command(
    OUTPUT "${NANITE_POST_CULL_SHADER}"
    COMMAND "${GLSLANG_VALIDATOR}" -V -D --target-env vulkan1.1
            -I"${NANITE_SHADER_DIR}" -S comp -e main
            -o "${NANITE_POST_CULL_SHADER}"
            "${NANITE_SHADER_DIR}/NanitePostCull.hlsl"
    DEPENDS
        "${NANITE_SHADER_DIR}/NaniteCommon.hlsl"
        "${NANITE_SHADER_DIR}/NanitePostCull.hlsl"
    VERBATIM
)

add_custom_command(
    OUTPUT "${NANITE_POST_SEED_SHADER}"
    COMMAND "${GLSLANG_VALIDATOR}" -V -D --target-env vulkan1.1
            -I"${NANITE_SHADER_DIR}" -S comp -e main
            -o "${NANITE_POST_SEED_SHADER}"
            "${NANITE_SHADER_DIR}/NanitePostSeed.hlsl"
    DEPENDS
        "${NANITE_SHADER_DIR}/NaniteCommon.hlsl"
        "${NANITE_SHADER_DIR}/NanitePostSeed.hlsl"
    VERBATIM
)

add_custom_command(
    OUTPUT "${NANITE_PREPARE_DISPATCH_SHADER}"
    COMMAND "${GLSLANG_VALIDATOR}" -V -D --target-env vulkan1.1
            -I"${NANITE_SHADER_DIR}" -S comp -e main
            -o "${NANITE_PREPARE_DISPATCH_SHADER}"
            "${NANITE_SHADER_DIR}/NanitePrepareDispatch.hlsl"
    DEPENDS
        "${NANITE_SHADER_DIR}/NaniteCommon.hlsl"
        "${NANITE_SHADER_DIR}/NanitePrepareDispatch.hlsl"
    VERBATIM
)

add_custom_command(
    OUTPUT "${NANITE_RASTER_BIN_SHADER}"
    COMMAND "${GLSLANG_VALIDATOR}" -V -D --target-env vulkan1.1
            -I"${NANITE_SHADER_DIR}" -S comp -e main
            -o "${NANITE_RASTER_BIN_SHADER}"
            "${NANITE_SHADER_DIR}/NaniteRasterBin.hlsl"
    DEPENDS
        "${NANITE_SHADER_DIR}/NaniteCommon.hlsl"
        "${NANITE_SHADER_DIR}/NaniteRasterBin.hlsl"
    VERBATIM
)

add_custom_command(
    OUTPUT "${NANITE_RASTER_VERTEX_SHADER}"
    COMMAND "${GLSLANG_VALIDATOR}" -V -D --target-env vulkan1.1
            -S vert -e NaniteVS
            -o "${NANITE_RASTER_VERTEX_SHADER}"
            "${NANITE_SHADER_DIR}/NaniteRaster.hlsl"
    DEPENDS "${NANITE_SHADER_DIR}/NaniteRaster.hlsl"
    VERBATIM
)

add_custom_command(
    OUTPUT "${NANITE_RASTER_PIXEL_SHADER}"
    COMMAND "${GLSLANG_VALIDATOR}" -V -D --target-env vulkan1.1
            -S frag -e NanitePS
            -o "${NANITE_RASTER_PIXEL_SHADER}"
            "${NANITE_SHADER_DIR}/NaniteRaster.hlsl"
    DEPENDS "${NANITE_SHADER_DIR}/NaniteRaster.hlsl"
    VERBATIM
)

add_custom_command(
    OUTPUT "${VERTEX_SHADER}"
    COMMAND "${GLSLANG_VALIDATOR}" -V --target-env vulkan1.1
            -S vert -o "${VERTEX_SHADER}"
            "${SHADER_SOURCE_DIR}/triangle.vert"
    DEPENDS "${SHADER_SOURCE_DIR}/triangle.vert"
    VERBATIM
)

add_custom_command(
    OUTPUT "${FRAGMENT_SHADER}"
    COMMAND "${GLSLANG_VALIDATOR}" -V --target-env vulkan1.1
            -S frag -o "${FRAGMENT_SHADER}"
            "${SHADER_SOURCE_DIR}/triangle.frag"
    DEPENDS "${SHADER_SOURCE_DIR}/triangle.frag"
    VERBATIM
)

add_custom_command(
    OUTPUT "${NANITE_SOFT_RASTER_SHADER}"
    COMMAND "${GLSLANG_VALIDATOR}" -V -D --target-env vulkan1.1
            -DNANITE_SOFT_VISIBILITY_PASS=0
            -I"${NANITE_SHADER_DIR}" -S comp -e main
            -o "${NANITE_SOFT_RASTER_SHADER}"
            "${NANITE_SHADER_DIR}/NaniteSoftRasterDepth.hlsl"
    DEPENDS
        "${NANITE_SHADER_DIR}/NaniteCommon.hlsl"
        "${NANITE_SHADER_DIR}/NaniteSoftRasterDepth.hlsl"
    VERBATIM
)

add_custom_command(
    OUTPUT "${NANITE_SOFT_VISIBILITY_SHADER}"
    COMMAND "${GLSLANG_VALIDATOR}" -V -D --target-env vulkan1.1
            -DNANITE_SOFT_VISIBILITY_PASS=1
            -I"${NANITE_SHADER_DIR}" -S comp -e main
            -o "${NANITE_SOFT_VISIBILITY_SHADER}"
            "${NANITE_SHADER_DIR}/NaniteSoftRasterDepth.hlsl"
    DEPENDS
        "${NANITE_SHADER_DIR}/NaniteCommon.hlsl"
        "${NANITE_SHADER_DIR}/NaniteSoftRasterDepth.hlsl"
    VERBATIM
)

add_custom_command(
    OUTPUT "${NANITE_SOFT_CHECK_SHADER}"
    COMMAND "${GLSLANG_VALIDATOR}" -V -D --target-env vulkan1.1
            -I"${NANITE_SHADER_DIR}" -S comp -e main
            -o "${NANITE_SOFT_CHECK_SHADER}"
            "${NANITE_SHADER_DIR}/NaniteSoftDepthCheck.hlsl"
    DEPENDS
        "${NANITE_SHADER_DIR}/NaniteCommon.hlsl"
        "${NANITE_SHADER_DIR}/NaniteSoftDepthCheck.hlsl"
    VERBATIM
)

add_custom_command(
    OUTPUT "${NANITE_VISIBILITY_RESOLVE_SHADER}"
    COMMAND "${GLSLANG_VALIDATOR}" -V -D --target-env vulkan1.1
            -I"${NANITE_SHADER_DIR}" -S comp -e main
            -o "${NANITE_VISIBILITY_RESOLVE_SHADER}"
            "${NANITE_SHADER_DIR}/NaniteResolveVisibility.hlsl"
    DEPENDS
        "${NANITE_SHADER_DIR}/NaniteCommon.hlsl"
        "${NANITE_SHADER_DIR}/NaniteResolveVisibility.hlsl"
    VERBATIM
)

add_custom_command(
    OUTPUT "${NANITE_VISIBILITY_SHADE_VERTEX_SHADER}"
    COMMAND "${GLSLANG_VALIDATOR}" -V -D --target-env vulkan1.1
            -I"${NANITE_SHADER_DIR}" -S vert -e NaniteVisibilityVS
            -o "${NANITE_VISIBILITY_SHADE_VERTEX_SHADER}"
            "${NANITE_SHADER_DIR}/NaniteVisibilityShade.hlsl"
    DEPENDS
        "${NANITE_SHADER_DIR}/NaniteCommon.hlsl"
        "${NANITE_SHADER_DIR}/NaniteVisibilityShade.hlsl"
    VERBATIM
)

add_custom_command(
    OUTPUT "${NANITE_VISIBILITY_SHADE_PIXEL_SHADER}"
    COMMAND "${GLSLANG_VALIDATOR}" -V -D --target-env vulkan1.1
            -I"${NANITE_SHADER_DIR}" -S frag -e NaniteVisibilityPS
            -o "${NANITE_VISIBILITY_SHADE_PIXEL_SHADER}"
            "${NANITE_SHADER_DIR}/NaniteVisibilityShade.hlsl"
    DEPENDS
        "${NANITE_SHADER_DIR}/NaniteCommon.hlsl"
        "${NANITE_SHADER_DIR}/NaniteVisibilityShade.hlsl"
    VERBATIM
)

add_custom_target(DiligentCoreVulkanShaders DEPENDS
    "${VERTEX_SHADER}"
    "${FRAGMENT_SHADER}"
    "${NANITE_INSTANCE_CULL_SHADER}"
    "${NANITE_PERSISTENT_CULL_SHADER}"
    "${NANITE_GENERATE_INDIRECT_SHADER}"
    "${NANITE_POST_CULL_SHADER}"
    "${NANITE_POST_SEED_SHADER}"
    "${NANITE_PREPARE_DISPATCH_SHADER}"
    "${NANITE_RASTER_BIN_SHADER}"
    "${NANITE_HIZ_BUILD_SHADER}"
    "${NANITE_RASTER_VERTEX_SHADER}"
    "${NANITE_RASTER_PIXEL_SHADER}"
    "${NANITE_SOFT_RASTER_SHADER}"
    "${NANITE_SOFT_VISIBILITY_SHADER}"
    "${NANITE_SOFT_CHECK_SHADER}"
    "${NANITE_VISIBILITY_RESOLVE_SHADER}"
    "${NANITE_VISIBILITY_SHADE_VERTEX_SHADER}"
    "${NANITE_VISIBILITY_SHADE_PIXEL_SHADER}")

set(DILIGENT_INCLUDE_DIRS
    "${DILIGENT_CORE_DIR}/Primitives/interface"
    "${DILIGENT_CORE_DIR}/Common/interface"
    "${DILIGENT_CORE_DIR}/Platforms/interface"
    "${DILIGENT_CORE_DIR}/Platforms/Basic/interface"
    "${DILIGENT_CORE_DIR}/Graphics/GraphicsAccessories/interface"
    "${DILIGENT_CORE_DIR}/Graphics/GraphicsEngine/interface"
    "${DILIGENT_CORE_DIR}/Graphics/GraphicsEngineVulkan/interface"
    "${DILIGENT_CORE_DIR}/Graphics/GraphicsTools/interface"
    "${DILIGENT_CORE_DIR}/ThirdParty/Vulkan-Headers/include"
    "${DILIGENT_CORE_DIR}/ThirdParty/volk"
    "${DILIGENT_CORE_BUILD_DIR}/Graphics/ShaderTools/GeneratedShaders"
)

if(APPLE)
    list(APPEND DILIGENT_INCLUDE_DIRS
        "${DILIGENT_CORE_DIR}/Platforms/Apple/interface")
elseif(WIN32)
    list(APPEND DILIGENT_INCLUDE_DIRS
        "${DILIGENT_CORE_DIR}/Platforms/Win32/interface"
        "${CMAKE_CURRENT_BINARY_DIR}/DiligentCore/Graphics/ShaderTools/GeneratedShaders")
endif()

if(APPLE)
    set(DEMO_PLATFORM_SOURCES main.mm)
else()
    set(DEMO_PLATFORM_SOURCES main_win32.cpp)
endif()

if(APPLE)
    add_executable(DiligentCoreVulkanDemo MACOSX_BUNDLE
        ${DEMO_PLATFORM_SOURCES}
        NanitePipelines.cpp
        NanitePipelines.hpp
        NaniteModel.cpp
        NaniteModel.hpp
        NaniteGpuScene.cpp
        NaniteGpuScene.hpp
        NaniteTypes.hpp
        DemoRenderer.cpp
        DemoRenderer.hpp
        "${VERTEX_SHADER}"
        "${FRAGMENT_SHADER}")
else()
    add_executable(DiligentCoreVulkanDemo WIN32
        ${DEMO_PLATFORM_SOURCES}
        NanitePipelines.cpp
        NanitePipelines.hpp
        NaniteModel.cpp
        NaniteModel.hpp
        NaniteGpuScene.cpp
        NaniteGpuScene.hpp
        NaniteTypes.hpp
        DemoRenderer.cpp
        DemoRenderer.hpp
        "${VERTEX_SHADER}"
        "${FRAGMENT_SHADER}")
endif()
add_dependencies(DiligentCoreVulkanDemo DiligentCoreVulkanShaders)
target_compile_features(DiligentCoreVulkanDemo PRIVATE cxx_std_17)
target_compile_definitions(DiligentCoreVulkanDemo PRIVATE
    DEMO_SHADER_DIR="${SHADER_BINARY_DIR}"
    DEMO_NANITE_MSL_DIR="${CMAKE_CURRENT_LIST_DIR}/Shaders/Nanite/msl"
    DEMO_ASSET_DIR="${CMAKE_CURRENT_LIST_DIR}/Assets"
    VULKAN_SUPPORTED=1)
if(APPLE)
    target_compile_definitions(DiligentCoreVulkanDemo PRIVATE
        PLATFORM_MACOS=1
        PLATFORM_APPLE=1)
elseif(WIN32)
    target_compile_definitions(DiligentCoreVulkanDemo PRIVATE PLATFORM_WIN32=1)
endif()
target_include_directories(DiligentCoreVulkanDemo PRIVATE ${DILIGENT_INCLUDE_DIRS})
target_include_directories(DiligentCoreVulkanDemo PRIVATE
    "${CMAKE_CURRENT_LIST_DIR}"
    "${MESHOPTIMIZER_DIR}/src"
    "${MESHOPTIMIZER_DIR}/demo"
    "${MESHOPTIMIZER_DIR}/extern")

target_link_libraries(DiligentCoreVulkanDemo PRIVATE meshoptimizer)
if(APPLE)
    target_link_libraries(DiligentCoreVulkanDemo PRIVATE
        "${DILIGENT_CORE_BUILD_DIR}/libDiligentCore.a"
        "${DILIGENT_CORE_BUILD_DIR}/ThirdParty/SPIRV-Cross/libspirv-cross-core.a"
        "${DILIGENT_CORE_BUILD_DIR}/ThirdParty/volk/libvolk.a"
        "${DILIGENT_CORE_BUILD_DIR}/ThirdParty/xxHash/build/cmake/libxxhash.a"
        /opt/homebrew/lib/libvulkan.dylib
        "-framework AppKit"
        "-framework QuartzCore"
        # UTType, for filtering the model open panel to OBJ and PLY. The older
        # allowedFileTypes takes plain strings and needs no framework, but it has
        # been deprecated since macOS 12.
        "-framework UniformTypeIdentifiers"
        "-framework Foundation"
        "-framework CoreFoundation"
        "-framework CoreGraphics"
        "-framework ImageIO")
    set_target_properties(DiligentCoreVulkanDemo PROPERTIES
        MACOSX_BUNDLE_BUNDLE_NAME "DiligentCore Vulkan Demo"
        MACOSX_BUNDLE_GUI_IDENTIFIER "local.diligentcore.vulkan.demo"
        MACOSX_BUNDLE_SHORT_VERSION_STRING "0.1"
        BUILD_RPATH "/opt/homebrew/lib"
        INSTALL_RPATH "/opt/homebrew/lib")
elseif(WIN32)
    if(DILIGENT_CORE_USE_SOURCE)
        target_link_libraries(DiligentCoreVulkanDemo PRIVATE
            Diligent-GraphicsEngineVk-static
            Vulkan::Vulkan)
    else()
        message(FATAL_ERROR
            "Windows builds require -DDILIGENT_CORE_USE_SOURCE=ON for this checkout")
    endif()
    target_link_libraries(DiligentCoreVulkanDemo PRIVATE user32 gdi32 comctl32)
endif()
