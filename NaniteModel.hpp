#pragma once

#include "NaniteTypes.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace Nanite
{

struct Position
{
    float x;
    float y;
    float z;
    float padding;
};

struct TexCoord
{
    float u;
    float v;
};

struct CpuScene
{
    std::uint64_t SourceTriangleCount = 0;
    std::vector<Position> Positions;
    // Smooth per-vertex normals, one per entry in Positions, generated from the
    // source triangles. Simplification only ever removes vertices, so the same
    // normal stays valid at every LOD level, which is what keeps a level switch
    // from also switching the shading.
    std::vector<Position> Normals;
    // glTF UVs, kept parallel to Positions. Legacy OBJ/PLY inputs receive zero
    // UVs and continue to use the generated PBR fallback mapping.
    std::vector<TexCoord> TexCoords;
    std::vector<std::uint32_t> Indices;
    // One material id per index in the post-cluster index stream. The current
    // visibility payload stays 32-bit; shading resolves this side-band id after
    // it has decoded the winning triangle.
    std::vector<std::uint32_t> MaterialIndices;
    std::vector<Material> Materials;
    std::vector<std::string> AlbedoTexturePaths;
    std::vector<std::string> NormalTexturePaths;
    std::vector<std::string> MetallicRoughnessTexturePaths;
    // Meshlet-local payload used by the software rasterizer. Each cluster owns
    // a compact vertex table and a parallel local index stream; the hardware
    // path keeps using Indices so this optimization cannot change its output.
    std::vector<Position> ClusterPositions;
    std::vector<std::uint32_t> ClusterLocalIndices;
    // First element of each cluster's local index range. This is separate from
    // Cluster::FirstIndex, which belongs to the original hardware index stream.
    std::vector<std::uint32_t> ClusterLocalIndexOffsets;
    std::vector<Instance> Instances;
    // Optional per-instance source triangle counts. Multi-prototype scenes use
    // this to report logical geometry without pretending every instance shares
    // one mesh. Legacy scenes leave it empty and use SourceTriangleCount.
    std::vector<std::uint64_t> InstanceTriangleCounts;
    bool CustomInstanceLayout = false;
    std::vector<DagNode> Nodes;
    std::vector<ClusterGroup> ClusterGroups;
    std::vector<Cluster> Clusters;
    // Largest Cluster::IndexCount in the scene. The raster pass draws every
    // visible cluster as one instance of a single indirect command, so all
    // clusters share one vertex count and the shorter ones collapse their tail
    // into degenerate triangles.
    std::uint32_t MaxClusterIndexCount = 0;
};

// Path of the model the scene is built from: the Stanford Happy Buddha by
// default. NANITE_MODEL overrides it with either an absolute path or a file name
// resolved against the asset directory.
std::string ResolveModelPath();

// OBJ, PLY, or glTF, chosen by extension. OBJ/PLY keep their legacy position
// path; glTF also imports normals, UVs, material factors, and external texture
// paths for the post-Visibility-Buffer PBR pass.
//
// clusterTriangles of 0 defers to NANITE_CLUSTER_TRIANGLES and then to the
// built-in default, so a caller that has no opinion does not have to know one.
CpuScene LoadModelAndBuildScene(const std::string& path,
                                std::uint64_t instanceCount = 0,
                                std::size_t clusterTriangles = 0);
CpuScene BuildDemoScene();

// Positions the first activeCount instances in the compact cube in front of the
// camera, leaving the remaining entries untouched. Exposed because the cube side
// is derived from the count, so changing how many instances are active at
// runtime means recomputing and re-uploading the transforms rather than just
// shrinking a draw range.
//
// spacingHorizontal separates neighbouring columns and, equally, the depth
// layers; spacingVertical separates rows. Splitting them this way rather than
// per-axis is what lets the block be squeezed into a wall or stretched into a
// column, which changes how much screen area the same instance count covers and
// therefore which LOD levels get picked.
void LayoutInstances(std::vector<Instance>& instances,
                     std::uint64_t activeCount,
                     float spacingHorizontal = 2.0f,
                     float spacingVertical = 2.0f);

// Positions an explicit X by Y by Z instance grid. The index order is X fastest,
// then Y, then Z, so the first layer is a row-major XY plane.
void LayoutInstanceGrid(std::vector<Instance>& instances,
                        std::uint32_t countX,
                        std::uint32_t countY,
                        std::uint32_t countZ,
                        float spacingHorizontal = 2.0f,
                        float spacingVertical = 2.0f);

} // namespace Nanite
