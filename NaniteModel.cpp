#include "NaniteModel.hpp"

#include <meshoptimizer.h>
#define CGLTF_IMPLEMENTATION
#include <cgltf.h>
#define CLUSTERLOD_IMPLEMENTATION
#include <clusterlod.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <numeric>
#include <limits>
#include <random>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <utility>

namespace
{

struct Float3
{
    float x;
    float y;
    float z;
};

struct Bounds
{
    Float3 Min{
        std::numeric_limits<float>::max(),
        std::numeric_limits<float>::max(),
        std::numeric_limits<float>::max()};
    Float3 Max{
        std::numeric_limits<float>::lowest(),
        std::numeric_limits<float>::lowest(),
        std::numeric_limits<float>::lowest()};
};

void Extend(Bounds& bounds, Float3 point)
{
    bounds.Min.x = std::min(bounds.Min.x, point.x);
    bounds.Min.y = std::min(bounds.Min.y, point.y);
    bounds.Min.z = std::min(bounds.Min.z, point.z);
    bounds.Max.x = std::max(bounds.Max.x, point.x);
    bounds.Max.y = std::max(bounds.Max.y, point.y);
    bounds.Max.z = std::max(bounds.Max.z, point.z);
}

Float3 Center(const Bounds& bounds)
{
    return {
        (bounds.Min.x + bounds.Max.x) * 0.5f,
        (bounds.Min.y + bounds.Max.y) * 0.5f,
        (bounds.Min.z + bounds.Max.z) * 0.5f};
}

float Radius(const Bounds& bounds)
{
    const Float3 center = Center(bounds);
    const float dx = bounds.Max.x - center.x;
    const float dy = bounds.Max.y - center.y;
    const float dz = bounds.Max.z - center.z;
    return std::sqrt(dx * dx + dy * dy + dz * dz);
}

Nanite::DagNode MakeInternalNode(const std::vector<Nanite::DagNode>& nodes,
                                 const std::vector<std::uint32_t>& children,
                                 std::uint32_t childStart)
{
    Bounds bounds;
    for (const std::uint32_t childIndex : children)
    {
        const Nanite::DagNode& child = nodes[childIndex];
        Extend(bounds, {child.BBoxMin[0], child.BBoxMin[1], child.BBoxMin[2]});
        Extend(bounds, {child.BBoxMax[0], child.BBoxMax[1], child.BBoxMax[2]});
    }

    Nanite::DagNode node{};
    node.BBoxMin[0] = bounds.Min.x;
    node.BBoxMin[1] = bounds.Min.y;
    node.BBoxMin[2] = bounds.Min.z;
    node.BBoxMax[0] = bounds.Max.x;
    node.BBoxMax[1] = bounds.Max.y;
    node.BBoxMax[2] = bounds.Max.z;
    node.NodeData = (childStart << 1u) | (static_cast<std::uint32_t>(children.size()) << 28u);
    return node;
}

Nanite::DagNode MakeDagNode(const clodNode& source,
                            const std::vector<Nanite::ClusterGroup>& clusterGroups)
{
    if (source.child_count > 8u)
        throw std::runtime_error{"clusterlod hierarchy node exceeds GPU child limit"};

    Nanite::DagNode node{};
    for (int axis = 0; axis < 3; ++axis)
    {
        node.BBoxMin[axis] = source.bounds.center[axis] - source.bounds.radius;
        node.BBoxMax[axis] = source.bounds.center[axis] + source.bounds.radius;
    }
    if (source.group >= 0 &&
        static_cast<std::size_t>(source.group) < clusterGroups.size())
    {
        const Nanite::ClusterGroup& group = clusterGroups[static_cast<std::size_t>(source.group)];
        for (int axis = 0; axis < 3; ++axis)
        {
            node.BBoxMin[axis] = group.BBoxMin[axis];
            node.BBoxMax[axis] = group.BBoxMax[axis];
        }
    }
    if (source.group >= 0)
    {
        node.NodeData = 1u | (static_cast<std::uint32_t>(source.group) << 1u);
    }
    else
    {
        node.NodeData = (source.child_offset << 1u) |
            (static_cast<std::uint32_t>(source.child_count) << 28u);
    }
    return node;
}

// Fills DagNode::LodSphere and MaxGroupError bottom up. The pair has to dominate
// the group error test of every cluster group in the subtree, so the sphere is
// grown from the boxes of the descendant group spheres - an enclosing sphere is
// never farther from the camera than what it encloses - and the error is their
// maximum. Nodes with no group below them keep a zero error and are therefore
// always pruned, which is correct: they cannot emit anything either.
void AssignNodeLodBounds(std::vector<Nanite::DagNode>& nodes,
                         const std::vector<Nanite::ClusterGroup>& groups,
                         std::uint32_t rootIndex)
{
    // Explicit post-order over the tree: a node is only finished after all of its
    // children are, so it can read their results instead of recursing.
    std::vector<std::pair<std::uint32_t, bool>> stack;
    stack.emplace_back(rootIndex, false);
    while (!stack.empty())
    {
        const std::uint32_t nodeIndex = stack.back().first;
        const bool childrenDone = stack.back().second;
        stack.pop_back();
        if (nodeIndex >= nodes.size())
            throw std::runtime_error{"Nanite hierarchy references a node out of range"};

        Nanite::DagNode& node = nodes[nodeIndex];
        if ((node.NodeData & 1u) != 0u)
        {
            const std::uint32_t groupIndex = (node.NodeData >> 1u) & 0x00FFFFFFu;
            if (groupIndex >= groups.size())
                throw std::runtime_error{"Nanite hierarchy references a group out of range"};
            const Nanite::ClusterGroup& group = groups[groupIndex];
            for (int component = 0; component < 4; ++component)
                node.LodSphere[component] = group.BoundSphere[component];
            node.MaxGroupError = group.ParentError;
            continue;
        }

        const std::uint32_t childStart = (node.NodeData >> 1u) & 0x07FFFFFFu;
        const std::uint32_t childCount = (node.NodeData >> 28u) & 0xFu;
        if (!childrenDone)
        {
            stack.emplace_back(nodeIndex, true);
            for (std::uint32_t child = 0; child < childCount; ++child)
                stack.emplace_back(childStart + child, false);
            continue;
        }

        Bounds bounds;
        float error = 0.0f;
        for (std::uint32_t child = 0; child < childCount; ++child)
        {
            const Nanite::DagNode& childNode = nodes[childStart + child];
            const float radius = childNode.LodSphere[3];
            Extend(bounds, {childNode.LodSphere[0] - radius,
                            childNode.LodSphere[1] - radius,
                            childNode.LodSphere[2] - radius});
            Extend(bounds, {childNode.LodSphere[0] + radius,
                            childNode.LodSphere[1] + radius,
                            childNode.LodSphere[2] + radius});
            error = std::max(error, childNode.MaxGroupError);
        }
        const Float3 center = Center(bounds);
        node.LodSphere[0] = center.x;
        node.LodSphere[1] = center.y;
        node.LodSphere[2] = center.z;
        node.LodSphere[3] = Radius(bounds);
        node.MaxGroupError = error;
    }
}

Nanite::CpuScene BuildSceneFromPositionsAndIndices(
    std::vector<Nanite::Position> positions,
    std::vector<std::uint32_t> indices,
    std::uint64_t requestedInstanceCount,
    std::size_t requestedClusterTriangles,
    std::vector<Nanite::TexCoord> texCoords = {},
    std::vector<std::uint32_t> vertexMaterialIndices = {},
    std::vector<Nanite::Material> materials = {},
    std::vector<std::string> albedoTexturePaths = {},
    std::vector<std::string> normalTexturePaths = {},
    std::vector<std::string> metallicRoughnessTexturePaths = {})
{
    if (positions.empty() || indices.empty() || (indices.size() % 3) != 0)
        throw std::runtime_error{"Nanite scene requires positions and a triangle index list"};

    Bounds sourceBounds;
    for (const Nanite::Position& position : positions)
        Extend(sourceBounds, {position.x, position.y, position.z});

    const Float3 sourceCenter = Center(sourceBounds);
    const float sourceExtent = std::max({
        sourceBounds.Max.x - sourceBounds.Min.x,
        sourceBounds.Max.y - sourceBounds.Min.y,
        sourceBounds.Max.z - sourceBounds.Min.z});
    if (sourceExtent <= std::numeric_limits<float>::epsilon())
        throw std::runtime_error{"Nanite scene has zero spatial extent"};

    const float normalizationScale = 2.0f / sourceExtent;
    for (Nanite::Position& position : positions)
    {
        position.x = (position.x - sourceCenter.x) * normalizationScale;
        position.y = (position.y - sourceCenter.y) * normalizationScale;
        position.z = (position.z - sourceCenter.z) * normalizationScale;
    }

    Nanite::CpuScene scene;
    scene.SourceTriangleCount = indices.size() / 3u;
    scene.Positions = std::move(positions);
    scene.TexCoords = texCoords.size() == scene.Positions.size() ?
        std::move(texCoords) : std::vector<Nanite::TexCoord>(scene.Positions.size());
    if (vertexMaterialIndices.size() != scene.Positions.size())
        vertexMaterialIndices.assign(scene.Positions.size(), 0u);
    scene.Materials = std::move(materials);
    scene.AlbedoTexturePaths = std::move(albedoTexturePaths);
    scene.NormalTexturePaths = std::move(normalTexturePaths);
    scene.MetallicRoughnessTexturePaths = std::move(metallicRoughnessTexturePaths);
    scene.Indices.clear();
    scene.MaterialIndices.clear();

    // Smooth vertex normals, accumulated from the source triangles before any
    // simplification runs. The cross product is left unnormalized so a triangle
    // contributes in proportion to its area, which stops a fan of slivers from
    // outvoting the one large face next to it. A flat normal from screen-space
    // derivatives would have been less code, but faceting would then change with
    // the LOD level and exaggerate exactly the popping this shading exists to
    // judge.
    scene.Normals.assign(scene.Positions.size(), Nanite::Position{});
    for (std::size_t triangle = 0; triangle + 2u < indices.size(); triangle += 3u)
    {
        const Nanite::Position& a = scene.Positions[indices[triangle + 0u]];
        const Nanite::Position& b = scene.Positions[indices[triangle + 1u]];
        const Nanite::Position& c = scene.Positions[indices[triangle + 2u]];
        const float edge1[3] = {b.x - a.x, b.y - a.y, b.z - a.z};
        const float edge2[3] = {c.x - a.x, c.y - a.y, c.z - a.z};
        const float faceNormal[3] = {
            edge1[1] * edge2[2] - edge1[2] * edge2[1],
            edge1[2] * edge2[0] - edge1[0] * edge2[2],
            edge1[0] * edge2[1] - edge1[1] * edge2[0]};
        for (std::size_t corner = 0; corner < 3u; ++corner)
        {
            Nanite::Position& normal = scene.Normals[indices[triangle + corner]];
            normal.x += faceNormal[0];
            normal.y += faceNormal[1];
            normal.z += faceNormal[2];
        }
    }
    for (Nanite::Position& normal : scene.Normals)
    {
        const float length = std::sqrt(
            normal.x * normal.x + normal.y * normal.y + normal.z * normal.z);
        // A vertex no triangle references, or one whose faces cancel out, has no
        // direction to offer. Point it up rather than leaving a zero that would
        // reach the shader as a black pixel.
        if (length > std::numeric_limits<float>::epsilon())
        {
            normal.x /= length;
            normal.y /= length;
            normal.z /= length;
        }
        else
        {
            normal.y = 1.0f;
        }
    }

    // Cluster granularity. Smaller clusters give the conservative HZB test tighter
    // projected bounds and a finer LOD cut; larger ones amortize the per-cluster
    // cost of the whole GPU pipeline, since traversal, the HZB sample, the queue
    // entry and the draw instance are all priced per cluster rather than per
    // triangle. That per-cluster cost dominates by a wide margin: swept at 4096
    // instances with vsync off and a static camera, frame time was 26.5 ms at 16
    // triangles, 12.5 at 32, 11.6 at 128 and back to 13.3 at 256, while
    // persistentCull alone went 8.5 -> 1.7 -> 0.65 ms. 128 draws 44% more triangles
    // than 16 for the same error threshold and is still 2.3x faster. At 256 the
    // 128-vertex cap starts binding before the triangle count, so clusters come out
    // uneven (max708 vs mean558 indices) and the padding in the fixed-size draw
    // instance eats the remaining gain. Nanite uses 128 as well.
    // Zero means "whatever the environment or the default says". The GUI passes an
    // explicit count so the control panel wins over the variable it was seeded
    // from; the variable still works for a run started from a shell.
    if (requestedClusterTriangles == 0u)
    {
        if (const char* requestedTriangles = std::getenv("NANITE_CLUSTER_TRIANGLES"))
            requestedClusterTriangles =
                static_cast<std::size_t>(std::strtoull(requestedTriangles, nullptr, 10));
    }
    std::size_t clusterTriangles = requestedClusterTriangles == 0u ? 128u : requestedClusterTriangles;
    // clodDefaultConfig asserts this range, and an assert in a release build is no
    // check at all.
    if (clusterTriangles < 4u || clusterTriangles > 256u)
        throw std::runtime_error{"Cluster triangle count must be between 4 and 256"};
    clodConfig config = clodDefaultConfig(clusterTriangles);
    // Leave headroom so the triangle count is what bounds a cluster, not the vertex
    // count: 16 triangles never reference more than 48 distinct vertices. The 128
    // ceiling is meshoptimizer's practical limit for a meshlet and the number
    // Nanite uses, so past 32 triangles the vertex count does start to bind.
    config.max_vertices = std::min<std::size_t>(clusterTriangles * 4u, 128u);
    config.partition_size = 8;
    config.optimize_bounds = true;

    // Normals, not positions, are what the attribute term is for. Feeding it the
    // positions again - which is what this used to do - counted the position error
    // twice and never looked at the normal field at all, so the error the library
    // reported was blind to shading: the simplifier was free to collapse an edge
    // that kept the silhouette but flattened the normals, and the error said nothing
    // had happened.
    //
    // How much that mattered turned out to be the smaller half of the story. The
    // other bug fixed alongside it - focalY being read off the combined
    // view-projection, which scaled every projected error by cos(pitch) - is what
    // accounted for most of why the usable threshold came out far below one pixel,
    // because popping is judged from an angle. With that fixed, 1 px is usable again
    // and the normal term only has to cover the rest.
    //
    // 0.5 is meshoptimizer's own Nanite demo's figure and, tested on the bunny and
    // the Buddha at 1 px, the point where popping is not visible; 1.0 looks slightly
    // better still and costs far too much for it (2.6x the clusters on the bunny),
    // and 0 - the attribute term dropped entirely - does pop. A fixed number works
    // for any asset here only because the loader has already rescaled every mesh to
    // extent 2: the error is absolute, in mesh coordinate space, so a weight not tied
    // to the mesh scale would mean something different per model. meshoptimizer
    // documents the unit as "a change of 1/weight in the attribute over a distance d
    // counts as much as a change of d in position".
    //
    // The environment variable is a diagnostic for redoing that comparison, not a
    // parameter anyone is expected to set.
    float normalWeight = 0.5f;
    if (const char* requestedWeight = std::getenv("NANITE_NORMAL_WEIGHT"))
        normalWeight = std::max(std::strtof(requestedWeight, nullptr), 0.0f);
    const float attributeWeights[3] = {normalWeight, normalWeight, normalWeight};
    clodMesh mesh{};
    mesh.indices = indices.data();
    mesh.index_count = indices.size();
    mesh.vertex_count = scene.Positions.size();
    mesh.vertex_positions = &scene.Positions[0].x;
    mesh.vertex_positions_stride = sizeof(Nanite::Position);
    mesh.vertex_attributes = &scene.Normals[0].x;
    mesh.vertex_attributes_stride = sizeof(Nanite::Position);
    mesh.attribute_weights = attributeWeights;
    // Zero weights would still cost the simplifier the attribute bookkeeping for an
    // error term that can only come out zero, so drop the attributes instead.
    mesh.attribute_count = normalWeight > 0.0f ? 3u : 0u;

    std::vector<clodGroup> clodGroups;
    clodBuild(config, mesh, [&](clodGroup sourceGroup,
                                const clodCluster* sourceClusters,
                                size_t clusterCount) -> int {
        if (clusterCount == 0)
            throw std::runtime_error{"clusterlod emitted an empty group"};

        Nanite::ClusterGroup group{};
        group.BoundSphere[0] = sourceGroup.simplified.center[0];
        group.BoundSphere[1] = sourceGroup.simplified.center[1];
        group.BoundSphere[2] = sourceGroup.simplified.center[2];
        group.BoundSphere[3] = sourceGroup.simplified.radius;
        for (int axis = 0; axis < 3; ++axis)
        {
            group.BBoxMin[axis] = sourceGroup.simplified.center[axis] - sourceGroup.simplified.radius;
            group.BBoxMax[axis] = sourceGroup.simplified.center[axis] + sourceGroup.simplified.radius;
        }
        group.ParentError = sourceGroup.simplified.error;
        group.ClusterStart = static_cast<std::uint32_t>(scene.Clusters.size());
        group.ClusterCount = static_cast<std::uint32_t>(clusterCount);
        group.RefineGroupIndex = std::numeric_limits<std::uint32_t>::max();

        Bounds groupGeometryBounds;

        for (size_t clusterIndex = 0; clusterIndex < clusterCount; ++clusterIndex)
        {
            const clodCluster& sourceCluster = sourceClusters[clusterIndex];
            if (sourceCluster.index_count == 0 || (sourceCluster.index_count % 3) != 0)
                throw std::runtime_error{"clusterlod emitted an invalid cluster"};

            Nanite::Cluster cluster{};
            cluster.BoundSphere[0] = sourceCluster.bounds.center[0];
            cluster.BoundSphere[1] = sourceCluster.bounds.center[1];
            cluster.BoundSphere[2] = sourceCluster.bounds.center[2];
            cluster.BoundSphere[3] = sourceCluster.bounds.radius;
            for (int axis = 0; axis < 3; ++axis)
            {
                cluster.BBoxMin[axis] = sourceCluster.bounds.center[axis] - sourceCluster.bounds.radius;
                cluster.BBoxMax[axis] = sourceCluster.bounds.center[axis] + sourceCluster.bounds.radius;
            }
            cluster.MaxError = sourceCluster.bounds.error;
            cluster.IndexCount = static_cast<std::uint32_t>(sourceCluster.index_count);
            cluster.FirstIndex = static_cast<std::uint32_t>(scene.Indices.size());
            cluster.VertexOffset = 0;
            cluster.RefinedGroupIndex = sourceCluster.refined < 0 ?
                std::numeric_limits<std::uint32_t>::max() : static_cast<std::uint32_t>(sourceCluster.refined);

            // Keep the original index stream for the fixed-function path, but
            // also materialize the meshlet-local representation used by software
            // raster. clodLocalIndices is the same locality conversion used by
            // meshoptimizer's meshlet tooling: every cluster-local vertex is
            // transformed once per workgroup instead of once per triangle corner.
            if (sourceCluster.vertex_count > 255u || scene.ClusterPositions.size() > 0x00FFFFFFu)
                throw std::runtime_error{"cluster-local vertex payload exceeds packed offset"};
            std::vector<unsigned int> localVertices(sourceCluster.vertex_count);
            std::vector<unsigned char> localIndices(sourceCluster.index_count);
            const std::size_t localVertexCount = clodLocalIndices(
                localVertices.data(), localIndices.data(),
                sourceCluster.indices, sourceCluster.index_count);
            if (localVertexCount != sourceCluster.vertex_count || localVertexCount > 128u)
                throw std::runtime_error{"cluster-local vertex remap is invalid"};
            const std::uint32_t localVertexOffset =
                static_cast<std::uint32_t>(scene.ClusterPositions.size());
            const std::uint32_t localIndexOffset =
                static_cast<std::uint32_t>(scene.ClusterLocalIndices.size());
            for (std::size_t vertex = 0; vertex < localVertexCount; ++vertex)
                scene.ClusterPositions.push_back(scene.Positions[localVertices[vertex]]);
            for (const unsigned char localIndex : localIndices)
                scene.ClusterLocalIndices.push_back(static_cast<std::uint32_t>(localIndex));
            scene.ClusterLocalIndexOffsets.push_back(localIndexOffset);
            cluster.VertexOffset =
                (static_cast<std::uint32_t>(localVertexCount) << 24u) | localVertexOffset;

            // The library's bounding sphere is useful for LOD error, but its
            // center +/- radius cube is too loose for conservative HZB tests.
            // Build the visibility box from the vertices actually referenced
            // by this cluster, then union those boxes for the group.
            Bounds clusterGeometryBounds;
            for (size_t index = 0; index < sourceCluster.index_count; ++index)
            {
                const std::uint32_t vertexIndex = sourceCluster.indices[index];
                if (vertexIndex >= scene.Positions.size())
                    throw std::runtime_error{"clusterlod emitted an invalid vertex index"};
                const Nanite::Position& position = scene.Positions[vertexIndex];
                Extend(clusterGeometryBounds, {position.x, position.y, position.z});
            }
            for (int axis = 0; axis < 3; ++axis)
            {
                cluster.BBoxMin[axis] = axis == 0 ? clusterGeometryBounds.Min.x :
                    axis == 1 ? clusterGeometryBounds.Min.y : clusterGeometryBounds.Min.z;
                cluster.BBoxMax[axis] = axis == 0 ? clusterGeometryBounds.Max.x :
                    axis == 1 ? clusterGeometryBounds.Max.y : clusterGeometryBounds.Max.z;
            }
            Extend(groupGeometryBounds,
                   {clusterGeometryBounds.Min.x, clusterGeometryBounds.Min.y,
                    clusterGeometryBounds.Min.z});
            Extend(groupGeometryBounds,
                   {clusterGeometryBounds.Max.x, clusterGeometryBounds.Max.y,
                    clusterGeometryBounds.Max.z});
            scene.Indices.insert(
                scene.Indices.end(),
                sourceCluster.indices,
                sourceCluster.indices + sourceCluster.index_count);
            for (size_t index = 0; index < sourceCluster.index_count; ++index)
            {
                const std::uint32_t vertexIndex = sourceCluster.indices[index];
                scene.MaterialIndices.push_back(vertexMaterialIndices[vertexIndex]);
            }
            scene.Clusters.push_back(cluster);
        }

        for (int axis = 0; axis < 3; ++axis)
        {
            group.BBoxMin[axis] = axis == 0 ? groupGeometryBounds.Min.x :
                axis == 1 ? groupGeometryBounds.Min.y : groupGeometryBounds.Min.z;
            group.BBoxMax[axis] = axis == 0 ? groupGeometryBounds.Max.x :
                axis == 1 ? groupGeometryBounds.Max.y : groupGeometryBounds.Max.z;
        }

        clodGroups.push_back(sourceGroup);
        scene.ClusterGroups.push_back(group);
        return static_cast<int>(clodGroups.size() - 1);
    });

    if (clodGroups.empty())
        throw std::runtime_error{"clusterlod produced no groups"};

    const std::size_t levelCount = static_cast<std::size_t>(clodGroups.back().depth + 1);
    const std::size_t nodeBound = clodBuildHierarchyBound(clodGroups.size(), 8, levelCount);
    std::vector<clodNode> hierarchy(nodeBound);
    hierarchy.resize(clodBuildHierarchy(
        hierarchy.data(), clodGroups.data(), clodGroups.size(), 8, levelCount));
    if (hierarchy.size() < levelCount)
        throw std::runtime_error{"clusterlod produced an invalid hierarchy"};

    scene.Nodes.reserve(hierarchy.size() + levelCount);
    for (const clodNode& node : hierarchy)
        scene.Nodes.push_back(MakeDagNode(node, scene.ClusterGroups));

    std::vector<std::uint32_t> rootLevel(levelCount);
    std::iota(rootLevel.begin(), rootLevel.end(), 0u);
    while (rootLevel.size() > 1)
    {
        std::vector<std::uint32_t> nextLevel;
        for (std::size_t start = 0; start < rootLevel.size(); start += 8)
        {
            const std::size_t count = std::min<std::size_t>(8, rootLevel.size() - start);
            std::vector<std::uint32_t> children(
                rootLevel.begin() + start,
                rootLevel.begin() + start + count);
            nextLevel.push_back(static_cast<std::uint32_t>(scene.Nodes.size()));
            scene.Nodes.push_back(MakeInternalNode(scene.Nodes, children, children.front()));
        }
        rootLevel = std::move(nextLevel);
    }

    AssignNodeLodBounds(scene.Nodes, scene.ClusterGroups, rootLevel.front());

    Nanite::Instance instance{};
    instance.WorldMatrix[0] = 1.0f;
    instance.WorldMatrix[5] = 1.0f;
    instance.WorldMatrix[10] = 1.0f;
    instance.WorldMatrix[15] = 1.0f;
    Bounds sceneBounds;
    for (const Nanite::Position& position : scene.Positions)
        Extend(sceneBounds, {position.x, position.y, position.z});
    const Float3 sceneCenter = Center(sceneBounds);
    instance.LocalBoundSphere[0] = sceneCenter.x;
    instance.LocalBoundSphere[1] = sceneCenter.y;
    instance.LocalBoundSphere[2] = sceneCenter.z;
    instance.LocalBoundSphere[3] = Radius(sceneBounds);
    instance.BBoxMin[0] = sceneBounds.Min.x;
    instance.BBoxMin[1] = sceneBounds.Min.y;
    instance.BBoxMin[2] = sceneBounds.Min.z;
    instance.BBoxMax[0] = sceneBounds.Max.x;
    instance.BBoxMax[1] = sceneBounds.Max.y;
    instance.BBoxMax[2] = sceneBounds.Max.z;
    instance.RootNodeIndex = rootLevel.front();
    std::uint64_t targetInstanceCount = requestedInstanceCount > 0u ?
        requestedInstanceCount : 27u;
    bool stressRequested = false;
    if (requestedInstanceCount > 0u)
    {
        stressRequested = requestedInstanceCount > 64u;
    }
    else if (const char* stressInstances = std::getenv("NANITE_STRESS_INSTANCES"))
    {
        stressRequested = true;
        targetInstanceCount = std::strtoull(stressInstances, nullptr, 10);
        if (targetInstanceCount == 0u)
            throw std::runtime_error{"NANITE_STRESS_INSTANCES must be greater than zero"};
    }
    else if (const char* stressTriangles = std::getenv("NANITE_STRESS_TRIANGLES"))
    {
        stressRequested = true;
        const std::uint64_t requestedTriangles =
            std::strtoull(stressTriangles, nullptr, 10);
        const std::uint64_t sourceTriangles = indices.size() / 3u;
        if (requestedTriangles == 0u || sourceTriangles == 0u)
            throw std::runtime_error{"NANITE_STRESS_TRIANGLES must be greater than zero"};
        targetInstanceCount = (requestedTriangles + sourceTriangles - 1u) / sourceTriangles;
    }
    if (targetInstanceCount > std::numeric_limits<std::uint32_t>::max())
        throw std::runtime_error{"Nanite stress instance count exceeds uint32 indexing"};

    const char* stressLayout = std::getenv("NANITE_STRESS_LAYOUT");
    const bool behindLayout = stressLayout != nullptr &&
        std::string{stressLayout} == "behind";
    const bool cubeLayout = !behindLayout;

    scene.Instances.reserve(static_cast<std::size_t>(targetInstanceCount));
    // Keep the normal demo and stress mode in a compact cube directly in front
    // of the camera. This gives movement tests real depth ordering without
    // putting every instance into one long occlusion column.
    for (std::uint64_t copyIndex = 0; copyIndex < targetInstanceCount; ++copyIndex)
        scene.Instances.push_back(instance);
    scene.InstanceTriangleCounts.assign(
        scene.Instances.size(), scene.SourceTriangleCount);
    Nanite::LayoutInstances(scene.Instances, targetInstanceCount);
    const std::uint64_t logicalTriangles = scene.SourceTriangleCount * scene.Instances.size();
    // One indirect command covers every visible cluster, so the vertex count it
    // carries has to be the largest any cluster needs. The mean says how much of
    // that is spent on degenerate padding.
    std::uint64_t totalClusterIndices = 0;
    std::uint64_t totalClusterVertices = 0;
    for (const Nanite::Cluster& cluster : scene.Clusters)
    {
        scene.MaxClusterIndexCount = std::max(scene.MaxClusterIndexCount, cluster.IndexCount);
        totalClusterIndices += cluster.IndexCount;
        totalClusterVertices += cluster.VertexOffset >> 24u;
    }
    std::cerr << "Nanite scene: sourceTriangles=" << indices.size() / 3u
              << " logicalTriangles=" << logicalTriangles
              << " instances=" << scene.Instances.size()
              << " layout=" << (cubeLayout ? "cube" : "behind")
              << " clusterTriangles=" << clusterTriangles
              << " nodes=" << scene.Nodes.size()
              << " groups=" << scene.ClusterGroups.size()
              << " clusters=" << scene.Clusters.size()
              << " clusterIndices=max" << scene.MaxClusterIndexCount
              << ",mean"
              << (scene.Clusters.empty() ? 0u :
                  static_cast<unsigned>(totalClusterIndices / scene.Clusters.size()))
              << " localVertices=mean"
              << (scene.Clusters.empty() ? 0u :
                  static_cast<unsigned>(totalClusterVertices / scene.Clusters.size()))
              << '\n';
    return scene;
}

// Merge independently built Nanite prototypes into one GPU scene. Each
// prototype keeps its own DAG root and normalized object space; placement is
// expressed by the instance transform, so cluster bounds and LOD errors remain
// valid without rewriting every hierarchy bound.
std::uint32_t AppendPrototypeScene(Nanite::CpuScene& destination,
                                    const Nanite::CpuScene& source)
{
    if (source.Instances.empty() || source.Nodes.empty())
        throw std::runtime_error{"Nanite prototype has no instance or hierarchy"};

    const std::uint32_t positionOffset = static_cast<std::uint32_t>(destination.Positions.size());
    const std::uint32_t indexOffset = static_cast<std::uint32_t>(destination.Indices.size());
    const std::uint32_t clusterPositionOffset =
        static_cast<std::uint32_t>(destination.ClusterPositions.size());
    const std::uint32_t localIndexOffset =
        static_cast<std::uint32_t>(destination.ClusterLocalIndices.size());
    const std::uint32_t clusterOffset = static_cast<std::uint32_t>(destination.Clusters.size());
    const std::uint32_t groupOffset = static_cast<std::uint32_t>(destination.ClusterGroups.size());
    const std::uint32_t nodeOffset = static_cast<std::uint32_t>(destination.Nodes.size());
    const std::uint32_t materialOffset = static_cast<std::uint32_t>(destination.Materials.size());

    destination.Positions.insert(destination.Positions.end(),
                                 source.Positions.begin(), source.Positions.end());
    destination.Normals.insert(destination.Normals.end(),
                               source.Normals.begin(), source.Normals.end());
    destination.TexCoords.insert(destination.TexCoords.end(),
                                 source.TexCoords.begin(), source.TexCoords.end());
    for (std::uint32_t index : source.Indices)
        destination.Indices.push_back(index + positionOffset);
    for (std::uint32_t materialIndex : source.MaterialIndices)
        destination.MaterialIndices.push_back(materialIndex + materialOffset);
    destination.ClusterPositions.insert(destination.ClusterPositions.end(),
                                        source.ClusterPositions.begin(), source.ClusterPositions.end());
    destination.ClusterLocalIndices.insert(destination.ClusterLocalIndices.end(),
                                           source.ClusterLocalIndices.begin(),
                                           source.ClusterLocalIndices.end());
    for (std::uint32_t offset : source.ClusterLocalIndexOffsets)
        destination.ClusterLocalIndexOffsets.push_back(offset + localIndexOffset);

    for (Nanite::Cluster cluster : source.Clusters)
    {
        cluster.FirstIndex += indexOffset;
        const std::uint32_t vertexCount = cluster.VertexOffset >> 24u;
        const std::uint32_t vertexOffset = (cluster.VertexOffset & 0x00FFFFFFu) +
            clusterPositionOffset;
        if (vertexOffset > 0x00FFFFFFu)
            throw std::runtime_error{"Merged Nanite cluster-local vertex buffer exceeds packed offset"};
        cluster.VertexOffset = (vertexCount << 24u) | vertexOffset;
        if (cluster.RefinedGroupIndex != std::numeric_limits<std::uint32_t>::max())
            cluster.RefinedGroupIndex += groupOffset;
        destination.MaxClusterIndexCount =
            std::max(destination.MaxClusterIndexCount, cluster.IndexCount);
        destination.Clusters.push_back(cluster);
    }
    for (Nanite::ClusterGroup group : source.ClusterGroups)
    {
        group.ClusterStart += clusterOffset;
        if (group.RefineGroupIndex != std::numeric_limits<std::uint32_t>::max())
            group.RefineGroupIndex += groupOffset;
        destination.ClusterGroups.push_back(group);
    }
    for (Nanite::DagNode node : source.Nodes)
    {
        if ((node.NodeData & 1u) != 0u)
        {
            const std::uint32_t groupIndex = (node.NodeData >> 1u) & 0x00FFFFFFu;
            node.NodeData = 1u | ((groupIndex + groupOffset) << 1u);
        }
        else
        {
            const std::uint32_t childStart = (node.NodeData >> 1u) & 0x07FFFFFFu;
            const std::uint32_t childCount = (node.NodeData >> 28u) & 0xFu;
            node.NodeData = ((childStart + nodeOffset) << 1u) | (childCount << 28u);
        }
        destination.Nodes.push_back(node);
    }
    destination.Materials.insert(destination.Materials.end(),
                                 source.Materials.begin(), source.Materials.end());
    destination.AlbedoTexturePaths.insert(destination.AlbedoTexturePaths.end(),
                                          source.AlbedoTexturePaths.begin(),
                                          source.AlbedoTexturePaths.end());
    destination.NormalTexturePaths.insert(destination.NormalTexturePaths.end(),
                                          source.NormalTexturePaths.begin(),
                                          source.NormalTexturePaths.end());
    destination.MetallicRoughnessTexturePaths.insert(
        destination.MetallicRoughnessTexturePaths.end(),
        source.MetallicRoughnessTexturePaths.begin(),
        source.MetallicRoughnessTexturePaths.end());
    destination.SourceTriangleCount += source.SourceTriangleCount;
    return nodeOffset + source.Instances[0].RootNodeIndex;
}

Nanite::Instance MakePlacedPrototypeInstance(const Nanite::CpuScene& prototype,
                                              std::uint32_t rootNode,
                                              float scale,
                                              float x,
                                              float y,
                                              float z)
{
    Nanite::Instance instance = prototype.Instances.front();
    instance.RootNodeIndex = rootNode;
    instance.WorldMatrix[0] = scale;
    instance.WorldMatrix[1] = 0.0f;
    instance.WorldMatrix[2] = 0.0f;
    instance.WorldMatrix[4] = 0.0f;
    instance.WorldMatrix[5] = scale;
    instance.WorldMatrix[6] = 0.0f;
    instance.WorldMatrix[8] = 0.0f;
    instance.WorldMatrix[9] = 0.0f;
    instance.WorldMatrix[10] = scale;
    instance.WorldMatrix[12] = x;
    instance.WorldMatrix[13] = y;
    instance.WorldMatrix[14] = z;
    return instance;
}

// Like MakePlacedPrototypeInstance but lays the mesh flat on the ground: a
// rotation of -90 degrees about the X axis maps the authored XY map plane onto
// the world XZ ground and the authored Z relief onto world Y (up). Snow
// Mountain 2 is authored as a vertical plane (map extent in XY, elevation in
// Z), so without this rotation it stands up like a 90-degree wall. The signs
// matter: Z -> +Y keeps the peaks pointing up, while Z -> -Y would bury the
// peaks underground.
Nanite::Instance MakeFlatPlacementInstance(const Nanite::CpuScene& prototype,
                                           std::uint32_t rootNode,
                                           float scale,
                                           float x,
                                           float y,
                                           float z)
{
    Nanite::Instance instance = prototype.Instances.front();
    instance.RootNodeIndex = rootNode;
    instance.WorldMatrix[0] = scale;
    instance.WorldMatrix[1] = 0.0f;
    instance.WorldMatrix[2] = 0.0f;
    instance.WorldMatrix[4] = 0.0f;
    instance.WorldMatrix[5] = 0.0f;
    instance.WorldMatrix[6] = -scale;
    instance.WorldMatrix[8] = 0.0f;
    instance.WorldMatrix[9] = scale;
    instance.WorldMatrix[10] = 0.0f;
    instance.WorldMatrix[12] = x;
    instance.WorldMatrix[13] = y;
    instance.WorldMatrix[14] = z;
    return instance;
}

std::int32_t ParseObjIndex(const std::string& token, std::int32_t vertexCount)
{
    const size_t slash = token.find('/');
    const std::string vertexToken = token.substr(0, slash);
    const std::int32_t rawIndex = std::stoi(vertexToken);
    const std::int32_t resolved = rawIndex >= 0 ? rawIndex - 1 : vertexCount + rawIndex;
    if (resolved < 0 || resolved >= vertexCount)
        throw std::runtime_error{"OBJ face references an invalid vertex"};
    return resolved;
}

// Minimal PLY reader, added because every freely available model past a million
// triangles ships as PLY, and converting them to OBJ costs roughly 6x the disk
// and a text parse of every coordinate at startup.
enum class PlyFormat
{
    Ascii,
    BinaryLittleEndian,
    BinaryBigEndian,
};

enum class PlyType
{
    Int8, UInt8, Int16, UInt16, Int32, UInt32, Float32, Float64,
};

struct PlyProperty
{
    std::string Name;
    PlyType ValueType = PlyType::Float32;
    // Type of the leading count field, for a list property only.
    PlyType CountType = PlyType::UInt8;
    bool IsList = false;
};

struct PlyElement
{
    std::string Name;
    std::uint64_t Count = 0;
    std::vector<PlyProperty> Properties;
};

PlyType ParsePlyType(const std::string& name)
{
    if (name == "char" || name == "int8") return PlyType::Int8;
    if (name == "uchar" || name == "uint8") return PlyType::UInt8;
    if (name == "short" || name == "int16") return PlyType::Int16;
    if (name == "ushort" || name == "uint16") return PlyType::UInt16;
    if (name == "int" || name == "int32") return PlyType::Int32;
    if (name == "uint" || name == "uint32") return PlyType::UInt32;
    if (name == "float" || name == "float32") return PlyType::Float32;
    if (name == "double" || name == "float64") return PlyType::Float64;
    throw std::runtime_error{"PLY property has an unsupported type: " + name};
}

std::size_t PlyTypeSize(PlyType type)
{
    switch (type)
    {
    case PlyType::Int8:
    case PlyType::UInt8: return 1;
    case PlyType::Int16:
    case PlyType::UInt16: return 2;
    case PlyType::Int32:
    case PlyType::UInt32:
    case PlyType::Float32: return 4;
    case PlyType::Float64: return 8;
    }
    throw std::runtime_error{"PLY property has an unsupported type"};
}

// Every value comes back as a double: it holds a 32-bit integer of either sign
// exactly, so one path can carry both coordinates and vertex indices without a
// variant. Only float64 coordinates lose anything, and those get truncated to
// float immediately afterwards anyway.
double ReadPlyValue(std::istream& file, PlyType type, PlyFormat format)
{
    if (format == PlyFormat::Ascii)
    {
        double value = 0.0;
        if (!(file >> value))
            throw std::runtime_error{"PLY body ended early"};
        return value;
    }

    const std::size_t size = PlyTypeSize(type);
    unsigned char bytes[8] = {};
    if (!file.read(reinterpret_cast<char*>(bytes), static_cast<std::streamsize>(size)))
        throw std::runtime_error{"PLY body ended early"};
    // The file's byte order is a property of the file, not of this machine, so
    // reverse into native order rather than testing the host endianness.
    if (format == PlyFormat::BinaryBigEndian)
        std::reverse(bytes, bytes + size);

    switch (type)
    {
    case PlyType::Int8: return static_cast<double>(*reinterpret_cast<const std::int8_t*>(bytes));
    case PlyType::UInt8: return static_cast<double>(bytes[0]);
    case PlyType::Int16: return static_cast<double>(*reinterpret_cast<const std::int16_t*>(bytes));
    case PlyType::UInt16: return static_cast<double>(*reinterpret_cast<const std::uint16_t*>(bytes));
    case PlyType::Int32: return static_cast<double>(*reinterpret_cast<const std::int32_t*>(bytes));
    case PlyType::UInt32: return static_cast<double>(*reinterpret_cast<const std::uint32_t*>(bytes));
    case PlyType::Float32: return static_cast<double>(*reinterpret_cast<const float*>(bytes));
    case PlyType::Float64: return *reinterpret_cast<const double*>(bytes);
    }
    throw std::runtime_error{"PLY property has an unsupported type"};
}

std::vector<PlyElement> ReadPlyHeader(std::istream& file, PlyFormat& format)
{
    std::string line;
    if (!std::getline(file, line))
        throw std::runtime_error{"PLY file is empty"};
    // Written on machines that used CRLF, and read on one that does not, so the
    // carriage return has to come off every header line rather than only this one.
    auto trim = [](std::string& text) {
        while (!text.empty() && (text.back() == '\r' || text.back() == ' '))
            text.pop_back();
    };
    trim(line);
    if (line != "ply")
        throw std::runtime_error{"PLY file does not start with the ply magic"};

    bool formatSeen = false;
    std::vector<PlyElement> elements;
    while (std::getline(file, line))
    {
        trim(line);
        std::istringstream stream{line};
        std::string keyword;
        stream >> keyword;
        if (keyword == "end_header")
        {
            if (!formatSeen)
                throw std::runtime_error{"PLY header has no format line"};
            return elements;
        }
        if (keyword == "format")
        {
            std::string name;
            stream >> name;
            if (name == "ascii") format = PlyFormat::Ascii;
            else if (name == "binary_little_endian") format = PlyFormat::BinaryLittleEndian;
            else if (name == "binary_big_endian") format = PlyFormat::BinaryBigEndian;
            else throw std::runtime_error{"PLY has an unsupported format: " + name};
            formatSeen = true;
        }
        else if (keyword == "element")
        {
            PlyElement element;
            stream >> element.Name >> element.Count;
            if (!stream)
                throw std::runtime_error{"PLY element line is malformed"};
            elements.push_back(std::move(element));
        }
        else if (keyword == "property")
        {
            if (elements.empty())
                throw std::runtime_error{"PLY property appears before any element"};
            std::string typeName;
            stream >> typeName;
            PlyProperty property;
            if (typeName == "list")
            {
                std::string countType;
                std::string valueType;
                stream >> countType >> valueType >> property.Name;
                property.IsList = true;
                property.CountType = ParsePlyType(countType);
                property.ValueType = ParsePlyType(valueType);
            }
            else
            {
                stream >> property.Name;
                property.ValueType = ParsePlyType(typeName);
            }
            if (!stream)
                throw std::runtime_error{"PLY property line is malformed"};
            elements.back().Properties.push_back(std::move(property));
        }
        // Anything else is a comment or an extension this reader does not need.
    }
    throw std::runtime_error{"PLY header has no end_header"};
}

// Reads positions and triangles out of a PLY. Elements and properties this
// renderer has no use for still have to be walked, because in a binary body they
// occupy bytes and skipping them wrong desynchronizes everything after.
void LoadPlyMesh(const std::string& path,
                 std::vector<Nanite::Position>& positions,
                 std::vector<std::uint32_t>& indices)
{
    // Binary mode even for an ascii body: getline has to leave the stream exactly
    // at the first payload byte, and a text-mode translation would move it.
    std::ifstream file{path, std::ios::binary};
    if (!file)
        throw std::runtime_error{"Unable to open PLY model: " + path};

    PlyFormat format = PlyFormat::Ascii;
    const std::vector<PlyElement> elements = ReadPlyHeader(file, format);

    for (const PlyElement& element : elements)
    {
        const bool isVertex = element.Name == "vertex";
        const bool isFace = element.Name == "face";
        if (isVertex)
            positions.reserve(positions.size() + static_cast<std::size_t>(element.Count));
        if (isFace)
            indices.reserve(indices.size() + static_cast<std::size_t>(element.Count) * 3u);

        for (std::uint64_t row = 0; row < element.Count; ++row)
        {
            Nanite::Position position{};
            bool positionSeen[3] = {false, false, false};
            for (const PlyProperty& property : element.Properties)
            {
                if (!property.IsList)
                {
                    const double value = ReadPlyValue(file, property.ValueType, format);
                    if (isVertex)
                    {
                        const int axis = property.Name == "x" ? 0 :
                            property.Name == "y" ? 1 : property.Name == "z" ? 2 : -1;
                        if (axis >= 0)
                        {
                            (&position.x)[axis] = static_cast<float>(value);
                            positionSeen[axis] = true;
                        }
                    }
                    continue;
                }

                const double rawCount = ReadPlyValue(file, property.CountType, format);
                if (rawCount < 0.0 || rawCount > 4096.0)
                    throw std::runtime_error{"PLY list property has an implausible length"};
                const std::size_t listCount = static_cast<std::size_t>(rawCount);
                const bool wantFace = isFace && property.Name == "vertex_indices";
                std::uint32_t corners[3] = {};
                for (std::size_t item = 0; item < listCount; ++item)
                {
                    const double value = ReadPlyValue(file, property.ValueType, format);
                    if (!wantFace)
                        continue;
                    if (value < 0.0 || value >= static_cast<double>(positions.size()))
                        throw std::runtime_error{"PLY face references an invalid vertex"};
                    // Fan triangulation, same as the OBJ path: keep the first
                    // corner and slide a window over the rest.
                    if (item < 2u)
                    {
                        corners[item] = static_cast<std::uint32_t>(value);
                        continue;
                    }
                    corners[2] = static_cast<std::uint32_t>(value);
                    indices.push_back(corners[0]);
                    indices.push_back(corners[1]);
                    indices.push_back(corners[2]);
                    corners[1] = corners[2];
                }
            }
            if (isVertex)
            {
                if (!positionSeen[0] || !positionSeen[1] || !positionSeen[2])
                    throw std::runtime_error{"PLY vertex element has no x/y/z properties"};
                positions.push_back(position);
            }
        }
    }
}

struct GltfMeshData
{
    std::vector<Nanite::Position> Positions;
    std::vector<Nanite::Position> Normals;
    std::vector<Nanite::TexCoord> TexCoords;
    std::vector<std::uint32_t> Indices;
    std::vector<std::uint32_t> VertexMaterialIndices;
    std::vector<Nanite::Material> Materials;
    std::vector<std::string> AlbedoTexturePaths;
    std::vector<std::string> NormalTexturePaths;
    std::vector<std::string> MetallicRoughnessTexturePaths;
};

// Loads an OBJ mesh together with its UVs. The generic loader keeps positions
// and faces only; the snow-mountain terrain needs the authored texcoords so it
// can sample its color.png satellite texture. UVs are stored per-vertex in the
// OBJ (the face tokens carry the /v/vt form), so the vt list lines up with the
// position list and is copied straight through.
void LoadObjMeshWithUvs(const std::string& path, GltfMeshData& mesh)
{
    std::ifstream file{path};
    if (!file)
        throw std::runtime_error{"Unable to open OBJ model: " + path};

    std::vector<Nanite::TexCoord> uvs;
    std::string line;
    while (std::getline(file, line))
    {
        std::istringstream stream{line};
        std::string tag;
        stream >> tag;
        if (tag == "v")
        {
            Nanite::Position position{};
            stream >> position.x >> position.y >> position.z;
            if (!stream)
                throw std::runtime_error{"OBJ vertex is malformed"};
            mesh.Positions.push_back(position);
        }
        else if (tag == "vt")
        {
            Nanite::TexCoord uv{};
            stream >> uv.u >> uv.v;
            if (!stream)
                throw std::runtime_error{"OBJ texcoord is malformed"};
            uvs.push_back(uv);
        }
        else if (tag == "f")
        {
            std::vector<std::string> face;
            std::string token;
            while (stream >> token)
                face.push_back(token);
            if (face.size() < 3)
                continue;
            const std::int32_t vertexCount = static_cast<std::int32_t>(mesh.Positions.size());
            const std::uint32_t first = static_cast<std::uint32_t>(ParseObjIndex(face[0], vertexCount));
            for (size_t corner = 1; corner + 1 < face.size(); ++corner)
            {
                mesh.Indices.push_back(first);
                mesh.Indices.push_back(static_cast<std::uint32_t>(ParseObjIndex(face[corner], vertexCount)));
                mesh.Indices.push_back(static_cast<std::uint32_t>(ParseObjIndex(face[corner + 1], vertexCount)));
            }
        }
    }
    if (mesh.Positions.empty() || mesh.Indices.empty())
        throw std::runtime_error{"OBJ model contains no geometry: " + path};
    // BuildSceneFromPositionsAndIndices replaces a size mismatch with zero UVs.
    mesh.TexCoords = uvs.size() == mesh.Positions.size() ?
        std::move(uvs) : std::vector<Nanite::TexCoord>(mesh.Positions.size());
}

Float3 TransformGltfPoint(const float* matrix, Float3 point)
{
    return {
        matrix[0] * point.x + matrix[4] * point.y + matrix[8] * point.z + matrix[12],
        matrix[1] * point.x + matrix[5] * point.y + matrix[9] * point.z + matrix[13],
        matrix[2] * point.x + matrix[6] * point.y + matrix[10] * point.z + matrix[14]};
}

Float3 TransformGltfDirection(const float* matrix, Float3 direction)
{
    return {
        matrix[0] * direction.x + matrix[4] * direction.y + matrix[8] * direction.z,
        matrix[1] * direction.x + matrix[5] * direction.y + matrix[9] * direction.z,
        matrix[2] * direction.x + matrix[6] * direction.y + matrix[10] * direction.z};
}

std::string DirectoryOf(const std::string& path)
{
    const std::size_t slash = path.find_last_of("/\\");
    return slash == std::string::npos ? std::string{} : path.substr(0, slash + 1u);
}

std::string GltfImagePath(const std::string& gltfPath, const cgltf_texture_view& view)
{
    if (view.texture == nullptr || view.texture->image == nullptr ||
        view.texture->image->uri == nullptr)
        return {};
    const std::string uri = view.texture->image->uri;
    // The downloaded PolyHaven files use external JPGs. Embedded data URIs are
    // intentionally left for the generated fallback texture until a staging
    // decoder is needed for another asset.
    if (uri.rfind("data:", 0u) == 0u)
        return {};
    return DirectoryOf(gltfPath) + uri;
}

std::uint32_t AddGltfMaterial(const std::string& gltfPath,
                              const cgltf_material* source,
                              GltfMeshData& output,
                              std::unordered_map<const cgltf_material*, std::uint32_t>& materialMap)
{
    if (source != nullptr)
    {
        const auto found = materialMap.find(source);
        if (found != materialMap.end())
            return found->second;
    }

    Nanite::Material material{};
    material.BaseColor[0] = 1.0f;
    material.BaseColor[1] = 1.0f;
    material.BaseColor[2] = 1.0f;
    material.BaseColor[3] = 1.0f;
    material.Roughness = 0.8f;
    material.AmbientOcclusion = 1.0f;
    if (source != nullptr && source->has_pbr_metallic_roughness)
    {
        const cgltf_pbr_metallic_roughness& pbr = source->pbr_metallic_roughness;
        for (int component = 0; component < 4; ++component)
            material.BaseColor[component] = pbr.base_color_factor[component];
        material.Metallic = pbr.metallic_factor;
        material.Roughness = pbr.roughness_factor;
        material.AmbientOcclusion = source->occlusion_texture.texture != nullptr ?
            source->occlusion_texture.scale : 1.0f;
    }

    material.TextureIndex = static_cast<std::uint32_t>(output.Materials.size());
    const std::uint32_t materialIndex = material.TextureIndex;
    output.Materials.push_back(material);
    output.AlbedoTexturePaths.push_back(source != nullptr && source->has_pbr_metallic_roughness ?
        GltfImagePath(gltfPath, source->pbr_metallic_roughness.base_color_texture) : std::string{});
    output.NormalTexturePaths.push_back(source != nullptr ?
        GltfImagePath(gltfPath, source->normal_texture) : std::string{});
    output.MetallicRoughnessTexturePaths.push_back(
        source != nullptr && source->has_pbr_metallic_roughness ?
            GltfImagePath(gltfPath, source->pbr_metallic_roughness.metallic_roughness_texture) :
            std::string{});
    if (source != nullptr)
        materialMap.emplace(source, materialIndex);
    return materialIndex;
}

void AppendGltfAsset(const std::string& path,
                     Float3 scale,
                     Float3 translation,
                     GltfMeshData& output)
{
    cgltf_options options{};
    cgltf_data* data = nullptr;
    if (cgltf_parse_file(&options, path.c_str(), &data) != cgltf_result_success || data == nullptr)
        throw std::runtime_error{"Unable to parse glTF model: " + path};
    const auto releaseData = [&]() { cgltf_free(data); };
    if (cgltf_load_buffers(&options, data, path.c_str()) != cgltf_result_success)
    {
        releaseData();
        throw std::runtime_error{"Unable to load glTF buffers: " + path};
    }

    std::unordered_map<const cgltf_material*, std::uint32_t> materialMap;
    for (cgltf_size nodeIndex = 0; nodeIndex < data->nodes_count; ++nodeIndex)
    {
        const cgltf_node& node = data->nodes[nodeIndex];
        if (node.mesh == nullptr)
            continue;
        float nodeMatrix[16] = {};
        cgltf_node_transform_world(&node, nodeMatrix);
        for (cgltf_size primitiveMesh = 0; primitiveMesh < node.mesh->primitives_count; ++primitiveMesh)
        {
            const cgltf_primitive& primitive = node.mesh->primitives[primitiveMesh];
            if (primitive.type != cgltf_primitive_type_triangles)
                continue;
            const cgltf_accessor* positions = nullptr;
            const cgltf_accessor* normals = nullptr;
            const cgltf_accessor* texCoords = nullptr;
            for (cgltf_size attributeIndex = 0; attributeIndex < primitive.attributes_count; ++attributeIndex)
            {
                const cgltf_attribute& attribute = primitive.attributes[attributeIndex];
                if (attribute.type == cgltf_attribute_type_position && attribute.index == 0)
                    positions = attribute.data;
                else if (attribute.type == cgltf_attribute_type_normal && attribute.index == 0)
                    normals = attribute.data;
                else if (attribute.type == cgltf_attribute_type_texcoord && attribute.index == 0)
                    texCoords = attribute.data;
            }
            if (positions == nullptr)
                continue;
            const std::uint32_t materialIndex =
                AddGltfMaterial(path, primitive.material, output, materialMap);
            const std::size_t baseVertex = output.Positions.size();
            output.Positions.reserve(baseVertex + positions->count);
            output.Normals.reserve(baseVertex + positions->count);
            output.TexCoords.reserve(baseVertex + positions->count);
            output.VertexMaterialIndices.reserve(baseVertex + positions->count);
            for (cgltf_size vertex = 0; vertex < positions->count; ++vertex)
            {
                float positionValue[3] = {};
                if (!cgltf_accessor_read_float(positions, vertex, positionValue, 3))
                {
                    releaseData();
                    throw std::runtime_error{"glTF position accessor is unreadable: " + path};
                }
                Float3 transformed = TransformGltfPoint(nodeMatrix,
                    {positionValue[0], positionValue[1], positionValue[2]});
                transformed.x = transformed.x * scale.x + translation.x;
                transformed.y = transformed.y * scale.y + translation.y;
                transformed.z = transformed.z * scale.z + translation.z;
                output.Positions.push_back({transformed.x, transformed.y, transformed.z, 0.0f});

                float normalValue[3] = {0.0f, 1.0f, 0.0f};
                if (normals != nullptr)
                    cgltf_accessor_read_float(normals, vertex, normalValue, 3);
                const Float3 transformedNormal = TransformGltfDirection(nodeMatrix,
                    {normalValue[0], normalValue[1], normalValue[2]});
                output.Normals.push_back({transformedNormal.x, transformedNormal.y,
                                          transformedNormal.z, 0.0f});
                float uvValue[2] = {};
                if (texCoords != nullptr)
                    cgltf_accessor_read_float(texCoords, vertex, uvValue, 2);
                output.TexCoords.push_back({uvValue[0], uvValue[1]});
                output.VertexMaterialIndices.push_back(materialIndex);
            }

            if (primitive.indices != nullptr)
            {
                for (cgltf_size index = 0; index < primitive.indices->count; ++index)
                {
                    const cgltf_size localIndex = cgltf_accessor_read_index(primitive.indices, index);
                    if (localIndex >= positions->count)
                    {
                        releaseData();
                        throw std::runtime_error{"glTF index accessor is out of range: " + path};
                    }
                    output.Indices.push_back(static_cast<std::uint32_t>(baseVertex + localIndex));
                }
            }
            else
            {
                for (cgltf_size index = 0; index < positions->count; ++index)
                    output.Indices.push_back(static_cast<std::uint32_t>(baseVertex + index));
            }
        }
    }
    releaseData();
}

// Offline .nanite cache: the heavy clusterlod build for the multi-million
// triangle terrain meshes runs once and is stored as a binary sidecar next to
// the source OBJ. The runtime loads the cache instead of re-parsing and
// re-clustering every launch. The format is deliberately a flat POD dump:
// every array is written with a 4-char tag, element count and element size,
// and string vectors store length-prefixed UTF-8, so the loader can validate
// tags and element sizes before trusting the file.
void SaveCpuSceneCache(const Nanite::CpuScene& scene, const std::string& path)
{
    std::ofstream out(path, std::ios::binary);
    if (!out)
        throw std::runtime_error{"Unable to write nanite cache: " + path};
    const auto writeHeader = [&](const char* tag, std::uint64_t count, std::uint32_t elementSize) {
        out.write(tag, 4);
        out.write(reinterpret_cast<const char*>(&count), sizeof(count));
        out.write(reinterpret_cast<const char*>(&elementSize), sizeof(elementSize));
    };
    const auto writePodArray = [&](const char* tag, const void* data, std::size_t count, std::size_t elementSize) {
        writeHeader(tag, count, static_cast<std::uint32_t>(elementSize));
        if (count != 0u)
            out.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(count * elementSize));
    };
    const auto writeStringArray = [&](const char* tag, const std::vector<std::string>& values) {
        writeHeader(tag, values.size(), 0u);
        for (const std::string& value : values)
        {
            const std::uint32_t length = static_cast<std::uint32_t>(value.size());
            out.write(reinterpret_cast<const char*>(&length), sizeof(length));
            out.write(value.data(), static_cast<std::streamsize>(value.size()));
        }
    };
    const char* magic = "NAN1";
    out.write(magic, 4);
    out.write(reinterpret_cast<const char*>(&scene.SourceTriangleCount), sizeof(scene.SourceTriangleCount));
    writePodArray("posi", scene.Positions.data(), scene.Positions.size(), sizeof(Nanite::Position));
    writePodArray("norm", scene.Normals.data(), scene.Normals.size(), sizeof(Nanite::Position));
    writePodArray("texc", scene.TexCoords.data(), scene.TexCoords.size(), sizeof(Nanite::TexCoord));
    writePodArray("indi", scene.Indices.data(), scene.Indices.size(), sizeof(std::uint32_t));
    writePodArray("mtrI", scene.MaterialIndices.data(), scene.MaterialIndices.size(), sizeof(std::uint32_t));
    writePodArray("matr", scene.Materials.data(), scene.Materials.size(), sizeof(Nanite::Material));
    writeStringArray("sAlb", scene.AlbedoTexturePaths);
    writeStringArray("sNrm", scene.NormalTexturePaths);
    writeStringArray("sMtl", scene.MetallicRoughnessTexturePaths);
    writePodArray("cpos", scene.ClusterPositions.data(), scene.ClusterPositions.size(), sizeof(Nanite::Position));
    writePodArray("clin", scene.ClusterLocalIndices.data(), scene.ClusterLocalIndices.size(), sizeof(std::uint32_t));
    writePodArray("clof", scene.ClusterLocalIndexOffsets.data(), scene.ClusterLocalIndexOffsets.size(), sizeof(std::uint32_t));
    writePodArray("inst", scene.Instances.data(), scene.Instances.size(), sizeof(Nanite::Instance));
    writePodArray("itri", scene.InstanceTriangleCounts.data(), scene.InstanceTriangleCounts.size(), sizeof(std::uint64_t));
    out.write(reinterpret_cast<const char*>(&scene.CustomInstanceLayout), sizeof(scene.CustomInstanceLayout));
    writePodArray("node", scene.Nodes.data(), scene.Nodes.size(), sizeof(Nanite::DagNode));
    writePodArray("grp ", scene.ClusterGroups.data(), scene.ClusterGroups.size(), sizeof(Nanite::ClusterGroup));
    writePodArray("clus", scene.Clusters.data(), scene.Clusters.size(), sizeof(Nanite::Cluster));
    out.write(reinterpret_cast<const char*>(&scene.MaxClusterIndexCount), sizeof(scene.MaxClusterIndexCount));
    if (!out)
        throw std::runtime_error{"Failed to write nanite cache: " + path};
}

Nanite::CpuScene LoadCpuSceneCache(const std::string& path)
{
    std::ifstream in(path, std::ios::binary);
    if (!in)
        throw std::runtime_error{"Unable to open nanite cache: " + path};
    const auto readHeader = [&](const char* expectedTag, std::uint64_t* count, std::uint32_t* elementSize) {
        char tag[4] = {};
        in.read(tag, 4);
        if (std::memcmp(tag, expectedTag, 4) != 0)
            throw std::runtime_error{"nanite cache tag mismatch: " + std::string{tag, 4} + " != " + expectedTag};
        in.read(reinterpret_cast<char*>(count), sizeof(*count));
        in.read(reinterpret_cast<char*>(elementSize), sizeof(*elementSize));
    };
    Nanite::CpuScene scene;
    char magic[4] = {};
    in.read(magic, 4);
    if (std::memcmp(magic, "NAN1", 4) != 0)
        throw std::runtime_error{"nanite cache has an invalid magic"};
    in.read(reinterpret_cast<char*>(&scene.SourceTriangleCount), sizeof(scene.SourceTriangleCount));
    const auto readPodArray = [&](const char* tag, auto& vector, std::size_t expectedElementSize) {
        std::uint64_t count = 0u;
        std::uint32_t elementSize = 0u;
        readHeader(tag, &count, &elementSize);
        if (elementSize != expectedElementSize)
            throw std::runtime_error{"nanite cache element size mismatch for " + std::string{tag, 4}};
        vector.resize(static_cast<std::size_t>(count));
        if (count != 0u)
            in.read(reinterpret_cast<char*>(vector.data()), static_cast<std::streamsize>(count * elementSize));
    };
    const auto readStringArray = [&](const char* tag, std::vector<std::string>& values) {
        std::uint64_t count = 0u;
        std::uint32_t elementSize = 0u;
        readHeader(tag, &count, &elementSize);
        values.resize(static_cast<std::size_t>(count));
        for (std::string& value : values)
        {
            std::uint32_t length = 0u;
            in.read(reinterpret_cast<char*>(&length), sizeof(length));
            value.resize(length);
            if (length != 0u)
                in.read(value.data(), static_cast<std::streamsize>(length));
        }
    };
    readPodArray("posi", scene.Positions, sizeof(Nanite::Position));
    readPodArray("norm", scene.Normals, sizeof(Nanite::Position));
    readPodArray("texc", scene.TexCoords, sizeof(Nanite::TexCoord));
    readPodArray("indi", scene.Indices, sizeof(std::uint32_t));
    readPodArray("mtrI", scene.MaterialIndices, sizeof(std::uint32_t));
    readPodArray("matr", scene.Materials, sizeof(Nanite::Material));
    readStringArray("sAlb", scene.AlbedoTexturePaths);
    readStringArray("sNrm", scene.NormalTexturePaths);
    readStringArray("sMtl", scene.MetallicRoughnessTexturePaths);
    readPodArray("cpos", scene.ClusterPositions, sizeof(Nanite::Position));
    readPodArray("clin", scene.ClusterLocalIndices, sizeof(std::uint32_t));
    readPodArray("clof", scene.ClusterLocalIndexOffsets, sizeof(std::uint32_t));
    readPodArray("inst", scene.Instances, sizeof(Nanite::Instance));
    readPodArray("itri", scene.InstanceTriangleCounts, sizeof(std::uint64_t));
    in.read(reinterpret_cast<char*>(&scene.CustomInstanceLayout), sizeof(scene.CustomInstanceLayout));
    readPodArray("node", scene.Nodes, sizeof(Nanite::DagNode));
    readPodArray("grp ", scene.ClusterGroups, sizeof(Nanite::ClusterGroup));
    readPodArray("clus", scene.Clusters, sizeof(Nanite::Cluster));
    in.read(reinterpret_cast<char*>(&scene.MaxClusterIndexCount), sizeof(scene.MaxClusterIndexCount));
    if (!in)
        throw std::runtime_error{"nanite cache is truncated: " + path};
    return scene;
}

Nanite::CpuScene BuildCoastalComposition(std::size_t requestedClusterTriangles)
{
    const bool forceRebuild = std::getenv("NANITE_FORCE_REBUILD_CACHE") != nullptr;

    // Load an OBJ terrain as a CpuScene, caching the cluster build in a
    // .nanite sidecar so the heavy clusterlod build runs once (offline) and
    // every later launch loads the cache instead.
    const auto loadTerrain = [&](const char* objRelative, const std::string& albedoPath, const char* cacheRelative) {
        const std::string objPath = std::string{DEMO_ASSET_DIR} + "/External/" + objRelative;
        const std::string cachePath = std::string{DEMO_ASSET_DIR} + "/External/" + cacheRelative;
        if (!forceRebuild && std::ifstream(cachePath, std::ios::binary))
        {
            Nanite::CpuScene terrain = LoadCpuSceneCache(cachePath);
            std::cerr << "Nanite loaded cache: " << cachePath
                      << " triangles=" << terrain.SourceTriangleCount << '\n';
            return terrain;
        }
        GltfMeshData terrainMesh;
        LoadObjMeshWithUvs(objPath, terrainMesh);
        terrainMesh.Materials = {
            {{1.0f, 1.0f, 1.0f, 1.0f}, 0.0f, 0.85f, 1.0f, 0u},
        };
        terrainMesh.AlbedoTexturePaths = {albedoPath};
        terrainMesh.NormalTexturePaths = {""};
        terrainMesh.MetallicRoughnessTexturePaths = {""};
        terrainMesh.VertexMaterialIndices.assign(terrainMesh.Positions.size(), 0u);
        Nanite::CpuScene terrain = BuildSceneFromPositionsAndIndices(
            std::move(terrainMesh.Positions), std::move(terrainMesh.Indices), 1u,
            requestedClusterTriangles, std::move(terrainMesh.TexCoords),
            std::move(terrainMesh.VertexMaterialIndices), std::move(terrainMesh.Materials),
            std::move(terrainMesh.AlbedoTexturePaths), std::move(terrainMesh.NormalTexturePaths),
            std::move(terrainMesh.MetallicRoughnessTexturePaths));
        std::cerr << "Nanite built terrain cache: " << cachePath
                  << " triangles=" << terrain.SourceTriangleCount << '\n';
        SaveCpuSceneCache(terrain, cachePath);
        return terrain;
    };

    // Snow Mountain 1 (Mesher.obj, 676k triangles) shrunk to one fifth of its
    // original placement, and Snow Mountain 2 (SnowMountain.obj, 16.7M
    // triangles) as the larger range behind it.
    const Nanite::CpuScene snow1 = loadTerrain(
        "SnowMountain/Mesher.obj",
        std::string{DEMO_ASSET_DIR} + "/External/SnowMountain/color.png",
        "SnowMountain/Mesher.nanite");
    const Nanite::CpuScene snow2 = loadTerrain(
        "SnowMontain/SnowMountain_lite.obj",
        std::string{DEMO_ASSET_DIR} + "/External/SnowMontain/SnowMountain.png",
        "SnowMontain/SnowMountain_lite.nanite");

    Nanite::CpuScene scene;
    scene.CustomInstanceLayout = true;
    const std::uint32_t snow1Root = AppendPrototypeScene(scene, snow1);
    const std::uint32_t snow2Root = AppendPrototypeScene(scene, snow2);

    // A 35x35 checkerboard of the two mountain prototypes so the whole field
    // reads as one continuous range: even cells get Snow Mountain 1, odd cells
    // Snow Mountain 2, ~612/613 instances each. Both are shrunk 50% further
    // than their single-instance placement: Snow Mountain 1 is 12 units wide
    // (its peak reaches about 2.4), Snow Mountain 2 is 10 units wide and is
    // laid flat (its authored Z relief maps onto +Y) with no lift, so its
    // relief runs from ground level up to about 2.5 and the two ranges crest at
    // the same height.
    //
    // Each instance gets a deterministic jitter so the range reads as
    // hand-placed rather than a perfect grid: the vertical scale varies by
    // +/-15% and the XZ position by +/-10% of the cell spacing. The random
    // stream is seeded from a fixed constant so the exact same layout is
    // produced on every launch, which keeps light baking reproducible offline.
    constexpr int GridSide = 35;
    constexpr int GridCount = GridSide * GridSide;
    constexpr float CellSpacing = 14.0f;
    const float gridHalf = static_cast<float>(GridSide - 1) * CellSpacing * 0.5f;
    std::mt19937 jitterRng(20260806u);
    std::uniform_real_distribution<float> heightJitter(0.85f, 1.15f);
    std::uniform_real_distribution<float> positionJitter(-0.1f, 0.1f);
    scene.Instances.reserve(scene.Instances.size() + GridCount);
    for (int row = 0; row < GridSide; ++row)
    {
        for (int column = 0; column < GridSide; ++column)
        {
            const int cell = row * GridSide + column;
            const float gx = static_cast<float>(column) * CellSpacing - gridHalf +
                positionJitter(jitterRng) * CellSpacing;
            const float gz = static_cast<float>(row) * CellSpacing - gridHalf +
                positionJitter(jitterRng) * CellSpacing;
            Nanite::Instance placed = (cell & 1) == 0 ?
                MakePlacedPrototypeInstance(snow1, snow1Root, 12.0f, gx, 0.0f, gz) :
                MakeFlatPlacementInstance(snow2, snow2Root, 10.0f, gx, 0.0f, gz);
            // Vertical scale jitter: Snow Mountain 1 scales its Y axis,
            // Snow Mountain 2 maps its authored Z relief onto world Y.
            const float heightFactor = heightJitter(jitterRng);
            if ((cell & 1) == 0)
                placed.WorldMatrix[5] *= heightFactor;
            else
                placed.WorldMatrix[9] *= heightFactor;
            scene.Instances.push_back(placed);
            scene.InstanceTriangleCounts.push_back(
                (cell & 1) == 0 ? snow1.SourceTriangleCount : snow2.SourceTriangleCount);
        }
    }

    if (scene.Instances.empty() || scene.Nodes.empty() || scene.Clusters.empty())
        throw std::runtime_error{"Snow terrain composition produced no instances"};
    std::cerr << "Nanite snow terrain: snow1=" << snow1.SourceTriangleCount
              << " snow2=" << snow2.SourceTriangleCount
              << " instances=" << scene.Instances.size() << '\n';
    return scene;
}

} // namespace

namespace Nanite
{

void LayoutInstances(std::vector<Instance>& instances,
                     std::uint64_t activeCount,
                     float spacingHorizontal,
                     float spacingVertical)
{
    const char* stressLayout = std::getenv("NANITE_STRESS_LAYOUT");
    const bool cubeLayout = stressLayout == nullptr ||
        std::string{stressLayout} != "behind";
    const std::uint64_t count = std::min<std::uint64_t>(activeCount, instances.size());
    // The cube side follows the active count, so the block stays compact and
    // centred on the camera whatever the count is. That is also why a count
    // change cannot be done by only shrinking a range: every transform moves.
    const std::uint64_t cubeSide = cubeLayout ? static_cast<std::uint64_t>(
        std::ceil(std::cbrt(static_cast<double>(std::max<std::uint64_t>(count, 1u))))) : 1u;
    const std::uint64_t cubePlane = cubeSide * cubeSide;
    for (std::uint64_t index = 0; index < count; ++index)
    {
        Instance& copy = instances[static_cast<std::size_t>(index)];
        if (cubeLayout)
        {
            const std::uint64_t layer = index / cubePlane;
            const std::uint64_t planeIndex = index % cubePlane;
            const std::uint64_t row = planeIndex / cubeSide;
            const std::uint64_t column = planeIndex % cubeSide;
            const float center = static_cast<float>(cubeSide - 1u) * 0.5f;
            copy.WorldMatrix[12] = (static_cast<float>(column) - center) * spacingHorizontal;
            copy.WorldMatrix[13] = (static_cast<float>(row) - center) * spacingVertical;
            // The nearest layer stays at a fixed -2 whatever the spacing is, so
            // squeezing the block pulls the far side towards the camera instead of
            // sliding the whole thing through it.
            copy.WorldMatrix[14] = -2.0f - static_cast<float>(layer) * spacingHorizontal;
        }
        else
        {
            copy.WorldMatrix[12] = 0.0f;
            copy.WorldMatrix[13] = 0.0f;
            copy.WorldMatrix[14] = -2.0f * static_cast<float>(index);
        }
    }
}

void LayoutInstanceGrid(std::vector<Instance>& instances,
                        std::uint32_t countX,
                        std::uint32_t countY,
                        std::uint32_t countZ,
                        float spacingHorizontal,
                        float spacingVertical)
{
    const std::uint64_t total = static_cast<std::uint64_t>(countX) * countY * countZ;
    const std::uint64_t count = std::min<std::uint64_t>(total, instances.size());
    const float centerX = static_cast<float>(countX - 1u) * 0.5f;
    const float centerY = static_cast<float>(countY - 1u) * 0.5f;
    for (std::uint64_t index = 0; index < count; ++index)
    {
        const std::uint64_t plane = static_cast<std::uint64_t>(countX) * countY;
        const std::uint64_t layer = index / plane;
        const std::uint64_t planeIndex = index % plane;
        const std::uint64_t row = planeIndex / countX;
        const std::uint64_t column = planeIndex % countX;
        Instance& copy = instances[static_cast<std::size_t>(index)];
        copy.WorldMatrix[12] = (static_cast<float>(column) - centerX) * spacingHorizontal;
        copy.WorldMatrix[13] = (static_cast<float>(row) - centerY) * spacingVertical;
        copy.WorldMatrix[14] = -2.0f - static_cast<float>(layer) * spacingHorizontal;
    }
}

std::string ResolveModelPath()
{
    if (const char* scene = std::getenv("NANITE_SCENE");
        scene != nullptr && std::string{scene} == "coastal")
        return "coastal_scene";
    const char* requested = std::getenv("NANITE_MODEL");
    if (requested == nullptr || *requested == '\0')
        return std::string{DEMO_ASSET_DIR} + "/happy_vrip.ply";
    if (requested[0] == '/')
        return std::string{requested};
    return std::string{DEMO_ASSET_DIR} + "/" + requested;
}

CpuScene LoadModelAndBuildScene(const std::string& path,
                                std::uint64_t instanceCount,
                                std::size_t clusterTriangles)
{
    if (path == "coastal_scene")
        return BuildCoastalComposition(clusterTriangles);

    std::vector<Position> positions;
    std::vector<std::uint32_t> indices;

    // PLY is picked by extension rather than by sniffing the magic, so a wrong
    // extension fails in the parser with a clear message instead of silently
    // reading one format as the other.
    const bool isPly = path.size() >= 4u &&
        std::equal(path.end() - 4, path.end(), ".ply",
                   [](char left, char right) { return std::tolower(left) == right; });
    if (isPly)
    {
        LoadPlyMesh(path, positions, indices);
        return BuildSceneFromPositionsAndIndices(
            std::move(positions), std::move(indices), instanceCount, clusterTriangles);
    }

    std::ifstream file{path};
    if (!file)
        throw std::runtime_error{"Unable to open OBJ model: " + path};

    std::string line;
    while (std::getline(file, line))
    {
        std::istringstream stream{line};
        std::string tag;
        stream >> tag;
        if (tag == "v")
        {
            Nanite::Position position{};
            stream >> position.x >> position.y >> position.z;
            if (!stream)
                throw std::runtime_error{"OBJ vertex is malformed"};
            positions.push_back(position);
        }
        else if (tag == "f")
        {
            std::vector<std::string> face;
            std::string token;
            while (stream >> token)
                face.push_back(token);
            if (face.size() < 3)
                continue;

            const std::int32_t vertexCount = static_cast<std::int32_t>(positions.size());
            const std::uint32_t first = static_cast<std::uint32_t>(ParseObjIndex(face[0], vertexCount));
            for (size_t corner = 1; corner + 1 < face.size(); ++corner)
            {
                indices.push_back(first);
                indices.push_back(static_cast<std::uint32_t>(ParseObjIndex(face[corner], vertexCount)));
                indices.push_back(static_cast<std::uint32_t>(ParseObjIndex(face[corner + 1], vertexCount)));
            }
        }
    }
    return BuildSceneFromPositionsAndIndices(
        std::move(positions), std::move(indices), instanceCount, clusterTriangles);
}

CpuScene BuildDemoScene()
{
    const std::vector<Position> positions = {
        {-1.0f, -1.0f, -1.0f, 0.0f}, {1.0f, -1.0f, -1.0f, 0.0f},
        {1.0f, 1.0f, -1.0f, 0.0f}, {-1.0f, 1.0f, -1.0f, 0.0f},
        {-1.0f, -1.0f, 1.0f, 0.0f}, {1.0f, -1.0f, 1.0f, 0.0f},
        {1.0f, 1.0f, 1.0f, 0.0f}, {-1.0f, 1.0f, 1.0f, 0.0f},
    };
    const std::vector<std::uint32_t> indices = {
        0, 1, 2, 2, 3, 0,
        4, 6, 5, 6, 4, 7,
        0, 4, 5, 5, 1, 0,
        1, 5, 6, 6, 2, 1,
        2, 6, 7, 7, 3, 2,
        4, 0, 3, 3, 7, 4,
    };
    return BuildSceneFromPositionsAndIndices(positions, indices, 0u, 0u);
}

} // namespace Nanite
