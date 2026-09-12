#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wmissing-prototypes"
#pragma clang diagnostic ignored "-Wmissing-braces"

// MVK_WORKGROUP_SIZE 64 1 1
// MVK_ENTRY_POINT main0

#include <metal_stdlib>
#include <simd/simd.h>
#include <metal_atomic>

using namespace metal;

template<typename T, size_t Num>
struct spvUnsafeArray
{
    T elements[Num ? Num : 1];
    
    thread T& operator [] (size_t pos) thread
    {
        return elements[pos];
    }
    constexpr const thread T& operator [] (size_t pos) const thread
    {
        return elements[pos];
    }
    
    device T& operator [] (size_t pos) device
    {
        return elements[pos];
    }
    constexpr const device T& operator [] (size_t pos) const device
    {
        return elements[pos];
    }
    
    constexpr const constant T& operator [] (size_t pos) const constant
    {
        return elements[pos];
    }
    
    threadgroup T& operator [] (size_t pos) threadgroup
    {
        return elements[pos];
    }
    constexpr const threadgroup T& operator [] (size_t pos) const threadgroup
    {
        return elements[pos];
    }
};

struct NaniteCullingConstants
{
    float4x4 ViewProjMatrix;
    float4x4 PreviousViewProjMatrix;
    packed_float3 CameraWorldPosition;
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
    uint CullingPadding2;
    float RasterBinAreaCutoff;
    uint HybridRasterEnabled;
    uint ClusterRasterMode;
    uint CullingPadding4;
};

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
    uint DispatchedClusterTasks;
    uint DispatchedSeedTasks;
    uint DispatchedPostClusterTasks;
    uint SoftRasterTriangles;
    uint SoftRasterPixels;
    uint SoftRasterSkippedLarge;
    uint SoftRasterSkippedClip;
    uint SoftDepthMatched;
    uint SoftDepthMismatch;
    uint SoftDepthMissing;
    uint SoftDepthExtra;
    uint SoftDepthMaxDiff;
    uint SoftVisMissing;
    uint SoftVisKeyMismatch;
    uint SoftVisResolved;
    uint SoftVisNoCover;
    uint SoftVisDepthOff;
    uint SoftClusterCount;
    uint HardwareClusterCount;
    uint SoftPadding2;
};

struct g_QueueState
{
    NaniteQueueState _data[1];
};

struct NaniteSoftwareDrawIndexElement
{
    uint Value;
};

struct g_SoftwareDrawIndices
{
    NaniteSoftwareDrawIndexElement _data[1];
};

struct NaniteDrawInstanceData
{
    uint InstanceIndex;
    uint ClusterIndex;
    uint FirstIndex;
    uint IndexCount;
};

struct g_DrawInstanceData
{
    NaniteDrawInstanceData _data[1];
};

struct NaniteInstance
{
    float4x4 WorldMatrix;
    float4 LocalBoundSphere;
    packed_float3 BBoxMin;
    uint BBoxMinPadding;
    packed_float3 BBoxMax;
    uint BBoxMaxPadding;
    uint RootNodeIndex;
    uint Padding0;
    uint Padding1;
    uint Padding2;
};

struct g_Instances
{
    NaniteInstance _data[1];
};

struct NaniteCluster
{
    float4 BoundSphere;
    packed_float3 BBoxMin;
    uint BBoxMinPadding;
    packed_float3 BBoxMax;
    float MaxError;
    uint IndexCount;
    uint FirstIndex;
    uint VertexOffset;
    uint RefinedGroupIndex;
};

struct g_Clusters
{
    NaniteCluster _data[1];
};

struct g_ClusterLocalIndexOffsets
{
    uint _data[1];
};

struct NaniteVisibility
{
    uint Payload;
    uint Key;
};

struct g_Visibility
{
    NaniteVisibility _data[1];
};

struct g_SoftDepth
{
    uint _data[1];
};

struct g_ClusterPositionBuffer
{
    float3 _data[1];
};

struct NaniteClusterIndexElement
{
    uint Value;
};

struct g_ClusterIndexBuffer
{
    NaniteClusterIndexElement _data[1];
};

kernel void main0(constant NaniteCullingConstants& _26 [[buffer(0)]], device g_QueueState& g_QueueState_1 [[buffer(1)]], device g_SoftwareDrawIndices& g_SoftwareDrawIndices_1 [[buffer(2)]], device g_DrawInstanceData& g_DrawInstanceData_1 [[buffer(3)]], device g_Instances& g_Instances_1 [[buffer(4)]], device g_Clusters& g_Clusters_1 [[buffer(5)]], device g_ClusterLocalIndexOffsets& g_ClusterLocalIndexOffsets_1 [[buffer(6)]], device g_Visibility& g_Visibility_1 [[buffer(7)]], device g_SoftDepth& g_SoftDepth_1 [[buffer(8)]], device g_ClusterPositionBuffer& g_ClusterPositionBuffer_1 [[buffer(9)]], device g_ClusterIndexBuffer& g_ClusterIndexBuffer_1 [[buffer(10)]], uint3 gl_WorkGroupID [[threadgroup_position_in_grid]], uint3 gl_LocalInvocationID [[thread_position_in_threadgroup]])
{
    threadgroup spvUnsafeArray<float3, 128> s_ScreenVertices;
    do
    {
        if (gl_WorkGroupID.x >= ((_26.HybridRasterEnabled != 0u) ? g_QueueState_1._data[0].SoftClusterCount : g_QueueState_1._data[0].DrawCount))
        {
            break;
        }
        uint _835 = g_SoftwareDrawIndices_1._data[_26.MaxDrawCommands + gl_WorkGroupID.x].Value;
        uint _837 = (_26.HybridRasterEnabled != 0u) ? _835 : gl_WorkGroupID.x;
        uint _1418 = g_DrawInstanceData_1._data[_837].InstanceIndex;
        uint _1420 = g_DrawInstanceData_1._data[_837].ClusterIndex;
        uint _1424 = g_DrawInstanceData_1._data[_837].IndexCount;
        float4x4 _1427 = transpose(g_Instances_1._data[_1418].WorldMatrix);
        uint _1469 = g_Clusters_1._data[_1420].VertexOffset;
        uint _898 = _1469 & 16777215u;
        uint _901 = min((_1469 >> 24u), 128u);
        uint _905 = g_ClusterLocalIndexOffsets_1._data[_1420];
        uint _908 = _1424 / 3u;
        if (g_Visibility_1._data[0].Key == 4294967295u)
        {
            g_SoftDepth_1._data[0] = g_Visibility_1._data[0].Key;
        }
        float _919 = float(_26.ViewportWidth);
        float _922 = float(_26.ViewportHeight);
        int _926 = int(_26.ViewportWidth) - 1;
        int _930 = int(_26.ViewportHeight) - 1;
        for (uint _1472 = gl_LocalInvocationID.x; _1472 < _901; _1472 += 64u)
        {
            float4 _960 = _26.ViewProjMatrix * float4((float4(g_ClusterPositionBuffer_1._data[_898 + _1472], 1.0) * _1427).xyz, 1.0);
            if (_960.w <= 9.9999999747524270787835121154785e-07)
            {
                s_ScreenVertices[_1472] = float3(0.0, 0.0, -1.0);
                continue;
            }
            float _970 = 1.0 / _960.w;
            s_ScreenVertices[_1472] = float3((((_960.x * _970) * 0.5) + 0.5) * _919, (0.5 - ((_960.y * _970) * 0.5)) * _922, _960.z * _970);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint _1474;
        uint _1476;
        uint _1479;
        uint _1483;
        _1483 = 0u;
        _1479 = 0u;
        _1476 = 0u;
        _1474 = 0u;
        spvUnsafeArray<float3, 3> _785;
        uint _1514;
        uint _1518;
        uint _1519;
        uint _1523;
        for (uint _1473 = gl_LocalInvocationID.x; _1473 < _908; _1483 = _1523, _1479 = _1519, _1476 = _1518, _1474 = _1514, _1473 += 64u)
        {
            bool _1530;
            bool _1487;
            uint _1484 = 0u;
            bool _1488 = false;
            for (;;)
            {
                if (_1484 < 3u)
                {
                    uint _1016 = (_905 + (_1473 * 3u)) + _1484;
                    if (g_ClusterIndexBuffer_1._data[_1016].Value >= _901)
                    {
                        _1487 = true;
                        break;
                    }
                    _785[_1484] = s_ScreenVertices[g_ClusterIndexBuffer_1._data[_1016].Value];
                    _1530 = (_785[_1484].z < 0.0) ? true : _1488;
                    _1484++;
                    _1488 = _1530;
                    continue;
                }
                else
                {
                    _1487 = _1488;
                    break;
                }
            }
            if (_1487)
            {
                _1523 = _1483 + uint(1);
                _1519 = _1479;
                _1518 = _1476;
                _1514 = _1474;
                continue;
            }
            float _1075 = ((_785[1].x - _785[0].x) * (_785[2].y - _785[0].y)) - ((_785[1].y - _785[0].y) * (_785[2].x - _785[0].x));
            if (abs(_1075) < 9.999999717180685365747194737196e-10)
            {
                _1523 = _1483;
                _1519 = _1479;
                _1518 = _1476;
                _1514 = _1474;
                continue;
            }
            float _1083 = (_1075 < 0.0) ? (-1.0) : 1.0;
            float2 _1088 = fast::min(fast::min(_785[0].xy, _785[1].xy), _785[2].xy);
            float2 _1093 = fast::max(fast::max(_785[0].xy, _785[1].xy), _785[2].xy);
            int _1098 = max(int(floor(_1088.x)), 0);
            int _1103 = max(int(floor(_1088.y)), 0);
            int _1109 = min(int(floor(_1093.x)), _926);
            int _1115 = min(int(floor(_1093.y)), _930);
            if ((_1098 > _1109) || (_1103 > _1115))
            {
                _1523 = _1483;
                _1519 = _1479;
                _1518 = _1476;
                _1514 = _1474;
                continue;
            }
            if (((_1109 - _1098) > 4096) || ((_1115 - _1103) > 4096))
            {
                _1523 = _1483;
                _1519 = _1479 + uint(1);
                _1518 = _1476;
                _1514 = _1474;
                continue;
            }
            if (((_26.HybridRasterEnabled != 0u) && (_26.ClusterRasterMode == 0u)) && ((abs(_1075) * 0.5) > _26.RasterBinAreaCutoff))
            {
                _1523 = _1483;
                _1519 = _1479;
                _1518 = _1476;
                _1514 = _1474;
                continue;
            }
            float _1163 = 1.0 / _1075;
            // Evaluate the three edge functions once at the first sample and
            // then walk them with constant deltas. The old generated loop did
            // six multiplies and six subtracts for every candidate pixel;
            // incremental scan conversion keeps the same sample convention
            // while moving that work to the triangle/row setup.
            const float _dx01 = _785[1].x - _785[0].x;
            const float _dy01 = _785[1].y - _785[0].y;
            const float _dx12 = _785[2].x - _785[1].x;
            const float _dy12 = _785[2].y - _785[1].y;
            const float _dx20 = _785[0].x - _785[2].x;
            const float _dy20 = _785[0].y - _785[2].y;
            const float _sampleX0 = float(_1098) + 0.5;
            uint _1497;
            _1497 = _1476;
            uint _1496;
            for (int _1493 = _1103; _1493 <= _1115; _1493++, _1497 = _1496)
            {
                _1496 = _1497;
                uint _1529;
                const float _sampleY = float(_1493) + 0.5;
                float _edge0 = _dx01 * (_sampleY - _785[0].y) -
                    _dy01 * (_sampleX0 - _785[0].x);
                float _edge1 = _dx12 * (_sampleY - _785[1].y) -
                    _dy12 * (_sampleX0 - _785[1].x);
                float _edge2 = _dx20 * (_sampleY - _785[2].y) -
                    _dy20 * (_sampleX0 - _785[2].x);
                for (int _1494 = _1098; _1494 <= _1109; _1496 = _1529, _1494++)
                {
                    const bool _inside = !(((_edge0 * _1083) < 0.0) ||
                        ((_edge1 * _1083) < 0.0) || ((_edge2 * _1083) < 0.0));
                    if (_inside)
                    {
                        const float _1285 = ((_edge1 * _785[0].z) +
                            (_edge2 * _785[1].z) + (_edge0 * _785[2].z)) * _1163;
                        if (_1285 >= 0.0 && _1285 <= 1.0)
                        {
                            const uint _pixel = (uint(_1493) * _26.ViewportWidth) +
                                uint(_1494);
                            // Metal exposes the 64-bit min/max operation even
                            // though Vulkan's shaderBufferInt64Atomics feature
                            // is false on this device. Pack nearest depth and
                            // payload into one atomic word so they cannot tear.
                            const ulong _candidate =
                                (ulong(~as_type<uint>(_1285)) << 32) |
                                (ulong(((_837 + 1u) << 8) | (_1473 & 255u)));
                            atomic_max_explicit(
                                (device atomic_ulong*)&g_Visibility_1._data[_pixel],
                                _candidate,
                                memory_order_relaxed);
                            _1529 = _1496 + uint(1);
                        }
                        else
                        {
                            _1529 = _1496;
                        }
                    }
                    else
                    {
                        _1529 = _1496;
                    }
                    _edge0 -= _dy01;
                    _edge1 -= _dy12;
                    _edge2 -= _dy20;
                }
            }
            _1523 = _1483;
            _1519 = _1479;
            _1518 = _1497;
            _1514 = _1474 + uint(1);
        }
        if (_1474 != 0u)
        {
            uint _1326 = atomic_fetch_add_explicit((device atomic_uint*)&g_QueueState_1._data[0].SoftRasterTriangles, _1474, memory_order_relaxed);
        }
        if (_1476 != 0u)
        {
            uint _1333 = atomic_fetch_add_explicit((device atomic_uint*)&g_QueueState_1._data[0].SoftRasterPixels, _1476, memory_order_relaxed);
        }
        if (_1479 != 0u)
        {
            uint _1340 = atomic_fetch_add_explicit((device atomic_uint*)&g_QueueState_1._data[0].SoftRasterSkippedLarge, _1479, memory_order_relaxed);
        }
        if (_1483 != 0u)
        {
            uint _1347 = atomic_fetch_add_explicit((device atomic_uint*)&g_QueueState_1._data[0].SoftRasterSkippedClip, _1483, memory_order_relaxed);
        }
        break;
    } while(false);
}
