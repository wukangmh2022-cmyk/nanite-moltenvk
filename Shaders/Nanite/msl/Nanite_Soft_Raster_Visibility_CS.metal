#pragma clang diagnostic ignored "-Wmissing-prototypes"
#pragma clang diagnostic ignored "-Wmissing-braces"

// MVK_WORKGROUP_SIZE 64 1 1
// MVK_ENTRY_POINT main0

#include <metal_stdlib>
#include <simd/simd.h>

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

struct g_SoftDepth
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

kernel void main0(constant NaniteCullingConstants& _26 [[buffer(0)]], device g_QueueState& g_QueueState_1 [[buffer(1)]], device g_SoftwareDrawIndices& g_SoftwareDrawIndices_1 [[buffer(2)]], device g_DrawInstanceData& g_DrawInstanceData_1 [[buffer(3)]], device g_Instances& g_Instances_1 [[buffer(4)]], device g_Clusters& g_Clusters_1 [[buffer(5)]], device g_ClusterLocalIndexOffsets& g_ClusterLocalIndexOffsets_1 [[buffer(6)]], device g_ClusterPositionBuffer& g_ClusterPositionBuffer_1 [[buffer(7)]], device g_ClusterIndexBuffer& g_ClusterIndexBuffer_1 [[buffer(8)]], device g_SoftDepth& g_SoftDepth_1 [[buffer(9)]], device g_Visibility& g_Visibility_1 [[buffer(10)]], uint3 gl_WorkGroupID [[threadgroup_position_in_grid]], uint3 gl_LocalInvocationID [[thread_position_in_threadgroup]])
{
    // Native 64-bit depth+payload atomics are committed by the depth override;
    // this compatibility module keeps the reflected interface but does no second
    // coverage sweep.
    return;
    threadgroup spvUnsafeArray<float3, 128> s_ScreenVertices;
    do
    {
        if (gl_WorkGroupID.x >= ((_26.HybridRasterEnabled != 0u) ? g_QueueState_1._data[0].SoftClusterCount : g_QueueState_1._data[0].DrawCount))
        {
            break;
        }
        uint _794 = g_SoftwareDrawIndices_1._data[_26.MaxDrawCommands + gl_WorkGroupID.x].Value;
        uint _796 = (_26.HybridRasterEnabled != 0u) ? _794 : gl_WorkGroupID.x;
        uint _1349 = g_DrawInstanceData_1._data[_796].ClusterIndex;
        uint _1353 = g_DrawInstanceData_1._data[_796].IndexCount;
        uint _1398 = g_Clusters_1._data[_1349].VertexOffset;
        uint _857 = _1398 & 16777215u;
        uint _860 = min((_1398 >> 24u), 128u);
        uint _864 = g_ClusterLocalIndexOffsets_1._data[_1349];
        uint _867 = _1353 / 3u;
        float _870 = float(_26.ViewportWidth);
        float _873 = float(_26.ViewportHeight);
        int _877 = int(_26.ViewportWidth) - 1;
        int _881 = int(_26.ViewportHeight) - 1;
        for (uint _1401 = gl_LocalInvocationID.x; _1401 < _860; _1401 += 64u)
        {
            float4 _911 = _26.ViewProjMatrix * float4((g_Instances_1._data[g_DrawInstanceData_1._data[_796].InstanceIndex].WorldMatrix * float4(g_ClusterPositionBuffer_1._data[_857 + _1401], 1.0)).xyz, 1.0);
            if (_911.w <= 9.9999999747524270787835121154785e-07)
            {
                s_ScreenVertices[_1401] = float3(0.0, 0.0, -1.0);
                continue;
            }
            float _921 = 1.0 / _911.w;
            s_ScreenVertices[_1401] = float3((((_911.x * _921) * 0.5) + 0.5) * _870, (0.5 - ((_911.y * _921) * 0.5)) * _873, _911.z * _921);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        spvUnsafeArray<float3, 3> _746;
        for (uint _1402 = gl_LocalInvocationID.x; _1402 < _867; _1402 += 64u)
        {
            bool _1424;
            bool _1406;
            uint _1403 = 0u;
            bool _1407 = false;
            for (;;)
            {
                if (_1403 < 3u)
                {
                    uint _967 = (_864 + (_1402 * 3u)) + _1403;
                    if (g_ClusterIndexBuffer_1._data[_967].Value >= _860)
                    {
                        _1406 = true;
                        break;
                    }
                    _746[_1403] = s_ScreenVertices[g_ClusterIndexBuffer_1._data[_967].Value];
                    _1424 = (_746[_1403].z < 0.0) ? true : _1407;
                    _1403++;
                    _1407 = _1424;
                    continue;
                }
                else
                {
                    _1406 = _1407;
                    break;
                }
            }
            if (_1406)
            {
                continue;
            }
            float _1026 = ((_746[1].x - _746[0].x) * (_746[2].y - _746[0].y)) - ((_746[1].y - _746[0].y) * (_746[2].x - _746[0].x));
            if (abs(_1026) < 9.999999717180685365747194737196e-10)
            {
                continue;
            }
            float _1034 = (_1026 < 0.0) ? (-1.0) : 1.0;
            float2 _1039 = fast::min(fast::min(_746[0].xy, _746[1].xy), _746[2].xy);
            float2 _1044 = fast::max(fast::max(_746[0].xy, _746[1].xy), _746[2].xy);
            int _1049 = max(int(floor(_1039.x)), 0);
            int _1054 = max(int(floor(_1039.y)), 0);
            int _1060 = min(int(floor(_1044.x)), _877);
            int _1066 = min(int(floor(_1044.y)), _881);
            if ((_1049 > _1060) || (_1054 > _1066))
            {
                continue;
            }
            if (((_1060 - _1049) > 4096) || ((_1066 - _1054) > 4096))
            {
                continue;
            }
            if (((_26.HybridRasterEnabled != 0u) && (_26.ClusterRasterMode == 0u)) && ((abs(_1026) * 0.5) > _26.RasterBinAreaCutoff))
            {
                continue;
            }
            uint _1110 = ((_796 + 1u) << uint(8)) | (_1402 & 255u);
            float _1112 = 1.0 / _1026;
            for (int _1410 = _1054; _1410 <= _1066; _1410++)
            {
                for (int _1411 = _1049; _1411 <= _1060; _1411++)
                {
                    float _1129 = float(_1411) + 0.5;
                    float _1132 = float(_1410) + 0.5;
                    float _1156 = ((_746[1].x - _746[0].x) * (_1132 - _746[0].y)) - ((_746[1].y - _746[0].y) * (_1129 - _746[0].x));
                    float _1179 = ((_746[2].x - _746[1].x) * (_1132 - _746[1].y)) - ((_746[2].y - _746[1].y) * (_1129 - _746[1].x));
                    float _1202 = ((_746[0].x - _746[2].x) * (_1132 - _746[2].y)) - ((_746[0].y - _746[2].y) * (_1129 - _746[2].x));
                    if ((((_1156 * _1034) < 0.0) || ((_1179 * _1034) < 0.0)) || ((_1202 * _1034) < 0.0))
                    {
                        continue;
                    }
                    float _1234 = (((_1179 * _746[0].z) + (_1202 * _746[1].z)) + (_1156 * _746[2].z)) * _1112;
                    if ((_1234 < 0.0) || (_1234 > 1.0))
                    {
                        continue;
                    }
                    uint _1249 = (uint(_1410) * _26.ViewportWidth) + uint(_1411);
                    uint _1251 = as_type<uint>(_1234);
                    if (_1251 == g_SoftDepth_1._data[_1249])
                    {
                        g_Visibility_1._data[_1249].Payload = _1110;
                        g_Visibility_1._data[_1249].Key = ~_1251;
                    }
                }
            }
        }
        break;
    } while(false);
}
