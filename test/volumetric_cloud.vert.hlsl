struct FullscreenVertex
{
    float4 Position : SV_Position;
};

FullscreenVertex FullscreenVS(uint vertexID : SV_VertexID)
{
    const float2 positions[3] = {
        float2(-1.0, -1.0),
        float2(-1.0,  3.0),
        float2( 3.0, -1.0),
    };

    FullscreenVertex output;
    output.Position = float4(positions[vertexID], 0.0, 1.0);
    return output;
}
