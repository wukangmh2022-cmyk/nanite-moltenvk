#version 450

layout(location = 0) out vec3 Color;

void main()
{
    const vec2 Positions[3] = vec2[](
        vec2(0.0, -0.72),
        vec2(0.68, 0.52),
        vec2(-0.68, 0.52)
    );
    const vec3 Colors[3] = vec3[](
        vec3(0.98, 0.32, 0.18),
        vec3(0.18, 0.76, 0.98),
        vec3(0.96, 0.82, 0.20)
    );

    gl_Position = vec4(Positions[gl_VertexIndex], 0.0, 1.0);
    Color = Colors[gl_VertexIndex];
}
