#version 430

uniform float osg_FrameTime;
uniform mat4 p3d_ModelViewProjectionMatrix;
in vec4 p3d_Vertex;
in vec3 p3d_Normal;
in vec2 p3d_MultiTexCoord0;

out vec3 vtx_pos;
out vec2 vtx_uv;

void main() {
    vec4 pos = p3d_Vertex;

    // Main flow in the X-axis direction.
    pos.z += sin(pos.x * 0.15 + osg_FrameTime * 2.5) * 1.2;
    // Add subtle fluctuations along the Y-axis to eliminate monotony.
    pos.z += cos(pos.y * 0.3 + osg_FrameTime * 1.8) * 0.4;

    gl_Position = p3d_ModelViewProjectionMatrix * pos;
    vtx_pos = pos.xyz;
    vtx_uv = p3d_MultiTexCoord0;
}