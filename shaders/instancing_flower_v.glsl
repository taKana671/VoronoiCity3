#version 430

uniform mat4 p3d_ModelViewProjectionMatrix;
in vec4 p3d_Vertex;
in vec3 p3d_Normal;
in vec2 p3d_MultiTexCoord0;
in vec4 p3d_Color;

uniform float osg_FrameTime;

out vec2 texcoord;
out vec4 vertex_color;

struct Instanced {
    mat4 matrix;
};

// Enable the system to receive data from ShaderBuffer('DataBuffer', ...) sent from Panda3D (Python)
layout(std430) buffer instanced_object {
    Instanced nodes[];
};

void main() {
    float max_h = 1.0;

    // Get the matrix (transform) for this instance.
    mat4 transform = nodes[gl_InstanceID].matrix;
    
    // Copy the raw vertex coordinates.
    vec4 local_vertex = p3d_Vertex;

    // The model's original height(z) is divided by max_h to normalize it to a range of 0.0 to 1.0.
    float height_weight = clamp(local_vertex.z / max_h, 0.0, 1.0);
    height_weight = height_weight * height_weight;

    // In the standard specification for Panda3D (row-major), the translation information
    // for that object is directly embedded in the fourth line at the bottom.
    // transform[3][0]: x, [3][1]: y
    // To prevent all the flowers from swaying to the right or left at the same time.
    float wave_x = sin(osg_FrameTime * 3.0 + transform[3][0] * 0.12) * 0.15;
    float wave_y = cos(osg_FrameTime * 2.2 + transform[3][1] * 0.12) * 0.10;

    // Multiply by a weight based on height.
    local_vertex.x += wave_x * height_weight;
    local_vertex.y += wave_y * height_weight;

    // Finally, multiply the matrix by the clean vertices after all transformations are complete.
    vec4 vertex_position = transform * local_vertex;

    gl_Position = p3d_ModelViewProjectionMatrix * vertex_position;
    texcoord = p3d_MultiTexCoord0;
    vertex_color = p3d_Color;
}
