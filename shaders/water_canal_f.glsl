#version 430

in vec3 vtx_pos;
in vec2 vtx_uv;

uniform float osg_FrameTime;
uniform sampler2D water_tex;  // Water texture passed from Python.

out vec4 p4d_FragColor;

void main() {
    // Adjust the texture size(tiling)
    vec2 uv = vtx_uv * 8.0;

    // A slow, flowing motion in the lower right corner.
    vec2 uv_flow1 = uv + vec2(osg_FrameTime * 0.05, osg_FrameTime * 0.03);
    
    // Create a slightly faster motion flowing in the opposite direction—toward the top left—to
    // completely eliminate the image's edges and repetition.
    vec2 uv_flow2 = uv * 1.3 + vec2(-osg_FrameTime * 0.04, -osg_FrameTime * 0.05);

    // Blend the two flowing textures. mix(x, y, a) => x * (1.0 - a) + y * a
    vec4 tex1 = texture(water_tex, uv_flow1);
    vec4 tex2 = texture(water_tex, uv_flow2);
    vec4 final_tex = mix(tex1, tex2, 0.5);

    // Depending on the wave height (Z-coordinate), add a bit of contrast to the brightness.
    // clamp(x, minVal, maxVal) -> min(max(x, minVal), maxVal)
    float height_factor = clamp((vtx_pos.z + 1.6) / 3.2, 0.0, 1.0);
    final_tex.rgb += vec3(0.1, 0.3, 0.2) * height_factor * 0.2;

    // Make it semi-transparent.
    float alpha = 0.75;

    p4d_FragColor = vec4(final_tex.rgb, alpha);
}