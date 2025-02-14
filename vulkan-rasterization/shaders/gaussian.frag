#version 450

layout(location = 0) in vec3 fragColor;
layout(location = 1) in float fragOpacity;

layout(location = 0) out vec4 outColor;

void main() {
    // apply alpha blending (assumes sorted order)
    outColor = vec4(fragColor, fragOpacity);
}
