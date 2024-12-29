#version 450

struct Gaussian {
    vec2 point;
    vec3 color;
    mat2 inverseCov;
    float opacity;
    float minX, maxX, minY, maxY;
};

layout(std430, binding = 0) buffer GaussianInput {
    Gaussian gaussians[];
};

layout(std430, binding = 1) buffer DebugOutput {
    Gaussian debugOutput[];
};

layout(local_size_x = 1) in;

void main() {
    uint idx = gl_GlobalInvocationID.x;

    // Copy input Gaussian to debug output
    debugOutput[idx] = gaussians[idx];
}
