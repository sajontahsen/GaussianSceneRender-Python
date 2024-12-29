#version 450

struct Gaussian {
    float x, y;                 // Point position
    float r, g, b;              // RGB colors
    float ic11, ic12, ic21, ic22; // Inverse covariance matrix
    float opacity;              // Opacity
    float min_x, max_x, min_y, max_y; // Bounding ranges
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
