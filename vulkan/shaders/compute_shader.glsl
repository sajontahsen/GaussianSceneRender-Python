#version 450

layout(local_size_x = 16, local_size_y = 16) in;

struct Gaussian {
    float x, y;                 // Point position
    float r, g, b;              // RGB colors
    float ic11, ic12, ic21, ic22; // Inverse covariance matrix
    float opacity;              // Opacity
    float min_x, max_x, min_y, max_y; // Bounding ranges
};

layout(std430, binding = 0) readonly buffer GaussianBuffer {
    Gaussian gaussians[];
};

layout(std430, binding = 1) writeonly buffer ImageBuffer {
    vec4 pixels[]; // RGBA output
};

layout(push_constant) uniform PushConstants {
    ivec2 imageSize; // Image width and height
};

float compute_pixel_strength(vec2 pixel, vec2 point, mat2 inverse_covariance) {
    vec2 delta = pixel - point;
    float power = dot(delta, inverse_covariance * delta);
    return exp(-0.5 * power);
}

void main() {
    ivec2 pixelPos = ivec2(gl_GlobalInvocationID.xy);

    // Ensure we're within image bounds
    if (pixelPos.x >= imageSize.x || pixelPos.y >= imageSize.y) return;

    vec2 pixel = vec2(pixelPos);
    vec3 color = vec3(0.0);
    float totalWeight = 1.0;

    for (int i = 0; i < gaussians.length(); ++i) {
        Gaussian g = gaussians[i];

        // Check if the pixel is within the Gaussian's bounding box
        if (pixel.x < g.min_x || pixel.x > g.max_x ||
            pixel.y < g.min_y || pixel.y > g.max_y) {
            continue;
        }

        // Compute strength of the Gaussian at this pixel
        mat2 inverse_covariance = mat2(g.ic11, g.ic12, g.ic21, g.ic22);
        float strength = compute_pixel_strength(pixel, vec2(g.x, g.y), inverse_covariance);

        float alpha = min(0.99, g.opacity * strength);
        float weight = totalWeight * (1.0 - alpha);

        if (weight < 0.001) break;

        // Accumulate Gaussian contribution to the pixel color
        color += totalWeight * alpha * vec3(g.r, g.g, g.b);
        totalWeight = weight;
    }

    // Write the color to the image buffer
    uint index = pixelPos.y * imageSize.x + pixelPos.x;
    pixels[index] = vec4(color, 1.0); // RGBA
}
