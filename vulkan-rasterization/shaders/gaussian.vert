#version 450

layout(std140, binding = 1) uniform CameraBuffer {
    mat4 view;
    mat4 projection;
} camera;

struct Gaussian {
    vec3 position;  
    vec3 color;     
    mat3 covariance;
    float opacity;  
};

layout(std430, binding = 0) readonly buffer GaussianBuffer {
    Gaussian gaussians[];
};

layout(location = 0) out vec3 fragColor;
layout(location = 1) out float fragOpacity;

void main() {
    Gaussian g = gaussians[gl_VertexIndex];  

    // Convert 3D position to homogeneous coordinates
    vec4 worldPos = vec4(g.position, 1.0);

    // Transform to camera space
    vec4 cameraPos = camera.view * worldPos;

    // Transform to clip space 
    vec4 clipPos = camera.projection * cameraPos;

    // Pass data to the fragment shader
    fragColor = g.color;
    fragOpacity = g.opacity;

    // Output final vertex position
    gl_Position = clipPos;  

    gl_PointSize = 3.0; 
}
