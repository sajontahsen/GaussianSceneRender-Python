#version 450

layout(local_size_x = 16) in;

layout(binding = 0) buffer InputBuffer {
    float inputData[];
};

layout(binding = 1) buffer OutputBuffer {
    float outputData[];
};

void main() {
    uint idx = gl_GlobalInvocationID.x;
    outputData[idx] = inputData[idx] * inputData[idx];
}
