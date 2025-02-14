#include "vulkan_setup.h"
#include "gaussian_pipeline.h"
#include "file_loader.h"
#include <iostream>

int main() {
    VulkanSetup vulkan;
    vulkan.initVulkan();

    std::vector<Gaussian> gaussians = FileLoader::loadGaussianData("../assets/sorted_culled_gaussians.bin");
    CameraBuffer cameraData = FileLoader::loadCameraData("../assets/camera.bin");

    GaussianPipeline pipeline(vulkan);
    pipeline.createCameraBuffer(cameraData); 
    pipeline.createGaussianBuffer(gaussians); 
    pipeline.createPipeline();

    std::cout << "Rendering frame..." << std::endl;

    while (!glfwWindowShouldClose(vulkan.getWindow())) {
        glfwPollEvents();  
        pipeline.renderFrame();  
    }

    vulkan.cleanup();
    std::cout << "done" << std::endl;
    
    return 0;
}
