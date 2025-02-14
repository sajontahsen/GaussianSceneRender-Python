#pragma once
#include "vulkan_setup.h"
#include <vector>
#include <vulkan/vulkan.h>
#include <string>
#include <file_loader.h>

class GaussianPipeline {
public:
    GaussianPipeline(VulkanSetup &vkSetup);
    void createPipeline();
    void createRenderPass();  
    void createGaussianBuffer(const std::vector<Gaussian> &gaussians);
    void renderFrame();  
    void createCameraBuffer(CameraBuffer &cameraData);

    void cleanup();

    const int width = 1200;
    const int height = 800;
    const float scaleFactor = 1.00f;

private:
    VulkanSetup &vulkan;
    VkPipeline graphicsPipeline;
    VkPipelineLayout pipelineLayout;
    VkRenderPass renderPass;  
    VkBuffer gaussianBuffer;
    VkDeviceMemory gaussianBufferMemory;
    
    VkFramebuffer framebuffer;
    VkCommandBuffer commandBuffer;
    VkSemaphore imageAvailableSemaphore, renderFinishedSemaphore;
    VkFence renderFence;
    VkDescriptorSetLayout descriptorSetLayout; 
    VkDescriptorSet descriptorSet; 
    VkDescriptorSet cameraDescriptorSet;       
    VkDescriptorSetLayout cameraDescriptorSetLayout;  

    VkImage depthImage;               
    VkDeviceMemory depthImageMemory;  
    VkImageView depthImageView;       

    VkBuffer cameraBuffer;              
    VkDeviceMemory cameraBufferMemory;  
    
    uint32_t gaussianCount = 0; 
    
    void createFramebuffer();
    void createCommandBuffer(); 
    void createDepthResources();

    std::vector<char> readShaderFile(const std::string &filename);
    VkShaderModule createShaderModule(const std::vector<char> &code);

};
