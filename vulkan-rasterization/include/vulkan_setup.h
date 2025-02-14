#pragma once
#include <vulkan/vulkan.h>
#include <GLFW/glfw3.h> 
#include <vector>

class VulkanSetup {
public:
    void initVulkan();
    void cleanup();

    VkDevice getDevice() { return device; }
    VkQueue getGraphicsQueue() { return graphicsQueue; }
    VkCommandPool getCommandPool() { return commandPool; }
    VkImageView getSwapchainImageView() { return swapchainImageView; }
    VkSurfaceKHR getSurface() { return surface; } 
    GLFWwindow* getWindow() { return window; } 
    VkSwapchainKHR getSwapchain() { return swapchain; }  
    VkDescriptorPool getDescriptorPool() { return descriptorPool; } 

    uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties);

private:
    VkInstance instance;
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    VkDevice device;

    VkQueue graphicsQueue;
    VkCommandPool commandPool;
    VkSwapchainKHR swapchain;       
    VkSurfaceKHR surface;          
    VkImageView swapchainImageView; 
    GLFWwindow* window; 
    VkDescriptorPool descriptorPool; 
    
    uint32_t findQueueFamily(VkQueueFlagBits queueFlags);
    
    void createInstance();
    void createSurface();   
    void createWindow();  
    void pickPhysicalDevice();
    void createLogicalDevice();
    void createCommandPool();  
    void createSwapchain();
    bool isDeviceSuitable(VkPhysicalDevice device);
    void createDescriptorPool();
};
