#pragma once

#include <vulkan/vulkan.h>
#include <vector>
#include <string>

class VulkanSetup {
public:
    VulkanSetup();
    ~VulkanSetup();

    void createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, 
                      VkBuffer &buffer, VkDeviceMemory &bufferMemory);
    void copyBuffer(VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size);

    void createCommandPool();

    VkDevice device;
    VkQueue computeQueue;
    VkCommandPool commandPool;

    VkShaderModule createShaderModule(const std::vector<char>& code);


private:
    VkInstance instance;
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    VkDeviceMemory allocateMemory(VkBuffer buffer, VkMemoryPropertyFlags properties);
    uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties);

    void createInstance();
    void pickPhysicalDevice();
    void createLogicalDevice();
    
    uint32_t findQueueFamily(VkQueueFlagBits queueFlags);

};
