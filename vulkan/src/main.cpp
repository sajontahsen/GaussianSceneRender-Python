#include "vulkan_setup.hpp"
#include "utils.hpp"
#include <cstring>
#include <iostream>
#include <array>
#include <vector>

int main() {
    try {
        VulkanSetup vulkan;

        // Data setup
        std::vector<float> inputData = {1.0f, 2.0f, 3.0f, 4.0f};
        std::vector<float> outputData(inputData.size(), 0.0f);
        VkDeviceSize bufferSize = sizeof(float) * inputData.size();

        // Create buffers
        VkBuffer inputBuffer, outputBuffer;
        VkDeviceMemory inputBufferMemory, outputBufferMemory;

        vulkan.createBuffer(bufferSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, 
                            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, 
                            inputBuffer, inputBufferMemory);
        vulkan.createBuffer(bufferSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, 
                            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, 
                            outputBuffer, outputBufferMemory);

        // Copy data to input buffer
        void* data;
        vkMapMemory(vulkan.device, inputBufferMemory, 0, bufferSize, 0, &data);
        memcpy(data, inputData.data(), bufferSize);
        vkUnmapMemory(vulkan.device, inputBufferMemory);

        std::cout << "Buffers created and data uploaded." << std::endl;

        std::vector<char> computeShaderCode = readFile("../shaders/compute_shader.spv");
        VkShaderModule computeShaderModule = vulkan.createShaderModule(computeShaderCode);
        std::cout << "Shader module created." << std::endl;

        // Descriptor set layout
        VkDescriptorSetLayoutBinding bindings[2] = {};
        bindings[0].binding = 0;
        bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[0].descriptorCount = 1;
        bindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

        bindings[1] = bindings[0];
        bindings[1].binding = 1;

        VkDescriptorSetLayoutCreateInfo layoutInfo = {};
        layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layoutInfo.bindingCount = 2;
        layoutInfo.pBindings = bindings;

        VkDescriptorSetLayout descriptorSetLayout;
        vkCreateDescriptorSetLayout(vulkan.device, &layoutInfo, nullptr, &descriptorSetLayout);

        // Pipeline layout
        VkPipelineLayoutCreateInfo pipelineLayoutInfo = {};
        pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipelineLayoutInfo.setLayoutCount = 1;
        pipelineLayoutInfo.pSetLayouts = &descriptorSetLayout;

        VkPipelineLayout pipelineLayout;
        vkCreatePipelineLayout(vulkan.device, &pipelineLayoutInfo, nullptr, &pipelineLayout);

        // Compute pipeline creation
        VkComputePipelineCreateInfo pipelineInfo = {};
        pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        pipelineInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        pipelineInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        pipelineInfo.stage.module = computeShaderModule;
        pipelineInfo.stage.pName = "main";
        pipelineInfo.layout = pipelineLayout;

        VkPipeline computePipeline;
        if (vkCreateComputePipelines(vulkan.device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &computePipeline) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create compute pipeline!");
        }

        std::cout << "Compute pipeline created successfully." << std::endl;

        // Descriptor pool
        VkDescriptorPoolSize poolSize = {};
        poolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        poolSize.descriptorCount = 2;

        VkDescriptorPoolCreateInfo poolInfo = {};
        poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        poolInfo.poolSizeCount = 1;
        poolInfo.pPoolSizes = &poolSize;
        poolInfo.maxSets = 1;

        VkDescriptorPool descriptorPool;
        if (vkCreateDescriptorPool(vulkan.device, &poolInfo, nullptr, &descriptorPool) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create descriptor pool!");
        }

        // Allocate descriptor set
        VkDescriptorSetAllocateInfo allocInfo = {};
        allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocInfo.descriptorPool = descriptorPool;
        allocInfo.descriptorSetCount = 1;
        allocInfo.pSetLayouts = &descriptorSetLayout;

        VkDescriptorSet descriptorSet;
        if (vkAllocateDescriptorSets(vulkan.device, &allocInfo, &descriptorSet) != VK_SUCCESS) {
            throw std::runtime_error("Failed to allocate descriptor set!");
        }

        std::cout << "Descriptor set allocated successfully." << std::endl;

        // Descriptor buffer bindings
        VkDescriptorBufferInfo inputBufferInfo = {};
        inputBufferInfo.buffer = inputBuffer;
        inputBufferInfo.offset = 0;
        inputBufferInfo.range = bufferSize;

        VkDescriptorBufferInfo outputBufferInfo = {};
        outputBufferInfo.buffer = outputBuffer;
        outputBufferInfo.offset = 0;
        outputBufferInfo.range = bufferSize;

        std::array<VkWriteDescriptorSet, 2> descriptorWrites = {};

        descriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWrites[0].dstSet = descriptorSet;
        descriptorWrites[0].dstBinding = 0;
        descriptorWrites[0].dstArrayElement = 0;
        descriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        descriptorWrites[0].descriptorCount = 1;
        descriptorWrites[0].pBufferInfo = &inputBufferInfo;

        descriptorWrites[1] = descriptorWrites[0];
        descriptorWrites[1].dstBinding = 1;
        descriptorWrites[1].pBufferInfo = &outputBufferInfo;

        vkUpdateDescriptorSets(vulkan.device, descriptorWrites.size(), descriptorWrites.data(), 0, nullptr);

        std::cout << "Descriptor sets updated successfully." << std::endl;

        if (vulkan.commandPool == VK_NULL_HANDLE) {
            throw std::runtime_error("Command pool is not initialized!");
        }

        if (vulkan.device == VK_NULL_HANDLE) {
            throw std::runtime_error("Logical device is not initialized!");
        }

        std::cout << "Command pool and logical device verified." << std::endl;


        // Allocate command buffer
        VkCommandBufferAllocateInfo allocInfoCmd = {};
        allocInfoCmd.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocInfoCmd.commandPool = vulkan.commandPool;
        allocInfoCmd.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocInfoCmd.commandBufferCount = 1;

        VkCommandBuffer commandBuffer;
        if (vkAllocateCommandBuffers(vulkan.device, &allocInfoCmd, &commandBuffer) != VK_SUCCESS) {
            throw std::runtime_error("Failed to allocate command buffer!");
        }

        // Record commands
        VkCommandBufferBeginInfo beginInfo = {};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

        vkBeginCommandBuffer(commandBuffer, &beginInfo);

        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, computePipeline);
        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1, &descriptorSet, 0, nullptr);

        // Dispatch compute shader
        vkCmdDispatch(commandBuffer, (uint32_t)inputData.size() / 16 + 1, 1, 1);

        vkEndCommandBuffer(commandBuffer);

        std::cout << "Commands recorded successfully." << std::endl;

        VkSubmitInfo submitInfo = {};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &commandBuffer;

        if (vkQueueSubmit(vulkan.computeQueue, 1, &submitInfo, VK_NULL_HANDLE) != VK_SUCCESS) {
            throw std::runtime_error("Failed to submit compute command buffer!");
        }

        vkQueueWaitIdle(vulkan.computeQueue);
        std::cout << "Compute shader executed successfully." << std::endl;

        void* mappedData;
        vkMapMemory(vulkan.device, outputBufferMemory, 0, bufferSize, 0, &mappedData);
        std::memcpy(outputData.data(), mappedData, bufferSize);
        vkUnmapMemory(vulkan.device, outputBufferMemory);

        std::cout << "Output data: ";
        for (float val : outputData) {
            std::cout << val << " ";
        }
        std::cout << std::endl;


    } catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
