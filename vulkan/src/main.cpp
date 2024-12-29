#include "vulkan_setup.hpp"
#include "utils.hpp"
#include <cstring>
#include <iostream>
#include <array>
#include <vector>

#include <fstream>
#include <sstream>
#include <cstddef> 

struct Gaussian {
    float x, y;                 // Point position
    float r, g, b;              // RGB colors
    float ic11, ic12, ic21, ic22; // Inverse covariance matrix
    float opacity;              // Opacity
    float min_x, max_x, min_y, max_y; // Bounding ranges
};

std::vector<std::vector<float>> readCSV(const std::string& filename) {
    std::vector<std::vector<float>> data;
    std::ifstream file(filename);

    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " + filename);
    }

    std::string line;
    while (std::getline(file, line)) {
        std::vector<float> row;
        std::stringstream lineStream(line);
        std::string cell;

        while (std::getline(lineStream, cell, ',')) {
            row.push_back(std::stof(cell)); // Convert each value to float
        }

        data.push_back(row);
    }

    file.close();
    return data;
}

std::vector<Gaussian> loadGaussianCSV(const std::string& filename) {
    std::vector<Gaussian> gaussians;
    auto data = readCSV(filename); // Use the generic CSV reader

    for (const auto& row : data) {
        if (row.size() != 14) { // Ensure correct number of columns
            std::cerr << "Invalid row size: " << row.size() << "\n";
            continue;
        }

        // Map row to Gaussian structure
        gaussians.push_back(Gaussian{
            row[0], row[1],  // x, y
            row[2], row[3], row[4], // r, g, b
            row[5], row[6], row[7], row[8], // ic11, ic12, ic21, ic22
            row[9],          // opacity
            row[10], row[11], row[12], row[13] // min_x, max_x, min_y, max_y
        });
    }

    return gaussians;
}

void checkCPUMemoryAlignment() {
    std::cout << "Offsets in C++ Gaussian struct:\n";
    std::cout << "x: " << offsetof(Gaussian, x) << "\n";
    std::cout << "y: " << offsetof(Gaussian, y) << "\n";
    std::cout << "r: " << offsetof(Gaussian, r) << "\n";
    std::cout << "g: " << offsetof(Gaussian, g) << "\n";
    std::cout << "b: " << offsetof(Gaussian, b) << "\n";
    std::cout << "ic11: " << offsetof(Gaussian, ic11) << "\n";
    std::cout << "opacity: " << offsetof(Gaussian, opacity) << "\n";
    std::cout << "min_x: " << offsetof(Gaussian, min_x) << "\n";

    std::cout << "Total size of struct: " << sizeof(Gaussian) << " bytes\n";
}

int main() {
    try {
        checkCPUMemoryAlignment();

        VulkanSetup vulkan;

        // Data setup
        std::vector<Gaussian> gaussians = loadGaussianCSV("../processed_scene.csv");

        for (size_t i = 0; i < 5 && i < gaussians.size(); ++i) {
            const auto& g = gaussians[i];
            std::cout << "Gaussian " << i << ": "
                    << "x=" << g.x << ", y=" << g.y
                    << ", r=" << g.r << ", g=" << g.g << ", b=" << g.b
                    << ", ic11=" << g.ic11 << ", ic12=" << g.ic12
                    << ", ic21=" << g.ic21 << ", ic22=" << g.ic22
                    << ", opacity=" << g.opacity
                    << ", min_x=" << g.min_x << ", max_x=" << g.max_x
                    << ", min_y=" << g.min_y << ", max_y=" << g.max_y
                    << std::endl;
        }

        VkDeviceSize gaussianBufferSize = sizeof(Gaussian) * gaussians.size();

        // Create a buffer for Gaussian data
        VkBuffer gaussianBuffer;
        VkDeviceMemory gaussianBufferMemory;
        vulkan.createBuffer(gaussianBufferSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            gaussianBuffer, gaussianBufferMemory);

        // Copy data to input buffer
        void* data;
        vkMapMemory(vulkan.device, gaussianBufferMemory, 0, gaussianBufferSize, 0, &data);
        std::memcpy(data, gaussians.data(), gaussianBufferSize);
        vkUnmapMemory(vulkan.device, gaussianBufferMemory);

        std::cout << "Input buffers created and data uploaded." << std::endl;

        // read back and verify uploaded data
        void* verifyData;
        vkMapMemory(vulkan.device, gaussianBufferMemory, 0, gaussianBufferSize, 0, &verifyData);
        std::vector<Gaussian> uploadedData(gaussians.size());
        std::memcpy(uploadedData.data(), verifyData, gaussianBufferSize);
        vkUnmapMemory(vulkan.device, gaussianBufferMemory);

        for (size_t i = 0; i < 5; ++i) {
            const auto& g = uploadedData[i];
            std::cout << "Uploaded Gaussian " << i << ": "
                    << "x=" << g.x << ", y=" << g.y
                    << ", r=" << g.r << ", g=" << g.g << ", b=" << g.b
                    << ", ic11=" << g.ic11 << ", ic12=" << g.ic12
                    << ", ic21=" << g.ic21 << ", ic22=" << g.ic22
                    << ", opacity=" << g.opacity
                    << ", min_x=" << g.min_x << ", max_x=" << g.max_x
                    << ", min_y=" << g.min_y << ", max_y=" << g.max_y
                    << std::endl;
        }

        VkDeviceSize debugBufferSize = sizeof(Gaussian) * gaussians.size(); // Same size as input

        VkBuffer debugBuffer;
        VkDeviceMemory debugBufferMemory;
        vulkan.createBuffer(debugBufferSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            debugBuffer, debugBufferMemory);

        std::cout << "Debug buffer created." << std::endl;

        // load compute shader 
        std::vector<char> computeShaderCode = readFile("../shaders/compute_shader.spv");
        VkShaderModule computeShaderModule = vulkan.createShaderModule(computeShaderCode);
        std::cout << "Shader module created." << std::endl;

        // Descriptor set layout
        VkDescriptorSetLayoutBinding gaussianBinding = {};
        gaussianBinding.binding = 0;
        gaussianBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        gaussianBinding.descriptorCount = 1;
        gaussianBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

        VkDescriptorSetLayoutBinding debugBinding = gaussianBinding;
        debugBinding.binding = 1;

        std::array<VkDescriptorSetLayoutBinding, 2> bindings = {gaussianBinding, debugBinding};

        VkDescriptorSetLayoutCreateInfo layoutInfo = {};
        layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layoutInfo.bindingCount = bindings.size();
        layoutInfo.pBindings = bindings.data();

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
        VkDescriptorBufferInfo gaussianBufferInfo = {};
        gaussianBufferInfo.buffer = gaussianBuffer;
        gaussianBufferInfo.offset = 0;
        gaussianBufferInfo.range = gaussianBufferSize;

        VkDescriptorBufferInfo debugBufferInfo = {};
        debugBufferInfo.buffer = debugBuffer;
        debugBufferInfo.offset = 0;
        debugBufferInfo.range = debugBufferSize;

        std::array<VkWriteDescriptorSet, 2> descriptorWrites = {};

        descriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWrites[0].dstSet = descriptorSet;
        descriptorWrites[0].dstBinding = 0;
        descriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        descriptorWrites[0].descriptorCount = 1;
        descriptorWrites[0].pBufferInfo = &gaussianBufferInfo;

        descriptorWrites[1] = descriptorWrites[0];
        descriptorWrites[1].dstBinding = 1;
        descriptorWrites[1].pBufferInfo = &debugBufferInfo;

        vkUpdateDescriptorSets(vulkan.device, descriptorWrites.size(), descriptorWrites.data(), 0, nullptr);

        std::cout << "Descriptor sets updated." << std::endl;

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

        // Dispatch the compute shader (1 thread per Gaussian)
        vkCmdDispatch(commandBuffer, (uint32_t)gaussians.size(), 1, 1);

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
        
        void* debugData;
        vkMapMemory(vulkan.device, debugBufferMemory, 0, debugBufferSize, 0, &debugData);
        std::vector<Gaussian> debugOutput(gaussians.size());
        std::memcpy(debugOutput.data(), debugData, debugBufferSize);
        vkUnmapMemory(vulkan.device, debugBufferMemory);

        // Print the first few debug outputs
        for (size_t i = 0; i < std::min(debugOutput.size(), size_t(5)); ++i) {
            const auto& g = debugOutput[i];
            std::cout << "Debug Gaussian " << i << ": "
                    << "x=" << g.x << ", y=" << g.y
                    << ", r=" << g.r << ", g=" << g.g << ", b=" << g.b
                    << ", ic11=" << g.ic11 << ", ic12=" << g.ic12
                    << ", ic21=" << g.ic21 << ", ic22=" << g.ic22
                    << ", opacity=" << g.opacity
                    << ", min_x=" << g.min_x << ", max_x=" << g.max_x
                    << ", min_y=" << g.min_y << ", max_y=" << g.max_y
                    << std::endl;
        }


    } catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
