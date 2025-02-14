#include "gaussian_pipeline.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <cstring>
#include <array>
#include <iomanip>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

GaussianPipeline::GaussianPipeline(VulkanSetup &vkSetup) : vulkan(vkSetup) {}

std::vector<char> GaussianPipeline::readShaderFile(const std::string &filename) {
    std::ifstream file(filename, std::ios::ate | std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open shader file: " + filename);
    }

    size_t fileSize = (size_t)file.tellg();
    std::vector<char> buffer(fileSize);
    file.seekg(0);
    file.read(buffer.data(), fileSize);
    file.close();
    return buffer;
}

VkShaderModule GaussianPipeline::createShaderModule(const std::vector<char> &code) {
    VkShaderModuleCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    createInfo.codeSize = code.size();
    createInfo.pCode = reinterpret_cast<const uint32_t *>(code.data());

    VkShaderModule shaderModule;
    if (vkCreateShaderModule(vulkan.getDevice(), &createInfo, nullptr, &shaderModule) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create shader module!");
    }

    return shaderModule;
}

void GaussianPipeline::createPipeline() {
    std::cout << "Creating render pass..." << std::endl;
    createRenderPass();  
    std::cout << "Creating frame buffer..." << std::endl;
    createFramebuffer();   
    std::cout << "Creating command buffer..." << std::endl;
    createCommandBuffer(); 

    VkSemaphoreCreateInfo semaphoreInfo{};
    semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

    if (vkCreateSemaphore(vulkan.getDevice(), &semaphoreInfo, nullptr, &imageAvailableSemaphore) != VK_SUCCESS ||
        vkCreateSemaphore(vulkan.getDevice(), &semaphoreInfo, nullptr, &renderFinishedSemaphore) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create semaphores!");
    }

    std::cout << "reading shader code..." << std::endl;
    auto vertShaderCode = readShaderFile("../shaders/gaussian.vert.spv");
    auto fragShaderCode = readShaderFile("../shaders/gaussian.frag.spv");

    VkShaderModule vertShaderModule = createShaderModule(vertShaderCode);
    VkShaderModule fragShaderModule = createShaderModule(fragShaderCode);

    // Declare binding and attribute descriptions
    VkVertexInputBindingDescription bindingDescription{};
    bindingDescription.binding = 0;
    bindingDescription.stride = sizeof(Gaussian);  
    bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX; // One per vertex

    std::array<VkVertexInputAttributeDescription, 5> attributeDescriptions{};

    // Position (vec3)
    attributeDescriptions[0].binding = 0;
    attributeDescriptions[0].location = 0;  
    attributeDescriptions[0].format = VK_FORMAT_R32G32B32_SFLOAT;
    attributeDescriptions[0].offset = offsetof(Gaussian, position);

    // Color (vec3)
    attributeDescriptions[1].binding = 0;
    attributeDescriptions[1].location = 1;  
    attributeDescriptions[1].format = VK_FORMAT_R32G32B32_SFLOAT;
    attributeDescriptions[1].offset = offsetof(Gaussian, color);

    // Covariance Matrix (mat3x3) (3 rows, to be stored as 3 vec3 attributes)
    attributeDescriptions[2].binding = 0;
    attributeDescriptions[2].location = 2;
    attributeDescriptions[2].format = VK_FORMAT_R32G32B32_SFLOAT;
    attributeDescriptions[2].offset = offsetof(Gaussian, covariance);  // First row

    attributeDescriptions[3].binding = 0;
    attributeDescriptions[3].location = 3;
    attributeDescriptions[3].format = VK_FORMAT_R32G32B32_SFLOAT;
    attributeDescriptions[3].offset = offsetof(Gaussian, covariance) + sizeof(glm::vec3);  // Second row

    attributeDescriptions[4].binding = 0;
    attributeDescriptions[4].location = 4;
    attributeDescriptions[4].format = VK_FORMAT_R32G32B32_SFLOAT;
    attributeDescriptions[4].offset = offsetof(Gaussian, covariance) + 2 * sizeof(glm::vec3);  // Third row

    VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
    vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vertexInputInfo.vertexBindingDescriptionCount = 1;
    vertexInputInfo.pVertexBindingDescriptions = &bindingDescription;
    vertexInputInfo.vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescriptions.size());
    vertexInputInfo.pVertexAttributeDescriptions = attributeDescriptions.data();

    VkPipelineShaderStageCreateInfo vertShaderStageInfo{};
    vertShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    vertShaderStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
    vertShaderStageInfo.module = vertShaderModule;
    vertShaderStageInfo.pName = "main";

    VkPipelineShaderStageCreateInfo fragShaderStageInfo{};
    fragShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    fragShaderStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    fragShaderStageInfo.module = fragShaderModule;
    fragShaderStageInfo.pName = "main";

    VkPipelineShaderStageCreateInfo shaderStages[] = {vertShaderStageInfo, fragShaderStageInfo};

    VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
    inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_POINT_LIST;
    inputAssembly.primitiveRestartEnable = VK_FALSE;

    VkViewport viewport{};
    viewport.x = 0.0f;
    viewport.y = 0.0f;
    viewport.width = (float)width * scaleFactor;
    viewport.height = (float)height * scaleFactor;
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;

    VkRect2D scissor{};
    scissor.offset = {0, 0};
    scissor.extent = {static_cast<uint32_t>(width * scaleFactor), static_cast<uint32_t>(height * scaleFactor)};

    VkPipelineViewportStateCreateInfo viewportState{};
    viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewportState.viewportCount = 1;
    viewportState.pViewports = &viewport;
    viewportState.scissorCount = 1;
    viewportState.pScissors = &scissor;

    VkPipelineRasterizationStateCreateInfo rasterizer{};
    rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rasterizer.depthClampEnable = VK_FALSE;
    rasterizer.rasterizerDiscardEnable = VK_FALSE;
    rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
    rasterizer.lineWidth = 1.0f;
    rasterizer.cullMode = VK_CULL_MODE_BACK_BIT;
    rasterizer.frontFace = VK_FRONT_FACE_CLOCKWISE;
    rasterizer.depthBiasEnable = VK_FALSE;

    VkPipelineMultisampleStateCreateInfo multisampling{};
    multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisampling.sampleShadingEnable = VK_FALSE;
    multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    // Enable Depth Testing
    VkPipelineDepthStencilStateCreateInfo depthStencil{};
    depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    depthStencil.depthTestEnable = VK_TRUE;  
    depthStencil.depthWriteEnable = VK_FALSE; 
    depthStencil.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL; // Closer objects appear in front
    depthStencil.depthBoundsTestEnable = VK_FALSE;
    depthStencil.stencilTestEnable = VK_FALSE;

    VkPipelineColorBlendAttachmentState colorBlendAttachment{};
    colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    colorBlendAttachment.blendEnable = VK_TRUE;
    colorBlendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
    colorBlendAttachment.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    colorBlendAttachment.colorBlendOp = VK_BLEND_OP_ADD;
    colorBlendAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
    colorBlendAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
    colorBlendAttachment.alphaBlendOp = VK_BLEND_OP_ADD;

    VkPipelineColorBlendStateCreateInfo colorBlending{};
    colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    colorBlending.logicOpEnable = VK_FALSE;
    colorBlending.attachmentCount = 1;
    colorBlending.pAttachments = &colorBlendAttachment;

    // std::cout << "done upto blending and depth test..." << std::endl;

    // Gaussian Buffer (Binding 0)
    VkDescriptorSetLayoutBinding gaussiansBinding{};
    gaussiansBinding.binding = 0;
    gaussiansBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    gaussiansBinding.descriptorCount = 1;
    gaussiansBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

    // Camera Buffer (Binding 1)
    VkDescriptorSetLayoutBinding cameraBinding{};
    cameraBinding.binding = 1;
    cameraBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    cameraBinding.descriptorCount = 1;
    cameraBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

    // Combine Both Bindings into One Layout
    std::array<VkDescriptorSetLayoutBinding, 2> bindings = {gaussiansBinding, cameraBinding};

    VkDescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
    layoutInfo.pBindings = bindings.data();

    if (vkCreateDescriptorSetLayout(vulkan.getDevice(), &layoutInfo, nullptr, &descriptorSetLayout) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create descriptor set layout!");
    }

    // CREATE PIPELINE LAYOUT AFTER DESCRIPTOR SET LAYOUT
    VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount = 1;  // Using one descriptor set layout
    pipelineLayoutInfo.pSetLayouts = &descriptorSetLayout;

    if (vkCreatePipelineLayout(vulkan.getDevice(), &pipelineLayoutInfo, nullptr, &pipelineLayout) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create pipeline layout!");
    }

    // Allocate Descriptor Set for Both Buffers
    VkDescriptorSetAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = vulkan.getDescriptorPool();
    allocInfo.descriptorSetCount = 1; // Allocate only ONE descriptor set for both buffers
    allocInfo.pSetLayouts = &descriptorSetLayout;

    if (vkAllocateDescriptorSets(vulkan.getDevice(), &allocInfo, &descriptorSet) != VK_SUCCESS) {
        throw std::runtime_error("Failed to allocate descriptor set!");
    }

    // Update Gaussian Buffer Binding (Binding 0)
    VkDescriptorBufferInfo gaussianBufferInfo{};
    gaussianBufferInfo.buffer = gaussianBuffer;
    gaussianBufferInfo.offset = 0;
    gaussianBufferInfo.range = VK_WHOLE_SIZE;

    VkWriteDescriptorSet descriptorWrite{};
    descriptorWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descriptorWrite.dstSet = descriptorSet;
    descriptorWrite.dstBinding = 0;  // Gaussian is at Binding 0
    descriptorWrite.dstArrayElement = 0;
    descriptorWrite.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    descriptorWrite.descriptorCount = 1;
    descriptorWrite.pBufferInfo = &gaussianBufferInfo;

    // Update Camera Buffer Binding (Binding 1)
    VkDescriptorBufferInfo cameraBufferInfo{};
    cameraBufferInfo.buffer = cameraBuffer;
    cameraBufferInfo.offset = 0;
    cameraBufferInfo.range = sizeof(CameraBuffer);

    VkWriteDescriptorSet cameraDescriptorWrite{};
    cameraDescriptorWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    cameraDescriptorWrite.dstSet = descriptorSet;
    cameraDescriptorWrite.dstBinding = 1;  // Camera is at Binding 1
    cameraDescriptorWrite.dstArrayElement = 0;
    cameraDescriptorWrite.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    cameraDescriptorWrite.descriptorCount = 1;
    cameraDescriptorWrite.pBufferInfo = &cameraBufferInfo;

    // Update Descriptor Sets 
    std::array<VkWriteDescriptorSet, 2> descriptorWrites = {descriptorWrite, cameraDescriptorWrite};
    vkUpdateDescriptorSets(vulkan.getDevice(), static_cast<uint32_t>(descriptorWrites.size()), descriptorWrites.data(), 0, nullptr);

    VkGraphicsPipelineCreateInfo pipelineInfo{};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipelineInfo.stageCount = 2;
    pipelineInfo.pStages = shaderStages;
    pipelineInfo.pVertexInputState = &vertexInputInfo;
    pipelineInfo.pInputAssemblyState = &inputAssembly;
    pipelineInfo.pViewportState = &viewportState;
    pipelineInfo.pRasterizationState = &rasterizer;
    pipelineInfo.pMultisampleState = &multisampling;
    pipelineInfo.pColorBlendState = &colorBlending;
    pipelineInfo.pDepthStencilState = &depthStencil;

    if (pipelineLayout == VK_NULL_HANDLE) {
        throw std::runtime_error("Pipeline layout is NULL!");
    }
    pipelineInfo.layout = pipelineLayout;
    pipelineInfo.renderPass = renderPass;  

    std::cout << "Shader Stages: " << shaderStages[0].pName << ", " << shaderStages[1].pName << std::endl;
    std::cout << "Vertex Input State: " << vertexInputInfo.sType << std::endl;
    std::cout << "Input Assembly Topology: " << inputAssembly.topology << std::endl;
    std::cout << "Viewport: " << viewport.width << "x" << viewport.height << std::endl;
    std::cout << "Rasterization Mode: " << rasterizer.polygonMode << std::endl;
    std::cout << "Blending Enabled: " << colorBlendAttachment.blendEnable << std::endl;
    std::cout << "Pipeline Layout: " << pipelineLayout << std::endl;

    // std::cout << "done until graphicspipeline..." << std::endl;

    if (vkCreateGraphicsPipelines(vulkan.getDevice(), VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &graphicsPipeline) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create graphics pipeline!");
    }

    // std::cout << "done until pipeline..." << std::endl;
    
    vkDestroyShaderModule(vulkan.getDevice(), fragShaderModule, nullptr);
    vkDestroyShaderModule(vulkan.getDevice(), vertShaderModule, nullptr);
}

void GaussianPipeline::cleanup() {
    vkDestroyPipeline(vulkan.getDevice(), graphicsPipeline, nullptr);
    vkDestroyPipelineLayout(vulkan.getDevice(), pipelineLayout, nullptr);
}

void GaussianPipeline::createRenderPass() {
    VkAttachmentDescription colorAttachment{};
    colorAttachment.format = VK_FORMAT_B8G8R8A8_UNORM;  // Color buffer format
    colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
    colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    colorAttachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

    VkAttachmentReference colorAttachmentRef{};
    colorAttachmentRef.attachment = 0;
    colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    VkAttachmentDescription depthAttachment{};
    depthAttachment.format = VK_FORMAT_D32_SFLOAT;  // Depth format
    depthAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
    depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    depthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    depthAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    depthAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    depthAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    depthAttachment.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    // Depth attachment reference
    VkAttachmentReference depthAttachmentRef{};
    depthAttachmentRef.attachment = 1;  // Second attachment (after color)
    depthAttachmentRef.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    VkSubpassDescription subpass{};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments = &colorAttachmentRef;
    subpass.pDepthStencilAttachment = &depthAttachmentRef;  // Attach depth buffer

    std::array<VkAttachmentDescription, 2> attachments = {colorAttachment, depthAttachment};
    VkRenderPassCreateInfo renderPassInfo{};
    renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    renderPassInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
    renderPassInfo.pAttachments = attachments.data();
    renderPassInfo.subpassCount = 1;
    renderPassInfo.pSubpasses = &subpass;

    if (vkCreateRenderPass(vulkan.getDevice(), &renderPassInfo, nullptr, &renderPass) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create render pass!");
    }
}

void GaussianPipeline::createGaussianBuffer(const std::vector<Gaussian> &gaussians) {
    gaussianCount = static_cast<uint32_t>(gaussians.size());  // Store count

    VkDeviceSize bufferSize = sizeof(Gaussian) * gaussians.size();

    // Create Buffer
    VkBufferCreateInfo bufferInfo{};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = bufferSize;
    bufferInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    if (vkCreateBuffer(vulkan.getDevice(), &bufferInfo, nullptr, &gaussianBuffer) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create Gaussian buffer!");
    }

    VkMemoryRequirements memRequirements;
    vkGetBufferMemoryRequirements(vulkan.getDevice(), gaussianBuffer, &memRequirements);

    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memRequirements.size;
    allocInfo.memoryTypeIndex = vulkan.findMemoryType(memRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    if (vkAllocateMemory(vulkan.getDevice(), &allocInfo, nullptr, &gaussianBufferMemory) != VK_SUCCESS) {
        throw std::runtime_error("Failed to allocate Gaussian buffer memory!");
    }

    vkBindBufferMemory(vulkan.getDevice(), gaussianBuffer, gaussianBufferMemory, 0);

    // Upload Gaussian data
    void* data;
    vkMapMemory(vulkan.getDevice(), gaussianBufferMemory, 0, bufferSize, 0, &data);
    memcpy(data, gaussians.data(), (size_t)bufferSize);
    vkUnmapMemory(vulkan.getDevice(), gaussianBufferMemory);

}


void GaussianPipeline::createFramebuffer() {
    // std::cout << "into frame buffer..." << std::endl;

    // Ensure depth buffer exists before using
    createDepthResources();

    VkImageView swapchainImageView = vulkan.getSwapchainImageView();
    if (swapchainImageView == VK_NULL_HANDLE) {
        throw std::runtime_error("Swapchain image view is null!");
    }

    // Include depth buffer in framebuffer attachments
    std::array<VkImageView, 2> attachments = {swapchainImageView, depthImageView};

    VkFramebufferCreateInfo framebufferInfo{};
    framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
    framebufferInfo.renderPass = renderPass;
    framebufferInfo.attachmentCount = 2;  // Now has color + depth attachments
    framebufferInfo.pAttachments = attachments.data();
    framebufferInfo.width = (float)width * scaleFactor;
    framebufferInfo.height = (float)height * scaleFactor;
    framebufferInfo.layers = 1;

    // std::cout << "into into frame buffer..." << std::endl;

    if (vkCreateFramebuffer(vulkan.getDevice(), &framebufferInfo, nullptr, &framebuffer) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create framebuffer!");
    }

    // std::cout << "done Creating frame buffer..." << std::endl;
}


void GaussianPipeline::createCommandBuffer() {
    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.commandPool = vulkan.getCommandPool();

    if (vulkan.getCommandPool() == VK_NULL_HANDLE) {
        throw std::runtime_error("Command pool is null!");
    }

    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = 1;

    // std::cout << "start Creating command buffer..." << std::endl;
    if (vkAllocateCommandBuffers(vulkan.getDevice(), &allocInfo, &commandBuffer) != VK_SUCCESS) {
        throw std::runtime_error("Failed to allocate command buffer!");
    }

    VkFenceCreateInfo fenceInfo{};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;  // Start in signaled state

    if (vkCreateFence(vulkan.getDevice(), &fenceInfo, nullptr, &renderFence) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create render fence!");
    }

    // std::cout << "after allocate command buffer and create fence..." << std::endl;
}

void GaussianPipeline::renderFrame() {
    std::cout << "Rendering a frame..." << std::endl;

    // Wait for previous frame to finish
    vkWaitForFences(vulkan.getDevice(), 1, &renderFence, VK_TRUE, UINT64_MAX);
    vkResetFences(vulkan.getDevice(), 1, &renderFence);

    if (commandBuffer == VK_NULL_HANDLE) {
        throw std::runtime_error("Command buffer is null!");
    }

    // Reset command buffer before recording a new frame
    vkResetCommandBuffer(commandBuffer, 0);

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

    if (vkBeginCommandBuffer(commandBuffer, &beginInfo) != VK_SUCCESS) {
        throw std::runtime_error("Failed to begin recording command buffer!");
    }

    // std::cout << "Acquiring frame..." << std::endl;

    uint32_t imageIndex;
    VkSwapchainKHR swapchain = vulkan.getSwapchain();

    if (swapchain == VK_NULL_HANDLE) {
        throw std::runtime_error("Swapchain is NULL!");
    }

    VkResult acquireResult = vkAcquireNextImageKHR(
        vulkan.getDevice(), swapchain, UINT64_MAX, imageAvailableSemaphore, VK_NULL_HANDLE, &imageIndex  // Use semaphore
    );

    if (acquireResult != VK_SUCCESS) {
        throw std::runtime_error("Failed to acquire swapchain image!");
    }

    std::cout << "Acquired swapchain image index: " << imageIndex << std::endl;

    // Begin render pass
    VkRenderPassBeginInfo renderPassInfo{};
    renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    renderPassInfo.renderPass = renderPass;
    renderPassInfo.framebuffer = framebuffer;
    renderPassInfo.renderArea.offset = {0, 0};
    renderPassInfo.renderArea.extent = {static_cast<uint32_t>(width * scaleFactor), static_cast<uint32_t>(height * scaleFactor)};

    VkClearValue clearValues[2];
    clearValues[0].color = {{1.0f, 1.0f, 1.0f, 1.0f}};  // Background color 
    clearValues[1].depthStencil = {1.0f, 0};            // Depth clear value

    renderPassInfo.clearValueCount = 2;   // Two clear values (color + depth)
    renderPassInfo.pClearValues = clearValues;

    vkCmdBeginRenderPass(commandBuffer, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipeline);

    std::cout << "before descriptor set: " << imageIndex << std::endl;

    // Ensure descriptor set is valid before binding
    if (descriptorSet == VK_NULL_HANDLE) {
        throw std::runtime_error("Descriptor set is NULL!");
    }

    // Bind descriptor set before drawing
    VkDescriptorSet descriptorSets[] = {descriptorSet, cameraDescriptorSet};
    vkCmdBindDescriptorSets(
        commandBuffer,
        VK_PIPELINE_BIND_POINT_GRAPHICS,
        pipelineLayout,
        0, 
        1, descriptorSets,  // for both Gaussian and Camera
        0, nullptr
    );

    // std::cout << "after descriptor set: " << imageIndex << std::endl;

    // checking data in buffers    
    // std::cout << "First 5 Gaussians (GPU buffer check)" << std::endl;

    // void* gpuData;
    // vkMapMemory(vulkan.getDevice(), gaussianBufferMemory, 0, sizeof(Gaussian) * 5, 0, &gpuData);
    // Gaussian* gpuGaussians = static_cast<Gaussian*>(gpuData);

    // for (size_t i = 0; i < 2; i++) {
    //     const auto& g = gpuGaussians[i];
        
    //     std::cout << "Gaussian " << i << ":\n"
    //             << "  Position: ("
    //             << std::fixed << std::setprecision(6)  // Set decimal precision
    //             << reinterpret_cast<const float*>(&g.position)[0] << ", "
    //             << reinterpret_cast<const float*>(&g.position)[1] << ", "
    //             << reinterpret_cast<const float*>(&g.position)[2] << ")\n"
                
    //             << "  Color: ("
    //             << reinterpret_cast<const float*>(&g.color)[0] << ", "
    //             << reinterpret_cast<const float*>(&g.color)[1] << ", "
    //             << reinterpret_cast<const float*>(&g.color)[2] << ")\n"
                
    //             << "  Covariance:\n";

    //     const float* covar = reinterpret_cast<const float*>(&g.covariance);
    //     for (int row = 0; row < 3; row++) {
    //         std::cout << "    ";
    //         for (int col = 0; col < 3; col++) {
    //             std::cout << covar[row * 3 + col] << " ";
    //         }
    //         std::cout << "\n";
    //     }
        
    //     std::cout << "  Opacity: " << g.opacity << "\n\n";
    // }

    // vkUnmapMemory(vulkan.getDevice(), gaussianBufferMemory);

    // std::cout << "Camera Buffer (GPU Check):" << std::endl;

    // void* cameraDataGPU;
    // vkMapMemory(vulkan.getDevice(), cameraBufferMemory, 0, sizeof(CameraBuffer), 0, &cameraDataGPU);
    // CameraBuffer* gpuCameraBuffer = static_cast<CameraBuffer*>(cameraDataGPU);

    // std::cout << "View Matrix:" << std::endl;
    // for (int i = 0; i < 4; i++) {
    //     std::cout << "  ";
    //     for (int j = 0; j < 4; j++) {
    //         std::cout << std::fixed << std::setprecision(6) << gpuCameraBuffer->view[i][j] << " ";
    //     }
    //     std::cout << std::endl;
    // }

    // std::cout << "Projection Matrix:" << std::endl;
    // for (int i = 0; i < 4; i++) {
    //     std::cout << "  ";
    //     for (int j = 0; j < 4; j++) {
    //         std::cout << std::fixed << std::setprecision(6) << gpuCameraBuffer->projection[i][j] << " ";
    //     }
    //     std::cout << std::endl;
    // }

    // std::cout << "Image Size: (" << gpuCameraBuffer->imageSize.x << ", " << gpuCameraBuffer->imageSize.y << ")" << std::endl;

    // vkUnmapMemory(vulkan.getDevice(), cameraBufferMemory);

    // bind vertex data
    VkDeviceSize offsets[] = {0};
    vkCmdBindVertexBuffers(commandBuffer, 0, 1, &gaussianBuffer, offsets);
    vkCmdDraw(commandBuffer, gaussianCount, 1, 0, 0);  
    vkCmdEndRenderPass(commandBuffer);

    if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS) {
        throw std::runtime_error("Failed to record command buffer!");
    }

    // std::cout << "after cmddraw and render pass: " << imageIndex << std::endl;

    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    VkSemaphore waitSemaphores[] = {imageAvailableSemaphore};
    VkPipelineStageFlags waitStages[] = {VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};
    submitInfo.waitSemaphoreCount = 1;
    submitInfo.pWaitSemaphores = waitSemaphores;
    submitInfo.pWaitDstStageMask = waitStages;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer;
    VkSemaphore signalSemaphores[] = {renderFinishedSemaphore};
    submitInfo.signalSemaphoreCount = 1;
    submitInfo.pSignalSemaphores = signalSemaphores;

    if (vkQueueSubmit(vulkan.getGraphicsQueue(), 1, &submitInfo, VK_NULL_HANDLE) != VK_SUCCESS) {
        throw std::runtime_error("Failed to submit draw command buffer!");
    }

    VkPresentInfoKHR presentInfo{};
    presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    presentInfo.swapchainCount = 1;
    presentInfo.pSwapchains = &swapchain; 
    presentInfo.waitSemaphoreCount = 1;
    presentInfo.pWaitSemaphores = signalSemaphores;
    presentInfo.pImageIndices = &imageIndex;  

    if (vkQueuePresentKHR(vulkan.getGraphicsQueue(), &presentInfo) != VK_SUCCESS) {
        throw std::runtime_error("Failed to present swapchain image!");
    }

    std::cout << "Frame presented successfully!" << std::endl;
}

void GaussianPipeline::createDepthResources() {
    VkExtent2D extent = {static_cast<uint32_t>(width * scaleFactor), static_cast<uint32_t>(height * scaleFactor)}; // Match swapchain size
    VkFormat depthFormat = VK_FORMAT_D32_SFLOAT;

    // Create depth image
    VkImageCreateInfo imageInfo{};
    imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageInfo.imageType = VK_IMAGE_TYPE_2D;
    imageInfo.format = depthFormat;
    imageInfo.extent.width = extent.width;
    imageInfo.extent.height = extent.height;
    imageInfo.extent.depth = 1;
    imageInfo.mipLevels = 1;
    imageInfo.arrayLayers = 1;
    imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    imageInfo.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    if (vkCreateImage(vulkan.getDevice(), &imageInfo, nullptr, &depthImage) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create depth image!");
    }

    // Allocate memory for depth image
    VkMemoryRequirements memRequirements;
    vkGetImageMemoryRequirements(vulkan.getDevice(), depthImage, &memRequirements);

    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memRequirements.size;
    allocInfo.memoryTypeIndex = vulkan.findMemoryType(memRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    if (vkAllocateMemory(vulkan.getDevice(), &allocInfo, nullptr, &depthImageMemory) != VK_SUCCESS) {
        throw std::runtime_error("Failed to allocate depth image memory!");
    }

    vkBindImageMemory(vulkan.getDevice(), depthImage, depthImageMemory, 0);

    // Create depth image view
    VkImageViewCreateInfo viewInfo{};
    viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewInfo.image = depthImage;
    viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.format = depthFormat;
    viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
    viewInfo.subresourceRange.baseMipLevel = 0;
    viewInfo.subresourceRange.levelCount = 1;
    viewInfo.subresourceRange.baseArrayLayer = 0;
    viewInfo.subresourceRange.layerCount = 1;

    if (vkCreateImageView(vulkan.getDevice(), &viewInfo, nullptr, &depthImageView) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create depth image view!");
    }
}

void GaussianPipeline::createCameraBuffer(CameraBuffer &cameraData) {
    std::cout << "Creating camera buffer..." << std::endl;

    VkDeviceSize bufferSize = sizeof(CameraBuffer);

    VkBufferCreateInfo bufferInfo{};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = bufferSize;
    bufferInfo.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    if (vkCreateBuffer(vulkan.getDevice(), &bufferInfo, nullptr, &cameraBuffer) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create Camera buffer!");
    }

    // std::cout << "camera buffer memres..." << std::endl;

    VkMemoryRequirements memRequirements;
    vkGetBufferMemoryRequirements(vulkan.getDevice(), cameraBuffer, &memRequirements);

    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memRequirements.size;
    allocInfo.memoryTypeIndex = vulkan.findMemoryType(
        memRequirements.memoryTypeBits,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
    );

    if (vkAllocateMemory(vulkan.getDevice(), &allocInfo, nullptr, &cameraBufferMemory) != VK_SUCCESS) {
        throw std::runtime_error("Failed to allocate Camera buffer memory!");
    }
    // std::cout << "camera buffer alloced..." << std::endl;

    vkBindBufferMemory(vulkan.getDevice(), cameraBuffer, cameraBufferMemory, 0);
    // std::cout << "camera buffer bound..." << std::endl;

    // Upload Camera Data
    void* data;
    vkMapMemory(vulkan.getDevice(), cameraBufferMemory, 0, bufferSize, 0, &data);
    memcpy(data, &cameraData, bufferSize);
    vkUnmapMemory(vulkan.getDevice(), cameraBufferMemory);
    std::cout << "camera data uploaded..." << std::endl;
}
