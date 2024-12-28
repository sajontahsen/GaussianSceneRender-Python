cmake_minimum_required(VERSION 3.10)
project(VulkanCompute)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)

# Find Vulkan
find_package(Vulkan REQUIRED)

# Executable
add_executable(VulkanCompute src/main.cpp src/vulkan_setup.cpp)

# Include directories
target_include_directories(VulkanCompute PRIVATE ${Vulkan_INCLUDE_DIRS})

# Link Vulkan
target_link_libraries(VulkanCompute Vulkan::Vulkan)

