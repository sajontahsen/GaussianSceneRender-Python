cmake_minimum_required(VERSION 3.10)
project(VulkanGaussianSplatting)

find_package(Vulkan REQUIRED)
find_package(glfw3 REQUIRED) 

include_directories(include)

add_executable(gaussian_splatting
    src/main.cpp
    src/vulkan_setup.cpp
    src/gaussian_pipeline.cpp
    src/file_loader.cpp
)

target_link_libraries(gaussian_splatting Vulkan::Vulkan glfw) 
