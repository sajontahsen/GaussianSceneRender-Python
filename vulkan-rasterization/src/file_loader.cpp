#include "file_loader.h"
#include <fstream>
#include <sstream>
#include <iostream>

std::vector<Gaussian> FileLoader::loadGaussianData(const std::string &filename) {
    std::vector<Gaussian> gaussians;
    std::ifstream file(filename, std::ios::binary);

    if (!file.is_open()) {
        throw std::runtime_error("Failed to open Gaussian binary file!");
    }

    Gaussian g;
    while (file.read(reinterpret_cast<char*>(&g.position), sizeof(glm::vec3))) {
        file.read(reinterpret_cast<char*>(&g.color), sizeof(glm::vec3));
        file.read(reinterpret_cast<char*>(&g.covariance), sizeof(glm::mat3));
        file.read(reinterpret_cast<char*>(&g.opacity), sizeof(float));

        gaussians.push_back(g);
    }

    file.close();
    return gaussians;
}

CameraBuffer FileLoader::loadCameraData(const std::string &cameraFilename) {
    CameraBuffer cameraData;

    std::ifstream file(cameraFilename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open camera binary file: " + cameraFilename);
    }

    file.read(reinterpret_cast<char*>(&cameraData.view), sizeof(glm::mat4));         // 64 bytes
    file.read(reinterpret_cast<char*>(&cameraData.projection), sizeof(glm::mat4));   // 64 bytes
    file.read(reinterpret_cast<char*>(&cameraData.imageSize), sizeof(glm::ivec2));   // 8 bytes
    file.close();


    std::cout << "Camera Data Loaded Successfully!" << std::endl;

    return cameraData;
}
