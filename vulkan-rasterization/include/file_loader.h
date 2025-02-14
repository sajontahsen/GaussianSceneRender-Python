#pragma once
#include <vector>
#include <string>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

struct Gaussian {
    glm::vec3 position;    // 3D Position
    glm::vec3 color;       // RGB Color
    glm::mat3 covariance;  // 3x3 Covariance Matrix
    float opacity;         // Opacity
};

struct CameraBuffer {
    glm::mat4 view;
    glm::mat4 projection;
    glm::ivec2 imageSize;
};

class FileLoader {
public:
    static std::vector<Gaussian> loadGaussianData(const std::string &filename);
    static CameraBuffer loadCameraData(const std::string &cameraFilename);
};
