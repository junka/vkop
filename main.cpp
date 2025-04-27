#include <cstdint>
#include <iostream>
#include <memory>
#include <random>
#include <chrono>
#include <cmath>
#include <sys/types.h>
#include "VulkanDevice.hpp"
#include "VulkanInstance.hpp"

using namespace vkop;

int main() {
    try {
        auto phydevs = VulkanInstance::getVulkanInstance().getPhysicalDevices();
        for (auto pdev : phydevs) {
            auto dev = std::make_shared<VulkanDevice>(pdev);
            std::cout << dev->getDeviceName() << std::endl;

        }
    } catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}