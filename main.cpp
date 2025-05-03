#include <cstdint>
#include <memory>
#include <random>
#include <chrono>
#include <cmath>
#include <sys/types.h>
#include "VulkanDevice.hpp"
#include "VulkanInstance.hpp"

#include "logger.hpp"

using namespace vkop;

int main() {
    Logger::getInstance().setLevel(LOG_INFO);
    Logger::getInstance().enableFileOutput("log", true);
    try {
        auto phydevs = VulkanInstance::getVulkanInstance().getPhysicalDevices();
        for (auto pdev : phydevs) {
            auto dev = std::make_shared<VulkanDevice>(pdev);
            LOG_INFO("%s",dev->getDeviceName().c_str());
        }
    } catch (const std::exception &e) {
        LOG_ERROR("%s", e.what());
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}