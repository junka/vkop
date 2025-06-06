#ifndef VKOP_TESTS_HPP_
#define VKOP_TESTS_HPP_

#include <cmath>
#include <string>
#include <vector>

#include "Tensor.hpp"
#include "VulkanDevice.hpp"
#include "VulkanInstance.hpp"
#include "logger.hpp"
#include "OperatorFactory.hpp"

namespace vkop {
namespace tests {

class TestCase {
private:
    std::string name_;
public:
    TestCase() = delete;
    explicit TestCase(std::string name): name_(std::move(name)) {}
    TestCase(const TestCase &test) = delete;
    TestCase(const TestCase &&test) = delete;
    TestCase &operator=(const TestCase &) = delete;
    TestCase &operator=(const TestCase &&) = delete;

    virtual bool run_test(const std::vector<std::shared_ptr<core::Tensor<float>>> &inputs,
                    const std::vector<float> &expectedOutput)
    {
        try {
            auto phydevs = VulkanInstance::getVulkanInstance().getPhysicalDevices();
            for (auto *pdev : phydevs) {
                auto dev = std::make_shared<VulkanDevice>(pdev);
                if (dev->getDeviceName().find("llvmpipe") != std::string::npos) {
                    continue;
                }
                VkDevice device = dev->getLogicalDevice();
                VulkanCommandPool cmdpool(device, dev->getComputeQueueFamilyIndex());

                auto op = ops::OperatorFactory::get_instance().create(name_);
                if (!op) {
                    LOG_ERROR("Fail to create oprator");
                    return false;
                }
                op->set_runtime_device(pdev, dev, &cmdpool);

                auto output = std::make_shared<core::Tensor<float>>();
                op->execute(inputs, std::vector<std::shared_ptr<core::Tensor<float>>> {output});
                auto *out_ptr = output->data();
                for (int i = 0; i < output->num_elements(); i++) {
                    if (std::fabs(out_ptr[i] - expectedOutput[i]) > 0.01) {
                        LOG_ERROR("Test Fail at (%d): %f, %f", i, out_ptr[i], expectedOutput[i]);
                        return false;
                    }
                }
                LOG_INFO("Test Passed for operator: %s", name_.c_str());
            }
        } catch (const std::exception &e) {
            LOG_ERROR("%s\n", e.what());
            return false;
        }
        return true;
    }

};

} // namespace tests
} // namespace vkop

#endif // VKOP_TESTS_HPP_
