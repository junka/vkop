#ifndef VKOP_TESTS_HPP_
#define VKOP_TESTS_HPP_

#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

#include "core/Tensor.hpp"
#include "vulkan/VulkanDevice.hpp"
#include "vulkan/VulkanInstance.hpp"
#include "include/logger.hpp"
#include "ops/OperatorFactory.hpp"
#include "ops/Ops.hpp"

#include <torch/torch.h>

namespace vkop {
namespace tests {

using vkop::core::Tensor;

class TestCase {
private:
    std::string name_;
public:
    TestCase() = delete;
    explicit TestCase(std::string name): name_(std::move(name)) {
    }
    ~TestCase() = default;
    TestCase(const TestCase &test) = delete;
    TestCase(const TestCase &&test) = delete;
    TestCase &operator=(const TestCase &) = delete;
    TestCase &operator=(const TestCase &&) = delete;

    template <typename T>
    bool run_test(const std::vector<std::shared_ptr<core::ITensor>> &inputs,
        const std::vector<std::shared_ptr<core::ITensor>> &expect_outputs,
        const std::function<void(std::unique_ptr<ops::Operator> &)> &attribute_func = nullptr)
    {
        auto phydevs = VulkanInstance::getVulkanInstance().getPhysicalDevices();
        auto dev = std::make_shared<VulkanDevice>(phydevs[0]);

        LOG_INFO("%s",dev->getDeviceName().c_str());
        auto cmdpool = std::make_shared<VulkanCommandPool>(dev);
        auto cmd = std::make_shared<VulkanCommandBuffer>(cmdpool);

        auto op = ops::create_from_type(vkop::ops::convert_opstring_to_enum(name_), inputs[0]->num_dims() <= 2);
        if (!op) {
            LOG_ERROR("Fail to create operator");
            return false;
        }
        op->set_runtime_device(dev, cmdpool);

        if (attribute_func) {
            attribute_func(op);
        }
        std::vector<std::shared_ptr<core::ITensor>> outputs;
        for (const auto & expect_output : expect_outputs) {
            if (expect_output->dtype() == typeid(float)) {
                auto output = std::make_shared<Tensor<float>>(true);
                outputs.push_back(output);
            } else if (expect_output->dtype() == typeid(uint16_t)) {
                auto output = std::make_shared<Tensor<uint16_t>>(true);
                outputs.push_back(output);
            } else if (expect_output->dtype() == typeid(int)) {
                auto output = std::make_shared<Tensor<int>>(true);
                outputs.push_back(output);
            } else {
                LOG_ERROR("Unsupported output tensor type");
                return false;
            }
        }
        for (const auto &input : inputs) {
            if (!input || input->dtype() == typeid(int64_t)) {
                continue;
            }
            auto t = core::as_tensor<T>(input);
            if (input->num_dims() <= 2) {
                if (vkop::ops::OpType::CONV2D == op->get_type() || vkop::ops::OpType::BATCHNORM == op->get_type()) {
                    t->as_uniform_bufferview(dev);
                } else {
                    t->as_storage_buffer(dev);
                }
            } else {
                t->as_input_image(dev, nullptr);
            }
            t->copyToGPU(cmdpool);
        }
        cmd->wait(dev->getComputeQueue());
        cmd->begin();
        op->onExecute(inputs, outputs, cmd, 0);
        cmd->end();
        cmd->submit(dev->getComputeQueue());

        auto check_ret = [&] (int idx, auto type_tag) -> bool {
            using TT = decltype(type_tag);
            auto output = core::as_tensor<TT>(outputs[idx]);
            output->copyToCPU(cmdpool);
            auto oshape = output->getShape();
            output->print_tensor();
            auto expect = core::as_tensor<TT>(expect_outputs[idx]);
            for (int i = 0; i < output->num_elements(); i++) {
                if constexpr (std::is_same_v<TT, uint16_t>) {
                    float out_val = core::ITensor::fp16_to_fp32((*output)[i]);
                    float exp_val = core::ITensor::fp16_to_fp32((*expect)[i]);
                    // std::cout << i << ": " << out_val << " vs " << exp_val << std::endl;
                    if (std::isnan(out_val)) {
                        LOG_ERROR("Test Fail at1 (%d): Output is NaN, expected %f", i, exp_val);
                        return false;
                    }

                    float abs_exp = std::abs(exp_val);
                    float threshold = (abs_exp > 1.0F) ? (abs_exp * 0.02F) : 0.02F;

                    if (std::abs(out_val - exp_val) > threshold) {
                        LOG_ERROR("Test Fail at1 (%d): %f vs %f (threshold: %f)", i, out_val, exp_val, threshold);
                        return false;
                    }
                } else if (typeid(TT) == typeid(int)) {
                    // std::cout << i << ": " << (*output)[i] << " vs " << (*expect)[i] << std::endl;
                    if (std::isnan((*output)[i]) || (*output)[i] != (*expect)[i]) {
                        LOG_ERROR("Test Fail at2 (%d): %d, %d", i, (*output)[i], (*expect)[i]);
                        return false;
                    }
                } else {
                    float out_val = (*output)[i];
                    float exp_val = (*expect)[i];
                    // std::cout << i << ": " << out_val << " vs " << exp_val << std::endl;

                    if (std::isnan(out_val)) {
                        LOG_ERROR("Test Fail (%d): Output is NaN, expected %f", i, exp_val);
                        return false;
                    }

                    // Adaptive tolerance based on magnitude of expected value
                    float abs_exp = std::abs(exp_val);
                    float threshold;
                    if (abs_exp > 1.0F) {
                        // For larger values, use relative error (e.g., 1% of expected value)
                        threshold = abs_exp * 0.01F;  // 1% relative error
                    } else if (abs_exp > 0.001F) {
                        // For medium values, use mixed relative/absolute error
                        threshold = std::max(0.001F, abs_exp * 0.02F);
                    } else {
                        // For very small values, use absolute error
                        threshold = 0.002F;
                    }
                    
                    if (std::abs(out_val - exp_val) > threshold) {
                        LOG_ERROR("Test Fail (%d): %f vs %f (threshold: %f)", i, out_val, exp_val, threshold);
                        return false;
                    }
                }
            }
            return true;
        };

        for (size_t idx = 0; idx < outputs.size(); idx++) {
            if (outputs[idx]->dtype() == typeid(float)) {
                if (!check_ret(idx, float{})) {
                    return false;
                }
            } else if (outputs[idx]->dtype() == typeid(uint16_t)){
                if (!check_ret(idx, uint16_t{}))
                    return false;
            } else if (outputs[idx]->dtype() == typeid(int)){
                if (!check_ret(idx, int{}))
                    return false;
            }
        }
        LOG_INFO("Test Passed for operator: %s", name_.c_str());
        return true;
    }
};

} // namespace tests
} // namespace vkop

#endif // VKOP_TESTS_HPP_
