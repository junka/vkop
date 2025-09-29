#ifndef VKOP_TESTS_HPP_
#define VKOP_TESTS_HPP_

#include <cmath>
#include <cstring>
#include <string>
#include <vector>

#include "core/Tensor.hpp"
#include "vulkan/VulkanDevice.hpp"
#include "vulkan/VulkanInstance.hpp"
#include "include/logger.hpp"
#include "ops/OperatorFactory.hpp"
#include "ops/Ops.hpp"

namespace vkop {
namespace tests {

using vkop::core::Tensor;

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

    static vkop::ops::OpType convert_opstring_to_enum(const std::string &name) {
        if (name == "Add") return vkop::ops::OpType::ADD;
        if (name == "Sub") return vkop::ops::OpType::SUB;
        if (name == "Mul") return vkop::ops::OpType::MUL;
        if (name == "Div") return vkop::ops::OpType::DIV;
        if (name == "Atan") return vkop::ops::OpType::ATAN;
        if (name == "Erf") return vkop::ops::OpType::ERF;
        if (name == "Pow") return vkop::ops::OpType::POW;
        if (name == "BatchNorm") return vkop::ops::OpType::BATCHNORM;
        if (name == "Relu") return vkop::ops::OpType::RELU;
        if (name == "Softmax") return vkop::ops::OpType::SOFTMAX;
        if (name == "Tanh") return vkop::ops::OpType::TANH;
        if (name == "MatMul") return vkop::ops::OpType::MATMUL;
        if (name == "Conv2d" || name == "Conv") return vkop::ops::OpType::CONV2D;
        if (name == "MaxPool2d" || name == "MaxPool") return vkop::ops::OpType::MAXPOOL2D;
        if (name == "AvgPool2d") return vkop::ops::OpType::AVGPOOL2D;
        if (name == "Upsample2d") return vkop::ops::OpType::UPSAMPLE2D;
        if (name == "GridSample") return vkop::ops::OpType::GRIDSAMPLE;
        if (name == "Constant") return vkop::ops::OpType::CONSTANT;
        if (name == "Floor") return vkop::ops::OpType::FLOOR;
        if (name == "Resize") return vkop::ops::OpType::RESIZE;
        return vkop::ops::OpType::UNKNOWN;
    }

    template <typename T>
    bool run_test(const std::vector<std::shared_ptr<Tensor<T>>> &inputs,
        const std::vector<T> &expectedOutput,
        const std::function<void(std::unique_ptr<ops::Operator> &)> &attribute_func)
    {
        try {
            auto phydevs = VulkanInstance::getVulkanInstance().getPhysicalDevices();
            for (auto *pdev : phydevs) {
                auto dev = std::make_shared<VulkanDevice>(pdev);
                if (dev->getDeviceName().find("llvmpipe") != std::string::npos) {
                    continue;
                }
                auto *device = dev->getLogicalDevice();
                auto cmdpool = std::make_shared<VulkanCommandPool>(device, dev->getComputeQueueFamilyIndex());

                auto op = ops::OperatorFactory::get_instance().create(convert_opstring_to_enum(name_));
                if (!op) {
                    LOG_ERROR("Fail to create operator");
                    return false;
                }
                op->set_runtime_device(pdev, dev, cmdpool);

                // Apply the attribute function callback if provided
                if (attribute_func) {
                    attribute_func(op);
                }

                auto output = std::make_shared<Tensor<T>>();
                op->execute(inputs, std::vector<std::shared_ptr<Tensor<T>>> {output});
                auto *out_ptr = output->data();
                for (int i = 0; i < output->num_elements(); i++) {
                    std::cout << i<< ": " << out_ptr[i] << " vs " <<expectedOutput[i] << std::endl;
                    if (std::fabs(out_ptr[i] - expectedOutput[i]) > 0.001) {
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

    template <typename T>
    bool run_test(const std::vector<std::shared_ptr<Tensor<T>>> &inputs,
        const std::vector<T> &expectedOutput)
    {
        try {
            auto phydevs = VulkanInstance::getVulkanInstance().getPhysicalDevices();
            for (auto *pdev : phydevs) {
                auto dev = std::make_shared<VulkanDevice>(pdev);
                if (dev->getDeviceName().find("llvmpipe") != std::string::npos) {
                    continue;
                }
                auto *device = dev->getLogicalDevice();
                auto cmdpool = std::make_shared<VulkanCommandPool>(device, dev->getComputeQueueFamilyIndex());

                auto op = ops::OperatorFactory::get_instance().create(convert_opstring_to_enum(name_));
                if (!op) {
                    LOG_ERROR("Fail to create operator");
                    return false;
                }
                op->set_runtime_device(pdev, dev, cmdpool);

                auto output = std::make_shared<Tensor<T>>();
                op->execute(inputs, std::vector<std::shared_ptr<Tensor<T>>> {output});
                auto *out_ptr = output->data();
                for (int i = 0; i < output->num_elements(); i++) {
                    if (sizeof(T) == 2) {
                        if (std::fabs(float16_to_float32(out_ptr[i]) - float16_to_float32(expectedOutput[i])) > 0.01) {
                            LOG_ERROR("Test Fail at1 (%d): %f, %f", i, float16_to_float32(out_ptr[i]), float16_to_float32(expectedOutput[i]));
                            return false;
                        }
                    } else {
                        if (std::fabs(out_ptr[i] - expectedOutput[i]) > 0.001) {
                            LOG_ERROR("Test Fail at (%d): %f, %f", i, out_ptr[i], expectedOutput[i]);
                            return false;
                        }
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

    static uint16_t float32_to_float16(float f) {
        uint32_t x;
        std::memcpy(&x, &f, sizeof(x));

        uint32_t sign = (x >> 16) & 0x8000; // 符号位 (bit 15)
        uint32_t exp = (x >> 23) & 0xFF;    // 指数 (8 bits)
        uint32_t mantissa = x & 0x7FFFFF;   // 尾数 (23 bits)

        uint16_t h = 0;

        if (exp == 0) {
            // FP32 zero or denormal
            h = static_cast<uint16_t>(sign | (mantissa != 0 ? 1 : 0)); // 保持为 0 或最小正数
        } else if (exp == 0xFF) {
            // Inf or NaN
            h = static_cast<uint16_t>(sign | 0x7C00 | (mantissa ? 0x200 : 0));
        } else {
            int new_exp = exp - 127 + 15; // 调整偏移

            if (new_exp >= 31) {
                // 溢出 → Inf
                h = static_cast<uint16_t>(sign | 0x7C00);
            } else if (new_exp <= 0) {
                // 下溢 → 非规格化或 0
                if (new_exp < -10) {
                    h = static_cast<uint16_t>(sign); // underflow to zero
                } else {
                    // 生成非规格化数
                    uint32_t shifted_mantissa = mantissa | 0x800000; // 添加隐含位
                    shifted_mantissa >>= (1 - new_exp); // 右移
                    h = static_cast<uint16_t>(sign | (shifted_mantissa >> 13));
                }
            } else {
                // 正规数
                h = static_cast<uint16_t>(sign | (new_exp << 10) | (mantissa >> 13));
            }
        }

        return h;
    }

    static float float16_to_float32(uint16_t h) {
        uint32_t sign = (h & 0x8000) << 16; // 符号位
        uint32_t exponent = (h & 0x7C00);    // 指数位
        uint32_t mantissa = (h & 0x03FF);    // 尾数位
    
        if (exponent == 0x7C00) { // Inf 或 NaN
            exponent = 0x3FC00; // FP32 的 Inf/Nan 指数
        } else if (exponent != 0) { // 正规数
            exponent = (exponent >> 10) + (127 - 15); // 指数偏移调整
            exponent <<= 23;
        } else if (mantissa != 0) { // 非规格化数
            // 处理非规格化数（denormal）
            int shift = __builtin_clz(mantissa) - 22; // GCC 内置函数
            mantissa <<= shift;
            exponent = (127 - 15 - shift + 1) << 23;
        }
        // 否则为 0
    
        mantissa <<= 13; // 尾数左移
        auto ret = (sign | exponent | mantissa);
        float retfloat;
        std::memcpy(&retfloat, &ret, 4);
        return retfloat;
    }
};

} // namespace tests
} // namespace vkop

#endif // VKOP_TESTS_HPP_
