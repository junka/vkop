#ifndef VKOP_TESTS_HPP_
#define VKOP_TESTS_HPP_

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

#include "core/Tensor.hpp"
#include "vulkan/VulkanCommandPool.hpp"
#include "vulkan/VulkanDevice.hpp"
#include "vulkan/VulkanInstance.hpp"
#include "include/logger.hpp"
#include "ops/OperatorFactory.hpp"
#include "ops/Ops.hpp"

#include <gtest/gtest.h>
#include <torch/torch.h>

namespace vkop {
namespace tests {

using vkop::core::Tensor;

class TestEnv : public testing::Environment {
private:
    static std::shared_ptr<VulkanDevice> dev_;
    static std::shared_ptr<VulkanCommandPool> cmdpool_;
    static bool initialized_;

public:
    static void initialize() {
        if (initialized_) return;

        Logger::getInstance().setLevel(LOG_INFO);
        Logger::getInstance().enableFileOutput("log", false);

        const auto& phydevs = VulkanInstance::getVulkanInstance().getPhysicalDevices();
        // enumPhysicalDevices() already sorts real GPUs before llvmpipe,
        // so phydevs[0] is always the best available device.
        dev_ = std::make_shared<VulkanDevice>(phydevs[0]);
        LOG_INFO("Initialized Vulkan device: %s", dev_->getDeviceName().c_str());
        cmdpool_ = std::make_shared<VulkanCommandPool>(dev_);
        initialized_ = true;
    }

    static void cleanup() {
        if (!initialized_) return;
        cmdpool_.reset();
        dev_.reset();
        initialized_ = false;
    }

    void SetUp() override {
        initialize();
    }

    void TearDown() override {
        cleanup();
    }

    static std::shared_ptr<VulkanDevice> get_device() {
        return dev_;
    }

    static std::shared_ptr<VulkanCommandPool> get_command_pool() {
        return cmdpool_;
    }

    static bool is_initialized() {
        return initialized_;
    }
};


template<typename T>
class TestCase {
private:
    std::string name_;
protected:
    std::shared_ptr<VulkanDevice> dev_;
    std::shared_ptr<VulkanCommandPool> cmdpool_;
    torch::TensorOptions conf_;
public:
    TestCase() = delete;
    explicit TestCase(const std::string& name): name_(std::move(name)) {
        if (!TestEnv::is_initialized()) {
            TestEnv::initialize();
        }
        dev_ = TestEnv::get_device();
        cmdpool_ = TestEnv::get_command_pool();
        if constexpr (std::is_same_v<T, float>) {
            conf_ = torch::TensorOptions().dtype(torch::kFloat32);
        } else if constexpr (std::is_same_v<T, uint16_t>) {
            conf_ = torch::TensorOptions().dtype(torch::kFloat16);
        }
    }
    ~TestCase() = default;
    TestCase(const TestCase &test) = delete;
    TestCase(const TestCase &&test) = delete;
    TestCase &operator=(const TestCase &) = delete;
    TestCase &operator=(const TestCase &&) = delete;

    virtual torch::TensorOptions getTorchConf() const {
        return conf_;
    }

    template<typename TT>
    void fillTensorFromTorch(std::shared_ptr<Tensor<TT>>& tensor,
                             const torch::Tensor& torch_tensor) {
        auto cpu_tensor = torch_tensor.cpu().contiguous().flatten();
        std::vector<TT> data_vector;
        data_vector.reserve(cpu_tensor.numel());
        if constexpr (std::is_same_v<float, TT> || std::is_same_v<int, TT> ) {
            auto accessor = cpu_tensor.accessor<TT, 1>();
            for (int64_t i = 0; i < cpu_tensor.numel(); i++) {
                data_vector.push_back(accessor[i]);
            }
        } else if constexpr (std::is_same_v<uint16_t, TT>) {
            auto *data_ptr = cpu_tensor.data_ptr<at::Half>();
            const auto *uint16_ptr = reinterpret_cast<const uint16_t*>(data_ptr);

            for (int64_t i = 0; i < cpu_tensor.numel(); i++) {
                data_vector.push_back(uint16_ptr[i]);
            }
        }
        tensor->fillToCPU(data_vector);
    }
    virtual bool verify_output(const std::unique_ptr<ops::Operator> &op,
                               int idx,
                               const std::shared_ptr<core::ITensor> &output,
                               const std::shared_ptr<core::ITensor> &expect) {
        return verify_output_default(output, expect);
    }

  protected:
    template <typename TT>
    static bool verify_output_typed(
        const std::shared_ptr<Tensor<TT>> &output,
        const std::shared_ptr<Tensor<TT>> &expect) {
        for (int i = 0; i < output->num_elements(); i++) {
            if constexpr (std::is_same_v<TT, uint16_t>) {
                float out_val = core::ITensor::fp16_to_fp32((*output)[i]);
                float exp_val = core::ITensor::fp16_to_fp32((*expect)[i]);
                if (std::isnan(exp_val)) {
                    LOG_ERROR("Test Fail: Expected value is NaN at index %d", i);
                    return false;
                }
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
            } else if constexpr (std::is_same_v<TT, int>) {
                if (std::isnan((*output)[i]) || (*output)[i] != (*expect)[i]) {
                    LOG_ERROR("Test Fail at2 (%d): %d, %d", i, (*output)[i], (*expect)[i]);
                    return false;
                }
            } else {
                float out_val = (*output)[i];
                float exp_val = (*expect)[i];
                if (std::isnan(out_val)) {
                    LOG_ERROR("Test Fail (%d): Output is NaN, expected %f", i, exp_val);
                    return false;
                }
                float abs_exp = std::abs(exp_val);
                float threshold;
                if (abs_exp > 1.0F) {
                    threshold = abs_exp * 0.01F;
                } else if (abs_exp > 0.001F) {
                    threshold = std::max(0.003F, abs_exp * 0.02F);
                } else {
                    threshold = 0.002F;
                }
                if (std::abs(out_val - exp_val) > threshold) {
                    LOG_ERROR("Test Fail (%d): %f vs %f (threshold: %f)", i, out_val, exp_val, threshold);
                    return false;
                }
            }
        }
        return true;
    }

  private:
    // Dispatch the default typed comparison from a type-erased ITensor pair.
    bool verify_output_default(const std::shared_ptr<core::ITensor> &output,
                               const std::shared_ptr<core::ITensor> &expect) {
        if (output->dtype() == typeid(float)) {
            return verify_output_typed(core::as_tensor<float>(output),
                                       core::as_tensor<float>(expect));
        } else if (output->dtype() == typeid(uint16_t)) {
            return verify_output_typed(core::as_tensor<uint16_t>(output),
                                       core::as_tensor<uint16_t>(expect));
        } else if (output->dtype() == typeid(int)) {
            return verify_output_typed(core::as_tensor<int>(output),
                                       core::as_tensor<int>(expect));
        } else if (output->dtype() == typeid(int64_t)) {
            return verify_output_typed(core::as_tensor<int64_t>(output),
                                       core::as_tensor<int64_t>(expect));
        }
        return false;
    }

  public:
    bool run_test(const std::vector<std::shared_ptr<core::ITensor>> &inputs,
        const std::vector<std::shared_ptr<core::ITensor>> &expect_outputs,
        const std::function<void(std::unique_ptr<ops::Operator> &)> &attribute_func = nullptr)
    {

        // VKOP_BUFFER_BACKEND=1 selects the SSBO buffer backend (compact
        // row-major tensors of arbitrary rank) instead of the image path.
        bool use_buffer = [] {
            const char *env = std::getenv("VKOP_BUFFER_BACKEND");
            return env && env[0] == '1';
        }();
        // Softmax over a 1-D/2-D tensor has no image (image2DArray needs
        // >=3-D) representation, so it must use the buffer backend even when
        // VKOP_BUFFER_BACKEND is unset.
        if (vkop::ops::convert_opstring_to_enum(name_) ==
                vkop::ops::OpType::SOFTMAX &&
            inputs[0]->num_dims() <= 2) {
            use_buffer = true;
        }

        auto op = ops::create_from_type(vkop::ops::convert_opstring_to_enum(name_), typeid(T) == typeid(uint16_t) ? 1 : 0, dev_->is_support_nv_tensor_core() ? 2 : dev_->is_support_cooperate_matrix() ? 1: 0, use_buffer);
        if (!op) {
            LOG_ERROR("Fail to create operator");
            return false;
        }

        op->set_runtime_device(dev_, cmdpool_);

        if (attribute_func) {
            attribute_func(op);
        }
        std::vector<std::shared_ptr<core::ITensor>> outputs;
        for (const auto & expect_output : expect_outputs) {
            if (expect_output->dtype() == typeid(float)) {
                auto output = std::make_shared<Tensor<float>>(true);
                output->resize(expect_output->getShape());
                outputs.push_back(output);
            } else if (expect_output->dtype() == typeid(uint16_t)) {
                auto output = std::make_shared<Tensor<uint16_t>>(true);
                output->resize(expect_output->getShape());
                outputs.push_back(output);
            } else if (expect_output->dtype() == typeid(int)) {
                auto output = std::make_shared<Tensor<int>>(true);
                output->resize(expect_output->getShape());
                outputs.push_back(output);
            } else if (expect_output->dtype() == typeid(int64_t)) {
                auto output = std::make_shared<Tensor<int64_t>>(true);
                output->resize(expect_output->getShape());
                outputs.push_back(output);
            } else {
                LOG_ERROR("Unsupported output tensor type");
                return false;
            }
        }
        for (const auto &input : inputs) {
            if (!input) {
                continue;
            }
            if (input->dtype() == typeid(int64_t)) {
                // int64 tensors are read by shaders as ivec2[] (8-byte
                // stride) for Gather/ScatterND, but some ops (Reshape/Slice/
                // Expand shape meta-chain) read them on the host. Pass an
                // explicit src so copyToGPU keeps the CPU copy alive.
                auto t = core::as_tensor<int64_t>(input);
                auto keep = t->data();
                t->as_storage_buffer(dev_);
                t->copyToGPU(cmdpool_, keep.data());
                continue;
            }
            if (input->dtype() == typeid(int)) {
                auto t = core::as_tensor<int>(input);
                t->as_storage_buffer(dev_);
                t->copyToGPU(cmdpool_);
                continue;
            }
            if (input->dtype() == typeid(int8_t)) {
                // int8 weight-only quantized tensors (Conv2d weights) are
                // uploaded as compact SSBOs; the shader unpacks 4 bytes/uint.
                auto t = core::as_tensor<int8_t>(input);
                t->as_storage_buffer(dev_);
                t->copyToGPU(cmdpool_);
                continue;
            }
            if (use_buffer || op->get_type() == vkop::ops::OpType::GATHER || op->get_type() == vkop::ops::OpType::EXPAND) {
                // Buffer backend: SSBOs support any rank, no RGBA conversion.
                auto t = core::as_tensor<T>(input);
                t->as_storage_buffer(dev_);
                t->copyToGPU(cmdpool_);
                continue;
            }
            auto t = core::as_tensor<T>(input);
            assert(t);
            if (input->num_dims() <= 2) {
                if (vkop::ops::OpType::CONV2D == op->get_type() || vkop::ops::OpType::BATCHNORM == op->get_type()) {
                    t->as_uniform_bufferview(dev_);
                } else {
                    t->as_storage_buffer(dev_);
                }
            } else {
                t->as_input_image(dev_, nullptr);
            }
            t->copyToGPU(cmdpool_);
        }
        op->onExecute(inputs, outputs, 0);
        auto cmd = op->get_record();
        std::vector<VkSubmitInfo> info;
        info.push_back(cmd->buildSubmitInfo());
        VulkanCommandBuffer::submit(dev_->getComputeQueue(), info);
        cmd->wait();
        dev_->wait_all_done();

        auto check_ret = [&] (int idx, auto type_tag) -> bool {
            using TT = decltype(type_tag);
            auto output = core::as_tensor<TT>(outputs[idx]);
            output->copyToCPU(cmdpool_);
            output->print_tensor();
            auto expect = core::as_tensor<TT>(expect_outputs[idx]);
            if (vkop::ops::convert_opstring_to_enum(name_) == vkop::ops::OpType::TOPK) {
                output->resize(expect->getShape());
            }
            return verify_output(op, idx, outputs[idx], expect_outputs[idx]);
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
            } else if (outputs[idx]->dtype() == typeid(int64_t)) {
                if (!check_ret(idx, int64_t{}))
                    return false;
            } else {
                return false;
            }
        }
        LOG_INFO("Test Passed for operator: %s, type %s", name_.c_str(), typeid(T).name());
        return true;
    }
};

} // namespace tests
} // namespace vkop

#endif // VKOP_TESTS_HPP_
