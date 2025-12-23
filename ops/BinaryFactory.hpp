// Copyright 2025 @junka
#ifndef OPS_BINARY_FACTORY_HPP_
#define OPS_BINARY_FACTORY_HPP_

#include "Operator.hpp"

namespace vkop {
namespace ops {
namespace binary {
enum class ActivationMode {
    NONE,
    RELU,
    SIGMOID,
    TANH,
    HARDSWISH,
    MISH,
    RELU6,
    GATE_SIGMOID,
};
struct alignas(16) GPUBinParam {
    int activation;
    int scaler;
};
} // namespace binary
class BinaryFactory : public Operator {
  public:
    explicit BinaryFactory(OpType type, uint8_t *spv, uint32_t spv_len)
        : Operator(type, spv, spv_len,
                   (type == OpType::ADD) ? sizeof(binary::GPUBinParam) : 0) {
        n_imgs_ = 3;
        types_ = {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                  VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                  VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER};
        objs_.reserve(types_.size());
    }
    void setAttribute(const std::unordered_map<std::string, std::string>
                          &attributes) override {
        if (attributes.count("activation") != 0) {
            std::string activation = attributes.at("activation");
            if (activation == "Relu") {
                activation_ = binary::ActivationMode::RELU;
            } else if (activation == "Sigmoid") {
                activation_ = binary::ActivationMode::SIGMOID;
            } else if (activation == "Tanh") {
                activation_ = binary::ActivationMode::TANH;
            } else if (activation == "HardSwish") {
                activation_ = binary::ActivationMode::HARDSWISH;
            } else if (activation == "Mish") {
                activation_ = binary::ActivationMode::MISH;
            } else if (activation == "Relu6") {
                activation_ = binary::ActivationMode::RELU6;
            } else if (activation == "GateSigmoid") {
                activation_ = binary::ActivationMode::GATE_SIGMOID;
            } else {
                throw std::invalid_argument("Unsupported activation: " +
                                            activation);
            }
        }
    }

  private:
    void execute(
        const std::vector<std::shared_ptr<core::ITensor>> &inputs,
        const std::vector<std::shared_ptr<core::ITensor>> &outputs) override {
        std::vector<int> input_shape = inputs[0]->getShape();
        dispatch_by_dtype(outputs[0]->dtype(), [&](auto t) {
            using T = decltype(t);
            auto outputptr = core::as_tensor<T>(outputs[0]);
            if (outputptr->size() == 0) {
                outputptr->resize(input_shape);
            }
            auto output_image = outputptr->as_output_image(m_dev_, m_cmd_);
            objs_.emplace_back(output_image);
        });

        for (size_t i = 0; i <= 1; ++i) {
            dispatch_by_dtype(inputs[i]->dtype(), [&](auto t) {
                using T = decltype(t);
                auto inputptr = core::as_tensor<T>(inputs[i]);
                auto input_image = inputptr->as_input_image(m_dev_, m_cmd_);
                objs_.emplace_back(input_image);
            });
        }
        auto out_gpu_shape = inputs[0]->getGPUShape();
        if (type_ == OpType::ADD) {
            binary::GPUBinParam param = {static_cast<int>(activation_), 0};
            submit(&param, UP_DIV(out_gpu_shape[0], 16),
                   UP_DIV(out_gpu_shape[1], 16), out_gpu_shape[2]);
        } else {
            submit(nullptr, UP_DIV(out_gpu_shape[0], 16),
                   UP_DIV(out_gpu_shape[1], 16), out_gpu_shape[2]);
        }
    }

    binary::ActivationMode activation_ = binary::ActivationMode::NONE;
};

} // namespace ops
} // namespace vkop
#endif // OPS_ELEMENT_WISE_FACTORY_HPP_
