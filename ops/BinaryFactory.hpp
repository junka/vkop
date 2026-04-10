// Copyright 2025 @junka
#ifndef OPS_BINARY_FACTORY_HPP_
#define OPS_BINARY_FACTORY_HPP_

#include "Operator.hpp"
#include "core/Tensor.hpp"

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
    ivec4 shapea;
    ivec4 shapeb;
    int broadcast;
    int activation;
    int scaler;
};
} // namespace binary
class BinaryFactory : public Operator {
  public:
    explicit BinaryFactory(OpType type, uint8_t *spv, uint32_t spv_len)
        : Operator(type, spv, spv_len,
                   {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                    VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                    VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER},
                   (type == OpType::ADD) ? sizeof(binary::GPUBinParam) : 0) {
        param_.activation = static_cast<int>(binary::ActivationMode::NONE);
        param_.broadcast = 0;
    }

    void setAttribute(const std::unordered_map<std::string, std::string>
                          &attributes) override {
        if (attributes.count("activation") != 0) {
            std::string activation = attributes.at("activation");
            if (activation == "Relu") {
                param_.activation =
                    static_cast<int>(binary::ActivationMode::RELU);
            } else if (activation == "Sigmoid") {
                param_.activation =
                    static_cast<int>(binary::ActivationMode::SIGMOID);
            } else if (activation == "Tanh") {
                param_.activation =
                    static_cast<int>(binary::ActivationMode::TANH);
            } else if (activation == "HardSwish") {
                param_.activation =
                    static_cast<int>(binary::ActivationMode::HARDSWISH);
            } else if (activation == "Mish") {
                param_.activation =
                    static_cast<int>(binary::ActivationMode::MISH);
            } else if (activation == "Relu6") {
                param_.activation =
                    static_cast<int>(binary::ActivationMode::RELU6);
            } else if (activation == "GateSigmoid") {
                param_.activation =
                    static_cast<int>(binary::ActivationMode::GATE_SIGMOID);
            } else {
                throw std::invalid_argument("Unsupported activation: " +
                                            activation);
            }
        }
    }

  private:
    static std::vector<int>
    computeBroadcastShape(const std::vector<int> &shape1,
                          const std::vector<int> &shape2) {
        size_t max_dims = std::max(shape1.size(), shape2.size());
        std::vector<int> result_shape(max_dims, 1);

        for (int i = max_dims - 1; i >= 0; --i) {
            int idx1 = i - (max_dims - shape1.size());
            int idx2 = i - (max_dims - shape2.size());

            int dim1 = (idx1 >= 0) ? shape1[idx1] : 1;
            int dim2 = (idx2 >= 0) ? shape2[idx2] : 1;

            if (dim1 == 1 || dim2 == 1) {
                result_shape[i] = std::max(dim1, dim2);
            } else if (dim1 == dim2) {
                result_shape[i] = dim1;
            } else {
                throw std::runtime_error("Shapes are not broadcast-compatible");
            }
        }

        return result_shape;
    }
    void execute(
        const std::vector<std::shared_ptr<core::ITensor>> &inputs,
        const std::vector<std::shared_ptr<core::ITensor>> &outputs) override {

        std::vector<int> input_shape = inputs[0]->getShape();
        auto output_shape = inputs[1]->getShape();
        output_shape = computeBroadcastShape(input_shape, output_shape);

        dispatch_by_dtype(outputs[0]->dtype(), [&](auto t) {
            using T = decltype(t);
            auto outputptr = core::as_tensor<T>(outputs[0]);
            if (outputptr->size() == 0) {
                outputptr->resize(output_shape);
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

        auto out_gpu_shape = outputs[0]->getGPUShape();
        if (type_ != OpType::PRELU) {
            param_.shapea[0] = inputs[0]->get_batch();
            param_.shapeb[0] = inputs[1]->get_batch();
            param_.shapea[1] = inputs[0]->get_channel();
            param_.shapeb[1] = inputs[1]->get_channel();
            param_.shapea[2] = inputs[0]->get_height();
            param_.shapeb[2] = inputs[1]->get_height();
            param_.shapea[3] = inputs[0]->get_width();
            param_.shapeb[3] = inputs[1]->get_width();
            for (size_t i = 0; i < 4; i++) {
                if (param_.shapeb[i] != param_.shapea[i]) {
                    param_.broadcast = 1;
                    break;
                }
            }

            submit(&param_, UP_DIV(out_gpu_shape[0], 16),
                   UP_DIV(out_gpu_shape[1], 16), out_gpu_shape[2]);
        } else {
            submit(nullptr, UP_DIV(out_gpu_shape[0], 16),
                   UP_DIV(out_gpu_shape[1], 16), out_gpu_shape[2]);
        }
    }

    binary::GPUBinParam param_;
};

} // namespace ops
} // namespace vkop
#endif // OPS_ELEMENT_WISE_FACTORY_HPP_
