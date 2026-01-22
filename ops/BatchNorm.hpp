// Copyright 2025 @junka
#ifndef OPS_BATCHNORM2D_HPP_
#define OPS_BATCHNORM2D_HPP_

#include "Operator.hpp"
#include "ops/Conv2d.hpp"

#include <memory>
extern "C" {
extern unsigned char batchnorm_spv[];
extern unsigned int batchnorm_spv_len;
}
namespace vkop {
namespace ops {
namespace batchnorm {

// torch.nn.functional.batch_norm(input, running_mean, running_var, weight=None,
//                                bias=None, training=False, momentum=0.1,
//                                eps=1e-05)
struct alignas(16) GpuBatchNormParam {
    ivec4 outShape;
    float eps; // default 1e-5
    int activation;
};
} // namespace batchnorm

class BatchNorm : public Operator {
  public:
    BatchNorm()
        : Operator(OpType::BATCHNORM, batchnorm_spv, batchnorm_spv_len,
                   {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                    VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                    DESCRIPTOR_TYPE_UNIFORM},
                   sizeof(batchnorm::GpuBatchNormParam)) {}

    void setAttribute(const std::unordered_map<std::string, std::string>
                          &attributes) override {
        // skip training, training_mode, spatial, since we don't need them
        if (attributes.find("eps") != attributes.end()) {
            eps_ = std::stof(attributes.at("eps"));
        }

        if (attributes.find("activation") != attributes.end()) {
            std::string activation = attributes.at("activation");
            if (activation == "Relu") {
                activation_ = conv2d::ActivationMode::RELU;
            } else if (activation == "Sigmoid") {
                activation_ = conv2d::ActivationMode::SIGMOID;
            } else if (activation == "Tanh") {
                activation_ = conv2d::ActivationMode::TANH;
            } else if (activation == "HardSwish") {
                activation_ = conv2d::ActivationMode::HARDSWISH;
            } else if (activation == "Mish") {
                activation_ = conv2d::ActivationMode::MISH;
            } else if (activation == "Relu6") {
                activation_ = conv2d::ActivationMode::RELU6;
            } else if (activation == "GateSigmoid") {
                activation_ = conv2d::ActivationMode::SWISH;
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
        auto input_shape = inputs[0]->getShape();

        dispatch_by_dtype(outputs[0]->dtype(), [&](auto t) {
            using T = decltype(t);
            auto outputptr = core::as_tensor<T>(outputs[0]);
            if (outputptr->size() == 0) {
                outputptr->resize(input_shape);
            }
            auto output_image = outputptr->as_output_image(m_dev_, m_cmd_);
            objs_.emplace_back(output_image);
        });

        dispatch_by_dtype(inputs[0]->dtype(), [&](auto t) {
            using T = decltype(t);
            auto inputptr = core::as_tensor<T>(inputs[0]);
            auto input_image = inputptr->as_input_image(m_dev_, m_cmd_);
            objs_.emplace_back(input_image);
        });

        dispatch_by_dtype(inputs[1]->dtype(), [&](auto t) {
            using T = decltype(t);
            auto para = core::as_tensor<T>(inputs[1]);
            auto para_buffer = para->as_uniform_bufferview(m_dev_);
            objs_.emplace_back(para_buffer);
        });

        auto gpu_shape = outputs[0]->getGPUShape();
        batchnorm::GpuBatchNormParam para;
        para.eps = eps_;
        para.activation = static_cast<int>(activation_);
        para.outShape[0] = input_shape[0];
        para.outShape[1] = input_shape[1];
        para.outShape[2] = input_shape[2];
        para.outShape[3] = input_shape[3];

        submit(&para, UP_DIV(gpu_shape[0], 16), UP_DIV(gpu_shape[1], 16),
               gpu_shape[2]);
    }

    float eps_ = 1e-5;
    conv2d::ActivationMode activation_ = conv2d::ActivationMode::NONE;
};

} // namespace ops
} // namespace vkop
#endif // OPS_BATCHNORM2D_HPP_
