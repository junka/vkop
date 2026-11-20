// Copyright 2025 @junka
#ifndef OPS_BATCHNORM2D_HPP_
#define OPS_BATCHNORM2D_HPP_

#include "Operator.hpp"
#include "ops/BufferBase.hpp"
#include "ops/Conv2d.hpp"
#include "ops/PimplFacade.hpp"

#include <memory>
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

// Buffer PC (mirrors shaders/buffer/batchnorm.comp). The channel axis is the
// one the parameter tensor indexes, so the host folds the shape into
// (outer, channels, spatial) around it. `fp16` and `params_fp16` say whether
// each edge packs two halfs per word; they are independent because the merged
// parameter tensor keeps the original mean/var dtype. One module serves both.
struct alignas(16) BatchNormPC {
    int outer;      // product of the dims before the channel axis
    int channels;   // C
    int spatial;    // product of the dims after the channel axis
    int params_fp16;
    int fp16;
    int activation; // ACTIVATION_* from shaders/common/activation.comp
    float eps;
    int _pad0;
};
static_assert(sizeof(BatchNormPC) == 32, "BatchNormPC drift");
static_assert(sizeof(BatchNormPC) <= 128, "BatchNormPC PC overflow");
} // namespace batchnorm

class BatchNormImage : public Operator {
  public:
    BatchNormImage()
        : Operator(OpType::BATCHNORM,
                   {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                    VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                    DESCRIPTOR_TYPE_UNIFORM},
                   sizeof(batchnorm::GpuBatchNormParam)) {}

    void setAttribute(const std::unordered_map<std::string, std::string>
                          &attributes) override {
        // skip training, training_mode, spatial, since we don't need them
        if (attributes.find("epsilon") != attributes.end()) {
            eps_ = std::stof(attributes.at("epsilon"));
        } else if (attributes.find("eps") != attributes.end()) {
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
// Buffer factory — SSBO-backed batch norm over an arbitrary rank. The channel
// axis is axis 1 (ONNX NCHW and this repo's every batchnorm use); the host
// folds the other dims into (outer, spatial). The merged parameter tensor is a
// plain SSBO in the converter's 16-scalar-per-4-channel layout, exactly as the
// image path reads it through a uniform texel view. One kernel serves fp32 and
// fp16 data and parameters (the `fp16` / `params_fp16` PC fields), so it is not
// a dual-fp16 shader.
class BatchNormBuffer : public BufferFactory {
  public:
    explicit BatchNormBuffer(int fp16)
        : BufferFactory(OpType::BATCHNORM,
                        {DESCRIPTOR_TYPE_STORAGE, DESCRIPTOR_TYPE_STORAGE,
                         DESCRIPTOR_TYPE_STORAGE},
                        sizeof(batchnorm::BatchNormPC), fp16) {}

    void setAttribute(const std::unordered_map<std::string, std::string>
                          &attributes) override {
        auto get = [&](const char *key) -> const std::string * {
            auto it = attributes.find(key);
            return it == attributes.end() ? nullptr : &it->second;
        };
        if (const auto *s = get("epsilon")) {
            eps_ = std::stof(*s);
        } else if (const auto *s = get("eps")) {
            eps_ = std::stof(*s);
        }
        if (const auto *s = get("activation")) {
            if (*s == "Relu") {
                activation_ = conv2d::ActivationMode::RELU;
            } else if (*s == "Sigmoid") {
                activation_ = conv2d::ActivationMode::SIGMOID;
            } else if (*s == "Tanh") {
                activation_ = conv2d::ActivationMode::TANH;
            } else if (*s == "HardSwish") {
                activation_ = conv2d::ActivationMode::HARDSWISH;
            } else if (*s == "Mish") {
                activation_ = conv2d::ActivationMode::MISH;
            } else if (*s == "Relu6") {
                activation_ = conv2d::ActivationMode::RELU6;
            } else if (*s == "GateSigmoid") {
                activation_ = conv2d::ActivationMode::SWISH;
            } else {
                throw std::invalid_argument("Unsupported activation: " + *s);
            }
        }
    }

  private:
    void execute(
        const std::vector<std::shared_ptr<core::ITensor>> &inputs,
        const std::vector<std::shared_ptr<core::ITensor>> &outputs) override {
        auto in_shape = inputs[0]->getShape();
        if (in_shape.size() < 2) {
            throw std::invalid_argument(
                "BatchNorm (buffer backend): input must have a channel axis "
                "(rank >= 2)");
        }
        int channels = in_shape[1];
        int outer = in_shape[0];
        int spatial = 1;
        for (size_t i = 2; i < in_shape.size(); ++i) {
            spatial *= in_shape[i];
        }
        if (inputs.size() < 2 || !inputs[1]) {
            throw std::invalid_argument(
                "BatchNorm (buffer backend): the merged parameter tensor "
                "(input 1) is required; run the converter's batchnorm merge");
        }

        dispatch_by_dtype(outputs[0]->dtype(), [&](auto dummy) {
            using T = decltype(dummy);
            auto output = core::as_tensor<T>(outputs[0]);
            if (output->num_elements() != total_elems(in_shape)) {
                output->resize(in_shape);
            }
            bind_ssbo<T>(outputs[0], /*is_output=*/true);
        });
        dispatch_by_dtype(inputs[0]->dtype(), [&](auto dummy) {
            using T = decltype(dummy);
            bind_ssbo<T>(inputs[0], /*is_output=*/false);
        });
        dispatch_by_dtype(inputs[1]->dtype(), [&](auto dummy) {
            using T = decltype(dummy);
            bind_ssbo<T>(inputs[1], /*is_output=*/false);
        });

        batchnorm::BatchNormPC pc{};
        pc.outer = outer;
        pc.channels = channels;
        pc.spatial = spatial;
        pc.params_fp16 = (inputs[1]->dtype() == typeid(uint16_t)) ? 1 : 0;
        // Single module for both layouts: the packed-half flag follows the
        // tensor actually bound, not the constructor's precision hint.
        const bool data_fp16 = inputs[0]->dtype() == typeid(uint16_t);
        pc.fp16 = data_fp16 ? 1 : 0;
        pc.activation = static_cast<int>(activation_);
        pc.eps = eps_;

        int total = total_elems(in_shape);
        int nthreads = data_fp16 ? (total + 1) / 2 : total;
        submit(&pc, UP_DIV(nthreads, 256), 1, 1);
    }

    float eps_ = 1e-5F;
    conv2d::ActivationMode activation_ = conv2d::ActivationMode::NONE;
};

// PIMPL façade: buffer SSBO impl when backend_buffer is set, else image.
class BatchNorm : public PimplFacade {
  public:
    BatchNorm(int fp16, bool backend_buffer)
        : PimplFacade(OpType::BATCHNORM) {
        if (backend_buffer) {
            impl_ = std::make_unique<BatchNormBuffer>(fp16);
        } else {
            impl_ = std::make_unique<BatchNormImage>();
        }
    }
};

} // namespace ops
} // namespace vkop
#endif // OPS_BATCHNORM2D_HPP_
