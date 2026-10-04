// Copyright 2025 @junka
#ifndef OPS_LAYERNORM_HPP_
#define OPS_LAYERNORM_HPP_

#include "Operator.hpp"
#include "ops/BufferBase.hpp"
#include "ops/PimplFacade.hpp"

#include <memory>
#include <stdexcept>
extern "C" {
extern unsigned char image_layernorm_spv[];
extern unsigned int image_layernorm_spv_len;
extern unsigned char buffer_layernorm_spv[];
extern unsigned int buffer_layernorm_spv_len;
extern unsigned char buffer_layernorm_fp16_spv[];
extern unsigned int buffer_layernorm_fp16_spv_len;
}
namespace vkop {
namespace ops {
namespace layernorm {

// torch.nn.functional.layer_norm(input, normalized_shape, weight=None,
// bias=None, eps=1e-05)

struct alignas(16) GpuLayerNormParam {
    ivec4 outShape;
    ivec4 normalizedShape;
    float eps; // default 1e-5
    int normalizedDim;
    int innerSize;
};
} // namespace layernorm

class LayerNormImage : public Operator {
  public:
    LayerNormImage()
        : Operator(OpType::LAYERNORM, image_layernorm_spv,
                   image_layernorm_spv_len,
                   {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                    VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                    DESCRIPTOR_TYPE_STORAGE, DESCRIPTOR_TYPE_STORAGE},
                   sizeof(layernorm::GpuLayerNormParam)) {}

    void setAttribute(const std::unordered_map<std::string, std::string>
                          &attributes) override {
        if (attributes.find("eps") != attributes.end()) {
            eps_ = std::stof(attributes.at("eps"));
        }
        if (attributes.find("normalized_shape") != attributes.end()) {
            std::string norm_shape_str = attributes.at("normalized_shape");
            normalized_shape_ = parse_attr_list<int>(norm_shape_str);
        }
    }

  private:
    void execute(
        const std::vector<std::shared_ptr<core::ITensor>> &inputs,
        const std::vector<std::shared_ptr<core::ITensor>> &outputs) override {
        auto input_shape = inputs[0]->getShape();
        // Unknown rank (empty shape from the converter): fall back to the
        // output tensor's live shape, which the producer already set correctly.
        // This avoids OOB reads on input_shape[0..3] below.
        if (input_shape.empty()) {
            dispatch_by_dtype(outputs[0]->dtype(), [&](auto t) {
                using T = decltype(t);
                input_shape = core::as_tensor<T>(outputs[0])->getShape();
            });
        }
        dispatch_by_dtype(outputs[0]->dtype(), [&](auto t) {
            using T = decltype(t);
            auto outputptr = core::as_tensor<T>(outputs[0]);
            if (outputptr->num_elements() != total_elems(input_shape)) {
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
        for (size_t i = 1; i <= 2; ++i) {
            dispatch_by_dtype(inputs[i]->dtype(), [&](auto t) {
                using T = decltype(t);
                auto tensor = core::as_tensor<T>(inputs[i]);
                auto buffer = tensor->as_storage_buffer(m_dev_);
                objs_.emplace_back(buffer);
            });
        }
        int batch = input_shape[0];
        int depth = input_shape[1];
        int out_height = input_shape[2];
        int out_width = input_shape[3];

        layernorm::GpuLayerNormParam para;
        para.eps = eps_;
        para.outShape[0] = batch;
        para.outShape[1] = depth;
        para.outShape[2] = out_height;
        para.outShape[3] = out_width;
        para.normalizedDim = static_cast<int>(normalized_shape_.size());
        para.innerSize = 1;
        for (size_t i = 0; i < normalized_shape_.size(); i++) {
            para.normalizedShape[i] = normalized_shape_[i];
            para.innerSize *= normalized_shape_[i];
        }

        if (normalized_shape_.size() == 1) {
            submit(&para, batch, out_height, UP_DIV(depth, 4));
        } else if (normalized_shape_.size() == 2) {
            submit(&para, batch, 1, UP_DIV(depth, 4));
        } else {
            submit(&para, batch, 1, 1);
        }
    }

    float eps_ = 1e-5;
    std::vector<int> normalized_shape_;
};

// Buffer (SSBO) LayerNorm. Normalizes over the trailing normalized_shape
// (inner_size elements); weight & bias are SSBOs. fp32 and fp16 are separate
// shader builds — the fp16 one reads/writes packed half2 words in place. The
// previous fp16 route upcast on the host, which read back every activation:
// 65 LayerNorms per DiT decode step, 7.1 s of a 22 s step.
class LayerNormBuffer : public BufferFactory {
  public:
    explicit LayerNormBuffer(int fp16)
        : BufferFactory(OpType::LAYERNORM,
                        fp16 ? buffer_layernorm_fp16_spv : buffer_layernorm_spv,
                        fp16 ? buffer_layernorm_fp16_spv_len
                             : buffer_layernorm_spv_len,
                        {DESCRIPTOR_TYPE_STORAGE, DESCRIPTOR_TYPE_STORAGE,
                         DESCRIPTOR_TYPE_STORAGE, DESCRIPTOR_TYPE_STORAGE},
                        sizeof(LayerNormPC), fp16),
          fp16_(fp16) {}

    void setAttribute(const std::unordered_map<std::string, std::string>
                          &attributes) override {
        if (attributes.find("eps") != attributes.end()) {
            eps_ = std::stof(attributes.at("eps"));
        }
        if (attributes.find("epsilon") != attributes.end()) {
            eps_ = std::stof(attributes.at("epsilon"));
        }
        if (attributes.find("normalized_shape") != attributes.end()) {
            normalized_shape_ =
                parse_attr_list<int>(attributes.at("normalized_shape"));
        }
        if (attributes.find("axis") != attributes.end()) {
            axis_ = std::stol(attributes.at("axis"));
        }
    }

  private:
    void execute(
        const std::vector<std::shared_ptr<core::ITensor>> &inputs,
        const std::vector<std::shared_ptr<core::ITensor>> &outputs) override {
        auto shape = inputs[0]->getShape();
        int total = total_elems(shape);
        int inner_size = 1;
        if (!normalized_shape_.empty()) {
            // Explicit normalized_shape (ONNX opset<17 style): product of the
            // trailing dims to normalize over.
            for (int d : normalized_shape_) {
                inner_size *= d;
            }
        } else {
            // ONNX LayerNormalization (opset 17+) with axis instead of
            // normalized_shape: normalize over dims [axis, rank). The scale
            // (inputs[1]) carries the normalized shape, so its element count
            // is the inner size. This is the form Qwen3-VL's visual encoder
            // uses (axis=-1, scale [1024]).
            int rank = static_cast<int>(shape.size());
            int ax = axis_;
            if (ax < 0)
                ax += rank;
            inner_size = 1;
            for (int d = ax; d < rank; ++d) {
                inner_size *= shape[d];
            }
        }
        int outer_size = total / inner_size;

        // The shader build is fixed at construction, so the live dtype has to
        // match it: packed half2 read as float bits (or the reverse) is silent
        // garbage.
        if ((inputs[0]->dtype() == typeid(uint16_t)) != (fp16_ != 0)) {
            throw std::runtime_error(
                "LayerNorm: input dtype disagrees with the built fp16 variant");
        }
        if (fp16_) {
            // inner_size odd would start a slice mid-word, so the slice's two
            // end halves would live in a neighbour's word.
            if (inner_size % 2 != 0) {
                throw std::runtime_error(
                    "LayerNorm fp16: odd inner_size straddles packed half2 "
                    "words");
            }
        }
        // Both builds read weight/bias as words of x's own dtype.
        if (inputs[1]->dtype() != inputs[0]->dtype() ||
            inputs[2]->dtype() != inputs[0]->dtype()) {
            throw std::runtime_error(
                "LayerNorm: weight/bias dtype differs from the input");
        }

        dispatch_by_dtype(outputs[0]->dtype(), [&](auto dummy) {
            using T = decltype(dummy);
            auto output = core::as_tensor<T>(outputs[0]);
            if (output->num_elements() != total_elems(shape)) {
                output->resize(shape);
            }
            bind_ssbo<T>(outputs[0], /*is_output=*/true);
        });
        dispatch_by_dtype(inputs[0]->dtype(), [&](auto dummy) {
            using T = decltype(dummy);
            bind_ssbo<T>(inputs[0], /*is_output=*/false);
        });
        for (size_t i = 1; i <= 2; ++i) {
            dispatch_by_dtype(inputs[i]->dtype(), [&](auto dummy) {
                using T = decltype(dummy);
                bind_ssbo<T>(inputs[i], /*is_output=*/false);
            });
        }

        LayerNormPC pc{};
        pc.eps = eps_;
        pc.inner_size = inner_size;
        pc.outer_size = outer_size;
        submit(&pc, outer_size, 1, 1);
    }

    float eps_ = 1e-5f;
    std::vector<int> normalized_shape_;
    int axis_ = -1;
    int fp16_ = 0;
};

// PIMPL façade: buffer SSBO impl when backend_buffer is set, else image.
class LayerNorm : public PimplFacade {
  public:
    LayerNorm(int fp16, bool backend_buffer) : PimplFacade(OpType::LAYERNORM) {
        impl_ = backend_buffer ? std::unique_ptr<Operator>(
                                     std::make_unique<LayerNormBuffer>(fp16))
                               : std::make_unique<LayerNormImage>();
    }
};

} // namespace ops
} // namespace vkop
#endif // OPS_LAYERNORM_HPP_
