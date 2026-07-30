// Copyright 2025 @junka
#ifndef OPS_LAYERNORM_HPP_
#define OPS_LAYERNORM_HPP_

#include "Operator.hpp"
#include "ops/BufferBase.hpp"
#include "ops/PimplFacade.hpp"

#include <memory>
extern "C" {
extern unsigned char image_layernorm_spv[];
extern unsigned int image_layernorm_spv_len;
extern unsigned char buffer_layernorm_spv[];
extern unsigned int buffer_layernorm_spv_len;
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
// (inner_size elements); weight & bias are SSBOs. fp32-only.
class LayerNormBuffer : public BufferFactory {
  public:
    explicit LayerNormBuffer(int /*fp16*/)
        : BufferFactory(OpType::LAYERNORM, buffer_layernorm_spv,
                        buffer_layernorm_spv_len,
                        {DESCRIPTOR_TYPE_STORAGE, DESCRIPTOR_TYPE_STORAGE,
                         DESCRIPTOR_TYPE_STORAGE, DESCRIPTOR_TYPE_STORAGE},
                        sizeof(LayerNormPC)) {}

    void setAttribute(const std::unordered_map<std::string, std::string>
                          &attributes) override {
        if (attributes.find("eps") != attributes.end()) {
            eps_ = std::stof(attributes.at("eps"));
        }
        if (attributes.find("normalized_shape") != attributes.end()) {
            normalized_shape_ =
                parse_attr_list<int>(attributes.at("normalized_shape"));
        }
    }

  private:
    void execute(
        const std::vector<std::shared_ptr<core::ITensor>> &inputs,
        const std::vector<std::shared_ptr<core::ITensor>> &outputs) override {
        auto shape = inputs[0]->getShape();
        int total = total_elems(shape);
        int inner_size = 1;
        for (int d : normalized_shape_) {
            inner_size *= d;
        }
        int outer_size = total / inner_size;

        dispatch_by_dtype(outputs[0]->dtype(), [&](auto dummy) {
            using T = decltype(dummy);
            auto output = core::as_tensor<T>(outputs[0]);
            if (output->size() == 0) {
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
};

// PIMPL façade: buffer SSBO impl when backend_buffer is set, else image.
class LayerNorm : public PimplFacade {
  public:
    LayerNorm(int /*fp16*/, bool backend_buffer)
        : PimplFacade(OpType::LAYERNORM) {
        impl_ = backend_buffer ? std::unique_ptr<Operator>(
                                     std::make_unique<LayerNormBuffer>(0))
                               : std::make_unique<LayerNormImage>();
    }
};

} // namespace ops
} // namespace vkop
#endif // OPS_LAYERNORM_HPP_
