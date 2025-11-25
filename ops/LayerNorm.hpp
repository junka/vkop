// Copyright 2025 @junka
#ifndef OPS_LAYERNORM_HPP_
#define OPS_LAYERNORM_HPP_

#include "Operator.hpp"

#include "core/Tensor.hpp"
#include "include/logger.hpp"
#include "vulkan/VulkanBuffer.hpp"
#include "vulkan/VulkanCommandBuffer.hpp"
#include "vulkan/VulkanImage.hpp"
#include "vulkan/VulkanPipeline.hpp"
#include "vulkan/VulkanQueryPool.hpp"

#include <memory>

extern unsigned char layernorm_spv[];
extern unsigned int layernorm_spv_len;

namespace vkop {
namespace ops {
namespace layernorm {

using ivec4 = int[4];

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

class LayerNorm : public Operator {
  public:
    LayerNorm()
        : Operator(OpType::LAYERNORM, layernorm_spv, layernorm_spv_len,
                   sizeof(layernorm::GpuLayerNormParam)) {
        types_ = {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                  VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                  VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                  VK_DESCRIPTOR_TYPE_STORAGE_BUFFER};
    }
    void setAttribute(const std::unordered_map<std::string, std::string>
                          &attributes) override {
        if (attributes.find("eps") != attributes.end()) {
            eps_ = std::stof(attributes.at("eps"));
        }
        if (attributes.find("normalized_shape") != attributes.end()) {
            std::string norm_shape_str = attributes.at("normalized_shape");
            normalized_shape_ = parse_attr_list(norm_shape_str);
        }
    }

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
            // types_.emplace_back(output_image->getDescriptorType());
            objs_.emplace_back(output_image);
        });

        dispatch_by_dtype(inputs[0]->dtype(), [&](auto t) {
            using T = decltype(t);
            auto inputptr = core::as_tensor<T>(inputs[0]);
            auto input_image = inputptr->as_input_image(m_dev_, m_cmd_);
            // types_.emplace_back(input_image->getDescriptorType());
            objs_.emplace_back(input_image);
        });
        for (size_t i = 1; i <= 2; ++i) {
            dispatch_by_dtype(inputs[i]->dtype(), [&](auto t) {
                using T = decltype(t);
                auto tensor = core::as_tensor<T>(inputs[i]);
                auto buffer = tensor->as_storage_buffer(m_dev_);
                tensor->copyToGPU(m_dev_, m_cmdpool_);
                // types_.emplace_back(buffer->getDescriptorType());
                objs_.emplace_back(buffer);
            });
        }
        int batch = input_shape[0];
        int depth = input_shape[1];
        int out_height = input_shape[2];
        int out_width = input_shape[3];

        int realwidth = out_width * UP_DIV(depth, 4);
        int realheight = out_height * batch;

        layernorm::GpuLayerNormParam para;
        para.eps = eps_;
        para.outShape[0] = batch;
        para.outShape[1] = depth;
        para.outShape[2] = out_height;
        para.outShape[3] = out_width;
        para.normalizedDim = normalized_shape_.size();
        para.innerSize = 1;
        for (size_t i = 0; i < normalized_shape_.size(); i++) {
            para.normalizedShape[i] = normalized_shape_[i];
            para.innerSize *= normalized_shape_[i];
        }

        if (normalized_shape_.size() == 1) { // 归一化最后一个维度 W
            submit(&para, batch * UP_DIV(depth, 4), out_height);
        } else if (normalized_shape_.size() == 2) { // 归一化最后两个维度 HW
            submit(&para, batch, UP_DIV(depth, 4));
        } else { // 归一化所有维度 CHW
            submit(&para, realwidth, realheight);
        }
    }

  private:
    float eps_ = 1e-5;
    std::vector<int> normalized_shape_;
};

} // namespace ops
} // namespace vkop
#endif // OPS_LAYERNORM_HPP_
