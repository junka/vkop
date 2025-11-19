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

struct GpuLayerNormParam {
    ivec4 outShape;
    ivec4 normalizedShape;
    float eps; // default 1e-5
    int normalizedDim;
    int innerSize;
};
} // namespace layernorm

class LayerNorm : public Operator {
  public:
    LayerNorm() : Operator(OpType::LAYERNORM) {}
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
    template <typename T>
    void prepare(std::vector<std::shared_ptr<core::ITensor>> inputs,
                 std::vector<std::shared_ptr<core::ITensor>> outputs) {
        auto input = core::as_tensor<T>(inputs[0]);
        auto output = core::as_tensor<T>(outputs[0]);
        auto weight =
            (inputs.size() > 1) ? core::as_tensor<T>(inputs[1]) : nullptr;
        auto bias =
            (inputs.size() > 2) ? core::as_tensor<T>(inputs[2]) : nullptr;

        auto input_shape = input->getShape();
        if (output->size() == 0) {
            output->resize(input->getShape());
        }

        auto input_image = input->as_input_image(m_dev_, m_cmdpool_);
        auto output_image = output->as_output_image(m_dev_, m_cmdpool_);

        weightBuffer_ = weight->as_storage_buffer(m_dev_);
        biasBuffer_ = bias->as_storage_buffer(m_dev_);

        paramBuffer_ = std::make_shared<VulkanBuffer>(
            m_dev_, sizeof(layernorm::GpuLayerNormParam),
            VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

        types_ = {output_image->getDescriptorType(),
                  input_image->getDescriptorType(),
                  weightBuffer_->getDescriptorType(),
                  biasBuffer_->getDescriptorType(),
                  paramBuffer_->getDescriptorType()};
        objs_ = {output_image, input_image, weightBuffer_, biasBuffer_,
                 paramBuffer_};
    }

    void
    apply(const std::vector<std::shared_ptr<core::ITensor>> &inputs,
          const std::vector<std::shared_ptr<core::ITensor>> &outputs) override {
        if (inputs[0]->dtype() == typeid(float)) {
            prepare<float>(inputs, outputs);
        } else if (inputs[0]->dtype() == typeid(uint16_t)) {
            prepare<uint16_t>(inputs, outputs);
        } else {
            LOG_ERROR("Unsupported data type");
        }
    }

    void execute(
        const std::vector<std::shared_ptr<core::ITensor>> &inputs,
        const std::vector<std::shared_ptr<core::ITensor>> &outputs) override {
        if (inputs[0]->dtype() == typeid(float)) {
            auto input = core::as_tensor<float>(inputs[0]);
            auto output = core::as_tensor<float>(outputs[0]);

            auto weight = (inputs.size() > 1)
                              ? core::as_tensor<float>(inputs[1])
                              : nullptr;
            auto bias = (inputs.size() > 2) ? core::as_tensor<float>(inputs[2])
                                            : nullptr;

            auto input_shape = input->getShape();
            int batch = input_shape[0];
            int depth = input_shape[1];
            int out_height = input_shape[2];
            int out_width = input_shape[3];

            int realwidth = out_width * UP_DIV(depth, 4);
            int realheight = out_height * batch;

            auto *para = static_cast<layernorm::GpuLayerNormParam *>(
                paramBuffer_->getMappedMemory());
            para->eps = eps_;
            para->outShape[0] = batch;
            para->outShape[1] = depth;
            para->outShape[2] = out_height;
            para->outShape[3] = out_width;
            para->normalizedDim = normalized_shape_.size();
            para->innerSize = 1;
            for (size_t i = 0; i < normalized_shape_.size(); i++) {
                para->normalizedShape[i] = normalized_shape_[i];
                para->innerSize *= normalized_shape_[i];
            }
            paramBuffer_->unmapMemory();
            weight->copyToGPU(m_dev_, m_cmdpool_);
            bias->copyToGPU(m_dev_, m_cmdpool_);

            if (normalized_shape_.size() == 1) { // 归一化最后一个维度 W
                submit(layernorm_spv, layernorm_spv_len,
                       batch * UP_DIV(depth, 4), out_height);
            } else if (normalized_shape_.size() == 2) { // 归一化最后两个维度 HW
                submit(layernorm_spv, layernorm_spv_len, batch,
                       UP_DIV(depth, 4));
            } else { // 归一化所有维度 CHW
                submit(layernorm_spv, layernorm_spv_len, realwidth, realheight);
            }
        }
    }

  private:
    float eps_ = 1e-5;
    std::vector<int> normalized_shape_;
    std::shared_ptr<VulkanBuffer> weightBuffer_;
    std::shared_ptr<VulkanBuffer> biasBuffer_;
    std::shared_ptr<VulkanBuffer> paramBuffer_;
};

} // namespace ops
} // namespace vkop
#endif // OPS_LAYERNORM_HPP_
