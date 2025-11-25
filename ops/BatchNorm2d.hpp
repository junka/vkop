// Copyright 2025 @junka
#ifndef OPS_BATCHNORM2D_HPP_
#define OPS_BATCHNORM2D_HPP_

#include "Operator.hpp"

#include "core/Tensor.hpp"
#include "include/logger.hpp"
#include "vulkan/VulkanBuffer.hpp"
#include "vulkan/VulkanCommandBuffer.hpp"
#include "vulkan/VulkanImage.hpp"
#include "vulkan/VulkanPipeline.hpp"
#include "vulkan/VulkanQueryPool.hpp"

#include <memory>

extern unsigned char batchnorm2d_spv[];
extern unsigned int batchnorm2d_spv_len;

namespace vkop {
namespace ops {
namespace batchnorm {

using ivec4 = int[4];

// torch.nn.functional.batch_norm(input, running_mean, running_var, weight=None,
//                                bias=None, training=False, momentum=0.1,
//                                eps=1e-05)
struct alignas(16) GpuBatchNormParam {
    ivec4 outShape;
    float eps;      // default 1e-5
    float momentum; // default 0.1
};
} // namespace batchnorm

class BatchNorm2d : public Operator {
  public:
    BatchNorm2d()
        : Operator(OpType::BATCHNORM, batchnorm2d_spv, batchnorm2d_spv_len,
                   sizeof(batchnorm::GpuBatchNormParam)) {
        types_ = {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                  VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                  VK_DESCRIPTOR_TYPE_STORAGE_BUFFER};
    }
    void setAttribute(const std::unordered_map<std::string, std::string>
                          &attributes) override {
        attributes.find("training") != attributes.end()
            ? training_ = (attributes.at("align_corners") == "1" ||
                           attributes.at("align_corners") == "true")
            : training_ = false;
        if (attributes.find("eps") != attributes.end()) {
            eps_ = std::stof(attributes.at("eps"));
        }
        if (attributes.find("momentum") != attributes.end()) {
            momentum_ = std::stof(attributes.at("momentum"));
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

        if (inputs[0]->dtype() == typeid(float)) {
            auto running_mean = core::as_tensor<float>(inputs[1]);
            auto running_var = core::as_tensor<float>(inputs[2]);

            auto weight = (inputs.size() > 3)
                              ? core::as_tensor<float>(inputs[3])
                              : nullptr;
            auto bias = (inputs.size() > 4) ? core::as_tensor<float>(inputs[4])
                                            : nullptr;

            int batch = input_shape[0];
            int depth = input_shape[1];
            int out_height = input_shape[2];
            int out_width = input_shape[3];

            int realwidth = out_width * UP_DIV(depth, 4);
            int realheight = out_height * batch;

            batchnorm::GpuBatchNormParam para;
            para.eps = eps_;
            para.momentum = momentum_;
            para.outShape[0] = batch;
            para.outShape[1] = depth;
            para.outShape[2] = out_height;
            para.outShape[3] = out_width;

            tensorBuffer_ = std::make_shared<VulkanBuffer>(
                m_dev_, running_mean->size() * 4,
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                    VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

            // types_.emplace_back(tensorBuffer_->getDescriptorType());
            objs_.push_back(tensorBuffer_);

            auto *var_buffer =
                static_cast<float *>(tensorBuffer_->getMappedMemory());
            for (int i = 0; i < running_mean->num_elements(); i++) {
                *(var_buffer + 4 * i) = (*running_mean)[i];
                *(var_buffer + 4 * i + 1) = (*running_var)[i];
                if (inputs.size() > 3) {
                    *(var_buffer + 4 * i + 2) = (*weight)[i];
                } else {
                    *(var_buffer + 4 * i + 2) = 1.0F;
                }
                if (inputs.size() > 4) {
                    *(var_buffer + 4 * i + 3) = (*bias)[i];
                } else {
                    *(var_buffer + 4 * i + 3) = 0.0F;
                }
            }
            tensorBuffer_->unmapMemory();

            submit(&para, UP_DIV(realwidth, 16), UP_DIV(realheight, 16));
        }
    }

  private:
    bool training_ = false;
    float momentum_ = 0.1;
    float eps_ = 1e-5;

    std::shared_ptr<VulkanBuffer> tensorBuffer_;
};

} // namespace ops
} // namespace vkop
#endif // OPS_BATCHNORM2D_HPP_
