// Copyright 2025 @junka
#ifndef OPS_UNARY_FACTORY_HPP_
#define OPS_UNARY_FACTORY_HPP_

#include "Operator.hpp"

#include "core/Tensor.hpp"
#include "include/logger.hpp"
#include "vulkan/VulkanBuffer.hpp"
#include "vulkan/VulkanCommandBuffer.hpp"
#include "vulkan/VulkanCommandPool.hpp"
#include "vulkan/VulkanDevice.hpp"
#include "vulkan/VulkanImage.hpp"
#include "vulkan/VulkanPipeline.hpp"
#include "vulkan/VulkanQueryPool.hpp"

namespace vkop {
namespace ops {

class UnaryFactory : public Operator {
  public:
    explicit UnaryFactory(OpType type, uint8_t *spv, uint32_t spv_len)
        : Operator(type, spv, spv_len, 0) {
        types_ = {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                  VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER};
    };

    void execute(
        const std::vector<std::shared_ptr<core::ITensor>> &inputs,
        const std::vector<std::shared_ptr<core::ITensor>> &outputs) override {
        dispatch_by_dtype(outputs[0]->dtype(), [&](auto dummy) {
            using T = decltype(dummy);
            auto output = core::as_tensor<T>(outputs[0]);
            if (output->size() == 0) {
                output->resize(inputs[0]->getShape());
            }
            auto output_image = output->as_output_image(m_dev_, m_cmd_);
            // types_.emplace_back(output_image->getDescriptorType());
            objs_.emplace_back(output_image);
        });
        dispatch_by_dtype(inputs[0]->dtype(), [&](auto dummy) {
            using T = decltype(dummy);
            auto input = core::as_tensor<T>(inputs[0]);
            auto input_image = input->as_input_image(m_dev_, m_cmd_);

            // types_.emplace_back(input_image->getDescriptorType());
            objs_.emplace_back(input_image);
        });

        auto input_shape = inputs[0]->getShape();
        int batch = input_shape[0];
        int depth = input_shape[1];
        int out_height = input_shape[2];
        int out_width = input_shape[3];

        int realwidth = out_width * UP_DIV(depth, 4);
        int realheight = out_height * batch;

        submit(nullptr, UP_DIV(realwidth, 16), UP_DIV(realheight, 16));
    }
};

} // namespace ops
} // namespace vkop
#endif // OPS_UNARY_FACTORY_HPP_
