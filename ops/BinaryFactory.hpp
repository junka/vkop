// Copyright 2025 @junka
#ifndef OPS_BINARY_FACTORY_HPP_
#define OPS_BINARY_FACTORY_HPP_

#include "Operator.hpp"

#include "core/Tensor.hpp"
#include "include/logger.hpp"
#include "ops/Ops.hpp"
#include "vulkan/VulkanBuffer.hpp"
#include "vulkan/VulkanCommandBuffer.hpp"
#include "vulkan/VulkanImage.hpp"
#include "vulkan/VulkanPipeline.hpp"
#include "vulkan/VulkanQueryPool.hpp"

namespace vkop {
namespace ops {

class BinaryFactory : public Operator {
  public:
    explicit BinaryFactory(OpType type) : Operator(type) {}

    template <typename T>
    void prepare(std::vector<std::shared_ptr<core::ITensor>> inputs,
                 std::vector<std::shared_ptr<core::ITensor>> outputs) {
        auto input_a = core::as_tensor<T>(inputs[0]);
        auto output = core::as_tensor<T>(outputs[0]);
        auto input_b = core::as_tensor<T>(inputs[1]);
        auto input_shape = input_a->getShape();
        if (output->size() == 0) {
            output->resize(input_shape);
        }

        auto inputa_image = input_a->as_input_image(m_dev_, m_cmdpool_);
        auto inputb_image = input_b->as_input_image(m_dev_, m_cmdpool_);
        auto output_image = output->as_output_image(m_dev_, m_cmdpool_);

        types_ = {output_image->getDescriptorType(),
                  inputa_image->getDescriptorType(),
                  inputb_image->getDescriptorType()};
        objs_ = {output_image, inputa_image, inputb_image};
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
        std::vector<int> input_shape;
        if (inputs[0]->dtype() == typeid(float)) {
            auto input_a = core::as_tensor<float>(inputs[0]);
            auto output = core::as_tensor<float>(outputs[0]);
            input_shape = input_a->getShape();
        } else if (inputs[0]->dtype() == typeid(uint16_t)) {
            auto input_a = core::as_tensor<uint16_t>(inputs[0]);
            auto output = core::as_tensor<uint16_t>(outputs[0]);
            input_shape = input_a->getShape();
        } else {
            LOG_ERROR("Unsupported data type");
        }
        int batch = input_shape[0];
        int depth = input_shape[1];
        int out_height = input_shape[2];
        int out_width = input_shape[3];

        int realwidth = out_width * UP_DIV(depth, 4);
        int realheight = out_height * batch;
        submit(nullptr, 0, spv_, spv_len_, UP_DIV(realwidth, 16),
               UP_DIV(realheight, 16));
    }

  protected:
    void set_vulkan_spv(unsigned char *spv, unsigned int spv_len) {
        spv_ = spv;
        spv_len_ = spv_len;
    }

  private:
    unsigned char *spv_;
    unsigned int spv_len_;
};

} // namespace ops
} // namespace vkop
#endif // OPS_ELEMENT_WISE_FACTORY_HPP_
