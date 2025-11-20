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
    explicit UnaryFactory(OpType type) : Operator(type){};

    template <typename T>
    void prepare(std::vector<std::shared_ptr<core::ITensor>> inputs,
                 std::vector<std::shared_ptr<core::ITensor>> outputs) {
        auto input = core::as_tensor<T>(inputs[0]);
        auto output = core::as_tensor<T>(outputs[0]);
        if (output->size() == 0) {
            output->resize(input->getShape());
        }

        auto input_image = input->as_input_image(m_dev_, m_cmdpool_);
        auto output_image = output->as_output_image(m_dev_, m_cmdpool_);

        types_ = {output_image->getDescriptorType(),
                  input_image->getDescriptorType()};
        objs_ = {output_image, input_image};
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

            auto input_shape = input->getShape();
            int batch = input_shape[0];
            int depth = input_shape[1];
            int out_height = input_shape[2];
            int out_width = input_shape[3];

            int realwidth = out_width * UP_DIV(depth, 4);
            int realheight = out_height * batch;

            submit(nullptr, 0, spv_, spv_len_, UP_DIV(realwidth, 16),
                   UP_DIV(realheight, 16));
        }
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
#endif // OPS_UNARY_FACTORY_HPP_
