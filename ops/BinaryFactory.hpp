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
        auto input_shape = input_a->getTensorShape();
        if (output->size() == 0) {
            output->resize(input_shape);
        }

        auto inputa_image = input_a->as_input_image(m_dev_, m_cmdpool_);
        auto inputb_image = input_b->as_input_image(m_dev_, m_cmdpool_);
        auto output_image = output->as_output_image(m_dev_, m_cmdpool_);

        inputImages_ = {inputa_image, inputb_image};
        outputImage_ = output_image;
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
            auto input_a = core::as_tensor<float>(inputs[0]);
            auto output = core::as_tensor<float>(outputs[0]);
            auto input_b = core::as_tensor<float>(inputs[1]);

            auto input_shape = input_a->getTensorShape();
            int batch = input_shape[0];
            int depth = input_shape[1];
            int out_height = input_shape[2];
            int out_width = input_shape[3];

            int realwidth = out_width * UP_DIV(depth, 4);
            int realheight = out_height * batch;
            submit(spv_, spv_len_, realwidth, realheight);
        } else if (inputs[0]->dtype() == typeid(uint16_t)) {
            auto input_a = core::as_tensor<uint16_t>(inputs[0]);
            auto output = core::as_tensor<uint16_t>(outputs[0]);
            auto input_b = core::as_tensor<uint16_t>(inputs[1]);

            auto input_shape = input_a->getTensorShape();
            int batch = input_shape[0];
            int depth = input_shape[1];
            int out_height = input_shape[2];
            int out_width = input_shape[3];

            int realwidth = out_width * UP_DIV(depth, 4);
            int realheight = out_height * batch;
            if (output->size() == 0) {
                output->resize(input_a->getTensorShape());
            }
            submit(spv_, spv_len_, realwidth, realheight);
        } else {
            LOG_ERROR("Unsupported data type");
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

    void submit(const unsigned char *spv, unsigned int spv_len, int out_width,
                int out_height) override {
        std::vector<VkDescriptorType> types = {
            VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
            VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER};
        std::vector<std::shared_ptr<VulkanResource>> objs = {
            outputImage_, inputImages_[0], inputImages_[1]};
        VkDevice device = m_dev_->getLogicalDevice();
        VulkanPipeline pipeline(device, types, objs,
                                reinterpret_cast<const uint32_t *>(spv),
                                spv_len);

        VulkanCommandBuffer cmd2(device, m_cmdpool_->getCommandPool());
#ifdef USE_MEASURE_TIME
        VulkanQueryPool query_pool(device, 2, VK_QUERY_TYPE_TIMESTAMP);
#endif
        cmd2.begin();
        cmd2.bind(pipeline);
#ifdef USE_MEASURE_TIME
        query_pool.begin(cmd2.get());
#endif
        cmd2.dispatch(UP_DIV(out_width, 16), UP_DIV(out_height, 16));
#ifdef USE_MEASURE_TIME
        query_pool.end(cmd2.get());
#endif
        cmd2.end();
        cmd2.submit(m_dev_->getComputeQueue());
#ifdef USE_MEASURE_TIME
        auto r = query_pool.getResults();
        LOG_INFO("Time: %f s", static_cast<double>(r[1] - r[0]) * (1e-9) *
                                   m_dev_->getTimestampPeriod());
#endif
    }
};

} // namespace ops
} // namespace vkop
#endif // OPS_ELEMENT_WISE_FACTORY_HPP_
