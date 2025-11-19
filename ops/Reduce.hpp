// Copyright 2025 @junka
#ifndef OPS_REDUCE_HPP_
#define OPS_REDUCE_HPP_

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

extern unsigned char reduce_spv[];
extern unsigned int reduce_spv_len;

namespace vkop {
namespace ops {

enum class ReduceType {
    L1 = 0,
    L2,
    LOGSUM,
    LOGSUMEXP,
    MAX,
    MEAN,
    MIN,
    PROD,
    SUM,
    SUMSQUARE,
};

struct GpuReduceParam {
    int H;
    int W;
    int norm_type;
    int keepdims;
    int noop_with_empty_axes;
};

class Reduce : public Operator {
  public:
    Reduce() : Operator(OpType::REDUCE) {}

    void setAttribute(const std::unordered_map<std::string, std::string>
                          &attributes) override {
        if (attributes.find("norm_type") != attributes.end()) {
            norm_type_ = std::stoi(attributes.at("norm_type"));
        }
        // opset ver 18 will move this to inputs
        if (attributes.find("axis") != attributes.end()) {
            axis_ = parse_attr_list(attributes.at("axis"));
        }
        if (attributes.find("keepdims") != attributes.end()) {
            keepdims_ = std::stoi(attributes.at("keepdims"));
        }
        // only exist for opset ver 18
        if (attributes.find("noop_with_empty_axes") != attributes.end()) {
            noop_with_empty_axes_ =
                std::stoi(attributes.at("noop_with_empty_axes"));
        }
    }
    template <typename T>
    void prepare(std::vector<std::shared_ptr<core::ITensor>> inputs,
                 std::vector<std::shared_ptr<core::ITensor>> outputs) {
        auto input = core::as_tensor<T>(inputs[0]);
        auto output = core::as_tensor<T>(outputs[0]);
        if (output->size() == 0) {
            output->resize(input->getShape());
        }

        auto input_buffer = input->as_storage_buffer(m_dev_);
        auto output_buffer = output->as_storage_buffer(m_dev_);
        paramBuffer_ = std::make_shared<VulkanBuffer>(
            m_dev_, sizeof(GpuReduceParam), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

        types_ = {output_buffer->getDescriptorType(),
                  input_buffer->getDescriptorType(),
                  paramBuffer_->getDescriptorType()};
        objs_ = {output_buffer, input_buffer, paramBuffer_};
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
            int h = input_shape[0];
            int w = input_shape[1];

            auto *para = static_cast<struct GpuReduceParam *>(
                paramBuffer_->getMappedMemory());
            para->H = h;
            para->W = w;
            para->norm_type = norm_type_;
            para->keepdims = keepdims_;
            para->noop_with_empty_axes = noop_with_empty_axes_;
            paramBuffer_->unmapMemory();
            submit(reduce_spv, reduce_spv_len, UP_DIV(h, 16), UP_DIV(w, 16));
        }
    }

  private:
    std::shared_ptr<VulkanBuffer> paramBuffer_;
    int norm_type_ = 0;
    std::vector<int> axis_;
    // Keep the reduced dimension or not, default 1 means keep reduced
    // dimension.
    int keepdims_ = 1;
    // Defines behavior when axes is not provided or is empty.
    int noop_with_empty_axes_ = 0;
};

} // namespace ops
} // namespace vkop
#endif // OPS_REDUCE_HPP_
