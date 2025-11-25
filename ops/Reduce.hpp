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
namespace resize {
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
} // namespace resize
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
                 std::vector<std::shared_ptr<core::ITensor>> outputs) {}

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
            auto output_buffer = outputptr->as_storage_buffer(m_dev_);
            types_.emplace_back(output_buffer->getDescriptorType());
            objs_.emplace_back(output_buffer);
        });
        dispatch_by_dtype(inputs[0]->dtype(), [&](auto t) {
            using T = decltype(t);
            auto inputptr = core::as_tensor<T>(inputs[0]);
            auto input_buffer = inputptr->as_storage_buffer(m_dev_);
            types_.emplace_back(input_buffer->getDescriptorType());
            objs_.emplace_back(input_buffer);
        });

        int h = input_shape[0];
        int w = input_shape[1];

        resize::GpuReduceParam para;
        para.H = h;
        para.W = w;
        para.norm_type = norm_type_;
        para.keepdims = keepdims_;
        para.noop_with_empty_axes = noop_with_empty_axes_;

        submit(&para, sizeof(resize::GpuReduceParam), reduce_spv,
               reduce_spv_len, UP_DIV(h, 16), UP_DIV(w, 16));
    }

  private:
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
