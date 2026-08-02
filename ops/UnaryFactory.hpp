// Copyright 2025 @junka
#ifndef OPS_UNARY_FACTORY_HPP_
#define OPS_UNARY_FACTORY_HPP_

#include "ops/Operator.hpp"

namespace vkop {
namespace ops {

class UnaryFactory : public Operator {
  public:
    explicit UnaryFactory(OpType type, uint8_t *spv, uint32_t spv_len)
        : Operator(type, spv, spv_len,
                   {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                    VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER},
                   0) {};

  private:
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
            objs_.emplace_back(output_image);
        });
        dispatch_by_dtype(inputs[0]->dtype(), [&](auto dummy) {
            using T = decltype(dummy);
            auto input = core::as_tensor<T>(inputs[0]);
            auto input_image = input->as_input_image(m_dev_, m_cmd_);
            objs_.emplace_back(input_image);
        });

        auto out_gpu_shape = inputs[0]->getGPUShape();
        submit(nullptr, UP_DIV(out_gpu_shape[0], 16),
               UP_DIV(out_gpu_shape[1], 16), out_gpu_shape[2]);
    }
};

} // namespace ops
} // namespace vkop
#endif // OPS_UNARY_FACTORY_HPP_
