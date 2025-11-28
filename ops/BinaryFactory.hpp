// Copyright 2025 @junka
#ifndef OPS_BINARY_FACTORY_HPP_
#define OPS_BINARY_FACTORY_HPP_

#include "Operator.hpp"

namespace vkop {
namespace ops {

class BinaryFactory : public Operator {
  public:
    explicit BinaryFactory(OpType type, uint8_t *spv, uint32_t spv_len)
        : Operator(type, spv, spv_len, 0) {
        n_imgs_ = 3;
        types_ = {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                  VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                  VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER};
        objs_.reserve(types_.size());
    }

  private:
    void execute(
        const std::vector<std::shared_ptr<core::ITensor>> &inputs,
        const std::vector<std::shared_ptr<core::ITensor>> &outputs) override {
        std::vector<int> input_shape = inputs[0]->getShape();
        dispatch_by_dtype(outputs[0]->dtype(), [&](auto t) {
            using T = decltype(t);
            auto outputptr = core::as_tensor<T>(outputs[0]);
            if (outputptr->size() == 0) {
                outputptr->resize(input_shape);
            }
            auto output_image = outputptr->as_output_image(m_dev_, m_cmd_);
            objs_.emplace_back(output_image);
        });

        for (size_t i = 0; i <= 1; ++i) {
            dispatch_by_dtype(inputs[i]->dtype(), [&](auto t) {
                using T = decltype(t);
                auto inputptr = core::as_tensor<T>(inputs[i]);
                auto input_image = inputptr->as_input_image(m_dev_, m_cmd_);
                objs_.emplace_back(input_image);
            });
        }
        int batch = input_shape[0];
        int depth = input_shape[1];
        int out_height = input_shape[2];
        int out_width = input_shape[3];

        int realheight = out_height * batch;
        submit(nullptr, UP_DIV(out_width, 16), UP_DIV(realheight, 16),
               UP_DIV(depth, 4));
    }
};

} // namespace ops
} // namespace vkop
#endif // OPS_ELEMENT_WISE_FACTORY_HPP_
