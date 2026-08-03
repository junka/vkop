// Copyright 2026 @junka
#ifndef OPS_WHERE_HPP_
#define OPS_WHERE_HPP_

#include "core/Tensor.hpp"
#include "ops/Operator.hpp"
#include "ops/PimplFacade.hpp"
#include <cmath>
#include <numeric>

extern "C" {
extern unsigned char image_where_spv[];
extern unsigned int image_where_spv_len;
}
namespace vkop {
namespace ops {
namespace where {} // namespace where

class WhereImage : public Operator {
  public:
    explicit WhereImage()
        : Operator(OpType::WHERE, image_where_spv, image_where_spv_len,
                   {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    VK_DESCRIPTOR_TYPE_STORAGE_BUFFER},
                   0) {}

  private:
    void execute(
        const std::vector<std::shared_ptr<core::ITensor>> &inputs,
        const std::vector<std::shared_ptr<core::ITensor>> &outputs) override {

        std::vector<int> out_shape = outputs[0]->getShape();
        if (out_shape.empty()) {
            // all tensor contains scalar value
            auto inshape = inputs[0]->getShape();
            out_shape = inshape;
        }

        dispatch_by_dtype(outputs[0]->dtype(), [&](auto dummy) {
            using T = decltype(dummy);
            auto output = core::as_tensor<T>(outputs[0]);
            if (output->size() == 0) {
                output->resize(out_shape);
            }
            auto output_buffer = output->as_storage_buffer(m_dev_, m_cmd_);
            objs_.emplace_back(output_buffer);
        });

        for (const auto &in : inputs) {
            dispatch_by_dtype(in->dtype(), [&](auto dummy) {
                using T = decltype(dummy);
                auto input = core::as_tensor<T>(in);
                auto input_buffer = input->as_storage_buffer(m_dev_, m_cmd_);
                objs_.emplace_back(input_buffer);
            });
        }

        auto total_size = std::accumulate(out_shape.begin(), out_shape.end(), 1,
                                          std::multiplies<>());
        submit(nullptr, UP_DIV(total_size, 256), 1, 1);
    }
};
// PIMPL façade: buffer SSBO impl when backend_buffer is set, else image.
class Where : public PimplFacade {
  public:
    Where(int /*fp16*/, bool /*backend_buffer*/) : PimplFacade(OpType::WHERE) {
        /* backend_buffer unused: already SSBO */
        // Already an SSBO impl (descriptor + tensor + shader all use storage
        // buffers); no image path exists for this op.
        impl_ = std::make_unique<WhereImage>();
    }
};

} // namespace ops
} // namespace vkop
#endif // OPS_WHERE_HPP_
