// Copyright 2026 @junka
#ifndef OPS_EXPAND_HPP_
#define OPS_EXPAND_HPP_

#include "core/Tensor.hpp"
#include "ops/Operator.hpp"
#include "ops/PimplFacade.hpp"
#include <cmath>
#include <numeric>

extern "C" {
extern unsigned char image_expand_spv[];
extern unsigned int image_expand_spv_len;
}
namespace vkop {
namespace ops {
namespace expand {
struct GpuExpandParam {
    ivec4 inshape;
    uint32_t shape_length;
    int fp16;
};
} // namespace expand

class ExpandImage : public Operator {
  public:
    explicit ExpandImage()
        : Operator(OpType::EXPAND, image_expand_spv, image_expand_spv_len,
                   {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    VK_DESCRIPTOR_TYPE_STORAGE_BUFFER},
                   sizeof(expand::GpuExpandParam)) {
        param_.fp16 = 0;
    }

  private:
    void execute(
        const std::vector<std::shared_ptr<core::ITensor>> &inputs,
        const std::vector<std::shared_ptr<core::ITensor>> &outputs) override {

        std::vector<int> inshape = inputs[0]->getShape();
        std::vector<int> out_shape = outputs[0]->getShape();
        if (out_shape.size() == 0) {
            auto input = core::as_tensor<int>(inputs[1]);
            input->copyToCPU(m_cmdpool_);
            auto num = input->size();
            out_shape.resize(num);
            for (int i = 0; i < num; ++i) {
                out_shape[i] = static_cast<int>(input->data()[i]);
            }
            input->copyToGPU(m_cmdpool_);
        }
        dispatch_by_dtype(outputs[0]->dtype(), [&](auto dummy) {
            using T = decltype(dummy);
            auto output = core::as_tensor<T>(outputs[0]);
            if (output->size() == 0) {
                output->resize(out_shape);
            }
            auto output_buffer = output->as_storage_buffer(m_dev_, m_cmd_);
            objs_.emplace_back(output_buffer);
            if (typeid(uint16_t) == typeid(T)) {
                param_.fp16 = 1;
            }
        });

        dispatch_by_dtype(inputs[0]->dtype(), [&](auto dummy) {
            using T = decltype(dummy);
            auto input = core::as_tensor<T>(inputs[0]);
            auto input_buffer = input->as_storage_buffer(m_dev_, m_cmd_);
            objs_.emplace_back(input_buffer);
        });
        auto shapeinput = core::as_tensor<int>(inputs[1]);
        auto input_buffer = shapeinput->as_storage_buffer(m_dev_, m_cmd_);
        objs_.emplace_back(input_buffer);

        auto total_size = std::accumulate(out_shape.begin(), out_shape.end(), 1,
                                          std::multiplies<>());
        for (size_t i = 0; i < out_shape.size(); ++i) {
            if (static_cast<int>(inshape.size() - i - 1) >= 0) {
                param_.inshape[out_shape.size() - i - 1] =
                    inshape[inshape.size() - i - 1];
            } else {
                param_.inshape[out_shape.size() - i - 1] = 1;
            }
        }
        param_.shape_length = static_cast<uint32_t>(out_shape.size());
        submit(&param_, UP_DIV(total_size, 256), 1, 1);
    }

    expand::GpuExpandParam param_;
};
// PIMPL façade: buffer SSBO impl when backend_buffer is set, else image.
class Expand : public PimplFacade {
  public:
    Expand(int /*fp16*/, bool /*backend_buffer*/)
        : PimplFacade(OpType::EXPAND) {
        /* backend_buffer unused: already SSBO */
        // Already an SSBO impl (descriptor + tensor + shader all use storage
        // buffers); no image path exists for this op.
        impl_ = std::make_unique<ExpandImage>();
    }
};

} // namespace ops
} // namespace vkop
#endif // OPS_EXPAND_HPP_
