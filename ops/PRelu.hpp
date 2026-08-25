// Copyright 2025 @junka
#ifndef OPS_PRELU_HPP_
#define OPS_PRELU_HPP_

#include "BinaryFactory.hpp"
#include "ops/BufferBase.hpp"
#include "ops/PimplFacade.hpp"
extern "C" {
extern unsigned char image_prelu_spv[];
extern unsigned int image_prelu_spv_len;
extern unsigned char buffer_prelu_spv[];
extern unsigned int buffer_prelu_spv_len;
extern unsigned char buffer_prelu_fp16_spv[];
extern unsigned int buffer_prelu_fp16_spv_len;
}
namespace vkop {
namespace ops {

class PReluImage : public BinaryFactory {
  public:
    PReluImage()
        : BinaryFactory(OpType::PRELU, image_prelu_spv, image_prelu_spv_len) {}
};

// PReLU buffer op: f(x) = max(0,x) + slope*min(0,x). Slope is broadcast
// right-aligned against the input/output shape. 3 SSBOs (out, input, slope).
// Uses the BinaryElemPC push constant (activation field unused).
class PReluBuffer : public BufferFactory {
  public:
    explicit PReluBuffer(int fp16)
        : BufferFactory(OpType::PRELU,
                        fp16 ? buffer_prelu_fp16_spv : buffer_prelu_spv,
                        fp16 ? buffer_prelu_fp16_spv_len : buffer_prelu_spv_len,
                        {DESCRIPTOR_TYPE_STORAGE, DESCRIPTOR_TYPE_STORAGE,
                         DESCRIPTOR_TYPE_STORAGE},
                        sizeof(BinaryElemPC), fp16) {}

  private:
    void execute(
        const std::vector<std::shared_ptr<core::ITensor>> &inputs,
        const std::vector<std::shared_ptr<core::ITensor>> &outputs) override {
        auto shape = inputs[0]->getShape();
        auto slope_shape = inputs[1]->getShape();
        int total = total_elems(shape);

        dispatch_by_dtype(outputs[0]->dtype(), [&](auto dummy) {
            using T = decltype(dummy);
            auto output = core::as_tensor<T>(outputs[0]);
            if (output->num_elements() != total_elems(shape)) {
                output->resize(shape);
            }
            bind_ssbo<T>(outputs[0], /*is_output=*/true);
        });
        dispatch_by_dtype(inputs[0]->dtype(), [&](auto dummy) {
            using T = decltype(dummy);
            bind_ssbo<T>(inputs[0], /*is_output=*/false);
        });
        dispatch_by_dtype(inputs[1]->dtype(), [&](auto dummy) {
            using T = decltype(dummy);
            bind_ssbo<T>(inputs[1], /*is_output=*/false);
        });

        BinaryElemPC pc{};
        pc.rank = static_cast<int>(shape.size());
        fill_dims(pc.outDims, shape);
        fill_dims(pc.in0Dims, shape);
        fill_dims_broadcast(pc.in1Dims, slope_shape, pc.rank);
        pc.activation = static_cast<int>(BufferActivation::NONE);
        pc.broadcast = (slope_shape == shape) ? 0 : 1;
        pc.total = total;
        int nthreads = (fp16_ != 0) ? (total + 1) / 2 : total;
        submit(&pc, UP_DIV(nthreads, 256), 1, 1);
    }
};

class PRelu : public PimplFacade {
  public:
    PRelu(int fp16, bool backend_buffer) : PimplFacade(OpType::PRELU) {
        impl_ =
            backend_buffer
                ? std::unique_ptr<Operator>(std::make_unique<PReluBuffer>(fp16))
                : std::make_unique<PReluImage>();
    }
};

} // namespace ops
} // namespace vkop
#endif // OPS_PRELU_HPP_
