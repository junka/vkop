// Copyright 2026 @junka
#ifndef OPS_NONZERO_HPP_
#define OPS_NONZERO_HPP_

#include "ops/BufferBase.hpp"
#include <numeric>

extern "C" {
extern unsigned char buffer_nonzero_spv[];
extern unsigned int buffer_nonzero_spv_len;
}

namespace vkop {
namespace ops {

namespace nonzero {
struct alignas(16) NonZeroPC {
    int total;
    int _pad0;
    int _pad1;
    int _pad2;
};
} // namespace nonzero

// SSBO-only op: finds linear indices of all non-zero elements.
// Output: uint buffer where [0] = count, [1..count] = indices.
// The host must zero output[0] before dispatch and read back the count.
class NonZero : public BufferFactory {
  public:
    explicit NonZero()
        : BufferFactory(OpType::NONZERO, buffer_nonzero_spv,
                        buffer_nonzero_spv_len,
                        {DESCRIPTOR_TYPE_STORAGE, DESCRIPTOR_TYPE_STORAGE},
                        sizeof(nonzero::NonZeroPC)) {}

  private:
    void execute(
        const std::vector<std::shared_ptr<core::ITensor>> &inputs,
        const std::vector<std::shared_ptr<core::ITensor>> &outputs) override {
        auto shape = inputs[0]->getShape();
        int total = total_elems(shape);
        // Output: [0]=count + max `total` indices. Worst case: all non-zero.
        int out_size = total + 1;

        dispatch_by_dtype(outputs[0]->dtype(), [&](auto dummy) {
            using T = decltype(dummy);
            auto output = core::as_tensor<T>(outputs[0]);
            if (output->size() == 0) {
                output->resize(std::vector<int>{out_size});
            }
            bind_ssbo<T>(outputs[0], true);
        });
        // Input may be bool/int8/int64/float — bind on its actual dtype so
        // as_tensor<T> yields a non-null Tensor<T> (hardcoding float would
        // crash on the LLM's int8 image_pad_mask-derived input).
        dispatch_by_dtype(inputs[0]->dtype(), [&](auto dummy) {
            using T = decltype(dummy);
            bind_ssbo<T>(inputs[0], false);
        });

        // Zero the counter at output[0].
        // Since output is bound as SSBO (not zeroed), we use a fillBuffer
        // via the command buffer before dispatch.
        // Actually, the output tensor might be on GPU already. We need to
        // zero the first 4 bytes. Use a staging copy or memset.
        // Simplest: the host fills the output with zeros before copyToGPU.
        // But the output is created on GPU (toGPU). So we use a dummy
        // approach: dispatch a fillBuffer via the VulkanBuffer.
        // Actually, the output's VulkanBuffer is in objs_[0]. We can
        // dynamic_cast and fillBuffer.
        auto out_buf = std::dynamic_pointer_cast<VulkanBuffer>(objs_[0]);
        if (out_buf) {
            out_buf->fillBuffer(m_cmd_->get(), 0u, 4);
        }

        nonzero::NonZeroPC pc{};
        pc.total = total;
        submit(&pc, UP_DIV(total, 256), 1, 1);
    }
};

} // namespace ops
} // namespace vkop
#endif // OPS_NONZERO_HPP_
