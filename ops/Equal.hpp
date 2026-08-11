// Copyright 2026 @junka
#ifndef OPS_EQUAL_HPP_
#define OPS_EQUAL_HPP_

#include "ops/BufferBinaryFactory.hpp"
#include <numeric>

extern "C" {
extern unsigned char buffer_equal_spv[];
extern unsigned int buffer_equal_spv_len;
}

namespace vkop {
namespace ops {

// SSBO-only op: C[i] = (A[i] == B[i]) ? 1 : 0. Supports broadcasting.
// Output is uint (1=true, 0=false), matching Where's condition buffer.
class Equal : public BufferBinaryFactory {
  public:
    explicit Equal()
        : BufferBinaryFactory(OpType::EQUAL, buffer_equal_spv,
                              buffer_equal_spv_len, /*fp16=*/0) {}

  private:
    void execute(
        const std::vector<std::shared_ptr<core::ITensor>> &inputs,
        const std::vector<std::shared_ptr<core::ITensor>> &outputs) override {
        auto shape_a = inputs[0]->getShape();
        auto shape_b = inputs[1]->getShape();
        auto out_shape = computeBroadcastShape(shape_a, shape_b);
        int total = total_elems(out_shape);

        // int64 comparison runs on the CPU (Equal is a pure host op in the
        // LLM's shape meta-chain). The runtime allocates the output as
        // int64 when the inputs are int64, and Where's CPU branch reads the
        // result as `cond != 0`, so emit actual int64 1/0.
        if (inputs[0]->dtype() == typeid(int64_t)) {
            auto a = core::as_tensor<int64_t>(inputs[0]);
            auto b = core::as_tensor<int64_t>(inputs[1]);
            if (!a->has_cpu_data()) {
                a->copyToCPU(m_cmdpool_);
            }
            if (!b->has_cpu_data()) {
                b->copyToCPU(m_cmdpool_);
            }
            std::vector<int64_t> out(total);
            for (int i = 0; i < total; ++i) {
                int64_t av = (*a)[broadcast_index(shape_a, out_shape, i)];
                int64_t bv = (*b)[broadcast_index(shape_b, out_shape, i)];
                out[i] = (av == bv) ? 1 : 0;
            }
            auto output = core::as_tensor<int64_t>(outputs[0]);
            output->resize(out_shape);
            output->fillToCPU(out);
            objs_.emplace_back(output->as_storage_buffer(m_dev_, m_cmd_));
            output->copyToGPU(m_cmdpool_, out.data());
            return;
        }

        dispatch_by_dtype(outputs[0]->dtype(), [&](auto dummy) {
            using T = decltype(dummy);
            auto output = core::as_tensor<T>(outputs[0]);
            if (output->size() == 0) {
                output->resize(out_shape);
            }
            bind_ssbo<T>(outputs[0], true);
        });
        bind_ssbo<float>(inputs[0], false);
        bind_ssbo<float>(inputs[1], false);

        BinaryElemPC pc{};
        pc.rank = static_cast<int>(out_shape.size());
        fill_dims(pc.outDims, out_shape);
        fill_dims_broadcast(pc.in0Dims, shape_a, pc.rank);
        fill_dims_broadcast(pc.in1Dims, shape_b, pc.rank);
        pc.activation = 0;
        pc.broadcast = (shape_a == out_shape && shape_b == out_shape) ? 0 : 1;
        pc.total = total;
        submit(&pc, UP_DIV(total, 256), 1, 1);
    }
};

} // namespace ops
} // namespace vkop
#endif // OPS_EQUAL_HPP_
