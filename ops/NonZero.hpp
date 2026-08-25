// Copyright 2026 @junka
#ifndef OPS_NONZERO_HPP_
#define OPS_NONZERO_HPP_

#include "core/Tensor.hpp"
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

// SSBO op: ONNX NonZero. Returns the indices of the non-zero elements of the
// input, as an int64 tensor of shape [rank, num_nonzero] (column-major: the
// k-th nonzero element's coordinates occupy out[:, k], i.e. out[r*count+k]
// for axis r).
//
// Host-side implementation. The LLM's sole NonZero feeds a deep int64
// shape-meta chain (Transpose[1,0] -> Expand -> ScatterElements) that
// consumes the [rank, num_nonzero] layout directly; the old single-pass GPU
// shader wrote a flat [count, idx...] buffer with shape [total+1], which
// broke Transpose's perm=[1,0] (rank-1 input read OOB) and fed garbage
// scatter indices. The input (image_pad_mask-derived) is tiny and bool/int8,
// and the count is only known after scanning the data — so compute on the
// host, set the exact [rank, count] shape, and upload. (All 6 decode rounds
// have an all-False mask -> count=0 -> empty [1,0] output -> empty scatter,
// which is the correct no-op.)
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
        int rank = static_cast<int>(shape.size());
        if (rank == 0) {
            rank = 1; // scalar input: treat as rank-1 [1]
            shape = {1};
        }
        int total = total_elems(shape);

        // Pull the input to the host and collect the multi-dim coordinates of
        // every non-zero element. bool/int8 share the int8_t storage repr in
        // the runtime; float/int64 are also supported by dispatch_by_dtype.
        std::vector<std::vector<int64_t>>
            coords; // coords[k] = coord of k-th nz
        dispatch_by_dtype(inputs[0]->dtype(), [&](auto dummy) {
            using T = decltype(dummy);
            auto input = core::as_tensor<T>(inputs[0]);
            if (!input->has_cpu_data()) {
                input->copyToCPU(m_cmdpool_);
            }
            // Precompute per-axis strides (row-major) for coordinate decode.
            std::vector<int64_t> stride(rank, 1);
            for (int d = rank - 2; d >= 0; --d) {
                stride[d] = stride[d + 1] * shape[d + 1];
            }
            for (int i = 0; i < total; ++i) {
                if (static_cast<double>((*input)[i]) != 0.0) {
                    std::vector<int64_t> c(rank);
                    int rem = i;
                    for (int d = 0; d < rank; ++d) {
                        c[d] = rem / static_cast<int>(stride[d]);
                        rem = rem % static_cast<int>(stride[d]);
                    }
                    coords.push_back(std::move(c));
                }
            }
        });

        int count = static_cast<int>(coords.size());
        // ONNX NonZero output: [rank, count], column-major. For rank==1 this
        // is just [1, count] == a flat [count] of linear indices in memory.
        std::vector<int> out_shape = {rank, count};
        std::vector<int64_t> out(static_cast<size_t>(rank) * count);
        for (int k = 0; k < count; ++k) {
            for (int r = 0; r < rank; ++r) {
                out[static_cast<size_t>(r) * count + k] = coords[k][r];
            }
        }

        auto output = core::as_tensor<int64_t>(outputs[0]);
        output->resize(out_shape);
        output->fillToCPU(out);
        objs_.emplace_back(output->as_storage_buffer(m_dev_, m_cmd_));
        output->copyToGPU(m_cmdpool_, out.data());
    }
};

} // namespace ops
} // namespace vkop
#endif // OPS_NONZERO_HPP_
