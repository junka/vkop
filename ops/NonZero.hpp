// Copyright 2026 @junka
#ifndef OPS_NONZERO_HPP_
#define OPS_NONZERO_HPP_

#include "core/Tensor.hpp"
#include "ops/BufferBase.hpp"
#include <cstdio>
#include <cstdlib>
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
//
// STABLE cache: the mask is phase-invariant (fixed for the whole decode
// phase — the image doesn't change between tokens), so two consecutive
// matching readbacks promote to STABLE and skip the copyToCPU entirely,
// reusing the cached [rank, count] output. invalidate_shape_cache() resets
// at the prefill->decode boundary.
class NonZero : public BufferFactory {
  public:
    explicit NonZero()
        : BufferFactory(OpType::NONZERO, buffer_nonzero_spv,
                        buffer_nonzero_spv_len,
                        {DESCRIPTOR_TYPE_STORAGE, DESCRIPTOR_TYPE_STORAGE},
                        sizeof(nonzero::NonZeroPC)) {}

    enum class LearnState { LEARNING, CONFIRMING, STABLE, DYNAMIC };
    LearnState nz_state_ = LearnState::LEARNING;
    // Fingerprint of the input: shape + flattened data (as double for
    // dtype-agnostic comparison). Small (mask is tiny).
    std::vector<int> cached_shape_;
    std::vector<double> cached_input_;
    // Cached output data + shape.
    std::vector<int> cached_out_shape_;
    std::vector<int64_t> cached_out_;
    static constexpr int STABLE_VERIFY_INTERVAL = 8;
    int stable_rounds_ = 0;

    void invalidate_shape_cache() override {
        nz_state_ = LearnState::LEARNING;
        cached_shape_.clear();
        cached_input_.clear();
        cached_out_shape_.clear();
        cached_out_.clear();
        stable_rounds_ = 0;
    }

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

        // STABLE fast path: reuse cached output, skip readback entirely.
        // The mask is phase-invariant (image fixed for the whole decode
        // phase), so the [rank, count] output is byte-identical when the
        // input matches.
        if (nz_state_ == LearnState::STABLE &&
            stable_rounds_ < STABLE_VERIFY_INTERVAL) {
            stable_rounds_++;
            auto output = core::as_tensor<int64_t>(outputs[0]);
            output->resize(cached_out_shape_);
            output->fillToCPU(cached_out_);
            objs_.emplace_back(output->as_storage_buffer(m_dev_, m_cmd_));
            output->copyToGPUDeferred(m_cmd_);
            if (host_shape_enabled())
                output->set_host_authoritative();
            if (std::getenv("VKOP_RB_TRACE")) {
                fprintf(stderr, "[rbtrace] NonZero STABLE skip (count=%d)\n",
                        cached_out_shape_.size() > 1 ? cached_out_shape_[1]
                                                     : -1);
            }
            return;
        }

        // Readback path (LEARNING/CONFIRMING/DYNAMIC, or STABLE verify).
        // Pull the input to the host and collect the multi-dim coordinates of
        // every non-zero element. bool/int8 share the int8_t storage repr in
        // the runtime; float/int64 are also supported by dispatch_by_dtype.
        std::vector<std::vector<int64_t>>
            coords; // coords[k] = coord of k-th nz
        std::vector<double> input_fp(total);
        dispatch_by_dtype(inputs[0]->dtype(), [&](auto dummy) {
            using T = decltype(dummy);
            auto input = core::as_tensor<T>(inputs[0]);
            // Unconditional readback: a cross-round-recycled GPU input may
            // have stale CPU data_ (see SqueezeUnsqueeze/ScatterElements fix).
            input->copyToCPU(m_cmdpool_);
            // Precompute per-axis strides (row-major) for coordinate decode.
            std::vector<int64_t> stride(rank, 1);
            for (int d = rank - 2; d >= 0; --d) {
                stride[d] = stride[d + 1] * shape[d + 1];
            }
            for (int i = 0; i < total; ++i) {
                double v = static_cast<double>((*input)[i]);
                input_fp[i] = v;
                if (v != 0.0) {
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
        if (std::getenv("VKOP_RB_TRACE")) {
            fprintf(stderr, "[rbtrace] NonZero shape=");
            for (int d : shape)
                fprintf(stderr, "%d,", d);
            fprintf(stderr, " total=%d count=%d state=%d\n", total, count,
                    static_cast<int>(nz_state_));
        }
        // ONNX NonZero output: [rank, count], column-major. For rank==1 this
        // is just [1, count] == a flat [count] of linear indices in memory.
        std::vector<int> out_shape = {rank, count};
        std::vector<int64_t> out(static_cast<size_t>(rank) * count);
        for (int k = 0; k < count; ++k) {
            for (int r = 0; r < rank; ++r) {
                out[static_cast<size_t>(r) * count + k] = coords[k][r];
            }
        }

        // Advance the learn state machine. Fingerprint = shape + input data.
        bool match = (shape == cached_shape_ && input_fp == cached_input_);
        if (nz_state_ == LearnState::LEARNING) {
            cached_shape_ = shape;
            cached_input_ = input_fp;
            cached_out_shape_ = out_shape;
            cached_out_ = out;
            nz_state_ = LearnState::CONFIRMING;
        } else if (nz_state_ == LearnState::CONFIRMING) {
            if (match) {
                nz_state_ = LearnState::STABLE;
                stable_rounds_ = 0;
            } else {
                nz_state_ = LearnState::DYNAMIC;
                cached_shape_ = shape;
                cached_input_ = input_fp;
                cached_out_shape_ = out_shape;
                cached_out_ = out;
            }
        } else if (nz_state_ == LearnState::STABLE) {
            // Verify round: drift -> DYNAMIC.
            if (!match) {
                nz_state_ = LearnState::DYNAMIC;
                cached_shape_ = shape;
                cached_input_ = input_fp;
                cached_out_shape_ = out_shape;
                cached_out_ = out;
            } else {
                stable_rounds_ = 0;
            }
        } else { // DYNAMIC
            cached_shape_ = shape;
            cached_input_ = input_fp;
            cached_out_shape_ = out_shape;
            cached_out_ = out;
        }

        auto output = core::as_tensor<int64_t>(outputs[0]);
        output->resize(out_shape);
        output->fillToCPU(out);
        objs_.emplace_back(output->as_storage_buffer(m_dev_, m_cmd_));
        output->copyToGPUDeferred(m_cmd_);
        if (host_shape_enabled())
            output->set_host_authoritative();
    }
};

} // namespace ops
} // namespace vkop
#endif // OPS_NONZERO_HPP_
