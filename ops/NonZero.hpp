// Copyright 2026 @junka
#ifndef OPS_NONZERO_HPP_
#define OPS_NONZERO_HPP_

#include "core/Tensor.hpp"
#include "ops/BufferBase.hpp"
#include <cstdio>
#include <cstdlib>
#include <numeric>

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
// at the prefill->decode boundary (and on every new prompt via
// Runtime::invalidate_replay). STABLE trusts the phase contract and reuses
// the cached output unconditionally — there is NO periodic verify readback.
// Rationale: within a decode phase the image_pad_mask is byte-identical every
// round (the driver writes [1,1] all-zero on every decode step), so a verify
// would never observe drift; the only point where the mask legitimately
// changes is a phase boundary, which already resets the state via
// invalidate_shape_cache(). A verify readback here would only pay a sync stall
// (it splits the graph into an extra segment) to confirm something the phase
// contract already guarantees.
class NonZero : public BufferFactory {
  public:
    explicit NonZero()
        : BufferFactory(OpType::NONZERO,
                        {DESCRIPTOR_TYPE_STORAGE, DESCRIPTOR_TYPE_STORAGE},
                        sizeof(nonzero::NonZeroPC)) {}

    // Fingerprint of the REUSABLE result: the mask's shape and values, plus
    // the output they produced. A round whose fingerprint matches the previous
    // one can reuse that output verbatim.
    //
    // A comparison, not a learning machine. The machine here promoted to
    // STABLE after two matching rounds and then trusted the mask never to
    // change, with no verify. Comparison gets the same steady-state win (one
    // cheap readback of a tiny mask per round) without depending on the phase
    // contract, and without state that can only advance inside execute().
    std::vector<int> cached_shape_;
    std::vector<double> cached_input_;
    std::vector<int> cached_out_shape_;
    std::vector<int64_t> cached_out_;
    // Whether the last round's fingerprint matched (drives the replay and
    // host-work answers).
    bool nz_reusable_ = false;
    bool invalidate_pending_ = false;
    // Mask values read once because the mask is a read-only leaf.
    std::vector<double> leaf_mask_values_;
    bool leaf_mask_valid_ = false;

    void invalidate_shape_cache() override {
        // Reuse is decided by comparing the mask with last round's, so a phase
        // boundary needs no reset for correctness -- but the cached OUTPUT was
        // produced by the previous phase, so refuse one reuse to be safe.
        invalidate_pending_ = true;
        nz_reusable_ = false;
    }
    bool has_host_side_effects() const override { return true; }

    // Replay answers. In STABLE this op re-fills and re-uploads the SAME cached
    // bytes every round, so the recording it produces is identical round to
    // round — a replay re-runs that recorded upload and reproduces exactly the
    // right buffer contents. What a replay skips is the host-side half (data_ +
    // host_authoritative), which refresh_dynamic_params() redoes.
    //
    // LEARNING/CONFIRMING/DYNAMIC do a real readback whose result can differ;
    // those keep the conservative answer.
    bool recording_changes_per_round() const override {
        return !nz_reusable_;
    }
    bool did_host_work_last_round() const override {
        // Not `host_work_observed_ ? host_work_last_round_ : ...`: execute()
        // sets that flag unconditionally at its top, so it is always observed
        // and always true, which would mask this answer.
        return !nz_reusable_;
    }
    bool refresh_dynamic_params(
        const std::vector<std::shared_ptr<core::ITensor>> & /*inputs*/,
        const std::vector<std::shared_ptr<core::ITensor>> &outputs) override {
        if (!nz_reusable_ || cached_out_.empty() ||
            outputs.empty() || !outputs[0]) {
            return false;
        }
        auto output = core::as_tensor<int64_t>(outputs[0]);
        if (!output) return false;
        if (output->getShape() != cached_out_shape_) {
            output->resize(cached_out_shape_);
        }
        output->fillToCPU(cached_out_);
        // Mirror execute()'s tail exactly: it ends with copyToGPUDeferred(),
        // which marks the tensor device-resident. Skipping that here leaves
        // converted_ false on a replayed segment while the buffer genuinely
        // holds the bytes, and a downstream op that consults is_on_GPU() then
        // takes the CPU path.
        output->toGPU();
        if (host_shape_enabled()) output->set_host_authoritative();
        if (std::getenv("VKOP_REF_CMP")) {
            auto r = output->gpu_buffer();
            fprintf(stderr,
                    "[ref-cmp] op=%s refresh: shape=%zu/%zu data=%zu gpu=%d "
                    "auth=%d conv=%d hostcopy=%d\n",
                    get_name().c_str(), output->getShape().size(),
                    cached_out_shape_.size(),
                    output->getShape().empty()
                        ? 0
                        : (size_t)output->getShape().back(),
                    r ? 1 : 0, (int)output->get_host_authoritative(),
                    (int)output->is_on_GPU(),
                    (int)output->get_host_copy_valid());
        }
        return true;
    }

  private:
    void execute(
        const std::vector<std::shared_ptr<core::ITensor>> &inputs,
        const std::vector<std::shared_ptr<core::ITensor>> &outputs) override {
        // Host work here is the mask readback + host coordinate scan below;
        // the STABLE fast path re-fills the cached output on the host instead
        // (still host work, still skipped by a replay — so NonZero stays
        // dynamic either way).
        set_did_host_work_last_round(true);
        auto shape = inputs[0]->getShape();
        int rank = static_cast<int>(shape.size());
        if (rank == 0) {
            rank = 1; // scalar input: treat as rank-1 [1]
            shape = {1};
        }
        int total = total_elems(shape);

        // Read the mask's values and compare with last round's. Equal means the
        // [rank, count] output is byte-identical, so it can be re-filled from
        // the cache instead of scanned again.
        //
        // The mask is tiny, and the readback is free whenever the producer
        // left the bytes on the host (a leaf mask, or one produced by another
        // cpu-path shape op) -- which is every case this chain actually hits.
        // Comparing is what the removed learning machine asserted instead: it
        // promoted to "stable" after two matching rounds and then trusted the
        // mask never to change, with no verify, keyed on a phase contract
        // every driver had to honour.
        // A read-only leaf mask cannot change, so its values (and therefore
        // this op's output) are a constant of the node: read once, then reuse
        // forever with no readback at all.
        const bool leaf_mask = inputs[0]->is_read_only_leaf();
        std::vector<double> input_fp;
        if (leaf_mask && leaf_mask_valid_) {
            input_fp = leaf_mask_values_;
        } else {
            input_fp.resize(total);
            dispatch_by_dtype(inputs[0]->dtype(), [&](auto dummy) {
                using T = decltype(dummy);
                auto input = core::as_tensor<T>(inputs[0]);
                // Unconditional readback: a cross-round-recycled GPU input may
                // have stale CPU data_ (see SqueezeUnsqueeze/ScatterElements
                // fix).
                input->copyToCPU(m_cmdpool_);
                for (int i = 0; i < total; ++i) {
                    input_fp[i] = static_cast<double>((*input)[i]);
                }
            });
            if (leaf_mask) {
                leaf_mask_values_ = input_fp;
                leaf_mask_valid_ = true;
            }
        }
        nz_reusable_ = !cached_out_.empty() && !invalidate_pending_ &&
                       shape == cached_shape_ && input_fp == cached_input_;
        invalidate_pending_ = false;
        if (nz_reusable_) {
            set_did_host_work_last_round(false);
            auto output = core::as_tensor<int64_t>(outputs[0]);
            output->resize(cached_out_shape_);
            output->fillToCPU(cached_out_);
            objs_.emplace_back(output->as_storage_buffer(m_dev_, m_cmd_));
            output->copyToGPUDeferred(m_cmd_);
            if (host_shape_enabled())
                output->set_host_authoritative();
            return;
        }

        // Different mask: rescan.
        // Collect the multi-dim coordinates of every non-zero element.
        // bool/int8 share the int8_t storage repr in the runtime; float/int64
        // are also supported by dispatch_by_dtype.
        std::vector<std::vector<int64_t>>
            coords; // coords[k] = coord of k-th nz
        {
            // Precompute per-axis strides (row-major) for coordinate decode.
            std::vector<int64_t> stride(rank, 1);
            for (int d = rank - 2; d >= 0; --d) {
                stride[d] = stride[d + 1] * shape[d + 1];
            }
            for (int i = 0; i < total; ++i) {
                if (input_fp[i] == 0.0) continue;
                std::vector<int64_t> c(rank);
                int rem = i;
                for (int d = 0; d < rank; ++d) {
                    c[d] = rem / static_cast<int>(stride[d]);
                    rem = rem % static_cast<int>(stride[d]);
                }
                coords.push_back(std::move(c));
            }
        }

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

        // Remember this round's fingerprint and result. The next round
        // reuses the output if the mask is byte-identical, which the fast path
        // at the top of execute() checks; nothing is promoted or predicted.
        cached_shape_ = shape;
        cached_input_ = input_fp;
        cached_out_shape_ = out_shape;
        cached_out_ = out;

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
