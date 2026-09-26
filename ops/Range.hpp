// Copyright 2026 @junka
#ifndef OPS_RANGE_HPP_
#define OPS_RANGE_HPP_

#include "core/Tensor.hpp"
#include "ops/BufferBase.hpp"
#include "ops/Operator.hpp"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <numeric>

extern "C" {
extern unsigned char buffer_range_spv[];
extern unsigned int buffer_range_spv_len;
}
namespace vkop {
namespace ops {

namespace range {
struct GpuRangeParam {
    bool fp16;
};
} // namespace range

// SSBO-only op: generates a 1-D sequence [start, start+delta, ...].
class Range : public Operator {
  public:
    explicit Range()
        : Operator(OpType::RANGE, buffer_range_spv, buffer_range_spv_len,
                   {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    VK_DESCRIPTOR_TYPE_STORAGE_BUFFER},
                   sizeof(range::GpuRangeParam)) {
        param_.fp16 = false;
    }

    // The 6 Range instances in the LLM decode graph (position-index /
    // deepstack shape-meta chains) read back start/limit/delta every round,
    // but those scalars are STABLE across decode rounds (e.g. arange(0,64),
    // arange(0,1)). Same auto-learning state machine as Reshape/Expand:
    //   LEARNING (round 0): readback, store params.
    //   CONFIRMING (round 1): readback, compare. match -> STABLE. differ ->
    //   DYNAMIC. STABLE (round 2+): reuse cached params + cached output data,
    //   no copyToCPU. DYNAMIC: readback every round.
    // invalidate_shape_cache() (prefill->decode boundary) resets to LEARNING.
    enum class LearnState { LEARNING, CONFIRMING, STABLE, DYNAMIC };
    LearnState range_state_ = LearnState::LEARNING;
    // Cached scalar params (last readback). Used as the fingerprint.
    int64_t cached_start_ = 0, cached_limit_ = 0, cached_delta_ = 0;
    // Cached int64 output data (the arange result). STABLE lets us skip not
    // only the readback but also the recompute + re-upload, reusing the GPU
    // buffer bound last round.
    std::vector<int64_t> cached_out_;
    std::vector<int> cached_out_shape_;
    // Verification interval: even at STABLE, re-read every N rounds to catch
    // drift (a shape-meta producer whose value changes mid-phase).
    static constexpr int STABLE_VERIFY_INTERVAL = 8;
    int stable_rounds_ = 0;

    void invalidate_shape_cache() override {
        range_state_ = LearnState::LEARNING;
        cached_out_.clear();
        cached_out_shape_.clear();
        stable_rounds_ = 0;
    }

  private:
    void execute(
        const std::vector<std::shared_ptr<core::ITensor>> &inputs,
        const std::vector<std::shared_ptr<core::ITensor>> &outputs) override {

        // Read start/limit/delta once (cached). Returns the scalar params +
        // whether a readback actually happened this round (for the fp path,
        // which still needs out_shape; for the int64 path, STABLE reuses the
        // cached output data wholesale).
        // The fp readback block above previously ran dispatch_by_dtype on
        // inputs[0] (which for int64 Range ALSO ran, causing a redundant 3×
        // copyToCPU before the int64 block re-read them). Unified here so each
        // round reads at most once.
        bool is_int64 = inputs[0]->dtype() == typeid(int64_t);
        int64_t start = 0, limit = 0, delta = 0;
        int inums = 0;
        std::vector<int> out_shape;

        // STABLE fast path for int64: reuse cached output data, skip readback
        // entirely. The arange result is byte-identical when start/limit/delta
        // are stable, so the GPU buffer from last round is still valid.
        if (is_int64 && range_state_ == LearnState::STABLE) {
            if (stable_rounds_ < STABLE_VERIFY_INTERVAL) {
                stable_rounds_++;
                // Reuse cached output: re-bind the SSBO (descriptor needs it
                // each recording), but skip copyToCPU + recompute + re-upload.
                auto output = core::as_tensor<int64_t>(outputs[0]);
                if (output->num_elements() != total_elems(cached_out_shape_)) {
                    output->resize(cached_out_shape_);
                }
                output->fillToCPU(cached_out_);
                objs_.emplace_back(output->as_storage_buffer(m_dev_, m_cmd_));
                output->copyToGPUDeferred(m_cmd_);
                if (host_shape_enabled())
                    output->set_host_authoritative();
                if (std::getenv("VKOP_RB_TRACE")) {
                    fprintf(
                        stderr, "[rbtrace] Range i64 STABLE skip (inums=%d)\n",
                        cached_out_shape_.empty() ? -1 : cached_out_shape_[0]);
                }
                return;
            }
            // Verify: fall through to readback + compare.
        }

        // Readback path (LEARNING/CONFIRMING/DYNAMIC, or STABLE verify).
        if (is_int64) {
            auto start_t = core::as_tensor<int64_t>(inputs[0]);
            start_t->copyToCPU(m_cmdpool_);
            auto limit_t = core::as_tensor<int64_t>(inputs[1]);
            limit_t->copyToCPU(m_cmdpool_);
            auto delta_t = core::as_tensor<int64_t>(inputs[2]);
            delta_t->copyToCPU(m_cmdpool_);
            start = start_t->at(0);
            limit = limit_t->at(0);
            delta = delta_t->at(0);
        } else {
            dispatch_by_dtype(inputs[0]->dtype(), [&](auto dummy) {
                using T = decltype(dummy);
                auto input0 = core::as_tensor<T>(inputs[0]);
                input0->copyToCPU(m_cmdpool_);
                start = input0->at(0);
                auto input1 = core::as_tensor<T>(inputs[1]);
                input1->copyToCPU(m_cmdpool_);
                limit = input1->at(0);
                auto input2 = core::as_tensor<T>(inputs[2]);
                input2->copyToCPU(m_cmdpool_);
                delta = input2->at(0);
            });
        }

        // ONNX count = max(0, ceil((limit-start)/delta)) with the division
        // rounding away from zero when sign(delta)==sign(range). The old
        // `ceil((limit-start)/abs(delta))` gave a NEGATIVE count for a
        // descending range (e.g. start=3,limit=-8,delta=-1 -> -11), which then
        // underflowed size_ and crashed buffer creation.
        if (is_int64) {
            int64_t range = limit - start;
            int64_t i_inums = 0;
            if (range == 0) {
                i_inums = 0;
            } else if (range > 0) {
                i_inums = (delta > 0) ? (range + delta - 1) / delta : 0;
            } else {
                i_inums = (delta < 0) ? (range - delta - 1) / delta : 0;
            }
            inums = static_cast<int>(i_inums);
        } else {
            double range =
                static_cast<double>(limit) - static_cast<double>(start);
            if (range > 0 && delta > 0) {
                inums = static_cast<int>(
                    std::ceil(range / static_cast<double>(delta)));
            } else if (range < 0 && delta < 0) {
                inums = static_cast<int>(
                    std::ceil(range / static_cast<double>(delta)));
            }
        }
        out_shape = {inums};

        // Advance the learn state machine for int64 (the cached-output reuse
        // only applies there; fp Range re-dispatches the GPU shader each round
        // anyway, so caching the params gains nothing on fp).
        if (is_int64) {
            if (range_state_ == LearnState::LEARNING) {
                cached_start_ = start;
                cached_limit_ = limit;
                cached_delta_ = delta;
                range_state_ = LearnState::CONFIRMING;
            } else if (range_state_ == LearnState::CONFIRMING) {
                if (start == cached_start_ && limit == cached_limit_ &&
                    delta == cached_delta_) {
                    range_state_ = LearnState::STABLE;
                    stable_rounds_ = 0;
                } else {
                    range_state_ = LearnState::DYNAMIC;
                    cached_start_ = start;
                    cached_limit_ = limit;
                    cached_delta_ = delta;
                }
            } else if (range_state_ == LearnState::STABLE) {
                // Verify round: if drift, drop to DYNAMIC.
                if (!(start == cached_start_ && limit == cached_limit_ &&
                      delta == cached_delta_)) {
                    range_state_ = LearnState::DYNAMIC;
                    cached_start_ = start;
                    cached_limit_ = limit;
                    cached_delta_ = delta;
                } else {
                    stable_rounds_ = 0; // reset verify timer
                }
            } else { // DYNAMIC
                cached_start_ = start;
                cached_limit_ = limit;
                cached_delta_ = delta;
            }
        }

        if (std::getenv("VKOP_RB_TRACE")) {
            fprintf(stderr,
                    "[rbtrace] Range %s start=%lld limit=%lld delta=%lld -> "
                    "inums=%d state=%d\n",
                    is_int64 ? "i64" : "fp", (long long)start, (long long)limit,
                    (long long)delta, inums, static_cast<int>(range_state_));
        }

        // int64 range runs on the CPU (all 6 instances are part of the shape
        // meta-chain, e.g. position indices). Generate [start, start+delta,
        // ..].
        if (is_int64) {
            std::vector<int64_t> out(static_cast<size_t>(inums));
            for (int64_t i = 0; i < inums; ++i) {
                out[static_cast<size_t>(i)] = start + i * delta;
            }
            cached_out_ = out; // refresh cache (cheap vector copy)
            cached_out_shape_ = out_shape;
            auto output = core::as_tensor<int64_t>(outputs[0]);
            if (output->num_elements() != total_elems(out_shape)) {
                output->resize(out_shape);
            }
            output->fillToCPU(out);
            objs_.emplace_back(output->as_storage_buffer(m_dev_, m_cmd_));
            output->copyToGPUDeferred(m_cmd_);
            if (host_shape_enabled())
                output->set_host_authoritative();
            return;
        }

        dispatch_by_dtype(outputs[0]->dtype(), [&](auto dummy) {
            using T = decltype(dummy);
            auto output = core::as_tensor<T>(outputs[0]);
            if (output->num_elements() != total_elems(out_shape)) {
                output->resize(out_shape);
            }
            auto output_buffer = output->as_storage_buffer(m_dev_, m_cmd_);
            objs_.emplace_back(output_buffer);
            if (typeid(uint16_t) == typeid(T)) {
                param_.fp16 = true;
            }
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
        submit(&param_, UP_DIV(total_size, 256), 1, 1);
    }

    range::GpuRangeParam param_;
};

} // namespace ops
} // namespace vkop
#endif // OPS_RANGE_HPP_
