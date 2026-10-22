// Copyright 2026 @junka
#ifndef OPS_RANGE_HPP_
#define OPS_RANGE_HPP_

#include "core/Tensor.hpp"
#include "ops/BufferBase.hpp"
#include "ops/Operator.hpp"
#include <cmath>
#include <cstdio>
#include <cstdlib>

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
    // Single-impl op: always the SSBO path (no image twin).
    bool uses_buffer_backend() const override { return true; }

    explicit Range()
        : Operator(OpType::RANGE,
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

        // Read start/limit/delta once per round: as int64 for an int32/int64
        // Range, as double for an fp16/fp32 one. The value generation below
        // is on the host for every dtype (there is no integer kernel to call),
        // so this readback is the only thing the GPU knows about a Range.
        bool is_int64 = inputs[0]->dtype() == typeid(int64_t);
        bool is_int32 = inputs[0]->dtype() == typeid(int);
        if (!is_int64 && !is_int32 && inputs[0]->dtype() != typeid(float) &&
            inputs[0]->dtype() != typeid(uint16_t)) {
            throw std::runtime_error(
                std::string("Range: start/limit/delta must be int32, int64, "
                            "fp16 or fp32, got ") +
                core::elem_name(inputs[0]->elem_kind()));
        }
        int64_t start = 0, limit = 0, delta = 0;
        // The fp path reads its params as double. Both fp widths can carry
        // fractional values that an int64_t would truncate, and an fp16
        // operand holds its bits packed two-per-word, so the value has to go
        // through fp16_to_fp32 rather than be read as an integer.
        double f_start = 0, f_limit = 0, f_delta = 0;
        int inums = 0;
        std::vector<int> out_shape;

        // STABLE fast path for int64: reuse cached output data, skip readback
        // entirely. The arange result is byte-identical when start/limit/delta
        // are stable, so the GPU buffer from last round is still valid.
        // (The cached bytes are int64, so this only serves an int64 output.)
        if (is_int64 && outputs[0]->dtype() == typeid(int64_t) &&
            range_state_ == LearnState::STABLE) {
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
        } else if (is_int32) {
            // int32 params are integers too: read them into the same exact
            // integer path (the shader used to serve these, and the fp path
            // below would hand back a float output for an int32 Range).
            auto read_i32 = [&](size_t idx) -> int64_t {
                auto t = core::as_tensor<int>(inputs[idx]);
                t->copyToCPU(m_cmdpool_);
                return static_cast<int64_t>(t->at(0));
            };
            start = read_i32(0);
            limit = read_i32(1);
            delta = read_i32(2);
        } else {
            dispatch_by_dtype(inputs[0]->dtype(), [&](auto dummy) {
                using T = decltype(dummy);
                auto read_one = [&](size_t idx) -> double {
                    auto t = core::as_tensor<T>(inputs[idx]);
                    t->copyToCPU(m_cmdpool_);
                    if constexpr (std::is_same_v<T, uint16_t>) {
                        return static_cast<double>(
                            core::ITensor::fp16_to_fp32(t->at(0)));
                    } else {
                        return static_cast<double>(t->at(0));
                    }
                };
                f_start = read_one(0);
                f_limit = read_one(1);
                f_delta = read_one(2);
            });
        }

        // ONNX count = max(0, ceil((limit-start)/delta)) with the division
        // rounding away from zero when sign(delta)==sign(range). The old
        // `ceil((limit-start)/abs(delta))` gave a NEGATIVE count for a
        // descending range (e.g. start=3,limit=-8,delta=-1 -> -11), which then
        // underflowed size_ and crashed buffer creation.
        if (is_int64 || is_int32) {
            if (delta == 0) {
                throw std::runtime_error(
                    "Range: delta must not be 0 (ONNX requires a non-zero "
                    "step)");
            }
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
            if (f_delta == 0.0) {
                throw std::runtime_error(
                    "Range: delta must not be 0 (ONNX requires a non-zero "
                    "step)");
            }
            double range = f_limit - f_start;
            if ((range > 0 && f_delta > 0) || (range < 0 && f_delta < 0)) {
                inums = static_cast<int>(std::ceil(range / f_delta));
            }
        }
        out_shape = {inums};

        // Advance the learn state machine for int64 (the cached-output reuse
        // only applies there: the cache holds int64 values, so it cannot serve
        // an fp output. An fp Range still reads back and regenerates each
        // round, which is what it did before the host-side generation too).
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

        // Integer ranges run on the CPU (all 6 instances in the LLM graph are
        // part of the shape meta-chain, e.g. position indices). Generate
        // [start, start+delta, ..] exactly, then store at the output's own
        // width — an int32-param Range with an fp output is a shape the tests
        // use and the conversion is exact for int32 magnitudes.
        if (is_int64 || is_int32) {
            std::vector<int64_t> ivals(static_cast<size_t>(inums));
            for (int64_t i = 0; i < inums; ++i) {
                ivals[static_cast<size_t>(i)] = start + i * delta;
            }
            if (outputs[0]->dtype() == typeid(int64_t)) {
                cached_out_ = ivals; // refresh cache (cheap vector copy)
                cached_out_shape_ = out_shape;
            }
            dispatch_by_dtype(outputs[0]->dtype(), [&](auto dummy) {
                using T = decltype(dummy);
                if constexpr (std::is_same_v<T, int64_t> ||
                              std::is_same_v<T, int>) {
                    std::vector<T> out(ivals.size());
                    for (size_t i = 0; i < ivals.size(); ++i) {
                        out[i] = static_cast<T>(ivals[i]);
                    }
                    auto output = core::as_tensor<T>(outputs[0]);
                    if (output->num_elements() != total_elems(out_shape)) {
                        output->resize(out_shape);
                    }
                    output->fillToCPU(out);
                    objs_.emplace_back(
                        output->as_storage_buffer(m_dev_, m_cmd_));
                    output->copyToGPUDeferred(m_cmd_);
                } else if constexpr (std::is_same_v<T, float> ||
                                     std::is_same_v<T, uint16_t>) {
                    std::vector<float> out(ivals.size());
                    for (size_t i = 0; i < ivals.size(); ++i) {
                        out[i] = static_cast<float>(ivals[i]);
                    }
                    auto output = core::as_tensor<T>(outputs[0]);
                    if (output->num_elements() != total_elems(out_shape)) {
                        output->resize(out_shape);
                    }
                    output->fillFP32ToCPU(out);
                    objs_.emplace_back(
                        output->as_storage_buffer(m_dev_, m_cmd_));
                    output->copyToGPUDeferred(m_cmd_);
                } else {
                    throw std::runtime_error(
                        std::string("Range: output dtype ") +
                        core::elem_name(outputs[0]->elem_kind()) +
                        " cannot hold an integer range");
                }
            });
            if (host_shape_enabled())
                outputs[0]->set_host_authoritative();
            return;
        }

        // The fp path is generated on the host too, for the same reason the
        // integer one is: the kernel cannot express it. range.comp binds all
        // three params as `int data[]` and does `float start =
        // float(uStart.data[0])`, i.e. it reinterprets the fp32/fp16 bit
        // pattern as an integer (1.0f is 0x3F800000 -> 1.065e9), and the host
        // truncated the params to int64_t besides. Both widths now come out of
        // one generator, so they agree by construction. range.comp stays in
        // the archive (the converter still records a shader per node) but is
        // no longer dispatched by this op.
        {
            std::vector<float> out(static_cast<size_t>(inums));
            for (int i = 0; i < inums; ++i) {
                out[static_cast<size_t>(i)] = static_cast<float>(
                    f_start + static_cast<double>(i) * f_delta);
            }
            const auto &odt = outputs[0]->dtype();
            if (odt == typeid(uint16_t)) {
                auto output = core::as_tensor<uint16_t>(outputs[0]);
                if (output->num_elements() != total_elems(out_shape)) {
                    output->resize(out_shape);
                }
                output->fillFP32ToCPU(out);
                objs_.emplace_back(output->as_storage_buffer(m_dev_, m_cmd_));
                output->copyToGPUDeferred(m_cmd_);
            } else if (odt == typeid(float)) {
                auto output = core::as_tensor<float>(outputs[0]);
                if (output->num_elements() != total_elems(out_shape)) {
                    output->resize(out_shape);
                }
                output->fillFP32ToCPU(out);
                objs_.emplace_back(output->as_storage_buffer(m_dev_, m_cmd_));
                output->copyToGPUDeferred(m_cmd_);
            } else {
                throw std::runtime_error(
                    std::string("Range: output dtype ") +
                    core::elem_name(outputs[0]->elem_kind()) +
                    " cannot hold a floating-point range");
            }
            if (host_shape_enabled())
                outputs[0]->set_host_authoritative();
            return;
        }
    }

    range::GpuRangeParam param_;
};

} // namespace ops
} // namespace vkop
#endif // OPS_RANGE_HPP_
