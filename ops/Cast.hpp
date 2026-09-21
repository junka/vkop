// Copyright 2026 @junka
#ifndef OPS_CAST_HPP_
#define OPS_CAST_HPP_

#include "ops/BufferBase.hpp"
#include <cstdio>
#include <cstdlib>

extern "C" {
extern unsigned char buffer_cast_spv[];
extern unsigned int buffer_cast_spv_len;
}

namespace vkop {
namespace ops {

namespace cast {
struct alignas(16) CastPC {
    int mode;  // 0 = fp32->fp16, 1 = fp16->fp32
    int total; // input element count
    int _pad0;
    int _pad1;
};
} // namespace cast

// SSBO-only op: ONNX Cast (fp32 <-> fp16). The model has 227 Cast nodes
// converting between float32 (to=1) and float16 (to=10); there are no
// int64 casts. Every input is GPU-produced (fp16 hidden states, fp32 rotary
// products), so this must be a GPU shader op — reading back a GPU-produced
// input during execute() would see stale data (the runtime records all
// command buffers before submitting any).
class Cast : public BufferFactory {
  public:
    explicit Cast()
        : BufferFactory(OpType::CAST, buffer_cast_spv, buffer_cast_spv_len,
                        {DESCRIPTOR_TYPE_STORAGE, DESCRIPTOR_TYPE_STORAGE},
                        sizeof(cast::CastPC)) {}

    // The int64->fp32 Cast reads back position-id data every round. In decode,
    // position_ids = [v,v,v] where v = current_token_position, growing by +1
    // each round. LINEAR_GROW predicts v+1 and skips the readback, verifying
    // every 8 rounds. STABLE handles the case where the value is constant.
    // invalidate_shape_cache() resets at phase boundaries.
    enum class LearnState {
        LEARNING,
        CONFIRMING,
        STABLE,
        DYNAMIC,
        LINEAR_GROW
    };
    LearnState cast_state_ = LearnState::LEARNING;
    std::vector<int64_t> cached_in_;
    static constexpr int LINEAR_VERIFY_INTERVAL = 8;
    int linear_rounds_ = 0;

    void invalidate_shape_cache() override {
        cast_state_ = LearnState::LEARNING;
        cached_in_.clear();
        linear_rounds_ = 0;
    }

    void setAttribute(const std::unordered_map<std::string, std::string>
                          &attributes) override {
        if (attributes.find("to") != attributes.end()) {
            to_ = std::stol(attributes.at("to"));
        }
    }

  private:
    void execute(
        const std::vector<std::shared_ptr<core::ITensor>> &inputs,
        const std::vector<std::shared_ptr<core::ITensor>> &outputs) override {
        auto shape = inputs[0]->getShape();
        int total = total_elems(shape);

        // int64 -> fp32 cast (the rotary position-id Cast). int64 tensors are
        // CPU-resident during recording (shape-meta domain), so cast on the
        // host: read the int64 input, write fp32 values, resize + re-upload.
        // The buffer_cast shader only handles fp32<->fp16, not int64.
        if (inputs[0]->dtype() == typeid(int64_t) &&
            outputs[0]->dtype() == typeid(float)) {
            // LINEAR_GROW fast path: predict each element +1, skip readback.
            if (cast_state_ == LearnState::LINEAR_GROW &&
                linear_rounds_ < LINEAR_VERIFY_INTERVAL) {
                linear_rounds_++;
                std::vector<int64_t> predicted = cached_in_;
                for (auto &v : predicted)
                    v += 1;
                cached_in_ = predicted;
                std::vector<float> out(total);
                for (int i = 0; i < total && i < (int)predicted.size(); ++i)
                    out[i] = static_cast<float>(predicted[i]);
                auto output = core::as_tensor<float>(outputs[0]);
                output->resize(shape);
                output->fillToCPU(out);
                objs_.emplace_back(output->as_storage_buffer(m_dev_, m_cmd_));
                output->copyToGPUDeferred(m_cmd_);
                if (std::getenv("VKOP_RB_TRACE")) {
                    fprintf(
                        stderr,
                        "[rbtrace] Cast i64->fp32 LINEAR_GROW skip (v=%lld)\n",
                        predicted.empty() ? -1LL : (long long)predicted[0]);
                }
                return;
            }

            auto src = core::as_tensor<int64_t>(inputs[0]);
            // Unconditional readback: a cross-round-recycled GPU input may
            // have stale CPU data_ (see SqueezeUnsqueeze/ScatterElements fix).
            src->copyToCPU(m_cmdpool_);
            int src_avail = src->num_elements();
            std::vector<int64_t> in_vec(src_avail); // capture for state machine
            for (int i = 0; i < src_avail; ++i)
                in_vec[i] = (*src)[i];
            if (std::getenv("VKOP_RB_TRACE")) {
                fprintf(stderr, "[rbtrace] Cast i64->fp32 shape=");
                for (int d : shape)
                    fprintf(stderr, "%d,", d);
                fprintf(stderr, " total=%d vals=", total);
                for (int i = 0; i < src_avail && i < 8; ++i)
                    fprintf(stderr, "%lld,", (long long)in_vec[i]);
                fprintf(stderr, " state=%d\n", static_cast<int>(cast_state_));
            }

            // Advance the learn state machine. Detect uniform +1 growth across
            // all elements (the position-id pattern).
            auto all_plus_one = [](const std::vector<int64_t> &prev,
                                   const std::vector<int64_t> &cur) {
                if (prev.size() != cur.size() || prev.empty())
                    return false;
                for (size_t i = 0; i < prev.size(); ++i)
                    if (cur[i] != prev[i] + 1)
                        return false;
                return true;
            };
            if (cast_state_ == LearnState::LEARNING) {
                cached_in_ = in_vec;
                cast_state_ = LearnState::CONFIRMING;
            } else if (cast_state_ == LearnState::CONFIRMING) {
                if (in_vec == cached_in_) {
                    cast_state_ = LearnState::STABLE;
                } else if (all_plus_one(cached_in_, in_vec)) {
                    cast_state_ = LearnState::LINEAR_GROW;
                    cached_in_ = in_vec;
                    linear_rounds_ = 0;
                } else {
                    cast_state_ = LearnState::DYNAMIC;
                    cached_in_ = in_vec;
                }
            } else if (cast_state_ == LearnState::LINEAR_GROW) {
                // Verify round: predicted was cached_in_+1; check actual.
                std::vector<int64_t> expected = cached_in_;
                for (auto &v : expected)
                    v += 1;
                if (in_vec == expected) {
                    cached_in_ = in_vec;
                    linear_rounds_ = 0;
                } else if (all_plus_one(cached_in_, in_vec)) {
                    cached_in_ = in_vec;
                    linear_rounds_ = 0;
                } else {
                    cast_state_ = LearnState::DYNAMIC;
                    cached_in_ = in_vec;
                }
            } else if (cast_state_ == LearnState::STABLE) {
                if (in_vec != cached_in_) {
                    if (all_plus_one(cached_in_, in_vec)) {
                        cast_state_ = LearnState::LINEAR_GROW;
                        cached_in_ = in_vec;
                        linear_rounds_ = 0;
                    } else {
                        cast_state_ = LearnState::DYNAMIC;
                        cached_in_ = in_vec;
                    }
                }
            } else { // DYNAMIC
                cached_in_ = in_vec;
            }

            std::vector<float> out(total);
            for (int i = 0; i < total && i < src_avail; ++i) {
                out[i] = static_cast<float>((*src)[i]);
            }
            auto output = core::as_tensor<float>(outputs[0]);
            output->resize(shape);
            output->fillToCPU(out);
            objs_.emplace_back(output->as_storage_buffer(m_dev_, m_cmd_));
            output->copyToGPUDeferred(m_cmd_);
            return;
        }

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

        cast::CastPC pc{};
        // mode 0 (f32->f16): each thread packs two fp32 words into one half2
        // output word. mode 1 (f16->f32): each thread unpacks one half2 input
        // word into two fp32 output words. Either way one thread per word.
        pc.mode = (inputs[0]->dtype() == typeid(uint16_t)) ? 1 : 0;
        pc.total = total;
        submit(&pc, UP_DIV((total + 1) / 2, 256), 1, 1);
    }

    int to_ = 0;
};

} // namespace ops
} // namespace vkop
#endif // OPS_CAST_HPP_
