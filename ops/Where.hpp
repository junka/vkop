// Copyright 2026 @junka
#ifndef OPS_WHERE_HPP_
#define OPS_WHERE_HPP_

#include "core/Tensor.hpp"
#include "ops/BufferBase.hpp"
#include "ops/Operator.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <numeric>

extern "C" {
extern unsigned char buffer_where_spv[];
extern unsigned int buffer_where_spv_len;
}
namespace vkop {
namespace ops {

// SSBO-only op: no image path. Selects elements from X or Y based on a
// boolean condition buffer.
class Where : public Operator {
  public:
    explicit Where()
        : Operator(OpType::WHERE, buffer_where_spv, buffer_where_spv_len,
                   {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    VK_DESCRIPTOR_TYPE_STORAGE_BUFFER},
                   0) {}

    // Phase-boundary reset: re-learn input stability. The int64 Where in the
    // shape-meta chain reads cond/X/Y whose values are almost always
    // round-invariant across decode (only the rare kv_len-derived one changes).
    // After two matching readbacks per input, skip its copyToCPU and reuse the
    // cached value. Mirrors ReshapeBuffer's auto-learning cache.
    void invalidate_shape_cache() override {
        for (auto &s : in_cache_state_)
            s = InLearnState::LEARNING;
    }

  private:
    void execute(
        const std::vector<std::shared_ptr<core::ITensor>> &inputs,
        const std::vector<std::shared_ptr<core::ITensor>> &outputs) override {

        std::vector<int> out_shape = outputs[0]->getShape();
        // The graph's recorded output shape is often stale for the int64
        // shape-meta chain (symbolic dims resolved to a max, not the runtime
        // value). For int64 Where, recompute the broadcasted output shape from
        // the authoritative inputs (cond/X/Y) — the CPU loop below needs the
        // true shape to broadcast against.
        if (inputs[0]->dtype() == typeid(int64_t)) {
            std::vector<int> bcast;
            for (const auto &in : inputs) {
                auto s = in->getShape();
                if (s.empty())
                    continue;
                if (bcast.empty()) {
                    bcast = s;
                } else {
                    size_t m = std::max(bcast.size(), s.size());
                    std::vector<int> nb(m, 1);
                    for (size_t i = 0; i < m; ++i) {
                        int a = (i < bcast.size()) ? bcast[bcast.size() - 1 - i]
                                                   : 1;
                        int b = (i < s.size()) ? s[s.size() - 1 - i] : 1;
                        nb[m - 1 - i] = std::max(a, b);
                    }
                    bcast = nb;
                }
            }
            if (!bcast.empty())
                out_shape = bcast;
        }
        if (out_shape.empty()) {
            auto inshape = inputs[0]->getShape();
            out_shape = inshape;
        }

        // int64 Where runs on the CPU (all 66 instances are part of the
        // shape meta-chain: cond = Equal int64, X = ConstantOfShape, Y =
        // Concat int64). cond, X, and Y are broadcast against out_shape;
        // cond nonzero selects X, else Y.
        if (inputs[0]->dtype() == typeid(int64_t)) {
            auto cond = core::as_tensor<int64_t>(inputs[0]);
            auto x = core::as_tensor<int64_t>(inputs[1]);
            auto y = core::as_tensor<int64_t>(inputs[2]);
            // Auto-learning per-input readback cache: each of cond/X/Y is a
            // tiny int64 tensor whose values are almost always round-invariant
            // across decode. After two matching readbacks, skip copyToCPU and
            // reuse the cached host vector. invalidate_shape_cache() (phase
            // boundary) resets all to LEARNING.
            read_int64_cached(cond, 0);
            read_int64_cached(x, 1);
            read_int64_cached(y, 2);
            int total = total_elems(out_shape);
            std::vector<int64_t> out(total);
            for (int i = 0; i < total; ++i) {
                int64_t cv = (*cond)[broadcast_index(inputs[0]->getShape(),
                                                     out_shape, i)];
                out[i] = (cv != 0) ? (*x)[broadcast_index(inputs[1]->getShape(),
                                                          out_shape, i)]
                                   : (*y)[broadcast_index(inputs[2]->getShape(),
                                                          out_shape, i)];
            }
            auto output = core::as_tensor<int64_t>(outputs[0]);
            output->resize(out_shape);
            output->fillToCPU(out);
            objs_.emplace_back(output->as_storage_buffer(m_dev_, m_cmd_));
            output->copyToGPUDeferred(m_cmd_);
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
        submit(nullptr, UP_DIV(total_size, 256), 1, 1);
    }

    // Read an int64 input into its host data_ with a per-input auto-learning
    // cache. STABLE inputs skip copyToCPU (the ~3.2ms submit+wait) and reuse
    // the cached host vector. The op still recomputes its output on CPU every
    // round from the (cached or freshly-read) host data — only the readback is
    // skipped, so downstream correctness is unaffected.
    void read_int64_cached(std::shared_ptr<core::Tensor<int64_t>> &t, int idx) {
        if (in_cache_state_[idx] == InLearnState::STABLE) {
            // Reuse cached host data_ (already populated on the STABLE round).
            // Restore it onto the tensor in case a prior op cleared data_.
            if (cached_in_[idx] && !cached_in_[idx]->empty()) {
                t->fillToCPU(*cached_in_[idx]);
            }
            return;
        }
        // Unconditional readback: a cross-round-recycled GPU input may have
        // stale CPU data_ (see SqueezeUnsqueeze/ScatterElements fix).
        t->copyToCPU(m_cmdpool_);
        const std::vector<int64_t> &cur = t->data();
        if (in_cache_state_[idx] == InLearnState::LEARNING) {
            *learned_in_[idx] = cur;
            in_cache_state_[idx] = InLearnState::CONFIRMING;
        } else if (in_cache_state_[idx] == InLearnState::CONFIRMING) {
            if (cur == *learned_in_[idx]) {
                in_cache_state_[idx] = InLearnState::STABLE;
                *cached_in_[idx] = cur;
            } else {
                in_cache_state_[idx] = InLearnState::DYNAMIC;
            }
        }
        // DYNAMIC: stay DYNAMIC (readback every round, no caching).
    }

    // --- per-input readback cache (runtime auto-learning) ---
    enum class InLearnState { LEARNING, CONFIRMING, STABLE, DYNAMIC };
    std::array<InLearnState, 3> in_cache_state_ = {{InLearnState::LEARNING,
                                                    InLearnState::LEARNING,
                                                    InLearnState::LEARNING}};
    std::array<std::shared_ptr<std::vector<int64_t>>, 3> learned_in_ = {
        {std::make_shared<std::vector<int64_t>>(),
         std::make_shared<std::vector<int64_t>>(),
         std::make_shared<std::vector<int64_t>>()}};
    std::array<std::shared_ptr<std::vector<int64_t>>, 3> cached_in_ = {
        {std::make_shared<std::vector<int64_t>>(),
         std::make_shared<std::vector<int64_t>>(),
         std::make_shared<std::vector<int64_t>>()}};
};

} // namespace ops
} // namespace vkop
#endif // OPS_WHERE_HPP_
