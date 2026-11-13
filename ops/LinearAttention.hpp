// Copyright 2025 @junka
#ifndef OPS_LINEAR_ATTENTION_HPP_
#define OPS_LINEAR_ATTENTION_HPP_

#include "Operator.hpp"
#include "ops/BufferBase.hpp"
#include "ops/PimplFacade.hpp"
#include <cstring>

namespace vkop {
namespace ops {

// Gated delta rule (linear attention), recurrent form. This is the core of
// Qwen3.5's `Qwen3_5GatedDeltaNet`, matching HF's
// `torch_recurrent_gated_delta_rule` for a single sequence:
//
//     qn = l2norm(q) / sqrt(Dk)        kn = l2norm(k)
//     S  = S * exp(g)                  (g <= 0; the decay)
//     kv = (S * kn).sum(dk)            [Dv]
//     delta = (v - kv) * beta
//     S  = S + kn (x) delta            (rank-1, the delta rule)
//     out = (S * qn).sum(dk)           [Dv]
//
// The projections, the causal conv, the A/dt_bias decay and the gated RMSNorm
// stay ordinary graph ops; this op is only the recurrence, because that is the
// part no existing kernel expresses.
//
// Interface:
//   inputs[0] q        [Hk, S, Dk] fp32 (NOT expanded: the kernel indexes
//                       value head h to key head h / (Hv/Hk), which is
//                       repeat_interleave's layout without the copy)
//   inputs[1] k        [Hk, S, Dk] fp32
//   inputs[2] v        [H, S, Dv] fp32
//   inputs[3] g        [H, S]     fp32, log-space decay (<= 0)
//   inputs[4] beta     [H, S]     fp32
//   inputs[5] state_in [H, Dk, Dv] fp32 - the recurrent state, a graph input
//   outputs[0] out        [H, S, Dv] fp32
//   outputs[1] state_out  [H, Dk, Dv] fp32 - may alias state_in; each lane
//                          rewrites only the column it read.
//
// Buffer-only. q/k/v/g/beta and `out` follow the tensor dtype (three builds:
// fp32, native fp16, packed half2); the recurrent STATE stays fp32 in every
// build, because HF converts the whole recurrence to fp32 and a half state
// would drift over a long decode.
class LinearAttentionBuffer : public BufferFactory {
  public:
    explicit LinearAttentionBuffer(int fp16)
        : BufferFactory(OpType::LINEAR_ATTENTION,
                        std::vector<VkDescriptorType>{
                            DESCRIPTOR_TYPE_STORAGE, // out
                            DESCRIPTOR_TYPE_STORAGE, // q
                            DESCRIPTOR_TYPE_STORAGE, // k
                            DESCRIPTOR_TYPE_STORAGE, // v
                            DESCRIPTOR_TYPE_STORAGE, // g
                            DESCRIPTOR_TYPE_STORAGE, // beta
                            DESCRIPTOR_TYPE_STORAGE, // state in
                            DESCRIPTOR_TYPE_STORAGE, // state out
                        },
                        sizeof(LinearAttentionPC), fp16) {}

  private:
    bool supports_bda_variant() const override { return true; }
    size_t bda_pc_size() const override { return sizeof(LinearAttentionBdaPC); }

    void execute(
        const std::vector<std::shared_ptr<core::ITensor>> &inputs,
        const std::vector<std::shared_ptr<core::ITensor>> &outputs) override {
        if (inputs.size() < 6 || outputs.empty()) {
            throw std::runtime_error(
                "LinearAttention: needs q, k, v, g, beta, state");
        }
        for (int i = 0; i < 6; i++) {
            if (!inputs[i]) {
                throw std::runtime_error("LinearAttention: a required input is "
                                         "unbound");
            }
        }
        // Accept the exporter's layout: [B, S, H, D] with B == 1. HF keeps a
        // batch axis through the recurrence, and the graph is batch-1 (the
        // Conv1d custom op assumes it too), so the op accepts rank 4 and
        // treats the leading axis as 1 rather than making the exporter drip a
        // Squeeze/Unsqueeze pair around every layer.
        const auto q_shape = inputs[0]->getShape();
        const bool rank4 = q_shape.size() == 4;
        if (!rank4 && q_shape.size() != 3) {
            throw std::runtime_error(
                "LinearAttention: q must be [B,S,H,D] (B==1) or [S,H,D]");
        }
        if (rank4 && q_shape[0] != 1) {
            throw std::runtime_error(
                "LinearAttention: batch must be 1 (got " +
                std::to_string(q_shape[0]) + ")");
        }
        // rank 4 is [B,S,H,D]; rank 3 is [H,S,D] -- the layouts differ in the
        // ORDER of the first two axes, not just in a missing leading one.
        const int kv_heads = q_shape[rank4 ? 2 : 0];
        const int seq = q_shape[rank4 ? 1 : 1];
        const int dk = q_shape[rank4 ? 3 : 2];
        const auto v_shape = inputs[2]->getShape();
        if (v_shape.size() != q_shape.size() || v_shape[1] != seq) {
            throw std::runtime_error(
                "LinearAttention: v must be [B,S,Hv,Dv] matching q's S");
        }
        const int heads = v_shape[rank4 ? 2 : 0];
        const int dv = v_shape[rank4 ? 3 : 2];
        if (dk <= 0 || dv <= 0 || heads <= 0 || kv_heads <= 0 || seq <= 0) {
            throw std::runtime_error("LinearAttention: empty extent");
        }
        // Validity of the head ratio, checked on the host so the shader's
        // integer divide cannot pick a key head out of range.
        if (heads % kv_heads != 0) {
            throw std::runtime_error(
                "LinearAttention: Hv must be a multiple of Hk (got " +
                std::to_string(heads) + " and " + std::to_string(kv_heads) +
                ")");
        }
        if (dk > 128 || dv > 128) {
            throw std::runtime_error(
                "LinearAttention: head dims above 128 are not supported (the "
                "workgroup is 128 lanes and the state column lives in "
                "registers)");
        }

        // Data buffers follow each tensor's dtype; the state stays fp32.
        const std::vector<int> o_shape =
            rank4 ? std::vector<int>{1, seq, heads, dv}
                  : std::vector<int>{heads, seq, dv};
        std::shared_ptr<VulkanBuffer> out_buf, q_buf, k_buf, v_buf, g_buf,
            b_buf;
        dispatch_by_dtype(outputs[0]->dtype(), [&](auto tag) {
            using T = decltype(tag);
            auto out = core::as_tensor<T>(outputs[0]);
            if (out->num_elements() != heads * seq * dv) {
                out->resize(o_shape);
            } else if (out->getShape() != o_shape) {
                out->reshape_view(o_shape);
            }
            out_buf = bind_ssbo<T>(outputs[0], /*is_output=*/true);
        });
        dispatch_by_dtype(inputs[0]->dtype(), [&](auto tag) {
            using T = decltype(tag);
            q_buf = bind_ssbo<T>(inputs[0], false);
        });
        dispatch_by_dtype(inputs[1]->dtype(), [&](auto tag) {
            using T = decltype(tag);
            k_buf = bind_ssbo<T>(inputs[1], false);
        });
        dispatch_by_dtype(inputs[2]->dtype(), [&](auto tag) {
            using T = decltype(tag);
            v_buf = bind_ssbo<T>(inputs[2], false);
        });
        dispatch_by_dtype(inputs[3]->dtype(), [&](auto tag) {
            using T = decltype(tag);
            g_buf = bind_ssbo<T>(inputs[3], false);
        });
        dispatch_by_dtype(inputs[4]->dtype(), [&](auto tag) {
            using T = decltype(tag);
            b_buf = bind_ssbo<T>(inputs[4], false);
        });

        const std::vector<int> s_shape{heads, dk, dv};
        auto st_in = core::as_tensor<float>(inputs[5]);
        if (!st_in) {
            throw std::runtime_error(
                "LinearAttention: state_in is not an fp32 tensor");
        }
        if (st_in->num_elements() != heads * dk * dv) {
            st_in->resize(s_shape);
        }
        auto state_in_buf = bind_ssbo<float>(inputs[5], /*is_output=*/false);
        // state_out may BE state_in (present aliases past). Bound as an output
        // so the storage helper marks it written; the kernel's lane writes
        // only the column it has already read.
        std::shared_ptr<VulkanBuffer> state_out_buf;
        if (outputs.size() > 1 && outputs[1]) {
            auto so = core::as_tensor<float>(outputs[1]);
            if (!so) {
                throw std::runtime_error(
                    std::string("LinearAttention: state_out is not an fp32 "
                                "tensor (got ") +
                    outputs[1]->dtype().name() + ", kind " +
                    std::to_string(static_cast<int>(outputs[1]->elem_kind())) +
                    ", file said conv_state_out/recurrent_state_out fp32, "
                    "rank=" + std::to_string(outputs[1]->getShape().size()) +
                    ")");
            }
            if (so->num_elements() != heads * dk * dv) {
                so->resize(s_shape);
            }
            state_out_buf = bind_ssbo<float>(outputs[1], /*is_output=*/true);
        } else {
            state_out_buf = state_in_buf;
        }

        LinearAttentionPC pc{};
        pc.heads = heads;
        pc.kv_heads = kv_heads;
        pc.seq = seq;
        pc.dk = dk;
        pc.dv = dv;
        pc.eps_bits = eps_bits();
        // Rank 4 is the exporter's [1, S, H, D]: a token's heads are adjacent
        // in memory. Rank 3 is the hand-built [H, S, D] the tests use.
        pc.seq_major = rank4 ? 1 : 0;

        // One workgroup per head; the lanes are its value columns.
        const int groups = heads;
        if (bda_ready() && out_buf && q_buf && k_buf && v_buf && g_buf &&
            b_buf && state_in_buf && state_out_buf &&
            out_buf->hasDeviceAddress() && q_buf->hasDeviceAddress() &&
            k_buf->hasDeviceAddress() && v_buf->hasDeviceAddress() &&
            g_buf->hasDeviceAddress() && b_buf->hasDeviceAddress() &&
            state_in_buf->hasDeviceAddress() &&
            state_out_buf->hasDeviceAddress()) {
            LinearAttentionBdaPC bpc{};
            std::memcpy(&bpc, &pc, offsetof(LinearAttentionBdaPC, out_addr));
            bpc.out_addr = out_buf->getDeviceAddress();
            bpc.q_addr = q_buf->getDeviceAddress();
            bpc.k_addr = k_buf->getDeviceAddress();
            bpc.v_addr = v_buf->getDeviceAddress();
            bpc.g_addr = g_buf->getDeviceAddress();
            bpc.beta_addr = b_buf->getDeviceAddress();
            bpc.state_in_addr = state_in_buf->getDeviceAddress();
            bpc.state_out_addr = state_out_buf->getDeviceAddress();
            submit_bda(&bpc, sizeof(bpc), groups, 1, 1);
            return;
        }
        submit(&pc, groups, 1, 1);
    }

    // l2norm epsilon, the value HF's `l2norm(x, eps=1e-6)` uses, carried as
    // float bits so the push constant stays all-int.
    static int eps_bits() {
        const float eps = 1e-6F;
        int bits;
        std::memcpy(&bits, &eps, sizeof(bits));
        return bits;
    }
};

// PIMPL façade: buffer-only, so both branches build the buffer leaf.
class LinearAttention : public PimplFacade {
  public:
    LinearAttention(int fp16, bool /*backend_buffer*/)
        : PimplFacade(OpType::LINEAR_ATTENTION) {
        impl_ = std::make_unique<LinearAttentionBuffer>(fp16);
    }
};

} // namespace ops
} // namespace vkop
#endif // OPS_LINEAR_ATTENTION_HPP_
