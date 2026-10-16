// Copyright 2026 @junka
#ifndef OPS_ATTENTION_HPP_
#define OPS_ATTENTION_HPP_

#include "ops/BufferBase.hpp"
#include "ops/PimplFacade.hpp"
#include <memory>
#include <string>
#include <unordered_map>

extern "C" {
extern unsigned char buffer_attention_spv[];
extern unsigned int buffer_attention_spv_len;
extern unsigned char buffer_attention_fp16_spv[];
extern unsigned int buffer_attention_fp16_spv_len;
}

namespace vkop {
namespace ops {

// Push constant for shaders/buffer/attention.comp. `kv_len` is deliberately in
// the PC: the record-once replay fingerprint hashes the PC bytes, and on the
// decode path kv_len is the ONLY thing that changes round to round (handles and
// the dispatch grid are constant) — without it the runtime would replay a
// stale recording. See Operator::fingerprint_matches.
struct alignas(16) AttentionPC {
    float scale;  // 1/sqrt(head_dim), folded by the converter
    int batch;    // B
    int heads;    // H (query heads)
    int kv_heads; // Hkv
    int seq;      // S (query rows per batch)
    int kv_len;   // kv (key/value positions)
    int head_dim; // D
    int rep;      // H / Hkv (GQA group size)
    int causal;   // 0/1
    int has_mask; // 0/1
    int _pad0;
    int _pad1;
};
static_assert(sizeof(AttentionPC) <= 128, "AttentionPC PC overflow");

// Fused attention: P = softmax(Q·Kᵀ*scale + mask); Y = P·V, one dispatch.
// Replaces the converter's MatMul(QKᵀ) -> [*scale +bias] -> Softmax ->
// MatMul(·V) chain (fuse_into_attention). GQA is served without materializing
// repeated K/V (h_kv = h / rep). Buffer-backend only.
//
// Bindings: 0=Y, 1=Q, 2=K, 3=V, 4=mask (dummy when absent). All tensors share
// one dtype (fp16 on the LLM path, fp32 in tests). One workgroup per query row
// (B*H*S rows); Q is staged in shared memory and the KV row is swept in
// TILE_KV chunks with a running max/sum, so kv is unbounded and shared memory
// stays ~3 KB. The host gates head_dim (even, <= 256). Batch/seq/kv_len are
// read from the bound tensors at execute() time, so symbolic dims in the
// converted graph are fine — only head_dim/heads must be statically known for
// the shader's compile-time envelope.
class AttentionBuffer : public BufferFactory {
  public:
    explicit AttentionBuffer(int fp16)
        : BufferFactory(OpType::ATTENTION,
                        fp16 ? buffer_attention_fp16_spv : buffer_attention_spv,
                        fp16 ? buffer_attention_fp16_spv_len
                             : buffer_attention_spv_len,
                        std::vector<VkDescriptorType>{
                            DESCRIPTOR_TYPE_STORAGE, DESCRIPTOR_TYPE_STORAGE,
                            DESCRIPTOR_TYPE_STORAGE, DESCRIPTOR_TYPE_STORAGE,
                            DESCRIPTOR_TYPE_STORAGE},
                        sizeof(AttentionPC), fp16),
          fp16_(fp16) {
        update_after_bind_ = true;
    }

    void setAttribute(const std::unordered_map<std::string, std::string>
                          &attributes) override {
        if (attributes.find("scale") != attributes.end()) {
            scale_ = std::stof(attributes.at("scale"));
        }
        if (attributes.find("num_heads") != attributes.end()) {
            num_heads_ =
                static_cast<int>(std::stol(attributes.at("num_heads")));
        }
        if (attributes.find("num_kv_heads") != attributes.end()) {
            num_kv_heads_ =
                static_cast<int>(std::stol(attributes.at("num_kv_heads")));
        }
        if (attributes.find("causal") != attributes.end()) {
            causal_ = static_cast<int>(std::stol(attributes.at("causal")));
        }
    }

  private:
    // (B, H, S, D) -> a 4-element shape check helper.
    static int dim(const std::vector<int> &s, int i) {
        return (i >= 0 && i < static_cast<int>(s.size())) ? s[i] : 1;
    }

    void execute(
        const std::vector<std::shared_ptr<core::ITensor>> &inputs,
        const std::vector<std::shared_ptr<core::ITensor>> &outputs) override {
        // Inputs: 0=Q, 1=K, 2=V, 3=mask (optional).
        auto q_shape = inputs[0]->getShape();
        auto k_shape = inputs[1]->getShape();
        if (q_shape.size() != 4 || k_shape.size() != 4) {
            throw std::runtime_error(
                "Attention: Q and K must be rank-4 [B,H,S,D] / [B,Hkv,kv,D]");
        }
        int batch = dim(q_shape, 0);
        int heads = dim(q_shape, 1);
        int seq = dim(q_shape, 2);
        int head_dim = dim(q_shape, 3);
        int kv_heads = dim(k_shape, 1);
        int kv_len = dim(k_shape, 2);

        // GQA: kv_heads divides heads; group-contiguous h_kv = h / rep.
        if (kv_heads <= 0 || heads % kv_heads != 0) {
            throw std::runtime_error(
                "Attention: num_heads must be a multiple of num_kv_heads");
        }
        // head_dim caps are the shader's compile-time envelope (MAX_HD, and the
        // half2 store needs an even dim). kv_len is NOT capped: the kernel
        // sweeps the KV row in TILE_KV chunks with a running max/sum, so shared
        // memory no longer scales with the context. A violation is a hard
        // error, never a silent wrong answer.
        if (head_dim <= 0 || head_dim > 256 || (head_dim % 2) != 0) {
            throw std::runtime_error(
                "Attention: head_dim must be even and <= 256");
        }
        if (kv_len <= 0) {
            throw std::runtime_error("Attention: kv_len must be > 0");
        }
        const bool has_mask = inputs.size() > 3 && inputs[3] != nullptr;

        // Y has Q's shape; resize if the runtime made a placeholder.
        dispatch_by_dtype(outputs[0]->dtype(), [&](auto type_tag) {
            using T = decltype(type_tag);
            auto output = core::as_tensor<T>(outputs[0]);
            if (output->num_elements() != total_elems(q_shape)) {
                output->resize(q_shape);
            }
            bind_ssbo<T>(outputs[0], /*is_output=*/true); // binding 0
        });
        dispatch_by_dtype(inputs[0]->dtype(), [&](auto type_tag) {
            using T = decltype(type_tag);
            bind_ssbo<T>(inputs[0], /*is_output=*/false); // binding 1 (Q)
        });
        dispatch_by_dtype(inputs[1]->dtype(), [&](auto type_tag) {
            using T = decltype(type_tag);
            bind_ssbo<T>(inputs[1], /*is_output=*/false); // binding 2 (K)
        });
        dispatch_by_dtype(inputs[2]->dtype(), [&](auto type_tag) {
            using T = decltype(type_tag);
            bind_ssbo<T>(inputs[2], /*is_output=*/false); // binding 3 (V)
        });
        // binding 4: additive mask [B,1,S,kv], or a dummy to keep the set
        // bound.
        if (has_mask) {
            dispatch_by_dtype(inputs[3]->dtype(), [&](auto type_tag) {
                using T = decltype(type_tag);
                bind_ssbo<T>(inputs[3], /*is_output=*/false);
            });
        } else {
            objs_.emplace_back(dummy_buffer_);
        }

        AttentionPC pc{};
        pc.scale = scale_;
        pc.batch = batch;
        pc.heads = heads;
        pc.kv_heads = kv_heads;
        pc.seq = seq;
        pc.kv_len = kv_len;
        pc.head_dim = head_dim;
        pc.rep = heads / kv_heads;
        pc.causal = causal_;
        pc.has_mask = has_mask ? 1 : 0;
        // One workgroup per query row.
        submit(&pc, batch * heads * seq, 1, 1);
    }

    int fp16_;
    float scale_ = 1.0f;
    int num_heads_ = 0;    // cross-check only; shapes are authoritative
    int num_kv_heads_ = 0; // ditto
    int causal_ = 0;
};

// PIMPL façade. Buffer-only (no image impl), like RMSNorm.
class Attention : public PimplFacade {
  public:
    Attention(int fp16, bool backend_buffer) : PimplFacade(OpType::ATTENTION) {
        if (!backend_buffer) {
            throw std::runtime_error(
                "Attention is buffer-backend only (no image impl).");
        }
        impl_ =
            std::unique_ptr<Operator>(std::make_unique<AttentionBuffer>(fp16));
    }
};

} // namespace ops
} // namespace vkop
#endif // OPS_ATTENTION_HPP_
