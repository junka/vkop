// Copyright 2026 @junka
#ifndef OPS_ATTENTION_HPP_
#define OPS_ATTENTION_HPP_

#include "ops/BufferBase.hpp"
#include "ops/PimplFacade.hpp"
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <unordered_map>

namespace vkop {
namespace ops {

// Push constant for shaders/buffer/attention.comp. `kv_len` is deliberately in
// the PC: the record-once replay fingerprint hashes the PC bytes, and on the
// decode path kv_len is the ONLY thing that changes round to round (handles and
// the attention part of the dispatch grid are constant) — without it the
// runtime would replay a stale recording. Same reason `kv_past` and the two
// strides live here: they all move with kv_len. See
// Operator::fingerprint_matches.
struct alignas(16) AttentionPC {
    float scale;       // 1/sqrt(head_dim), folded by the converter
    int batch;         // B
    int heads;         // H (query heads)
    int kv_heads;      // Hkv
    int seq;           // S (query rows per batch)
    int kv_len;        // kv (total positions: kv_past + S with a cache)
    int head_dim;      // D
    int rep;           // H / Hkv (GQA group size)
    int causal;        // 0/1
    int has_mask;      // 0/1
    int mask_causal;   // 0/1: that mask is the standard causal one
    int kv_past;       // positions already in the cache (0 without one)
    int has_cache;     // 0/1
    int src_stride;    // cache_in row stride (kv pitch of that SSBO)
    int dst_stride;    // cache_out row stride
    int inplace;       // 0/1: cache_out aliases cache_in (detected, not asked)
    int ring;          // 0/1: sliding-window ring over the cache rows
    int ring_head;     // physical slot of logical row 0 (0 when ring == 0)
    float cache_scale; // fp8 cache dequant/quant scale (1.0 for an fp16 cache)
    int cache_fp8;     // 0/1: the cache holds E4M3 bytes, not Q-dtype elements
    int _pad0;
    int _pad1;
};
static_assert(sizeof(AttentionPC) <= 128, "AttentionPC PC overflow");

// Fused attention: P = softmax(Q·Kᵀ*scale + mask); Y = P·V, one dispatch.
// Replaces the converter's MatMul(QKᵀ) -> [*scale +bias] -> Softmax ->
// MatMul(·V) chain (fuse_into_attention). GQA is served without materializing
// repeated K/V (h_kv = h / rep). Buffer-backend only.
//
// Two forms, selected by the `has_cache` attribute:
//
//   no cache — bindings 0=Y, 1=Q, 2=K, 3=V, 4=mask (dummy when absent), with
//   K/V holding all kv positions ([B,Hkv,kv,D]). One workgroup per query row
//   (B*H*S rows).
//
//   cache — bindings 0=Y, 1=Q, 2=K_new, 3=V_new, 4=mask, 5=cache_in,
//   6=cache_out. K_new/V_new hold only the S fresh rows; positions
//   [0, kv_past) are read from cache_in, a [1,2,nkv,<cap>,hd] SSBO (K and V
//   stacked on dim 1). The last `ncopy` workgroups of the same dispatch write
//   cache_out (`present`) for the whole of [0, kv_len): the history copied from
//   cache_in, then the new rows. This is what replaces the graph's
//   Gather/Concat/Unsqueeze rotation, which copied the entire past every decode
//   round (7 dispatches/layer, linear in kv_len — 48-70% of a round at
//   kv_len 1024..4096). When cache_out IS cache_in (the driver gives past and
//   present one preallocated buffer) the history copy is a no-op and only the
//   fresh rows are written — see `inplace`, which is worked out from the bound
//   buffers at execute() time, never from an attribute.
//
// The host gates head_dim (even, <= 256) and the shape-level invariants and
// THROWS on a violation, never a silent wrong answer. Batch/seq/kv_len come
// from the bound tensors at execute() time, so symbolic dims in the converted
// graph are fine — only head_dim/heads must be statically known for the
// shader's compile-time envelope.
class AttentionBuffer : public BufferFactory {
  public:
    explicit AttentionBuffer(int fp16)
        : BufferFactory(OpType::ATTENTION,
                        std::vector<VkDescriptorType>{
                            DESCRIPTOR_TYPE_STORAGE, DESCRIPTOR_TYPE_STORAGE,
                            DESCRIPTOR_TYPE_STORAGE, DESCRIPTOR_TYPE_STORAGE,
                            DESCRIPTOR_TYPE_STORAGE, DESCRIPTOR_TYPE_STORAGE,
                            DESCRIPTOR_TYPE_STORAGE},
                        sizeof(AttentionPC), fp16) {
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
        if (attributes.find("mask_causal") != attributes.end()) {
            mask_causal_ =
                static_cast<int>(std::stol(attributes.at("mask_causal")));
        }
        if (const char *e = getenv("VKOP_ATTN_MASK_CAUSAL")) {
            mask_causal_ = (e[0] == '1') ? 1 : 0;
        }
        if (attributes.find("has_cache") != attributes.end()) {
            has_cache_ =
                static_cast<int>(std::stol(attributes.at("has_cache")));
        }
        if (attributes.find("has_ring") != attributes.end()) {
            ring_ = static_cast<int>(std::stol(attributes.at("has_ring")));
        }
        // fp8 cache: the cache SSBO holds E4M3 bytes and the op does the
        // decode/encode itself, which is what lets an fp8 cache be written in
        // place (past == present) instead of through a fp16 intermediate. The
        // converter sets this only after folding the graph's Q/DQ pair into
        // this op, so an attribute without that fold cannot occur.
        if (attributes.find("cache_scale") != attributes.end()) {
            cache_scale_ = std::stof(attributes.at("cache_scale"));
        }
    }

  private:
    // (B, H, S, D) -> a 4-element shape check helper.
    static int dim(const std::vector<int> &s, int i) {
        return (i >= 0 && i < static_cast<int>(s.size())) ? s[i] : 1;
    }

    // Rows a KV SSBO can hold at the packed layout: the buffer is preallocated
    // to the session's max context, so its capacity — not the tensor's logical
    // kv_len — is the row pitch the kernel must use. That is exactly what makes
    // appending cheap: the rows already written never move as kv_len grows.
    // The capacity, not the logical length, is the answer even when the buffer
    // is the smaller of the two: `at_least` is the caller's logical floor, and
    // returning it here would make the kernel address at a pitch the allocation
    // does not have. `elem_bytes` is the element width the shader binds (2 for
    // the fp16 half2 SSBO, 4 for fp32); anything else means the cache is not
    // the same dtype as Q, which the caller rejects before we get here.
    static int row_capacity(const std::shared_ptr<VulkanBuffer> &buf,
                            int elem_bytes, int row_elems, int at_least) {
        if (!buf || elem_bytes <= 0 || row_elems <= 0) {
            return at_least;
        }
        const int64_t rows =
            static_cast<int64_t>(buf->getSize()) / (elem_bytes * row_elems);
        return static_cast<int>(rows > 0 ? rows : at_least);
    }

    void execute(
        const std::vector<std::shared_ptr<core::ITensor>> &inputs,
        const std::vector<std::shared_ptr<core::ITensor>> &outputs) override {
        // Inputs: 0=Q, 1=K, 2=V, then [cache,] [mask].
        auto q_shape = inputs[0]->getShape();
        auto k_shape = inputs[1]->getShape();
        if (q_shape.size() != 4 || k_shape.size() != 4) {
            throw std::runtime_error(
                "Attention: Q and K must be rank-4 [B,H,S,D] / [B,Hkv,kv,D]");
        }
        const bool has_cache = has_cache_ != 0;
        int batch = dim(q_shape, 0);
        int heads = dim(q_shape, 1);
        int seq = dim(q_shape, 2);
        int head_dim = dim(q_shape, 3);
        int kv_heads = dim(k_shape, 1);
        int kv_new = dim(k_shape, 2); // fresh rows: S with a cache, kv without

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

        int kv_past = 0;
        int cache_idx = -1;
        if (has_cache) {
            cache_idx = 3;
            if (inputs.size() < 4 || !inputs[3]) {
                throw std::runtime_error(
                    "Attention: has_cache is set but input 3 is not bound");
            }
            auto c_shape = inputs[3]->getShape();
            if (c_shape.size() != 5 || dim(c_shape, 0) != 1 ||
                dim(c_shape, 1) != 2) {
                throw std::runtime_error(
                    "Attention: cache must be rank-5 [1,2,Hkv,kv,D]");
            }
            if (dim(c_shape, 2) != kv_heads || dim(c_shape, 4) != head_dim) {
                throw std::runtime_error(
                    "Attention: cache head/dim axes disagree with K");
            }
            if (batch != 1) {
                throw std::runtime_error(
                    "Attention: the KV cache form requires batch == 1");
            }
            if (kv_new != seq) {
                throw std::runtime_error(
                    "Attention: with a cache, K must hold exactly seq rows");
            }
            kv_past = dim(c_shape, 3);
            if (kv_past < 0) {
                throw std::runtime_error("Attention: cache kv dim is unknown");
            }
        }
        const int kv_len = kv_past + kv_new;
        if (kv_len <= 0) {
            throw std::runtime_error("Attention: kv_len must be > 0");
        }
        if (has_cache && outputs.size() < 2) {
            throw std::runtime_error(
                "Attention: the KV cache form needs a present output");
        }
        const int mask_idx = has_cache ? 4 : 3;
        // An int32 tensor in the mask slot is the ring head, not a mask: the
        // converter appends kv_ring_head after the mask, so a graph that has
        // the ring but no additive bias puts it at index 4. The mask is
        // fp16/fp32, the ring head int32, so the dtype tells them apart.
        const bool has_mask = static_cast<int>(inputs.size()) > mask_idx &&
                              inputs[mask_idx] &&
                              inputs[mask_idx]->dtype() != typeid(int);
        // Q/K/V/mask must all be Q's element type: the dispatch below picks ONE
        // T from Q and binds every operand through as_tensor<T>(), which
        // returns null on a mismatch — and bind_ssbo dereferences it, so a
        // graph whose dtype propagation left K or V at another width segfaults
        // here instead of reporting. (That is reachable: an fp8 KV graph whose
        // Quantize/Dequantize pair has been ablated through
        // VKOP_NO_FUSIONS=fuse_kv_cache_append leaves the fp8 bytes flowing
        // into the concat that feeds K.) The cache is deliberately exempt: it
        // has its own dispatch below and may be 1-byte fp8 while Q/K/V stay
        // fp16.
        for (int i = 1; i <= 2; ++i) {
            if (inputs[i] && inputs[i]->dtype() != inputs[0]->dtype()) {
                throw std::runtime_error(
                    std::string("Attention: K/V must have Q's element type — "
                                "K/V at a different width would bind a null "
                                "buffer (Q is ") +
                    core::elem_name(inputs[0]->elem_kind()) + ", K/V is " +
                    core::elem_name(inputs[i]->elem_kind()) + ")");
            }
        }
        if (has_mask && inputs[mask_idx]->dtype() != inputs[0]->dtype()) {
            throw std::runtime_error(
                std::string("Attention: the attention bias must have Q's "
                            "element type (Q is ") +
                core::elem_name(inputs[0]->elem_kind()) + ", the bias is " +
                core::elem_name(inputs[mask_idx]->elem_kind()) + ")");
        }
        // The kernel reads the additive bias as one flat [B,S,kv] block,
        // indexed (b*S+s)*kv + j (attention.comp). A per-head [B,H,S,kv]
        // bias, a [1,1,1,kv] row broadcast, or a scalar all read the wrong
        // element — or past the end — and nothing downstream can tell.
        if (has_mask) {
            const size_t want = static_cast<size_t>(batch) *
                                static_cast<size_t>(seq) *
                                static_cast<size_t>(kv_len);
            const size_t got =
                static_cast<size_t>(total_elems(inputs[mask_idx]->getShape()));
            if (got != want) {
                throw std::runtime_error(
                    std::string("Attention: the additive bias must be the flat "
                                "[B,S,kv] form the kernel indexes as "
                                "(b*S+s)*kv+j — got ") +
                    std::to_string(got) + " elements, expected " +
                    std::to_string(want));
            }
        }
        // An fp8/fp4 cache rides its own 1-byte container through the graph's
        // Q/DQ pair; this op computes in the Q dtype, so a cache of a different
        // width is a wiring bug, not something to guess a reinterpret for. The
        // one exception is the fp8 cache this op now serves itself (see
        // cache_scale_): there the cache IS E4M3 bytes and the op does the
        // decode/encode inline, so a byte container is what it expects.
        const bool cache_fp8 =
            has_cache && inputs[3]->dtype() == typeid(int8_t);
        if (has_cache && !cache_fp8 &&
            inputs[3]->dtype() != inputs[0]->dtype()) {
            throw std::runtime_error(
                "Attention: cache dtype must match Q (quantized caches are "
                "served by the graph's Quantize/Dequantize pair)");
        }
        if (cache_fp8 && !(cache_scale_ > 0.0f)) {
            throw std::runtime_error(
                "Attention: an fp8 cache needs a positive `cache_scale`");
        }
        // The fp8 kernel copies four cache bytes per lane, so a row has to be
        // a whole number of words. Every real head_dim is a multiple of 16.
        if (cache_fp8 && (head_dim % 4) != 0) {
            throw std::runtime_error(
                "Attention: an fp8 cache needs head_dim % 4 == 0");
        }

        // Sliding-window ring. `kv_ring_head` (the int32 input the converter
        // appends after the mask) carries the physical slot of logical row 0;
        // the op is otherwise stateless per dispatch and derives `kv_past` from
        // the cache shape alone. The driver keeps kv_len <= W by shrinking the
        // cache shape before a run that would overflow — so the ring is pure
        // addressing, never a row move, and the preallocated past/present alias
        // stays intact. Read on the host: this is a 1-element int32 metadata
        // input, filled by the driver each round, not a device buffer the
        // shader indexes. Located by dtype (the mask is fp16/fp32, this is
        // int32) so it works whether or not the graph carries a mask.
        int ring = 0;
        int ring_head = 0;
        int ring_idx = -1;
        if (has_cache && ring_ != 0) {
            for (size_t k = 4; k < inputs.size(); ++k) {
                if (inputs[k] && inputs[k]->dtype() == typeid(int)) {
                    ring_idx = static_cast<int>(k);
                    break;
                }
            }
            if (ring_idx < 0) {
                throw std::runtime_error("Attention: has_ring is set but no "
                                         "int32 kv_ring_head input "
                                         "is bound");
            }
            auto ht = core::as_tensor<int>(inputs[ring_idx]);
            if (!ht || ht->num_elements() < 1) {
                throw std::runtime_error(
                    "Attention: kv_ring_head must be int32 with >= 1 element");
            }
            if (ht->has_cpu_data()) {
                ring_head = ht->data()[0];
            } else {
                ht->copyToCPU(m_cmdpool_);
                ring_head = ht->at(0);
            }
            ring = 1;
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
        pc.mask_causal = mask_causal_;
        pc.kv_past = kv_past;
        pc.has_cache = has_cache ? 1 : 0;
        pc.ring = ring;
        pc.ring_head = ring_head;
        pc.cache_scale = cache_fp8 ? cache_scale_ : 1.0f;
        pc.cache_fp8 = cache_fp8 ? 1 : 0;

        int grid = batch * heads * seq;
        dispatch_by_dtype(inputs[0]->dtype(), [&](auto type_tag) {
            using T = decltype(type_tag);
            // Y has Q's shape; resize if the runtime made a placeholder.
            auto output = core::as_tensor<T>(outputs[0]);
            if (output->num_elements() != total_elems(q_shape)) {
                output->resize(q_shape);
            }
            bind_ssbo<T>(outputs[0], /*is_output=*/true); // binding 0
            bind_ssbo<T>(inputs[0], /*is_output=*/false); // binding 1 (Q)
            bind_ssbo<T>(inputs[1], /*is_output=*/false); // binding 2 (K)
            bind_ssbo<T>(inputs[2], /*is_output=*/false); // binding 3 (V)
            if (has_mask) {
                bind_ssbo<T>(inputs[mask_idx], /*is_output=*/false); // 4
            } else {
                objs_.emplace_back(dummy_buffer_);
            }

            if (!has_cache) {
                objs_.emplace_back(dummy_buffer_); // binding 5
                objs_.emplace_back(dummy_buffer_); // binding 6
                return;
            }
            // The cache is bound as its OWN element type, not the compute
            // dtype: an fp8 cache is E4M3 bytes in an int8_t container while
            // Q/K/V/Y stay fp16, and binding it through as_tensor<T> would
            // return null and then address at the wrong pitch. Everything
            // downstream of the binding (row capacity, the copy grid) is
            // expressed in elements and elem_bytes, so the two cases share
            // one body.
            dispatch_by_dtype(inputs[cache_idx]->dtype(), [&](auto cache_tag) {
                using C = decltype(cache_tag);
                const int row_elems = 2 * kv_heads * head_dim;
                const int elem_bytes = static_cast<int>(sizeof(C));
                // The copy below counts uint words and a word holds
                // 4/elem_bytes elements, so only the cache layouts the shader
                // writes are possible here: the 2-byte packed half2, the
                // 4-byte fp32 and the 1-byte fp8. An 8-byte element would
                // truncate that ratio to 0 and divide by it -- which clang
                // constant-folds in this lambda's int64_t instantiation even
                // though no such dtype reaches here, so state the contract
                // instead.
                if (elem_bytes != 1 && elem_bytes != 2 && elem_bytes != 4) {
                    throw std::runtime_error(
                        "Attention: the fused KV cache path takes a 1-, 2- "
                        "or 4-byte element");
                }
                // cache_in: the history. Its buffer's row capacity is the pitch
                // the previous round wrote at (the same kernel), so read at
                // that.
                auto cbuf =
                    bind_ssbo<C>(inputs[cache_idx], /*is_output=*/false);
                pc.src_stride =
                    row_capacity(cbuf, elem_bytes, row_elems, kv_past);
                // cache_out: present, logical shape [1,2,Hkv,kv_len,D].
                auto pres = core::as_tensor<C>(outputs[1]);
                std::vector<int> p_shape{1, 2, kv_heads, kv_len, head_dim};
                if (pres->num_elements() != total_elems(p_shape)) {
                    pres->resize(p_shape);
                }
                auto pbuf = bind_ssbo<C>(outputs[1], /*is_output=*/true);
                pc.dst_stride =
                    row_capacity(pbuf, elem_bytes, row_elems, kv_len);
                // `present` may BE the cache: the driver hands past and present
                // one preallocated buffer (Tensor::alias_gpu_buffer), so the
                // history is already where the next round will read it and the
                // copy is pure overhead. Detected from the two bound buffers
                // rather than from an attribute, so the flag and the wiring
                // cannot disagree — same buffer skips the history copy,
                // different buffers runs it. Never an assumption that could
                // yield a silently stale cache.
                pc.inplace = (cbuf && pbuf && cbuf.get() == pbuf.get()) ? 1 : 0;
                // Ring invariants: the history must stay put (a ring only
                // rotates the addressing, so past and present have to be one
                // buffer), and the logical window must fit the physical one —
                // the whole point of the shape-shrink protocol is that kv_len
                // never exceeds W.
                if (ring) {
                    if (pc.inplace == 0) {
                        throw std::runtime_error(
                            "Attention: the KV ring requires present to alias "
                            "past (inplace); the driver must hand both one "
                            "buffer");
                    }
                    if (kv_len > pc.src_stride) {
                        throw std::runtime_error(
                            "Attention: KV ring window (kv_len=" +
                            std::to_string(kv_len) +
                            ") exceeds the cache capacity (" +
                            std::to_string(pc.src_stride) + ")");
                    }
                    if (!ring_logged_) {
                        std::fprintf(stderr, "[attn] kv ring active (W=%d%s)\n",
                                     pc.src_stride, cache_fp8 ? ", fp8" : "");
                        ring_logged_ = true;
                    }
                }
                // The copy half of the dispatch: one uint word per lane, and a
                // word holds 4/sizeof(C) elements (2 for the packed fp16 cache,
                // 4 for the fp8 one, 1 for fp32). Dividing by elem_bytes here
                // instead would under-dispatch fp32 fourfold and leave
                // `present` half-written — and over-dispatch fp8 just as badly.
                const int elems_per_word = 4 / elem_bytes;
                // Under the ring the history is already in place and only the
                // fresh rows are written — the same rows the kernel's copy
                // covers — so the dispatch is sized to them rather than to the
                // whole window.
                const int64_t copy_rows = ring ? kv_new : kv_len;
                const int64_t words = static_cast<int64_t>(row_elems) *
                                      copy_rows / elems_per_word;
                grid += static_cast<int>((words + 255) / 256);
            });
        });

        submit(&pc, grid, 1, 1);
    }

    float scale_ = 1.0F;
    int num_heads_ = 0;    // cross-check only; shapes are authoritative
    int num_kv_heads_ = 0; // ditto
    int causal_ = 0;
    int has_cache_ = 0;
    // Sliding-window ring (see execute()). Off unless the converter declares
    // it, so a graph without the ring is byte-identical to before.
    int ring_ = 0;
    bool ring_logged_ = false;
    // fp8 cache scale (see setAttribute). 0 = no fp8 cache folded in, which is
    // also the check that rejects an fp8 cache bound without one.
    float cache_scale_ = 0.0F;
    // Declared invariant: the bound mask is the standard causal one (0 at
    // j <= kv_len - seq + s, fp16 min above). It lets the kernel stop its kv
    // sweep at the query row's diagonal, which halves a causal prefill. Only
    // an assertion if the declaration is wrong: whatever builds the bias --
    // llm_chat's causal_bias(), an exporter writing the mask -- is the only
    // party that knows, so it has to say so. Set from the op attribute, with
    // an env override for A/B without a reconversion.
    int mask_causal_ = 0;
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
