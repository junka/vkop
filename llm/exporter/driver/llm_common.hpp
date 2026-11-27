// junka @ 2026
// Pieces llm_chat and llm_serve both need, kept in one place so the two
// front-ends cannot drift: architecture inference, the embedding table, input
// fill/upload helpers, the GPU argmax registration, and DecodeInputs -- the
// per-step decode input writer that uses host-imported (zero-copy) storage
// where the device allows it and one batched upload otherwise.
//
// Everything here is header-only and has no state beyond what its callers
// own, so including it changes nothing about how a binary is linked.

#pragma once

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <fstream>
#include <functional>
#include <limits>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "core/Tensor.hpp"
#include "core/runtime.hpp"
#include "vulkan/VulkanDevice.hpp"

namespace vkop::llm {

using core::as_tensor;
using core::ElemKind;
using core::ITensor;
using core::Runtime;

// Qwen uses torch.finfo(float16).min, not -inf, as the causal-mask fill.
constexpr uint16_t FP16_MIN = 0xFBFF;  // -65504

inline float fp16_to_f32(uint16_t h) { return ITensor::fp16_to_fp32(h); }

// ------------------------------------------------------ 16-bit float kinds ----
// An --act bf16 artifact (onnx2vkop --act bf16) declares its float inputs,
// its KV cache and (unless kept) its outputs bfloat16. bf16 rides the same
// uint16_t container as fp16, so the container cannot tell the two apart:
// every host read and write below goes by the tensor's elem_kind().
//
// The host side keeps producing fp16 (embed_tokens.bin is fp16, the causal
// mask is fp16's -65504, vision features are fp16); the conversion happens
// once, where the bits enter a bf16 input: fp16 -> fp32 (exact) -> bf16 with
// round-to-nearest-even (core::float_to_bf16, the shaders' own rounding).
inline bool is_half16(ElemKind k) {
    return k == ElemKind::kFloat16 || k == ElemKind::kBFloat16;
}

inline float half_to_f32(uint16_t h, ElemKind kind) {
    return kind == ElemKind::kBFloat16 ? core::bf16_to_float(h)
                                       : fp16_to_f32(h);
}

// `n` fp16 words as `kind` words. fp16 is returned as-is (no copy); bf16 is
// converted into `scratch`.
inline const uint16_t *fp16_as_kind(ElemKind kind, const uint16_t *src,
                                    size_t n, std::vector<uint16_t> &scratch) {
    if (kind != ElemKind::kBFloat16) return src;
    scratch.resize(n);
    for (size_t i = 0; i < n; ++i)
        scratch[i] = core::float_to_bf16(fp16_to_f32(src[i]));
    return scratch.data();
}

// ------------------------------------------------------------ architecture ----
struct ModelArch {
    int hidden = 2048;
    int nkv = 8;
    int hd = 128;
    // Number of layers that carry a KV cache. Equal to the layer count for a
    // dense graph, smaller for a hybrid one (Qwen3.5-27B: 16 of 64).
    int nlayers = 28;
    // The layer index of each KV cache slot, ascending. The graph input is
    // `past_key_values_<layer>`, and for a hybrid model those numbers are
    // sparse (Qwen3.5-27B: 3, 7, ... 63). Dense models get 0..nlayers-1.
    std::vector<int> kv_layers;
    int position_ids_dims = 2;   // 2 = text RoPE, 3 = multimodal MRoPE
    bool has_deepstack = false;  // deepstack_embeds_{0,1,2}
    bool has_image_pad_mask = false;
    ElemKind cache_kind = ElemKind::kFloat16;
    // inputs_embeds' kind: kBFloat16 for an --act bf16 artifact.
    ElemKind act_kind = ElemKind::kFloat16;

    std::string kv_in(int i) const {
        return "past_key_values_" + std::to_string(kv_layers[i]);
    }
    std::string kv_out(int i) const {
        return "present_key_values_" + std::to_string(kv_layers[i]);
    }
};

// Read the architecture off the loaded graph's inputs. The cache layers are
// enumerated from the input names rather than probed 0, 1, 2, ..., because a
// hybrid decoder has no past_key_values_0.
inline ModelArch infer_arch(const std::shared_ptr<Runtime> &rt,
                            const char *tag) {
    ModelArch a;
    const std::string kv_prefix = "past_key_values_";
    for (const auto &name : rt->ListInputNames()) {
        if (name.rfind(kv_prefix, 0) != 0) continue;
        const std::string digits = name.substr(kv_prefix.size());
        // A suffix like `3_scale` belongs to a quantized-cache variant.
        if (digits.empty() ||
            digits.find_first_not_of("0123456789") != std::string::npos)
            continue;
        a.kv_layers.push_back(std::atoi(digits.c_str()));
    }
    std::sort(a.kv_layers.begin(), a.kv_layers.end());
    if (a.kv_layers.empty())
        throw std::runtime_error("model has no past_key_values_* inputs");
    a.nlayers = static_cast<int>(a.kv_layers.size());
    auto pk0 = rt->GetInput(a.kv_in(0));
    if (!pk0)
        throw std::runtime_error("model declares " + a.kv_in(0) +
                                 " but the runtime has no such input");
    const auto s = pk0->getShape();  // (B, 2, nkv, kv_len, hd)
    if (s.size() >= 3) a.nkv = static_cast<int>(s[2]);
    if (s.size() >= 5) a.hd = static_cast<int>(s[4]);
    a.cache_kind = pk0->elem_kind();
    if (a.cache_kind != ElemKind::kFloat16 &&
        a.cache_kind != ElemKind::kBFloat16 &&
        a.cache_kind != ElemKind::kFloat8E4M3FN &&
        a.cache_kind != ElemKind::kFloat8E5M2 &&
        a.cache_kind != ElemKind::kInt4 && a.cache_kind != ElemKind::kNF4) {
        std::fprintf(stderr, "[%s] cache dtype %s unsupported, using fp16\n",
                     tag, core::elem_name(a.cache_kind));
        a.cache_kind = ElemKind::kFloat16;
    }
    auto ie = rt->GetInput("inputs_embeds");
    const auto es = ie->getShape();  // (B, q, hidden)
    if (es.size() >= 3) a.hidden = static_cast<int>(es.back());
    a.act_kind = ie->elem_kind() == ElemKind::kBFloat16 ? ElemKind::kBFloat16
                                                        : ElemKind::kFloat16;
    auto pid = rt->GetInput("position_ids");
    a.position_ids_dims = pid ? static_cast<int>(pid->getShape().size()) : 2;
    a.has_deepstack = rt->GetInput("deepstack_embeds_0") != nullptr;
    a.has_image_pad_mask = rt->GetInput("image_pad_mask") != nullptr;
    return a;
}

// ---------------------------------------------------------------- embedding ----
// embed_tokens.bin is raw fp16 [vocab, hidden]; vocab comes from the file size.
inline std::pair<std::vector<uint16_t>, int>
load_embed_table(const std::string &path, int hidden, const char *tag) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error("cannot open " + path);
    const auto bytes = static_cast<size_t>(f.tellg());
    f.seekg(0);
    const size_t row = static_cast<size_t>(hidden) * sizeof(uint16_t);
    if (row == 0) throw std::runtime_error("hidden must be > 0");
    if (bytes % row != 0)
        std::fprintf(stderr,
                     "[%s] embed table %zu bytes not a multiple of hidden*2=%zu "
                     "(model hidden=%d); reading whole rows only\n",
                     tag, bytes, row, hidden);
    const int vocab = static_cast<int>(bytes / row);
    std::vector<uint16_t> buf(static_cast<size_t>(vocab) * hidden);
    f.read(reinterpret_cast<char *>(buf.data()),
           static_cast<std::streamsize>(buf.size() * sizeof(uint16_t)));
    if (!f) throw std::runtime_error("short read on " + path);
    return {std::move(buf), vocab};
}

// ids -> (1, L, hidden) fp16 rows. An out-of-range id becomes a zero row.
inline std::vector<uint16_t> embed_lookup(const std::vector<uint16_t> &table,
                                          const std::vector<uint32_t> &ids,
                                          int vocab, int hidden) {
    std::vector<uint16_t> out(ids.size() * hidden, 0);
    for (size_t i = 0; i < ids.size(); ++i) {
        if (ids[i] >= static_cast<uint32_t>(vocab)) {
            std::fprintf(stderr, "[embed] token %u >= vocab %d (zeroed)\n",
                         ids[i], vocab);
            continue;
        }
        std::memcpy(&out[i * hidden],
                    &table[static_cast<size_t>(ids[i]) * hidden],
                    hidden * sizeof(uint16_t));
    }
    return out;
}

// ------------------------------------------------------------------ inputs ----
// `data` is fp16. A bf16 input (--act bf16) gets it rounded to bf16 (RNE);
// an fp16 input gets the bits as they are.
inline void fill_fp16_input(const std::shared_ptr<Runtime> &rt,
                            const std::string &name,
                            const std::vector<uint32_t> &shape,
                            const uint16_t *data) {
    rt->ResizeInput(name, shape);
    auto tg = as_tensor<uint16_t>(rt->GetInput(name));
    if (tg->num_elements() <= 0) return;
    std::vector<uint16_t> scratch;
    tg->fillToCPU(fp16_as_kind(tg->elem_kind(), data,
                               static_cast<size_t>(tg->num_elements()),
                               scratch));
}

inline void fill_i64_input(const std::shared_ptr<Runtime> &rt,
                           const std::string &name,
                           const std::vector<uint32_t> &shape,
                           const int64_t *data) {
    rt->ResizeInput(name, shape);
    auto tg = as_tensor<int64_t>(rt->GetInput(name));
    if (tg->num_elements() > 0) tg->fillToCPU(data);
}

// With `batch`, the copy is only recorded into it; the caller ends, submits
// and waits once for every input and then resets the staging pool.
inline void upload_input(const std::shared_ptr<VulkanCommandPool> &cp,
                         const std::shared_ptr<ITensor> &t,
                         const std::shared_ptr<VulkanCommandBuffer> &batch =
                             nullptr) {
    if (t->dtype() == typeid(int64_t))
        as_tensor<int64_t>(t)->copyToGPU(cp, nullptr, nullptr, batch);
    else if (t->dtype() == typeid(int))
        as_tensor<int>(t)->copyToGPU(cp, nullptr, nullptr, batch);
    else if (t->dtype() == typeid(int8_t))
        as_tensor<int8_t>(t)->copyToGPU(cp, nullptr, nullptr, batch);
    else if (t->dtype() == typeid(float))
        as_tensor<float>(t)->copyToGPU(cp, nullptr, nullptr, batch);
    else
        as_tensor<uint16_t>(t)->copyToGPU(cp, nullptr, nullptr, batch);
}

// (1,1,q,kv) causal bias: query i sees keys [0, i + (kv - q)].
inline std::vector<uint16_t> causal_bias(int q, int kv) {
    std::vector<uint16_t> m(static_cast<size_t>(q) * kv, 0);
    for (int i = 0; i < q; ++i) {
        const int threshold = i + (kv - q);
        for (int j = threshold + 1; j < kv; ++j)
            m[static_cast<size_t>(i) * kv + j] = FP16_MIN;
    }
    return m;
}

// ------------------------------------------------------------ prefix reuse ----
// Default prompt tokens per prefill dispatch. The graph's logits output is
// (1, q, vocab): at q = 8192 and a 152k vocab that is ~2.5 GB, which still
// fits, and an ~11k agent prompt splits into two dispatches. Every chunk
// re-runs lm_head, so larger is faster as long as the buffer fits.
constexpr int kDefaultPrefillChunk = 8192;

// A span type with no members, for callers that have no image spans.
struct NoSpan {
    int start = 0, count = 0, block = 0;
};

// How many leading KV rows computed for `cov_ids` can be reused for `ids`.
//
// The longest common prefix, because causality makes any shared run reusable:
// row i's K/V depends only on ids[0..i] at absolute position i, and the rows
// past the divergence point are simply overwritten by this round's prefill
// (attention never reads beyond past_len). Three cuts on top of the plain
// id comparison:
//   - at least one token stays outside, or there is no Run() to produce the
//     first logits row;
//   - the mm types must agree (an image placeholder id is the same token for
//     every image, so the id alone does not prove the row is the same);
//   - an image span may not straddle the cut, and every span inside the
//     prefix must be the very same span (start, count, block) on both sides:
//     the same placeholder ids with a different image behind them give
//     different K/V, and that error never shows itself.
template <typename Span = NoSpan>
inline int reusable_prefix(const std::vector<uint32_t> &cov_ids,
                           const std::vector<uint32_t> &ids,
                           const std::vector<int32_t> *cov_mm = nullptr,
                           const std::vector<int32_t> *mm = nullptr,
                           const std::vector<Span> *cov_spans = nullptr,
                           const std::vector<Span> *spans = nullptr) {
    if (ids.empty()) return 0;
    size_t limit = std::min(cov_ids.size(), ids.size() - 1);
    if (cov_mm && mm)
        limit = std::min({limit, cov_mm->size(), mm->size()});
    size_t match = 0;
    while (match < limit && cov_ids[match] == ids[match] &&
           (!cov_mm || !mm || (*cov_mm)[match] == (*mm)[match]))
        ++match;
    // A span that starts inside the prefix must end inside it and have an
    // identical twin on the other side; otherwise cut at its start. Spans do
    // not overlap, so cutting at one span's start cannot invalidate an
    // earlier one, and a single pass with min() is enough.
    auto check = [&match](const std::vector<Span> *mine,
                          const std::vector<Span> *other) {
        if (!mine) return;
        size_t cut = match;
        for (const auto &s : *mine) {
            if (static_cast<size_t>(s.start) >= match) continue;
            bool ok = static_cast<size_t>(s.start + s.count) <= match;
            if (ok) {
                ok = false;
                if (other)
                    for (const auto &o : *other)
                        if (o.start == s.start && o.count == s.count &&
                            o.block == s.block) {
                            ok = true;
                            break;
                        }
            }
            if (!ok) cut = std::min(cut, static_cast<size_t>(s.start));
        }
        match = cut;
    };
    check(cov_spans, spans);
    check(spans, cov_spans);
    return static_cast<int>(match);
}

// ------------------------------------------------------------- GPU top-k ----
// TopK registered as a post-process over 'logits': recorded into Run's own
// command buffer, so no step pays an extra submit. Outputs are published as
// post_TopK0 (values, fp16) and post_TopK1 (int32 ids), each (rows, k) with
// rows sorted descending (ties in no fixed order). k = 1 is GPU argmax; a
// larger k also serves sampling (Sampler::pick_topk). Needs the archive to
// carry the TopK kernel (onnx2vkop --extra-shaders TopK); without it
// RegisterPostProcess throws before the graph is touched and the caller
// keeps the host path. VKOP_GPU_ARGMAX=0 forces the host path (A/B arm).
// Returns the k that is live, or 0.
inline int register_gpu_topk(const std::shared_ptr<Runtime> &rt,
                             const char *tag, int k = 1) {
    if (const char *e = std::getenv("VKOP_GPU_ARGMAX"); e && e[0] == '0') {
        std::printf("[%s] argmax: host scan (VKOP_GPU_ARGMAX=0)\n", tag);
        return 0;
    }
    auto logits = rt->GetOutput("logits");
    if (!logits) return 0;
    if (logits->elem_kind() == ElemKind::kBFloat16) {
        // TopK has no bf16 kernel. onnx2vkop --act bf16 keeps `logits` at the
        // export's width (fp16_graph._BF16_KEEP_OUTPUTS) for exactly this; an
        // artifact that did not gets the host scan, which decodes bf16.
        std::printf("[%s] argmax: host scan (bf16 logits: no bf16 TopK)\n",
                    tag);
        return 0;
    }
    // The kernel keeps k per workgroup in shared memory and pads an odd k
    // for fp16; stay well inside both.
    k = std::max(1, std::min(k, 128));
    if (k > 1 && (k & 1)) ++k;
    try {
        rt->RegisterPostProcess(ops::OpType::TOPK, {{"k", std::to_string(k)}},
                                {logits},
                                {std::make_shared<core::Tensor<uint16_t>>(),
                                 std::make_shared<core::Tensor<int>>()});
        std::printf("[%s] argmax: GPU TopK(k=%d) post-process\n", tag, k);
        return k;
    } catch (const std::exception &e) {
        std::printf("[%s] argmax: host scan (%s)\n", tag, e.what());
        return 0;
    }
}

inline bool register_gpu_argmax(const std::shared_ptr<Runtime> &rt,
                                const char *tag) {
    return register_gpu_topk(rt, tag, 1) > 0;
}

// ------------------------------------------------------------- sampling ----
struct SampleParams {
    float temperature = 0.0f;         // <= 0: greedy
    int top_k = 0;                    // <= 0: keep the whole vocabulary
    float top_p = 1.0f;               // >= 1: no nucleus cut
    float repetition_penalty = 1.0f;  // 1: off
};

// Two entry points with the same distribution:
//
//   pick()      -- the whole fp16 logits row on the host.
//   pick_topk() -- only the K largest logits (GPU TopK post-process), which
//                  is what makes non-greedy decode cheap: K values instead of
//                  a 151936-wide row read back and scanned every step.
//
// pick_topk() is exact, not a truncation. Every token outside the K
// candidates has a logit <= the K-th candidate's (`bound`), and neither the
// repetition penalty nor the temperature can lift it above that. So:
//   - top_k = k: the k-th candidate strictly above `bound` proves the top-k
//     set is the first k candidates; both paths then run the same float ops
//     in the same order, so the token is bit-identical to pick().
//   - top_p only: nucleus over the whole vocabulary. The unseen tail weighs
//     at most (V-K)*exp(bound-max); when the cut lands on the same candidate
//     with tail mass 0 and with the bound, the cut is known and the draw is
//     the same as pick()'s.
//   - temperature only: rejection. Draw over candidates + the tail's bound;
//     a hit in the candidates is final. A hit in the tail reads the full row,
//     and lands in the real tail or (for the slack between the bound and the
//     real tail) redraws over the full row. The result has exactly the
//     softmax distribution; only the RNG stream differs from pick().
// Anything else falls back to pick() on the full row (`full_row`).
class Sampler {
  public:
    explicit Sampler(uint64_t seed = 0) : rng_(seed) {}

    // How the full logits row decodes (the `logits` output's elem_kind):
    // fp16 by default, bf16 for an artifact that left logits bf16. The K
    // candidate values of pick_topk come from the TopK post-process, which
    // only runs on fp16 logits, so they stay fp16.
    void set_logits_kind(ElemKind kind) {
        logits_kind_ = kind == ElemKind::kBFloat16 ? kind : ElemKind::kFloat16;
    }

    // Counters for the log line: how many picks needed the full row.
    int fast_picks = 0, full_picks = 0;

    // `history` is the set of ids this round has generated (unique, first
    // seen order); prompt tokens are deliberately not penalised.
    int pick(const uint16_t *row, int vocab,
             const std::vector<uint32_t> &history, const SampleParams &sp) {
        ++full_picks;
        if (sp.temperature <= 0.0f && sp.repetition_penalty == 1.0f) {
            int best = 0;
            float bestv = -1e30f;
            for (int i = 0; i < vocab; ++i) {
                const float v = half_to_f32(row[i], logits_kind_);
                if (v > bestv) { bestv = v; best = i; }
            }
            return best;
        }
        logits_.resize(static_cast<size_t>(vocab));
        for (int i = 0; i < vocab; ++i) logits_[i] = half_to_f32(row[i], logits_kind_);
        apply_penalty_(history, vocab, sp);
        // max_element takes the first maximum, the strict-`>` scan's rule.
        if (sp.temperature <= 0.0f)
            return static_cast<int>(
                std::max_element(logits_.begin(), logits_.end()) -
                logits_.begin());

        const float inv_t = 1.0f / sp.temperature;
        for (float &v : logits_) v *= inv_t;
        idx_.resize(static_cast<size_t>(vocab));
        for (int i = 0; i < vocab; ++i) idx_[i] = i;
        // Descending, ties by index: a total order, so the candidate path can
        // reproduce it exactly.
        const auto better = [this](int a, int b) {
            return logits_[a] > logits_[b] ||
                   (logits_[a] == logits_[b] && a < b);
        };
        int k = vocab;
        if (sp.top_k > 0 && sp.top_k < vocab) {
            k = sp.top_k;
            std::nth_element(idx_.begin(), idx_.begin() + k, idx_.end(),
                             better);
        }
        // A cut set (top-k or nucleus) is sorted; the plain-temperature walk
        // keeps index order -- sorting 152k entries per token is not free.
        if (k < vocab || nucleus_(sp))
            std::sort(idx_.begin(), idx_.begin() + k, better);
        float maxv = -std::numeric_limits<float>::infinity();
        for (int i = 0; i < k; ++i) maxv = std::max(maxv, logits_[idx_[i]]);
        return draw_(k, maxv, sp, [this](int i) { return logits_[idx_[i]]; },
                     [this](int i) { return idx_[i]; });
    }

    // `vals`/`idx`: the K largest logits of the row (fp16) and their vocab
    // ids, in any order. `full_row` reads the whole row back when needed.
    int pick_topk(const uint16_t *vals, const int *idx, int K, int vocab,
                  const std::vector<uint32_t> &history, const SampleParams &sp,
                  const std::function<const uint16_t *()> &full_row) {
        auto fallback = [&] { return pick(full_row(), vocab, history, sp); };
        if (K <= 0 || K >= vocab) return fallback();
        cand_.resize(static_cast<size_t>(K));
        float bound = std::numeric_limits<float>::infinity();
        for (int i = 0; i < K; ++i) {
            cand_[i] = {fp16_to_f32(vals[i]), idx[i]};
            bound = std::min(bound, cand_[i].v);
            if (idx[i] < 0 || idx[i] >= vocab) return fallback();
        }
        if (sp.repetition_penalty != 1.0f)
            for (const uint32_t id : history)
                for (auto &c : cand_)
                    if (static_cast<uint32_t>(c.id) == id)
                        c.v = c.v > 0.0f ? c.v / sp.repetition_penalty
                                         : c.v * sp.repetition_penalty;
        const auto better = [](const Cand &a, const Cand &b) {
            return a.v > b.v || (a.v == b.v && a.id < b.id);
        };
        if (sp.temperature <= 0.0f) {
            const Cand best = *std::min_element(cand_.begin(), cand_.end(),
                                                better);
            if (!(best.v > bound)) return fallback();
            ++fast_picks;
            return best.id;
        }
        const float inv_t = 1.0f / sp.temperature;
        for (auto &c : cand_) c.v *= inv_t;
        bound *= inv_t;
        std::sort(cand_.begin(), cand_.end(), better);
        const float maxv = cand_[0].v;
        if (!(maxv > bound)) return fallback();
        auto val = [this](int i) { return cand_[i].v; };
        auto id = [this](int i) { return cand_[i].id; };

        if (sp.top_k > 0 && sp.top_k < vocab) {
            if (sp.top_k > K || !(cand_[sp.top_k - 1].v > bound))
                return fallback();
            ++fast_picks;
            return draw_(sp.top_k, maxv, sp, val, id);
        }
        // Upper bound on the unseen tail's mass; the 1.001 covers the
        // rounding of a sum of (V-K) terms against one product.
        const float tail_bound = static_cast<float>(vocab - K) *
                                 std::exp(bound - maxv) * 1.001f;
        if (nucleus_(sp)) {
            const int n = nucleus_cut_(K, maxv, sp.top_p, tail_bound);
            if (n < 0) return fallback();
            ++fast_picks;
            return draw_(n, maxv, sp, val, id, /*nucleus_n=*/true);
        }
        // Temperature only: rejection against the tail bound.
        float zc = 0.0f;
        for (int i = 0; i < K; ++i) zc += std::exp(cand_[i].v - maxv);
        std::uniform_real_distribution<float> u(0.0f, 1.0f);
        float r = u(rng_) * (zc + tail_bound);
        if (r < zc) {
            ++fast_picks;
            for (int i = 0; i < K; ++i) {
                r -= std::exp(cand_[i].v - maxv);
                if (r <= 0.0f) return cand_[i].id;
            }
            return cand_[K - 1].id;
        }
        // The draw fell past the candidates: resolve it against the real
        // tail (same penalty and scaling as pick()).
        ++full_picks;
        const uint16_t *row = full_row();
        logits_.resize(static_cast<size_t>(vocab));
        for (int i = 0; i < vocab; ++i) logits_[i] = half_to_f32(row[i], logits_kind_);
        apply_penalty_(history, vocab, sp);
        in_cand_.assign(static_cast<size_t>(vocab), 0);
        for (const auto &c : cand_) in_cand_[c.id] = 1;
        r -= zc;
        for (int j = 0; j < vocab; ++j) {
            if (in_cand_[j]) continue;
            r -= std::exp(logits_[j] * inv_t - maxv);
            if (r <= 0.0f) return j;
        }
        // The slack between the bound and the real tail: redraw exactly.
        --full_picks;
        return pick(row, vocab, history, sp);
    }

  private:
    struct Cand {
        float v;
        int id;
    };

    static bool nucleus_(const SampleParams &sp) {
        return sp.top_p > 0.0f && sp.top_p < 1.0f;
    }

    void apply_penalty_(const std::vector<uint32_t> &history, int vocab,
                        const SampleParams &sp) {
        if (sp.repetition_penalty == 1.0f) return;
        for (const uint32_t id : history) {
            if (id >= static_cast<uint32_t>(vocab)) continue;
            float &v = logits_[id];
            v = v > 0.0f ? v / sp.repetition_penalty
                         : v * sp.repetition_penalty;
        }
    }

    // The nucleus size over the sorted candidates when the true total mass
    // is anywhere in [Zc, Zc + tail_bound]; -1 when that range moves the cut.
    int nucleus_cut_(int K, float maxv, float top_p, float tail_bound) {
        probs_.resize(static_cast<size_t>(K));
        float zc = 0.0f;
        for (int i = 0; i < K; ++i) {
            probs_[i] = std::exp(cand_[i].v - maxv);
            zc += probs_[i];
        }
        auto cut = [&](float z) {
            float cum = 0.0f;
            for (int i = 0; i < K; ++i) {
                cum += probs_[i] / z;
                if (cum >= top_p) return i + 1;
            }
            return -1;
        };
        const int lo = cut(zc), hi = cut(zc + tail_bound);
        return (lo > 0 && lo == hi) ? lo : -1;
    }

    // Softmax over the first `k` entries (sorted order) and one draw, with
    // pick()'s exact float sequence. With nucleus_n the k entries ARE the
    // nucleus already; otherwise the nucleus cut (if any) is applied here.
    template <typename Val, typename Id>
    int draw_(int k, float maxv, const SampleParams &sp, Val val, Id id,
              bool nucleus_n = false) {
        probs_.resize(static_cast<size_t>(k));
        float sum = 0.0f;
        for (int i = 0; i < k; ++i) {
            const float e = std::exp(val(i) - maxv);
            probs_[i] = e;
            sum += e;
        }
        int n = k;
        if (!nucleus_n && nucleus_(sp)) {
            float cum = 0.0f;
            for (int i = 0; i < k; ++i) {
                cum += probs_[i] / sum;
                if (cum >= sp.top_p) { n = i + 1; break; }
            }
        }
        if (n != k || nucleus_n) {
            sum = 0.0f;
            for (int i = 0; i < n; ++i) sum += probs_[i];
        }
        std::uniform_real_distribution<float> u(0.0f, 1.0f);
        float r = u(rng_) * sum;
        for (int i = 0; i < n; ++i) {
            r -= probs_[i];
            if (r <= 0.0f) return id(i);
        }
        return id(n - 1);
    }

    std::mt19937_64 rng_;
    ElemKind logits_kind_ = ElemKind::kFloat16;
    std::vector<float> logits_;
    std::vector<int> idx_;
    std::vector<float> probs_;
    std::vector<Cand> cand_;
    std::vector<uint8_t> in_cand_;
};

// VKOP_COMPONENT_TIMER=1: time the argmax readback and the decode upload
// directly; end-to-end wall cannot resolve ~1 ms on a thermally noisy laptop.
inline bool component_timer_on() {
    static const bool on = [] {
        const char *e = std::getenv("VKOP_COMPONENT_TIMER");
        return e && e[0] == '1';
    }();
    return on;
}

// ------------------------------------------------------------ DecodeInputs ----
// Writes the q_len=1 inputs of one decode step: inputs_embeds, position_ids,
// attention_bias (all zero, only its length moves), and the deepstack /
// image_pad_mask zeros a multimodal graph declares.
//
// install() puts host-imported storage under those inputs, sized to their
// decode capacity; after that a step is a memcpy into device-visible host
// memory with no staging submit. Inputs that are not imported take one
// batched upload (one submit + wait for all of them). Call install() once per
// round after prefill: a prefill shape larger than the import's capacity
// drops it (Tensor::recreate_storage_buffer), and re-installing an import
// that survived is free (as_host_imported_buffer keeps a large-enough block).
//
// Knobs: VKOP_DECODE_ZEROCOPY=0 disables imports. Bisection aids:
// VKOP_ZC_ONLY=a,b imports only the named inputs; VKOP_ZC_UPLOAD=1 also
// uploads on top of the imports (isolates binding bugs from content bugs).
class DecodeInputs {
  public:
    DecodeInputs(std::shared_ptr<Runtime> rt, std::shared_ptr<VulkanDevice> dev,
                 std::shared_ptr<VulkanCommandPool> cp, const ModelArch &arch)
        : rt_(std::move(rt)), dev_(std::move(dev)), cp_(std::move(cp)),
          arch_(arch) {
        ds_zero_.assign(static_cast<size_t>(arch_.hidden), 0);
        ds_shape_ = {1u, static_cast<uint32_t>(arch_.hidden)};
    }

    const std::vector<uint32_t> &ds_shape() const { return ds_shape_; }
    const std::vector<uint16_t> &ds_zero() const { return ds_zero_; }

    // Returns whether the imports are in place. Any failure leaves the
    // batched-upload path, which is always correct.
    bool install(int max_kv) {
        zerocopy_ = false;
        if (const char *e = std::getenv("VKOP_DECODE_ZEROCOPY");
            e && std::strcmp(e, "0") == 0)
            return false;
        if (!dev_->is_support_host_import()) {
            if (!reported_unavailable_) {
                reported_unavailable_ = true;
                std::printf("[perf] host import unavailable, decode keeps "
                            "batched uploads\n");
            }
            return false;
        }
        const char *only_env = std::getenv("VKOP_ZC_ONLY");
        const std::string only = only_env ? only_env : "";
        auto want = [&](const char *name) {
            return only.empty() || only.find(name) != std::string::npos;
        };
        auto zero_import = [&](const std::string &name, size_t max_elems) {
            auto t = rt_->GetInput(name);
            if (!t || t->dtype() != typeid(uint16_t)) return;
            auto g = as_tensor<uint16_t>(t);
            g->as_host_imported_buffer(dev_, max_elems);
            std::memset(g->host_import_ptr(), 0, g->host_import_bytes());
        };
        try {
            if (want("inputs_embeds"))
                zero_import("inputs_embeds", static_cast<size_t>(arch_.hidden));
            if (want("attention_bias"))
                zero_import("attention_bias", static_cast<size_t>(max_kv));
            if (arch_.has_deepstack && want("deepstack")) {
                size_t ne = 1;
                for (auto d : ds_shape_) ne *= d;
                for (int d = 0; d < 3; ++d)
                    zero_import("deepstack_embeds_" + std::to_string(d), ne);
            }
            if (arch_.has_image_pad_mask) {
                auto t = rt_->GetInput("image_pad_mask");
                if (t && t->dtype() == typeid(int8_t)) {
                    auto g = as_tensor<int8_t>(t);
                    g->as_host_imported_buffer(dev_, 1);
                    // posix_memalign does not zero, and decode never writes
                    // this input again: without the memset the mask a step
                    // reads is whatever the allocator handed back.
                    std::memset(g->host_import_ptr(), 0, g->host_import_bytes());
                }
            }
            if (want("position_ids")) {
                auto pid = rt_->GetInput("position_ids");
                if (pid && pid->dtype() == typeid(int64_t))
                    as_tensor<int64_t>(pid)->as_host_imported_buffer(dev_, 3);
            }
        } catch (const std::exception &e) {
            std::printf("[perf] host import failed (%s), decode keeps batched "
                        "uploads\n", e.what());
            return false;
        }
        if (!reported_ok_) {
            reported_ok_ = true;
            std::printf("[perf] decode inputs host-imported: per-step upload "
                        "submits eliminated\n");
        }
        zerocopy_ = true;
        return true;
    }

    bool zerocopy() const { return zerocopy_; }

    // One decode step's inputs. `emb` is one (1,1,hidden) row; `pos` the
    // absolute position (rope delta included); `kv_len` the attention_bias
    // length (cache rows read this step + 1).
    void step(const uint16_t *emb, int64_t pos, int kv_len) {
        const bool upload_too =
            zerocopy_ && std::getenv("VKOP_ZC_UPLOAD") != nullptr;
        bool ie_imp, pid_imp, ab_imp, ds_imp[3] = {true, true, true},
                                      pm_imp = true;
        {
            rt_->ResizeInput("inputs_embeds",
                             {1u, 1u, static_cast<uint32_t>(arch_.hidden)});
            auto ie = as_tensor<uint16_t>(rt_->GetInput("inputs_embeds"));
            ie_imp = ie->is_host_imported();
            // `emb` is an fp16 row of embed_tokens.bin; a bf16 graph gets it
            // rounded to bf16 here (hidden words per step, ~microseconds).
            emb = fp16_as_kind(ie->elem_kind(), emb,
                               static_cast<size_t>(arch_.hidden), emb_bits_);
            if (ie_imp)
                std::memcpy(ie->host_import_ptr(), emb,
                            static_cast<size_t>(arch_.hidden) * sizeof(uint16_t));
            else
                ie->fillToCPU(emb);
        }
        {
            int64_t p3[3] = {pos, pos, pos};
            const bool three = arch_.position_ids_dims == 3;
            if (three)
                rt_->ResizeInput("position_ids", {3u, 1u, 1u});
            else
                rt_->ResizeInput("position_ids", {1u, 1u});
            auto t = as_tensor<int64_t>(rt_->GetInput("position_ids"));
            pid_imp = t->is_host_imported();
            // position_ids is also read on the host by Gather's index fast
            // path (data_, not the bound buffer), so an imported tensor needs
            // both the mapping and data_ written.
            if (pid_imp)
                std::memcpy(t->host_import_ptr(), p3,
                            (three ? 3 : 1) * sizeof(int64_t));
            t->fillToCPU(p3);
        }
        {
            rt_->ResizeInput("attention_bias",
                             {1u, 1u, 1u, static_cast<uint32_t>(kv_len)});
            auto ab = as_tensor<uint16_t>(rt_->GetInput("attention_bias"));
            ab_imp = ab->is_host_imported();
            // Imported storage was zeroed to capacity at install.
            if (!ab_imp) {
                if (zeros_.size() < static_cast<size_t>(kv_len))
                    zeros_.assign(static_cast<size_t>(kv_len), 0);
                ab->fillToCPU(zeros_.data());
            }
        }
        if (arch_.has_deepstack) {
            for (int d = 0; d < 3; ++d) {
                const std::string nm = "deepstack_embeds_" + std::to_string(d);
                rt_->ResizeInput(nm, ds_shape_);
                auto t = as_tensor<uint16_t>(rt_->GetInput(nm));
                ds_imp[d] = t->is_host_imported();
                if (!ds_imp[d]) t->fillToCPU(ds_zero_.data());
            }
        }
        if (arch_.has_image_pad_mask) {
            rt_->ResizeInput("image_pad_mask", {1u, 1u});
            auto t = as_tensor<int8_t>(rt_->GetInput("image_pad_mask"));
            pm_imp = t->is_host_imported();
            if (!pm_imp) {
                int8_t zero = 0;
                t->fillToCPU(&zero);
            }
        }

        // ds_imp/pm_imp default to true so a text model, which has neither
        // input, never submits an empty batch -- that would hand back exactly
        // the submit the imports saved.
        const bool need = !ie_imp || !pid_imp || !ab_imp || !ds_imp[0] ||
                          !ds_imp[1] || !ds_imp[2] || !pm_imp;
        if (!upload_too && !need) return;
        const auto t0 = std::chrono::steady_clock::now();
        auto batch = std::make_shared<VulkanCommandBuffer>(cp_);
        batch->begin();
        if (upload_too || !ie_imp)
            upload_input(cp_, rt_->GetInput("inputs_embeds"), batch);
        if (upload_too || !pid_imp)
            upload_input(cp_, rt_->GetInput("position_ids"), batch);
        if (upload_too || !ab_imp)
            upload_input(cp_, rt_->GetInput("attention_bias"), batch);
        if (arch_.has_deepstack)
            for (int d = 0; d < 3; ++d)
                if (upload_too || !ds_imp[d])
                    upload_input(cp_,
                                 rt_->GetInput("deepstack_embeds_" +
                                               std::to_string(d)),
                                 batch);
        if (arch_.has_image_pad_mask && (upload_too || !pm_imp))
            upload_input(cp_, rt_->GetInput("image_pad_mask"), batch);
        batch->end();
        batch->submit(dev_->getComputeQueue());
        batch->wait();
        cp_->getStagingBufferPool()->reset();
        if (component_timer_on())
            std::printf("[p1t] upload %.3fms\n",
                        std::chrono::duration<double, std::milli>(
                            std::chrono::steady_clock::now() - t0)
                            .count());
    }

  private:
    std::shared_ptr<Runtime> rt_;
    std::shared_ptr<VulkanDevice> dev_;
    std::shared_ptr<VulkanCommandPool> cp_;
    ModelArch arch_;
    std::vector<uint16_t> ds_zero_;
    std::vector<uint32_t> ds_shape_;
    std::vector<uint16_t> zeros_;
    std::vector<uint16_t> emb_bits_;  // the bf16 copy of a decode row
    bool zerocopy_ = false;
    bool reported_ok_ = false;
    bool reported_unavailable_ = false;
};

}  // namespace vkop::llm
