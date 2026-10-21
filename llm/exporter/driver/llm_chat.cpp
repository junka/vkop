// junka @ 2026
// End-to-end conversational driver for llm.vkopbin (Qwen3 / Qwen3-VL LLM,
// buffer backend). Text in → generated text out, no .npy dumps required.
//
// 架构参数 (NLAYERS / NKV / HD / HIDDEN) 在 LoadModel 之后从 runtime inputs
// 动态推断 —— 同一个驱动同时支持：
//   · Qwen3-VL-2B 多模态：3D MRoPE position_ids + deepstack_embeds_*
//     + image_pad_mask（有 --image/--visual 时自动启用）
//   · Qwen3-8B / Qwen3-4B / 其他纯文本：2D 标准 RoPE，无视觉 I/O
//
// Reuses:
//   - llm/tokenizer   (BBPE encode/decode + chat template)
//   - llm/exporter/driver/conversation.hpp (多轮上下文：token 级历史 + 整段重 prefill)
//   - llm/exporter/driver/kv_cache.hpp     (KV 预分配 / 每轮 reset / present→past)
//   - llm.vkopbin     (LLM graph, KV-cache as explicit I/O)
//   - embed_tokens.bin (standalone [vocab, hidden] fp16 embedding table,
//     exported by qwen3vl_export_onnx.py 或 qwen3_export_onnx.py)
//
// Usage:
//   llm_chat <model.vkopbin> <embed_tokens.bin> <tokenizer.bin> [max_new]
//            [--image <img>]... --visual <visual.vkopbin>
//   (--image 可重复，多图按给出顺序与 prompt 里的 image_pad 标记一一对应；
//    这些图属于第一轮，之后的轮次靠历史复用它们)
//   (then type prompts on stdin, Ctrl-D to quit)
//   同进程里的每条输入都是下一轮的上下文（VKOP_RAW_PROMPT=1 关掉，每行独立）。
//
// Build: `make llm_chat` (ENABLE_LLM_CHAT is on by default; `make` builds it
// along with the rest). See the ENABLE_LLM_CHAT block in CMakeLists.txt.

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <memory>
#include <string>
#include <vector>
#include <fstream>
#include <stdexcept>
#include <cmath>
#include <cstdlib>
#include <chrono>

#include "vulkan/VulkanDevice.hpp"
#include "vulkan/VulkanInstance.hpp"
#include "include/logger.hpp"
#include "core/Tensor.hpp"
#include "core/runtime.hpp"
#include "tokenizer.hpp"
#include "image_preproc.hpp"
#include "conversation.hpp"
#include "kv_cache.hpp"

#define STB_IMAGE_IMPLEMENTATION
#include "include/stb_image.h"

using vkop::VulkanInstance;
using vkop::VulkanDevice;
using vkop::VulkanCommandBuffer;
using vkop::core::ITensor;
using vkop::core::Runtime;
using vkop::core::as_tensor;
using vkop::core::ElemKind;
using vkop::core::elem_name;
using vkop::export_::Conversation;
using vkop::export_::ImageBlock;
using vkop::export_::KVCache;
using vkop::export_::RenderedContext;
using vkop::export_::get_rope_index;
using vkop::export_::preprocess_image_noresize;
using qwen::Tokenizer;

namespace {

// ---- profiling helpers ----
struct PerfMetrics {
    // Prefill (current round)
    double prefill_ms = 0;
    int prefill_tokens = 0;

    // Decode (current round)
    int decode_steps = 0;
    double total_decode_ms = 0;
    std::vector<double> per_token_ms;  // per-step latency for percentile calc

    // Session-wide accumulators
    int total_prompt_tokens = 0;
    int total_generated_tokens = 0;
    int total_rounds = 0;

    double prefill_tok_per_sec() const {
        return prefill_ms > 0 ? (prefill_tokens / (prefill_ms / 1000.0)) : 0;
    }
    double decode_tok_per_sec() const {
        return total_decode_ms > 0 ? (decode_steps / (total_decode_ms / 1000.0)) : 0;
    }
    double avg_decode_ms() const {
        return decode_steps > 0 ? (total_decode_ms / decode_steps) : 0;
    }

    // Percentile from per_token_ms (sorted internally).
    double percentile(double p) const {
        if (per_token_ms.empty()) return 0;
        auto sorted = per_token_ms;
        std::sort(sorted.begin(), sorted.end());
        double idx = (p / 100.0) * (sorted.size() - 1);
        size_t lo = static_cast<size_t>(idx);
        size_t hi = lo + 1;
        if (hi >= sorted.size()) return sorted.back();
        double frac = idx - lo;
        return sorted[lo] * (1 - frac) + sorted[hi] * frac;
    }
};

// Legacy fallback 默认值 (Qwen3-VL-2B)。真正的值在 LoadModel 之后从 runtime
// inputs 的 shape 动态推断，存入 struct ModelArch 里。以下常量只用于
// argmax_last_token 里 shape 不可靠时的 defensive fallback（极罕见）。
constexpr int LEGACY_HIDDEN = 2048;
constexpr int LEGACY_NKV = 8;
constexpr int LEGACY_HD = 128;
constexpr int LEGACY_NLAYERS = 28;

// 架构参数 —— 在 LoadModel 之后由 infer_model_arch() 填充。
// 纯文本 Qwen3 的 position_ids 是 2D (B, q)；多模态是 3D (3, B, q) MRoPE。
struct ModelArch {
    int hidden = LEGACY_HIDDEN;
    int nkv = LEGACY_NKV;
    int hd = LEGACY_HD;
    int nlayers = LEGACY_NLAYERS;
    int position_ids_dims = 2;   // 2 = 纯文本 2D RoPE, 3 = 多模态 3D MRoPE
    bool has_deepstack = false;  // deepstack_embeds_{0,1,2} 输入存在
    bool has_image_pad_mask = false;
    // KV cache 存储语义：从 past_key_values_0 的 elem_kind() 读。fp16 是历史
    // 默认；fp8 (E4M3/E5M2) 时 KVCache 用 int8_t 容器承载字节，显存减半。
    ElemKind cache_kind = ElemKind::kFloat16;
};

// 结束符 / 图像占位符的 id 由 Tokenizer 从注册表查（Conversation 也走同一入口），
// 这里不写死 —— 换 checkpoint 时那些 id 会变。
// Qwen3-VL uses torch.finfo(float16).min ≈ -65504 as the causal mask fill, not
// -inf, so softmax keeps a tiny but finite distinction. -65504 = 0xFBFF.
constexpr uint16_t FP16_MIN = 0xFBFF;

// ---- fp16 helpers (match ITensor::fp16_to_fp32 / fp32_to_fp16) ----
inline float fp16_to_f32(uint16_t h) { return ITensor::fp16_to_fp32(h); }

// 从 Runtime 的 inputs 动态推断架构参数。
// 纯文本 Qwen3 (position_ids 2D, 无 deepstack_embeds / image_pad_mask)
// 和多模态 Qwen3-VL (position_ids 3D, 有 deepstack_embeds / image_pad_mask)
// 都能自动识别。
ModelArch infer_model_arch(const std::shared_ptr<Runtime>& rt) {
    ModelArch arch;

    // ---- NLAYERS: 数 past_key_values_i 的个数 ----
    int nlayers = 0;
    for (;; ++nlayers) {
        auto t = rt->GetInput("past_key_values_" + std::to_string(nlayers));
        if (!t) break;
    }
    if (nlayers == 0) throw std::runtime_error("no past_key_values_* inputs in model");
    arch.nlayers = nlayers;

    // ---- NKV / HD: 从 past_key_values_0 shape ----
    // getShape() 在 ITensor 基类上，与容器类型无关 —— fp8 cache 的 past 是
    // Tensor<int8_t>，但仍能直接读 shape，无需 as_tensor 到具体容器。
    auto pk0 = rt->GetInput("past_key_values_0");
    auto pk0_shape = pk0->getShape();  // (B, 2, nkv, kv_len, hd)
    if (pk0_shape.size() < 5) {
        std::fprintf(stderr, "[arch] past_key_values_0 shape is %zu dims, expect 5\n",
                     pk0_shape.size());
    }
    arch.nkv = pk0_shape.size() >= 3 ? static_cast<int>(pk0_shape[2]) : LEGACY_NKV;
    arch.hd = pk0_shape.size() >= 5 ? static_cast<int>(pk0_shape[4]) : LEGACY_HD;

    // ---- cache_kind: 从 past_key_values_0 的 elem_kind() 读 ----
    // fp16 (默认) / fp8 (E4M3/E5M2)。shape 读取仍走 as_tensor<uint16_t> 是因为
    // getShape() 在 ITensor 接口上、与容器类型无关；elem_kind() 才记录语义。
    // 4-bit (kInt4/kNF4) 也走 int8_t 容器，但一个字节两个值 —— KVCache 内部按
    // cache_kind 分派，逻辑元素数不变（见 kv_cache.hpp）。
    arch.cache_kind = pk0->elem_kind();
    if (arch.cache_kind != ElemKind::kFloat16 &&
        arch.cache_kind != ElemKind::kFloat8E4M3FN &&
        arch.cache_kind != ElemKind::kFloat8E5M2 &&
        arch.cache_kind != ElemKind::kInt4 &&
        arch.cache_kind != ElemKind::kNF4) {
        std::fprintf(stderr, "[arch] past_key_values_0 cache dtype %s unsupported, "
                     "falling back to fp16\n", elem_name(arch.cache_kind));
        arch.cache_kind = ElemKind::kFloat16;
    }

    // ---- HIDDEN: 从 inputs_embeds shape ----
    auto emb = rt->GetInput("inputs_embeds");
    auto emb_g = as_tensor<uint16_t>(emb);
    auto emb_shape = emb_g->getShape();  // (B, q, hidden)
    arch.hidden = emb_shape.size() >= 3
        ? static_cast<int>(emb_shape[emb_shape.size() - 1]) : LEGACY_HIDDEN;

    // ---- position_ids dims ----
    auto pid = rt->GetInput("position_ids");
    // position_ids 是 int64，cast 到 int64 tensor 看 shape
    auto pid_shape = pid ? rt->GetInput("position_ids")->getShape()
                         : std::vector<int>{};
    arch.position_ids_dims = static_cast<int>(pid_shape.size());  // 2 或 3

    // ---- 视觉 I/O 是否存在 ----
    arch.has_deepstack = (rt->GetInput("deepstack_embeds_0") != nullptr);
    arch.has_image_pad_mask = (rt->GetInput("image_pad_mask") != nullptr);

    return arch;
}

// Read the whole embed_tokens.bin (raw fp16, [vocab, hidden]) into memory.
// hidden 必须等于模型的 HIDDEN（runtime inputs_embeds 的最后一维）。
// 若 embed_tokens.bin 的 hidden 不匹配，尝试按模型 hidden 重新推断 vocab。
std::pair<std::vector<uint16_t>, int>
load_embed_table(const std::string& path, int hidden) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error("cannot open " + path);
    auto bytes = static_cast<size_t>(f.tellg());
    f.seekg(0);
    const size_t row_bytes = hidden * sizeof(uint16_t);
    if (row_bytes == 0) throw std::runtime_error("hidden must be > 0");
    if (bytes % row_bytes != 0) {
        // 可能是旧模型导出的 embed_tokens.bin（不同 hidden），
        // 给出提示但仍按当前 hidden 推断 vocab。
        std::fprintf(stderr,
            "[warn] embed_tokens.bin size %zu not divisible by hidden*2=%zu, "
            "model hidden=%d — will read max integer vocab, tail bytes ignored\n",
            bytes, row_bytes, hidden);
    }
    int vocab = static_cast<int>(bytes / row_bytes);
    std::vector<uint16_t> buf(static_cast<size_t>(vocab) * hidden);
    f.read(reinterpret_cast<char*>(buf.data()),
           static_cast<std::streamsize>(buf.size() * sizeof(uint16_t)));
    if (!f) throw std::runtime_error("short read on " + path);
    return {std::move(buf), vocab};
}

// Lookup L token ids → fp16 embedding rows, laid out as (1, L, hidden).
std::vector<uint16_t> embed_lookup(const std::vector<uint16_t>& table,
                                   const std::vector<uint32_t>& ids,
                                   int vocab, int hidden) {
    std::vector<uint16_t> out(ids.size() * hidden, 0);
    for (size_t i = 0; i < ids.size(); ++i) {
        uint32_t id = ids[i];
        if (id >= static_cast<uint32_t>(vocab)) {
            std::fprintf(stderr, "[embed] token id %u >= vocab %d, zeroing\n", id, vocab);
            continue;
        }
        std::memcpy(&out[i * hidden], &table[static_cast<size_t>(id) * hidden],
                    hidden * sizeof(uint16_t));
    }
    return out;
}

// Fill a uint16_t (fp16) input tensor from a host buffer, after resizing it.
void fill_fp16_input(const std::shared_ptr<Runtime>& rt,
                     const std::string& name,
                     const std::vector<uint32_t>& shape,
                     const uint16_t* data) {
    rt->ResizeInput(name, shape);
    auto t = rt->GetInput(name);
    auto tg = as_tensor<uint16_t>(t);
    if (tg->num_elements() > 0) tg->fillToCPU(data);
}

// Fill an int64 input tensor from a host buffer.
void fill_i64_input(const std::shared_ptr<Runtime>& rt,
                    const std::string& name,
                    const std::vector<uint32_t>& shape,
                    const int64_t* data) {
    rt->ResizeInput(name, shape);
    auto t = rt->GetInput(name);
    auto tg = as_tensor<int64_t>(t);
    if (tg->num_elements() > 0) tg->fillToCPU(data);
}

// Fill the ring-head input ([1] int32, host-authoritative — the Attention op
// reads it through has_cpu_data(), so there is nothing to upload).
void fill_i32_input(const std::shared_ptr<Runtime>& rt,
                    const std::string& name, int value) {
    auto t = rt->GetInput(name);
    if (!t) return;
    if (t->getShape() != std::vector<int>{1}) rt->ResizeInput(name, {1u});
    auto tg = as_tensor<int>(rt->GetInput(name));
    if (tg->num_elements() > 0) tg->fillToCPU(&value);
}

// Upload whatever dtype input to GPU.
void upload_input(const std::shared_ptr<vkop::VulkanCommandPool>& cmdpool,
                  const std::shared_ptr<ITensor>& t) {
    if (t->dtype() == typeid(int64_t)) as_tensor<int64_t>(t)->copyToGPU(cmdpool);
    else if (t->dtype() == typeid(int)) as_tensor<int>(t)->copyToGPU(cmdpool);
    else if (t->dtype() == typeid(int8_t)) as_tensor<int8_t>(t)->copyToGPU(cmdpool);
    else if (t->dtype() == typeid(float)) as_tensor<float>(t)->copyToGPU(cmdpool);
    else as_tensor<uint16_t>(t)->copyToGPU(cmdpool);
}

// Argmax over the last-position logits: logits is (1, q, vocab) fp16, take
// [0, q-1, *]. Returns the vocab index of the max.
// embed_vocab 是 embed_tokens.bin 的行数：logits 最后一维正常时它和 shape.back()
// 相等（同一张表），只在 shape 记不住时用作兜底，所以不能写死某个模型的词表大小。
int argmax_last_token(const std::shared_ptr<Runtime>& rt,
                      const std::shared_ptr<vkop::VulkanCommandPool>& cmdpool,
                      int embed_vocab) {
    auto logits = rt->GetOutput("logits");
    if (!logits) throw std::runtime_error("no 'logits' output");
    auto lg = as_tensor<uint16_t>(logits);
    lg->copyToCPU(cmdpool);
    const uint16_t* p = reinterpret_cast<const uint16_t*>(lg->data().data());
    // logits shape (1, q, vocab): the converter records it; num_elements =
    // q*vocab. We want the last q row → offset (q-1)*vocab. Derive vocab from
    // the tensor shape (last dim).
    auto shape = lg->getShape();
    int vocab = shape.back();
    int q = shape.size() >= 2 ? shape[shape.size() - 2] : 1;
    int total = lg->num_elements();
    // Defensive: if shape is unreliable, assume total = q*vocab and vocab from
    // the known Qwen3-VL size.
    if (vocab <= 0 || q <= 0) {
        vocab = embed_vocab;
        q = total / vocab;
    }
    const uint16_t* row = p + (q - 1) * vocab;
    int best = 0;
    float bestv = -1e30f;
    for (int i = 0; i < vocab; ++i) {
        float v = fp16_to_f32(row[i]);
        if (v > bestv) { bestv = v; best = i; }
    }
    if (std::getenv("VKOP_CHATDBG")) {
        // top5 for sanity vs ORT.
        std::vector<std::pair<float,int>> tp;
        tp.reserve(vocab);
        for (int i = 0; i < vocab; ++i) tp.push_back({fp16_to_f32(row[i]), i});
        std::partial_sort(tp.begin(), tp.begin()+5, tp.end(),
                          [](auto&a,auto&b){return a.first>b.first;});
        std::printf("  [logits] q=%d vocab=%d top5: ", q, vocab);
        for (int i = 0; i < 5; ++i) std::printf("[%d %.3g] ", tp[i].second, tp[i].first);
        std::printf("\n");
    }
    return best;
}

// Build the causal attention_bias (1,1,q,kv) fp16: upper-triangular above the
// diagonal = FP16_MIN, else 0. For prefill q=kv=L; this is the only place a
// non-zero bias is needed (decode rounds use all-zero full-history masks).
std::vector<uint16_t> causal_bias(int q, int kv) {
    std::vector<uint16_t> m(static_cast<size_t>(q) * kv, 0);
    for (int i = 0; i < q; ++i) {
        // mask position j (key) from query i if j > i + (kv - q)  (causal:
        // query i sees keys [0, i + (kv-q)]). Equivalent to triu(diagonal =
        // kv-q+1).
        int threshold = i + (kv - q);
        for (int j = threshold + 1; j < kv; ++j) {
            m[static_cast<size_t>(i) * kv + j] = FP16_MIN;
        }
    }
    return m;
}

// One image's visual outputs: image_features + 3 deepstack tensors, each
// (n_img, HIDDEN) fp16, plus grid_thw for rope_index.
struct VisualFeatures {
    std::vector<uint16_t> image_features;   // (n_img * HIDDEN)
    std::vector<uint16_t> deepstack[3];     // each (n_img * HIDDEN)
    int n_img = 0;
    int grid_t = 0, grid_h = 0, grid_w = 0;
};

// Visual encoder session: visual.vkopbin is loaded once and reused for every
// image in the session (each run only refills pixel_values + grid_thw and
// re-reads the 4 outputs). The Runtime owns the tensor set, so features must be
// copied out before the next run — run() returns them by value.
struct VisualEngine {
    std::shared_ptr<Runtime> vrt;
    std::shared_ptr<vkop::VulkanCommandPool> cmdpool;

    VisualEngine(const std::shared_ptr<vkop::VulkanCommandPool>& cp,
                 const std::string& visual_vkopbin)
        : cmdpool(cp) {
        vrt = std::make_shared<Runtime>(cmdpool, visual_vkopbin, /*precision=*/1);
        vrt->set_backend_buffer(true);
        std::printf("=== LoadModel (visual) ===\n");
        vrt->LoadModel();
        std::printf("=== LoadModel (visual) done ===\n");
        check_visual_graph(*vrt);
    }

    // The graph takes grid_thw as an input, so the pos_embed interpolation,
    // rotary tables and everything else derived from the grid are computed
    // inside it at run time (see qwen3vl_visual_dynamic.py). The only grid-free
    // constants left are the patch geometry, which image_preproc.hpp must
    // describe exactly. Verify before the first image so a mismatch fails
    // loudly at load instead of silently mis-slicing pixel_values.
    //
    // The row dim (kRow) is a real graph constant; pixel_values' leading dim
    // and image_features' rows are symbolic ("seq_len"/"num_image_patches") and
    // the runtime substitutes 1 for a dynamic dim at load, so they carry no
    // information here — the driver ResizeInput's the first one per image.
    static void check_visual_graph(Runtime& rt) {
        auto pv = rt.GetInput("pixel_values");
        if (!pv)
            throw std::runtime_error("visual graph has no pixel_values input");
        auto g = rt.GetInput("grid_thw");
        if (!g)
            throw std::runtime_error(
                "visual graph has no grid_thw input — it is a static-grid "
                "export; re-export with the current qwen3vl_export_onnx.py");
        if (g->dtype() != typeid(int64_t))
            throw std::runtime_error("visual grid_thw input is not int64");
        auto s = pv->getShape();
        const int row = vkop::export_::kRow;
        if (s.size() != 2 || s[1] != row) {
            char buf[384];
            std::snprintf(buf, sizeof(buf),
                          "visual graph pixel_values [%d,%d] does not match "
                          "compiled constants (want [seq_len,%d] with "
                          "patch=%d temporal=%d chans=%d). The graph was "
                          "exported for a different vision config — re-export "
                          "it with a matching model.",
                          s.empty() ? 0 : s[0], s.size() > 1 ? s[1] : 0, row,
                          vkop::export_::kPatch, vkop::export_::kTemporalPatch,
                          vkop::export_::kInChans);
            throw std::runtime_error(buf);
        }
        std::printf("[visual] graph pixel_values [%d,%d] + grid_thw int64 "
                    "matches patch=%d temporal=%d merge=%d chans=%d (dynamic "
                    "grid)\n",
                    s[0], s[1], vkop::export_::kPatch,
                    vkop::export_::kTemporalPatch, vkop::export_::kMerge,
                    vkop::export_::kInChans);
    }

    VisualFeatures run(const std::string& image_path) {
        VisualFeatures vf;
        int w = 0, h = 0, c = 0;
        unsigned char* img = stbi_load(image_path.c_str(), &w, &h, &c, 3);
        if (!img) throw std::runtime_error("stbi_load failed: " + image_path);
        std::printf("[visual] image %s  %dx%d ch=%d\n", image_path.c_str(), w, h, c);
        auto pr = preprocess_image_noresize(img, h, w, 3);
        stbi_image_free(img);
        if (pr.pixel_values_fp16.empty())
            throw std::runtime_error("image not divisible by patch*merge (16*2=32)");
        vf.grid_t = pr.grid_t; vf.grid_h = pr.grid_h; vf.grid_w = pr.grid_w;
        // n_img (LLM image-token count) = visual pixel seq_len / merge^2. The
        // visual encoder consumes seq_len = grid_t*grid_h*grid_w patches
        // (pre-merge) but emits image_features of (n_img, HIDDEN) where n_img
        // = that / merge^2 (the spatial merge happens inside the visual
        // graph). For 224x224: seq_len=196, n_img=49.
        vf.n_img = pr.seq_len / (vkop::export_::kMerge * vkop::export_::kMerge);
        std::printf("[visual] grid_thw=[%d,%d,%d] n_img=%d seq_len=%d\n",
                    vf.grid_t, vf.grid_h, vf.grid_w, vf.n_img, pr.seq_len);
        if (pr.seq_len != pr.grid_t * pr.grid_h * pr.grid_w) {
            char buf[256];
            std::snprintf(buf, sizeof(buf),
                          "preprocess produced seq_len=%d but grid=[%d,%d,%d] "
                          "(=%d patches)",
                          pr.seq_len, pr.grid_t, pr.grid_h, pr.grid_w,
                          pr.grid_t * pr.grid_h * pr.grid_w);
            throw std::runtime_error(buf);
        }

        // grid_thw drives the whole grid-derived half of the graph (pos_embed
        // interpolation table lookup, rotary, merge order), so it is a real
        // input now — one row per image, (1,3) int64 {t,h,w}.
        const int64_t gthw[3] = {vf.grid_t, vf.grid_h, vf.grid_w};
        fill_i64_input(vrt, "grid_thw", {1u, 3u}, gthw);
        upload_input(cmdpool, vrt->GetInput("grid_thw"));
        vrt->ResizeInput("pixel_values",
                         {static_cast<uint32_t>(pr.seq_len),
                          static_cast<uint32_t>(pr.row)});
        auto t = vrt->GetInput("pixel_values");
        as_tensor<uint16_t>(t)->fillToCPU(pr.pixel_values_fp16.data());
        as_tensor<uint16_t>(t)->copyToGPU(cmdpool);
        double ms = vrt->Run();
        vrt->ReadResult();
        std::printf("[visual] run %.1fms\n", ms);

        const char* outs[] = {"image_features", "deepstack_features_0",
                              "deepstack_features_1", "deepstack_features_2"};
        std::vector<uint16_t>* dsts[] = {&vf.image_features, &vf.deepstack[0],
                                         &vf.deepstack[1], &vf.deepstack[2]};
        for (int i = 0; i < 4; ++i) {
            auto o = vrt->GetOutput(outs[i]);
            if (!o) throw std::runtime_error(std::string("no visual output ") + outs[i]);
            auto og = as_tensor<uint16_t>(o);
            og->copyToCPU(cmdpool);
            *dsts[i] = og->data();
        }
        return vf;
    }
};

// VKOP_RAW_PROMPT 调试路径：不过 chat template、也不累积历史，把一行输入当成
// 独立序列（对齐 dump_llm_decode.py 那种 proc(text=[text]) 的参考）。这里只做
// HF processor 的 image pad 展开，凑出 Conversation::render() 同款的序列描述。
RenderedContext render_raw(const std::vector<uint32_t>& ids, uint32_t image_pad_id,
                           const std::vector<ImageBlock>& blocks) {
    RenderedContext r;
    size_t next_block = 0;
    for (uint32_t id : ids) {
        if (image_pad_id && id == image_pad_id && next_block < blocks.size()) {
            const ImageBlock& blk = blocks[next_block];
            r.spans.push_back({static_cast<int>(r.ids.size()), blk.n_img,
                               static_cast<int>(next_block)});
            for (int k = 0; k < blk.n_img; ++k) r.ids.push_back(id);
            ++next_block;
        } else {
            r.ids.push_back(id);
        }
    }
    r.mm_types.assign(r.ids.size(), 0);
    for (const auto& s : r.spans)
        for (int i = 0; i < s.count; ++i) r.mm_types[s.start + i] = 1;
    return r;
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 4) {
        std::fprintf(stderr,
            "usage: %s <llm.vkopbin> <embed_tokens.bin> <tokenizer.bin> [max_new] "
            "[--image <img>]... --visual <visual.vkopbin> [--max-kv N] "
            "[--kv-kind {auto,fp16,fp8e4m3,fp8e5m2,int4,nf4}]\n",
            argv[0]);
        return 1;
    }
    const std::string model_path = argv[1];
    const std::string embed_path = argv[2];
    const std::string tok_path = argv[3];
    int max_new = 64;
    int max_kv = 8192;
    std::vector<std::string> image_paths;
    std::string visual_path;
    std::string kv_kind_opt = "auto";
    // Parse optional positional max_new + --image/--visual flags (any order).
    // --image is repeatable: order defines the image→prompt binding.
    bool have_max = false;
    for (int i = 4; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--image" && i + 1 < argc) { image_paths.push_back(argv[++i]); }
        else if (a == "--visual" && i + 1 < argc) { visual_path = argv[++i]; }
        else if (a == "--max-kv" && i + 1 < argc) { max_kv = std::atoi(argv[++i]); }
        else if (a == "--kv-kind" && i + 1 < argc) { kv_kind_opt = argv[++i]; }
        else if (!a.empty() && a[0] != '-' && !have_max) {
            max_new = std::atoi(a.c_str()); have_max = true;
        }
    }
    if (max_kv <= 0) {
        std::fprintf(stderr, "[error] --max-kv must be > 0 (got %d)\n", max_kv);
        return 1;
    }
    if (max_kv < max_new) {
        std::fprintf(stderr, "[error] --max-kv %d < max_new %d — the window must "
                     "hold at least one generation\n", max_kv, max_new);
        return 1;
    }
    const bool multimodal = !image_paths.empty() && !visual_path.empty();
    if (!image_paths.empty() && visual_path.empty()) {
        std::fprintf(stderr,
            "[warn] --image 提供了但没有 --visual —— 没有视觉编码器，图片被忽略\n");
    }

    Logger::getInstance().setLevel(LOG_INFO);
    const auto& phydevs = VulkanInstance::getVulkanInstance().getPhysicalDevices();
    if (phydevs.empty()) { std::printf("no vulkan device\n"); return -1; }
    auto dev = std::make_shared<VulkanDevice>(phydevs[0]);
    if (dev->getDeviceName().find("llvmpipe") != std::string::npos) {
        std::printf("no valid vulkan device\n"); return -1;
    }
    std::printf("GPU: %s\n", dev->getDeviceName().c_str());
    auto cmdpool = std::make_shared<vkop::VulkanCommandPool>(dev);

    // Tokenizer.
    Tokenizer tok;
    tok.load(tok_path);
    std::printf("[tok] loaded %s\n", tok_path.c_str());

    // Runtime + model. LoadModel 必须在 load_embed_table 之前 —— embed_tokens.bin
    // 的 hidden 必须匹配模型的 HIDDEN（runtime inputs_embeds 的最后一维）。
    // 先 LoadModel → 推断架构参数 → 再 load embed_table。
    auto rt = std::make_shared<Runtime>(cmdpool, model_path, /*precision=*/1);
    rt->set_backend_buffer(true);
    std::printf("=== LoadModel ===\n");
    rt->LoadModel();
    std::printf("=== LoadModel done ===\n");

    // 从 runtime inputs 动态推断架构参数（纯文本 vs 多模态 自动识别）。
    ModelArch arch = infer_model_arch(rt);
    std::printf("[arch] hidden=%d nlayers=%d nkv=%d hd=%d "
                "position_ids=%dD %s %s\n",
                arch.hidden, arch.nlayers, arch.nkv, arch.hd,
                arch.position_ids_dims,
                arch.has_deepstack ? "has_deepstack" : "no_deepstack",
                arch.has_image_pad_mask ? "has_image_pad_mask" : "no_image_pad_mask");

    // --kv-kind：覆盖 KVCache 的元素宽度。默认 auto = 跟随模型的
    // past_key_values_0（见 infer_model_arch）。cache 宽度必须与图的 cache
    // 张量一致 —— 不一致时 vkop 会静默误读，所以这里硬警告。
    if (kv_kind_opt != "auto") {
        const ElemKind pk0_kind = rt->GetInput("past_key_values_0")->elem_kind();
        static const std::pair<const char*, ElemKind> kKinds[] = {
            {"fp16", ElemKind::kFloat16},
            {"fp8e4m3", ElemKind::kFloat8E4M3FN},
            {"fp8e5m2", ElemKind::kFloat8E5M2},
            {"int4", ElemKind::kInt4},
            {"nf4", ElemKind::kNF4},
        };
        bool ok = false;
        for (const auto& k : kKinds) {
            if (kv_kind_opt == k.first) { arch.cache_kind = k.second; ok = true; break; }
        }
        if (!ok) {
            std::fprintf(stderr, "[error] --kv-kind %s 未知（auto/fp16/fp8e4m3/"
                         "fp8e5m2/int4/nf4）\n", kv_kind_opt.c_str());
            return 1;
        }
        if (arch.cache_kind != pk0_kind) {
            std::fprintf(stderr,
                "[warn] --kv-kind %s 覆盖了模型的 cache dtype %s —— 两者不一致会"
                "误读 KV 缓冲，除非模型文件确实按该 dtype 导出\n",
                kv_kind_opt.c_str(), elem_name(pk0_kind));
        }
    }
    std::printf("[arch] kv cache dtype = %s\n", elem_name(arch.cache_kind));

    // Embedding table: vocab inferred from file size，按 arch.hidden 推断。
    auto [embed_table, vocab] = load_embed_table(embed_path, arch.hidden);
    std::printf("[embed] loaded %s  vocab=%d  hidden=%d  (~%.0fMB, %zu rows)\n",
                embed_path.c_str(), vocab, arch.hidden,
                static_cast<double>(vocab) * arch.hidden * 2 / 1e6,
                embed_table.size() / arch.hidden);

    // Visual features (multimodal only): one --image arg per image, all
    // encoded once up-front (the same images apply to the whole session).
    // HF concatenates per-image features in prompt order into the single
    // (total_img_tokens, hidden) deepstack/image_features input, so do that
    // here too and keep only the flat buffers + the per-image token counts.
    bool have_visual = false;
    std::vector<VisualFeatures> vfs;
    std::vector<uint16_t> img_feat_cat;          // (total_img, hidden)
    std::vector<uint16_t> ds_cat[3];             // each (total_img, hidden)
    std::vector<int> feat_row0;                  // 每张图在拼接特征里的首行
    int total_img = 0;
    if (multimodal) {
        if (!arch.has_deepstack || !arch.has_image_pad_mask) {
            std::fprintf(stderr,
                "[warn] --image/--visual 提供了，但模型没有 deepstack_embeds / "
                "image_pad_mask 输入 —— 纯文本模型不支持多模态输入，忽略 image\n");
        } else {
            VisualEngine venc(cmdpool, visual_path);
            for (const auto& p : image_paths) {
                auto vf = venc.run(p);
                feat_row0.push_back(total_img);
                img_feat_cat.insert(img_feat_cat.end(), vf.image_features.begin(),
                                    vf.image_features.end());
                for (int d = 0; d < 3; ++d)
                    ds_cat[d].insert(ds_cat[d].end(), vf.deepstack[d].begin(),
                                     vf.deepstack[d].end());
                total_img += vf.n_img;
                vfs.push_back(std::move(vf));
            }
            have_visual = true;
            std::printf("[visual] %zu images, total_img=%d tokens, "
                        "image_features=%zu elems\n",
                        image_paths.size(), total_img, img_feat_cat.size());
        }
    }

    // KV cache：每层 past/present 的 buffer 一次开好（max_kv 上界），之后每轮
    // 只做逻辑 resize + present→past 回填。生命周期见 kv_cache.hpp。
    // --max-kv 可调：它同时是滑窗环的表观容量 W（见 Attention 的 ring）。
    KVCache kv(rt, cmdpool, arch.nlayers, arch.nkv, arch.hd, max_kv,
               arch.cache_kind);

    // ---- 滑窗环（graph 侧带 kv_ring_head 时启用，见 ops/Attention.hpp）----
    // KV 变成 W 行的一个环：逻辑行 j 落在物理槽 (kv_head + j) % W，覆盖的绝对
    // 位置是 [kv_base, kv_base + kv_resident)。三条会话级不变量：
    //   · kv_resident + S <= W —— 本轮 Run **之前**必须成立，否则新行会写进
    //     本轮查询还要读的最旧几行（kernel 是原地写）。这由窗口起点在
    //     prefill 之前就定死来保证（见下面的 ring 分支）：S = L - kv_base 且
    //     kv_base >= L - W，所以 S <= W 恒成立；
    //   · 图的 kv_past 是 kv_resident（cache 的行数），而 position_ids 用的是
    //     绝对位置 kv_base + kv_resident —— 两者在 base 移动后就分岔了；
    //   · 丢行只动 kv_head/kv_base，一个字都不搬，这正是精力在 ring 上而不是
    //     在 shape 推断上的原因。
    const bool ring = rt->GetInput("kv_ring_head") != nullptr;
    const int W = max_kv;
    int kv_resident = 0;   // KV 里现在有几行
    int kv_head = 0;       // 逻辑行 0 所在的物理槽
    int kv_base = 0;       // 逻辑行 0 的绝对位置
    if (ring) {
        std::printf("[ctx] KV 滑窗环 W=%d 行：序列可以无限长，最旧的行让位\n", W);
    } else if (max_kv < 8192) {
        std::printf("[ctx] KV 上限 %d 行（图不支持滑窗环：序列不得超出）\n", max_kv);
    }

    // 会话上下文：图片块先登记（视觉塔已跑完），每轮的 user/assistant turn 累积
    // 在 Conversation 里，每轮整段重新 prefill。预算 = KV 上界减去本轮最多要生成
    // 的 token 数，超预算整对（user+assistant）丢最旧的；VKOP_MAX_CTX 可覆盖。
    Conversation conv(tok);
    for (const auto& vf : vfs) {
        conv.addBlock({vf.n_img, vf.grid_t, vf.grid_h, vf.grid_w});
    }
    int max_ctx = max_kv - max_new;
    if (const char* e = std::getenv("VKOP_MAX_CTX")) {
        const int v = std::atoi(e);
        if (v > 0) max_ctx = v;
    }
    // raw 模式（对齐参考 dump）不过 chat template；没有模板时也没法累积历史，
    // 两种情况都退化成「每条输入独立」。
    const bool raw_mode = std::getenv("VKOP_RAW_PROMPT") != nullptr;
    const bool multi_turn = !raw_mode && conv.has_template();
    if (raw_mode || !multi_turn) {
        std::printf("[ctx] 单轮模式（%s）：不累积历史\n",
                    raw_mode ? "VKOP_RAW_PROMPT" : "tokenizer.bin 无 chat template");
    } else {
        std::printf("[ctx] 多轮上下文已启用，预算 %d tokens（KV 上界 %d）\n",
                    max_ctx, max_kv);
    }
    if (!raw_mode && have_visual && !conv.has_template()) {
        // 图像内容项要靠 chat template 的占位格式渲染，没模板就渲染不出图。
        std::fprintf(stderr,
            "[warn] tokenizer.bin 没有 chat template，无法渲染图像内容 —— 忽略 --image\n");
        have_visual = false;
    }

    // Per-round reusable zero buffers (deepstack + decode attention_bias +
    // image_pad_mask). deepstack_embeds_{0,1,2}: (1, hidden) fp16 zeros.
    std::vector<uint16_t> ds_zero(arch.hidden, 0);
    std::vector<uint32_t> ds_shape = {1u, static_cast<uint32_t>(arch.hidden)};

    // Profiling state (opt-in via VKOP_PROFILE=1).
    const bool profile_mode = std::getenv("VKOP_PROFILE") != nullptr;
    PerfMetrics session_metrics;

    // ---- 跨轮 KV 前缀复用 ----
    // 上一轮结束时 KV 里装的是哪一段序列（token 级）。下一轮的 render() 只要以
    // 它为前缀，就能从 kv_len=P 接着 prefill，只算新增的那几个 token —— 多轮
    // 对话里省掉的是「把整段历史重新 encode + 重新过一遍 attention」。
    //
    // 为什么数值上成立：解码器自回归，位置 i 的 K/V 只由 ≤i 的 token 决定，
    // 后面再接什么都不影响它（attention 的 mask 决定读出什么，不影响写进去的
    // K/V）。Conversation::addAssistantTurn 特意把「生成停在停止符」那轮的收尾
    // 补齐，就是为了让下一轮序列是上一轮的合法扩展；否则前缀对不上，这里会
    // 直接放弃复用。
    struct KvCoverage {
        std::vector<uint32_t> ids;    // 模型实际吃进去过的 token（prefill 的
                                      // ids + decode 每步喂进去的那个）
        std::vector<int32_t> mm_types;
        std::vector<RenderedContext::Span> spans;
    };
    KvCoverage covered;
    // 默认开（前提检查不通过就自动退回整段 prefill，只会慢不会错）；
    // VKOP_NO_KV_REUSE=1 强制整段，用来 A/B 出「复用是否逐位相同」。
    const bool kv_reuse = std::getenv("VKOP_NO_KV_REUSE") == nullptr;

    // REPL loop.
    {
        std::string stops;
        for (uint32_t id : conv.stop_ids()) {
            stops += (stops.empty() ? "" : ",") + std::to_string(id);
        }
        std::printf("\n=== ready (max_new=%d, stop=[%s]). type a prompt, Ctrl-D to quit ===\n\n",
                    max_new, stops.c_str());
    }
    std::string line;
    int turn = 0;
    while (std::getline(std::cin, line)) {
        if (line.empty()) continue;
        // Phase boundary decode→prefill: previous prompt's decode left
        // STABLE shape caches (Reshape/Range/Slice/Expand/Cast/...) keyed to
        // its q_len=1/past_len shapes. A new prompt has a different L, so
        // blind reuse would corrupt the graph — reset every op's cache, and
        // do it BEFORE any input is filled (ResizeInput+fillToCPU).
        rt->invalidate_replay();

        // 本轮的序列描述：会话模式走 Conversation（含历史），raw 模式把这一行
        // 当成独立序列。会话图片属于第一轮，之后的轮不再重复贴图。
        RenderedContext rctx;
        if (raw_mode) {
            rctx = render_raw(tok.encode(line), conv.image_pad_id(), conv.blocks());
        } else {
            if (!multi_turn) conv.clear();
            const int n_img_this_turn =
                (turn == 0 && have_visual && conv.has_template())
                    ? static_cast<int>(vfs.size()) : 0;
            conv.addUserTurn(line, /*block_first=*/0, n_img_this_turn);
            // 环模式下丢轮次是多余的：窗口本身就是那条界，而且丢了轮次会让
            // 下一轮的 ids 对不上 kv_base，跨轮复用直接失效（退化成整段重算）。
            if (!ring && conv.trimToBudget(max_ctx)) {
                std::printf("[ctx] 超出预算 %d，已丢掉最旧的一问一答（当前历史 %d tokens）\n",
                            max_ctx, conv.context_len());
            }
            rctx = conv.render();
        }
        const std::vector<uint32_t>& ids = rctx.ids;
        const int img_count = rctx.image_tokens();
        std::printf("[prompt] %zu tokens, %zu image span(s) (history %d turns)\n",
                    ids.size(), rctx.spans.size(), static_cast<int>(conv.size()));
        std::fflush(stdout);
        if (std::getenv("VKOP_CHATDBG")) {   // 与 HF/ORT 逐 token 对齐时用
            std::printf("  [ids]");
            for (uint32_t id : ids) std::printf(" %u", id);
            std::printf("\n");
        }
        // 环模式下序列可以无限长：超出的部分丢最旧的若干行（见 kv_base）。
        // 无环时仍是硬上界 —— buffer 装不下就是装不下。
        if (!ring && static_cast<int>(ids.size()) + max_new > max_kv) {
            std::fprintf(stderr,
                "[error] 序列 %zu tokens + 最多生成 %d 超出 KV 上限 %d —— "
                "调大 --max-kv / 调小 max_new, 或设 VKOP_MAX_CTX\n",
                ids.size(), max_new, max_kv);
            break;
        }

        // ---- Prefill (q_len = L, kv_len = 0) ----
        int L = static_cast<int>(ids.size());
        std::printf("[prefill] L=%d building inputs...\n", L); std::fflush(stdout);

        // ---- 本轮从哪接着算 ----
        // 两个「前缀长度」在这一轮里是不同的东西，环模式下尤其要分开：
        //   P     —— 图里 kv_past 的值，也就是从 cache 读几行 = kv_resident；
        //   P_abs —— 新算的那几个 token 的**绝对**位置起点 = kv_base + P。
        // 无环时 base 恒为 0，两者相等，代码路径与从前逐位一致。
        //
        // 复用要求 ids 以「上一轮 KV 装的那段序列」为前缀，且前缀里的 mm 类型
        // 与图像 span 逐位一致 —— 占位 token 相同但配错图，K/V 就是错的，而
        // 这种错不会自己暴露。任何一条不满足都退回整段 prefill（P=0）。
        int P = 0, P_abs = 0;
        if (ring) {
            // 环：先把窗口推到本轮结束时的位置。本轮算完 KV 只留最后 ≤W 行，
            // 所以比窗口更旧的 token 既不在环里、也不会被任何一个查询看到 ——
            // **根本不必算**。窗口起点 nb = max(kv_base, L-W) 因此必须在算之前
            // 就定下来，而不是让每块按「装不下就丢最旧的」临时丢（那样某几行会
            // 带着已经被丢掉的 token 的上下文，结果就随块长变了）。
            //
            // 起点可以越过 KV 里已经算过的那一段（那时那些行作废，resident 归
            // 零、从头的槽位重写）。「只能丢已经写进 KV 的行」约束的是**槽位
            // 复用**，不是起点：起跳只是让被跳过的 token 不进 KV，没有槽位冲突。
            // 整张图要么留要么整个丢 —— 从图片 span 中间起跳的话，后缀里那几个
            // 占位 token 的 MRoPE 位置就没法从「纯文本序列」重新推出来。
            int nb = std::max(kv_base, L - W);
            for (const auto& sp : rctx.spans) {
                if (sp.start < nb && nb < sp.start + sp.count) nb = sp.start + sp.count;
            }
            if (nb > kv_base + kv_resident) {
                // 起点跨过了 KV 的末尾：中间那些 token 本轮不参与（它们在窗口
                // 之外，没有算好的行可以复用），旧行一并作废。
                kv_resident = 0;
                kv_head = 0;
            } else {
                // 起点落在已有行里：把它们让出去，一个字都不搬。
                const int drop = nb - kv_base;
                kv_head = (kv_head + drop) % W;
                kv_resident -= drop;
            }
            kv_base = nb;
            // KV 里那 kv_resident 行必须逐位就是 ids[kv_base ...) —— 序列只增
            // 不减，但 assistant 轮在模板里会被补收尾串，所以还是要核对一遍。
            bool ok = kv_reuse && kv_resident > 0 &&
                      kv_resident <= L - kv_base &&
                      covered.ids.size() >= static_cast<size_t>(kv_base + kv_resident) &&
                      covered.mm_types.size() >=
                          static_cast<size_t>(kv_base + kv_resident);
            for (int i = 0; ok && i < kv_resident; ++i) {
                ok = covered.ids[kv_base + i] == ids[kv_base + i] &&
                     covered.mm_types[kv_base + i] == rctx.mm_types[kv_base + i];
            }
            size_t j = 0;
            for (const auto& s : covered.spans) {
                if (!ok) break;
                if (j >= rctx.spans.size()) { ok = false; break; }
                if (s.start + s.count <= kv_base) { ++j; continue; }  // 已丢
                if (s.start >= kv_base + kv_resident) break;          // 窗口之后
                ok = rctx.spans[j].start == s.start &&
                     rctx.spans[j].count == s.count &&
                     rctx.spans[j].block == s.block;
                ++j;
            }
            for (; ok && j < rctx.spans.size(); ++j) {
                ok = rctx.spans[j].start >= kv_base + kv_resident;
            }
            if (ok) {
                P = kv_resident;
            } else {
                if (kv_resident > 0) {
                    std::printf("[prefill] 环内前缀对不上（KV 覆盖 %d..%d）"
                                " — 整段重算\n", kv_base, kv_base + kv_resident);
                }
                // 只作废行，不动 base：base 就是窗口起点，重算的是 [base, L)。
                kv_resident = 0;
                P = 0;
            }
            // 本轮要算的绝对区间是 [kv_base + kv_resident, L)：复用成功就接着
            // 已有行往下算，否则从窗口起点整段重算。窗口起点 ≥ L-W，所以
            // S = L - P_abs <= W —— 环里的 prefill 永远装得下。
            P_abs = kv_base + kv_resident;
        } else if (kv_reuse && !covered.ids.empty() && covered.ids.size() < ids.size() &&
                   Conversation::prefixMatch(covered.ids, ids)) {
            bool ok = covered.mm_types.size() == covered.ids.size();
            for (int i = 0; ok && i < static_cast<int>(covered.ids.size()); ++i) {
                ok = covered.mm_types[i] == rctx.mm_types[i];
            }
            size_t j = 0;
            for (const auto& s : covered.spans) {
                if (!ok) break;
                // span 跨在边界上就没法只复用前半段：整段重算。
                ok = s.start + s.count <= static_cast<int>(covered.ids.size()) &&
                     j < rctx.spans.size();
                if (!ok) break;
                const auto& n = rctx.spans[j++];
                ok = n.start == s.start && n.count == s.count && n.block == s.block;
            }
            for (; ok && j < rctx.spans.size(); ++j) {
                ok = rctx.spans[j].start >= static_cast<int>(covered.ids.size());
            }
            if (ok) P = static_cast<int>(covered.ids.size());
        }
        if (!ring) P_abs = P;
        const int S = L - P_abs;
        // 本轮 decode 实际喂进去的 token（prefill 的 ids 之外的部分）—— 和
        // past_len 一起长，本轮结束后就是 KV 覆盖的完整序列。
        std::vector<uint32_t> consumed_gen;
        consumed_gen.reserve(static_cast<size_t>(max_new));
        if (P > 0) {
            std::printf("[prefill] reusing %d kv positions (base=%d, abs=%d), "
                        "%d to compute\n", P, kv_base, P_abs, S);
        } else if (kv_reuse && !covered.ids.empty()) {
            std::printf("[prefill] no reusable prefix (previous kv covered %zu "
                        "tokens) — full prefill\n", covered.ids.size());
        }
        // mm_token_type_ids (1, L): 0=text, 1=image. attention_mask all 1.
        // 序列里图片位置的判定由 Conversation/render_raw 一起算好了。
        const std::vector<int32_t>& mtt = rctx.mm_types;
        std::vector<int8_t> amask(L, 1);

        // ---- 分块 prefill ----
        // 窗口起点在上面就定死了，所以 P + S <= W 恒成立，这个 loop 默认只跑
        // 一次；VKOP_PREFILL_CHUNK 把它压小用来 A/B（同一条序列、同一个窗口、
        // 不同块边界必须得到同一个结果 —— 这是「窗口只由 W 决定」的直接检验），
        // 也用来给 lm_head 的 logits buffer 封顶。每块的丢行因此恒为 0，留着
        // 只是兜底。
        bool chunked = ring && P + S > W;
        int C_max = std::max(1, W / 2);
        // 调试用：把块长压小并强制分块（W 不变，所以一块行都不会丢）。这样
        // 「分块 vs 单块」能在同一条序列上直接对比 —— 没有窗口损失时两者应当
        // 只差一个 softmax 归约顺序。
        if (const char* e = std::getenv("VKOP_PREFILL_CHUNK")) {
            const int v = std::atoi(e);
            if (v > 0) {
                C_max = v;
                chunked = ring && S > C_max;
            }
        }
        const int n_chunks = chunked ? (S + C_max - 1) / C_max : 1;
        if (chunked) {
            std::printf("[prefill] 分块 prefill：本轮 %d 个 token，每块 ≤%d，"
                        "共 %d 块（窗口 %d 行，起点 base=%d）\n",
                        S, C_max, n_chunks, W, kv_base);
            std::fflush(stdout);
        }
        double ms = 0, prefill_chunk_ms = 0;
        int64_t rope_delta = 0;   // 每块都算（同值），decode 循环还要用
        for (int blk = 0; blk < n_chunks; ++blk) {
        // 本块的绝对起点 A、长度 C，以及本块能从 cache 读到几行 Pc。
        const int A = chunked ? P_abs + blk * C_max : P_abs;
        const int C = chunked ? std::min(C_max, P_abs + S - A) : S;
        if (chunked) {
            // 丢的只能是已经写进 KV 的最旧那些行。kv_resident + C <= W 是
            // Attention 侧的硬约束：新的 C 行要写进还空着的槽。
            const int drop = std::max(0, kv_resident + C - W);
            kv_head = (kv_head + drop) % W;
            kv_resident -= drop;
            kv_base += drop;
        }
        const int Pc = chunked ? kv_resident : P;

        // inputs_embeds (1, C, hidden): embed this chunk's tokens, then scatter
        // the visual image_features rows into the image-pad positions (多模态).
        // 取的是 [A, A+C) 与 span 的**交集** —— 图比窗口还大时块边界必然切在
        // 图中间，那时从该图第 i = j - sp.start 行取起。
        const std::vector<uint32_t> chunk_ids(ids.begin() + A, ids.begin() + A + C);
        auto emb = embed_lookup(embed_table, chunk_ids, vocab, arch.hidden);
        std::vector<uint16_t> ds_turn;   // (img_in_chunk, hidden)
        int img_in_chunk = 0;
        for (const auto& sp : rctx.spans) {
            img_in_chunk += std::max(0, std::min(sp.start + sp.count, A + C) -
                                            std::max(sp.start, A));
        }
        if (have_visual && img_in_chunk > 0) {
            ds_turn.assign(static_cast<size_t>(img_in_chunk) * arch.hidden, 0);
            int feat_row = 0;   // running row index into emb / ds_turn
            for (const auto& sp : rctx.spans) {
                const int lo = std::max(sp.start, A);
                const int hi = std::min(sp.start + sp.count, A + C);
                if (hi <= lo) continue;
                const int row0 = feat_row0[sp.block];
                for (int j = lo; j < hi; ++j, ++feat_row) {
                    const int i = j - sp.start;   // 该图内的第几行
                    std::memcpy(&emb[static_cast<size_t>(j - A) * arch.hidden],
                                &img_feat_cat[static_cast<size_t>(row0 + i) * arch.hidden],
                                arch.hidden * sizeof(uint16_t));
                    for (int d = 0; d < 3; ++d)
                        std::memcpy(&ds_turn[static_cast<size_t>(feat_row) * arch.hidden],
                                    &ds_cat[d][static_cast<size_t>(row0 + i) * arch.hidden],
                                    arch.hidden * sizeof(uint16_t));
                }
            }
        }
        fill_fp16_input(rt, "inputs_embeds", {1u, static_cast<uint32_t>(C),
                      static_cast<uint32_t>(arch.hidden)}, emb.data());
        std::printf("  inputs_embeds ok\n"); std::fflush(stdout);

        // position_ids: 纯文本 2D (1, S) 或 多模态 3D (3, 1, S) MRoPE.
        // 纯文本用 arange；多模态走 get_rope_index 拿 rope_delta（decode 时用）。
        // 位置是绝对位置，所以复用前缀时两种形式都要带上 P 的偏移 —— 3D 更是
        // 必须在**整段** ids 上算一次（MRoPE 的位置是顺着序列走出来的），再切
        // 尾部 S 个，才能保证前缀那几行的位置和上一轮算出的一模一样。
        rope_delta = 0;
        if (arch.position_ids_dims == 3) {
            // 本轮实际用到的图（按出现顺序）的 grid_thw，从 ImageBlock 现取。
            std::vector<int> span_grid;
            for (const auto& sp : rctx.spans) {
                const ImageBlock& blk = conv.blocks()[sp.block];
                span_grid.insert(span_grid.end(),
                                 {blk.grid_t, blk.grid_h, blk.grid_w});
            }
            auto ri = get_rope_index(reinterpret_cast<const int64_t*>(ids.data()),
                                     mtt.data(), amask.data(),
                                     have_visual && img_count > 0
                                         ? span_grid.data() : nullptr,
                                     have_visual && img_count > 0
                                         ? static_cast<int>(rctx.spans.size()) : 0,
                                     /*B=*/1, L);
            std::vector<int64_t> pos3(static_cast<size_t>(3) * C);
            for (int c = 0; c < 3; ++c) {
                for (int i = 0; i < C; ++i) {
                    pos3[static_cast<size_t>(c) * C + i] =
                        ri.pos_ids[static_cast<size_t>(c) * L + A + i];
                }
            }
            fill_i64_input(rt, "position_ids", {3u, 1u, static_cast<uint32_t>(C)},
                           pos3.data());
            rope_delta = ri.rope_delta[0];
        } else {
            // 2D 标准 RoPE: (1, S) = arange(P_abs, P_abs+S)。无 rope_delta。
            // 环模式下 P_abs = kv_base + kv_resident，位置仍是绝对位置 ——
            // RoPE 只吃 position_ids，图自己的 kv_past 只是「读几行 cache」。
            std::vector<int64_t> pos(static_cast<size_t>(C));
            for (int i = 0; i < C; ++i) pos[i] = A + i;
            fill_i64_input(rt, "position_ids", {1u, static_cast<uint32_t>(C)},
                           pos.data());
        }
        std::printf("  position_ids ok (%dD, rope_delta=%lld)\n",
                    arch.position_ids_dims, (long long)rope_delta); std::fflush(stdout);

        // attention_bias (1, 1, C, Pc+C) causal: 查询 i 在本块的坐标系里看的是
        // [0, Pc+i]（Pc = cache 的行数）。causal_bias 的 threshold 就是按 kv-q
        // 算的，分块后每块都是同一个式子。绝对位置只通过 position_ids 进 RoPE。
        {
            auto ab = causal_bias(C, Pc + C);
            fill_fp16_input(rt, "attention_bias", {1u, 1u, static_cast<uint32_t>(C),
                          static_cast<uint32_t>(Pc + C)}, ab.data());
        }
        std::printf("  attention_bias ok\n"); std::fflush(stdout);

        // deepstack_embeds_{0,1,2}: 仅多模态模型有此输入。多图时 HF 把每张
        // 图的 deepstack 按 prompt 顺序拼成 (total_img, hidden) 一个输入。
        if (arch.has_deepstack) {
            if (have_visual && img_in_chunk > 0) {
                for (int d = 0; d < 3; ++d)
                    fill_fp16_input(rt, "deepstack_embeds_" + std::to_string(d),
                                    {static_cast<uint32_t>(img_in_chunk),
                                     static_cast<uint32_t>(arch.hidden)},
                                    ds_turn.data());
            } else {
                for (int d = 0; d < 3; ++d)
                    fill_fp16_input(rt, "deepstack_embeds_" + std::to_string(d),
                                    ds_shape, ds_zero.data());
            }
            std::printf("  deepstack ok\n"); std::fflush(stdout);
        }

        // image_pad_mask (1, S): 仅多模态模型有此输入。
        if (arch.has_image_pad_mask) {
            std::vector<int8_t> mask(C, 0);
            for (int i = 0; i < C; ++i)
                mask[i] = mtt[A + i] ? 1 : 0;
            rt->ResizeInput("image_pad_mask", {1u, static_cast<uint32_t>(C)});
            auto t = rt->GetInput("image_pad_mask");
            auto tg = as_tensor<int8_t>(t);
            if (tg->num_elements() > 0) tg->fillToCPU(mask.data());
            std::printf("  image_pad_mask ok\n"); std::fflush(stdout);
        }

        // past：复用前缀只是把逻辑 kv_len 设成 Pc（buffer 里那 Pc 行是上一轮/
        // 上一块写下的，一个字都不动），只有本轮的**第一块**在整段重算时才需要
        // 归零 + 重新上传。环模式下 Pc 是窗口内的行数，`present==past` 那块
        // buffer 里的行本来就是按时间排的，所以只靠 kv_ring_head 给槽位偏移。
        if (Pc > 0) {
            kv.resume_from(Pc);
        } else if (blk == 0) {
            kv.reset_for_prefill();
            kv.upload();
        } else {
            kv.resume_from(0);   // 窗口丢空了：buffer 内容无所谓，读 0 行
        }
        // 环头必须在 Run() 之前写进去：它和 kv_len 一样是本轮的调度输入，变了就
        // 得让 replay 失效（PushConstant 的指纹会自己做到）。
        if (ring) fill_i32_input(rt, "kv_ring_head", kv_head);
        std::printf("  past_kv resize ok (kv_len=%d, head=%d)\n", Pc,
                    ring ? kv_head : 0); std::fflush(stdout);

        // Upload all inputs (跳过模型没有的 deepstack / image_pad_mask).
        upload_input(cmdpool, rt->GetInput("inputs_embeds"));
        upload_input(cmdpool, rt->GetInput("position_ids"));
        upload_input(cmdpool, rt->GetInput("attention_bias"));
        if (arch.has_deepstack) {
            for (int d = 0; d < 3; ++d)
                upload_input(cmdpool, rt->GetInput("deepstack_embeds_" + std::to_string(d)));
        }
        if (arch.has_image_pad_mask)
            upload_input(cmdpool, rt->GetInput("image_pad_mask"));
        std::printf("  upload ok, calling Run()...\n"); std::fflush(stdout);

        auto t_prefill_start = std::chrono::steady_clock::now();
        double run_ms = rt->Run();
        auto t_prefill_end = std::chrono::steady_clock::now();
        ms += run_ms;
        if (profile_mode) {
            prefill_chunk_ms += std::chrono::duration<double, std::milli>(
                t_prefill_end - t_prefill_start).count();
            session_metrics.prefill_tokens = L;
        }
        std::printf("  Run done %.1fms\n", run_ms); std::fflush(stdout);
        // Prefill (q_len=C) and decode (q_len=1) have entirely different
        // shapes; the replay cache from prefill must not bleed into decode —
        // and a following chunk of another length must not reuse it either.
        rt->invalidate_replay();
        // Wait for GPU compute to finish, but do NOT ReadResult() — that would
        // copyToCPU all 28 present_kv outputs (28 sync points) which we don't
        // need: feedback_kv does device→device, and argmax reads only logits.
        cmdpool->getVulkanDevice()->wait_all_done();

        if (chunked) {
            kv_resident += C;   // 本块的 C 行已经写进环里
            std::printf("[prefill] 第 %d/%d 块: A=%d C=%d Pc=%d → kv=%d "
                        "(base=%d, head=%d) %.1fms\n",
                        blk + 1, n_chunks, A, C, Pc, kv_resident, kv_base,
                        kv_head, run_ms);
            std::fflush(stdout);
        }
        }   // for (blk)
        if (profile_mode) session_metrics.prefill_ms = prefill_chunk_ms;

        // Optional named-intermediate dump (mirrors llm_driver's VKOP_DUMP_TENSORS).
        // VKOP_DUMP_ROUND=0 restricts to prefill; "*" dumps every fp16 tensor.
        // Output matches dump_ort_intermediates.py for line-by-line topological diff.
        if (const char *d = std::getenv("VKOP_DUMP_TENSORS")) {
            std::string s(d);
            std::vector<std::pair<std::string, std::shared_ptr<ITensor>>> items;
            if (s == "*") {
                items = rt->ListTensors();
            } else {
                std::stringstream ss(s);
                std::string nm;
                while (std::getline(ss, nm, ',')) {
                    if (nm.empty()) continue;
                    items.push_back({nm, rt->GetTensor(nm)});
                }
            }
            for (auto &it : items) {
                const std::string &nm = it.first;
                auto &tns = it.second;
                if (!tns) { std::printf("[%s] NOT FOUND\n", nm.c_str()); continue; }
                if (tns->dtype() == typeid(int64_t)) {
                    if (!std::getenv("VKOP_DUMP_INT64") ||
                        std::getenv("VKOP_DUMP_INT64")[0] != '1') continue;
                    auto tg = as_tensor<int64_t>(tns);
                    if (!tg->has_gpu_buffer()) continue;
                    tg->copyToCPU(cmdpool);
                    const int64_t *p = tg->data().data();
                    int ne = tg->num_elements();
                    std::printf("[%s] ne=%d int64=[", nm.c_str(), ne);
                    for (int i = 0; i < 16 && i < ne; ++i) std::printf("%lld,", (long long)p[i]);
                    std::printf("]\n");
                    continue;
                }
                if (tns->dtype() == typeid(float)) {
                    auto tg = as_tensor<float>(tns);
                    if (!tg->has_gpu_buffer()) { std::printf("[%s] no GPU buffer\n", nm.c_str()); continue; }
                    tg->copyToCPU(cmdpool);
                    const float *p = tg->data().data();
                    int ne = tg->num_elements();
                    int nan=0,inf=0,zero=0; float mn=1e30f,mx=-1e30f;
                    for(int i=0;i<ne;++i){float v=p[i];
                        if(std::isnan(v))nan++;else if(std::isinf(v))inf++;
                        if(v==0.f)zero++;
                        if(!std::isnan(v)&&!std::isinf(v)){if(v>mx)mx=v;if(v<mn)mn=v;}}
                    std::printf("[%s] ne=%d nan=%d inf=%d zero=%d min=%.4g max=%.4g first=[",
                                nm.c_str(), ne, nan, inf, zero, mn, mx);
                    for(int i=0;i<16&&i<ne;++i)std::printf("%.4g,",p[i]);
                    std::printf("]\n");
                    if (const char *off_env = std::getenv("VKOP_DUMP_OFF")) {
                        std::string oe(off_env);
                        auto colon = oe.find(':');
                        if (colon != std::string::npos && oe.substr(0,colon)==nm) {
                            int off = std::stoi(oe.substr(colon+1));
                            std::printf("[%s@%d] ", nm.c_str(), off);
                            for(int i=0;i<16&&off+i<ne;++i)std::printf("%.4g,",p[off+i]);
                            std::printf("]\n");
                        }
                    }
                    continue;
                }
                if (tns->dtype() != typeid(uint16_t)) continue;
                auto tg = as_tensor<uint16_t>(tns);
                if (!tg->has_gpu_buffer()) { std::printf("[%s] no GPU buffer\n", nm.c_str()); continue; }
                tg->copyToCPU(cmdpool);
                const uint16_t *p = reinterpret_cast<const uint16_t*>(tg->data().data());
                int ne = tg->num_elements();
                int nan=0,inf=0,zero=0; float mn=1e30f,mx=-1e30f;
                for(int i=0;i<ne;++i){float v=ITensor::fp16_to_fp32(p[i]);
                    if(std::isnan(v))nan++;else if(std::isinf(v))inf++;
                    if(v==0.f)zero++;
                    if(!std::isnan(v)&&!std::isinf(v)){if(v>mx)mx=v;if(v<mn)mn=v;}}
                std::printf("[%s] ne=%d nan=%d inf=%d zero=%d min=%.4g max=%.4g first=[",
                            nm.c_str(), ne, nan, inf, zero, mn, mx);
                for(int i=0;i<16&&i<ne;++i)std::printf("%04x,",p[i]);
                std::printf("]\n");
                // DEBUG: dump 16 elements at a configurable offset (VKOP_DUMP_OFF=name:off)
                if (const char *off_env = std::getenv("VKOP_DUMP_OFF")) {
                    std::string oe(off_env);
                    auto colon = oe.find(':');
                    if (colon != std::string::npos && oe.substr(0,colon)==nm) {
                        int off = std::stoi(oe.substr(colon+1));
                        std::printf("[%s@%d] ", nm.c_str(), off);
                        for(int i=0;i<16&&off+i<ne;++i)std::printf("%.4g,",ITensor::fp16_to_fp32(p[off+i]));
                        std::printf("]\n");
                    }
                }
            }
            std::fflush(stdout);
        }

        int next_id = argmax_last_token(rt, cmdpool, vocab);
        std::printf("[prefill] %.1fms  → token %d  ", ms, next_id);
        std::vector<uint32_t> out_ids = {static_cast<uint32_t>(next_id)};
        std::printf("%s\n", tok.decode({static_cast<uint32_t>(next_id)}).c_str());
        std::fflush(stdout);

        // Feed KV cache back for decode rounds.
        // 无环：feedback() 报的就是 KV 里现在的行数（== L）。环模式下它是
        // 窗口内的行数：单块时就是 P+S；分块时块循环里已经跟着丢行减过了，
        // 行数**不会**超过 W（那些越界的行在块边界上就被丢掉了）。
        int past_len;
        if (ring) {
            if (!chunked) kv_resident = P + S;
            past_len = kv_resident;
        } else {
            past_len = kv.feedback();   // == L，KV 现在装着整段 prompt
        }
        int dec_base = kv_base;   // 本轮 decode 期间逻辑行 0 的绝对位置

        // ---- Decode loop (q_len = 1) ----
        for (int step = 1; step < max_new; ++step) {
            if (conv.isStop(static_cast<uint32_t>(next_id))) {
                std::printf("[done] 本轮结束\n");
                break;
            }
            // cur_emb (1, 1, hidden) from the single next_id.
            auto cur_emb = embed_lookup(embed_table,
                                        {static_cast<uint32_t>(next_id)},
                                        vocab, arch.hidden);
            // 这个 token 马上要被模型吃进 KV，记下来（下一轮判断能否复用前缀）。
            consumed_gen.push_back(static_cast<uint32_t>(next_id));
            fill_fp16_input(rt, "inputs_embeds", {1u, 1u, static_cast<uint32_t>(arch.hidden)},
                          cur_emb.data());

            // 环：新的一行要进去，就得先给最旧的一行腾槽。丢的是 cache 第 0
            // 行（kv_base 那一行），头上那一格挪一格、行数减一，绝对位置前移
            // —— 一步里只动 host 侧的三个整数。每一步都要把 past 的逻辑
            // kv_len 摆成「本步从 cache 读几行」，非环路径是 feedback() 在做。
            if (ring) {
                if (past_len + 1 > W) {
                    kv_head = (kv_head + 1) % W;
                    past_len -= 1;
                    dec_base += 1;
                }
                kv_resident = past_len;
                kv.resume_from(kv_resident);
                fill_i32_input(rt, "kv_ring_head", kv_head);
            }

            // position_ids: 2D 纯文本 (1, 1) = 绝对位置；3D 多模态 (3, 1, 1)
            // = 绝对位置 + rope_delta（MRoPE delta）。环模式下图的 kv_past 是
            // 窗口内的序号，绝对位置是 dec_base + past_len。
            {
                int64_t p = static_cast<int64_t>(dec_base + past_len) + rope_delta;
                if (arch.position_ids_dims == 3) {
                    int64_t pos[3] = {p, p, p};
                    fill_i64_input(rt, "position_ids", {3u, 1u, 1u}, pos);
                } else {
                    fill_i64_input(rt, "position_ids", {1u, 1u}, &p);
                }
            }
            // attention_bias (1, 1, 1, past_len+1) all zero (full history).
            {
                std::vector<uint16_t> ab(static_cast<size_t>(past_len + 1), 0);
                fill_fp16_input(rt, "attention_bias", {1u, 1u, 1u,
                              static_cast<uint32_t>(past_len + 1)}, ab.data());
            }
            // deepstack zeros / image_pad_mask false（仅模型有这些输入时才写）。
            if (arch.has_deepstack) {
                for (int d = 0; d < 3; ++d)
                    fill_fp16_input(rt, "deepstack_embeds_" + std::to_string(d),
                                    ds_shape, ds_zero.data());
            }
            if (arch.has_image_pad_mask) {
                std::vector<int8_t> mask(1, 0);
                rt->ResizeInput("image_pad_mask", {1u, 1u});
                auto t = rt->GetInput("image_pad_mask");
                as_tensor<int8_t>(t)->fillToCPU(mask.data());
            }

            // Upload.
            upload_input(cmdpool, rt->GetInput("inputs_embeds"));
            upload_input(cmdpool, rt->GetInput("position_ids"));
            upload_input(cmdpool, rt->GetInput("attention_bias"));
            if (arch.has_deepstack) {
                for (int d = 0; d < 3; ++d)
                    upload_input(cmdpool, rt->GetInput("deepstack_embeds_" + std::to_string(d)));
            }
            if (arch.has_image_pad_mask)
                upload_input(cmdpool, rt->GetInput("image_pad_mask"));

            auto t0 = std::chrono::steady_clock::now();
            ms = rt->Run();
            auto t1 = std::chrono::steady_clock::now();
            next_id = argmax_last_token(rt, cmdpool, vocab);
            out_ids.push_back(static_cast<uint32_t>(next_id));
            double run_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            if (profile_mode) {
                session_metrics.decode_steps++;
                session_metrics.total_decode_ms += run_ms;
                session_metrics.per_token_ms.push_back(run_ms);
            }
            std::printf("[r%d] %.1fms  past_len=%d pos=%lld%s  → %d  %s\n", step,
                        run_ms, past_len,
                        (long long)(dec_base + past_len + rope_delta),
                        ring ? " ring" : "",
                        next_id,
                        tok.decode({static_cast<uint32_t>(next_id)}).c_str());
            std::fflush(stdout);

            // 非环：present→past 的逻辑长度跟进（== past_len+1）。环：自己记，
            // 末尾把会话级的三个整数交回去给下一轮。
            if (ring) {
                past_len += 1;
                kv_resident = past_len;
            } else {
                past_len = kv.feedback();
            }
        }
        if (ring) kv_base = dec_base;   // 下一轮接在这里

        std::printf("\n=== full decode ===\n%s\n\n", tok.decode(out_ids).c_str());

        // Update session-wide metrics.
        if (profile_mode) {
            session_metrics.total_prompt_tokens += L;
            int generated = static_cast<int>(out_ids.size()) - 1;  // first token from prefill
            session_metrics.total_generated_tokens += generated;
            session_metrics.total_rounds++;

            // Per-round summary.
            std::printf("[profile] round %d:\n", turn + 1);
            std::printf("  prefill: %d tokens in %.1fms (%.1f tok/s)\n",
                        session_metrics.prefill_tokens, session_metrics.prefill_ms,
                        session_metrics.prefill_tok_per_sec());
            std::printf("  decode:  %d tokens in %.1fms (%.1f tok/s, avg %.1fms/token)\n",
                        session_metrics.decode_steps, session_metrics.total_decode_ms,
                        session_metrics.decode_tok_per_sec(), session_metrics.avg_decode_ms());
            std::printf("  kv cache: %d/%d (%.1f%%)\n",
                        static_cast<int>(ids.size()) + session_metrics.decode_steps,
                        max_kv,
                        (static_cast<double>(ids.size()) + session_metrics.decode_steps) / max_kv * 100.0);
            std::printf("\n");
            std::fflush(stdout);
        }

        // 回复原样进历史（不重新分词）：下一轮整段重新 prefill 时，模型看到的
        // 就是它自己上一轮写下的 token。
        if (!raw_mode) {
            conv.addAssistantTurn(out_ids);
            ++turn;
        }

        // KV 现在覆盖的序列：prefill 的 ids + decode 每步喂进去的 token（停止符
        // 被生成出来但没喂进去，所以不在其中 —— 与 past_len 一致）。下一轮拿它
        // 做前缀校验；模板补齐收尾串（addAssistantTurn）正是为了让下一轮的 ids
        // 是它的合法扩展。
        covered.ids = ids;
        covered.ids.insert(covered.ids.end(), consumed_gen.begin(),
                           consumed_gen.end());
        covered.mm_types = rctx.mm_types;
        covered.mm_types.resize(covered.ids.size(), 0); // 生成出来的都是文本
        covered.spans = rctx.spans;
    }

    // Session-wide summary on exit.
    if (profile_mode && session_metrics.total_rounds > 0) {
        std::printf("\n[profile] session summary (%d rounds):\n", session_metrics.total_rounds);
        std::printf("  total prompt tokens: %d\n", session_metrics.total_prompt_tokens);
        std::printf("  total generated tokens: %d\n", session_metrics.total_generated_tokens);
        std::printf("  avg prefill tok/s: %.1f\n",
                    session_metrics.total_prompt_tokens /
                    (session_metrics.prefill_ms / 1000.0 * session_metrics.total_rounds));
        std::printf("  avg decode tok/s: %.1f\n",
                    session_metrics.total_generated_tokens /
                    (session_metrics.total_decode_ms / 1000.0));
        std::printf("  p50 decode latency: %.1fms\n", session_metrics.percentile(50));
        std::printf("  p90 decode latency: %.1fms\n", session_metrics.percentile(90));
        std::printf("  p99 decode latency: %.1fms\n", session_metrics.percentile(99));
        std::printf("\n");
        std::fflush(stdout);
    }

    return 0;
}
