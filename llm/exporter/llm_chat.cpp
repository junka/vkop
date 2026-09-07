// junka @ 2026
// End-to-end conversational driver for llm.vkopbin (Qwen3-VL-2B LLM, buffer
// backend). Text in → generated text out, no .npy dumps required.
//
// Reuses:
//   - llm/tokenizer   (BBPE encode/decode + chat template)
//   - llm.vkopbin     (the 3649-node LLM graph, KV-cache as explicit I/O)
//   - embed_tokens.bin (standalone [vocab, 2048] fp16 embedding table, exported
//     by qwen3vl_export_onnx.py — the graph takes inputs_embeds, not input_ids,
//     so the embedding lookup is done host-side here)
//
// The embedding lookup is a pure 1:1 row gather (token_id → row of HIDDEN fp16),
// so it is done on the CPU (EmbeddingForward op is for multi-hot reduce, overkill
// here and wants float). For L prefill tokens + 1 decode token the cost is nil.
//
// Text-only (delta=0): position_ids = [0..L-1] replicated across the 3 MRoPE
// axes; image_pad_mask all false; deepstack_embeds all zero. Multimodal would
// need get_rope_index's M-RoPE delta + visual features — out of scope v1.
//
// Usage:
//   llm_chat <model.vkopbin> <embed_tokens.bin> <tokenizer.bin> [max_new]
//   (then type prompts on stdin, Ctrl-D to quit)
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

#define STB_IMAGE_IMPLEMENTATION
#include "include/stb_image.h"

using vkop::VulkanInstance;
using vkop::VulkanDevice;
using vkop::VulkanCommandBuffer;
using vkop::core::ITensor;
using vkop::core::Runtime;
using vkop::core::as_tensor;
using vkop::export_::get_rope_index;
using vkop::export_::preprocess_image_noresize;
using qwen::Tokenizer;

namespace {

constexpr int HIDDEN = 2048;
constexpr int NKV = 8;
constexpr int HD = 128;
constexpr int NLAYERS = 28;
constexpr uint32_t IM_END = 151645;
constexpr uint32_t IMAGE_PAD = 151655;  // <|image_pad|>
constexpr uint16_t FP16_NEG_INF = 0xFC00;  // -inf in fp16 (use finfo.min -65504)
// Qwen3-VL uses torch.finfo(float16).min ≈ -65504 as the causal mask fill, not
// -inf, so softmax keeps a tiny but finite distinction. -65504 = 0xFBFF.
constexpr uint16_t FP16_MIN = 0xFBFF;

// ---- fp16 helpers (match ITensor::fp16_to_fp32 / fp32_to_fp16) ----
inline float fp16_to_f32(uint16_t h) { return ITensor::fp16_to_fp32(h); }

// Read the whole embed_tokens.bin (raw fp16, [vocab, HIDDEN]) into memory.
// vocab is inferred from the file size (bytes / (HIDDEN*2)) so the driver does
// not depend on a tokenizer vocab accessor. Returns (buffer, vocab).
std::pair<std::vector<uint16_t>, int>
load_embed_table(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error("cannot open " + path);
    auto bytes = static_cast<size_t>(f.tellg());
    f.seekg(0);
    if (bytes % (HIDDEN * sizeof(uint16_t)) != 0)
        throw std::runtime_error("embed_tokens.bin size not a multiple of HIDDEN*2");
    int vocab = static_cast<int>(bytes / (HIDDEN * sizeof(uint16_t)));
    std::vector<uint16_t> buf(static_cast<size_t>(vocab) * HIDDEN);
    f.read(reinterpret_cast<char*>(buf.data()),
           static_cast<std::streamsize>(buf.size() * sizeof(uint16_t)));
    if (!f) throw std::runtime_error("short read on " + path);
    return {std::move(buf), vocab};
}

// Lookup L token ids → fp16 embedding rows, laid out as (1, L, HIDDEN).
std::vector<uint16_t> embed_lookup(const std::vector<uint16_t>& table,
                                   const std::vector<uint32_t>& ids, int vocab) {
    std::vector<uint16_t> out(ids.size() * HIDDEN, 0);
    for (size_t i = 0; i < ids.size(); ++i) {
        uint32_t id = ids[i];
        if (id >= static_cast<uint32_t>(vocab)) {
            std::fprintf(stderr, "[embed] token id %u >= vocab %d, zeroing\n", id, vocab);
            continue;
        }
        std::memcpy(&out[i * HIDDEN], &table[static_cast<size_t>(id) * HIDDEN],
                    HIDDEN * sizeof(uint16_t));
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
int argmax_last_token(const std::shared_ptr<Runtime>& rt,
                      const std::shared_ptr<vkop::VulkanCommandPool>& cmdpool) {
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
        vocab = 151936;
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

// Copy present_key_values_{i} output → past_key_values_{i} input for the next
// round, entirely on the GPU (device→device, no CPU round-trip). present shape
// is (1,2,NKV,kv_len,128); past for next round takes the same shape (kv_len
// already includes the just-appended token). All 28 layers' copies are recorded
// into ONE command buffer and submitted with a single wait — vs the old path
// which did 28 separate copyToCPU+copyToGPU cycles (56 sync points).
//
// Both past and present buffers are pre-allocated to MAX_KV (see
// preallocate_buffer in LoadModel setup), so ResizeInput on past keeps the
// same physical VkBuffer (prealloc_keep_) and the device→device copy writes
// the logical region into the reused buffer.
void feedback_kv(const std::shared_ptr<Runtime>& rt,
                 const std::shared_ptr<vkop::VulkanCommandPool>& cmdpool) {
    auto dev = cmdpool->getVulkanDevice();
    // First pass: derive kv_len + ResizeInput past (logical shape only; buffer
    // reused via prealloc_keep_). Must happen before the copy pass because
    // ResizeInput sets converted_=false (off-GPU), and as_storage_buffer below
    // re-marks the buffer for the copy.
    std::vector<int> kv_lens(NLAYERS);
    for (int i = 0; i < NLAYERS; ++i) {
        auto pres = rt->GetOutput("present_key_values_" + std::to_string(i));
        auto pres_t = as_tensor<uint16_t>(pres);
        int kv_len = pres_t->num_elements() / (2 * NKV * HD);
        kv_lens[i] = kv_len;
        std::vector<uint32_t> u32shape = {
            1u, 2u, static_cast<uint32_t>(NKV),
            static_cast<uint32_t>(kv_len), static_cast<uint32_t>(HD)};
        rt->ResizeInput("past_key_values_" + std::to_string(i), u32shape);
    }
    // Single command buffer for all 28 layers' device→device copies.
    VulkanCommandBuffer cmd(cmdpool);
    cmd.begin();
    for (int i = 0; i < NLAYERS; ++i) {
        auto pres = as_tensor<uint16_t>(
            rt->GetOutput("present_key_values_" + std::to_string(i)));
        auto past = as_tensor<uint16_t>(
            rt->GetInput("past_key_values_" + std::to_string(i)));
        // as_storage_buffer reuses the pre-allocated VkBuffer (prealloc_keep_,
        // size >= aligned). present: read barrier for the copy source; past:
        // the copyStageBufferToBuffer call below adds the write barrier.
        auto pres_buf = pres->as_storage_buffer(dev, nullptr);
        auto past_buf = past->as_storage_buffer(dev, nullptr);
        // present was written by the Concat compute shader last round; barrier
        // it to TRANSFER_READ before the copy. size = logical bytes (kv_len
        // region); the buffer is oversized but only the logical region holds
        // valid data.
        VkDeviceSize copy_bytes = static_cast<VkDeviceSize>(
            2 * NKV * kv_lens[i] * HD * sizeof(uint16_t));
        if (copy_bytes == 0) {
            // kv_len==0 (shouldn't happen post-prefill, but guard anyway):
            // nothing to copy, just mark past on-GPU.
            continue;
        }
        pres_buf->transferReadBarrier(cmd.get(), copy_bytes, 0);
        // past->copyStageBufferToBuffer barriers past to TRANSFER_WRITE, copies
        // from present's VkBuffer, then restores past's access. The source is
        // the present VkBuffer (already barriered above).
        past_buf->copyStageBufferToBuffer(cmd.get(), pres_buf->getBuffer(),
                                          0, copy_bytes, 0);
        // past is now populated on the GPU; mark on-GPU so the next Run()'s
        // input binding reuses the buffer without a host upload.
        past->toGPU();
    }
    cmd.end();
    cmd.submit(dev->getComputeQueue());
    cmd.wait();  // single sync point for all 28 layers
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

// Visual encoder run: load image, C++-preprocess, run visual.vkopbin, return
// the 4 outputs (image_features + 3 deepstack), each (n_img, HIDDEN) fp16.
// Also returns grid_thw (T,H,W) for rope_index.
struct VisualFeatures {
    std::vector<uint16_t> image_features;   // (n_img * HIDDEN)
    std::vector<uint16_t> deepstack[3];     // each (n_img * HIDDEN)
    int n_img = 0;
    int grid_t = 0, grid_h = 0, grid_w = 0;
};

VisualFeatures run_visual(const std::shared_ptr<vkop::VulkanCommandPool>& cmdpool,
                          const std::string& visual_vkopbin,
                          const std::string& image_path) {
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
    // visual encoder consumes seq_len = grid_t*grid_h*grid_w patches (pre-merge)
    // but emits image_features of (n_img, HIDDEN) where n_img = that / merge^2
    // (the spatial merge happens inside the visual graph). For 224x224:
    // seq_len=196, n_img=49.
    vf.n_img = pr.seq_len / (vkop::export_::kMerge * vkop::export_::kMerge);
    std::printf("[visual] grid_thw=[%d,%d,%d] n_img=%d seq_len=%d\n",
                vf.grid_t, vf.grid_h, vf.grid_w, vf.n_img, pr.seq_len);

    auto vrt = std::make_shared<Runtime>(cmdpool, visual_vkopbin, /*precision=*/1);
    vrt->set_backend_buffer(true);
    vrt->LoadModel();
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

// Expand the single <|image_pad|> token (151655) in `ids` into n_img copies,
// so the sequence length matches the visual feature count (HF processor does
// this expansion based on grid_thw). Returns the expanded ids + the index of
// the first image token (for mm_token_type_ids / image_pad_mask / scatter).
// Returns {ids unchanged, -1} if no image token present.
struct ExpandedIds {
    std::vector<uint32_t> ids;
    int img_start = -1;   // first image-token index in the expanded sequence
    int img_count = 0;    // number of image tokens (0 if none)
};
ExpandedIds expand_image_token(const std::vector<uint32_t>& ids, int n_img) {
    ExpandedIds ex;
    int single = -1;
    for (size_t i = 0; i < ids.size(); ++i) {
        if (ids[i] == IMAGE_PAD) { single = static_cast<int>(i); break; }
    }
    if (single < 0 || n_img <= 0) {
        ex.ids = ids;
        return ex;
    }
    ex.ids.reserve(ids.size() - 1 + n_img);
    for (int i = 0; i < single; ++i) ex.ids.push_back(ids[i]);
    ex.img_start = single;
    ex.img_count = n_img;
    for (int i = 0; i < n_img; ++i) ex.ids.push_back(IMAGE_PAD);
    for (size_t i = single + 1; i < ids.size(); ++i) ex.ids.push_back(ids[i]);
    return ex;
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 4) {
        std::fprintf(stderr,
            "usage: %s <llm.vkopbin> <embed_tokens.bin> <tokenizer.bin> [max_new] "
            "[--image <img> --visual <visual.vkopbin>]\n",
            argv[0]);
        return 1;
    }
    const std::string model_path = argv[1];
    const std::string embed_path = argv[2];
    const std::string tok_path = argv[3];
    int max_new = 64;
    std::string image_path, visual_path;
    // Parse optional positional max_new + --image/--visual flags (any order).
    bool have_max = false;
    for (int i = 4; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--image" && i + 1 < argc) { image_path = argv[++i]; }
        else if (a == "--visual" && i + 1 < argc) { visual_path = argv[++i]; }
        else if (!a.empty() && a[0] != '-' && !have_max) {
            max_new = std::atoi(a.c_str()); have_max = true;
        }
    }
    const bool multimodal = !image_path.empty() && !visual_path.empty();

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

    // Embedding table. vocab inferred from file size.
    auto [embed_table, vocab] = load_embed_table(embed_path);
    std::printf("[embed] loaded %s  vocab=%d  hidden=%d  (~%.0fMB, %zu rows)\n",
                embed_path.c_str(), vocab, HIDDEN,
                static_cast<double>(vocab) * HIDDEN * 2 / 1e6,
                embed_table.size() / HIDDEN);

    // Visual features (multimodal only). Precomputed once — the single --image
    // arg applies to the whole session.
    bool have_visual = false;
    VisualFeatures vf;
    if (multimodal) {
        vf = run_visual(cmdpool, visual_path, image_path);
        have_visual = true;
        std::printf("[visual] image_features=%zu elems, deepstack=[%zu,%zu,%zu]\n",
                    vf.image_features.size(), vf.deepstack[0].size(),
                    vf.deepstack[1].size(), vf.deepstack[2].size());
    }

    // Runtime + model.
    auto rt = std::make_shared<Runtime>(cmdpool, model_path, /*precision=*/1);
    rt->set_backend_buffer(true);
    std::printf("=== LoadModel ===\n");
    rt->LoadModel();
    std::printf("=== LoadModel done ===\n");

    // Pre-allocate the KV-cache buffers (past_key_values_i inputs +
    // present_key_values_i outputs) to a max size once, so they are NOT
    // reallocated every round as kv_len grows by 1. Each buffer holds
    // (1, 2, NKV, MAX_KV, HD) fp16 = 2*8*MAX_KV*128 elements. With
    // prealloc_keep_, make_vkbuff reuses the buffer (>= check) and
    // recreate_storage_buffer skips the drop. The logical shape is still
    // set per-round via ResizeInput (past) / Concat's resize (present);
    // only the physical VkBuffer is oversized. MAX_KV covers prefill L +
    // max_new decode tokens with headroom.
    {
        auto dev = cmdpool->getVulkanDevice();
        // A generous upper bound; prompts are short and max_new caps decode.
        const int MAX_KV = 8192;
        std::size_t kv_elems =
            static_cast<std::size_t>(2) * NKV * MAX_KV * HD;
        for (int i = 0; i < NLAYERS; ++i) {
            auto pin = rt->GetInput("past_key_values_" + std::to_string(i));
            auto pout = rt->GetOutput("present_key_values_" +
                                      std::to_string(i));
            as_tensor<uint16_t>(pin)->preallocate_buffer(dev, kv_elems);
            as_tensor<uint16_t>(pout)->preallocate_buffer(dev, kv_elems);
        }
    }

    // Per-round reusable zero buffers (deepstack + decode attention_bias +
    // image_pad_mask). deepstack_embeds_{0,1,2}: (1, HIDDEN) fp16 zeros.
    std::vector<uint16_t> ds_zero(HIDDEN, 0);
    std::vector<uint32_t> ds_shape = {1u, static_cast<uint32_t>(HIDDEN)};

    // REPL loop.
    std::printf("\n=== ready (max_new=%d, IM_END=%u). type a prompt, Ctrl-D to quit ===\n\n",
                max_new, IM_END);
    std::string line;
    while (std::getline(std::cin, line)) {
        if (line.empty()) continue;
        std::string prompt;
        if (std::getenv("VKOP_RAW_PROMPT")) {
            // Bypass chat template: encode the literal input (for matching the
            // reference dump_llm_decode.py, which uses proc(text=[text])).
            prompt = line;
        } else {
            // Render chat: user turn + generation prompt for assistant.
            // Multimodal: insert an image content item before the text.
            std::vector<qwen::ChatMessage> msgs;
            if (have_visual) {
                msgs = {{/*role=*/"user", /*contents=*/{
                    {/*type=*/"image", ""}, {/*type=*/"text", line}}}};
            } else {
                msgs = {{/*role=*/"user", /*contents=*/{{/*type=*/"text", line}}}};
            }
            prompt = tok.apply_chat_template(msgs, /*add_generation_prompt=*/true);
            if (prompt.empty()) {
                // No chat template baked in → fall back to raw text.
                prompt = line;
            }
        }
        std::vector<uint32_t> raw_ids = tok.encode(prompt);
        // Expand the single <|image_pad|> token into n_img copies (HF processor
        // does this based on grid_thw). No-op when no visual.
        ExpandedIds ex = have_visual
            ? expand_image_token(raw_ids, vf.n_img)
            : ExpandedIds{raw_ids, -1, 0};
        std::vector<uint32_t> ids = ex.ids;
        const int img_start = ex.img_start;
        const int img_count = ex.img_count;
        std::printf("[prompt] %zu tokens (image tokens %d..%d)\n", ids.size(),
                    img_start, img_start + img_count - 1);
        std::fflush(stdout);

        // ---- Prefill (q_len = L, kv_len = 0) ----
        int L = static_cast<int>(ids.size());
        std::printf("[prefill] L=%d building inputs...\n", L); std::fflush(stdout);

        // mm_token_type_ids (1, L): 0=text, 1=image. attention_mask all 1.
        std::vector<int32_t> mtt(L, 0);
        std::vector<int8_t> amask(L, 1);
        for (int i = 0; i < img_count; ++i) mtt[img_start + i] = 1;

        // inputs_embeds (1, L, HIDDEN): embed all ids, then scatter the visual
        // image_features rows into the image-pad positions.
        auto emb = embed_lookup(embed_table, ids, vocab);
        if (have_visual) {
            for (int i = 0; i < img_count; ++i) {
                std::memcpy(&emb[(img_start + i) * HIDDEN],
                            &vf.image_features[static_cast<size_t>(i) * HIDDEN],
                            HIDDEN * sizeof(uint16_t));
            }
        }
        fill_fp16_input(rt, "inputs_embeds", {1u, static_cast<uint32_t>(L),
                      static_cast<uint32_t>(HIDDEN)}, emb.data());
        std::printf("  inputs_embeds ok\n"); std::fflush(stdout);

        // position_ids (3, 1, L) via get_rope_index (text-only collapses to
        // arange; multimodal uses the M-RoPE delta). rope_delta drives decode.
        int64_t rope_delta = 0;
        {
            int grid_thw_flat[3] = {vf.grid_t, vf.grid_h, vf.grid_w};
            auto ri = get_rope_index(reinterpret_cast<const int64_t*>(ids.data()),
                                     mtt.data(), amask.data(),
                                     have_visual ? grid_thw_flat : nullptr,
                                     have_visual ? 1 : 0, /*B=*/1, L);
            fill_i64_input(rt, "position_ids", {3u, 1u, static_cast<uint32_t>(L)},
                           ri.pos_ids.data());
            rope_delta = ri.rope_delta[0];
        }
        std::printf("  position_ids ok (rope_delta=%lld)\n",
                    (long long)rope_delta); std::fflush(stdout);
        // attention_bias (1, 1, L, L) causal.
        {
            auto ab = causal_bias(L, L);
            fill_fp16_input(rt, "attention_bias", {1u, 1u, static_cast<uint32_t>(L),
                          static_cast<uint32_t>(L)}, ab.data());
        }
        std::printf("  attention_bias ok\n"); std::fflush(stdout);
        // deepstack_embeds_{0,1,2} (n_img, HIDDEN): visual features when
        // multimodal, else (1, HIDDEN) zeros.
        if (have_visual) {
            for (int d = 0; d < 3; ++d)
                fill_fp16_input(rt, "deepstack_embeds_" + std::to_string(d),
                                {static_cast<uint32_t>(img_count),
                                 static_cast<uint32_t>(HIDDEN)},
                                vf.deepstack[d].data());
        } else {
            for (int d = 0; d < 3; ++d)
                fill_fp16_input(rt, "deepstack_embeds_" + std::to_string(d),
                                ds_shape, ds_zero.data());
        }
        std::printf("  deepstack ok\n"); std::fflush(stdout);
        // image_pad_mask (1, L): true at image positions, false elsewhere.
        {
            std::vector<int8_t> mask(L, 0);
            for (int i = 0; i < img_count; ++i) mask[img_start + i] = 1;
            rt->ResizeInput("image_pad_mask", {1u, static_cast<uint32_t>(L)});
            auto t = rt->GetInput("image_pad_mask");
            auto tg = as_tensor<int8_t>(t);
            if (tg->num_elements() > 0) tg->fillToCPU(mask.data());
        }
        std::printf("  image_pad_mask ok\n"); std::fflush(stdout);
        // past_key_values_{i} (1,2,NKV,0,128) empty.
        for (int i = 0; i < NLAYERS; ++i) {
            std::string n = "past_key_values_" + std::to_string(i);
            rt->ResizeInput(n, {1u, 2u, static_cast<uint32_t>(NKV), 0u,
                              static_cast<uint32_t>(HD)});
        }
        std::printf("  past_kv resize ok\n"); std::fflush(stdout);
        // Upload all inputs.
        for (int i = 0; i < NLAYERS; ++i)
            upload_input(cmdpool, rt->GetInput("past_key_values_" + std::to_string(i)));
        upload_input(cmdpool, rt->GetInput("inputs_embeds"));
        upload_input(cmdpool, rt->GetInput("position_ids"));
        upload_input(cmdpool, rt->GetInput("attention_bias"));
        for (int d = 0; d < 3; ++d)
            upload_input(cmdpool, rt->GetInput("deepstack_embeds_" + std::to_string(d)));
        upload_input(cmdpool, rt->GetInput("image_pad_mask"));
        std::printf("  upload ok, calling Run()...\n"); std::fflush(stdout);

        double ms = rt->Run();
        std::printf("  Run done %.1fms\n", ms); std::fflush(stdout);
        // Wait for GPU compute to finish, but do NOT ReadResult() — that would
        // copyToCPU all 28 present_kv outputs (28 sync points) which we don't
        // need: feedback_kv does device→device, and argmax reads only logits.
        cmdpool->getVulkanDevice()->wait_all_done();

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

        int next_id = argmax_last_token(rt, cmdpool);
        std::printf("[prefill] %.1fms  → token %d  ", ms, next_id);
        std::vector<uint32_t> out_ids = {static_cast<uint32_t>(next_id)};
        std::printf("%s\n", tok.decode({static_cast<uint32_t>(next_id)}).c_str());
        std::fflush(stdout);

        // Feed KV cache back for decode rounds.
        feedback_kv(rt, cmdpool);
        int past_len = L;  // KV now holds L tokens.

        // ---- Decode loop (q_len = 1) ----
        for (int step = 1; step < max_new; ++step) {
            if (static_cast<uint32_t>(next_id) == IM_END) {
                std::printf("[done] IM_END\n");
                break;
            }
            // cur_emb (1, 1, HIDDEN) from the single next_id.
            auto cur_emb = embed_lookup(embed_table, {static_cast<uint32_t>(next_id)}, vocab);
            fill_fp16_input(rt, "inputs_embeds", {1u, 1u, static_cast<uint32_t>(HIDDEN)},
                          cur_emb.data());
            // position_ids (3, 1, 1) = past_len + rope_delta (MRoPE delta;
            // 0 for text-only, nonzero for multimodal image prompts).
            {
                int64_t p = static_cast<int64_t>(past_len) + rope_delta;
                int64_t pos[3] = {p, p, p};
                fill_i64_input(rt, "position_ids", {3u, 1u, 1u}, pos);
            }
            // attention_bias (1, 1, 1, past_len+1) all zero (full history).
            {
                std::vector<uint16_t> ab(static_cast<size_t>(past_len + 1), 0);
                fill_fp16_input(rt, "attention_bias", {1u, 1u, 1u,
                              static_cast<uint32_t>(past_len + 1)}, ab.data());
            }
            // deepstack zeros (unchanged) + image_pad_mask (1,1) false.
            for (int d = 0; d < 3; ++d)
                fill_fp16_input(rt, "deepstack_embeds_" + std::to_string(d), ds_shape, ds_zero.data());
            {
                std::vector<int8_t> mask(1, 0);
                rt->ResizeInput("image_pad_mask", {1u, 1u});
                auto t = rt->GetInput("image_pad_mask");
                as_tensor<int8_t>(t)->fillToCPU(mask.data());
            }
            // Upload.
            upload_input(cmdpool, rt->GetInput("inputs_embeds"));
            upload_input(cmdpool, rt->GetInput("position_ids"));
            upload_input(cmdpool, rt->GetInput("attention_bias"));
            for (int d = 0; d < 3; ++d)
                upload_input(cmdpool, rt->GetInput("deepstack_embeds_" + std::to_string(d)));
            upload_input(cmdpool, rt->GetInput("image_pad_mask"));

            auto t0 = std::chrono::steady_clock::now();
            ms = rt->Run();
            // Skip ReadResult (28 present_kv copyToCPU). argmax reads logits
            // (self-syncing transferReadBarrier+wait); feedback_kv is
            // device→device. One device wait ensures Run's shaders finished.
            cmdpool->getVulkanDevice()->wait_all_done();
            auto t1 = std::chrono::steady_clock::now();
            next_id = argmax_last_token(rt, cmdpool);
            out_ids.push_back(static_cast<uint32_t>(next_id));
            double run_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            std::printf("[r%d] %.1fms  past_len=%d pos=%lld  → %d  %s\n", step,
                        run_ms, past_len, (long long)(past_len + rope_delta),
                        next_id,
                        tok.decode({static_cast<uint32_t>(next_id)}).c_str());
            std::fflush(stdout);

            feedback_kv(rt, cmdpool);
            past_len += 1;
        }

        std::printf("\n=== full decode ===\n%s\n\n", tok.decode(out_ids).c_str());
    }
    return 0;
}
