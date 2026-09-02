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

#include "vulkan/VulkanDevice.hpp"
#include "vulkan/VulkanInstance.hpp"
#include "include/logger.hpp"
#include "core/Tensor.hpp"
#include "core/runtime.hpp"
#include "tokenizer.hpp"

using vkop::VulkanInstance;
using vkop::VulkanDevice;
using vkop::core::ITensor;
using vkop::core::Runtime;
using vkop::core::as_tensor;
using qwen::Tokenizer;

namespace {

constexpr int HIDDEN = 2048;
constexpr int NKV = 8;
constexpr int HD = 128;
constexpr int NLAYERS = 28;
constexpr uint32_t IM_END = 151645;
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
// round. present shape is (1,2,NKV,kv_len,128); past for next round takes the
// same shape (kv_len already includes the just-appended token).
void feedback_kv(const std::shared_ptr<Runtime>& rt,
                 const std::shared_ptr<vkop::VulkanCommandPool>& cmdpool) {
    for (int i = 0; i < NLAYERS; ++i) {
        std::string pres_name = "present_key_values_" + std::to_string(i);
        std::string past_name = "past_key_values_" + std::to_string(i);
        auto pres = rt->GetOutput(pres_name);
        if (!pres) throw std::runtime_error("no output " + pres_name);
        auto pres_t = as_tensor<uint16_t>(pres);
        // present was copyToCPU'd by ReadResult; data() holds the bytes.
        auto shape = pres_t->getShape();
        if (std::getenv("VKOP_CHATDBG")) {
            std::printf("[kvdbg] layer%d pres shape=[", i);
            for (size_t k = 0; k < shape.size(); ++k) std::printf("%d ", shape[k]);
            std::printf("] ne=%d first=", pres_t->num_elements());
            if (pres_t->num_elements() > 0)
                std::printf("%04x", (int)pres_t->data()[0]);
            std::printf("\n");
        }
        // The recorded output shape may carry -1 (dynamic kv_len); derive the
        // concrete shape from num_elements: (1,2,NKV,kv_len,128) where
        // kv_len = ne / (2*NKV*128).
        std::vector<uint32_t> u32shape;
        u32shape.push_back(1);
        u32shape.push_back(2);
        u32shape.push_back(static_cast<uint32_t>(NKV));
        int kv_len = pres_t->num_elements() / (2 * NKV * HD);
        u32shape.push_back(static_cast<uint32_t>(kv_len));
        u32shape.push_back(static_cast<uint32_t>(HD));
        if (std::getenv("VKOP_CHATDBG"))
            std::printf("[kvdbg] layer%d derived kv_len=%d\n", i, kv_len);
        // Resize + fill the past input with present's host data, then upload.
        rt->ResizeInput(past_name, u32shape);
        auto past = rt->GetInput(past_name);
        auto past_t = as_tensor<uint16_t>(past);
        if (past_t->num_elements() > 0) {
            past_t->fillToCPU(pres_t->data().data());
        }
        upload_input(cmdpool, past);
        if (std::getenv("VKOP_CHATDBG") && i == 0) {
            // Read back the uploaded past to confirm the SSBO holds present's data.
            past_t->copyToCPU(cmdpool);
            std::printf("[kvdbg] layer0 past after upload: ne=%d first=%04x "
                        "(expect %04x)\n", past_t->num_elements(),
                        past_t->num_elements() ? (int)past_t->data()[0] : -1,
                        pres_t->num_elements() ? (int)pres_t->data()[0] : -1);
        }
    }
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

} // namespace

int main(int argc, char** argv) {
    if (argc < 4) {
        std::fprintf(stderr,
            "usage: %s <model.vkopbin> <embed_tokens.bin> <tokenizer.bin> [max_new]\n",
            argv[0]);
        return 1;
    }
    const std::string model_path = argv[1];
    const std::string embed_path = argv[2];
    const std::string tok_path = argv[3];
    int max_new = (argc > 4) ? std::atoi(argv[4]) : 64;

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

    // Runtime + model.
    auto rt = std::make_shared<Runtime>(cmdpool, model_path, /*precision=*/1);
    rt->set_backend_buffer(true);
    std::printf("=== LoadModel ===\n");
    rt->LoadModel();
    std::printf("=== LoadModel done ===\n");

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
            std::vector<qwen::ChatMessage> msgs = {
                {/*role=*/"user", /*contents=*/{{/*type=*/"text", line}}}
            };
            prompt = tok.apply_chat_template(msgs, /*add_generation_prompt=*/true);
            if (prompt.empty()) {
                // No chat template baked in → fall back to raw text.
                prompt = line;
            }
        }
        std::vector<uint32_t> ids = tok.encode(prompt);
        std::printf("[prompt] %zu tokens\n", ids.size());
        std::fflush(stdout);

        // ---- Prefill (q_len = L, kv_len = 0) ----
        int L = static_cast<int>(ids.size());
        std::printf("[prefill] L=%d building inputs...\n", L); std::fflush(stdout);
        // inputs_embeds (1, L, HIDDEN)
        auto emb = embed_lookup(embed_table, ids, vocab);
        fill_fp16_input(rt, "inputs_embeds", {1u, static_cast<uint32_t>(L),
                      static_cast<uint32_t>(HIDDEN)}, emb.data());
        std::printf("  inputs_embeds ok\n"); std::fflush(stdout);
        // position_ids (3, 1, L) = [0..L-1] on each of the 3 MRoPE axes.
        {
            std::vector<int64_t> pos(3 * L);
            for (int i = 0; i < L; ++i)
                for (int a = 0; a < 3; ++a) pos[a * L + i] = i;
            fill_i64_input(rt, "position_ids", {3u, 1u, static_cast<uint32_t>(L)}, pos.data());
        }
        std::printf("  position_ids ok\n"); std::fflush(stdout);
        // attention_bias (1, 1, L, L) causal.
        {
            auto ab = causal_bias(L, L);
            fill_fp16_input(rt, "attention_bias", {1u, 1u, static_cast<uint32_t>(L),
                          static_cast<uint32_t>(L)}, ab.data());
        }
        std::printf("  attention_bias ok\n"); std::fflush(stdout);
        // deepstack_embeds_{0,1,2} (1, HIDDEN) zeros.
        for (int d = 0; d < 3; ++d)
            fill_fp16_input(rt, "deepstack_embeds_" + std::to_string(d), ds_shape, ds_zero.data());
        std::printf("  deepstack ok\n"); std::fflush(stdout);
        // image_pad_mask (1, L) all false (bool → int8_t 0).
        {
            std::vector<int8_t> mask(L, 0);
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
        std::printf("  Run done %.1fms, ReadResult...\n", ms); std::fflush(stdout);
        rt->ReadResult();

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
            // position_ids (3, 1, 1) = past_len (delta=0 text-only).
            {
                int64_t pos[3] = {past_len, past_len, past_len};
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

            ms = rt->Run();
            rt->ReadResult();
            next_id = argmax_last_token(rt, cmdpool);
            out_ids.push_back(static_cast<uint32_t>(next_id));
            std::printf("[r%d] %.1fms  past_len=%d pos=%d  → %d  %s\n", step, ms,
                        past_len, past_len, next_id,
                        tok.decode({static_cast<uint32_t>(next_id)}).c_str());
            std::fflush(stdout);

            feedback_kv(rt, cmdpool);
            past_len += 1;
        }

        std::printf("\n=== full decode ===\n%s\n\n", tok.decode(out_ids).c_str());
    }
    return 0;
}
