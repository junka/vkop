// junka @ 2026
// Image generation driver for Qwen-Image-2.1 (text tower + DiT + VAE decoder).
// Text prompt -> tokenized -> text tower -> DiT prefill -> denoising loop ->
// VAE decode -> PNG. Four graphs are loaded one at a time and released between
// stages (each DiT/text-tower graph is ~14 GB; two never fit in 36 GB).
//
// Usage:
//   image_gen <dit_prefill.vkopbin> <dit_decode.vkopbin> <prompt>
//             [steps] [seed] [--size 512|1024] [--ref DIR]
//
// Without --ref everything is computed here; the text tower's three artifacts
// and the VAE graph come from the env vars listed in the usage error below.
// Build: cmake -DENABLE_IMAGE_GEN=ON .. && make image_gen

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <algorithm>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <map>
#include <chrono>
#include <random>
#include <fstream>
#include <sys/stat.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#include "vulkan/VulkanDevice.hpp"
#include "vulkan/VulkanInstance.hpp"
#include "vulkan/VulkanCommandPool.hpp"
#include "core/Tensor.hpp"
#include "core/runtime.hpp"
#include "model/load.hpp"
#include "tokenizer.hpp"

#include "qwen_image.hpp"

using namespace qimg;

using vkop::VulkanInstance;
using vkop::VulkanDevice;
using vkop::VulkanCommandBuffer;
using vkop::VulkanCommandPool;
using vkop::core::ITensor;
using vkop::core::Runtime;
using vkop::core::as_tensor;

namespace {

// ---- Save raw fp32 tensor to file ----
void save_raw(const char* path, const float* data, int numel) {
    FILE* f = fopen(path, "wb");
    if (!f) {
        fprintf(stderr, "Cannot open %s for writing\n", path);
        return;
    }
    fwrite(data, sizeof(float), numel, f);
    fclose(f);
    printf("[save] Saved %d floats (%.1f MB) to %s\n", 
           numel, numel * 4.0f / 1e6f, path);
}

// ---- ORT reference alignment (--ref DIR) ----
// Load a binary file into a byte vector.
bool read_file_bytes(const std::string& path, std::vector<uint8_t>& out) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    f.seekg(0, std::ios::end);
    out.resize(f.tellg());
    f.seekg(0);
    f.read(reinterpret_cast<char*>(out.data()), out.size());
    return true;
}

struct RefEntry {
    std::vector<uint32_t> dims;
    int elem_size = 0;
    std::vector<uint8_t> bytes;
};

// Parse DIR/shapes.txt ("<name> d0xd1x... <elem_size>") and slurp every file.
bool load_ref_dir(const std::string& dir,
                  std::map<std::string, RefEntry>& ref) {
    std::ifstream shapes(dir + "/shapes.txt");
    if (!shapes) {
        fprintf(stderr, "[ref] missing %s/shapes.txt\n", dir.c_str());
        return false;
    }
    std::string name, dimstr;
    int esz;
    while (shapes >> name >> dimstr >> esz) {
        RefEntry e;
        e.elem_size = esz;
        size_t pos = 0;
        while (pos < dimstr.size()) {
            size_t nxt = dimstr.find('x', pos);
            if (nxt == std::string::npos) nxt = dimstr.size();
            e.dims.push_back(std::stoul(dimstr.substr(pos, nxt - pos)));
            pos = nxt + 1;
        }
        if (!read_file_bytes(dir + "/" + name, e.bytes)) {
            fprintf(stderr, "[ref] missing file %s/%s\n", dir.c_str(), name.c_str());
            return false;
        }
        ref[name] = std::move(e);
    }
    printf("[ref] loaded %zu tensors from %s\n", ref.size(), dir.c_str());
    return true;
}

} // anonymous namespace

int main(int argc, char** argv) {
    g_run_start = std::chrono::high_resolution_clock::now();
    if (argc < 4) {
        fprintf(stderr, 
                "Usage: %s <dit_prefill.vkopbin> <dit_decode.vkopbin> <prompt> "
                "[steps] [seed] [--size 512|1024] [--ref DIR]\n"
                "  no --ref: the prompt is tokenized and encoded on the GPU by the\n"
                "  text tower; its files come from the env (defaults shown):\n"
                "    TEXT_ENCODER_VKOPBIN=text_encoder.vkopbin\n"
                "    TEXT_ENCODER_EMBEDS=text_encoder_embeds.bin\n"
                "    TOKENIZER_BIN=llm/tokenizer/qwen3_vl.bin\n", argv[0]);
        return 1;
    }
    
    const char* prefill_path = argv[1];
    const char* decode_path = argv[2];
    const char* prompt = argv[3];
    
    int steps = 20;
    uint32_t seed = 42;
    int img_size = 512;
    std::string ref_dir; // ORT reference dir for numeric alignment (--ref DIR)

    // Parse optional args
    for (int i = 4; i < argc; ++i) {
        if (strcmp(argv[i], "--size") == 0 && i + 1 < argc) {
            img_size = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--ref") == 0 && i + 1 < argc) {
            ref_dir = argv[++i];
        } else if (isdigit(argv[i][0])) {
            if (steps == 20) {
                steps = atoi(argv[i]);
            } else {
                seed = atoi(argv[i]);
            }
        }
    }
    bool ref_mode = !ref_dir.empty();
    std::map<std::string, RefEntry> ref;
    if (ref_mode && !load_ref_dir(ref_dir, ref)) return 1;

    // 文本塔的三份产物。图和查表和 DiT 的 vkopbin 一样放在 image/exporter/，
    // tokenizer 在 llm/tokenizer/ 下 —— 而 BASELINE.md 的命令既可能在
    // image/exporter 里跑也可能在仓库根跑，所以按候选顺序取第一个存在的
    // （显式给 env 时以 env 为准）。
    auto env_or = [](const char* var, const char* def) {
        const char* v = getenv(var);
        return std::string(v && *v ? v : def);
    };
    const std::string te_path =
        env_or("TEXT_ENCODER_VKOPBIN", "text_encoder.vkopbin");
    const std::string te_table_path =
        env_or("TEXT_ENCODER_EMBEDS", "text_encoder_embeds.bin");
    std::string tok_path = getenv("TOKENIZER_BIN") ? getenv("TOKENIZER_BIN") : "";
    if (tok_path.empty())
        for (const char* c : {"llm/tokenizer/qwen3_vl.bin",
                              "../../llm/tokenizer/qwen3_vl.bin"}) {
            struct stat st;
            if (stat(c, &st) == 0) { tok_path = c; break; }
        }
    if (tok_path.empty()) tok_path = "llm/tokenizer/qwen3_vl.bin";
    
    printf("[init] Qwen-Image-2.1 generation driver\n");
    printf("  prompt: \"%s\"\n", prompt);
    printf("  steps: %d, seed: %u, size: %dx%d\n", steps, seed, img_size, img_size);
    
    // Latent dimensions: B×C×H×W where C=64, H=W=img_size/16
    int latent_h = img_size / 16;
    int latent_w = img_size / 16;
    int latent_c = 64; // Will be adjusted based on model's actual input shape
    
    printf("  latent shape: 1 x %d x %d x %d\n", latent_c, latent_h, latent_w);
    
    // Initialize Vulkan runtime (mirrors llm_chat pattern)
    const auto& phydevs = VulkanInstance::getVulkanInstance().getPhysicalDevices();
    if (phydevs.empty()) { fprintf(stderr, "no vulkan device\n"); return 1; }
    auto dev = std::make_shared<VulkanDevice>(phydevs[0]);
    if (dev->getDeviceName().find("llvmpipe") != std::string::npos) {
        fprintf(stderr, "no valid vulkan device\n"); return 1;
    }
    
    auto cmdpool = std::make_shared<VulkanCommandPool>(dev);
    
    // fillToCPU only stages data on the host; the input SSBO must be uploaded
    // explicitly before Run (same fill+copyToGPU pairing as llm_chat).
    auto fill16 = [&](const std::shared_ptr<ITensor>& t,
                      const std::vector<uint16_t>& d) {
        auto tt = as_tensor<uint16_t>(t);
        tt->fillToCPU(d.data());
        tt->copyToGPU(cmdpool);
    };
    auto fill32 = [&](const std::shared_ptr<ITensor>& t,
                      const std::vector<float>& d) {
        auto tt = as_tensor<float>(t);
        tt->fillToCPU(d.data());
        tt->copyToGPU(cmdpool);
    };
    
    // Fill a graph input (fp16 or fp32 by element size) from a reference file.
    auto ref_fill = [&](Runtime* rt, const std::string& input,
                        const std::string& file) -> bool {
        auto it = ref.find(file);
        if (it == ref.end()) {
            fprintf(stderr, "[ref] missing tensor %s\n", file.c_str());
            return false;
        }
        rt->ResizeInput(input, it->second.dims);
        auto t = rt->GetInput(input);
        if (!t) {
            fprintf(stderr, "[ref] no input '%s' in model\n", input.c_str());
            return false;
        }
        const uint8_t* p = it->second.bytes.data();
        size_t n = it->second.bytes.size();
        if (it->second.elem_size == 2) {
            std::vector<uint16_t> d(n / 2);
            memcpy(d.data(), p, n);
            fill16(t, d);
        } else {
            std::vector<float> d(n / 4);
            memcpy(d.data(), p, n);
            fill32(t, d);
        }
        return true;
    };
    
    // Load models (using buffer backend like llm_chat)
    // 文本塔排在 DiT 之前、用完即释放：它和 prefill 各要 13.9/13.8 GB，36 GB
    // 统一内存里两段不能同时在。
    PromptEmbeds text_pe;
    if (!ref_mode) {
        if (!encode_prompt(te_path, te_table_path, tok_path, prompt, cmdpool, dev,
                           text_pe))
            return 1;
    }

    printf("[load] Loading DiT prefill graph...\n");
    auto prefill_rt = std::make_shared<Runtime>(cmdpool, prefill_path, /*precision=*/1);
    prefill_rt->set_backend_buffer(true);
    prefill_rt->LoadModel();
    printf("[load] Prefill loaded (+%.2fs)\n", stage_s());

    // Header-only shape probe: the flatbuffer is mmap'd, so reading input dims
    // costs nothing and lets the decode graph stay unloaded until the prefill is
    // done. Each 7.12B graph is ~13 GB of device weights; both at once plus
    // activations do not fit in 36 GB.
    auto header_dims = [](const char* path, bool input,
                          const std::string& name) -> std::vector<int32_t> {
        vkop::load::VkModel m(path);
        const auto& list = input ? m.inputs : m.outputs;
        for (const auto& s : list)
            if (s.name == name) return s.dims;
        return {};
    };

    // [probe] report-only: checksum the initializers embedded in each vkopbin
    // so silently-zeroed weights are caught before any GPU run.
    if (std::getenv("VKOP_DUMP_WEIGHTS")) {
        auto f16 = [](const uint8_t* p) {
            _Float16 h;
            memcpy(&h, p, 2);
            return (float)h;
        };
        for (const char* wp : {prefill_path, decode_path}) {
            vkop::load::VkModel m(wp);
            printf("[weights] %s: %zu entries\n", wp, m.initializers.size());
            for (const auto& kv : m.initializers) {
                const auto& init = kv.second;
                size_t cnt = 1;
                for (auto d : init.dims) cnt *= d;
                size_t esz = init.dtype == "float16" || init.dtype == "bfloat16" ? 2
                           : init.dtype == "float32" || init.dtype == "int32" ? 4
                           : init.dtype == "int64" ? 8 : 0;
                auto oit = m.initializer_offsets.find(kv.first);
                if (!m.initializer_memory || esz == 0 || oit == m.initializer_offsets.end()) {
                    printf("    %-40s %s elems=%zu SKIP\n", kv.first.c_str(),
                           init.dtype.c_str(), cnt);
                    continue;
                }
                const uint8_t* base = m.initializer_memory + oit->second;
                size_t nz = 0;
                double sum = 0;
                float vmin = 1e30f, vmax = -1e30f;
                for (size_t i = 0; i < cnt; ++i) {
                    const uint8_t* p = base + i * esz;
                    float v = 0.0f;
                    if (init.dtype == "float16") {
                        v = f16(p);
                    } else if (init.dtype == "float32") {
                        memcpy(&v, p, 4);
                    } else if (init.dtype == "int64") {
                        int64_t q;
                        memcpy(&q, p, 8);
                        v = (float)(q % 1000000);
                    } else if (init.dtype == "int32") {
                        int32_t q;
                        memcpy(&q, p, 4);
                        v = (float)q;
                    }
                    if (v != 0.0f) ++nz;
                    sum += v;
                    vmin = std::min(vmin, v);
                    vmax = std::max(vmax, v);
                }
                printf("    %-40s %s elems=%zu nz=%zu sum=%.6g min=%.4g max=%.4g\n",
                       kv.first.c_str(), init.dtype.c_str(), cnt, nz, sum, vmin, vmax);
            }
        }
    }
    
    // Detect latent channel dimension from the decode model's header
    int target_len = latent_h * latent_w;
    auto target_latents_dims = header_dims(decode_path, true, "target_latents");
    if (target_latents_dims.size() >= 3 && target_latents_dims[2] > 0) {
        latent_c = target_latents_dims[2]; // Last dim is C in BLC format
        printf("[model] Detected latent_c=%d from decode model\n", latent_c);
    }
    
    int latent_n = 1 * latent_c * target_len;
    std::vector<float> latent(latent_n);
    if (ref_mode) {
        // Reference mode: the latent AND all per-step inputs come from ORT's
        // ref/ dir so vkop and ORT consume bit-identical data.
        const auto& e = ref["latent_init.raw"];
        target_len = (int)e.dims[1];
        latent_c = (int)e.dims[2];
        latent_n = target_len * latent_c;
        latent.assign(reinterpret_cast<const float*>(e.bytes.data()),
                      reinterpret_cast<const float*>(e.bytes.data()) + latent_n);
        printf("[ref] latent from latent_init.raw: target_len=%d latent_c=%d\n",
               target_len, latent_c);
    } else {
        generate_random_latent(latent.data(), latent_n, seed);
    }
    
    // σ 调度。ref 模式读 gen_dit_ref.py 写的 sigmas.raw（长度 steps+1，末尾是
    // 终值 0），其余情况由 official_sigmas 现算 —— 同一条公式，所以两边可以
    // 逐位对拍。模型的 timestep 输入就是位移后的 σ 本身（pipeline 传 t/1000，
    // 而 t = σ·1000），Euler 更新是 x += (σ_next − σ)·v。
    std::vector<float> sigmas;
    if (ref_mode && ref.count("sigmas.raw")) {
        const auto &e = ref["sigmas.raw"];
        sigmas.assign(reinterpret_cast<const float *>(e.bytes.data()),
                      reinterpret_cast<const float *>(e.bytes.data()) +
                          e.bytes.size() / 4);
        if (steps != (int)sigmas.size() - 1) {
            printf("[ref] steps=%d -> %zu (from sigmas.raw)\n", steps,
                   sigmas.size() - 1);
            steps = (int)sigmas.size() - 1;
        }
        printf("[ref] official schedule: %zu sigmas, σ0=%.4f μ-shifted\n",
               sigmas.size(), sigmas[0]);
    } else if (!ref_mode) {
        sigmas = official_sigmas(steps, target_len);
    }
    // Re-anchoring the latent on ORT's per-step state isolates each decode step
    // (no divergence carry-over) — right for numeric alignment, wrong for a
    // real sample, which must follow vkop's own trajectory. VKOP_REF_FREE=1
    // turns re-anchoring off.
    const bool ref_free_run =
        ref_mode && getenv("VKOP_REF_FREE") != nullptr;
    
    const int hd = kRopeHeadDim;  // cos/sin 的列数
    RopeTables rope;  // 非 ref 路径在 prefill 前算好，decode 每一步都要用
    
    // ---- KV Cache Integration ----
    // Prefill outputs present_kv_* tensors, decode needs past_kv_* inputs
    // Dynamically detect the number of KV layers from the model
    
    // prefix_len 是 DiT 侧固定消费的文本段行数（下面从图里读），valid_len 是其中
    // 真实 token 的行数，剩下的行是补零 —— 由文本塔现算。
    int prefix_len = 64;
    int valid_len = 64;
    if (!ref_mode) valid_len = text_pe.valid_len;
    
    printf("[prefill] Running prompt encoding to get initial KV cache...\n");
    
    // Prepare prefill inputs
    int seq_len = prefix_len;
    
    // Detect context_dim from prefill model's prompt_embeds input
    auto prompt_embeds_check = prefill_rt->GetInput("prompt_embeds");
    int context_dim = 4096; // Default for full model
    int graph_text_len = 0; // 图自己要的文本段行数
    if (prompt_embeds_check) {
        auto check_shape = prompt_embeds_check->getShape();
        if (check_shape.size() >= 3 && check_shape[2] > 0) {
            context_dim = check_shape[2];
            graph_text_len = check_shape[1];
            printf("[model] Detected context_dim=%d from prefill model\n", context_dim);
        }
    }
    
    if (ref_mode) {
        // Bit-identical inputs from the ORT reference dir.
        const auto& pe = ref["prompt_embeds.raw"];
        seq_len = prefix_len = (int)pe.dims[1];
        context_dim = (int)pe.dims[2];
        if (!ref_fill(prefill_rt.get(), "prompt_embeds", "prompt_embeds.raw")) return 1;
        if (!ref_fill(prefill_rt.get(), "cos", "cos_prefill.raw")) return 1;
        if (!ref_fill(prefill_rt.get(), "sin", "sin_prefill.raw")) return 1;
        prefill_rt->ResizeInput("timestep_zero", {1u});
        std::vector<float> timestep_zero_data(1, 0.0f);
        fill32(prefill_rt->GetInput("timestep_zero"), timestep_zero_data);
        if (!ref_fill(prefill_rt.get(), "attention_bias", "bias_prefill.raw")) return 1;
        printf("[ref] prefill inputs loaded (seq_len=%d ctx_dim=%d)\n",
               seq_len, context_dim);
    } else {
        // prompt_embeds: [1, prefix_len, context_dim] fp16，来自文本塔（查表 ->
        // 36 层 -> 去 system 段 -> 右补零）。文本塔的图长 P = drop_idx + prefix_len
        // 和 DiT 的 prefix_len 是同一批导出时定死的，行数或宽度对不上就是拿错了
        // 配对 —— 必须在这里停，而不是把 64 行的向量塞进要 56 行的输入。
        prefix_len = seq_len = graph_text_len;
        if (text_pe.ctx_dim != context_dim || text_pe.prefix_len != prefix_len) {
            fprintf(stderr,
                    "[error] 文本塔给出 (%d,%d)，prefill 图要 (%d,%d)；"
                    "text_encoder 和 dit_*_static 不是同一批导出的\n",
                    text_pe.prefix_len, text_pe.ctx_dim, prefix_len, context_dim);
            return 1;
        }
        prefill_rt->ResizeInput("prompt_embeds", {1u, (uint32_t)seq_len, (uint32_t)context_dim});
        auto prompt_embeds_t = prefill_rt->GetInput("prompt_embeds");
        if (!prompt_embeds_t) {
            fprintf(stderr, "[error] Failed to get prompt_embeds input tensor\n");
            return 1;
        }
        fill16(prompt_embeds_t, text_pe.data);
        
        // joint rope 的两段表：文本行 + 图像行（图像段的 frame 位置按 valid_len
        // 冻结，见 build_rope）。fp32 算完再整体落 fp16，与 gen_dit_ref.py 的
        // `torch.cos(...).astype(np.float16)` 同一舍入点。
        rope = build_rope(prefix_len, valid_len, latent_h, latent_w);
        printf("[rope] txt %d 行 + img %d 行, axes={%d,%d,%d}, theta=%g\n",
               prefix_len, target_len, kAxesRope[0], kAxesRope[1], kAxesRope[2],
               (double)kRopeTheta);
        // 把驱动自己算的表和 gen_dit_ref.py --real-rope --schedule diffusers 的
        // 产物逐位对拍用（"纯 vkop 出图"这条路上没有任何东西保证公式搬对了）。
        if (const char* dd = getenv("VKOP_DUMP_TABLES")) {
            mkdir(dd, 0755);
            auto w = [&](const char* name, const std::vector<uint16_t>& v) {
                FILE* f = fopen((std::string(dd) + "/" + name).c_str(), "wb");
                if (f) { fwrite(v.data(), 2, v.size(), f); fclose(f); }
            };
            w("cos_prefill.raw", rope.cos_text);
            w("sin_prefill.raw", rope.sin_text);
            w("cos_decode.raw", rope.cos_img);
            w("sin_decode.raw", rope.sin_img);
            FILE* f = fopen((std::string(dd) + "/sigmas.raw").c_str(), "wb");
            if (f) { fwrite(sigmas.data(), 4, sigmas.size(), f); fclose(f); }
            printf("[tables] -> %s\n", dd);
        }
        prefill_rt->ResizeInput("cos", {(uint32_t)seq_len, (uint32_t)hd});
        fill16(prefill_rt->GetInput("cos"), rope.cos_text);
        prefill_rt->ResizeInput("sin", {(uint32_t)seq_len, (uint32_t)hd});
        fill16(prefill_rt->GetInput("sin"), rope.sin_text);
        
        // timestep_zero: [1]
        prefill_rt->ResizeInput("timestep_zero", {1u});
        auto timestep_zero_t = prefill_rt->GetInput("timestep_zero");
        std::vector<float> timestep_zero_data(1, 0.0f);
        fill32(timestep_zero_t, timestep_zero_data);
        
        // attention_bias: [1, 1, prefix_len, prefix_len] with causal mask (fp16)
        prefill_rt->ResizeInput("attention_bias", {1u, 1u, (uint32_t)seq_len, (uint32_t)seq_len});
        auto attn_bias_t = prefill_rt->GetInput("attention_bias");
        std::vector<float> attn_bias_prefill(seq_len * seq_len, 0.0f);
        for (int i = 0; i < seq_len; ++i)
            for (int j = i + 1; j < seq_len; ++j)
                attn_bias_prefill[i * seq_len + j] = -65504.0f;
        std::vector<uint16_t> attn_bias_fp16(attn_bias_prefill.size());
        for (size_t i = 0; i < attn_bias_prefill.size(); ++i)
            attn_bias_fp16[i] = ITensor::fp32_to_fp16(attn_bias_prefill[i]);
        fill16(attn_bias_t, attn_bias_fp16);
    }
    
    // [probe] VKOP_REF_SKIP_PREFILL=1 (ref mode only): don't execute the vkop
    // prefill at all — decode KV already comes from ORT files. If the
    // VK_ERROR_DEVICE_LOST crash disappears, the prefill's mis-shaped
    // present_kv_1 output (256-elem SSBO written as 4096) is the fault source.
    const bool skip_prefill_run =
        ref_mode && getenv("VKOP_REF_SKIP_PREFILL") != nullptr;

    // Run prefill
    auto t_prefill_start = std::chrono::high_resolution_clock::now();
    if (!skip_prefill_run) {
        prefill_rt->Run();
        prefill_rt->ReadResult(); // drain GPU queues + read back real outputs
    }
    auto t_prefill_end = std::chrono::high_resolution_clock::now();
    double t_prefill = std::chrono::duration<double>(t_prefill_end - t_prefill_start).count();
    
    // Count how many present_kv_* outputs exist
    int num_kv_layers = 0;
    while (true) {
        std::string kv_name = "present_kv_" + std::to_string(num_kv_layers);
        auto kv_out = prefill_rt->GetOutput(kv_name);
        if (!kv_out) break;
        num_kv_layers++;
    }
    
    printf("[prefill] Done in %.3fs (+%.2fs), found %d KV layers\n", t_prefill,
           stage_s(), num_kv_layers);
    
    if (num_kv_layers == 0) {
        fprintf(stderr, "[error] No present_kv outputs found in prefill model\n");
        return 1;
    }
    
    // Extract present_kv outputs and store as past_kv for decode
    std::vector<std::vector<uint16_t>> past_kv_cache(num_kv_layers);
    std::vector<std::vector<int>> past_kv_shapes(num_kv_layers); // Store shapes for decode resize
    
    for (int layer = 0; !skip_prefill_run && layer < num_kv_layers; ++layer) {
        std::string kv_name = "present_kv_" + std::to_string(layer);
        auto kv_out = prefill_rt->GetOutput(kv_name);
        if (!kv_out) {
            fprintf(stderr, "[error] Missing output: %s\n", kv_name.c_str());
            return 1;
        }
        
        auto kv_tensor = as_tensor<uint16_t>(kv_out);
        kv_tensor->copyToCPU(cmdpool);
        const auto& kv_data = kv_tensor->data();
        past_kv_cache[layer].assign(kv_data.begin(), kv_data.end());
        
        // Store the shape for later use in decode
        past_kv_shapes[layer] = kv_tensor->getShape();
        
        if (ref_mode) {
            // Dump vkop's own prefill KV so it can be compared against ORT's.
            mkdir("ref_out", 0755);
            auto shp = kv_tensor->getShape();
            printf("[dbg] present_kv_%d data_n=%zu shape=", layer, kv_data.size());
            for (size_t d = 0; d < shp.size(); ++d) printf("%d,", shp[d]);
            printf("\n");
            std::string out_path = "ref_out/vkop_present_kv_" + std::to_string(layer) + ".raw";
            FILE* fp = fopen(out_path.c_str(), "wb");
            if (fp) {
                fwrite(kv_data.data(), sizeof(uint16_t), kv_data.size(), fp);
                fclose(fp);
            }
        }
    }
    
    if (ref_mode && !getenv("VKOP_REF_VK_KV")) {
        // Use ORT's KV as decode inputs so the decode graph is isolated from
        // any prefill divergence (both compared separately).
        for (int layer = 0; layer < num_kv_layers; ++layer) {
            const auto& e = ref["ort_present_kv_" + std::to_string(layer) + ".raw"];
            const uint16_t* p = reinterpret_cast<const uint16_t*>(e.bytes.data());
            past_kv_cache[layer].assign(p, p + e.bytes.size() / 2);
            past_kv_shapes[layer] =
                std::vector<int>(e.dims.begin(), e.dims.end());
        }
        printf("[ref] decode past_kv taken from ORT prefill outputs\n");
    } else if (ref_mode) {
        printf("[ref] decode past_kv taken from vkop's own prefill\n");
    }
    
    printf("[kv] Cached %d KV layers (prefix_len=%d)\n", num_kv_layers, prefix_len);

    // Sequential loading (same strategy as the ORT baseline): the prefill graph
    // is released before the decode graph takes its weights, so a 7.12B model
    // needs one graph's worth of device memory at a time.
    auto t_pf_rel = std::chrono::high_resolution_clock::now();
    prefill_rt.reset();

    // Force GPU to finish and release MoltenVK/Metal resources before loading
    // the next large model. On UMA (M5 Max), wait_all_done ensures Metal can
    // reclaim GPU heap back to system RAM.
    cmdpool->getVulkanDevice()->wait_all_done();
    cmdpool->reset();
    printf("[mem] Prefill released in %.2fs (+%.2fs), device idle\n",
           std::chrono::duration<double>(
               std::chrono::high_resolution_clock::now() - t_pf_rel)
               .count(),
           stage_s());

    // VAE_ONLY mode: skip DiT decode entirely, load latent from file.
    const char* vae_only_env = getenv("VAE_ONLY");
    bool vae_only_mode = vae_only_env && vae_only_env[0] != '0';

    if (!vae_only_mode) {
        printf("[load] Loading DiT decode graph...\n");
        auto decode_rt = std::make_shared<Runtime>(cmdpool, decode_path, /*precision=*/1);
        decode_rt->set_backend_buffer(true);
        try {
            decode_rt->LoadModel();
            printf("[load] Decode loaded (+%.2fs)\n", stage_s());
        } catch (const std::exception& e) {
            fprintf(stderr, "[error] Failed to load decode model: %s\n", e.what());
            return 1;
        }

        printf("[gen] Starting denoising loop (%d steps)...\n", steps);
        auto t_start = std::chrono::high_resolution_clock::now();
    
    // Denoising loop with real DiT decode inference
    for (int step = 0; step < steps; ++step) {
        float t, sigma;
        if (!sigmas.empty()) {
            sigma = sigmas[step];
            t = sigma; // pipeline feeds timestep = t/1000, and t = sigma·1000
        } else {
            // 只有 ref 目录里没有 sigmas.raw（gen_dit_ref.py 的 naive 调度）才走到
            // 这里：那条参考循环用的就是 t=linspace(1,0,N,endpoint=False)、σ=1-t。
            t = 1.0f - (float)step / (float)steps;
            sigma = 1.0f - t;
        }

        if (ref_mode && step > 0 && !ref_free_run) {
            // Re-anchor the latent on ORT's post-step state so every step
            // compares decode in isolation (no divergence carry-over).
            std::string lf = "ort_latent_after_step" + std::to_string(step - 1) + ".raw";
            auto it = ref.find(lf);
            if (it != ref.end()) {
                const uint16_t* p = reinterpret_cast<const uint16_t*>(it->second.bytes.data());
                latent.resize(it->second.bytes.size() / 2);
                for (size_t i = 0; i < latent.size(); ++i) {
                    latent[i] = ITensor::fp16_to_fp32(p[i]);
                }
            }
        }
        
        // Resize and fill target_latents: [1, target_len, latent_c]
        decode_rt->ResizeInput("target_latents", {1u, (uint32_t)target_len, (uint32_t)latent_c});
        auto target_lat = decode_rt->GetInput("target_latents");
        std::vector<uint16_t> latent_fp16(latent_n);
        for (int i = 0; i < latent_n; ++i) {
            latent_fp16[i] = ITensor::fp32_to_fp16(latent[i]);
        }
        fill16(target_lat, latent_fp16);
        
        // Resize and fill timestep: [1]
        decode_rt->ResizeInput("timestep", {1u});
        auto timestep_t = decode_rt->GetInput("timestep");
        std::vector<float> timestep_data(1, t);
        fill32(timestep_t, timestep_data);
        
        // rope、past_kv 和 attention_bias 每步都一样，只在第 0 步上传一次。
        // 之前每步都 ResizeInput + copyToGPU：ResizeInput 会 recreate_storage_buffer，
        // 所以每步要重新分配 32 个 KV buffer 再同步上传，主机侧每步约 115 ms
        // （KV 90 ms、bias 21 ms），占一步的 7%。这些输入在 decode 图里只读：
        // 图里没有带 in-place cache 的 Attention，不 resize 时 buffer 就跨 Run 保留。
        // ref 模式每步按参考文件重灌，保持原样。
        const bool upload_static = ref_mode || step == 0;
        if (ref_mode) {
            if (!ref_fill(decode_rt.get(), "cos", "cos_decode.raw")) return 1;
            if (!ref_fill(decode_rt.get(), "sin", "sin_decode.raw")) return 1;
        } else if (upload_static) {
            decode_rt->ResizeInput("cos", {(uint32_t)target_len, (uint32_t)hd});
            fill16(decode_rt->GetInput("cos"), rope.cos_img);
            decode_rt->ResizeInput("sin", {(uint32_t)target_len, (uint32_t)hd});
            fill16(decode_rt->GetInput("sin"), rope.sin_img);
        }
        
        // Feed all past_kv inputs (dynamic count and shape based on prefill outputs)
        for (int layer = 0; upload_static && layer < num_kv_layers; ++layer) {
            std::string kv_name = "past_kv_" + std::to_string(layer);
            const auto& kv_shape = past_kv_shapes[layer];
            decode_rt->ResizeInput(kv_name, {
                (uint32_t)kv_shape[0], 
                (uint32_t)kv_shape[1], 
                (uint32_t)kv_shape[2], 
                (uint32_t)kv_shape[3], 
                (uint32_t)kv_shape[4]
            });
            auto kv_t = decode_rt->GetInput(kv_name);
            fill16(kv_t, past_kv_cache[layer]);
        }
        
        // attention_bias: [1, 1, target_len, kv_len] (fp16)
        if (ref_mode) {
            if (!ref_fill(decode_rt.get(), "attention_bias", "bias_decode.raw")) return 1;
        } else if (upload_static) {
            int total_kv_len = prefix_len + target_len;
            decode_rt->ResizeInput("attention_bias", {1u, 1u, (uint32_t)target_len, (uint32_t)total_kv_len});
            auto attn_bias_decode_t = decode_rt->GetInput("attention_bias");
            std::vector<float> attn_bias_decode(target_len * total_kv_len, 0.0f);
            // 文本 KV 段的前 valid_len 列是真 token，剩下 prefix_len-valid_len 列是
            // 补零。图像段对文本是**双向**注意力，零向量当 key 会实打实地改变注意力
            // 权重，所以这些列必须屏蔽（-65504 是 fp16 最小值，softmax 后为 0）。
            // 与 gen_dit_ref.py 的 bias_t[0,0,:,valid_len:prefix_len] 同一口径。
            for (int i = 0; i < target_len; ++i)
                for (int j = valid_len; j < prefix_len; ++j)
                    attn_bias_decode[i * total_kv_len + j] = -65504.0f;
            if (valid_len < prefix_len)
                printf("[bias] decode: 屏蔽 %d 个 prompt 补零列\n", prefix_len - valid_len);
            std::vector<uint16_t> attn_bias_decode_fp16(attn_bias_decode.size());
            for (size_t i = 0; i < attn_bias_decode.size(); ++i) {
                attn_bias_decode_fp16[i] = ITensor::fp32_to_fp16(attn_bias_decode[i]);
            }
            fill16(attn_bias_decode_t, attn_bias_decode_fp16);
        }
        
        if (ref_mode) {
            // [probe] Read the inputs back FROM THE GPU before Run to see
            // whether what the shaders will consume is byte-identical to the
            // ref files the host intended to upload.
            const char* probe_names[] = {"target_latents", "timestep", "cos",
                                         "sin", "attention_bias"};
            for (const char* pn : probe_names) {
                auto t = decode_rt->GetInput(pn);
                std::string f = "ref_out/echo_step" + std::to_string(step) +
                                "_" + pn + ".raw";
                FILE* fp = fopen(f.c_str(), "wb");
                if (!fp) continue;
                if (t->dtype() == typeid(float)) {
                    auto tt = as_tensor<float>(t);
                    tt->copyToCPU(cmdpool);
                    fwrite(tt->data().data(), sizeof(float),
                           tt->data().size(), fp);
                } else {
                    auto tt = as_tensor<uint16_t>(t);
                    tt->copyToCPU(cmdpool);
                    fwrite(tt->data().data(), sizeof(uint16_t),
                           tt->data().size(), fp);
                }
                fclose(fp);
            }
            for (int layer = 0; layer < num_kv_layers; ++layer) {
                auto t = decode_rt->GetInput("past_kv_" + std::to_string(layer));
                auto tt = as_tensor<uint16_t>(t);
                tt->copyToCPU(cmdpool);
                std::string f = "ref_out/echo_step" + std::to_string(step) +
                                "_past_kv_" + std::to_string(layer) + ".raw";
                FILE* fp = fopen(f.c_str(), "wb");
                if (!fp) continue;
                fwrite(tt->data().data(), sizeof(uint16_t),
                       tt->data().size(), fp);
                fclose(fp);
            }
        }

        // Run decode
        decode_rt->Run();
        decode_rt->ReadResult();
        
        // Get output: sample [1, target_len, latent_c]
        auto sample_out = decode_rt->GetOutput("sample");
        auto sample_tensor = as_tensor<uint16_t>(sample_out);
        sample_tensor->copyToCPU(cmdpool);
        const auto& sample_data = sample_tensor->data();
        
        // Convert fp16 back to fp32 and compute velocity
        std::vector<float> velocity(latent_n);
        for (int i = 0; i < latent_n; ++i) {
            velocity[i] = ITensor::fp16_to_fp32(sample_data[i]);
        }
        
        if (ref_mode) {
            // Dump for the compare script.
            mkdir("ref_out", 0755);
            auto shp = sample_tensor->getShape();
            uint64_t sum = 0;
            for (size_t i = 0; i < std::min<size_t>(sample_data.size(), 64); ++i)
                sum += sample_data[i];
            printf("[dbg] step %d data_n=%zu head64sum=%llu shape=", step,
                   sample_data.size(), (unsigned long long)sum);
            for (size_t d = 0; d < shp.size(); ++d) printf("%d,", shp[d]);
            printf("\n");
            std::string out_path = "ref_out/vkop_velocity_step" + std::to_string(step) + ".raw";
            FILE* fp = fopen(out_path.c_str(), "wb");
            if (fp) {
                fwrite(sample_data.data(), sizeof(uint16_t),
                       std::min<size_t>(sample_data.size(), latent_n), fp);
                fclose(fp);
            }
        }
        
        // Euler step：官方调度下 x += (σ_next − σ)·v；naive 那条（老 ref 目录）
        // 沿用 x -= σ·v，与 gen_dit_ref.py 的 naive 循环一致。
        if (!sigmas.empty()) {
            const float d = sigmas[step + 1] - sigmas[step];
            for (int i = 0; i < latent_n; ++i)
                latent[i] += d * velocity[i];
        } else {
            for (int i = 0; i < latent_n; ++i)
                latent[i] -= sigma * velocity[i];
        }
        
        if ((step + 1) % 5 == 0 || step == 0) {
            printf("  Step %d/%d (t=%.3f, σ=%.3f)\n", 
                   step + 1, steps, t, sigma);
        }
    }
    
    auto t_end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(t_end - t_start).count();
    printf("[gen] Denoising complete: %.1fs (%.1f s/step, +%.2fs)\n",
           elapsed, elapsed / steps, stage_s());
    {
        // Release the decode graph here so its teardown is measured; leaving it
        // to the scope end hides ~14 GB of device frees inside the VAE load.
        auto t_dc_rel = std::chrono::high_resolution_clock::now();
        decode_rt.reset();
        cmdpool->getVulkanDevice()->wait_all_done();
        cmdpool->reset();
        printf("[mem] decode released in %.2fs (+%.2fs)\n",
               std::chrono::duration<double>(
                   std::chrono::high_resolution_clock::now() - t_dc_rel)
                   .count(),
               stage_s());
    }
    
    // Save output
    char out_path[256];
    snprintf(out_path, sizeof(out_path), "latent_out_%dx%d.raw", img_size, img_size);
    save_raw(out_path, latent.data(), latent_n);
    
    printf("[done] Raw latent saved (%d elements).\n", (int)latent_n);
} else {
    // VAE_ONLY mode: load latent from file instead of running DiT decode.
    const char* latent_file = getenv("VAE_LATENT_FILE");
    std::string lf = latent_file ? latent_file : "ref_full/ort_latent_after_step0.raw";
    printf("[vae-only] Loading latent from %s\n", lf.c_str());
    FILE* fp = fopen(lf.c_str(), "rb");
    if (!fp) {
        fprintf(stderr, "[vae-only] cannot open %s\n", lf.c_str());
        return 1;
    }
    // Read fp16 latent [1, target_len, latent_c].
    size_t n_fp16 = (size_t)1 * target_len * latent_c;
    std::vector<uint16_t> latent_fp16(n_fp16);
    if (fread(latent_fp16.data(), sizeof(uint16_t), n_fp16, fp) != n_fp16) {
        fprintf(stderr, "[vae-only] short read from %s\n", lf.c_str());
        fclose(fp);
        return 1;
    }
    fclose(fp);
    // Convert to fp32.
    latent.resize(n_fp16);
    for (size_t i = 0; i < n_fp16; ++i) {
        latent[i] = vkop::core::ITensor::fp16_to_fp32(latent_fp16[i]);
    }
    printf("[vae-only] Loaded %zu fp32 values\n", latent.size());
}

// ---- VAE Decode & PNG output ----
    // Latent shape: [1, target_len, latent_c] in BLC layout.
    // VAE expects: [1, latent_c, 1, H_lat, W_lat] in NCDHW where
    //   H_lat * W_lat = target_len. For target_len=1024 -> 32x32 grid.
    int vae_latent_h = 0, vae_latent_w = 0;
    for (int s = 1; s <= target_len; ++s) {
        if (target_len % s == 0) {
            vae_latent_h = s;
            vae_latent_w = target_len / s;
            if (std::abs(vae_latent_h - vae_latent_w) <= 1)
                break;
        }
    }
    if (vae_latent_h == 0) {
        fprintf(stderr, "[vae] cannot factor target_len=%d\n", target_len);
        return 1;
    }
    printf("[vae] Latent grid: %dx%d (target_len=%d, latent_c=%d)\n",
           vae_latent_h, vae_latent_w, target_len, latent_c);

    // Reshape BLC [1, target_len, latent_c] -> NCDHW [1, latent_c, 1, H, W].
    std::vector<float> latent_ncdhw((size_t)1 * latent_c * 1 * vae_latent_h *
                                    vae_latent_w);
    for (int t = 0; t < target_len; ++t) {
        int lh = t / vae_latent_w;
        int lw = t % vae_latent_w;
        for (int c = 0; c < latent_c; ++c) {
            latent_ncdhw[((size_t)c * vae_latent_h + lh) * vae_latent_w + lw] =
                latent[(size_t)t * latent_c + c] * kLatentsStd[c] + kLatentsMean[c];
        }
    }

    // Load VAE decoder.
    const char* vae_path_env = getenv("VAE_VKOPBIN");
    std::string vae_path_str =
        vae_path_env ? vae_path_env : "vae_decoder_512.vkopbin";
    const char* vae_path = vae_path_str.c_str();

    printf("[load] Loading VAE decoder: %s\n", vae_path);
    auto vae_rt =
        std::make_shared<Runtime>(cmdpool, vae_path, /*precision=*/1);
    vae_rt->set_backend_buffer(true);
    try {
        vae_rt->LoadModel();
        printf("[load] VAE loaded (+%.2fs)\n", stage_s());
    } catch (const std::exception& e) {
        fprintf(stderr, "[vae] LoadModel failed: %s\n", e.what());
        return 1;
    }

    // Set latent input by name.
    auto vae_input = vae_rt->GetInput("latent");
    if (!vae_input) {
        fprintf(stderr, "[vae] no 'latent' input found\n");
        return 1;
    }
    // Resize and fill the VAE's input tensor.
    // An fp16-converted VAE (onnx2vkop -q fp16) takes and returns fp16.
    const std::vector<int> latent_shape{1, latent_c, 1, vae_latent_h,
                                        vae_latent_w};
    if (vae_input->dtype() == typeid(float)) {
        auto vae_latent_tensor = vkop::core::as_tensor<float>(vae_input);
        vae_latent_tensor->resize(latent_shape);
        vae_latent_tensor->fillToCPU(latent_ncdhw.data());
        vae_latent_tensor->copyToGPU(cmdpool);
    } else {
        std::vector<uint16_t> h(latent_ncdhw.size());
        for (size_t i = 0; i < h.size(); ++i)
            h[i] = vkop::core::ITensor::fp32_to_fp16(latent_ncdhw[i]);
        auto vae_latent_tensor = vkop::core::as_tensor<uint16_t>(vae_input);
        vae_latent_tensor->resize(latent_shape);
        vae_latent_tensor->fillToCPU(h);
        vae_latent_tensor->copyToGPU(cmdpool);
    }

    // Run VAE decode.
    printf("[vae] Running decode...\n");
    auto t_vae_start = std::chrono::high_resolution_clock::now();
    vae_rt->Run();
    auto t_vae_end = std::chrono::high_resolution_clock::now();
    double vae_elapsed =
        std::chrono::duration<double>(t_vae_end - t_vae_start).count();
    printf("[time] vae decode %.2fs (+%.2fs)\n", vae_elapsed, stage_s());

    // Read decoded output.
    auto vae_output = vae_rt->GetOutput("decoded");
    if (!vae_output) {
        fprintf(stderr, "[vae] no 'decoded' output\n");
        return 1;
    }
    const std::vector<int> decoded_shape = vae_output->getShape();
    std::vector<float> decoded_data;
    if (vae_output->dtype() == typeid(float)) {
        auto t = vkop::core::as_tensor<float>(vae_output);
        t->copyToCPU(cmdpool);
        decoded_data.assign(t->data().begin(), t->data().end());
    } else {
        auto t = vkop::core::as_tensor<uint16_t>(vae_output);
        t->copyToCPU(cmdpool);
        decoded_data.resize(t->num_elements());
        for (size_t i = 0; i < decoded_data.size(); ++i)
            decoded_data[i] = vkop::core::ITensor::fp16_to_fp32(t->data()[i]);
    }
    if (decoded_shape.size() < 4) {
        fprintf(stderr, "[vae] unexpected output rank %zu\n",
                decoded_shape.size());
        return 1;
    }
    // Shape is [1, 4, 1, H_img, W_img] in NCDHW.
    int img_h = decoded_shape[decoded_shape.size() - 2];
    int img_w = decoded_shape[decoded_shape.size() - 1];
    int img_c = decoded_shape[1]; // C dimension
    printf("[vae] Decoded: [%d,%d,%d,%d,%d] -> image %dx%d x%d\n",
           (int)decoded_shape[0], (int)decoded_shape[1],
           (int)decoded_shape[2], img_h, img_w, img_h, img_w, img_c);


    char png_path[256];
    snprintf(png_path, sizeof(png_path), "output_%dx%d.png", img_h, img_w);
    if (save_latent_as_png(decoded_data, img_h, img_w, png_path)) {
        printf("[png] Saved %s (%dx%d RGB, +%.2fs)\n", png_path, img_w, img_h,
               stage_s());
    } else {
        fprintf(stderr, "[png] Failed to write %s\n", png_path);
    }

    return 0;
}
