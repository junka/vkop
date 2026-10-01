// junka @ 2026
// Image generation driver for Qwen-Image-2.1 (DiT + VAE decoder).
// Text prompt -> tokenized -> DiT prefill -> denoising loop -> VAE decode -> PNG
//
// Usage:
//   image_gen <dit_prefill.vkopbin> <dit_decode.vkopbin> <vae_decoder.onnx> \
//             <prompt> [steps] [seed] [--size 512|1024]
//
// Build: cmake -DENABLE_IMAGE_GEN=ON .. && make image_gen

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <map>
#include <chrono>
#include <random>
#include <fstream>
#include <sys/stat.h>

#include "vulkan/VulkanDevice.hpp"
#include "vulkan/VulkanInstance.hpp"
#include "vulkan/VulkanCommandPool.hpp"
#include "core/Tensor.hpp"
#include "core/runtime.hpp"
#include "model/load.hpp"

using vkop::VulkanInstance;
using vkop::VulkanDevice;
using vkop::VulkanCommandBuffer;
using vkop::VulkanCommandPool;
using vkop::core::ITensor;
using vkop::core::Runtime;
using vkop::core::as_tensor;

namespace {

// ---- Scheduler: FlowMatchEulerDiscreteScheduler (simplified) ----
struct SchedulerConfig {
    int num_train_timesteps = 1000;
    float base_shift = 0.5f;
    float max_shift = 0.9f;
    float shift = 1.0f;
    float shift_terminal = 0.02f;
    bool use_dynamic_shifting = true;
};

class FlowMatchEulerScheduler {
public:
    explicit FlowMatchEulerScheduler(const SchedulerConfig& cfg) : cfg_(cfg) {}
    
    // Compute timestep schedule for given number of steps
    std::vector<float> compute_timesteps(int num_steps) {
        std::vector<float> timesteps(num_steps);
        for (int i = 0; i < num_steps; ++i) {
            // Linear spacing from 1.0 to 0.0
            timesteps[i] = 1.0f - static_cast<float>(i) / num_steps;
        }
        return timesteps;
    }
    
    // Euler step: x_{t-1} = x_t - sigma * velocity
    // For flow matching: velocity ≈ model_output
    void step(float* latent, const float* velocity, float sigma, int numel) {
        for (int i = 0; i < numel; ++i) {
            latent[i] -= sigma * velocity[i];
        }
    }
    
private:
    SchedulerConfig cfg_;
};

// ---- Random latent generator ----
void generate_random_latent(float* data, int numel, uint32_t seed) {
    std::mt19937 gen(seed);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (int i = 0; i < numel; ++i) {
        data[i] = dist(gen);
    }
}

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
    if (argc < 4) {
        fprintf(stderr, 
                "Usage: %s <dit_prefill.vkopbin> <dit_decode.vkopbin> <prompt> "
                "[steps] [seed] [--size 512|1024]\n", argv[0]);
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
    printf("[load] Loading DiT prefill graph...\n");
    auto prefill_rt = std::make_shared<Runtime>(cmdpool, prefill_path, /*precision=*/1);
    prefill_rt->set_backend_buffer(true);
    prefill_rt->LoadModel();
    printf("[load] Prefill loaded\n");

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
    
    // Setup scheduler
    SchedulerConfig sched_cfg;
    FlowMatchEulerScheduler scheduler(sched_cfg);
    auto timesteps = scheduler.compute_timesteps(steps);
    
    // Prepare RoPE embeddings for target sequence
    int hd = 128; // head dimension for RoPE
    std::vector<float> cos_vec(target_len * hd);
    std::vector<float> sin_vec(target_len * hd);
    for (int i = 0; i < target_len; ++i) {
        for (int j = 0; j < hd; ++j) {
            float freq = i * 0.01f + j * 0.001f; // Simplified frequency
            cos_vec[i * hd + j] = std::cos(freq);
            sin_vec[i * hd + j] = std::sin(freq);
        }
    }
    
    // Attention bias (zeros for now)
    int kv_len = target_len;
    int attn_bias_size = target_len * kv_len;
    std::vector<float> attn_bias(attn_bias_size, 0.0f);
    
    // ---- KV Cache Integration ----
    // Prefill outputs present_kv_* tensors, decode needs past_kv_* inputs
    // Dynamically detect the number of KV layers from the model
    
    int prefix_len = 64; // Default sequence length for prefill (adjust based on model)
    
    printf("[prefill] Running prompt encoding to get initial KV cache...\n");
    
    // Prepare prefill inputs
    int seq_len = prefix_len;
    
    // Detect context_dim from prefill model's prompt_embeds input
    auto prompt_embeds_check = prefill_rt->GetInput("prompt_embeds");
    int context_dim = 4096; // Default for full model
    if (prompt_embeds_check) {
        auto check_shape = prompt_embeds_check->getShape();
        if (check_shape.size() >= 3 && check_shape[2] > 0) {
            context_dim = check_shape[2];
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
        // Resize and fill prompt_embeds: [1, prefix_len, context_dim] (fp16)
        prefill_rt->ResizeInput("prompt_embeds", {1u, (uint32_t)seq_len, (uint32_t)context_dim});
        auto prompt_embeds_t = prefill_rt->GetInput("prompt_embeds");
        if (!prompt_embeds_t) {
            fprintf(stderr, "[error] Failed to get prompt_embeds input tensor\n");
            return 1;
        }
        std::vector<float> prompt_embeds_data(seq_len * context_dim, 0.02f);
        std::vector<uint16_t> prompt_embeds_fp16(prompt_embeds_data.size());
        for (size_t i = 0; i < prompt_embeds_data.size(); ++i) {
            prompt_embeds_fp16[i] = ITensor::fp32_to_fp16(prompt_embeds_data[i]);
        }
        fill16(prompt_embeds_t, prompt_embeds_fp16);
        
        // Resize and fill cos/sin: [prefix_len, 128]
        prefill_rt->ResizeInput("cos", {(uint32_t)seq_len, 128u});
        auto cos_prefill_t = prefill_rt->GetInput("cos");
        std::vector<uint16_t> cos_prefill_fp16(seq_len * hd);
        for (int i = 0; i < seq_len; ++i) {
            for (int j = 0; j < hd; ++j) {
                float freq = i * 0.01f + j * 0.001f;
                cos_prefill_fp16[i * hd + j] = ITensor::fp32_to_fp16(std::cos(freq));
            }
        }
        fill16(cos_prefill_t, cos_prefill_fp16);
        
        prefill_rt->ResizeInput("sin", {(uint32_t)seq_len, 128u});
        auto sin_prefill_t = prefill_rt->GetInput("sin");
        std::vector<uint16_t> sin_prefill_fp16(seq_len * hd);
        for (int i = 0; i < seq_len; ++i) {
            for (int j = 0; j < hd; ++j) {
                float freq = i * 0.01f + j * 0.001f;
                sin_prefill_fp16[i * hd + j] = ITensor::fp32_to_fp16(std::sin(freq));
            }
        }
        fill16(sin_prefill_t, sin_prefill_fp16);
        
        // timestep_zero: [1]
        prefill_rt->ResizeInput("timestep_zero", {1u});
        auto timestep_zero_t = prefill_rt->GetInput("timestep_zero");
        std::vector<float> timestep_zero_data(1, 0.0f);
        fill32(timestep_zero_t, timestep_zero_data);
        
        // attention_bias: [1, 1, prefix_len, prefix_len] with causal mask (fp16)
        prefill_rt->ResizeInput("attention_bias", {1u, 1u, (uint32_t)seq_len, (uint32_t)seq_len});
        auto attn_bias_t = prefill_rt->GetInput("attention_bias");
        std::vector<float> attn_bias_prefill(seq_len * seq_len, 0.0f);
        for (int i = 0; i < seq_len; ++i) {
            for (int j = i + 1; j < seq_len; ++j) {
                attn_bias_prefill[i * seq_len + j] = -65504.0f;
            }
        }
        std::vector<uint16_t> attn_bias_fp16(attn_bias_prefill.size());
        for (size_t i = 0; i < attn_bias_prefill.size(); ++i) {
            attn_bias_fp16[i] = ITensor::fp32_to_fp16(attn_bias_prefill[i]);
        }
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
    
    printf("[prefill] Done in %.3fs, found %d KV layers\n", t_prefill, num_kv_layers);
    
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
    
    if (ref_mode) {
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
    }
    
    printf("[kv] Cached %d KV layers (prefix_len=%d)\n", num_kv_layers, prefix_len);

    // Sequential loading (same strategy as the ORT baseline): the prefill graph
    // is released before the decode graph takes its weights, so a 7.12B model
    // needs one graph's worth of device memory at a time.
    prefill_rt.reset();
    printf("[load] Loading DiT decode graph...\n");
    auto decode_rt = std::make_shared<Runtime>(cmdpool, decode_path, /*precision=*/1);
    decode_rt->set_backend_buffer(true);
    try {
        decode_rt->LoadModel();
        printf("[load] Decode loaded\n");
    } catch (const std::exception& e) {
        fprintf(stderr, "[error] Failed to load decode model: %s\n", e.what());
        return 1;
    }

    printf("[gen] Starting denoising loop (%d steps)...\n", steps);
    auto t_start = std::chrono::high_resolution_clock::now();
    
    // Denoising loop with real DiT decode inference
    for (int step = 0; step < steps; ++step) {
        float t = timesteps[step];
        float sigma = 1.0f - t; // Simplified sigma schedule
        
        if (ref_mode && step > 0) {
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
        
        if (ref_mode) {
            if (!ref_fill(decode_rt.get(), "cos", "cos_decode.raw")) return 1;
            if (!ref_fill(decode_rt.get(), "sin", "sin_decode.raw")) return 1;
        } else {
            // Resize and fill cos/sin: [target_len, 128]
            decode_rt->ResizeInput("cos", {(uint32_t)target_len, 128u});
            auto cos_t = decode_rt->GetInput("cos");
            std::vector<uint16_t> cos_decode_fp16(target_len * hd);
            for (int i = 0; i < target_len; ++i) {
                for (int j = 0; j < hd; ++j) {
                    float freq = (prefix_len + i) * 0.01f + j * 0.001f;
                    cos_decode_fp16[i * hd + j] = ITensor::fp32_to_fp16(std::cos(freq));
                }
            }
            fill16(cos_t, cos_decode_fp16);
            
            decode_rt->ResizeInput("sin", {(uint32_t)target_len, 128u});
            auto sin_t = decode_rt->GetInput("sin");
            std::vector<uint16_t> sin_decode_fp16(target_len * hd);
            for (int i = 0; i < target_len; ++i) {
                for (int j = 0; j < hd; ++j) {
                    float freq = (prefix_len + i) * 0.01f + j * 0.001f;
                    sin_decode_fp16[i * hd + j] = ITensor::fp32_to_fp16(std::sin(freq));
                }
            }
            fill16(sin_t, sin_decode_fp16);
        }
        
        // Feed all past_kv inputs (dynamic count and shape based on prefill outputs)
        for (int layer = 0; layer < num_kv_layers; ++layer) {
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
        } else {
            int total_kv_len = prefix_len + target_len;
            decode_rt->ResizeInput("attention_bias", {1u, 1u, (uint32_t)target_len, (uint32_t)total_kv_len});
            auto attn_bias_decode_t = decode_rt->GetInput("attention_bias");
            std::vector<float> attn_bias_decode(target_len * total_kv_len, 0.0f);
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
        
        // Euler step
        scheduler.step(latent.data(), velocity.data(), sigma, latent_n);
        
        if ((step + 1) % 5 == 0 || step == 0) {
            printf("  Step %d/%d (t=%.3f, σ=%.3f)\n", 
                   step + 1, steps, t, sigma);
        }
    }
    
    auto t_end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(t_end - t_start).count();
    printf("[gen] Denoising complete: %.1fs (%.1f s/step)\n", 
           elapsed, elapsed / steps);
    
    // Save output
    char out_path[256];
    snprintf(out_path, sizeof(out_path), "latent_out_%dx%d.raw", img_size, img_size);
    save_raw(out_path, latent.data(), latent_n);
    
    printf("[done] Raw latent saved. Next steps:\n");
    printf("  1. Implement DiT decode inference (blocked by missing ops)\n");
    printf("  2. Add VAE decoder (ONNX or upstream diffusers)\n");
    printf("  3. Convert latent to PNG\n");
    
    return 0;
}
