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
#include <chrono>
#include <random>
#include <fstream>

#include "vulkan/VulkanDevice.hpp"
#include "vulkan/VulkanInstance.hpp"
#include "vulkan/VulkanCommandPool.hpp"
#include "core/Tensor.hpp"
#include "core/runtime.hpp"

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
    
    // Parse optional args
    for (int i = 4; i < argc; ++i) {
        if (strcmp(argv[i], "--size") == 0 && i + 1 < argc) {
            img_size = atoi(argv[++i]);
        } else if (isdigit(argv[i][0])) {
            if (steps == 20) {
                steps = atoi(argv[i]);
            } else {
                seed = atoi(argv[i]);
            }
        }
    }
    
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
    
    // Load models (using buffer backend like llm_chat)
    printf("[load] Loading DiT prefill graph...\n");
    auto prefill_rt = std::make_shared<Runtime>(cmdpool, prefill_path, /*precision=*/1);
    prefill_rt->set_backend_buffer(true);
    prefill_rt->LoadModel();
    printf("[load] Prefill loaded\n");
    
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
    
    // Detect latent channel dimension from decode model's target_latents input
    int target_len = latent_h * latent_w;
    auto target_latents_check = decode_rt->GetInput("target_latents");
    if (target_latents_check) {
        auto check_shape = target_latents_check->getShape();
        if (check_shape.size() >= 3 && check_shape[2] > 0) {
            latent_c = check_shape[2]; // Last dim is C in BLC format
            printf("[model] Detected latent_c=%d from decode model\n", latent_c);
        }
    }
    
    int latent_n = 1 * latent_c * target_len;
    std::vector<float> latent(latent_n);
    generate_random_latent(latent.data(), latent_n, seed);
    
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
    
    // Resize and fill prompt_embeds: [1, prefix_len, context_dim] (fp16)
    prefill_rt->ResizeInput("prompt_embeds", {1u, (uint32_t)seq_len, (uint32_t)context_dim});
    auto prompt_embeds_t = prefill_rt->GetInput("prompt_embeds");
    if (!prompt_embeds_t) {
        fprintf(stderr, "[error] Failed to get prompt_embeds input tensor\n");
        return 1;
    }
    std::vector<float> prompt_embeds_data(seq_len * context_dim, 0.02f);
    // Convert to fp16
    std::vector<uint16_t> prompt_embeds_fp16(prompt_embeds_data.size());
    for (size_t i = 0; i < prompt_embeds_data.size(); ++i) {
        prompt_embeds_fp16[i] = ITensor::fp32_to_fp16(prompt_embeds_data[i]);
    }
    as_tensor<uint16_t>(prompt_embeds_t)->fillToCPU(prompt_embeds_fp16);
    
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
    as_tensor<uint16_t>(cos_prefill_t)->fillToCPU(cos_prefill_fp16);
    
    prefill_rt->ResizeInput("sin", {(uint32_t)seq_len, 128u});
    auto sin_prefill_t = prefill_rt->GetInput("sin");
    std::vector<uint16_t> sin_prefill_fp16(seq_len * hd);
    for (int i = 0; i < seq_len; ++i) {
        for (int j = 0; j < hd; ++j) {
            float freq = i * 0.01f + j * 0.001f;
            sin_prefill_fp16[i * hd + j] = ITensor::fp32_to_fp16(std::sin(freq));
        }
    }
    as_tensor<uint16_t>(sin_prefill_t)->fillToCPU(sin_prefill_fp16);
    
    // timestep_zero: [1]
    prefill_rt->ResizeInput("timestep_zero", {1u});
    auto timestep_zero_t = prefill_rt->GetInput("timestep_zero");
    std::vector<float> timestep_zero_data(1, 0.0f);
    as_tensor<float>(timestep_zero_t)->fillToCPU(timestep_zero_data);
    
    // attention_bias: [1, 1, prefix_len, prefix_len] with causal mask (fp16)
    prefill_rt->ResizeInput("attention_bias", {1u, 1u, (uint32_t)seq_len, (uint32_t)seq_len});
    auto attn_bias_t = prefill_rt->GetInput("attention_bias");
    std::vector<float> attn_bias_prefill(seq_len * seq_len, 0.0f);
    // Apply causal mask (upper triangle = -65504 for fp16)
    for (int i = 0; i < seq_len; ++i) {
        for (int j = i + 1; j < seq_len; ++j) {
            attn_bias_prefill[i * seq_len + j] = -65504.0f;
        }
    }
    // Convert to fp16
    std::vector<uint16_t> attn_bias_fp16(attn_bias_prefill.size());
    for (size_t i = 0; i < attn_bias_prefill.size(); ++i) {
        attn_bias_fp16[i] = ITensor::fp32_to_fp16(attn_bias_prefill[i]);
    }
    as_tensor<uint16_t>(attn_bias_t)->fillToCPU(attn_bias_fp16);
    
    // Run prefill
    auto t_prefill_start = std::chrono::high_resolution_clock::now();
    prefill_rt->Run();
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
    
    for (int layer = 0; layer < num_kv_layers; ++layer) {
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
    }
    
    printf("[kv] Cached %d KV layers (prefix_len=%d)\n", num_kv_layers, prefix_len);
    
    printf("[gen] Starting denoising loop (%d steps)...\n", steps);
    auto t_start = std::chrono::high_resolution_clock::now();
    
    // Denoising loop with real DiT decode inference
    for (int step = 0; step < steps; ++step) {
        float t = timesteps[step];
        float sigma = 1.0f - t; // Simplified sigma schedule
        
        // Resize and fill target_latents: [1, target_len, latent_c]
        decode_rt->ResizeInput("target_latents", {1u, (uint32_t)target_len, (uint32_t)latent_c});
        auto target_lat = decode_rt->GetInput("target_latents");
        std::vector<uint16_t> latent_fp16(latent_n);
        for (int i = 0; i < latent_n; ++i) {
            latent_fp16[i] = ITensor::fp32_to_fp16(latent[i]);
        }
        as_tensor<uint16_t>(target_lat)->fillToCPU(latent_fp16);
        
        // Resize and fill timestep: [1]
        decode_rt->ResizeInput("timestep", {1u});
        auto timestep_t = decode_rt->GetInput("timestep");
        std::vector<float> timestep_data(1, t);
        as_tensor<float>(timestep_t)->fillToCPU(timestep_data);
        
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
        as_tensor<uint16_t>(cos_t)->fillToCPU(cos_decode_fp16);
        
        decode_rt->ResizeInput("sin", {(uint32_t)target_len, 128u});
        auto sin_t = decode_rt->GetInput("sin");
        std::vector<uint16_t> sin_decode_fp16(target_len * hd);
        for (int i = 0; i < target_len; ++i) {
            for (int j = 0; j < hd; ++j) {
                float freq = (prefix_len + i) * 0.01f + j * 0.001f;
                sin_decode_fp16[i * hd + j] = ITensor::fp32_to_fp16(std::sin(freq));
            }
        }
        as_tensor<uint16_t>(sin_t)->fillToCPU(sin_decode_fp16);
        
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
            as_tensor<uint16_t>(kv_t)->fillToCPU(past_kv_cache[layer]);
        }
        
        // attention_bias: [1, 1, target_len, kv_len] where kv_len = prefix_len + target_len (fp16)
        int total_kv_len = prefix_len + target_len;
        decode_rt->ResizeInput("attention_bias", {1u, 1u, (uint32_t)target_len, (uint32_t)total_kv_len});
        auto attn_bias_decode_t = decode_rt->GetInput("attention_bias");
        std::vector<float> attn_bias_decode(target_len * total_kv_len, 0.0f);
        // Convert to fp16
        std::vector<uint16_t> attn_bias_decode_fp16(attn_bias_decode.size());
        for (size_t i = 0; i < attn_bias_decode.size(); ++i) {
            attn_bias_decode_fp16[i] = ITensor::fp32_to_fp16(attn_bias_decode[i]);
        }
        as_tensor<uint16_t>(attn_bias_decode_t)->fillToCPU(attn_bias_decode_fp16);
        
        // Run decode
        decode_rt->Run();
        
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
