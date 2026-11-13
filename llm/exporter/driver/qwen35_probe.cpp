// Copyright 2026 @junka
// Minimal end-to-end check for an exported qwen3_5 .vkopbin.
//
// llm_chat assumes `past_key_values_{0..N-1}` are numbered consecutively, which
// is true for a dense Qwen3 graph and false for qwen3_5 -- only the
// full_attention layers have a KV cache, so the names are sparse and the
// linear-attention layers carry `conv_state_*` / `recurrent_state_*` instead.
// This driver therefore walks the runtime's own input list rather than probing
// names: it fills every input with a deterministic pattern, runs one round, and
// reports the logits plus the first NaN it sees. That is enough to prove the
// graph loads, every op resolves a kernel, and nothing produces garbage.
//
// Usage: qwen35_probe <model.vkopbin> [seq_len]
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "core/Tensor.hpp"
#include "core/runtime.hpp"
#include "include/logger.hpp"
#include "vulkan/VulkanCommandBuffer.hpp"
#include "vulkan/VulkanCommandPool.hpp"
#include "vulkan/VulkanDevice.hpp"
#include "vulkan/VulkanInstance.hpp"

using vkop::VulkanCommandPool;
using vkop::VulkanDevice;
using vkop::VulkanInstance;
using vkop::core::Runtime;
using vkop::core::Tensor;

int main(int argc, char **argv) {
    if (argc < 2) {
        std::printf("usage: %s <model.vkopbin> [seq_len]\n", argv[0]);
        return 2;
    }
    const std::string model_path = argv[1];
    // The graph's seq extent is dynamic, so the dummy length is whatever the
    // model declares; the argument only documents that a caller may pass one.
    (void)argc;
    (void)argv;

    Logger::getInstance().setLevel(LOG_ERROR);
    const auto &phys = VulkanInstance::getVulkanInstance().getPhysicalDevices();
    if (phys.empty()) {
        std::printf("SKIP: no Vulkan device\n");
        return 0;
    }
    auto dev = std::make_shared<VulkanDevice>(phys[0]);
    auto pool = std::make_shared<VulkanCommandPool>(dev);
    std::printf("[probe] GPU: %s\n", dev->getDeviceName().c_str());

    auto rt = std::make_shared<Runtime>(pool, model_path, /*precision=*/1);
    // Must precede LoadModel: it decides which shader family each op builds,
    // and several qwen3_5 ops (RotaryEmbedding, LinearAttention, Conv1d) are
    // buffer-backend only.
    rt->set_backend_buffer(true);
    rt->LoadModel();

    const auto in_names = rt->ListInputNames();
    const auto out_names = rt->ListOutputNames();
    std::printf("[probe] %zu inputs / %zu outputs\n", in_names.size(),
                out_names.size());

    // Fill every graph input with a small deterministic pattern. The point is
    // "does it run and stay finite", not "is the answer right" -- correctness
    // is checked against HF by the exporter's self-check and by
    // check_qwen3_5_ops.py's contract test.
    int filled = 0;
    for (const auto &name : in_names) {
        auto t = rt->GetInput(name);
        if (!t) continue;
        auto shape = t->getShape();
        auto kind = t->elem_kind();
        if (kind == vkop::core::ElemKind::kFloat16) {
            auto ft = vkop::core::as_tensor<uint16_t>(t);
            if (!ft) continue;
            // attention_bias is declared [1,1,q,kv] with dynamic axes, so the
            // loader sees [1,1,1,1]. The op indexes it as (b*S+s)*kv+j over
            // THIS round's q and kv, so size it the way the graph's own
            // attention does: q == the inputs_embeds seq, kv == q + the KV
            // cache's past length (0 on the first round).
            if (name.rfind("attention_bias", 0) == 0 && shape.size() == 4) {
                int q = 1, kv = 1;
                if (auto e = rt->GetInput("inputs_embeds")) {
                    const auto es = e->getShape();
                    if (es.size() >= 2) q = es[1];
                }
                for (const auto &nm : rt->ListInputNames()) {
                    if (nm.rfind("past_key_values", 0) != 0) continue;
                    auto p = rt->GetInput(nm);
                    if (!p) continue;
                    const auto ps = p->getShape();
                    if (ps.size() == 5) {
                        kv = q + (ps[3] > 0 ? ps[3] : 0);
                        break;
                    }
                }
                if (kv <= 0) kv = q;
                const int want_bias = q * kv;
                if (ft->num_elements() != want_bias) {
                    ft->resize(std::vector<int>{1, 1, q, kv});
                    shape = ft->getShape();
                }
            }
            const size_t n = ft->num_elements();
            std::vector<uint16_t> v(n);
            for (size_t i = 0; i < n; i++) {
                const float x = std::sin(static_cast<float>(i) * 0.017F) * 0.5F;
                v[i] = vkop::core::ITensor::fp32_to_fp16(x);
            }
            if (name.rfind("attention_bias", 0) == 0 && shape.size() == 4) {
                // Causal mask over the last two axes.
                const int q = shape[2], kv = shape[3];
                const uint16_t neg =
                    vkop::core::ITensor::fp32_to_fp16(-65504.0F);
                for (int a = 0; a < q; a++)
                    for (int b = 0; b < kv; b++)
                        if (b > a) v[a * kv + b] = neg;
            }
            ft->fillToCPU(v);
            ft->as_storage_buffer(dev);
            ft->copyToGPU(pool);
            filled++;
        } else if (kind == vkop::core::ElemKind::kFloat32) {
            auto ft = vkop::core::as_tensor<float>(t);
            if (!ft) continue;
            const size_t n = ft->num_elements();
            std::vector<float> v(n, 0.0F); // conv/recurrent state starts at 0
            ft->fillToCPU(v);
            ft->as_storage_buffer(dev);
            ft->copyToGPU(pool);
            filled++;
        } else if (kind == vkop::core::ElemKind::kInt64) {
            auto ft = vkop::core::as_tensor<int64_t>(t);
            if (!ft) continue;
            const size_t n = ft->num_elements();
            std::vector<int64_t> v(n);
            // position_ids is (3,B,q): every plane holds the same ramp.
            const auto &sh = t->getShape();
            const int q = sh.empty() ? static_cast<int>(n) : sh.back();
            for (size_t i = 0; i < n; i++)
                v[i] = static_cast<int64_t>(i % static_cast<size_t>(q));
            ft->fillToCPU(v);
            ft->as_storage_buffer(dev);
            ft->copyToGPU(pool);
            filled++;
        } else {
            std::printf("[probe] input %s kind=%d left untouched\n", name.c_str(),
                        static_cast<int>(kind));
        }
    }
    std::printf("[probe] filled %d input(s)\n", filled);

    const double ms = rt->Run();
    std::printf("[probe] Run() = %.1f ms\n", ms);

    int bad = 0;
    for (const auto &name : out_names) {
        auto t = rt->GetOutput(name);
        if (!t) continue;
        if (t->elem_kind() != vkop::core::ElemKind::kFloat16) continue;
        auto ft = vkop::core::as_tensor<uint16_t>(t);
        if (!ft) continue;
        ft->copyToCPU(pool);
        const auto &d = ft->data();
        int nan = 0, inf = 0;
        float mn = 1e30F, mx = -1e30F;
        for (auto raw : d) {
            const float x = vkop::core::ITensor::fp16_to_fp32(raw);
            if (std::isnan(x)) { nan++; continue; }
            if (std::isinf(x)) { inf++; continue; }
            mn = std::min(mn, x);
            mx = std::max(mx, x);
        }
        std::printf("[probe] %-28s n=%zu nan=%d inf=%d min=%.4g max=%.4g\n",
                    name.c_str(), d.size(), nan, inf, mn, mx);
        if (nan || inf) bad++;
    }
    std::printf(bad ? "[probe] RESULT: %d output(s) non-finite\n"
                    : "[probe] RESULT: all outputs finite\n", bad);
    std::fflush(nullptr);
    return bad == 0 ? 0 : 1;
}
