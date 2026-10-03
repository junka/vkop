// junka @ 2026
// Minimal text-encoder driver: loads text_encoder.vkopbin, runs the Qwen3-VL text
// tower over a raw fp16 inputs_embeds [1,P,4096] and writes the raw fp16
// hidden_states (pre-final-norm) plus a numeric fingerprint.
//
// Usage: te_gen <text_encoder.vkopbin> <embeds.raw> <hidden_out.raw> [P]
//
// P defaults to 78 (= drop_idx 14 + prefix_len 64, the length the ONNX was traced
// at; this graph is fully static, so a different P needs a re-export).
//
// Build: cmake -DENABLE_IMAGE_GEN=ON .. && make te_gen
// Run:   see image/exporter/BASELINE.md for the DYLD/VK_ICD env the repo needs.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "core/Tensor.hpp"
#include "core/runtime.hpp"
#include "vulkan/VulkanCommandPool.hpp"
#include "vulkan/VulkanDevice.hpp"
#include "vulkan/VulkanInstance.hpp"

using vkop::VulkanCommandPool;
using vkop::VulkanDevice;
using vkop::VulkanInstance;
using vkop::core::as_tensor;
using vkop::core::ITensor;
using vkop::core::Runtime;

static std::vector<uint16_t> read_fp16(const char* path, size_t n) {
    FILE* f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "cannot read %s\n", path);
        exit(1);
    }
    std::vector<uint16_t> v(n);
    if (fread(v.data(), sizeof(uint16_t), n, f) != n) {
        fprintf(stderr, "short read of %s (expected %zu fp16 values)\n", path, n);
        exit(1);
    }
    fclose(f);
    return v;
}

int main(int argc, char** argv) {
    if (argc < 4) {
        fprintf(stderr, "Usage: %s <text_encoder.vkopbin> <embeds.raw> <out.raw> [P]\n",
                argv[0]);
        return 1;
    }
    const int P = argc > 4 ? atoi(argv[4]) : 78;
    const int HIDDEN = 4096;

    auto embeds = read_fp16(argv[2], (size_t)P * HIDDEN);

    const auto& phydevs = VulkanInstance::getVulkanInstance().getPhysicalDevices();
    if (phydevs.empty()) {
        fprintf(stderr, "no vulkan device\n");
        return 1;
    }
    auto device = std::make_shared<VulkanDevice>(phydevs[0]);
    auto cmdpool = std::make_shared<VulkanCommandPool>(device);

    printf("[load] %s\n", argv[1]);
    auto rt = std::make_shared<Runtime>(cmdpool, argv[1], /*precision=*/1);
    rt->set_backend_buffer(true);  // same as image_gen: shape ops are SSBO-only
    auto t0 = std::chrono::steady_clock::now();
    rt->LoadModel();
    const double load_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    printf("[time] load %.2fs\n", load_s);

    auto input = rt->GetInput("inputs_embeds");
    if (!input) {
        fprintf(stderr, "no 'inputs_embeds' input\n");
        return 1;
    }
    auto shp = input->getShape();
    printf("[io] inputs_embeds shape:");
    for (size_t d = 0; d < shp.size(); ++d) printf(" %u", shp[d]);
    printf(" elems=%zu\n", (size_t)P * HIDDEN);
    int want = P * HIDDEN;
    int got = 1;
    for (size_t d = 0; d < shp.size(); ++d) got *= (int)shp[d];
    if (got != want) {
        fprintf(stderr, "[io] graph wants %d elems, %s has %d (P mismatch -> re-export)\n",
                want, argv[2], got);
        return 1;
    }
    as_tensor<uint16_t>(input)->fillToCPU(embeds.data());
    as_tensor<uint16_t>(input)->copyToGPU(cmdpool);

    auto trun = std::chrono::steady_clock::now();
    rt->Run();
    rt->ReadResult();
    printf("[time] run %.2fs\n",
           std::chrono::duration<double>(std::chrono::steady_clock::now() - trun).count());

    auto output = rt->GetOutput("hidden_states");
    if (!output) {
        fprintf(stderr, "no 'hidden_states' output\n");
        return 1;
    }
    auto oshp = output->getShape();
    int on = 1;
    for (size_t d = 0; d < oshp.size(); ++d) on *= (int)oshp[d];
    printf("[io] hidden_states shape:");
    for (size_t d = 0; d < oshp.size(); ++d) printf(" %u", oshp[d]);
    printf(" (%d elems)\n", on);

    auto tt = as_tensor<uint16_t>(output);
    tt->copyToCPU(cmdpool);
    const auto& d = tt->data();
    const size_t n = std::min<size_t>(on, d.size());
    std::vector<uint16_t> out(d.begin(), d.begin() + n);

    double sumsq = 0.0, absmax = 0.0, nan = 0.0;
    for (uint16_t h : out) {
        const float v = ITensor::fp16_to_fp32(h);
        if (!std::isfinite(v)) {
            ++nan;
            continue;
        }
        sumsq += (double)v * v;
        absmax = std::max(absmax, (double)std::abs(v));
    }
    printf("[gpu] absmax=%.4f rms=%.4f nonfinite=%.0f\n", absmax,
           std::sqrt(sumsq / std::max<size_t>(1, out.size())), nan);

    FILE* g = fopen(argv[3], "wb");
    if (!g) {
        fprintf(stderr, "cannot write %s\n", argv[3]);
        return 1;
    }
    fwrite(out.data(), sizeof(uint16_t), out.size(), g);
    fclose(g);
    printf("[done] wrote %s (%zu B fp16)\n", argv[3], out.size() * sizeof(uint16_t));
    return 0;
}
