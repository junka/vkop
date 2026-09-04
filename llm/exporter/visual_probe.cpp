// junka @ 2026
// Visual encoder probe: load visual.vkopbin, feed a pre-dumped pixel_values
// (196,1536) fp16, run, and dump the 4 outputs (image_features +
// deepstack_features_{0,1,2}) to /tmp/vkop_*.bin for ORT comparison.
//
// This is the stage-0/stage-1 verification harness: it exercises the full
// visual graph including the patch_embed Conv3D (5-D Conv) to confirm the
// Conv2d buffer-backend 5-D extension produces ORT-aligned outputs.
//
// Usage:
//   visual_probe <visual.vkopbin> <pv.bin> [seq_len]
//   pv.bin is a raw fp16 [seq_len, 1536] buffer (dumped by ORT or Python).
//   Outputs: /tmp/vkop_image_features.bin, /tmp/vkop_deepstack_features_{0,1,2}.bin

#include <cstdint>
#include <cstdio>
#include <vector>
#include <fstream>
#include <memory>
#include <string>
#include <sstream>
#include <cmath>

#include "vulkan/VulkanDevice.hpp"
#include "vulkan/VulkanInstance.hpp"
#include "include/logger.hpp"
#include "core/Tensor.hpp"
#include "core/runtime.hpp"

using vkop::VulkanInstance;
using vkop::VulkanDevice;
using vkop::core::Runtime;
using vkop::core::as_tensor;
using vkop::core::ITensor;

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s <visual.vkopbin> <pv.bin> [seq_len]\n",
                     argv[0]);
        return 1;
    }
    const std::string model_path = argv[1];
    const std::string pv_path = argv[2];
    int seq_len = (argc > 3) ? std::atoi(argv[3]) : 196;
    int row = 1536;

    Logger::getInstance().setLevel(LOG_INFO);
    const auto& phydevs =
        vkop::VulkanInstance::getVulkanInstance().getPhysicalDevices();
    if (phydevs.empty()) { std::printf("no vulkan device\n"); return -1; }
    auto dev = std::make_shared<vkop::VulkanDevice>(phydevs[0]);
    if (dev->getDeviceName().find("llvmpipe") != std::string::npos) {
        std::printf("no valid vulkan device\n"); return -1;
    }
    std::printf("GPU: %s\n", dev->getDeviceName().c_str());
    auto cmdpool = std::make_shared<vkop::VulkanCommandPool>(dev);

    // Load pixel_values (fp16).
    std::ifstream f(pv_path, std::ios::binary | std::ios::ate);
    if (!f) { std::fprintf(stderr, "cannot open %s\n", pv_path.c_str()); return 1; }
    auto sz = f.tellg();
    f.seekg(0);
    std::vector<uint16_t> pv(sz / sizeof(uint16_t));
    f.read(reinterpret_cast<char*>(pv.data()), sz);
    std::printf("[pv] %s  seq_len=%d row=%d  elems=%zu\n", pv_path.c_str(),
                seq_len, row, pv.size());

    // Runtime + visual model.
    auto rt = std::make_shared<Runtime>(cmdpool, model_path, /*precision=*/1);
    rt->set_backend_buffer(true);
    std::printf("=== LoadModel ===\n");
    rt->LoadModel();
    std::printf("=== LoadModel done ===\n");

    // Fill pixel_values input (seq_len, 1536) fp16.
    rt->ResizeInput("pixel_values",
                    {static_cast<uint32_t>(seq_len),
                     static_cast<uint32_t>(row)});
    auto t = rt->GetInput("pixel_values");
    auto tg = as_tensor<uint16_t>(t);
    tg->fillToCPU(pv.data());
    tg->copyToGPU(cmdpool);
    std::printf("[pv] uploaded\n");

    // Run.
    double ms = rt->Run();
    rt->ReadResult();
    std::printf("[run] %.1fms\n", ms);

    // Optional named-intermediate dump (VKOP_DUMP_TENSORS='*' or 'name1,name2').
    // Mirrors llm_chat's dump for topo diff vs ORT (/tmp/ort_visual_all.txt).
    if (const char* d = std::getenv("VKOP_DUMP_TENSORS")) {
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
        for (auto& it : items) {
            const std::string& nm = it.first;
            auto& tns = it.second;
            if (!tns) { std::printf("[%s] NOT FOUND\n", nm.c_str()); continue; }
            if (tns->dtype() == typeid(int64_t)) {
                if (!std::getenv("VKOP_DUMP_INT64")) continue;
                auto tg = as_tensor<int64_t>(tns);
                if (!tg->has_gpu_buffer()) continue;
                tg->copyToCPU(cmdpool);
                const int64_t* p = tg->data().data();
                int ne = tg->num_elements();
                std::printf("[%s] ne=%d int64=[", nm.c_str(), ne);
                for (int i = 0; i < 16 && i < ne; ++i)
                    std::printf("%lld,", (long long)p[i]);
                std::printf("]\n");
                continue;
            }
            if (tns->dtype() == typeid(float)) {
                auto tg = as_tensor<float>(tns);
                if (!tg->has_gpu_buffer()) continue;
                tg->copyToCPU(cmdpool);
                const float* p = tg->data().data();
                int ne = tg->num_elements();
                float mn = 1e30f, mx = -1e30f, sum = 0;
                for (int i = 0; i < ne; ++i) {
                    float v = p[i];
                    if (!std::isnan(v) && !std::isinf(v)) {
                        if (v > mx) mx = v;
                        if (v < mn) mn = v;
                        sum += v;
                    }
                }
                std::printf("[%s] ne=%d min=%.4g max=%.4g mean=%.4g first=%.4g\n",
                            nm.c_str(), ne, mn, mx, sum / std::max(1, ne),
                            ne ? p[0] : 0.f);
                continue;
            }
            if (tns->dtype() != typeid(uint16_t)) continue;
            auto tg = as_tensor<uint16_t>(tns);
            if (!tg->has_gpu_buffer()) continue;
            // DBG: print underlying VkBuffer handle for aliasing cross-ref
            if (std::getenv("VKOP_DUMP_SAMPLE")) {
                auto buf = tg->as_storage_buffer(dev);
                std::printf("[buf:%s this=%p h=%p sz=%zu]\n", nm.c_str(),
                            (void*)tns.get(), (void*)buf->getBuffer(),
                            buf->getSize());
            }
            tg->copyToCPU(cmdpool);
            const uint16_t* p =
                reinterpret_cast<const uint16_t*>(tg->data().data());
            int ne = tg->num_elements();
            float mn = 1e30f, mx = -1e30f, sum = 0;
            for (int i = 0; i < ne; ++i) {
                float v = ITensor::fp16_to_fp32(p[i]);
                if (!std::isnan(v) && !std::isinf(v)) {
                    if (v > mx) mx = v;
                    if (v < mn) mn = v;
                    sum += v;
                }
            }
            std::printf("[%s] ne=%d min=%.4g max=%.4g mean=%.4g first=%.4g\n",
                        nm.c_str(), ne, mn, mx, sum / std::max(1, ne),
                        ne ? ITensor::fp16_to_fp32(p[0]) : 0.f);
            if (std::getenv("VKOP_DUMP_RAW")) {
                int nraw = std::getenv("VKOP_DUMP_RAWN") ? std::atoi(std::getenv("VKOP_DUMP_RAWN")) : 8;
                std::printf("  raw=[");
                for (int i = 0; i < nraw && i < ne; ++i)
                    std::printf("%.4g ", ITensor::fp16_to_fp32(p[i]));
                std::printf("]\n");
            }
            // DBG: sample words at several offsets to see which gids wrote.
            if (std::getenv("VKOP_DUMP_SAMPLE") && !std::getenv("VKOP_DUMP_INT64") &&
                tns->dtype() == typeid(uint16_t)) {
                int offs[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 100, 200702, 200704, 200706, 250000, 300000, 301055};
                const uint32_t* wp = reinterpret_cast<const uint32_t*>(p);
                int nwords = ne / 2;
                std::printf("  words=[");
                for (int off : offs) {
                    if (off < nwords) {
                        uint32_t u = wp[off];
                        float bf;
                        std::memcpy(&bf, &u, sizeof(bf));
                        std::printf("%d:%.4g(u%u) ", off, bf, u);
                    }
                }
                std::printf("]\n");
                // Find first non-zero word and last non-zero word.
                int first_nz = -1, last_nz = -1;
                for (int i = 0; i < nwords; ++i) {
                    if (wp[i] != 0) {
                        if (first_nz < 0) first_nz = i;
                        last_nz = i;
                    }
                }
                std::printf("  nzrange: first=%d last=%d (nwords=%d)\n",
                            first_nz, last_nz, nwords);
            }
        }
    }

    // Dump 4 outputs.
    const char* outs[] = {"image_features", "deepstack_features_0",
                          "deepstack_features_1", "deepstack_features_2"};
    for (const char* name : outs) {
        auto o = rt->GetOutput(name);
        if (!o) { std::fprintf(stderr, "no output %s\n", name); continue; }
        auto og = as_tensor<uint16_t>(o);
        og->copyToCPU(cmdpool);
        const auto& data = og->data();
        // stats in fp32
        float mn = 1e30f, mx = -1e30f, sum = 0;
        for (uint16_t h : data) {
            float v = ITensor::fp16_to_fp32(h);
            if (v < mn) mn = v;
            if (v > mx) mx = v;
            sum += v;
        }
        std::printf("[%s] ne=%d shape=[", name, og->num_elements());
        for (size_t k = 0; k < og->getShape().size(); ++k)
            std::printf("%d ", og->getShape()[k]);
        std::printf("] min=%.4f max=%.4f mean=%.4f\n", mn, mx,
                    sum / std::max(1, og->num_elements()));
        // save to /tmp/vkop_<name>.bin
        std::string outp = std::string("/tmp/vkop_") + name + ".bin";
        std::ofstream of(outp, std::ios::binary);
        of.write(reinterpret_cast<const char*>(data.data()),
                 data.size() * sizeof(uint16_t));
        std::printf("  saved %s\n", outp.c_str());
    }
    return 0;
}
