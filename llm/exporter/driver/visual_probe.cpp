// junka @ 2026
// Visual encoder probe: load visual.vkopbin, feed pixel_values + grid_thw, run,
// and dump the 4 outputs (image_features + deepstack_features_{0,1,2}).
//
// Stage-0/stage-1 verification harness: exercises the full visual graph incl.
// the patch_embed Conv3D (5-D Conv) to confirm the Conv2d buffer-backend 5-D
// extension produces ORT-aligned outputs, AND that the C++ image preprocessor
// (image_preproc.hpp) reproduces HF's pixel_values layout.
//
// The graph is dynamic-grid: every grid-derived quantity (pos_embed
// interpolation, rotary tables, merge order) is computed in-graph from the
// grid_thw input, so any size whose sides are multiples of patch*merge runs.
//
// Usage:
//   visual_probe <visual.vkopbin> --pv <pv.bin> [seq_len] [--grid h,w]
//   visual_probe <visual.vkopbin> --image <img> [--grid h,w]
//     --ref-pv <dir>     diff C++ pixel_values vs <dir>/pv.bin (preproc check)
//     --ref-out <dir>    diff 4 vkop outputs vs <dir>/<name>.bin (encoder check)
//   --grid defaults to the C++ preprocess's grid for --image, and to
//   sqrt(seq_len) for --pv when seq_len is a perfect square (196 -> 14x14).
//   Outputs: /tmp/vkop_image_features.bin, /tmp/vkop_deepstack_features_{0,1,2}.bin

#define STB_IMAGE_IMPLEMENTATION
#include "include/stb_image.h"

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
#include "image_preproc.hpp"

using vkop::VulkanInstance;
using vkop::VulkanDevice;
using vkop::core::Runtime;
using vkop::core::as_tensor;
using vkop::core::ITensor;

// Load a raw fp16 file into a vector<uint16_t>.
static std::vector<uint16_t> load_raw_fp16(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) { std::fprintf(stderr, "cannot open %s\n", path.c_str()); return {}; }
    auto sz = f.tellg();
    f.seekg(0);
    std::vector<uint16_t> v(sz / sizeof(uint16_t));
    f.read(reinterpret_cast<char*>(v.data()), sz);
    return v;
}

// Load a raw fp16 file, return fp32 stats for diff printing.
static std::vector<uint16_t> load_ref(const std::string& dir, const std::string& name) {
    std::string p = dir + "/" + name + ".bin";
    std::ifstream f(p, std::ios::binary | std::ios::ate);
    if (!f) return {};
    auto sz = f.tellg();
    f.seekg(0);
    std::vector<uint16_t> v(sz / sizeof(uint16_t));
    f.read(reinterpret_cast<char*>(v.data()), sz);
    return v;
}

// Diff two fp16 buffers: print max_abs / mean_abs / count of mismatches beyond tol.
static void diff_fp16(const std::string& label, const uint16_t* a, const uint16_t* b,
                      int n, float tol) {
    if (n <= 0) { std::printf("[diff:%s] n=%d (skipped)\n", label.c_str(), n); return; }
    float max_abs = 0, sum_abs = 0;
    int over = 0;
    for (int i = 0; i < n; ++i) {
        float va = ITensor::fp16_to_fp32(a[i]);
        float vb = ITensor::fp16_to_fp32(b[i]);
        float d = std::fabs(va - vb);
        if (d > max_abs) max_abs = d;
        sum_abs += d;
        if (d > tol) ++over;
    }
    std::printf("[diff:%s] n=%d max_abs=%.6f mean_abs=%.6f over_tol=%d (tol=%.4f)\n",
                label.c_str(), n, max_abs, sum_abs / n, over, tol);
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr,
            "usage: %s <visual.vkopbin> --pv <pv.bin> [seq_len] [--grid h,w] "
            "[--ref-out <dir>]\n"
            "       %s <visual.vkopbin> --image <img> [--grid h,w] "
            "[--ref-pv <dir>] [--ref-out <dir>]\n",
            argv[0], argv[0]);
        return 1;
    }
    const std::string model_path = argv[1];

    // Parse remaining args: --pv/--image pick input mode; optional seq_len,
    // --grid h,w, --ref-pv / --ref-out for diffing.
    std::string pv_path, image_path, ref_pv_dir, ref_out_dir;
    int seq_len = 196; // default for the export's 224x224 trace example
    int grid_h = 0, grid_w = 0;
    bool have_seq = false;
    for (int i = 2; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--pv" && i + 1 < argc) { pv_path = argv[++i]; }
        else if (a == "--image" && i + 1 < argc) { image_path = argv[++i]; }
        else if (a == "--ref-pv" && i + 1 < argc) { ref_pv_dir = argv[++i]; }
        else if (a == "--ref-out" && i + 1 < argc) { ref_out_dir = argv[++i]; }
        else if (a == "--grid" && i + 1 < argc) {
            if (std::sscanf(argv[++i], "%d,%d", &grid_h, &grid_w) != 2) {
                std::fprintf(stderr, "--grid wants h,w (e.g. 14,14)\n");
                return 1;
            }
        }
        else if (!a.empty() && a[0] != '-' && !have_seq) {
            seq_len = std::atoi(a.c_str()); have_seq = true;
        }
        else {
            std::fprintf(stderr, "unknown arg: %s\n", argv[i]);
            return 1;
        }
    }
    if (pv_path.empty() && image_path.empty()) {
        std::fprintf(stderr, "need --pv <pv.bin> or --image <img>\n");
        return 1;
    }
    // --ref-pv diffs the C++ preprocess output against HF's pv.bin, which only
    // exists on the --image path; on --pv the pv *is* the reference.
    if (!ref_pv_dir.empty() && image_path.empty()) {
        std::fprintf(stderr,
                     "[warn] --ref-pv is ignored without --image (the --pv input "
                     "is already the reference)\n");
    }
    const int row = 1536;

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

    // Build pixel_values (fp16). Either raw load or C++ preprocess.
    std::vector<uint16_t> pv;
    if (!image_path.empty()) {
        int w = 0, h = 0, c = 0;
        unsigned char* img = stbi_load(image_path.c_str(), &w, &h, &c, 3);
        if (!img) {
            std::fprintf(stderr, "stbi_load failed: %s: %s\n", image_path.c_str(),
                         stbi_failure_reason());
            return 1;
        }
        std::printf("[image] %s  %dx%d ch=%d\n", image_path.c_str(), w, h, c);
        auto pr = vkop::export_::preprocess_image_noresize(img, h, w, 3);
        stbi_image_free(img);
        if (pr.pixel_values_fp16.empty()) {
            std::fprintf(stderr,
                "preprocess failed: image %dx%d not divisible by patch*merge=%d\n",
                w, h, vkop::export_::kPatch * vkop::export_::kMerge);
            return 1;
        }
        pv = std::move(pr.pixel_values_fp16);
        seq_len = pr.seq_len;
        grid_h = pr.grid_h; grid_w = pr.grid_w;
        std::printf("[preproc] grid_thw=[%d,%d,%d] seq_len=%d row=%d\n",
                    pr.grid_t, pr.grid_h, pr.grid_w, pr.seq_len, pr.row);
        // Optional: diff C++ pv vs HF reference pv.
        if (!ref_pv_dir.empty()) {
            auto ref = load_ref(ref_pv_dir, "pv");
            if (ref.empty()) {
                std::fprintf(stderr, "no ref pv at %s/pv.bin\n", ref_pv_dir.c_str());
            } else {
                int n = std::min<int>(ref.size(), pv.size());
                diff_fp16("pixel_values", pv.data(), ref.data(), n, 1e-3f);
                if ((int)ref.size() != (int)pv.size()) {
                    std::printf("[diff:pixel_values] SIZE MISMATCH cpp=%zu ref=%zu\n",
                                pv.size(), ref.size());
                }
            }
        }
    } else {
        pv = load_raw_fp16(pv_path);
        if (pv.empty()) return 1;
    }
    std::printf("[pv] seq_len=%d row=%d elems=%zu\n", seq_len, row, pv.size());

    // Runtime + visual model.
    auto rt = std::make_shared<Runtime>(cmdpool, model_path, /*precision=*/1);
    rt->set_backend_buffer(true);
    std::printf("=== LoadModel ===\n");
    rt->LoadModel();
    std::printf("=== LoadModel done ===\n");

    // Feed grid_thw (1,3) int64 — the dynamic-grid graph's second input. An
    // archived static-grid export has no such input: its grid-derived tensors
    // are baked in, so the image must reduce to the export's own grid and
    // there is nothing to feed. Handled rather than rejected so a
    // static-vs-dynamic timing A/B stays runnable with one binary.
    const int64_t gthw[3] = {1, grid_h, grid_w};
    auto gt_in = rt->GetInput("grid_thw");
    if (gt_in && gt_in->dtype() != typeid(int64_t)) {
        std::fprintf(stderr, "grid_thw input is not int64\n");
        return 1;
    }
    if (!gt_in) {
        std::printf("[grid] graph has no grid_thw input (static-grid export); "
                    "its grid is baked in, feeding pixel_values only\n");
    } else {
        if (grid_h == 0 || grid_w == 0) {
            int s = static_cast<int>(std::lround(std::sqrt(static_cast<double>(seq_len))));
            if (s * s != seq_len) {
                std::fprintf(stderr,
                    "seq_len=%d is not a perfect square: pass --grid h,w\n", seq_len);
                return 1;
            }
            grid_h = grid_w = s;
        }
        if (grid_h * grid_w != seq_len) {
            std::fprintf(stderr, "grid %dx%d = %d patches but pixel_values has %d rows\n",
                         grid_h, grid_w, grid_h * grid_w, seq_len);
            return 1;
        }
        std::printf("[grid] thw=[1,%d,%d]\n", grid_h, grid_w);
        rt->ResizeInput("grid_thw", {1u, 3u});
        auto gt = as_tensor<int64_t>(rt->GetInput("grid_thw"));
        gt->fillToCPU(gthw);
        gt->copyToGPU(cmdpool);
    }

    // Fill pixel_values input (seq_len, 1536) fp16.
    rt->ResizeInput("pixel_values",
                    {static_cast<uint32_t>(seq_len),
                     static_cast<uint32_t>(row)});
    auto t = rt->GetInput("pixel_values");
    auto tg = as_tensor<uint16_t>(t);
    tg->fillToCPU(pv.data());
    tg->copyToGPU(cmdpool);
    std::printf("[pv] uploaded (pixel_values + grid_thw)\n");

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
                if (const char* di = std::getenv("VKOP_DUMP_INT64");
                    !di || di[0] != '1') continue;
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
                int nnan = 0, ninf = 0;
                for (int i = 0; i < ne; ++i) {
                    float v = p[i];
                    if (std::isnan(v)) { ++nnan; continue; }
                    if (std::isinf(v)) { ++ninf; continue; }
                    if (v > mx) mx = v;
                    if (v < mn) mn = v;
                    sum += v;
                }
                std::printf("[%s] ne=%d min=%.4g max=%.4g mean=%.4g first=%.4g "
                            "nan=%d inf=%d\n",
                            nm.c_str(), ne, mn, mx, sum / std::max(1, ne),
                            ne ? p[0] : 0.f, nnan, ninf);
                if (std::getenv("VKOP_DUMP_RAW")) {
                    std::string fname = nm;
                    for (auto& ch : fname)
                        if (ch == '/' || ch == '.') ch = '_';
                    std::ofstream of("/tmp/vkopdump_" + fname + ".f32.bin",
                                     std::ios::binary);
                    of.write(reinterpret_cast<const char*>(p), ne * sizeof(float));
                }
                continue;
            }
            if (tns->dtype() != typeid(uint16_t)) continue;
            auto tg = as_tensor<uint16_t>(tns);
            if (!tg->has_gpu_buffer()) continue;
            tg->copyToCPU(cmdpool);
            const uint16_t* p =
                reinterpret_cast<const uint16_t*>(tg->data().data());
            int ne = tg->num_elements();
            float mn = 1e30f, mx = -1e30f, sum = 0;
            int nnan = 0, ninf = 0;
            for (int i = 0; i < ne; ++i) {
                float v = ITensor::fp16_to_fp32(p[i]);
                if (std::isnan(v)) { ++nnan; continue; }
                if (std::isinf(v)) { ++ninf; continue; }
                if (v > mx) mx = v;
                if (v < mn) mn = v;
                sum += v;
            }
            std::printf("[%s] ne=%d min=%.4g max=%.4g mean=%.4g first=%.4g "
                        "nan=%d inf=%d\n",
                        nm.c_str(), ne, mn, mx, sum / std::max(1, ne),
                        ne ? ITensor::fp16_to_fp32(p[0]) : 0.f, nnan, ninf);
            // Optional: raw dump for element-wise ORT diff. Writes the fp16
            // buffer to /tmp/vkopdump_<sanitized_name>.bin.
            if (std::getenv("VKOP_DUMP_RAW")) {
                std::string fname = nm;
                for (auto& ch : fname)
                    if (ch == '/' || ch == '.') ch = '_';
                std::string outp = "/tmp/vkopdump_" + fname + ".bin";
                std::ofstream of(outp, std::ios::binary);
                of.write(reinterpret_cast<const char*>(p),
                         ne * sizeof(uint16_t));
            }
        }
    }

    // Dump 4 outputs + optional ORT diff.
    const char* outs[] = {"image_features", "deepstack_features_0",
                          "deepstack_features_1", "deepstack_features_2"};
    bool any_diff = false;
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
        // Optional ORT diff.
        if (!ref_out_dir.empty()) {
            auto ref = load_ref(ref_out_dir, name);
            if (ref.empty()) {
                std::printf("  [no ref %s/%s.bin]\n", ref_out_dir.c_str(), name);
            } else {
                int n = std::min<int>(ref.size(), data.size());
                diff_fp16(name, data.data(), ref.data(), n, 0.05f);
                if ((int)ref.size() != (int)data.size()) {
                    std::printf("  [diff:%s] SIZE MISMATCH vkop=%zu ref=%zu\n",
                                name, data.size(), ref.size());
                    any_diff = true;
                }
            }
        }
    }
    return any_diff ? 2 : 0;
}
