// junka @ 2025
// End-to-end driver for llm.vkopbin (Qwen3-VL-2B LLM, buffer backend).
//
// Loads the 3649-node model, fills the 35 prefill inputs from .npy files
// dumped by llm/exporter/dump_llm_inputs.py, runs prefill, and compares the
// logits output against reference_logits.npy (ONNX Runtime).
//
// Usage:
//   llm_driver <model.vkopbin> <inputs_dump_dir>
//
// Build (from repo root):
//   g++ -std=c++17 -I. -Imodel/generated llm/exporter/llm_driver.cpp \
//       build/libvkop.a build/model/libvload.a -o build/llm_driver \
//       $(pkg-config --libs vulkan) -lpthread
//
// Env: VKOP_BUFFER_BACKEND=1 is not needed — we call set_backend_buffer(true).

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <cmath>
#include <cstdlib>

#include "vulkan/VulkanDevice.hpp"
#include "vulkan/VulkanInstance.hpp"
#include "include/logger.hpp"
#include "core/Tensor.hpp"
#include "core/runtime.hpp"

using vkop::VulkanInstance;
using vkop::VulkanDevice;
using vkop::core::ITensor;
using vkop::core::Runtime;
using vkop::core::as_tensor;

namespace {

// --- minimal NumPy .npy reader (little-endian, C-contiguous) ---
struct NpyArray {
    std::vector<uint32_t> shape;
    std::string dtype;   // "<f2","<i8","|b1",...
    std::vector<uint8_t> data;
};

NpyArray load_npy(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + path);
    std::vector<uint8_t> buf((std::istreambuf_iterator<char>(f)),
                             std::istreambuf_iterator<char>());
    // magic: \x93NUMPY
    if (buf.size() < 10 || buf[0] != 0x93 || buf[6] != 0x01) {
        throw std::runtime_error("bad npy magic in " + path);
    }
    // version 1.0: header_len is 2 bytes at offset 8
    uint16_t header_len = static_cast<uint16_t>(buf[8]) |
                          (static_cast<uint16_t>(buf[9]) << 8);
    size_t header_start = 10;
    size_t data_start = header_start + header_len;
    if (data_start > buf.size()) throw std::runtime_error("truncated npy header");
    std::string header(buf.begin() + header_start, buf.begin() + data_start);

    NpyArray arr;
    // parse 'descr' and 'fortran_order' and 'shape' from the python-dict header
    auto find_val = [&](const std::string& key) -> std::string {
        auto kp = header.find(key);
        if (kp == std::string::npos) return "";
        auto colon = header.find(':', kp);
        auto q1 = header.find('\'', colon);
        auto q2 = header.find('\'', q1 + 1);
        if (q1 == std::string::npos || q2 == std::string::npos) return "";
        return header.substr(q1 + 1, q2 - q1 - 1);
    };
    arr.dtype = find_val("descr");
    // shape: NumPy writes a tuple in parentheses, e.g. "(3, 1, 1)".
    auto sb = header.find('(');
    auto se = header.find(')', sb);
    if (sb != std::string::npos && se != std::string::npos) {
        std::string shp = header.substr(sb + 1, se - sb - 1);
        std::stringstream ss(shp);
        std::string tok;
        while (std::getline(ss, tok, ',')) {
            // trim
            size_t a = tok.find_first_not_of(' ');
            if (a == std::string::npos) continue;
            arr.shape.push_back(static_cast<uint32_t>(std::stoul(tok.substr(a))));
        }
    }
    arr.data.assign(buf.begin() + data_start, buf.end());
    return arr;
}

size_t npy_elem_count(const std::vector<uint32_t>& shape) {
    size_t n = 1;
    for (auto d : shape) n *= d;
    return n;
}

// Fill a typed Tensor<T> from the npy raw bytes (resize to npy shape first).
template <typename T>
void fill_typed(const std::shared_ptr<ITensor>& t, const NpyArray& arr) {
    auto typed = as_tensor<T>(t);
    typed->resize(arr.shape);
    size_t n = npy_elem_count(arr.shape);
    // fillToCPU reserves CPU staging and memcpy's n*sizeof(T) bytes.
    if (n > 0) {
        typed->fillToCPU(reinterpret_cast<const T*>(arr.data.data()));
    }
    // n==0 leaves an empty CPU tensor (kv_len=0 past_key_values, etc.).
}

// Dispatch fill by the input tensor's runtime dtype. The runtime created each
// input as Tensor<int64_t/int/int8_t/float/uint16_t> per the model's dtype
// string; we resize to the npy shape and recreate the SSBO, then stage the
// data on CPU (upload happens via copyToGPU after).
void fill_input(const std::shared_ptr<Runtime>& rt, const std::string& name,
                const NpyArray& arr) {
    auto t = rt->GetInput(name);
    if (!t) throw std::runtime_error("unknown input " + name);
    rt->ResizeInput(name, arr.shape);
    t = rt->GetInput(name);
    if (t->dtype() == typeid(int64_t)) {
        fill_typed<int64_t>(t, arr);
    } else if (t->dtype() == typeid(int)) {
        fill_typed<int>(t, arr);
    } else if (t->dtype() == typeid(int8_t)) {
        fill_typed<int8_t>(t, arr);
    } else if (t->dtype() == typeid(float)) {
        fill_typed<float>(t, arr);
    } else {
        fill_typed<uint16_t>(t, arr);
    }
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s <model.vkopbin> <inputs_dump_dir>\n", argv[0]);
        return 1;
    }
    const std::string model_path = argv[1];
    const std::string dump_dir = argv[2];

    Logger::getInstance().setLevel(LOG_INFO);
    const auto& phydevs = VulkanInstance::getVulkanInstance().getPhysicalDevices();
    if (phydevs.empty()) { std::printf("no vulkan device\n"); return -1; }
    auto dev = std::make_shared<VulkanDevice>(phydevs[0]);
    if (dev->getDeviceName().find("llvmpipe") != std::string::npos) {
        std::printf("no valid vulkan device\n"); return -1;
    }
    std::printf("GPU: %s\n", dev->getDeviceName().c_str());
    auto cmdpool = std::make_shared<vkop::VulkanCommandPool>(dev);

    auto rt = std::make_shared<Runtime>(cmdpool, model_path, /*precision=*/1);
    rt->set_backend_buffer(true);
    std::printf("=== LoadModel ===\n");
    rt->LoadModel();
    std::printf("=== LoadModel done ===\n");

    // Read input name list (one per line, written by dump_llm_inputs.py).
    std::ifstream nf(dump_dir + "/input_names.txt");
    if (!nf) { std::fprintf(stderr, "no input_names.txt in %s\n", dump_dir.c_str()); return -1; }
    std::vector<std::string> input_names;
    std::string line;
    while (std::getline(nf, line)) {
        if (!line.empty()) input_names.push_back(line);
    }
    std::printf("Filling %zu inputs from %s\n", input_names.size(), dump_dir.c_str());
    for (const auto& name : input_names) {
        NpyArray arr = load_npy(dump_dir + "/" + name + ".npy");
        try {
            fill_input(rt, name, arr);
        } catch (const std::exception& e) {
            std::fprintf(stderr, "fill_input(%s) failed: %s\n", name.c_str(), e.what());
            return -1;
        }
        // Upload CPU staging -> SSBO. copyToGPU with no src reads the tensor's
        // own CPU staging (filled above) and clears it. Skip empty inputs
        // (kv_len=0 past_key_values): copyToGPUBuffer would assert on a 0-byte
        // staging alloc, and there's nothing to upload anyway.
        auto t = rt->GetInput(name);
        if (t->size() == 0) {
            std::printf("  skip empty input %s\n", name.c_str());
            continue;
        }
        if (t->dtype() == typeid(int64_t)) as_tensor<int64_t>(t)->copyToGPU(cmdpool);
        else if (t->dtype() == typeid(int)) as_tensor<int>(t)->copyToGPU(cmdpool);
        else if (t->dtype() == typeid(int8_t)) as_tensor<int8_t>(t)->copyToGPU(cmdpool);
        else if (t->dtype() == typeid(float)) as_tensor<float>(t)->copyToGPU(cmdpool);
        else as_tensor<uint16_t>(t)->copyToGPU(cmdpool);
    }

    std::printf("=== inputs uploaded ===\n");

    std::printf("=== Run ===\n");
    double ms = rt->Run();
    std::printf("Run took %.2f ms\n", ms);
    rt->ReadResult();

    // Dump + NaN-check named intermediates (VKOP_DUMP_TENSORS=comma list).
    if (const char *d = std::getenv("VKOP_DUMP_TENSORS")) {
        std::string s(d), name;
        std::stringstream ss(s);
        while (std::getline(ss, name, ',')) {
            if (name.empty()) continue;
            auto t = rt->GetTensor(name);
            if (!t) { std::printf("[dump] %s: NOT FOUND\n", name.c_str()); continue; }
            auto tg = as_tensor<uint16_t>(t);
            tg->copyToCPU(cmdpool);
            int ne = tg->num_elements();
            int nan_cnt = 0, inf_cnt = 0, zero_cnt = 0;
            float maxv = -1e30f, minv = 1e30f;
            const uint16_t *p = reinterpret_cast<const uint16_t*>(tg->data().data());
            for (int i = 0; i < ne; ++i) {
                float v = ITensor::fp16_to_fp32(p[i]);
                if (std::isnan(v)) nan_cnt++;
                else if (std::isinf(v)) inf_cnt++;
                if (v == 0.f) zero_cnt++;
                if (!std::isnan(v) && !std::isinf(v)) {
                    if (v > maxv) maxv = v;
                    if (v < minv) minv = v;
                }
            }
            std::printf("[dump] %s: ne=%d nan=%d inf=%d zero=%d min=%.4g max=%.4g first=[",
                        name.c_str(), ne, nan_cnt, inf_cnt, zero_cnt, minv, maxv);
            for (int i = 0; i < 4 && i < ne; ++i) std::printf("%04x,", p[i]);
            std::printf("]\n"); fflush(stdout);
        }
    }

    // Compare logits against ONNX Runtime reference.
    NpyArray ref = load_npy(dump_dir + "/reference_logits.npy");
    auto logits = rt->GetOutput("logits");
    if (!logits) { std::fprintf(stderr, "no 'logits' output\n"); return -1; }
    auto lg = as_tensor<uint16_t>(logits);
    lg->copyToCPU(cmdpool);
    std::printf("logits gpu=%d cpu=%d ne=%d firstbits=[",
                (int)lg->has_gpu_buffer(), (int)lg->has_cpu_data(),
                (int)lg->num_elements());
    for (int i = 0; i < 8 && i < lg->num_elements(); ++i) {
        std::printf("%04x,", (unsigned)reinterpret_cast<const uint16_t*>(lg->data().data())[i]);
    }
    std::printf("]\n"); fflush(stdout);
    // reference is fp16 -> uint16_t bits; convert to fp32 for argmax/diff.
    size_t n = npy_elem_count(ref.shape);
    std::printf("logits: out_n=%d ref_n=%zu\n", lg->num_elements(), n);
    if (size_t(lg->num_elements()) != n) {
        std::fprintf(stderr, "logits size mismatch\n");
        return -1;
    }
    int argmax_out = -1, argmax_ref = -1;
    float maxv_out = -1e30f, maxv_ref = -1e30f;
    double sum_abs_diff = 0;
    float max_abs_diff = 0;
    // Top-5 buckets for vkop and ref logits.
    std::vector<std::pair<float,int>> vk, rf;
    vk.reserve(n); rf.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        float fo = ITensor::fp16_to_fp32(reinterpret_cast<const uint16_t*>(lg->data().data())[i]);
        float fr = ITensor::fp16_to_fp32(reinterpret_cast<const uint16_t*>(ref.data.data())[i]);
        if (fo > maxv_out) { maxv_out = fo; argmax_out = int(i); }
        if (fr > maxv_ref) { maxv_ref = fr; argmax_ref = int(i); }
        float d = std::fabs(fo - fr);
        sum_abs_diff += d;
        if (d > max_abs_diff) max_abs_diff = d;
        vk.push_back({fo, (int)i});
        rf.push_back({fr, (int)i});
    }
    auto topp = [](std::vector<std::pair<float,int>> &v) {
        std::partial_sort(v.begin(), v.begin()+std::min<size_t>(8,v.size()), v.end(),
                          [](auto&a,auto&b){return a.first>b.first;});
    };
    topp(vk); topp(rf);
    std::printf("vkop top8: ");
    for (int i=0;i<8;++i) std::printf("[%d %.4g] ", vk[i].second, vk[i].first);
    std::printf("\nref  top8: ");
    for (int i=0;i<8;++i) std::printf("[%d %.4g] ", rf[i].second, rf[i].first);
    // value at ref argmax and vkop argmax in both
    float vk_at_refarg = (argmax_ref>=0 && argmax_ref<(int)n)
        ? ITensor::fp16_to_fp32(reinterpret_cast<const uint16_t*>(lg->data().data())[argmax_ref]) : 0.f;
    float rf_at_vkarg = (argmax_out>=0 && argmax_out<(int)n)
        ? ITensor::fp16_to_fp32(reinterpret_cast<const uint16_t*>(ref.data.data())[argmax_out]) : 0.f;
    std::printf("\nvk@refarg(%d)=%.4g ref@vkarg(%d)=%.4g\n", argmax_ref, vk_at_refarg, argmax_out, rf_at_vkarg);
    std::printf("argmax: vkop=%d  ref=%d\n", argmax_out, argmax_ref);
    std::printf("max_abs_diff=%.4g  mean_abs_diff=%.4g\n", max_abs_diff, sum_abs_diff / n);
    std::printf("=== %s ===\n", (argmax_out == argmax_ref) ? "MATCH" : "DIFF");
    return (argmax_out == argmax_ref) ? 0 : 2;
}

