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

// Upload one input tensor's CPU staging to its SSBO. Skip empty tensors
// (kv_len=0 past_key_values): copyToGPUBuffer would assert on a 0-byte stage.
void upload_input(const std::shared_ptr<vkop::VulkanCommandPool>& cmdpool,
                  const std::shared_ptr<ITensor>& t) {
    if (t->size() == 0) return;
    if (t->dtype() == typeid(int64_t)) as_tensor<int64_t>(t)->copyToGPU(cmdpool);
    else if (t->dtype() == typeid(int)) as_tensor<int>(t)->copyToGPU(cmdpool);
    else if (t->dtype() == typeid(int8_t)) as_tensor<int8_t>(t)->copyToGPU(cmdpool);
    else if (t->dtype() == typeid(float)) as_tensor<float>(t)->copyToGPU(cmdpool);
    else as_tensor<uint16_t>(t)->copyToGPU(cmdpool);
}

// Fill + upload every input for one round from <round_dir>/input_names.txt.
// Each round's KV-cache past_key_values_{i} grows (kv_len increments), so we
// re-resize + recreate the SSBO + re-upload every round.
void load_round_inputs(const std::shared_ptr<Runtime>& rt,
                       const std::shared_ptr<vkop::VulkanCommandPool>& cmdpool,
                       const std::string& round_dir,
                       std::vector<std::string>* input_names_out) {
    std::ifstream nf(round_dir + "/input_names.txt");
    if (!nf) throw std::runtime_error("no input_names.txt in " + round_dir);
    std::vector<std::string> names;
    std::string line;
    while (std::getline(nf, line)) {
        if (!line.empty()) names.push_back(line);
    }
    for (const auto& name : names) {
        NpyArray arr = load_npy(round_dir + "/" + name + ".npy");
        fill_input(rt, name, arr);
        upload_input(cmdpool, rt->GetInput(name));
    }
    if (input_names_out) *input_names_out = names;
}

// Compare the runtime's 'logits' output against <round_dir>/reference_logits.npy.
// Returns the vkop argmax (== ref argmax on success). Prints top8 + diffs.
int compare_round(const std::shared_ptr<Runtime>& rt,
                  const std::shared_ptr<vkop::VulkanCommandPool>& cmdpool,
                  const std::string& round_dir) {
    NpyArray ref = load_npy(round_dir + "/reference_logits.npy");
    auto logits = rt->GetOutput("logits");
    if (!logits) throw std::runtime_error("no 'logits' output");
    auto lg = as_tensor<uint16_t>(logits);
    lg->copyToCPU(cmdpool);
    const uint16_t* out_bits = reinterpret_cast<const uint16_t*>(lg->data().data());
    size_t n = npy_elem_count(ref.shape);
    if (size_t(lg->num_elements()) != n) {
        std::fprintf(stderr, "[%s] logits size mismatch out=%d ref=%zu\n",
                     round_dir.c_str(), lg->num_elements(), n);
        return -1;
    }
    int argmax_out = -1, argmax_ref = -1;
    float maxv_out = -1e30f, maxv_ref = -1e30f;
    double sum_abs_diff = 0;
    float max_abs_diff = 0;
    std::vector<std::pair<float,int>> vk, rf;
    vk.reserve(n); rf.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        float fo = ITensor::fp16_to_fp32(out_bits[i]);
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
    std::printf("  vkop top8: ");
    for (int i=0;i<8;++i) std::printf("[%d %.4g] ", vk[i].second, vk[i].first);
    std::printf("\n  ref  top8: ");
    for (int i=0;i<8;++i) std::printf("[%d %.4g] ", rf[i].second, rf[i].first);
    std::printf("\n  argmax: vkop=%d  ref=%d  max_abs_diff=%.4g  mean_abs_diff=%.4g  %s\n",
                argmax_out, argmax_ref, max_abs_diff, sum_abs_diff / n,
                (argmax_out == argmax_ref) ? "MATCH" : "DIFF");
    return argmax_out;
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s <model.vkopbin> <inputs_dump_dir>\n", argv[0]);
        std::fprintf(stderr, "  inputs_dump_dir contains round0/ round1/ ... subdirs\n");
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

    // Discover round directories (round0 = prefill, round1.. = decode steps).
    std::vector<int> rounds;
    for (int t = 0; ; ++t) {
        std::string rd = dump_dir + "/round" + std::to_string(t);
        std::ifstream f(rd + "/input_names.txt");
        if (!f) break;
        rounds.push_back(t);
    }
    if (rounds.empty()) {
        // Back-compat: a flat dump (no round subdirs) → treat dump_dir as round0.
        std::ifstream f(dump_dir + "/input_names.txt");
        if (!f) {
            std::fprintf(stderr, "no round subdirs and no input_names.txt in %s\n", dump_dir.c_str());
            return -1;
        }
        rounds.push_back(-1);  // sentinel: load from dump_dir directly
    }
    std::printf("Found %zu round(s)\n", rounds.size());

    int n_match = 0, n_round = 0;
    for (int t : rounds) {
        std::string rd = (t < 0) ? dump_dir : dump_dir + "/round" + std::to_string(t);
        std::printf("\n=== round%d (%s) ===\n", n_round, rd.c_str());
        try {
            load_round_inputs(rt, cmdpool, rd, nullptr);
        } catch (const std::exception& e) {
            std::fprintf(stderr, "load_round_inputs failed: %s\n", e.what());
            return -1;
        }
        // Optional upload-verify: readback inputs_embeds + past_key_values_0
        // and compare their first elements / norms against the npy source, to
        // confirm round-to-round input updates actually landed on the SSBOs.
        if (const char *v = std::getenv("VKOP_VERIFY_INPUTS")) {
            auto verify = [&](const std::string& tname, bool is_fp16) {
                auto t = rt->GetInput(tname);
                if (!t) { std::printf("[vin] %s: NOT FOUND\n", tname.c_str()); return; }
                NpyArray src = load_npy(rd + "/" + tname + ".npy");
                if (is_fp16) {
                    auto tg = as_tensor<uint16_t>(t);
                    if (tg->has_gpu_buffer()) tg->copyToCPU(cmdpool);
                    const uint16_t* p = reinterpret_cast<const uint16_t*>(tg->data().data());
                    const uint16_t* s = reinterpret_cast<const uint16_t*>(src.data.data());
                    int ne = std::min<int>(4, tg->num_elements());
                    float maxd = 0;
                    for (size_t i = 0; i < npy_elem_count(src.shape); ++i) {
                        float d = std::fabs(ITensor::fp16_to_fp32(p[i]) - ITensor::fp16_to_fp32(s[i]));
                        if (d > maxd) maxd = d;
                    }
                    std::printf("[vin] %s: ne=%d maxdiff_vs_npy=%.4g first=[", tname.c_str(),
                                tg->num_elements(), maxd);
                    for (int i = 0; i < ne; ++i) std::printf("%04x,", p[i]);
                    std::printf("]\n");
                }
            };
            verify("inputs_embeds", true);
            verify("past_key_values_0", true);
            verify("attention_bias", true);
        }
        double ms = rt->Run();
        rt->ReadResult();
        std::printf("Run took %.2f ms\n", ms);

        // Optional named-intermediate dump (VKOP_DUMP_TENSORS=comma list, or
        // "*" to dump every fp16 tensor). VKOP_DUMP_ROUND=<n> restricts to one
        // round. Output matches dump_ort_intermediates.py for line-by-line diff.
        if (const char *d = std::getenv("VKOP_DUMP_TENSORS")) {
            int dump_round = -1;
            if (const char *r = std::getenv("VKOP_DUMP_ROUND")) dump_round = std::atoi(r);
            if (dump_round < 0 || dump_round == n_round) {
                std::string s(d);
                // Build (name, tensor) list to dump.
                std::vector<std::pair<std::string, std::shared_ptr<ITensor>>> items;
                if (s == "*") {
                    items = rt->ListTensors();
                } else {
                    std::stringstream ss(s);
                    std::string nm;
                    while (std::getline(ss, nm, ',')) {
                        if (nm.empty()) continue;
                        auto t = rt->GetTensor(nm);
                        items.push_back({nm, t});
                    }
                }
                for (auto &it : items) {
                    const std::string &nm = it.first;
                    auto &tns = it.second;
                    if (!tns) { std::printf("[%s] NOT FOUND\n", nm.c_str()); continue; }
                    if (tns->dtype() == typeid(int64_t)) {
                        // int64 shape-meta dump (opt-in via VKOP_DUMP_INT64=1).
                        if (!std::getenv("VKOP_DUMP_INT64") ||
                            std::getenv("VKOP_DUMP_INT64")[0] != '1') continue;
                        auto tg = as_tensor<int64_t>(tns);
                        if (!tg->has_gpu_buffer()) { std::printf("[%s] no GPU buffer\n", nm.c_str()); continue; }
                        tg->copyToCPU(cmdpool);
                        const int64_t *p = tg->data().data();
                        int ne = tg->num_elements();
                        std::printf("[%s] ne=%d int64=[", nm.c_str(), ne);
                        for (int i = 0; i < 16 && i < ne; ++i) std::printf("%lld,", (long long)p[i]);
                        std::printf("]\n");
                        continue;
                    }
                    if (tns->dtype() == typeid(float)) {
                        // fp32 dump (opt-in via VKOP_DUMP_FP32=1).
                        if (!std::getenv("VKOP_DUMP_FP32") ||
                            std::getenv("VKOP_DUMP_FP32")[0] != '1') continue;
                        auto tg = as_tensor<float>(tns);
                        if (!tg->has_gpu_buffer()) { std::printf("[%s] no GPU buffer\n", nm.c_str()); continue; }
                        tg->copyToCPU(cmdpool);
                        const float *p = tg->data().data();
                        int ne = tg->num_elements();
                        int nan=0,inf=0,zero=0; float mn=1e30f,mx=-1e30f;
                        for (int i=0;i<ne;++i){float v=p[i];
                            if(std::isnan(v))nan++;else if(std::isinf(v))inf++;
                            if(v==0.f)zero++;
                            if(!std::isnan(v)&&!std::isinf(v)){if(v>mx)mx=v;if(v<mn)mn=v;}}
                        std::printf("[%s] ne=%d nan=%d inf=%d zero=%d min=%.4g max=%.4g first=[",
                                    nm.c_str(), ne, nan, inf, zero, mn, mx);
                        for(int i=0;i<8&&i<ne;++i)std::printf("%.4g,",p[i]);
                        std::printf("]\n");
                        continue;
                    }
                    if (tns->dtype() != typeid(uint16_t)) continue;  // fp16 only, match ORT script
                    auto tg = as_tensor<uint16_t>(tns);
                    if (!tg->has_gpu_buffer()) { std::printf("[%s] no GPU buffer\n", nm.c_str()); continue; }
                    tg->copyToCPU(cmdpool);
                    const uint16_t *p = reinterpret_cast<const uint16_t*>(tg->data().data());
                    int ne = tg->num_elements();
                    int nan=0,inf=0,zero=0; float mn=1e30f,mx=-1e30f;
                    for (int i=0;i<ne;++i){float v=ITensor::fp16_to_fp32(p[i]);
                        if(std::isnan(v))nan++;else if(std::isinf(v))inf++;
                        if(v==0.f)zero++;
                        if(!std::isnan(v)&&!std::isinf(v)){if(v>mx)mx=v;if(v<mn)mn=v;}}
                    std::printf("[%s] ne=%d nan=%d inf=%d zero=%d min=%.4g max=%.4g first=[",
                                nm.c_str(), ne, nan, inf, zero, mn, mx);
                    for(int i=0;i<4&&i<ne;++i)std::printf("%04x,",p[i]);
                    std::printf("]\n");
                }
                fflush(stdout);
            }
        }

        int argmax_out = compare_round(rt, cmdpool, rd);
        if (argmax_out >= 0) {
            // Re-fetch ref argmax for the match tally.
            NpyArray ref = load_npy(rd + "/reference_logits.npy");
            int argmax_ref = -1; float mv = -1e30f;
            for (size_t i = 0; i < npy_elem_count(ref.shape); ++i) {
                float fr = ITensor::fp16_to_fp32(reinterpret_cast<const uint16_t*>(ref.data.data())[i]);
                if (fr > mv) { mv = fr; argmax_ref = int(i); }
            }
            if (argmax_out == argmax_ref) ++n_match;
        }
        ++n_round;
        fflush(stdout);
    }

    std::printf("\n=== summary: %d/%d rounds MATCH ===\n", n_match, n_round);
    return (n_match == n_round) ? 0 : 2;
}

