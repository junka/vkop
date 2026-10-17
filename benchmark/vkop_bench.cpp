// junka @ 2026
// vkop-bench -- trtexec-style generic benchmark driver for any .vkopbin.
//
// Loads a model, sizes its graph inputs to something the runtime can
// dispatch on, fills them with deterministic pseudo-random data, warms
// up, then times N iterations of Runtime::Run() and prints a percentile
// table. This is the "give me ms/throughput for this vkopbin" path
// trtexec covers with `--iterations / --warmUp / --duration`, and the
// reason it exists here is that every prior vkop benchmark was
// model-specific (llm_chat / image_gen / cache_probe) and needed a
// driver-side harness to feed the tokens or the images or the KV cache.
//
// What it does NOT do: verify numerics. Inputs are random, so outputs
// are garbage. The point is timing and load-phase measurement, not
// correctness; use llm_driver / compare_vkop_ort.py for that.
//
// Usage:
//   vkop-bench model.vkopbin
//       [--iters N] [--warmup N] [--precision fp16|fp32]
//       [--backend buffer|image]
//       [--shape name:d1xd2x...]      (may repeat; required for dynamic dims)
//       [--seed N]
//
// Dynamic-shape inputs: LoadModel turns every -1 sentinel into a
// placeholder dim of 1 (see core/runtime.cpp:186ff), so a raw run without
// --shape would execute a [1,1,...] kernel. --shape rebinds the tensor
// via ResizeInput to a concrete dim and is required for anything
// meaningful on an LLM.
//
// Composition with the pipeline cache / shader archive / stripped build:
// the binary respects VK_PIPELINE_CACHE, VKOP_SHADER_ARCHIVE, and
// MESA_SHADER_CACHE_DISABLE the same way llm_chat does. Bench in the
// deployment's env, not this binary's default.

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <typeinfo>
#include <unordered_map>
#include <vector>

#include "core/Tensor.hpp"
#include "core/runtime.hpp"
#include "include/logger.hpp"
#include "vulkan/VulkanDevice.hpp"
#include "vulkan/VulkanInstance.hpp"

using vkop::VulkanDevice;
using vkop::VulkanInstance;
using vkop::core::as_tensor;
using vkop::core::ITensor;
using vkop::core::Runtime;

namespace {

struct Options {
    std::string model_path;
    int iters = 10;
    int warmup = 3;
    int precision = 1; // 1 = fp16 (matches llm_chat / cache_probe default)
    bool buffer_backend = true;
    uint32_t seed = 0xc0ffee;
    // name -> concrete dims. Anything here overrides LoadModel's
    // placeholder of 1 for a dynamic (-1) input dim.
    std::unordered_map<std::string, std::vector<uint32_t>> shapes;
    bool list_inputs_only = false;
    bool time_load_only = false;
};

void usage(const char *argv0) {
    std::fprintf(stderr,
                 "Usage: %s model.vkopbin [options]\n"
                 "  --iters N             timed iterations (default 10)\n"
                 "  --warmup N            warmup iterations before timing (default 3)\n"
                 "  --precision fp16|fp32 graph precision (default fp16)\n"
                 "  --backend buffer|image  runtime backend (default buffer)\n"
                 "  --shape name:d1xd2x...   resize an input; may repeat\n"
                 "  --seed N                fill RNG seed (default 0xc0ffee)\n"
                 "  --list-inputs         print each input's name/dtype/shape and exit\n"
                 "  --time-load-only      measure LoadModel only, no Run()\n",
                 argv0);
}

bool parse_shape_spec(const std::string &spec, std::string &name,
                      std::vector<uint32_t> &dims, std::string &err) {
    auto colon = spec.find(':');
    if (colon == std::string::npos) {
        err = "expected name:d1xd2..., got " + spec;
        return false;
    }
    name = spec.substr(0, colon);
    std::stringstream ss(spec.substr(colon + 1));
    std::string tok;
    while (std::getline(ss, tok, 'x')) {
        if (tok.empty()) {
            err = "empty dim in " + spec;
            return false;
        }
        try {
            dims.push_back(static_cast<uint32_t>(std::stoul(tok)));
        } catch (...) {
            err = "bad dim " + tok + " in " + spec;
            return false;
        }
    }
    if (dims.empty()) {
        err = "no dims in " + spec;
        return false;
    }
    return true;
}

bool parse_args(int argc, char **argv, Options &o) {
    if (argc < 2) {
        usage(argv[0]);
        return false;
    }
    std::string first = argv[1];
    if (first == "-h" || first == "--help") {
        usage(argv[0]);
        return false;
    }
    o.model_path = first;
    for (int i = 2; i < argc; ++i) {
        std::string a = argv[i];
        auto need_next = [&](const char *flag) -> const char * {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "%s needs a value\n", flag);
                std::exit(2);
            }
            return argv[++i];
        };
        if (a == "--iters") {
            o.iters = std::atoi(need_next("--iters"));
        } else if (a == "--warmup") {
            o.warmup = std::atoi(need_next("--warmup"));
        } else if (a == "--precision") {
            std::string p = need_next("--precision");
            if (p == "fp16")
                o.precision = 1;
            else if (p == "fp32")
                o.precision = 0;
            else {
                std::fprintf(stderr, "--precision must be fp16|fp32\n");
                return false;
            }
        } else if (a == "--backend") {
            std::string b = need_next("--backend");
            if (b == "buffer")
                o.buffer_backend = true;
            else if (b == "image")
                o.buffer_backend = false;
            else {
                std::fprintf(stderr, "--backend must be buffer|image\n");
                return false;
            }
        } else if (a == "--seed") {
            o.seed = static_cast<uint32_t>(std::strtoul(need_next("--seed"), nullptr, 0));
        } else if (a == "--shape") {
            std::string spec = need_next("--shape");
            std::string name;
            std::vector<uint32_t> dims;
            std::string err;
            if (!parse_shape_spec(spec, name, dims, err)) {
                std::fprintf(stderr, "--shape %s\n", err.c_str());
                return false;
            }
            o.shapes[name] = std::move(dims);
        } else if (a == "--list-inputs") {
            o.list_inputs_only = true;
        } else if (a == "--time-load-only") {
            o.time_load_only = true;
        } else if (a == "-h" || a == "--help") {
            usage(argv[0]);
            return false;
        } else {
            std::fprintf(stderr, "unknown arg: %s\n", a.c_str());
            usage(argv[0]);
            return false;
        }
    }
    return true;
}

// Fill an ITensor with deterministic data whose dtype matches the tensor.
// fp16 / fp32 get values in [-0.5, +0.5] (LLM-shaped magnitudes, so the
// kernels don't produce infinities that change timing), int64 / int32 get
// small non-negative indices (so downstream Gather/Reshape stay in-range),
// int8 gets [-64, +64] (won't overflow a QDQ accumulator). Not the same
// values a real driver would feed; that's fine because we don't verify
// outputs, we time dispatches.
template <typename T>
std::vector<T> make_fill(int n, uint32_t seed);

template <>
std::vector<float> make_fill<float>(int n, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    std::vector<float> out(n);
    for (auto &v : out)
        v = dist(rng);
    return out;
}

template <>
std::vector<uint16_t> make_fill<uint16_t>(int n, uint32_t seed) {
    // fp16 stored as uint16_t in vkop's Tensor path; encode float bits
    // rounded to half precision so the shader reads realistic exponent
    // ranges instead of arbitrary bit patterns (which would be NaN or
    // denormal, forcing driver-side slow paths).
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    std::vector<uint16_t> out(n);
    for (auto &v : out) {
        float f = dist(rng);
        // Manual float -> half, no need to link half.h here.
        uint32_t bits;
        std::memcpy(&bits, &f, 4);
        uint32_t sign = (bits >> 16) & 0x8000u;
        int32_t exp = ((bits >> 23) & 0xff) - 127 + 15;
        uint32_t mant = bits & 0x7fffffu;
        uint16_t h;
        if (exp <= 0) {
            h = static_cast<uint16_t>(sign);
        } else if (exp >= 31) {
            h = static_cast<uint16_t>(sign | 0x7c00u);
        } else {
            h = static_cast<uint16_t>(sign | (exp << 10) | (mant >> 13));
        }
        v = h;
    }
    return out;
}

template <>
std::vector<int64_t> make_fill<int64_t>(int n, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int64_t> dist(0, 8);
    std::vector<int64_t> out(n);
    for (auto &v : out)
        v = dist(rng);
    return out;
}

template <>
std::vector<int> make_fill<int>(int n, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> dist(0, 8);
    std::vector<int> out(n);
    for (auto &v : out)
        v = dist(rng);
    return out;
}

template <>
std::vector<int8_t> make_fill<int8_t>(int n, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> dist(-64, 63);
    std::vector<int8_t> out(n);
    for (auto &v : out)
        v = static_cast<int8_t>(dist(rng));
    return out;
}

// Product of a shape vector. Empty shape -> 1 (scalar).
int64_t element_count(const std::vector<int> &shape) {
    int64_t n = 1;
    for (int d : shape) {
        n *= static_cast<int64_t>(std::max(0, d));
    }
    return n;
}

// Resize + fill + upload one input. Every dispatch is dtype-gated
// (matching llm_chat's upload_input); an unknown dtype falls through to
// uint16 as the fp16 default.
void prep_input(const std::shared_ptr<Runtime> &rt,
                const std::shared_ptr<vkop::VulkanCommandPool> &cmdpool,
                const std::string &name, const Options &o, uint32_t salt) {
    auto t = rt->GetInput(name);
    if (!t) {
        std::fprintf(stderr, "no such input: %s\n", name.c_str());
        return;
    }
    auto it = o.shapes.find(name);
    if (it != o.shapes.end()) {
        rt->ResizeInput(name, it->second);
    }
    // Seed salted by a per-input hash so different inputs of the same
    // size don't get identical patterns (which would let the GPU's L2
    // cache hit harder than reality and skew the number).
    uint32_t seed = o.seed;
    for (char c : name)
        seed = seed * 33u + static_cast<uint32_t>(c);
    seed ^= salt;

    const int n = static_cast<int>(element_count(t->getShape()));
    if (n <= 0) {
        return;
    }
    const auto &dt = t->dtype();
    if (dt == typeid(float)) {
        auto tt = as_tensor<float>(t);
        auto buf = make_fill<float>(n, seed);
        tt->fillToCPU(buf);
        tt->copyToGPU(cmdpool);
    } else if (dt == typeid(int64_t)) {
        auto tt = as_tensor<int64_t>(t);
        auto buf = make_fill<int64_t>(n, seed);
        tt->fillToCPU(buf);
        tt->copyToGPU(cmdpool);
    } else if (dt == typeid(int)) {
        auto tt = as_tensor<int>(t);
        auto buf = make_fill<int>(n, seed);
        tt->fillToCPU(buf);
        tt->copyToGPU(cmdpool);
    } else if (dt == typeid(int8_t)) {
        auto tt = as_tensor<int8_t>(t);
        auto buf = make_fill<int8_t>(n, seed);
        tt->fillToCPU(buf);
        tt->copyToGPU(cmdpool);
    } else {
        // uint16_t is vkop's fp16 storage; anything else lands here too.
        auto tt = as_tensor<uint16_t>(t);
        auto buf = make_fill<uint16_t>(n, seed);
        tt->fillToCPU(buf);
        tt->copyToGPU(cmdpool);
    }
}

struct RunStats {
    double min_ms, max_ms, avg_ms, p50_ms, p90_ms, p99_ms;
    int iters;
};

RunStats summarize(std::vector<double> samples) {
    std::sort(samples.begin(), samples.end());
    auto pick = [&](double q) {
        if (samples.empty())
            return 0.0;
        size_t idx = static_cast<size_t>(q * (samples.size() - 1) + 0.5);
        if (idx >= samples.size())
            idx = samples.size() - 1;
        return samples[idx];
    };
    double sum = std::accumulate(samples.begin(), samples.end(), 0.0);
    return {samples.front(),
            samples.back(),
            sum / samples.size(),
            pick(0.50),
            pick(0.90),
            pick(0.99),
            static_cast<int>(samples.size())};
}

} // namespace

int main(int argc, char **argv) {
    Options o;
    if (!parse_args(argc, argv, o))
        return 2;

    Logger::getInstance().setLevel(LOG_INFO);
    const auto &phydevs = VulkanInstance::getVulkanInstance().getPhysicalDevices();
    if (phydevs.empty()) {
        std::fprintf(stderr, "no vulkan device\n");
        return 1;
    }
    auto dev = std::make_shared<VulkanDevice>(phydevs[0]);
    auto cmdpool = std::make_shared<vkop::VulkanCommandPool>(dev);

    auto t_load_0 = std::chrono::steady_clock::now();
    auto rt = std::make_shared<Runtime>(cmdpool, o.model_path, o.precision);
    rt->set_backend_buffer(o.buffer_backend);
    rt->LoadModel();
    auto t_load_1 = std::chrono::steady_clock::now();
    double load_ms = std::chrono::duration<double, std::milli>(t_load_1 - t_load_0).count();

    std::printf("=== vkop-bench ===\n");
    std::printf("model:    %s\n", o.model_path.c_str());
    std::printf("gpu:      %s\n", dev->getDeviceName().c_str());
    std::printf("backend:  %s\n", o.buffer_backend ? "buffer" : "image");
    std::printf("prec:     %s\n", o.precision ? "fp16" : "fp32");
    std::printf("load:     %.1f ms\n", load_ms);
    std::printf("pipeline-cache: %s\n",
                std::getenv("VK_PIPELINE_CACHE") ? "on" : "off");
    std::printf("shader-archive: %s\n",
                std::getenv("VKOP_SHADER_ARCHIVE") ? std::getenv("VKOP_SHADER_ARCHIVE") : "embedded");

    auto names = rt->ListInputNames();
    if (o.list_inputs_only) {
        std::printf("\ninputs (%zu):\n", names.size());
        for (const auto &name : names) {
            auto t = rt->GetInput(name);
            auto sh = t->getShape();
            std::printf("  %-40s n=%lld shape=[", name.c_str(),
                        (long long)element_count(sh));
            for (size_t i = 0; i < sh.size(); ++i)
                std::printf("%s%d", i ? "," : "", sh[i]);
            std::printf("]\n");
        }
        return 0;
    }
    if (o.time_load_only) {
        return 0;
    }

    // Pre-fill every input once. ResizeInput recreates the SSBO/image
    // backing, so a shape not declared via --shape falls back to
    // LoadModel's placeholder [1, 1, ...] -- the model still runs, just
    // on a tiny tensor, which is not what you want but never crashes.
    for (size_t i = 0; i < names.size(); ++i) {
        prep_input(rt, cmdpool, names[i], o, static_cast<uint32_t>(i));
    }
    dev->wait_all_done();

    for (int w = 0; w < o.warmup; ++w) {
        rt->Run();
    }
    dev->wait_all_done();

    std::vector<double> samples;
    samples.reserve(o.iters);
    for (int i = 0; i < o.iters; ++i) {
        auto t0 = std::chrono::steady_clock::now();
        rt->Run();
        auto t1 = std::chrono::steady_clock::now();
        samples.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    }
    dev->wait_all_done();

    auto s = summarize(samples);
    std::printf("\nrun (%d iters after %d warmup):\n", o.iters, o.warmup);
    std::printf("  min  %8.3f ms\n", s.min_ms);
    std::printf("  p50  %8.3f ms\n", s.p50_ms);
    std::printf("  p90  %8.3f ms\n", s.p90_ms);
    std::printf("  p99  %8.3f ms\n", s.p99_ms);
    std::printf("  max  %8.3f ms\n", s.max_ms);
    std::printf("  avg  %8.3f ms\n", s.avg_ms);
    std::printf("  throughput %8.2f runs/s\n", 1000.0 / s.avg_ms);
    return 0;
}
