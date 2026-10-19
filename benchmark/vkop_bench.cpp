// junka @ 2026
// vkop-bench -- trtexec-style generic benchmark + verify driver for any
// .vkopbin.
//
// Loads a model, sizes its graph inputs to something the runtime can
// dispatch on, warms up, then times N iterations of Runtime::Run() and
// prints a percentile table. Inputs default to deterministic pseudo-random
// fill (fast path: no data on disk needed); --load-inputs points at a
// directory of .npy files to run against real values instead, and
// --save-outputs dumps every output tensor to a matching directory so a
// numpy script can diff against an ORT reference without going through
// llm_driver's fixed 35-input LLM contract.
//
// The .npy reader is minimal (little-endian, C-contiguous, v1.0 header),
// the same subset llm_driver reads. Non-contiguous or big-endian files
// raise with a clear message rather than silently reading shuffled bytes.
//
// Usage:
//   vkop-bench model.vkopbin
//       [--iters N] [--warmup N] [--precision fp16|fp32]
//       [--backend buffer|image]
//       [--shape name:d1xd2x...]      (may repeat; required for dynamic dims)
//       [--seed N]
//       [--load-inputs DIR]           (per-input <DIR>/<name>.npy, random for misses)
//       [--save-outputs DIR]          (raw bytes to <DIR>/<name>.bin after the last run)
//
// Dynamic-shape inputs: LoadModel turns every -1 sentinel into a
// placeholder dim of 1 (see core/runtime.cpp:186ff), so a raw run without
// --shape would execute a [1,1,...] kernel. --shape rebinds the tensor
// via ResizeInput to a concrete dim. With --load-inputs the npy's own
// shape wins over --shape (the file records its dims).
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
#include <exception>
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
    // Real input tensors read from <load_inputs_dir>/<name>.npy; anything
    // missing falls back to the pseudo-random fill. Same path convention
    // llm_driver uses, so the same dump_dir works in either tool.
    std::string load_inputs_dir;
    // After the last iteration, dump every graph output tensor as raw
    // bytes to <save_outputs_dir>/<name>.bin. Shape and dtype are known
    // from the vkopbin; a numpy reader can reconstruct via
    // np.frombuffer(...).reshape(shape).astype(dtype).
    std::string save_outputs_dir;
    // JSON dump of per-node GPU timings from the last Run(), written as JSON. Enables
    // the runtime's submit-side timestamp query pool (same as
    // VKOP_SUBMIT_PROF=1) before warmup so the last run is populated.
    std::string profile_out_path;
    // Physical-device selector. VulkanInstance sorts discrete > integrated >
    // virtual > CPU, so --device 0 is "the fastest real GPU present"; --device
    // 1 hits the integrated iGPU on a laptop with both. --list-devices
    // enumerates in that same order without loading any model.
    int device_index = 0;
    bool list_devices_only = false;
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
                 "  --load-inputs DIR     fill inputs from DIR/<name>.npy; misses fall back to random\n"
                 "  --save-outputs DIR    write each output to DIR/<name>.bin after the last run\n"
                 "  --profile-out FILE    dump per-node GPU timings as JSON (enables submit profiling)\n"
                 "  --device N            pick physical device N (sorted discrete>iGPU>cpu; default 0)\n"
                 "  --list-devices        enumerate physical devices and exit\n"
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
    // --list-devices does not need a model; accept it as argv[1] and start
    // option parsing at argv[2] like normal, but skip the "no model path"
    // check below.
    int start = 1;
    if (first == "--list-devices") {
        o.list_devices_only = true;
    } else {
        o.model_path = first;
        start = 2;
    }
    for (int i = start; i < argc; ++i) {
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
        } else if (a == "--load-inputs") {
            o.load_inputs_dir = need_next("--load-inputs");
        } else if (a == "--save-outputs") {
            o.save_outputs_dir = need_next("--save-outputs");
        } else if (a == "--profile-out") {
            o.profile_out_path = need_next("--profile-out");
        } else if (a == "--device") {
            o.device_index = std::atoi(need_next("--device"));
        } else if (a == "--list-devices") {
            o.list_devices_only = true;
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

size_t npy_count(const std::vector<uint32_t> &shape);
void upload_by_dtype(const std::shared_ptr<vkop::VulkanCommandPool> &cmdpool,
                     const std::shared_ptr<ITensor> &t);
void prep_input(const std::shared_ptr<Runtime> &rt,
                const std::shared_ptr<vkop::VulkanCommandPool> &cmdpool,
                const std::string &name, const Options &o, uint32_t salt);

// ---- NumPy .npy reader (little-endian, C-contiguous, v1.0 header) ----
// Same subset llm_driver consumes. Version 2.0 headers (uint32 header
// length) and big-endian / fortran-order files are refused loudly rather
// than silently mis-read.
struct NpyArray {
    std::vector<uint32_t> shape;
    std::string descr; // "<f2", "<i8", "<f4", "|i1", "<u2", ...
    std::vector<uint8_t> bytes;
};

NpyArray load_npy(const std::string &path) {
    std::ifstream f(path, std::ios::binary);
    if (!f)
        throw std::runtime_error("cannot open " + path);
    std::vector<uint8_t> buf((std::istreambuf_iterator<char>(f)),
                             std::istreambuf_iterator<char>());
    if (buf.size() < 10 || buf[0] != 0x93 ||
        std::memcmp(buf.data() + 1, "NUMPY", 5) != 0)
        throw std::runtime_error("bad npy magic in " + path);
    if (buf[6] != 1)
        throw std::runtime_error("npy v" + std::to_string(buf[6]) +
                                 " not supported (only 1.x), in " + path);
    uint16_t hdr_len = static_cast<uint16_t>(buf[8]) |
                       (static_cast<uint16_t>(buf[9]) << 8);
    size_t hdr_start = 10;
    size_t data_start = hdr_start + hdr_len;
    if (data_start > buf.size())
        throw std::runtime_error("truncated npy header in " + path);
    std::string header(buf.begin() + hdr_start, buf.begin() + data_start);
    if (header.find("'fortran_order': True") != std::string::npos)
        throw std::runtime_error(
            "fortran-order npy not supported in " + path);

    NpyArray arr;
    auto find_str = [&](const char *key) -> std::string {
        auto kp = header.find(key);
        if (kp == std::string::npos)
            return "";
        auto colon = header.find(':', kp);
        auto q1 = header.find('\'', colon);
        auto q2 = header.find('\'', q1 + 1);
        if (q1 == std::string::npos || q2 == std::string::npos)
            return "";
        return header.substr(q1 + 1, q2 - q1 - 1);
    };
    arr.descr = find_str("descr");
    auto sb = header.find('(');
    auto se = header.find(')', sb);
    if (sb != std::string::npos && se != std::string::npos) {
        std::stringstream ss(header.substr(sb + 1, se - sb - 1));
        std::string tok;
        while (std::getline(ss, tok, ',')) {
            size_t a = tok.find_first_not_of(" \t");
            if (a == std::string::npos)
                continue;
            arr.shape.push_back(
                static_cast<uint32_t>(std::stoul(tok.substr(a))));
        }
    }
    arr.bytes.assign(buf.begin() + data_start, buf.end());
    return arr;
}

// Fill one input from an NpyArray. Resizes the tensor to the npy's shape
// so an input whose .vkopbin recorded a dynamic dim still lands at the
// concrete size from disk. dtype dispatch mirrors llm_driver's fill_input;
// the npy descr is used only to sanity-check we aren't silently reinterpreting.
template <typename T>
void fill_typed_from_npy(const std::shared_ptr<ITensor> &t, const NpyArray &a,
                         const std::string &name) {
    if (a.bytes.size() < sizeof(T) * npy_count(a.shape)) {
        throw std::runtime_error("npy " + name + " short read: file " +
                                 std::to_string(a.bytes.size()) +
                                 " bytes, need more");
    }
    auto typed = as_tensor<T>(t);
    typed->resize(a.shape);
    size_t n = npy_count(a.shape);
    if (n > 0) {
        typed->fillToCPU(reinterpret_cast<const T *>(a.bytes.data()));
    }
}

size_t npy_count(const std::vector<uint32_t> &shape) {
    size_t n = 1;
    for (auto d : shape)
        n *= d;
    return n;
}

// Resize + fill + upload one input from an .npy if load_inputs_dir has it,
// else fall through to the deterministic pseudo-random path.
void prep_input_from_dir_or_random(const std::shared_ptr<Runtime> &rt,
                                   const std::shared_ptr<vkop::VulkanCommandPool> &cmdpool,
                                   const std::string &name, const Options &o,
                                   uint32_t salt, bool &served_from_npy) {
    served_from_npy = false;
    if (!o.load_inputs_dir.empty()) {
        std::string path = o.load_inputs_dir + "/" + name + ".npy";
        std::ifstream probe(path);
        if (probe) {
            probe.close();
            NpyArray arr = load_npy(path);
            auto t = rt->GetInput(name);
            if (!t)
                throw std::runtime_error("no such input: " + name);
            rt->ResizeInput(name, arr.shape);
            t = rt->GetInput(name);
            const auto &dt = t->dtype();
            if (dt == typeid(int64_t))
                fill_typed_from_npy<int64_t>(t, arr, name);
            else if (dt == typeid(int))
                fill_typed_from_npy<int>(t, arr, name);
            else if (dt == typeid(int8_t))
                fill_typed_from_npy<int8_t>(t, arr, name);
            else if (dt == typeid(float))
                fill_typed_from_npy<float>(t, arr, name);
            else
                fill_typed_from_npy<uint16_t>(t, arr, name);
            upload_by_dtype(cmdpool, t);
            served_from_npy = true;
            return;
        }
    }
    prep_input(rt, cmdpool, name, o, salt);
}

void upload_by_dtype(const std::shared_ptr<vkop::VulkanCommandPool> &cmdpool,
                     const std::shared_ptr<ITensor> &t) {
    if (t->size() == 0)
        return;
    const auto &dt = t->dtype();
    if (dt == typeid(int64_t))
        as_tensor<int64_t>(t)->copyToGPU(cmdpool);
    else if (dt == typeid(int))
        as_tensor<int>(t)->copyToGPU(cmdpool);
    else if (dt == typeid(int8_t))
        as_tensor<int8_t>(t)->copyToGPU(cmdpool);
    else if (dt == typeid(float))
        as_tensor<float>(t)->copyToGPU(cmdpool);
    else
        as_tensor<uint16_t>(t)->copyToGPU(cmdpool);
}

// Write every graph output's bytes to <dir>/<name>.bin. Shape and dtype
// stay implicit (from the .vkopbin + inspect); a consumer reconstructs
// via np.frombuffer(open(...,'rb').read(), dtype).reshape(shape).
void save_outputs(const std::shared_ptr<Runtime> &rt,
                  const std::shared_ptr<vkop::VulkanCommandPool> &cmdpool,
                  const std::string &dir) {
    auto names = rt->ListOutputNames();
    for (const auto &name : names) {
        auto t = rt->GetOutput(name);
        if (!t) {
            std::fprintf(stderr, "save: no output %s\n", name.c_str());
            continue;
        }
        const auto &dt = t->dtype();
        std::string path = dir + "/" + name + ".bin";
        std::ofstream f(path, std::ios::binary);
        if (!f) {
            std::fprintf(stderr, "save: cannot write %s\n", path.c_str());
            continue;
        }
        // Read GPU buffer back to CPU before writing.
        if (dt == typeid(int64_t)) {
            auto tt = as_tensor<int64_t>(t);
            tt->copyToCPU(cmdpool);
            const auto &d = tt->data();
            f.write(reinterpret_cast<const char *>(d.data()),
                    d.size() * sizeof(int64_t));
        } else if (dt == typeid(int)) {
            auto tt = as_tensor<int>(t);
            tt->copyToCPU(cmdpool);
            const auto &d = tt->data();
            f.write(reinterpret_cast<const char *>(d.data()),
                    d.size() * sizeof(int));
        } else if (dt == typeid(int8_t)) {
            auto tt = as_tensor<int8_t>(t);
            tt->copyToCPU(cmdpool);
            const auto &d = tt->data();
            f.write(reinterpret_cast<const char *>(d.data()),
                    d.size() * sizeof(int8_t));
        } else if (dt == typeid(float)) {
            auto tt = as_tensor<float>(t);
            tt->copyToCPU(cmdpool);
            const auto &d = tt->data();
            f.write(reinterpret_cast<const char *>(d.data()),
                    d.size() * sizeof(float));
        } else {
            auto tt = as_tensor<uint16_t>(t);
            tt->copyToCPU(cmdpool);
            const auto &d = tt->data();
            f.write(reinterpret_cast<const char *>(d.data()),
                    d.size() * sizeof(uint16_t));
        }
        std::printf("  saved %s -> %s\n", name.c_str(), path.c_str());
    }
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
    // Validate --device before the --list-devices early return: `vkop-bench
    // --list-devices --device 99` used to print the list and exit 0, silently
    // ignoring an index that would have been rejected on any other path.
    if (o.device_index < 0 ||
        static_cast<size_t>(o.device_index) >= phydevs.size()) {
        std::fprintf(stderr,
                     "--device %d out of range (0..%zu); use --list-devices\n",
                     o.device_index, phydevs.size() - 1);
        return 2;
    }
    if (o.list_devices_only) {
        std::printf("%d physical device(s), in vkop's preferred order:\n",
                    static_cast<int>(phydevs.size()));
        for (size_t i = 0; i < phydevs.size(); ++i) {
            VkPhysicalDeviceProperties props;
            // VulkanLib.hpp installs per-TU static function pointers inside
            // namespace vkop (the loader is dlopen-based, so the global
            // vkGetPhysicalDeviceProperties symbol is deliberately not
            // linked). Reach the vkop-scoped alias for the same entry point.
            vkop::vkGetPhysicalDeviceProperties(phydevs[i], &props);
            const char *type = "UNKNOWN";
            switch (props.deviceType) {
            case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU:
                type = "DISCRETE";
                break;
            case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU:
                type = "INTEGRATED";
                break;
            case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU:
                type = "VIRTUAL";
                break;
            case VK_PHYSICAL_DEVICE_TYPE_CPU:
                type = "CPU";
                break;
            default:
                break;
            }
            std::printf("  [%zu] %-11s %s  (vendor=0x%04x device=0x%04x)\n", i,
                        type, props.deviceName, props.vendorID, props.deviceID);
        }
        return 0;
    }
    if (o.model_path.empty()) {
        std::fprintf(stderr, "no model path (also accepts --list-devices)\n");
        return 2;
    }
    auto dev = std::make_shared<VulkanDevice>(phydevs[o.device_index]);
    auto cmdpool = std::make_shared<vkop::VulkanCommandPool>(dev);

    auto t_load_0 = std::chrono::steady_clock::now();
    auto rt = std::make_shared<Runtime>(cmdpool, o.model_path, o.precision);
    rt->set_backend_buffer(o.buffer_backend);
    // A .vkopbin records no backend flag that predicts this failure: a graph
    // converted for the other backend throws where a tensor is bound for the
    // layout the runtime chose (as_input_image() in core/Tensor.hpp), so it
    // can surface in LoadModel, in the input pre-fill, or in Run().
    // `vkop-bench model/onnx_models/resnet18.vkopbin` on the default buffer
    // backend used to abort uncaught with "Expected VulkanImage but got
    // something else." (rc=134), which reads as a runtime bug rather than a
    // flag mismatch. Name the phase, the model and the switch to flip.
    auto backend_mismatch = [&](const char *phase, const std::exception &e) {
        std::fprintf(stderr,
                    "vkop-bench: %s(%s) failed: %s\n"
                    "           if this vkopbin was converted for the other "
                    "backend, pass --backend %s\n",
                    phase, o.model_path.c_str(), e.what(),
                    o.buffer_backend ? "image" : "buffer");
    };
    try {
        rt->LoadModel();
    } catch (const std::exception &e) {
        backend_mismatch("LoadModel", e);
        return 2;
    }
    auto t_load_1 = std::chrono::steady_clock::now();
    double load_ms = std::chrono::duration<double, std::milli>(t_load_1 - t_load_0).count();

    // VKOP_SHAPE_DEBUG=1 asks the view ops how they did: `applied` is a pure
    // metadata reshape, `rejected` is the case that today silently keeps the
    // old rank (the reason a Conv2dImage input can arrive with rank 0), and
    // `buffer_aliases` counts outputs that share the producer's VkBuffer
    // instead of round-tripping bytes. Scope guard so both the --time-load-only
    // exit and the normal one report.
    struct ShapeViewReport {
        ~ShapeViewReport() {
            if (!vkop::core::ITensor::shape_debug_enabled()) {
                return;
            }
            const auto &s = vkop::core::ITensor::shape_view_stats();
            std::printf("shape-view: applied=%llu rejected=%llu buffer_aliases=%llu\n",
                        static_cast<unsigned long long>(s.views_applied),
                        static_cast<unsigned long long>(s.views_rejected),
                        static_cast<unsigned long long>(s.buffer_aliases));
        }
    } shape_view_report;

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
    // With --load-inputs, each tensor is filled from DIR/<name>.npy when
    // the file exists (its shape wins over --shape); otherwise the random
    // fill path is taken so a partial dump still runs.
    //
    // Wall time and byte total for this phase become h2d_ms / h2d_bytes
    // in the profile JSON. This is the transfer cost the "steady-state
    // decode 110ms" number elides: every input's first upload is here,
    // not inside Run(). For a llm.vkopbin decode bench at past=512 the
    // KV caches alone are ~28 MB of H2D before the first iteration.
    int from_npy = 0;
    double h2d_ms = 0.0;
    uint64_t h2d_bytes = 0;
    auto t_h2d_0 = std::chrono::steady_clock::now();
    try {
        for (size_t i = 0; i < names.size(); ++i) {
            bool served = false;
            prep_input_from_dir_or_random(rt, cmdpool, names[i], o,
                                          static_cast<uint32_t>(i), served);
            if (served)
                ++from_npy;
            auto t = rt->GetInput(names[i]);
            if (t)
                h2d_bytes += static_cast<uint64_t>(t->size());
        }
    } catch (const std::exception &e) {
        backend_mismatch("input pre-fill", e);
        return 2;
    }
    dev->wait_all_done();
    {
        auto t_h2d_1 = std::chrono::steady_clock::now();
        h2d_ms = std::chrono::duration<double, std::milli>(t_h2d_1 - t_h2d_0).count();
    }
    if (!o.load_inputs_dir.empty()) {
        std::printf("inputs: %d from npy, %d random\n", from_npy,
                    static_cast<int>(names.size()) - from_npy);
    }
    // Enabling profiling has to happen BEFORE the first Run(), where the
    // query pool is allocated. Post-warmup would leave the pool null for
    // every measurement iteration.
    if (!o.profile_out_path.empty()) {
        rt->enable_submit_profiling();
    }
    dev->wait_all_done();

    std::vector<double> samples;
    samples.reserve(o.iters);
    try {
        for (int w = 0; w < o.warmup; ++w) {
            rt->Run();
        }
        dev->wait_all_done();

        for (int i = 0; i < o.iters; ++i) {
            auto t0 = std::chrono::steady_clock::now();
            rt->Run();
            auto t1 = std::chrono::steady_clock::now();
            samples.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
        }
    } catch (const std::exception &e) {
        backend_mismatch("Run", e);
        return 2;
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

    // Outputs land after the LAST run; a --load-inputs verification dump
    // matches whatever the final iteration produced. If the caller only
    // wants a correctness dump (no timing), --iters 1 --warmup 0 gives
    // exactly one Run() before save. Wall time and byte total become
    // d2h_ms / d2h_bytes in the profile JSON. Done before the transfer
    // print so D2H sits next to H2D in both stdout and the JSON.
    double d2h_ms = 0.0;
    uint64_t d2h_bytes = 0;
    if (!o.save_outputs_dir.empty()) {
        std::printf("\nsaving outputs to %s\n", o.save_outputs_dir.c_str());
        auto t_d2h_0 = std::chrono::steady_clock::now();
        for (const auto &n : rt->ListOutputNames()) {
            auto t = rt->GetOutput(n);
            if (t)
                d2h_bytes += static_cast<uint64_t>(t->size());
        }
        save_outputs(rt, cmdpool, o.save_outputs_dir);
        dev->wait_all_done();
        auto t_d2h_1 = std::chrono::steady_clock::now();
        d2h_ms = std::chrono::duration<double, std::milli>(t_d2h_1 - t_d2h_0).count();
    }

    std::printf("\ntransfers:\n");
    std::printf("  h2d  %8.3f ms  %12llu bytes  %8.2f MB/s\n", h2d_ms,
                (unsigned long long)h2d_bytes,
                h2d_ms > 0 ? h2d_bytes / (h2d_ms * 1e3) : 0.0);
    if (d2h_ms > 0 || d2h_bytes > 0) {
        std::printf("  d2h  %8.3f ms  %12llu bytes  %8.2f MB/s\n", d2h_ms,
                    (unsigned long long)d2h_bytes,
                    d2h_ms > 0 ? d2h_bytes / (d2h_ms * 1e3) : 0.0);
    }

    // JSON dump of per-node GPU timings from the LAST profiled Run(). Same
    // data the runtime prints to stderr under VKOP_SUBMIT_PROF, but as a
    // machine-readable form so --profile-out runs can be diffed against
    // baseline runs, sorted by op, or folded into a flamegraph.
    if (!o.profile_out_path.empty()) {
        auto &timings = rt->last_run_timings();
        std::ofstream pf(o.profile_out_path);
        if (!pf) {
            std::fprintf(stderr, "profile: cannot write %s\n",
                         o.profile_out_path.c_str());
        } else {
            double total = 0.0;
            for (const auto &t : timings)
                total += t.ms;
            auto esc = [](const std::string &s) {
                std::string out;
                out.reserve(s.size());
                for (char c : s) {
                    if (c == '"' || c == '\\')
                        out.push_back('\\');
                    out.push_back(c);
                }
                return out;
            };
            pf << "{\n";
            pf << "  \"timestamp_period_ns\": " << rt->timestamp_period_ns()
               << ",\n";
            pf << "  \"total_gpu_ms\": " << total << ",\n";
            pf << "  \"device_index\": " << o.device_index << ",\n";
            pf << "  \"h2d_ms\": " << h2d_ms << ",\n";
            pf << "  \"h2d_bytes\": " << h2d_bytes << ",\n";
            pf << "  \"d2h_ms\": " << d2h_ms << ",\n";
            pf << "  \"d2h_bytes\": " << d2h_bytes << ",\n";
            pf << "  \"node_count\": " << timings.size() << ",\n";
            pf << "  \"nodes\": [\n";
            for (size_t i = 0; i < timings.size(); ++i) {
                const auto &t = timings[i];
                pf << "    {\"op_type\": \"" << esc(t.op_type) << "\", "
                   << "\"name\": \"" << esc(t.name) << "\", "
                   << "\"level\": " << t.level << ", "
                   << "\"ms\": " << t.ms << "}";
                if (i + 1 < timings.size())
                    pf << ",";
                pf << "\n";
            }
            pf << "  ]\n";
            pf << "}\n";
            std::printf("\nprofile: wrote %zu nodes to %s (total %.3f ms)\n",
                        timings.size(), o.profile_out_path.c_str(), total);
        }
    }
    return 0;
}
