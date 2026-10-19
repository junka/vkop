// junka @ 2025
// End-to-end driver for llm.vkopbin (Qwen3-VL-2B LLM, buffer backend).
//
// Loads the 3649-node model, fills the 35 prefill inputs from .npy files
// dumped by llm/exporter/export/dump_llm_inputs.py, runs prefill, and compares the
// logits output against reference_logits.npy (ONNX Runtime).
//
// Usage:
//   llm_driver <model.vkopbin> <inputs_dump_dir>
//
// Build (from repo root):
//   g++ -std=c++17 -I. -Imodel/generated llm/exporter/driver/llm_driver.cpp
//       build/libvkop.a build/model/libvload.a -o build/llm_driver
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
#include "kv_cache.hpp"

using vkop::VulkanInstance;
using vkop::VulkanDevice;
using vkop::core::ITensor;
using vkop::core::Runtime;
using vkop::core::as_tensor;
using vkop::export_::KVCache;

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
//
// `skip_kv` drops the past_key_values_* inputs: in the cache form the caller
// stages those itself, at the cache's own row pitch (see stage_cache_history).
void load_round_inputs(const std::shared_ptr<Runtime>& rt,
                       const std::shared_ptr<vkop::VulkanCommandPool>& cmdpool,
                       const std::string& round_dir,
                       std::vector<std::string>* input_names_out,
                       bool skip_kv = false) {
    std::ifstream nf(round_dir + "/input_names.txt");
    if (!nf) throw std::runtime_error("no input_names.txt in " + round_dir);
    std::vector<std::string> names;
    std::string line;
    while (std::getline(nf, line)) {
        if (!line.empty()) names.push_back(line);
    }
    for (const auto& name : names) {
        if (skip_kv && name.rfind("past_key_values_", 0) == 0) continue;
        NpyArray arr = load_npy(round_dir + "/" + name + ".npy");
        fill_input(rt, name, arr);
        upload_input(cmdpool, rt->GetInput(name));
    }
    if (input_names_out) *input_names_out = names;
}

// Does the converted model carry the fused KV-cache form (i.e. each layer's
// present_key_values_i is written by the attention op itself, aliasing the
// preallocated past, rather than by the graph's Gather/Concat/Unsqueeze
// rotation)?
//
// Asking the loaded graph would need a runtime accessor for "is there an
// Attention node with has_cache set", which does not exist, so read it off the
// file instead: the converter emits the `has_cache` attribute only when
// fuse_kv_cache_append fired, and a vkopbin serialises the whole graph — node
// names and attributes — ahead of the weight blob. Verified on an 814 MB model:
// all node metadata lands in the first megabyte, so an 8 MB header read is both
// sufficient and cheap. (The roadmap's own check for a stale binary is the same
// `grep has_cache`.)
//
// The obvious-looking alternative does NOT work: present_key_values_0 is a graph
// output in *both* forms, since the rotation's Unsqueeze/Concat produces it too,
// so keying on that flags every model.
bool model_has_cache_form(const std::string& model_path) {
    std::ifstream f(model_path, std::ios::binary);
    if (!f) return false;
    constexpr std::size_t kScan = 8u << 20;
    std::vector<char> buf(kScan);
    f.read(buf.data(), static_cast<std::streamsize>(kScan));
    const std::size_t n = static_cast<std::size_t>(f.gcount());
    const std::string needle = "has_cache";
    return std::search(buf.begin(), buf.begin() + n, needle.begin(),
                       needle.end()) != buf.begin() + n;
}

// Stage one layer's history into its preallocated cache buffer.
//
// The reference bundle hands us the history the way the rotation-form graph
// wants it: <round_dir>/past_key_values_<layer>.npy, shape [1,2,nkv,kv_len,hd],
// i.e. rows packed at pitch kv_len. The cache is [1,2,nkv,cap,hd] and the
// kernel reads it at pitch `cap` (AttentionBuffer::row_capacity takes the row
// pitch from the bound buffer's capacity, never from the tensor's logical
// kv_len) — so the rows have to be laid out at stride cap. Within one (c,h)
// plane both layouts are contiguous in j, so each plane is one memcpy; only the
// plane stride differs. Getting this wrong does not crash, it reads the wrong
// K/V rows, which is exactly what this driver exists to catch.
//
// The staging tensor is resized to the cap-shaped logical length so its CPU
// staging is cap-sized (fillToCPU copies num_elements() worth), uploaded, and
// then handed back to the true kv_len by the caller via KVCache::resume_from.
// Returns the history length.
int stage_cache_history(const std::shared_ptr<Runtime>& rt,
                        const std::shared_ptr<vkop::VulkanCommandPool>& cmdpool,
                        const std::string& round_dir, int layer, int cap) {
    const std::string name = "past_key_values_" + std::to_string(layer);
    NpyArray arr = load_npy(round_dir + "/" + name + ".npy");
    if (arr.shape.size() != 5) {
        throw std::runtime_error(name + ": expected a rank-5 [1,2,nkv,kv,hd] npy");
    }
    const int nkv = static_cast<int>(arr.shape[2]);
    const int kv_len = static_cast<int>(arr.shape[3]);
    const int hd = static_cast<int>(arr.shape[4]);
    if (kv_len > cap) {
        throw std::runtime_error(name + ": kv_len " + std::to_string(kv_len) +
                                 " exceeds the cache cap " + std::to_string(cap) +
                                 " (raise VKOP_KV_MAX)");
    }
    const uint16_t* src = reinterpret_cast<const uint16_t*>(arr.data.data());
    // Zero the staging so the untouched tail of the cap-sized buffer is
    // deterministic; the kernel never reads rows >= kv_len.
    std::vector<uint16_t> stage(static_cast<size_t>(2) * nkv * cap * hd, 0);
    // Index the K and V halves of every head as one flat run of 2*nkv blocks.
    // These flatten the same way in both layouts — (c,h) then (j,d) — so the
    // only difference is the block stride, cap*hd against kv_len*hd. Each block
    // is kv_len rows of hd, contiguous in both, so it is a single memcpy.
    const size_t src_stride = static_cast<size_t>(kv_len) * hd;
    const size_t dst_stride = static_cast<size_t>(cap) * hd;
    const size_t block = src_stride; // kv_len * hd elements
    for (int blk = 0; blk < 2 * nkv; ++blk) {
        std::memcpy(stage.data() + blk * dst_stride, src + blk * src_stride,
                    block * sizeof(uint16_t));
    }
    // Row-major [cap][hd] == [kv_len][hd] then a tail: the block's own memory
    // order does not change, only where the next block starts. That is what
    // makes the layout a relayout and not a transpose.
    rt->ResizeInput(name, {1u, 2u, static_cast<uint32_t>(nkv),
                           static_cast<uint32_t>(cap), static_cast<uint32_t>(hd)});
    auto typed = as_tensor<uint16_t>(rt->GetInput(name));
    typed->fillToCPU(stage.data());
    typed->copyToGPU(cmdpool);
    return kv_len;
}

// Compare the runtime's 'logits' output against <round_dir>/reference_logits.npy.
// Returns the vkop argmax (== ref argmax on success). Prints top8 + diffs.

// The window of the reference array to compare against, as (offset, count).
//
// A model converted with VKOP_LOGITS_LAST_ROW=<logits tensor> computes only
// the LM head's last row — row L-1 is all a greedy loop reads, and the other
// L-1 rows are a 1.13 GB buffer at L=1860 plus 14% of the prefill. The
// reference npy is still the full [1,L,vocab] ORT produced from the unshrunk
// graph, so the comparison has to shrink to its last row too; otherwise this
// driver's size check rejects exactly the graph the flag exists to produce.
//
// Same env name as the conversion flag, but only its presence matters here:
// which tensors to shrink was decided when the .vkopbin was written.
bool logits_last_row_only() {
    const char *v = std::getenv("VKOP_LOGITS_LAST_ROW");
    return v != nullptr && *v != '\0' && std::strcmp(v, "0") != 0;
}

void ref_window(const NpyArray &ref, size_t &offset, size_t &count) {
    count = npy_elem_count(ref.shape);
    offset = 0;
    if (logits_last_row_only() && ref.shape.size() >= 2) {
        count = ref.shape.back();
        offset = npy_elem_count(ref.shape) - count;
    }
}

int compare_round(const std::shared_ptr<Runtime>& rt,
                  const std::shared_ptr<vkop::VulkanCommandPool>& cmdpool,
                  const std::string& round_dir) {
    NpyArray ref = load_npy(round_dir + "/reference_logits.npy");
    auto logits = rt->GetOutput("logits");
    if (!logits) throw std::runtime_error("no 'logits' output");
    auto lg = as_tensor<uint16_t>(logits);
    lg->copyToCPU(cmdpool);
    const uint16_t* out_bits = reinterpret_cast<const uint16_t*>(lg->data().data());
    size_t ref_off = 0, n = 0;
    ref_window(ref, ref_off, n);
    if (size_t(lg->num_elements()) != n) {
        std::fprintf(stderr, "[%s] logits size mismatch out=%d ref=%zu\n",
                     round_dir.c_str(), lg->num_elements(), n);
        return -1;
    }
    const uint16_t* ref_bits =
        reinterpret_cast<const uint16_t*>(ref.data.data()) + ref_off;
    int argmax_out = -1, argmax_ref = -1;
    float maxv_out = -1e30f, maxv_ref = -1e30f;
    double sum_abs_diff = 0;
    float max_abs_diff = 0;
    std::vector<std::pair<float,int>> vk, rf;
    vk.reserve(n); rf.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        float fo = ITensor::fp16_to_fp32(out_bits[i]);
        float fr = ITensor::fp16_to_fp32(ref_bits[i]);
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

    // ---- KV-cache form ----------------------------------------------------
    // The converted model may carry the fused KV-cache form (present_key_values_i
    // graph outputs aliasing a preallocated past). llm_driver's normal path
    // cannot exercise that: it resizes past_key_values_i to the round's own
    // kv_len, so the buffer pitch equals the logical length and the kernel's
    // src_stride/dst_stride are degenerate. VKOP_KV_CACHE=1 preallocates each
    // layer's cache at VKOP_KV_MAX (default 8192) instead and stages the history
    // at that pitch — the cap > kv_len case, on the real 28-layer model against
    // the real reference logits.
    //
    // Opt-in, and a no-op when the model has no cache form, so the default gate
    // is untouched. Note the reference rounds are independent states (round0
    // past=0, round1 past=1024, ...), not a chain, so the history is re-staged
    // from the npy each round rather than carried across rounds by feedback();
    // this covers the cache-form *kernel* path, not cross-round persistence.
    std::unique_ptr<KVCache> kv;
    int cache_cap = 0;
    const bool cache_form = model_has_cache_form(model_path);
    const bool cache_wanted = std::getenv("VKOP_KV_CACHE") != nullptr;
    const bool cache_mode = cache_form && cache_wanted;
    if (cache_wanted && !cache_form) {
        // Refuse rather than mis-run: on the rotation form the driver's default
        // path is already correct, and preallocating here would feed the
        // rotation's contiguous Gather reads a cap-pitched buffer.
        std::fprintf(stderr,
                     "[kvcache] VKOP_KV_CACHE=1 but %s has no has_cache attribute"
                     " (it is the rotation form) — rerun without the flag\n",
                     model_path.c_str());
        return -1;
    }
    if (cache_form && !cache_wanted) {
        std::printf("[kvcache] model has the cache form; set VKOP_KV_CACHE=1 to "
                    "run it (preallocated past at VKOP_KV_MAX, default 8192)\n");
    }
    if (cache_mode) {
        const std::string rd0 =
            (rounds[0] < 0) ? dump_dir : dump_dir + "/round" + std::to_string(rounds[0]);
        std::ifstream nf(rd0 + "/input_names.txt");
        std::string ln;
        int nlayers = 0;
        while (std::getline(nf, ln)) {
            if (ln.rfind("past_key_values_", 0) == 0) ++nlayers;
        }
        if (nlayers == 0) {
            std::fprintf(stderr, "[kvcache] no past_key_values_* inputs to cache\n");
            return -1;
        }
        NpyArray p0 = load_npy(rd0 + "/past_key_values_0.npy");
        const int nkv = static_cast<int>(p0.shape[2]);
        const int hd = static_cast<int>(p0.shape[4]);
        cache_cap = std::getenv("VKOP_KV_MAX")
                        ? std::atoi(std::getenv("VKOP_KV_MAX"))
                        : 8192;
        kv = std::make_unique<KVCache>(rt, cmdpool, nlayers, nkv, hd, cache_cap,
                                       vkop::core::ElemKind::kFloat16);
        std::printf("[kvcache] layers=%d nkv=%d hd=%d cap=%d\n", nlayers, nkv, hd,
                    cache_cap);
    }

    int n_match = 0, n_round = 0;
    for (int t : rounds) {
        std::string rd = (t < 0) ? dump_dir : dump_dir + "/round" + std::to_string(t);
        std::printf("\n=== round%d (%s) ===\n", n_round, rd.c_str());
        try {
            load_round_inputs(rt, cmdpool, rd, nullptr, /*skip_kv=*/cache_mode);
        } catch (const std::exception& e) {
            std::fprintf(stderr, "load_round_inputs failed: %s\n", e.what());
            return -1;
        }
        if (kv) {
            try {
                int past_len = 0;
                for (int i = 0; i < kv->nlayers(); ++i) {
                    past_len = stage_cache_history(rt, cmdpool, rd, i, cache_cap);
                }
                // Logical lengths back to the true history; the bytes stay put.
                kv->resume_from(past_len);
                std::printf("[kvcache] staged past_len=%d at pitch %d\n", past_len,
                            cache_cap);
            } catch (const std::exception& e) {
                std::fprintf(stderr, "[kvcache] staging failed: %s\n", e.what());
                return -1;
            }
        }
        // Optional upload-verify: readback inputs_embeds + past_key_values_0
        // and compare their first elements / norms against the npy source, to
        // confirm round-to-round input updates actually landed on the SSBOs.
        if (const char *v = std::getenv("VKOP_VERIFY_INPUTS")) {
            (void)v;  // presence-only gate
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
                    // VKOP_DUMP_RAW=<dir>: raw little-endian fp16 + a .shape
                    // sidecar, so a tensor can be diffed element-by-element in
                    // numpy instead of by the four-value preview above. The
                    // name's slashes become '_' ("/lm_head/MatMul__lastrow" ->
                    // "lm_head_MatMul__lastrow").
                    if (const char *raw = std::getenv("VKOP_DUMP_RAW")) {
                        std::string base = nm;
                        for (char &c : base) if (c == '/') c = '_';
                        std::string path = std::string(raw) + "/" + base;
                        std::ofstream fp(path + ".f16", std::ios::binary);
                        if (fp) fp.write(reinterpret_cast<const char*>(p),
                                         std::streamsize(ne) * 2);
                        std::ofstream sp(path + ".shape");
                        if (sp) {
                            sp << tns->getShape().size();
                            for (int d : tns->getShape()) sp << " " << d;
                            sp << " " << ne << "\n";
                        }
                    }
                }
                fflush(stdout);
            }
        }

        int argmax_out = compare_round(rt, cmdpool, rd);
        if (argmax_out >= 0) {
            // Re-fetch ref argmax for the match tally, over the same window
            // compare_round used — a model converted with
            // VKOP_LOGITS_LAST_ROW only carries the last row.
            NpyArray ref = load_npy(rd + "/reference_logits.npy");
            size_t ref_off = 0, ref_n = 0;
            ref_window(ref, ref_off, ref_n);
            const uint16_t *rb =
                reinterpret_cast<const uint16_t*>(ref.data.data()) + ref_off;
            int argmax_ref = -1; float mv = -1e30f;
            for (size_t i = 0; i < ref_n; ++i) {
                float fr = ITensor::fp16_to_fp32(rb[i]);
                if (fr > mv) { mv = fr; argmax_ref = int(i); }
            }
            if (argmax_out == argmax_ref) ++n_match;
        }
        // present→past. The bytes are already in place (the two are one buffer);
        // this only carries the logical length forward, which is what a chained
        // caller would need. Here the next round re-stages from its own npy, so
        // it is exercised for its side effects on the shape, not chained.
        if (kv) {
            std::printf("[kvcache] feedback -> past_len=%d\n", kv->feedback());
        }
        ++n_round;
        fflush(stdout);
    }

    std::printf("\n=== summary: %d/%d rounds MATCH ===\n", n_match, n_round);
    return (n_match == n_round) ? 0 : 2;
}

