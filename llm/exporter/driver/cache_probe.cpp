// junka @ 2026
// cache_probe —— 在一个「只有 attention + KV cache」的最小图上跑多轮，
// 用来真正执行 cache 形式（llm.vkopbin 比 8fb2862 早，本地一直没有任何
// 一个图走过这条路径）。
//
// 图由 /tmp/gen_cache_graph.py 生成：Q/K/V 都是图输入（不带权重），
// past_key_values_0 进、present_key_values_0 出，中间是 HF 那 7 个节点的
// KV 轮转。因此
//
//   VKOP_NO_FUSIONS=fuse_kv_cache_append ./cache_probe ...
//
// 就是同一条调度上「轮转」对「cache 形式」的 A/B：前者每轮拷贝整段历史，
// 后者只在 cache 里写 S 行。
//
// 每轮结束用 CPU 侧参考（fp32 在线 softmax 的朴素实现）对 y 做逐元素比对，
// 因为这条路径此前完全没有跑过。
//
// Usage:
//   cache_probe <cache_block.vkopbin> [--max-kv N] [--prefill S0] [--rounds N]
//               [--window W] [--no-check] [--dump]
//
// --window W turns the cache into a sliding-window ring of W rows: the driver
// lets rounds run past W (the guard is off), advancing the slot of logical row 0
// and shrinking the reported history each round, so the kernel has to wrap. The
// graph must have been generated with VKOP_RING=1 (it carries the int32
// kv_ring_head input the fused Attention reads). The CPU reference mirrors the
// kernel's semantics — attend the resident rows causally — which for the last
// query row of a round is exactly a per-query sliding window of size W.

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <cstdlib>
#include <chrono>
#include <memory>
#include <string>
#include <vector>

#include "vulkan/VulkanDevice.hpp"
#include "vulkan/VulkanInstance.hpp"
#include "include/logger.hpp"
#include "core/DType.hpp"
#include "core/Tensor.hpp"
#include "core/runtime.hpp"
#include "kv_cache.hpp"

using vkop::VulkanInstance;
using vkop::VulkanDevice;
using vkop::VulkanCommandBuffer;
using vkop::core::ITensor;
using vkop::core::Runtime;
using vkop::core::as_tensor;
using vkop::core::ElemKind;
using vkop::export_::KVCache;

namespace {

// 确定性伪随机：round 和 index 决定每个 fp16 值，这样 A/B 两次跑喂进去的
// 输入逐位相同。范围 ±0.5，和真实 Q/K/V 的量级接近。
uint16_t rnd_half(uint32_t seed) {
    uint32_t x = seed * 2654435761u + 0x9e3779b9u;
    x ^= x >> 15;
    x *= 0x85ebca6bu;
    x ^= x >> 13;
    float f = (static_cast<float>(x & 0xffffu) / 65535.0f - 0.5f);
    // fp16 round-trip through the same helper the runtime uses
    uint32_t bits;
    std::memcpy(&bits, &f, sizeof(bits));
    // fp32 -> fp16 (RNE), no denormal concern at this magnitude
    uint32_t sign = (bits >> 16) & 0x8000u;
    int32_t exp = static_cast<int32_t>((bits >> 23) & 0xff) - 127 + 15;
    uint32_t mant = (bits >> 13) & 0x3ffu;
    if (exp <= 0) return static_cast<uint16_t>(sign);
    if (exp >= 31) return static_cast<uint16_t>(sign | 0x7c00u);
    return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exp) << 10) | mant);
}

float half_to_float(uint16_t h) {
    uint32_t sign = (h >> 15) & 1u, exp = (h >> 10) & 0x1fu, mant = h & 0x3ffu;
    uint32_t bits;
    if (exp == 0) {
        if (mant == 0) bits = sign << 31;
        else {
            // subnormal
            int e = -1;
            do { ++e; mant <<= 1; } while ((mant & 0x400u) == 0);
            mant &= 0x3ffu;
            bits = (sign << 31) | (static_cast<uint32_t>(127 - 15 - e) << 23) |
                   (mant << 13);
        }
    } else if (exp == 31) {
        bits = (sign << 31) | 0x7f800000u | (mant << 13);
    } else {
        bits = (sign << 31) | ((exp - 15 + 127) << 23) | (mant << 13);
    }
    float f;
    std::memcpy(&f, &bits, sizeof(f));
    return f;
}

void fill_input(const std::shared_ptr<Runtime>& rt, const std::string& name,
                const std::vector<uint32_t>& shape, const std::vector<uint16_t>& data) {
    rt->ResizeInput(name, shape);
    auto t = rt->GetInput(name);
    auto tg = as_tensor<uint16_t>(t);
    if (tg->num_elements() > 0 && !data.empty()) tg->fillToCPU(data.data());
}

void upload(const std::shared_ptr<vkop::VulkanCommandPool>& cmdpool,
            const std::shared_ptr<ITensor>& t) {
    // fp16 graph inputs (q/k/v/bias) are the common case; the ring head is
    // int32, so dispatch on the container rather than assuming one.
    if (t->dtype() == typeid(int)) as_tensor<int>(t)->copyToGPU(cmdpool);
    else as_tensor<uint16_t>(t)->copyToGPU(cmdpool);
}

void fill_i32_input(const std::shared_ptr<Runtime>& rt, const std::string& name,
                    const std::vector<uint32_t>& shape, int value) {
    auto t = rt->GetInput(name);
    if (!t) return;
    auto tg = as_tensor<int>(t);
    // The ring head is fixed shape [1]; resizing it every round would churn its
    // storage buffer (and, on this driver, disturb the other inputs' buffers)
    // for no reason.
    std::vector<uint32_t> cur(shape.size());
    for (size_t i = 0; i < shape.size(); ++i) {
        cur[i] = static_cast<uint32_t>(tg->getShape()[i]);
    }
    if (cur != shape) rt->ResizeInput(name, shape);
    if (tg->num_elements() > 0) tg->fillToCPU(&value);
}

// fp32 -> fp16 (RNE, no denormal concern at this magnitude).
uint16_t f32_to_half(float v) {
    uint32_t bits;
    std::memcpy(&bits, &v, sizeof(bits));
    uint32_t sign = (bits >> 16) & 0x8000u;
    int32_t e = static_cast<int32_t>((bits >> 23) & 0xff) - 127 + 15;
    uint32_t mant = (bits >> 13) & 0x3ffu;
    return static_cast<uint16_t>(sign | (static_cast<uint32_t>(e) << 10) | mant);
}

// ---- fp8 E4M3, mirroring shaders/buffer/attention.comp -------------------
// The probe's reference has to reproduce the cache's quantization exactly: with
// an fp8 cache the kernel stores encode(x/scale) and reads decode*scale, so a
// reference fed the raw fp16 K/V would differ from the kernel by the fp8 grid
// step (~3% at these magnitudes), swamping the read/write bug this probe exists
// to catch. Both functions are the shader's, transliterated.
uint32_t e4m3_encode(float x) {
    const uint32_t sign = (x < 0.0f) ? 0x80u : 0x00u;
    const float a = std::fabs(x);
    uint32_t code;
    if (a >= 0.015625f) {
        uint32_t bits;
        std::memcpy(&bits, &a, sizeof(bits));
        int32_t e32 = static_cast<int32_t>((bits >> 23) & 0xffu) - 127;
        uint32_t eb = static_cast<uint32_t>(e32 + 127) << 23;
        float p;
        std::memcpy(&p, &eb, sizeof(p));
        int32_t m_i = static_cast<int32_t>(std::nearbyint((a / p - 1.0f) * 8.0f));
        if (m_i >= 8) {
            m_i = 0;
            e32 += 1;
        }
        const int32_t e_code = e32 + 7;
        code = (e_code >= 15)
                   ? 0x7Eu
                   : ((static_cast<uint32_t>(e_code) << 3) |
                      static_cast<uint32_t>(m_i));
    } else {
        code = static_cast<uint32_t>(std::nearbyint(a / 0.001953125f));
    }
    return sign | (code > 0x7Eu ? 0x7Eu : code);
}

float e4m3_decode(uint32_t b) {
    const uint32_t m = b & 0x7Fu;
    float v;
    if (m > 7u) {
        const uint32_t bits = (m + 960u) << 20u;
        std::memcpy(&v, &bits, sizeof(v));
    } else {
        v = static_cast<float>(m) * 0.001953125f;
    }
    return (b & 0x80u) != 0u ? -v : v;
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr,
                     "Usage: %s <cache_block.vkopbin> [--max-kv N] [--prefill S0]\n"
                     "          [--rounds N] [--window W] [--scale S] [--check-max N]\n"
                     "          [--step S] [--no-check] [--dump]\n",
                     argv[0]);
        return 1;
    }
    const std::string model_path = argv[1];
    int max_kv = 4096, prefill = 1, rounds = 8, check_max = 64, window = 0;
    int step = 0; // S for rounds after the first (0 = one row, the decode case)
    float kv_scale = 0.1f;
    bool check = true, dump = false;
    for (int i = 2; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--max-kv" && i + 1 < argc) max_kv = std::atoi(argv[++i]);
        else if (a == "--prefill" && i + 1 < argc) prefill = std::atoi(argv[++i]);
        else if (a == "--step" && i + 1 < argc) step = std::atoi(argv[++i]);
        else if (a == "--rounds" && i + 1 < argc) rounds = std::atoi(argv[++i]);
        else if (a == "--check-max" && i + 1 < argc) check_max = std::atoi(argv[++i]);
        else if (a == "--window" && i + 1 < argc) window = std::atoi(argv[++i]);
        else if (a == "--scale" && i + 1 < argc) kv_scale = std::atof(argv[++i]);
        else if (a == "--no-check") check = false;
        else if (a == "--dump") dump = true;
    }
    // The ring size IS the cache capacity: W rows preallocated, W rows resident.
    const bool ring = window > 0;
    if (ring) max_kv = window;

    Logger::getInstance().setLevel(LOG_INFO);
    const auto& phydevs = VulkanInstance::getVulkanInstance().getPhysicalDevices();
    if (phydevs.empty()) {
        std::printf("no vulkan device\n");
        return 1;
    }
    auto dev = std::make_shared<VulkanDevice>(phydevs[0]);
    std::printf("GPU: %s\n", dev->getDeviceName().c_str());
    auto cmdpool = std::make_shared<vkop::VulkanCommandPool>(dev);

    auto rt = std::make_shared<Runtime>(cmdpool, model_path, /*precision=*/1);
    rt->set_backend_buffer(true);
    rt->LoadModel();

    // Shapes come from the graph itself. The head counts and head dim are
    // static in the export (only batch/seq/kv are symbolic — and those are
    // baked to 1 in the recorded metadata), so they survive round-tripping.
    auto q_t = rt->GetInput("q");
    auto kn_t = rt->GetInput("k_new");
    if (!q_t || !kn_t) {
        std::fprintf(stderr, "graph has no q / k_new input\n");
        return 1;
    }
    auto qs = q_t->getShape();
    auto ks = kn_t->getShape();
    if (qs.size() != 4 || ks.size() != 4) {
        std::fprintf(stderr, "q / k_new are not rank 4\n");
        return 1;
    }
    int H = static_cast<int>(qs[1]);
    int Hkv = static_cast<int>(ks[1]);
    int D = static_cast<int>(qs[3]);
    if (std::getenv("CACHE_H")) H = std::atoi(std::getenv("CACHE_H"));
    if (std::getenv("CACHE_HKV")) Hkv = std::atoi(std::getenv("CACHE_HKV"));
    if (std::getenv("CACHE_D")) D = std::atoi(std::getenv("CACHE_D"));
    if (H <= 0 || Hkv <= 0 || D <= 0 || H % Hkv != 0) {
        std::fprintf(stderr, "bad arch from graph: H=%d Hkv=%d D=%d\n", H, Hkv, D);
        return 1;
    }

    std::printf("[arch] H=%d Hkv=%d D=%d max_kv=%d prefill=%d rounds=%d check=%d%s\n",
                H, Hkv, D, max_kv, prefill, rounds, check ? 1 : 0,
                ring ? " ring" : "");

    // The cache's storage format is the graph's business: a fused fp8 cache
    // (past declared E4M3, the Q/DQ pair folded into the op as `cache_scale`)
    // arrives as an int8_t container, and the probe's KVCache must preallocate
    // in the same width or the alias and the shader's addressing disagree.
    auto pk0 = rt->GetInput("past_key_values_0");
    const ElemKind cache_kind = pk0 ? pk0->elem_kind() : ElemKind::kFloat16;
    const bool cache_fp8 = cache_kind == ElemKind::kFloat8E4M3FN ||
                           cache_kind == ElemKind::kFloat8E5M2;
    if (cache_fp8 && !(kv_scale > 0.0f)) {
        std::fprintf(stderr, "[error] fp8 cache needs --scale > 0\n");
        return 1;
    }
    std::printf("[cache] kind=%s%s\n", vkop::core::elem_name(cache_kind),
                cache_fp8 ? " (reference quantizes to match)" : "");

    KVCache kv(rt, cmdpool, /*nlayers=*/1, Hkv, D, max_kv, cache_kind);
    const bool have_ring_in = rt->GetInput("kv_ring_head") != nullptr;
    if (ring && !have_ring_in) {
        std::fprintf(stderr, "[error] --window needs a graph with kv_ring_head "
                             "(regenerate with VKOP_RING=1)\n");
        return 1;
    }

    // ---- CPU reference state: the K/V history, fp32, indexed by absolute
    // position mod max_kv (the same ring the cache uses; without --window the
    // absolute positions stay below max_kv so the modulo is the identity). ----
    std::vector<float> kh(static_cast<size_t>(max_kv) * Hkv * D, 0.0f);
    std::vector<float> vh(static_cast<size_t>(max_kv) * Hkv * D, 0.0f);
    // The same rows again, unquantized. The kernel reads positions >= kv_past
    // straight from the fp16 K/V — only the *cache* is fp8 — so a reference that
    // took everything from `kh` would be measuring the quantization step on
    // this round's own fresh rows, not the cache path.
    std::vector<float> kraw(kh.size(), 0.0f);
    std::vector<float> vraw(vh.size(), 0.0f);
    // Ring bookkeeping. `resident` is the history rows the next round reports as
    // kv_past; `head` is the slot of logical row 0; `abs_written` is how many
    // absolute positions are in the cache. Logical row j holds absolute position
    // (abs_written - resident) + j. Without --window, head stays 0, resident
    // equals abs_written and the absolute base is 0 — the ring is the identity.
    int resident = 0, head = 0;
    int abs_written = 0;
    const float scale = 1.0f / std::sqrt(static_cast<float>(D));
    double worst = 0.0;

    // What the cache holds for a fresh fp16 value: the identity for fp16, and
    // the fp8 grid point the kernel will encode it to otherwise. The reference
    // has to traffic in the *stored* value, or every round would differ from
    // the kernel by the quantization step.
    auto stored = [&](float x) -> float {
        if (!cache_fp8) return x;
        return e4m3_decode(e4m3_encode(x / kv_scale)) * kv_scale;
    };

    auto run_round = [&](int S, int r) {
        // Make room for the S fresh rows: drop the oldest (resident + S - W)
        // before the run, so kv_len = resident + S never exceeds the buffer and
        // the preallocated past/present alias survives.
        const int drop = std::max(0, resident + S - max_kv);
        head = (head + drop) % max_kv;
        resident -= drop;
        const int base = abs_written - resident; // absolute position of row 0
        const int kv_len = resident + S;
        std::vector<uint16_t> q(static_cast<size_t>(H) * S * D);
        std::vector<uint16_t> kn(static_cast<size_t>(Hkv) * S * D);
        std::vector<uint16_t> vn(static_cast<size_t>(Hkv) * S * D);
        std::vector<uint16_t> bias(static_cast<size_t>(S) * kv_len);
        for (size_t i = 0; i < q.size(); ++i)
            q[i] = rnd_half(static_cast<uint32_t>(r) * 1000003u + static_cast<uint32_t>(i));
        for (size_t i = 0; i < kn.size(); ++i)
            kn[i] = rnd_half(static_cast<uint32_t>(r) * 2000003u + static_cast<uint32_t>(i));
        for (size_t i = 0; i < vn.size(); ++i)
            vn[i] = rnd_half(static_cast<uint32_t>(r) * 3000003u + static_cast<uint32_t>(i));
        for (int s = 0; s < S; ++s) {
            for (int j = 0; j < kv_len; ++j) {
                const bool masked = j > resident + s;
                bias[static_cast<size_t>(s) * kv_len + j] = f32_to_half(masked ? -10000.0f : 0.0f);
            }
        }

        if (r == 0) {
            kv.reset_for_prefill();
            kv.upload();
        } else if (ring) {
            // The buffer already holds the resident rows at the right slots —
            // only the reported history length moves (never an upload, which
            // would overwrite the history with stale staging bytes).
            kv.resume_from(resident);
        }
        // A ring-capable graph carries kv_ring_head even when the driver is not
        // windowing: head 0 makes the slot map the identity, but the input must
        // still be written — an unfilled int32 input is read back as garbage.
        // Host-only: the op reads it through has_cpu_data(), never the shader,
        // so there is no GPU copy to make.
        if (have_ring_in) {
            fill_i32_input(rt, "kv_ring_head", {1u}, head);
        }
        auto t0 = std::chrono::steady_clock::now();
        fill_input(rt, "q", {1u, static_cast<uint32_t>(H), static_cast<uint32_t>(S),
                             static_cast<uint32_t>(D)}, q);
        fill_input(rt, "k_new", {1u, static_cast<uint32_t>(Hkv), static_cast<uint32_t>(S),
                                 static_cast<uint32_t>(D)}, kn);
        fill_input(rt, "v_new", {1u, static_cast<uint32_t>(Hkv), static_cast<uint32_t>(S),
                                 static_cast<uint32_t>(D)}, vn);
        fill_input(rt, "attention_bias",
                   {1u, 1u, static_cast<uint32_t>(S), static_cast<uint32_t>(kv_len)}, bias);
        upload(cmdpool, rt->GetInput("q"));
        upload(cmdpool, rt->GetInput("k_new"));
        upload(cmdpool, rt->GetInput("v_new"));
        upload(cmdpool, rt->GetInput("attention_bias"));

        double ms = rt->Run();
        dev->wait_all_done();
        auto t1 = std::chrono::steady_clock::now();
        double wall = std::chrono::duration<double, std::milli>(t1 - t0).count();
        rt->invalidate_replay();
        // present→past: with the shared buffer this is only the logical length
        // moving forward, and it is what the *next* round's kv_past comes from.
        // Without it every round would think it has no history (kv_past=0) and
        // attend over its own single row. Under the ring the length is clamped
        // instead (see the drop above), so the host sets it next round via
        // resume_from — feedback() here would report the unclamped kv_len.
        if (!ring) kv.feedback();

        // ---- reference ----
        // The CPU history is maintained unconditionally (a big prefill must be
        // folded in so a *later* decode round can be compared against it) but
        // the O(S·kv·D)-per-head recompute only runs when this round is small:
        // a 4096-row prefill would be minutes of scalar math to re-derive a
        // number the S=1 round right after it checks just as well.
        // Fresh rows land at their absolute positions, mod max_kv — the same
        // slot map the cache uses, so a wrapped round overwrites the evicted
        // rows in the reference exactly as the kernel does.
        for (int hk = 0; hk < Hkv; ++hk)
            for (int s = 0; s < S; ++s) {
                const int a = (base + resident + s) % max_kv;
                for (int d = 0; d < D; ++d) {
                    const size_t i = (static_cast<size_t>(a) * Hkv + hk) * D + d;
                    kraw[i] = half_to_float(kn[(static_cast<size_t>(hk) * S + s) * D + d]);
                    vraw[i] = half_to_float(vn[(static_cast<size_t>(hk) * S + s) * D + d]);
                    kh[i] = stored(kraw[i]);
                    vh[i] = stored(vraw[i]);
                }
            }
        // Row j of the window as the kernel sees it: the cache (quantized when
        // the cache is fp8) for the history, the raw fp16 rows for j >= resident.
        auto k_at = [&](int j, int hk, int d) -> float {
            const int a = (base + j) % max_kv;
            const std::vector<float>& src = (j < resident) ? kh : kraw;
            return src[(static_cast<size_t>(a) * Hkv + hk) * D + d];
        };
        auto v_at = [&](int j, int hk, int d) -> float {
            const int a = (base + j) % max_kv;
            const std::vector<float>& src = (j < resident) ? vh : vraw;
            return src[(static_cast<size_t>(a) * Hkv + hk) * D + d];
        };
        const bool do_check = check && S <= check_max;
        std::vector<float> ref;
        const uint16_t* yp = nullptr;
        double worst_here = 0.0;
        int worst_at = -1;
        if (do_check) {
            auto y = rt->GetOutput("y");
            as_tensor<uint16_t>(y)->copyToCPU(cmdpool);
            yp = reinterpret_cast<const uint16_t*>(as_tensor<uint16_t>(y)->data().data());
            ref.assign(static_cast<size_t>(H) * S * D, 0.0f);
            const int rep = H / Hkv;
            for (int h = 0; h < H; ++h) {
                const int hk = h / rep;
                for (int s = 0; s < S; ++s) {
                    // Causal over the resident rows: logical j in [0, resident+s],
                    // i.e. absolute [base, base+resident+s]. For the last query
                    // row of a round this is exactly a W-wide sliding window.
                    const int n = resident + s + 1;
                    std::vector<float> sc(n);
                    float m = -1e30f;
                    for (int j = 0; j < n; ++j) {
                        float dot = 0.0f;
                        for (int d = 0; d < D; ++d)
                            dot += half_to_float(q[(static_cast<size_t>(h) * S + s) * D + d]) *
                                   k_at(j, hk, d);
                        sc[j] = dot * scale +
                                half_to_float(bias[static_cast<size_t>(s) * kv_len + j]);
                        m = std::max(m, sc[j]);
                    }
                    float l = 0.0f;
                    for (int j = 0; j < n; ++j) { sc[j] = std::exp(sc[j] - m); l += sc[j]; }
                    for (int d = 0; d < D; ++d) {
                        float acc = 0.0f;
                        for (int j = 0; j < n; ++j)
                            acc += sc[j] * v_at(j, hk, d);
                        ref[(static_cast<size_t>(h) * S + s) * D + d] = acc / l;
                    }
                }
            }
            for (size_t i = 0; i < ref.size(); ++i) {
                const double d = std::fabs(static_cast<double>(half_to_float(yp[i])) -
                                           static_cast<double>(ref[i]));
                if (d > worst_here) { worst_here = d; worst_at = static_cast<int>(i); }
            }
            worst = std::max(worst, worst_here);
        }

        std::printf("  round %d: S=%d hist=%d head=%d kv=%d  vkop=%.1fms wall=%.1fms",
                    r, S, resident, ring ? head : 0, kv_len, ms, wall);
        if (do_check) {
            std::printf("  maxdiff=%.5f @%d", worst_here, worst_at);
        } else {
            std::printf("  (no check, S>%d)", check_max);
        }
        std::printf("\n");
        if (do_check && dump && r < 2) {
            std::printf("    y[0..7] =");
            for (int i = 0; i < 8; ++i) std::printf(" %.5f", half_to_float(yp[i]));
            std::printf("\n    ref     =");
            for (int i = 0; i < 8; ++i) std::printf(" %.5f", ref[i]);
            std::printf("\n");
        }
        abs_written += S;
        resident += S; // the rows the next round reads back as history
    };

    // Round 0 is the prefill of `prefill` rows; the rest are decode rounds
    // (one row each), unless --step says otherwise — a multi-row append is what
    // the driver's chunked prefill does, and it is the only case where the drop
    // and the fresh rows are both wide.
    for (int r = 0; r < rounds; ++r) {
        int S = (r == 0) ? prefill : (step > 0 ? step : 1);
        // With the ring the point is to run *past* the capacity: the drop below
        // evicts instead of stopping.
        if (!ring && abs_written + S > max_kv) {
            std::printf("[stop] past+S=%d > max_kv=%d\n", abs_written + S, max_kv);
            break;
        }
        run_round(S, r);
    }
    if (check) std::printf("[check] worst |vkop-ref| over all rounds = %.5f\n", worst);
    std::printf("[done]\n");
    dev->wait_all_done();
    return 0;
}
