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
//               [--no-check] [--dump]

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
#include "core/Tensor.hpp"
#include "core/runtime.hpp"
#include "kv_cache.hpp"

using vkop::VulkanInstance;
using vkop::VulkanDevice;
using vkop::VulkanCommandBuffer;
using vkop::core::ITensor;
using vkop::core::Runtime;
using vkop::core::as_tensor;
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
    as_tensor<uint16_t>(t)->copyToGPU(cmdpool);
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr,
                     "Usage: %s <cache_block.vkopbin> [--max-kv N] [--prefill S0]\n"
                     "          [--rounds N] [--no-check] [--dump]\n", argv[0]);
        return 1;
    }
    const std::string model_path = argv[1];
    int max_kv = 4096, prefill = 1, rounds = 8, check_max = 64;
    bool check = true, dump = false;
    for (int i = 2; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--max-kv" && i + 1 < argc) max_kv = std::atoi(argv[++i]);
        else if (a == "--prefill" && i + 1 < argc) prefill = std::atoi(argv[++i]);
        else if (a == "--rounds" && i + 1 < argc) rounds = std::atoi(argv[++i]);
        else if (a == "--check-max" && i + 1 < argc) check_max = std::atoi(argv[++i]);
        else if (a == "--no-check") check = false;
        else if (a == "--dump") dump = true;
    }

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

    std::printf("[arch] H=%d Hkv=%d D=%d max_kv=%d prefill=%d rounds=%d check=%d\n",
                H, Hkv, D, max_kv, prefill, rounds, check ? 1 : 0);

    KVCache kv(rt, cmdpool, /*nlayers=*/1, Hkv, D, max_kv);

    // ---- CPU reference state: the full K/V history, fp32 ----
    std::vector<float> kh(static_cast<size_t>(max_kv) * Hkv * D, 0.0f);
    std::vector<float> vh(static_cast<size_t>(max_kv) * Hkv * D, 0.0f);
    int past = 0;
    const float scale = 1.0f / std::sqrt(static_cast<float>(D));
    double worst = 0.0;

    auto run_round = [&](int S, int r) {
        const int kv_len = past + S;
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
                const bool masked = j > past + s;
                const float v = masked ? -10000.0f : 0.0f;
                // fp32 -> fp16 via the same rounding as rnd_half's tail
                uint32_t bits; std::memcpy(&bits, &v, sizeof(bits));
                uint32_t sign = (bits >> 16) & 0x8000u;
                int32_t e = static_cast<int32_t>((bits >> 23) & 0xff) - 127 + 15;
                uint32_t mant = (bits >> 13) & 0x3ffu;
                uint16_t h = static_cast<uint16_t>(sign | (static_cast<uint32_t>(e) << 10) | mant);
                bias[static_cast<size_t>(s) * kv_len + j] = h;
            }
        }

        if (r == 0) { kv.reset_for_prefill(); kv.upload(); }
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
        // attend over its own single row.
        kv.feedback();

        // ---- reference ----
        // The CPU history is maintained unconditionally (a big prefill must be
        // folded in so a *later* decode round can be compared against it) but
        // the O(S·kv·D)-per-head recompute only runs when this round is small:
        // a 4096-row prefill would be minutes of scalar math to re-derive a
        // number the S=1 round right after it checks just as well.
        for (int hk = 0; hk < Hkv; ++hk)
            for (int s = 0; s < S; ++s)
                for (int d = 0; d < D; ++d) {
                    kh[(static_cast<size_t>(past + s) * Hkv + hk) * D + d] =
                        half_to_float(kn[(static_cast<size_t>(hk) * S + s) * D + d]);
                    vh[(static_cast<size_t>(past + s) * Hkv + hk) * D + d] =
                        half_to_float(vn[(static_cast<size_t>(hk) * S + s) * D + d]);
                }
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
                    const int n = past + s + 1;
                    std::vector<float> sc(n);
                    float m = -1e30f;
                    for (int j = 0; j < n; ++j) {
                        float dot = 0.0f;
                        for (int d = 0; d < D; ++d)
                            dot += half_to_float(q[(static_cast<size_t>(h) * S + s) * D + d]) *
                                   kh[(static_cast<size_t>(j) * Hkv + hk) * D + d];
                        sc[j] = dot * scale +
                                half_to_float(bias[static_cast<size_t>(s) * kv_len + j]);
                        m = std::max(m, sc[j]);
                    }
                    float l = 0.0f;
                    for (int j = 0; j < n; ++j) { sc[j] = std::exp(sc[j] - m); l += sc[j]; }
                    for (int d = 0; d < D; ++d) {
                        float acc = 0.0f;
                        for (int j = 0; j < n; ++j)
                            acc += sc[j] * vh[(static_cast<size_t>(j) * Hkv + hk) * D + d];
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

        std::printf("  round %d: S=%d past=%d kv=%d  vkop=%.1fms wall=%.1fms",
                    r, S, past, kv_len, ms, wall);
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
        past += S;
    };

    // Round 0 is the prefill of `prefill` rows; the rest are decode rounds.
    for (int r = 0; r < rounds; ++r) {
        int S = (r == 0) ? prefill : 1;
        if (past + S > max_kv) {
            std::printf("[stop] past+S=%d > max_kv=%d\n", past + S, max_kv);
            break;
        }
        run_round(S, r);
    }
    if (check) std::printf("[check] worst |vkop-ref| over all rounds = %.5f\n", worst);
    std::printf("[done]\n");
    dev->wait_all_done();
    return 0;
}
