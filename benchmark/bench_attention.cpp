// Fused Attention microbenchmark: the row kernel (attention.comp, one
// workgroup per query row) against the tiled many-row kernel
// (attention_flash.comp), same process, same inputs, interleaved.
//
// Each arm is one real Attention op (ops::create_from_type) recorded once
// through the replay cache with a GPU timestamp pair inside its own command
// buffer (Operator::set_prof_query, the VKOP_SUBMIT_PROF mechanism). A sample
// is one submit + wait + timestamp read; the two arms alternate sample by
// sample so a concurrent GPU user or a clock change hits both, and the median
// is reported. The outputs of both arms are compared with each other and,
// for two heads, with an fp32 CPU reference.
//
// Run:
//   DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib
//   VK_ICD_FILENAMES=/opt/homebrew/etc/vulkan/icd.d/MoltenVK_icd.json
//   ./build/benchmark/bench_attention [samples]
// Env: BENCH_SHAPES=dit,llm,small (comma list; default all).
//      BENCH_ACT_BF16=1: arm 0 is the fp16 op on fp16 tensors, arm 1 the
//      _bf16 op on the same values as bf16 tensors (--act bf16), both on the
//      tiled route the op picks (fp16: the coop8 build on Apple, bf16: the
//      SIMT -DBF16 build). BENCH_ATTN_FP16_COOP8=0 pins arm 0 to the SIMT
//      tiled kernel too, for the like-for-like kernel A/B. Arm 1's error
//      column includes the bf16 rounding of its inputs.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "core/Tensor.hpp"
#include "include/dev_shaders.hpp"
#include "include/logger.hpp"
#include "ops/OperatorFactory.hpp"
#include "vulkan/VulkanBuffer.hpp"
#include "vulkan/VulkanCommandBuffer.hpp"
#include "vulkan/VulkanCommandPool.hpp"
#include "vulkan/VulkanDevice.hpp"
#include "vulkan/VulkanInstance.hpp"
#include "vulkan/VulkanQueryPool.hpp"

using namespace vkop;

namespace {

using T16 = core::Tensor<uint16_t>;

struct Case {
    const char *name;
    const char *group;
    int B, H, Hkv, S, kv, D;
    bool causal;      // causal attribute (no mask tensor)
    bool mask;        // additive [B,1,S,kv] mask tensor
    bool mask_causal; // that mask is the causal one (declared)
    int masked_tail;  // non-causal mask: last N columns at -65504
    bool seq_major;
};

uint32_t g_rng = 0x2468ace1u;
float frand() {
    g_rng = g_rng * 1664525u + 1013904223u;
    return static_cast<float>((g_rng >> 8) & 0xFFFF) / 65536.0f * 2.0f - 1.0f;
}

std::shared_ptr<T16> make_tensor(const std::vector<int> &shape,
                                 std::vector<float> &host, float amp, std::shared_ptr<VulkanDevice> dev,

                                 const std::shared_ptr<VulkanCommandPool> &cp) {
    auto t = std::make_shared<T16>(shape);
    t->reserveOnCPU();
    host.resize(static_cast<size_t>(t->num_elements()));
    for (size_t i = 0; i < host.size(); ++i) {
        uint16_t h = core::ITensor::fp32_to_fp16(frand() * amp);
        t->data()[i] = h;
        host[i] = core::ITensor::fp16_to_fp32(h);
    }
    t->as_storage_buffer(dev);
    t->copyToGPU(cp);
    return t;
}

double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v.empty() ? 0.0 : v[v.size() / 2];
}

} // namespace

int main(int argc, char **argv) {
    Logger::getInstance().setLevel(LOG_ERROR);
    vkop::dev::seed_shader_archive_from_build_tree();
    const auto &phydevs =
        VulkanInstance::getVulkanInstance().getPhysicalDevices();
    if (phydevs.empty()) {
        printf("no vulkan device\n");
        return 1;
    }
    auto dev = std::make_shared<VulkanDevice>(phydevs[0]);
    printf("GPU: %s\n", dev->getDeviceName().c_str());
    auto cmdpool = std::make_shared<VulkanCommandPool>(dev);
    auto queue = dev->getComputeQueue(0);
    const int samples = argc > 1 ? std::atoi(argv[1]) : 15;
    std::string want = "dit,llm,small,few";
    if (const char *e = std::getenv("BENCH_SHAPES")) want = e;
    const bool coop_ab = std::getenv("BENCH_COOP8") != nullptr;
    const bool act_bf16 = std::getenv("BENCH_ACT_BF16") != nullptr;
    const char *f16c8 = std::getenv("BENCH_ATTN_FP16_COOP8");

    std::vector<Case> cases = {
        // Qwen-Image DiT joint attention: 1024 image + 64 text tokens, the
        // text padding masked through the additive bias.
        {"dit", "dit", 1, 32, 32, 1024, 1088, 128, false, true, false, 24,
         false},
        // Same shape without the bias: what the mask reads cost.
        {"dit_nomask", "dit", 1, 32, 32, 1024, 1088, 128, false, false, false,
         0, false},
        {"dit_seqmaj", "dit", 1, 32, 32, 1024, 1088, 128, false, true, false,
         24, true},
        // MiniMax-H3 FL2VA DiT at 512x288x56f: 2687 joint rows, 56 heads,
        // no mask (BENCH_SHAPES=h3).
        {"h3_2687", "h3", 1, 56, 56, 2687, 2687, 128, false, false, false, 0,
         false},
        // LLM prefill (GQA 16/8, D=128): causal attribute and causal mask.
        {"llm512c", "llm", 1, 16, 8, 512, 512, 128, true, false, false, 0,
         false},
        {"llm1024m", "llm", 1, 16, 8, 1024, 1024, 128, false, true, true, 0,
         false},
        {"llm2048c", "llm", 1, 16, 8, 2048, 2048, 128, true, false, false, 0,
         false},
        {"llm2048m", "llm", 1, 16, 8, 2048, 2048, 128, false, true, true, 0,
         false},
        // ViT-like bidirectional, D=64 and D=80.
        {"vit1024d64", "llm", 1, 16, 16, 1024, 1024, 64, false, false, false,
         0, true},
        {"vit1024d80", "llm", 1, 16, 16, 1024, 1024, 80, false, false, false,
         0, true},
        // Threshold sweep: short prefill chunks against a history.
        {"s16", "small", 1, 16, 8, 16, 1040, 128, true, false, false, 0, false},
        {"s32", "small", 1, 16, 8, 32, 1056, 128, true, false, false, 0, false},
        {"s48", "small", 1, 16, 8, 48, 1072, 128, true, false, false, 0, false},
        {"s64", "small", 1, 16, 8, 64, 1088, 128, true, false, false, 0, false},
        {"s128", "small", 1, 16, 8, 128, 1152, 128, true, false, false, 0,
         false},
        {"s256", "small", 1, 16, 8, 256, 256, 128, true, false, false, 0,
         false},
        // Few heads against a long history: few row blocks to spread.
        {"h2s64", "few", 1, 2, 2, 64, 4096, 128, false, false, false, 0,
         false},
        {"h2s128", "few", 1, 2, 2, 128, 4096, 128, false, false, false, 0,
         false},
        {"h4s256", "few", 1, 4, 4, 256, 4096, 128, false, false, false, 0,
         false},
        {"h8s64", "few", 1, 8, 8, 64, 4096, 128, false, false, false, 0,
         false},
        {"h16s64", "few", 1, 16, 16, 64, 4096, 128, false, false, false, 0,
         false},
    };

    printf("%-11s %5s %5s %4s | %9s %9s %7s | %10s %10s %10s\n", "case", "S",
           "kv", "D", "row ms", "tiled ms", "speedup", "max|d|o-n",
           "max|e|row", "max|e|tile");
    for (const auto &c : cases) {
        if (want.find(c.group) == std::string::npos) continue;
        std::vector<float> hq, hk, hv, hm;
        std::vector<int> qs = c.seq_major
                                  ? std::vector<int>{c.B, c.S, c.H, c.D}
                                  : std::vector<int>{c.B, c.H, c.S, c.D};
        std::vector<int> ks = c.seq_major
                                  ? std::vector<int>{c.B, c.kv, c.Hkv, c.D}
                                  : std::vector<int>{c.B, c.Hkv, c.kv, c.D};
        auto Q = make_tensor(qs, hq, 2.0f, dev, cmdpool);
        auto K = make_tensor(ks, hk, 2.0f, dev, cmdpool);
        auto V = make_tensor(ks, hv, 1.0f, dev, cmdpool);
        std::vector<std::shared_ptr<core::ITensor>> in{Q, K, V};
        std::shared_ptr<T16> M;
        if (c.mask) {
            M = std::make_shared<T16>(std::vector<int>{c.B, 1, c.S, c.kv});
            M->reserveOnCPU();
            hm.resize(static_cast<size_t>(M->num_elements()));
            const int past = c.kv - c.S;
            for (int b = 0; b < c.B; ++b)
                for (int s = 0; s < c.S; ++s)
                    for (int j = 0; j < c.kv; ++j) {
                        float v = 0.0f;
                        if (c.mask_causal) {
                            v = (j <= past + s) ? 0.0f : -65504.0f;
                        } else if (j >= c.kv - c.masked_tail) {
                            v = -65504.0f;
                        }
                        size_t i = (static_cast<size_t>(b) * c.S + s) * c.kv + j;
                        M->data()[i] = core::ITensor::fp32_to_fp16(v);
                        hm[i] = v;
                    }
            M->as_storage_buffer(dev);
            M->copyToGPU(cmdpool);
            in.push_back(M);
        }
        // bf16 copies of every input (same values, bf16-rounded), for the
        // BENCH_ACT_BF16 arm.
        std::vector<std::shared_ptr<core::ITensor>> in_bf;
        if (act_bf16) {
            for (const auto &t : in) {
                auto src = std::dynamic_pointer_cast<T16>(t);
                auto b = std::make_shared<T16>(src->getShape());
                b->reserveOnCPU();
                const std::vector<float> &h =
                    t == Q ? hq : t == K ? hk : t == V ? hv : hm;
                for (size_t i = 0; i < h.size(); ++i)
                    b->data()[i] = core::float_to_bf16(h[i]);
                b->set_elem_kind(core::ElemKind::kBFloat16);
                b->as_storage_buffer(dev);
                b->copyToGPU(cmdpool);
                in_bf.push_back(b);
            }
        }
        const float scale = 1.0f / std::sqrt(static_cast<float>(c.D));
        std::unordered_map<std::string, std::string> attrs{
            {"scale", std::to_string(scale)},
            {"num_heads", std::to_string(c.H)},
            {"num_kv_heads", std::to_string(c.Hkv)},
            {"causal", c.causal ? "1" : "0"},
            {"mask_causal", c.mask_causal ? "1" : "0"},
            {"seq_major", c.seq_major ? "1" : "0"}};

        struct Arm {
            std::shared_ptr<ops::Operator> op;
            std::shared_ptr<T16> out;
            std::unique_ptr<VulkanQueryPool> ts;
            std::shared_ptr<VulkanCommandBuffer> cmd;
            std::vector<double> ms;
            std::vector<double> wall;
        } arms[2];
        bool ok = true;
        for (int a = 0; a < 2; ++a) {
            auto &arm = arms[a];
            auto at = attrs;
            // BENCH_COOP8=1: arm 0 is the SIMT tiled kernel, arm 1 the
            // coopmat one (MoltenVK). Default: row kernel against tiled.
            const bool bf_arm = act_bf16 && a == 1;
            if (act_bf16) {
                at["flash"] = "1";
                if (a == 0 && f16c8 != nullptr) at["flash_coop8"] = f16c8;
            } else if (coop_ab) {
                at["flash"] = "1";
                at["flash_coop8"] = a == 0 ? "0" : "1";
            } else {
                at["flash"] = a == 0 ? "0" : "1";
            }
            arm.op = ops::create_from_type(ops::OpType::ATTENTION,
                                           bf_arm ? 2 : 1, 0, true);
            arm.op->set_runtime_device(dev, cmdpool);
            arm.op->setAttribute(at);
            arm.op->enable_replay(true);
            arm.out = std::make_shared<T16>(
                c.seq_major ? std::vector<int>{c.B, c.S, c.H, c.D}
                            : std::vector<int>{c.B, c.H, c.S, c.D});
            if (bf_arm) arm.out->set_elem_kind(core::ElemKind::kBFloat16);
            arm.out->as_storage_buffer(dev);
            arm.ts = std::make_unique<VulkanQueryPool>(dev->getLogicalDevice(),
                                                       2);
            arm.op->set_prof_query(arm.ts->get(), 0);
            std::vector<std::shared_ptr<core::ITensor>> out{arm.out};
            const auto &ins = bf_arm ? in_bf : in;
            arm.op->onExecute(ins, out, 0);
            arm.op->onExecute(ins, out, 0);
            if (act_bf16) {
                printf("  arm %d (%s): route %s\n", a, bf_arm ? "bf16" : "fp16",
                       arm.op->route_tag().c_str());
            }
            arm.cmd = arm.op->get_record();
            if (!arm.cmd) {
                printf("%s: no recorded cmd\n", c.name);
                ok = false;
                break;
            }
            arm.cmd->submit(queue);
            dev->wait_all_done();
        }
        if (!ok) continue;
        // Warm-up, then interleaved samples.
        for (int w = 0; w < 2; ++w)
            for (auto &arm : arms) {
                arm.cmd->submit(queue);
                dev->wait_all_done();
            }
        const double period = dev->getTimestampPeriod() * 1e-6;
        for (int sidx = 0; sidx < samples; ++sidx) {
            for (int a = 0; a < 2; ++a) {
                auto &arm = arms[(sidx & 1) ? 1 - a : a];
                arm.ts->reset();
                auto w0 = std::chrono::steady_clock::now();
                arm.cmd->submit(queue);
                dev->wait_all_done();
                arm.wall.push_back(std::chrono::duration<double, std::milli>(
                                       std::chrono::steady_clock::now() - w0)
                                       .count());
                auto t = arm.ts->getResults();
                if (t[1] > t[0])
                    arm.ms.push_back(static_cast<double>(t[1] - t[0]) * period);
            }
        }
        // Outputs.
        std::vector<float> yo, yn;
        for (int a = 0; a < 2; ++a) {
            arms[a].out->reserveOnCPU();
            arms[a].out->copyToCPU(cmdpool);
            auto &dst = a == 0 ? yo : yn;
            dst.resize(static_cast<size_t>(arms[a].out->num_elements()));
            for (size_t i = 0; i < dst.size(); ++i)
                dst[i] = (act_bf16 && a == 1)
                             ? core::bf16_to_float(arms[a].out->data()[i])
                             : core::ITensor::fp16_to_fp32(arms[a].out->data()[i]);
        }
        double d_on = 0.0;
        for (size_t i = 0; i < yo.size(); ++i)
            d_on = std::max(d_on, static_cast<double>(std::fabs(yo[i] - yn[i])));
        // fp32 CPU reference on heads 0 and H-1, every 4th row.
        auto qidx = [&](int h, int s, int d) {
            return c.seq_major ? ((static_cast<size_t>(s) * c.H + h) * c.D + d)
                               : ((static_cast<size_t>(h) * c.S + s) * c.D + d);
        };
        auto kidx = [&](int hk, int j, int d) {
            return c.seq_major
                       ? ((static_cast<size_t>(j) * c.Hkv + hk) * c.D + d)
                       : ((static_cast<size_t>(hk) * c.kv + j) * c.D + d);
        };
        double e_old = 0.0, e_new = 0.0;
        const int past = c.kv - c.S;
        std::vector<double> sc(c.kv);
        for (int h : {0, c.H - 1}) {
            int hg = h / (c.H / c.Hkv);
            for (int s = 0; s < c.S; s += 4) {
                double mx = -1e300;
                for (int j = 0; j < c.kv; ++j) {
                    double dot = 0.0;
                    for (int d = 0; d < c.D; ++d)
                        dot += static_cast<double>(hq[qidx(h, s, d)]) *
                               hk[kidx(hg, j, d)];
                    double x = dot * scale;
                    if (c.mask) x += hm[static_cast<size_t>(s) * c.kv + j];
                    if (c.causal && j > past + s) x = -1e300;
                    sc[j] = x;
                    mx = std::max(mx, x);
                }
                double l = 0.0;
                for (int j = 0; j < c.kv; ++j) {
                    sc[j] = std::exp(sc[j] - mx);
                    l += sc[j];
                }
                for (int d = 0; d < c.D; ++d) {
                    double y = 0.0;
                    for (int j = 0; j < c.kv; ++j)
                        y += sc[j] * hv[kidx(hg, j, d)];
                    y /= l;
                    size_t oi = qidx(h, s, d);
                    e_old = std::max(e_old, std::fabs(y - yo[oi]));
                    e_new = std::max(e_new, std::fabs(y - yn[oi]));
                }
            }
        }
        double mo = median(arms[0].ms), mn = median(arms[1].ms);
        double wo = median(arms[0].wall), wn = median(arms[1].wall);
        printf("%-11s %5d %5d %4d | %9.3f %9.3f %6.2fx | %10.2e %10.2e %10.2e"
               " | wall %7.3f %7.3f %5.2fx\n",
               c.name, c.S, c.kv, c.D, mo, mn, mo / mn, d_on, e_old, e_new, wo,
               wn, wo / wn);
        fflush(stdout);
    }
    return 0;
}
