// Copyright 2026 @junka
// End-to-end probe for the three qwen3_5 linear-attention ops:
//   CumSum          (prefix sum along an axis)
//   Conv1d          (causal depthwise conv with a rolling state)
//   LinearAttention (the gated delta rule recurrence with a matrix state)
//
// Each is driven through the real op class against a plain CPU reference
// implementation of the same math, so this is an accuracy test, not a smoke
// test. Everything here is fp32 (all three ops are fp32-only by design) and
// buffer-backed.
//
// Skips cleanly when there is no Vulkan device. The BDA/descriptor split is
// exercised by running the same cases with VKOP_BDA=1 and VKOP_BDA unset.

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <vector>

#include "core/Tensor.hpp"
#include "include/dev_shaders.hpp"
#include "include/logger.hpp"
#include "ops/OperatorFactory.hpp"
#include "ops/Ops.hpp"
#include "vulkan/VulkanCommandBuffer.hpp"
#include "vulkan/VulkanCommandPool.hpp"
#include "vulkan/VulkanDevice.hpp"
#include "vulkan/VulkanInstance.hpp"

using vkop::VulkanCommandBuffer;
using vkop::VulkanCommandPool;
using vkop::VulkanDevice;
using vkop::VulkanInstance;
using vkop::core::Tensor;
using vkop::ops::OpType;

namespace {

int failures = 0;

void check(bool cond, const char *what) {
    if (!cond) {
        std::printf("FAIL: %s\n", what);
        failures++;
    }
}

// Tolerant compare: a fixed absolute floor plus a relative term, so a value
// near zero is not held to a ratio the fp32 arithmetic cannot deliver.
void expect_close(const char *what, size_t i, float got, float want,
                  float rel = 2e-4F, float abs = 1e-5F) {
    const float err = std::fabs(got - want);
    if (err > abs + rel * std::fabs(want)) {
        std::printf("FAIL: %s[%zu] got %.6f want %.6f (err %.3e)\n", what, i,
                    got, want, err);
        failures++;
    }
}

template <typename T>
std::shared_ptr<Tensor<T>> make(std::shared_ptr<VulkanDevice> &dev,
                                std::shared_ptr<VulkanCommandPool> &pool,
                                const std::vector<int> &shape,
                                const std::vector<float> &vals) {
    auto t = std::make_shared<Tensor<T>>(shape);
    std::vector<T> bytes(vals.size());
    for (size_t i = 0; i < vals.size(); i++) {
        bytes[i] = static_cast<T>(vals[i]);
    }
    t->fillToCPU(bytes);
    t->as_storage_buffer(dev);
    t->copyToGPU(pool);
    return t;
}

template <typename T>
std::shared_ptr<Tensor<T>> make_out(std::shared_ptr<VulkanDevice> &dev,
                                    const std::vector<int> &shape,
                                    size_t n) {
    auto t = std::make_shared<Tensor<T>>(shape);
    t->as_storage_buffer(dev);
    t->toGPU();
    (void)n;
    return t;
}

// Run an op over (inputs -> outputs) once and synchronise.
void run(std::shared_ptr<VulkanDevice> &dev,
         vkop::ops::Operator *op,
         const std::vector<std::shared_ptr<vkop::core::ITensor>> &ins,
         const std::vector<std::shared_ptr<vkop::core::ITensor>> &outs) {
    op->onExecute(ins, outs, 0);
    auto cmd = op->get_record();
    std::vector<VkSubmitInfo> info{cmd->buildSubmitInfo()};
    VulkanCommandBuffer::submit(dev->getComputeQueue(), info);
    cmd->wait();
    dev->wait_all_done();
}

std::vector<float> readback(std::shared_ptr<VulkanCommandPool> &pool,
                            const std::shared_ptr<Tensor<float>> &t) {
    t->copyToCPU(pool);
    const auto &d = t->data();
    return std::vector<float>(d.begin(), d.end());
}

// ---------------------------------------------------------------- CumSum ---

void test_cumsum(std::shared_ptr<VulkanDevice> &dev,
                 std::shared_ptr<VulkanCommandPool> &pool) {
    // [2, 3, 4] summed along axis 1: two outer slices, three steps, stride 4.
    const std::vector<int> shape{2, 3, 4};
    std::vector<float> x(24);
    for (size_t i = 0; i < x.size(); i++) {
        x[i] = std::sin(static_cast<float>(i) * 0.7F) * 2.0F;
    }
    const int axis = 1;
    const int outer = shape[0], len = shape[1], stride = shape[2];
    std::vector<float> want(24, 0.0F);
    for (int o = 0; o < outer; o++) {
        for (int s = 0; s < stride; s++) {
            float acc = 0.0F;
            for (int i = 0; i < len; i++) {
                const int idx = (o * len + i) * stride + s;
                acc += x[idx];
                want[idx] = acc;
            }
        }
    }

    auto in = make<float>(dev, pool, shape, x);
    auto out = make_out<float>(dev, shape, x.size());
    auto op = vkop::ops::create_from_type(OpType::CUMSUM, 0, 0, true);
    check(op != nullptr, "CumSum constructed");
    if (!op) return;
    op->setAttribute({{"axis", std::to_string(axis)}});
    op->set_runtime_device(dev, pool);
    run(dev, op.get(), {in}, {out});
    const auto got = readback(pool, out);
    for (size_t i = 0; i < got.size(); i++) {
        expect_close("CumSum", i, got[i], want[i]);
    }
    std::printf("PASS: CumSum [2,3,4] axis=1 (bda=%d)\n",
                op->bda_pipeline_ready() ? 1 : 0);

    // A second axis (last) to catch a stride-1 assumption.
    std::vector<float> want_m2(24, 0.0F);
    const int axis2 = 2;
    const int outer2 = shape[0] * shape[1], len2 = shape[2];
    for (int o = 0; o < outer2; o++) {
        float acc = 0.0F;
        for (int i = 0; i < len2; i++) {
            acc += x[o * len2 + i];
            want_m2[o * len2 + i] = acc;
        }
    }
    auto in2 = make<float>(dev, pool, shape, x);
    auto out2 = make_out<float>(dev, shape, x.size());
    auto op2 = vkop::ops::create_from_type(OpType::CUMSUM, 0, 0, true);
    op2->setAttribute({{"axis", std::to_string(axis2)}});
    op2->set_runtime_device(dev, pool);
    run(dev, op2.get(), {in2}, {out2});
    const auto got2 = readback(pool, out2);
    for (size_t i = 0; i < got2.size(); i++) {
        expect_close("CumSum-axis2", i, got2[i], want_m2[i]);
    }
    std::printf("PASS: CumSum [2,3,4] axis=2\n");
}

// ---------------------------------------------------------------- Conv1d ---

// One round of the causal depthwise convolution, CPU side. `state` is
// [C, K-1] and is updated in place (the caller reuses it next round).
void conv1d_ref(const std::vector<float> &x, const std::vector<float> &w,
                int C, int S, int K, std::vector<float> &state,
                std::vector<float> &y) {
    const int pad = K - 1;
    for (int c = 0; c < C; c++) {
        for (int t = 0; t < S; t++) {
            float acc = 0.0F;
            for (int k = 0; k < K; k++) {
                const int pos = t - (K - 1) + k;
                float v;
                if (pos >= 0) {
                    v = x[c * S + pos];
                } else {
                    const int si = pad + pos;
                    v = si >= 0 ? state[c * pad + si] : 0.0F;
                }
                acc += v * w[c * K + k];
            }
            y[c * S + t] = acc;
        }
    }
    // Roll the state: the last pad samples of (state ++ x).
    for (int c = 0; c < C; c++) {
        std::vector<float> h;
        h.reserve(pad + S);
        for (int i = 0; i < pad; i++) h.push_back(state[c * pad + i]);
        for (int t = 0; t < S; t++) h.push_back(x[c * S + t]);
        for (int j = 0; j < pad; j++) {
            state[c * pad + j] = h[S + j];
        }
    }
}

void test_conv1d(std::shared_ptr<VulkanDevice> &dev,
                 std::shared_ptr<VulkanCommandPool> &pool) {
    const int C = 6, S = 5, K = 4;
    const int pad = K - 1;
    std::vector<float> w(C * K), x(C * S), state(C * pad, 0.0F);
    for (size_t i = 0; i < w.size(); i++) w[i] = 0.3F + 0.1F * (i % 5);
    for (size_t i = 0; i < x.size(); i++) x[i] = std::cos(i * 0.4F);
    auto state_ref = state;

    auto w_t = make<float>(dev, pool, {C, K}, w);
    auto x_t = make<float>(dev, pool, {C, S}, x);
    auto st_t = make<float>(dev, pool, {C, pad}, state);
    auto y_t = make_out<float>(dev, {C, S}, C * S);
    auto so_t = make_out<float>(dev, {C, pad}, C * pad);

    auto op = vkop::ops::create_from_type(OpType::CONV1D, 0, 0, true);
    check(op != nullptr, "Conv1d constructed");
    if (!op) return;
    op->setAttribute({{"kernel", std::to_string(K)}});
    op->set_runtime_device(dev, pool);

    // Two rounds through the SAME state tensor: the second round is where a
    // broken state roll-over shows up (the first round reads a zero state).
    for (int round = 0; round < 2; round++) {
        std::vector<float> want(C * S);
        conv1d_ref(x, w, C, S, K, state_ref, want);
        // state_out aliases state_in, like the driver's present==past.
        run(dev, op.get(), {x_t, w_t, st_t}, {y_t, st_t});
        const auto got = readback(pool, y_t);
        for (size_t i = 0; i < got.size(); i++) {
            expect_close("Conv1d.y", i, got[i], want[i]);
        }
        const auto got_state = readback(pool, st_t);
        for (size_t i = 0; i < got_state.size(); i++) {
            expect_close("Conv1d.state", i, got_state[i], state_ref[i]);
        }
    }
    std::printf("PASS: Conv1d [C=%d,S=%d,K=%d] x2 rounds in-place state "
                "(bda=%d)\n", C, S, K, op->bda_pipeline_ready() ? 1 : 0);
}

// ------------------------------------------------------- LinearAttention ---

// HF torch_recurrent_gated_delta_rule, single sequence, CPU side.
void linear_attention_ref(const std::vector<float> &q,
                          const std::vector<float> &k,
                          const std::vector<float> &v,
                          const std::vector<float> &g,
                          const std::vector<float> &beta, int H, int S, int Dk,
                          int Dv, std::vector<float> &state,
                          std::vector<float> &out) {
    const float eps = 1e-6F;
    for (int h = 0; h < H; h++) {
        const size_t sb = static_cast<size_t>(h) * Dk * Dv;
        for (int t = 0; t < S; t++) {
            const size_t qb = (static_cast<size_t>(h) * S + t) * Dk;
            const size_t vb = (static_cast<size_t>(h) * S + t) * Dv;
            // l2 norms
            float qss = 0.0F, kss = 0.0F;
            for (int i = 0; i < Dk; i++) {
                qss += q[qb + i] * q[qb + i];
                kss += k[qb + i] * k[qb + i];
            }
            const float qinv = 1.0F / std::sqrt(qss + eps) / std::sqrt(Dk);
            const float kinv = 1.0F / std::sqrt(kss + eps);
            const float decay = std::exp(g[static_cast<size_t>(h) * S + t]);
            const float bt = beta[static_cast<size_t>(h) * S + t];
            // S = S * decay
            for (int i = 0; i < Dk; i++)
                for (int j = 0; j < Dv; j++)
                    state[sb + i * Dv + j] *= decay;
            // kv[dv] = sum_dk S[dk,dv] * kn[dk]
            std::vector<float> kv(Dv, 0.0F);
            for (int i = 0; i < Dk; i++) {
                const float kn = k[qb + i] * kinv;
                for (int j = 0; j < Dv; j++)
                    kv[j] += state[sb + i * Dv + j] * kn;
            }
            // delta = (v - kv) * beta; S += kn (x) delta; out = S^T qn
            for (int i = 0; i < Dk; i++) {
                const float kn = k[qb + i] * kinv;
                const float qn = q[qb + i] * qinv;
                for (int j = 0; j < Dv; j++) {
                    const float delta = (v[vb + j] - kv[j]) * bt;
                    state[sb + i * Dv + j] += kn * delta;
                }
                (void)qn;
            }
            for (int j = 0; j < Dv; j++) {
                float acc = 0.0F;
                for (int i = 0; i < Dk; i++) {
                    acc += state[sb + i * Dv + j] * (q[qb + i] * qinv);
                }
                out[vb + j] = acc;
            }
        }
    }
}

void test_linear_attention(std::shared_ptr<VulkanDevice> &dev,
                           std::shared_ptr<VulkanCommandPool> &pool) {
    // Two shapes: a small non-trivial one (fast CPU reference), then the real
    // Qwen3.5-27B geometry (Hv=48, Dk=Dv=128, several tokens in prefill).
    struct Shape { int H, S, Dk, Dv; };
    const Shape shapes[] = {{2, 3, 8, 8}, {48, 4, 128, 128}};
    for (const auto &sh : shapes) {
        const int H = sh.H, S = sh.S, Dk = sh.Dk, Dv = sh.Dv;
        std::vector<float> q(H * S * Dk), k(H * S * Dk), v(H * S * Dv),
            g(H * S), beta(H * S), state(H * Dk * Dv, 0.0F);
        for (size_t i = 0; i < q.size(); i++)
            q[i] = std::sin(i * 0.31F) * 0.7F;
        for (size_t i = 0; i < k.size(); i++)
            k[i] = std::cos(i * 0.23F) * 0.5F;
        for (size_t i = 0; i < v.size(); i++)
            v[i] = std::sin(i * 0.17F + 1.0F);
        for (size_t i = 0; i < g.size(); i++)
            g[i] = -0.2F - 0.05F * static_cast<float>(i % 3); // <= 0
        for (size_t i = 0; i < beta.size(); i++)
            beta[i] = 0.5F + 0.1F * static_cast<float>(i % 4);

        auto state_ref = state;
        std::vector<float> want(H * S * Dv, 0.0F);
        linear_attention_ref(q, k, v, g, beta, H, S, Dk, Dv, state_ref, want);

        auto q_t = make<float>(dev, pool, {H, S, Dk}, q);
        auto k_t = make<float>(dev, pool, {H, S, Dk}, k);
        auto v_t = make<float>(dev, pool, {H, S, Dv}, v);
        auto g_t = make<float>(dev, pool, {H, S}, g);
        auto b_t = make<float>(dev, pool, {H, S}, beta);
        auto st_t = make<float>(dev, pool, {H, Dk, Dv}, state);
        auto o_t = make_out<float>(dev, {H, S, Dv}, H * S * Dv);

        auto op =
            vkop::ops::create_from_type(OpType::LINEAR_ATTENTION, 0, 0, true);
        check(op != nullptr, "LinearAttention constructed");
        if (!op) return;
        op->set_runtime_device(dev, pool);

        run(dev, op.get(), {q_t, k_t, v_t, g_t, b_t, st_t}, {o_t, st_t});
        const auto got = readback(pool, o_t);
        for (size_t i = 0; i < got.size(); i++) {
            expect_close("LinearAttention.out", i, got[i], want[i], 2e-3F,
                         1e-5F);
        }
        const auto got_state = readback(pool, st_t);
        for (size_t i = 0; i < got_state.size(); i++) {
            expect_close("LinearAttention.state", i, got_state[i],
                         state_ref[i], 2e-3F, 1e-5F);
        }
        std::printf("PASS: LinearAttention H=%d S=%d Dk=%d Dv=%d in-place "
                    "state (bda=%d)\n", H, S, Dk, Dv,
                    op->bda_pipeline_ready() ? 1 : 0);
    }
}

} // namespace

int main() {
    Logger::getInstance().setLevel(LOG_ERROR);
    vkop::dev::seed_shader_archive_from_build_tree();
    const auto &phydevs =
        VulkanInstance::getVulkanInstance().getPhysicalDevices();
    if (phydevs.empty()) {
        std::printf("SKIP: no Vulkan device\n");
        return 0;
    }
    auto dev = std::make_shared<VulkanDevice>(phydevs[0]);
    auto pool = std::make_shared<VulkanCommandPool>(dev);
    std::printf("GPU: %s, BDA=%d, VKOP_BDA=%s\n", dev->getDeviceName().c_str(),
                dev->is_support_buffer_device_address() ? 1 : 0,
                std::getenv("VKOP_BDA") ? std::getenv("VKOP_BDA") : "(unset)");

    test_cumsum(dev, pool);
    test_conv1d(dev, pool);
    test_linear_attention(dev, pool);

    std::printf(failures ? "RESULT: %d FAILURE(S)\n" : "RESULT: all pass\n",
                failures);
    return failures == 0 ? 0 : 1;
}
