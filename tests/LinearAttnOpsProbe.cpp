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

// float -> storage element. uint16_t is vkop's fp16 storage, so a plain
// static_cast would truncate the value to an integer.
template <typename T> T to_storage(float v) {
    if constexpr (std::is_same_v<T, uint16_t>) {
        return vkop::core::ITensor::fp32_to_fp16(v);
    } else {
        return static_cast<T>(v);
    }
}

template <typename T> float from_storage(T v) {
    if constexpr (std::is_same_v<T, uint16_t>) {
        return vkop::core::ITensor::fp16_to_fp32(v);
    } else {
        return static_cast<float>(v);
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
        bytes[i] = to_storage<T>(vals[i]);
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

template <typename T>
std::vector<float> readback(std::shared_ptr<VulkanCommandPool> &pool,
                            const std::shared_ptr<Tensor<T>> &t) {
    t->copyToCPU(pool);
    const auto &d = t->data();
    std::vector<float> out(d.size());
    for (size_t i = 0; i < d.size(); i++) {
        out[i] = from_storage<T>(d[i]);
    }
    return out;
}

// One CumSum run at a given storage dtype.
template <typename T>
void cumsum_case(std::shared_ptr<VulkanDevice> &dev,
                 std::shared_ptr<VulkanCommandPool> &pool,
                 const std::vector<int> &shape, const std::vector<float> &x,
                 int axis, const std::vector<float> &want, const char *tag,
                 int fp16) {
    auto in = make<T>(dev, pool, shape, x);
    auto out = make_out<T>(dev, shape, x.size());
    auto op = vkop::ops::create_from_type(OpType::CUMSUM, fp16, 0, true);
    check(op != nullptr, "CumSum constructed");
    if (!op) return;
    op->setAttribute({{"axis", std::to_string(axis)}});
    op->set_runtime_device(dev, pool);
    run(dev, op.get(), {in}, {out});
    const auto got = readback<T>(pool, out);
    if (std::getenv("CUM_DEBUG")) {
        for (size_t i = 0; i < got.size() && i < 12; i++)
            std::printf("  [%s axis=%d] i=%zu got=%f want=%f\n", tag, axis, i,
                        got[i], want[i]);
    }
    for (size_t i = 0; i < got.size(); i++) {
        // fp16 storage costs one rounding of the result, so the tolerance is
        // per-element relative rather than the fp32 default.
        expect_close(tag, i, got[i], want[i], fp16 ? 2e-3F : 2e-4F, 1e-3F);
    }
    std::printf("PASS: CumSum %s [2,3,4] axis=%d (bda=%d)\n", tag, axis,
                op->bda_pipeline_ready() ? 1 : 0);
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

    // fp32 and fp16 share the same reference: the kernel accumulates in fp32
    // either way, so the only difference is the storage round-trip.
    cumsum_case<float>(dev, pool, shape, x, axis, want, "fp32", 0);
    cumsum_case<uint16_t>(dev, pool, shape, x, axis, want, "fp16", 1);

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
    cumsum_case<float>(dev, pool, shape, x, axis2, want_m2, "fp32", 0);
    cumsum_case<uint16_t>(dev, pool, shape, x, axis2, want_m2, "fp16", 1);
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

// One Conv1d run at a given DATA dtype. The state stays fp32 in every build,
// which is the contract ops/Conv1d.hpp implements.
template <typename T>
void conv1d_case(std::shared_ptr<VulkanDevice> &dev,
                 std::shared_ptr<VulkanCommandPool> &pool, const char *tag,
                 int fp16) {
    const int C = 6, S = 5, K = 4;
    const int pad = K - 1;
    std::vector<float> w(C * K), x(C * S), state(C * pad, 0.0F);
    for (size_t i = 0; i < w.size(); i++) w[i] = 0.3F + 0.1F * (i % 5);
    for (size_t i = 0; i < x.size(); i++) x[i] = std::cos(i * 0.4F);
    auto state_ref = state;

    auto w_t = make<T>(dev, pool, {C, K}, w);
    auto x_t = make<T>(dev, pool, {C, S}, x);
    auto st_t = make<float>(dev, pool, {C, pad}, state); // state is always fp32
    auto y_t = make_out<T>(dev, {C, S}, C * S);

    auto op = vkop::ops::create_from_type(OpType::CONV1D, fp16, 0, true);
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
        const auto got = readback<T>(pool, y_t);
        for (size_t i = 0; i < got.size(); i++) {
            expect_close(tag, i, got[i], want[i], fp16 ? 2e-3F : 2e-4F, 2e-3F);
        }
        const auto got_state = readback<float>(pool, st_t);
        // The state holds the last K-1 INPUTS, so with fp16 data it carries the
        // rounded inputs while the reference used the fp32 ones. That is the
        // storage's own precision, not an error in the roll-over.
        for (size_t i = 0; i < got_state.size(); i++) {
            expect_close("Conv1d.state", i, got_state[i], state_ref[i],
                         2e-4F, fp16 ? 5e-3F : 1e-5F);
        }
    }
    std::printf("PASS: Conv1d %s [C=%d,S=%d,K=%d] x2 rounds in-place state "
                "(bda=%d)\n", tag, C, S, K,
                op->bda_pipeline_ready() ? 1 : 0);
}

void test_conv1d(std::shared_ptr<VulkanDevice> &dev,
                 std::shared_ptr<VulkanCommandPool> &pool) {
    conv1d_case<float>(dev, pool, "fp32", 0);
    conv1d_case<uint16_t>(dev, pool, "fp16", 1);
}

// ------------------------------------------------------- LinearAttention ---

// HF torch_recurrent_gated_delta_rule, single sequence, CPU side.
void linear_attention_ref(const std::vector<float> &q,
                          const std::vector<float> &k,
                          const std::vector<float> &v,
                          const std::vector<float> &g,
                          const std::vector<float> &beta, int Hk, int Hv,
                          int S, int Dk, int Dv, std::vector<float> &state,
                          std::vector<float> &out) {
    const float eps = 1e-6F;
    const int rep = Hv / Hk;
    for (int h = 0; h < Hv; h++) {
        const int kh = h / rep;  // repeat_interleave's layout
        const size_t sb = static_cast<size_t>(h) * Dk * Dv;
        const size_t kqb = static_cast<size_t>(kh) * S * Dk;
        for (int t = 0; t < S; t++) {
            const size_t qb = kqb + static_cast<size_t>(t) * Dk;
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

template <typename T>
void linear_attention_shapes(std::shared_ptr<VulkanDevice> &dev,
                             std::shared_ptr<VulkanCommandPool> &pool,
                             const char *tag, int fp16) {
    // Two shapes: a small non-trivial one (fast CPU reference), then the real
    // Qwen3.5-27B geometry (Hv=48, Dk=Dv=128, several tokens in prefill).
    // Hk is the key head count; Hv may be larger and is what the op's output
    // uses (the kernel re-derives the key head as h / (Hv/Hk)).
    struct Shape { int Hk, Hv, S, Dk, Dv; };
    const Shape shapes[] = {{2, 2, 3, 8, 8}, {16, 48, 4, 128, 128}};
    for (const auto &sh : shapes) {
        const int Hk = sh.Hk, Hv = sh.Hv, S = sh.S, Dk = sh.Dk, Dv = sh.Dv;
        std::vector<float> q(Hk * S * Dk), k(Hk * S * Dk), v(Hv * S * Dv),
            g(Hv * S), beta(Hv * S), state(Hv * Dk * Dv, 0.0F);
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
        std::vector<float> want(Hv * S * Dv, 0.0F);
        linear_attention_ref(q, k, v, g, beta, Hk, Hv, S, Dk, Dv, state_ref,
                             want);

        // Data follows the dtype under test; the recurrent state is always
        // fp32 (ops/LinearAttention.hpp's contract).
        auto q_t = make<T>(dev, pool, {Hk, S, Dk}, q);
        auto k_t = make<T>(dev, pool, {Hk, S, Dk}, k);
        auto v_t = make<T>(dev, pool, {Hv, S, Dv}, v);
        auto g_t = make<T>(dev, pool, {Hv, S}, g);
        auto b_t = make<T>(dev, pool, {Hv, S}, beta);
        auto st_t = make<float>(dev, pool, {Hv, Dk, Dv}, state);
        auto o_t = make_out<T>(dev, {Hv, S, Dv}, Hv * S * Dv);

        auto op = vkop::ops::create_from_type(OpType::LINEAR_ATTENTION, fp16, 0,
                                             true);
        check(op != nullptr, "LinearAttention constructed");
        if (!op) return;
        op->set_runtime_device(dev, pool);

        run(dev, op.get(), {q_t, k_t, v_t, g_t, b_t, st_t}, {o_t, st_t});
        const auto got = readback<T>(pool, o_t);
        for (size_t i = 0; i < got.size(); i++) {
            // The fp16 build reads half-rounded q/k/v, and the recurrence
            // amplifies that through the state, so its bound is looser.
            expect_close(tag, i, got[i], want[i], fp16 ? 2e-2F : 2e-3F,
                         fp16 ? 5e-3F : 1e-5F);
        }
        const auto got_state = readback<float>(pool, st_t);
        for (size_t i = 0; i < got_state.size(); i++) {
            expect_close("LinearAttention.state", i, got_state[i],
                         state_ref[i], fp16 ? 2e-2F : 2e-3F,
                         fp16 ? 5e-3F : 1e-5F);
        }
        std::printf("PASS: LinearAttention %s Hk=%d Hv=%d S=%d Dk=%d Dv=%d "
                    "in-place state (bda=%d)\n", tag, Hk, Hv, S, Dk, Dv,
                    op->bda_pipeline_ready() ? 1 : 0);
    }
}

void test_linear_attention(std::shared_ptr<VulkanDevice> &dev,
                           std::shared_ptr<VulkanCommandPool> &pool) {
    linear_attention_shapes<float>(dev, pool, "fp32", 0);
    linear_attention_shapes<uint16_t>(dev, pool, "fp16", 1);
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
