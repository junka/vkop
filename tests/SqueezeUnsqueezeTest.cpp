// Copyright 2026 @junka
// Squeeze / Unsqueeze runtime view-op tests (ops/SqueezeUnsqueeze.hpp).
// Pure shape-metadata ops: element bytes are unchanged, only the rank/shape
// differs. Branches covered here:
//   - axes from input[1] (opset>=13, int64) vs the "axes" attribute
//     (opset<13, bracketed "[a, b]" serialization — see BufferRankTest's
//     Transpose attr note).
//   - Squeeze: explicit axes, negative axes, empty axes (remove all size-1
//     dims).
//   - Unsqueeze: negatives normalize against the OUTPUT rank (the
//     Unsqueeze([1,1,20], axes=[-1]) -> [1,1,20,1] rule that misordered the
//     rotary ScatterND index concat when it used the input rank).
//   - GPU-alias fast path (input has an SSBO → output shares the VkBuffer)
//     vs the host-only byte-copy fallback.
//   - int64 data (the rotary shape-meta views).
//   - the axes auto-learning cache (LEARNING → CONFIRMING → STABLE) across
//     three executes on ONE operator instance — STABLE rounds skip the
//     axes copyToCPU readback and must still produce the right view.
//
// Data runs under backend_buffer=true so float inputs are SSBOs (the alias
// path needs input->has_gpu_buffer()); the host-only case deliberately skips
// the upload.

#include <cstdint>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

#include "setup.hpp"
#include "core/Tensor.hpp"
#include "include/logger.hpp"
#include "ops/OperatorFactory.hpp"
#include "ops/Ops.hpp"

#include <gtest/gtest.h>
#include <torch/torch.h>

using vkop::core::Tensor;

namespace {

// ---- su_* helpers -----------------------------------------------------------

// Named static: Tensor::as_storage_buffer takes the device by non-const
// lvalue reference, so a temporary from a getter will not bind.
std::shared_ptr<vkop::VulkanDevice> &su_dev() {
    static std::shared_ptr<vkop::VulkanDevice> dev = [] {
        if (!vkop::tests::TestEnv::is_initialized())
            vkop::tests::TestEnv::initialize();
        return vkop::tests::TestEnv::get_device();
    }();
    return dev;
}
std::shared_ptr<vkop::VulkanCommandPool> su_cmdpool() {
    return vkop::tests::TestEnv::get_command_pool();
}

template <typename T>
std::shared_ptr<Tensor<T>> su_make(const std::vector<int> &shape,
                                   const std::vector<T> &vals) {
    auto t = std::make_shared<Tensor<T>>(shape);
    t->fillToCPU(vals);
    return t;
}

template <typename T>
void su_upload(std::shared_ptr<Tensor<T>> &t) {
    auto keep = t->data();
    t->as_storage_buffer(su_dev());
    t->copyToGPU(su_cmdpool(), keep.data());
}

// int64 axes input tensor (opset 13). Empty vector → shape {0} (ONNX
// "axes omitted": Squeeze removes all size-1 dims).
std::shared_ptr<Tensor<int64_t>> su_axes(const std::vector<int64_t> &axes) {
    auto t = std::make_shared<Tensor<int64_t>>(
        std::vector<int>{static_cast<int>(axes.size())});
    t->fillToCPU(axes);
    su_upload(t);
    return t;
}

std::string su_axes_attr(const std::vector<int> &axes) {
    std::string s = "[";
    for (size_t i = 0; i < axes.size(); ++i) {
        if (i)
            s += ", ";
        s += std::to_string(axes[i]);
    }
    return s + "]";
}

std::unique_ptr<vkop::ops::Operator> su_make_op(vkop::ops::OpType type) {
    auto op = vkop::ops::create_from_type(type, 0, 0, /*backend_buffer=*/true);
    if (!op) {
        LOG_ERROR("create_from_type(%d) returned null",
                  static_cast<int>(type));
        return nullptr;
    }
    op->set_runtime_device(su_dev(), su_cmdpool());
    return op;
}

void su_run(vkop::ops::Operator *op) {
    auto cmd = op->get_record();
    std::vector<VkSubmitInfo> info{cmd->buildSubmitInfo()};
    vkop::VulkanCommandBuffer::submit(su_dev()->getComputeQueue(), info);
    cmd->wait();
    su_dev()->wait_all_done();
}

// One view case: run `rounds` executes on the SAME operator instance
// (rounds>1 exercises the axes auto-learning state machine), then check the
// last output's shape and element bytes against the input.
template <typename T>
bool su_view_case(vkop::ops::OpType type, const std::vector<int> &in_shape,
                  const std::vector<T> &in_vals,
                  const std::vector<int> &expect_shape,
                  const std::vector<int64_t> &axes_vals,
                  const std::string &axes_attr, int rounds = 1) {
    auto data = su_make<T>(in_shape, in_vals);
    su_upload(data);

    std::vector<std::shared_ptr<vkop::core::ITensor>> ins{data};
    if (!axes_attr.empty()) {
        // opset<13 form: attribute only, no axes input.
    } else {
        auto ax = su_axes(axes_vals);
        ins.push_back(ax);
    }

    auto op = su_make_op(type);
    if (!op)
        return false;
    if (!axes_attr.empty())
        op->setAttribute({{"axes", axes_attr}});

    // Keep every round's output alive until the case ends: on the alias path
    // they share the input's VkBuffer, and freeing a mid-round output while
    // later rounds still read the input would exercise the shared-buffer
    // destructor ordering.
    std::vector<std::shared_ptr<Tensor<T>>> outs;
    std::shared_ptr<Tensor<T>> out;
    for (int r = 0; r < rounds; ++r) {
        // Pre-size the output at the INPUT shape: only the op's own
        // resize(out_shape) may fix it — a test that "helpfully" resized to
        // the expectation would not catch a wrong out_shape.
        out = std::make_shared<Tensor<T>>(in_shape);
        out->as_storage_buffer(su_dev());
        out->toGPU();
        outs.push_back(out);
        op->onExecute(ins, {out}, 0);
        su_run(op.get());
        out->copyToCPU(su_cmdpool());
    }

    auto got_shape = out->getShape();
    if (got_shape != expect_shape) {
        LOG_ERROR("view shape mismatch: got %zu dims, want %zu dims",
                  got_shape.size(), expect_shape.size());
        for (size_t i = 0; i < got_shape.size(); ++i)
            LOG_ERROR("  got[%zu]=%d", i, got_shape[i]);
        return false;
    }
    const auto &o = out->data();
    if (o.size() != in_vals.size()) {
        LOG_ERROR("view element count mismatch: %zu vs %zu", o.size(),
                  in_vals.size());
        return false;
    }
    for (size_t i = 0; i < o.size(); ++i) {
        if constexpr (std::is_same_v<T, float>) {
            if (o[i] != in_vals[i]) {
                LOG_ERROR("view byte mismatch at %zu: %f vs %f", i, o[i],
                          in_vals[i]);
                return false;
            }
        } else {
            if (o[i] != in_vals[i]) {
                LOG_ERROR("view mismatch at %zu: %lld vs %lld", i,
                          static_cast<long long>(o[i]),
                          static_cast<long long>(in_vals[i]));
                return false;
            }
        }
    }
    return true;
}

// ---- Squeeze ----------------------------------------------------------------

TEST(SqueezeUnsqueezeTest, SqueezeAxesInputOpset13) {
    // [1,8,1,128] axes {0,2} → [8,128] (rotary-style 4-D squeeze, alias path).
    const std::vector<int> shape{1, 8, 1, 128};
    std::vector<float> vals(shape[1] * shape[3]);
    for (size_t i = 0; i < vals.size(); ++i)
        vals[i] = static_cast<float>(i) * 0.25f - 3.0f;
    EXPECT_TRUE(su_view_case<float>(vkop::ops::OpType::SQUEEZE, shape, vals,
                                    {8, 128}, {0, 2}, ""));
}

TEST(SqueezeUnsqueezeTest, SqueezeNegativeAxes) {
    // [4,1,1] axes {-1} → [4,1]: negative normalizes against the INPUT rank
    // (removes the last dim only, not every size-1 dim).
    const std::vector<int> shape{4, 1, 1};
    std::vector<float> vals{1.f, 2.f, 3.f, 4.f};
    EXPECT_TRUE(su_view_case<float>(vkop::ops::OpType::SQUEEZE, shape, vals,
                                    {4, 1}, {-1}, ""));
}

TEST(SqueezeUnsqueezeTest, SqueezeEmptyAxesRemovesAllOnes) {
    // Empty axes input → remove ALL size-1 dims: [1,1,6,1] → [6].
    const std::vector<int> shape{1, 1, 6, 1};
    std::vector<float> vals{10.f, 20.f, 30.f, 40.f, 50.f, 60.f};
    EXPECT_TRUE(su_view_case<float>(vkop::ops::OpType::SQUEEZE, shape, vals,
                                    {6}, {}, ""));
}

TEST(SqueezeUnsqueezeTest, SqueezeAxesAttributeOpset12) {
    // opset<13: axes via attribute, single input.
    const std::vector<int> shape{1, 3, 1, 4};
    std::vector<float> vals(12);
    for (size_t i = 0; i < vals.size(); ++i)
        vals[i] = static_cast<float>(i);
    EXPECT_TRUE(su_view_case<float>(vkop::ops::OpType::SQUEEZE, shape, vals,
                                    {3, 4}, {}, su_axes_attr({0, 2})));
}

// ---- Unsqueeze ---------------------------------------------------------------

TEST(SqueezeUnsqueezeTest, UnsqueezeNegativeAxisUsesOutputRank) {
    // The documented regression: Unsqueeze([1,1,20], axes=[-1]) must give
    // [1,1,20,1] (out-rank normalization), NOT [1,1,1,20].
    const std::vector<int> shape{1, 1, 20};
    std::vector<float> vals(20);
    for (size_t i = 0; i < vals.size(); ++i)
        vals[i] = static_cast<float>(i) + 0.5f;
    EXPECT_TRUE(su_view_case<float>(vkop::ops::OpType::UNSQUEEZE, shape, vals,
                                    {1, 1, 20, 1}, {-1}, ""));
}

TEST(SqueezeUnsqueezeTest, UnsqueezeRotaryShape) {
    // [8,128] axes {0,2} → [1,8,1,128].
    const std::vector<int> shape{8, 128};
    std::vector<float> vals(8 * 128);
    auto torch_vals = torch::randn({static_cast<int64_t>(vals.size())});
    auto acc = torch_vals.accessor<float, 1>();
    for (size_t i = 0; i < vals.size(); ++i)
        vals[i] = acc[i];
    EXPECT_TRUE(su_view_case<float>(vkop::ops::OpType::UNSQUEEZE, shape, vals,
                                    {1, 8, 1, 128}, {0, 2}, ""));
}

TEST(SqueezeUnsqueezeTest, SqueezeInt64Data) {
    // int64 shape-meta view: [2,1,3] axes {1} → [2,3], values verbatim
    // (64-bit must survive the alias byte-for-byte).
    const std::vector<int> shape{2, 1, 3};
    std::vector<int64_t> vals{1, -2, 3000000000LL, 4, 5, -60000000000LL};
    EXPECT_TRUE(su_view_case<int64_t>(vkop::ops::OpType::SQUEEZE, shape, vals,
                                      {2, 3}, {1}, ""));
}

// ---- branch coverage ---------------------------------------------------------

TEST(SqueezeUnsqueezeTest, HostOnlyInputFallsBackToByteCopy) {
    // No upload → input has no vkobj_ → the CPU-fallback branch (copyToCPU +
    // fillToCPU + copyToGPUDeferred), not the alias path.
    const std::vector<int> shape{1, 5, 1};
    std::vector<float> vals{1.f, 3.f, 5.f, 7.f, 9.f};

    auto data = su_make<float>(shape, vals); // host-only: NOT uploaded

    auto ax = su_axes({0, 2});
    auto out = std::make_shared<Tensor<float>>(shape);
    out->as_storage_buffer(su_dev());
    out->toGPU();

    auto op = su_make_op(vkop::ops::OpType::SQUEEZE);
    ASSERT_NE(op, nullptr);
    op->onExecute({data, ax}, {out}, 0);
    su_run(op.get());
    out->copyToCPU(su_cmdpool());

    EXPECT_EQ(out->getShape(), std::vector<int>({5}));
    const auto &o = out->data();
    ASSERT_EQ(o.size(), vals.size());
    for (size_t i = 0; i < vals.size(); ++i)
        EXPECT_FLOAT_EQ(o[i], vals[i]) << "host-fallback mismatch at " << i;
}

TEST(SqueezeUnsqueezeTest, AxesCacheStableAcrossRounds) {
    // Three executes on ONE op instance: round0 LEARNING, round1 CONFIRMING
    // (axes match → STABLE), round2 skips the axes readback and must reuse
    // the cached axes. Same expectation each round.
    const std::vector<int> shape{1, 6, 1, 7};
    std::vector<float> vals(42);
    for (size_t i = 0; i < vals.size(); ++i)
        vals[i] = static_cast<float>(i) * 1.5f;
    EXPECT_TRUE(su_view_case<float>(vkop::ops::OpType::SQUEEZE, shape, vals,
                                    {6, 7}, {0, 2}, "", /*rounds=*/3));
}

} // namespace
