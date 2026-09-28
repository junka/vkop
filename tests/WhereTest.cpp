// Copyright 2026 @junka
// Where op tests. Where is SSBO-only (no image path), so every case drives
// create_from_type(..., backend_buffer=true) like BufferRankTest — no env var
// needed for the backend. Two shader/data paths are covered:
//   - fp32 GPU path (buffer_where_spv / where.comp): per-index select, no
//     broadcast in the shader → same-shape cond/X/Y only.
//   - int64 path (the shape-meta chain added in 67e6b10): host-evaluated
//     cpuWhereInt64 by default (VKOP_HOST_SHAPE on), GPU where_int64.comp
//     baseline with VKOP_HOST_SHAPE=0. Both must produce identical bytes, so
//     the same reference works in either mode; host_shape_enabled() caches
//     its env read in a function-local static, so the explicit GPU-baseline
//     TEST below setenv()s before the first call in its process (ctest runs
//     each gtest as its own process; in a single-process run the first int64
//     TEST wins and the rest simply duplicate one path).

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

// ---- wt_* helpers (mirror BufferRankTest's brt_* set) ----------------------

// Named static: Tensor::as_storage_buffer takes the device by non-const
// lvalue reference, so a temporary from a getter will not bind.
std::shared_ptr<vkop::VulkanDevice> &wt_dev() {
    static std::shared_ptr<vkop::VulkanDevice> dev = [] {
        if (!vkop::tests::TestEnv::is_initialized())
            vkop::tests::TestEnv::initialize();
        return vkop::tests::TestEnv::get_device();
    }();
    return dev;
}
std::shared_ptr<vkop::VulkanCommandPool> wt_cmdpool() {
    return vkop::tests::TestEnv::get_command_pool();
}

template <typename T>
std::shared_ptr<Tensor<T>> wt_make(const std::vector<int> &shape,
                                   const std::vector<T> &vals) {
    auto t = std::make_shared<Tensor<T>>(shape);
    t->fillToCPU(vals);
    return t;
}

template <typename T>
void wt_upload(std::shared_ptr<Tensor<T>> &t) {
    // Keep the CPU copy alive for int64: the host-path Where reads data_
    // directly (copyToCPU with host data is a free early-return there).
    auto keep = t->data();
    t->as_storage_buffer(wt_dev());
    t->copyToGPU(wt_cmdpool(), keep.data());
}

template <typename T>
std::shared_ptr<Tensor<T>> wt_make_out(const std::vector<int> &shape) {
    auto t = std::make_shared<Tensor<T>>(shape);
    t->as_storage_buffer(wt_dev());
    t->toGPU();
    return t;
}

std::unique_ptr<vkop::ops::Operator> wt_make_op() {
    auto op = vkop::ops::create_from_type(vkop::ops::OpType::WHERE, 0, 0,
                                          /*backend_buffer=*/true);
    if (!op) {
        LOG_ERROR("create_from_type(WHERE) returned null");
        return nullptr;
    }
    op->set_runtime_device(wt_dev(), wt_cmdpool());
    return op;
}

void wt_run(vkop::ops::Operator *op) {
    auto cmd = op->get_record();
    std::vector<VkSubmitInfo> info{cmd->buildSubmitInfo()};
    vkop::VulkanCommandBuffer::submit(wt_dev()->getComputeQueue(), info);
    cmd->wait();
    wt_dev()->wait_all_done();
}

// ---- fp32 GPU path ----------------------------------------------------------

TEST(WhereTest, FloatSameShapeBuffer) {
    const std::vector<int> shape{2, 3, 4};
    const int n = 2 * 3 * 4;
    std::vector<float> cond(n), xs(n), ys(n), expect(n);
    auto torch_x = torch::randn({n});
    auto torch_y = torch::randn({n});
    auto xc = torch_x.accessor<float, 1>();
    auto yc = torch_y.accessor<float, 1>();
    for (int i = 0; i < n; ++i) {
        cond[i] = (i % 3 == 0) ? 1.0f : 0.0f;
        xs[i] = xc[i];
        ys[i] = yc[i];
        expect[i] = (cond[i] != 0.0f) ? xs[i] : ys[i];
    }

    auto c = wt_make<float>(shape, cond);
    auto x = wt_make<float>(shape, xs);
    auto y = wt_make<float>(shape, ys);
    wt_upload(c);
    wt_upload(x);
    wt_upload(y);
    auto out = wt_make_out<float>(shape);

    auto op = wt_make_op();
    ASSERT_NE(op, nullptr);
    op->onExecute({c, x, y}, {out}, 0);
    wt_run(op.get());
    out->copyToCPU(wt_cmdpool());

    auto o = out->data();
    for (int i = 0; i < n; ++i)
        ASSERT_NEAR(o[i], expect[i], 1e-5f) << "fp32 Where mismatch at " << i;
}

// ---- int64 shape-meta path (host mode by default; GPU baseline below) ------

// cond/X/Y same shape, large and negative values: the int64 path is a
// verbatim 8-byte copy (no float truncation allowed).
bool wt_int64_same_shape_case() {
    const std::vector<int> shape{5};
    std::vector<int64_t> cond{1, 0, 1, 0, 1};
    std::vector<int64_t> xs{1, 2, -999, 4, 5000000000LL};
    std::vector<int64_t> ys{10, 20, 30, 40, -70000000000LL};
    std::vector<int64_t> expect(5);
    for (int i = 0; i < 5; ++i)
        expect[i] = cond[i] ? xs[i] : ys[i];

    auto c = wt_make<int64_t>(shape, cond);
    auto x = wt_make<int64_t>(shape, xs);
    auto y = wt_make<int64_t>(shape, ys);
    wt_upload(c);
    wt_upload(x);
    wt_upload(y);
    auto out = wt_make_out<int64_t>(shape);

    auto op = wt_make_op();
    if (!op)
        return false;
    op->onExecute({c, x, y}, {out}, 0);
    wt_run(op.get());
    out->copyToCPU(wt_cmdpool());

    auto o = out->data();
    for (int i = 0; i < 5; ++i) {
        if (o[i] != expect[i]) {
            LOG_ERROR("int64 Where mismatch at %d: %lld vs %lld", i,
                      static_cast<long long>(o[i]),
                      static_cast<long long>(expect[i]));
            return false;
        }
    }
    return true;
}

TEST(WhereTest, Int64SameShape) {
    EXPECT_TRUE(wt_int64_same_shape_case());
}

// cond [1,3] broadcasts against X/Y [2,3] → out [2,3]. Where recomputes the
// broadcast output shape from the live inputs (the recorded shape is stale in
// the meta-chain), so the output tensor is created at the resolved shape.
bool wt_int64_broadcast_case() {
    const std::vector<int> cshape{1, 3}, xyshape{2, 3};
    std::vector<int64_t> cond{1, 0, 1};
    std::vector<int64_t> xs{11, 12, 13, 14, 15, 16};
    std::vector<int64_t> ys{21, 22, 23, 24, 25, 26};
    std::vector<int64_t> expect(6);
    for (int i = 0; i < 6; ++i) {
        int ci = i % 3; // row broadcast over dim 0 of cond (size 1)
        expect[i] = cond[ci] ? xs[i] : ys[i];
    }

    auto c = wt_make<int64_t>(cshape, cond);
    auto x = wt_make<int64_t>(xyshape, xs);
    auto y = wt_make<int64_t>(xyshape, ys);
    wt_upload(c);
    wt_upload(x);
    wt_upload(y);
    auto out = wt_make_out<int64_t>(xyshape);

    auto op = wt_make_op();
    if (!op)
        return false;
    op->onExecute({c, x, y}, {out}, 0);
    wt_run(op.get());
    out->copyToCPU(wt_cmdpool());

    auto o = out->data();
    for (int i = 0; i < 6; ++i) {
        if (o[i] != expect[i]) {
            LOG_ERROR("int64 broadcast Where mismatch at %d: %lld vs %lld", i,
                      static_cast<long long>(o[i]),
                      static_cast<long long>(expect[i]));
            return false;
        }
    }
    return true;
}

TEST(WhereTest, Int64CondBroadcast) {
    EXPECT_TRUE(wt_int64_broadcast_case());
}

// GPU where_int64.comp baseline (the VKOP_HOST_SHAPE=0 A/B mode). Setenv
// before the first host_shape_enabled() call in this process wins; the other
// two int64 TESTs are path-agnostic (both modes must match the same
// reference), so an order where this runs last is harmless.
TEST(WhereTest, Int64GpuShaderBaseline) {
    setenv("VKOP_HOST_SHAPE", "0", 1);
    EXPECT_TRUE(wt_int64_same_shape_case());
}

} // namespace
