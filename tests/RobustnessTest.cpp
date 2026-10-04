// Copyright 2026 @junka
// Robustness tests: stress the runtime with "destructive" inputs that don't
// necessarily have a valid mathematical result but must not crash or corrupt
// memory. Covers empty tensors, rank-0 scalars, NaN/Inf values, and illegal
// axis indices.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <exception>
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

// ---- rt_* helpers (mirror BufferRankTest's brt_* set) ----------------------

std::shared_ptr<vkop::VulkanDevice> &rt_dev() {
    static std::shared_ptr<vkop::VulkanDevice> dev = [] {
        if (!vkop::tests::TestEnv::is_initialized())
            vkop::tests::TestEnv::initialize();
        return vkop::tests::TestEnv::get_device();
    }();
    return dev;
}
std::shared_ptr<vkop::VulkanCommandPool> rt_cmdpool() {
    return vkop::tests::TestEnv::get_command_pool();
}

template <typename T>
std::shared_ptr<Tensor<T>> rt_make(const std::vector<int> &shape,
                                   const std::vector<T> &vals) {
    auto t = std::make_shared<Tensor<T>>(shape);
    t->fillToCPU(vals);
    return t;
}

template <typename T>
void rt_upload(std::shared_ptr<Tensor<T>> &t) {
    auto keep = t->data();
    t->as_storage_buffer(rt_dev());
    t->copyToGPU(rt_cmdpool(), keep.data());
}

std::unique_ptr<vkop::ops::Operator> rt_make_op(vkop::ops::OpType type) {
    auto op = vkop::ops::create_from_type(type, 0, 0, /*backend_buffer=*/true);
    if (!op) {
        LOG_ERROR("create_from_type(%d) returned null",
                  static_cast<int>(type));
        return nullptr;
    }
    op->set_runtime_device(rt_dev(), rt_cmdpool());
    return op;
}

void rt_run(vkop::ops::Operator *op) {
    auto cmd = op->get_record();
    std::vector<VkSubmitInfo> info{cmd->buildSubmitInfo()};
    vkop::VulkanCommandBuffer::submit(rt_dev()->getComputeQueue(), info);
    cmd->wait();
    rt_dev()->wait_all_done();
}

// The scatter guards must throw for the reason they claim, so the message is
// part of the assertion (EXPECT_THROW alone would pass on any unrelated error).
template <typename Fn>
void expect_throw_contains(Fn &&fn, const std::string &needle) {
    try {
        fn();
    } catch (const std::exception &e) {
        EXPECT_NE(std::string(e.what()).find(needle), std::string::npos)
            << "message was: " << e.what();
        return;
    }
    ADD_FAILURE() << "expected a throw containing \"" << needle << "\"";
}

// ---- Empty / Rank-0 --------------------------------------------------------

TEST(RobustnessTest, ConcatEmptyInput) {
    // [1, 0] + [1, 3] along axis 1 → [1, 3]. Should not crash on 0-dim input.
    auto in1 = rt_make<float>({1, 0}, {});
    auto in2 = rt_make<float>({1, 3}, {1.f, 2.f, 3.f});
    rt_upload(in1);
    rt_upload(in2);
    auto out = std::make_shared<Tensor<float>>(std::vector<int>{1, 3});
    out->as_storage_buffer(rt_dev());
    out->toGPU();

    auto op = rt_make_op(vkop::ops::OpType::CONCAT);
    ASSERT_NE(op, nullptr);
    op->onExecute({in1, in2}, {out}, 0);
    rt_run(op.get());
    out->copyToCPU(rt_cmdpool());

    EXPECT_EQ(out->num_elements(), 3);
    const auto &o = out->data();
    EXPECT_FLOAT_EQ(o[0], 1.f);
    EXPECT_FLOAT_EQ(o[2], 3.f);
}

TEST(RobustnessTest, ReshapeToScalar) {
    // [1] → [] (rank-0). The output tensor should have size 1 but num_dims 0.
    auto in = rt_make<float>({1}, {42.f});
    rt_upload(in);
    // Pre-size to rank-0 shape: empty vector.
    auto out = std::make_shared<Tensor<float>>(std::vector<int>{});
    out->as_storage_buffer(rt_dev());
    out->toGPU();

    auto op = rt_make_op(vkop::ops::OpType::RESHAPE);
    ASSERT_NE(op, nullptr);
    // Provide shape input for rank-0: an empty int64 tensor.
    auto shape_in = std::make_shared<Tensor<int64_t>>(std::vector<int>{0});
    shape_in->fillToCPU({});
    rt_upload(shape_in);

    op->onExecute({in, shape_in}, {out}, 0);
    rt_run(op.get());
    out->copyToCPU(rt_cmdpool());

    EXPECT_EQ(out->getShape().size(), 0u);
    EXPECT_EQ(out->num_elements(), 1);
    EXPECT_FLOAT_EQ(out->data()[0], 42.f);
}

// ---- NaN / Inf -------------------------------------------------------------

TEST(RobustnessTest, MatMulWithNaN) {
    // If one element is NaN, the whole row/col product usually becomes NaN.
    // We just verify it doesn't crash and propagates NaN.
    std::vector<float> vals_a{1.f, std::nanf("")};
    std::vector<float> vals_b{2.f, 3.f};
    auto a = rt_make<float>({1, 2}, vals_a);
    auto b = rt_make<float>({2, 1}, vals_b);
    rt_upload(a);
    rt_upload(b);
    auto out = std::make_shared<Tensor<float>>(std::vector<int>{1, 1});
    out->as_storage_buffer(rt_dev());
    out->toGPU();

    auto op = rt_make_op(vkop::ops::OpType::MATMUL);
    ASSERT_NE(op, nullptr);
    op->onExecute({a, b}, {out}, 0);
    rt_run(op.get());
    out->copyToCPU(rt_cmdpool());

    EXPECT_TRUE(std::isnan(out->data()[0]));
}

TEST(RobustnessTest, ReluWithInf) {
    // -Inf should stay -Inf (ReLU output is 0 for x < 0, but -Inf is a special
    // case in some implementations; we just verify it doesn't crash and
    // produces a finite or well-defined result). +Inf should stay +Inf.
    auto in = rt_make<float>({2}, {-std::numeric_limits<float>::infinity(),
                                   std::numeric_limits<float>::infinity()});
    rt_upload(in);
    auto out = std::make_shared<Tensor<float>>(std::vector<int>{2});
    out->as_storage_buffer(rt_dev());
    out->toGPU();

    auto op = rt_make_op(vkop::ops::OpType::RELU);
    ASSERT_NE(op, nullptr);
    op->onExecute({in}, {out}, 0);
    rt_run(op.get());
    out->copyToCPU(rt_cmdpool());

    // ReLU(-Inf) is mathematically 0.0 (since -Inf < 0).
    EXPECT_FLOAT_EQ(out->data()[0], 0.f);
    EXPECT_EQ(out->data()[1], std::numeric_limits<float>::infinity());
}

// ---- Illegal Axis ----------------------------------------------------------

TEST(RobustnessTest, TransposeIllegalAxis) {
    // For a rank-2 tensor [2, 3], perm [2, 0] is out of bounds.
    // Current implementation might segfault or produce garbage. We expect it
    // to either throw or at least not crash the process.
    auto in = rt_make<float>({2, 3}, {1.f, 2.f, 3.f, 4.f, 5.f, 6.f});
    rt_upload(in);
    auto out = std::make_shared<Tensor<float>>(std::vector<int>{2, 3});
    out->as_storage_buffer(rt_dev());
    out->toGPU();

    auto op = rt_make_op(vkop::ops::OpType::TRANSPOSE);
    ASSERT_NE(op, nullptr);
    op->setAttribute({{"perm", "[2, 0]"}});

    // In a robust implementation this should throw. Currently it may just
    // access OOB memory. We wrap in EXPECT_NO_THROW to document current
    // behavior, but ideally this should be changed to EXPECT_THROW.
    EXPECT_NO_THROW({
        op->onExecute({in}, {out}, 0);
        rt_run(op.get());
    });
}

// ---- Scatter: shape contract + out-of-range writes -------------------------
// The converter's view-fold pass once deleted a rank-increasing Unsqueeze
// without fixing up the consumer, so ScatterND read indices.shape[-1] as
// index_rank against the *pre-view* rank: every scatter target landed outside
// the output buffer and fp32 rope values were written into the neighbouring
// fp16 weight blob (silent, decode all-NaN). Nothing downstream can detect that
// pattern, so both scatter ops now check their shapes on the host and drop
// out-of-range writes in the shader. These are the tests for that.

// data/indices/updates with the exact /rotary_emb shapes that caused it.
TEST(RobustnessTest, ScatterNDIndexRankOverflowThrows) {
    auto data = rt_make<float>({1, 1, 64}, std::vector<float>(64, 0.f));
    auto indices = rt_make<int64_t>({1, 1, 21}, std::vector<int64_t>(21, 0));
    auto updates = rt_make<float>({1, 1, 20}, std::vector<float>(20, 0.f));
    auto out = std::make_shared<Tensor<float>>(std::vector<int>{1, 1, 64});

    auto op = rt_make_op(vkop::ops::OpType::SCATTER_ND);
    ASSERT_NE(op, nullptr);
    expect_throw_contains(
        [&] { op->onExecute({data, indices, updates}, {out}, 0); },
        "index_rank=21 num_tuples=1");
}

// index_rank in range, but updates does not match
// indices.shape[:-1] + data.shape[index_rank:].
TEST(RobustnessTest, ScatterNDUpdatesSizeMismatchThrows) {
    auto data = rt_make<float>({1, 1, 64}, std::vector<float>(64, 0.f));
    // [2,1,1,3] -> index_rank 3, num_tuples 2, slice_size 1 => 2 updates elems
    auto indices = rt_make<int64_t>({2, 1, 1, 3}, std::vector<int64_t>(6, 0));
    auto updates = rt_make<float>({3, 1, 1}, std::vector<float>(3, 0.f));
    auto out = std::make_shared<Tensor<float>>(std::vector<int>{1, 1, 64});

    auto op = rt_make_op(vkop::ops::OpType::SCATTER_ND);
    ASSERT_NE(op, nullptr);
    expect_throw_contains(
        [&] { op->onExecute({data, indices, updates}, {out}, 0); },
        "updates_total=3");
}

// The op takes the (row, col) split width from indices and reads updates at the
// same flat gid, so indices and updates must agree on rank and size.
TEST(RobustnessTest, ScatterElementsRankMismatchThrows) {
    auto data = rt_make<float>({4}, {1.f, 2.f, 3.f, 4.f});
    auto indices = rt_make<int64_t>({1, 4}, {0, 0, 1, 1});
    auto updates = rt_make<float>({4}, {5.f, 6.f, 7.f, 8.f});
    auto out = std::make_shared<Tensor<float>>(std::vector<int>{4});

    auto op = rt_make_op(vkop::ops::OpType::SCATTER_ELEMENTS);
    ASSERT_NE(op, nullptr);
    expect_throw_contains(
        [&] { op->onExecute({data, indices, updates}, {out}, 0); },
        "equal rank, equal trailing");
}

TEST(RobustnessTest, ScatterNDTargetOutOfRangeDropped) {
    // Valid shapes, but the second tuple indexes past the end of data. The
    // in-range write must land and the stray one must not touch anything else.
    std::vector<float> vals(64);
    for (int i = 0; i < 64; ++i)
        vals[i] = static_cast<float>(i);
    auto data = rt_make<float>({1, 1, 64}, vals);
    auto indices = rt_make<int64_t>({2, 1, 1, 3}, {0, 0, 3, 0, 0, 999});
    auto updates = rt_make<float>({2, 1, 1}, {100.f, 200.f});
    rt_upload(data);
    rt_upload(indices);
    rt_upload(updates);
    auto out = std::make_shared<Tensor<float>>(std::vector<int>{1, 1, 64});
    out->as_storage_buffer(rt_dev());
    out->toGPU();

    auto op = rt_make_op(vkop::ops::OpType::SCATTER_ND);
    ASSERT_NE(op, nullptr);
    op->onExecute({data, indices, updates}, {out}, 0);
    rt_run(op.get());
    out->copyToCPU(rt_cmdpool());

    EXPECT_FLOAT_EQ(out->data()[3], 100.f);
    for (int i = 0; i < 64; ++i) {
        if (i == 3)
            continue;
        EXPECT_FLOAT_EQ(out->data()[i], static_cast<float>(i)) << "at " << i;
    }
}

TEST(RobustnessTest, ScatterElementsTargetOutOfRangeDropped) {
    // data [2,2], cols 2: rows 0 and 9 -> targets 0,1 (in range) and 18,19
    // (out of range, dropped). Output is seeded from data, so the untouched
    // elements keep their original values.
    auto data = rt_make<float>({2, 2}, {1.f, 2.f, 3.f, 4.f});
    auto indices = rt_make<int64_t>({2, 2}, {0, 0, 9, 9});
    auto updates = rt_make<float>({2, 2}, {7.f, 8.f, 9.f, 10.f});
    rt_upload(data);
    rt_upload(indices);
    rt_upload(updates);
    auto out = std::make_shared<Tensor<float>>(std::vector<int>{2, 2});
    out->as_storage_buffer(rt_dev());
    out->toGPU();

    auto op = rt_make_op(vkop::ops::OpType::SCATTER_ELEMENTS);
    ASSERT_NE(op, nullptr);
    op->setAttribute({{"axis", "0"}, {"reduction", "none"}});
    op->onExecute({data, indices, updates}, {out}, 0);
    rt_run(op.get());
    out->copyToCPU(rt_cmdpool());

    const std::vector<float> ref{7.f, 8.f, 3.f, 4.f};
    for (int i = 0; i < 4; ++i) {
        EXPECT_FLOAT_EQ(out->data()[i], ref[i]) << "at " << i;
    }
}

} // namespace
