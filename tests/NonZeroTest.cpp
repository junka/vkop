// Copyright 2025 @junka
#include "setup.hpp"
#include "core/Tensor.hpp"
#include "ops/OperatorFactory.hpp"
#include "ops/Ops.hpp"
#include <gtest/gtest.h>
#include <torch/torch.h>

using vkop::core::Tensor;
using vkop::tests::TestCase;
namespace ops = vkop::ops;

// NonZero has a dynamic output size (depends on the number of non-zero
// elements), so it can't use TestCase::run_test which pre-allocates the
// output from expect_outputs->getShape(). Instead it uses the direct
// construction pattern (like BufferBackendTest).
//
// The op emits an int64 tensor of shape [rank, count] (ONNX column-major:
// the k-th nonzero's coordinates occupy out[:, k]). For a rank-1 input this
// is [1, count] == a flat [count] of linear indices.

namespace {

static void run_op(ops::OpType type,
                   const std::vector<std::shared_ptr<vkop::core::ITensor>> &inputs,
                   const std::vector<std::shared_ptr<vkop::core::ITensor>> &outputs) {
    auto dev = vkop::tests::TestEnv::get_device();
    auto cmdpool = vkop::tests::TestEnv::get_command_pool();
    auto op = ops::create_from_type(type, 0, 0, true);
    op->set_runtime_device(dev, cmdpool);
    for (auto &t : inputs) {
        if (!t || t->dtype() == typeid(int64_t)) continue;
        if (t->dtype() == typeid(float)) {
            vkop::core::as_tensor<float>(t)->as_storage_buffer(dev);
            vkop::core::as_tensor<float>(t)->copyToGPU(cmdpool);
        } else if (t->dtype() == typeid(int)) {
            vkop::core::as_tensor<int>(t)->as_storage_buffer(dev);
            vkop::core::as_tensor<int>(t)->copyToGPU(cmdpool);
        }
    }
    op->onExecute(inputs, outputs, 0);
    auto cmd = op->get_record();
    std::vector<VkSubmitInfo> info{cmd->buildSubmitInfo()};
    vkop::VulkanCommandBuffer::submit(dev->getComputeQueue(), info);
    cmd->wait();
    dev->wait_all_done();
    for (auto &t : outputs) {
        if (t->dtype() == typeid(int64_t)) {
            auto out = vkop::core::as_tensor<int64_t>(t);
            out->copyToCPU(cmdpool);
        }
    }
}

// Rank-1 helper: returns the linear indices of the non-zero elements of a
// 1-D float input. The op writes [1, count] int64 which, flattened, is exactly
// [idx0, idx1, ...] (count entries, no leading count slot).
static std::vector<int64_t> nonzero_indices(const std::vector<float> &data) {
    int n = static_cast<int>(data.size());
    auto tin = std::make_shared<Tensor<float>>(std::vector<int>{n});
    tin->fillToCPU(data);
    // Output shape is [rank, count] = [1, count]; count is only known after the
    // scan, so pre-allocate worst case [1, n] and let the op resize.
    auto tout = std::make_shared<Tensor<int64_t>>(std::vector<int>{1, n});
    tout->fillToCPU(std::vector<int64_t>(n, 0));

    run_op(ops::OpType::NONZERO, {tin}, {tout});

    auto out = tout->getShape();
    int count = (out.size() >= 2) ? out[1] : 0;
    std::vector<int64_t> result(count);
    for (int i = 0; i < count; ++i) {
        result[i] = (*tout)[i];
    }
    return result;
}

TEST(NonZeroTest, Basic) {
    // Input: [0, 3, 0, 0, 7, 0, 9] -> non-zero at indices 1, 4, 6
    std::vector<float> data = {0.0f, 3.0f, 0.0f, 0.0f, 7.0f, 0.0f, 9.0f};
    auto idx = nonzero_indices(data);
    ASSERT_EQ(idx.size(), 3u);
    EXPECT_EQ(idx[0], 1);
    EXPECT_EQ(idx[1], 4);
    EXPECT_EQ(idx[2], 6);
}

TEST(NonZeroTest, AllNonZero) {
    std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    auto idx = nonzero_indices(data);
    ASSERT_EQ(idx.size(), data.size());
    for (size_t i = 0; i < data.size(); ++i) {
        EXPECT_EQ(idx[i], static_cast<int64_t>(i));
    }
}

TEST(NonZeroTest, AllZero) {
    std::vector<float> data = {0.0f, 0.0f, 0.0f};
    auto idx = nonzero_indices(data);
    EXPECT_TRUE(idx.empty());
}

} // namespace
