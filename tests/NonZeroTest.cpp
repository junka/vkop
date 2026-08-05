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
    for (auto &t : outputs) {
        if (t->dtype() == typeid(int)) {
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
        if (t->dtype() == typeid(int)) {
            vkop::core::as_tensor<int>(t)->copyToCPU(cmdpool);
        }
    }
}

TEST(NonZeroTest, Basic) {
    // Input: [0, 3, 0, 0, 7, 0, 9] -> non-zero at indices 1, 4, 6
    std::vector<float> data = {0.0f, 3.0f, 0.0f, 0.0f, 7.0f, 0.0f, 9.0f};
    int n = 7;
    auto tin = std::make_shared<Tensor<float>>(std::vector<int>{n});
    tin->fillToCPU(data);
    // Output: [count, idx0, idx1, ...] — worst case n+1
    auto tout = std::make_shared<Tensor<int>>(std::vector<int>{n + 1}); tout->fillToCPU(std::vector<int>(n + 1, 0));

    run_op(ops::OpType::NONZERO, {tin}, {tout});

    EXPECT_EQ((*tout)[0], 3);
    EXPECT_EQ((*tout)[1], 1);
    EXPECT_EQ((*tout)[2], 4);
    EXPECT_EQ((*tout)[3], 6);
}

TEST(NonZeroTest, AllNonZero) {
    std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    int n = 5;
    auto tin = std::make_shared<Tensor<float>>(std::vector<int>{n});
    tin->fillToCPU(data);
    auto tout = std::make_shared<Tensor<int>>(std::vector<int>{n + 1}); tout->fillToCPU(std::vector<int>(n + 1, 0));

    run_op(ops::OpType::NONZERO, {tin}, {tout});

    EXPECT_EQ((*tout)[0], n);
    for (int i = 0; i < n; ++i) {
        EXPECT_EQ((*tout)[i + 1], i);
    }
}

TEST(NonZeroTest, AllZero) {
    std::vector<float> data = {0.0f, 0.0f, 0.0f};
    int n = 3;
    auto tin = std::make_shared<Tensor<float>>(std::vector<int>{n});
    tin->fillToCPU(data);
    auto tout = std::make_shared<Tensor<int>>(std::vector<int>{n + 1}); tout->fillToCPU(std::vector<int>(n + 1, 0));

    run_op(ops::OpType::NONZERO, {tin}, {tout});

    EXPECT_EQ((*tout)[0], 0);
}

} // namespace
