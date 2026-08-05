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

// ScatterElements has an in-place semantic (data is both input and output),
// and the reduction='add' path needs the initial data on the GPU. These
// don't fit TestCase::run_test's "create fresh output from expected shape"
// model. We use the direct construction pattern instead, mirroring the
// setup.hpp run_test flow but with the data tensor passed as both input
// and output.

namespace {

static void run_scatter(
    const std::string &reduction,
    const std::vector<std::shared_ptr<vkop::core::ITensor>> &inputs,
    const std::vector<std::shared_ptr<vkop::core::ITensor>> &outputs) {
    auto dev = vkop::tests::TestEnv::get_device();
    auto cmdpool = vkop::tests::TestEnv::get_command_pool();
    auto op = ops::create_from_type(ops::OpType::SCATTER_ELEMENTS, 0, 0, true);
    op->set_runtime_device(dev, cmdpool);
    op->setAttribute({{"axis", "0"}, {"reduction", reduction}});

    // Upload all tensors (data/indices/updates) — data is also the output.
    for (auto &t : inputs) {
        if (!t) continue;
        if (t->dtype() == typeid(int)) {
            vkop::core::as_tensor<int>(t)->as_storage_buffer(dev);
            vkop::core::as_tensor<int>(t)->copyToGPU(cmdpool);
        } else if (t->dtype() == typeid(float)) {
            vkop::core::as_tensor<float>(t)->as_storage_buffer(dev);
            vkop::core::as_tensor<float>(t)->copyToGPU(cmdpool);
        }
    }
    for (auto &t : outputs) {
        if (t->dtype() == typeid(float)) {
            vkop::core::as_tensor<float>(t)->as_storage_buffer(dev);
            vkop::core::as_tensor<float>(t)->copyToGPU(cmdpool);
        }
    }

    op->onExecute(inputs, outputs, 0);
    auto cmd = op->get_record();
    std::vector<VkSubmitInfo> info{cmd->buildSubmitInfo()};
    vkop::VulkanCommandBuffer::submit(dev->getComputeQueue(), info);
    cmd->wait();
    dev->wait_all_done();
    for (auto &t : outputs) {
        if (t->dtype() == typeid(float)) {
            vkop::core::as_tensor<float>(t)->copyToCPU(cmdpool);
        }
    }
}

static void fill_float(std::shared_ptr<Tensor<float>> &t, const torch::Tensor &tt) {
    auto flat = tt.cpu().contiguous().flatten();
    const float *p = flat.data_ptr<float>();
    t->fillToCPU(std::vector<float>(p, p + flat.numel()));
}

TEST(ScatterElementsTest, Overwrite) {
    auto data = torch::zeros({4, 3});
    auto indices = torch::tensor({0, 2, 1}).to(torch::kInt64);
    auto updates = torch::tensor({{10.0f, 20.0f, 30.0f},
                                   {40.0f, 50.0f, 60.0f},
                                   {70.0f, 80.0f, 90.0f}});
    auto ref = data.clone();
    ref.scatter_(0, indices.unsqueeze(1).expand({3, 3}), updates);

    auto tout = std::make_shared<Tensor<float>>(std::vector<int>{4, 3});
    auto tidx = std::make_shared<Tensor<int>>(std::vector<int>{3});
    auto tupd = std::make_shared<Tensor<float>>(std::vector<int>{3, 3});
    fill_float(tout, data);
    tidx->fillToCPU(std::vector<int>{0, 2, 1});
    fill_float(tupd, updates);

    run_scatter("none", {tout, tidx, tupd}, {tout});

    for (int i = 0; i < 12; ++i) {
        EXPECT_NEAR((*tout)[i], ref.flatten()[i].item<float>(), 0.01f)
            << "mismatch at " << i;
    }
}

TEST(ScatterElementsTest, AddReduction) {
    auto data = torch::tensor({{1.0f, 1.0f, 1.0f},
                                {2.0f, 2.0f, 2.0f},
                                {3.0f, 3.0f, 3.0f},
                                {4.0f, 4.0f, 4.0f}});
    auto indices = torch::tensor({0, 2, 0}).to(torch::kInt64);
    auto updates = torch::tensor({{10.0f, 20.0f, 30.0f},
                                   {40.0f, 50.0f, 60.0f},
                                   {100.0f, 200.0f, 300.0f}});
    auto ref = data.clone();
    ref.scatter_(0, indices.unsqueeze(1).expand({3, 3}), updates, "add");

    auto tout = std::make_shared<Tensor<float>>(std::vector<int>{4, 3});
    auto tidx = std::make_shared<Tensor<int>>(std::vector<int>{3});
    auto tupd = std::make_shared<Tensor<float>>(std::vector<int>{3, 3});
    fill_float(tout, data);
    tidx->fillToCPU(std::vector<int>{0, 2, 0});
    fill_float(tupd, updates);

    run_scatter("add", {tout, tidx, tupd}, {tout});

    for (int i = 0; i < 12; ++i) {
        EXPECT_NEAR((*tout)[i], ref.flatten()[i].item<float>(), 0.01f)
            << "mismatch at " << i;
    }
}

} // namespace
