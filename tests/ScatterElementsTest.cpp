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
        } else if (t->dtype() == typeid(int64_t)) {
            vkop::core::as_tensor<int64_t>(t)->as_storage_buffer(dev);
            vkop::core::as_tensor<int64_t>(t)->copyToGPU(cmdpool);
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
    auto tidx = std::make_shared<Tensor<int64_t>>(std::vector<int>{3});
    auto tupd = std::make_shared<Tensor<float>>(std::vector<int>{3, 3});
    fill_float(tout, data);
    tidx->fillToCPU(std::vector<int64_t>{0, 2, 1});
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
    auto tidx = std::make_shared<Tensor<int64_t>>(std::vector<int>{3});
    auto tupd = std::make_shared<Tensor<float>>(std::vector<int>{3, 3});
    fill_float(tout, data);
    tidx->fillToCPU(std::vector<int64_t>{0, 2, 0});
    fill_float(tupd, updates);

    run_scatter("add", {tout, tidx, tupd}, {tout});

    for (int i = 0; i < 12; ++i) {
        EXPECT_NEAR((*tout)[i], ref.flatten()[i].item<float>(), 0.01f)
            << "mismatch at " << i;
    }
}

// fp16 variant of the AddReduction test. Exercises the word-level CAS shader
// path (buffer_scatter_elements_fp16_spv). cols=3 is ODD, so update-slot
// (gid&1) and target-slot (target&1) DIVERGE for some threads — this is the
// case that breaks a naive "same slot" assumption.
static void run_scatter_fp16(
    const std::string &reduction,
    const std::vector<std::shared_ptr<vkop::core::ITensor>> &inputs,
    const std::vector<std::shared_ptr<vkop::core::ITensor>> &outputs) {
    auto dev = vkop::tests::TestEnv::get_device();
    auto cmdpool = vkop::tests::TestEnv::get_command_pool();
    auto op = ops::create_from_type(ops::OpType::SCATTER_ELEMENTS, 1, 0, true);
    op->set_runtime_device(dev, cmdpool);
    op->setAttribute({{"axis", "0"}, {"reduction", reduction}});

    for (auto &t : inputs) {
        if (!t) continue;
        if (t->dtype() == typeid(int64_t)) {
            vkop::core::as_tensor<int64_t>(t)->as_storage_buffer(dev);
            vkop::core::as_tensor<int64_t>(t)->copyToGPU(cmdpool);
        } else if (t->dtype() == typeid(uint16_t)) {
            vkop::core::as_tensor<uint16_t>(t)->as_storage_buffer(dev);
            vkop::core::as_tensor<uint16_t>(t)->copyToGPU(cmdpool);
        }
    }
    for (auto &t : outputs) {
        if (t->dtype() == typeid(uint16_t)) {
            auto ot = vkop::core::as_tensor<uint16_t>(t);
            ot->as_storage_buffer(dev);
            // Only upload if the output carries host data (in-place case where
            // output==data input). A fresh distinct output has no data — the op
            // GPU-copies data into it via as_storage_buffer + vkCmdCopyBuffer.
            if (ot->has_cpu_data()) {
                ot->copyToGPU(cmdpool);
            }
        }
    }

    op->onExecute(inputs, outputs, 0);
    auto cmd = op->get_record();
    std::vector<VkSubmitInfo> info{cmd->buildSubmitInfo()};
    vkop::VulkanCommandBuffer::submit(dev->getComputeQueue(), info);
    cmd->wait();
    dev->wait_all_done();
    for (auto &t : outputs) {
        if (t->dtype() == typeid(uint16_t)) {
            vkop::core::as_tensor<uint16_t>(t)->copyToCPU(cmdpool);
        }
    }
}

TEST(ScatterElementsTest, AddReductionFp16) {
    // cols=3 (ODD) to stress the divergent-slot fp16 CAS path.
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

    auto tout = std::make_shared<Tensor<uint16_t>>(std::vector<int>{4, 3});
    auto tidx = std::make_shared<Tensor<int64_t>>(std::vector<int>{3});
    auto tupd = std::make_shared<Tensor<uint16_t>>(std::vector<int>{3, 3});
    std::vector<float> data_vec(data.data_ptr<float>(),
                                data.data_ptr<float>() + data.numel());
    std::vector<float> upd_vec(updates.data_ptr<float>(),
                               updates.data_ptr<float>() + updates.numel());
    tout->fillFP32ToCPU(data_vec);
    tidx->fillToCPU(std::vector<int64_t>{0, 2, 0});
    tupd->fillFP32ToCPU(upd_vec);

    run_scatter_fp16("add", {tout, tidx, tupd}, {tout});

    for (int i = 0; i < 12; ++i) {
        float got = vkop::core::ITensor::fp16_to_fp32((*tout)[i]);
        EXPECT_NEAR(got, ref.flatten()[i].item<float>(), 1.0f)
            << "mismatch at " << i << " got=" << got;
    }
}

TEST(ScatterElementsTest, OverwriteFp16EvenCols) {
    // cols=4 (EVEN) — the real LLM shape parity (2048). update-slot and
    // target-slot always match here, but 'none' reduction still needs the
    // word-level CAS (a sibling thread may own the other half).
    auto data = torch::zeros({4, 4});
    auto indices = torch::tensor({0, 2, 1}).to(torch::kInt64);
    auto updates = torch::tensor({{10.0f, 20.0f, 30.0f, 40.0f},
                                   {50.0f, 60.0f, 70.0f, 80.0f},
                                   {90.0f, 100.0f, 110.0f, 120.0f}});
    auto ref = data.clone();
    ref.scatter_(0, indices.unsqueeze(1).expand({3, 4}), updates);

    auto tout = std::make_shared<Tensor<uint16_t>>(std::vector<int>{4, 4});
    auto tidx = std::make_shared<Tensor<int64_t>>(std::vector<int>{3});
    auto tupd = std::make_shared<Tensor<uint16_t>>(std::vector<int>{3, 4});
    std::vector<float> data_vec(data.data_ptr<float>(),
                                data.data_ptr<float>() + data.numel());
    std::vector<float> upd_vec(updates.data_ptr<float>(),
                               updates.data_ptr<float>() + updates.numel());
    tout->fillFP32ToCPU(data_vec);
    tidx->fillToCPU(std::vector<int64_t>{0, 2, 1});
    tupd->fillFP32ToCPU(upd_vec);

    run_scatter_fp16("none", {tout, tidx, tupd}, {tout});

    for (int i = 0; i < 16; ++i) {
        float got = vkop::core::ITensor::fp16_to_fp32((*tout)[i]);
        EXPECT_NEAR(got, ref.flatten()[i].item<float>(), 1.0f)
            << "mismatch at " << i << " got=" << got;
    }
}

// Distinct-output variant: the runtime allocates a FRESH output tensor per
// node (output != data input). The GPU path must seed the output with a
// device->device copy of the data buffer before scattering, else non-scattered
// positions read as zero. This mirrors the real LLM model wiring.
TEST(ScatterElementsTest, AddReductionFp16DistinctOutput) {
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

    // data input tensor (stays unmodified — distinct from output)
    auto tdata = std::make_shared<Tensor<uint16_t>>(std::vector<int>{4, 3});
    auto tidx = std::make_shared<Tensor<int64_t>>(std::vector<int>{3});
    auto tupd = std::make_shared<Tensor<uint16_t>>(std::vector<int>{3, 3});
    // FRESH output tensor (never seeded on host — the op must GPU-copy data in)
    auto tout = std::make_shared<Tensor<uint16_t>>(std::vector<int>{4, 3});
    std::vector<float> data_vec(data.data_ptr<float>(),
                                data.data_ptr<float>() + data.numel());
    std::vector<float> upd_vec(updates.data_ptr<float>(),
                               updates.data_ptr<float>() + updates.numel());
    tdata->fillFP32ToCPU(data_vec);
    tidx->fillToCPU(std::vector<int64_t>{0, 2, 0});
    tupd->fillFP32ToCPU(upd_vec);

    run_scatter_fp16("add", {tdata, tidx, tupd}, {tout});
    // tdata was uploaded (host data cleared); read it back to verify it wasn't
    // clobbered by the distinct-output GPU copy.
    tdata->copyToCPU(vkop::tests::TestEnv::get_command_pool());

    for (int i = 0; i < 12; ++i) {
        float got = vkop::core::ITensor::fp16_to_fp32((*tout)[i]);
        EXPECT_NEAR(got, ref.flatten()[i].item<float>(), 1.0f)
            << "mismatch at " << i << " got=" << got;
    }
    // The data input must be UNMODIFIED (distinct output → no in-place write).
    for (int i = 0; i < 12; ++i) {
        float got = vkop::core::ITensor::fp16_to_fp32((*tdata)[i]);
        EXPECT_NEAR(got, data.flatten()[i].item<float>(), 1.0f)
            << "data input clobbered at " << i << " got=" << got;
    }
}

} // namespace
