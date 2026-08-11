// Copyright 2026 @junka
#include "setup.hpp"
#include "core/Tensor.hpp"
#include "ops/OperatorFactory.hpp"
#include "ops/Ops.hpp"
#include <gtest/gtest.h>
#include <torch/torch.h>

using vkop::core::Tensor;
using vkop::tests::TestCase;
namespace ops = vkop::ops;

// Cast is a GPU buffer shader op (fp32 <-> fp16, driven by the "to"
// attribute). Unlike the symmetric TestCase::run_test flow it has two
// asymmetric dtype directions (f32 input -> f16 output and f16 input -> f32
// output), so we use the direct construction pattern and compare the raw
// fp16 bits against torch's reference conversion.

namespace {

template <typename T>
static void upload(std::shared_ptr<Tensor<T>> t) {
    auto dev = vkop::tests::TestEnv::get_device();
    auto cmdpool = vkop::tests::TestEnv::get_command_pool();
    t->as_storage_buffer(dev);
    t->copyToGPU(cmdpool);
}

// Run one Cast pass with the given "to" attribute.
static void run_cast(const std::string &to,
                     const std::shared_ptr<vkop::core::ITensor> &input,
                     const std::shared_ptr<vkop::core::ITensor> &output) {
    auto dev = vkop::tests::TestEnv::get_device();
    auto cmdpool = vkop::tests::TestEnv::get_command_pool();
    auto op = ops::create_from_type(ops::OpType::CAST, 0, 0, true);
    op->set_runtime_device(dev, cmdpool);
    op->setAttribute({{"to", to}});

    // Upload the input CPU data to its SSBO. The output is written entirely
    // by the shader — it is created GPU-side and the op allocates its SSBO in
    // execute(); copying CPU data (which it has none of) would dereference an
    // empty vector.
    if (input->dtype() == typeid(float)) {
        upload(vkop::core::as_tensor<float>(input));
    } else if (input->dtype() == typeid(uint16_t)) {
        upload(vkop::core::as_tensor<uint16_t>(input));
    }

    op->onExecute({input}, {output}, 0);
    auto cmd = op->get_record();
    std::vector<VkSubmitInfo> info{cmd->buildSubmitInfo()};
    vkop::VulkanCommandBuffer::submit(dev->getComputeQueue(), info);
    cmd->wait();
    dev->wait_all_done();

    if (output->dtype() == typeid(float)) {
        vkop::core::as_tensor<float>(output)->copyToCPU(cmdpool);
    } else if (output->dtype() == typeid(uint16_t)) {
        vkop::core::as_tensor<uint16_t>(output)->copyToCPU(cmdpool);
    }
}

static std::vector<uint16_t> fp16_bits(const torch::Tensor &t) {
    auto cpu = t.cpu().contiguous();
    const auto *p = reinterpret_cast<const uint16_t *>(cpu.data_ptr<at::Half>());
    return std::vector<uint16_t>(p, p + cpu.numel());
}

TEST(CastTest, F32ToF16) {
    auto torch_a = torch::tensor({{1.0f, 0.5f, -2.0f, 3.25f, 0.0f, -0.25f, 100.0f},
                                  {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f},
                                  {1.0e4f, -1.0e4f, 65504.0f, 1.5f, 2.5f, 3.5f, 4.5f}});
    auto ref = torch_a.to(torch::kFloat16);
    int n = static_cast<int>(torch_a.numel());

    auto tin = std::make_shared<Tensor<float>>(std::vector<int>{3, 7});
    tin->fillToCPU(std::vector<float>(torch_a.data_ptr<float>(),
                                      torch_a.data_ptr<float>() + n));
    auto tout = std::make_shared<Tensor<uint16_t>>(std::vector<int>{3, 7}, true);
    auto expected = fp16_bits(ref);

    run_cast("10", tin, tout);

    for (int i = 0; i < n; ++i) {
        EXPECT_EQ((*tout)[i], expected[i])
            << "mismatch at " << i << " (raw fp16 bits)";
    }
}

TEST(CastTest, F16ToF32) {
    auto torch_h = torch::tensor({{1.0f, 0.5f, -2.0f, 3.25f, 0.0f},
                                  {0.1f, 0.2f, 0.3f, 0.4f, 0.5f}}).to(torch::kFloat16);
    auto ref = torch_h.to(torch::kFloat32);
    int n = static_cast<int>(torch_h.numel());

    auto tin = std::make_shared<Tensor<uint16_t>>(std::vector<int>{2, 5});
    tin->fillToCPU(fp16_bits(torch_h));
    auto tout = std::make_shared<Tensor<float>>(std::vector<int>{2, 5}, true);

    run_cast("1", tin, tout);

    for (int i = 0; i < n; ++i) {
        EXPECT_FLOAT_EQ((*tout)[i], ref.flatten()[i].item<float>())
            << "mismatch at " << i;
    }
}

TEST(CastTest, OddTotal) {
    // Odd element count exercises the OOB guard on the packed half2 word.
    auto torch_a = torch::tensor({1.0f, 2.0f, 3.0f, 4.0f, 5.0f});
    auto ref = torch_a.to(torch::kFloat16);
    int n = 5;

    auto tin = std::make_shared<Tensor<float>>(std::vector<int>{5});
    tin->fillToCPU(std::vector<float>(torch_a.data_ptr<float>(),
                                      torch_a.data_ptr<float>() + n));
    auto tout = std::make_shared<Tensor<uint16_t>>(std::vector<int>{5}, true);
    auto expected = fp16_bits(ref);

    run_cast("10", tin, tout);

    for (int i = 0; i < n; ++i) {
        EXPECT_EQ((*tout)[i], expected[i])
            << "mismatch at " << i << " (raw fp16 bits)";
    }
}

} // namespace
