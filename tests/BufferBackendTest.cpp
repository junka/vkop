// Copyright 2025 @junka
// Explicit canary for the buffer (SSBO) backend. Constructs each ported
// buffer op directly via create_from_type(use_buffer=true) on a small
// {4,5} 2-D tensor and checks against a libtorch reference. Catches
// buffer-path-only regressions even when VKOP_BUFFER_BACKEND is unset.

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

template <typename T>
void fill(std::shared_ptr<Tensor<T>> &t, const torch::Tensor &tt) {
    auto flat = tt.cpu().contiguous().flatten();
    std::vector<T> v;
    if constexpr (std::is_same_v<T, uint16_t>) {
        const auto *p = reinterpret_cast<const uint16_t *>(flat.data_ptr<at::Half>());
        v.assign(p, p + flat.numel());
    } else {
        auto a = flat.accessor<T, 1>();
        v.assign(a.data(), a.data() + flat.numel());
    }
    t->fillToCPU(v);
}

template <typename T>
bool close_to_torch(const std::shared_ptr<Tensor<T>> &out, const torch::Tensor &ref) {
    auto flat = ref.cpu().contiguous().flatten();
    for (int i = 0; i < out->num_elements(); ++i) {
        float got, exp;
        if constexpr (std::is_same_v<T, uint16_t>) {
            got = vkop::core::ITensor::fp16_to_fp32((*out)[i]);
            exp = vkop::core::ITensor::fp16_to_fp32(
                reinterpret_cast<const uint16_t *>(flat.data_ptr<at::Half>())[i]);
        } else {
            got = float((*out)[i]);
            exp = float(flat[i].template item<T>());
        }
        float thr = std::max(std::abs(exp) * 0.02f, 0.02f);
        if (std::abs(got - exp) > thr) {
            LOG_ERROR("mismatch %d: %f vs %f", i, got, exp);
            return false;
        }
    }
    return true;
}

// Drive one op through the buffer backend and compare to a reference.
template <typename T>
bool run_unary(vkop::ops::OpType type, int op_code,
               const std::function<torch::Tensor(const torch::Tensor &)> &ref) {
    std::vector<int64_t> sh = {4, 5};
    auto torch_in = torch::randn(sh, torch::TensorOptions().dtype(
        std::is_same_v<T, uint16_t> ? torch::kFloat16 : torch::kFloat32));
    auto torch_out = ref(torch_in);

    auto input = std::make_shared<Tensor<T>>(std::vector<int>{4, 5});
    fill(input, torch_in);
    auto output = std::make_shared<Tensor<T>>(std::vector<int>{4, 5});

    auto op = vkop::ops::create_from_type(type, std::is_same_v<T, uint16_t> ? 1 : 0, 0, /*backend_buffer=*/true);
    if (!op) return false;
    auto dev = vkop::tests::TestEnv::get_device();
    auto cmdpool = vkop::tests::TestEnv::get_command_pool();
    op->set_runtime_device(dev, cmdpool);

    input->as_storage_buffer(dev);
    input->copyToGPU(cmdpool);
    output->as_storage_buffer(dev);
    output->toGPU();
    op->onExecute({input}, {output}, 0);
    auto cmd = op->get_record();
    std::vector<VkSubmitInfo> info{cmd->buildSubmitInfo()};
    vkop::VulkanCommandBuffer::submit(dev->getComputeQueue(), info);
    cmd->wait();
    dev->wait_all_done();
    output->copyToCPU(cmdpool);
    return close_to_torch(output, torch_out);
}

TEST(BufferBackendTest, UnaryRelu) {
    EXPECT_TRUE(run_unary<float>(vkop::ops::OpType::RELU, 0,
        [](const torch::Tensor &x) { return torch::relu(x); }));
}
TEST(BufferBackendTest, UnaryNeg) {
    EXPECT_TRUE(run_unary<float>(vkop::ops::OpType::NEG, 0,
        [](const torch::Tensor &x) { return -x; }));
}
TEST(BufferBackendTest, UnarySqrt) {
    EXPECT_TRUE(run_unary<float>(vkop::ops::OpType::SQRT, 0,
        [](const torch::Tensor &x) { return torch::abs(x).sqrt(); }));
}

} // namespace

// Binary (with broadcast) and prelu canaries.
template <typename T>
bool run_binary(vkop::ops::OpType type, int op_code,
                const std::function<torch::Tensor(const torch::Tensor &,
                                                  const torch::Tensor &)> &ref) {
    auto opt = torch::TensorOptions().dtype(
        std::is_same_v<T, uint16_t> ? torch::kFloat16 : torch::kFloat32);
    auto a = torch::randn({4, 5}, opt);
    auto b = torch::randn({5}, opt);
    auto torch_out = ref(a, b);

    auto in0 = std::make_shared<Tensor<T>>(std::vector<int>{4, 5});
    auto in1 = std::make_shared<Tensor<T>>(std::vector<int>{5});
    fill(in0, a);
    fill(in1, b);
    auto out = std::make_shared<Tensor<T>>(std::vector<int>{4, 5});

    auto op = vkop::ops::create_from_type(type,
        std::is_same_v<T, uint16_t> ? 1 : 0, 0, true);
    auto dev = vkop::tests::TestEnv::get_device();
    auto cmdpool = vkop::tests::TestEnv::get_command_pool();
    op->set_runtime_device(dev, cmdpool);
    in0->as_storage_buffer(dev); in0->copyToGPU(cmdpool);
    in1->as_storage_buffer(dev); in1->copyToGPU(cmdpool);
    out->as_storage_buffer(dev); out->toGPU();
    op->onExecute({in0, in1}, {out}, 0);
    auto cmd = op->get_record();
    std::vector<VkSubmitInfo> info{cmd->buildSubmitInfo()};
    vkop::VulkanCommandBuffer::submit(dev->getComputeQueue(), info);
    cmd->wait(); dev->wait_all_done();
    out->copyToCPU(cmdpool);
    return close_to_torch(out, torch_out);
}

TEST(BufferBackendTest, BinaryAdd) {
    EXPECT_TRUE(run_binary<float>(vkop::ops::OpType::ADD, 0,
        [](const torch::Tensor &a, const torch::Tensor &b) { return a + b; }));
}
TEST(BufferBackendTest, BinaryMul) {
    EXPECT_TRUE(run_binary<float>(vkop::ops::OpType::MUL, 0,
        [](const torch::Tensor &a, const torch::Tensor &b) { return a * b; }));
}
TEST(BufferBackendTest, BinaryAddFp16) {
    EXPECT_TRUE(run_binary<uint16_t>(vkop::ops::OpType::ADD, 0,
        [](const torch::Tensor &a, const torch::Tensor &b) { return a + b; }));
}

// Reshape canary: {2,3,4} -> {6,4}.
TEST(BufferBackendTest, Reshape) {
    auto torch_in = torch::randn({2, 3, 4});
    auto input = std::make_shared<Tensor<float>>(std::vector<int>{2, 3, 4});
    fill(input, torch_in);
    auto shape_tensor = std::make_shared<Tensor<int64_t>>(std::vector<int64_t>{6, 4});
    shape_tensor->fillToCPU(std::vector<int64_t>{6, 4});
    shape_tensor->toGPU();
    auto output = std::make_shared<Tensor<float>>(std::vector<int>{6, 4});

    auto op = vkop::ops::create_from_type(vkop::ops::OpType::RESHAPE, 0, 0, true);
    auto dev = vkop::tests::TestEnv::get_device();
    auto cmdpool = vkop::tests::TestEnv::get_command_pool();
    op->set_runtime_device(dev, cmdpool);
    input->as_storage_buffer(dev); input->copyToGPU(cmdpool);
    output->as_storage_buffer(dev); output->toGPU();
    op->onExecute({input, shape_tensor}, {output}, 0);
    auto cmd = op->get_record();
    std::vector<VkSubmitInfo> info{cmd->buildSubmitInfo()};
    vkop::VulkanCommandBuffer::submit(dev->getComputeQueue(), info);
    cmd->wait(); dev->wait_all_done();
    output->copyToCPU(cmdpool);
    EXPECT_TRUE(close_to_torch(output, torch_in.view({6, 4})));
}
