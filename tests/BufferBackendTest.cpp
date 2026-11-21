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

// Broadcast forms the binary kernel's affine path (broadcast == 3, index =
// (lin / div) % mod) must get right: trailing, per-row (div > 1), middle,
// scalar, a broadcast A operand, an odd fp16 element count; plus one
// non-affine pair ([4,1,6] vs [1,5,1]) that must stay on the N-d path.
template <typename T>
bool run_binary_shapes(vkop::ops::OpType type, const std::vector<int> &sa,
                       const std::vector<int> &sb,
                       const std::function<torch::Tensor(
                           const torch::Tensor &, const torch::Tensor &)> &ref) {
    auto opt = torch::TensorOptions().dtype(
        std::is_same_v<T, uint16_t> ? torch::kFloat16 : torch::kFloat32);
    auto to64 = [](const std::vector<int> &v) {
        return std::vector<int64_t>(v.begin(), v.end());
    };
    auto a = torch::randn(to64(sa), opt);
    auto b = torch::randn(to64(sb), opt);
    auto torch_out = ref(a, b);
    std::vector<int> so(torch_out.sizes().begin(), torch_out.sizes().end());

    auto in0 = std::make_shared<Tensor<T>>(sa);
    auto in1 = std::make_shared<Tensor<T>>(sb);
    fill(in0, a);
    fill(in1, b);
    auto out = std::make_shared<Tensor<T>>(so);

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

TEST(BufferBackendTest, BinaryAffineBroadcast) {
    const std::vector<std::pair<std::vector<int>, std::vector<int>>> cases = {
        {{3, 7, 5}, {5}},          // trailing
        {{6, 9}, {6, 1}},          // per-row, div > 1
        {{2, 5, 7}, {1, 5, 1}},    // middle run
        {{3, 7, 5}, {1}},          // scalar
        {{1, 5, 1}, {2, 5, 7}},    // broadcast A
        {{3, 1, 7}, {3, 5, 7}},    // A per-outer-row: not contiguous -> N-d
        {{4, 1, 6}, {1, 5, 1}},    // both broadcast, not affine -> N-d
        {{1, 1, 33}, {1, 1, 1}},   // odd total
        // General N-d path with coalesced dims (fill_dims_coalesced):
        {{2, 3, 4, 5}, {2, 1, 4, 1}},        // alternating: stays rank 4
        {{2, 1, 3, 4, 5}, {1, 1, 3, 1, 1}},  // size-1 dims dropped
        {{2, 3, 4, 5, 6}, {2, 3, 1, 1, 6}},  // runs merge to rank 3
        {{4, 1, 6, 2}, {1, 5, 1, 2}},        // both broadcast, mixed
    };
    auto mul = [](const torch::Tensor &a, const torch::Tensor &b) {
        return a * b;
    };
    auto mx = [](const torch::Tensor &a, const torch::Tensor &b) {
        return torch::max(a, b);
    };
    for (const auto &[sa, sb] : cases) {
        EXPECT_TRUE(run_binary_shapes<float>(vkop::ops::OpType::MUL, sa, sb, mul));
        EXPECT_TRUE(run_binary_shapes<uint16_t>(vkop::ops::OpType::MUL, sa, sb, mul));
        EXPECT_TRUE(run_binary_shapes<uint16_t>(vkop::ops::OpType::MAX, sa, sb, mx));
    }
}

// fill_dims_coalesced merges output dims every input treats alike and drops
// size-1 dims; anything it cannot read as a broadcast falls back unchanged.
TEST(BufferBackendTest, CoalescedBroadcastDims) {
    int od[8], a[8], b[8];
    std::vector<int> out{1, 1024, 32, 128}, sa{1, 1024, 32, 128}, sb{1, 1, 32, 128};
    int r = vkop::ops::fill_dims_coalesced(od, out, {&a, &b}, {&sa, &sb});
    EXPECT_EQ(r, 2);
    EXPECT_EQ(od[0], 1024); EXPECT_EQ(od[1], 4096);
    EXPECT_EQ(a[0], 1024);  EXPECT_EQ(a[1], 4096);
    EXPECT_EQ(b[0], 1);     EXPECT_EQ(b[1], 4096);

    std::vector<int> out2{2, 3, 4, 5}, sc{3, 1};
    std::vector<int> sd{4, 1};
    r = vkop::ops::fill_dims_coalesced(od, out2, {&a}, {&sd});
    EXPECT_EQ(r, 3); // {6 | 4 | 5} with input {1 | 4 | 1}
    EXPECT_EQ(od[0], 6); EXPECT_EQ(od[1], 4); EXPECT_EQ(od[2], 5);
    EXPECT_EQ(a[0], 1);  EXPECT_EQ(a[1], 4);  EXPECT_EQ(a[2], 1);

    // Not a broadcast ({3,1} against trailing {4,5}): plain fill, full rank.
    r = vkop::ops::fill_dims_coalesced(od, out2, {&a}, {&sc});
    EXPECT_EQ(r, 4);
    EXPECT_EQ(a[0], 1); EXPECT_EQ(a[1], 1); EXPECT_EQ(a[2], 3); EXPECT_EQ(a[3], 1);

    // All-ones output collapses to rank 1.
    std::vector<int> ones{1, 1, 1}, one{1};
    r = vkop::ops::fill_dims_coalesced(od, ones, {&a}, {&one});
    EXPECT_EQ(r, 1); EXPECT_EQ(od[0], 1); EXPECT_EQ(a[0], 1);
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
