// Copyright 2025 @junka
#include "setup.hpp"
#include "core/Tensor.hpp"
#include <gtest/gtest.h>
#include <torch/torch.h>

using vkop::core::Tensor;
using vkop::tests::TestCase;

namespace {

// Equal: binary op, same shape. Inherits TestCase<float> and uses run_test.
class EqualSameShape : public TestCase<float> {
public:
    std::shared_ptr<Tensor<float>> inputa, inputb, output;
    torch::Tensor torch_a, torch_b, torch_out;

    EqualSameShape() : TestCase<float>("Equal") {
        torch_a = torch::tensor({{1.0f, 2.0f, 3.0f, 4.0f},
                                  {5.0f, 6.0f, 7.0f, 8.0f},
                                  {1.0f, 2.0f, 3.0f, 0.0f}});
        torch_b = torch::tensor({{1.0f, 0.0f, 3.0f, 0.0f},
                                  {0.0f, 6.0f, 0.0f, 8.0f},
                                  {1.0f, 2.0f, 0.0f, 0.0f}});
        torch_out = (torch_a == torch_b).to(torch::kFloat32);

        inputa = std::make_shared<Tensor<float>>(std::vector<int>{3, 4});
        inputb = std::make_shared<Tensor<float>>(std::vector<int>{3, 4});
        output = std::make_shared<Tensor<float>>(std::vector<int>{3, 4});
        fillTensorFromTorch(inputa, torch_a);
        fillTensorFromTorch(inputb, torch_b);
        fillTensorFromTorch(output, torch_out);
    }
    bool run_test() { return TestCase<float>::run_test({inputa, inputb}, {output}); }
};

class EqualBroadcast : public TestCase<float> {
public:
    std::shared_ptr<Tensor<float>> inputa, inputb, output;
    torch::Tensor torch_a, torch_b, torch_out;

    EqualBroadcast() : TestCase<float>("Equal") {
        torch_a = torch::randn({3, 4});
        torch_b = torch::randn({4});
        torch_out = (torch_a == torch_b).to(torch::kFloat32);

        inputa = std::make_shared<Tensor<float>>(std::vector<int>{3, 4});
        inputb = std::make_shared<Tensor<float>>(std::vector<int>{4});
        output = std::make_shared<Tensor<float>>(std::vector<int>{3, 4});
        fillTensorFromTorch(inputa, torch_a);
        fillTensorFromTorch(inputb, torch_b);
        fillTensorFromTorch(output, torch_out);
    }
    bool run_test() { return TestCase<float>::run_test({inputa, inputb}, {output}); }
};

TEST(EqualTest, Basic) { EXPECT_TRUE(EqualSameShape().run_test()); }
TEST(EqualTest, Broadcast) { EXPECT_TRUE(EqualBroadcast().run_test()); }

} // namespace
