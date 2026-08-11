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

// Shape is a CPU-only op (no shader): it reads the input's host shape
// metadata via getShape() and writes an int64 tensor. There is no GPU
// dispatch involved — the output int64 values are produced on the host and
// pushed up to the GPU SSBO for downstream int64 consumers (all of which
// read the CPU copy via as_tensor<int64_t>()).
//
// run_test() still records+submits the op's (empty) command buffer, which is
// harmless for a pipeline-less op.

namespace {

class ShapeTestFixture : public TestCase<float> {
  public:
    std::shared_ptr<Tensor<float>> input;
    std::shared_ptr<Tensor<int64_t>> output;

    explicit ShapeTestFixture(const std::vector<int> &shape)
        : TestCase<float>("Shape") {
        int n = 1;
        for (int d : shape) n *= d;
        input = std::make_shared<Tensor<float>>(shape);
        input->fillToCPU(std::vector<float>(n, 1.0f));
        std::vector<int64_t> dims(shape.begin(), shape.end());
        output = std::make_shared<Tensor<int64_t>>(
            std::vector<int>{static_cast<int>(shape.size())});
        output->fillToCPU(dims);
    }

    bool run_test() { return TestCase<float>::run_test({input}, {output}); }
};

TEST(ShapeTest, Rank3) { EXPECT_TRUE(ShapeTestFixture({2, 3, 4}).run_test()); }

TEST(ShapeTest, Rank1) { EXPECT_TRUE(ShapeTestFixture({7}).run_test()); }

TEST(ShapeTest, Rank5) {
    EXPECT_TRUE(ShapeTestFixture({1, 1, 1, 1, 64}).run_test());
}

} // namespace
