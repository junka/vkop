#include <vector>
#include <random>
#include <cmath>

#include "setup.hpp"
#include "core/Tensor.hpp"
#include "include/logger.hpp"

using vkop::core::Tensor;
using vkop::tests::TestCase;

namespace {

class AtanTest : public TestCase {
public:
    std::vector<int> input_shape_;

    std::shared_ptr<Tensor<float>> input;
    std::shared_ptr<Tensor<float>> output;

    explicit AtanTest(std::vector<int> input_shape):TestCase("Atan"), input_shape_(std::move(input_shape)) {
        initTestdata();
    }
private:
    void initTestdata()
    {
        input = std::make_shared<Tensor<float>>(input_shape_);
        input->reserveOnCPU();
        output = std::make_shared<Tensor<float>>(input_shape_);
        output->reserveOnCPU();
        
        std::random_device rd{};
        std::mt19937 gen{rd()};
        gen.seed(1024);
        std::normal_distribution<> input_dist{0.0F, 1.0F};
        for (int i = 0; i < input->num_elements(); i++) {
            (*input)[i] = input_dist(gen);
            (*output)[i] = std::atan((*input)[i]);
        }
    }
};
}

TEST(AtanTest, AtanComprehensiveTest) {

    std::vector<std::vector<int>> test_cases = {
        {10, 5, 64, 64},
        {1, 3, 128, 128},
        {2, 4, 32, 32}
    };

    for (const auto& t : test_cases) {
        AtanTest atantest(t);
        EXPECT_TRUE (atantest.run_test<float>({atantest.input}, {atantest.output}));
    }
}