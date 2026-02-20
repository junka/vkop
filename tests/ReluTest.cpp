
#include <vector>

#include "UnaryTest.hpp"
#include "include/logger.hpp"

using vkop::tests::UnaryTest;

namespace {

template <typename T>
class ReluTest : public vkop::tests::UnaryTest<T> {
public:
    explicit ReluTest(const std::vector<int> &shape): vkop::tests::UnaryTest<T>("Relu", shape) {
        auto torch_output = torch::relu(this->torch_input);
        this->fillTensorFromTorch(this->output, torch_output);
    }
};
}

TEST(ReluTest, ReluComprehensiveTest) {
    std::vector<std::tuple<std::vector<int>>> test_cases = {
        {{1, 3, 64, 64}},
    };
    for (const auto &test_case : test_cases) {
        auto [shape] = test_case;
        ReluTest<float> relutest(shape);
        EXPECT_TRUE(relutest.run_test());
    }
}