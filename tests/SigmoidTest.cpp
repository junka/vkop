#include <vector>

#include "UnaryTest.hpp"
#include "include/logger.hpp"

using vkop::tests::UnaryTest;

namespace {

template <typename T>
class SigmoidTest : public vkop::tests::UnaryTest<T> {
public:
    explicit SigmoidTest(const std::vector<int> &shape): vkop::tests::UnaryTest<T>("Sigmoid", shape) {
        auto torch_output = torch::sigmoid(this->torch_input);
        this->fillTensorFromTorch(this->output, torch_output);
    }
};
}

TEST(SigmoidTest, SigmoidComprehensiveTest) {
    std::vector<std::tuple<std::vector<int>>> test_cases = {
        {{1, 3, 64, 64}},
    };
    for (const auto &test_case : test_cases) {
        auto [shape] = test_case;
        SigmoidTest<float> sigest(shape);
        EXPECT_TRUE(sigest.run_test());

        SigmoidTest<uint16_t> sigest1(shape);
        EXPECT_TRUE(sigest1.run_test());
    }
}