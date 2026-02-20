#include <vector>

#include "BinaryTest.hpp"
#include "include/logger.hpp"

using vkop::tests::BinaryTest;

namespace {
    

template<typename T>
class PowTest : public BinaryTest<T> {
public:
    explicit PowTest(std::vector<int> input_shape): BinaryTest<T>("Pow", std::move(input_shape)) {
        auto in = this->torch_inputa.abs() + 1e-6;
        this->fillTensorFromTorch(this->inputa, in);
        auto torch_output = torch::pow(in,this->torch_inputb);
        this->fillTensorFromTorch(this->output, torch_output);
    }
};
}

TEST(PowTest, PowComprehensiveTest) {

    const std::vector<std::vector<int>> test_cases = {
        {10, 5, 64, 64},
        {1, 3, 128, 128},
        {2, 4, 32, 32}
    };
    for (const auto& t : test_cases) {
        PowTest<float> powtest(t);
        EXPECT_TRUE(powtest.run_test());
    }
}