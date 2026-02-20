#include <vector>

#include "BinaryTest.hpp"
#include "include/logger.hpp"

using vkop::tests::BinaryTest;

namespace {
    

template<typename T>
class MulTest : public BinaryTest<T> {
public:
    explicit MulTest(std::vector<int> input_shape): BinaryTest<T>("Mul", std::move(input_shape)) {
        auto torch_output = this->torch_inputa * this->torch_inputb;
        this->fillTensorFromTorch(this->output, torch_output);
    }
};
}

TEST(MulTest, MulComprehensiveTest) {

    const std::vector<std::vector<int>> test_cases = {
        {10, 5, 64, 64},
        {1, 3, 128, 128},
        {2, 4, 32, 32}
    };
    for (const auto& t : test_cases) {
        MulTest<float> multest(t);
        EXPECT_TRUE(multest.run_test());
    }
}