#include <vector>

#include "BinaryTest.hpp"
#include "include/logger.hpp"

using vkop::tests::BinaryTest;

namespace {
    

template<typename T>
class PReluTest : public BinaryTest<T> {
public:
    explicit PReluTest(std::vector<int> input_shape): BinaryTest<T>("PRelu", std::move(input_shape)) {
        // auto torch_output = torch::prelu(this->torch_inputa,this->torch_inputb);
        auto torch_output = torch::where(this->torch_inputa > 0, this->torch_inputa, this->torch_inputa * this->torch_inputb);
        this->fillTensorFromTorch(this->output, torch_output);
    }
};
}

TEST(PReluTest, PReluComprehensiveTest) {
    const std::vector<std::vector<int>> test_cases = {
        {10, 5, 64, 64},
        {1, 3, 128, 128},
        {2, 4, 32, 32}
    };
    for (const auto& t : test_cases) {
        PReluTest<float> prelutest(t);
        EXPECT_TRUE(prelutest.run_test());

        PReluTest<uint16_t> prelutest1(t);
        EXPECT_TRUE(prelutest1.run_test());

    }
}
