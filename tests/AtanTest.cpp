
#include <vector>

#include "UnaryTest.hpp"
#include "include/logger.hpp"

using vkop::tests::UnaryTest;

namespace {

template <typename T>
class AtanTest : public vkop::tests::UnaryTest<T> {
public:
    explicit AtanTest(const std::vector<int> &shape): vkop::tests::UnaryTest<T>("Atan", shape) {
        auto torch_output = torch::atan(this->torch_input);
        this->fillTensorFromTorch(this->output, torch_output);
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
        AtanTest<float> atantest(t);
        EXPECT_TRUE(atantest.run_test());

        AtanTest<uint16_t> atantest1(t);
        EXPECT_TRUE(atantest1.run_test());
    }
}