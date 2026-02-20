
#include <vector>

#include "UnaryTest.hpp"
#include "include/logger.hpp"

using vkop::tests::UnaryTest;

namespace {

template <typename T>
class SoftplusTest : public vkop::tests::UnaryTest<T> {
public:
    explicit SoftplusTest(const std::vector<int> &shape): vkop::tests::UnaryTest<T>("Softplus", shape) {
        auto torch_output = torch::softplus(this->torch_input);
        this->fillTensorFromTorch(this->output, torch_output);
    }
};
}


TEST(SoftplusTest, SoftplusComprehensiveTest) {
    std::vector<std::vector<int>> test_cases = {
        {10, 5, 64, 64},
        {1, 3, 128, 128},
        {2, 4, 32, 32}
    };
    for (const auto& t : test_cases) {
        SoftplusTest<float> sptest(t);
        EXPECT_TRUE(sptest.run_test());
    }
}