
#include <vector>

#include "UnaryTest.hpp"
#include "include/logger.hpp"

using vkop::tests::UnaryTest;

namespace {

template <typename T>
class ErfTest : public vkop::tests::UnaryTest<T> {
public:
    explicit ErfTest(const std::vector<int> &shape): vkop::tests::UnaryTest<T>("Erf", shape) {
        auto torch_output = torch::erf(this->torch_input);
        this->fillTensorFromTorch(this->output, torch_output);
    }
};
}

TEST(ErfTest, ErfComprehensiveTest) {
    std::vector<std::tuple<std::vector<int>>> test_cases = {
        {{1, 3, 64, 64}},
    };
    for (const auto &test_case : test_cases) {
        auto [shape] = test_case;
        ErfTest<float> erftest(shape);
        EXPECT_TRUE(erftest.run_test());
    }
}
