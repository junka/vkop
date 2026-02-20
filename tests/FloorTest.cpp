
#include <vector>

#include "UnaryTest.hpp"
#include "include/logger.hpp"

using vkop::tests::UnaryTest;

namespace {

template <typename T>
class FloorTest : public vkop::tests::UnaryTest<T> {
public:
    explicit FloorTest(const std::vector<int> &shape): vkop::tests::UnaryTest<T>("Floor", shape) {
        auto torch_output = torch::floor(this->torch_input);
        this->fillTensorFromTorch(this->output, torch_output);
    }
};
}

TEST(FloorTest, FloorComprehensiveTest) {
    std::vector<std::tuple<std::vector<int>>> test_cases = {
        {{1, 3, 64, 64}},
    };
    for (const auto &test_case : test_cases) {
        auto [shape] = test_case;
        FloorTest<float> floortest(shape);
        EXPECT_TRUE(floortest.run_test());
    }
}
