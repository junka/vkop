#include <vector>

#include "BinaryTest.hpp"
#include "include/logger.hpp"

using vkop::tests::BinaryTest;

namespace {
    

template<typename T>
class DivTest : public BinaryTest<T> {
public:
    explicit DivTest(std::vector<int> input_shape): BinaryTest<T>("Div", std::move(input_shape)) {
        auto torch_output = this->torch_inputa / this->torch_inputb;
        this->fillTensorFromTorch(this->output, torch_output);
    }
private:
};
}


TEST(DivTest, DivComprehensiveTest) {

    const std::vector<std::vector<int>> test_cases = {
        {10, 5, 64, 64},
        {1, 3, 128, 128},
        {2, 4, 32, 32}
    };
    for (const auto& t : test_cases) {
        DivTest<float> divtest(t);
        EXPECT_TRUE (divtest.run_test());
    }
}