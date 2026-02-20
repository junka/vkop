#include <vector>

#include "BinaryTest.hpp"
#include "include/logger.hpp"

using vkop::tests::BinaryTest;

namespace {
    

template<typename T>
class SubTest : public BinaryTest<T> {
public:
    explicit SubTest(std::vector<int> input_shape): BinaryTest<T>("Sub", std::move(input_shape)) {
        auto torch_output = this->torch_inputa - this->torch_inputb;
        this->fillTensorFromTorch(this->output, torch_output);
    }
};
}

TEST(SubTest, SubComprehensiveTest) {
    const std::vector<std::vector<int>> test_cases = {
        {10, 5, 64, 64},
        {1, 3, 128, 128},
        {2, 4, 32, 32}
    };
    for (const auto& t : test_cases) {
        SubTest<float> subtest(t);
        EXPECT_TRUE(subtest.run_test());

        SubTest<uint16_t> subtest1(t);
        EXPECT_TRUE(subtest1.run_test());

    }
}
