#include <vector>

#include "BinaryTest.hpp"
#include "include/logger.hpp"

using vkop::tests::BinaryTest;

namespace {
    

template<typename T>
class AddTest : public BinaryTest<T> {
public:
    explicit AddTest(std::vector<int> input_shape): BinaryTest<T>("Add", std::move(input_shape)) {
        auto torch_output = this->torch_inputa + this->torch_inputb;
        this->fillTensorFromTorch(this->output, torch_output);
    }
};
}

TEST(AddTest, AddComprehensiveTest) {
    const std::vector<std::vector<int>> test_cases = {
        {10, 5, 64, 64},
        {1, 3, 128, 128},
        {2, 4, 32, 32}
    };

    for (const auto& t : test_cases) {
        AddTest<float> addtest(t);
        EXPECT_TRUE (addtest.run_test());
    
        AddTest<uint16_t> addtest1(t);
        EXPECT_TRUE (addtest1.run_test());
    }
}