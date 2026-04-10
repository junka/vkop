#include <vector>

#include "BinaryTest.hpp"
#include "include/logger.hpp"

using vkop::tests::BinaryTest;

namespace {
    

template<typename T>
class AddTest : public BinaryTest<T> {
public:
    explicit AddTest(std::vector<int> input_shape): BinaryTest<T>("Add", std::move(input_shape)) {
        std::cout << this->torch_inputa << std::endl;
        std::cout << this->torch_inputb << std::endl;
        auto torch_output = this->torch_inputa + this->torch_inputb;
        std::cout << torch_output << std::endl;
        this->fillTensorFromTorch(this->output, torch_output);
    }
    AddTest(const std::vector<int>& shapea, const std::vector<int>& shapeb): BinaryTest<T>("Add", shapea, shapeb) {
        std::cout << this->torch_inputa << std::endl;
        std::cout << this->torch_inputb << std::endl;
        auto torch_output = this->torch_inputa + this->torch_inputb;
        std::cout << torch_output << std::endl;
        this->fillTensorFromTorch(this->output, torch_output);
    };
};

TEST(AddTest, AddComprehensiveTest) {
    const std::vector<std::vector<int>> test_cases = {
        {10, 5, 64, 64},
        {1, 3, 128, 128},
        {2, 4, 32, 32}
    };

    for (const auto& t : test_cases) {
        AddTest<float> addtest(t);
        EXPECT_TRUE(addtest.run_test());
    
        AddTest<uint16_t> addtest1(t);
        EXPECT_TRUE(addtest1.run_test());
    }
}


TEST(AddTest, AddBroadcastTest) {
    const std::vector<std::tuple<std::vector<int>, std::vector<int>>> test_cases = {
        {{1, 3, 4, 4}, {1,1,1}},
        {{1, 1, 1, 1}, {2, 4, 32, 32}},
        {{10, 5, 64, 64}, {1, 1, 1, 1}},
    };

    for (const auto& t : test_cases) {
        auto [ashape, bshape] = t;
        AddTest<float> addtest(ashape, bshape);
        EXPECT_TRUE(addtest.run_test());
    
        AddTest<uint16_t> addtest1(ashape, bshape);
        EXPECT_TRUE(addtest1.run_test());
    }
}

} // namespace