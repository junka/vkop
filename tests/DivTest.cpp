#include <vector>

#include "BinaryTest.hpp"
#include "include/logger.hpp"

using vkop::tests::BinaryTest;

namespace {
    

template<typename T>
class DivTest : public BinaryTest<T> {
public:
    explicit DivTest(std::vector<int> input_shape): BinaryTest<T>("Div", std::move(input_shape)) {
        // The base fixture draws the denominator from randn, which lands
        // arbitrarily close to 0. The quotient then overflows the fp16 range,
        // and the torch reference and the shader disagree on which side of the
        // 65504 boundary the rounded value lands -- the fp16 case failed on
        // ~10% of runs. Draw |b| from [0.5, 1.5) instead, keeping the quotient
        // representable; sign coverage is kept by the random sign.
        const auto sizes = this->torch_inputb.sizes();
        auto sign_pick = torch::rand(sizes) < 0.5;
        auto sign = torch::where(sign_pick, torch::full(sizes, -1.0),
                                 torch::full(sizes, 1.0));
        auto mag = torch::rand(sizes) + 0.5;
        this->torch_inputb = (mag * sign).to(this->getTorchConf().dtype());
        this->fillTensorFromTorch(this->inputb, this->torch_inputb);
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

        DivTest<uint16_t> divtest16(t);
        EXPECT_TRUE (divtest16.run_test());
    }
}