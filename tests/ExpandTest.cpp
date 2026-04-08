#include <cstdint>
#include <vector>

#include "setup.hpp"
#include "core/Tensor.hpp"
#include "include/logger.hpp"

using vkop::core::Tensor;
using vkop::tests::TestCase;

namespace {

template<typename T>
class ExpandTest : public TestCase<T> {
public:
    std::shared_ptr<Tensor<T>> output;
    std::shared_ptr<Tensor<T>> input;
    std::shared_ptr<Tensor<int>> toshape;
    std::vector<int> shape_;
    std::vector<int> outshape_;

    ExpandTest(std::vector<int> &shape, std::vector<int> &outshape): TestCase<T>("Expand"), shape_(shape), outshape_(outshape) {
        initTestdata();
    }

private:
    void initTestdata()
    {
        std::vector<int64_t> ishape(shape_.begin(), shape_.end());
        auto torch_input = torch::randn(ishape, this->getTorchConf());

        std::vector<int64_t> outshape(outshape_.begin(), outshape_.end());
        auto torch_output = torch_input.expand(outshape);

        std::cout << "torch_input: " << torch_input << std::endl;

        std::cout << "torch_output: " << torch_output << std::endl;


        input = std::make_shared<Tensor<T>>(shape_);
        this->fillTensorFromTorch(input, torch_input);
        toshape = std::make_shared<Tensor<int>>(outshape_);
        toshape->fillToCPU(outshape_);
        
        output = std::make_shared<Tensor<T>>(outshape_);
        this->fillTensorFromTorch(output, torch_output);
    }
};
}

TEST(ExpandTest, ExpandComprehensiveTest) {
    std::vector<std::tuple<std::vector<int>, std::vector<int>>> test_cases = {
        {{3, 9, 1}, {3, 9, 3}},
        {{3, 8}, {2, 3, 8}},
        {{2}, {2, 12, 2}},
    };
    for (const auto& test_case : test_cases) {
        auto [shape, outshape] = test_case;
        LOG_INFO("Testing Expand fp32");
        ExpandTest<float> expand_test(shape, outshape);
        const std::vector<std::shared_ptr<vkop::core::ITensor>> inputs = {
            expand_test.input,
            expand_test.toshape,
        };
        EXPECT_TRUE(expand_test.run_test(inputs, {expand_test.output}));

        LOG_INFO("Testing Expand fp16");
        ExpandTest<uint16_t> expand_test1(shape, outshape);
        const std::vector<std::shared_ptr<vkop::core::ITensor>> inputs1 = {
            expand_test1.input,
            expand_test1.toshape,
        };
        EXPECT_TRUE(expand_test1.run_test(inputs1, {expand_test1.output}));
    }
}