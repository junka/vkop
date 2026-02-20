#include <vector>

#include "setup.hpp"
#include "core/Tensor.hpp"
#include "include/logger.hpp"

using vkop::core::Tensor;
using vkop::tests::TestCase;
using vkop::ops::Reshape;

namespace {

template <typename T>
class ReshapeTest : public TestCase<T> {
public:
    std::vector<int> input_shape_;
    std::vector<int> reshape_shape_;

    std::unordered_map<std::string, std::string> attributes;
    std::shared_ptr<Tensor<T>> input;
    std::shared_ptr<Tensor<int64_t>> shape;
    std::shared_ptr<Tensor<T>> output;

    ReshapeTest(std::vector<int> &inputshape, std::vector<int> &reshapeshape, bool allowzero):TestCase<T>("Reshape"), input_shape_(inputshape), reshape_shape_(reshapeshape) {
        attributes = {
            {"allowzero", allowzero ? "1" : "0"},
        };
        initTestdata();
    }
private:
    void initTestdata()
    {
        input = std::make_shared<Tensor<T>>(input_shape_);
        shape = std::make_shared<Tensor<int64_t>>(reshape_shape_.size());
        std::vector<int64_t> reshape(reshape_shape_.begin(), reshape_shape_.end());
        shape->fillToCPU(reshape);
        output = std::make_shared<Tensor<T>>(reshape_shape_);

        std::vector<int64_t> inshape(input_shape_.begin(), input_shape_.end());
        auto torch_input = torch::randn(inshape, this->getTorchConf());
        std::cout << torch_input << std::endl;
        auto torch_output = torch::reshape(torch_input, reshape);
        this->fillTensorFromTorch(input, torch_input);
        this->fillTensorFromTorch(output, torch_output);
        printf("=======input==============\n");
        input->print_tensor();
        printf("=========ouput============\n");
        output->print_tensor();
    }
};
}

TEST(ReshapeTest, ReshapeComprehensiveTest) {

    const std::vector<std::tuple<std::vector<int>, std::vector<int>, bool>> test_cases = {
        {{1, 8, 4, 4}, {8, 4, 4}, false},
        {{1, 8, 4, 4}, {1, 4, 8, 4}, false},
        {{1, 8, 4, 4}, {1, 8, 16}, false},
        {{1, 2, 64, 400}, {1, 128, 20, 20}, false},
    };
    for (const auto &test_case : test_cases) {
        auto [input_shape, reshape_shape, allowzero] = test_case;

        LOG_INFO("Testing Reshape");
        ReshapeTest<float> reshape_test(input_shape, reshape_shape, allowzero);
        EXPECT_TRUE(reshape_test.run_test({reshape_test.input, reshape_test.shape}, {reshape_test.output}, [&reshape_test](std::unique_ptr<vkop::ops::Operator> &op) {
            auto *reshape_op = dynamic_cast<Reshape *>(op.get());
            if (!reshape_op) {
                LOG_ERROR("Failed to cast operator to Reshape");
                return;
            }
            reshape_op->setAttribute(reshape_test.attributes);
        }));

        ReshapeTest<uint16_t> reshape_test1(input_shape, reshape_shape, allowzero);
        EXPECT_TRUE(reshape_test1.run_test({reshape_test1.input, reshape_test1.shape}, {reshape_test1.output}, [&reshape_test1](std::unique_ptr<vkop::ops::Operator> &op) {
            auto *reshape_op = dynamic_cast<Reshape *>(op.get());
            if (!reshape_op) {
                LOG_ERROR("Failed to cast operator to Reshape");
                return;
            }
            reshape_op->setAttribute(reshape_test1.attributes);
        }));
    }
}