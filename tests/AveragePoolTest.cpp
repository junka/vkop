#include <vector>

#include "setup.hpp"
#include "core/Tensor.hpp"
#include "include/logger.hpp"
#include "ops/AveragePool.hpp"

using vkop::core::Tensor;
using vkop::tests::TestCase;
using vkop::ops::AveragePool;

namespace {
class AveragePoolTest : public TestCase {
public:
    std::vector<int> input_shape_;
    std::vector<int> kernel_shape_;
    std::vector<int> strides_;
    std::vector<int> pads_;
    bool count_include_pad_;

    std::shared_ptr<Tensor<float>> input;
    std::shared_ptr<Tensor<float>> output;
    std::unordered_map<std::string, std::string> attributes;

    AveragePoolTest(std::vector<int> input_shape, std::vector<int> kernel_shape, std::vector<int> strides, std::vector<int> pads, bool count_include_pad):
        TestCase("AveragePool"), input_shape_(std::move(input_shape)), kernel_shape_(std::move(kernel_shape)),
        strides_(std::move(strides)), pads_(std::move(pads)) {
        attributes = {
            {"auto_pad", "NOTSET"},
            {"pads", "[" + std::to_string(pads_[0]) + "," + std::to_string(pads_[1]) + "," + std::to_string(pads_[2]) + "," + std::to_string(pads_[3]) + "]"},
            {"strides", "[" + std::to_string(strides_[0]) + "," + std::to_string(strides_[1]) + "]"},
            {"kernel_shape", "[" + std::to_string(kernel_shape_[0]) + "," + std::to_string(kernel_shape_[1]) + "]"},
            {"count_include_pad", std::to_string(count_include_pad_? 1 : 0)},
        };
        initTestdata();
    }
private:
    void initTestdata()
    {
        torch::manual_seed(42);
        auto torch_input = torch::randn({input_shape_[0], input_shape_[1], input_shape_[2], input_shape_[3]});
        std::vector<int64_t> kernel_sizes = {kernel_shape_[0], kernel_shape_[1]};
        std::vector<int64_t> strides = {strides_[0], strides_[1]};
        std::vector<int64_t> paddings = {pads_[0], pads_[2]};

        bool ceil_mode = attributes.count("ceil_mode") ? (std::stoi(attributes.at("ceil_mode")) != 0) : false;
        bool count_include_pad = attributes.count("count_include_pad") ? (std::stoi(attributes.at("count_include_pad")) != 0) : true;

        auto torch_output = torch::avg_pool2d(torch_input,
            torch::ArrayRef<int64_t>(kernel_sizes),
            torch::ArrayRef<int64_t>(strides),
            torch::ArrayRef<int64_t>(paddings),
            ceil_mode,
            count_include_pad
        );

        std::vector<int> output_shape = {};
        output_shape.reserve(torch_output.dim());
        for (int i = 0; i < torch_output.dim(); i++) {
            output_shape.push_back(torch_output.size(i));
        }

        input = std::make_shared<Tensor<float>>(input_shape_);
        fillTensorFromTorch(input, torch_input);

        output = std::make_shared<Tensor<float>>(output_shape);
        fillTensorFromTorch(output, torch_output);

        printf("\n===Input==============\n");
        std::cout << torch_input << std::endl;

        printf("\n===Output==============\n");
        printf("[ %d, %d, %d, %d ]\n", output_shape[0], output_shape[1], output_shape[2], output_shape[3]);
        std::cout << torch_output << std::endl;
    }
};
}

TEST(AveragePoolTest, AveragePoolComprehensiveTest) {

    std::vector<std::tuple<std::vector<int>, std::vector<int>, std::vector<int>, std::vector<int>, bool>> test_configs = {
        {{1, 3, 32, 32}, {4, 8}, {2, 4}, {0, 0, 0, 0}, true},
        {{1, 3, 32, 32}, {4, 8}, {2, 4}, {1, 1, 1, 1}, false},
        {{1, 8, 28, 28}, {5, 5}, {1, 1}, {2, 2, 2, 2}, true},
    };
    for (const auto& config : test_configs) {
        auto [input_shape, kernel_shape, strides, pads, count_include_pad] = config;
        LOG_INFO("Testing AveragePool with input shape: [ %d, %d, %d, %d ], kernel shape: [ %d, %d ], strides: [ %d, %d ], pads: [ %d, %d, %d, %d ], count_include_pad: %s",
            input_shape[0], input_shape[1], input_shape[2], input_shape[3],
            kernel_shape[0], kernel_shape[1],
            strides[0], strides[1], pads[0], pads[1], pads[2], pads[3],
            count_include_pad ? "true" : "false");
        AveragePoolTest aptest(input_shape, kernel_shape, strides, pads, count_include_pad);
        EXPECT_TRUE (aptest.run_test<float>({aptest.input}, {aptest.output},
            [&aptest](std::unique_ptr<vkop::ops::Operator> &op) {
            auto *ap_op = dynamic_cast<AveragePool *>(op.get());
            if (!ap_op) {
                LOG_ERROR("Failed to cast operator to AveragePool");
                return;
            }
            ap_op->setAttribute(aptest.attributes);
        }));
    }
}