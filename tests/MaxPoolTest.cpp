#include <string>
#include <unordered_map>
#include <vector>
#include <cmath>

#include "setup.hpp"
#include "core/Tensor.hpp"
#include "include/logger.hpp"
#include "ops/Maxpool2d.hpp"

using vkop::core::Tensor;
using vkop::tests::TestCase;
using vkop::ops::Maxpool2d;

namespace {
#if USE_CPP_REFER
template<typename T>
void maxpool2d_reference(const std::shared_ptr<Tensor<T>>& input, std::shared_ptr<Tensor<T>>& output, std::vector<int> shape,
                    int kernel_size, int stride_size, int pad_size) {
    int batch = shape[0];
    int channels = shape[1];
    int height = shape[2];
    int width = shape[3];
    int kernel_h = kernel_size;
    int kernel_w = kernel_size;
    int stride_height = stride_size;
    int stride_width = stride_size;
    int pad_height = pad_size;
    int pad_width = pad_size;

    int padded_height = height + (2 * pad_height);
    int padded_width = width + (2 * pad_width);

    int out_height = ((padded_height - kernel_h) / stride_height) + 1;
    int out_width = ((padded_width - kernel_w) / stride_width) + 1;

    for (int b = 0; b < batch; ++b) {
        for (int c = 0; c < channels; ++c) {
            for (int oh = 0; oh < out_height; ++oh) {
                for (int ow = 0; ow < out_width; ++ow) {
                    float max_val = -std::numeric_limits<float>::infinity();
                    for (int ph = 0; ph < kernel_h; ++ph) {
                        for (int pw = 0; pw < kernel_w; ++pw) {
                            int ih = (oh * stride_height) + ph - pad_height;
                            int iw = (ow * stride_width) + pw - pad_width;
                            if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
                                int input_idx = (((b * channels + c) * height + ih) * width) + iw;
                                if constexpr(std::is_same_v<T, float>) {
                                    max_val = std::max(max_val, (*input)[input_idx]);
                                } else {
                                    max_val = std::max(max_val, Tensor<T>::fp16_to_fp32((*input)[input_idx]));
                                }
                            }
                        }
                    }
                    int output_idx = (((b * channels + c) * out_height + oh) * out_width) + ow;
                    if constexpr(std::is_same_v<T, float>) {
                        (*output)[output_idx] = max_val;
                    } else {
                        (*output)[output_idx] = Tensor<T>::fp32_to_fp16(max_val);
                    }
                }
            }
        }
    }
}
#endif

template<typename T>
class MaxpoolTest : public TestCase<T> {
public:
    std::shared_ptr<Tensor<T>> input;
    std::shared_ptr<Tensor<T>> output;
    std::vector<int> input_shape_;
    int kernel_size_ = 3;
    int stride_ = 2;
    int pad_ = 1;

    std::unordered_map<std::string, std::string> attributes;

    MaxpoolTest(const std::vector<int>& input_shape, int kernel_size, int stride, int pad) : TestCase<T>("MaxPool"), input_shape_(input_shape), kernel_size_(kernel_size), stride_(stride), pad_(pad) {
        attributes = {
            {"kernel_shape", std::to_string(kernel_size_)},
            {"strides", std::to_string(stride_)},
            {"pads", std::to_string(pad_)},
            {"dilations", "1"},
            {"ceil_mode", "0"},
            {"storage_order", "1"}
        };
        initTestdata();
    }

private:
    void initTestdata() {
        input = std::make_shared<Tensor<T>>(input_shape_);
        output = std::make_shared<Tensor<T>>(input_shape_);

        torch::manual_seed(42);
        auto torch_input = torch::randn({input_shape_[0], input_shape_[1], input_shape_[2], input_shape_[3]}, this->getTorchConf());
        auto torch_output = torch::max_pool2d(torch_input, {kernel_size_, kernel_size_}, {stride_, stride_}, {pad_, pad_}, 1, false);
        printf("=======output tensor=======");
        std::cout << torch_output << std::endl;
        this->fillTensorFromTorch(input, torch_input);
        this->fillTensorFromTorch(output, torch_output);
    }
};
}


TEST(MaxPoolTest, MaxPoolComprehensiveTest) {
    const std::vector<std::tuple<std::vector<int>, int, int, int>> test_cases = {
        {{1, 3, 224, 224}, 3, 2, 1},
        {{1, 3, 16, 16}, 4, 2, 1},
    };

    for (const auto &test_case : test_cases) {
        auto [input_shape, kernel_size, stride, pad] = test_case;
        LOG_INFO("Testing Maxpool2d with input shape: [%d, %d, %d, %d], kernel_size: %d, stride: %d, pad: %d",
                input_shape[0], input_shape[1], input_shape[2], input_shape[3], kernel_size, stride, pad);
        LOG_INFO("Running test for fp32 ...");
        MaxpoolTest<float> maxtest(input_shape, kernel_size, stride, pad);
        EXPECT_TRUE(maxtest.run_test({maxtest.input}, {maxtest.output},
            [&maxtest](std::unique_ptr<vkop::ops::Operator> &op) {
                auto *maxpool_op = dynamic_cast<Maxpool2d *>(op.get());
                if (!maxpool_op) {
                    LOG_ERROR("Failed to cast operator to Maxpool2d");
                    return;
                }
                maxpool_op->setAttribute(maxtest.attributes);
            }));

        LOG_INFO("Running test for fp16 ...");
        MaxpoolTest<uint16_t> maxtest2(input_shape, kernel_size, stride, pad);
        EXPECT_TRUE(maxtest2.run_test({maxtest2.input}, {maxtest2.output},
            [&maxtest2](std::unique_ptr<vkop::ops::Operator> &op) {
                auto *maxpool_op = dynamic_cast<Maxpool2d *>(op.get());
                if (!maxpool_op) {
                    LOG_ERROR("Failed to cast operator to Maxpool2d");
                    return;
                }
                maxpool_op->setAttribute(maxtest2.attributes);
            }));
    }
}