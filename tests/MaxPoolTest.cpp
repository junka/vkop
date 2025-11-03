#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>
#include <random>
#include <cmath>
#include <stack>

#include "setup.hpp"
#include "core/Tensor.hpp"
#include "include/logger.hpp"
#include "ops/Maxpool2d.hpp"

using vkop::core::Tensor;
using vkop::tests::TestCase;
using vkop::ops::Maxpool2d;

namespace {
    void maxpool2d_reference(const float *input, float *output, std::vector<int> shape,
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

        int padded_height = height + 2 * pad_height;
        int padded_width = width + 2 * pad_width;

        int out_height = (padded_height - kernel_h) / stride_height + 1;
        int out_width = (padded_width - kernel_w) / stride_width + 1;

        for (int b = 0; b < batch; ++b) {
            for (int c = 0; c < channels; ++c) {
                for (int oh = 0; oh < out_height; ++oh) {
                    for (int ow = 0; ow < out_width; ++ow) {
                        float max_val = -std::numeric_limits<float>::infinity();
                        for (int ph = 0; ph < kernel_h; ++ph) {
                            for (int pw = 0; pw < kernel_w; ++pw) {
                                int ih = oh * stride_height + ph - pad_height;
                                int iw = ow * stride_width + pw - pad_width;
                                if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
                                    int input_idx = ((b * channels + c) * height + ih) * width + iw;
                                    max_val = std::max(max_val, input[input_idx]);
                                }
                            }
                        }
                        int output_idx = ((b * channels + c) * out_height + oh) * out_width + ow;
                        output[output_idx] = max_val;
                    }
                }
            }
        }
    }

class MaxpoolTest : public TestCase {
public:
    std::shared_ptr<Tensor<float>> input;
    std::vector<float> expectedOutput;
    int stride_ = 1;
    int kernel_size_ = 4;
    int pad_ = 1;

    std::unordered_map<std::string, std::string> attributes = {
        {"kernel_shape", std::to_string(kernel_size_)},
        {"strides", std::to_string(stride_)},
        {"pads", std::to_string(pad_)},
        {"dilations", "1"},
        {"ceil_mode", "0"},
        {"storage_order", "1"}
    };

    MaxpoolTest() : TestCase("MaxPool") {
        initTestdata();
    }

private:
    void initTestdata() {
        std::vector<int> t = {1, 4, 8, 8};
        input = std::make_shared<Tensor<float>>(t);

        auto *input_ptr = input->data();
        expectedOutput.resize(input->num_elements());

        std::random_device rd{};
        std::mt19937 gen{rd()};
        gen.seed(1024);
        std::normal_distribution<> input_dist{-3.0F, 6.0F};
        for (int i = 0; i < input->num_elements(); i++) {
            input_ptr[i] = input_dist(gen);
        }

        maxpool2d_reference(input_ptr, expectedOutput.data(), t, kernel_size_, stride_, pad_);
    }
};
}

int main() {
    Logger::getInstance().setLevel(LOG_INFO);
    Logger::getInstance().enableFileOutput("log", false);

    MaxpoolTest maxtest;
    if (!maxtest.run_test({maxtest.input}, maxtest.expectedOutput,
        [&maxtest](std::unique_ptr<vkop::ops::Operator> &op) {
            auto *maxpool_op = dynamic_cast<Maxpool2d *>(op.get());
            if (!maxpool_op) {
                LOG_ERROR("Failed to cast operator to Softmax");
                return;
            }
            maxpool_op->setAttribute(maxtest.attributes);
        })) {
        return -1;
    }

    return 0;
}