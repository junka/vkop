#include <cstdint>
#include <vector>
#include <random>

#include "setup.hpp"
#include "core/Tensor.hpp"
#include "include/logger.hpp"

using vkop::core::Tensor;
using vkop::tests::TestCase;

namespace {

std::vector<float> batch_norm_2d(std::shared_ptr<Tensor<float>> &input, int batch, int channels, int height, int width,
                                 const std::vector<float>& scale, const std::vector<float>& bias, const std::vector<float>& mean,
                                 const std::vector<float>& variance, float epsilon) {
    std::vector<float> output(input->num_elements());
    int spatial_size = height * width;

    for (int n = 0; n < batch; ++n) {
        for (int c = 0; c < channels; ++c) {
            float inv_std = 1.0F / std::sqrt(variance[c] + epsilon);
            for (int h = 0; h < height; ++h) {
                for (int w = 0; w < width; ++w) {
                    int idx = n * channels * spatial_size + c * spatial_size + h * width + w;
                    output[idx] = scale[c] * (input->data()[idx] - mean[c]) * inv_std + bias[c];
                }
            }
        }
    }

    return output;
}

class BatchNorm2dTest : public TestCase {
public:
    std::shared_ptr<Tensor<float>> input;
    std::vector<float> expectedOutput;

    BatchNorm2dTest():TestCase("BatchNorm2d") {
        initTestdata();
    }
private:
    void initTestdata()
    {
        std::vector<int> t = {
            1, 3, 64, 64
        };
        input = std::make_shared<Tensor<float>>(t);

        auto *input_ptr = input->data();
        expectedOutput.resize(input->num_elements());

        std::random_device rd{};
        std::mt19937 gen{rd()};
        gen.seed(1024);
        std::normal_distribution<> inputa_dist{3.0F, 25.0F};
        std::normal_distribution<> b_dist{1.0F, 16.0F};
        for (int i = 0; i < input->num_elements(); i++) {
            input_ptr[i] = inputa_dist(gen);
        }
        std::vector<float> scale(t[1]);
        std::vector<float> bias(t[1]);
        std::vector<float> mean(t[1]);
        std::vector<float> variance(t[1]);
        for (int c = 0; c < t[1]; c++) {
            scale[c] = b_dist(gen);
            bias[c] = b_dist(gen);
            mean[c] = b_dist(gen);
            variance[c] = b_dist(gen);
        }
        expectedOutput = batch_norm_2d(input, t[0], t[1], t[2], t[3],
                    scale, bias, mean, variance, 1e-5);
    }
};
}

int main() {
    Logger::getInstance().setLevel(LOG_INFO);
    Logger::getInstance().enableFileOutput("log", false);

    BatchNorm2dTest bntest;
    if (!bntest.run_test({bntest.input}, bntest.expectedOutput)) {
        return -1;
    }

    return 0;
}