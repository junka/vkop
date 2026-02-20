#include <memory>
#include <utility>
#include <vector>

#include "setup.hpp"
#include "core/Tensor.hpp"
#include "include/logger.hpp"
#include "ops/BatchNorm.hpp"

using vkop::core::Tensor;
using vkop::tests::TestCase;
using vkop::ops::BatchNorm;
#define USE_CPP_REFER 0
namespace {
#if USE_CPP_REFER

std::vector<float> batch_norm_2d(std::shared_ptr<Tensor<float>> &input, int batch, int channels, int height, int width,
                                 const std::vector<float>& weight, const std::vector<float>& bias, std::shared_ptr<Tensor<float>>& mean,
                                 std::shared_ptr<Tensor<float>>& variance, float epsilon) {
    std::vector<float> output(input->num_elements());
    int spatial_size = height * width;

    for (int n = 0; n < batch; ++n) {
        for (int c = 0; c < channels; ++c) {
            float inv_std = 1.0F / std::sqrt(variance->data()[c] + epsilon);
            for (int h = 0; h < height; ++h) {
                for (int w = 0; w < width; ++w) {
                    int idx = n * channels * spatial_size + c * spatial_size + h * width + w;
                    output[idx] = weight[c] * (input->data()[idx] - mean->data()[c]) * inv_std + bias[c];
                }
            }
        }
    }

    return output;
}
#endif

template <typename T>
class BatchNormTest : public TestCase<T> {
public:
    std::vector<int> input_shape_;
    const std::unordered_map<std::string, std::string> param = {
        {"eps", "1e-5"}
    };
    std::shared_ptr<Tensor<float>> input;
    std::shared_ptr<Tensor<float>> output;
    std::shared_ptr<Tensor<float>> para;

    explicit BatchNormTest(std::vector<int> input_shape):TestCase<T>("BatchNorm"), input_shape_(std::move(input_shape)) {
        initTestdata();
    }
private:
    void initTestdata()
    {
        torch::manual_seed(42);
        auto torch_input = torch::randn({input_shape_[0], input_shape_[1], input_shape_[2], input_shape_[3]});

        auto torch_weight = torch::ones({input_shape_[1]});  // gamma
        auto torch_bias = torch::zeros({input_shape_[1]});  // beta
        auto torch_running_mean = torch::zeros({input_shape_[1]});  // running mean
        auto torch_running_var = torch::ones({input_shape_[1]});    // running var

        float eps = param.count("eps") ? std::stof(param.at("eps")) : 1e-5F;
        float momentum = param.count("momentum") ? std::stof(param.at("momentum")) : 0.1F;

        auto torch_output = torch::batch_norm(
            torch_input,
            torch_weight,
            torch_bias,
            torch_running_mean,
            torch_running_var,
            false,
            momentum,
            eps,
            false
        );

        std::vector<int> output_shape = {};
        output_shape.reserve(torch_output.dim());
        for (int i = 0; i < torch_output.dim(); i++) {
            output_shape.push_back(torch_output.size(i));
        }

        std::vector<float> torch_mean;
        std::vector<float> torch_var;
        auto mean_cpu = torch_running_mean.cpu().contiguous();
        auto var_cpu = torch_running_var.cpu().contiguous();
        for (int i = 0; i < mean_cpu.size(0); i++) {
            torch_mean.push_back(mean_cpu[i].item<float>());
            torch_var.push_back(var_cpu[i].item<float>());
        }

        printf("torch output size: [%d, %d, %d, %d]\n", output_shape[0], output_shape[1], output_shape[2], output_shape[3]);
        printf("=======mean ============\n");
        for (int i = 0; i < output_shape[1]; i++) {
            printf("%.4f, ", torch_mean[i]);
        }
        printf("\n=======var ============\n");
        for (int i = 0; i < output_shape[1]; i++) {
            printf("%.4f, ", torch_var[i]);
        }
        printf("\n===Input==============\n");
        std::cout << torch_input << std::endl;

        printf("\n===Output==============\n");
        std::cout << torch_output << std::endl;

        input = std::make_shared<Tensor<float>>(input_shape_);
        this->fillTensorFromTorch(input, torch_input);

        output = std::make_shared<Tensor<float>>(output_shape);
        this->fillTensorFromTorch(output, torch_output);

        para = std::make_shared<Tensor<float>>(std::vector<int>{UP_DIV(input_shape_[1], 4) * 16});
        para->reserveOnCPU();
        // fill every element as vec4
        int c = input->get_channel();
        int c4 = UP_DIV(c, 4);
        for (int i = 0; i < c4; i++) {
            for (int j = 0; j < 4; j++) {
                if (i * 4 + j >= c) {
                    break;
                }
                (*para)[(i * 16) + j] = 1.0F;
                (*para)[(i * 16) + 4 + j] = 0.0F;
                (*para)[(i * 16) + 8 + j] = torch_mean[(i * 4) + j];
                (*para)[(i * 16) + 12 + j] = torch_var[(i * 4) + j];
            }
        }
    }
};
}

TEST(BatchNormTest, BatchNormComprehensiveTest) {

    std::vector<std::vector<int>> test_configs = {
        {1, 3, 32, 32},
        {2, 8, 28, 28},
        {4, 16, 16, 16},
    };
    for (auto config : test_configs) {
        LOG_INFO("======test config: [%d, %d, %d, %d]\n", config[0], config[1], config[2], config[3]);
        BatchNormTest<float> bntest(config);

#if USE_CPP_REFER
        printf("\n===verify C++ refer ==========\n");
        std::vector<float> torch_weight(bntest.input_shape_[1], 1.0F);
        std::vector<float> torch_bias(bntest.input_shape_[1], 0.0F);
        auto bout = batch_norm_2d(bntest.input, bntest.input_shape_[0], bntest.input_shape_[1],
                    bntest.input_shape_[2], bntest.input_shape_[3], torch_weight, torch_bias,
                    bntest.mean, bntest.var, 1e-5);
        for (int i = 0; i < bntest.input_shape_[0]; i++) {
            printf("[\n");
            for (int j = 0; j < bntest.input_shape_[1]; j++) {
                printf("[\n");
                for (int k = 0; k < bntest.input_shape_[2]; k++) {
                    printf("[");
                    for (int l = 0; l < bntest.input_shape_[3]; l++) {
                        int idx = i * bntest.input_shape_[1] * bntest.input_shape_[2] * bntest.input_shape_[3] +
                                j * bntest.input_shape_[2] * bntest.input_shape_[3] +
                                k * bntest.input_shape_[3] +
                                l;
                        printf("%.4f, ", bout[idx]);
                        if (fabs(bout[idx] - (*bntest.output)[idx]) > 1e-3) {
                            printf("  <--mismatch ");
                        }
                    }
                    printf("],\n");
                }
                printf("],\n");
            }
            printf("]\n");
        }
#endif

    EXPECT_TRUE (bntest.run_test({bntest.input, bntest.para}, {bntest.output},
        [&bntest](std::unique_ptr<vkop::ops::Operator> &op) {
            auto *batchnorm_op = dynamic_cast<BatchNorm *>(op.get());
            if (!batchnorm_op) {
                LOG_ERROR("Failed to cast operator to BatchNorm");
                return;
            }
            batchnorm_op->setAttribute(bntest.param);
        }));
    }

}