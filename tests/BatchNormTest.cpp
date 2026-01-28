#include <memory>
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

class BatchNormTest : public TestCase {
public:
    std::vector<int> input_shape_ = {
        1, 5, 2, 2
    };
    const std::unordered_map<std::string, std::string> param = {
        {"eps", "1e-5"}
    };
    std::shared_ptr<Tensor<float>> input;
    std::shared_ptr<Tensor<float>> output;
    std::shared_ptr<Tensor<float>> para;

    BatchNormTest():TestCase("BatchNorm") {
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
        auto input_cpu = torch_input.cpu().contiguous();
        std::vector<float> input_vector;
        input_vector.reserve(input_cpu.numel());
        auto input_accessor = input_cpu.accessor<float, 4>();
        for (int i = 0; i < input_shape_[0]; i++) {
            for (int j = 0; j < input_shape_[1]; j++) {
                for (int k = 0; k < input_shape_[2]; k++) {
                    for (int l = 0; l < input_shape_[3]; l++) {
                        input_vector.push_back(input_accessor[i][j][k][l]);
                    }
                }
            }
        }
        input->fillToCPU(input_vector);

        output = std::make_shared<Tensor<float>>(output_shape);
        auto output_cpu = torch_output.cpu().contiguous();
        std::vector<float> output_vector;
        output_vector.reserve(output_cpu.numel());
        auto output_accessor = output_cpu.accessor<float, 4>();
        for (int i = 0; i < output_shape[0]; i++) {
            for (int j = 0; j < output_shape[1]; j++) {
                for (int k = 0; k < output_shape[2]; k++) {
                    for (int l = 0; l < output_shape[3]; l++) {
                        output_vector.push_back(output_accessor[i][j][k][l]);
                    }
                }
            }
        }
        output->fillToCPU(output_vector);

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

int main() {
    Logger::getInstance().setLevel(LOG_INFO);
    Logger::getInstance().enableFileOutput("log", false);

    BatchNormTest bntest;

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

    if (!bntest.run_test<float>({bntest.input, bntest.para}, {bntest.output},
        [&bntest](std::unique_ptr<vkop::ops::Operator> &op) {
            auto *batchnorm_op = dynamic_cast<BatchNorm *>(op.get());
            if (!batchnorm_op) {
                LOG_ERROR("Failed to cast operator to BatchNorm");
                return;
            }
            batchnorm_op->setAttribute(bntest.param);
        })) {
        return -1;
    }

    return 0;
}