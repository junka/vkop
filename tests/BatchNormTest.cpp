#include <cstdint>
#include <memory>
#include <vector>
#include <random>

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
        1, 3, 4, 4
    };
    const std::unordered_map<std::string, std::string> param = {
        {"momentum", "0.1"}, {"eps", "1e-5"}
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
        std::vector<std::vector<int>> shapes;
        std::vector<int> num_feature(1);
        num_feature[0] = input_shape_[1];
        shapes.push_back(input_shape_);
        shapes.push_back(num_feature);
        shapes.push_back(num_feature);
        shapes.push_back(num_feature);
        shapes.push_back(num_feature);


        std::tuple<std::vector<std::vector<float>>, std::vector<int>> k = TestCase::execute_torch_operator("batch_norm", shapes, param);
        std::vector<std::vector<float>> torch_tensors = std::get<0>(k);
        std::vector<int> output_shape = std::get<1>(k);
        auto torch_output = torch_tensors[0];
        auto torch_input = torch_tensors[1];
        auto torch_mean = torch_tensors[2];
        auto torch_var = torch_tensors[3];

        printf("torch output size: [%d, %d, %d, %d]\n", output_shape[0], output_shape[1], output_shape[2], output_shape[3]);
#if 1
        printf("=======mean ============\n");
        for (int i = 0; i < output_shape[1]; i++) {
            printf("%.4f, ", torch_mean[i]);
        }
        printf("\n=======var ============\n");
        for (int i = 0; i < output_shape[1]; i++) {
            printf("%.4f, ", torch_var[i]);
        }
        printf("\n===Input==============\n");
        for (int i = 0; i < output_shape[0]; i++) {
            printf("[\n");
            for (int j = 0; j < output_shape[1]; j++) {
                printf("[\n");
                for (int k = 0; k < output_shape[2]; k++) {
                    printf("[");
                    for (int l = 0; l < output_shape[3]; l++) {
                        int idx = i * output_shape[1] * output_shape[2] * output_shape[3] +
                                j * output_shape[2] * output_shape[3] +
                                k * output_shape[3] +
                                l;
                        printf("%.4f, ", torch_input[idx]);
                    }
                    printf("],\n");
                }
                printf("],\n");
            }
            printf("]\n");
        }

        printf("\n===Output==============\n");
        for (int i = 0; i < output_shape[0]; i++) {
            for (int j = 0; j < output_shape[1]; j++) {
                for (int k = 0; k < output_shape[2]; k++) {
                    printf("[");
                    for (int l = 0; l < output_shape[3]; l++) {
                        int idx = i * output_shape[1] * output_shape[2] * output_shape[3] +
                                j * output_shape[2] * output_shape[3] +
                                k * output_shape[3] +
                                l;
                        printf("%.4f, ", torch_output[idx]);
                    }
                    printf("]\n");
                }
                printf("\n");
            }
            printf("\n");
        }
#endif
        input = std::make_shared<Tensor<float>>(input_shape_);
        input->fillToCPU(torch_input);
        // mean = std::make_shared<Tensor<float>>(num_feature);
        // mean->fillToCPU(torch_mean);
        // var = std::make_shared<Tensor<float>>(num_feature);
        // var->fillToCPU(torch_var);
        output = std::make_shared<Tensor<float>>(output_shape);
        output->fillToCPU(torch_output);
        para = std::make_shared<Tensor<float>>(std::vector<int>{input_shape_[1], 4});
        para->reserveOnCPU();
        for (int i = 0; i < output_shape[1]; i++) {
            (*para)[i * 4] = torch_mean[i];
            (*para)[i * 4 + 1] = torch_var[i];
            (*para)[i * 4 + 2] = 1.0F;
            (*para)[i * 4 + 3] = 0.0F;
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