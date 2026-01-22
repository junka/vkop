#include <memory>
#include <vector>

#include "setup.hpp"
#include "core/Tensor.hpp"
#include "include/logger.hpp"
#include "ops/LayerNorm.hpp"

using vkop::core::Tensor;
using vkop::tests::TestCase;
using vkop::ops::LayerNorm;
#define USE_CPP_REFER 1
namespace {
#if USE_CPP_REFER

std::vector<float> layer_norm(std::shared_ptr<Tensor<float>> &input, std::vector<int> input_shape, std::vector<int> nshape,
                              std::shared_ptr<Tensor<float>>& weight, std::shared_ptr<Tensor<float>>& bias, float epsilon) {
    size_t total_size = input->num_elements();
    std::vector<float> output(total_size);

    int rank = input_shape.size();
    int norm_rank = nshape.size();

    if (norm_rank > rank) {
        throw std::invalid_argument("normalized_shape has more dims than input");
    }

    for (int i = 0; i < norm_rank; ++i) {
        if (input_shape[rank - norm_rank + i] != nshape[i]) {
            throw std::invalid_argument("normalized_shape does not match trailing dims of input");
        }
    }

    int inner_size = 1;
    for (int s : nshape) {
        inner_size *= s;
    }

    int outer_size = total_size / inner_size;

    bool has_weight = (inner_size == weight->num_elements());
    bool has_bias = (inner_size == bias->num_elements());

    if (weight->num_elements() != inner_size) {
        throw std::invalid_argument("weight size must match product of normalized_shape");
    }
    if (bias->num_elements() != inner_size) {
        throw std::invalid_argument("bias size must match product of normalized_shape");
    }

    for (int outer_idx = 0; outer_idx < outer_size; ++outer_idx) {
        const int in_block_offset = outer_idx * inner_size;
        float*       out_block = output.data() + (outer_idx * inner_size);

        float sum = 0.0F;
        for (int i = 0; i < inner_size; ++i) {
            sum += (*input)[in_block_offset+i];
        }
        float mean = sum / inner_size;

        float var = 0.0F;
        for (int i = 0; i < inner_size; ++i) {
            float diff = (*input)[in_block_offset+i] - mean;
            var += diff * diff;
        }
        var /= inner_size;
        float inv_std = 1.0F / std::sqrt(var + epsilon);

        for (int i = 0; i < inner_size; ++i) {
            float normalized_val = ((*input)[in_block_offset+i] - mean) * inv_std;
            if (has_weight) normalized_val *= (*weight)[i];
            if (has_bias)   normalized_val += (*bias)[i];
            out_block[i] = normalized_val;
        }
    }

    return output;
}
#endif

class LayerNormTest : public TestCase {
public:
    std::vector<int> input_shape_ = {
        2, 5, 4, 4
    };
    std::vector<int> normalized_shape_ = {4, 4};
    const std::unordered_map<std::string, std::string> param = {
        {"eps", "1e-5"}, {"normalized_shape", "[4, 4]"}
    };
    std::shared_ptr<Tensor<float>> input;
    std::shared_ptr<Tensor<float>> weight;
    std::shared_ptr<Tensor<float>> bias;
    std::shared_ptr<Tensor<float>> output;

    LayerNormTest():TestCase("LayerNorm") {
        initTestdata();
    }
private:
    void initTestdata()
    {
        std::vector<std::vector<int>> shapes;
        shapes.push_back(input_shape_);
        shapes.push_back(normalized_shape_);
        shapes.push_back(normalized_shape_);

        std::tuple<std::vector<std::vector<float>>, std::vector<int>> k = TestCase::execute_torch_operator("layer_norm", shapes, param);
        std::vector<std::vector<float>> torch_tensors = std::get<0>(k);
        std::vector<int> output_shape = std::get<1>(k);
        const auto& torch_output = torch_tensors[0];
        const auto& torch_input = torch_tensors[1];
        const auto& torch_weight = torch_tensors[2];
        const auto& torch_bias = torch_tensors[3];

        printf("torch output size: [%d, %d, %d, %d]\n", output_shape[0], output_shape[1], output_shape[2], output_shape[3]);
#if 1
        int size = 1;
        for (int s : normalized_shape_) {
            size *= s;
        }
        printf("=======weight ============\n");
        for (int i = 0; i < size; i++) {
            printf("%.4f, ", torch_weight[i]);
        }
        printf("\n=======bias ============\n");
        for (int i = 0; i < size; i++) {
            printf("%.4f, ", torch_bias[i]);
        }
        printf("\n===Input==============\n");
        for (int i = 0; i < output_shape[0]; i++) {
            printf("[\n");
            for (int j = 0; j < output_shape[1]; j++) {
                printf("[\n");
                for (int k = 0; k < output_shape[2]; k++) {
                    printf("[");
                    for (int l = 0; l < output_shape[3]; l++) {
                        int idx = (i * output_shape[1] * output_shape[2] * output_shape[3]) +
                                (j * output_shape[2] * output_shape[3]) +
                                (k * output_shape[3]) +
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
                        int idx = (i * output_shape[1] * output_shape[2] * output_shape[3]) +
                                (j * output_shape[2] * output_shape[3]) +
                                (k * output_shape[3]) +
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
        weight = std::make_shared<Tensor<float>>(normalized_shape_);
        weight->fillToCPU(torch_weight);
        bias = std::make_shared<Tensor<float>>(normalized_shape_);
        bias->fillToCPU(torch_bias);
        output = std::make_shared<Tensor<float>>(output_shape);
        output->fillToCPU(torch_output);
    }
};
}

int main() {
    Logger::getInstance().setLevel(LOG_INFO);
    Logger::getInstance().enableFileOutput("log", false);

    LayerNormTest lntest;

#if USE_CPP_REFER
    printf("\n===verify C++ refer ==========\n");
    auto bout = layer_norm(lntest.input, lntest.input_shape_, lntest.normalized_shape_, lntest.weight, lntest.bias,
                1e-5);
    for (int i = 0; i < lntest.input_shape_[0]; i++) {
        printf("[\n");
        for (int j = 0; j < lntest.input_shape_[1]; j++) {
            printf("[\n");
            for (int k = 0; k < lntest.input_shape_[2]; k++) {
                printf("[");
                for (int l = 0; l < lntest.input_shape_[3]; l++) {
                    int idx = (i * lntest.input_shape_[1] * lntest.input_shape_[2] * lntest.input_shape_[3]) +
                              (j * lntest.input_shape_[2] * lntest.input_shape_[3]) +
                              (k * lntest.input_shape_[3]) + l;
                    printf("%.4f, ", bout[idx]);
                    if (fabs(bout[idx] - (*lntest.output)[idx]) > 1e-3) {
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

    if (!lntest.run_test<float>({lntest.input, lntest.weight, lntest.bias}, {lntest.output},
        [&lntest](std::unique_ptr<vkop::ops::Operator> &op) {
            auto *batchnorm_op = dynamic_cast<LayerNorm *>(op.get());
            if (!batchnorm_op) {
                LOG_ERROR("Failed to cast operator to LayerNorm");
                return;
            }
            batchnorm_op->setAttribute(lntest.param);
        })) {
        return -1;
    }

    return 0;
}