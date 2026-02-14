#include <cmath>
#include <iostream>
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
        torch::manual_seed(42);
        auto torch_input = torch::randn({input_shape_[0], input_shape_[1], input_shape_[2], input_shape_[3]});
        auto torch_weight = torch::randn({normalized_shape_[0], normalized_shape_[1]});
        auto torch_bias = torch::randn({normalized_shape_[0], normalized_shape_[1]});

        auto torch_output = torch::layer_norm(torch_input, torch::IntArrayRef({normalized_shape_[0], normalized_shape_[1]}), torch_weight, torch_bias, 1e-5);

        std::vector<int> output_shape = {};
        output_shape.reserve(torch_output.dim());
        for (int i = 0; i < torch_output.dim(); i++) {
            output_shape.push_back(torch_output.size(i));
        }

        printf("torch output size: [%d, %d, %d, %d]\n", output_shape[0], output_shape[1], output_shape[2], output_shape[3]);
        printf("=======weight ============\n");
        std::cout << torch_weight << std::endl;
        printf("\n=======bias ============\n");
        std::cout << torch_bias << std::endl;
        printf("\n===Input==============\n");
        std::cout << torch_input << std::endl;

        printf("\n===Output==============\n");
        std::cout << torch_output << std::endl;

        input = std::make_shared<Tensor<float>>(input_shape_);
        fillTensorFromTorch(input, torch_input);

        weight = std::make_shared<Tensor<float>>(normalized_shape_);
        fillTensorFromTorch(weight, torch_weight);

        bias = std::make_shared<Tensor<float>>(normalized_shape_);
        fillTensorFromTorch(bias, torch_bias);

        output = std::make_shared<Tensor<float>>(output_shape);
        fillTensorFromTorch(output, torch_output);
    }
};
}

TEST(LayerNormTest, LayerNormComprehensiveTest) {

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
                    if (std::fabs(bout[idx] - (*lntest.output)[idx]) > 1e-3) {
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

    EXPECT_TRUE(lntest.run_test<float>({lntest.input, lntest.weight, lntest.bias}, {lntest.output},
        [&lntest](std::unique_ptr<vkop::ops::Operator> &op) {
        auto *batchnorm_op = dynamic_cast<LayerNorm *>(op.get());
        if (!batchnorm_op) {
            LOG_ERROR("Failed to cast operator to LayerNorm");
            return;
        }
        batchnorm_op->setAttribute(lntest.param);
    }));

}