#include <unordered_map>
#include <vector>
#include <cmath>

#include "setup.hpp"
#include "core/Tensor.hpp"
#include "include/logger.hpp"
#include "ops/Softmax.hpp"

using vkop::core::Tensor;
using vkop::tests::TestCase;
using vkop::ops::Softmax;

#define USE_CPP_REFER 0

namespace {


    
/* When implementing the Softmax function, subtracting the maximum value for numerical stability is a common technique.
The purpose is to avoid numerical overflow or precision issues when computing exponentials.
In practice, if the input values x_i are large (e.g., close to 100 or higher), computing 
exp(x_i) can cause numerical overflow because the exponential function grows extremely rapidly. Conversely, if 
x_i is very small, exp(x_i) may underflow to zero, leading to loss of precision.

 y subtracting the maximum value max_val from the input vector, the Softmax formula becomes:
        softmax(x_i) = exp(x_i - max_val) / sum_j exp(x_j - max_val)

 This transformation does not change the Softmax output, as subtracting a constant from all elements preserves their relative proportions. However, it significantly improves numerical stability:
    Prevents overflow: : x_i - max_val  the exponentiated values remain bounded (at most 1), avoiding overflow 
    Improves precision:Shifting the inputs so their maximum is zero centers the values closer to zero, reducing floating-point rounding errors and mitigating underflow for smaller components.
*/
#if USE_CPP_REFER
void softmax_nd(const float* input, float* output,
                 int N, int C, int H, int W, int axis) {
    int dims[4] = {N, C, H, W};

    if (axis < 0 || axis >= 4) {
        std::cerr << "Invalid axis: must be in [0, 1, 2, 3]" << std::endl;
        return;
    }

    int group_size = dims[axis];
    int total_elements = N * C * H * W;
    int total_groups = total_elements / group_size;

    int strides[4];
    strides[3] = 1;           // W
    strides[2] = W;           // H
    strides[1] = H * W;       // C
    strides[0] = C * H * W;   // N

    for (int g = 0; g < total_groups; ++g) {
        int indices[4] = {0};
        int temp = g;
        int skip_dim = axis;

        for (int d = 3; d >= 0; --d) {
            if (d == skip_dim) continue;
            int dim_size = dims[d];
            indices[d] = temp % dim_size;
            temp /= dim_size;
        }

        std::vector<float> vals(group_size);
        std::vector<int>  idxs(group_size);

        for (int i = 0; i < group_size; ++i) {
            indices[skip_dim] = i;

            // flat index: n*strides[0] + c*strides[1] + h*strides[2] + w*strides[3]
            int flat_idx = 0;
            for (int d = 0; d < 4; ++d) {
                flat_idx += indices[d] * strides[d];
            }
            idxs[i] = flat_idx;
            vals[i] = input[flat_idx];
        }

        float max_val = -INFINITY;
        for (int i = 0; i < group_size; ++i) {
            if (vals[i] > max_val)
                max_val = vals[i];
        }

        float sum_exp = 0.0F;
        std::vector<float> exp_vals(group_size);
        for (int i = 0; i < group_size; ++i) {
            exp_vals[i] = std::exp(vals[i] - max_val);
            sum_exp += exp_vals[i];
        }

        for (int i = 0; i < group_size; ++i) {
            output[idxs[i]] = exp_vals[i] / sum_exp;
        }
    }
}
#endif

template <typename T>
class SoftmaxTest : public TestCase<T> {
public:
    std::vector<int> input_shape_;
    std::shared_ptr<Tensor<T>> input;
    std::shared_ptr<Tensor<T>> output;
    int axis_ = 0;
    std::unordered_map<std::string, std::string> dim;

    SoftmaxTest(const std::vector<int>& input_shape, int axis):TestCase<T>("Softmax"), input_shape_(input_shape), axis_(axis) {
        dim = {{"dim", std::to_string(axis_)}};
        initTestData();
    }
private:
    void initTestData() {
        std::vector<std::vector<int>> shapes;
        shapes.push_back(input_shape_);

        torch::manual_seed(42);
        torch::Tensor torch_input;
        if (input_shape_.size() == 2) {
            torch_input = torch::randn({input_shape_[0], input_shape_[1]}, this->getTorchConf());
        } else if (input_shape_.size() == 1) {
            torch_input = torch::randn({input_shape_[0]}, this->getTorchConf());
        } else {
            torch_input = torch::randn({input_shape_[0], input_shape_[1], input_shape_[2], input_shape_[3]}, this->getTorchConf());
        }

        int axis = 0;
        if (dim.count("axis") > 0) {
            axis = std::stoi(dim.at("axis"));
        } else if (dim.count("dim") > 0) {
            axis = std::stoi(dim.at("dim"));
        }

        if (axis < 0) {
            axis += torch_input.dim();
        }
        auto torch_output = torch::softmax(torch_input, axis);

        std::vector<int> output_shape = {};
        output_shape.reserve(torch_output.dim());
        for (int i = 0; i < torch_output.dim(); i++) {
            output_shape.push_back(torch_output.size(i));
        }

        printf("torch output size: [");
        for (auto &s: output_shape) {
            printf("%d, ", s);
        }
        printf("]\n");
#if USE_CPP_REFER
        printf("\n===verify C++ refer ==========\n");
        softmax_nd(torch_input.data(), torch_output.data(),
                    softtest.input_shape_[0], softtest.input_shape_[1],
                    softtest.input_shape_[2], softtest.input_shape_[3],
                    softtest.axis_);
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
                        printf("%.4f, ", torch_output[idx]);
                    }
                    printf("],\n");
                }
                printf("],\n");
            }
            printf("]\n");
        }
#endif

        // printf("\n===Input==============\n");
        // std::cout << torch_input << std::endl;

        // printf("\n===Output==============\n");
        // std::cout << torch_output << std::endl;

        input = std::make_shared<Tensor<T>>(input_shape_);
        this->fillTensorFromTorch(input, torch_input);

        output = std::make_shared<Tensor<T>>(output_shape);
        this->fillTensorFromTorch(output, torch_output);
    }
};
}

TEST(SoftmaxTest, SoftmaxComprehensiveTest) {

    const std::vector<std::tuple<std::vector<int>, int>> test_cases = {
        {{1, 10, 7, 7}, 1},
        {{1, 6, 8, 8}, 3},
        {{2, 4, 10, 10}, 2},
        {{1, 8, 6, 6}, 2},
        {{1, 16, 5, 5}, 1},
        {{4, 1000}, 1},
    };
    for (const auto &test_case: test_cases) {
        auto [input_shape, axis] = test_case;

        LOG_INFO("Testing FP32");
        SoftmaxTest<float> softtest(input_shape, axis);

        EXPECT_TRUE(softtest.run_test({softtest.input}, {softtest.output},
            [&softtest](std::unique_ptr<vkop::ops::Operator> &op) {
                auto *softmax_op = dynamic_cast<Softmax *>(op.get());
                if (!softmax_op) {
                    LOG_ERROR("Failed to cast operator to Softmax");
                    return;
                }
                softmax_op->setAttribute(softtest.dim);
            }));

        LOG_INFO("Testing FP16");
        SoftmaxTest<uint16_t> softtest2(input_shape, axis);
        EXPECT_TRUE(softtest2.run_test({softtest2.input}, {softtest2.output},
            [&softtest2](std::unique_ptr<vkop::ops::Operator> &op) {
                auto *softmax_op = dynamic_cast<Softmax *>(op.get());
                if (!softmax_op) {
                    LOG_ERROR("Failed to cast operator to Softmax");
                    return;
                }
                softmax_op->setAttribute(softtest2.dim);
            }));
    }
}