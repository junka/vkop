#include <cstdint>
#include <unordered_map>
#include <vector>
#include <random>
#include <cmath>
#include <stack>

#include "setup.hpp"
#include "core/Tensor.hpp"
#include "include/logger.hpp"
#include "ops/Softmax.hpp"

using vkop::core::Tensor;
using vkop::tests::TestCase;
using vkop::ops::Softmax;

#define USE_CPP_REFER 0

namespace {


    
/* 在实现 Softmax 函数时，寻找最大值用于数值稳定性是一个常见的技巧，
    目的是避免在计算指数时出现数值溢出或精度问题。
    Softmax 的公式在实际计算中, 如果 x_i 的值很大(例如接近 100 或更高), exp(x_i) 会导致数值溢出,
    因为指数函数增长非常快. 同样如果 x_i 的值很小, exp(x_i) 可能会接近 0, 导致精度损失.

    最大值的作用
    通过减去输入向量中的最大值 max_val, 公式变为:
        softmax(x_i) = exp(x_i - max_val) / sum_j exp(x_j - max_val)

    这样做不会改变 Softmax 的结果, 因为减去一个常数不会影响相对比例, 但可以显著提高数值稳定性:

    防止溢出: x_i - max_val 的值会变小, 避免了 exp(x_i) 计算时的溢出.
    提高精度: 减去最大值后, 输入值的范围更接近 0, 减少了浮点数计算中的精度损失.
*/
#if USE_CPP_REFER
void softmax_nd(const float* input, float* output,
                 int N, int C, int H, int W, int axis) {
    int dims[4] = {N, C, H, W};

    // 验证 axis
    if (axis < 0 || axis >= 4) {
        std::cerr << "Invalid axis: must be in [0, 1, 2, 3]" << std::endl;
        return;
    }

    // 计算 total_groups 和 group_size
    int group_size = dims[axis];
    int total_elements = N * C * H * W;
    int total_groups = total_elements / group_size;

    // 预计算 strides（NCHW 格式）
    int strides[4];
    strides[3] = 1;           // W
    strides[2] = W;           // H
    strides[1] = H * W;       // C
    strides[0] = C * H * W;   // N

    // 遍历每一个 group（即每个非 axis 维度的组合）
    for (int g = 0; g < total_groups; ++g) {
        // 分解 group index -> (n, c, h, w)，跳过 axis 维度
        int indices[4] = {0};
        int temp = g;
        int skip_dim = axis;

        // 从后往前分解非 axis 维度的索引
        for (int d = 3; d >= 0; --d) {
            if (d == skip_dim) continue;
            int dim_size = dims[d];
            indices[d] = temp % dim_size;
            temp /= dim_size;
        }

        // 找到当前 group 中沿 axis 维度的所有值
        std::vector<float> vals(group_size);
        std::vector<int>  idxs(group_size); // 存储实际 flat 索引

        for (int i = 0; i < group_size; ++i) {
            // 设置 axis 维度为 i
            indices[skip_dim] = i;

            // 计算 flat index: n*strides[0] + c*strides[1] + h*strides[2] + w*strides[3]
            int flat_idx = 0;
            for (int d = 0; d < 4; ++d) {
                flat_idx += indices[d] * strides[d];
            }
            idxs[i] = flat_idx;
            vals[i] = input[flat_idx];
        }

        // Step 1: 找最大值用于数值稳定
        float max_val = -INFINITY;
        for (int i = 0; i < group_size; ++i) {
            if (vals[i] > max_val)
                max_val = vals[i];
        }

        // Step 2: 计算 exp(x_i - max_val) 和 sum
        float sum_exp = 0.0F;
        std::vector<float> exp_vals(group_size);
        for (int i = 0; i < group_size; ++i) {
            exp_vals[i] = std::exp(vals[i] - max_val);
            sum_exp += exp_vals[i];
        }

        // Step 3: 归一化并写入 output
        for (int i = 0; i < group_size; ++i) {
            output[idxs[i]] = exp_vals[i] / sum_exp;
        }
    }
}
#endif

class SoftmaxTest : public TestCase {
public:
    std::vector<int> input_shape_ = {12, 15, 8, 8};
    std::shared_ptr<Tensor<float>> input;
    std::shared_ptr<Tensor<float>> output;
    int axis_ = 1;
    const std::unordered_map<std::string, std::string> dim = {{"dim", std::to_string(axis_)}};

    SoftmaxTest():TestCase("Softmax") {
        initTestData();
    }
private:
    void initTestData() {
        std::vector<std::vector<int>> shapes;
        shapes.push_back(input_shape_);

        std::tuple<std::vector<std::vector<float>>, std::vector<int>> k = TestCase::execute_torch_operator("softmax", shapes, dim);
        std::vector<std::vector<float>> torch_tensors = std::get<0>(k);
        auto torch_output = torch_tensors[0];
        auto torch_input = torch_tensors[1];
        std::vector<int> output_shape = std::get<1>(k);

        printf("torch output size: [%d, %d, %d, %d]\n", output_shape[0], output_shape[1], output_shape[2], output_shape[3]);
#if 1
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
        output = std::make_shared<Tensor<float>>(output_shape);
        output->fillToCPU(torch_output);
    }
};
}

int main() {
    Logger::getInstance().setLevel(LOG_INFO);
    Logger::getInstance().enableFileOutput("log", false);

    SoftmaxTest softtest;

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

    if (!softtest.run_test<float>({softtest.input}, {softtest.output},
        [&softtest](std::unique_ptr<vkop::ops::Operator> &op) {
            auto *softmax_op = dynamic_cast<Softmax *>(op.get());
            if (!softmax_op) {
                LOG_ERROR("Failed to cast operator to Softmax");
                return;
            }
            softmax_op->setAttribute(softtest.dim);
        })) {
        return -1;
    }

    return 0;
}