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

namespace {

void softmax_nd(const float* input, float* output,
                 int N, int C, int H, int W, int axis) {
    int dims[4] = {N, C, H, W};

    // 如果 axis 不合法
    if (axis < 0 || axis >= 4) {
        std::cerr << "Invalid axis: must be in [0, 1, 2, 3]" << std::endl;
        return;
    }

    // 计算 softmax group size 和 total groups
    int group_size = dims[axis];
    int total_groups = 1;

    for (int i = 0; i < 4; ++i) {
        if (i != axis) {
            total_groups *= dims[i];
        }
    }

    // if (group_size == 1) {
    //     for (int i = 0; i < N * C * H * W; ++i) {
    //         output[i] = 1.0F;
    //     }
    //     return;
    // }
    /* 在实现 Softmax 函数时，寻找最大值用于数值稳定性是一个常见的技巧，
        目的是避免在计算指数时出现数值溢出或精度问题。
        Softmax 的公式在实际计算中, 如果 x_i 的值很大(例如接近 100 或更高), exp(x_i) 会导致数值溢出,
        因为指数函数增长非常快. 同样如果 x_i 的值很小, exp(x_i) 可能会接近 0, 导致精度损失.

        最大值的作用
        通过减去输入向量中的最大值 max_val, 公式变为:

        这样做不会改变 Softmax 的结果, 因为减去一个常数不会影响相对比例, 但可以显著提高数值稳定性:

        防止溢出: x_i - max_val 的值会变小, 避免了 exp(x_i) 计算时的溢出.
        提高精度: 减去最大值后, 输入值的范围更接近 0, 减少了浮点数计算中的精度损失.
    */
    // 遍历每一个 softmax group
    for (int g = 0; g < total_groups; ++g) {
        // 找最大值用于数值稳定
        float max_val = -INFINITY;
        for (int i = 0; i < group_size; ++i) {
            int idx = 0;
            int temp = g;
            int stride = group_size;

            for (int d = 3; d >= 0; --d) {
                if (d == axis) {
                    idx += i * (temp % dims[d]) * stride / dims[d];
                    stride *= dims[d];
                    temp /= dims[d];
                } else {
                    idx += (temp % dims[d]) * stride;
                    stride *= dims[d];
                    temp /= dims[d];
                }
            }
            float val = input[idx];
            if (val > max_val)
                max_val = val;
        }

        // 计算指数和
        float sum_exp = 0.0F;
        std::vector<float> exp_vals(group_size);

        for (int i = 0; i < group_size; ++i) {
            int idx = 0;
            int temp = g;
            int stride = group_size;
            for (int d = 3; d >= 0; --d) {
                if (d == axis) {
                    idx += i * (temp % dims[d]) * stride / dims[d];
                    stride *= dims[d];
                    temp /= dims[d];
                } else {
                    idx += (temp % dims[d]) * stride;
                    stride *= dims[d];
                    temp /= dims[d];
                }
            }
            float val = std::exp(input[idx] - max_val);
            exp_vals[i] = val;
            sum_exp += val;
        }

        // 归一化并写入 output
        for (int i = 0; i < group_size; ++i) {
            int idx = 0;
            int temp = g;
            int stride = group_size;
            for (int d = 3; d >= 0; --d) {
                if (d == axis) {
                    idx += i * (temp % dims[d]) * stride / dims[d];
                    stride *= dims[d];
                    temp /= dims[d];
                } else {
                    idx += (temp % dims[d]) * stride;
                    stride *= dims[d];
                    temp /= dims[d];
                }
            }
            output[idx] = exp_vals[i] / sum_exp;
        }
    }
}
class SoftmaxTest : public TestCase {
public:
    std::shared_ptr<Tensor<float>> input;
    std::vector<float> expectedOutput;
    int axis_ = 0;
    const std::unordered_map<std::string, std::string> axis = {{"axis", "0"}};

    SoftmaxTest():TestCase("Softmax") {
        initTestdata();
    }
private:
    void initTestdata()
    {
        std::vector<int> t = {
            1, 3, 4, 4
        };
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
        softmax_nd(input_ptr, expectedOutput.data(), t[0], t[1], t[2], t[3], axis_);
    }
};
}

int main() {
    Logger::getInstance().setLevel(LOG_INFO);
    Logger::getInstance().enableFileOutput("log", false);

    SoftmaxTest softtest;
    if (!softtest.run_test({softtest.input}, softtest.expectedOutput, 
        [&softtest](std::unique_ptr<vkop::ops::Operator> &op) {
            auto *softmax_op = dynamic_cast<Softmax *>(op.get());
            if (!softmax_op) {
                LOG_ERROR("Failed to cast operator to Softmax");
                return;
            }
            softmax_op->setAttribute(softtest.axis);
        })) {
        return -1;
    }

    return 0;
}