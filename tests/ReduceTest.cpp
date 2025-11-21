#include <cstdint>
#include <vector>
#include <random>
#include <cmath>
#include <stack>

#include "setup.hpp"
#include "core/Tensor.hpp"
#include "include/logger.hpp"
#include "ops/Reduce.hpp"

using vkop::core::Tensor;
using vkop::tests::TestCase;
using vkop::ops::Reduce;

namespace {
#ifdef USE_CPP_REF
void reference_reduce(const std::shared_ptr<Tensor<float>>& input,
    std::vector<float> &output,
    const std::vector<int>& axes_input,
    bool keepdims = false,
    ReduceType reduce_type = ReduceType::SUM
) {

    const int ndim = static_cast<int>(input->num_dims());
    assert(ndim >= 2 && ndim <= 4); // 2D~4D

    // 规范化 axes（支持负索引）
    std::vector<int> axes = axes_input;
    for (int& ax : axes) {
        if (ax < 0) ax += ndim;
        assert(ax >= 0 && ax < ndim);
    }
    std::sort(axes.begin(), axes.end());
    axes.erase(std::unique(axes.begin(), axes.end()), axes.end());

    std::vector<bool> is_reduced(ndim, false);
    for (int ax : axes) is_reduced[ax] = true;

    // 构建输出 shape
    std::vector<int64_t> out_shape;
    int64_t reduced_size = 1; // 被规约维度的总元素数（用于 MEAN）
    for (int i = 0; i < ndim; ++i) {
        if (is_reduced[i]) {
            reduced_size *= input->getShape()[i];
            if (keepdims) out_shape.push_back(1);
        } else {
            out_shape.push_back(input->getShape()[i]);
        }
    }

    size_t out_numel = 1;
    for (int64_t s : out_shape) out_numel *= s;

    auto in_strides = input.strides();
    auto out_strides = output.strides();

    // 预计算：每个输出位置对应的“未规约维度”坐标映射
    // 我们将输出 index 映射到输入的一个 base 坐标（被规约维度设为0）

    for (size_t out_idx_flat = 0; out_idx_flat < out_numel; ++out_idx_flat) {
        // 将 flat 输出索引转为多维坐标
        std::vector<int64_t> out_coord(out_shape.size());
        size_t tmp = out_idx_flat;
        for (int i = static_cast<int>(out_shape.size()) - 1; i >= 0; --i) {
            out_coord[i] = tmp % out_shape[i];
            tmp /= out_shape[i];
        }

        // 构建输入 base 坐标（未规约维度从 out_coord 取，规约维度先设为0）
        std::vector<int64_t> base_in_coord(ndim, 0);
        for (int i = 0, j = 0; i < ndim; ++i) {
            if (!is_reduced[i]) {
                base_in_coord[i] = (keepdims && input.shape[i] == 1) ? 0 : out_coord[j++];
            }
        }

        // 初始化 accumulator
        double acc = 0.0;
        bool first = true;
        float max_val_for_logsumexp = -std::numeric_limits<float>::infinity();

        // Step 1: 遍历所有被规约维度的组合
        std::function<void(int)> iterate_reduced_dims = [&](int dim) {
            if (dim == ndim) {
                // 计算输入线性偏移
                size_t in_offset = 0;
                for (int d = 0; d < ndim; ++d) {
                    in_offset += base_in_coord[d] * in_strides[d];
                }
                float val = input.data[in_offset];

                // 根据 reduce_type 累加
                switch (reduce_type) {
                    case ReduceType::L1:
                        acc += std::abs(val);
                        break;
                    case ReduceType::L2:
                    case ReduceType::SUMSQUARE:
                        acc += static_cast<double>(val) * val;
                        break;
                    case ReduceType::SUM:
                    case ReduceType::MEAN:
                        acc += val;
                        break;
                    case ReduceType::PROD:
                        acc = first ? val : acc * val;
                        break;
                    case ReduceType::MAX:
                        acc = first ? val : std::max(acc, static_cast<double>(val));
                        break;
                    case ReduceType::MIN:
                        acc = first ? val : std::min(acc, static_cast<double>(val));
                        break;
                    case ReduceType::LOGSUMEXP:
                        if (first || val > max_val_for_logsumexp) {
                            max_val_for_logsumexp = val;
                        }
                        // 先不累加，后面统一处理
                        break;
                    case ReduceType::LOGSUM:
                        // 先不处理，后面用 exp(val) 累加
                        break;
                }
                first = false;
                return;
            }

            if (is_reduced[dim]) {
                for (int64_t k = 0; k < input.shape[dim]; ++k) {
                    base_in_coord[dim] = k;
                    iterate_reduced_dims(dim + 1);
                }
            } else {
                iterate_reduced_dims(dim + 1);
            }
        };

        iterate_reduced_dims(0);

        // Step 2: 后处理（如 sqrt, mean, logsumexp 等）
        if (reduce_type == ReduceType::L2) {
            acc = std::sqrt(acc);
        } else if (reduce_type == ReduceType::MEAN) {
            acc /= static_cast<double>(reduced_size);
        } else if (reduce_type == ReduceType::LOGSUM) {
            // LOGSUM = log(sum(exp(x))) — 注意：不是 log(sum(x))
            // 但 ONNX ReduceLogSum 定义是：log(sum(|x|))？需确认！
            // 实际 ONNX: ReduceLogSum = log(sum(x))，但要求 x > 0
            // 这里按标准数学定义：log(sum(exp(x))) 是 LogSumExp
            // 而 ReduceLogSum = log(sum(x)) —— 但通常 x 应为正
            // 为安全，我们按 ONNX spec：https://onnx.ai/onnx/operators/onnx__ReduceLogSum.html
            // "log(sum(xi))" —— 所以先 sum(x)，再 log
            // 但若 sum(x) <= 0，log 无定义 → 返回 -inf
            if (acc <= 0.0) {
                acc = -std::numeric_limits<double>::infinity();
            } else {
                acc = std::log(acc);
            }
        } else if (reduce_type == ReduceType::LOGSUMEXP) {
            // 数值稳定版：log(sum(exp(x))) = m + log(sum(exp(x - m)))
            double sum_exp = 0.0;
                    // 重新遍历计算 exp(x - m)
                    std::function<void(int)> compute_sum_exp = [&](int dim) {
                        if (dim == ndim) {
                            size_t in_offset = 0;
                            for (int d = 0; d < ndim; ++d) {
                                in_offset += base_in_coord[d] * in_strides[d];
                            }
                            float val = input.data[in_offset];
                            sum_exp += std::exp(static_cast<double>(val) - max_val_for_logsumexp);
                            return;
                        }
                        if (is_reduced[dim]) {
                            for (int64_t k = 0; k < input.shape[dim]; ++k) {
                                base_in_coord[dim] = k;
                                compute_sum_exp(dim + 1);
                            }
                        } else {
                            compute_sum_exp(dim + 1);
                        }
                    };
                    compute_sum_exp(0);
                    acc = static_cast<double>(max_val_for_logsumexp) + std::log(sum_exp);
        }

        output.data[out_idx_flat] = static_cast<float>(acc);
    }

    return output;
}
#endif
class ReduceTest : public TestCase {
public:
    std::shared_ptr<Tensor<float>> input;
    std::vector<float> expectedOutput;

    ReduceTest():TestCase("Reduce") {
        initTestdata();
    }
private:
    void initTestdata()
    {
        std::vector<int> t = {
            64, 64
        };
        input = std::make_shared<Tensor<float>>(t);
        input->reserveOnCPU();

        expectedOutput.resize(input->num_elements());

        std::random_device rd{};
        std::mt19937 gen{rd()};
        gen.seed(1024);
        std::normal_distribution<> input_dist{-4.0F, 6.0F};
        for (int i = 0; i < input->num_elements(); i++) {
            (*input)[i] = input_dist(gen);
        }
        // reference_reduce(input, expectedOutput, 0, 0);
        for (int i = 0; i < input->num_elements(); i++) {
            printf("%.4f " , expectedOutput[i]);
        }
    }
};
}

int main() {
    Logger::getInstance().setLevel(LOG_INFO);
    Logger::getInstance().enableFileOutput("log", false);

    ReduceTest reducetest;
    if (!reducetest.run_test({reducetest.input}, reducetest.expectedOutput)) {
        return -1;
    }

    return 0;
}