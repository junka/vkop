#include <cstdint>
#include <vector>

#include "setup.hpp"
#include "core/Tensor.hpp"
#include "include/logger.hpp"
#include "ops/Reduce.hpp"

using vkop::core::Tensor;
using vkop::tests::TestCase;
using vkop::ops::Reduce;
#define USE_CPP_REF
namespace {
#ifdef USE_CPP_REF
void reference_reduce(const std::shared_ptr<Tensor<float>>& input,
    std::shared_ptr<Tensor<float>> &output,
    const std::vector<int>& axes_input,
    bool keepdims = true,
    vkop::ops::reduce::ReduceType reduce_type = vkop::ops::reduce::ReduceType::SUM
) {

    const int ndim = static_cast<int>(input->num_dims());
    assert(ndim >= 2 && ndim <= 4); // 2D~4D

    std::vector<int> axes = axes_input;
    for (int& ax : axes) {
        if (ax < 0) ax += ndim;
        assert(ax >= 0 && ax < ndim);
    }
    std::sort(axes.begin(), axes.end());
    axes.erase(std::unique(axes.begin(), axes.end()), axes.end());

    std::vector<bool> is_reduced(ndim, false);
    for (int ax : axes) is_reduced[ax] = true;

    std::vector<int64_t> out_shape;
    int64_t reduced_size = 1;
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

    // Calculate strides manually based on NCHW format
    std::vector<size_t> in_strides(ndim);
    std::vector<size_t> out_strides(out_shape.size());

    size_t temp_stride = 1;
    for (int i = ndim - 1; i >= 0; --i) {
        in_strides[i] = temp_stride;
        temp_stride *= input->getShape()[i];
    }
    
    temp_stride = 1;
    for (int i = out_shape.size() - 1; i >= 0; --i) {
        out_strides[i] = temp_stride;
        temp_stride *= out_shape[i];
    }

    // For each output position, calculate the corresponding input positions to reduce
    for (size_t out_idx_flat = 0; out_idx_flat < out_numel; ++out_idx_flat) {
        std::vector<int64_t> out_coord(out_shape.size());
        size_t tmp = out_idx_flat;
        for (int i = static_cast<int>(out_shape.size()) - 1; i >= 0; --i) {
            out_coord[i] = tmp % out_shape[i];
            tmp /= out_shape[i];
        }

        std::vector<int64_t> in_coord(ndim, 0);
        int out_dim_idx = 0;
        for (int i = 0; i < ndim; ++i) {
            if (is_reduced[i]) {
                if (keepdims) {
                    in_coord[i] = out_coord[out_dim_idx++]; // This will be 0 for reduced dims with keepdims
                } else {
                    in_coord[i] = 0; // Reduced dimensions start at 0
                }
            } else {
                in_coord[i] = out_coord[out_dim_idx++]; // Non-reduced dimensions take values from output
            }
        }

        // accumulator
        double acc = 0.0;
        bool first = true;
        float max_val_for_logsumexp = -std::numeric_limits<float>::infinity();

        // Step 1
        std::function<void(int, std::vector<int64_t>&)> iterate_reduced_dims = 
        [&](int dim, std::vector<int64_t>& current_coord) {
            if (dim == ndim) {
                size_t in_offset = 0;
                for (int d = 0; d < ndim; ++d) {
                    in_offset += current_coord[d] * in_strides[d];
                }
                float val = input->data()[in_offset];

                switch (reduce_type) {
                    case vkop::ops::reduce::ReduceType::L1:
                        acc += std::abs(val);
                        break;
                    case vkop::ops::reduce::ReduceType::L2:
                        acc += static_cast<double>(val) * val;
                        break;
                    case vkop::ops::reduce::ReduceType::LOGSUMEXP:
                        if (first || val > max_val_for_logsumexp) {
                            max_val_for_logsumexp = val;
                        }
                        break;
                    case vkop::ops::reduce::ReduceType::MAX:
                        acc = first ? val : std::max(acc, static_cast<double>(val));
                        break;
                    case vkop::ops::reduce::ReduceType::MEAN:
                        acc += val;
                        break;
                    case vkop::ops::reduce::ReduceType::MIN:
                        acc = first ? val : std::min(acc, static_cast<double>(val));
                        break;
                    case vkop::ops::reduce::ReduceType::PROD:
                        acc = first ? val : acc * val;
                        break;
                    case vkop::ops::reduce::ReduceType::SUM:
                    case vkop::ops::reduce::ReduceType::LOGSUM:
                        acc += val;
                        break;
                    case vkop::ops::reduce::ReduceType::SUMSQUARE:
                        acc += static_cast<double>(val) * val;
                        break;
                }
                first = false;
                return;
            }

            if (is_reduced[dim]) {
                for (int64_t k = 0; k < input->getShape()[dim]; ++k) {
                    current_coord[dim] = k;
                    iterate_reduced_dims(dim + 1, current_coord);
                }
            } else {
                iterate_reduced_dims(dim + 1, current_coord);
            }
        };

        // Start iteration with initial coordinate
        iterate_reduced_dims(0, in_coord);

        // Step 2: sqrt, mean, logsumexp 
        if (reduce_type == vkop::ops::reduce::ReduceType::L2) {
            acc = std::sqrt(acc);
        } else if (reduce_type == vkop::ops::reduce::ReduceType::MEAN) {
            acc /= static_cast<double>(reduced_size);
        } else if (reduce_type == vkop::ops::reduce::ReduceType::LOGSUM) {
            acc = std::log(std::max(acc, 1e-12));  // Take log of sum
        } else if (reduce_type == vkop::ops::reduce::ReduceType::LOGSUMEXP) {
            // log(sum(exp(x))) = m + log(sum(exp(x - m)))
            double sum_exp = 0.0;
            std::vector<int64_t> temp_coord = in_coord; // Create temporary coordinate vector for re-computation
            
            std::function<void(int)> compute_sum_exp = [&](int dim) {
                if (dim == ndim) {
                    size_t in_offset = 0;
                    for (int d = 0; d < ndim; ++d) {
                        in_offset += temp_coord[d] * in_strides[d];
                    }
                    float val = input->data()[in_offset];
                    sum_exp += std::exp(static_cast<double>(val) - max_val_for_logsumexp);
                    return;
                }
                if (is_reduced[dim]) {
                    for (int64_t k = 0; k < input->getShape()[dim]; ++k) {
                        temp_coord[dim] = k;
                        compute_sum_exp(dim + 1);
                    }
                } else {
                    compute_sum_exp(dim + 1);
                }
            };
            compute_sum_exp(0);
            acc = static_cast<double>(max_val_for_logsumexp) + std::log(sum_exp);
        }

        (*output)[out_idx_flat] = static_cast<float>(acc);
    }

}

#endif

template<typename T>
class ReduceTest : public TestCase<T> {
public:
    std::shared_ptr<Tensor<T>> input;
    std::shared_ptr<Tensor<T>> output;
    std::vector<int> t;
    std::vector<int> axes;

    std::unordered_map<std::string, std::string> attributes = {
        {"axes", "[0,1]"},
        {"reduce_op", "mean"},
        {"keepdims", "1"}
    };

    explicit ReduceTest(const std::vector<int> &shape, const std::vector<int> &axes):TestCase<T>("Reduce"), t(shape), axes(axes) {
    }

    void initTestdata(const std::string& reduce_op)
    {
        attributes["reduce_op"] = reduce_op;
        input = std::make_shared<Tensor<T>>(t);

        std::vector<int> output_shape = Reduce::calculateOutputShape(input->getShape(), axes, attributes.at("keepdims") == "1");
        output = std::make_shared<Tensor<T>>(output_shape);

        std::vector<int64_t> inshape(t.begin(), t.end());
        std::vector<int64_t> axeshape(axes.begin(), axes.end());
        torch::Tensor torch_input;
        if (attributes.at("reduce_op") == "log_sum" || attributes.at("reduce_op") == "log_sum_exp") {
            // For log operations, ensure positive values
            torch_input = torch::rand(inshape, this->getTorchConf()) + 0.01;  // Small positive values
        } else if (attributes.at("reduce_op") == "prod") {
            // For product operations, use smaller values to prevent overflow
            torch_input = torch::rand(inshape, this->getTorchConf()) * 0.5 + 0.5;  // Values in [0.5, 1.0]
        } else {
            // For other operations, use normal distribution
            torch_input = torch::randn(inshape, this->getTorchConf());
        }
        this->fillTensorFromTorch(input, torch_input);

        // vkop::ops::reduce::ReduceType reduce_type = vkop::ops::reduce::ReduceType::SUM;

        torch::Tensor torch_output;
        bool keepdims = (attributes.at("keepdims") == "1");
        if (attributes.at("reduce_op") == "l1_norm") {
            // reduce_type = vkop::ops::reduce::ReduceType::L1;
            torch_output = torch::sum(torch::abs(torch_input), axeshape, keepdims);
        } else if (attributes.at("reduce_op") == "l2_norm") {
            // reduce_type = vkop::ops::reduce::ReduceType::L2;
            torch_output = torch::sqrt(torch::sum(torch::square(torch_input), axeshape, keepdims));
        } else if (attributes.at("reduce_op") == "log_sum") {
            // reduce_type = vkop::ops::reduce::ReduceType::LOGSUM;
            constexpr double kEps = 1e-12;
            torch_output = torch::log(torch::sum(torch_input, axeshape, keepdims) + kEps);
        } else if (attributes.at("reduce_op") == "log_sum_exp") {
            // reduce_type = vkop::ops::reduce::ReduceType::LOGSUMEXP;
            torch_output = torch::logsumexp(torch_input, axeshape, keepdims);
        } else if (attributes.at("reduce_op") == "max") {
            // reduce_type = vkop::ops::reduce::ReduceType::MAX;
            torch_output = torch::amax(torch_input, axeshape, keepdims);
        } else if (attributes.at("reduce_op") == "mean") {
            // reduce_type = vkop::ops::reduce::ReduceType::MEAN;
            torch_output = torch::mean(torch_input, axeshape, keepdims);
        } else if (attributes.at("reduce_op") == "min") {
            // reduce_type = vkop::ops::reduce::ReduceType::MIN;
            torch_output = torch::amin(torch_input, axeshape, keepdims);
        } else if (attributes.at("reduce_op") == "prod") {
            // reduce_type = vkop::ops::reduce::ReduceType::PROD;
            torch_output = torch_input.clone();
            // Sort axes in descending order to avoid index shifting issues
            std::vector<int64_t> sorted_axes = axeshape;
            std::sort(sorted_axes.begin(), sorted_axes.end(), std::greater<int64_t>());
            for (int64_t axis : sorted_axes) {
                torch_output = torch::prod(torch_output, axis, keepdims);
            }
        } else if (attributes.at("reduce_op") == "sum") {
            // reduce_type = vkop::ops::reduce::ReduceType::SUM;
            torch_output = torch::sum(torch_input, axeshape, keepdims);
        } else if (attributes.at("reduce_op") == "sum_square") {
            // reduce_type = vkop::ops::reduce::ReduceType::SUMSQUARE;
            torch_output = torch::sum(torch::square(torch_input), axeshape, keepdims);
        }
        this->fillTensorFromTorch(output, torch_output);
#if 0
        reference_reduce(input, output, axes, attributes.at("keepdims") == "1", reduce_type);
#endif
        input->print_tensor();
        output->print_tensor();
    }
};
}

TEST(ReduceTest, ReduceComprehensiveTest) {
    const std::vector<std::tuple<std::vector<int>, std::vector<int>>> test_cases = {
        {{1,3, 4, 4}, {0, 1}},
    };
    for (const auto &test_case : test_cases) {
        auto [t, axes] = test_case;
        ReduceTest<float> reducetest(t, axes);
        for (const auto *reduceop: {"l1_norm", "l2_norm", "log_sum", "log_sum_exp", "max", "mean", "min", "prod", "sum", "sum_square"}) {
            reducetest.initTestdata(reduceop);
            LOG_INFO("Testing fp32 reduce op: %s", reduceop);
            EXPECT_TRUE(reducetest.run_test({reducetest.input}, {reducetest.output},
                [&reducetest](std::unique_ptr<vkop::ops::Operator> &op) {
                auto *reduce_op = dynamic_cast<Reduce *>(op.get());
                if (!reduce_op) {
                    LOG_ERROR("Failed to cast operator to Reduce");
                    return;
                }
                reduce_op->setAttribute(reducetest.attributes);
            }));
        }

        ReduceTest<uint16_t> reducetest1(t, axes);
        for (const auto *reduceop: {"l1_norm", "l2_norm", "log_sum", "log_sum_exp", "max", "mean", "min", "prod", "sum", "sum_square"}) {
            reducetest1.initTestdata(reduceop);
            LOG_INFO("Testing fp16 reduce op: %s", reduceop);
            EXPECT_TRUE(reducetest1.run_test({reducetest1.input}, {reducetest1.output},
                [&reducetest1](std::unique_ptr<vkop::ops::Operator> &op) {
                auto *reduce_op = dynamic_cast<Reduce *>(op.get());
                if (!reduce_op) {
                    LOG_ERROR("Failed to cast operator to Reduce");
                    return;
                }
                reduce_op->setAttribute(reducetest1.attributes);
            }));
        }

    }
}