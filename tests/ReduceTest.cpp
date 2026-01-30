#include <cstdint>
#include <vector>
#include <random>
#include <cmath>

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
class ReduceTest : public TestCase {
public:
    std::shared_ptr<Tensor<float>> input;
    std::shared_ptr<Tensor<float>> output;
    std::vector<int> axes = {0,1};

    std::unordered_map<std::string, std::string> attributes = {
        {"axes", "[0,1]"},
        {"reduce_op", "mean"},
        {"keepdims", "1"}
    };

    explicit ReduceTest():TestCase("Reduce") {
    }

    void initTestdata(const std::string& reduce_op)
    {
        attributes["reduce_op"] = reduce_op;
        std::vector<int> t = {
            1, 3, 4,4
        };
        input = std::make_shared<Tensor<float>>(t);

        std::vector<int> output_shape = Reduce::calculateOutputShape(input->getShape(), axes, attributes.at("keepdims") == "1");
        output = std::make_shared<Tensor<float>>(output_shape);
        input->reserveOnCPU();
        output->reserveOnCPU();

        std::random_device rd{};
        std::mt19937 gen{rd()};
        gen.seed(1024);
        std::normal_distribution<> input_dist{-4.0F, 6.0F};
        for (int i = 0; i < input->num_elements(); i++) {
            (*input)[i] = input_dist(gen);
        }
        vkop::ops::reduce::ReduceType reduce_type = vkop::ops::reduce::ReduceType::SUM;
        if (attributes.at("reduce_op") == "l1_norm") {
            reduce_type = vkop::ops::reduce::ReduceType::L1;
        } else if (attributes.at("reduce_op") == "l2_norm") {
            reduce_type = vkop::ops::reduce::ReduceType::L2;
        } else if (attributes.at("reduce_op") == "log_sum") {
            reduce_type = vkop::ops::reduce::ReduceType::LOGSUM;
        } else if (attributes.at("reduce_op") == "log_sum_exp") {
            reduce_type = vkop::ops::reduce::ReduceType::LOGSUMEXP;
        } else if (attributes.at("reduce_op") == "max") {
            reduce_type = vkop::ops::reduce::ReduceType::MAX;
        } else if (attributes.at("reduce_op") == "mean") {
            reduce_type = vkop::ops::reduce::ReduceType::MEAN;
        } else if (attributes.at("reduce_op") == "min") {
            reduce_type = vkop::ops::reduce::ReduceType::MIN;
        } else if (attributes.at("reduce_op") == "prod") {
            reduce_type = vkop::ops::reduce::ReduceType::PROD;
        } else if (attributes.at("reduce_op") == "sum") {
            reduce_type = vkop::ops::reduce::ReduceType::SUM;
        } else if (attributes.at("reduce_op") == "sum_square") {
            reduce_type = vkop::ops::reduce::ReduceType::SUMSQUARE;
        }

        reference_reduce(input, output, axes, attributes.at("keepdims") == "1", reduce_type);
        printTensorWithShape(*input, input->getShape());
        printTensorWithShape(*output, output->getShape());
    }
private:
    static void printTensorWithShape(const Tensor<float>& tensor, const std::vector<int>& shape) {
        if (shape.empty()) return;
        
        // For small tensors, print with full hierarchical structure
        size_t total_elements = 1;
        for (int dim : shape) {
            total_elements *= dim;
        }
        
        // Limit printing for large tensors
        if (total_elements > 1000) {
            printf("tensor shape: ");
            for (size_t i = 0; i < shape.size(); ++i) {
                printf("%d", shape[i]);
                if (i < shape.size() - 1) printf("x");
            }
            printf(", total elements: %zu\n", total_elements);
            printf("First 10 elements: ");
            for (size_t i = 0; i < std::min(total_elements, static_cast<size_t>(10)); ++i) {
                printf("%.4f ", tensor[i]);
            }
            printf("\n");
            return;
        }
        
        // For small tensors, print with hierarchical structure
        printf("tensor shape: [");
        for (int i : shape) {
            printf("%d, ", i);
        }
        printf("]\n");
        
        if (shape.size() == 1) {
            printf("[");
            for (int i = 0; i < shape[0]; ++i) {
                printf("%.4f", tensor[i]);
                if (i < shape[0] - 1) printf(", ");
            }
            printf("]\n");
        } else if (shape.size() == 2) {
            for (int i = 0; i < shape[0]; ++i) {
                printf("[");
                for (int j = 0; j < shape[1]; ++j) {
                    size_t idx = (i * shape[1]) + j;
                    printf("%.4f", tensor[idx]);
                    if (j < shape[1] - 1) printf(", ");
                }
                printf("]\n");
            }
        } else if (shape.size() == 3) {
            for (int i = 0; i < shape[0]; ++i) {
                printf("Channel %d:\n", i);
                for (int j = 0; j < shape[1]; ++j) {
                    printf("  [");
                    for (int k = 0; k < shape[2]; ++k) {
                        size_t idx = (i * shape[1] * shape[2]) + (j * shape[2]) + k;
                        printf("%.4f", tensor[idx]);
                        if (k < shape[2] - 1) printf(", ");
                    }
                    printf("]\n");
                }
            }
        } else if (shape.size() == 4) {
            for (int n = 0; n < shape[0]; ++n) {
                printf("[");
                for (int c = 0; c < shape[1]; ++c) {
                    printf(" [");
                    for (int h = 0; h < shape[2]; ++h) {
                        printf("    [");
                        for (int w = 0; w < shape[3]; ++w) {
                            size_t idx = (n * shape[1] * shape[2] * shape[3]) + 
                                         (c * shape[2] * shape[3]) + 
                                         (h * shape[3]) + w;
                            printf("%.4f", tensor[idx]);
                            if (w < shape[3] - 1) printf(", ");
                        }
                        printf("]\n");
                    }
                    printf("]\n");
                }
                printf("]\n");
            }
        } else {
            // For tensors with more than 4 dimensions, just print flat
            printf("[");
            for (size_t i = 0; i < total_elements; ++i) {
                printf("%.4f", tensor[i]);
                if (i < total_elements - 1) printf(", ");
            }
            printf("]\n");
        }
    }
};
}

int main() {
    Logger::getInstance().setLevel(LOG_INFO);
    Logger::getInstance().enableFileOutput("log", false);
    
    ReduceTest reducetest;
    for (const auto *reduceop: {"l1_norm", "l2_norm", "log_sum", "log_sum_exp", "max", "mean", "min", "prod", "sum", "sum_square"}) {
        reducetest.initTestdata(reduceop);
        if (!reducetest.run_test<float>({reducetest.input}, {reducetest.output},
            [&reducetest](std::unique_ptr<vkop::ops::Operator> &op) {
            auto *reduce_op = dynamic_cast<Reduce *>(op.get());
            if (!reduce_op) {
                LOG_ERROR("Failed to cast operator to Reduce");
                return;
            }
            reduce_op->setAttribute(reducetest.attributes);
        })) {
            return -1;
        }
    }

    return 0;
}