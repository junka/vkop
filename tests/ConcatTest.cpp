#include <cstdint>
#include <vector>
#include <random>

#include "setup.hpp"
#include "core/Tensor.hpp"
#include "include/logger.hpp"
#include "ops/Concat.hpp"

using vkop::core::Tensor;
using vkop::tests::TestCase;
using vkop::ops::Concat;

namespace {
void concat(const std::vector<std::shared_ptr<Tensor<float>>> &inputs,
            std::vector<float> &output, int axis) {
    size_t num_inputs = inputs.size();
    std::vector<int> output_shape = inputs[0]->getShape();
    for (size_t i = 1; i < num_inputs; ++i) {
        output_shape[axis] += inputs[i]->getShape()[axis];
    }

    size_t output_size = std::accumulate(output_shape.begin(), output_shape.end(), 1, std::multiplies<>());
    output.resize(output_size);

    int ndim = inputs[0]->num_dims();
    std::vector<size_t> out_strides(ndim);
    out_strides[ndim - 1] = 1;
    for (int i = ndim - 2; i >= 0; --i) {
        out_strides[i] = out_strides[i + 1] * output_shape[i + 1];
    }

    size_t current_offset = 0;
    for (size_t i = 0; i < num_inputs; ++i) {
        auto in_shape = inputs[i]->getShape();
        size_t input_size = inputs[i]->num_elements();
        auto input_ptr = inputs[i]->data();

        std::vector<size_t> in_strides(ndim);
        in_strides[ndim - 1] = 1;
        for (int i = ndim - 2; i >= 0; --i) {
            in_strides[i] = in_strides[i + 1] * in_shape[i + 1];
        }

        std::vector<int> indices(ndim, 0);
        for (size_t idx = 0; idx < input_size; ++idx) {
            size_t out_idx = 0;
            for (int d = 0; d < ndim; ++d) {
                int coord = indices[d];
                if (d == axis) {
                    coord += current_offset;
                }
                out_idx += coord * out_strides[d];
            }
            output[out_idx] = input_ptr[idx];

            int carry = 1;
            for (int d = ndim - 1; d >= 0 && carry; --d) {
                indices[d] += carry;
                if (indices[d] >= in_shape[d]) {
                    indices[d] = 0;
                } else {
                    carry = 0;
                }
            }
        }

        current_offset += in_shape[axis];
    }
}

class ConcatTest : public TestCase {
public:
    std::shared_ptr<Tensor<float>> input1;
    std::shared_ptr<Tensor<float>> input2;
    std::shared_ptr<Tensor<float>> input3;
    std::vector<float> expectedOutput;
    int axis_ = 0;

    std::unordered_map<std::string, std::string> attributes = {
        {"axis", std::to_string(axis_)}
    };
    ConcatTest():TestCase("Concat") {
        initTestdata();
    }
private:
    void initTestdata()
    {
        std::vector<int> t1 = {
            3, 2, 4
        };
        std::vector<int> t2 = {
            1, 2, 4
        };
        std::vector<int> t3 = {
            4, 2, 4
        };


        input1 = std::make_shared<Tensor<float>>(t1);
        input1->reserveOnCPU();
        input2 = std::make_shared<Tensor<float>>(t2);
        input2->reserveOnCPU();
        input3 = std::make_shared<Tensor<float>>(t3);
        input3->reserveOnCPU();

        std::vector<int> to = {
            8, 2, 4
        };

        std::random_device rd{};
        std::mt19937 gen{rd()};
        gen.seed(1024);
        std::normal_distribution<> input_dist{0.0F, 1.0F};
        for (int i = 0; i < input1->num_elements(); i++) {
            (*input1)[i] = input_dist(gen);
        }
        for (int i = 0; i < input2->num_elements(); i++) {
            (*input2)[i] = input_dist(gen);
        }
        for (int i = 0; i < input3->num_elements(); i++) {
            (*input3)[i] = input_dist(gen);
        }
        concat({input1, input2, input3}, expectedOutput, axis_);
#if 0
        printf("=====================\n");
        for (int n = 0; n < t1[0]; n++) {
            printf("[\n");
            for (int c = 0; c < t1[1]; c++) {
                printf("[\n");
                for (int h = 0; h < t1[2]; h++) {
                    printf("[");
                    for (int w = 0; w < t1[3]; w++) {
                        int idx = n * t1[1] * t1[2] * t1[3] + c * t1[2] * t1[3] +
                                  h * t1[3] + w;
                        printf("%f ", (*input1)[idx]);
                    }
                    printf("]\n");
                }
                printf("]\n");
            }
            printf("]\n");
        }
        printf("=====================\n");
        for (int n = 0; n < t2[0]; n++) {
            printf("[\n");
            for (int c = 0; c < t2[1]; c++) {
                printf("[\n");
                for (int h = 0; h < t2[2]; h++) {
                    printf("[");
                    for (int w = 0; w < t2[3]; w++) {
                        int idx = n * t2[1] * t2[2] * t2[3] + c * t2[2] * t2[3] +
                                  h * t2[3] + w;
                        printf("%f ", (*input2)[idx]);
                    }
                    printf("]\n");
                }
                printf("]\n");
            }
            printf("]\n");
        }
        printf("=====================\n");
        for (int n = 0; n < t3[0]; n++) {
            printf("[\n");
            for (int c = 0; c < t3[1]; c++) {
                printf("[\n");
                for (int h = 0; h < t3[2]; h++) {
                    printf("[");
                    for (int w = 0; w < t3[3]; w++) {
                        int idx = n * t3[1] * t3[2] * t3[3] + c * t3[2] * t3[3] +
                                  h * t3[3] + w;
                        printf("%f ", (*input3)[idx]);
                    }
                    printf("]\n");
                }
                printf("]\n");
            }
            printf("]\n");
        }
        printf("=====================\n");
        for (int n = 0; n < to[0]; n++) {
            printf("[\n");
            for (int c = 0; c < to[1]; c++) {
                printf("[\n");
                for (int h = 0; h < to[2]; h++) {
                    printf("[");
                    for (int w = 0; w < to[3]; w++) {
                        int idx = n * to[1] * to[2] * to[3] + c * to[2] * to[3] +
                                  h * to[3] + w;
                        printf("%f ", expectedOutput[idx]);
                    }
                    printf("]\n");
                }
                printf("]\n");
            }
            printf("]\n");
        }
#endif
    }
};
}

int main() {
    Logger::getInstance().setLevel(LOG_INFO);
    Logger::getInstance().enableFileOutput("log", false);

    ConcatTest cctest;
    if (!cctest.run_test({cctest.input1, cctest.input2, cctest.input3}, cctest.expectedOutput, [&cctest] (std::unique_ptr<vkop::ops::Operator> &op) {
        auto *conv_op = dynamic_cast<Concat *>(op.get());
        if (!conv_op) {
            LOG_ERROR("Failed to cast operator to Conv2d");
            return;
        }
        conv_op->setAttribute(cctest.attributes);
    })) {
        return -1;
    }

    return 0;
}