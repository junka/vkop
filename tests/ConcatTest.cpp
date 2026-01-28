#include <cstdint>
#include <cstdio>
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
            std::shared_ptr<Tensor<float>> &output, int axis) {
    size_t num_inputs = inputs.size();
    std::vector<int> output_shape = inputs[0]->getShape();
    for (size_t i = 1; i < num_inputs; ++i) {
        output_shape[axis] += inputs[i]->getShape()[axis];
    }
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
            (*output)[out_idx] = input_ptr[idx];

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
    std::shared_ptr<Tensor<float>> output;
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
        output = std::make_shared<Tensor<float>>(to);
        output->reserveOnCPU();

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
        concat({input1, input2, input3}, {output}, axis_);
        print_tensor<float>(input1);
        print_tensor<float>(input2);
        print_tensor<float>(input3);
        print_tensor<float>(output);
    }
};
}

int main() {
    Logger::getInstance().setLevel(LOG_INFO);
    Logger::getInstance().enableFileOutput("log", false);

    ConcatTest cctest;
    if (!cctest.run_test<float>({cctest.input1, cctest.input2, cctest.input3}, {cctest.output}, [&cctest] (std::unique_ptr<vkop::ops::Operator> &op) {
        auto *concat_op = dynamic_cast<Concat *>(op.get());
        if (!concat_op) {
            LOG_ERROR("Failed to cast operator to Concat");
            return;
        }
        concat_op->setAttribute(cctest.attributes);
    })) {
        return -1;
    }

    return 0;
}