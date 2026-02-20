#include <cstdint>
#include <cstdio>
#include <vector>

#include "setup.hpp"
#include "core/Tensor.hpp"
#include "include/logger.hpp"
#include "ops/Concat.hpp"

using vkop::core::Tensor;
using vkop::tests::TestCase;
using vkop::ops::Concat;

namespace {
#ifdef USE_CPP_REF
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
#endif
template<typename T>
class ConcatTest : public TestCase<T> {
public:
    std::shared_ptr<Tensor<float>> input1;
    std::shared_ptr<Tensor<float>> input2;
    std::shared_ptr<Tensor<float>> input3;
    std::shared_ptr<Tensor<float>> output;
    std::vector<int> t1;
    std::vector<int> t2;
    std::vector<int> t3;
    int axis_ = 0;

    std::unordered_map<std::string, std::string> attributes;
    ConcatTest(const std::vector<int> &t1, const std::vector<int> &t2, const std::vector<int> &t3, int axis):TestCase<T>("Concat"), t1(t1), t2(t2), t3(t3), axis_(axis) {
        attributes = {
            {"axis", std::to_string(axis_)}
        };
        initTestdata();
    }
private:
    void initTestdata()
    {
        input1 = std::make_shared<Tensor<float>>(t1);
        input2 = std::make_shared<Tensor<float>>(t2);
        input3 = std::make_shared<Tensor<float>>(t3);

        std::vector<int64_t> t1shape(t1.begin(), t1.end());
        std::vector<int64_t> t2shape(t2.begin(), t2.end());
        std::vector<int64_t> t3shape(t3.begin(), t3.end());

        std::vector<int> to = t1;  // Start with first input shape
        to[axis_] = t1[axis_] + t2[axis_] + t3[axis_];

        auto torch_input1 = torch::randn(t1shape, this->getTorchConf());
        auto torch_input2 = torch::randn(t2shape, this->getTorchConf());
        auto torch_input3 = torch::randn(t3shape, this->getTorchConf());
        output = std::make_shared<Tensor<float>>(to);
        this->fillTensorFromTorch(input1, torch_input1);
        this->fillTensorFromTorch(input2, torch_input2);
        this->fillTensorFromTorch(input3, torch_input3);

        auto torch_output = torch::cat({torch_input1, torch_input2, torch_input3}, axis_);
        this->fillTensorFromTorch(output, torch_output);

#if 0
        concat({input1, input2, input3}, {output}, axis_);
#endif
        input1->print_tensor();
        input2->print_tensor();
        input3->print_tensor();
        output->print_tensor();
    }
};
}

TEST(ConcatTest, ConcatComprehensiveTest) {

    std::vector<std::tuple<std::vector<int>, std::vector<int>, std::vector<int>, int>> test_cases = {
        {{3, 2, 4}, {1, 2, 4}, {4, 2, 4}, 0},
    };
    for (const auto &test_case : test_cases) {
        auto [t1, t2, t3, axis] = test_case;
        ConcatTest<float> cctest(t1, t2, t3, axis);
        EXPECT_TRUE (cctest.run_test({cctest.input1, cctest.input2, cctest.input3}, {cctest.output}, [&cctest] (std::unique_ptr<vkop::ops::Operator> &op) {
            auto *concat_op = dynamic_cast<Concat *>(op.get());
            if (!concat_op) {
                LOG_ERROR("Failed to cast operator to Concat");
                return;
            }
            concat_op->setAttribute(cctest.attributes);
        }));
    }

}