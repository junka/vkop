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
    std::vector<std::shared_ptr<vkop::core::ITensor>> inputs;

    std::shared_ptr<Tensor<T>> output;
    std::vector<std::vector<int>> shapes;
    int axis_ = 0;

    std::unordered_map<std::string, std::string> attributes;
    ConcatTest(const std::vector<std::vector<int>> &shapes, int axis):TestCase<T>("Concat"), shapes(shapes), axis_(axis) {
        attributes = {
            {"axis", std::to_string(axis_)}
        };
        initTestdata();
    }
    ~ConcatTest() {
        inputs.clear();
        output.reset();
    }
private:
    void initTestdata()
    {
        std::vector<int> output_shape;  // Start with first input shape
        std::vector<torch::Tensor> torch_inputs;

        output_shape = shapes[0];
        for (auto &shape : shapes) {
            auto input = std::make_shared<Tensor<T>>(shape);
            std::vector<int64_t> inshape(shape.begin(), shape.end());
            output_shape[axis_] += shape[axis_];
            auto torch_input = torch::randn(inshape, this->getTorchConf());
            torch_inputs.emplace_back(torch_input);
            this->fillTensorFromTorch(input, torch_input);
            inputs.emplace_back(input);
        }

        output = std::make_shared<Tensor<T>>(output_shape);

        auto torch_output = torch::cat(torch_inputs, axis_);
        this->fillTensorFromTorch(output, torch_output);

#if 0
        concat(inputs, {output}, axis_);
#endif
        output->print_tensor();
    }
};
}

TEST(ConcatTest, ConcatComprehensiveTest) {
    const std::vector<std::tuple<std::vector<std::vector<int>>, int>> test_cases = {
        {{{3, 2, 4}, {1, 2, 4}, {4, 2, 4}}, 0},
        {{{1, 16, 16, 16}, {1, 16, 16, 16}, {1, 16, 16, 16}}, 1},
        {{{1, 2, 16}, {1, 2, 16}}, 1},
        {{{1, 2, 16}, {1, 2, 16}}, 0},
        {{{1, 14, 64},{1, 14, 16},{1, 14, 40}}, 2},
        {{{1, 4, 84}, {1, 80, 84}}, 1},

    };
    for (const auto &test_case : test_cases) {
        auto [shapes, axis] = test_case;

        LOG_INFO("Testing FP32");
        ConcatTest<float> cctest(shapes, axis);
        EXPECT_TRUE (cctest.run_test(cctest.inputs, {cctest.output}, [&cctest] (std::unique_ptr<vkop::ops::Operator> &op) {
            auto *concat_op = dynamic_cast<Concat *>(op.get());
            if (!concat_op) {
                LOG_ERROR("Failed to cast operator to Concat");
                return;
            }
            concat_op->setAttribute(cctest.attributes);
        }));

        LOG_INFO("Testing FP16");
        ConcatTest<uint16_t> cctest1(shapes, axis);
        EXPECT_TRUE (cctest1.run_test(cctest1.inputs, {cctest1.output}, [&cctest1] (std::unique_ptr<vkop::ops::Operator> &op) {
            auto *concat_op = dynamic_cast<Concat *>(op.get());
            if (!concat_op) {
                LOG_ERROR("Failed to cast operator to Concat");
                return;
            }
            concat_op->setAttribute(cctest1.attributes);
        }));
    }

}