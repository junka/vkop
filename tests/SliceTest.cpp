#include <vector>

#include "setup.hpp"
#include "core/Tensor.hpp"
#include "include/logger.hpp"
#include "ops/Slice.hpp"

using vkop::core::Tensor;
using vkop::tests::TestCase;
using vkop::ops::Slice;

namespace {
#if 0
std::vector<float> slice_cpu(const std::shared_ptr<Tensor<float>>& input,
    const std::vector<int64_t>& starts, const std::vector<int64_t>& ends,
    const std::vector<int64_t>& steps, const std::vector<int64_t>& axes) {
    auto input_shape = input->getShape();
    size_t rank = input_shape.size();

    if (starts.size() != ends.size() || starts.size() != steps.size() || starts.size() != axes.size()) {
        throw std::invalid_argument("Mismatch in size of starts, ends, steps, and axes");
    }

    auto output_shape = Slice::CalculateOutputShape<int64_t>(
        input_shape, starts, ends, axes, steps
    );
    for (size_t i = 0; i < rank; ++i)
        printf("%d ", output_shape[0][i]);
    printf("\n");

    std::vector<int64_t> input_strides(rank, 1);
    for (int i = rank - 2; i >= 0; --i) {
        input_strides[i] = input_strides[i + 1] * input_shape[i + 1];
    }

    size_t output_size = 1;
    for (auto dim : output_shape[0]) {
        output_size *= dim;
    }
    std::vector<float> output(output_size);

    std::vector<int64_t> output_indices(rank, 0);
    for (size_t i = 0; i < output_size; ++i) {
        int64_t input_index = 0;
        for (size_t j = 0; j < rank; ++j) {
            auto it = std::find(axes.begin(), axes.end(), j);
            if (it != axes.end()) {
                size_t axis_idx = std::distance(axes.begin(), it);
                int64_t start = starts[axis_idx];
                int64_t step = steps[axis_idx];
                input_index += (output_indices[j] * step + start) * input_strides[j];
            } else {
                input_index += output_indices[j] * input_strides[j];
            }
        }
        output[i] = (*input)[input_index];

        for (int j = rank - 1; j >= 0; --j) {
            if (++output_indices[j] < output_shape[0][j]) {
                break;
            }
            output_indices[j] = 0;
        }
    }
    // 层级打印输出 input
    printf("Input:\n");
    std::vector<int64_t> current_input_indices(rank, 0);
    for (int i = 0; i < input->num_elements(); ++i) {
        for (size_t j = 0; j < rank; ++j) {
            printf("%ld ", current_input_indices[j]);
        }
        printf(": %f\n", (*input)[i]);

        // Increment current input indices
        for (int j = rank - 1; j >= 0; --j) {
            if (++current_input_indices[j] < input_shape[j]) {
                break;
            }
            current_input_indices[j] = 0;
        }
    }
    // 层级打印输出 output
    printf("Output:\n");
    std::vector<int64_t> current_indices(rank, 0);
    for (size_t i = 0; i < output_size; ++i) {
        for (size_t j = 0; j < rank; ++j) {
            printf("%ld ", current_indices[j]);
        }
        printf(": %f\n", output[i]);

        // Increment current indices
        for (int j = rank - 1; j >= 0; --j) {
            if (++current_indices[j] < output_shape[0][j]) {
                break;
            }
            current_indices[j] = 0;
        }
    }

    return output;
}
#endif

template<typename T>
class SliceTest : public TestCase<T> {
public:
    std::vector<int> input_shape_;
    std::vector<int64_t> starts;
    std::vector<int64_t> ends;
    std::vector<int64_t> steps;
    std::vector<int64_t> axes;

    std::shared_ptr<Tensor<T>> input;
    std::shared_ptr<Tensor<int64_t>> starts_;
    std::shared_ptr<Tensor<int64_t>> ends_;
    std::shared_ptr<Tensor<int64_t>> axes_;
    std::shared_ptr<Tensor<int64_t>> steps_;
    std::shared_ptr<Tensor<T>> output;

    SliceTest(std::vector<int> &shape, std::vector<int64_t> &starts, std::vector<int64_t>& ends, std::vector<int64_t>& steps, std::vector<int64_t>& axes):TestCase<T>("Slice"), input_shape_(shape), starts(starts), ends(ends), steps(steps), axes(axes) {
        initTestdata();
    }
private:
    void initTestdata()
    {
        input = std::make_shared<Tensor<T>>(input_shape_);
        starts_ = std::make_shared<Tensor<int64_t>>(starts.size());
        starts_->fillToCPU(starts);
        ends_ = std::make_shared<Tensor<int64_t>>(ends.size());
        ends_->fillToCPU(ends);
        axes_ = std::make_shared<Tensor<int64_t>>(axes.size());
        axes_->fillToCPU(axes);
        steps_ = std::make_shared<Tensor<int64_t>>(steps.size());
        steps_->fillToCPU(steps);

        std::vector<int64_t> inshape(input_shape_.begin(), input_shape_.end());
        auto torch_input = torch::randn(inshape, this->getTorchConf());
        
        std::cout << "torch_input: " << torch_input << std::endl;
        std::vector<std::vector<int>> ret = Slice::CalculateOutputShape<int64_t>(
            input_shape_, starts, ends, axes, steps
        );
        auto oshape = ret[0];
        auto ostart = ret[1];
        auto oend = ret[2];
        auto ostep = ret[3];

        output = std::make_shared<Tensor<T>>(oshape);
#if 0
        auto result = slice_cpu(input, starts, ends, steps, axes);
#endif
        int64_t ndim = input->num_dims();
        std::vector<torch::indexing::TensorIndex> indices;
        indices.reserve(ndim);
        for (int64_t i = 0; i < ndim; ++i) {
            indices.emplace_back(torch::indexing::Slice(
                ostart[i],
                oend[i],
                ostep[i]
            ));
        }
        auto torch_output = torch_input.index(indices);
        std::cout << "torch_output: " << torch_output << std::endl;
        this->fillTensorFromTorch(input, torch_input);
        this->fillTensorFromTorch(output, torch_output);
        // output->print_tensor();
    }
};
}

TEST(SliceTest, SliceComprehensiveTest) {
    std::vector<std::tuple<std::vector<int>, std::vector<int64_t>, std::vector<int64_t>, std::vector<int64_t>, std::vector<int64_t>>> test_cases = {
        {{6, 4, 4}, {0, 2}, {2, 3}, {1, 2}, {0, 2}},
    };
    for (const auto& test_case : test_cases) {
        auto [input_shape, starts, ends, steps, axes] = test_case;
        for (size_t i = 0; i < starts.size(); ++i) {
            LOG_INFO("range %d - %d, step %d, axes %d", starts[i], ends[i], steps[i], axes[i]);
        }
        LOG_INFO("Testing Slice fp32");
        SliceTest<float> slice_test(input_shape, starts, ends, steps, axes);
        const std::vector<std::shared_ptr<vkop::core::ITensor>> inputs = {
            slice_test.input,
            slice_test.starts_,
            slice_test.ends_,
            slice_test.axes_,
            slice_test.steps_,
        };
        EXPECT_TRUE(slice_test.run_test(inputs, {slice_test.output}));


        LOG_INFO("Testing Slice fp16");
        SliceTest<uint16_t> slice_test1(input_shape, starts, ends, steps, axes);
        const std::vector<std::shared_ptr<vkop::core::ITensor>> inputs1 = {
            slice_test1.input,
            slice_test1.starts_,
            slice_test1.ends_,
            slice_test1.axes_,
            slice_test1.steps_,
        };
        EXPECT_TRUE(slice_test1.run_test(inputs1, {slice_test1.output}));
    }
}