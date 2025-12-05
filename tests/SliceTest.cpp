#include <vector>
#include <random>

#include "setup.hpp"
#include "core/Tensor.hpp"
#include "include/logger.hpp"
#include "ops/Slice.hpp"

using vkop::core::Tensor;
using vkop::tests::TestCase;
using vkop::ops::Slice;

namespace {

std::vector<float> slice_cpu(const std::shared_ptr<Tensor<float>>& input,
    const std::vector<int64_t>& starts, const std::vector<int64_t>& ends,
    const std::vector<int64_t>& steps, const std::vector<int64_t>& axes) {
    auto input_shape = input->getShape();
    size_t rank = input_shape.size();

    if (starts.size() != ends.size() || starts.size() != steps.size() || starts.size() != axes.size()) {
        throw std::invalid_argument("Mismatch in size of starts, ends, steps, and axes");
    }

    auto output_shape = Slice::CalculateOutputShape<int64_t>(
        input_shape,
        starts, ends, axes, steps
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

class SliceTest : public TestCase {
public:
    std::vector<int> input_shape_ = {
        1, 6, 4, 4
    };
    const std::vector<int64_t> starts = {0, 2};
    const std::vector<int64_t> ends = {2, 4};
    const std::vector<int64_t> steps = {1, 2};
    const std::vector<int64_t> axes = {1, 3};

    std::shared_ptr<Tensor<float>> input;
    std::shared_ptr<Tensor<int64_t>> starts_;
    std::shared_ptr<Tensor<int64_t>> ends_;
    std::shared_ptr<Tensor<int64_t>> axes_;
    std::shared_ptr<Tensor<int64_t>> steps_;
    std::vector<float> expectedOutput;

    SliceTest():TestCase("Slice") {
        initTestdata();
    }
private:
    void initTestdata()
    {
        input = std::make_shared<Tensor<float>>(input_shape_);
        input->reserveOnCPU();
        starts_ = std::make_shared<Tensor<int64_t>>(starts.size());
        starts_->fillToCPU(starts);
        ends_ = std::make_shared<Tensor<int64_t>>(ends.size());
        ends_->fillToCPU(ends);
        axes_ = std::make_shared<Tensor<int64_t>>(axes.size());
        axes_->fillToCPU(axes);
        steps_ = std::make_shared<Tensor<int64_t>>(steps.size());
        steps_->fillToCPU(steps);

        std::random_device rd{};
        std::mt19937 gen{rd()};
        gen.seed(1024);
        std::normal_distribution<> input_dist{-1.0F, 1.0F};
        for (int i = 0; i < input->num_elements(); i++) {
            auto a = input_dist(gen);
            (*input)[i] = a;
        }
        expectedOutput = slice_cpu(input, starts, ends, steps, axes);
    }
};
}


int main()
{
    Logger::getInstance().setLevel(LOG_INFO);
    Logger::getInstance().enableFileOutput("log", false);

    SliceTest slice_test;
    const std::vector<std::shared_ptr<vkop::core::ITensor>> inputs = {
        slice_test.input,
        slice_test.starts_,
        slice_test.ends_,
        slice_test.axes_,
        slice_test.steps_,
    };
    if (!slice_test.run_test(inputs, slice_test.expectedOutput)) {
        return -1;
    }
    return 0;
}