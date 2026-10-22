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

// A rank-2 slice on the buffer path. The buffer variant has always been
// rank-generic (it pads the unused int[6] lanes), so this is the regression
// guard for the shape math that the image variant used to skip entirely.
TEST(SliceTest, SliceRank2Buffer) {
    vkop::tests::ScopedBufferBackend buffer;
    std::vector<int> input_shape{6, 4};
    std::vector<int64_t> starts{0, 2};
    std::vector<int64_t> ends{6, 4};
    std::vector<int64_t> steps{2, 1};
    std::vector<int64_t> axes{0, 1};

    SliceTest<float> slice_test(input_shape, starts, ends, steps, axes);
    const std::vector<std::shared_ptr<vkop::core::ITensor>> inputs = {
        slice_test.input, slice_test.starts_, slice_test.ends_,
        slice_test.axes_,  slice_test.steps_};
    EXPECT_TRUE(slice_test.run_test(inputs, {slice_test.output}));
}

// Pure math behind the zero-copy fast path: contiguous_slice_range decides
// whether a slice is one contiguous byte run and, if so, its offset/length;
// offset_view_aligned is the descriptor-alignment + fp16 word-parity gate. A
// wrong answer here is silently-wrong data (a bad offset binds the wrong
// bytes), so pin the interesting shapes.
TEST(SliceTest, ContiguousRangeMath) {
    using vkop::ops::contiguous_slice_range;
    using vkop::ops::offset_view_aligned;
    VkDeviceSize bo = 0, bl = 0;
    // Outer-axis slice of {6,4,4}: [2:5] is 3*4*4 == 48 contiguous fp32 at
    // element 32 -> byte 128.
    EXPECT_TRUE(contiguous_slice_range({6, 4, 4}, {3, 4, 4}, {2, 0, 0},
                                       {1, 1, 1}, sizeof(float), bo, bl));
    EXPECT_EQ(bo, VkDeviceSize(128));
    EXPECT_EQ(bl, VkDeviceSize(48) * sizeof(float));
    // Any step != 1 is strided -> never a contiguous block.
    EXPECT_FALSE(contiguous_slice_range({6, 4, 4}, {3, 4, 4}, {0, 0, 0},
                                        {2, 1, 1}, sizeof(float), bo, bl));
    // A partial innermost slice leaves gaps between the selected rows.
    EXPECT_FALSE(contiguous_slice_range({6, 4, 4}, {6, 2, 4}, {0, 1, 0},
                                        {1, 1, 1}, sizeof(float), bo, bl));
    // Slicing two axes at once interleaves -> not contiguous.
    EXPECT_FALSE(contiguous_slice_range({6, 4, 4}, {2, 2, 4}, {1, 1, 0},
                                        {1, 1, 1}, sizeof(float), bo, bl));
    // Alignment: 4 bytes is the floor (fp16 packs 2 half/uint word, so an odd
    // fp16 element would straddle two words), the device limit raises it.
    EXPECT_TRUE(offset_view_aligned(128, 16));
    EXPECT_FALSE(offset_view_aligned(3, 16)); // not a multiple of the limit
    EXPECT_FALSE(offset_view_aligned(2, 1));  // below the 4-byte floor
    EXPECT_TRUE(offset_view_aligned(4, 1));
}

// Zero-copy contiguous fast path end to end. Slicing the OUTERMOST axis with
// step 1 selects a contiguous byte run of the input, so SliceBuffer binds the
// input's VkBuffer at that offset (VkDescriptorBufferInfo.offset, transparent
// to the shader) instead of dispatching slice.comp. The real assertion is
// correctness: the borrowed view has to read back exactly its window, which
// exercises the offset forwarding in VulkanBuffer::copyBufferToStageBuffer —
// read at the parent's start instead of +range_offset and the output would be
// the input's head, not the slice.
TEST(SliceTest, SliceOuterAxisContiguousBuffer) {
    vkop::tests::ScopedBufferBackend buffer;
    std::vector<int> input_shape{6, 4, 4};
    std::vector<int64_t> starts{2};
    std::vector<int64_t> ends{5};
    std::vector<int64_t> steps{1};
    std::vector<int64_t> axes{0};

    LOG_INFO("Testing Slice outer-axis fast path fp32");
    SliceTest<float> fp32(input_shape, starts, ends, steps, axes);
    EXPECT_TRUE(fp32.run_test(
        {fp32.input, fp32.starts_, fp32.ends_, fp32.axes_, fp32.steps_},
        {fp32.output}));

    LOG_INFO("Testing Slice outer-axis fast path fp16");
    SliceTest<uint16_t> fp16(input_shape, starts, ends, steps, axes);
    EXPECT_TRUE(fp16.run_test(
        {fp16.input, fp16.starts_, fp16.ends_, fp16.axes_, fp16.steps_},
        {fp16.output}));
}

// ONNX requires a non-zero step. 0 used to be silently rewritten to 1, which
// turned a malformed Slice into a plausible-looking full-axis copy.
TEST(SliceTest, SliceStepZeroThrows) {
    const std::vector<int> shape{4, 4};
    const std::vector<int64_t> starts{0};
    const std::vector<int64_t> ends{4};
    const std::vector<int64_t> axes{1};
    const std::vector<int64_t> steps{0};
    EXPECT_THROW(Slice::CalculateOutputShape<int64_t>(shape, starts, ends, axes,
                                                      steps),
                 std::invalid_argument);
}

// The buffer shader walks `rank` lanes of an int[6], so rank 7 would read past
// the array into the neighbouring push-constant fields and slice the wrong
// elements. It has to fail instead.
TEST(SliceTest, SliceRank7BufferThrows) {
    vkop::tests::ScopedBufferBackend buffer;
    std::vector<int> input_shape{1, 1, 1, 1, 1, 2, 4};
    std::vector<int64_t> starts{0};
    std::vector<int64_t> ends{2};
    std::vector<int64_t> steps{1};
    std::vector<int64_t> axes{5};

    SliceTest<float> slice_test(input_shape, starts, ends, steps, axes);
    const std::vector<std::shared_ptr<vkop::core::ITensor>> inputs = {
        slice_test.input, slice_test.starts_, slice_test.ends_,
        slice_test.axes_,  slice_test.steps_};
    EXPECT_THROW(slice_test.run_test(inputs, {slice_test.output}),
                 std::runtime_error);
}