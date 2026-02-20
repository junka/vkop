#include <cstdint>
#include <vector>

#include "setup.hpp"
#include "core/Tensor.hpp"
#include "include/logger.hpp"
#include "ops/Split.hpp"

using vkop::core::Tensor;
using vkop::tests::TestCase;
using vkop::ops::Split;

namespace {
#ifdef CPP_REF
void split_cpu(const std::shared_ptr<Tensor<float>>& input,
               const std::vector<std::shared_ptr<Tensor<float>>>& outputs,
               const std::vector<int64_t>& split_shape,
               int axis) {
    auto input_shape = input->getShape();
    size_t rank = input_shape.size();

    assert(axis >= 0 && static_cast<size_t>(axis) < rank);
    assert(outputs.size() == split_shape.size());

    int64_t total_split = 0;
    for (int64_t s : split_shape) {
        total_split += s;
    }
    assert(total_split == input_shape[axis]);

    size_t num_outputs = outputs.size();

    int64_t outer_size = 1;
    for (int i = 0; i < axis; ++i) {
        outer_size *= input_shape[i];
    }

    int64_t inner_size = 1;
    for (size_t i = axis + 1; i < rank; ++i) {
        inner_size *= input_shape[i];
    }

    int64_t input_axis_size = input_shape[axis];
    int64_t offset = 0;

    for (size_t i = 0; i < num_outputs; ++i) {
        int64_t slice_size = split_shape[i];
        const auto& output = outputs[i];

        for (int64_t outer = 0; outer < outer_size; ++outer) {
             for (int64_t j = 0; j < slice_size; ++j) {
                for (int64_t inner = 0; inner < inner_size; ++inner) {
                    int64_t out_index = (outer * (slice_size * inner_size)) + (j * inner_size) + inner;
                    int64_t in_index  = (outer * (input_axis_size * inner_size)) + ((offset + j) * inner_size) + inner;
                    (*output)[out_index] = (*input)[in_index];
                }
            }
        }

        offset += slice_size;
    }
}
#endif

template <typename T>
class SplitTest : public TestCase<T> {
public:
    std::vector<int> input_shape_;
    int axis_ = 2;
    std::unordered_map<std::string, std::string> attributes;

    std::vector<int64_t> split_shape;

    std::shared_ptr<Tensor<T>> input;
    std::shared_ptr<Tensor<int64_t>> split_;
    std::vector<std::shared_ptr<Tensor<T>>> outputs;

    SplitTest(std::vector<int> &shape, int axis, const std::vector<int64_t> &splits):TestCase<T>("Split"), input_shape_(shape), axis_(axis), split_shape(splits) {
        attributes["axis"] = std::to_string(axis_);

        initTestdata();
    }
private:
    void initTestdata()
    {
        torch::manual_seed(42);
        std::vector<int64_t> inshape(input_shape_.begin(), input_shape_.end());
        auto torch_input = torch::randn(inshape, this->getTorchConf());
        input = std::make_shared<Tensor<T>>(input_shape_);
        this->fillTensorFromTorch(input, torch_input);

        split_ = std::make_shared<Tensor<int64_t>>(split_shape.size());
        split_->fillToCPU(split_shape);
        auto torch_outputs = torch::split_with_sizes(torch_input, split_shape, axis_);
        std::cout << torch_input << std::endl;
        int num_outputs = split_shape.size();
        std::vector<int> shapes;
        for (int i = 0; i < num_outputs; i++) {
            auto shape = input_shape_;
            shape[axis_] = split_shape[i];
            auto output = std::make_shared<Tensor<T>>(shape);
            std::cout << torch_outputs[i] << std::endl;
            this->fillTensorFromTorch(output, torch_outputs[i]);
            outputs.push_back(output);
        }

        input->print_tensor();
        printf("=============================\n");
        for (auto &output : outputs) {
            output->print_tensor();
        }
    }
};
}

TEST(SplitTest, SplitComprehensiveTest) {
    const std::vector<std::tuple<std::vector<int>, int, std::vector<int64_t>>> test_cases = {
        {{8, 4, 8}, 2, {2, 6}},
        {{8, 4, 8}, 1, {1, 3}},
        {{1, 2, 8, 8}, 2, {5, 3}},
    };

    for (const auto &test_case : test_cases) {
        auto [input_shape, axis, split_shape] = test_case;
        LOG_INFO("Testing input shape: [%d, %d, %d], axis: %d, split shape: [%ld, %ld]", input_shape[0], input_shape[1], input_shape[2], axis, split_shape[0], split_shape[1]);
        LOG_INFO("Testing FP32");
        SplitTest<float> split_test(input_shape, axis, split_shape);
        const std::vector<std::shared_ptr<vkop::core::ITensor>> inputs = {
            split_test.input,
            split_test.split_,
        };
        std::vector<std::shared_ptr<vkop::core::ITensor>> outputs;
        outputs.reserve(split_test.outputs.size());
        for (auto &output : split_test.outputs) {
            outputs.push_back(output);
        };
        EXPECT_TRUE(split_test.run_test(inputs, outputs, [&split_test] (std::unique_ptr<vkop::ops::Operator> &op) {
            auto *split_op = dynamic_cast<Split *>(op.get());
            if (!split_op) {
                LOG_ERROR("Failed to cast operator to Split");
                return;
            }
            split_op->setAttribute(split_test.attributes);
        }));

        LOG_INFO("Testing FP16");        
        SplitTest<uint16_t> split_test1(input_shape, axis, split_shape);
        const std::vector<std::shared_ptr<vkop::core::ITensor>> inputs1 = {
            split_test1.input,
            split_test1.split_,
        };
        std::vector<std::shared_ptr<vkop::core::ITensor>> outputs1;
        outputs1.reserve(split_test1.outputs.size());
        for (auto &output : split_test1.outputs) {
            outputs1.push_back(output);
        };
        EXPECT_TRUE(split_test1.run_test(inputs1, outputs1, [&split_test1] (std::unique_ptr<vkop::ops::Operator> &op) {
            auto *split_op = dynamic_cast<Split *>(op.get());
            if (!split_op) {
                LOG_ERROR("Failed to cast operator to Split");
                return;
            }
            split_op->setAttribute(split_test1.attributes);
        }));
    }
}