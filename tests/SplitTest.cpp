#include <vector>
#include <random>

#include "setup.hpp"
#include "core/Tensor.hpp"
#include "include/logger.hpp"
#include "ops/Split.hpp"

using vkop::core::Tensor;
using vkop::tests::TestCase;
using vkop::ops::Split;

namespace {

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
class SplitTest : public TestCase {
public:
    std::vector<int> input_shape_ = {
        8, 4, 8
    };
    int axis_ = 2;
    std::unordered_map<std::string, std::string> attributes = {
        {"axis", std::to_string(axis_)}
    };

    const std::vector<int64_t> split_shape = {2, 6};

    std::shared_ptr<Tensor<float>> input;
    std::shared_ptr<Tensor<int64_t>> split_;
    std::vector<std::shared_ptr<Tensor<float>>> outputs;

    SplitTest():TestCase("Split") {
        initTestdata();
    }
private:
    void initTestdata()
    {
        input = std::make_shared<Tensor<float>>(input_shape_);
        input->reserveOnCPU();
        split_ = std::make_shared<Tensor<int64_t>>(split_shape.size());
        split_->fillToCPU(split_shape);

        int num_outputs = split_shape.size();
        std::vector<int> shapes;
        for (int i = 0; i < num_outputs; i++) {
            auto shape = input_shape_;
            shape[axis_] = split_shape[i];
            auto output = std::make_shared<Tensor<float>>(shape);
            output->reserveOnCPU();
            outputs.emplace_back(output);
        }

        std::random_device rd{};
        std::mt19937 gen{rd()};
        gen.seed(1024);
        std::normal_distribution<> input_dist{-1.0F, 1.0F};
        for (int i = 0; i < input->num_elements(); i++) {
            auto a = input_dist(gen);
            (*input)[i] = a;
        }
        split_cpu(input, outputs, split_shape, axis_);
        input->print_tensor();
        printf("=============================\n");
        for (auto &output : outputs) {
            output->print_tensor();
        }
    }
};
}

TEST(SplitTest, SplitComprehensiveTest) {
    SplitTest split_test;
    const std::vector<std::shared_ptr<vkop::core::ITensor>> inputs = {
        split_test.input,
        split_test.split_,
    };
    std::vector<std::shared_ptr<vkop::core::ITensor>> outputs;
    outputs.reserve(split_test.outputs.size());
    for (auto &output : split_test.outputs) {
        outputs.push_back(output);
    };
    EXPECT_TRUE(split_test.run_test<float>(inputs, outputs, [&split_test] (std::unique_ptr<vkop::ops::Operator> &op) {
        auto *split_op = dynamic_cast<Split *>(op.get());
        if (!split_op) {
            LOG_ERROR("Failed to cast operator to Conv2d");
            return;
        }
        split_op->setAttribute(split_test.attributes);
    }));
}