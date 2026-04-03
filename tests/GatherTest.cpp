#include <vector>

#include "setup.hpp"
#include "core/Tensor.hpp"
#include "include/logger.hpp"
#include "ops/Gather.hpp"

using vkop::core::Tensor;
using vkop::tests::TestCase;
using vkop::ops::Gather;

namespace {

template<typename T>
class GatherTest : public TestCase<T> {
public:
    std::vector<int> input_shape_;
    std::vector<int> indices_shape_;
    std::vector<int> indices_data_;
    int axis;

    std::shared_ptr<Tensor<T>> input;
    std::shared_ptr<Tensor<int>> indices;
    std::shared_ptr<Tensor<T>> output;
    std::unordered_map<std::string, std::string> attributes;

    GatherTest(std::vector<int> &input_shape, int axis, std::vector<int>& indices_shape, std::vector<int>& indices_data): TestCase<T>("Gather"),input_shape_(input_shape), indices_shape_(indices_shape), indices_data_(indices_data), axis(axis) {
        attributes = {
            {"axis", std::to_string(axis)},
        };
        initTestdata();
    }
    torch::Tensor onnx_gather(const torch::Tensor& data, const torch::Tensor& indices, int axis) {
        if (axis < 0) {
            axis += data.dim();
        }

        std::vector<torch::indexing::TensorIndex> index_list;
        index_list.reserve(data.dim());

        for (int64_t i = 0; i < axis; ++i) {
            index_list.emplace_back(torch::indexing::Slice()); 
        }

        index_list.emplace_back(indices);

        for (int64_t i = axis + 1; i < data.dim(); ++i) {
            index_list.emplace_back(torch::indexing::Slice());
        }

        return data.index(index_list);
    }

private:
    void initTestdata()
    {
        input = std::make_shared<Tensor<T>>(input_shape_);

        indices = std::make_shared<Tensor<int>>(indices_shape_);
        indices->fillToCPU(indices_data_);
        indices->print_tensor();

        std::vector<int64_t> inshape(input_shape_.begin(), input_shape_.end());
        std::vector<int64_t> indshape(indices_shape_.begin(), indices_shape_.end());
        auto torch_input = torch::randn(inshape, this->getTorchConf());
        auto torch_indices = torch::tensor(indices_data_, torch::TensorOptions().dtype(torch::kInt64)).reshape(indshape);

        std::cout << "axis " << axis << std::endl;
        std::cout << "torch_input: " << torch_input << std::endl;
        std::cout << "torch_indices: " << torch_indices << std::endl;
        auto torch_output = onnx_gather(torch_input, torch_indices, axis);
        std::cout << "torch_output: " << torch_output << std::endl;
        this->fillTensorFromTorch(input, torch_input);

        std::vector<int> output_shape;
        for (auto s : torch_output.sizes()) {
            output_shape.push_back(static_cast<int>(s));
        }
        output = std::make_shared<Tensor<T>>(output_shape);
        this->fillTensorFromTorch(output, torch_output);
        // output->print_tensor();
    }
};
}

TEST(GatherTest, GatherComprehensiveTest) {
    std::vector<std::tuple<std::vector<int>, int, std::vector<int>, std::vector<int>>> test_cases = {
        {{3, 4}, 0, {3}, {1, 0, 2}},
        {{3, 4}, 1, {3, 2}, {1, 0, 2, 1, 0, 3}},  // Shape: [3, 4], axis: 1, indices shape: [3, 2], indices: [[1, 0], [2, 1], [0, 3]]
        {{2, 4}, 0, {1, 2, 1}, {0, 1}},         // Shape: [2, 4], axis: 0, indices shape: [1, 2, 1], indices: [[[0], [1]]]
        {{2, 3, 4}, 2, {2, 3}, {0, 1, 1, 0, 1, 2}}  // Shape: [2, 3, 4], axis: 2, indices shape: [2, 3, 2], indices: complex
    };
    for (const auto& test_case : test_cases) {
        auto [input_shape, axis, indices_shape, indices_data] = test_case;

        LOG_INFO("Testing Gather fp32");
        GatherTest<float> gather_test(input_shape, axis, indices_shape, indices_data);
        const std::vector<std::shared_ptr<vkop::core::ITensor>> inputs = {
            gather_test.input,
            gather_test.indices
        };
        EXPECT_TRUE(gather_test.run_test(inputs, {gather_test.output}, [&gather_test](std::unique_ptr<vkop::ops::Operator> &op){
            auto *gather_op = dynamic_cast<Gather *>(op.get());
            if (!gather_op) {
                LOG_ERROR("Failed to cast operator to Gather");
                return;
            }
            gather_op->setAttribute(gather_test.attributes);
        }));


        LOG_INFO("Testing Gather fp16");
        GatherTest<uint16_t> gather_test1(input_shape, axis, indices_shape, indices_data);
        const std::vector<std::shared_ptr<vkop::core::ITensor>> inputs1 = {
            gather_test1.input,
            gather_test1.indices
        };
        EXPECT_TRUE(gather_test1.run_test(inputs1, {gather_test1.output}, [&gather_test1](std::unique_ptr<vkop::ops::Operator> &op){
            auto *gather_op = dynamic_cast<Gather *>(op.get());
            if (!gather_op) {
                LOG_ERROR("Failed to cast operator to Gather");
                return;
            }
            gather_op->setAttribute(gather_test1.attributes);
        }));
    }
}