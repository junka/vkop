#include <vector>

#include "setup.hpp"
#include "core/Tensor.hpp"
#include "include/logger.hpp"

using vkop::core::Tensor;
using vkop::tests::TestCase;

namespace {

template<typename T>
class RangeTest : public TestCase<T> {
public:
    int start_;
    int limit_;
    int delta_;
    std::shared_ptr<Tensor<T>> output;
    std::shared_ptr<Tensor<int>> input0;
    std::shared_ptr<Tensor<int>> input1;
    std::shared_ptr<Tensor<int>> input2;

    RangeTest(int start, int limit, int delta_): TestCase<T>("Range"), start_(start), limit_(limit), delta_(delta_) {
        initTestdata();
    }

private:
    void initTestdata()
    {
        input0 = std::make_shared<Tensor<int>>(1);
        input0->fillToCPU({start_});
        input1 = std::make_shared<Tensor<int>>(1);
        input1->fillToCPU({limit_});
        input2 = std::make_shared<Tensor<int>>(1);
        input2->fillToCPU({delta_});
        input2->print_tensor();

        auto torch_range = torch::arange(start_, limit_, delta_, this->getTorchConf());
        std::vector<int> output_shape(torch_range.dim());
        for (int i = 0; i < torch_range.dim(); ++i) {
            output_shape[i] = static_cast<int>(torch_range.size(i));
        }

        output = std::make_shared<Tensor<T>>(output_shape);
        std::cout << "range output: " << torch_range << std::endl;
        this->fillTensorFromTorch(output, torch_range);
    }
};
}

TEST(RangeTest, RangeComprehensiveTest) {
    std::vector<std::tuple<int, int, int>> test_cases = {
        {3, 9, 1},
        {3, -8, -1},
        {2, 12, 2},
        {-2, 20, 9},
    };
    for (const auto& test_case : test_cases) {
        auto [start, limit, delta] = test_case;
        LOG_INFO("start: %d, limit: %d, delta: %d", start, limit, delta);

        LOG_INFO("Testing Range fp32");
        RangeTest<float> range_test(start, limit, delta);
        const std::vector<std::shared_ptr<vkop::core::ITensor>> inputs = {
            range_test.input0,
            range_test.input1,
            range_test.input2
        };
        EXPECT_TRUE(range_test.run_test(inputs, {range_test.output}));

        LOG_INFO("Testing Range fp16");
        RangeTest<uint16_t> range_test1(start, limit, delta);
        const std::vector<std::shared_ptr<vkop::core::ITensor>> inputs1 = {
            range_test1.input0,
            range_test1.input1,
            range_test1.input2
        };
        EXPECT_TRUE(range_test1.run_test(inputs1, {range_test1.output}));
    }
}