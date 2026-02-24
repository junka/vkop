#include <vector>

#include "setup.hpp"
#include "core/Tensor.hpp"
#include "include/logger.hpp"

using vkop::core::Tensor;
using vkop::tests::TestCase;

namespace {
#ifdef USE_CPP_REF
template<typename T>
void global_average_pool(const std::shared_ptr<Tensor<T>>& input,
                          std::shared_ptr<Tensor<T>>& output) {
    auto shape = input->getShape();
    int n = shape[0];
    int c = shape[1];
    int num_elements = input->num_elements() / n / c;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < c; j++) {
            float sum = 0.0F;
            for (int k = 0; k < num_elements; k++) {
                if constexpr (std::is_same_v<T, float>) {
                    sum += (*input)[(i * c * num_elements) + (j * num_elements) + k];
                } else if constexpr (std::is_same_v<T, uint16_t>) {
                    sum += vkop::core::ITensor::fp16_to_fp32((*input)[(i * c * num_elements) + (j * num_elements) + k]);
                }
            }
            if constexpr (std::is_same_v<T, float>) {
                (*output)[(i * c) + j] = sum / num_elements;
            } else if constexpr (std::is_same_v<T, uint16_t>) {
                (*output)[(i * c) + j] = vkop::core::ITensor::fp32_to_fp16(sum / num_elements);
            }
        }
    }
}
#endif

template<typename T>
class GlobalAveragePoolTest : public TestCase<T> {
public:
    std::shared_ptr<Tensor<T>> input;
    std::shared_ptr<Tensor<T>> output;
    std::vector<int> shape_;

    explicit GlobalAveragePoolTest(const std::vector<int> &t):TestCase<T>("GlobalAveragePool"), shape_(t) {
        initTestdata();
    }
private:
    void initTestdata()
    {
        input = std::make_shared<Tensor<T>>(shape_);
        output = std::make_shared<Tensor<T>>(std::vector<int>{shape_[0], shape_[1]});

        std::vector<int64_t> inshape(shape_.begin(), shape_.end());
        auto torch_input = torch::rand(inshape, this->getTorchConf());
        std::vector<int64_t> dims;
        auto ndim = torch_input.dim();
        for (int64_t i = 2; i < ndim; ++i) {
            dims.push_back(i);
        }
        auto torch_output = torch::mean(torch_input, dims, true);
        this->fillTensorFromTorch(input, torch_input);
        this->fillTensorFromTorch(output, torch_output);
        std::cout << torch_input << std::endl;
        std::cout << torch_output << std::endl;
#if 0
        global_average_pool(input, output);
#endif
    }
};
}

TEST(GlobalAveragePoolTest, GlobalAveragePoolComprehensiveTest) {

    const std::vector<std::tuple<std::vector<int>>> testcases = {
        {{1, 3, 360, 512}},
        {{1, 3, 4, 8}},
        {{1, 4, 6, 1}},
        {{1, 4, 6}},
    };

    for (const auto &t : testcases) {
        auto [shape] = t;
        std::string str = "[";
        for (auto i : shape) {
            str += std::to_string(i) + ", ";
        }
        str += "]";
        LOG_INFO("Testing globalAveragePool: %s", str.c_str());
        LOG_INFO("Testing fp32");

        GlobalAveragePoolTest<float> gaptest(shape);
        EXPECT_TRUE(gaptest.run_test({gaptest.input}, {gaptest.output}));

        LOG_INFO("Testing fp16");

        GlobalAveragePoolTest<uint16_t> gaptest2(shape);
        EXPECT_TRUE(gaptest2.run_test({gaptest2.input}, {gaptest2.output}));
    }
}