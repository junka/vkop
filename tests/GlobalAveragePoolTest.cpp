#include <vector>
#include <random>
#include <cmath>

#include "setup.hpp"
#include "core/Tensor.hpp"
#include "include/logger.hpp"

using vkop::core::Tensor;
using vkop::tests::TestCase;
#define USE_CPP_REF
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
class GlobalAveragePoolTest : public TestCase {
public:
    std::shared_ptr<Tensor<T>> input;
    std::shared_ptr<Tensor<T>> output;
    std::vector<int> shape_;

    explicit GlobalAveragePoolTest(const std::vector<int> &t):TestCase("GlobalAveragePool"), shape_(t) {
        initTestdata();
    }
private:
    void initTestdata()
    {
        input = std::make_shared<Tensor<T>>(shape_);
        output = std::make_shared<Tensor<T>>(std::vector<int>{shape_[0], shape_[1]});
        input->reserveOnCPU();
        output->reserveOnCPU();

        std::random_device rd{};
        std::mt19937 gen{rd()};
        gen.seed(1024);
        std::normal_distribution<> input_dist{-1.0F, 6.0F};
        for (int i = 0; i < input->num_elements(); i++) {
            if constexpr (std::is_same_v<T, uint16_t>) {
                (*input)[i] = vkop::core::ITensor::fp32_to_fp16(input_dist(gen));
            } else if constexpr (std::is_same_v<T, float>) {
                (*input)[i] = input_dist(gen);
            }
        }
        global_average_pool(input, output);
    }
};
}

TEST(GlobalAveragePoolTest, GlobalAveragePoolComprehensiveTest) {

    const std::vector<std::tuple<std::vector<int>>> testcases = {
        {{1, 3, 360, 512}},
        {{1, 3, 4, 8}},
    };

    for (const auto &t : testcases) {
        auto [shape] = t;
        LOG_INFO("Testing globalAveragePool: [%d, %d, %d, %d]", shape[0], shape[1], shape[2], shape[3]);
        LOG_INFO("Testing fp32");

        GlobalAveragePoolTest<float> gaptest(shape);
        EXPECT_TRUE(gaptest.run_test<float>({gaptest.input}, {gaptest.output}));

        LOG_INFO("Testing fp16");

        GlobalAveragePoolTest<uint16_t> gaptest2(shape);
        EXPECT_TRUE(gaptest2.run_test<uint16_t>({gaptest2.input}, {gaptest2.output}));
    }
}