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
void global_average_pool(const std::shared_ptr<Tensor<float>>& input,
                          std::shared_ptr<Tensor<float>>& output) {
    auto shape = input->getShape();
    int n = shape[0];
    int c = shape[1];
    int num_elements = input->num_elements() / n / c;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < c; j++) {
            float sum = 0.0F;
            for (int k = 0; k < num_elements; k++) {
                sum += (*input)[(i * c * num_elements) + (j * num_elements) + k];
            }
            (*output)[(i * c) + j] = sum / num_elements;
        }
    }

}
#endif
class GlobalAveragePoolTest : public TestCase {
public:
    std::shared_ptr<Tensor<float>> input;
    std::shared_ptr<Tensor<float>> output;

    GlobalAveragePoolTest():TestCase("GlobalAveragePool") {
        initTestdata();
    }
private:
    void initTestdata()
    {
        std::vector<int> t = {
            1, 3, 4, 8
        };
        input = std::make_shared<Tensor<float>>(t);
        output = std::make_shared<Tensor<float>>(std::vector<int>{t[0], t[1]});
        input->reserveOnCPU();
        output->reserveOnCPU();

        std::random_device rd{};
        std::mt19937 gen{rd()};
        gen.seed(1024);
        std::normal_distribution<> input_dist{-4.0F, 6.0F};
        for (int i = 0; i < input->num_elements(); i++) {
            (*input)[i] = input_dist(gen);
        }
        global_average_pool(input, output);
        for (int i = 0; i < input->getShape()[0]; i++) {
            for (int j = 0; j < input->getShape()[1]; j++) {
                printf("%.4f " , (*output)[(i * input->getShape()[1]) + j]);
            }
            printf("\n");
        }
    }
};
}

int main() {
    Logger::getInstance().setLevel(LOG_INFO);
    Logger::getInstance().enableFileOutput("log", false);

    GlobalAveragePoolTest gaptest;
    if (!gaptest.run_test<float>({gaptest.input}, {gaptest.output})) {
        return -1;
    }

    return 0;
}