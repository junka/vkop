#include <vector>
#include <random>

#include "setup.hpp"
#include "core/Tensor.hpp"
#include "include/logger.hpp"

using vkop::core::Tensor;
using vkop::tests::TestCase;

namespace {

class SigmoidTest : public TestCase {
public:
    std::shared_ptr<Tensor<float>> input;
    std::shared_ptr<Tensor<float>> output;

    SigmoidTest():TestCase("Sigmoid") {
        initTestdata();
    }
private:
    void initTestdata()
    {
        std::vector<int> t = {
            1, 6, 64, 64
        };

        input = std::make_shared<Tensor<float>>(t);
        output = std::make_shared<Tensor<float>>(t);
        input->reserveOnCPU();
        output->reserveOnCPU();
        
        std::random_device rd{};
        std::mt19937 gen{rd()};
        gen.seed(1024);
        std::normal_distribution<> input_dist{0.0F, 1.0F};
        for (int i = 0; i < input->num_elements(); i++) {
            (*input)[i] = input_dist(gen);
            (*output)[i] = ((*input)[i] > 0) ? 1.0F / (1.0F + exp(-(*input)[i])) : exp((*input)[i]) / (1.0F + exp((*input)[i]));
        }
    }
};
}

TEST(SigmoidTest, SigmoidComprehensiveTest) {
    SigmoidTest sigest;
    EXPECT_TRUE(sigest.run_test<float>({sigest.input}, {sigest.output}));
}