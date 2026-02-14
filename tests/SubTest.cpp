#include <vector>
#include <random>

#include "setup.hpp"
#include "core/Tensor.hpp"
#include "include/logger.hpp"

using vkop::core::Tensor;
using vkop::tests::TestCase;

namespace {

class SubTest : public TestCase {
public:
    std::shared_ptr<Tensor<float>> inputa;
    std::shared_ptr<Tensor<float>> inputb;
    std::shared_ptr<Tensor<float>> output;

    SubTest():TestCase("Sub") {
        initTestdata();
    }
private:
    void initTestdata()
    {
        std::vector<int> t = {
            1, 7, 64, 64
        };
        inputa = std::make_shared<Tensor<float>>(t);
        inputb = std::make_shared<Tensor<float>>(t);
        output = std::make_shared<Tensor<float>>(t);
        inputa->reserveOnCPU();
        inputb->reserveOnCPU();
        output->reserveOnCPU();
        
        std::random_device rd{};
        std::mt19937 gen{rd()};
        gen.seed(1024);
        std::normal_distribution<> inputa_dist{0.0F, 1.0F};
        std::normal_distribution<> inputb_dist{1.0F, 2.0F};
        for (int i = 0; i < inputa->num_elements(); i++) {
            (*inputa)[i] = inputa_dist(gen);
            (*inputb)[i] = inputa_dist(gen);
            (*output)[i] = (*inputa)[i] - (*inputb)[i];
        }
    }
};
}

TEST(SubTest, SubComprehensiveTest) {

    SubTest subtest;
    EXPECT_TRUE(subtest.run_test<float>({subtest.inputa, subtest.inputb}, {subtest.output}));
}