#include <vector>
#include <random>
#include <cmath>

#include "setup.hpp"
#include "core/Tensor.hpp"
#include "include/logger.hpp"

using vkop::core::Tensor;
using vkop::tests::TestCase;

namespace {

class FloorTest : public TestCase {
public:
    std::shared_ptr<Tensor<float>> input;
    std::shared_ptr<Tensor<float>> output;

    FloorTest():TestCase("Floor") {
        initTestdata();
    }
private:
    void initTestdata()
    {
        std::vector<int> t = {
            3, 6, 64, 64
        };
        input = std::make_shared<Tensor<float>>(t);
        input->reserveOnCPU();
        output = std::make_shared<Tensor<float>>(t);
        output->reserveOnCPU();

        std::random_device rd{};
        std::mt19937 gen{rd()};
        gen.seed(1024);
        std::normal_distribution<> input_dist{-4.0F, 6.0F};
        for (int i = 0; i < input->num_elements(); i++) {
            (*input)[i] = input_dist(gen);
            (*output)[i] = std::floor((*input)[i]);
        }
    }
};
}

TEST(FloorTest, FloorComprehensiveTest) {

    FloorTest floortest;
    EXPECT_TRUE(floortest.run_test<float>({floortest.input}, {floortest.output}));
}