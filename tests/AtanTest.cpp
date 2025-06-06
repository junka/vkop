#include <cstdint>
#include <vector>
#include <random>
#include <cmath>

#include "setup.hpp"
#include "Tensor.hpp"
#include "logger.hpp"

using vkop::core::Tensor;
using vkop::tests::TestCase;

namespace {

class AtanTest : public TestCase {
public:
    std::shared_ptr<Tensor<float>> input;
    std::vector<float> expectedOutput;

    AtanTest():TestCase("Atan") {
        initTestdata();
    }
private:
    void initTestdata()
    {
        std::vector<int> t = {
            1, 3, 64, 64
        };

        input = std::make_shared<Tensor<float>>(t);

        auto *input_ptr = input->data();
        expectedOutput.resize(input->num_elements());
        
        std::random_device rd{};
        std::mt19937 gen{rd()};
        gen.seed(1024);
        std::normal_distribution<> input_dist{0.0F, 1.0F};
        for (int i = 0; i < input->num_elements(); i++) {
            input_ptr[i] = input_dist(gen);
            expectedOutput[i] = std::atan(input_ptr[i]);
        }
    }
};
}

int main() {
    Logger::getInstance().setLevel(LOG_INFO);
    Logger::getInstance().enableFileOutput("log", false);

    AtanTest atantest;
    if (!atantest.run_test({atantest.input}, atantest.expectedOutput)) {
        return -1;
    }

    return 0;
}