#include <cstdint>
#include <vector>
#include <random>

#include "setup.hpp"
#include "core/Tensor.hpp"
#include "include/logger.hpp"

using vkop::core::Tensor;
using vkop::tests::TestCase;

namespace {
    

class AddTest : public TestCase {
public:
    std::vector<int> input_shape_ = {
        10, 5, 64, 64
    };
    std::shared_ptr<Tensor<float>> inputa;
    std::shared_ptr<Tensor<float>> inputb;
    std::vector<float> expectedOutput;

    AddTest():TestCase("Add") {
        initTestdata();
    }
private:
    void initTestdata()
    {
        inputa = std::make_shared<Tensor<float>>(input_shape_);
        inputb = std::make_shared<Tensor<float>>(input_shape_);

        auto *inputa_ptr = inputa->data();
        auto *inputb_ptr = inputb->data();
        expectedOutput.resize(inputa->num_elements());
        
        std::random_device rd{};
        std::mt19937 gen{rd()};
        gen.seed(1024);
        std::normal_distribution<> inputa_dist{-1.0F, 1.0F};
        std::normal_distribution<> inputb_dist{1.0F, 2.0F};
        for (int i = 0; i < inputa->num_elements(); i++) {
            auto a = inputa_dist(gen);
            auto b = inputb_dist(gen);
            inputa_ptr[i] = a;
            inputb_ptr[i] = b;
            expectedOutput[i] = a+b;
        }
    }
};
}

int main() {
    Logger::getInstance().setLevel(LOG_INFO);
    Logger::getInstance().enableFileOutput("log", false);

    AddTest addtest;
    if (!addtest.run_test({addtest.inputa, addtest.inputb}, addtest.expectedOutput)) {
        return -1;
    }

    return 0;
}