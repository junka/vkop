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
        1, 3, 64, 64
    };
    std::shared_ptr<Tensor<uint16_t>> inputa;
    std::shared_ptr<Tensor<uint16_t>> inputb;
    std::vector<uint16_t> expectedOutput;

    AddTest():TestCase("Add") {
        initTestdata();
    }
private:
    void initTestdata()
    {
        inputa = std::make_shared<Tensor<uint16_t>>(input_shape_);
        inputb = std::make_shared<Tensor<uint16_t>>(input_shape_);

        auto *inputa_ptr = inputa->data();
        auto *inputb_ptr = inputb->data();
        expectedOutput.resize(inputa->num_elements());
        
        std::random_device rd{};
        std::mt19937 gen{rd()};
        gen.seed(1024);
        std::normal_distribution<> inputa_dist{0.0F, 1.0F};
        std::normal_distribution<> inputb_dist{1.0F, 2.0F};
        for (int i = 0; i < inputa->num_elements(); i++) {
            auto a = inputa_dist(gen);
            auto b = inputb_dist(gen);
            inputa_ptr[i] = float32_to_float16(a);
            inputb_ptr[i] = float32_to_float16(b);
            expectedOutput[i] = float32_to_float16(a+b);
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