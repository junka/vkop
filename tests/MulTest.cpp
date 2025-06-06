#include <cstdint>
#include <vector>
#include <random>

#include "setup.hpp"
#include "Tensor.hpp"
#include "logger.hpp"

using vkop::core::Tensor;
using vkop::tests::TestCase;

namespace {

class MulTest : public TestCase {
public:
    int batch_;
    int depth_;
    int height_;
    int width_;
    std::shared_ptr<Tensor<float>> inputa;
    std::shared_ptr<Tensor<float>> inputb;
    std::vector<float> expectedOutput;

    MulTest():TestCase("Mul") {
        initTestdata();
    }
private:
    void initTestdata()
    {
        std::vector<int> t = {
            1, 3, 64, 64
        };
        batch_ = t[0];
        depth_ = t[1];
        height_ = t[2];
        width_ = t[3];
        inputa = std::make_shared<Tensor<float>>(batch_, depth_, height_, width_);
        inputb = std::make_shared<Tensor<float>>(batch_, depth_, height_, width_);

        auto *inputa_ptr = inputa->data();
        auto *inputb_ptr = inputb->data();
        expectedOutput.resize(inputa->num_elements());
        
        std::random_device rd{};
        std::mt19937 gen{rd()};
        gen.seed(1024);
        std::normal_distribution<> inputa_dist{0.0F, 1.0F};
        std::normal_distribution<> inputb_dist{1.0F, 2.0F};
        for (int i = 0; i < inputa->num_elements(); i++) {
            inputa_ptr[i] = inputa_dist(gen);
            inputb_ptr[i] = inputa_dist(gen);
            expectedOutput[i] = inputa_ptr[i] * inputb_ptr[i];
        }
    }
};
}

int main() {
    Logger::getInstance().setLevel(LOG_INFO);
    Logger::getInstance().enableFileOutput("log", false);

    MulTest multest;
    if (!multest.run_test({multest.inputa, multest.inputb}, multest.expectedOutput)) {
        return -1;
    }

    return 0;
}