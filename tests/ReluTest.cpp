#include <cstdint>
#include <vector>
#include <random>
#include <cmath>
#include <stack>

#include "setup.hpp"
#include "Tensor.hpp"
#include "logger.hpp"

using vkop::core::Tensor;
using vkop::tests::TestCase;

namespace {

float reference_relu(float val)
{
    return std::fmax(val, 0.0F);
}

class ReluTest : public TestCase {
public:
    int batch_;
    int depth_;
    int height_;
    int width_;
    std::shared_ptr<Tensor<float>> input;
    std::vector<float> expectedOutput;

    ReluTest():TestCase("Relu") {
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
        input = std::make_shared<Tensor<float>>(batch_, depth_, height_, width_);

        auto *input_ptr = input->data();
        expectedOutput.resize(input->num_elements());

        std::random_device rd{};
        std::mt19937 gen{rd()};
        gen.seed(1024);
        std::normal_distribution<> input_dist{-4.0F, 6.0F};
        for (int i = 0; i < input->num_elements(); i++) {
            input_ptr[i] = input_dist(gen);
            expectedOutput[i] = reference_relu(input_ptr[i]);
        }
    }
};
}

int main() {
    Logger::getInstance().setLevel(LOG_INFO);
    Logger::getInstance().enableFileOutput("log", false);

    ReluTest relutest;
    if (!relutest.run_test({relutest.input}, relutest.expectedOutput)) {
        return -1;
    }

    return 0;
}