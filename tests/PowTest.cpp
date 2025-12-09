#include <cstdint>
#include <vector>
#include <random>
#include <cmath>

#include "setup.hpp"
#include "core/Tensor.hpp"
#include "include/logger.hpp"

using vkop::core::Tensor;
using vkop::tests::TestCase;

namespace {

class PowTest : public TestCase {
public:
    std::shared_ptr<Tensor<float>> inputa;
    std::shared_ptr<Tensor<float>> inputb;
    std::shared_ptr<Tensor<float>> output;

    PowTest():TestCase("Pow") {
        initTestdata();
    }
private:
    void initTestdata()
    {
        std::vector<int> t = {
            1, 5, 64, 64
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
        std::normal_distribution<> inputa_dist{1.0F, 0.5F};
        std::normal_distribution<> inputb_dist{0.0F, 1.5F};
        for (int i = 0; i < inputa->num_elements(); i++) {
            (*inputa)[i] = inputa_dist(gen);
            (*inputb)[i] = inputa_dist(gen);
            (*output)[i] = std::pow((*inputa)[i], (*inputb)[i]);
        }
    }
};
}

int main() {
    Logger::getInstance().setLevel(LOG_INFO);
    Logger::getInstance().enableFileOutput("log", false);

    PowTest powtest;
    if (!powtest.run_test<float>({powtest.inputa, powtest.inputb}, {powtest.output})) {
        return -1;
    }

    return 0;
}