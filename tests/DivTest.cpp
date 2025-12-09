#include <cstdint>
#include <vector>
#include <random>
#include <iomanip>

#include "setup.hpp"
#include "core/Tensor.hpp"
#include "include/logger.hpp"

using vkop::core::Tensor;
using vkop::tests::TestCase;

namespace {

class DivTest : public TestCase {
public:
    std::shared_ptr<Tensor<float>> inputa;
    std::shared_ptr<Tensor<float>> inputb;
    std::shared_ptr<Tensor<float>> output;

    DivTest():TestCase("Div") {
        initTestdata();
    }
private:
    void initTestdata()
    {
        std::vector<int> t = {
            4, 1, 64, 64
        };
        inputa = std::make_shared<Tensor<float>>(t);
        inputb = std::make_shared<Tensor<float>>(t);
        inputa->reserveOnCPU();
        inputb->reserveOnCPU();
        output = std::make_shared<Tensor<float>>(t);
        output->reserveOnCPU();
        
        std::random_device rd{};
        std::mt19937 gen{rd()};
        gen.seed(1024);
        std::normal_distribution<> inputa_dist{0.2F, 3.0F};
        std::normal_distribution<> inputb_dist{2.0F, 4.0F};
        for (int i = 0; i < inputa->num_elements(); i++) {
            (*inputa)[i] = inputa_dist(gen);
            (*inputb)[i] = inputa_dist(gen);
            (*output)[i] = (*inputa)[i] / (*inputb)[i];
            std::cout << "i " << i << ": "<< std::setprecision(15) << (*inputa)[i];
            std::cout << ", " << std::setprecision(15) <<  (*inputb)[i];
            std::cout << " -> " << std::setprecision(15) << (*output)[i] << "\n";
        }
    }
};
}

int main() {
    Logger::getInstance().setLevel(LOG_INFO);
    Logger::getInstance().enableFileOutput("log", false);

    DivTest divtest;
    if (!divtest.run_test<float>({divtest.inputa, divtest.inputb}, {divtest.output})) {
        return -1;
    }

    return 0;
}