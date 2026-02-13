#include <vector>
#include <random>
#include <cmath>

#include "setup.hpp"
#include "core/Tensor.hpp"
#include "include/logger.hpp"

using vkop::core::Tensor;
using vkop::tests::TestCase;

namespace {

class SoftplusTest : public TestCase {
public:
    std::shared_ptr<Tensor<float>> input;
    std::shared_ptr<Tensor<float>> output;

    SoftplusTest():TestCase("Softplus") {
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
            (*output)[i] = (std::log1p(std::exp((*input)[i])));
        }
    }
};
}

int main() {
    Logger::getInstance().setLevel(LOG_INFO);
    Logger::getInstance().enableFileOutput("log", false);
    vkop::tests::TestEnv::initialize();

    SoftplusTest sptest;
    if (!sptest.run_test<float>({sptest.input}, {sptest.output})) {
        return -1;
    }

    vkop::tests::TestEnv::cleanup();
    return 0;
}