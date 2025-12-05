#include <vector>
#include <random>

#include "setup.hpp"
#include "core/Tensor.hpp"
#include "include/logger.hpp"

using vkop::core::Tensor;
using vkop::tests::TestCase;

namespace {
class ReshapeTest : public TestCase {
public:
    std::vector<int> input_shape_ = {
        1, 6, 4, 4
    };
    const std::unordered_map<std::string, std::string> param = {
        {"allowzero ", "0"}
    };
    std::shared_ptr<Tensor<float>> input;
    std::shared_ptr<Tensor<int64_t>> shape;
    std::vector<float> expectedOutput;

    ReshapeTest():TestCase("Reshape") {
        initTestdata();
    }
private:
    void initTestdata()
    {
        input = std::make_shared<Tensor<float>>(input_shape_);
        input->reserveOnCPU();
        shape = std::make_shared<Tensor<int64_t>>(4);
        expectedOutput.resize(input->num_elements());
        std::random_device rd{};
        std::mt19937 gen{rd()};
        gen.seed(1024);
        std::normal_distribution<> input_dist{-1.0F, 1.0F};
        for (int i = 0; i < input->num_elements(); i++) {
            auto a = input_dist(gen);
            (*input)[i] = a;
            expectedOutput[i] = a;
        }
        (*shape)[0] = 1;
        (*shape)[1] = 3;
        (*shape)[2] = 8;
        (*shape)[3] = 4;
    }
};
}


int main()
{
    Logger::getInstance().setLevel(LOG_INFO);
    Logger::getInstance().enableFileOutput("log", false);

    ReshapeTest reshape_test;
    if (!reshape_test.run_test({reshape_test.input, reshape_test.shape}, reshape_test.expectedOutput)) {
        return -1;
    }
    return 0;
}