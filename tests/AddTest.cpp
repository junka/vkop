#include <cstdint>
#include <vector>
#include <random>

#include "setup.hpp"
#include "core/Tensor.hpp"
#include "include/logger.hpp"

using vkop::core::Tensor;
using vkop::core::ITensor;
using vkop::tests::TestCase;

namespace {
    

template<typename T>
class AddTest : public TestCase {
public:
    std::vector<int> input_shape_ = {
        10, 5, 64, 64
    };
    std::shared_ptr<Tensor<T>> inputa;
    std::shared_ptr<Tensor<T>> inputb;
    std::vector<T> expectedOutput;

    AddTest():TestCase("Add") {
        initTestdata();
    }
private:
    void initTestdata()
    {
        inputa = std::make_shared<Tensor<T>>(input_shape_);
        inputb = std::make_shared<Tensor<T>>(input_shape_);
        inputa->reserveOnCPU();
        inputb->reserveOnCPU();

        expectedOutput.resize(inputa->num_elements());
        std::random_device rd{};
        std::mt19937 gen{rd()};
        gen.seed(1024);
        std::normal_distribution<> inputa_dist{-1.0F, 1.0F};
        std::normal_distribution<> inputb_dist{1.0F, 2.0F};
        for (int i = 0; i < inputa->num_elements(); i++) {
            auto a = inputa_dist(gen);
            auto b = inputb_dist(gen);
            if (typeid(T) == typeid(uint16_t)) {
                (*inputa)[i] = ITensor::fp32_to_fp16(a);
                (*inputb)[i] = ITensor::fp32_to_fp16(b);
                expectedOutput[i] = ITensor::fp32_to_fp16(a+b);
            } else {
                (*inputa)[i] = a;
                (*inputb)[i] = b;
                expectedOutput[i] = a+b;
            }
        }
    }
};
}

int main() {
    Logger::getInstance().setLevel(LOG_INFO);
    Logger::getInstance().enableFileOutput("log", false);
#ifdef FP16
    AddTest<uint16_t> addtest;
#else
    AddTest<float> addtest;
#endif
    if (!addtest.run_test({addtest.inputa, addtest.inputb}, addtest.expectedOutput)) {
        return -1;
    }

    return 0;
}