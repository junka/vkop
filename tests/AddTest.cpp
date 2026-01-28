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
    std::vector<int> input_shape_;
    std::shared_ptr<Tensor<T>> inputa;
    std::shared_ptr<Tensor<T>> inputb;
    std::shared_ptr<Tensor<T>> output;

    explicit AddTest(std::vector<int> input_shape):TestCase("Add"), input_shape_(std::move(input_shape)) {
        initTestdata();
    }
private:
    void initTestdata()
    {
        inputa = std::make_shared<Tensor<T>>(input_shape_);
        inputb = std::make_shared<Tensor<T>>(input_shape_);
        inputa->reserveOnCPU();
        inputb->reserveOnCPU();
        output = std::make_shared<Tensor<T>>(input_shape_);
        output->reserveOnCPU();

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
                (*output)[i] = ITensor::fp32_to_fp16(a+b);
            } else {
                (*inputa)[i] = a;
                (*inputb)[i] = b;
                (*output)[i] = a+b;
            }
        }
    }
};
}

int main() {
    Logger::getInstance().setLevel(LOG_INFO);
    Logger::getInstance().enableFileOutput("log", false);
    std::vector<std::vector<int>> test_cases = {
        {10, 5, 64, 64},
        {1, 3, 128, 128},
        {2, 4, 32, 32}
    };

    for (const auto& t : test_cases) {
        AddTest<uint16_t> addtest(t);
        if (!addtest.run_test<uint16_t>({addtest.inputa, addtest.inputb}, {addtest.output})) {
            return -1;
        }
    }

    return 0;
}