#include <cstdint>
#include <string>
#include <vector>
#include <random>
#include <cmath>
#include <stack>

#include "setup.hpp"
#include "core/Tensor.hpp"
#include "include/logger.hpp"
#include "ops/Matmul.hpp"

using vkop::core::Tensor;
using vkop::tests::TestCase;
void reference_matmul(const std::shared_ptr<Tensor<float>> &inputa, const std::shared_ptr<Tensor<float>> &inputb, float *output, int M, int N, int K) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0F;
            for (int k = 0; k < K; k++) {
                sum += (*inputa)[i * K + k] * (*inputb)[k * N + j];
            }
            output[i * N + j] = sum;
        }
    }
}

namespace {

class MatMulTest : public TestCase {
public:
    std::shared_ptr<Tensor<float>> inputa;
    std::shared_ptr<Tensor<float>> inputb;
    std::vector<float> expectedOutput;

    MatMulTest() : TestCase("MatMul") {
        initTestdata();
    }

private:
    void initTestdata() {
        std::vector<int> t1 = {4, 8};
        std::vector<int> t2 = {8, 6};
        inputa = std::make_shared<Tensor<float>>(t1);
        inputb = std::make_shared<Tensor<float>>(t2);
        inputa->reserveOnCPU();
        inputb->reserveOnCPU();

        expectedOutput.resize(t1[0] * t2[1]);

        std::random_device rd{};
        std::mt19937 gen{rd()};
        gen.seed(1024);
        std::normal_distribution<> input_dist{-3.0F, 6.0F};
        for (int i = 0; i < inputa->num_elements(); i++) {
            (*inputa)[i] = input_dist(gen);
        }
        for (int i = 0; i < inputb->num_elements(); i++) {
            (*inputb)[i] = input_dist(gen);
        }

        printf("M %d, N %d, K %d\n", t1[0], t2[1], t1[1]);
        printf("==============================================================\n");
        printf("Input A:\n");
        for (int i = 0; i < t1[0]; i++) {
            for (int j = 0; j < t1[1]; j++) {
                printf("%f ", (*inputa)[i * t1[1] + j]);
            }
            printf("\n");
        }
        printf("\n");
        printf("Input B:\n");
        for (int i = 0; i < t2[0]; i++) {
            for (int j = 0; j < t2[1]; j++) {
                printf("%f ", (*inputb)[i * t2[1] + j]);
            }
            printf("\n");
        }

        reference_matmul(inputa, inputb, expectedOutput.data(), t1[0], t2[1], t1[1]);
        printf("\n");
        printf("Output:\n");
        for (int i = 0; i < t1[0]; i++) {
            for (int j = 0; j < t2[1]; j++) {
                printf("%f ", expectedOutput[i * t2[1] + j]);
            }
            printf("\n");
        }
        printf("\n");
    }
};
}

int main() {
    Logger::getInstance().setLevel(LOG_INFO);
    Logger::getInstance().enableFileOutput("log", false);

    MatMulTest mmtest;
    if (!mmtest.run_test({mmtest.inputa, mmtest.inputb}, mmtest.expectedOutput)) {
        return -1;
    }

    return 0;
}