#include <string>
#include <vector>
#include <random>
#include <cmath>

#include "setup.hpp"
#include "core/Tensor.hpp"
#include "include/logger.hpp"

using vkop::core::Tensor;
using vkop::tests::TestCase;
void reference_matmul(const std::shared_ptr<Tensor<float>> &inputa, const std::shared_ptr<Tensor<float>> &inputb, std::shared_ptr<Tensor<float>> &output) {
    int M = inputa->get_height(); // A: [..., M, K]
    int K = inputa->get_width();
    int N = inputb->get_width();
    auto batch = inputa->get_batch();
    auto chan = inputa->get_channel();
    printf("M: %d, K: %d, N: %d\n", M, K, N);

    for (int b = 0; b < batch; b++) {
        for (int c = 0; c < chan; c++) {
            for (int i = 0; i < M; i++) {
                for (int j = 0; j < N; j++) {
                    float sum = 0.0F;
                    size_t idxc = (b * chan * M * N) + (c * M * N) + (i * N) + j;
                    for (int k = 0; k < K; k++) {
                        // M * k, K * N
                        size_t idxa = (b * chan * M * K) + (c * M * K) + (i * K) + k;
                        size_t idxb = (b * chan * K * N) + (c * K * N) + (k * N) + j;
                        sum += (*inputa)[idxa] * (*inputb)[idxb];
                    }
                    (*output)[idxc] = sum;
                }
            }
        }
    }
}

namespace {

class MatMulTest : public TestCase {
public:
    std::shared_ptr<Tensor<float>> inputa;
    std::shared_ptr<Tensor<float>> inputb;
    std::shared_ptr<Tensor<float>> output;

    MatMulTest() : TestCase("MatMul") {
        initTestdata();
    }

private:
    void initTestdata() {
        std::vector<int> t1 = {3, 4, 1};
        std::vector<int> t2 = {3, 1, 6};
        inputa = std::make_shared<Tensor<float>>(t1);
        inputb = std::make_shared<Tensor<float>>(t2);
        size_t rank = t1.size();
        std::vector<int> to = {3, t1[rank-2], t2[rank-1]};
        int kk = t1[rank-1];
        output = std::make_shared<Tensor<float>>(to);
        inputa->reserveOnCPU();
        inputb->reserveOnCPU();
        output->reserveOnCPU();

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

        printf("M %d, N %d, K %d\n", to[rank-2], to[rank-1], kk);
        printf("==============================================================\n");
        printf("Input A:\n");
        auto shapea = inputa->getShape();
        printf("%d %d %d %d\n", shapea[0], shapea[1], shapea[2], shapea[3]);
        inputa->print_tensor();
        printf("Input B:\n");
        auto shapeb = inputb->getShape();
        inputb->print_tensor();
        reference_matmul(inputa, inputb, output);
        printf("Output:\n");
        output->print_tensor();
    }
};
}

int main() {
    Logger::getInstance().setLevel(LOG_INFO);
    Logger::getInstance().enableFileOutput("log", false);

    MatMulTest mmtest;
    if (!mmtest.run_test<float>({mmtest.inputa, mmtest.inputb}, {mmtest.output})) {
        return -1;
    }

    return 0;
}