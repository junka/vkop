#include <cstdint>
#include <string>
#include <vector>
#include <random>
#include <cmath>
#include <stack>

#include "setup.hpp"
#include "core/Tensor.hpp"
#include "include/logger.hpp"
#include "ops/Gemm.hpp"

using vkop::core::Tensor;
using vkop::tests::TestCase;
using vkop::ops::Gemm;
void reference_gemm(const std::shared_ptr<Tensor<float>> &inputa, const std::shared_ptr<Tensor<float>> &inputb,
        const std::shared_ptr<Tensor<float>> &inputc, std::shared_ptr<Tensor<float>> &output,
        int M, int N, int K, float alpha, float beta, bool transA, bool transB, bool has_bias) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0F;
            for (int k = 0; k < K; ++k) {
                float a_val;
                float b_val;

                if (transA) {
                    // inputa is stored as [K][M], so A^T[i][k] = inputa[k][i]
                    a_val = (*inputa)[k * M + i];
                } else {
                    // inputa is stored as [M][K]
                    a_val = (*inputa)[i * K + k];
                }

                if (transB) {
                    // inputb is stored as [N][K], so B^T[k][j] = inputb[j][k]
                    b_val = (*inputb)[j * K + k];
                } else {
                    // inputb is stored as [K][N]
                    b_val = (*inputb)[k * N + j];
                }

                sum += a_val * b_val;
            }

            sum *= alpha;
            if (has_bias && inputc != nullptr) {
                sum += beta * (*inputc)[i * N + j];
            }

            (*output)[i * N + j] = sum;
        }
    }
}

namespace {

class GemmTest : public TestCase {
public:
    std::shared_ptr<Tensor<float>> inputa;
    std::shared_ptr<Tensor<float>> inputb;
    std::shared_ptr<Tensor<float>> inputc;
    std::shared_ptr<Tensor<float>> output;

    float alpha = 0.9F;
    float beta = 0.1F;
    bool transA = false;
    bool transB = false;

    std::unordered_map<std::string, std::string> attr = {
        {"alpha", "0.9"},
        {"beta", "0.1"},
        {"transA", transA ? "1" : "0"},
        {"transB", transB ? "1" : "0"},
    };
    GemmTest() : TestCase("Gemm") {
        initTestdata();
    }

private:
    void initTestdata() {
        std::vector<int> t1 = {4, 8};
        std::vector<int> t2 = {8, 6};
        inputa = std::make_shared<Tensor<float>>(t1);
        inputb = std::make_shared<Tensor<float>>(t2);
        int m = transA ? t1[1] : t1[0];
        int ka = transA ? t1[0] : t1[1];
        // int kb = transB ? t2[1] : t2[0];
        int n = transB ? t2[0] : t2[1];
        int k = ka;
        inputc = std::make_shared<Tensor<float>>(std::vector<int>{m, n});
        output = std::make_shared<Tensor<float>>(std::vector<int>{t1[0], t2[1]});
        inputa->reserveOnCPU();
        inputb->reserveOnCPU();
        inputc->reserveOnCPU();
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
        for (int i = 0; i < inputc->num_elements(); ++i) {
            (*inputc)[i] = input_dist(gen);
        }
        printf("M %d, N %d, K %d\n", m, n, k);
        printf("==============================================================\n");
        printf("Input A:\n");
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < k; j++) {
                printf("%f ", (*inputa)[i * k + j]);
            }
            printf("\n");
        }
        printf("\n");
        printf("Input B:\n");
        for (int i = 0; i < k; i++) {
            for (int j = 0; j < n; j++) {
                printf("%f ", (*inputb)[i * n + j]);
            }
            printf("\n");
        }
        printf("\n");
        printf("Input C:\n");
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                printf("%f ", (*inputc)[i * n + j]);
            }
            printf("\n");
        }

        reference_gemm(inputa, inputb, inputc, output, m, n, k, alpha, beta, transA, transB, true);
        printf("\n");
        printf("Output:\n");
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                printf("%f ", (*output)[i * n + j]);
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

    GemmTest gmtest;
    if (!gmtest.run_test<float>({gmtest.inputa, gmtest.inputb, gmtest.inputc}, {gmtest.output},
        [&gmtest](std::unique_ptr<vkop::ops::Operator> &op) {
            auto *gemm_op = dynamic_cast<Gemm *>(op.get());
            if (!gemm_op) {
                LOG_ERROR("Failed to cast operator to Gemm");
                return;
            }
            gemm_op->setAttribute(gmtest.attr);
        })) {
        return -1;
    }

    return 0;
}