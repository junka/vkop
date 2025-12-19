#include <cstdint>
#include <string>
#include <vector>
#include <random>
#include <cmath>

#include "setup.hpp"
#include "core/Tensor.hpp"
#include "include/logger.hpp"
#include "ops/Gemm.hpp"

using vkop::core::Tensor;
using vkop::tests::TestCase;
using vkop::ops::Gemm;
using vkop::core::ITensor;

namespace {
template <typename T>
void reference_gemm(const std::shared_ptr<Tensor<T>> &inputa, const std::shared_ptr<Tensor<T>> &inputb,
        const std::shared_ptr<Tensor<T>> &inputc, std::shared_ptr<Tensor<float>> &output,
        int M, int N, int K, float alpha, float beta, bool transA, bool transB, bool has_bias) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0F;
            for (int k = 0; k < K; ++k) {
                float a_val;
                float b_val;

                if (transA) {
                    // inputa is stored as [K][M], so A^T[i][k] = inputa[k][i]
                    if (typeid(T) == typeid(uint16_t)) {
                        a_val = ITensor::fp16_to_fp32((*inputa)[(k * M) + i]);
                    } else {
                        a_val = (*inputa)[(k * M) + i];
                    }
                } else {
                    // inputa is stored as [M][K]
                    if (typeid(T) == typeid(uint16_t)) {
                        a_val = ITensor::fp16_to_fp32((*inputa)[(i * K) + k]);
                    } else {
                        a_val = (*inputa)[(i * K) + k];
                    }
                }

                if (transB) {
                    // inputb is stored as [N][K], so B^T[k][j] = inputb[j][k]
                    if (typeid(T) == typeid(uint16_t)) {
                        b_val = ITensor::fp16_to_fp32((*inputb)[(j * K) + k]);
                    } else {
                        b_val = (*inputb)[(j * K) + k];
                    }
                } else {
                    // inputb is stored as [K][N]
                    if (typeid(T) == typeid(uint16_t)) {
                        b_val = ITensor::fp16_to_fp32((*inputb)[(k * N) + j]);
                    } else {
                        b_val = (*inputb)[(k * N) + j];
                    }
                }

                sum += a_val * b_val;
            }

            sum *= alpha;
            if (has_bias && inputc != nullptr) {
                if (typeid(T) == typeid(uint16_t)) {
                    sum += beta * ITensor::fp16_to_fp32((*inputc)[(i * N) + j]);
                } else {
                   sum += beta * (*inputc)[(i * N) + j];
                }
            }
            (*output)[(i * N) + j] = sum;
        }
    }
}

template <typename T>
class GemmTest : public TestCase {
public:
    std::shared_ptr<Tensor<T>> inputa;
    std::shared_ptr<Tensor<T>> inputb;
    std::shared_ptr<Tensor<T>> inputc;
    std::shared_ptr<Tensor<float>> output;

    float alpha = 1.0F;
    float beta = 1.0F;
    bool transA = false;
    bool transB = true;

    std::unordered_map<std::string, std::string> attr = {
        {"alpha", "1"},
        {"beta", "1"},
        {"transA", transA ? "1" : "0"},
        {"transB", transB ? "1" : "0"},
    };
    GemmTest() : TestCase("Gemm") {
        initTestdata();
    }

private:
    void initTestdata() {
        std::vector<int> t1 = {1, 2048};
        std::vector<int> t2 = {1000, 2048};
        inputa = std::make_shared<Tensor<T>>(t1);
        inputb = std::make_shared<Tensor<T>>(t2);
        int m = transA ? t1[1] : t1[0];
        int ka = transA ? t1[0] : t1[1];
        // int kb = transB ? t2[1] : t2[0];
        int n = transB ? t2[0] : t2[1];
        int k = ka;
        inputc = std::make_shared<Tensor<T>>(std::vector<int>{m, n});
        output = std::make_shared<Tensor<float>>(std::vector<int>{t1[0], t2[1]});
        inputa->reserveOnCPU();
        inputb->reserveOnCPU();
        inputc->reserveOnCPU();
        output->reserveOnCPU();

        std::random_device rd{};
        std::mt19937 gen{rd()};
        gen.seed(1024);
        std::normal_distribution<> input_dist{-3.0F, 2.0F};
        for (int i = 0; i < inputa->num_elements(); i++) {
            if (typeid(T) == typeid(uint16_t)) {
                (*inputa)[i] = ITensor::fp32_to_fp16(input_dist(gen));
            } else {
                (*inputa)[i] = input_dist(gen);
            }
        }
        for (int i = 0; i < inputb->num_elements(); i++) {
            if (typeid(T) == typeid(uint16_t)) {
                (*inputb)[i] = ITensor::fp32_to_fp16(input_dist(gen));
            } else {
                (*inputb)[i] = input_dist(gen);
            }
        }
        for (int i = 0; i < inputc->num_elements(); ++i) {
            if (typeid(T) == typeid(uint16_t)) {
                (*inputc)[i] = ITensor::fp32_to_fp16(input_dist(gen));
            } else {
                (*inputc)[i] = input_dist(gen);
            }
        }
        printf("M %d, N %d, K %d\n", m, n, k);
        printf("==============================================================\n");
        printf("Input A:\n");
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < k; j++) {
                if (typeid(T) == typeid(uint16_t)) {
                    std::cout << ITensor::fp16_to_fp32((*inputa)[(i * k) + j]) << " ";
                } else {
                    std::cout << (*inputa)[(i * k) + j] << " ";
                }
            }
            printf("\n");
        }
        printf("\n");
        printf("Input B:\n");
        for (int i = 0; i < k; i++) {
            for (int j = 0; j < n; j++) {
                if (typeid(T) == typeid(uint16_t)) {
                    std::cout << ITensor::fp16_to_fp32((*inputb)[(i * n) + j]) << " ";
                } else {
                    std::cout << (*inputb)[(i * n) + j] << " ";
                }
            }
            printf("\n");
        }
        printf("\n");
        printf("Input C:\n");
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (typeid(T) == typeid(uint16_t)) {
                    std::cout << ITensor::fp16_to_fp32((*inputc)[(i * n) + j]) << " ";
                } else {
                    std::cout << (*inputc)[(i * n) + j] << " ";
                }
            }
            printf("\n");
        }

        reference_gemm(inputa, inputb, inputc, output, m, n, k, alpha, beta, transA, transB, true);
        printf("\n");
        printf("Output:\n");
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                std::cout << (*output)[(i * n) + j] << " ";
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
    GemmTest<float> gmtest1;
    if (!gmtest1.run_test<float>({gmtest1.inputa, gmtest1.inputb, gmtest1.inputc}, {gmtest1.output},
        [&gmtest1](std::unique_ptr<vkop::ops::Operator> &op) {
            auto *gemm_op = dynamic_cast<Gemm *>(op.get());
            if (!gemm_op) {
                LOG_ERROR("Failed to cast operator to Gemm");
                return;
            }
            gemm_op->setAttribute(gmtest1.attr);
        })) {
        return -1;
    }

    GemmTest<uint16_t> gmtest;
    if (!gmtest.run_test<uint16_t>({gmtest.inputa, gmtest.inputb, gmtest.inputc}, {gmtest.output},
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