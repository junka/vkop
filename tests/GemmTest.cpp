#include <vector>

#include "setup.hpp"
#include "core/Tensor.hpp"
#include "include/logger.hpp"
#include "ops/Gemm.hpp"

using vkop::core::Tensor;
using vkop::tests::TestCase;
using vkop::ops::Gemm;

namespace {
#ifdef USE_CPP_REF
template <typename T>
void reference_gemm(const std::shared_ptr<Tensor<T>> &inputa, const std::shared_ptr<Tensor<T>> &inputb,
        const std::shared_ptr<Tensor<T>> &inputc, std::shared_ptr<Tensor<T>> &output,
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
            if constexpr (std::is_same_v<T, float>) {
                (*output)[(i * N) + j] = sum;
            } else if constexpr (std::is_same_v<T, uint16_t>) {
                (*output)[(i * N) + j] = ITensor::fp32_to_fp16(sum);
            }
        }
    }
}
#endif

template <typename T>
class GemmTest : public TestCase<T> {
public:
    std::shared_ptr<Tensor<T>> inputa;
    std::shared_ptr<Tensor<T>> inputb;
    std::shared_ptr<Tensor<T>> inputc;
    std::shared_ptr<Tensor<T>> output;

    std::vector<int> t1;
    std::vector<int> t2;
    float alpha = 1.0F;
    float beta = 1.0F;
    bool transA;
    bool transB;

    std::unordered_map<std::string, std::string> attr;

    GemmTest(const std::vector<int> &inshapeA, const std::vector<int> &inshapeB, float alpha, float beta, bool transA, bool transB) : TestCase<T>("Gemm"),
     t1(inshapeA), t2(inshapeB), alpha(alpha), beta(beta), transA(transA), transB(transB) {
        attr = {
            {"alpha", "1"},
            {"beta", "1"},
            {"transA", transA ? "1" : "0"},
            {"transB", transB ? "1" : "0"},
        };
        initTestdata();
    }

private:
    void initTestdata() {
        inputa = std::make_shared<Tensor<T>>(t1);
        inputb = std::make_shared<Tensor<T>>(t2);
        int m = transA ? t1[1] : t1[0];
        int ka = transA ? t1[0] : t1[1];
        // int kb = transB ? t2[1] : t2[0];
        int n = transB ? t2[0] : t2[1];
        int k = ka;
        inputc = std::make_shared<Tensor<T>>(std::vector<int>{m, n});
        output = std::make_shared<Tensor<T>>(std::vector<int>{t1[0], t2[1]});

        torch::manual_seed(42);
        std::vector<int64_t> t1shape(t1.begin(), t1.end());
        std::vector<int64_t> t2shape(t2.begin(), t2.end());
        auto torch_inputa = torch::randn(t1shape, this->getTorchConf());
        auto torch_inputb = torch::randn(t2shape, this->getTorchConf());
        auto torch_inputc = torch::randn({m, n}, this->getTorchConf());

        this->fillTensorFromTorch(inputa, torch_inputa);
        this->fillTensorFromTorch(inputb, torch_inputb);
        this->fillTensorFromTorch(inputc, torch_inputc);
        printf("M %d, N %d, K %d\n", m, n, k);
        printf("==============================================================\n");
        printf("Input A:\n");
        inputa->print_tensor();
        printf("Input B:\n");
        inputb->print_tensor();

        if (transA) torch_inputa = torch_inputa.t();
        if (transB) torch_inputb = torch_inputb.t();

        auto torch_ouptput = alpha * torch::matmul(torch_inputa, torch_inputb);

        if (beta != 0.0F && torch_inputc.numel() > 0) {
            torch_ouptput = torch_ouptput + beta * torch_inputc;
        }
        this->fillTensorFromTorch(output, torch_ouptput);
        printf("output:\n");
        output->print_tensor();

    }
};
}

TEST(GemmTest, GemmComprehensiveTest) {

    const std::vector<std::tuple<std::vector<int>, std::vector<int>, float, float, bool, bool>> testcases = {
        {{1, 20}, {20, 16}, 1.0F, 1.0F, false, false},
        {{1, 2048}, {1000, 2048}, 1.0F, 1.0F, false, true},
    };

    for (const auto &testcase : testcases) {
        auto [inshapeA, inshapeB, alpha, beta, transA, transB] = testcase;

        LOG_INFO("Testing [%d, %d], [%d, %d], %f, %f, %d, %d", inshapeA[0], inshapeA[1], inshapeB[0], inshapeB[1], alpha, beta, transA, transB);
        LOG_INFO("Testing FP32");
        GemmTest<float> gmtest1(inshapeA, inshapeB, alpha, beta, transA, transB);
        EXPECT_TRUE(gmtest1.run_test({gmtest1.inputa, gmtest1.inputb, gmtest1.inputc}, {gmtest1.output},
            [&gmtest1](std::unique_ptr<vkop::ops::Operator> &op) {
                auto *gemm_op = dynamic_cast<Gemm *>(op.get());
                if (!gemm_op) {
                    LOG_ERROR("Failed to cast operator to Gemm");
                    return;
                }
                gemm_op->setAttribute(gmtest1.attr);
            }));
        LOG_INFO("Testing FP16");
        GemmTest<uint16_t> gmtest(inshapeA, inshapeB, alpha, beta, transA, transB);
        EXPECT_TRUE(gmtest.run_test({gmtest.inputa, gmtest.inputb, gmtest.inputc}, {gmtest.output},
            [&gmtest](std::unique_ptr<vkop::ops::Operator> &op) {
                auto *gemm_op = dynamic_cast<Gemm *>(op.get());
                if (!gemm_op) {
                    LOG_ERROR("Failed to cast operator to Gemm");
                    return;
                }
                gemm_op->setAttribute(gmtest.attr);
            }));
    }
}