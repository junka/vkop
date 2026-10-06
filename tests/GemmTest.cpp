#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <tuple>
#include <type_traits>
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
        // ONNX Gemm C is 1-D [N], broadcast across all M rows (the shader
        // reads bias by column only). A [M,N] C would make rows i>=1 read
        // C[j] (row 0) while the expected adds C[i*N+j] -> row-indexed
        // mismatch (regression guard in c4251a6).
        inputc = std::make_shared<Tensor<T>>(std::vector<int>{n});
        output = std::make_shared<Tensor<T>>(std::vector<int>{m, n});

        torch::manual_seed(42);
        std::vector<int64_t> t1shape(t1.begin(), t1.end());
        std::vector<int64_t> t2shape(t2.begin(), t2.end());
        auto torch_inputa = torch::randn(t1shape, this->getTorchConf());
        auto torch_inputb = torch::randn(t2shape, this->getTorchConf());
        auto torch_inputc = torch::randn({n}, this->getTorchConf());

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
        // M>1 fp16 transB + bias case: the existing fp16 transB case only used
        // M=1, so a row-1+ bias-indexing bug (C read as [M][N] instead of 1-D
        // [N] broadcast) never surfaced there. Small M/N/K keeps it fast.
        {{8, 64}, {128, 64}, 1.0F, 1.0F, false, true},
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

namespace {
// int8 weight-only Gemm (buffer backend, where the int8 kernel lives). B is the
// quantized weight and its per-output-column fp32 scale (amax/127 reduced over
// K) is the LAST input, the same convention the optimizer appends and Conv2d
// uses:
//   [A, B_int8, scale]        no bias
//   [A, B_int8, C, scale]     with bias
// transB picks the physical layout ([K, N] when 0, [N, K] when 1) while the
// scale always holds N entries. The reference is computed from the DEQUANTIZED
// weight, so this pins the kernel's byte unpack and its fold-the-scale-out-of-
// the-K-loop accumulate rather than re-measuring quantization noise.
template <typename T>
class GemmInt8Test : public TestCase<T> {
public:
    std::unordered_map<std::string, std::string> attr;
    std::shared_ptr<Tensor<T>> inputa;
    std::shared_ptr<Tensor<int8_t>> weight;
    std::shared_ptr<Tensor<float>> scale_data;
    std::shared_ptr<Tensor<T>> inputc;
    std::shared_ptr<Tensor<T>> output;

    GemmInt8Test(int m, int k, int n, bool transB, bool has_bias)
        : TestCase<T>("Gemm"), m_(m), k_(k), n_(n), transB_(transB),
          has_bias_(has_bias) {
        attr = {{"alpha", "1"},
                {"beta", "1"},
                {"transA", "0"},
                {"transB", transB_ ? "1" : "0"}};
        initTestData();
    }

    // The quantization noise itself is out of scope here (the reference uses
    // the dequantized weight), but the folded scale turns the accumulate into a
    // different summation order, and the fp16 output rounds twice: keep the
    // loose bound int8 Conv2d uses.
    bool verify_output(const std::unique_ptr<vkop::ops::Operator> &op, int idx,
                       const std::shared_ptr<vkop::core::ITensor> &output,
                       const std::shared_ptr<vkop::core::ITensor> &expect)
        override {
        auto out = vkop::core::as_tensor<T>(output);
        auto exp = vkop::core::as_tensor<T>(expect);
        for (int i = 0; i < out->num_elements(); i++) {
            float ov = to_float((*out)[i]);
            float ev = to_float((*exp)[i]);
            if (std::isnan(ov)) {
                LOG_ERROR("int8 Gemm NaN at %d, expected %f", i, ev);
                return false;
            }
            float threshold = std::max(0.05F, std::abs(ev) * 0.05F);
            if (std::abs(ov - ev) > threshold) {
                LOG_ERROR("int8 Gemm Fail (%d): %f vs %f (thr %f)", i, ov, ev,
                          threshold);
                return false;
            }
        }
        return true;
    }

private:
    int m_, k_, n_;
    bool transB_, has_bias_;

    static float to_float(T v) {
        if constexpr (std::is_same_v<T, uint16_t>) {
            return vkop::core::ITensor::fp16_to_fp32(v);
        } else {
            return v;
        }
    }

    // Everything is generated in fp32 and only the tensors this build stores
    // are cast; the reference then reads back exactly the stored values, so the
    // comparison is against what the kernel actually sees.
    static torch::Tensor store(const torch::Tensor &t) {
        if constexpr (std::is_same_v<T, uint16_t>) {
            return t.to(torch::kFloat16);
        } else {
            return t;
        }
    }

    static torch::Tensor as_ref(const torch::Tensor &t) {
        return t.to(torch::kFloat32);
    }

    void initTestData() {
        auto f32 = torch::TensorOptions().dtype(torch::kFloat32);
        torch::manual_seed(42);

        auto a = store(torch::randn({m_, k_}, f32));
        // transB=1 stores the weight as [N, K], so the axis that is NOT N is
        // the one reduced for the per-column amax.
        auto w_src =
            torch::randn(transB_ ? std::vector<int64_t>{n_, k_}
                                 : std::vector<int64_t>{k_, n_}, f32);
        const int reduce_axis = transB_ ? 1 : 0;
        auto amax = std::get<0>(w_src.abs().max(reduce_axis, true));
        auto scale = amax / 127.0;
        scale = torch::where(scale == 0, torch::ones_like(scale), scale);
        auto q = torch::round(w_src / scale).to(torch::kInt8);
        auto deq = q.to(torch::kFloat32) * scale.to(torch::kFloat32);
        auto b = transB_ ? deq.t() : deq; // [K, N]

        auto y = torch::matmul(as_ref(a), b);
        auto bias = store(torch::randn({n_}, f32));
        if (has_bias_) {
            y = y + as_ref(bias);
        }

        inputa = std::make_shared<Tensor<T>>(std::vector<int>{m_, k_});
        this->fillTensorFromTorch(inputa, a);
        weight = std::make_shared<Tensor<int8_t>>(
            std::vector<int>{transB_ ? n_ : k_, transB_ ? k_ : n_});
        auto cpu_q = q.cpu().contiguous().flatten();
        auto *qptr = cpu_q.data_ptr<int8_t>();
        weight->fillToCPU(std::vector<int8_t>(qptr, qptr + cpu_q.numel()));
        scale_data = std::make_shared<Tensor<float>>(std::vector<int>{n_});
        scale_data->fillToCPU(to_std_vector(scale.cpu().contiguous().flatten()));
        if (has_bias_) {
            inputc = std::make_shared<Tensor<T>>(std::vector<int>{n_});
            this->fillTensorFromTorch(inputc, bias);
        }
        output = std::make_shared<Tensor<T>>(std::vector<int>{m_, n_});
        this->fillTensorFromTorch(output, store(y));

        LOG_INFO("int8 Gemm M %d, N %d, K %d, transB %d, bias %d, fp16 %d", m_,
                 n_, k_, transB_ ? 1 : 0, has_bias_ ? 1 : 0,
                 std::is_same_v<T, uint16_t> ? 1 : 0);
    }

    static std::vector<float> to_std_vector(const torch::Tensor &t) {
        auto acc = t.accessor<float, 1>();
        std::vector<float> v;
        v.reserve(t.numel());
        for (int64_t i = 0; i < t.numel(); i++) v.push_back(acc[i]);
        return v;
    }
};

template <typename T>
void run_gemm_int8(const std::vector<std::tuple<int, int, int, bool, bool>> &cases) {
    // fp16 output packs two columns per word, so every N here is even (the
    // odd-N tail is a known pre-existing limitation of gemm16, not something
    // the int8 path introduces).
    for (const auto &tc : cases) {
        auto [m, k, n, transB, has_bias] = tc;
        GemmInt8Test<T> t(m, k, n, transB, has_bias);
        std::vector<std::shared_ptr<vkop::core::ITensor>> inputs =
            has_bias ? std::vector<std::shared_ptr<vkop::core::ITensor>>{
                           t.inputa, t.weight, t.inputc, t.scale_data}
                     : std::vector<std::shared_ptr<vkop::core::ITensor>>{
                           t.inputa, t.weight, t.scale_data};
        EXPECT_TRUE(t.run_test(inputs, {t.output},
            [&t](std::unique_ptr<vkop::ops::Operator> &op) {
                auto *gemm_op = dynamic_cast<Gemm *>(op.get());
                if (!gemm_op) {
                    LOG_ERROR("Failed to cast operator to Gemm");
                    return;
                }
                gemm_op->setAttribute(t.attr);
            }));
    }
}
} // namespace

TEST(GemmTest, GemmInt8WeightOnlyBuffer) {
    vkop::tests::ScopedBufferBackend buffer;
    const std::vector<std::tuple<int, int, int, bool, bool>> cases = {
        {1, 2048, 1000, true, false}, // M=1 GEMV, transB=1, no bias
        {8, 64, 128, true, true},     // transB=1 with bias
        {4, 32, 16, false, false},    // [K, N] layout, no bias
        {5, 20, 18, false, true},     // [K, N] layout with bias, odd M
    };
    LOG_INFO("int8 Gemm, FP32");
    run_gemm_int8<float>(cases);
    LOG_INFO("int8 Gemm, FP16");
    run_gemm_int8<uint16_t>(cases);
}

// Same contract on the Gemm side: the scale indexes Y's columns, so a K-long
// scale (quantized along the wrong axis) must be refused rather than folded
// into the accumulate.
TEST(GemmTest, GemmInt8WrongAxisScaleThrows) {
    vkop::tests::ScopedBufferBackend buffer;
    GemmInt8Test<float> t(/*m=*/4, /*k=*/32, /*n=*/16, /*transB=*/false,
                          /*has_bias=*/false);
    auto wrong = std::make_shared<Tensor<float>>(std::vector<int>{32});
    wrong->fillToCPU(std::vector<float>(32, 1.0F));
    EXPECT_THROW(t.run_test({t.inputa, t.weight, wrong}, {t.output},
                            [&t](std::unique_ptr<vkop::ops::Operator> &op) {
                                op->setAttribute(t.attr);
                            }),
                 std::runtime_error);
}