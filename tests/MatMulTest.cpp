#include <algorithm>
#include <stdexcept>
#include <tuple>
#include <vector>
#include <cmath>

#include "setup.hpp"
#include "core/Tensor.hpp"
#include "include/logger.hpp"

using vkop::core::Tensor;
using vkop::tests::TestCase;

#ifdef USE_CPP_REF
static void reference_matmul(const std::shared_ptr<Tensor<float>> &inputa, const std::shared_ptr<Tensor<float>> &inputb, std::shared_ptr<Tensor<float>> &output) {
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
#endif
namespace {

template<typename T>
class MatMulTest : public TestCase<T> {
public:
    std::vector<int> t1;
    std::vector<int> t2;
    std::shared_ptr<Tensor<T>> inputa;
    std::shared_ptr<Tensor<T>> inputb;
    std::shared_ptr<Tensor<T>> output;

    MatMulTest(const std::vector<int> &t1, const std::vector<int> &t2) : TestCase<T>("MatMul"), t1(t1), t2(t2) {
        initTestdata();
    }

private:
    void initTestdata() {
        int batch = t1[0];
        inputa = std::make_shared<Tensor<T>>(t1);
        inputb = std::make_shared<Tensor<T>>(t2);
        size_t rank = t1.size();
        std::vector<int> to = {batch, t1[rank-2], t2[rank-1]};
        int kk = t1[rank-1];
        output = std::make_shared<Tensor<T>>(to);
        std::vector<int64_t> t1shape(t1.begin(), t1.end());
        std::vector<int64_t> t2shape(t2.begin(), t2.end());

        auto torch_in1 = torch::randn(t1shape, this->getTorchConf());
        auto torch_in2 = torch::randn(t2shape, this->getTorchConf());

        auto torch_output = torch::matmul(torch_in1, torch_in2);
        this->fillTensorFromTorch(inputa, torch_in1);
        this->fillTensorFromTorch(inputb, torch_in2);
        this->fillTensorFromTorch(output, torch_output);

        printf("M %d, N %d, K %d\n", to[rank-2], to[rank-1], kk);
        printf("==============================================================\n");
        printf("Input A:\n");
        auto shapea = inputa->getShape();
        inputa->print_tensor();
        printf("Input B:\n");
        auto shapeb = inputb->getShape();
        inputb->print_tensor();
#if 0
        reference_matmul(inputa, inputb, output);
#endif
        printf("Output:\n");
        output->print_tensor();
    }
};
}

TEST(MatMulTest, MatMulComprehensiveTest) {
    const std::vector<std::tuple<std::vector<int>, std::vector<int>>> test_cases = {
        {{5, 4, 1}, {5, 1, 6}},
        {{3, 4, 5}, {3, 5, 6}},
        {{3, 8, 16}, {3, 16, 32}},
        {{3, 15, 15}, {3, 15, 15}},
        {{3, 16, 16}, {3, 16, 16}},
    };
    for (const auto &test_case : test_cases) {
        auto [t1, t2] = test_case;
        MatMulTest<float> mmtest(t1, t2);
        EXPECT_TRUE(mmtest.run_test({mmtest.inputa, mmtest.inputb}, {mmtest.output}));

        MatMulTest<uint16_t> mmtest1(t1, t2);
        EXPECT_TRUE(mmtest1.run_test({mmtest1.inputa, mmtest1.inputb}, {mmtest1.output}));
    }
}

namespace {
// int8 weight-only MatMul (buffer backend, where the int8 kernel lives). B is
// the quantized weight and its per-output-column fp32 scale (amax/127 reduced
// over K) is the LAST input — [A, B_int8, scale], the convention the optimizer
// appends. transB picks the layout ([K, N] when 0, [N, K] when 1) while the
// scale always holds N entries.
//
// A batched call needs a batched B: the kernel offsets B by batch index *
// K * N, so a rank-2 B under a rank-3 A would read past the matrix. That is
// how the converter delivers it too (the broadcast Expand is materialized),
// which is why batched_b repeats one quantized matrix — the per-column scale
// is then the same for every slice by construction.
//
// The reference is computed from the DEQUANTIZED weight, so this pins the byte
// unpack and the fold-the-scale-out-of-the-K-loop, not quantization noise.
template <typename T>
class MatMulInt8Test : public TestCase<T> {
public:
    std::unordered_map<std::string, std::string> attr;
    std::shared_ptr<Tensor<T>> inputa;
    std::shared_ptr<Tensor<int8_t>> weight;
    std::shared_ptr<Tensor<float>> scale_data;
    std::shared_ptr<Tensor<T>> output;

    MatMulInt8Test(int batch, int m, int k, int n, bool transB, bool batched_b)
        : TestCase<T>("MatMul"), batch_(batch), m_(m), k_(k), n_(n),
          transB_(transB), batched_b_(batched_b) {
        attr = {{"transB", transB_ ? "1" : "0"}};
        initTestData();
    }

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
                LOG_ERROR("int8 MatMul NaN at %d, expected %f", i, ev);
                return false;
            }
            float threshold = std::max(0.05F, std::abs(ev) * 0.05F);
            if (std::abs(ov - ev) > threshold) {
                LOG_ERROR("int8 MatMul Fail (%d): %f vs %f (thr %f)", i, ov, ev,
                          threshold);
                return false;
            }
        }
        return true;
    }

  private:
    int batch_, m_, k_, n_;
    bool transB_, batched_b_;

    static float to_float(T v) {
        if constexpr (std::is_same_v<T, uint16_t>) {
            return vkop::core::ITensor::fp16_to_fp32(v);
        } else {
            return v;
        }
    }

    static torch::Tensor store(const torch::Tensor &t) {
        if constexpr (std::is_same_v<T, uint16_t>) {
            return t.to(torch::kFloat16);
        } else {
            return t;
        }
    }

    void initTestData() {
        auto f32 = torch::TensorOptions().dtype(torch::kFloat32);
        torch::manual_seed(42);

        std::vector<int64_t> a_shape = shape({batch_, m_, k_});
        auto a = store(torch::randn(a_shape, f32));

        auto w_src = torch::randn(
            transB_ ? std::vector<int64_t>{n_, k_}
                    : std::vector<int64_t>{k_, n_},
            f32);
        const int reduce_axis = transB_ ? 1 : 0;
        auto amax = std::get<0>(w_src.abs().max(reduce_axis, true));
        auto scale = amax / 127.0;
        scale = torch::where(scale == 0, torch::ones_like(scale), scale);
        auto q = torch::round(w_src / scale).to(torch::kInt8);
        auto deq = q.to(torch::kFloat32) * scale.to(torch::kFloat32);
        auto b = transB_ ? deq.t() : deq; // [K, N]

        auto y = torch::matmul(a.to(torch::kFloat32), b);

        inputa = std::make_shared<Tensor<T>>(to_ints(a_shape));
        this->fillTensorFromTorch(inputa, a);

        auto cpu_q = q.cpu().contiguous().flatten();
        auto *qptr = cpu_q.data_ptr<int8_t>();
        std::vector<int8_t> bytes(qptr, qptr + cpu_q.numel());
        if (batched_b_) {
            const size_t one = bytes.size();
            bytes.resize(one * static_cast<size_t>(batch_));
            for (int i = 1; i < batch_; i++) {
                std::copy_n(bytes.begin(), one, bytes.begin() + i * one);
            }
        }
        std::vector<int> w_ints =
            to_ints(shape(batched_b_ ? std::vector<int64_t>{batch_, transB_ ? n_ : k_,
                                                             transB_ ? k_ : n_}
                                     : std::vector<int64_t>{transB_ ? n_ : k_,
                                                            transB_ ? k_ : n_}));
        weight = std::make_shared<Tensor<int8_t>>(w_ints);
        weight->fillToCPU(bytes);

        scale_data = std::make_shared<Tensor<float>>(std::vector<int>{n_});
        auto cpu_scale = scale.cpu().contiguous().flatten();
        auto sacc = cpu_scale.accessor<float, 1>();
        std::vector<float> svec;
        svec.reserve(cpu_scale.numel());
        for (int64_t i = 0; i < cpu_scale.numel(); i++) svec.push_back(sacc[i]);
        scale_data->fillToCPU(svec);

        output = std::make_shared<Tensor<T>>(to_ints(y.sizes().vec()));
        this->fillTensorFromTorch(output, store(y));

        LOG_INFO("int8 MatMul batch %d, M %d, N %d, K %d, transB %d, "
                 "batched B %d, fp16 %d",
                 batch_, m_, n_, k_, transB_ ? 1 : 0, batched_b_ ? 1 : 0,
                 std::is_same_v<T, uint16_t> ? 1 : 0);
    }

    // Drop a leading batch of 1 so the un-batched cases stay plain 2-D.
    static std::vector<int64_t> shape(const std::vector<int64_t> &s) {
        if (s.size() == 3 && s[0] == 1) {
            return {s[1], s[2]};
        }
        return s;
    }

    static std::vector<int> to_ints(const std::vector<int64_t> &s) {
        return std::vector<int>(s.begin(), s.end());
    }
};

template <typename T>
void run_matmul_int8(
    const std::vector<std::tuple<int, int, int, int, bool, bool>> &cases) {
    for (const auto &tc : cases) {
        auto [batch, m, k, n, transB, batched_b] = tc;
        MatMulInt8Test<T> t(batch, m, k, n, transB, batched_b);
        EXPECT_TRUE(t.run_test({t.inputa, t.weight, t.scale_data}, {t.output},
            [&t](std::unique_ptr<vkop::ops::Operator> &op) {
                op->setAttribute(t.attr);
            }));
    }
}
} // namespace

TEST(MatMulTest, MatMulSplitKGemvBuffer) {
    vkop::tests::ScopedBufferBackend buffer;
    // Decode-shaped calls (one A row, long K), where the kernel splits K across
    // the workgroup lanes the single row leaves idle. The cases are chosen to
    // cover the slicing edges rather than more sizes: K a multiple of 16 slices,
    // K that is not (the tail slices go empty), an N whose last 16-quad block is
    // partial, more than one output row, and a broadcast batch.
    const std::vector<std::tuple<std::vector<int>, std::vector<int>>> cases = {
        {{1, 1, 512}, {1, 512, 256}},
        {{1, 1, 258}, {1, 258, 260}},
        {{1, 8, 260}, {1, 260, 64}},
        {{2, 1, 300}, {2, 300, 128}},
    };
    for (const auto &test_case : cases) {
        auto [t1, t2] = test_case;
        MatMulTest<uint16_t> mmtest(t1, t2);
        EXPECT_TRUE(mmtest.run_test({mmtest.inputa, mmtest.inputb},
                                    {mmtest.output}));
    }
}

TEST(MatMulTest, MatMulInt8WeightOnlyBuffer) {
    vkop::tests::ScopedBufferBackend buffer;
    const std::vector<std::tuple<int, int, int, int, bool, bool>> cases = {
        {1, 4, 32, 16, false, false},  // even N, plain 2-D weight
        {1, 8, 20, 7, false, false},   // odd N: fp32 direct store, fp16
                                        // reduce->pack
        {1, 1, 64, 12, true, false},   // transB=1 GEMV over [N, K]
        {1, 2, 22, 12, true, false},   // even N, K % 4 != 0: the word-aligned
                                        // 4-taps-per-load nest does not apply
        {1, 5, 10, 14, false, false},  // even N, transB=0: a column's bytes are
                                       // N apart, so one tap per extract
        {2, 3, 24, 9, false, true},    // broadcast weight materialized, odd N
        {2, 4, 16, 6, true, true},     // transB=1 with a batched weight
        // Decode shapes: one A row over a long K, which the host routes to the
        // split-K GEMV (K sliced across the workgroup's idle lanes). K divisible
        // and not divisible by the slice length, a partial last 16-quad block,
        // and more than one row all have to stay exact.
        {1, 1, 512, 256, false, false},
        {1, 1, 258, 260, false, false},
        {1, 3, 258, 32, false, false},
        {2, 1, 300, 128, false, true},
    };
    LOG_INFO("int8 MatMul, FP32");
    run_matmul_int8<float>(cases);
    LOG_INFO("int8 MatMul, FP16");
    run_matmul_int8<uint16_t>(cases);
}

// A scale quantized along K instead of N is the failure the converter gate is
// meant to prevent, but a graph that got it wrong must not be computed with:
// the kernel indexes uScale by output column, so 32 entries for a 16-column
// output silently multiplies every column by some other column's step. The
// host has to refuse the call.
TEST(MatMulTest, MatMulInt8WrongAxisScaleThrows) {
    vkop::tests::ScopedBufferBackend buffer;
    MatMulInt8Test<float> t(/*batch=*/1, /*m=*/4, /*k=*/32, /*n=*/16,
                            /*transB=*/false, /*batched_b=*/false);
    auto wrong = std::make_shared<Tensor<float>>(std::vector<int>{32});
    wrong->fillToCPU(std::vector<float>(32, 1.0F));
    EXPECT_THROW(t.run_test({t.inputa, t.weight, wrong}, {t.output},
                            [&t](std::unique_ptr<vkop::ops::Operator> &op) {
                                op->setAttribute(t.attr);
                            }),
                 std::runtime_error);
}