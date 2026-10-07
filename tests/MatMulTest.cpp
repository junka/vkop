#include <algorithm>
#include <cstring>
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

// fp8 weight-only MatMul (buffer backend): the same multi-input convention as
// int8 — [A, B_fp8, scale], one fp32 scale per output column — but each byte
// holds sign | exponent | mantissa instead of a two's-complement value, because
// MoltenVK has no shader-float8 extension and the kernel unpacks it with bit
// math (f8_val in shaders/buffer/matmul.comp). The scale is amax/max_finite
// (448 for E4M3, 57344 for E5M2), so a column's largest weight lands on the
// layout's largest finite code.
//
// The bytes come from libtorch's own Float8 cast, which knows nothing about that
// bit math or about the converter's encoder. Three independent implementations
// of one grid agreeing is the point: if the kernel decoded a different grid than
// the one the writer emits, the product here would be wrong.
//
// Every byte path int8 has, fp8 rides — which is the whole design — so the case
// list below mirrors the int8 one shape for shape.
template <typename T>
class MatMulFp8Test : public TestCase<T> {
  public:
    std::unordered_map<std::string, std::string> attr;
    std::shared_ptr<Tensor<T>> inputa;
    std::shared_ptr<Tensor<int8_t>> weight;
    std::shared_ptr<Tensor<float>> scale_data;
    std::shared_ptr<Tensor<T>> output;

    // Payload coverage, filled in by initTestData.
    int subnormal_codes_ = 0;
    int zero_codes_ = 0;

    MatMulFp8Test(int batch, int m, int k, int n, bool e5m2, bool transB,
                  bool batched_b, bool wide = false)
        : TestCase<T>("MatMul"), batch_(batch), m_(m), k_(k), n_(n),
          e5m2_(e5m2), transB_(transB), batched_b_(batched_b), wide_(wide) {
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
                LOG_ERROR("fp8 MatMul NaN at %d, expected %f", i, ev);
                return false;
            }
            float threshold = std::max(0.05F, std::abs(ev) * 0.05F);
            if (std::abs(ov - ev) > threshold) {
                LOG_ERROR("fp8 MatMul Fail (%d): %f vs %f (thr %f)", i, ov, ev,
                          threshold);
                return false;
            }
        }
        return true;
    }

  private:
    int batch_, m_, k_, n_;
    bool e5m2_, transB_, batched_b_, wide_;

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

        const auto fp8 = e5m2_ ? c10::ScalarType::Float8_e5m2
                               : c10::ScalarType::Float8_e4m3fn;
        const float max_finite = e5m2_ ? 57344.0F : 448.0F;

        std::vector<int64_t> a_shape = shape({batch_, m_, k_});
        auto a = store(torch::randn(a_shape, f32));

        auto w_src = torch::randn(
            transB_ ? std::vector<int64_t>{n_, k_}
                    : std::vector<int64_t>{k_, n_},
            f32);
        if (wide_) {
            // A spread wide enough, in one column, for the column's absolute
            // scale to push a real share of its weights under the layout's
            // smallest normal (f8_val's subnormal arm) and under half its step
            // (the zero byte). Neither is reachable on randn weights. E5M2 needs
            // more decades than E4M3 because its range is 57344-wide but its
            // smallest normal is 2^-14, so its subnormal band starts a further
            // 10^4 below the top.
            const double spread = e5m2_ ? 14.0 : 8.0;
            w_src = w_src * torch::pow(10.0,
                                       torch::rand(w_src.sizes().vec(), f32) *
                                           -spread);
        }
        // N entries either way; transB only says which physical axis of the
        // stored matrix N runs along.
        auto amax = std::get<0>(w_src.abs().max(transB_ ? 1 : 0, false));
        auto scale =
            torch::where(amax == 0, torch::ones_like(amax), amax / max_finite);
        auto bcast = scale.reshape(transB_ ? std::vector<int64_t>{n_, 1}
                                           : std::vector<int64_t>{1, n_});
        // The clamp is the converter's: past the largest finite code the only
        // encodings left are Inf/NaN, which no kernel decodes.
        auto q8 = (w_src / bcast).clamp(-max_finite, max_finite).to(fp8);
        auto deq = q8.to(torch::kFloat32) * bcast;
        auto b = transB_ ? deq.t() : deq; // [K, N]
        auto y = torch::matmul(a.to(torch::kFloat32), b);

        // How much of the payload sits in the layout's subnormal band (a nonzero
        // magnitude below its smallest normal) or below it (a zero byte). Counted
        // on the normalized values rather than the bytes, and recorded, so the
        // wide cases can prove they reach those two arms of f8_val instead of
        // assuming a particular magnitude distribution did.
        {
            const float min_normal = e5m2_ ? 1.0F / 16384.0F : 1.0F / 64.0F;
            auto norm = (deq / bcast).abs();
            subnormal_codes_ = static_cast<int>(
                norm.lt(min_normal).logical_and(norm.gt(0)).sum().item<int64_t>());
            zero_codes_ = static_cast<int>(norm.eq(0).sum().item<int64_t>());
        }

        inputa = std::make_shared<Tensor<T>>(to_ints(a_shape));
        this->fillTensorFromTorch(inputa, a);

        // The fp8 byte pattern is the payload; reinterpreting it as int8 is how
        // a host with no fp8 container moves it.
        auto cpu_bytes =
            q8.view(c10::ScalarType::Char).cpu().contiguous().flatten();
        auto *bptr = cpu_bytes.data_ptr<int8_t>();
        std::vector<int8_t> bytes(bptr, bptr + cpu_bytes.numel());
        if (batched_b_) {
            const size_t one = bytes.size();
            bytes.resize(one * static_cast<size_t>(batch_));
            for (int i = 1; i < batch_; i++) {
                std::copy_n(bytes.begin(), one, bytes.begin() + i * one);
            }
        }
        std::vector<int> w_ints = to_ints(shape(
            batched_b_
                ? std::vector<int64_t>{batch_, transB_ ? n_ : k_,
                                       transB_ ? k_ : n_}
                : std::vector<int64_t>{transB_ ? n_ : k_, transB_ ? k_ : n_}));
        weight = std::make_shared<Tensor<int8_t>>(w_ints);
        // int8_t is the container, not the element type: the kernel has to be
        // told which of the two byte grids to decode against.
        weight->set_elem_kind(
            e5m2_ ? vkop::core::ElemKind::kFloat8E5M2
                  : vkop::core::ElemKind::kFloat8E4M3FN);
        weight->fillToCPU(bytes);

        scale_data = std::make_shared<Tensor<float>>(std::vector<int>{n_});
        auto cpu_scale = scale.cpu().contiguous().flatten();
        auto sacc = cpu_scale.accessor<float, 1>();
        std::vector<float> svec;
        svec.reserve(static_cast<size_t>(cpu_scale.numel()));
        for (int64_t i = 0; i < cpu_scale.numel(); i++)
            svec.push_back(sacc[i]);
        scale_data->fillToCPU(svec);

        output = std::make_shared<Tensor<T>>(to_ints(y.sizes().vec()));
        this->fillTensorFromTorch(output, store(y));

        LOG_INFO("fp8 (%s) MatMul batch %d, M %d, N %d, K %d, transB %d, "
                 "batched B %d, wide %d, fp16 %d, %d subnormal / %d zero bytes",
                 e5m2_ ? "e5m2" : "e4m3", batch_, m_, n_, k_, transB_ ? 1 : 0,
                 batched_b_ ? 1 : 0, wide_ ? 1 : 0,
                 std::is_same_v<T, uint16_t> ? 1 : 0, subnormal_codes_,
                 zero_codes_);
    }

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
void run_matmul_fp8(
    const std::vector<std::tuple<int, int, int, int, bool, bool, bool>> &cases) {
    for (const auto &tc : cases) {
        auto [batch, m, k, n, transB, batched_b, wide] = tc;
        for (const bool e5m2 : {false, true}) {
            MatMulFp8Test<T> t(batch, m, k, n, e5m2, transB, batched_b, wide);
            if (wide) {
                // A wide case that quantized to nothing but normals would test
                // the same arm as an ordinary one and leave the claim in its
                // comment unbacked.
                EXPECT_GT(t.subnormal_codes_, 0);
                EXPECT_GT(t.zero_codes_, 0);
            }
            EXPECT_TRUE(t.run_test({t.inputa, t.weight, t.scale_data},
                                   {t.output},
                                   [&t](
                                       std::unique_ptr<vkop::ops::Operator> &op) {
                                       op->setAttribute(t.attr);
                                   }));
        }
    }
}

// The NF4 codebook the shader carries: 16 quantiles of a unit normal, extreme
// codes exactly -1 and +1. Duplicated here on purpose — if the test used the
// shader's table the comparison would be vacuous.
static const float kNf4Codebook[16] = {
    -1.0f, -0.6961928009986877f, -0.5250730514526367f, -0.3949174189567566f,
    -0.2844413814544678f, -0.1847814998626709f, -0.0910967006323811f, 0.0f,
    0.0795802986717224f, 0.1601973110246658f, 0.2447470557689667f,
    0.3361703515052795f, 0.4407098295211792f, 0.5626170039176941f,
    0.7229568369388580f, 1.0f};

// 4-bit weight-only MatMul (buffer backend). B is a nibble-packed weight: two
// values per byte, the even element in the LOW nibble, so element i of the
// row-major matrix lives in byte i/2 at nibble i%2. The LAST input is the fp32
// dequant scale table laid out [K/group, N] — one absmax per (K group, output
// column) — which is the multi-input convention the optimizer appends:
//   int4 : [A, B_int4, scale]   nibble = signed value, scale = amax/7
//   nf4  : [A, B_nf4,  scale]   nibble = codebook index, scale = amax
// A group's taps are summed before its scale is applied, so the reference
// dequantizes the same way instead of comparing against the original weight.
template <typename T>
class MatMulW4Test : public TestCase<T> {
  public:
    std::unordered_map<std::string, std::string> attr;
    std::shared_ptr<Tensor<T>> inputa;
    std::shared_ptr<Tensor<int8_t>> weight;
    std::shared_ptr<Tensor<float>> scale_data;
    std::shared_ptr<Tensor<T>> output;

    MatMulW4Test(int batch, int m, int k, int n, int group, bool nf4,
                 bool batched_b = false, bool transB = false)
        : TestCase<T>("MatMul"), batch_(batch), m_(m), k_(k), n_(n),
          group_(group), nf4_(nf4), batched_b_(batched_b), transB_(transB) {
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
                LOG_ERROR("4bit MatMul NaN at %d, expected %f", i, ev);
                return false;
            }
            float threshold = std::max(0.05F, std::abs(ev) * 0.05F);
            if (std::abs(ov - ev) > threshold) {
                LOG_ERROR("4bit MatMul Fail (%d): %f vs %f (thr %f)", i, ov, ev,
                          threshold);
                return false;
            }
        }
        return true;
    }

  private:
    int batch_, m_, k_, n_, group_;
    bool nf4_, batched_b_, transB_;

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
        auto i64 = torch::TensorOptions().dtype(torch::kInt64);
        torch::manual_seed(42);

        const int n_groups = k_ / group_;
        std::vector<int64_t> a_shape = shape({batch_, m_, k_});
        auto a = store(torch::randn(a_shape, f32));

        // Quantize on the [K, N] view: the kernel's unpack is defined for that
        // layout only (a transB case below exists to be refused).
        auto w_src = torch::randn({k_, n_}, f32);
        auto w3 = w_src.reshape({n_groups, group_, n_});
        auto amax = std::get<0>(w3.abs().max(1, true)); // [n_groups, 1, N]
        torch::Tensor codes, deq3;
        torch::Tensor scale = amax;
        if (nf4_) {
            // Nearest codebook entry to the group-normalized value; the scale
            // the kernel multiplies by is the group absmax itself.
            auto cb = torch::from_blob(const_cast<float *>(kNf4Codebook), {16},
                                       f32)
                          .clone();
            auto norm = w3 / amax;
            codes = (norm.unsqueeze(-1) - cb.reshape({1, 1, 1, 16}))
                        .abs()
                        .argmin(-1);
            deq3 = cb.index_select(0, codes.reshape({-1}))
                       .reshape({n_groups, group_, n_}) *
                   amax;
        } else {
            scale = amax / 7.0; // symmetric int4 reaches +7; -8 stays unused
            codes = (w3 / scale).round().clamp(-8, 7).to(torch::kInt64);
            deq3 = codes.to(torch::kFloat32) * scale;
        }
        auto deq = deq3.reshape({k_, n_});
        auto y = torch::matmul(a.to(torch::kFloat32), deq);

        inputa = std::make_shared<Tensor<T>>(to_ints(a_shape));
        this->fillTensorFromTorch(inputa, a);

        // Pack the CODES nibble-wise: element i in byte i/2, low nibble for
        // even i. A batched weight repeats the same matrix, so its bytes repeat.
        auto flat_codes = codes.reshape({-1}).cpu();
        auto cacc = flat_codes.accessor<int64_t, 1>();
        const int64_t values = flat_codes.numel();
        std::vector<int8_t> bytes(static_cast<size_t>((values + 1) / 2), 0);
        for (int64_t i = 0; i < values; i += 2) {
            const uint32_t lo = static_cast<uint32_t>(cacc[i]) & 0xFu;
            const uint32_t hi = i + 1 < values
                                    ? (static_cast<uint32_t>(cacc[i + 1]) & 0xFu)
                                    : 0u;
            bytes[static_cast<size_t>(i / 2)] =
                static_cast<int8_t>(lo | (hi << 4));
        }
        std::vector<int8_t> wbytes = bytes;
        if (batched_b_) {
            for (int i = 1; i < batch_; i++)
                wbytes.insert(wbytes.end(), bytes.begin(), bytes.end());
        }
        std::vector<int> w_ints = to_ints(shape(
            batched_b_ ? std::vector<int64_t>{batch_, k_, n_}
                       : std::vector<int64_t>{k_, n_}));
        weight = std::make_shared<Tensor<int8_t>>(w_ints);
        weight->set_elem_kind(nf4_ ? vkop::core::ElemKind::kNF4
                                   : vkop::core::ElemKind::kInt4);
        // Two values per byte: the payload is half the element count, while the
        // dims keep describing the logical matrix (what the kernel indexes).
        weight->set_payload_bytes(static_cast<int>(wbytes.size()));
        weight->fillToCPU(wbytes);

        // A broadcast batch shares one weight, so one [n_groups, N] table serves
        // every slice — it is NOT repeated per batch.
        scale_data = std::make_shared<Tensor<float>>(
            std::vector<int>{n_groups * n_});
        auto cpu_scale = scale.reshape({-1}).cpu().contiguous();
        auto sacc = cpu_scale.accessor<float, 1>();
        std::vector<float> svec;
        svec.reserve(static_cast<size_t>(cpu_scale.numel()));
        for (int64_t i = 0; i < cpu_scale.numel(); i++)
            svec.push_back(sacc[i]);
        scale_data->fillToCPU(svec);

        output = std::make_shared<Tensor<T>>(to_ints(y.sizes().vec()));
        this->fillTensorFromTorch(output, store(y));

        LOG_INFO("4bit MatMul %s batch %d, M %d, N %d, K %d, group %d, "
                 "batched B %d, transB %d, fp16 %d",
                 nf4_ ? "nf4" : "int4", batch_, m_, n_, k_, group_,
                 batched_b_ ? 1 : 0, transB_ ? 1 : 0,
                 std::is_same_v<T, uint16_t> ? 1 : 0);
    }

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

// Cases are picked for the edges, not for coverage of sizes: a group count
// below / equal to / above the 16 K slices the split-K GEMV hands out, a last
// 16-quad block that is partial, a single group (the table degenerates to one
// scale per column), more than one A row, and a materialized broadcast batch.
template <typename T>
void run_matmul_w4(
    const std::vector<std::tuple<int, int, int, int, int, bool>> &cases) {
    for (const auto &tc : cases) {
        auto [batch, m, k, n, group, batched_b] = tc;
        for (const bool nf4 : {false, true}) {
            MatMulW4Test<T> t(batch, m, k, n, group, nf4, batched_b);
            EXPECT_TRUE(t.run_test(
                {t.inputa, t.weight, t.scale_data}, {t.output},
                [&t](std::unique_ptr<vkop::ops::Operator> &op) {
                    op->setAttribute(t.attr);
                }));
        }
    }
}

// NVFP4's nibble, decoded the way the buffer kernel does it: the magnitude
// field's exponent and mantissa slid into an fp32's, with only exponent 0 on the
// subnormal term. Restated from the shader's expression rather than read from a
// grid table, so this checks the kernel's arithmetic instead of agreeing with it
// twice.
static float e2m1_bits(uint32_t c) {
    const uint32_t m = c & 7u;
    float v;
    if (m > 1u) {
        const uint32_t bits = (((m + 252u) >> 1u) << 23u) | ((m & 1u) << 22u);
        std::memcpy(&v, &bits, sizeof(v));
    } else {
        v = static_cast<float>(m) * 0.5F;
    }
    return (c & 8u) != 0u ? -v : v;
}

// The same nibble's encoding, from libm's round-to-nearest-even instead of
// numpy's: exponent from repeated halving, one mantissa bit rounded out of the
// [1,2) significand, a carry bumping the exponent and clearing it. Below 1.0 the
// grid is the subnormal step 0.5, and a carry there lands on the smallest normal
// on its own (2 * 0.5 IS 1.0), so no extra case.
static uint32_t e2m1_encode(float v) {
    const uint32_t sign = std::signbit(v) ? 8u : 0u;
    const float a = std::fabs(v);
    if (a < 1.0F) return sign | static_cast<uint32_t>(std::nearbyint(a * 2.0F));
    int e = 0;
    float q = a;
    while (q >= 2.0F) {
        q *= 0.5F;
        ++e;
    }
    float mb = std::nearbyint((q - 1.0F) * 2.0F);
    if (mb > 1.5F) {
        mb = 0.0F;
        ++e;
    }
    return sign | (static_cast<uint32_t>(e + 1) << 1) |
           static_cast<uint32_t>(mb);
}

// 4-bit weight-only MatMul with NVFP4's two scale levels (buffer backend). B is
// an E2M1 nibble payload like int4's, but the dequant table is one fp8 E4M3 byte
// per (16-value block of K, output column) and the tensor has one fp32 factor on
// top, so the graph carries FOUR inputs:
//   nvfp4 : [A, B_4bit, block_scale(n_blocks * N, fp8), global_scale(1, fp32)]
// The factor is the absmax of the whole weight over 448 * 6, which puts the
// largest block's scale exactly on e4m3's top byte — the reason one byte can
// cover a block at all. A block's taps are summed before its scale, so the
// reference dequantizes the same way.
template <typename T>
class MatMulNvfp4Test : public TestCase<T> {
  public:
    std::unordered_map<std::string, std::string> attr;
    std::shared_ptr<Tensor<T>> inputa;
    std::shared_ptr<Tensor<int8_t>> weight;
    std::shared_ptr<Tensor<int8_t>> block_scale;
    std::shared_ptr<Tensor<float>> global_scale;
    std::shared_ptr<Tensor<T>> output;
    // How many codes the block-scale rounding pushed past the grid's top, and how
    // many block scales landed in e4m3's subnormal band or on zero, so a case can
    // prove it exercised those arms rather than assuming a distribution did.
    int saturated_ = 0;
    int subnormal_scales_ = 0;
    int dead_blocks_ = 0;

    MatMulNvfp4Test(int batch, int m, int k, int n, bool batched_b = false,
                    bool wide = false)
        : TestCase<T>("MatMul"), batch_(batch), m_(m), k_(k), n_(n),
          batched_b_(batched_b), wide_(wide) {
        attr = {{"transB", "0"}};
        initTestData();
    }

    bool verify_output(const std::unique_ptr<vkop::ops::Operator> &op, int idx,
                       const std::shared_ptr<vkop::core::ITensor> &output,
                       const std::shared_ptr<vkop::core::ITensor> &expect)
        override {
        auto out = vkop::core::as_tensor<T>(output);
        auto exp = vkop::core::as_tensor<T>(expect);
        for (int i = 0; i < out->num_elements(); i++) {
            const float ov = to_float((*out)[i]);
            const float ev = to_float((*exp)[i]);
            if (std::isnan(ov)) {
                LOG_ERROR("nvfp4 MatMul NaN at %d, expected %f", i, ev);
                return false;
            }
            const float threshold = std::max(0.05F, std::abs(ev) * 0.05F);
            if (std::abs(ov - ev) > threshold) {
                LOG_ERROR("nvfp4 MatMul Fail (%d): %f vs %f (thr %f)", i, ov, ev,
                          threshold);
                return false;
            }
        }
        return true;
    }

  private:
    static constexpr int kBlock = 16;

    int batch_, m_, k_, n_;
    bool batched_b_, wide_;

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

        const int n_blocks = k_ / kBlock;
        std::vector<int64_t> a_shape = shape({batch_, m_, k_});
        auto a = store(torch::randn(a_shape, f32));

        auto w_src = torch::randn({k_, n_}, f32);
        if (wide_) {
            // A spread of decades ACROSS the K blocks of a column, so the
            // tensor-wide factor leaves some blocks with a scale below e4m3's
            // smallest normal and some with none at all. Neither arm is reachable
            // on randn weights, and the second one is the format's own way of
            // erasing a block, so it has to be exercised and counted rather than
            // assumed away.
            const int n_blocks_spread = k_ / kBlock;
            w_src = w_src.reshape({n_blocks_spread, kBlock, n_}) *
                    torch::pow(10.0, torch::rand({n_blocks_spread, 1, n_}, f32) *
                                            -7.0)
                        .reshape({n_blocks_spread, 1, n_});
            w_src = w_src.reshape({k_, n_});
        }
        auto w3 = w_src.reshape({n_blocks, kBlock, n_});
        const float amax_t = w_src.abs().max().item<float>();
        const float g = amax_t / (448.0F * 6.0F);
        auto amax_b = std::get<0>(w3.abs().max(1, true)); // [n_blocks, 1, N]
        // The block scale as the format stores it: e4m3 rounded, so torch's own
        // cast is the oracle for the converter's hand-written encoder.
        auto s_real = amax_b / (6.0F * g);
        auto s8 = s_real.to(c10::ScalarType::Float8_e4m3fn);
        auto s_eff = s8.to(torch::kFloat32) * g; // [n_blocks, 1, N]
        // A block scale that rounds to zero erases its block — the format's own
        // answer, which the wide cases below reach. Dividing by it would be a
        // nan, so encode against a stand-in: the dequantized product is zero
        // either way, and that is what both the kernel and this oracle give.
        auto divide_by = torch::where(s_eff == 0, torch::ones_like(s_eff), s_eff);
        {
            // What the two arms of the format's own scale grid did, counted here
            // so a wide case can assert it reached them: e4m3's smallest normal is
            // 2^-6, below it a scale is a multiple of 2^-9, and under half that
            // the byte is zero and the block it covers decodes to nothing.
            auto sv = s8.to(torch::kFloat32).reshape({-1}).cpu();
            auto av = amax_b.reshape({-1}).cpu();
            auto *sp = sv.data_ptr<float>();
            auto *ap = av.data_ptr<float>();
            for (int64_t i = 0; i < sv.numel(); i++) {
                const float mag = std::fabs(sp[i]);
                if (mag > 0.0F && mag < 1.0F / 64.0F)
                    subnormal_scales_++;
                if (mag == 0.0F && ap[i] > 0.0F)
                    dead_blocks_++;
            }
        }
        auto codes3 = torch::empty(
            {n_blocks, kBlock, n_},
            torch::TensorOptions().dtype(torch::kUInt8));
        auto *cd = codes3.data_ptr<uint8_t>();
        {
            // Count on the ratio, before the clip: a block whose byte rounded down
            // leaves its own absmax above the grid's top, and that is the saturation
            // the case is meant to have exercised. A value that merely equals 6.0 is
            // on the grid and encodes without clipping.
            auto norm = w3 / divide_by;
            auto cpu_norm = norm.cpu().contiguous();
            auto acc = cpu_norm.accessor<float, 3>();
            for (int b = 0; b < n_blocks; b++)
                for (int i = 0; i < kBlock; i++)
                    for (int j = 0; j < n_; j++) {
                        const float v = acc[b][i][j];
                        if (std::fabs(v) > 6.0F)
                            saturated_++;
                        cd[(b * kBlock + i) * n_ + j] =
                            static_cast<uint8_t>(e2m1_encode(
                                std::max(-6.0F, std::min(6.0F, v))));
                    }
        }
        auto deq3 = codes3.to(torch::kFloat32);
        {
            // Decode through the kernel's bit arithmetic, not the nibble itself.
            auto *p = deq3.data_ptr<float>();
            for (int64_t i = 0; i < deq3.numel(); i++)
                p[i] = e2m1_bits(cd[i]);
        }
        deq3 = deq3 * s_eff;
        auto deq = deq3.reshape({k_, n_});
        auto y = torch::matmul(a.to(torch::kFloat32), deq);

        inputa = std::make_shared<Tensor<T>>(to_ints(a_shape));
        this->fillTensorFromTorch(inputa, a);

        // Pack the CODES nibble-wise: element i in byte i/2, low nibble for even
        // i. A batched weight repeats the same matrix, so its bytes repeat.
        const int64_t values = k_ * static_cast<int64_t>(n_);
        std::vector<int8_t> bytes(static_cast<size_t>((values + 1) / 2), 0);
        for (int64_t i = 0; i < values; i += 2) {
            const uint32_t lo = cd[i] & 0xFu;
            const uint32_t hi =
                i + 1 < values ? (cd[i + 1] & 0xFu) : 0u;
            bytes[static_cast<size_t>(i / 2)] =
                static_cast<int8_t>(lo | (hi << 4));
        }
        std::vector<int8_t> wbytes = bytes;
        if (batched_b_) {
            for (int i = 1; i < batch_; i++)
                wbytes.insert(wbytes.end(), bytes.begin(), bytes.end());
        }
        std::vector<int> w_ints = to_ints(shape(
            batched_b_ ? std::vector<int64_t>{batch_, k_, n_}
                       : std::vector<int64_t>{k_, n_}));
        weight = std::make_shared<Tensor<int8_t>>(w_ints);
        // int8_t is the container, not the element type: the kernel has to be
        // told its nibble is an E2M1 code and not an int4 value or an NF4 index.
        weight->set_elem_kind(vkop::core::ElemKind::kFloat4E2M1);
        weight->set_payload_bytes(static_cast<int>(wbytes.size()));
        weight->fillToCPU(wbytes);

        // The block scale table travels as the bytes torch just wrote, one per
        // (block, column) row-major — the same view the kernel reads four of them
        // out of one word.
        auto cpu_s8 =
            s8.view(c10::ScalarType::Char).cpu().contiguous().flatten();
        auto *sptr = cpu_s8.data_ptr<int8_t>();
        std::vector<int8_t> sbytes(sptr, sptr + cpu_s8.numel());
        block_scale = std::make_shared<Tensor<int8_t>>(
            std::vector<int>{n_blocks * n_});
        block_scale->set_elem_kind(vkop::core::ElemKind::kFloat8E4M3FN);
        block_scale->fillToCPU(sbytes);

        global_scale = std::make_shared<Tensor<float>>(std::vector<int>{1});
        global_scale->fillToCPU(std::vector<float>{g});

        output = std::make_shared<Tensor<T>>(to_ints(y.sizes().vec()));
        this->fillTensorFromTorch(output, store(y));

        LOG_INFO("nvfp4 MatMul batch %d, M %d, N %d, K %d, blocks %d, batched "
                 "B %d, wide %d, fp16 %d, %d saturated, %d subnormal scales, "
                 "%d dead blocks",
                 batch_, m_, n_, k_, n_blocks, batched_b_ ? 1 : 0,
                 wide_ ? 1 : 0, std::is_same_v<T, uint16_t> ? 1 : 0, saturated_,
                 subnormal_scales_, dead_blocks_);
    }

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
void run_matmul_nvfp4(
    const std::vector<std::tuple<int, int, int, int, bool, bool>> &cases) {
    for (const auto &tc : cases) {
        auto [batch, m, k, n, batched_b, wide] = tc;
        MatMulNvfp4Test<T> t(batch, m, k, n, batched_b, wide);
        if (wide) {
            // A wide case that reached neither arm would test the same thing an
            // ordinary one does and leave its comment's claim unbacked.
            EXPECT_GT(t.subnormal_scales_, 0);
            EXPECT_GT(t.dead_blocks_, 0);
        }
        EXPECT_GT(t.saturated_, 0) << "no value saturated, so the grid's top "
                                      "arm was never read";
        EXPECT_TRUE(t.run_test(
            {t.inputa, t.weight, t.block_scale, t.global_scale}, {t.output},
            [&t](std::unique_ptr<vkop::ops::Operator> &op) {
                op->setAttribute(t.attr);
            }));
    }
}
} // namespace

TEST(MatMulTest, MatMulInt4Nf4WeightOnlyBuffer) {
    vkop::tests::ScopedBufferBackend buffer;
    const std::vector<std::tuple<int, int, int, int, int, bool>> cases = {
        {1, 4, 32, 16, 8, false},    // 4 groups, plain quad path
        {1, 1, 256, 64, 64, false},  // decode: 4 groups over 16 slices
        {1, 1, 256, 64, 16, false},  // decode: exactly 16 groups
        {1, 1, 256, 64, 8, false},   // decode: 32 groups, 2 per slice
        {1, 1, 320, 104, 32, false}, // partial last 16-quad block
        {1, 3, 96, 24, 24, false},   // a few rows, 4 groups
        {2, 1, 128, 32, 128, true},  // one group: per-column scales, batched A
        {1, 16, 64, 16, 64, false},  // prefill-shaped M
        {2, 4, 64, 16, 32, true},    // materialized broadcast weight
    };
    LOG_INFO("4bit MatMul, FP32");
    run_matmul_w4<float>(cases);
    LOG_INFO("4bit MatMul, FP16");
    run_matmul_w4<uint16_t>(cases);
}

// The contracts the converter is supposed to guarantee, checked here so a graph
// that breaks one fails instead of computing a wrong answer: a packed row must
// start on a word (N % 8), a group must be walkable in A's half2 pairs (even),
// the table must be a whole number of N-rows whose count divides K, and only
// the [K, N] layout is unpacked.
TEST(MatMulTest, MatMulInt4ContractThrows) {
    vkop::tests::ScopedBufferBackend buffer;
    {
        MatMulW4Test<float> t(1, 4, 32, 12, 8, false); // N % 8 != 0
        EXPECT_THROW(t.run_test({t.inputa, t.weight, t.scale_data}, {t.output},
                                [&t](std::unique_ptr<vkop::ops::Operator> &op) {
                                    op->setAttribute(t.attr);
                                }),
                     std::runtime_error);
    }
    {
        MatMulW4Test<float> t(1, 4, 40, 16, 5, false); // odd group
        EXPECT_THROW(t.run_test({t.inputa, t.weight, t.scale_data}, {t.output},
                                [&t](std::unique_ptr<vkop::ops::Operator> &op) {
                                    op->setAttribute(t.attr);
                                }),
                     std::runtime_error);
    }
    {
        // transB: the same bytes read as [N, K] would need different
        // addressing, so the host refuses rather than unpacking a transpose.
        MatMulW4Test<float> t(1, 4, 32, 16, 8, false, false, true);
        EXPECT_THROW(t.run_test({t.inputa, t.weight, t.scale_data}, {t.output},
                                [&t](std::unique_ptr<vkop::ops::Operator> &op) {
                                    op->setAttribute(t.attr);
                                }),
                     std::runtime_error);
    }
    {
        // A scale row count that does not divide K: 24 entries over 16 columns
        // says 1.5 groups, which no grouping of K could have produced.
        MatMulW4Test<float> t(1, 4, 32, 16, 8, false);
        auto wrong = std::make_shared<Tensor<float>>(std::vector<int>{24});
        wrong->fillToCPU(std::vector<float>(24, 1.0F));
        EXPECT_THROW(t.run_test({t.inputa, t.weight, wrong}, {t.output},
                                [&t](std::unique_ptr<vkop::ops::Operator> &op) {
                                    op->setAttribute(t.attr);
                                }),
                     std::runtime_error);
    }
}

TEST(MatMulTest, MatMulNvfp4WeightOnlyBuffer) {
    vkop::tests::ScopedBufferBackend buffer;
    // The block is fixed at 16, so the edges are the same ones the split-K GEMV
    // and the quad kernel care about: K below / at / above the 16 slices a
    // workgroup hands out, a partial last 16-quad block, more than one A row, a
    // materialized broadcast batch, and a weight spread across decades so the
    // fp8 scale table reaches its own subnormal and zero arms.
    const std::vector<std::tuple<int, int, int, int, bool, bool>> cases = {
        {1, 4, 32, 16, false, false},    // 2 blocks, prefill-shaped M
        {1, 1, 128, 64, false, false},   // decode: 8 blocks over 16 slices
        {1, 1, 256, 64, false, false},   // decode: exactly 16 blocks
        {1, 1, 512, 64, false, false},   // decode: 32 blocks, 2 per slice
        {1, 1, 320, 104, false, false},  // partial last 16-quad block
        {1, 3, 96, 24, false, false},    // 6 blocks, N not a multiple of 16
        {2, 1, 128, 32, true, false},    // broadcast batch
        {1, 16, 64, 16, false, false},   // 4 blocks, large M
        {2, 4, 64, 16, true, false},     // materialized broadcast weight
        {1, 4, 64, 16, false, true},     // decades-wide: subnormal + dead scales
    };
    LOG_INFO("nvfp4 MatMul, FP32");
    run_matmul_nvfp4<float>(cases);
    LOG_INFO("nvfp4 MatMul, FP16");
    run_matmul_nvfp4<uint16_t>(cases);
}

// NVFP4's own contract, on top of the shared 4-bit gates: four inputs, an e4m3
// block table, a single fp32 factor, and a table whose row count says 16 values
// per block. A graph that breaks one of these is a different format, and reading
// it as this one would apply every scale to the wrong span of values.
TEST(MatMulTest, MatMulNvfp4ContractThrows) {
    vkop::tests::ScopedBufferBackend buffer;
    const int kBlock = 16;
    {
        // The global factor missing: a three-input graph is int4's convention.
        MatMulNvfp4Test<float> t(1, 4, 32, 16);
        EXPECT_THROW(t.run_test({t.inputa, t.weight, t.block_scale}, {t.output},
                                [&t](std::unique_ptr<vkop::ops::Operator> &op) {
                                    op->setAttribute(t.attr);
                                }),
                     std::runtime_error);
    }
    {
        // A block scale left as fp32 costs four bytes where the format promises
        // one, so the byte count the kernel walks would be wrong fourfold.
        MatMulNvfp4Test<float> t(1, 4, 32, 16);
        const int n_blocks = 32 / kBlock;
        auto fp32_scales = std::make_shared<Tensor<float>>(
            std::vector<int>{n_blocks * 16});
        fp32_scales->fillToCPU(
            std::vector<float>(static_cast<size_t>(n_blocks * 16), 1.0F));
        EXPECT_THROW(
            t.run_test(
                {t.inputa, t.weight, fp32_scales, t.global_scale}, {t.output},
                [&t](std::unique_ptr<vkop::ops::Operator> &op) {
                    op->setAttribute(t.attr);
                }),
            std::runtime_error);
    }
    {
        // The factor is not a byte: it is the one fp32 that lets the table be
        // bytes at all, so an fp8 or half here is a mislabeled graph.
        MatMulNvfp4Test<float> t(1, 4, 32, 16);
        EXPECT_THROW(
            t.run_test(
                {t.inputa, t.weight, t.block_scale, t.block_scale}, {t.output},
                [&t](std::unique_ptr<vkop::ops::Operator> &op) {
                    op->setAttribute(t.attr);
                }),
            std::runtime_error);
    }
    {
        // More than one fp32 where the format defines exactly one.
        MatMulNvfp4Test<float> t(1, 4, 32, 16);
        auto two_floats = std::make_shared<Tensor<float>>(std::vector<int>{2});
        two_floats->fillToCPU(std::vector<float>{1.0F, 1.0F});
        EXPECT_THROW(
            t.run_test(
                {t.inputa, t.weight, t.block_scale, two_floats}, {t.output},
                [&t](std::unique_ptr<vkop::ops::Operator> &op) {
                    op->setAttribute(t.attr);
                }),
            std::runtime_error);
    }
    {
        // A table that says 32 values per block: the row count divides K and the
        // columns cleanly, so only the format's own block length can reject it.
        MatMulNvfp4Test<float> t(1, 4, 32, 16);
        auto half_table = std::make_shared<Tensor<int8_t>>(
            std::vector<int>{16});
        half_table->set_elem_kind(vkop::core::ElemKind::kFloat8E4M3FN);
        half_table->fillToCPU(std::vector<int8_t>(16, 0x3c));
        EXPECT_THROW(
            t.run_test(
                {t.inputa, t.weight, half_table, t.global_scale}, {t.output},
                [&t](std::unique_ptr<vkop::ops::Operator> &op) {
                    op->setAttribute(t.attr);
                }),
            std::runtime_error);
    }
    {
        MatMulNvfp4Test<float> t(1, 4, 32, 12); // N % 8 != 0
        EXPECT_THROW(
            t.run_test(
                {t.inputa, t.weight, t.block_scale, t.global_scale}, {t.output},
                [&t](std::unique_ptr<vkop::ops::Operator> &op) {
                    op->setAttribute(t.attr);
                }),
            std::runtime_error);
    }
    {
        // 48 bytes over 16 columns says 1.5 blocks of 16 for K = 24... which no
        // grouping produced, and 32 % 3 is not a block: the row count has to
        // divide K before the block length can even be asked about.
        MatMulNvfp4Test<float> t(1, 4, 32, 16);
        auto ragged = std::make_shared<Tensor<int8_t>>(std::vector<int>{48});
        ragged->set_elem_kind(vkop::core::ElemKind::kFloat8E4M3FN);
        ragged->fillToCPU(std::vector<int8_t>(48, 0x3c));
        EXPECT_THROW(
            t.run_test(
                {t.inputa, t.weight, ragged, t.global_scale}, {t.output},
                [&t](std::unique_ptr<vkop::ops::Operator> &op) {
                    op->setAttribute(t.attr);
                }),
            std::runtime_error);
    }
}

TEST(MatMulTest, MatMulSplitKGemvBuffer) {
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
        // Compute-bound shapes: from m >= 12 with K % 16 == 0 and N % 4 == 0 the
        // host puts a byte weight in the shared-memory tile, which stages the
        // decoded value and applies the column scale in the epilogue. The 64-row
        // tile edge, several k-tiles, a partial last 64-column block (only the
        // quads at 128 and 132 are live) and the batched z extent all have to
        // hold. The per-column scales differ enough that a column paired with
        // another column's scale fails here.
        {1, 12, 16, 4, false, false},    // the m >= 12 gate exactly
        {1, 13, 32, 16, false, false},   // one A row past a 64-row tile
        {1, 70, 64, 136, false, false},  // K over several k-tiles, partial block
        {2, 16, 32, 24, false, true},    // batched, so the tile grid's z matters
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

// The same shapes the int8 list runs, because fp8 is meant to reach the same
// kernels through the same [A, B, scale] contract — a byte-grid change should
// cost the dispatch layer nothing. Both layouts are run per shape (E4M3 and
// E5M2), since they differ only in two shader constants and a case that used one
// would leave the other's shift widths unverified. The last two add a weight
// column spanning 1e-8..1, which is the only way the subnormal branch and the
// rounds-to-zero byte ever execute.
TEST(MatMulTest, MatMulFp8WeightOnlyBuffer) {
    vkop::tests::ScopedBufferBackend buffer;
    const std::vector<std::tuple<int, int, int, int, bool, bool, bool>> cases =
        {
            {1, 4, 32, 16, false, false, false},  // even N, plain 2-D weight
            {1, 8, 20, 7, false, false, false},   // odd N
            {1, 1, 64, 12, true, false, false},   // transB=1 GEMV over [N, K]
            {1, 2, 22, 12, true, false, false},   // K % 4 != 0
            {1, 5, 10, 14, false, false, false},  // one tap per byte extract
            {2, 3, 24, 9, false, true, false},    // materialized broadcast, odd N
            {2, 4, 16, 6, true, true, false},     // transB=1 with a batched weight
            {1, 1, 512, 256, false, false, false},  // decode: split-K GEMV
            {1, 1, 258, 260, false, false, false},  // decode: K not a multiple of
                                                   // the slice length
            {1, 3, 258, 32, false, false, false},   // decode: several rows
            {2, 1, 300, 128, false, true, false},   // decode: batched
            {1, 4, 32, 16, false, false, true},    // wide spread: subnormals
            {1, 1, 256, 64, false, false, true},   // wide spread under split-K
            // The tiled byte kernel (see the int8 list): fp8 reaches it through
            // the same loader and only f8_val differs, so the same shapes run.
            {1, 12, 16, 4, false, false, false},
            {1, 13, 32, 16, false, false, false},
            {1, 70, 64, 136, false, false, false},
            {1, 64, 48, 32, false, false, true},   // wide spread inside a tile
        };
    LOG_INFO("fp8 MatMul, FP32");
    run_matmul_fp8<float>(cases);
    LOG_INFO("fp8 MatMul, FP16");
    run_matmul_fp8<uint16_t>(cases);
}

// fp8 inherits int8's two host-side contracts, checked on the fp8 tensor so the
// shared gate is proven for both formats rather than only the one that predates
// it: a byte weight with no scale has no dequantization at all, and a scale that
// is not one entry per output column mixes columns silently.
TEST(MatMulTest, MatMulFp8ContractThrows) {
    vkop::tests::ScopedBufferBackend buffer;
    {
        MatMulFp8Test<float> t(1, 4, 32, 16, false, false, false);
        EXPECT_THROW(t.run_test({t.inputa, t.weight}, {t.output},
                                [&t](std::unique_ptr<vkop::ops::Operator> &op) {
                                    op->setAttribute(t.attr);
                                }),
                     std::runtime_error);
    }
    {
        MatMulFp8Test<float> t(1, 4, 32, 16, false, false, false);
        auto wrong = std::make_shared<Tensor<float>>(std::vector<int>{32});
        wrong->fillToCPU(std::vector<float>(32, 1.0F));
        EXPECT_THROW(t.run_test({t.inputa, t.weight, wrong}, {t.output},
                                [&t](std::unique_ptr<vkop::ops::Operator> &op) {
                                    op->setAttribute(t.attr);
                                }),
                     std::runtime_error);
    }
}