// Copyright 2026 @junka
// bfloat16 ACTIVATIONS (--act bf16): what runs, and what is refused.
//
// The bf16 activation domain is an allow-list (core/Bf16Gate.hpp) of kernels
// whose _bf16 build was verified to decode and encode bf16 end to end. Every
// other op -- including ones whose archive HAS a _bf16 key that still decodes
// fp16 -- must fail loudly, at LoadModel and again at the op's own execute,
// instead of computing on bf16 words read as fp16.
//
// Two halves:
//   * refusals: an op with no bf16 kernel (AveragePool), an operand-kind
//     mix, the MatMul down_act fold -- both as single ops and as .vkopbin
//     graphs through Runtime::LoadModel (the unary family, Cast,
//     Gather's bf16 output, FusedElemwise, Expand and Concat have bf16
//     kernels: tests/Bf16ElementwiseTest.cpp; Transpose, Pad, Rotary, Resize,
//     pools, reductions, InstanceNorm, PRelu, CumSum, ScatterElements:
//     tests/Bf16DataMoveTest.cpp; Reshape/Slice: the byte movers below);
//   * the allow-listed kernels against an fp32 reference computed from the
//     same bf16 inputs: a binary op, RMSNorm, Softmax, every MatMul route a
//     bf16 activation takes (GEMVs, fused, the 64x64 and 128x128 tiles,
//     tile-split-K, the Apple coop8 GEMM, reduce_pack, the epi_bias and
//     attn_softmax epilogues, the SwiGLU gate/up fold) and Attention's row,
//     flash and split-KV kernels -- with values past fp16's 65504 and masks
//     of finfo(bf16).min / -inf.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "core/Bf16Gate.hpp"
#include "core/DType.hpp"
#include "core/KernelSelect.hpp"
#include "core/runtime.hpp"
#include "ops/OperatorFactory.hpp"
#include "setup.hpp"

#include <gtest/gtest.h>

namespace bf16act {

using vkop::core::bf16_to_float;
using vkop::core::ElemKind;
using vkop::core::float_to_bf16;
using vkop::core::ITensor;
using vkop::core::Tensor;
using vkop::ops::OpType;
using Attrs = std::unordered_map<std::string, std::string>;

struct Env {
    std::shared_ptr<vkop::VulkanDevice> dev;
    std::shared_ptr<vkop::VulkanCommandPool> cmdpool;
    Env() {
        if (!vkop::tests::TestEnv::is_initialized())
            vkop::tests::TestEnv::initialize();
        dev = vkop::tests::TestEnv::get_device();
        cmdpool = vkop::tests::TestEnv::get_command_pool();
    }
};

size_t numel(const std::vector<int> &shape) {
    size_t n = 1;
    for (int d : shape) n *= static_cast<size_t>(d);
    return n;
}

// Deterministic values in [-1, 1), rounded to bf16 so the reference and the
// kernel start from identical inputs.
std::vector<float> bf16_values(size_t n, uint32_t seed) {
    std::vector<float> v(n);
    uint32_t s = seed * 2654435761u + 12345u;
    for (size_t i = 0; i < n; ++i) {
        s = s * 1664525u + 1013904223u;
        const float u = static_cast<float>(s >> 8) / 16777216.0F; // [0, 1)
        v[i] = bf16_to_float(float_to_bf16(2.0F * u - 1.0F));
    }
    return v;
}

std::shared_ptr<Tensor<uint16_t>> bf16_in(Env &e, const std::vector<int> &shape,
                                          const std::vector<float> &vals) {
    auto t = std::make_shared<Tensor<uint16_t>>(shape);
    t->set_elem_kind(ElemKind::kBFloat16);
    std::vector<uint16_t> bits(vals.size());
    for (size_t i = 0; i < vals.size(); ++i) bits[i] = float_to_bf16(vals[i]);
    t->fillToCPU(bits);
    t->as_storage_buffer(e.dev);
    t->copyToGPU(e.cmdpool);
    return t;
}

std::shared_ptr<Tensor<uint16_t>> fp16_in(Env &e, const std::vector<int> &shape,
                                          const std::vector<float> &vals) {
    auto t = std::make_shared<Tensor<uint16_t>>(shape);
    std::vector<uint16_t> bits(vals.size());
    for (size_t i = 0; i < vals.size(); ++i)
        bits[i] = ITensor::fp32_to_fp16(vals[i]);
    t->fillToCPU(bits);
    t->as_storage_buffer(e.dev);
    t->copyToGPU(e.cmdpool);
    return t;
}

template <typename T>
std::shared_ptr<Tensor<T>> out(Env &e, const std::vector<int> &shape,
                               ElemKind kind) {
    auto t = std::make_shared<Tensor<T>>(shape);
    t->set_elem_kind(kind);
    t->as_storage_buffer(e.dev);
    t->toGPU();
    return t;
}

std::vector<float> read_bf16(Env &e, const std::shared_ptr<Tensor<uint16_t>> &t) {
    t->copyToCPU(e.cmdpool);
    std::vector<float> v(t->num_elements());
    for (size_t i = 0; i < v.size(); ++i) v[i] = bf16_to_float((*t)[i]);
    return v;
}

// fp16 = 2 is what Runtime hands an op whose data is a bf16 activation
// (core::op_shader_fp16); the op then resolves its _bf16 module.
std::unique_ptr<vkop::ops::Operator> make_op(Env &e, OpType type, int fp16,
                                             const Attrs &attrs = {}) {
    const int tc = e.dev->is_support_nv_tensor_core()      ? 2
                   : e.dev->is_support_cooperate_matrix() ? 1
                                                          : 0;
    auto op = vkop::ops::create_from_type(type, fp16, tc,
                                          /*backend_buffer=*/true);
    op->set_runtime_device(e.dev, e.cmdpool);
    if (!attrs.empty()) op->setAttribute(attrs);
    return op;
}

void submit(Env &e, vkop::ops::Operator *op) {
    auto cmd = op->get_record();
    std::vector<VkSubmitInfo> info{cmd->buildSubmitInfo()};
    vkop::VulkanCommandBuffer::submit(e.dev->getComputeQueue(), info);
    cmd->wait();
    e.dev->wait_all_done();
}

// Runs `fn`, demands a std::runtime_error whose text contains `needle`.
void expect_refused(const std::function<void()> &fn, const std::string &needle) {
    try {
        fn();
    } catch (const std::runtime_error &err) {
        EXPECT_NE(std::string(err.what()).find(needle), std::string::npos)
            << "message was: " << err.what();
        return;
    }
    ADD_FAILURE() << "expected a refusal containing '" << needle << "'";
}

// |got - ref| within bf16's output rounding (2^-8 relative) plus the fp32
// accumulation's slack. Decoding a bf16 word as fp16 is off by orders of
// magnitude, so this cannot pass by accident.
int count_bad(const std::vector<float> &got, const std::vector<double> &ref,
              double rtol, double atol, double *worst) {
    int bad = 0;
    *worst = 0.0;
    for (size_t i = 0; i < ref.size(); ++i) {
        const double d = std::fabs(static_cast<double>(got[i]) - ref[i]);
        *worst = std::max(*worst, std::isfinite(d) ? d : 1e30);
        if (!(d <= rtol * std::fabs(ref[i]) + atol)) ++bad;
    }
    return bad;
}

} // namespace bf16act

using namespace bf16act;

// ---------------------------------------------------------------- the table

TEST(Bf16ActivationTest, AllowListIsTheVerifiedKernels) {
    using vkop::core::bf16_activation_op;
    for (OpType t : {OpType::ADD, OpType::SUB, OpType::MUL, OpType::DIV,
                     OpType::POW, OpType::MIN, OpType::MAX, OpType::MOD,
                     OpType::SPLIT, OpType::LAYERNORM, OpType::RMSNORM,
                     OpType::SOFTMAX, OpType::MATMUL, OpType::ATTENTION,
                     // byte movers (ByteMoversKeepBf16Bits below)
                     OpType::SHAPE, OpType::SQUEEZE, OpType::UNSQUEEZE,
                     OpType::RESHAPE, OpType::SLICE,
                     // tests/Bf16ElementwiseTest.cpp
                     OpType::RELU, OpType::SIGMOID, OpType::CLIP,
                     OpType::EXPAND, OpType::CONCAT, OpType::FUSED_ELEMWISE,
                     OpType::CAST, OpType::GATHER,
                     // tests/Bf16DataMoveTest.cpp
                     OpType::TRANSPOSE, OpType::ROTARY_EMBEDDING, OpType::PAD,
                     OpType::RESIZE, OpType::MAXPOOL2D,
                     OpType::GLOBALAVERAGEPOOL, OpType::REDUCE,
                     OpType::REDUCEMEAN, OpType::INSTANCENORM, OpType::PRELU,
                     OpType::CUMSUM, OpType::SCATTER_ELEMENTS}) {
        EXPECT_TRUE(bf16_activation_op(t)) << vkop::ops::convert_optype_to_string(t);
    }
    // Each of these has a _bf16 key in the archive (or falls back to the fp32
    // module) and still decodes fp16 somewhere, or has no dual build at all.
    for (OpType t : {OpType::GEMM, OpType::CONV2D, OpType::TOPK,
                     OpType::AVERAGEPOOL, OpType::WHERE}) {
        EXPECT_FALSE(bf16_activation_op(t)) << vkop::ops::convert_optype_to_string(t);
    }
    // bf16 WEIGHT slots are a separate role and stay as they were.
    EXPECT_TRUE(vkop::core::bf16_weight_slot(OpType::MATMUL, 1));
    EXPECT_TRUE(vkop::core::bf16_weight_slot(OpType::GEMM, 1));
    EXPECT_TRUE(vkop::core::bf16_weight_slot(OpType::GATHER, 0));
    EXPECT_FALSE(vkop::core::bf16_weight_slot(OpType::MATMUL, 0));
    EXPECT_FALSE(vkop::core::is_float_elem(ElemKind::kBFloat16));
}

// ------------------------------------------------------ single-op refusals

TEST(Bf16ActivationTest, GatedOpRefusesBf16) {
    Env e;
    auto x = bf16_in(e, {1, 2, 2, 8}, bf16_values(32, 1));
    auto y = out<uint16_t>(e, {1, 2, 1, 4}, ElemKind::kBFloat16);
    auto op = make_op(e, OpType::AVERAGEPOOL, 2,
                      {{"kernel_shape", "[2, 2]"}, {"strides", "[2, 2]"}});
    expect_refused([&] { op->onExecute({x}, {y}, 0); },
                   "vkop: no bf16 activation kernel for AveragePool");
}

TEST(Bf16ActivationTest, PerOperandOpsStillNeedTheirOutputBuild) {
    // FusedElemwise and Gather build for their OUTPUT format: a bf16 output
    // from an op built for fp16 would pack fp16 bits.
    Env e;
    auto table = bf16_in(e, {4, 8}, bf16_values(32, 4));
    auto idx = std::make_shared<Tensor<int64_t>>(std::vector<int>{2});
    idx->fillToCPU(std::vector<int64_t>{3, 1});
    auto keep = idx->data();
    idx->as_storage_buffer(e.dev);
    idx->copyToGPU(e.cmdpool, keep.data());
    auto y = out<uint16_t>(e, {2, 8}, ElemKind::kBFloat16);
    auto op = make_op(e, OpType::GATHER, 1, {{"axis", "0"}});
    expect_refused([&] { op->onExecute({table, idx}, {y}, 0); },
                   "built for fp16");
    // ...and an fp16 operand of a NON-per-operand op is still a mix.
    auto x = bf16_in(e, {4, 8}, bf16_values(32, 3));
    auto f = fp16_in(e, {4, 8}, bf16_values(32, 2));
    auto cat = make_op(e, OpType::CONCAT, 2, {{"axis", "0"}});
    auto z2 = out<uint16_t>(e, {8, 8}, ElemKind::kBFloat16);
    expect_refused([&] { cat->onExecute({x, f}, {z2}, 0); },
                   "mixes bfloat16 activations with a float16 input 1");
}

TEST(Bf16ActivationTest, Fp16BesideBf16Refused) {
    // The _bf16 module decodes every 16-bit operand as bf16, so an fp16
    // operand beside a bf16 one is read as noise.
    Env e;
    auto x = bf16_in(e, {4, 8}, bf16_values(32, 5));
    auto b = fp16_in(e, {4, 8}, bf16_values(32, 6));
    auto y = out<uint16_t>(e, {4, 8}, ElemKind::kBFloat16);
    auto op = make_op(e, OpType::ADD, 2);
    expect_refused([&] { op->onExecute({x, b}, {y}, 0); },
                   "mixes bfloat16 activations with a float16 input 1");
}

TEST(Bf16ActivationTest, OpBuiltForFp16RefusesBf16) {
    Env e;
    auto x = bf16_in(e, {4, 8}, bf16_values(32, 7));
    auto b = bf16_in(e, {4, 8}, bf16_values(32, 8));
    auto y = out<uint16_t>(e, {4, 8}, ElemKind::kBFloat16);
    auto op = make_op(e, OpType::ADD, 1);
    expect_refused([&] { op->onExecute({x, b}, {y}, 0); }, "built for fp16");
}

TEST(Bf16ActivationTest, MatMulWithoutABf16RouteRefused) {
    Env e;
    // The SwiGLU down fold (matmul_down_act.comp) reads fp16 only.
    {
        auto a = bf16_in(e, {1, 64}, bf16_values(64, 9));
        auto b = bf16_in(e, {64, 8}, bf16_values(64 * 8, 10));
        auto y = out<uint16_t>(e, {1, 8}, ElemKind::kBFloat16);
        auto op = make_op(e, OpType::MATMUL, 2, {{"down_act", "1"}});
        expect_refused([&] { op->onExecute({a, b}, {y}, 0); },
                       "down_act SwiGLU fold");
    }
    // A quantized/fp16 weight beside a bf16 activation: the _bf16 module
    // would decode the fp16 B as bf16.
    {
        auto a = bf16_in(e, {1, 256}, bf16_values(256, 14));
        auto b = fp16_in(e, {256, 64}, bf16_values(256 * 64, 15));
        auto y = out<uint16_t>(e, {1, 64}, ElemKind::kBFloat16);
        auto op = make_op(e, OpType::MATMUL, 2);
        expect_refused([&] { op->onExecute({a, b}, {y}, 0); },
                       "mixes bfloat16 activations with a float16 input 1");
    }
}

// ------------------------------------------ allow-listed ops vs fp32 reference

TEST(Bf16ActivationTest, BinaryAddMatchesFp32) {
    Env e;
    struct Case {
        std::vector<int> a, b, y;
    };
    for (const Case &c : {Case{{3, 70}, {3, 70}, {3, 70}},
                          Case{{3, 70}, {70}, {3, 70}},
                          Case{{2, 3, 5}, {1, 3, 1}, {2, 3, 5}}}) {
        const auto av = bf16_values(numel(c.a), 21);
        const auto bv = bf16_values(numel(c.b), 22);
        auto a = bf16_in(e, c.a, av);
        auto b = bf16_in(e, c.b, bv);
        auto y = out<uint16_t>(e, c.y, ElemKind::kBFloat16);
        auto op = make_op(e, OpType::ADD, 2);
        op->onExecute({a, b}, {y}, 0);
        submit(e, op.get());
        // Right-aligned broadcast, written out for the three cases above.
        std::vector<double> ref(numel(c.y));
        const int rank = static_cast<int>(c.y.size());
        for (size_t i = 0; i < ref.size(); ++i) {
            size_t rem = i, ia = 0, ib = 0, sa = 1, sb = 1;
            std::vector<int> idx(rank);
            for (int d = rank - 1; d >= 0; --d) {
                idx[d] = static_cast<int>(rem % c.y[d]);
                rem /= c.y[d];
            }
            for (int d = rank - 1; d >= 0; --d) {
                const int da = d - (rank - static_cast<int>(c.a.size()));
                if (da >= 0) {
                    ia += (c.a[da] == 1 ? 0 : idx[d]) * sa;
                    sa *= c.a[da];
                }
                const int db = d - (rank - static_cast<int>(c.b.size()));
                if (db >= 0) {
                    ib += (c.b[db] == 1 ? 0 : idx[d]) * sb;
                    sb *= c.b[db];
                }
            }
            ref[i] = static_cast<double>(av[ia]) + bv[ib];
        }
        double worst = 0;
        EXPECT_EQ(count_bad(read_bf16(e, y), ref, 8e-3, 1e-6, &worst), 0)
            << "Add bf16 a=" << c.a.size() << "-D worst " << worst;
    }
}

TEST(Bf16ActivationTest, RMSNormMatchesFp32) {
    Env e;
    const int rows = 6, d = 64;
    auto xv = bf16_values(rows * d, 31);
    for (auto &v : xv) v = bf16_to_float(float_to_bf16(v * 300.0F)); // > fp16's
    const auto wv = bf16_values(d, 32);                              // comfort
    auto x = bf16_in(e, {rows, d}, xv);
    auto w = bf16_in(e, {d}, wv);
    auto y = out<uint16_t>(e, {rows, d}, ElemKind::kBFloat16);
    auto op = make_op(e, OpType::RMSNORM, 2, {{"eps", "1e-6"}, {"axis", "-1"}});
    op->onExecute({x, w}, {y}, 0);
    submit(e, op.get());
    std::vector<double> ref(rows * d);
    for (int r = 0; r < rows; ++r) {
        double ss = 0;
        for (int i = 0; i < d; ++i) ss += double(xv[r * d + i]) * xv[r * d + i];
        const double inv = 1.0 / std::sqrt(ss / d + 1e-6);
        for (int i = 0; i < d; ++i) ref[r * d + i] = xv[r * d + i] * inv * wv[i];
    }
    double worst = 0;
    EXPECT_EQ(count_bad(read_bf16(e, y), ref, 1.2e-2, 1e-3, &worst), 0)
        << "RMSNorm bf16 worst " << worst;
}

TEST(Bf16ActivationTest, SoftmaxMatchesFp32) {
    Env e;
    struct Case {
        std::vector<int> shape;
        int axis;
    };
    // Innermost even axis (the packed row pass) and a middle axis (the fp32
    // scratch + pack pass).
    for (const Case &c : {Case{{4, 70}, -1}, Case{{2, 6, 4}, 1}}) {
        auto xv = bf16_values(numel(c.shape), 41);
        for (auto &v : xv) v = bf16_to_float(float_to_bf16(v * 8.0F));
        auto x = bf16_in(e, c.shape, xv);
        auto y = out<uint16_t>(e, c.shape, ElemKind::kBFloat16);
        auto op = make_op(e, OpType::SOFTMAX, 2,
                          {{"axis", std::to_string(c.axis)}});
        op->onExecute({x}, {y}, 0);
        submit(e, op.get());
        const int rank = static_cast<int>(c.shape.size());
        const int ax = c.axis < 0 ? c.axis + rank : c.axis;
        int outer = 1, inner = 1;
        for (int i = 0; i < ax; ++i) outer *= c.shape[i];
        for (int i = ax + 1; i < rank; ++i) inner *= c.shape[i];
        const int n = c.shape[ax];
        std::vector<double> ref(xv.size());
        for (int o = 0; o < outer; ++o) {
            for (int in = 0; in < inner; ++in) {
                double mx = -1e30, sum = 0;
                for (int j = 0; j < n; ++j)
                    mx = std::max(mx, double(xv[(o * n + j) * inner + in]));
                for (int j = 0; j < n; ++j)
                    sum += std::exp(xv[(o * n + j) * inner + in] - mx);
                for (int j = 0; j < n; ++j) {
                    const size_t k = (o * n + j) * inner + in;
                    ref[k] = std::exp(xv[k] - mx) / sum;
                }
            }
        }
        double worst = 0;
        EXPECT_EQ(count_bad(read_bf16(e, y), ref, 1.2e-2, 1e-5, &worst), 0)
            << "Softmax bf16 axis " << c.axis << " worst " << worst;
    }
}

// ------------------------------------------ MatMul / Attention bf16 routes
//
// Every MatMul route a bf16 activation takes now runs the same kernel family
// the fp16 graph would (the selector walks the same table; only the KHR
// cooperative-matrix GEMMs and the down_act fold stay fenced), so these tests
// assert two things per shape: the bf16 route EQUALS the fp16 route for the
// same shape, and the bf16 result matches an fp64 reference computed from the
// same bf16-rounded inputs -- including inputs past fp16's 65504, which a
// half-staged kernel would clamp.

namespace bf16act {

// finfo(torch.bfloat16).min and -inf, the two additive-mask values a bf16
// graph carries. Neither exists in fp16.
const float kBf16Min = bf16_to_float(0xFF7Fu);
const float kNegInf = -std::numeric_limits<float>::infinity();

std::shared_ptr<Tensor<uint16_t>> half_in(Env &e, bool bf16,
                                          const std::vector<int> &shape,
                                          const std::vector<float> &vals) {
    return bf16 ? bf16_in(e, shape, vals) : fp16_in(e, shape, vals);
}

std::vector<float> read_half(Env &e, bool bf16,
                             const std::shared_ptr<Tensor<uint16_t>> &t) {
    if (bf16) return read_bf16(e, t);
    t->copyToCPU(e.cmdpool);
    std::vector<float> v(t->num_elements());
    for (size_t i = 0; i < v.size(); ++i)
        v[i] = ITensor::fp16_to_fp32((*t)[i]);
    return v;
}

// Runs one MatMul on 16-bit operands: bf16 (the _bf16 module) or fp16.
struct MMRun {
    std::vector<float> y;
    std::string route;
};
MMRun run_matmul(Env &e, bool bf16, const std::vector<int> &ashape,
                 const std::vector<float> &av, const std::vector<int> &bshape,
                 const std::vector<float> &bv, const std::vector<int> &yshape,
                 const Attrs &attrs, const std::vector<int> *biasshape = nullptr,
                 const std::vector<float> *biasv = nullptr) {
    auto a = half_in(e, bf16, ashape, av);
    auto b = half_in(e, bf16, bshape, bv);
    auto y = out<uint16_t>(e, yshape,
                           bf16 ? ElemKind::kBFloat16 : ElemKind::kFloat16);
    std::vector<std::shared_ptr<ITensor>> ins{a, b};
    if (biasshape != nullptr) ins.push_back(half_in(e, bf16, *biasshape, *biasv));
    auto op = make_op(e, OpType::MATMUL, bf16 ? 2 : 1, attrs);
    op->onExecute(ins, {y}, 0);
    submit(e, op.get());
    return {read_half(e, bf16, y), op->route_tag()};
}

// The route the fp16 graph takes for this shape (same attributes, small
// in-range inputs): the bf16 route must be the same one now.
std::string fp16_route(Env &e, const std::vector<int> &ashape,
                       const std::vector<int> &bshape,
                       const std::vector<int> &yshape, const Attrs &attrs,
                       const std::vector<int> *biasshape = nullptr) {
    std::vector<float> bias;
    if (biasshape != nullptr) bias.assign(numel(*biasshape), 0.0F);
    return run_matmul(e, false, ashape, bf16_values(numel(ashape), 1), bshape,
                      bf16_values(numel(bshape), 2), yshape, attrs, biasshape,
                      biasshape != nullptr ? &bias : nullptr)
        .route;
}

// |got - ref| <= rtol*|ref| + atol_i, atol_i = max(atol, slack*sum_k|a||b|)
// per element: the fp32 accumulation (and, on coop8, the scaled stage's
// subnormal grid) is bounded by the magnitude of the products, not of their
// sum. A clamped 3e5 entry misses by ~2.3e5*|b|, far outside this.
int count_bad_mm(const std::vector<float> &got, const std::vector<double> &ref,
                 const std::vector<double> &mag, double rtol, double atol,
                 double slack, double *worst) {
    int bad = 0;
    *worst = 0.0;
    for (size_t i = 0; i < ref.size(); ++i) {
        const double d = std::fabs(static_cast<double>(got[i]) - ref[i]);
        const double tol = rtol * std::fabs(ref[i]) + std::max(atol, slack * mag[i]);
        *worst = std::max(*worst, std::isfinite(d) ? d / tol : 1e30);
        if (!(d <= tol)) ++bad;
    }
    return bad;
}

// Plain [batch, M, K] x [K, N] (or [N, K]) reference with per-element
// magnitude sum_k |a||b|. B is shared over the batch when bbatch == 1.
void mm_ref(int batch, int bbatch, int m, int k, int n, bool transB,
            const std::vector<float> &av, const std::vector<float> &bv,
            std::vector<double> *ref, std::vector<double> *mag) {
    ref->assign(static_cast<size_t>(batch) * m * n, 0.0);
    mag->assign(ref->size(), 0.0);
    for (int bt = 0; bt < batch; ++bt) {
        const size_t boff = static_cast<size_t>(bbatch == 1 ? 0 : bt) * k * n;
        for (int i = 0; i < m; ++i)
            for (int j = 0; j < n; ++j) {
                double s = 0, g = 0;
                for (int kk = 0; kk < k; ++kk) {
                    const double a = av[(static_cast<size_t>(bt) * m + i) * k + kk];
                    const double b = transB ? bv[boff + static_cast<size_t>(j) * k + kk]
                                            : bv[boff + static_cast<size_t>(kk) * n + j];
                    s += a * b;
                    g += std::fabs(a * b);
                }
                (*ref)[(static_cast<size_t>(bt) * m + i) * n + j] = s;
                (*mag)[(static_cast<size_t>(bt) * m + i) * n + j] = g;
            }
    }
}

// Sprinkles bf16 values past fp16's range: A rows `rows` get one entry each
// (alternating sign, 7e4 .. 3e6), B columns `cols` one each (2e5).
void add_outliers(std::vector<float> *av, int m, int k, const std::vector<int> &rows,
                  std::vector<float> *bv, int n, bool transB,
                  const std::vector<int> &cols, int batch = 1) {
    for (int bt = 0; bt < batch; ++bt)
        for (size_t r = 0; r < rows.size(); ++r) {
            const float v = (r & 1 ? -1.0F : 1.0F) * (7.0e4F * float(1 + 7 * r));
            const int kk = static_cast<int>((r * 37 + 5) % k);
            (*av)[(static_cast<size_t>(bt) * m + rows[r]) * k + kk] =
                bf16_to_float(float_to_bf16(v));
        }
    for (size_t c = 0; c < cols.size(); ++c) {
        const int kk = static_cast<int>((c * 53 + 11) % k);
        const float v = bf16_to_float(float_to_bf16(c & 1 ? -2.0e5F : 2.5e5F));
        if (transB)
            (*bv)[static_cast<size_t>(cols[c]) * k + kk] = v;
        else
            (*bv)[static_cast<size_t>(kk) * n + cols[c]] = v;
    }
}

} // namespace bf16act

TEST(Bf16ActivationTest, MatMulRoutesMatchFp32) {
    Env e;
    // Apple with coop8 left on: the "coop8" shapes must really reach it.
    const char *c8env = std::getenv("VKOP_COOP8");
    const bool apple = vkop::core::device_supports_apple_coop8(
                           e.dev->device_traits()) &&
                       (c8env == nullptr || std::atoi(c8env) != 0);
    struct Case {
        int batch, m, k, n;
        bool transB;
        bool outliers; // A rows / B columns past 65504
        const char *why;
    };
    const std::vector<Case> cases = {
        {1, 1, 512, 256, false, false, "decode: split-K GEMV"},
        {1, 1, 258, 260, false, false, "decode: split-K GEMV, K tail"},
        {1, 1, 512, 4096, false, true, "decode: WIDE split-K GEMV"},
        {1, 3, 64, 12, false, false, "even N, K < 256: fused even-N kernel"},
        {1, 2, 64, 12, true, false, "fused even-N, transB"},
        {2, 3, 32, 10, false, false, "batched A, shared 2-D weight"},
        {1, 20, 128, 96, false, true, "64x64 SIMT tile"},
        {1, 20, 128, 96, true, true, "64x64 SIMT tile, transB"},
        {1, 16, 1024, 64, false, true, "tile split-K"},
        {1, 130, 256, 192, false, true, "tile split-K, M > 128"},
        {1, 512, 256, 1024, false, false, "Apple coop8 GEMM"},
        {1, 512, 256, 1024, false, true, "Apple coop8 GEMM, outliers"},
        {1, 128, 128, 128, true, true, "Apple coop8 GEMM, transB, outliers"},
        {3, 70, 128, 64, false, true, "coop8 shape, batched A, outliers"},
        {1, 256, 64, 2056, false, true, "128x128 tile"},
        {1, 4, 64, 7, false, true, "odd N: reduce + bf16 pack"},
        {2, 5, 96, 33, true, false, "odd N, transB, batched"},
    };
    for (const Case &c : cases) {
        const std::vector<int> ashape =
            c.batch > 1 ? std::vector<int>{c.batch, c.m, c.k}
                        : std::vector<int>{c.m, c.k};
        const std::vector<int> bshape = c.transB ? std::vector<int>{c.n, c.k}
                                                 : std::vector<int>{c.k, c.n};
        const std::vector<int> yshape =
            c.batch > 1 ? std::vector<int>{c.batch, c.m, c.n}
                        : std::vector<int>{c.m, c.n};
        auto av = bf16_values(numel(ashape), 51 + c.m);
        auto bv = bf16_values(numel(bshape), 52 + c.n);
        if (c.outliers) {
            std::vector<int> rows{0};
            if (c.m > 3) rows.push_back(3);
            if (c.m > 70) rows.push_back(70);
            if (c.m > 129) rows.push_back(129);
            if (c.m > 300) rows.push_back(300);
            if (c.m > 500) rows.push_back(511);
            std::vector<int> cols{c.n - 1};
            if (c.n > 5) cols.push_back(5);
            add_outliers(&av, c.m, c.k, rows, &bv, c.n, c.transB, cols, c.batch);
        }
        const Attrs attrs{{"transB", c.transB ? "1" : "0"}};
        const std::string want = fp16_route(e, ashape, bshape, yshape, attrs);
        const MMRun r = run_matmul(e, true, ashape, av, bshape, bv, yshape, attrs);
        EXPECT_EQ(r.route, want) << c.why << ": bf16 and fp16 routes differ";
        if (apple && std::string(c.why).find("coop8") != std::string::npos) {
            EXPECT_EQ(r.route, "coop8") << c.why;
        }
        std::vector<double> ref, mag;
        mm_ref(c.batch, 1, c.m, c.k, c.n, c.transB, av, bv, &ref, &mag);
        double worst = 0;
        const int bad = count_bad_mm(r.y, ref, mag, 8e-3, 0.03, 2e-3, &worst);
        EXPECT_EQ(bad, 0) << c.why << " (route " << r.route << ") m=" << c.m
                          << " k=" << c.k << " n=" << c.n << " worst/tol "
                          << worst;
        LOG_INFO("bf16 MatMul %s: route %s (fp16 %s), worst err/tol %g", c.why,
                 r.route.c_str(), want.c_str(), worst);
    }
}

// The 512-lane split-K builds are opt-in (VKOP_KSPLIT512=1, read once per
// process), so this only runs in a process started with it.
TEST(Bf16ActivationTest, MatMulKsplit512MatchesFp32) {
    if (!vkop::core::matmul_knobs().ksplit512_enabled) {
        GTEST_SKIP() << "run with VKOP_KSPLIT512=1";
    }
    Env e;
    struct Case {
        int m, k, n;
    } cases[] = {{1, 2048, 256}, {1, 2048, 4096}, {2, 1024, 512}};
    for (const Case &c : cases) {
        auto av = bf16_values(c.m * c.k, 61);
        auto bv = bf16_values(c.k * c.n, 62);
        add_outliers(&av, c.m, c.k, {0}, &bv, c.n, false, {3});
        const MMRun r = run_matmul(e, true, {c.m, c.k}, av, {c.k, c.n}, bv,
                                   {c.m, c.n}, {});
        EXPECT_NE(r.route.find("512"), std::string::npos) << r.route;
        std::vector<double> ref, mag;
        mm_ref(1, 1, c.m, c.k, c.n, false, av, bv, &ref, &mag);
        double worst = 0;
        EXPECT_EQ(count_bad_mm(r.y, ref, mag, 8e-3, 0.03, 2e-3, &worst), 0)
            << "route " << r.route << " worst/tol " << worst;
    }
}

// The attention QK^T epilogue: out = acc * scale + bias, with a bf16 bias
// carrying finfo(bf16).min and -inf. The fp16 decode of the bias read those as
// NaN; now -inf stays -inf and finfo.min stays finite.
TEST(Bf16ActivationTest, MatMulEpiBiasBf16Mask) {
    Env e;
    struct Case {
        int heads, m, k, n;
        bool transB;
    } cases[] = {{2, 4, 64, 8, true},    {2, 20, 128, 96, true},
                 {2, 3, 64, 7, true},    {4, 1, 512, 64, true},
                 {2, 70, 64, 128, false}};
    for (const Case &c : cases) {
        const int B = 1;
        const int batch = B * c.heads;
        const std::vector<int> ashape{batch, c.m, c.k};
        const std::vector<int> bshape = c.transB
                                            ? std::vector<int>{batch, c.n, c.k}
                                            : std::vector<int>{batch, c.k, c.n};
        const std::vector<int> yshape{batch, c.m, c.n};
        const std::vector<int> biasshape{B, c.m, c.n};
        auto av = bf16_values(numel(ashape), 71);
        auto bv = bf16_values(numel(bshape), 72);
        std::vector<float> bias = bf16_values(numel(biasshape), 73);
        for (int i = 0; i < c.m; ++i)
            for (int j = 0; j < c.n; ++j) {
                float &v = bias[static_cast<size_t>(i) * c.n + j];
                if (j > i + c.n / 2) v = kBf16Min;      // causal-style
                if (i == c.m - 1 && j % 3 == 0) v = kNegInf;
            }
        const float scale = 0.125F;
        const Attrs attrs{{"transB", c.transB ? "1" : "0"},
                          {"epi_bias", "1"},
                          {"epi_scale", std::to_string(scale)}};
        const std::string want = fp16_route(e, ashape, bshape, yshape, attrs, &biasshape);
        const MMRun r = run_matmul(e, true, ashape, av, bshape, bv, yshape,
                                   attrs, &biasshape, &bias);
        EXPECT_EQ(r.route, want);
        std::vector<double> ref, mag;
        // B is per-batch here (bbatch = batch).
        mm_ref(batch, batch, c.m, c.k, c.n, c.transB, av, bv, &ref, &mag);
        int bad = 0;
        for (int bt = 0; bt < batch; ++bt)
            for (int i = 0; i < c.m; ++i)
                for (int j = 0; j < c.n; ++j) {
                    const size_t o = (static_cast<size_t>(bt) * c.m + i) * c.n + j;
                    const double bb = bias[static_cast<size_t>(i) * c.n + j];
                    const double want_v = ref[o] * scale + bb;
                    const float got = r.y[o];
                    bool ok;
                    if (std::isinf(want_v)) {
                        ok = std::isinf(got) && got < 0;
                    } else {
                        ok = std::fabs(got - want_v) <=
                             8e-3 * std::fabs(want_v) + 0.02 + 2e-3 * mag[o] * scale;
                    }
                    if (!ok && bad++ < 5)
                        ADD_FAILURE() << "epi_bias (route " << r.route << ") at "
                                      << bt << "," << i << "," << j << ": got "
                                      << got << " want " << want_v;
                }
        EXPECT_EQ(bad, 0) << "route " << r.route << " m=" << c.m << " n=" << c.n;
        LOG_INFO("bf16 MatMul epi_bias m=%d n=%d: route %s", c.m, c.n,
                 r.route.c_str());
    }
}

// The fused QK^T + scale + bias + softmax row kernel (attn_softmax). Head 0's
// rows mix real scores, finfo(bf16).min and -inf; head 1's bias masks every
// column with finfo.min, whose reference softmax is uniform (all scores equal
// once the mask swamps them) -- the old -1e30 running max gave zeros there.
TEST(Bf16ActivationTest, MatMulAttnSoftmaxBf16Mask) {
    Env e;
    const int B = 2, H = 2, m = 1, k = 64, n = 128;
    const int batch = B * H;
    const std::vector<int> ashape{batch, m, k};
    const std::vector<int> bshape{batch, n, k};
    const std::vector<int> yshape{batch, m, n};
    const std::vector<int> biasshape{B, m, n};
    auto av = bf16_values(numel(ashape), 81);
    auto bv = bf16_values(numel(bshape), 82);
    std::vector<float> bias(numel(biasshape), 0.0F);
    for (int j = 0; j < n; ++j) {
        bias[j] = j < 40 ? 0.0F : (j % 2 ? kBf16Min : kNegInf);
        bias[n + j] = kBf16Min;
    }
    const float scale = 0.125F;
    const Attrs attrs{{"transB", "1"},
                      {"epi_bias", "1"},
                      {"epi_scale", std::to_string(scale)},
                      {"attn_softmax", "1"}};
    const std::string want = fp16_route(e, ashape, bshape, yshape, attrs, &biasshape);
    const MMRun r = run_matmul(e, true, ashape, av, bshape, bv, yshape, attrs,
                               &biasshape, &bias);
    EXPECT_EQ(r.route, want);
    EXPECT_EQ(r.route, "attn_softmax");
    std::vector<double> ref, mag;
    mm_ref(batch, batch, m, k, n, true, av, bv, &ref, &mag);
    std::vector<double> sm(ref.size());
    for (int bt = 0; bt < batch; ++bt) {
        const int bb = bt / H;
        std::vector<double> x(n);
        double mx = -std::numeric_limits<double>::infinity();
        for (int j = 0; j < n; ++j) {
            x[j] = ref[static_cast<size_t>(bt) * n + j] * scale +
                   double(bias[static_cast<size_t>(bb) * n + j]);
            // fp32 cannot tell finfo.min + O(1) apart: neither can the GPU.
            x[j] = double(static_cast<float>(x[j]));
            mx = std::max(mx, x[j]);
        }
        double s = 0;
        for (int j = 0; j < n; ++j) s += std::exp(x[j] - mx);
        for (int j = 0; j < n; ++j)
            sm[static_cast<size_t>(bt) * n + j] = std::exp(x[j] - mx) / s;
    }
    double worst = 0;
    EXPECT_EQ(count_bad(r.y, sm, 1.2e-2, 1e-4, &worst), 0)
        << "attn_softmax worst " << worst;
    // The fully masked head: uniform 1/n.
    EXPECT_NEAR(r.y[static_cast<size_t>(2) * n + 5], 1.0 / n, 1e-4);
}

// The SwiGLU gate/up fold: B interleaves Wg and Wu four columns at a time and
// the result is silu(x.Wg) * (x.Wu), [M, N_up]. Each shape reaches a
// different epilogue: the WIDE GEMV's, coop8's, and the scratch +
// matmul_swiglu_pack pass behind the tile.
TEST(Bf16ActivationTest, MatMulSwigluFoldMatchesFp32) {
    Env e;
    struct Case {
        int m, k, nup;
        const char *why;
    } cases[] = {{1, 256, 512, "decode GEMV + swiglu_pack"},
                 {1, 256, 2048, "decode WIDE GEMV epilogue"},
                 {64, 64, 64, "coop8 epilogue"},
                 {20, 128, 48, "tile + swiglu_pack"},
                 {3, 64, 16, "fused + swiglu_pack"}};
    for (const Case &c : cases) {
        const int n = 2 * c.nup;
        auto av = bf16_values(static_cast<size_t>(c.m) * c.k, 91);
        auto bv = bf16_values(static_cast<size_t>(c.k) * n, 92);
        const Attrs attrs{{"swiglu", "1"}};
        const std::string want =
            fp16_route(e, {c.m, c.k}, {c.k, n}, {c.m, c.nup}, attrs);
        const MMRun r = run_matmul(e, true, {c.m, c.k}, av, {c.k, n}, bv,
                                   {c.m, c.nup}, attrs);
        EXPECT_EQ(r.route, want) << c.why;
        std::vector<double> ref, mag;
        mm_ref(1, 1, c.m, c.k, n, false, av, bv, &ref, &mag);
        std::vector<double> h(static_cast<size_t>(c.m) * c.nup);
        for (int i = 0; i < c.m; ++i)
            for (int j = 0; j < c.nup; ++j) {
                const int oct = j / 4, q = j % 4;
                // The kernel rounds the plain product to the output width
                // before the pack pass applies silu on the routes without an
                // epilogue; the reference uses exact products, so allow for it.
                const double g = ref[static_cast<size_t>(i) * n + oct * 8 + q];
                const double u = ref[static_cast<size_t>(i) * n + oct * 8 + 4 + q];
                h[static_cast<size_t>(i) * c.nup + j] = g / (1.0 + std::exp(-g)) * u;
            }
        double worst = 0;
        EXPECT_EQ(count_bad(r.y, h, 2.5e-2, 0.03, &worst), 0)
            << c.why << " (route " << r.route << ") worst " << worst;
        LOG_INFO("bf16 SwiGLU fold %s: route %s", c.why, r.route.c_str());
    }
}

// ----------------------------------------------------------- Attention

namespace bf16act {

// Q [1,H,S,D], K/V [1,Hkv,kv,D], optional mask [1,1,S,kv]; fp64 reference.
// A row whose scores are all -inf has no reference softmax (0/0); the kernels
// store zeros for it, which is what is checked there.
std::vector<double> attn_ref(int H, int Hkv, int S, int D, int kv,
                             const std::vector<float> &q, const std::vector<float> &k,
                             const std::vector<float> &v,
                             const std::vector<float> *mask, float scale,
                             bool causal) {
    const int rep = H / Hkv;
    std::vector<double> y(static_cast<size_t>(H) * S * D, 0.0);
    for (int h = 0; h < H; ++h)
        for (int s = 0; s < S; ++s) {
            std::vector<double> x(kv);
            double mx = -std::numeric_limits<double>::infinity();
            for (int j = 0; j < kv; ++j) {
                double dot = 0;
                for (int d = 0; d < D; ++d)
                    dot += double(q[(static_cast<size_t>(h) * S + s) * D + d]) *
                           k[(static_cast<size_t>(h / rep) * kv + j) * D + d];
                double sc = dot * scale;
                if (mask) sc += (*mask)[static_cast<size_t>(s) * kv + j];
                sc = double(static_cast<float>(sc)); // the kernels' fp32 sum
                if (causal && j > kv - S + s) sc = -std::numeric_limits<double>::infinity();
                x[j] = sc;
                mx = std::max(mx, sc);
            }
            if (std::isinf(mx)) continue; // all -inf: zeros
            double l = 0;
            for (int j = 0; j < kv; ++j) l += std::exp(x[j] - mx);
            for (int j = 0; j < kv; ++j) {
                const double p = std::exp(x[j] - mx) / l;
                for (int d = 0; d < D; ++d)
                    y[(static_cast<size_t>(h) * S + s) * D + d] +=
                        p * v[(static_cast<size_t>(h / rep) * kv + j) * D + d];
            }
        }
    return y;
}

// A bf16 mask exercising every form: real values, finfo(bf16).min columns,
// -inf columns, one row fully finfo.min (uniform reference) and, when
// `neg_inf_row`, one row fully -inf (zeros).
std::vector<float> attn_mask(int S, int kv, bool neg_inf_row) {
    std::vector<float> m = bf16_values(static_cast<size_t>(S) * kv, 99);
    for (int s = 0; s < S; ++s)
        for (int j = 0; j < kv; ++j) {
            float &x = m[static_cast<size_t>(s) * kv + j];
            if ((j + s) % 7 == 3) x = kBf16Min;
            if ((j + 2 * s) % 11 == 5) x = kNegInf;
            if (s == 1) x = kBf16Min;
            if (neg_inf_row && s == S - 1) x = kNegInf;
        }
    return m;
}

struct AttnRun {
    std::vector<float> y;
    std::string route;
};

} // namespace bf16act

// Row kernel and the tiled (flash) kernel, bf16 Q/K/V/mask/Y. V carries
// entries past fp16's range (outlier channel), so a half stage would clamp
// them; the output then sits past 65504 as well. Each shape is run against the
// fp16 graph's route for the same shape: bf16 now reaches the flash kernel
// (its SIMT -DBF16 build; the coop8 build has no bf16 form).
TEST(Bf16ActivationTest, AttentionRoutesMatchFp32) {
    Env e;
    struct Case {
        int H, Hkv, S, D, kv;
        bool mask, causal;
        const char *why;
    } cases[] = {
        {4, 2, 3, 64, 300, true, false, "row kernel, multi-tile kv"},
        {4, 4, 1, 128, 77, true, false, "row kernel, decode"},
        {16, 8, 128, 64, 128, true, false, "flash"},
        {16, 8, 130, 128, 200, false, true, "flash, causal with a history"},
        {16, 16, 96, 64, 131, true, false, "flash, odd kv mask"},
    };
    for (const Case &c : cases) {
        const float scale = 1.0F / std::sqrt(static_cast<float>(c.D));
        auto q = bf16_values(static_cast<size_t>(c.H) * c.S * c.D, 101);
        auto k = bf16_values(static_cast<size_t>(c.Hkv) * c.kv * c.D, 102);
        auto v = bf16_values(static_cast<size_t>(c.Hkv) * c.kv * c.D, 103);
        for (size_t i = 7; i < v.size(); i += static_cast<size_t>(c.D) * 5)
            v[i] = bf16_to_float(float_to_bf16(1.5e5F));
        std::vector<float> mask = attn_mask(c.S, c.kv, c.S > 2);
        const Attrs attrs{{"scale", std::to_string(scale)},
                          {"num_heads", std::to_string(c.H)},
                          {"num_kv_heads", std::to_string(c.Hkv)},
                          {"causal", c.causal ? "1" : "0"}};
        auto run = [&](bool bf16) {
            std::vector<float> qv = q, kvv = k, vv = v, mv = mask;
            if (!bf16) { // in-range data: only the route matters here
                vv = bf16_values(v.size(), 3);
                for (float &x : mv) x = std::max(x, -65504.0F);
            }
            std::vector<std::shared_ptr<ITensor>> ins{
                half_in(e, bf16, {1, c.H, c.S, c.D}, qv),
                half_in(e, bf16, {1, c.Hkv, c.kv, c.D}, kvv),
                half_in(e, bf16, {1, c.Hkv, c.kv, c.D}, vv)};
            if (c.mask) ins.push_back(half_in(e, bf16, {1, 1, c.S, c.kv}, mv));
            auto y = out<uint16_t>(e, {1, c.H, c.S, c.D},
                                   bf16 ? ElemKind::kBFloat16 : ElemKind::kFloat16);
            auto op = make_op(e, OpType::ATTENTION, bf16 ? 2 : 1, attrs);
            op->onExecute(ins, {y}, 0);
            submit(e, op.get());
            return AttnRun{read_half(e, bf16, y), op->route_tag()};
        };
        const AttnRun f16 = run(false);
        const AttnRun r = run(true);
        // The coop8 flash build is fp16-only: bf16 takes the SIMT flash.
        const std::string want = f16.route == "flash_coop8" ? "flash" : f16.route;
        EXPECT_EQ(r.route, want) << c.why;
        const auto ref = attn_ref(c.H, c.Hkv, c.S, c.D, c.kv, q, k, v,
                                  c.mask ? &mask : nullptr, scale, c.causal);
        double worst = 0;
        // rtol: bf16 output; atol: P (fp16 in the flash stage) times the
        // 1.5e5 outlier channel.
        EXPECT_EQ(count_bad(r.y, ref, 1.2e-2, 0.02 + 1.5e5 * 2e-3, &worst), 0)
            << c.why << " (route " << r.route << ") worst " << worst;
        LOG_INFO("bf16 Attention %s: route %s (fp16 %s), worst |err| %g", c.why,
                 r.route.c_str(), f16.route.c_str(), worst);
    }
}

// H3's unfused graph materializes BF16 QK and BF16 softmax; the experimental
// fold transposes its already-scaled K back and uses online attention instead.
TEST(Bf16ActivationTest, H3AttentionMatchesUnfusedBf16) {
    Env e;
    for (const auto &c : std::vector<std::pair<int, int>>{{56, 1018}, {2, 1019}}) {
        const int H = c.first, S = c.second, D = 128;
        const std::vector<int> shape{1, H, S, D};
        const std::vector<int> kt_shape{1, H, D, S};
        const std::vector<int> score_shape{1, H, S, S};
        auto qv = bf16_values(numel(shape), 201);
        auto kv = bf16_values(numel(shape), 202);
        auto vv = bf16_values(numel(shape), 203);
        std::vector<float> ktv(kv.size());
        for (int h = 0; h < H; ++h)
            for (int s = 0; s < S; ++s)
                for (int d = 0; d < D; ++d)
                    ktv[(static_cast<size_t>(h) * D + d) * S + s] =
                        kv[(static_cast<size_t>(h) * S + s) * D + d];

        auto q = bf16_in(e, shape, qv);
        auto kt = bf16_in(e, kt_shape, ktv);
        auto v = bf16_in(e, shape, vv);
        auto score = out<uint16_t>(e, score_shape, ElemKind::kBFloat16);
        auto prob = out<uint16_t>(e, score_shape, ElemKind::kBFloat16);
        auto baseline = out<uint16_t>(e, shape, ElemKind::kBFloat16);
        auto qk = make_op(e, OpType::MATMUL, 2);
        qk->onExecute({q, kt}, {score}, 0);
        submit(e, qk.get());
        auto sx = make_op(e, OpType::SOFTMAX, 2, {{"axis", "-1"}});
        sx->onExecute({score}, {prob}, 0);
        submit(e, sx.get());
        auto pv = make_op(e, OpType::MATMUL, 2);
        pv->onExecute({prob, v}, {baseline}, 0);
        submit(e, pv.get());

        auto k = out<uint16_t>(e, shape, ElemKind::kBFloat16);
        auto tp = make_op(e, OpType::TRANSPOSE, 2, {{"perm", "[0, 1, 3, 2]"}});
        tp->onExecute({kt}, {k}, 0);
        submit(e, tp.get());
        auto fused = out<uint16_t>(e, shape, ElemKind::kBFloat16);
        auto attention = make_op(e, OpType::ATTENTION, 2,
                                 {{"h3_self_attention", "1"}});
        attention->onExecute({q, k, v}, {fused}, 0);
        submit(e, attention.get());
        EXPECT_EQ(attention->route_tag(), "flash");

        const auto ref = read_bf16(e, baseline);
        const auto got = read_bf16(e, fused);
        double max_abs = 0.0, sum_abs = 0.0;
        size_t nonfinite = 0;
        for (size_t i = 0; i < got.size(); ++i) {
            if (!std::isfinite(ref[i]) || !std::isfinite(got[i])) {
                ++nonfinite;
                continue;
            }
            const double err = std::fabs(static_cast<double>(got[i]) - ref[i]);
            max_abs = std::max(max_abs, err);
            sum_abs += err;
        }
        // Independent CPU references on rows distributed across heads and
        // sequence: one rounds both intermediates to BF16, one does not.
        double baseline_cpu_max = 0.0, fused_cpu_max = 0.0;
        double expected_rounding_max = 0.0;
        int baseline_bad = 0, fused_bad = 0;
        constexpr int kSamples = 8;
        std::vector<float> scores(S), probs(S), bf16_probs(S);
        for (int sample = 0; sample < kSamples; ++sample) {
            const int h = (H - 1) * sample / (kSamples - 1);
            const int row = (S - 1) * sample / (kSamples - 1);
            const size_t qbase = (static_cast<size_t>(h) * S + row) * D;
            float full_max = -std::numeric_limits<float>::infinity();
            float rounded_max = full_max;
            for (int j = 0; j < S; ++j) {
                const size_t kbase = (static_cast<size_t>(h) * S + j) * D;
                float dot = 0.0F;
                for (int d = 0; d < D; ++d)
                    dot += qv[qbase + d] * kv[kbase + d];
                scores[j] = dot;
                full_max = std::max(full_max, dot);
                rounded_max = std::max(
                    rounded_max, bf16_to_float(float_to_bf16(dot)));
            }
            float full_sum = 0.0F, rounded_sum = 0.0F;
            for (int j = 0; j < S; ++j) {
                probs[j] = std::exp(scores[j] - full_max);
                bf16_probs[j] = std::exp(
                    bf16_to_float(float_to_bf16(scores[j])) - rounded_max);
                full_sum += probs[j];
                rounded_sum += bf16_probs[j];
            }
            for (int j = 0; j < S; ++j) {
                probs[j] /= full_sum;
                bf16_probs[j] =
                    bf16_to_float(float_to_bf16(bf16_probs[j] / rounded_sum));
            }
            for (int d = 0; d < D; ++d) {
                float full = 0.0F, rounded = 0.0F;
                for (int j = 0; j < S; ++j) {
                    const float value = vv[(static_cast<size_t>(h) * S + j) * D + d];
                    full += probs[j] * value;
                    rounded += bf16_probs[j] * value;
                }
                full = bf16_to_float(float_to_bf16(full));
                rounded = bf16_to_float(float_to_bf16(rounded));
                const double be = std::fabs(ref[qbase + d] - rounded);
                const double fe = std::fabs(got[qbase + d] - full);
                baseline_cpu_max = std::max(baseline_cpu_max, be);
                fused_cpu_max = std::max(fused_cpu_max, fe);
                expected_rounding_max = std::max(
                    expected_rounding_max, double(std::fabs(full - rounded)));
                if (!(be <= 0.004 + 0.01 * std::fabs(rounded))) ++baseline_bad;
                if (!(fe <= 0.004 + 0.01 * std::fabs(full))) ++fused_bad;
            }
        }
        std::fprintf(stderr,
                     "H3 attention H=%d S=%d route=%s max_abs=%g mean_abs=%g "
                     "nonfinite=%zu sampled_cpu_baseline=%g sampled_cpu_fused=%g "
                     "sampled_rounding_difference=%g\n",
                     H, S, attention->route_tag().c_str(), max_abs,
                     sum_abs / got.size(), nonfinite, baseline_cpu_max,
                     fused_cpu_max, expected_rounding_max);
        EXPECT_EQ(nonfinite, 0u);
        EXPECT_EQ(baseline_bad, 0);
        EXPECT_EQ(fused_bad, 0);
    }
}

TEST(Bf16ActivationTest, H3AttentionRejectsMismatchedDynamicShapes) {
    Env e;
    auto q = bf16_in(e, {1, 2, 4, 128}, bf16_values(2 * 4 * 128, 211));
    auto k = bf16_in(e, {1, 1, 4, 128}, bf16_values(4 * 128, 212));
    auto v = bf16_in(e, {1, 2, 4, 128}, bf16_values(2 * 4 * 128, 213));
    auto y = out<uint16_t>(e, {1, 2, 4, 128}, ElemKind::kBFloat16);
    auto op = make_op(e, OpType::ATTENTION, 2, {{"h3_self_attention", "1"}});
    expect_refused([&] { op->onExecute({q, k, v}, {y}, 0); },
                   "H3 self-attention");
}

// Split-KV decode on a bf16 KV cache: the -DBF16 build of attention.comp's
// KV_SPLIT pair, with the history in the cache and one fresh row.
TEST(Bf16ActivationTest, AttentionSplitKvDecodeBf16) {
    Env e;
    const int H = 16, Hkv = 8, D = 128, past = 2000, S = 1, kv = past + S;
    const float scale = 1.0F / std::sqrt(static_cast<float>(D));
    auto q = bf16_values(static_cast<size_t>(H) * D, 111);
    auto kall = bf16_values(static_cast<size_t>(Hkv) * kv * D, 112);
    auto vall = bf16_values(static_cast<size_t>(Hkv) * kv * D, 113);
    for (size_t i = 3; i < vall.size(); i += static_cast<size_t>(D) * 9)
        vall[i] = bf16_to_float(float_to_bf16(-1.2e5F));
    std::vector<float> mask(kv, 0.0F);
    for (int j = 0; j < kv; ++j)
        if (j % 5 == 1) mask[j] = (j % 2) ? kBf16Min : kNegInf;
    // past: [1,2,Hkv,past,D] (K then V); fresh K/V: [1,Hkv,1,D].
    std::vector<float> cache(static_cast<size_t>(2) * Hkv * past * D);
    std::vector<float> kn(static_cast<size_t>(Hkv) * D), vn(kn.size());
    for (int h = 0; h < Hkv; ++h)
        for (int j = 0; j < kv; ++j)
            for (int d = 0; d < D; ++d) {
                const size_t src = (static_cast<size_t>(h) * kv + j) * D + d;
                if (j < past) {
                    cache[(static_cast<size_t>(h) * past + j) * D + d] = kall[src];
                    cache[(static_cast<size_t>(Hkv + h) * past + j) * D + d] = vall[src];
                } else {
                    kn[static_cast<size_t>(h) * D + d] = kall[src];
                    vn[static_cast<size_t>(h) * D + d] = vall[src];
                }
            }
    auto pastt = bf16_in(e, {1, 2, Hkv, past, D}, cache);
    auto pres = out<uint16_t>(e, {1, 2, Hkv, kv, D}, ElemKind::kBFloat16);
    auto y = out<uint16_t>(e, {1, H, S, D}, ElemKind::kBFloat16);
    auto op = make_op(e, OpType::ATTENTION, 2,
                      {{"scale", std::to_string(scale)},
                       {"num_heads", std::to_string(H)},
                       {"num_kv_heads", std::to_string(Hkv)},
                       {"causal", "1"},
                       {"has_cache", "1"}});
    op->onExecute({bf16_in(e, {1, H, S, D}, q), bf16_in(e, {1, Hkv, S, D}, kn),
                   bf16_in(e, {1, Hkv, S, D}, vn), pastt,
                   bf16_in(e, {1, 1, S, kv}, mask)},
                  {y, pres}, 0);
    submit(e, op.get());
    const char *split_env = std::getenv("VKOP_ATTN_SPLIT");
    if (split_env == nullptr || split_env[0] != '0') {
        EXPECT_EQ(op->route_tag(), "split_kv");
    }
    const auto ref =
        attn_ref(H, Hkv, S, D, kv, q, kall, vall, &mask, scale, true);
    double worst = 0;
    EXPECT_EQ(count_bad(read_bf16(e, y), ref, 1.2e-2, 0.02 + 1.2e5 * 1e-3, &worst), 0)
        << "split-KV worst " << worst;
    // present = history + the fresh row, bit-exact (a word copy).
    const auto pv = read_bf16(e, pres);
    int bad = 0;
    for (int c2 = 0; c2 < 2; ++c2)
        for (int h = 0; h < Hkv; ++h)
            for (int j = 0; j < kv; ++j)
                for (int d = 0; d < D; ++d) {
                    const size_t src = (static_cast<size_t>(h) * kv + j) * D + d;
                    const float want = c2 == 0 ? kall[src] : vall[src];
                    const float got =
                        pv[((static_cast<size_t>(c2) * Hkv + h) * kv + j) * D + d];
                    if (got != want) ++bad;
                }
    EXPECT_EQ(bad, 0) << "present cache";
}

// Softmax over a bf16 row that is entirely finfo(bf16).min (uniform) or
// entirely -inf (zeros, not NaN), beside an ordinary row.
TEST(Bf16ActivationTest, SoftmaxFullyMaskedBf16Rows) {
    Env e;
    const int rows = 3, n = 64;
    std::vector<float> x = bf16_values(static_cast<size_t>(rows) * n, 121);
    for (int j = 0; j < n; ++j) {
        x[n + j] = kBf16Min;
        x[2 * n + j] = kNegInf;
    }
    auto xi = bf16_in(e, {rows, n}, x);
    auto y = out<uint16_t>(e, {rows, n}, ElemKind::kBFloat16);
    auto op = make_op(e, OpType::SOFTMAX, 2, {{"axis", "-1"}});
    op->onExecute({xi}, {y}, 0);
    submit(e, op.get());
    const auto got = read_bf16(e, y);
    double mx = -1e300, s = 0;
    for (int j = 0; j < n; ++j) mx = std::max(mx, double(x[j]));
    for (int j = 0; j < n; ++j) s += std::exp(x[j] - mx);
    for (int j = 0; j < n; ++j) {
        EXPECT_NEAR(got[j], std::exp(x[j] - mx) / s, 1e-2 * std::exp(x[j] - mx) / s + 1e-5);
        EXPECT_NEAR(got[n + j], 1.0 / n, 1e-4) << "finfo.min row, col " << j;
        EXPECT_EQ(got[2 * n + j], 0.0F) << "-inf row, col " << j;
    }
}

// ------------------------------------------------------- LoadModel (graphs)

namespace bf16act {
std::shared_ptr<vkop::core::Runtime> load(const std::string &file) {
    auto rt = std::make_shared<vkop::core::Runtime>(
        vkop::tests::TestEnv::get_command_pool(),
        std::string(BF16_GATE_DATA) + "/" + file, 0);
    rt->set_backend_buffer(true);
    rt->LoadModel();
    return rt;
}
} // namespace bf16act

TEST(Bf16ActivationTest, LoadModelRunsAnAllowListedGraph) {
    Env e;
    auto rt = load("bf16_gate_add.vkopbin");
    auto x = vkop::core::as_tensor<uint16_t>(rt->GetInput("x"));
    auto y = vkop::core::as_tensor<uint16_t>(rt->GetOutput("y"));
    ASSERT_TRUE(x && y);
    EXPECT_EQ(x->elem_kind(), ElemKind::kBFloat16);
    // A bf16 graph output used to be born fp16 (no bf16 branch in the output
    // loop), so the producer resolved its _fp16 module.
    EXPECT_EQ(y->elem_kind(), ElemKind::kBFloat16);
    std::vector<float> xv(16);
    std::vector<uint16_t> bits(16);
    for (int i = 0; i < 16; ++i) {
        // Exact in bf16 (its step is 2 in [256, 512)); x + b then rounds,
        // and the kernel's round-to-nearest-even pack must land on the same
        // bf16 as the host mirror (core::float_to_bf16), bit for bit.
        xv[i] = 256.0F + 2.0F * i;
        bits[i] = float_to_bf16(xv[i]);
    }
    x->fillToCPU(bits);
    x->copyToGPU(e.cmdpool);
    rt->Run();
    rt->ReadResult();
    const auto got = read_bf16(e, y);
    for (int i = 0; i < 16; ++i) {
        const float b = (static_cast<float>(i) - 7.0F) / 8.0F;
        const float want = bf16_to_float(float_to_bf16(xv[i] + b));
        EXPECT_EQ(got[i], want) << "element " << i;
    }
}

TEST(Bf16ActivationTest, LoadModelRefusesGatedGraphs) {
    expect_refused([] { load("bf16_gate_avgpool.vkopbin"); },
                   "vkop: no bf16 activation kernel for AveragePool "
                   "'pool0'");
    expect_refused([] { load("bf16_gate_mixed.vkopbin"); },
                   "vkop: Add 'add_mixed' mixes bfloat16 activations with a "
                   "float16 input 1");
}

// The graphs the gate used to refuse, run through LoadModel: the loader's
// dtype inference (Cast to=16, Gather's recorded bf16 output) and
// op_shader_fp16 have to land each node on its bf16 route.
TEST(Bf16ActivationTest, LoadModelRunsTheNewBf16Graphs) {
    Env e;
    // Relu(x bf16) -> y bf16. x holds values past fp16's range.
    {
        auto rt = load("bf16_gate_relu.vkopbin");
        auto x = vkop::core::as_tensor<uint16_t>(rt->GetInput("x"));
        auto y = vkop::core::as_tensor<uint16_t>(rt->GetOutput("y"));
        ASSERT_TRUE(x && y);
        EXPECT_EQ(y->elem_kind(), ElemKind::kBFloat16);
        std::vector<uint16_t> bits(16);
        for (int i = 0; i < 16; ++i)
            bits[i] = float_to_bf16((i - 8) * 70000.0F);
        x->fillToCPU(bits);
        x->copyToGPU(e.cmdpool);
        rt->Run();
        rt->ReadResult();
        const auto got = read_bf16(e, y);
        for (int i = 0; i < 16; ++i)
            EXPECT_EQ(got[i], bf16_to_float(bits[i]) > 0
                                  ? bf16_to_float(bits[i])
                                  : 0.0F)
                << "relu element " << i;
    }
    // Cast(x bf16, to=FLOAT) -> y fp32: exact widening.
    {
        auto rt = load("bf16_gate_cast.vkopbin");
        auto x = vkop::core::as_tensor<uint16_t>(rt->GetInput("x"));
        auto y = vkop::core::as_tensor<float>(rt->GetOutput("y"));
        ASSERT_TRUE(x && y);
        std::vector<uint16_t> bits(16);
        for (int i = 0; i < 16; ++i)
            bits[i] = float_to_bf16((i - 8) * 12345.0F);
        x->fillToCPU(bits);
        x->copyToGPU(e.cmdpool);
        rt->Run();
        rt->ReadResult();
        y->copyToCPU(e.cmdpool);
        for (int i = 0; i < 16; ++i)
            EXPECT_EQ((*y)[i], bf16_to_float(bits[i])) << "cast element " << i;
    }
    // Cast(x fp32, to=BFLOAT16) -> y bf16: the loader's to=16 output.
    {
        auto rt = load("bf16_gate_cast16.vkopbin");
        auto x = vkop::core::as_tensor<float>(rt->GetInput("x"));
        auto y = vkop::core::as_tensor<uint16_t>(rt->GetOutput("y"));
        ASSERT_TRUE(x && y);
        EXPECT_EQ(y->elem_kind(), ElemKind::kBFloat16);
        std::vector<float> xv(16);
        for (int i = 0; i < 16; ++i) xv[i] = (i - 8) * 10001.37F;
        x->fillToCPU(xv);
        x->copyToGPU(e.cmdpool);
        rt->Run();
        rt->ReadResult();
        y->copyToCPU(e.cmdpool);
        for (int i = 0; i < 16; ++i)
            EXPECT_EQ((*y)[i], float_to_bf16(xv[i])) << "cast16 element " << i;
    }
    // Gather(table bf16, idx) -> y bf16: a bit copy of rows 3 and 1.
    {
        auto rt = load("bf16_gate_gather.vkopbin");
        auto idx = vkop::core::as_tensor<int64_t>(rt->GetInput("idx"));
        auto y = vkop::core::as_tensor<uint16_t>(rt->GetOutput("y"));
        ASSERT_TRUE(idx && y);
        EXPECT_EQ(y->elem_kind(), ElemKind::kBFloat16);
        std::vector<int64_t> iv{3, 1};
        idx->fillToCPU(iv);
        idx->copyToGPU(e.cmdpool, iv.data());
        rt->Run();
        rt->ReadResult();
        const auto got = read_bf16(e, y);
        for (int c = 0; c < 8; ++c) {
            EXPECT_EQ(got[c], float(3 * 8 + c));
            EXPECT_EQ(got[8 + c], float(1 * 8 + c));
        }
    }
}

// ------------------------------------------------------------ byte movers

// Reshape and Slice never decode an element: a bf16 tensor must come out bit
// for bit, and with its bf16 KIND (the next op's gate keys on it).
TEST(Bf16ActivationTest, ByteMoversKeepBf16Bits) {
    Env e;
    // A non-trivial pattern, including NaN/-inf/-0 words that an fp16
    // decode/encode would canonicalize or round.
    std::vector<uint16_t> bits(48);
    for (size_t i = 0; i < bits.size(); ++i)
        bits[i] = static_cast<uint16_t>(0x3F80u + 37u * i);
    bits[3] = 0x7FC1; // NaN with payload
    bits[4] = 0xFF80; // -inf
    bits[5] = 0x8000; // -0
    auto make = [&](const std::vector<int> &shape) {
        auto t = std::make_shared<Tensor<uint16_t>>(shape);
        t->set_elem_kind(ElemKind::kBFloat16);
        t->fillToCPU(bits);
        t->as_storage_buffer(e.dev);
        t->copyToGPU(e.cmdpool);
        return t;
    };
    auto i64 = [&](const std::vector<int64_t> &v) {
        auto t = std::make_shared<Tensor<int64_t>>(
            std::vector<int>{static_cast<int>(v.size())});
        t->fillToCPU(v);
        auto keep = t->data();
        t->as_storage_buffer(e.dev);
        t->copyToGPU(e.cmdpool, keep.data());
        return t;
    };
    // Slice on the inner axis of [6, 8]: columns 1..6 step 2 -> [6, 3]. Not
    // contiguous, so it runs slice.comp's _bf16 module (an odd row length
    // also exercises the half-word split).
    {
        auto x = make({6, 8});
        auto y = out<uint16_t>(e, {6, 3}, ElemKind::kBFloat16);
        auto op = make_op(e, OpType::SLICE, 2);
        op->onExecute({x, i64({1}), i64({7}), i64({1}), i64({2})}, {y}, 0);
        submit(e, op.get());
        y->copyToCPU(e.cmdpool);
        EXPECT_EQ(y->elem_kind(), ElemKind::kBFloat16);
        for (int r = 0; r < 6; ++r)
            for (int c = 0; c < 3; ++c)
                EXPECT_EQ((*y)[r * 3 + c], bits[r * 8 + 1 + 2 * c])
                    << "Slice r=" << r << " c=" << c;
    }
    // Reshape [6, 8] -> [4, 12]: an alias of the input buffer.
    {
        auto x = make({6, 8});
        auto y = out<uint16_t>(e, {4, 12}, ElemKind::kBFloat16);
        auto op = make_op(e, OpType::RESHAPE, 2);
        op->onExecute({x, i64({4, 12})}, {y}, 0);
        submit(e, op.get());
        y->copyToCPU(e.cmdpool);
        EXPECT_EQ(y->elem_kind(), ElemKind::kBFloat16);
        ASSERT_EQ(y->num_elements(), 48);
        for (int i = 0; i < 48; ++i)
            EXPECT_EQ((*y)[i], bits[i]) << "Reshape element " << i;
    }
}

// The same movers inside a loaded graph: Reshape -> Slice -> (output s, and
// Add -> y). LoadModel's gate admits them, the runtime gives their outputs
// the bf16 kind, and the Add after them runs (it would refuse an fp16-kind
// input beside its bf16 bias).
TEST(Bf16ActivationTest, LoadModelMovesBf16BitsThroughByteMovers) {
    Env e;
    auto rt = load("bf16_gate_movers.vkopbin");
    auto x = vkop::core::as_tensor<uint16_t>(rt->GetInput("x"));
    auto s = vkop::core::as_tensor<uint16_t>(rt->GetOutput("s"));
    auto y = vkop::core::as_tensor<uint16_t>(rt->GetOutput("y"));
    ASSERT_TRUE(x && s && y);
    EXPECT_EQ(s->elem_kind(), ElemKind::kBFloat16);
    EXPECT_EQ(y->elem_kind(), ElemKind::kBFloat16);
    std::vector<uint16_t> bits(16);
    for (int i = 0; i < 16; ++i) bits[i] = float_to_bf16(256.0F + 2.0F * i);
    x->fillToCPU(bits);
    x->copyToGPU(e.cmdpool);
    rt->Run();
    rt->ReadResult();
    s->copyToCPU(e.cmdpool);
    const auto got = read_bf16(e, y);
    // x viewed as [4, 4], columns 1..2.
    for (int r = 0; r < 4; ++r) {
        for (int c = 0; c < 2; ++c) {
            const uint16_t src = bits[r * 4 + 1 + c];
            EXPECT_EQ((*s)[r * 2 + c], src) << "s r=" << r << " c=" << c;
            const float b = (static_cast<float>(r * 2 + c) - 3.0F) / 8.0F;
            EXPECT_EQ(got[r * 2 + c],
                      bf16_to_float(float_to_bf16(bf16_to_float(src) + b)))
                << "y r=" << r << " c=" << c;
        }
    }
}
