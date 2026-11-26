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
//   * refusals: a unary op, Transpose, Cast, a Gather writing a bf16 output,
//     an operand-kind mix, MatMul shapes/folds with no bf16 route -- both as
//     single ops and as .vkopbin graphs through Runtime::LoadModel;
//   * the allow-listed kernels against an fp32 reference computed from the
//     same bf16 inputs: a binary op, RMSNorm, Softmax and every MatMul route a
//     bf16 activation may take (including the shapes that would otherwise go
//     to the Apple coop8 GEMM, the 128x128 tile and tile-split-K).

#include <cmath>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "core/Bf16Gate.hpp"
#include "core/DType.hpp"
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
                     OpType::SOFTMAX, OpType::MATMUL, OpType::ATTENTION}) {
        EXPECT_TRUE(bf16_activation_op(t)) << vkop::ops::convert_optype_to_string(t);
    }
    // Each of these has a _bf16 key in the archive (or falls back to the fp32
    // module) and still decodes fp16: the old archive-lookup gate let them in.
    for (OpType t : {OpType::RELU, OpType::SIGMOID, OpType::TRANSPOSE,
                     OpType::EXPAND, OpType::CONCAT, OpType::SLICE,
                     OpType::RESHAPE, OpType::CAST, OpType::GATHER,
                     OpType::GEMM, OpType::FUSED_ELEMWISE, OpType::REDUCE,
                     OpType::ROTARY_EMBEDDING, OpType::CLIP, OpType::CONV2D,
                     OpType::INSTANCENORM, OpType::CUMSUM, OpType::TOPK}) {
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

TEST(Bf16ActivationTest, UnaryOpRefusesBf16) {
    Env e;
    auto x = bf16_in(e, {4, 8}, bf16_values(32, 1));
    auto y = out<uint16_t>(e, {4, 8}, ElemKind::kBFloat16);
    auto op = make_op(e, OpType::RELU, 2);
    expect_refused([&] { op->onExecute({x}, {y}, 0); },
                   "vkop: no bf16 activation kernel for Relu");
}

TEST(Bf16ActivationTest, TransposeRefusesBf16) {
    Env e;
    auto x = bf16_in(e, {4, 8}, bf16_values(32, 2));
    auto y = out<uint16_t>(e, {8, 4}, ElemKind::kBFloat16);
    auto op = make_op(e, OpType::TRANSPOSE, 2, {{"perm", "[1, 0]"}});
    expect_refused([&] { op->onExecute({x}, {y}, 0); },
                   "vkop: no bf16 activation kernel for Transpose");
}

TEST(Bf16ActivationTest, CastRefusesBf16) {
    // cast.comp's fp16 -> fp32 path would decode the bf16 words as fp16.
    Env e;
    auto x = bf16_in(e, {4, 8}, bf16_values(32, 3));
    auto y = out<float>(e, {4, 8}, ElemKind::kFloat32);
    auto op = make_op(e, OpType::CAST, 0, {{"to", "1"}});
    expect_refused([&] { op->onExecute({x}, {y}, 0); },
                   "vkop: no bf16 activation kernel for Cast");
}

TEST(Bf16ActivationTest, GatherWithBf16OutputRefused) {
    // A bf16 TABLE is a weight and stays legal with an fp16/fp32 output
    // (GatherTest.GatherBf16Table); a bf16 OUTPUT is an activation, and
    // gather.comp -DBF16_TABLE writes fp32/fp16 only.
    Env e;
    auto table = bf16_in(e, {4, 8}, bf16_values(32, 4));
    auto idx = std::make_shared<Tensor<int64_t>>(std::vector<int>{2});
    idx->fillToCPU(std::vector<int64_t>{3, 1});
    auto keep = idx->data();
    idx->as_storage_buffer(e.dev);
    idx->copyToGPU(e.cmdpool, keep.data());
    auto y = out<uint16_t>(e, {2, 8}, ElemKind::kBFloat16);
    auto op = make_op(e, OpType::GATHER, 2, {{"axis", "0"}});
    expect_refused([&] { op->onExecute({table, idx}, {y}, 0); },
                   "vkop: no bf16 activation kernel for Gather");
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
    // Odd N: the only route is reduce_pack, whose pack pass writes fp16.
    {
        auto a = bf16_in(e, {4, 64}, bf16_values(4 * 64, 9));
        auto b = bf16_in(e, {64, 7}, bf16_values(64 * 7, 10));
        auto y = out<uint16_t>(e, {4, 7}, ElemKind::kBFloat16);
        auto op = make_op(e, OpType::MATMUL, 2);
        expect_refused([&] { op->onExecute({a, b}, {y}, 0); },
                       "vkop: no bf16 activation kernel for MatMul");
    }
    // The QK^T epilogue (mask/bias read) has no bf16 form.
    {
        auto a = bf16_in(e, {4, 64}, bf16_values(4 * 64, 11));
        auto b = bf16_in(e, {64, 8}, bf16_values(64 * 8, 12));
        auto bias = bf16_in(e, {4, 8}, bf16_values(32, 13));
        auto y = out<uint16_t>(e, {4, 8}, ElemKind::kBFloat16);
        auto op = make_op(e, OpType::MATMUL, 2,
                          {{"epi_bias", "1"}, {"epi_scale", "0.5"}});
        expect_refused([&] { op->onExecute({a, b, bias}, {y}, 0); },
                       "epi_bias");
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

TEST(Bf16ActivationTest, MatMulRoutesMatchFp32) {
    Env e;
    struct Case {
        int batch, m, k, n;
        bool transB;
        const char *why;
    };
    const std::vector<Case> cases = {
        {1, 1, 512, 256, false, "decode: split-K GEMV"},
        {1, 1, 258, 260, false, "decode: split-K GEMV, K tail"},
        {1, 1, 512, 4096, false, "decode: WIDE split-K GEMV"},
        {1, 3, 64, 12, false, "even N, K < 256: fused even-N kernel"},
        {1, 2, 64, 12, true, "fused even-N, transB"},
        {2, 3, 32, 10, false, "batched A, shared 2-D weight"},
        {1, 20, 128, 96, false, "64x64 SIMT tile"},
        {1, 20, 128, 96, true, "64x64 SIMT tile, transB"},
        {1, 16, 1024, 64, false, "fp16 would take tile split-K"},
        {1, 130, 256, 192, false, "fp16 would take the Apple coop8 GEMM"},
        {3, 70, 128, 64, false, "coop8 shape, batched A"},
        {1, 256, 64, 2056, false, "fp16 would take the 128x128 tile"},
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
        const auto av = bf16_values(numel(ashape), 51 + c.m);
        const auto bv = bf16_values(numel(bshape), 52 + c.n);
        auto a = bf16_in(e, ashape, av);
        auto b = bf16_in(e, bshape, bv);
        auto y = out<uint16_t>(e, yshape, ElemKind::kBFloat16);
        auto op = make_op(e, OpType::MATMUL, 2,
                          {{"transB", c.transB ? "1" : "0"}});
        op->onExecute({a, b}, {y}, 0);
        submit(e, op.get());
        const std::string route = op->route_tag();
        EXPECT_TRUE(route.rfind("coop", 0) != 0 && route != "tiled_ks" &&
                    route != "reduce_pack" && route.find("512") == std::string::npos)
            << c.why << ": bf16 took route " << route;
        std::vector<double> ref(numel(yshape));
        for (int bt = 0; bt < c.batch; ++bt)
            for (int i = 0; i < c.m; ++i)
                for (int j = 0; j < c.n; ++j) {
                    double s = 0;
                    for (int kk = 0; kk < c.k; ++kk) {
                        const float bvv = c.transB ? bv[j * c.k + kk]
                                                   : bv[kk * c.n + j];
                        s += double(av[(bt * c.m + i) * c.k + kk]) * bvv;
                    }
                    ref[(bt * c.m + i) * c.n + j] = s;
                }
        double worst = 0;
        const int bad = count_bad(read_bf16(e, y), ref, 1.6e-2, 0.03, &worst);
        EXPECT_EQ(bad, 0) << c.why << " (route " << route << ") m=" << c.m
                          << " k=" << c.k << " n=" << c.n << " worst " << worst;
        LOG_INFO("bf16 MatMul %s: route %s, worst |err| %g", c.why,
                 route.c_str(), worst);
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
    expect_refused([] { load("bf16_gate_relu.vkopbin"); },
                   "vkop: no bf16 activation kernel for Relu 'relu0'");
    expect_refused([] { load("bf16_gate_cast.vkopbin"); },
                   "vkop: no bf16 activation kernel for Cast 'cast0'");
    expect_refused([] { load("bf16_gate_gather.vkopbin"); },
                   "vkop: no bf16 activation kernel for Gather 'gather0'");
    expect_refused([] { load("bf16_gate_mixed.vkopbin"); },
                   "vkop: Add 'add_mixed' mixes bfloat16 activations with a "
                   "float16 input 1");
}
