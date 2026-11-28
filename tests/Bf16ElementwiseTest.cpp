// Copyright 2026 @junka
// bfloat16 ACTIVATION kernels for the elementwise / mover ops GLM-Edge's graph
// needs: the unary family, Clip, Cast, FusedElemwise, Gather with a bf16
// output, Expand and Concat.
//
// Every case runs the op's real bf16 route (the op created with fp16 = 2, as
// Runtime does for a bf16 activation) against an fp32/double reference taken
// from the SAME bf16-rounded inputs, so the only error allowed is the kernel's
// own arithmetic plus the bf16 rounding of the result. The inputs include what
// bf16 exists for and fp16 cannot hold -- magnitudes past 65504 -- plus NaN,
// +-Inf, -0 and an odd element count (the tail half-word). Decoding a bf16 word
// with the fp16 unpack (the bug these kernels had) is off by orders of
// magnitude on every one of them.

#include <cmath>
#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "core/Bf16Gate.hpp"
#include "core/DType.hpp"
#include "ops/OperatorFactory.hpp"
#include "setup.hpp"

#include <gtest/gtest.h>

namespace bf16ew {

using vkop::core::bf16_to_float;
using vkop::core::ElemKind;
using vkop::core::float_to_bf16;
using vkop::core::ITensor;
using vkop::core::Tensor;
using vkop::ops::OpType;
using Attrs = std::unordered_map<std::string, std::string>;

constexpr float kInf = std::numeric_limits<float>::infinity();
const float kNaN = std::numeric_limits<float>::quiet_NaN();

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

float rbf(float v) { return bf16_to_float(float_to_bf16(v)); }

// fp16 conversions from the compiler's own IEEE half (RNE, NaN kept NaN): the
// host helper's portable branch maps NaN to Inf.
uint16_t f32_to_f16(float f) {
    _Float16 h = static_cast<_Float16>(f);
    uint16_t b;
    std::memcpy(&b, &h, sizeof(b));
    return b;
}
float f16_to_f32(uint16_t b) {
    _Float16 h;
    std::memcpy(&h, &b, sizeof(b));
    return static_cast<float>(h);
}

// Deterministic values in [-scale, scale), rounded to bf16.
std::vector<float> values(size_t n, uint32_t seed, float scale) {
    std::vector<float> v(n);
    uint32_t s = seed * 2654435761u + 12345u;
    for (size_t i = 0; i < n; ++i) {
        s = s * 1664525u + 1013904223u;
        const float u = static_cast<float>(s >> 8) / 16777216.0F;
        v[i] = rbf(scale * (2.0F * u - 1.0F));
    }
    return v;
}

// The edge cases, spliced over the front of a vector: past fp16's ceiling,
// fp32-range magnitudes, -0, Inf, NaN, tiny values.
std::vector<float> with_specials(std::vector<float> v) {
    const float sp[] = {1.0F,   2.0F,   -3.5F,   100.0F,  65504.0F, 70000.0F,
                        -1e5F,  3e38F,  -2e30F,  -0.0F,   0.0F,     kInf,
                        -kInf,  kNaN,   1e-30F,  -7e-20F, 0.5F,     -0.5F};
    for (size_t i = 0; i < sizeof(sp) / sizeof(sp[0]) && i < v.size(); ++i)
        v[i] = rbf(sp[i]);
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
    for (size_t i = 0; i < vals.size(); ++i) bits[i] = f32_to_f16(vals[i]);
    t->fillToCPU(bits);
    t->as_storage_buffer(e.dev);
    t->copyToGPU(e.cmdpool);
    return t;
}

std::shared_ptr<Tensor<float>> fp32_in(Env &e, const std::vector<int> &shape,
                                       const std::vector<float> &vals) {
    auto t = std::make_shared<Tensor<float>>(shape);
    t->fillToCPU(vals);
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

std::vector<uint16_t> read_bits(Env &e, const std::shared_ptr<Tensor<uint16_t>> &t) {
    t->copyToCPU(e.cmdpool);
    std::vector<uint16_t> v(t->num_elements());
    for (size_t i = 0; i < v.size(); ++i) v[i] = (*t)[i];
    return v;
}

std::vector<float> read_bf16(Env &e, const std::shared_ptr<Tensor<uint16_t>> &t) {
    auto b = read_bits(e, t);
    std::vector<float> v(b.size());
    for (size_t i = 0; i < v.size(); ++i) v[i] = bf16_to_float(b[i]);
    return v;
}

std::vector<float> read_f32(Env &e, const std::shared_ptr<Tensor<float>> &t) {
    t->copyToCPU(e.cmdpool);
    std::vector<float> v(t->num_elements());
    for (size_t i = 0; i < v.size(); ++i) v[i] = (*t)[i];
    return v;
}

std::unique_ptr<vkop::ops::Operator> make_op(Env &e, OpType type, int fp16,
                                             const Attrs &attrs = {}) {
    auto op = vkop::ops::create_from_type(type, fp16, 0,
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

// One element against its reference. NaN must be NaN, an infinity must be the
// same infinity, a finite value within rtol*|ref| + atol (the reference is
// first rounded to bf16, which is what the kernel stores).
bool close(float got, double ref, double rtol, double atol) {
    if (std::isnan(ref)) return std::isnan(got);
    const float rb = rbf(static_cast<float>(ref));
    if (std::isinf(rb)) return got == rb;
    if (!std::isfinite(got)) return false;
    return std::fabs(double(got) - rb) <= rtol * std::fabs(rb) + atol;
}

int count_bad(const std::vector<float> &got, const std::vector<double> &ref,
              double rtol, double atol, std::string *first) {
    int bad = 0;
    for (size_t i = 0; i < ref.size(); ++i) {
        if (!close(got[i], ref[i], rtol, atol)) {
            if (bad++ == 0 && first) {
                *first = "element " + std::to_string(i) + ": got " +
                         std::to_string(got[i]) + " want " +
                         std::to_string(ref[i]);
            }
        }
    }
    return bad;
}

} // namespace bf16ew

using namespace bf16ew;

// ------------------------------------------------------------------- gate

TEST(Bf16ElementwiseTest, AllowListHasTheseKernels) {
    using vkop::core::bf16_activation_op;
    for (OpType t :
         {OpType::SIN, OpType::COS, OpType::SIGMOID, OpType::TANH, OpType::ERF,
          OpType::SQRT, OpType::RELU, OpType::NEG, OpType::FLOOR, OpType::ATAN,
          OpType::SOFTPLUS, OpType::CLIP, OpType::FUSED_ELEMWISE, OpType::CAST,
          OpType::GATHER, OpType::EXPAND, OpType::CONCAT}) {
        EXPECT_TRUE(bf16_activation_op(t))
            << vkop::ops::convert_optype_to_string(t);
    }
    // The per-operand slots: anything float there is the kernel's own read.
    using vkop::core::bf16_other_float_ok;
    using vkop::core::ElemKind;
    for (ElemKind k : {ElemKind::kFloat16, ElemKind::kFloat32}) {
        EXPECT_TRUE(bf16_other_float_ok(OpType::CAST, false, 0, k));
        EXPECT_TRUE(bf16_other_float_ok(OpType::CAST, true, 0, k));
        EXPECT_TRUE(bf16_other_float_ok(OpType::FUSED_ELEMWISE, false, 3, k));
        EXPECT_TRUE(bf16_other_float_ok(OpType::GATHER, false, 0, k));
        EXPECT_FALSE(bf16_other_float_ok(OpType::GATHER, false, 1, k));
        EXPECT_FALSE(bf16_other_float_ok(OpType::GATHER, true, 0, k));
        EXPECT_TRUE(bf16_other_float_ok(OpType::CLIP, false, 1, k));
        EXPECT_FALSE(bf16_other_float_ok(OpType::CLIP, false, 0, k));
        EXPECT_FALSE(bf16_other_float_ok(OpType::ADD, false, 1, k));
    }
}

// --------------------------------------------------------------- unary family

TEST(Bf16ElementwiseTest, UnaryFamilyMatchesFp32) {
    Env e;
    struct U {
        OpType type;
        double (*ref)(double);
        float scale; // input magnitude of the random part
        double rtol, atol;
        bool nan_ok; // the GPU result for a NaN input is defined
    };
    const std::vector<U> ops = {
        {OpType::NEG, [](double x) { return -x; }, 1e6F, 0, 0, true},
        {OpType::RELU, [](double x) { return x > 0 ? x : 0.0; }, 1e6F, 0, 0,
         false},
        {OpType::FLOOR, [](double x) { return std::floor(x); }, 300.0F, 0, 0,
         true},
        {OpType::SQRT, [](double x) { return std::sqrt(x); }, 1e6F, 8e-3, 0,
         true},
        {OpType::SIGMOID, [](double x) { return 1.0 / (1.0 + std::exp(-x)); },
         20.0F, 8e-3, 1e-6, true},
        {OpType::TANH, [](double x) { return std::tanh(x); }, 8.0F, 8e-3, 1e-5,
         true},
        {OpType::ERF, [](double x) { return std::erf(x); }, 4.0F, 8e-3, 2e-6,
         true},
        {OpType::ATAN, [](double x) { return std::atan(x); }, 1e3F, 8e-3,
         1e-6, true},
        {OpType::SOFTPLUS,
         [](double x) {
             return std::max(x, 0.0) + std::log1p(std::exp(-std::fabs(x)));
         },
         200.0F, 8e-3, 1e-6, true},
        {OpType::SIN, [](double x) { return std::sin(x); }, 100.0F, 8e-3,
         2e-3, false},
        {OpType::COS, [](double x) { return std::cos(x); }, 100.0F, 8e-3,
         2e-3, false},
    };
    // 37 * 3 = 111: odd, so the last word carries one element.
    const std::vector<int> shape = {3, 37};
    for (const U &u : ops) {
        const std::string name = vkop::ops::convert_optype_to_string(u.type);
        auto xv = with_specials(values(numel(shape), 7, u.scale));
        auto x = bf16_in(e, shape, xv);
        auto y = out<uint16_t>(e, shape, ElemKind::kBFloat16);
        auto op = make_op(e, u.type, 2);
        op->onExecute({x}, {y}, 0);
        submit(e, op.get());
        const auto got = read_bf16(e, y);
        std::vector<double> ref(xv.size());
        for (size_t i = 0; i < xv.size(); ++i) ref[i] = u.ref(xv[i]);
        // Sin/Cos of a huge argument (1e5, 3e38, ...) depends on the
        // device's range reduction; GLSL leaves it unspecified. Torch's
        // reference phases are in range, so only the in-range part is held to
        // a tolerance and the rest must merely be within [-1, 1]. Of a
        // non-finite argument it is undefined too (MoltenVK's fast math
        // returns 0 for sin(+-Inf) / sin(NaN), in every build), so those are
        // skipped.
        int bad = 0;
        std::string first;
        for (size_t i = 0; i < xv.size(); ++i) {
            const bool trig = u.type == OpType::SIN || u.type == OpType::COS;
            if (std::isnan(xv[i]) && !u.nan_ok) continue;
            if (trig && !std::isfinite(xv[i])) continue;
            if (trig && std::isfinite(xv[i]) && std::fabs(xv[i]) > 1000.0F) {
                if (!(got[i] >= -1.0F && got[i] <= 1.0F)) {
                    if (bad++ == 0)
                        first = "huge-arg element " + std::to_string(i);
                }
                continue;
            }
            if (!close(got[i], ref[i], u.rtol, u.atol)) {
                if (bad++ == 0) {
                    first = "element " + std::to_string(i) + " x=" +
                            std::to_string(xv[i]) + ": got " +
                            std::to_string(got[i]) + " want " +
                            std::to_string(ref[i]);
                }
            }
        }
        EXPECT_EQ(bad, 0) << name << " bf16: " << first;
        // Neg is a sign flip: bit exact, -0 included.
        if (u.type == OpType::NEG) {
            const auto bits = read_bits(e, y);
            for (size_t i = 0; i < xv.size(); ++i) {
                if (std::isnan(xv[i])) continue;
                EXPECT_EQ(bits[i], float_to_bf16(-xv[i])) << "Neg element " << i;
            }
        }
    }
}

// The review's reproducer: buffer_neg_bf16 on [1, 2, -3.5, 100, ...] gave
// [-1.875, -2, 2.1875, -3.39, ...] -- an fp16 decode of bf16 words.
TEST(Bf16ElementwiseTest, NegIsNotAnFp16Decode) {
    Env e;
    const std::vector<float> xv = {1.0F, 2.0F, -3.5F, 100.0F, 7.0F};
    auto x = bf16_in(e, {5}, xv);
    auto y = out<uint16_t>(e, {5}, ElemKind::kBFloat16);
    auto op = make_op(e, OpType::NEG, 2);
    op->onExecute({x}, {y}, 0);
    submit(e, op.get());
    const auto got = read_bf16(e, y);
    for (size_t i = 0; i < xv.size(); ++i) EXPECT_EQ(got[i], -xv[i]) << i;
}

TEST(Bf16ElementwiseTest, ClipMatchesFp32) {
    Env e;
    const std::vector<int> shape = {5, 13}; // odd
    auto xv = with_specials(values(numel(shape), 11, 2e5F));
    // Bounds past fp16's range, given as fp32 and bf16 scalars: Clip reads
    // them on the host by their own kind.
    for (bool bf_bounds : {false, true}) {
        auto x = bf16_in(e, shape, xv);
        std::shared_ptr<ITensor> lo, hi;
        if (bf_bounds) {
            lo = bf16_in(e, {1}, {-70000.0F});
            hi = bf16_in(e, {1}, {1e5F});
        } else {
            lo = fp32_in(e, {1}, {-70000.0F});
            hi = fp32_in(e, {1}, {1e5F});
        }
        auto y = out<uint16_t>(e, shape, ElemKind::kBFloat16);
        auto op = make_op(e, OpType::CLIP, 2);
        op->onExecute({x, lo, hi}, {y}, 0);
        submit(e, op.get());
        const auto got = read_bf16(e, y);
        const double l = rbf(-70000.0F), h = rbf(1e5F);
        std::vector<double> ref(xv.size());
        for (size_t i = 0; i < xv.size(); ++i)
            ref[i] = std::isnan(xv[i]) ? xv[i]
                                       : std::min(std::max(double(xv[i]), l), h);
        std::string first;
        EXPECT_EQ(count_bad(got, ref, 0, 0, &first), 0)
            << "Clip bf16 (bounds " << (bf_bounds ? "bf16" : "fp32")
            << "): " << first;
    }
}

// ---------------------------------------------------------------------- Cast

TEST(Bf16ElementwiseTest, CastFp32ToBf16RoundsToNearestEven) {
    Env e;
    for (int n : {64, 37}) {
        std::vector<float> xv = values(n, 21, 1e5F);
        // Unrounded inputs: the cast does the rounding. Ties (0x3F808000 is
        // exactly half way between two bf16), NaN, Inf, -0, past 65504.
        for (auto &v : xv) v *= 1.0001F;
        const uint32_t ties[] = {0x3F808000u, 0x3F818000u, 0xC0008000u};
        for (int i = 0; i < 3; ++i) std::memcpy(&xv[i], &ties[i], 4);
        xv[3] = kNaN;
        xv[4] = -kInf;
        xv[5] = -0.0F;
        xv[6] = 3.3e38F; // rounds past the largest bf16: +Inf
        xv[7] = 70001.0F;
        auto x = fp32_in(e, {n}, xv);
        auto y = out<uint16_t>(e, {n}, ElemKind::kBFloat16);
        auto op = make_op(e, OpType::CAST, 0, {{"to", "16"}});
        op->onExecute({x}, {y}, 0);
        submit(e, op.get());
        const auto bits = read_bits(e, y);
        for (int i = 0; i < n; ++i)
            EXPECT_EQ(bits[i], float_to_bf16(xv[i]))
                << "n=" << n << " element " << i << " x=" << xv[i];
    }
}

TEST(Bf16ElementwiseTest, CastBf16ToFp32IsExact) {
    Env e;
    const int n = 37;
    auto xv = with_specials(values(n, 22, 3e5F));
    auto x = bf16_in(e, {n}, xv);
    auto y = out<float>(e, {n}, ElemKind::kFloat32);
    auto op = make_op(e, OpType::CAST, 2, {{"to", "1"}});
    op->onExecute({x}, {y}, 0);
    submit(e, op.get());
    const auto got = read_f32(e, y);
    for (int i = 0; i < n; ++i) {
        if (std::isnan(xv[i])) {
            EXPECT_TRUE(std::isnan(got[i])) << i;
        } else {
            uint32_t a, b;
            std::memcpy(&a, &got[i], 4);
            std::memcpy(&b, &xv[i], 4);
            EXPECT_EQ(a, b) << "element " << i << " x=" << xv[i];
        }
    }
}

TEST(Bf16ElementwiseTest, CastBetweenFp16AndBf16) {
    Env e;
    const int n = 37;
    // fp16 -> bf16: every fp16 value, RNE to bf16's 8 bits.
    {
        std::vector<float> xv = values(n, 23, 60000.0F);
        for (auto &v : xv) v = f16_to_f32(f32_to_f16(v * 1.0003F));
        xv[0] = kNaN;
        xv[1] = -kInf;
        xv[2] = -0.0F;
        xv[3] = 65504.0F;
        xv[4] = f16_to_f32(1); // smallest fp16 subnormal
        auto x = fp16_in(e, {n}, xv);
        auto y = out<uint16_t>(e, {n}, ElemKind::kBFloat16);
        auto op = make_op(e, OpType::CAST, 1, {{"to", "16"}});
        op->onExecute({x}, {y}, 0);
        submit(e, op.get());
        const auto bits = read_bits(e, y);
        for (int i = 0; i < n; ++i) {
            if (std::isnan(xv[i])) {
                EXPECT_TRUE(std::isnan(bf16_to_float(bits[i]))) << i;
                continue;
            }
            EXPECT_EQ(bits[i], float_to_bf16(xv[i]))
                << "fp16->bf16 element " << i << " x=" << xv[i];
        }
    }
    // bf16 -> fp16: in-range values round to fp16 (exact: 8 bits fit in 11);
    // past 65504 the result is +-Inf, like torch's .to(float16).
    {
        auto xv = values(n, 24, 60000.0F);
        xv[0] = kNaN;
        xv[1] = kInf;
        xv[2] = -0.0F;
        xv[3] = rbf(70000.0F);
        xv[4] = rbf(-1e6F);
        auto x = bf16_in(e, {n}, xv);
        auto y = out<uint16_t>(e, {n}, ElemKind::kFloat16);
        auto op = make_op(e, OpType::CAST, 2, {{"to", "10"}});
        op->onExecute({x}, {y}, 0);
        submit(e, op.get());
        y->copyToCPU(e.cmdpool);
        for (int i = 0; i < n; ++i) {
            const float got = f16_to_f32((*y)[i]);
            if (std::isnan(xv[i])) {
                EXPECT_TRUE(std::isnan(got)) << i;
                continue;
            }
            EXPECT_EQ((*y)[i], f32_to_f16(xv[i]))
                << "bf16->fp16 element " << i << " x=" << xv[i]
                << " got " << got;
        }
    }
}

TEST(Bf16ElementwiseTest, CastBf16ToBf16IsAWordCopy) {
    Env e;
    const int n = 37;
    auto xv = with_specials(values(n, 25, 1e6F));
    auto x = bf16_in(e, {n}, xv);
    auto y = out<uint16_t>(e, {n}, ElemKind::kBFloat16);
    auto op = make_op(e, OpType::CAST, 2, {{"to", "16"}});
    op->onExecute({x}, {y}, 0);
    submit(e, op.get());
    const auto bits = read_bits(e, y);
    for (int i = 0; i < n; ++i) EXPECT_EQ(bits[i], float_to_bf16(xv[i])) << i;
}

TEST(Bf16ElementwiseTest, CastToAttributeMustMatchABf16Output) {
    Env e;
    auto x = fp32_in(e, {4}, {1, 2, 3, 4});
    auto y = out<uint16_t>(e, {4}, ElemKind::kBFloat16);
    auto op = make_op(e, OpType::CAST, 0, {{"to", "10"}});
    try {
        op->onExecute({x}, {y}, 0);
        ADD_FAILURE() << "to=10 with a bf16 output must be refused";
    } catch (const std::runtime_error &err) {
        EXPECT_NE(std::string(err.what()).find("to=10"), std::string::npos)
            << err.what();
    }
}

// ------------------------------------------------------------- FusedElemwise

namespace bf16ew {
// Runs one FusedElemwise program; inputs are pre-built tensors.
void run_fused(Env &e, const std::string &ops, const std::string &scalars,
               const std::vector<std::shared_ptr<ITensor>> &ins,
               const std::shared_ptr<ITensor> &y, int fp16) {
    const auto shp = y->getShape();
    std::string out_shp = "[";
    for (size_t i = 0; i < shp.size(); ++i)
        out_shp += (i ? "," : "") + std::to_string(shp[i]);
    out_shp += "]";
    auto op = make_op(e, OpType::FUSED_ELEMWISE, fp16,
                      {{"ops", ops},
                       {"input_shapes", "[]"},
                       {"out_shape", out_shp},
                       {"rank", std::to_string(shp.size())},
                       {"scalars", scalars}});
    op->onExecute(ins, {y}, 0);
    submit(e, op.get());
}
} // namespace bf16ew

// GLM-Edge's MLP: silu(gate) * up, both bf16 -- the chain
// fuse_elemwise_chain emits for the SwiGLU block ({SILU, MUL}).
TEST(Bf16ElementwiseTest, FusedSiluMulMatchesFp32) {
    Env e;
    for (const std::vector<int> &shape :
         {std::vector<int>{4, 64}, std::vector<int>{3, 37}}) {
        auto gv = with_specials(values(numel(shape), 31, 12.0F));
        auto uv = values(numel(shape), 32, 9000.0F); // products past 65504
        gv[0] = rbf(-200.0F);                        // exp(-x) overflows
        gv[1] = rbf(200.0F);
        auto g = bf16_in(e, shape, gv);
        auto u = bf16_in(e, shape, uv);
        auto y = out<uint16_t>(e, shape, ElemKind::kBFloat16);
        // reg2 = silu(reg0); reg0 = reg2 * reg1.
        run_fused(e, "[14,2,0,0, 3,0,2,1]", "[]", {g, u}, y, 2);
        const auto got = read_bf16(e, y);
        std::vector<double> ref(gv.size());
        for (size_t i = 0; i < gv.size(); ++i) {
            const double x = gv[i];
            ref[i] = x / (1.0 + std::exp(-x)) * uv[i]; // silu(-inf) = NaN
        }
        std::string first;
        EXPECT_EQ(count_bad(got, ref, 1.2e-2, 1e-4, &first), 0)
            << "silu*mul bf16 " << shape.size() << "-D: " << first;
    }
}

// The rotary table's tail: an fp32 cos/sin (computed wide) scaled and cast to
// the activation dtype -- a fused {MUL scalar, CAST} with an fp32 leaf and a
// bf16 terminal.
TEST(Bf16ElementwiseTest, FusedFp32LeafToBf16Terminal) {
    Env e;
    const std::vector<int> shape = {1, 7, 1, 33}; // odd total
    std::vector<float> xv(numel(shape));
    for (size_t i = 0; i < xv.size(); ++i)
        xv[i] = std::cos(0.37F * static_cast<float>(i)) * 1.0001F;
    xv[3] = 80000.0F;
    xv[4] = kNaN;
    auto x = fp32_in(e, shape, xv);
    auto y = out<uint16_t>(e, shape, ElemKind::kBFloat16);
    const float scale = 1.25F;
    uint32_t sb;
    std::memcpy(&sb, &scale, 4);
    // reg1 = reg0 * scalar0; reg0 = cast(reg1).
    run_fused(e, "[3,1,0,100, 11,0,1,0]", "[1.25]", {x}, y, 2);
    const auto bits = read_bits(e, y);
    for (size_t i = 0; i < xv.size(); ++i) {
        const float want = xv[i] * scale;
        if (std::isnan(want)) {
            EXPECT_TRUE(std::isnan(bf16_to_float(bits[i]))) << i;
            continue;
        }
        EXPECT_EQ(bits[i], float_to_bf16(want)) << "element " << i;
    }
}

// The other direction: a bf16 leaf read by an fp32-terminal chain (fp32 build,
// per-input dtype code 2), and an fp16 leaf beside a bf16 one in a bf16 chain.
TEST(Bf16ElementwiseTest, FusedMixedLeaves) {
    Env e;
    const std::vector<int> shape = {5, 9};
    auto av = with_specials(values(numel(shape), 41, 3e5F));
    {
        auto a = bf16_in(e, shape, av);
        auto y = out<float>(e, shape, ElemKind::kFloat32);
        // reg0 = reg0 * 2 (an fp32 chain over a bf16 leaf).
        run_fused(e, "[3,0,0,100]", "[2.0]", {a}, y, 0);
        const auto got = read_f32(e, y);
        for (size_t i = 0; i < av.size(); ++i) {
            const float want = av[i] * 2.0F;
            if (std::isnan(want)) {
                EXPECT_TRUE(std::isnan(got[i])) << i;
            } else {
                EXPECT_EQ(got[i], want) << "bf16 leaf -> fp32, element " << i;
            }
        }
    }
    {
        auto bv = values(numel(shape), 42, 1000.0F);
        auto a = bf16_in(e, shape, av);
        auto b = fp16_in(e, shape, bv);
        auto y = out<uint16_t>(e, shape, ElemKind::kBFloat16);
        // reg0 = reg0 + reg1.
        run_fused(e, "[1,0,0,1]", "[]", {a, b}, y, 2);
        const auto got = read_bf16(e, y);
        std::vector<double> ref(av.size());
        for (size_t i = 0; i < av.size(); ++i)
            ref[i] = double(av[i]) + f16_to_f32(f32_to_f16(bv[i]));
        std::string first;
        EXPECT_EQ(count_bad(got, ref, 8e-3, 0, &first), 0)
            << "bf16 + fp16 leaves -> bf16: " << first;
    }
}

// Broadcast operands, every load path: same shape (word load), a per-row
// column (affine), a scalar, and a non-affine middle-axis broadcast.
TEST(Bf16ElementwiseTest, FusedBroadcastsMatchFp32) {
    Env e;
    const std::vector<int> ys = {3, 5, 7}; // 105, odd
    const std::vector<int> as = {3, 5, 7}, bs = {3, 1, 1}, cs = {1},
                           ds = {3, 1, 7};
    auto av = with_specials(values(numel(as), 51, 1e4F));
    auto bv = values(numel(bs), 52, 50.0F);
    auto cv = values(1, 53, 3.0F);
    auto dv = values(numel(ds), 54, 1e3F);
    auto a = bf16_in(e, as, av);
    auto b = bf16_in(e, bs, bv);
    auto c = bf16_in(e, cs, cv);
    auto d = bf16_in(e, ds, dv);
    auto y = out<uint16_t>(e, ys, ElemKind::kBFloat16);
    // reg4 = a*b; reg4 = reg4 + c; reg0 = reg4 - d.
    run_fused(e, "[3,4,0,1, 1,4,4,2, 2,0,4,3]", "[]", {a, b, c, d}, y, 2);
    const auto got = read_bf16(e, y);
    std::vector<double> ref(numel(ys));
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 5; ++j)
            for (int k = 0; k < 7; ++k) {
                const size_t o = (i * 5 + j) * 7 + k;
                // fp32 register arithmetic, like the kernel.
                float r = av[o] * bv[i];
                r = r + cv[0];
                r = r - dv[i * 7 + k];
                ref[o] = r;
            }
    std::string first;
    EXPECT_EQ(count_bad(got, ref, 8e-3, 0, &first), 0) << first;
}

// ---------------------------------------------------------------------- Gather

TEST(Bf16ElementwiseTest, GatherBf16OutputFromEveryTable) {
    Env e;
    const int rows = 6, cols = 9; // odd row length
    auto tv = with_specials(values(rows * cols, 61, 1e5F));
    auto idx = std::make_shared<Tensor<int64_t>>(std::vector<int>{3});
    idx->fillToCPU(std::vector<int64_t>{5, 0, -4}); // -4 wraps to row 2
    auto keep = idx->data();
    idx->as_storage_buffer(e.dev);
    idx->copyToGPU(e.cmdpool, keep.data());
    const int rows_out[3] = {5, 0, 2};
    for (int fmt = 0; fmt < 3; ++fmt) {
        std::shared_ptr<ITensor> table;
        std::vector<float> want_rowsrc(tv);
        if (fmt == 0) {
            table = bf16_in(e, {rows, cols}, tv);
        } else if (fmt == 1) {
            // fp16 can't hold the large values; keep them in fp16 range.
            for (auto &v : want_rowsrc)
                v = std::isfinite(v) && std::fabs(v) > 60000.0F
                        ? std::copysign(1000.5F, v)
                        : v * 1.0001F;
            for (auto &v : want_rowsrc) v = f16_to_f32(f32_to_f16(v));
            table = fp16_in(e, {rows, cols}, want_rowsrc);
        } else {
            for (auto &v : want_rowsrc) v *= 1.0001F; // unrounded
            table = fp32_in(e, {rows, cols}, want_rowsrc);
        }
        auto y = out<uint16_t>(e, {3, cols}, ElemKind::kBFloat16);
        auto op = make_op(e, OpType::GATHER, 2, {{"axis", "0"}});
        op->onExecute({table, idx}, {y}, 0);
        submit(e, op.get());
        const auto bits = read_bits(e, y);
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < cols; ++c) {
                const float src = want_rowsrc[rows_out[r] * cols + c];
                const uint16_t got = bits[r * cols + c];
                if (std::isnan(src)) {
                    EXPECT_TRUE(std::isnan(bf16_to_float(got)));
                    continue;
                }
                EXPECT_EQ(got, float_to_bf16(src))
                    << "table fmt " << fmt << " row " << r << " col " << c;
            }
    }
}

// bf16 table -> bf16 output along axis 0 with rows of 16-byte chunks takes
// gather.comp's row fast path (one thread per chunk): bit-exact rows, a
// negative index wraps once, an out-of-range index gives a zero row.
TEST(Bf16ElementwiseTest, GatherBf16RowFastPath) {
    Env e;
    const int rows = 6, cols = 40;
    auto tv = with_specials(values(rows * cols, 67, 1e5F));
    auto table = bf16_in(e, {rows, cols}, tv);
    const std::vector<int64_t> iv{5, 0, -4, 6, -7, 3, 3};
    auto idx = std::make_shared<Tensor<int64_t>>(
        std::vector<int>{static_cast<int>(iv.size())});
    idx->fillToCPU(iv);
    auto keep = idx->data();
    idx->as_storage_buffer(e.dev);
    idx->copyToGPU(e.cmdpool, keep.data());
    const int n = static_cast<int>(iv.size());
    auto y = out<uint16_t>(e, {n, cols}, ElemKind::kBFloat16);
    auto op = make_op(e, OpType::GATHER, 2, {{"axis", "0"}});
    op->onExecute({table, idx}, {y}, 0);
    submit(e, op.get());
    const auto bits = read_bits(e, y);
    for (int r = 0; r < n; ++r) {
        int64_t src = iv[r] < 0 ? iv[r] + rows : iv[r];
        for (int c = 0; c < cols; ++c) {
            const uint16_t got = bits[r * cols + c];
            if (src < 0 || src >= rows) {
                EXPECT_EQ(got, 0) << "row " << r << " col " << c;
                continue;
            }
            const float v = tv[src * cols + c];
            if (std::isnan(v)) {
                EXPECT_TRUE(std::isnan(bf16_to_float(got)));
                continue;
            }
            EXPECT_EQ(got, float_to_bf16(v)) << "row " << r << " col " << c;
        }
    }
}

// The WEIGHT role is unchanged: a bf16 table with an fp16/fp32 output still
// takes the -DBF16_TABLE modules and is not a bf16 activation.
TEST(Bf16ElementwiseTest, GatherBf16WeightTableUnchanged) {
    Env e;
    const int rows = 4, cols = 8;
    auto tv = values(rows * cols, 71, 100.0F);
    auto table = bf16_in(e, {rows, cols}, tv);
    auto idx = std::make_shared<Tensor<int64_t>>(std::vector<int>{2});
    idx->fillToCPU(std::vector<int64_t>{3, 1});
    auto keep = idx->data();
    idx->as_storage_buffer(e.dev);
    idx->copyToGPU(e.cmdpool, keep.data());
    EXPECT_FALSE(vkop::core::has_bf16_activation(
        OpType::GATHER, {table, idx},
        {out<float>(e, {2, cols}, ElemKind::kFloat32)}));
    {
        auto y = out<float>(e, {2, cols}, ElemKind::kFloat32);
        auto op = make_op(e, OpType::GATHER, 0, {{"axis", "0"}});
        op->onExecute({table, idx}, {y}, 0);
        submit(e, op.get());
        const auto got = read_f32(e, y);
        for (int c = 0; c < cols; ++c) {
            EXPECT_EQ(got[c], tv[3 * cols + c]);
            EXPECT_EQ(got[cols + c], tv[1 * cols + c]);
        }
    }
    {
        auto y = out<uint16_t>(e, {2, cols}, ElemKind::kFloat16);
        auto op = make_op(e, OpType::GATHER, 1, {{"axis", "0"}});
        op->onExecute({table, idx}, {y}, 0);
        submit(e, op.get());
        y->copyToCPU(e.cmdpool);
        for (int c = 0; c < cols; ++c) {
            EXPECT_EQ((*y)[c], f32_to_f16(tv[3 * cols + c]));
            EXPECT_EQ((*y)[cols + c], f32_to_f16(tv[1 * cols + c]));
        }
    }
}

// ------------------------------------------------------------ Expand / Concat

TEST(Bf16ElementwiseTest, ExpandMovesBf16Exactly) {
    Env e;
    struct Case {
        std::vector<int> in, outs;
    };
    // Odd total (per-element path) and an even, innermost-copied shape (the
    // whole-word path).
    for (const Case &c : {Case{{3, 1, 5}, {3, 3, 5}}, Case{{2, 1, 6}, {2, 3, 6}},
                          Case{{1, 7}, {5, 7}}}) {
        auto xv = with_specials(values(numel(c.in), 81, 1e6F));
        auto x = bf16_in(e, c.in, xv);
        auto shape = std::make_shared<Tensor<int>>(
            std::vector<int>{static_cast<int>(c.outs.size())});
        shape->fillToCPU(c.outs);
        auto y = out<uint16_t>(e, c.outs, ElemKind::kBFloat16);
        auto op = make_op(e, OpType::EXPAND, 2);
        op->onExecute({x, shape}, {y}, 0);
        submit(e, op.get());
        const auto bits = read_bits(e, y);
        const int r = static_cast<int>(c.outs.size());
        for (size_t o = 0; o < bits.size(); ++o) {
            size_t rem = o, ii = 0, stride = 1;
            std::vector<int> coord(r);
            for (int d = r - 1; d >= 0; --d) {
                coord[d] = static_cast<int>(rem % c.outs[d]);
                rem /= c.outs[d];
            }
            for (int d = r - 1; d >= 0; --d) {
                ii += (c.in[d] == 1 ? 0 : coord[d]) * stride;
                stride *= c.in[d];
            }
            EXPECT_EQ(bits[o], float_to_bf16(xv[ii])) << "out " << o;
        }
    }
}

TEST(Bf16ElementwiseTest, ConcatMovesBf16Exactly) {
    Env e;
    struct Case {
        std::vector<int> a, b;
        int axis;
    };
    // Even blocks (whole-word copy) and odd blocks (atomic half-word path).
    for (const Case &c : {Case{{2, 4}, {2, 2}, 1}, Case{{2, 3}, {2, 2}, 1},
                          Case{{3, 5}, {2, 5}, 0}}) {
        auto av = with_specials(values(numel(c.a), 91, 1e6F));
        auto bv = values(numel(c.b), 92, 3e5F);
        auto a = bf16_in(e, c.a, av);
        auto b = bf16_in(e, c.b, bv);
        std::vector<int> ys = c.a;
        ys[c.axis] += c.b[c.axis];
        auto y = out<uint16_t>(e, ys, ElemKind::kBFloat16);
        auto op = make_op(e, OpType::CONCAT, 2,
                          {{"axis", std::to_string(c.axis)}});
        op->onExecute({a, b}, {y}, 0);
        submit(e, op.get());
        const auto bits = read_bits(e, y);
        const int rows = ys[0], cols = ys[1];
        for (int i = 0; i < rows; ++i)
            for (int j = 0; j < cols; ++j) {
                float want;
                if (c.axis == 1) {
                    want = j < c.a[1] ? av[i * c.a[1] + j]
                                      : bv[i * c.b[1] + (j - c.a[1])];
                } else {
                    want = i < c.a[0] ? av[i * cols + j]
                                      : bv[(i - c.a[0]) * cols + j];
                }
                EXPECT_EQ(bits[i * cols + j], float_to_bf16(want))
                    << "concat axis " << c.axis << " (" << i << "," << j << ")";
            }
    }
}
