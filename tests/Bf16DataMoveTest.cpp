// Copyright 2026 @junka
// bfloat16 ACTIVATIONS through the data-movement and reduction kernels:
// Transpose and RotaryEmbedding (GLM-Edge's hot path), then Pad, Resize,
// MaxPool, GlobalAveragePool, Reduce/ReduceMean, InstanceNormalization, Clip,
// PRelu, CumSum and ScatterElements.
//
// Each _bf16 module is checked against a reference computed from the SAME
// bf16-rounded inputs: a host reference for Transpose (bit-exact, including NaN
// and denormal payloads) and RotaryEmbedding (GLM-Edge's head_dim 128, partial
// rotary, the folded-Transpose layout, an fp32 cos/sin table), and the op's own
// fp32 kernel for the rest -- the fp32 builds are covered by each op's own
// suite, so "bf16 run == fp32 run rounded to bf16" isolates exactly the bf16
// decode/encode. Inputs reach past fp16's 65504 wherever the op is not
// range-limited, so an fp16 decode of a bf16 word cannot pass by accident.

#include <cmath>
#include <cstring>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "core/Bf16Gate.hpp"
#include "core/DType.hpp"
#include "ops/OperatorFactory.hpp"
#include "setup.hpp"

#include <gtest/gtest.h>

namespace bf16dm {

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

// Deterministic values in [-scale, scale), rounded to bf16.
std::vector<float> vals(size_t n, uint32_t seed, float scale = 1.0F) {
    std::vector<float> v(n);
    uint32_t s = seed * 2654435761u + 12345u;
    for (size_t i = 0; i < n; ++i) {
        s = s * 1664525u + 1013904223u;
        const float u = static_cast<float>(s >> 8) / 16777216.0F; // [0, 1)
        v[i] = bf16_to_float(float_to_bf16(scale * (2.0F * u - 1.0F)));
    }
    return v;
}

std::shared_ptr<Tensor<uint16_t>> bf16_bits_in(Env &e,
                                               const std::vector<int> &shape,
                                               const std::vector<uint16_t> &b) {
    auto t = std::make_shared<Tensor<uint16_t>>(shape);
    t->set_elem_kind(ElemKind::kBFloat16);
    t->fillToCPU(b);
    t->as_storage_buffer(e.dev);
    t->copyToGPU(e.cmdpool);
    return t;
}

std::shared_ptr<Tensor<uint16_t>> bf16_in(Env &e, const std::vector<int> &shape,
                                          const std::vector<float> &v) {
    std::vector<uint16_t> bits(v.size());
    for (size_t i = 0; i < v.size(); ++i) bits[i] = float_to_bf16(v[i]);
    return bf16_bits_in(e, shape, bits);
}

std::shared_ptr<Tensor<float>> f32_in(Env &e, const std::vector<int> &shape,
                                      const std::vector<float> &v) {
    auto t = std::make_shared<Tensor<float>>(shape);
    t->fillToCPU(v);
    t->as_storage_buffer(e.dev);
    t->copyToGPU(e.cmdpool);
    return t;
}

std::shared_ptr<Tensor<uint16_t>> f16_in(Env &e, const std::vector<int> &shape,
                                         const std::vector<float> &v) {
    auto t = std::make_shared<Tensor<uint16_t>>(shape);
    std::vector<uint16_t> bits(v.size());
    for (size_t i = 0; i < v.size(); ++i) bits[i] = ITensor::fp32_to_fp16(v[i]);
    t->fillToCPU(bits);
    t->as_storage_buffer(e.dev);
    t->copyToGPU(e.cmdpool);
    return t;
}

std::shared_ptr<Tensor<int64_t>> i64_in(Env &e, const std::vector<int> &shape,
                                        const std::vector<int64_t> &v) {
    auto t = std::make_shared<Tensor<int64_t>>(shape);
    t->fillToCPU(v);
    auto keep = t->data();
    t->as_storage_buffer(e.dev);
    t->copyToGPU(e.cmdpool, keep.data());
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

std::vector<uint16_t> read_bits(Env &e,
                                const std::shared_ptr<Tensor<uint16_t>> &t) {
    t->copyToCPU(e.cmdpool);
    std::vector<uint16_t> v(t->num_elements());
    for (size_t i = 0; i < v.size(); ++i) v[i] = (*t)[i];
    return v;
}

std::vector<float> read_bf16(Env &e,
                             const std::shared_ptr<Tensor<uint16_t>> &t) {
    auto b = read_bits(e, t);
    std::vector<float> v(b.size());
    for (size_t i = 0; i < b.size(); ++i) v[i] = bf16_to_float(b[i]);
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

// One input of a twin run: its shape and values, and whether it is a float
// operand (bf16 in the bf16 run, fp32 in the reference run) or an int64 one.
struct In {
    std::vector<int> shape;
    std::vector<float> v;
    bool i64 = false;
    bool keep_f32 = false; // float operand that stays fp32 in BOTH runs
};

// Run `type` twice over the same bf16-rounded inputs: the _bf16 module
// (fp16 = 2, every float operand bf16) and the fp32 module. Returns the two
// outputs, the bf16 one decoded.
struct Twin {
    std::vector<float> bf16, f32;
    std::vector<int> shape;
};
Twin twin(Env &e, OpType type, const Attrs &attrs, const std::vector<In> &ins,
          const std::vector<int> &yshape) {
    Twin r;
    for (int pass = 0; pass < 2; ++pass) {
        const bool b = pass == 0;
        std::vector<std::shared_ptr<ITensor>> inputs;
        for (const In &in : ins) {
            if (in.i64) {
                std::vector<int64_t> iv(in.v.begin(), in.v.end());
                inputs.push_back(i64_in(e, in.shape, iv));
            } else if (b && !in.keep_f32) {
                inputs.push_back(bf16_in(e, in.shape, in.v));
            } else {
                inputs.push_back(f32_in(e, in.shape, in.v));
            }
        }
        auto op = make_op(e, type, b ? 2 : 0, attrs);
        if (b) {
            auto y = out<uint16_t>(e, yshape, ElemKind::kBFloat16);
            op->onExecute(inputs, {y}, 0);
            submit(e, op.get());
            r.bf16 = read_bf16(e, y);
            r.shape = y->getShape();
        } else {
            auto y = out<float>(e, yshape, ElemKind::kFloat32);
            op->onExecute(inputs, {y}, 0);
            submit(e, op.get());
            r.f32 = read_f32(e, y);
        }
    }
    return r;
}

// bf16 output vs the fp32 kernel's output. rel = 0 demands the bf16 result
// to be exactly the fp32 result rounded to bf16 (a pure mover / selection);
// otherwise |got - ref| <= rel*|ref| + abs.
int count_bad(const std::vector<float> &got, const std::vector<float> &ref,
              double rel, double abs_tol, double *worst) {
    int bad = 0;
    *worst = 0;
    if (got.size() != ref.size()) return static_cast<int>(ref.size()) + 1;
    for (size_t i = 0; i < ref.size(); ++i) {
        if (rel == 0.0) {
            const float want = bf16_to_float(float_to_bf16(ref[i]));
            const bool same = (std::isnan(want) && std::isnan(got[i])) ||
                              want == got[i];
            if (!same) ++bad;
            continue;
        }
        const double d = std::fabs(double(got[i]) - double(ref[i]));
        *worst = std::max(*worst, std::isfinite(d) ? d : 1e30);
        if (!(d <= rel * std::fabs(double(ref[i])) + abs_tol)) ++bad;
    }
    return bad;
}

// Host N-d transpose of raw 16-bit words.
std::vector<uint16_t> host_transpose(const std::vector<uint16_t> &x,
                                     const std::vector<int> &shape,
                                     const std::vector<int> &perm) {
    const int rank = static_cast<int>(shape.size());
    std::vector<int> oshape(rank), istride(rank, 1);
    for (int i = rank - 2; i >= 0; --i)
        istride[i] = istride[i + 1] * shape[i + 1];
    for (int i = 0; i < rank; ++i) oshape[i] = shape[perm[i]];
    std::vector<uint16_t> y(x.size());
    for (size_t o = 0; o < y.size(); ++o) {
        size_t rem = o, src = 0;
        for (int i = rank - 1; i >= 0; --i) {
            const int c = static_cast<int>(rem % oshape[i]);
            rem /= oshape[i];
            src += static_cast<size_t>(c) * istride[perm[i]];
        }
        y[o] = x[src];
    }
    return y;
}

std::string perm_attr(const std::vector<int> &p) {
    std::string s = "[";
    for (size_t i = 0; i < p.size(); ++i)
        s += (i ? ", " : "") + std::to_string(p[i]);
    return s + "]";
}

} // namespace bf16dm

using namespace bf16dm;

// ------------------------------------------------------------- Transpose

TEST(Bf16DataMoveTest, TransposeMovesRawBf16Words) {
    Env e;
    struct Case {
        std::vector<int> shape, perm;
        const char *why;
    };
    const std::vector<Case> cases = {
        {{2, 4}, {1, 0}, "review repro: [2,4] 1..8"},
        {{1, 5, 16, 128}, {0, 2, 1, 3}, "GLM q [B,S,H,D]->[B,H,S,D] (chunk4)"},
        {{1, 16, 5, 128}, {0, 2, 1, 3}, "GLM attn out [B,H,S,D]->[B,S,H,D]"},
        {{1, 4, 7, 128}, {0, 1, 3, 2}, "K^T, even rows"},
        {{3, 5, 7}, {2, 0, 1}, "odd rows, odd total"},
        {{5, 7}, {1, 0}, "odd x odd"},
        {{2, 3, 6}, {1, 0, 2}, "kept innermost 6 (not chunk4)"},
        {{1, 1, 4, 8}, {0, 2, 1, 3}, "flat identity: aliased view"},
    };
    for (const Case &c : cases) {
        const size_t n = numel(c.shape);
        std::vector<uint16_t> bits(n);
        if (c.shape == std::vector<int>{2, 4}) {
            for (size_t i = 0; i < n; ++i)
                bits[i] = float_to_bf16(static_cast<float>(i + 1));
        } else {
            const auto v = vals(n, 7 + static_cast<uint32_t>(n), 3.0e5F);
            for (size_t i = 0; i < n; ++i) bits[i] = float_to_bf16(v[i]);
            // Payloads a decode/encode round trip would not keep: a NaN with
            // a non-canonical payload, -Inf, a bf16 denormal, -0.
            bits[0] = 0x7FC1;
            if (n > 1) bits[1] = 0xFF80;
            if (n > 2) bits[2] = 0x0001;
            if (n > 3) bits[3] = 0x8000;
        }
        std::vector<int> oshape(c.shape.size());
        for (size_t i = 0; i < c.shape.size(); ++i)
            oshape[i] = c.shape[c.perm[i]];
        auto x = bf16_bits_in(e, c.shape, bits);
        auto y = out<uint16_t>(e, oshape, ElemKind::kBFloat16);
        auto op = make_op(e, OpType::TRANSPOSE, 2,
                          {{"perm", perm_attr(c.perm)}});
        op->onExecute({x}, {y}, 0);
        submit(e, op.get());
        const auto want = host_transpose(bits, c.shape, c.perm);
        const auto got = read_bits(e, y);
        ASSERT_EQ(got.size(), want.size()) << c.why;
        int bad = 0;
        for (size_t i = 0; i < want.size(); ++i) bad += got[i] != want[i];
        EXPECT_EQ(bad, 0) << c.why;
        if (c.shape == std::vector<int>{2, 4}) {
            const float ref[8] = {1, 5, 2, 6, 3, 7, 4, 8};
            for (int i = 0; i < 8; ++i)
                EXPECT_EQ(bf16_to_float(got[i]), ref[i]) << "element " << i;
        }
    }
}

// The same word mover serves fp16: the raw move keeps its bits too.
TEST(Bf16DataMoveTest, TransposeFp16StillBitExact) {
    Env e;
    const std::vector<int> shape = {3, 5, 7}, perm = {2, 0, 1};
    const auto v = vals(numel(shape), 99, 100.0F);
    auto x = f16_in(e, shape, v);
    x->copyToCPU(e.cmdpool);
    std::vector<uint16_t> bits(numel(shape));
    for (size_t i = 0; i < bits.size(); ++i) bits[i] = (*x)[i];
    auto y = out<uint16_t>(e, {7, 3, 5}, ElemKind::kFloat16);
    auto op = make_op(e, OpType::TRANSPOSE, 1, {{"perm", perm_attr(perm)}});
    op->onExecute({x}, {y}, 0);
    submit(e, op.get());
    EXPECT_EQ(read_bits(e, y), host_transpose(bits, shape, perm));
}

// ------------------------------------------------------- RotaryEmbedding

namespace bf16dm {
// Host half-split RoPE, [B, H, S, D] output; x read at `xshape` layout
// (untransposed = [B, S, H, D]); table rows (b*S + s)*rd.
std::vector<double> host_rope(const std::vector<float> &x,
                              const std::vector<float> &cs,
                              const std::vector<float> &sn, int B, int H, int S,
                              int D, int rd, bool untransposed) {
    std::vector<double> y(static_cast<size_t>(B) * H * S * D);
    const int half = rd / 2;
    auto xat = [&](int b, int h, int s, int d) {
        const size_t i = untransposed
                             ? ((static_cast<size_t>(b) * S + s) * H + h) * D + d
                             : ((static_cast<size_t>(b) * H + h) * S + s) * D + d;
        return double(x[i]);
    };
    for (int b = 0; b < B; ++b)
        for (int h = 0; h < H; ++h)
            for (int s = 0; s < S; ++s)
                for (int d = 0; d < D; ++d) {
                    const size_t o =
                        ((static_cast<size_t>(b) * H + h) * S + s) * D + d;
                    if (d >= rd) {
                        y[o] = xat(b, h, s, d);
                        continue;
                    }
                    const size_t ci = (static_cast<size_t>(b) * S + s) * rd + d;
                    const double c = cs[ci], si = sn[ci];
                    y[o] = d < half ? xat(b, h, s, d) * c -
                                          xat(b, h, s, d + half) * si
                                    : xat(b, h, s, d) * c +
                                          xat(b, h, s, d - half) * si;
                }
    return y;
}

// HF-style table: cos/sin of position * inv_freq, half-repeated (cat(f, f)).
void rope_table(int S, int rd, int pos0, std::vector<float> &cs,
                std::vector<float> &sn, bool round_bf16) {
    cs.assign(static_cast<size_t>(S) * rd, 0.0F);
    sn = cs;
    for (int s = 0; s < S; ++s)
        for (int d = 0; d < rd; ++d) {
            const int i = d % (rd / 2);
            const double inv = 1.0 / std::pow(10000.0, 2.0 * i / rd);
            const double a = (pos0 + s) * inv;
            float c = static_cast<float>(std::cos(a));
            float si = static_cast<float>(std::sin(a));
            if (round_bf16) {
                c = bf16_to_float(float_to_bf16(c));
                si = bf16_to_float(float_to_bf16(si));
            }
            cs[s * rd + d] = c;
            sn[s * rd + d] = si;
        }
}
} // namespace bf16dm

TEST(Bf16DataMoveTest, RotaryMatchesHostReference) {
    Env e;
    struct Case {
        int H, S, D, rd;
        bool untransposed, table_f32;
        float scale;
        const char *why;
    };
    const std::vector<Case> cases = {
        {16, 5, 128, 128, false, false, 1.0F, "GLM-Edge q prefill"},
        {4, 5, 128, 128, false, false, 1.0F, "GLM-Edge k prefill (kv heads)"},
        {16, 1, 128, 128, false, false, 1.0F, "GLM-Edge q decode"},
        {16, 5, 128, 128, true, false, 1.0F, "folded Transpose (untransposed)"},
        {4, 3, 128, 96, false, false, 1.0F, "partial rotary 96 of 128 (Phi-4)"},
        {16, 5, 128, 128, false, true, 1.0F, "fp32 cos/sin beside bf16 X"},
        {3, 7, 64, 64, false, false, 2.0e5F, "values past fp16's 65504"},
    };
    for (const Case &c : cases) {
        const int B = 1;
        const std::vector<int> xshape =
            c.untransposed ? std::vector<int>{B, c.S, c.H, c.D}
                           : std::vector<int>{B, c.H, c.S, c.D};
        const std::vector<int> yshape = {B, c.H, c.S, c.D};
        const auto xv = vals(numel(xshape), 17 + c.H + c.S, c.scale);
        std::vector<float> cs, sn;
        rope_table(c.S, c.rd, 3, cs, sn, /*round_bf16=*/!c.table_f32);
        const std::vector<int> tshape = {B, 1, c.S, c.rd};
        auto x = bf16_in(e, xshape, xv);
        std::shared_ptr<ITensor> ct, st;
        if (c.table_f32) {
            ct = f32_in(e, tshape, cs);
            st = f32_in(e, tshape, sn);
        } else {
            ct = bf16_in(e, tshape, cs);
            st = bf16_in(e, tshape, sn);
        }
        auto y = out<uint16_t>(e, yshape, ElemKind::kBFloat16);
        Attrs attrs;
        if (c.untransposed) attrs["input_untransposed"] = "1";
        auto op = make_op(e, OpType::ROTARY_EMBEDDING, 2, attrs);
        op->onExecute({x, ct, st}, {y}, 0);
        submit(e, op.get());
        const auto ref =
            host_rope(xv, cs, sn, B, c.H, c.S, c.D, c.rd, c.untransposed);
        const auto got = read_bf16(e, y);
        ASSERT_EQ(got.size(), ref.size()) << c.why;
        int bad = 0;
        double worst = 0;
        for (size_t i = 0; i < ref.size(); ++i) {
            // One bf16 rounding of an fp32 result: 2^-8 relative, plus the
            // cancellation slack of a two-term sum.
            const double d = std::fabs(got[i] - ref[i]);
            worst = std::max(worst, d);
            if (!(d <= 4.0e-3 * std::fabs(ref[i]) + 4.0e-3 * c.scale)) ++bad;
        }
        EXPECT_EQ(bad, 0) << c.why << " worst " << worst;
    }
}

TEST(Bf16DataMoveTest, RotaryRefusesAnFp16TableBesideBf16X) {
    // The 16-bit table form is X's own format, so an fp16 table beside bf16 X
    // has no reader; an fp32 one does (cs_fp32) and is admitted above.
    Env e;
    auto x = bf16_in(e, {1, 2, 3, 8}, vals(48, 1));
    auto ct = f16_in(e, {1, 1, 3, 8}, vals(24, 2));
    auto st = f16_in(e, {1, 1, 3, 8}, vals(24, 3));
    auto y = out<uint16_t>(e, {1, 2, 3, 8}, ElemKind::kBFloat16);
    auto op = make_op(e, OpType::ROTARY_EMBEDDING, 2);
    try {
        op->onExecute({x, ct, st}, {y}, 0);
        ADD_FAILURE() << "an fp16 cos table beside bf16 X was accepted";
    } catch (const std::runtime_error &err) {
        EXPECT_NE(std::string(err.what()).find("float16 input 1"),
                  std::string::npos)
            << err.what();
    }
    // One rule (bf16_other_float_ok): cos/sin may be fp32 but not fp16;
    // Pad's constant any float; nothing else on these ops.
    using vkop::core::bf16_other_float_ok;
    const auto f32 = ElemKind::kFloat32, f16 = ElemKind::kFloat16;
    EXPECT_TRUE(bf16_other_float_ok(OpType::ROTARY_EMBEDDING, false, 1, f32));
    EXPECT_TRUE(bf16_other_float_ok(OpType::ROTARY_EMBEDDING, false, 2, f32));
    EXPECT_FALSE(bf16_other_float_ok(OpType::ROTARY_EMBEDDING, false, 1, f16));
    EXPECT_FALSE(bf16_other_float_ok(OpType::ROTARY_EMBEDDING, false, 0, f32));
    EXPECT_TRUE(bf16_other_float_ok(OpType::PAD, false, 2, f16));
    EXPECT_FALSE(bf16_other_float_ok(OpType::PAD, false, 0, f32));
    EXPECT_FALSE(bf16_other_float_ok(OpType::ADD, false, 1, f32));
}

// ------------------------------- the rest: _bf16 module vs the fp32 module

namespace bf16dm {
void expect_twin(Env &e, OpType type, const Attrs &attrs,
                 const std::vector<In> &ins, const std::vector<int> &yshape,
                 double rel, double abs_tol, const std::string &why) {
    const Twin r = twin(e, type, attrs, ins, yshape);
    double worst = 0;
    const int bad = count_bad(r.bf16, r.f32, rel, abs_tol, &worst);
    EXPECT_EQ(bad, 0) << why << ": " << bad << " of " << r.f32.size()
                      << " off, worst " << worst;
}
} // namespace bf16dm

TEST(Bf16DataMoveTest, PadMatchesFp32) {
    Env e;
    const std::vector<int> s = {2, 3, 5};
    const auto x = vals(numel(s), 3, 2.0e5F);
    // Constant 1e5 is past fp16's range: an fp16-encoded constant would be
    // Inf (and decoded as bf16, garbage).
    expect_twin(e, OpType::PAD,
                {{"pads", "[0, 1, 2, 0, 1, 1]"}, {"mode", "constant"},
                 {"constant_value", "100000.0"}},
                {{s, x}}, {2, 5, 8}, 0.0, 0.0, "Pad constant");
    expect_twin(e, OpType::PAD, {{"pads", "[0, 0, 2, 0, 0, 1]"},
                                 {"mode", "reflect"}},
                {{s, x}}, {2, 3, 8}, 0.0, 0.0, "Pad reflect");
}

TEST(Bf16DataMoveTest, ResizeMatchesFp32) {
    Env e;
    const std::vector<int> s = {1, 3, 5, 6};
    const auto x = vals(numel(s), 4, 2.0e5F);
    expect_twin(e, OpType::RESIZE,
                {{"mode", "nearest"}, {"scales", "[1, 1, 2, 2]"}}, {{s, x}},
                {1, 3, 10, 12}, 0.0, 0.0, "Resize nearest x2");
    expect_twin(e, OpType::RESIZE,
                {{"mode", "nearest"}, {"sizes", "[1, 3, 7, 9]"}}, {{s, x}},
                {1, 3, 7, 9}, 0.0, 0.0, "Resize nearest to odd sizes");
    expect_twin(e, OpType::RESIZE,
                {{"mode", "linear"}, {"scales", "[1, 1, 2, 2]"}}, {{s, x}},
                {1, 3, 10, 12}, 8e-3, 1.0, "Resize bilinear");
}

TEST(Bf16DataMoveTest, MaxPoolMatchesFp32) {
    Env e;
    const std::vector<int> s = {1, 3, 7, 9};
    // All-negative and past the old -100000 sentinel: the max of a window of
    // -3e5 values must be one of them, not the sentinel.
    auto x = vals(numel(s), 5, 1.0e5F);
    for (auto &v : x) v = bf16_to_float(float_to_bf16(-2.0e5F + v));
    expect_twin(e, OpType::MAXPOOL2D,
                {{"kernel_shape", "[3, 3]"}, {"strides", "[2, 2]"},
                 {"pads", "[1, 1, 1, 1]"}, {"ceil_mode", "0"}},
                {{s, x}}, {1, 3, 4, 5}, 0.0, 0.0, "MaxPool 3x3/2 pad 1");
}

TEST(Bf16DataMoveTest, GlobalAveragePoolMatchesFp32) {
    Env e;
    const std::vector<int> s = {1, 5, 6, 7};
    const auto x = vals(numel(s), 6, 2.0e5F);
    expect_twin(e, OpType::GLOBALAVERAGEPOOL, {}, {{s, x}}, {1, 5}, 8e-3, 1.0,
                "GlobalAveragePool odd channel count");
}

TEST(Bf16DataMoveTest, ReduceMatchesFp32) {
    Env e;
    const std::vector<int> s = {3, 5, 6};
    const auto x = vals(numel(s), 7, 2.0e5F);
    struct Case {
        const char *op, *axes;
        std::vector<int> y;
        double rel;
    };
    for (const Case &c : {Case{"sum", "[2]", {3, 5, 1}, 8e-3},
                          Case{"mean", "[1]", {3, 1, 6}, 8e-3},
                          Case{"max", "[2]", {3, 5, 1}, 0.0},
                          Case{"min", "[0, 2]", {1, 5, 1}, 0.0},
                          Case{"sum_square", "[2]", {3, 5, 1}, 8e-3},
                          Case{"log_sum_exp", "[2]", {3, 5, 1}, 8e-3}}) {
        expect_twin(e, OpType::REDUCE,
                    {{"reduce_op", c.op}, {"axes", c.axes}, {"keepdims", "1"}},
                    {{s, x}}, c.y, c.rel, 1e-3,
                    std::string("Reduce ") + c.op + " " + c.axes);
    }
    // Max over values all below -1e30: the old -1e30 init returned the init.
    std::vector<float> big(numel(s));
    for (size_t i = 0; i < big.size(); ++i)
        big[i] = bf16_to_float(float_to_bf16(-1.0e32F * (1.0F + i % 7)));
    expect_twin(e, OpType::REDUCE,
                {{"reduce_op", "max"}, {"axes", "[2]"}, {"keepdims", "1"}},
                {{s, big}}, {3, 5, 1}, 0.0, 0.0, "Reduce max below -1e30");
    expect_twin(e, OpType::REDUCEMEAN, {{"axes", "[-1]"}, {"keepdims", "1"}},
                {{s, x}}, {3, 5, 1}, 8e-3, 1e-3, "ReduceMean last axis");
}

TEST(Bf16DataMoveTest, InstanceNormMatchesFp32) {
    Env e;
    const std::vector<int> s = {2, 3, 4, 6};
    auto x = vals(numel(s), 8, 3.0e4F);
    for (auto &v : x) v = bf16_to_float(float_to_bf16(v + 1.0e5F));
    expect_twin(e, OpType::INSTANCENORM, {{"epsilon", "1e-5"}},
                {{s, x}, {{3}, vals(3, 9)}, {{3}, vals(3, 10)}}, s, 1.6e-2,
                2e-2, "InstanceNorm");
}

TEST(Bf16DataMoveTest, ClipMatchesFp32) {
    Env e;
    const std::vector<int> s = {5, 7};
    const auto x = vals(numel(s), 11, 4.0e5F);
    expect_twin(e, OpType::CLIP, {{"min", "-150000"}, {"max", "70000"}},
                {{s, x}}, s, 0.0, 0.0, "Clip attribute bounds");
    // Bounds as tensor inputs: bf16 ones decode as bf16 on the host, and fp32
    // ones are an admitted slot beside a bf16 X.
    expect_twin(e, OpType::CLIP, {},
                {{s, x}, {{1}, {-150528.0F}}, {{1}, {70144.0F}}}, s, 0.0, 0.0,
                "Clip bf16 bound tensors");
    expect_twin(e, OpType::CLIP, {},
                {{s, x}, {{1}, {-150000.0F}, false, true},
                 {{1}, {70000.0F}, false, true}},
                s, 0.0, 0.0, "Clip fp32 bound tensors");
}

TEST(Bf16DataMoveTest, PReluMatchesFp32) {
    Env e;
    const std::vector<int> s = {2, 3, 5};
    const auto x = vals(numel(s), 12, 2.0e5F);
    expect_twin(e, OpType::PRELU, {}, {{s, x}, {{3, 1}, vals(3, 13)}}, s, 8e-3,
                1e-3, "PRelu broadcast slope");
    expect_twin(e, OpType::PRELU, {}, {{s, x}, {s, vals(numel(s), 14)}}, s,
                8e-3, 1e-3, "PRelu full slope");
}

TEST(Bf16DataMoveTest, CumSumMatchesFp32) {
    Env e;
    struct Case {
        std::vector<int> s;
        const char *axis;
        bool excl, rev;
    };
    // Odd axis_stride (3, 5) puts the two lines a thread owns in different
    // words: the packed-word layout would race there.
    for (const Case &c : {Case{{4, 6}, "1", false, false},
                          Case{{6, 3}, "0", false, false},
                          Case{{2, 7, 5}, "1", true, false},
                          Case{{3, 9}, "-1", false, true}}) {
        const auto x = vals(numel(c.s), 15, 3.0e4F);
        expect_twin(e, OpType::CUMSUM,
                    {{"axis", c.axis},
                     {"exclusive", c.excl ? "1" : "0"},
                     {"reverse", c.rev ? "1" : "0"}},
                    {{c.s, x}}, c.s, 1.6e-2, 1.0e3,
                    std::string("CumSum axis ") + c.axis);
    }
}

TEST(Bf16DataMoveTest, ScatterElementsMatchesFp32) {
    Env e;
    const std::vector<int> ds = {4, 5}, is = {2, 5};
    const auto data = vals(numel(ds), 16, 2.0e5F);
    const auto upd = vals(numel(is), 17, 2.0e5F);
    const std::vector<float> idx = {3, 0, 2, 1, 3, 0, 2, 1, 0, 2};
    for (const char *red : {"none", "add", "max"}) {
        expect_twin(e, OpType::SCATTER_ELEMENTS,
                    {{"axis", "0"}, {"reduction", red}},
                    {{ds, data}, {is, idx, true}, {is, upd}}, ds,
                    std::string(red) == "add" ? 8e-3 : 0.0, 2e3,
                    std::string("ScatterElements axis 0 ") + red);
    }
    // axis 1 on an odd row width: neighbouring columns share words across rows.
    const std::vector<int> ds1 = {3, 5}, is1 = {3, 2};
    const std::vector<float> idx1 = {4, 1, 0, 3, 2, 4};
    expect_twin(e, OpType::SCATTER_ELEMENTS, {{"axis", "1"}},
                {{ds1, vals(15, 18, 2.0e5F)}, {is1, idx1, true},
                 {is1, vals(6, 19, 2.0e5F)}},
                ds1, 0.0, 0.0, "ScatterElements axis 1");
}

