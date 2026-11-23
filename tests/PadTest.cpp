// Copyright 2026 @junka
// Pad op tests. Pad is SSBO-only (no image twin; the target device exposes no
// image backend), so every case drives
// create_from_type(OpType::PAD, fp16, 0, /*backend_buffer=*/true).
//
// Coverage:
//   - constant pad, 2-D and 4-D, asymmetric begin/end per axis, fp32 + fp16
//   - reflect on a width-5 axis padded by 2, which pins the ONNX/PyTorch
//     numbers exactly: 2 1 0 1 2 3 4 3 2 (the edge element is NOT repeated —
//     that is what edge/symmetric do, and getting it wrong is invisible on a
//     symmetric input, so the reference here is a ramp)
//   - edge pad, and a rank-5 case (the video VAE is 5-D), fp32 + fp16
//   - symmetric, since the shader implements it and it is one line to check
//   - an attribute-only `pads` spelling, and the opset-18 `pads` input
//   - error cases: a pads vector of the wrong length must throw, and so must
//     a reflect pad that reaches the input's size (ONNX forbids it).

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include "setup.hpp"
#include "core/Tensor.hpp"
#include "include/logger.hpp"
#include "ops/OperatorFactory.hpp"
#include "ops/Ops.hpp"

#include <gtest/gtest.h>

using vkop::core::Tensor;

namespace {

// ---- device / tensor helpers ---------------------------------------------

std::shared_ptr<vkop::VulkanDevice> &pt_dev() {
    static std::shared_ptr<vkop::VulkanDevice> dev = [] {
        if (!vkop::tests::TestEnv::is_initialized())
            vkop::tests::TestEnv::initialize();
        return vkop::tests::TestEnv::get_device();
    }();
    return dev;
}
std::shared_ptr<vkop::VulkanCommandPool> pt_cmdpool() {
    return vkop::tests::TestEnv::get_command_pool();
}

template <typename T>
std::shared_ptr<Tensor<T>> pt_make(const std::vector<int> &shape,
                                   const std::vector<T> &vals) {
    auto t = std::make_shared<Tensor<T>>(shape);
    t->fillToCPU(vals);
    return t;
}

template <typename T>
void pt_upload(std::shared_ptr<Tensor<T>> &t) {
    t->as_storage_buffer(pt_dev());
    t->copyToGPU(pt_cmdpool());
}

template <typename T>
std::shared_ptr<Tensor<T>> pt_make_out(const std::vector<int> &shape) {
    auto t = std::make_shared<Tensor<T>>(shape);
    t->as_storage_buffer(pt_dev());
    t->toGPU();
    return t;
}

std::unique_ptr<vkop::ops::Operator>
pt_make_op(bool fp16, const std::unordered_map<std::string, std::string> &attrs) {
    auto op = vkop::ops::create_from_type(vkop::ops::OpType::PAD, fp16 ? 1 : 0,
                                          0, /*backend_buffer=*/true);
    if (!op) {
        LOG_ERROR("create_from_type(PAD) returned null");
        return nullptr;
    }
    op->set_runtime_device(pt_dev(), pt_cmdpool());
    if (!attrs.empty())
        op->setAttribute(attrs);
    return op;
}

void pt_run(vkop::ops::Operator *op) {
    auto cmd = op->get_record();
    std::vector<VkSubmitInfo> info{cmd->buildSubmitInfo()};
    vkop::VulkanCommandBuffer::submit(pt_dev()->getComputeQueue(), info);
    cmd->wait();
    pt_dev()->wait_all_done();
}

// ---- host reference -------------------------------------------------------
// Written as torch/ONNX define it: build, per axis, the list of INPUT indices
// the padded axis reads, then gather. Deliberately a different shape of code
// from the shader's single-pass coordinate map (the shader folds the per-axis
// map into the linear-index walk), so an arithmetic slip does not reproduce on
// both sides.

enum class RefMode { Constant, Edge, Reflect, Symmetric };

// Input index for output position `c` on an axis of size `D` with `b` begin
// and `e` end pad. Returns -1 for a constant-pad position.
int ref_index(int c, int D, int b, RefMode mode) {
    if (c < b) {
        switch (mode) {
        case RefMode::Constant:
            return -1;
        case RefMode::Edge:
            return 0;
        case RefMode::Reflect:
            return b - c; // mirror about the first interior element
        case RefMode::Symmetric:
            return b - c - 1; // mirror about the edge, repeating it
        }
    }
    if (c >= b + D) {
        int j = c - b - D;
        switch (mode) {
        case RefMode::Constant:
            return -1;
        case RefMode::Edge:
            return D - 1;
        case RefMode::Reflect:
            return D - 2 - j; // mirror about the last interior element
        case RefMode::Symmetric:
            return D - 1 - j;
        }
    }
    return c - b;
}

// The constant pad element in the ELEMENT storage type. For fp16 that is the
// packed half bits, not an integer truncation of the value — static_cast on a
// uint16_t storage would turn 3.25 into 3 and agree with nothing.
template <typename T> T ref_constant(float value) {
    if constexpr (std::is_same_v<T, uint16_t>) {
        return vkop::core::ITensor::fp32_to_fp16(value);
    } else {
        return static_cast<T>(value);
    }
}

// Reference output for a row-major tensor, with per-axis begin pads and the
// end pads implied by `out_shape`.
template <typename T>
std::vector<T> pad_ref(const std::vector<T> &in, const std::vector<int> &in_shape,
                       const std::vector<int> &out_shape,
                       const std::vector<int> &begins, RefMode mode,
                       float const_value) {
    const int rank = static_cast<int>(in_shape.size());
    std::vector<std::vector<int>> axis_map(static_cast<size_t>(rank));
    for (int d = 0; d < rank; ++d) {
        axis_map[d].resize(static_cast<size_t>(out_shape[d]));
        for (int c = 0; c < out_shape[d]; ++c) {
            axis_map[d][c] = ref_index(c, in_shape[d], begins[d], mode);
        }
    }
    int total = 1;
    for (int d = 0; d < rank; ++d) total *= out_shape[d];
    std::vector<T> out(static_cast<size_t>(total));
    std::vector<int> in_stride(static_cast<size_t>(rank), 1);
    std::vector<int> out_stride(static_cast<size_t>(rank), 1);
    for (int d = rank - 2; d >= 0; --d) {
        in_stride[d] = in_stride[d + 1] * in_shape[d + 1];
        out_stride[d] = out_stride[d + 1] * out_shape[d + 1];
    }
    std::vector<int> in_coord(static_cast<size_t>(rank), 0);
    for (int o = 0; o < total; ++o) {
        bool padded = false;
        int in_lin = 0;
        for (int d = 0; d < rank; ++d) {
            int c = (o / out_stride[d]) % out_shape[d];
            in_coord[d] = axis_map[d][c];
            if (in_coord[d] < 0) padded = true;
            in_lin += std::max(in_coord[d], 0) * in_stride[d];
        }
        out[static_cast<size_t>(o)] =
            padded ? ref_constant<T>(const_value)
                   : in[static_cast<size_t>(in_lin)];
    }
    return out;
}

RefMode ref_mode_of(const std::string &mode) {
    if (mode == "edge") return RefMode::Edge;
    if (mode == "reflect") return RefMode::Reflect;
    if (mode == "symmetric") return RefMode::Symmetric;
    return RefMode::Constant;
}

// Run one Pad and compare against pad_ref. `pads` is the ONNX attribute vector
// (rank*2: begins then ends).
template <typename T>
bool pt_pad_case(const std::vector<int> &in_shape, const std::vector<T> &in_vals,
                 const std::vector<int> &pads, const std::string &mode,
                 float const_value) {
    const int rank = static_cast<int>(in_shape.size());
    EXPECT_EQ(static_cast<int>(pads.size()), 2 * rank);

    std::vector<int> out_shape(in_shape);
    for (int d = 0; d < rank; ++d) {
        out_shape[d] += pads[d] + pads[d + rank];
    }
    std::vector<int> begins(pads.begin(), pads.begin() + rank);

    const auto expect = pad_ref<T>(in_vals, in_shape, out_shape, begins,
                                   ref_mode_of(mode), const_value);

    const bool fp16 = std::is_same_v<T, uint16_t>;
    std::unordered_map<std::string, std::string> attrs;
    attrs["pads"] = "[";
    for (size_t i = 0; i < pads.size(); ++i) {
        if (i) attrs["pads"] += ", ";
        attrs["pads"] += std::to_string(pads[i]);
    }
    attrs["pads"] += "]";
    attrs["mode"] = mode;
    attrs["constant_value"] = std::to_string(const_value);

    auto x = pt_make<T>(in_shape, in_vals);
    pt_upload(x);
    auto out = pt_make_out<T>(out_shape);

    auto op = pt_make_op(fp16, attrs);
    if (!op) return false;
    // A throw here is a real failure, not a silent false: gtest reports the
    // message, which names the offending pads/axis.
    op->onExecute({x}, {out}, 0);
    pt_run(op.get());
    out->copyToCPU(pt_cmdpool());

    if (out->num_elements() != static_cast<int>(expect.size())) {
        LOG_ERROR("Pad output element count %d != expected %zu",
                  out->num_elements(), expect.size());
        return false;
    }
    bool ok = true;
    for (int i = 0; i < out->num_elements(); ++i) {
        float got = fp16 ? vkop::core::ITensor::fp16_to_fp32((*out)[i])
                         : static_cast<float>((*out)[i]);
        float exp = fp16 ? vkop::core::ITensor::fp16_to_fp32(expect[static_cast<size_t>(i)])
                         : static_cast<float>(expect[static_cast<size_t>(i)]);
        if (std::abs(got - exp) > 1e-3F) {
            LOG_ERROR("Pad mismatch at %d: got %f expected %f", i, got, exp);
            ok = false;
        }
    }
    return ok;
}

// A 0,1,2,... ramp keeps every element distinct, so a wrong reflection axis or
// a swapped begin/end pad is visible in the numbers instead of cancelling.
template <typename T>
std::vector<T> ramp(size_t n) {
    std::vector<T> v(n);
    for (size_t i = 0; i < n; ++i) {
        v[i] = static_cast<T>(static_cast<float>(i) - 10.0F);
    }
    return v;
}

template <typename Col>
std::vector<uint16_t> to_fp16(const std::vector<Col> &v) {
    std::vector<uint16_t> out(v.size());
    for (size_t i = 0; i < v.size(); ++i) {
        out[i] = vkop::core::ITensor::fp32_to_fp16(static_cast<float>(v[i]));
    }
    return out;
}

} // namespace

// ---- constant -------------------------------------------------------------

TEST(PadTest, ConstantFp32_2D_Asymmetric) {
    // [2,3] padded by (begin, end) = ((1,0),(2,1)) → [3,6].
    const std::vector<int> shape{2, 3};
    EXPECT_TRUE(pt_pad_case<float>(shape, ramp<float>(6), {1, 2, 0, 1},
                                   "constant", 0.0F));
}

TEST(PadTest, ConstantFp32_2D_NonZeroValue) {
    const std::vector<int> shape{2, 3};
    EXPECT_TRUE(pt_pad_case<float>(shape, ramp<float>(6), {0, 1, 2, 0},
                                   "constant", -7.5F));
}

TEST(PadTest, ConstantFp32_4D_Asymmetric) {
    // [1,2,3,2] with a different pad on every axis, both ends.
    const std::vector<int> shape{1, 2, 3, 2};
    EXPECT_TRUE(pt_pad_case<float>(shape, ramp<float>(12), {0, 1, 2, 1,
                                                            2, 0, 1, 0},
                                   "constant", 3.25F));
}

TEST(PadTest, ConstantFp16_4D_Asymmetric) {
    const std::vector<int> shape{1, 2, 3, 2};
    EXPECT_TRUE(pt_pad_case<uint16_t>(shape, to_fp16(ramp<float>(12)),
                                      {0, 1, 2, 1, 2, 0, 1, 0}, "constant",
                                      3.25F));
}

// ---- reflect: the case the video VAE needs --------------------------------

TEST(PadTest, ReflectFp32_Width5Pad2ExactNumbers) {
    // The reference vector from the tree's earlier reflect probe: a width-5
    // ramp padded by 2 reflects as 2 1 0 1 2 3 4 3 2 (edge NOT repeated).
    const std::vector<int> shape{1, 1, 5};
    const std::vector<float> in{10.F, 11.F, 12.F, 13.F, 14.F};
    EXPECT_TRUE(pt_pad_case<float>(shape, in, {0, 0, 2, 0, 0, 2}, "reflect",
                                   0.0F));

    // Pin the actual bytes as well, so a test-side reference bug cannot make
    // both sides agree on something wrong.
    auto x = pt_make<float>(shape, in);
    pt_upload(x);
    auto out = pt_make_out<float>({1, 1, 9});
    auto op = pt_make_op(false, {{"pads", "[0, 0, 2, 0, 0, 2]"},
                                 {"mode", "reflect"}});
    ASSERT_NE(op, nullptr);
    op->onExecute({x}, {out}, 0);
    pt_run(op.get());
    out->copyToCPU(pt_cmdpool());
    const std::vector<float> expect{12.F, 11.F, 10.F, 11.F, 12.F,
                                    13.F, 14.F, 13.F, 12.F};
    ASSERT_EQ(out->num_elements(), static_cast<int>(expect.size()));
    for (int i = 0; i < out->num_elements(); ++i) {
        EXPECT_FLOAT_EQ((*out)[i], expect[static_cast<size_t>(i)]) << "at " << i;
    }
}

TEST(PadTest, ReflectFp32_RampNumbers) {
    // Same shape on a ramp: 2 1 0 1 2 3 4 3 2 (offset by -10 like ramp<>).
    const std::vector<int> shape{1, 1, 5};
    const std::vector<float> in{-10.F, -9.F, -8.F, -7.F, -6.F};
    auto x = pt_make<float>(shape, in);
    pt_upload(x);
    auto out = pt_make_out<float>({1, 1, 9});
    auto op = pt_make_op(false, {{"pads", "[0, 0, 2, 0, 0, 2]"},
                                 {"mode", "reflect"}});
    ASSERT_NE(op, nullptr);
    op->onExecute({x}, {out}, 0);
    pt_run(op.get());
    out->copyToCPU(pt_cmdpool());
    const std::vector<float> expect{-8.F, -9.F, -10.F, -9.F, -8.F,
                                    -7.F, -6.F, -7.F, -8.F};
    ASSERT_EQ(out->num_elements(), static_cast<int>(expect.size()));
    for (int i = 0; i < out->num_elements(); ++i) {
        EXPECT_FLOAT_EQ((*out)[i], expect[static_cast<size_t>(i)]) << "at " << i;
    }
}

TEST(PadTest, ReflectFp32_4D_Asymmetric) {
    const std::vector<int> shape{1, 2, 4, 3};
    EXPECT_TRUE(pt_pad_case<float>(shape, ramp<float>(24), {0, 1, 2, 1,
                                                            0, 0, 1, 2},
                                   "reflect", 0.0F));
}

TEST(PadTest, ReflectFp16_4D) {
    const std::vector<int> shape{2, 3, 4, 3};
    EXPECT_TRUE(pt_pad_case<uint16_t>(shape, to_fp16(ramp<float>(72)),
                                      {1, 0, 2, 1, 0, 1, 1, 2}, "reflect",
                                      0.0F));
}

// The VAE is 5-D and pads its spatial axes only.
TEST(PadTest, ReflectFp32_Rank5_SpatialOnly) {
    const std::vector<int> shape{1, 2, 3, 5, 4};
    EXPECT_TRUE(pt_pad_case<float>(shape, ramp<float>(120), {0, 0, 0, 2, 1,
                                                             0, 0, 0, 2, 1},
                                   "reflect", 0.0F));
}

TEST(PadTest, ReflectFp16_Rank5_SpatialOnly) {
    const std::vector<int> shape{1, 2, 3, 5, 4};
    EXPECT_TRUE(pt_pad_case<uint16_t>(shape, to_fp16(ramp<float>(120)),
                                      {0, 0, 0, 2, 1, 0, 0, 0, 2, 1},
                                      "reflect", 0.0F));
}

// ---- edge -----------------------------------------------------------------

TEST(PadTest, EdgeFp32_Width5Pad2) {
    // Edge repeats the boundary element: 10 10 10 11 12 13 14 14 14.
    const std::vector<int> shape{1, 1, 5};
    const std::vector<float> in{10.F, 11.F, 12.F, 13.F, 14.F};
    EXPECT_TRUE(pt_pad_case<float>(shape, in, {0, 0, 2, 0, 0, 2}, "edge",
                                   0.0F));

    auto x = pt_make<float>(shape, in);
    pt_upload(x);
    auto out = pt_make_out<float>({1, 1, 9});
    auto op = pt_make_op(false, {{"pads", "[0, 0, 2, 0, 0, 2]"},
                                 {"mode", "edge"}});
    ASSERT_NE(op, nullptr);
    op->onExecute({x}, {out}, 0);
    pt_run(op.get());
    out->copyToCPU(pt_cmdpool());
    const std::vector<float> expect{10.F, 10.F, 10.F, 11.F, 12.F,
                                    13.F, 14.F, 14.F, 14.F};
    ASSERT_EQ(out->num_elements(), static_cast<int>(expect.size()));
    for (int i = 0; i < out->num_elements(); ++i) {
        EXPECT_FLOAT_EQ((*out)[i], expect[static_cast<size_t>(i)]) << "at " << i;
    }
}

TEST(PadTest, EdgeFp16_Rank5) {
    const std::vector<int> shape{1, 2, 3, 4, 5};
    EXPECT_TRUE(pt_pad_case<uint16_t>(shape, to_fp16(ramp<float>(120)),
                                      {0, 0, 0, 1, 2, 1, 0, 1, 1, 2}, "edge",
                                      0.0F));
}

// ---- symmetric (implemented alongside the other three) -------------------

TEST(PadTest, SymmetricFp32_Width5Pad2) {
    // Symmetric repeats the edge: 11 10 10 11 12 13 14 14 13.
    const std::vector<int> shape{1, 1, 5};
    const std::vector<float> in{10.F, 11.F, 12.F, 13.F, 14.F};
    EXPECT_TRUE(pt_pad_case<float>(shape, in, {0, 0, 2, 0, 0, 2}, "symmetric",
                                   0.0F));
}

// ---- pads supplied as an input tensor (ONNX opset-2..10 spelling) --------

TEST(PadTest, PadsFromInputTensor) {
    // No `pads` attribute: the op reads the 2nd input (int64).
    const std::vector<int> shape{2, 3};
    const std::vector<float> in{-10.F, -9.F, -8.F, -7.F, -6.F, -5.F};
    auto x = pt_make<float>(shape, in);
    pt_upload(x);
    auto pads = pt_make<int64_t>({4}, {1, 0, 0, 2});
    pt_upload(pads);
    auto out = pt_make_out<float>({3, 5});

    auto op = pt_make_op(false, {{"mode", "constant"},
                                 {"constant_value", "1"}});
    ASSERT_NE(op, nullptr);
    op->onExecute({x, pads}, {out}, 0);
    pt_run(op.get());
    out->copyToCPU(pt_cmdpool());

    const std::vector<float> expect{
        1.F,  1.F,  1.F,  1.F,  1.F,   //
        -10.F, -9.F, -8.F, 1.F, 1.F,   //
        -7.F, -6.F, -5.F, 1.F, 1.F,
    };
    ASSERT_EQ(out->num_elements(), static_cast<int>(expect.size()));
    for (int i = 0; i < out->num_elements(); ++i) {
        EXPECT_FLOAT_EQ((*out)[i], expect[static_cast<size_t>(i)]) << "at " << i;
    }
}

// ---- error cases ---------------------------------------------------------

TEST(PadTest, WrongPadsLengthThrows) {
    // pads has 3 entries for a rank-2 input: 2*rank = 4 is required.
    auto op = pt_make_op(false, {{"pads", "[1, 0, 0]"}});
    ASSERT_NE(op, nullptr);
    auto x = pt_make<float>({2, 3}, ramp<float>(6));
    pt_upload(x);
    auto out = pt_make_out<float>({3, 4});

    bool threw = false;
    try {
        op->onExecute({x}, {out}, 0);
    } catch (const std::exception &e) {
        threw = true;
        EXPECT_NE(std::string(e.what()).find("pads"), std::string::npos)
            << "message was: " << e.what();
    }
    EXPECT_TRUE(threw);
}

TEST(PadTest, ReflectPadNotSmallerThanInputThrows) {
    // ONNX forbids a reflect pad >= the input size on that axis.
    auto op = pt_make_op(false, {{"pads", "[0, 5, 0, 0]"}, {"mode", "reflect"}});
    ASSERT_NE(op, nullptr);
    auto x = pt_make<float>({2, 4}, ramp<float>(8));
    pt_upload(x);
    auto out = pt_make_out<float>({3, 9});

    bool threw = false;
    try {
        op->onExecute({x}, {out}, 0);
    } catch (const std::exception &e) {
        threw = true;
        EXPECT_NE(std::string(e.what()).find("reflect"), std::string::npos)
            << "message was: " << e.what();
    }
    EXPECT_TRUE(threw);
}

TEST(PadTest, UnknownModeThrows) {
    bool threw = false;
    try {
        auto op = pt_make_op(false, {{"mode", "wrap"}});
        (void)op;
    } catch (const std::exception &e) {
        threw = true;
        EXPECT_NE(std::string(e.what()).find("mode"), std::string::npos)
            << "message was: " << e.what();
    }
    EXPECT_TRUE(threw);
}
