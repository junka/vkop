// Copyright 2026 @junka
// Clip op tests. Clip is buffer-only (no image kernel), so every case drives
// create_from_type(OpType::CLIP, fp16, 0, /*backend_buffer=*/true) like
// WhereTest -- no env var needed for the backend. Reference is torch::clamp,
// which is the op this comes from (torch.clamp(x, min=..) / clamp(x, max=..)).
//
// What the cases are aimed at, in order of how easy each is to get wrong:
//
//   * the ASYMMETRIC forms. ONNX Clip may leave either bound absent, and an
//     absent bound is unbounded. An implementation that fills the missing side
//     with a sentinel (+-FLT_MAX, 0, ...) passes the two-bounds case and
//     quietly clamps the one-bound cases. Both sides are therefore exercised
//     alone, and through BOTH spellings the op accepts:
//       - opset <= 6 attributes  (min= / max= on the node), and
//       - opset >= 11 inputs     (inputs[1]/inputs[2], with the missing side
//         as a NULL entry -- what torch.clamp exports: `in: x, , val_1`).
//   * values EXACTLY at each bound (the boundary must be inclusive) and values
//     far outside it.
//   * NaN. ONNX does not specify Clip's NaN behaviour, so the reference here is
//     torch.clamp, which passes NaN through. The kernel has an explicit isnan()
//     early-out for that: GLSL's max/min are only required to return the
//     non-NaN operand, so a bare max(x, lo) would turn NaN into a bound on
//     some drivers.
//   * fp32 and the packed-half2 fp16 build, on a rank-1 and a rank-4 shape.

#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <vector>

#include "setup.hpp"
#include "core/ShaderKeys.hpp"
#include "core/Tensor.hpp"
#include "include/logger.hpp"
#include "ops/OperatorFactory.hpp"
#include "ops/Ops.hpp"

#include <gtest/gtest.h>

using vkop::core::Tensor;

namespace {

// ---- helpers ---------------------------------------------------------------

// Named static: Tensor::as_storage_buffer takes the device by non-const lvalue
// reference, so a temporary from a getter will not bind.
std::shared_ptr<vkop::VulkanDevice> &clip_dev() {
    static std::shared_ptr<vkop::VulkanDevice> dev = [] {
        if (!vkop::tests::TestEnv::is_initialized()) {
            vkop::tests::TestEnv::initialize();
        }
        return vkop::tests::TestEnv::get_device();
    }();
    return dev;
}
std::shared_ptr<vkop::VulkanCommandPool> clip_cmdpool() {
    return vkop::tests::TestEnv::get_command_pool();
}

template <typename T> float to_f32(T v) {
    if constexpr (std::is_same_v<T, uint16_t>) {
        return vkop::core::ITensor::fp16_to_fp32(v);
    } else {
        return static_cast<float>(v);
    }
}

template <typename T> T from_f32(float v) {
    if constexpr (std::is_same_v<T, uint16_t>) {
        return vkop::core::ITensor::fp32_to_fp16(v);
    } else {
        return static_cast<T>(v);
    }
}

// A tensor uploaded and ready to be read by the op. The CPU copy is kept when
// the value is needed later (the reference comparison does not need it).
template <typename T>
std::shared_ptr<Tensor<T>> clip_tensor(const std::vector<int> &shape,
                                       const std::vector<float> &values) {
    auto t = std::make_shared<Tensor<T>>(shape);
    std::vector<T> v;
    v.reserve(values.size());
    for (float f : values) {
        v.push_back(from_f32<T>(f));
    }
    t->fillToCPU(v);
    t->as_storage_buffer(clip_dev());
    t->copyToGPU(clip_cmdpool());
    return t;
}

std::unique_ptr<vkop::ops::Operator> clip_op(int fp16) {
    auto op = vkop::ops::create_from_type(vkop::ops::OpType::CLIP, fp16, 0,
                                          /*backend_buffer=*/true);
    if (!op) {
        LOG_ERROR("create_from_type(CLIP) returned null");
        return nullptr;
    }
    op->set_runtime_device(clip_dev(), clip_cmdpool());
    return op;
}

void clip_run(vkop::ops::Operator *op) {
    auto cmd = op->get_record();
    std::vector<VkSubmitInfo> info{cmd->buildSubmitInfo()};
    vkop::VulkanCommandBuffer::submit(clip_dev()->getComputeQueue(), info);
    cmd->wait();
    clip_dev()->wait_all_done();
}

struct ClipCase {
    std::vector<int> shape;
    std::vector<float> values;
    // Reference bounds (torch::clamp).
    bool ref_has_min = false;
    bool ref_has_max = false;
    float ref_min = 0.0F;
    float ref_max = 0.0F;
    // How the op is told about them.
    bool attr_form = true;  // true: setAttribute; false: input tensors
};

// Drive one Clip case through the buffer backend and compare element-for-
// element against torch::clamp over the same values. NaN is compared as NaN
// (the reference passes it through, and so must the kernel).
template <typename T> bool run_clip_case(const ClipCase &c) {
    const int n = static_cast<int>(c.values.size());

    // Reference. Build torch from the SAME rounded values the kernel will
    // read (so fp16 rounding of the input is not scored as an op error).
    auto opt = torch::TensorOptions().dtype(
        std::is_same_v<T, uint16_t> ? torch::kFloat16 : torch::kFloat32);
    std::vector<float> rounded;
    rounded.reserve(c.values.size());
    for (float f : c.values) {
        rounded.push_back(to_f32<T>(from_f32<T>(f)));
    }
    auto torch_in = torch::tensor(rounded, opt);
    // torch::clamp REJECTS a call with neither bound ("At least one of 'min'
    // or 'max' must not be None"), so the unbounded identity case has no
    // reference to call -- the reference is the input itself (an ONNX Clip
    // with both bounds absent copies its data through).
    torch::Tensor torch_out;
    if (!c.ref_has_min && !c.ref_has_max) {
        torch_out = torch_in.clone();
    } else {
        torch_out = torch::clamp(
            torch_in,
            c.ref_has_min ? std::optional<at::Scalar>(at::Scalar(c.ref_min))
                          : std::optional<at::Scalar>(),
            c.ref_has_max ? std::optional<at::Scalar>(at::Scalar(c.ref_max))
                          : std::optional<at::Scalar>());
    }

    auto input = clip_tensor<T>(c.shape, c.values);
    auto output = std::make_shared<Tensor<T>>(c.shape);
    output->as_storage_buffer(clip_dev());
    output->toGPU();

    auto op = clip_op(std::is_same_v<T, uint16_t> ? 1 : 0);
    if (!op) {
        return false;
    }

    std::vector<std::shared_ptr<vkop::core::ITensor>> inputs{input};
    if (c.attr_form) {
        std::unordered_map<std::string, std::string> attrs;
        if (c.ref_has_min) {
            attrs["min"] = std::to_string(c.ref_min);
        }
        if (c.ref_has_max) {
            attrs["max"] = std::to_string(c.ref_max);
        }
        op->setAttribute(attrs);
    } else {
        // ONNX opset >= 11 form: the bound travels as an input, and the ABSENT
        // side is a null entry (the exporter writes an empty input name).
        if (c.ref_has_min) {
            inputs.push_back(
                clip_tensor<T>({}, {c.ref_min})); // rank-0 scalar
        } else {
            inputs.push_back(nullptr);
        }
        if (c.ref_has_max) {
            inputs.push_back(clip_tensor<T>({}, {c.ref_max}));
        }
        // An absent max needs no third entry at all; an absent min does, or the
        // max would land in the min slot.
        if (!c.ref_has_max && c.ref_has_min) {
            inputs.push_back(nullptr);
        }
    }

    op->onExecute(inputs, {output}, 0);
    clip_run(op.get());
    output->copyToCPU(clip_cmdpool());

    // The reference in float, from the same dtype torch computed in (kept in a
    // named tensor: a pointer into a temporary would dangle).
    auto ref_flat = torch_out.cpu().contiguous().flatten();
    std::vector<float> expect(static_cast<size_t>(n));
    if constexpr (std::is_same_v<T, uint16_t>) {
        const auto *p = reinterpret_cast<const uint16_t *>(
            ref_flat.template data_ptr<at::Half>());
        for (int i = 0; i < n; ++i) {
            expect[static_cast<size_t>(i)] =
                vkop::core::ITensor::fp16_to_fp32(p[i]);
        }
    } else {
        const auto *p = ref_flat.template data_ptr<float>();
        for (int i = 0; i < n; ++i) {
            expect[static_cast<size_t>(i)] = p[i];
        }
    }

    const auto &ov = output->data();
    for (int i = 0; i < n; ++i) {
        const float got = to_f32<T>(ov[static_cast<size_t>(i)]);
        const float exp = expect[static_cast<size_t>(i)];
        if (std::isnan(exp)) {
            if (!std::isnan(got)) {
                LOG_ERROR("Clip: NaN at %d became %f (must pass through)", i,
                          got);
                return false;
            }
            continue;
        }
        if (std::isnan(got)) {
            LOG_ERROR("Clip: expected %f at %d, got NaN", exp, i);
            return false;
        }
        // fp16 storage + fp16 reference: the only difference left is the
        // kernel's clamp arithmetic, which is exact for these values; a half
        // ulp of slack covers the reference's own rounding.
        const float tol = std::is_same_v<T, uint16_t> ? 1e-3F : 0.0F;
        if (std::abs(got - exp) > tol) {
            LOG_ERROR("Clip: mismatch at %d: %f vs %f", i, got, exp);
            return false;
        }
    }
    return true;
}

// Values that bracket realistic clamps: far below, exactly at, just inside,
// just above, far above -- plus NaN.
const std::vector<float> kProbe = {
    -1000.0F, -5.0F, -1.0F, -0.5F, -0.0F, 0.0F, 0.25F,
    0.5F,     1.0F,  1.5F, 5.0F,  1000.0F, std::numeric_limits<float>::quiet_NaN(),
};

ClipCase make_case(const std::vector<int> &shape, bool has_min, float vmin,
                   bool has_max, float vmax, bool attr_form) {
    ClipCase c;
    c.shape = shape;
    int total = 1;
    for (int d : shape) total *= d;
    c.values.reserve(static_cast<size_t>(total));
    for (int i = 0; i < total; ++i) {
        c.values.push_back(kProbe[static_cast<size_t>(i) % kProbe.size()]);
    }
    c.ref_has_min = has_min;
    c.ref_min = vmin;
    c.ref_has_max = has_max;
    c.ref_max = vmax;
    c.attr_form = attr_form;
    return c;
}

} // namespace

// ---- both bounds ------------------------------------------------------------

TEST(ClipTest, BothBoundsFp32Rank1) {
    // rank-1, exactly at and far outside both bounds.
    EXPECT_TRUE(run_clip_case<float>(make_case({14}, true, -1.0F, true, 1.5F,
                                               /*attr_form=*/true)));
}

TEST(ClipTest, BothBoundsFp32Rank4) {
    EXPECT_TRUE(run_clip_case<float>(
        make_case({2, 3, 2, 2}, true, -0.5F, true, 0.5F, /*attr_form=*/true)));
}

TEST(ClipTest, BothBoundsFp16Rank4) {
    EXPECT_TRUE(run_clip_case<uint16_t>(
        make_case({2, 2, 3, 2}, true, -1.0F, true, 1.5F, /*attr_form=*/true)));
}

// ---- only min / only max (the asymmetric cases) ----------------------------

TEST(ClipTest, OnlyMinAttrFp32) {
    // The naive sentinel implementation leaves max at +FLT_MAX or 0. With
    // max = 0 the positives would be clipped away; with max = +FLT_MAX this
    // case passes, so the value set includes positives both sides of 0.
    EXPECT_TRUE(run_clip_case<float>(
        make_case({14}, true, 0.25F, false, 0.0F, /*attr_form=*/true)));
}

TEST(ClipTest, OnlyMaxAttrFp32) {
    EXPECT_TRUE(run_clip_case<float>(
        make_case({14}, false, 0.0F, true, -0.25F, /*attr_form=*/true)));
}

TEST(ClipTest, OnlyMinAttrFp16) {
    EXPECT_TRUE(run_clip_case<uint16_t>(
        make_case({2, 3, 2, 2}, true, 0.5F, false, 0.0F, /*attr_form=*/true)));
}

TEST(ClipTest, OnlyMaxAttrFp16) {
    EXPECT_TRUE(run_clip_case<uint16_t>(
        make_case({2, 3, 2, 2}, false, 0.0F, true, 1.5F, /*attr_form=*/true)));
}

// ---- the same asymmetric forms through the ONNX >= 11 INPUT spelling --------
// torch.clamp exports this form; with one bound it writes an EMPTY input name
// for the missing side, which the runtime turns into a null input entry.

TEST(ClipTest, OnlyMinInputFp32) {
    EXPECT_TRUE(run_clip_case<float>(
        make_case({2, 7}, true, -1.0F, false, 0.0F, /*attr_form=*/false)));
}

TEST(ClipTest, OnlyMaxInputFp32) {
    EXPECT_TRUE(run_clip_case<float>(
        make_case({2, 7}, false, 0.0F, true, 0.5F, /*attr_form=*/false)));
}

TEST(ClipTest, BothBoundsInputFp32) {
    EXPECT_TRUE(run_clip_case<float>(
        make_case({2, 7}, true, -0.5F, true, 1.5F, /*attr_form=*/false)));
}

TEST(ClipTest, OnlyMinInputFp16) {
    EXPECT_TRUE(run_clip_case<uint16_t>(
        make_case({4, 4}, true, 0.25F, false, 0.0F, /*attr_form=*/false)));
}

TEST(ClipTest, OnlyMaxInputFp16) {
    EXPECT_TRUE(run_clip_case<uint16_t>(
        make_case({4, 4}, false, 0.0F, true, -0.25F, /*attr_form=*/false)));
}

// ---- unbounded on both sides ----------------------------------------------
// ONNX allows a Clip with neither bound (an identity). It falls out of the same
// flags: nothing is applied, and the output must equal the input bit-for-bit.

TEST(ClipTest, NoBoundsFp32IsIdentity) {
    EXPECT_TRUE(run_clip_case<float>(
        make_case({2, 5}, false, 0.0F, false, 0.0F, /*attr_form=*/true)));
}

// ---- NaN ------------------------------------------------------------------
// Pinned explicitly rather than left to the shared case: this is the one
// behaviour ONNX does not specify, and the chosen answer is torch.clamp's --
// pass NaN through unchanged, never clamp it to a bound.

TEST(ClipTest, NanPassesThroughBothBounds) {
    ClipCase c;
    c.shape = {5};
    c.values = {std::numeric_limits<float>::quiet_NaN(), -5.0F, 0.0F, 5.0F,
                std::numeric_limits<float>::quiet_NaN()};
    c.ref_has_min = true;
    c.ref_min = -1.0F;
    c.ref_has_max = true;
    c.ref_max = 1.0F;
    EXPECT_TRUE(run_clip_case<float>(c));
}

TEST(ClipTest, NanPassesThroughMinOnly) {
    ClipCase c;
    c.shape = {3};
    c.values = {std::numeric_limits<float>::quiet_NaN(), -5.0F, 5.0F};
    c.ref_has_min = true;
    c.ref_min = 0.0F;
    EXPECT_TRUE(run_clip_case<float>(c));
}

TEST(ClipTest, NanPassesThroughFp16) {
    ClipCase c;
    c.shape = {4};
    c.values = {std::numeric_limits<float>::quiet_NaN(), -5.0F, 0.5F, 5.0F};
    c.ref_has_min = true;
    c.ref_min = -1.0F;
    c.ref_has_max = true;
    c.ref_max = 1.0F;
    EXPECT_TRUE(run_clip_case<uint16_t>(c));
}

// ---- the shader really is the dual-build pair ------------------------------
// Guards the table wiring this op depends on: the fp16 request must resolve a
// DIFFERENT key than the fp32 one, or the op would run the fp32 module over
// packed half words. (The C++/Python tables are cross-checked by pytest; this
// pins what the op itself will ask for.)

TEST(ClipTest, Fp16ResolvesThePackedBuild) {
    EXPECT_EQ(vkop::core::shader_key_for(vkop::ops::OpType::CLIP, true, 0),
              "buffer_clip");
    EXPECT_EQ(vkop::core::shader_key_for(vkop::ops::OpType::CLIP, true, 1),
              "buffer_clip_fp16");
    // Buffer-only: no image key exists, so an image request falls back to the
    // buffer spelling rather than producing an "image_clip" phantom.
    EXPECT_EQ(vkop::core::shader_key_for(vkop::ops::OpType::CLIP, false, 0),
              "buffer_clip");
}
