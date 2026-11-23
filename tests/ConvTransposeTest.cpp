// Copyright 2026 @junka
// ConvTranspose op tests (buffer / SSBO backend only).
//
// The op exists for the MiniMax-H3 audio VAE (BigVGAN), whose export has a
// stack of 1-D ConvTranspose upsamplers. Every numeric expectation here comes
// from libtorch's conv_transpose1d / conv_transpose2d — this tree links
// libtorch for exactly this reason, and a hand-written reference would only
// re-derive whatever mistake the kernel makes.
//
// What is pinned:
//   * 1-D stride 1, no padding (the plain form);
//   * 1-D stride > 1 WITH output_padding — the off-by-one case, where
//     output_padding extends the output without changing which input taps can
//     reach it;
//   * 1-D depthwise (groups == channels; BigVGAN's upsamplers) and grouped
//     with 1 < groups < channels, so the weight's OCG axis is exercised
//     independently of the depthwise shortcut;
//   * 2-D, and 2-D with stride 2 + output_padding 1;
//   * asymmetric `pads` (begin != end). torch cannot express that directly,
//     so the reference is the same conv_transpose at padding 0 cropped by
//     [begin, end) on each axis — which is exactly what ONNX padding means;
//   * the `output_shape` attribute, both the natural size and a trimmed one;
//   * a bias-free node;
//   * fp32 and fp16 across those rows;
//   * a weight whose channel axes disagree with `group` must throw rather
//     than compute garbage.
//
// fp16 expectations are built from values already rounded to half, so the
// torch reference walks precisely the numbers the GPU walks.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#include "core/Tensor.hpp"
#include "include/logger.hpp"
#include "setup.hpp"
#include "ops/OperatorFactory.hpp"
#include "ops/Ops.hpp"

#include <gtest/gtest.h>

using vkop::core::Tensor;

namespace {

struct CtDev {
    std::shared_ptr<vkop::VulkanDevice> dev;
    std::shared_ptr<vkop::VulkanCommandPool> cmdpool;
    CtDev() {
        if (!vkop::tests::TestEnv::is_initialized())
            vkop::tests::TestEnv::initialize();
        dev = vkop::tests::TestEnv::get_device();
        cmdpool = vkop::tests::TestEnv::get_command_pool();
    }
};

// Round an fp32 torch tensor through fp16 and back, so an fp16 run and its
// reference see identical inputs (the GPU is handed the half values).
torch::Tensor ct_half_rounded(torch::Tensor t) {
    return t.to(torch::kFloat16).to(torch::kFloat32);
}

std::vector<int> ct_shape_of(const torch::Tensor &t) {
    std::vector<int> s;
    for (auto d : t.sizes()) {
        s.push_back(static_cast<int>(d));
    }
    return s;
}

template <typename T>
std::shared_ptr<Tensor<T>> ct_make(const torch::Tensor &src) {
    auto t = std::make_shared<Tensor<T>>(ct_shape_of(src));
    auto flat = src.cpu().contiguous().flatten();
    std::vector<T> v;
    v.reserve(flat.numel());
    if constexpr (std::is_same_v<T, uint16_t>) {
        flat = flat.to(torch::kFloat16);
        const auto *p =
            reinterpret_cast<const uint16_t *>(flat.data_ptr<at::Half>());
        v.assign(p, p + flat.numel());
    } else {
        flat = flat.to(torch::kFloat32);
        const auto *p = flat.data_ptr<float>();
        v.assign(p, p + flat.numel());
    }
    t->fillToCPU(v);
    return t;
}

template <typename T>
bool ct_close(const std::shared_ptr<Tensor<T>> &got, const torch::Tensor &want,
              float rtol, float atol) {
    auto flat = want.cpu().contiguous().flatten();
    if (got->num_elements() != static_cast<int>(flat.numel())) {
        LOG_ERROR("ConvTranspose: output has %d elems, reference %lld",
                  got->num_elements(), static_cast<long long>(flat.numel()));
        return false;
    }
    auto w = flat.accessor<float, 1>();
    // Report the actual error magnitude, not just a pass/fail: a grouped or
    // output_padding row that is "close" and one that is exact should not read
    // the same in the log.
    float max_abs = 0.0F;
    float max_rel = 0.0F;
    int worst = -1;
    for (int i = 0; i < got->num_elements(); ++i) {
        float g = 0.0F;
        if constexpr (std::is_same_v<T, uint16_t>) {
            g = vkop::core::ITensor::fp16_to_fp32((*got)[i]);
        } else {
            g = (*got)[i];
        }
        const float e = w[i];
        const float thr = std::max(std::abs(e) * rtol, atol);
        if (!std::isnan(g)) {
            const float aerr = std::abs(g - e);
            const float rerr = aerr / std::max(1e-6F, std::abs(e));
            if (aerr > max_abs) {
                max_abs = aerr;
                worst = i;
            }
            max_rel = std::max(max_rel, rerr);
        }
        if (std::isnan(g) || std::abs(g - e) > thr) {
            LOG_ERROR("ConvTranspose mismatch at %d: got %f want %f (thr %f)",
                      i, g, e, thr);
            return false;
        }
    }
    LOG_INFO("ConvTranspose ok: n=%d max_abs_err=%.3e max_rel_err=%.3e",
             got->num_elements(), max_abs, max_rel);
    (void)worst;
    return true;
}

template <typename T>
std::shared_ptr<Tensor<T>> ct_make_out(const std::vector<int> &shape,
                                       CtDev &d) {
    auto t = std::make_shared<Tensor<T>>(shape);
    t->as_storage_buffer(d.dev);
    t->toGPU();
    return t;
}

void ct_run_op(vkop::ops::Operator *op, CtDev &d) {
    auto cmd = op->get_record();
    std::vector<VkSubmitInfo> info{cmd->buildSubmitInfo()};
    vkop::VulkanCommandBuffer::submit(d.dev->getComputeQueue(), info);
    cmd->wait();
    d.dev->wait_all_done();
}

// Reference signature: (input, weight, bias) -> expected output. `bias` is an
// undefined tensor when the case has no bias.
using CtRef = std::function<torch::Tensor(const torch::Tensor &,
                                          const torch::Tensor &,
                                          const torch::Tensor &)>;

template <typename T>
void ct_upload(std::shared_ptr<Tensor<T>> &t, CtDev &d) {
    t->as_storage_buffer(d.dev);
    t->copyToGPU(d.cmdpool);
}

// The typed body: builds the tensors, runs the op, compares. Split from
// ct_case only so the fp16/fp32 choice can pick the element type once.
template <typename T>
bool ct_case_typed(const torch::Tensor &x, const torch::Tensor &w,
                   const torch::Tensor &bias, bool with_bias,
                   const std::vector<std::string> &attr_keys,
                   const std::vector<std::string> &attr_vals,
                   const torch::Tensor &want, bool fp16) {
    CtDev d;

    auto op = vkop::ops::create_from_type(vkop::ops::OpType::CONVTRANSPOSE,
                                          fp16 ? 1 : 0, 0,
                                          /*backend_buffer=*/true);
    if (!op) {
        LOG_ERROR("ConvTranspose: create_from_type returned null");
        return false;
    }
    if (!op->uses_buffer_backend()) {
        LOG_ERROR("ConvTranspose: op does not serve the buffer backend");
        return false;
    }
    op->set_runtime_device(d.dev, d.cmdpool);
    std::unordered_map<std::string, std::string> attrs;
    for (size_t i = 0; i < attr_keys.size(); ++i) {
        attrs[attr_keys[i]] = attr_vals[i];
    }
    op->setAttribute(attrs);

    auto xin = ct_make<T>(x);
    auto wt = ct_make<T>(w);
    ct_upload(xin, d);
    ct_upload(wt, d);

    std::vector<std::shared_ptr<vkop::core::ITensor>> inputs{xin, wt};
    std::shared_ptr<Tensor<T>> bi;
    if (with_bias) {
        bi = ct_make<T>(bias);
        ct_upload(bi, d);
        inputs.push_back(bi);
    }

    auto out = ct_make_out<T>(ct_shape_of(want), d);
    op->onExecute(inputs, {out}, 0);
    ct_run_op(op.get(), d);
    out->copyToCPU(d.cmdpool);

    const float rtol = fp16 ? 0.02F : 0.01F;
    const float atol = fp16 ? 0.02F : 0.002F;
    return ct_close(out, want, rtol, atol);
}

// One numeric row.
bool ct_case(const std::vector<int> &in_shape,
             const std::vector<int> &w_shape, bool with_bias, int groups,
             const std::vector<std::pair<std::string, std::string>> &attrs,
             const CtRef &ref, bool fp16) {
    torch::manual_seed(20260101);

    // libtorch's randn takes an IntArrayRef, so the shapes are widened here
    // rather than relying on a vector<int> -> IntArrayRef conversion (there is
    // none).
    auto i64 = [](const std::vector<int> &v) {
        return std::vector<int64_t>(v.begin(), v.end());
    };
    const auto opts = torch::TensorOptions().dtype(torch::kFloat32);
    auto x = torch::randn(i64(in_shape), opts);
    auto w = torch::randn(i64(w_shape), opts);
    auto bias = torch::randn(std::vector<int64_t>{w_shape[1] * groups}, opts);
    if (fp16) {
        x = ct_half_rounded(x);
        w = ct_half_rounded(w);
        bias = ct_half_rounded(bias);
    }
    auto want = ref(x, w, with_bias ? bias : torch::Tensor());
    if (!want.defined()) {
        LOG_ERROR("ConvTranspose: reference produced nothing");
        return false;
    }

    std::vector<std::string> keys, vals;
    for (const auto &kv : attrs) {
        keys.push_back(kv.first);
        vals.push_back(kv.second);
    }
    return fp16 ? ct_case_typed<uint16_t>(x, w, bias, with_bias, keys, vals,
                                          want, true)
                : ct_case_typed<float>(x, w, bias, with_bias, keys, vals, want,
                                       false);
}

// A '['-bracket attribute list, the form the model loader produces.
std::string ct_list(const std::vector<int> &v) {
    std::string s = "[";
    for (size_t i = 0; i < v.size(); ++i) {
        s += std::to_string(v[i]);
        if (i + 1 < v.size()) {
            s += ", ";
        }
    }
    return s + "]";
}

// Crops the first `begin` and last `end` elements of each spatial axis — the
// ONNX meaning of pads=[begin, end] for a transposed convolution, and the only
// way to express an asymmetric pad through torch.
torch::Tensor ct_crop(const torch::Tensor &t, const std::vector<int> &begin,
                      const std::vector<int> &end) {
    std::vector<torch::indexing::TensorIndex> ix;
    ix.emplace_back(torch::indexing::Slice());
    ix.emplace_back(torch::indexing::Slice());
    for (size_t d = 0; d < begin.size(); ++d) {
        const int64_t dim = static_cast<int64_t>(t.dim()) -
                            static_cast<int64_t>(begin.size()) +
                            static_cast<int64_t>(d);
        ix.emplace_back(torch::indexing::Slice(
            begin[d], t.size(dim) - end[d]));
    }
    return t.index(ix);
}

// ---- 1-D references -------------------------------------------------------

// ONNX ConvTranspose1d with symmetric padding == torch conv_transpose1d.
CtRef ct_ref_1d(int stride, int pad, int out_pad, int groups) {
    return [=](const torch::Tensor &x, const torch::Tensor &w,
               const torch::Tensor &b) {
        return torch::conv_transpose1d(x, w, b, torch::IntArrayRef({stride}),
                                       torch::IntArrayRef({pad}),
                                       torch::IntArrayRef({out_pad}), groups,
                                       torch::IntArrayRef({1}));
    };
}

// Asymmetric pads: run at padding 0 and crop, so begin != end is expressible.
CtRef ct_ref_1d_asym(int stride, int dil, int pad_begin, int pad_end,
                     int groups) {
    return [=](const torch::Tensor &x, const torch::Tensor &w,
               const torch::Tensor &b) {
        auto full = torch::conv_transpose1d(
            x, w, b, torch::IntArrayRef({stride}), torch::IntArrayRef({0}),
            torch::IntArrayRef({0}), groups, torch::IntArrayRef({dil}));
        return ct_crop(full, {pad_begin}, {pad_end});
    };
}

CtRef ct_ref_2d(int stride, int pad, int out_pad, int groups,
                int dil) {
    return [=](const torch::Tensor &x, const torch::Tensor &w,
               const torch::Tensor &b) {
        return torch::conv_transpose2d(
            x, w, b, torch::IntArrayRef({stride, stride}),
            torch::IntArrayRef({pad, pad}),
            torch::IntArrayRef({out_pad, out_pad}), groups,
            torch::IntArrayRef({dil, dil}));
    };
}

// ---- rows ----------------------------------------------------------------

// 1-D, stride 1, no padding, bias, fp32 + fp16.
void ct_row_1d_plain(bool fp16) {
    const std::vector<int> x_shape{2, 3, 8};
    const std::vector<int> w_shape{3, 4, 3}; // [IC, OCG, K] -> OC 4
    EXPECT_TRUE(ct_case(x_shape, w_shape, /*with_bias=*/true, 1,
                        {{"strides", ct_list({1})}, {"pads", ct_list({0, 0})},
                         {"group", "1"}},
                        ct_ref_1d(1, 0, 0, 1), fp16));
}

// 1-D upsampler with stride > 1 and output_padding: the off-by-one case.
void ct_row_1d_strided(bool fp16) {
    const std::vector<int> x_shape{1, 4, 6};
    const std::vector<int> w_shape{4, 8, 4}; // [IC, OCG, K] -> OC 8
    EXPECT_TRUE(ct_case(x_shape, w_shape, /*with_bias=*/true, 1,
                        {{"strides", ct_list({2})}, {"pads", ct_list({1, 1})},
                         {"output_padding", ct_list({1})}, {"group", "1"}},
                        ct_ref_1d(2, 1, 1, 1), fp16));
    // stride 3, output_padding 2, no padding at all.
    EXPECT_TRUE(ct_case({1, 2, 5}, {2, 6, 3}, /*with_bias=*/true, 1,
                        {{"strides", ct_list({3})}, {"pads", ct_list({0, 0})},
                         {"output_padding", ct_list({2})}, {"group", "1"}},
                        ct_ref_1d(3, 0, 2, 1), fp16));
}

// 1-D depthwise (BigVGAN's upsamplers: groups == channels).
void ct_row_1d_depthwise(bool fp16) {
    const int C = 6;
    EXPECT_TRUE(ct_case({1, C, 9}, {C, 1, 4}, /*with_bias=*/true, C,
                        {{"strides", ct_list({2})}, {"pads", ct_list({1, 1})},
                         {"output_padding", ct_list({1})},
                         {"group", std::to_string(C)}},
                        ct_ref_1d(2, 1, 1, C), fp16));
}

// 1-D grouped, groups strictly between 1 and C.
void ct_row_1d_grouped(bool fp16) {
    const int G = 2;
    EXPECT_TRUE(ct_case({2, 4, 7}, {4, 3, 3}, /*with_bias=*/true, G,
                        {{"strides", ct_list({2})}, {"pads", ct_list({0, 0})},
                         {"output_padding", ct_list({0})},
                         {"group", std::to_string(G)}},
                        ct_ref_1d(2, 0, 0, G), fp16));
}

// 1-D asymmetric pads: begin != end.
void ct_row_1d_asymmetric_pads(bool fp16) {
    const std::vector<int> pads{1, 3}; // begin 1, end 3
    EXPECT_TRUE(ct_case({1, 3, 8}, {3, 5, 4}, /*with_bias=*/true, 1,
                        {{"strides", ct_list({1})}, {"pads", ct_list(pads)},
                         {"group", "1"}},
                        ct_ref_1d_asym(1, 1, pads[0], pads[1], 1), fp16));
    // Same idea with stride 2 and a dilation of 2, so the crop and the tap
    // spacing are both exercised at once.
    EXPECT_TRUE(ct_case({1, 2, 6}, {2, 4, 3}, /*with_bias=*/false, 1,
                        {{"strides", ct_list({2})}, {"pads", ct_list({2, 1})},
                         {"dilations", ct_list({2})}, {"group", "1"}},
                        ct_ref_1d_asym(2, 2, 2, 1, 1), fp16));
}

// 1-D with the `output_shape` attribute trimming the natural output.
void ct_row_1d_output_shape(bool fp16) {
    // natural: 2*(6-1) + 0 + (3-1) + 1 = 13; ask for 12, so total padding 1.
    // ONNX's else branch splits that as pad_begin = 1, pad_end = 0, i.e. the
    // FIRST element is dropped -- verified against onnxruntime, where pads
    // [0,0] and [1,0] give byte-identical output once output_shape is set
    // (output_shape overrides pads). Cropping the tail instead would shift
    // every value by one position, which is the off-by-one this row exists to
    // catch.
    EXPECT_TRUE(ct_case({1, 3, 6}, {3, 4, 3}, /*with_bias=*/true, 1,
                        {{"strides", ct_list({2})},
                         {"output_shape", ct_list({12})}, {"group", "1"}},
                        [](const torch::Tensor &x, const torch::Tensor &w,
                           const torch::Tensor &b) {
                            auto full = torch::conv_transpose1d(
                                x, w, b, torch::IntArrayRef({2}),
                                torch::IntArrayRef({0}),
                                torch::IntArrayRef({0}), 1,
                                torch::IntArrayRef({1}));
                            return full.index(
                                {torch::indexing::Slice(),
                                 torch::indexing::Slice(),
                                 torch::indexing::Slice(1, 13)});
                        },
                        fp16));
    // And the same node WITH an explicit `pads` set: onnxruntime ignores it
    // when output_shape is present, so the expected output is identical. A
    // kernel that rejected or honoured the pads here would disagree.
    EXPECT_TRUE(ct_case({1, 3, 6}, {3, 4, 3}, /*with_bias=*/true, 1,
                        {{"strides", ct_list({2})}, {"pads", ct_list({0, 0})},
                         {"output_shape", ct_list({12})}, {"group", "1"}},
                        [](const torch::Tensor &x, const torch::Tensor &w,
                           const torch::Tensor &b) {
                            auto full = torch::conv_transpose1d(
                                x, w, b, torch::IntArrayRef({2}),
                                torch::IntArrayRef({0}),
                                torch::IntArrayRef({0}), 1,
                                torch::IntArrayRef({1}));
                            return full.index(
                                {torch::indexing::Slice(),
                                 torch::indexing::Slice(),
                                 torch::indexing::Slice(1, 13)});
                        },
                        fp16));
}

// 2-D, stride 1, padding 1, and 2-D upsampler with output_padding.
void ct_row_2d(bool fp16) {
    EXPECT_TRUE(ct_case({1, 3, 5, 6}, {3, 2, 3, 3}, /*with_bias=*/true, 1,
                        {{"strides", ct_list({1, 1})},
                         {"pads", ct_list({1, 1, 1, 1})}, {"group", "1"}},
                        ct_ref_2d(1, 1, 0, 1, 1), fp16));
    EXPECT_TRUE(ct_case({1, 4, 4, 3}, {4, 2, 3, 3}, /*with_bias=*/true, 2,
                        {{"strides", ct_list({2, 2})},
                         {"pads", ct_list({1, 1, 1, 1})},
                         {"output_padding", ct_list({1, 1})},
                         {"group", "2"}},
                        ct_ref_2d(2, 1, 1, 2, 1), fp16));
}

// The kernel must actually run, not silently fall back: the op resolves
// buffer_convtranspose from the build archive and builds a pipeline from it.
void ct_row_shader_resolves(bool fp16) {
    CtDev d;
    auto op = vkop::ops::create_from_type(vkop::ops::OpType::CONVTRANSPOSE,
                                          fp16 ? 1 : 0, 0,
                                          /*backend_buffer=*/true);
    ASSERT_NE(op, nullptr);
    op->set_runtime_device(d.dev, d.cmdpool);
    // set_runtime_device resolves the primary spv; a missing key throws with
    // "this libvkop carries no built-in SPIR-V" naming the op. Reaching here
    // without a throw means the archive carried the key.
    SUCCEED();
}

} // namespace

// fp32 rows.
TEST(ConvTransposeTest, OneDimensionalPlainFp32) { ct_row_1d_plain(false); }
TEST(ConvTransposeTest, OneDimensionalStridedOutputPaddingFp32) {
    ct_row_1d_strided(false);
}
TEST(ConvTransposeTest, OneDimensionalDepthwiseFp32) {
    ct_row_1d_depthwise(false);
}
TEST(ConvTransposeTest, OneDimensionalGroupedFp32) {
    ct_row_1d_grouped(false);
}
TEST(ConvTransposeTest, OneDimensionalAsymmetricPadsFp32) {
    ct_row_1d_asymmetric_pads(false);
}
TEST(ConvTransposeTest, OneDimensionalOutputShapeFp32) {
    ct_row_1d_output_shape(false);
}
TEST(ConvTransposeTest, TwoDimensionalFp32) { ct_row_2d(false); }

// fp16 rows: same shapes, fp16 build.
TEST(ConvTransposeTest, OneDimensionalPlainFp16) { ct_row_1d_plain(true); }
TEST(ConvTransposeTest, OneDimensionalStridedOutputPaddingFp16) {
    ct_row_1d_strided(true);
}
TEST(ConvTransposeTest, OneDimensionalDepthwiseFp16) {
    ct_row_1d_depthwise(true);
}
TEST(ConvTransposeTest, OneDimensionalGroupedFp16) {
    ct_row_1d_grouped(true);
}
TEST(ConvTransposeTest, OneDimensionalAsymmetricPadsFp16) {
    ct_row_1d_asymmetric_pads(true);
}
TEST(ConvTransposeTest, OneDimensionalOutputShapeFp16) {
    ct_row_1d_output_shape(true);
}
TEST(ConvTransposeTest, TwoDimensionalFp16) { ct_row_2d(true); }

// The archive key resolves for both builds (i.e. CMake compiled and packed
// buffer/convtranspose and its -DFP16 twin).
TEST(ConvTransposeTest, ShaderKeyResolvesFp32) { ct_row_shader_resolves(false); }
TEST(ConvTransposeTest, ShaderKeyResolvesFp16) { ct_row_shader_resolves(true); }

// The image backend must be refused by name instead of silently building the
// buffer op, exactly like every other buffer-only façade.
TEST(ConvTransposeTest, ImageBackendRefused) {
    EXPECT_THROW(
        vkop::ops::create_from_type(vkop::ops::OpType::CONVTRANSPOSE, 0, 0,
                                    /*backend_buffer=*/false),
        std::runtime_error);
}

// A weight whose channel axes do not agree with `group` must throw before any
// dispatch: computing anyway would read the wrong channels and produce a
// plausible-looking wrong answer.
TEST(ConvTransposeTest, WeightInconsistentWithGroupsThrows) {
    CtDev d;
    // weight first dim (input channels) = 4, but the input has 4 and group 3
    // does not divide it.
    auto op = vkop::ops::create_from_type(vkop::ops::OpType::CONVTRANSPOSE, 0,
                                          0, /*backend_buffer=*/true);
    ASSERT_NE(op, nullptr);
    op->set_runtime_device(d.dev, d.cmdpool);
    op->setAttribute({{"strides", ct_list({2})}, {"group", "3"}});

    auto x = ct_make<float>(torch::zeros({1, 4, 8}));
    auto w = ct_make<float>(torch::zeros({4, 1, 3}));
    ct_upload(x, d);
    ct_upload(w, d);
    auto out = ct_make_out<float>({1, 3, 20}, d);
    EXPECT_THROW(op->onExecute({x, w}, {out}, 0), std::runtime_error);
}

// A Conv-shaped weight fed to ConvTranspose has its two channel axes swapped:
// its dim 0 is OC, not IC, so it no longer matches the input's channel count
// and must be rejected rather than silently read down the wrong axis.
TEST(ConvTransposeTest, ConvStyleWeightRejected) {
    CtDev d;
    auto op = vkop::ops::create_from_type(vkop::ops::OpType::CONVTRANSPOSE, 0,
                                          0, /*backend_buffer=*/true);
    ASSERT_NE(op, nullptr);
    op->set_runtime_device(d.dev, d.cmdpool);
    op->setAttribute({{"strides", ct_list({1})}, {"group", "1"}});

    auto x = ct_make<float>(torch::zeros({1, 4, 8}));
    // Conv layout [OC=8, IC/group=1, K=3] against a 4-channel input: dim 0 is
    // 8, but ConvTranspose requires it to be IC (4).
    auto w = ct_make<float>(torch::zeros({8, 1, 3}));
    ct_upload(x, d);
    ct_upload(w, d);
    auto out = ct_make_out<float>({1, 1, 10}, d);
    EXPECT_THROW(op->onExecute({x, w}, {out}, 0), std::runtime_error);
}
