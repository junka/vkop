// Copyright 2026 @junka
// InstanceNormalization buffer (SSBO) backend tests.
//
// ONNX InstanceNormalization is what torch's nn.GroupNorm exports to, so the
// GroupNorm case at the bottom is the thing this op exists to unblock (the
// MiniMax-H3 VAE encoder's norm layers). Coverage:
//   - rank-3 [N,C,L] and rank-4 [N,C,H,W], N>1 and C>1 (N=C=1 hides indexing
//     bugs: channel 0 of batch 0 is a slice whose offset is 0 and whose
//     channel index is 0, i.e. every index arithmetic bug still looks right);
//   - non-trivial scale/bias (never all-ones / all-zeros);
//   - epsilon variation;
//   - fp32 and fp16, with the fp16 tolerance the rest of the suite uses;
//   - the GroupNorm-shaped case: a rank-3 [N, G, C/G*spatial] InstanceNorm
//     against torch::nn functional group_norm, which is exactly the
//     Reshape -> InstanceNormalization form the exporter emits;
//   - error cases: wrong scale length, wrong bias length, rank < 3, and an odd
//     spatial extent in the fp16 build.
// The reference is torch's own kernels (at::instance_norm / at::group_norm),
// so its biased variance is what the shader must reproduce -- dividing by M-1
// instead of M is the classic way to be off by a factor and would fail here.
//
// Backend: create_from_type(..., backend_buffer=true). InstanceNorm is
// buffer-only, so asking for the image impl throws; that is asserted too.

#include <cmath>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "setup.hpp"
#include "core/Tensor.hpp"
#include "include/logger.hpp"
#include "ops/OperatorFactory.hpp"
#include "ops/Ops.hpp"

#include <ATen/ops/group_norm.h>
#include <ATen/ops/instance_norm.h>

#include <gtest/gtest.h>

using vkop::core::Tensor;

namespace {

// ---- in_* helpers (mirror BufferRankTest's brt_* set) ----------------------

std::shared_ptr<vkop::VulkanDevice> &in_dev() {
    static std::shared_ptr<vkop::VulkanDevice> dev = [] {
        if (!vkop::tests::TestEnv::is_initialized())
            vkop::tests::TestEnv::initialize();
        return vkop::tests::TestEnv::get_device();
    }();
    return dev;
}
std::shared_ptr<vkop::VulkanCommandPool> in_cmdpool() {
    return vkop::tests::TestEnv::get_command_pool();
}

template <typename T>
std::shared_ptr<Tensor<T>> in_make(const std::vector<int> &shape,
                                   const std::vector<float> &vals) {
    auto t = std::make_shared<Tensor<T>>(shape);
    if constexpr (std::is_same_v<T, uint16_t>) {
        // fillFP32ToCPU rounds once per element, the same rounding the fp16
        // reference carries (see the comparison below).
        std::vector<float> v = vals;
        t->fillFP32ToCPU(v);
    } else {
        t->fillToCPU(vals);
    }
    return t;
}

template <typename T>
void in_upload(std::shared_ptr<Tensor<T>> &t) {
    auto keep = t->data();
    t->as_storage_buffer(in_dev());
    t->copyToGPU(in_cmdpool(), keep.data());
}

template <typename T>
std::shared_ptr<Tensor<T>> in_make_out(const std::vector<int> &shape) {
    auto t = std::make_shared<Tensor<T>>(shape);
    t->as_storage_buffer(in_dev());
    t->toGPU();
    return t;
}

void in_run(vkop::ops::Operator *op) {
    auto cmd = op->get_record();
    std::vector<VkSubmitInfo> info{cmd->buildSubmitInfo()};
    vkop::VulkanCommandBuffer::submit(in_dev()->getComputeQueue(), info);
    cmd->wait();
    in_dev()->wait_all_done();
}

// Round a host array through fp16 and back. The fp16 row's tensors hold the
// rounded values, so its reference must be computed from the SAME values --
// otherwise the comparison measures the input rounding, not the kernel.
std::vector<float> in_round_fp16(const std::vector<float> &v) {
    std::vector<float> r(v.size());
    for (size_t i = 0; i < v.size(); ++i) {
        r[i] = vkop::core::ITensor::fp16_to_fp32(
            vkop::core::ITensor::fp32_to_fp16(v[i]));
    }
    return r;
}

// Compare against a torch reference. `rtol`/`atol` are absolute/relative on the
// fp32 values (fp16 readings are widened first). The fp32 row uses a tight
// tolerance: the shader's two-pass (mean, then variance around it) reduction
// is exact enough that anything looser would hide a biased-vs-unbiased or
// wrong-slice bug.
template <typename T>
bool in_close(const std::shared_ptr<Tensor<T>> &out, const torch::Tensor &ref,
              float rtol, float atol) {
    auto flat = ref.cpu().contiguous().flatten();
    if (flat.scalar_type() != at::kFloat) flat = flat.to(at::kFloat);
    auto refc = flat.contiguous();
    auto refv = refc.accessor<float, 1>();
    const int n = out->num_elements();
    if (n != static_cast<int>(refc.numel())) {
        LOG_ERROR("InstanceNorm size mismatch: out=%d ref=%lld", n,
                  static_cast<long long>(refc.numel()));
        return false;
    }
    float worst_err = 0.0f;
    int worst_i = -1;
    for (int i = 0; i < n; ++i) {
        float got;
        if constexpr (std::is_same_v<T, uint16_t>) {
            got = vkop::core::ITensor::fp16_to_fp32((*out)[i]);
        } else {
            got = static_cast<float>((*out)[i]);
        }
        const float exp = refv[i];
        const float thr = std::max(std::abs(exp) * rtol, atol);
        const float err = std::abs(got - exp);
        if (err > worst_err) {
            worst_err = err;
            worst_i = i;
        }
        if (err > thr) {
            LOG_ERROR("InstanceNorm mismatch at %d: got=%f exp=%f (thr=%f)",
                      i, got, exp, thr);
            return false;
        }
    }
    LOG_INFO("InstanceNorm ok: max |err| %g at %d", worst_err, worst_i);
    return true;
}

// Drive one InstanceNorm dispatch. `x` is [N, C, *spatial]; `scale`/`bias` are
// [C] raw host arrays in the test's element type. Returns the output tensor
// (nullptr on a failed dispatch), so a caller can chain the GroupNorm tail.
template <typename T>
std::shared_ptr<Tensor<T>> in_run_case(
    const std::vector<int> &shape, const std::vector<float> &x,
    const std::vector<float> &scale, const std::vector<float> &bias,
    float eps, const std::unordered_map<std::string, std::string> &attrs = {}) {
    auto input = in_make<T>(shape, x);
    auto scale_t = in_make<T>({static_cast<int>(scale.size())}, scale);
    auto bias_t = in_make<T>({static_cast<int>(bias.size())}, bias);
    in_upload(input);
    in_upload(scale_t);
    in_upload(bias_t);
    auto output = in_make_out<T>(shape);

    auto op = vkop::ops::create_from_type(vkop::ops::OpType::INSTANCENORM,
                                          std::is_same_v<T, uint16_t> ? 1 : 0,
                                          0, /*backend_buffer=*/true);
    if (!op) {
        LOG_ERROR("create_from_type(INSTANCENORM) returned null");
        return nullptr;
    }
    op->set_runtime_device(in_dev(), in_cmdpool());
    std::unordered_map<std::string, std::string> a{{"epsilon",
                                                    std::to_string(eps)}};
    for (const auto &kv : attrs) a[kv.first] = kv.second;
    op->setAttribute(a);
    op->onExecute({input, scale_t, bias_t}, {output}, 0);
    in_run(op.get());
    output->copyToCPU(in_cmdpool());
    return output;
}

// Deterministic, non-trivial data with a per-(n, c) offset: an instance norm
// whose slice means are all zero cannot tell a correct mean subtraction from a
// dropped one, and one whose scale/bias are 1/0 cannot tell the affine apart
// from a no-op.
struct Case {
    std::vector<int> shape;
    float eps;
};

std::vector<float> in_make_x(const std::vector<int> &shape) {
    int total = 1;
    for (int d : shape) total *= d;
    const int channels = shape[1];
    const int spatial = total / (shape[0] * channels);
    std::vector<float> x(total);
    for (int idx = 0; idx < total; ++idx) {
        const int c = (idx / spatial) % channels;
        // Per-channel offset (1.5 * c) plus a hashed spread, so mean != 0 and
        // var != 1 for every slice.
        const float base = 1.5f * static_cast<float>(c) - 2.0f;
        const float wig = std::sin(static_cast<float>(idx) * 0.37f) * 1.3f +
                          std::cos(static_cast<float>(idx) * 0.11f) * 0.6f;
        x[idx] = base + wig;
    }
    return x;
}

std::vector<float> in_make_scale(int channels) {
    std::vector<float> s(channels);
    for (int c = 0; c < channels; ++c) {
        s[c] = 0.5f + 0.7f * static_cast<float>(c) - 0.25f * (c % 3);
    }
    return s;
}

std::vector<float> in_make_bias(int channels) {
    std::vector<float> b(channels);
    for (int c = 0; c < channels; ++c) {
        b[c] = -0.9f + 0.4f * static_cast<float>(c);
    }
    return b;
}

// torch reference for the same problem. at::instance_norm with
// use_input_stats=true computes the mean and BIASED variance over each
// (n, c) slice's spatial extent, which is ONNX's definition.
torch::Tensor in_ref_instance_norm(const std::vector<int> &shape,
                                   const std::vector<float> &x,
                                   const std::vector<float> &scale,
                                   const std::vector<float> &bias, float eps) {
    const int channels = shape[1];
    auto xt = torch::from_blob(const_cast<float *>(x.data()),
                               std::vector<int64_t>(shape.begin(), shape.end()),
                               at::TensorOptions().dtype(at::kFloat))
                  .clone();
    auto st = torch::from_blob(const_cast<float *>(scale.data()),
                               {channels},
                               at::TensorOptions().dtype(at::kFloat))
                  .clone();
    auto bt = torch::from_blob(const_cast<float *>(bias.data()),
                               {channels},
                               at::TensorOptions().dtype(at::kFloat))
                  .clone();
    return at::instance_norm(xt, st, bt, /*running_mean=*/{}, /*running_var=*/{},
                             /*use_input_stats=*/true, /*momentum=*/0.1,
                             /*eps=*/eps, /*cudnn_enabled=*/false);
}

// One full case: shape + eps, checked against torch in both dtypes.
// The fp32 band (1e-4 relative/absolute) is loose enough to absorb the
// different summation ORDER torch and the kernel use, and tight enough that
// the bugs this op can have are caught by two orders of magnitude: an
// M-1 variance, a wrong slice, a dropped mean, or a scale/bias indexed by
// slice instead of channel all move values by >= 1e-2.
bool in_check_case(const Case &c, float fp32_rtol = 1e-4f,
                   float fp32_atol = 1e-4f, float fp16_rtol = 2e-2f,
                   float fp16_atol = 2e-2f) {
    const int channels = c.shape[1];
    auto x = in_make_x(c.shape);
    auto scale = in_make_scale(channels);
    auto bias = in_make_bias(channels);
    auto ref = in_ref_instance_norm(c.shape, x, scale, bias, c.eps);

    auto out = in_run_case<float>(c.shape, x, scale, bias, c.eps);
    if (!out) return false;
    if (!in_close<float>(out, ref, fp32_rtol, fp32_atol)) {
        LOG_ERROR("  (fp32, shape %s eps %g)",
                  vkop::core::shape_to_string(c.shape).c_str(), c.eps);
        return false;
    }

    // The fp16 row's reference is computed in fp32 from the SAME fp16-rounded
    // inputs the kernel sees, then rounded once to half: the kernel widens
    // half to fp32 for the reduce and rounds only on store, so this is the
    // exact arithmetic it performs. Comparing against the unrounded reference
    // instead would measure the rounding of the OUTPUT, not the kernel.
    const auto x16 = in_round_fp16(x);
    const auto scale16 = in_round_fp16(scale);
    const auto bias16 = in_round_fp16(bias);
    auto ref16 = in_ref_instance_norm(c.shape, x16, scale16, bias16, c.eps);
    auto out16 = in_run_case<uint16_t>(c.shape, x, scale, bias, c.eps);
    if (!out16) return false;
    if (!in_close<uint16_t>(out16, ref16.to(at::kHalf), fp16_rtol, fp16_atol)) {
        LOG_ERROR("  (fp16, shape %s eps %g)",
                  vkop::core::shape_to_string(c.shape).c_str(), c.eps);
        return false;
    }
    return true;
}

} // namespace

// ---- shapes ----------------------------------------------------------------

TEST(InstanceNormTest, Rank3MultiBatchMultiChannel) {
    // [N, C, L] with N>1 and C>1: batch * C slices, so both the batch offset
    // and the channel index feed the slice arithmetic.
    EXPECT_TRUE(in_check_case({{2, 3, 8}, 1e-5f}));
}

TEST(InstanceNormTest, OddSpatialExtentFp32) {
    // L=7: a slice that does not fill a whole number of half2 words. The fp32
    // build handles any extent; the fp16 build requires an EVEN one (its write
    // pass owns whole words and two workgroups would race on a shared word) and
    // throws otherwise -- see Fp16OddSpatialExtentThrows.
    const std::vector<int> shape{2, 3, 7};
    const int channels = shape[1];
    auto x = in_make_x(shape);
    auto scale = in_make_scale(channels);
    auto bias = in_make_bias(channels);
    auto ref = in_ref_instance_norm(shape, x, scale, bias, 1e-5f);
    auto out = in_run_case<float>(shape, x, scale, bias, 1e-5f);
    ASSERT_TRUE(out);
    EXPECT_TRUE(in_close<float>(out, ref, 1e-4f, 1e-4f));
}

TEST(InstanceNormTest, Rank4MultiBatchMultiChannel) {
    // [N, C, H, W]: the GroupNorm-relevant layout (video VAE feature maps).
    EXPECT_TRUE(in_check_case({{2, 3, 4, 5}, 1e-5f}));
}

TEST(InstanceNormTest, Rank5SpatialProduct) {
    // [N, C, D, H, W]: reduction extent is the product of dims 2..rank-1.
    EXPECT_TRUE(in_check_case({{2, 2, 2, 3, 4}, 1e-5f}));
}

TEST(InstanceNormTest, SingleBatchSingleChannel) {
    // The degenerate case: every indexing bug still looks right here, so it is
    // a canary, not the main coverage.
    EXPECT_TRUE(in_check_case({{1, 1, 32}, 1e-5f}));
}

// ---- epsilon ---------------------------------------------------------------

TEST(InstanceNormTest, EpsilonVariation) {
    // A near-constant slice makes the eps term dominate, which is the only way
    // to see whether the attribute reached the shader at all (and whether a
    // dropped eps would divide by ~0).
    const std::vector<int> shape{1, 2, 16};
    std::vector<float> x(32);
    for (int i = 0; i < 32; ++i) x[i] = 3.0f + 1e-4f * static_cast<float>(i % 4);
    auto scale = in_make_scale(2);
    auto bias = in_make_bias(2);
    for (float eps : {1e-5f, 1e-3f, 0.5f}) {
        auto ref = in_ref_instance_norm(shape, x, scale, bias, eps);
        auto out = in_run_case<float>(shape, x, scale, bias, eps);
        ASSERT_TRUE(out);
        EXPECT_TRUE(in_close<float>(out, ref, 1e-4f, 1e-4f)) << "eps=" << eps;
    }
}

// ---- the GroupNorm shape ---------------------------------------------------

// nn.GroupNorm(C, G) exports as
//   Reshape [N,C,*spatial] -> [N,G,C/G*spatial]
//   -> InstanceNormalization(scale=[G], bias=[G], epsilon)
//   -> Reshape back -> Mul(weight[C]) -> Add(bias[C])
// (verified by exporting a tiny one at opset 17). So the op really only ever
// sees a rank-3 [N, G, C/G*spatial] tensor, and THIS test is that exact
// problem: run InstanceNorm on the reshaped tensor and compare against
// at::group_norm on the original layout. No GroupNorm-specific path exists.
TEST(InstanceNormTest, GroupNormEquivalentFp32) {
    const int N = 2, C = 8, G = 4, H = 5, W = 3;
    const int spatial = H * W;
    const int group_channels = C / G; // dims[2] of the InstanceNorm input
    const float eps = 1e-5f;

    // A GroupNorm-shaped problem: per-(n, c) offsets and a varied spread, so
    // grouping channels correctly matters.
    const int total = N * C * spatial;
    std::vector<float> x(total);
    for (int i = 0; i < total; ++i) {
        const int c = (i / spatial) % C;
        x[i] = 1.1f * static_cast<float>(c) - 3.0f +
               std::sin(static_cast<float>(i) * 0.23f) * 1.7f;
    }

    // GroupNorm's affine is per CHANNEL ([C]), while the InstanceNorm the graph
    // emits carries ones/zeros ([G]) and the affine is applied afterwards by
    // the separate Mul/Add. Take the exported form literally: ones/zeros here.
    std::vector<float> one(static_cast<size_t>(G), 1.0f);
    std::vector<float> zero(static_cast<size_t>(G), 0.0f);

    auto ref = at::group_norm(
        torch::from_blob(x.data(), {N, C, H, W},
                         at::TensorOptions().dtype(at::kFloat))
            .clone(),
        G, /*weight=*/{}, /*bias=*/{}, eps, /*cudnn_enabled=*/false);

    // What the exporter hands the InstanceNorm node.
    const std::vector<int> in_shape{N, G, group_channels * spatial};
    auto out = in_run_case<float>(in_shape, x, one, zero, eps);
    ASSERT_TRUE(out);
    EXPECT_TRUE(in_close<float>(out, ref, 1e-4f, 1e-4f))
        << "GroupNorm decomposition through a rank-3 InstanceNormalization "
           "must reproduce at::group_norm";
}

TEST(InstanceNormTest, GroupNormEquivalentFp16) {
    // The same problem in fp16, which is the dtype the VAE encoder runs in.
    const int N = 2, C = 4, G = 2, H = 4, W = 2;
    const int spatial = H * W;
    const int group_channels = C / G;
    const float eps = 1e-5f;

    const int total = N * C * spatial;
    std::vector<float> x(total);
    for (int i = 0; i < total; ++i) {
        const int c = (i / spatial) % C;
        x[i] = 0.8f * static_cast<float>(c) + 0.5f +
               std::cos(static_cast<float>(i) * 0.41f) * 1.2f;
    }
    std::vector<float> one(static_cast<size_t>(G), 1.0f);
    std::vector<float> zero(static_cast<size_t>(G), 0.0f);

    // Same contract as in_check_case: the kernel sees fp16-rounded inputs and
    // accumulates in fp32, so the reference is computed in fp32 from the
    // rounded values and rounded once on output.
    const auto x16 = in_round_fp16(x);
    auto ref16 = at::group_norm(
        torch::from_blob(const_cast<float *>(x16.data()), {N, C, H, W},
                         at::TensorOptions().dtype(at::kFloat))
            .clone(),
        G, /*weight=*/{}, /*bias=*/{}, eps, /*cudnn_enabled=*/false);

    const std::vector<int> in_shape{N, G, group_channels * spatial};
    auto out = in_run_case<uint16_t>(in_shape, x, one, zero, eps);
    ASSERT_TRUE(out);
    auto out16_ref = ref16.to(at::kHalf);
    EXPECT_TRUE(in_close<uint16_t>(out, out16_ref, 2e-2f, 2e-2f));
}

// ---- affine and the shader build ------------------------------------------

TEST(InstanceNormTest, NonTrivialAffine) {
    // A per-channel scale/bias that differs across channels is what proves the
    // shader indexes scale/bias by channel and not by slice.
    EXPECT_TRUE(in_check_case({{3, 4, 6}, 1e-5f}));
}

// ---- errors ----------------------------------------------------------------

namespace {

// A dispatch that must throw, with the reason named in the message.
template <typename Fn>
void in_expect_throw(Fn &&fn, const std::string &needle) {
    try {
        fn();
    } catch (const std::exception &e) {
        EXPECT_NE(std::string(e.what()).find(needle), std::string::npos)
            << "message was: " << e.what();
        return;
    }
    ADD_FAILURE() << "expected a throw containing \"" << needle << "\"";
}

} // namespace

TEST(InstanceNormTest, WrongScaleLengthThrows) {
    const std::vector<int> shape{2, 3, 4};
    auto x = in_make_x(shape);
    auto scale = in_make_scale(2); // 2 elements for 3 channels
    auto bias = in_make_bias(3);
    auto input = in_make<float>(shape, x);
    auto scale_t = in_make<float>({2}, scale);
    auto bias_t = in_make<float>({3}, bias);
    in_upload(input);
    in_upload(scale_t);
    in_upload(bias_t);
    auto output = in_make_out<float>(shape);

    auto op = vkop::ops::create_from_type(vkop::ops::OpType::INSTANCENORM, 0, 0,
                                          /*backend_buffer=*/true);
    ASSERT_NE(op, nullptr);
    op->set_runtime_device(in_dev(), in_cmdpool());
    in_expect_throw(
        [&] { op->onExecute({input, scale_t, bias_t}, {output}, 0); },
        "scale has 2 elements but the input has 3 channels");
}

TEST(InstanceNormTest, WrongBiasLengthThrows) {
    const std::vector<int> shape{2, 3, 4};
    auto x = in_make_x(shape);
    auto scale = in_make_scale(3);
    auto bias = in_make_bias(4); // 4 elements for 3 channels
    auto input = in_make<float>(shape, x);
    auto scale_t = in_make<float>({3}, scale);
    auto bias_t = in_make<float>({4}, bias);
    in_upload(input);
    in_upload(scale_t);
    in_upload(bias_t);
    auto output = in_make_out<float>(shape);

    auto op = vkop::ops::create_from_type(vkop::ops::OpType::INSTANCENORM, 0, 0,
                                          /*backend_buffer=*/true);
    ASSERT_NE(op, nullptr);
    op->set_runtime_device(in_dev(), in_cmdpool());
    in_expect_throw(
        [&] { op->onExecute({input, scale_t, bias_t}, {output}, 0); },
        "bias has 4 elements but the input has 3 channels");
}

TEST(InstanceNormTest, RankTooLowThrows) {
    // Rank 2 has no spatial extent to normalize over; reading dims[2..] would
    // be past the shape.
    const std::vector<int> shape{2, 3};
    auto input = in_make<float>(shape, std::vector<float>(6, 1.0f));
    auto scale_t = in_make<float>({3}, in_make_scale(3));
    auto bias_t = in_make<float>({3}, in_make_bias(3));
    in_upload(input);
    in_upload(scale_t);
    in_upload(bias_t);
    auto output = in_make_out<float>(shape);

    auto op = vkop::ops::create_from_type(vkop::ops::OpType::INSTANCENORM, 0, 0,
                                          /*backend_buffer=*/true);
    ASSERT_NE(op, nullptr);
    op->set_runtime_device(in_dev(), in_cmdpool());
    in_expect_throw(
        [&] { op->onExecute({input, scale_t, bias_t}, {output}, 0); },
        "rank 3..8");
}

TEST(InstanceNormTest, Fp16OddSpatialExtentThrows) {
    // An odd slice starts mid-word, so its last half would share a word with
    // the next slice and two workgroups would read-modify-write it.
    const std::vector<int> shape{1, 2, 5};
    auto input = in_make<uint16_t>(shape, std::vector<float>(10, 1.0f));
    auto scale_t = in_make<uint16_t>({2}, in_make_scale(2));
    auto bias_t = in_make<uint16_t>({2}, in_make_bias(2));
    in_upload(input);
    in_upload(scale_t);
    in_upload(bias_t);
    auto output = in_make_out<uint16_t>(shape);

    auto op = vkop::ops::create_from_type(vkop::ops::OpType::INSTANCENORM, 1, 0,
                                          /*backend_buffer=*/true);
    ASSERT_NE(op, nullptr);
    op->set_runtime_device(in_dev(), in_cmdpool());
    in_expect_throw(
        [&] { op->onExecute({input, scale_t, bias_t}, {output}, 0); },
        "odd spatial extent");
}

TEST(InstanceNormTest, ImageBackendIsUnavailable) {
    // Buffer-only by construction: no image kernel exists, and silently
    // substituting one would run a different op's shader.
    EXPECT_THROW(
        vkop::ops::create_from_type(vkop::ops::OpType::INSTANCENORM, 0, 0,
                                    /*backend_buffer=*/false),
        std::exception);
}
