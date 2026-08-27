// Copyright 2026 @junka
// All-rank buffer (SSBO) backend tests. Drives each shape-critical buffer op
// through ranks 0..max (per op) against a libtorch reference, under
// create_from_type(..., backend_buffer=true) — no env var needed. Covers the
// scalar (rank-0) and empty (0-dim) edge cases that the LLM (Qwen3-VL) hits:
//   - 56 Gather nodes with 5-D data [1,2,8,kv,128] + scalar index (KV-cache
//     slice). gather.comp is hard 4-D (ivec4) → these document the cap.
//   - Concat_5 mixes an empty (kv_len=0) input with a real one.
//   - Expand right-aligned broadcast to 5-D (rotary).
//   - Reshape to/from 5-D and to a rank-0 scalar.
//
// fp16 is exercised only for ops that ship an fp16 buffer shader (Reshape,
// Reduce, Expand, Range, Gather). Concat/Slice/Transpose/Cast are fp32-only
// structural shaders (see shaders/common/buffer_common.comp) → fp16 rows are
// skipped for them.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <vector>

#include "setup.hpp"
#include "core/Tensor.hpp"
#include "include/logger.hpp"
#include "ops/OperatorFactory.hpp"
#include "ops/Ops.hpp"

#include <gtest/gtest.h>
#include <torch/torch.h>

using vkop::core::Tensor;

namespace {

// ---- generic brt_fill / compare (fp32 + fp16) -------------------------------

template <typename T>
void brt_fill(std::shared_ptr<Tensor<T>> &t, const torch::Tensor &tt) {
    auto flat = tt.cpu().contiguous().flatten();
    std::vector<T> v;
    if constexpr (std::is_same_v<T, uint16_t>) {
        const auto *p =
            reinterpret_cast<const uint16_t *>(flat.data_ptr<at::Half>());
        v.assign(p, p + flat.numel());
    } else {
        auto a = flat.accessor<T, 1>();
        v.assign(a.data(), a.data() + flat.numel());
    }
    t->fillToCPU(v);
}

template <typename T>
bool brt_close_to_torch(const std::shared_ptr<Tensor<T>> &out,
                    const torch::Tensor &ref, float rtol = 0.02f,
                    float atol = 0.02f) {
    auto flat = ref.cpu().contiguous().flatten();
    int n = out->num_elements();
    if (n != flat.numel()) {
        LOG_ERROR("size mismatch: out=%d ref=%lld", n,
                  (long long)flat.numel());
        return false;
    }
    for (int i = 0; i < n; ++i) {
        float got, exp;
        if constexpr (std::is_same_v<T, uint16_t>) {
            got = vkop::core::ITensor::fp16_to_fp32((*out)[i]);
            exp = vkop::core::ITensor::fp16_to_fp32(
                reinterpret_cast<const uint16_t *>(flat.data_ptr<at::Half>())[i]);
        } else {
            got = float((*out)[i]);
            exp = float(flat[i].template item<T>());
        }
        float thr = std::max(std::abs(exp) * rtol, atol);
        if (std::abs(got - exp) > thr) {
            LOG_ERROR("mismatch %d: got=%f exp=%f (thr=%f)", i, got, exp, thr);
            return false;
        }
    }
    return true;
}

torch::TensorOptions brt_opt(bool fp16) {
    return torch::TensorOptions().dtype(fp16 ? torch::kFloat16 : torch::kFloat32);
}

template <typename T> torch::TensorOptions brt_torch_opt();
template <> torch::TensorOptions brt_torch_opt<float>() { return brt_opt(false); }
template <> torch::TensorOptions brt_torch_opt<uint16_t>() { return brt_opt(true); }

std::vector<int> brt_to_int_shape(const torch::Tensor &t) {
    std::vector<int> s;
    for (auto d : t.sizes())
        s.push_back(static_cast<int>(d));
    return s;
}

// Build an int64 tensor of the given shape from values.
std::shared_ptr<Tensor<int64_t>>
brt_make_i64(const std::vector<int64_t> &vals, const std::vector<int> &shape) {
    auto t = std::make_shared<Tensor<int64_t>>(shape);
    t->fillToCPU(vals);
    return t;
}

// Device + cmdpool handles (lazily-initialized via TestEnv).
struct Dev {
    std::shared_ptr<vkop::VulkanDevice> dev;
    std::shared_ptr<vkop::VulkanCommandPool> cmdpool;
    Dev() {
        if (!vkop::tests::TestEnv::is_initialized())
            vkop::tests::TestEnv::initialize();
        dev = vkop::tests::TestEnv::get_device();
        cmdpool = vkop::tests::TestEnv::get_command_pool();
    }
};

// Upload an fp32/fp16 data tensor as an SSBO (CPU copy discarded).
template <typename T>
void brt_upload(std::shared_ptr<Tensor<T>> &t, Dev &d) {
    t->as_storage_buffer(d.dev);
    t->copyToGPU(d.cmdpool);
}

// Upload an int64 tensor as an SSBO, keeping its CPU copy (downstream host
// reads — Gather/Slice/Expand shape meta-chain — need data_ alive).
void brt_upload_i64(std::shared_ptr<Tensor<int64_t>> &t, Dev &d) {
    auto keep = t->data();
    t->as_storage_buffer(d.dev);
    t->copyToGPU(d.cmdpool, keep.data());
}

// Allocate an output SSBO pre-sized to `shape` (off-GPU ctor → SSBO + mark
// on-GPU so copyToCPU later reads the shader-written buffer back).
template <typename T>
std::shared_ptr<Tensor<T>> brt_make_out(const std::vector<int> &shape, Dev &d) {
    auto t = std::make_shared<Tensor<T>>(shape);
    t->as_storage_buffer(d.dev);
    t->toGPU();
    return t;
}

// Create + setRuntimeDevice + (optional) attributes.
std::unique_ptr<vkop::ops::Operator>
brt_make_op(vkop::ops::OpType type, bool fp16,
        const std::unordered_map<std::string, std::string> &attrs, Dev &d) {
    auto op = vkop::ops::create_from_type(type, fp16 ? 1 : 0, 0,
                                          /*backend_buffer=*/true);
    if (!op) {
        LOG_ERROR("create_from_type returned null");
        return nullptr;
    }
    op->set_runtime_device(d.dev, d.cmdpool);
    if (!attrs.empty())
        op->setAttribute(attrs);
    return op;
}

void brt_run_op(vkop::ops::Operator *op, Dev &d) {
    auto cmd = op->get_record();
    std::vector<VkSubmitInfo> info{cmd->buildSubmitInfo()};
    vkop::VulkanCommandBuffer::submit(d.dev->getComputeQueue(), info);
    cmd->wait();
    d.dev->wait_all_done();
}

// =========================================================================
// Gather
// =========================================================================

// ONNX Gather via torch indexing: data.index([... indices at axis ...]).
torch::Tensor brt_ref_gather(const torch::Tensor &data, const torch::Tensor &idx,
                         int axis) {
    if (axis < 0)
        axis += data.dim();
    std::vector<torch::indexing::TensorIndex> ix;
    ix.reserve(data.dim());
    for (int i = 0; i < axis; ++i)
        ix.emplace_back(torch::indexing::Slice());
    ix.emplace_back(idx);
    for (int i = axis + 1; i < data.dim(); ++i)
        ix.emplace_back(torch::indexing::Slice());
    return data.index(ix);
}

template <typename T>
bool brt_gather_case(const std::vector<int> &data_shape,
                 const std::vector<int> &idx_shape,
                 const std::vector<int64_t> &idx_vals, int axis, bool fp16) {
    Dev d;
    auto torch_data = torch::randn(
        std::vector<int64_t>(data_shape.begin(), data_shape.end()),
        brt_torch_opt<T>());
    auto torch_idx =
        torch::tensor(idx_vals, torch::TensorOptions().dtype(torch::kInt64))
            .reshape(std::vector<int64_t>(idx_shape.begin(), idx_shape.end()));
    auto torch_out = brt_ref_gather(torch_data, torch_idx, axis);

    auto input = std::make_shared<Tensor<T>>(data_shape);
    brt_fill(input, torch_data);
    brt_upload(input, d);

    auto indices = brt_make_i64(idx_vals, idx_shape);
    brt_upload_i64(indices, d);

    auto output = brt_make_out<T>(brt_to_int_shape(torch_out), d);
    auto op = brt_make_op(vkop::ops::OpType::GATHER, fp16,
                      {{"axis", std::to_string(axis)}}, d);
    if (!op)
        return false;
    op->onExecute({input, indices}, {output}, 0);
    brt_run_op(op.get(), d);
    output->copyToCPU(d.cmdpool);
    return brt_close_to_torch(output, torch_out);
}

TEST(BufferRankTest, GatherScalarIndexRank0) {
    // 3-D data, scalar [] index at axis 0 → rank-2 output.
    EXPECT_TRUE(brt_gather_case<float>({2, 3, 4}, {}, {1}, 0, false));
    EXPECT_TRUE(brt_gather_case<uint16_t>({2, 3, 4}, {}, {1}, 0, true));
    // scalar index at axis 1 → rank-2 output (drops axis 1).
    EXPECT_TRUE(brt_gather_case<float>({2, 3, 4}, {}, {2}, 1, false));
    // scalar index at last axis → rank-2 output.
    EXPECT_TRUE(brt_gather_case<float>({2, 3, 4}, {}, {0}, 2, false));
}

TEST(BufferRankTest, GatherEmptyData) {
    // 5-D data with a 0 dim (kv_len=0), scalar idx axis 1 → [1,8,0,128] (ne=0).
    // Exercises the re-resize fix: output must be empty, not stale 1024 elems.
    EXPECT_TRUE(brt_gather_case<float>({1, 2, 8, 0, 128}, {}, {0}, 1, false));
}

TEST(BufferRankTest, Gather5D_DataRankExceedsIvec4Cap) {
    // The 56 LLM KV-cache gathers: 5-D data [1,2,8,kv,128], scalar idx axis 1.
    // gather.comp was ivec4 (4-D); inShape[4] overflowed into indicesShape.
    // Now migrated to IArr8 (rank ≤ 8) so 5-D data works.
    EXPECT_TRUE(brt_gather_case<float>({1, 2, 8, 3, 128}, {}, {0}, 1, false));
    EXPECT_TRUE(brt_gather_case<float>({1, 2, 8, 3, 128}, {1}, {1}, 1, false));
    // fp16 path (fp16 gather spv variant) on 5-D data.
    EXPECT_TRUE(brt_gather_case<uint16_t>({1, 2, 8, 3, 128}, {1}, {1}, 1, true));
}

TEST(BufferRankTest, GatherRank1IndexOn4D) {
    // 4-D data, 1-D index [1] axis 1 → rank-4 output (axis dim replaced by 1).
    EXPECT_TRUE(brt_gather_case<float>({1, 3, 4, 5}, {1}, {1}, 1, false));
    EXPECT_TRUE(brt_gather_case<uint16_t>({1, 3, 4, 5}, {1}, {2}, 1, true));
}

TEST(BufferRankTest, GatherMultiElementIndex) {
    // 3-D data, 2-elem 1-D index axis 2 → output [...,2,...].
    EXPECT_TRUE(brt_gather_case<float>({2, 3, 4}, {2}, {0, 3}, 2, false));
}

// =========================================================================
// Concat (fp32-only shader)
// =========================================================================

template <typename T>
bool brt_concat_case(const std::vector<std::vector<int>> &shapes, int axis,
                 bool /*fp16*/) {
    Dev d;
    std::vector<torch::Tensor> torch_in;
    std::vector<std::shared_ptr<vkop::core::ITensor>> inputs;
    for (const auto &s : shapes) {
        auto t = torch::randn(
            std::vector<int64_t>(s.begin(), s.end()), brt_torch_opt<T>());
        torch_in.push_back(t);
        auto in = std::make_shared<Tensor<T>>(s);
        brt_fill(in, t);
        brt_upload(in, d);
        inputs.push_back(in);
    }
    auto torch_out = torch::cat(torch_in, axis);
    auto output = brt_make_out<T>(brt_to_int_shape(torch_out), d);

    auto op = brt_make_op(vkop::ops::OpType::CONCAT, false,
                      {{"axis", std::to_string(axis)}}, d);
    if (!op)
        return false;
    op->onExecute(inputs, {output}, 0);
    brt_run_op(op.get(), d);
    output->copyToCPU(d.cmdpool);
    return brt_close_to_torch(output, torch_out);
}

TEST(BufferRankTest, ConcatRank5WithEmptyInput) {
    // The Concat_5 reproducer: empty (kv_len=0) + real, axis 2.
    // Result must equal the non-empty input.
    EXPECT_TRUE(brt_concat_case<float>({{1, 8, 0, 128}, {1, 8, 1, 128}}, 2, false));
    // Empty first then real along axis 1.
    EXPECT_TRUE(brt_concat_case<float>({{1, 0, 4}, {1, 3, 4}}, 1, false));
}

TEST(BufferRankTest, ConcatRank5AllAxes) {
    // For concat along axis `ax`, all non-axis dims must match between inputs.
    // Base shape [1,2,3,4,2]; the second input varies only the axis dim.
    std::vector<int> base = {1, 2, 3, 4, 2};
    for (int ax = 0; ax < 5; ++ax) {
        auto b = base;
        b[ax] = base[ax] + 1; // only the concat axis differs
        EXPECT_TRUE(brt_concat_case<float>({base, b}, ax, false));
    }
}

TEST(BufferRankTest, ConcatRank4AxisEachPosition) {
    // Base [2,3,4,5]; second input varies only the axis dim.
    std::vector<int> base = {2, 3, 4, 5};
    for (int ax = 0; ax < 4; ++ax) {
        auto b = base;
        b[ax] = base[ax] + 2;
        EXPECT_TRUE(brt_concat_case<float>({base, b}, ax, false));
    }
}

TEST(BufferRankTest, ConcatThreeInputsRank3) {
    EXPECT_TRUE(brt_concat_case<float>({{3, 2, 4}, {1, 2, 4}, {4, 2, 4}}, 0, false));
    EXPECT_TRUE(brt_concat_case<float>({{1, 4, 6}, {1, 4, 3}}, 2, false));
}

// =========================================================================
// Expand (fp16 via PC flag set from input dtype)
// =========================================================================

// ONNX Expand reference: output dim = max(in_dim, target_dim) right-aligned;
// an input dim of 1 broadcasts to the output dim; an input dim larger than the
// target dim is kept (torch.expand would reject this, so build it manually).
torch::Tensor ref_expand(const torch::Tensor &in,
                         const std::vector<int> &target_shape) {
    auto in_shape = in.sizes();
    int rank = static_cast<int>(std::max(in_shape.size(), target_shape.size()));
    std::vector<int64_t> out_shape(rank);
    std::vector<int64_t> in_padded(rank, 1);
    for (int i = 0; i < rank; ++i) {
        int ii = static_cast<int>(in_shape.size()) - 1 - i;
        int ti = static_cast<int>(target_shape.size()) - 1 - i;
        int id = (ii >= 0) ? static_cast<int>(in_shape[ii]) : 1;
        int td = (ti >= 0) ? target_shape[ti] : 1;
        int v = std::max(id, td);
        if (td == 0 || id == 0)
            v = 0;
        out_shape[rank - 1 - i] = v;
        in_padded[rank - 1 - i] = id;
    }
    // Reshape input to the right-aligned padded shape (1s in the new leading
    // dims), then expand to out_shape (now every in dim is 1 or == out dim).
    auto reshaped = in.reshape(in_padded);
    return reshaped.expand(out_shape).contiguous();
}

template <typename T>
bool brt_expand_case(const std::vector<int> &in_shape,
                 const std::vector<int> &target_shape, bool fp16) {
    Dev d;
    auto torch_in = torch::randn(
        std::vector<int64_t>(in_shape.begin(), in_shape.end()), brt_torch_opt<T>());
    auto torch_out = ref_expand(torch_in, target_shape);

    auto input = std::make_shared<Tensor<T>>(in_shape);
    brt_fill(input, torch_in);
    brt_upload(input, d);

    auto shape_t = brt_make_i64(
        std::vector<int64_t>(target_shape.begin(), target_shape.end()),
        {static_cast<int>(target_shape.size())});
    brt_upload_i64(shape_t, d);

    auto output = brt_make_out<T>(brt_to_int_shape(torch_out), d);
    auto op = brt_make_op(vkop::ops::OpType::EXPAND, fp16, {}, d);
    if (!op)
        return false;
    op->onExecute({input, shape_t}, {output}, 0);
    brt_run_op(op.get(), d);
    output->copyToCPU(d.cmdpool);
    return brt_close_to_torch(output, torch_out);
}

TEST(BufferRankTest, ExpandRightAlignedBroadcastTo5D) {
    // Rotary Expand: [8,1,128] -> [1,8,2,1,128] (right-aligned broadcast).
    EXPECT_TRUE(brt_expand_case<float>({8, 1, 128}, {1, 8, 2, 1, 128}, false));
    EXPECT_TRUE(brt_expand_case<uint16_t>({8, 1, 128}, {1, 8, 2, 1, 128}, true));
}

TEST(BufferRankTest, ExpandInputDimLargerThanTargetKept) {
    // ONNX Expand: output dim = max(in, target). An input dim larger than the
    // target dim is KEPT (not broadcast away).
    EXPECT_TRUE(brt_expand_case<float>({3, 9, 1}, {3, 9, 3}, false));
    EXPECT_TRUE(brt_expand_case<float>({3, 8}, {2, 3, 8}, false));
}

TEST(BufferRankTest, ExpandScalarBroadcast) {
    // rank-0 input (scalar) -> target [4].
    EXPECT_TRUE(brt_expand_case<float>({}, {4}, false));
}

// =========================================================================
// Reshape (fp16 buffer shader exists)
// =========================================================================

template <typename T>
bool brt_reshape_case(const std::vector<int> &in_shape,
                  const std::vector<int> &out_shape, bool fp16) {
    Dev d;
    auto torch_in = torch::randn(
        std::vector<int64_t>(in_shape.begin(), in_shape.end()), brt_torch_opt<T>());
    auto torch_out = torch_in.reshape(
        std::vector<int64_t>(out_shape.begin(), out_shape.end()));

    auto input = std::make_shared<Tensor<T>>(in_shape);
    brt_fill(input, torch_in);
    brt_upload(input, d);

    auto shape_t = brt_make_i64(
        std::vector<int64_t>(out_shape.begin(), out_shape.end()),
        {static_cast<int>(out_shape.size())});
    brt_upload_i64(shape_t, d);

    auto output = brt_make_out<T>(out_shape, d);
    auto op = brt_make_op(vkop::ops::OpType::RESHAPE, fp16, {}, d);
    if (!op)
        return false;
    op->onExecute({input, shape_t}, {output}, 0);
    brt_run_op(op.get(), d);
    output->copyToCPU(d.cmdpool);
    return brt_close_to_torch(output, torch_out);
}

TEST(BufferRankTest, Reshape5DToFlatAndBack) {
    EXPECT_TRUE(brt_reshape_case<float>({1, 2, 8, 3, 128}, {6144}, false));
    EXPECT_TRUE(brt_reshape_case<float>({6144}, {1, 2, 8, 3, 128}, false));
}

TEST(BufferRankTest, ReshapeToScalar) {
    // [1] -> [] (rank-0 scalar). Exercises the 1-elem-scalar Tensor fix.
    EXPECT_TRUE(brt_reshape_case<float>({1}, {}, false));
}

TEST(BufferRankTest, ReshapeRank3To2) {
    EXPECT_TRUE(brt_reshape_case<float>({2, 3, 4}, {6, 4}, false));
    EXPECT_TRUE(brt_reshape_case<uint16_t>({2, 3, 4}, {6, 4}, true));
}

// =========================================================================
// Slice (fp32-only shader; starts/ends/[axes]/[steps] as int64 inputs)
// =========================================================================

template <typename T>
bool brt_slice_case(const std::vector<int> &in_shape,
                const std::vector<int64_t> &starts,
                const std::vector<int64_t> &ends,
                const std::vector<int64_t> &axes,
                const std::vector<int64_t> &steps, bool fp16) {
    Dev d;
    auto torch_in = torch::randn(
        std::vector<int64_t>(in_shape.begin(), in_shape.end()), brt_torch_opt<T>());
    int rank = in_shape.size();

    std::vector<torch::indexing::TensorIndex> ix;
    for (int dd = 0; dd < rank; ++dd) {
        auto ait = std::find(axes.begin(), axes.end(), dd);
        if (ait == axes.end()) {
            ix.emplace_back(torch::indexing::Slice());
            continue;
        }
        int k = static_cast<int>(std::distance(axes.begin(), ait));
        int64_t step = (steps.size() > static_cast<size_t>(k)) ? steps[k] : 1;
        ix.emplace_back(torch::indexing::Slice(
            static_cast<int64_t>(starts[k]), static_cast<int64_t>(ends[k]),
            step));
    }
    auto torch_out = torch_in.index(ix);

    auto input = std::make_shared<Tensor<T>>(in_shape);
    brt_fill(input, torch_in);
    brt_upload(input, d);

    auto st = brt_make_i64(starts, {static_cast<int>(starts.size())});
    auto en = brt_make_i64(ends, {static_cast<int>(ends.size())});
    brt_upload_i64(st, d);
    brt_upload_i64(en, d);
    std::shared_ptr<Tensor<int64_t>> ax, sp;
    if (!axes.empty()) {
        ax = brt_make_i64(axes, {static_cast<int>(axes.size())});
        brt_upload_i64(ax, d);
    }
    if (!steps.empty()) {
        sp = brt_make_i64(steps, {static_cast<int>(steps.size())});
        brt_upload_i64(sp, d);
    }

    std::vector<std::shared_ptr<vkop::core::ITensor>> ins = {input, st, en};
    if (ax)
        ins.push_back(ax);
    if (sp)
        ins.push_back(sp);

    auto output = brt_make_out<T>(brt_to_int_shape(torch_out), d);
    auto op = brt_make_op(vkop::ops::OpType::SLICE, fp16, {}, d);
    if (!op)
        return false;
    op->onExecute(ins, {output}, 0);
    brt_run_op(op.get(), d);
    output->copyToCPU(d.cmdpool);
    return brt_close_to_torch(output, torch_out);
}

TEST(BufferRankTest, SliceRank4PerAxis) {
    // 4-D, slice axis 1 and 3 with steps.
    EXPECT_TRUE(brt_slice_case<float>({2, 6, 4, 8}, {1, 0}, {5, 8}, {1, 3},
                                  {1, 2}, false));
}

TEST(BufferRankTest, SliceRank3NegativeEnd) {
    // end beyond dim clamps; negative start.
    EXPECT_TRUE(
        brt_slice_case<float>({3, 7, 5}, {1, -2}, {2, 7}, {0, 1}, {}, false));
}

TEST(BufferRankTest, SliceEmptyResult) {
    // start==end → empty output along that axis.
    EXPECT_TRUE(brt_slice_case<float>({2, 4, 5}, {0}, {0}, {1}, {1}, false));
}

// =========================================================================
// Transpose (fp32-only shader; perm attribute; up to 8-D)
// =========================================================================

template <typename T>
bool brt_transpose_case(const std::vector<int> &in_shape,
                    const std::vector<int> &perm, bool fp16) {
    Dev d;
    auto torch_in = torch::randn(
        std::vector<int64_t>(in_shape.begin(), in_shape.end()), brt_torch_opt<T>());
    std::vector<int64_t> perm64(perm.begin(), perm.end());
    auto torch_out = torch_in.permute(perm64).contiguous();

    auto input = std::make_shared<Tensor<T>>(in_shape);
    brt_fill(input, torch_in);
    brt_upload(input, d);

    auto output = brt_make_out<T>(brt_to_int_shape(torch_out), d);
    // vkopbin serializes INTS attrs as "[a, b, c]" (bracketed, ", "-separated)
    // and Operator::parse_attr_list requires the brackets — a bare "0,3,2,1"
    // parses to an empty vector, which would leave perm_ empty and segfault.
    std::string perms = "[";
    for (size_t i = 0; i < perm.size(); ++i) {
        if (i)
            perms += ", ";
        perms += std::to_string(perm[i]);
    }
    perms += "]";
    auto op = brt_make_op(vkop::ops::OpType::TRANSPOSE, fp16, {{"perm", perms}}, d);
    if (!op)
        return false;
    op->onExecute({input}, {output}, 0);
    brt_run_op(op.get(), d);
    output->copyToCPU(d.cmdpool);
    return brt_close_to_torch(output, torch_out);
}

TEST(BufferRankTest, TransposeRank4Perms) {
    EXPECT_TRUE(brt_transpose_case<float>({2, 3, 4, 5}, {0, 3, 2, 1}, false));
    EXPECT_TRUE(brt_transpose_case<float>({2, 3, 4, 5}, {3, 2, 1, 0}, false));
    EXPECT_TRUE(brt_transpose_case<float>({1, 8, 2, 128}, {0, 2, 1, 3}, false));
}

TEST(BufferRankTest, TransposeRank2) {
    EXPECT_TRUE(brt_transpose_case<float>({4, 7}, {1, 0}, false));
}

// =========================================================================
// Reduce (fp16 buffer shader exists; reduce_op is lowercase)
// =========================================================================

template <typename T>
bool brt_reduce_case(const std::vector<int> &in_shape,
                 const std::string &reduce_op, const std::vector<int> &axes,
                 int keepdims, bool fp16) {
    Dev d;
    auto torch_in = torch::randn(
        std::vector<int64_t>(in_shape.begin(), in_shape.end()), brt_torch_opt<T>());
    std::vector<int64_t> axes64(axes.begin(), axes.end());

    torch::Tensor torch_out;
    if (reduce_op == "mean") {
        torch_out = torch_in.mean(axes64, /*keepdim=*/keepdims != 0);
    } else if (reduce_op == "sum") {
        torch_out = torch_in.sum(axes64, /*keepdim=*/keepdims != 0);
    } else if (reduce_op == "max") {
        // Tensor::max takes a single dim; use the free function for a
        // multi-axis max reduction.
        torch_out = torch::amax(torch_in, axes64, /*keepdim=*/keepdims != 0);
    } else if (reduce_op == "min") {
        torch_out = torch::amin(torch_in, axes64, /*keepdim=*/keepdims != 0);
    } else {
        return false;
    }
    torch_out = torch_out.contiguous();

    auto input = std::make_shared<Tensor<T>>(in_shape);
    brt_fill(input, torch_in);
    brt_upload(input, d);

    auto output = brt_make_out<T>(brt_to_int_shape(torch_out), d);
    // axes is an INTS attr → must be bracketed "[a, b]" for parse_attr_list.
    std::string axes_str = "[";
    for (size_t i = 0; i < axes.size(); ++i) {
        if (i)
            axes_str += ", ";
        axes_str += std::to_string(axes[i]);
    }
    axes_str += "]";
    auto op = brt_make_op(vkop::ops::OpType::REDUCE, fp16,
                      {{"reduce_op", reduce_op},
                       {"axes", axes_str},
                       {"keepdims", std::to_string(keepdims)}}, d);
    if (!op)
        return false;
    op->onExecute({input}, {output}, 0);
    brt_run_op(op.get(), d);
    output->copyToCPU(d.cmdpool);
    return brt_close_to_torch(output, torch_out, /*rtol=*/0.03, /*atol=*/0.03);
}

TEST(BufferRankTest, ReduceMeanRank4Keepdims) {
    EXPECT_TRUE(brt_reduce_case<float>({2, 3, 4, 5}, "mean", {1, 2}, 1, false));
    EXPECT_TRUE(brt_reduce_case<uint16_t>({2, 3, 4, 5}, "mean", {1, 2}, 1, true));
}

TEST(BufferRankTest, ReduceSumRank3NoKeepdims) {
    EXPECT_TRUE(brt_reduce_case<float>({3, 4, 5}, "sum", {0, 2}, 0, false));
}

TEST(BufferRankTest, ReduceMaxRank1) {
    EXPECT_TRUE(brt_reduce_case<float>({17}, "max", {0}, 1, false));
    EXPECT_TRUE(brt_reduce_case<float>({17}, "min", {0}, 0, false));
}

// =========================================================================
// Cast (fp32 <-> fp16; fp32-only shader, mode from dtype)
// =========================================================================

TEST(BufferRankTest, CastFp32ToFp16Rank4) {
    Dev d;
    auto torch_in = torch::randn({2, 3, 4, 5}, brt_opt(false));
    auto torch_out = torch_in.to(torch::kFloat16);

    auto input = std::make_shared<Tensor<float>>(std::vector<int>{2, 3, 4, 5});
    brt_fill(input, torch_in);
    brt_upload(input, d);

    auto output = brt_make_out<uint16_t>({2, 3, 4, 5}, d);
    auto op = brt_make_op(vkop::ops::OpType::CAST, /*fp16=*/false,
                      {{"to", "10"}}, d); // 10 = FLOAT16
    ASSERT_TRUE(op);
    op->onExecute({input}, {output}, 0);
    brt_run_op(op.get(), d);
    output->copyToCPU(d.cmdpool);
    EXPECT_TRUE(brt_close_to_torch(output, torch_out));
}

TEST(BufferRankTest, CastFp16ToFp32Rank2) {
    Dev d;
    auto torch_in = torch::randn({4, 7}, brt_opt(true));
    auto torch_out = torch_in.to(torch::kFloat32);

    auto input = std::make_shared<Tensor<uint16_t>>(std::vector<int>{4, 7});
    brt_fill(input, torch_in);
    brt_upload(input, d);

    auto output = brt_make_out<float>({4, 7}, d);
    auto op = brt_make_op(vkop::ops::OpType::CAST, /*fp16=*/false,
                      {{"to", "1"}}, d); // 1 = FLOAT
    ASSERT_TRUE(op);
    op->onExecute({input}, {output}, 0);
    brt_run_op(op.get(), d);
    output->copyToCPU(d.cmdpool);
    EXPECT_TRUE(brt_close_to_torch(output, torch_out));
}

// =========================================================================
// Range (scalar int64 start/limit/delta → CPU path)
// =========================================================================

TEST(BufferRankTest, RangeScalarStartLimitDelta) {
    Dev d;
    // start=0, limit=5, delta=1 → [0,1,2,3,4]
    auto start = brt_make_i64({0}, {});
    auto limit = brt_make_i64({5}, {});
    auto delta = brt_make_i64({1}, {});
    brt_upload_i64(start, d);
    brt_upload_i64(limit, d);
    brt_upload_i64(delta, d);

    auto output = brt_make_out<int64_t>({5}, d);
    auto op = brt_make_op(vkop::ops::OpType::RANGE, /*fp16=*/false, {}, d);
    ASSERT_TRUE(op);
    op->onExecute({start, limit, delta}, {output}, 0);
    brt_run_op(op.get(), d);
    output->copyToCPU(d.cmdpool);

    auto ref = torch::arange(0, 5, 1,
                             torch::TensorOptions().dtype(torch::kInt64));
    EXPECT_TRUE(brt_close_to_torch(output, ref, /*rtol=*/0.0, /*atol=*/0.0));
}

// =========================================================================
// Shape (CPU op; outputs int64 dims of the input)
// =========================================================================

TEST(BufferRankTest, ShapeRank4) {
    Dev d;
    auto torch_in = torch::randn({2, 3, 4, 5}, brt_opt(false));
    auto input = std::make_shared<Tensor<float>>(std::vector<int>{2, 3, 4, 5});
    brt_fill(input, torch_in);
    brt_upload(input, d);

    auto output = brt_make_out<int64_t>({4}, d);
    auto op = brt_make_op(vkop::ops::OpType::SHAPE, /*fp16=*/false, {}, d);
    ASSERT_TRUE(op);
    op->onExecute({input}, {output}, 0);
    brt_run_op(op.get(), d);
    output->copyToCPU(d.cmdpool);

    auto v = output->data();
    ASSERT_EQ(v.size(), 4u);
    EXPECT_EQ(v[0], 2);
    EXPECT_EQ(v[1], 3);
    EXPECT_EQ(v[2], 4);
    EXPECT_EQ(v[3], 5);
}

// =========================================================================
// RotaryEmbedding
// =========================================================================

// HF-style rotate_half reference (half-split, non-interleaved). cos/sin are
// [B, 1, S, head_dim] (half-repeated: cat(half, half)), broadcast over heads.
//   rotate_half(x) = concat(-x[half:], x[:half], dim=-1)
//   q_rot = x * cos + rotate_half(x) * sin
torch::Tensor brt_ref_rotary(const torch::Tensor &x, const torch::Tensor &cos,
                             const torch::Tensor &sin) {
    int d = (int)x.size(-1);
    int half = d / 2;
    auto x1 = x.index({torch::indexing::Slice(), torch::indexing::Slice(),
                       torch::indexing::Slice(),
                       torch::indexing::Slice(0, half)});
    auto x2 = x.index({torch::indexing::Slice(), torch::indexing::Slice(),
                       torch::indexing::Slice(),
                       torch::indexing::Slice(half, d)});
    auto rot = torch::cat({-x2, x1}, -1);
    return x * cos + rot * sin;
}

template <typename T>
bool brt_rotary_case(const std::vector<int> &xshape, bool fp16) {
    Dev d;
    int head_dim = xshape.back();
    int seq = xshape[xshape.size() - 2];
    int heads = xshape[xshape.size() - 3];
    int batch = xshape[0];
    // cos/sin: [B, 1, S, head_dim], half-repeated (cat(half, half)).
    int half = head_dim / 2;
    auto torch_x = torch::randn(
        std::vector<int64_t>(xshape.begin(), xshape.end()), brt_torch_opt<T>());
    auto cos_half = torch::randn({batch, 1, seq, half}, brt_torch_opt<T>());
    auto sin_half = torch::randn({batch, 1, seq, half}, brt_torch_opt<T>());
    auto torch_cos = torch::cat({cos_half, cos_half}, -1);
    auto torch_sin = torch::cat({sin_half, sin_half}, -1);
    auto torch_out = brt_ref_rotary(torch_x, torch_cos, torch_sin);

    auto input = std::make_shared<Tensor<T>>(xshape);
    brt_fill(input, torch_x);
    brt_upload(input, d);
    auto cos_t = std::make_shared<Tensor<T>>(brt_to_int_shape(torch_cos));
    brt_fill(cos_t, torch_cos);
    brt_upload(cos_t, d);
    auto sin_t = std::make_shared<Tensor<T>>(brt_to_int_shape(torch_sin));
    brt_fill(sin_t, torch_sin);
    brt_upload(sin_t, d);

    auto output = brt_make_out<T>(brt_to_int_shape(torch_out), d);
    auto op = brt_make_op(vkop::ops::OpType::ROTARY_EMBEDDING, fp16, {}, d);
    if (!op)
        return false;
    op->onExecute({input, cos_t, sin_t}, {output}, 0);
    brt_run_op(op.get(), d);
    output->copyToCPU(d.cmdpool);

    return brt_close_to_torch(output, torch_out, 0.02f, 0.02f);
}

TEST(BufferRankTest, RotaryEmbeddingPrefillAndDecode) {
    // Prefill shape: B=1, heads=16, S=5, head_dim=128 (q-heads).
    EXPECT_TRUE(brt_rotary_case<float>({1, 16, 5, 128}, false));
    EXPECT_TRUE(brt_rotary_case<uint16_t>({1, 16, 5, 128}, true));
    // Decode shape: B=1, heads=16, S=1, head_dim=128.
    EXPECT_TRUE(brt_rotary_case<float>({1, 16, 1, 128}, false));
    EXPECT_TRUE(brt_rotary_case<uint16_t>({1, 16, 1, 128}, true));
    // K-heads (8) — cos/sin still broadcast over heads.
    EXPECT_TRUE(brt_rotary_case<float>({1, 8, 3, 128}, false));
    EXPECT_TRUE(brt_rotary_case<uint16_t>({1, 8, 3, 128}, true));
}

// Sanity: cos=1, sin=0 -> q_rot = x*1 + rotate_half(x)*0 = x. Isolates
// indexing/binding from the rotate-half arithmetic.
template <typename T>
bool brt_rotary_identity_case(const std::vector<int> &xshape, bool fp16) {
    Dev d;
    auto torch_x = torch::randn(
        std::vector<int64_t>(xshape.begin(), xshape.end()), brt_torch_opt<T>());
    // Build cos/sin as [B,1,S,head_dim] of 1.0 / 0.0.
    int B = xshape[0], S = xshape[2], hd = xshape[3];
    auto cos_t = torch::ones({B, 1, S, hd}, brt_torch_opt<T>());
    auto sin_t = torch::zeros({B, 1, S, hd}, brt_torch_opt<T>());
    auto torch_out = torch_x; // cos=1, sin=0 => identity

    auto input = std::make_shared<Tensor<T>>(xshape);
    brt_fill(input, torch_x);
    brt_upload(input, d);
    auto cos_tt = std::make_shared<Tensor<T>>(brt_to_int_shape(cos_t));
    brt_fill(cos_tt, cos_t);
    brt_upload(cos_tt, d);
    auto sin_tt = std::make_shared<Tensor<T>>(brt_to_int_shape(sin_t));
    brt_fill(sin_tt, sin_t);
    brt_upload(sin_tt, d);

    auto output = brt_make_out<T>(brt_to_int_shape(torch_out), d);
    auto op = brt_make_op(vkop::ops::OpType::ROTARY_EMBEDDING, fp16, {}, d);
    if (!op)
        return false;
    op->onExecute({input, cos_tt, sin_tt}, {output}, 0);
    brt_run_op(op.get(), d);
    output->copyToCPU(d.cmdpool);
    return brt_close_to_torch(output, torch_out, 0.001f, 0.001f);
}

TEST(BufferRankTest, RotaryEmbeddingIdentity) {
    EXPECT_TRUE(brt_rotary_identity_case<float>({1, 4, 2, 8}, false));
}

} // namespace
