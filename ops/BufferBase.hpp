// Copyright 2025 @junka
#ifndef OPS_BUFFER_BASE_HPP_
#define OPS_BUFFER_BASE_HPP_

#include "core/Tensor.hpp"
#include "ops/Operator.hpp"

#include <algorithm>
#include <cstddef>
#include <numeric>

// Shared base + helpers + push-constant structs for the buffer (SSBO)
// backend ops. Each buffer op treats tensors as compact row-major
// contiguous data of arbitrary rank (<=8-D via the dims[8] push-constant
// convention), instead of the image path's NCHW->RGBA packing. Shapes flow
// through push constants (see shaders/common/buffer_common.comp for the
// matching GLSL convention); data flows through SSBOs.
//
// The elementwise (unary/binary) PC structs live here too so both
// BufferUnaryFactory and BufferBinaryFactory (and the per-op shaders) share
// one layout definition. See ops/BufferUnaryFactory.hpp and
// ops/BufferBinaryFactory.hpp for the factory bases that use them.

namespace vkop {
namespace ops {

// Elementwise unary push constant (mirrors shaders/buffer/<op>.comp). The
// per-op shader inlines its own math (no op_code switch), so the PC carries
// no op code — just the shape + element count.
struct alignas(16) UnaryElemPC {
    int rank;       // tensor rank
    int outDims[8]; // output shape (== input shape for unary)
    int in0Dims[8]; // input shape
    int total;      // output element count
    int _pad;
};
static_assert(sizeof(UnaryElemPC) <= 128, "UnaryElemPC PC overflow");

// Device-address variant of UnaryElemPC. The shader is buffer/<op>.comp built
// with -DBDA (shared body: shaders/common/unary_elementwise_body.comp); the
// same source carries the descriptor path too.
struct alignas(16) UnaryBdaPC {
    // Explicit fields, not `UnaryElemPC base`: its alignas(16) sizeof is 80
    // while the last scalar field ends at 76, which happens to match the
    // next 8-aligned offset today. Spelling the fields out keeps the layout
    // independent of that coincidence.
    int rank;
    int outDims[8];
    int in0Dims[8];
    int total;
    int _pad;
    VkDeviceAddress in_addr;
    VkDeviceAddress out_addr;
};
static_assert(offsetof(UnaryBdaPC, in_addr) == 80, "UnaryBdaPC layout drift");
static_assert(offsetof(UnaryBdaPC, out_addr) == 88, "UnaryBdaPC layout drift");
static_assert(sizeof(UnaryBdaPC) == 96, "UnaryBdaPC layout drift");
static_assert(sizeof(UnaryBdaPC) <= 128, "UnaryBdaPC PC overflow");

// Elementwise binary push constant (mirrors shaders/buffer/<op>.comp) with
// ONNX right-aligned broadcasting + optional post-fuse activation. No op
// code (per-op shader inlines the math).
struct alignas(16) BinaryElemPC {
    int rank;       // common rank for nd<->linear (== out rank)
    int outDims[8]; // left-aligned output shape
    int in0Dims[8]; // input A, right-aligned to `rank`
    int in1Dims[8]; // input B, right-aligned to `rank`
    int activation; // ACTIVATION_* from shaders/common/activation.comp (0=none)
    int broadcast;  // 1 if either input needs broadcasting
    int total;      // output element count
    int _pad;
};
static_assert(sizeof(BinaryElemPC) <= 128, "BinaryElemPC PC overflow");

// Device-address args for binary ops (buffer/<op>.comp built with -DBDA).
// BinaryElemPC is already 128 B, so the BDA push constant instead carries only
// the scalar fields + the address of this args buffer. Must match
// shaders/common/binary_bda_body.comp's ArgsBuf.
struct alignas(8) BinaryBdaArgs {
    int outDims[8];
    int in0Dims[8];
    int in1Dims[8];
    VkDeviceAddress out_addr;
    VkDeviceAddress in0_addr;
    VkDeviceAddress in1_addr;
};
static_assert(offsetof(BinaryBdaArgs, in0Dims) == 32, "BinaryBdaArgs drift");
static_assert(offsetof(BinaryBdaArgs, in1Dims) == 64, "BinaryBdaArgs drift");
static_assert(offsetof(BinaryBdaArgs, out_addr) == 96, "BinaryBdaArgs drift");
static_assert(offsetof(BinaryBdaArgs, in0_addr) == 104, "BinaryBdaArgs drift");
static_assert(offsetof(BinaryBdaArgs, in1_addr) == 112, "BinaryBdaArgs drift");
static_assert(sizeof(BinaryBdaArgs) == 120, "BinaryBdaArgs drift");

struct alignas(8) BinaryBdaPC {
    int rank;
    int activation;
    int broadcast;
    int total;
    VkDeviceAddress args_addr;
};
static_assert(offsetof(BinaryBdaPC, args_addr) == 16, "BinaryBdaPC drift");
static_assert(sizeof(BinaryBdaPC) == 24, "BinaryBdaPC drift");
static_assert(sizeof(BinaryBdaPC) <= 128, "BinaryBdaPC PC overflow");

struct alignas(16) ReshapePC {
    int rank_in;
    int inDims[8];
    int rank_out;
    int outDims[8];
    int total;
};

struct alignas(16) SlicePC {
    int rank;
    int inDims[6];
    int outDims[6];
    int starts[6];
    int ends[6];
    int steps[6];
};
static_assert(sizeof(SlicePC) <= 128, "SlicePC PC overflow");

struct alignas(16) ConcatPC {
    int axis;
    int rank;
    int inDims[8];
    int outDims[8];
    int offset;
    int _pad0;
    int _pad1;
    int _pad2;
};

struct alignas(16) SplitPC {
    int axis;
    int rank;
    int inDims[8];
    int outDims[8];
    int split;
    int _pad0;
    int _pad1;
    int _pad2;
};

struct alignas(16) TransposePC {
    int rank;
    int inDims[8];
    int outDims[8];
    int perm[8];
};

struct alignas(16) SoftmaxPC {
    int axis;
    int axis_size;
    int outer_size;
    int inner_size;
};

// Device-address variant (buffer/softmax.comp built with -DBDA). out/in/scratch
// addresses; the fp16 two-pass path reuses the same pipeline with a different
// PC (reduce writes scratch, pack reads scratch). The GLSL block is 40 bytes;
// the struct is 48 because alignas(16) pads the tail.
struct alignas(16) SoftmaxBdaPC {
    SoftmaxPC base;
    VkDeviceAddress out_addr;
    VkDeviceAddress in_addr;
    VkDeviceAddress scratch_addr;
};
static_assert(offsetof(SoftmaxBdaPC, out_addr) == 16, "SoftmaxBdaPC drift");
static_assert(offsetof(SoftmaxBdaPC, in_addr) == 24, "SoftmaxBdaPC drift");
static_assert(offsetof(SoftmaxBdaPC, scratch_addr) == 32, "SoftmaxBdaPC drift");
static_assert(sizeof(SoftmaxBdaPC) <= 128, "SoftmaxBdaPC PC overflow");

struct alignas(16) LayerNormPC {
    float eps;
    int inner_size;
    int outer_size;
    // 1 = a 4th input (binding 4) is bound and added to x before the
    // mean/variance reduce: LayerNorm(x + skip). The converter's
    // fuse_skip_layernorm folds the Add of a post-norm residual block into
    // this op, so no separate Add dispatch and no round-trip of x+skip.
    int has_skip;
};

// Device-address variant (buffer/layernorm.comp built with -DBDA). out/x/w/b
// addresses, then the skip slot (aliases x when has_skip == 0). The GLSL block
// is 56 bytes; the struct is 64 because alignas(16) pads the tail.
struct alignas(16) LayerNormBdaPC {
    LayerNormPC base;
    VkDeviceAddress out_addr;
    VkDeviceAddress in_addr;
    VkDeviceAddress weight_addr;
    VkDeviceAddress bias_addr;
    VkDeviceAddress skip_addr;
};
static_assert(offsetof(LayerNormBdaPC, out_addr) == 16, "LayerNormBdaPC drift");
static_assert(offsetof(LayerNormBdaPC, in_addr) == 24, "LayerNormBdaPC drift");
static_assert(offsetof(LayerNormBdaPC, weight_addr) == 32,
              "LayerNormBdaPC drift");
static_assert(offsetof(LayerNormBdaPC, bias_addr) == 40,
              "LayerNormBdaPC drift");
static_assert(offsetof(LayerNormBdaPC, skip_addr) == 48,
              "LayerNormBdaPC drift");
static_assert(sizeof(LayerNormBdaPC) <= 128, "LayerNormBdaPC PC overflow");

struct alignas(16) ReducePC {
    int rank;
    int inDims[8];
    int outDims[8];
    int axes_mask;
    int reduce_op;
    int keepdims;
    int _pad0;
    int _pad1;
};

// Device-address variants: two direct pointers fit in the remaining push
// constant space. The base struct's alignas(16) tail padding must NOT be
// copied here: std430 places the 8-byte pointers immediately after the last
// field (offset 88/104), not after the padded sizeof (96/112).
struct alignas(8) ReduceBdaPC {
    int rank;
    int inDims[8];
    int outDims[8];
    int axes_mask;
    int reduce_op;
    int keepdims;
    int _pad0;
    int _pad1;
    VkDeviceAddress out_addr;
    VkDeviceAddress in_addr;
};
static_assert(offsetof(ReduceBdaPC, out_addr) == 88, "ReduceBdaPC drift");
static_assert(sizeof(ReduceBdaPC) == 104, "ReduceBdaPC drift");

// Cumulative sum along one axis (shaders/buffer/cumsum.comp). One workgroup
// per "line" (every lane group fixed by the non-axis dims); within a line the
// elements sit `axis_stride` apart in linear index space and are walked in
// order by that workgroup, carrying a running total. The shape is not in the
// PC: the resolved (axis_len, axis_stride, outer) triple is everything the
// shader needs, and dropping the dims array keeps the BDA pointer offsets
// simple.
struct alignas(16) CumsumPC {
    int axis;        // resolved, >= 0 (diagnostic only)
    int axis_len;    // dims[axis]
    int axis_stride; // product of dims after the axis
    int outer;       // product of dims before the axis
    int exclusive;   // 1 = exclusive prefix sum
    int reverse;     // 1 = walk the line backwards
    int _pad0;
    int _pad1;
};
static_assert(sizeof(CumsumPC) == 32, "CumsumPC layout drift");

// Device-address variant: the two pointers follow the 32-byte scalar prefix.
struct alignas(8) CumsumBdaPC {
    int axis;
    int axis_len;
    int axis_stride;
    int outer;
    int exclusive;
    int reverse;
    int _pad0;
    int _pad1;
    VkDeviceAddress out_addr;
    VkDeviceAddress in_addr;
};
static_assert(offsetof(CumsumBdaPC, out_addr) == 32, "CumsumBdaPC drift");
static_assert(offsetof(CumsumBdaPC, in_addr) == 40, "CumsumBdaPC drift");
static_assert(sizeof(CumsumBdaPC) == 48, "CumsumBdaPC drift");

// Causal depthwise Conv1d (shaders/buffer/conv1d.comp). One thread per output
// element of a [C, S] channel-major tensor; each lane runs its channel's
// length-K filter over a causal window, taking the samples before t=0 from
// the conv state (a [C, K-1] graph I/O tensor) instead of zero padding. The
// same pipeline runs a second time with mode=1 to write the new state tail,
// which is what makes present-aliases-past race-free.
struct alignas(16) Conv1dPC {
    int channels;  // C
    int seq;       // S, new tokens this round
    int kernel;    // K
    int state_len; // K-1
    int total;     // C*S for mode 0, C for mode 1
    int mode;      // 0 = convolution, 1 = state update
    int _pad0;
    int _pad1;
};
static_assert(sizeof(Conv1dPC) == 32, "Conv1dPC layout drift");

struct alignas(8) Conv1dBdaPC {
    int channels;
    int seq;
    int kernel;
    int state_len;
    int total;
    int mode;
    int _pad0;
    int _pad1;
    VkDeviceAddress out_addr;
    VkDeviceAddress x_addr;
    VkDeviceAddress w_addr;
    VkDeviceAddress state_in_addr;
    VkDeviceAddress state_out_addr;
};
static_assert(offsetof(Conv1dBdaPC, out_addr) == 32, "Conv1dBdaPC drift");
static_assert(offsetof(Conv1dBdaPC, state_out_addr) == 64,
              "Conv1dBdaPC drift");
static_assert(sizeof(Conv1dBdaPC) == 72, "Conv1dBdaPC drift");

// Gated delta rule (shaders/buffer/linear_attention.comp). One workgroup per
// value head; lane t owns value-column t of that head's [Dk, Dv] state matrix,
// so the recurrence needs no cross-lane reduction and the state store rewrites
// only what the lane read.
struct alignas(16) LinearAttentionPC {
    int heads;    // Hv, value heads == workgroup count
    int kv_heads; // Hk; value head h reads q/k head (h / (Hv/Hk))
    int seq;      // tokens this round
    int dk;       // key head dim
    int dv;       // value head dim
    int eps_bits;  // l2norm epsilon as float bits
    int seq_major; // 1 = data is [S,H,D] in memory (exporter's [1,S,H,D])
    int _pad1;
};
static_assert(sizeof(LinearAttentionPC) == 32, "LinearAttentionPC drift");

struct alignas(8) LinearAttentionBdaPC {
    int heads;
    int kv_heads;
    int seq;
    int dk;
    int dv;
    int eps_bits;
    int seq_major;
    int _pad1;
    VkDeviceAddress out_addr;
    VkDeviceAddress q_addr;
    VkDeviceAddress k_addr;
    VkDeviceAddress v_addr;
    VkDeviceAddress g_addr;
    VkDeviceAddress beta_addr;
    VkDeviceAddress state_in_addr;
    VkDeviceAddress state_out_addr;
};
static_assert(offsetof(LinearAttentionBdaPC, out_addr) == 32,
              "LinearAttentionBdaPC drift");
static_assert(offsetof(LinearAttentionBdaPC, state_out_addr) == 88,
              "LinearAttentionBdaPC drift");
static_assert(sizeof(LinearAttentionBdaPC) == 96,
              "LinearAttentionBdaPC drift");

struct alignas(8) TransposeBdaPC {
    int rank;
    int inDims[8];
    int outDims[8];
    int perm[8];
    VkDeviceAddress out_addr;
    VkDeviceAddress in_addr;
};
static_assert(offsetof(TransposeBdaPC, out_addr) == 104,
              "TransposeBdaPC drift");
static_assert(sizeof(TransposeBdaPC) == 120, "TransposeBdaPC drift");

// Activation codes shared with shaders/common/activation.comp
enum class BufferActivation {
    NONE = 0,
    RELU = 1,
    SIGMOID = 2,
    TANH = 3,
    HSWISH = 4,
    MISH = 5,
    RELU6 = 6,
    SWISH = 7,
};

// Left-align `shape` into dims[8]: dims[0..rank-1] = shape, the rest = 1.
// The shader nd<->linear helpers iterate i in [0..rank-1], so the shape
// must live at the front of the array. For broadcast inputs of lower rank,
// the host right-aligns conceptually and pads leading dims with 1 — see
// fill_dims_broadcast.
inline void fill_dims(int (&dims)[8], const std::vector<int> &shape) {
    for (int i = 0; i < 8; ++i)
        dims[i] = 1;
    int r = static_cast<int>(std::min<size_t>(shape.size(), 8));
    for (int i = 0; i < r; ++i)
        dims[i] = shape[i];
}

// Right-align `shape` (rank < out_rank) into dims[8] against `out_rank`:
// the shape occupies the trailing slots, leading slots are 1. This is the
// ONNX broadcast convention for a lower-rank input.
inline void fill_dims_broadcast(int (&dims)[8], const std::vector<int> &shape,
                                int out_rank) {
    for (int i = 0; i < 8; ++i)
        dims[i] = 1;
    int r = static_cast<int>(std::min<size_t>(shape.size(), 8));
    for (int i = 0; i < r; ++i)
        dims[out_rank - r + i] = shape[i];
}

// 4-int variants for ops whose PC uses ivec4 shapes (rank ≤ 4 — e.g. the
// int64 Where GPU path, where 4× IArr8 would overflow the 128B PC limit).
inline void fill_dims4(int (&dims)[4], const std::vector<int> &shape) {
    for (int i = 0; i < 4; ++i)
        dims[i] = 1;
    int r = static_cast<int>(std::min<size_t>(shape.size(), 4));
    for (int i = 0; i < r; ++i)
        dims[i] = shape[i];
}

inline void fill_dims4_broadcast(int (&dims)[4], const std::vector<int> &shape,
                                 int out_rank) {
    for (int i = 0; i < 4; ++i)
        dims[i] = 1;
    int r = static_cast<int>(std::min<size_t>(shape.size(), 4));
    for (int i = 0; i < r; ++i)
        dims[out_rank - r + i] = shape[i];
}

inline int total_elems(const std::vector<int> &shape) {
    return static_cast<int>(
        std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<>()));
}

// Contiguous sub-range test for the zero-copy view fast path. Given the input
// shape, the Slice-selected output shape, and the full-rank per-axis
// starts/steps, decide whether the selected block is a CONTIGUOUS run of the
// input's row-major elements. On success, set `byte_offset` (byte position of
// the block's first element) and `byte_len`, and return true.
//
// With every step == 1 the output->input index map is affine and monotone, so
// the block is contiguous iff its element count equals the linear span between
// its first and last element (count == last - first + 1). A partial innermost-
// axis slice (rows with a gap) or slicing two axes at once fails this and falls
// back to the gather shader.
inline bool contiguous_slice_range(const std::vector<int> &in_shape,
                                   const std::vector<int> &out_shape,
                                   const std::vector<int> &starts,
                                   const std::vector<int> &steps,
                                   size_t elem_size, VkDeviceSize &byte_offset,
                                   VkDeviceSize &byte_len) {
    int rank = static_cast<int>(in_shape.size());
    if (static_cast<int>(out_shape.size()) != rank ||
        static_cast<int>(starts.size()) < rank ||
        static_cast<int>(steps.size()) < rank) {
        return false;
    }
    std::vector<int64_t> stride(rank, 1);
    for (int d = rank - 2; d >= 0; --d) {
        stride[d] = stride[d + 1] * in_shape[d + 1];
    }
    int64_t first = 0, last = 0, n = 1;
    for (int d = 0; d < rank; ++d) {
        if (steps[d] != 1) {
            return false; // a strided slice is never a contiguous block
        }
        int64_t s = starts[d];
        int64_t o = out_shape[d];
        if (o <= 0 || s < 0 || s + o > in_shape[d]) {
            return false;
        }
        n *= o;
        first += s * stride[d];
        last += (s + o - 1) * stride[d];
    }
    if (n != last - first + 1) {
        return false; // gaps between selected elements -> not contiguous
    }
    byte_offset = static_cast<VkDeviceSize>(first) * elem_size;
    byte_len = static_cast<VkDeviceSize>(n) * elem_size;
    return true;
}

// Contiguous sub-range test for the Gather zero-copy view fast path. A Gather
// along `axis` produces the contiguous run of the data's row-major elements
// [s, s+1, ..., s+k-1] (k = indices count) lifted by the inner block size iff
// (a) every dimension before `axis` is 1 — otherwise the outer dims stride
// between the gathered blocks and the output is not one run — and (b) the
// flattened indices are consecutive ascending. Then the output is exactly data
// elements [s*inner, (s+k)*inner), inner = prod(in_shape[axis+1..]). `idx` must
// already be host-resolved and normalized to >= 0 (see Gather::execute).
// Mirrors contiguous_slice_range: same "count == span" idea, expressed on the
// gather's index list instead of per-axis starts/steps.
inline bool contiguous_gather_range(const std::vector<int> &in_shape,
                                    const std::vector<int64_t> &idx, int axis,
                                    size_t elem_size, VkDeviceSize &byte_offset,
                                    VkDeviceSize &byte_len) {
    int rank = static_cast<int>(in_shape.size());
    if (axis < 0 || axis >= rank || idx.empty()) {
        return false;
    }
    for (int d = 0; d < axis; ++d) {
        if (in_shape[d] != 1) {
            return false; // an outer dim > 1 strides between blocks
        }
    }
    int64_t k = static_cast<int64_t>(idx.size());
    int64_t s = idx[0];
    if (s < 0 || s + k > in_shape[axis]) {
        return false;
    }
    for (int64_t i = 1; i < k; ++i) {
        if (idx[i] != s + i) {
            return false; // not consecutive ascending -> a real gather
        }
    }
    int64_t inner = 1;
    for (int d = axis + 1; d < rank; ++d) {
        inner *= in_shape[d];
    }
    byte_offset = static_cast<VkDeviceSize>(s * inner) * elem_size;
    byte_len = static_cast<VkDeviceSize>(k * inner) * elem_size;
    return true;
}

// True when a binding at `byte_offset` is legal for an SSBO AND preserves the
// fp16 word packing. Vulkan requires the offset to be a multiple of the
// device's minStorageBufferOffsetAlignment; the fp16 backed layout packs two
// halves per uint, so the offset must also be 4-byte aligned (an even element
// index) or a consumer's word read would straddle two halves of different
// words. Both fall out of requiring max(align, 4).
inline bool offset_view_aligned(VkDeviceSize byte_offset, VkDeviceSize align) {
    VkDeviceSize a = align > 4 ? align : 4;
    return (byte_offset % a) == 0;
}

// ONNX right-aligned broadcast: given the (lower-or-equal-rank) input shape
// `in_shape` and the broadcast `out_shape`, map a linear output index `lin`
// to the linear input index (broadcast dims / leading padding dims map to
// element 0). Shared by the CPU int64 compute branches (binary ops, Equal,
// Where, Expand). The backward pass computes both the coordinate
// decomposition and the input's row-major stride in one sweep.
inline int broadcast_index(const std::vector<int> &in_shape,
                           const std::vector<int> &out_shape, int lin) {
    int rank = static_cast<int>(out_shape.size());
    int offset = rank - static_cast<int>(in_shape.size());
    int s = 1;
    int idx = 0;
    for (int d = rank - 1; d >= 0; --d) {
        int in_dim = (d >= offset) ? in_shape[d - offset] : 1;
        int coord = lin % out_shape[d];
        lin /= out_shape[d];
        if (in_dim != 1) {
            idx += coord * s;
            s *= in_dim;
        }
    }
    return idx;
}

// Buffer op base: holds the SSBO bind helper; the Operator base does the
// descriptor/pipeline heavy lifting.
class BufferFactory : public Operator {
  protected:
    using Operator::Operator;

    bool uses_buffer_backend() const override { return true; }

    // Bind one tensor as an SSBO (read or write) and append to objs_.
    template <typename T>
    std::shared_ptr<VulkanBuffer>
    bind_ssbo(const std::shared_ptr<core::ITensor> &t, bool is_output) {
        auto tensor = core::as_tensor<T>(t);
        auto buf =
            tensor->as_storage_buffer(m_dev_, is_output ? m_cmd_ : nullptr);
        objs_.emplace_back(buf);
        return buf;
    }
};

} // namespace ops
} // namespace vkop

#endif // OPS_BUFFER_BASE_HPP_
