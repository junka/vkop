// Copyright 2025 @junka
#ifndef OPS_BUFFER_BASE_HPP_
#define OPS_BUFFER_BASE_HPP_

#include "core/Tensor.hpp"
#include "ops/Operator.hpp"

#include <algorithm>
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

struct alignas(16) LayerNormPC {
    float eps;
    int inner_size;
    int outer_size;
};

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

inline int total_elems(const std::vector<int> &shape) {
    return static_cast<int>(
        std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<>()));
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
