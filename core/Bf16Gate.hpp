// Copyright 2025 @junka
#ifndef CORE_BF16_GATE_HPP_
#define CORE_BF16_GATE_HPP_

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "core/Tensor.hpp"
#include "ops/Ops.hpp"

namespace vkop {
namespace core {

// bfloat16 reaches a kernel in two roles, and they are gated separately.
//
//  * WEIGHT: a constant bf16 tensor in a slot whose kernel decodes it itself
//    -- MatMul/Gemm B (para_.bf16 bit 0, gemm.comp fp16b == 2) and Gather's
//    table (gather.comp -DBF16_TABLE). The activations around it stay fp16 or
//    fp32, and the op picks its fp16/fp32 build as usual.
//  * ACTIVATION (--act bf16): the node runs its `_bf16` module,
//    which is the fp16 source compiled with -DFP16 -DBF16. That build is only
//    right where the shader actually routes every read and write through the
//    bf16 decode/encode, and most of them do not (a direct unpackHalf2x16, an
//    fp16-only secondary module, a host route hard-wired to an `_fp16` key).
//    So the activation role is an explicit allow-list of what was verified,
//    not "the archive has a _bf16 key".
//
// Everything else fails closed with an error naming the op and the node.

// (op, input slot) pairs that decode a bf16 WEIGHT.
bool bf16_weight_slot(ops::OpType type, size_t input_index);

// Ops whose `_bf16` build was verified to decode and encode bf16 end to end
// (binary elementwise body, Split, LayerNorm, RMSNorm, Softmax, MatMul on the
// routes core::select_matmul keeps for MatmulFeatures::act_bf16, Attention on
// its row kernel), plus the host-only byte movers (Shape, Squeeze, Unsqueeze),
// which never decode an element.
bool bf16_activation_op(ops::OpType type);

// True when the node carries a bf16 ACTIVATION: a bf16 input outside the
// weight slots above, or a bf16 output.
bool has_bf16_activation(
    ops::OpType type, const std::vector<std::shared_ptr<ITensor>> &inputs,
    const std::vector<std::shared_ptr<ITensor>> &outputs);

// The load/execute-time gate. Returns has_bf16_activation(); throws
//   "vkop: no bf16 activation kernel for <op> '<node>' ..."
// when a bf16 activation reaches an op outside the allow-list (or the image
// backend), and
//   "vkop: <op> '<node>' mixes bfloat16 activations with ..."
// when such a node also binds an fp16/fp32 operand or a quantized payload: the
// _bf16 module decodes EVERY 16-bit operand as bf16 and reads no other float
// width, so an fp16 tensor beside a bf16 one is a silent wrong answer.
// Integer operands (int32/int64 indices, shapes, split sizes) are fine.
bool check_bf16_activation(
    ops::OpType type, const std::string &op_name, const std::string &node_name,
    const std::vector<std::shared_ptr<ITensor>> &inputs,
    const std::vector<std::shared_ptr<ITensor>> &outputs, bool buffer_backend);

} // namespace core
} // namespace vkop

#endif // CORE_BF16_GATE_HPP_
