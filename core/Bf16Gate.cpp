// Copyright 2025 @junka
#include "core/Bf16Gate.hpp"

#include <stdexcept>

#include "core/DType.hpp"

namespace vkop {
namespace core {

bool bf16_weight_slot(ops::OpType type, size_t input_index) {
    switch (type) {
    case ops::OpType::MATMUL:
    case ops::OpType::GEMM:
        return input_index == 1;
    case ops::OpType::GATHER:
        return input_index == 0;
    default:
        return false;
    }
}

bool bf16_activation_op(ops::OpType type) {
    switch (type) {
    // shaders/common/binary_elementwise_body.comp: both operands and the
    // result go through bdec2/benc2, which pick the bf16 form under -DBF16.
    case ops::OpType::ADD:
    case ops::OpType::SUB:
    case ops::OpType::MUL:
    case ops::OpType::DIV:
    case ops::OpType::POW:
    case ops::OpType::MIN:
    case ops::OpType::MAX:
    case ops::OpType::MOD:
    // BUF_LOAD / BUF_PACK2 under -DBF16.
    case ops::OpType::SPLIT:
    case ops::OpType::LAYERNORM:
    case ops::OpType::RMSNORM:
    case ops::OpType::SOFTMAX:
    // Route-gated inside the op: MatMul takes every route but the KHR
    // cooperative-matrix GEMMs and the down_act fold (KernelSelect's
    // matmul_route_takes_bf16_act); Attention takes its row, split-KV and
    // SIMT flash kernels (_bf16 builds; not the coop8 flash build).
    case ops::OpType::MATMUL:
    case ops::OpType::ATTENTION:
    // shaders/common/unary_elementwise_body.comp: UDEC2/UENC2 (both sides).
    case ops::OpType::SIN:
    case ops::OpType::COS:
    case ops::OpType::SIGMOID:
    case ops::OpType::TANH:
    case ops::OpType::ERF:
    case ops::OpType::SQRT:
    case ops::OpType::RELU:
    case ops::OpType::NEG:
    case ops::OpType::FLOOR:
    case ops::OpType::ATAN:
    case ops::OpType::SOFTPLUS:
    // buffer/clip.comp: BUF_UNPACK2 / BUF_PACK2; bounds read on the host by
    // their own kind (bf16_other_float_ok).
    case ops::OpType::CLIP:
    // buffer/expand.comp: BUF_LOAD / BUF_PACK2 (or a whole-word copy).
    case ops::OpType::EXPAND:
    // buffer/concat.comp: a 16-bit mover (whole words or atomic halves).
    case ops::OpType::CONCAT:
    // Per-operand formats (bf16_other_float_ok): FusedElemwise decodes each
    // input by its own dtype code and packs the output by its build; Cast's
    // modes 3-6 and Gather's -DBF16_OUT modules convert between the formats.
    case ops::OpType::FUSED_ELEMWISE:
    case ops::OpType::CAST:
    case ops::OpType::GATHER:
    // Raw 16-bit word movers: Transpose moves each element's bits
    // (BUF_BITS16), Pad copies words and writes a host-encoded bf16 constant.
    case ops::OpType::TRANSPOSE:
    case ops::OpType::PAD:
    // Decode through BUF_LOAD / BUF_UNPACK2 and encode through BUF_PACK2,
    // which are bf16 under -DBF16 (shaders/common/buffer_common.comp).
    case ops::OpType::ROTARY_EMBEDDING:
    case ops::OpType::RESIZE:
    case ops::OpType::MAXPOOL2D:
    case ops::OpType::GLOBALAVERAGEPOOL:
    case ops::OpType::REDUCE:
    case ops::OpType::REDUCEMEAN:
    case ops::OpType::INSTANCENORM:
    case ops::OpType::PRELU:
    // Own codec under -DBF16: cumsum.comp binds uint16_t storage and goes
    // through bf16_to_float / float_to_bf16; scatter_elements.comp's word CAS
    // re-encodes only its own half with packBF16x2.
    case ops::OpType::CUMSUM:
    case ops::OpType::SCATTER_ELEMENTS:
    // Byte movers: alias or copy 16-bit words, never decode an element.
    // Shape/Squeeze/Unsqueeze are host aliases; Reshape aliases the input's
    // VkBuffer (ops/Reshape.hpp, dispatch_by_dtype over uint16_t, CPU
    // fallback copies uint16_t); Slice binds a sub-range or runs slice.comp,
    // whose -DFP16 body (and so the _bf16 build) moves raw 16-bit halves
    // (in_half) without unpacking them.
    case ops::OpType::SHAPE:
    case ops::OpType::SQUEEZE:
    case ops::OpType::UNSQUEEZE:
    case ops::OpType::RESHAPE:
    case ops::OpType::SLICE:
        return true;
    default:
        return false;
    }
}

bool bf16_other_float_ok(ops::OpType type, bool output, size_t index,
                         ElemKind kind) {
    const bool fp32 = kind == ElemKind::kFloat32;
    const bool fp16 = kind == ElemKind::kFloat16;
    if (!fp32 && !fp16) {
        return false;
    }
    switch (type) {
    // Converters: every operand is read or written at its own format.
    case ops::OpType::CAST:
    case ops::OpType::FUSED_ELEMWISE:
        return true;
    // The table: -DBF16_OUT reads a bf16, fp16 or fp32 table.
    case ops::OpType::GATHER:
        return !output && index == 0;
    // Host-read scalars (ClipBuffer::read_bound, Pad read_scalar_float),
    // decoded by their own kind; no kernel binds them.
    case ops::OpType::CLIP:
        return !output && (index == 1 || index == 2);
    case ops::OpType::PAD:
        return !output && index == 2;
    // cos/sin: an fp32 table is read at fp32 stride (cs_fp32); a 16-bit one
    // in X's own format, so fp16 beside bf16 X is not an operand set.
    case ops::OpType::ROTARY_EMBEDDING:
        return !output && fp32 && (index == 1 || index == 2);
    default:
        return false;
    }
}

namespace {
bool is_bf16(const std::shared_ptr<ITensor> &t) {
    return t != nullptr && t->elem_kind() == ElemKind::kBFloat16;
}

// What a bf16 node may bind beside its bf16 tensors: index/shape integers.
bool bf16_neighbour_ok(ElemKind kind) {
    return kind == ElemKind::kBFloat16 || kind == ElemKind::kInt32 ||
           kind == ElemKind::kInt64;
}

// An int8 or int4 weight-only MatMul, [A, W, scale_fp32]. matmul.comp's _bf16
// module decodes the byte B to fp32 itself (byte stage / GEMV decode, never
// through the bf16 unpack), applies the fp32 per-column scale in fp32, and
// writes the output at A's width; matmul_coop8.comp's _bf16 module stages the
// int8 codes as exact halves and applies the scale in its fp32 epilogue.
// Tests: Bf16ActivationTest.MatMulInt8WeightMatchesFp32.
bool bf16_int8_weight_slot(ops::OpType type,
                           const std::vector<std::shared_ptr<ITensor>> &inputs,
                           size_t index) {
    // int4 (signed nibbles, fp32 [K/group, N] scale table) decodes through
    // the same _bf16 module paths: Bf16ActivationTest.MatMulInt4WeightMatchesFp32.
    if (type != ops::OpType::MATMUL || inputs.size() != 3 || !inputs[1] ||
        (inputs[1]->elem_kind() != ElemKind::kInt8 &&
         inputs[1]->elem_kind() != ElemKind::kInt4)) {
        return false;
    }
    if (index == 1) {
        return true;
    }
    return index == 2 && inputs[2] &&
           inputs[2]->elem_kind() == ElemKind::kFloat32;
}
} // namespace

bool has_bf16_activation(
    ops::OpType type, const std::vector<std::shared_ptr<ITensor>> &inputs,
    const std::vector<std::shared_ptr<ITensor>> &outputs) {
    for (size_t i = 0; i < inputs.size(); ++i) {
        if (is_bf16(inputs[i]) && !bf16_weight_slot(type, i)) {
            return true;
        }
    }
    for (const auto &o : outputs) {
        if (is_bf16(o)) {
            return true;
        }
    }
    // MatMul's A decides its activation width even when B is the bf16 slot:
    // covered by the input loop (A is input 0, never a weight slot).
    return false;
}

bool bf16_build_matches(
    ops::OpType type, int fp16,
    const std::vector<std::shared_ptr<ITensor>> &outputs) {
    switch (type) {
    // No module (host-only) or one module for every format (Cast).
    case ops::OpType::SHAPE:
    case ops::OpType::SQUEEZE:
    case ops::OpType::UNSQUEEZE:
    case ops::OpType::CAST:
        return true;
    // The build is the OUTPUT's format; a bf16 input is decoded per operand.
    case ops::OpType::FUSED_ELEMWISE:
    case ops::OpType::GATHER: {
        bool out_bf16 = false;
        for (const auto &o : outputs) {
            out_bf16 = out_bf16 || is_bf16(o);
        }
        return out_bf16 ? fp16 == 2 : fp16 != 2;
    }
    default:
        return fp16 == 2;
    }
}

bool check_bf16_activation(
    ops::OpType type, const std::string &op_name, const std::string &node_name,
    const std::vector<std::shared_ptr<ITensor>> &inputs,
    const std::vector<std::shared_ptr<ITensor>> &outputs,
    bool buffer_backend) {
    if (!has_bf16_activation(type, inputs, outputs)) {
        return false;
    }
    // Name the first bf16 tensor, so the message says which edge carried it.
    std::string role;
    for (size_t i = 0; i < inputs.size() && role.empty(); ++i) {
        if (is_bf16(inputs[i]) && !bf16_weight_slot(type, i)) {
            role = "input " + std::to_string(i);
        }
    }
    for (size_t i = 0; i < outputs.size() && role.empty(); ++i) {
        if (is_bf16(outputs[i])) {
            role = "output " + std::to_string(i);
        }
    }
    const std::string where =
        op_name + " '" + (node_name.empty() ? op_name : node_name) + "'";
    if (!bf16_activation_op(type)) {
        throw std::runtime_error(
            "vkop: no bf16 activation kernel for " + where + " (" + role +
            " is bfloat16). bf16 activations run only on Add/Sub/Mul/Div/Pow/"
            "Min/Max/Mod, Split, LayerNorm, RMSNorm, Softmax, MatMul, "
            "Attention, the unary family (Sin/Cos/Sigmoid/Tanh/Erf/Sqrt/Relu/"
            "Neg/Floor/Atan/Softplus/Clip), FusedElemwise, Cast, Gather, "
            "Expand, Concat, Transpose, RotaryEmbedding, Pad, Resize, MaxPool, "
            "GlobalAveragePool, Reduce*, InstanceNormalization, PRelu, CumSum "
            "and ScatterElements (plus the byte-moving Shape/Squeeze/Unsqueeze/"
            "Reshape/Slice); this op's _bf16 build does not decode bf16. "
            "Reconvert with "
            "--act fp16, or keep this op out of the bf16 domain.");
    }
    if (!buffer_backend) {
        throw std::runtime_error("vkop: no bf16 activation kernel for " +
                                 where +
                                 " on the image backend (bf16 is buffer-only)");
    }
    auto mixed = [&](const std::shared_ptr<ITensor> &t, const char *side,
                     size_t i) {
        if (t == nullptr || bf16_neighbour_ok(t->elem_kind())) {
            return;
        }
        if (side[0] == 'i' && bf16_int8_weight_slot(type, inputs, i)) {
            return;
        }
        if (bf16_other_float_ok(type, side[0] == 'o', i, t->elem_kind())) {
            return;
        }
        throw std::runtime_error(
            "vkop: " + where + " mixes bfloat16 activations with a " +
            elem_name(t->elem_kind()) + " " + side + " " + std::to_string(i) +
            ": its _bf16 module decodes that operand as bf16, so every float "
            "operand of a bf16 node outside the own-format slots (Cast, "
            "FusedElemwise, Gather table, Clip bounds, Pad constant, fp32 "
            "Rotary cos/sin) must be bfloat16 (no quantized payload)");
    };
    for (size_t i = 0; i < inputs.size(); ++i) {
        mixed(inputs[i], "input", i);
    }
    for (size_t i = 0; i < outputs.size(); ++i) {
        mixed(outputs[i], "output", i);
    }
    return true;
}

} // namespace core
} // namespace vkop
