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
    // Route-gated inside the op: MatMul keeps to the GEMV / split-K /
    // split-K WIDE / even-N fused / 64x64 tile kernels (KernelSelect's
    // act_bf16), Attention to its row kernel (no flash, no split-KV).
    case ops::OpType::MATMUL:
    case ops::OpType::ATTENTION:
    // Host-only: alias or copy bytes, never decode an element.
    case ops::OpType::SHAPE:
    case ops::OpType::SQUEEZE:
    case ops::OpType::UNSQUEEZE:
        return true;
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
            "Min/Max/Mod, Split, LayerNorm, RMSNorm, Softmax, MatMul and "
            "Attention (plus the byte-moving Shape/Squeeze/Unsqueeze); this "
            "op's _bf16 build does not decode bf16. Reconvert with "
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
        throw std::runtime_error(
            "vkop: " + where + " mixes bfloat16 activations with a " +
            elem_name(t->elem_kind()) + " " + side + " " + std::to_string(i) +
            ": its _bf16 module decodes every 16-bit operand as bf16 and reads "
            "no other float width, so every float operand of a bf16 node must "
            "be bfloat16 (no fp16/fp32 beside it, no quantized payload)");
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
