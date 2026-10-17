// Copyright 2025 @junka
#ifndef CORE_SHADER_KEYS_HPP_
#define CORE_SHADER_KEYS_HPP_

#include <string>

#include "ops/Ops.hpp"

namespace vkop {
namespace core {

// Map (OpType, backend, fp16) to the shader's name inside vkop_shaders.bin.
// Names match the folder-derived symbol CMakeLists.txt bakes as C arrays
// today (shaders/buffer/add.comp -> buffer_add_spv[] -> archive key
// "buffer_add"). An empty return means "the runtime should leave the op's
// ctor-assigned SPIR-V pointer alone" -- either because the op has no GPU
// shader (Shape, SqueezeUnsqueeze run on the host), or because its fp16
// variant is not in DUAL_FP16_SHADERS and we do not want to feed the driver
// the wrong shader.
//
// The fp16 flag only affects names: an op whose buffer variant is not in the
// DUAL_FP16_SHADERS list falls back to the non-fp16 key even when fp16 is
// requested, matching what the leaf ctor already does.
inline std::string shader_key_for(vkop::ops::OpType type, bool backend_buffer,
                                  int fp16) {
    using T = vkop::ops::OpType;
    struct Spec {
        const char *image;
        const char *buffer;
        bool image_fp16;
        bool buffer_fp16;
    };
    // Entries mirror the extern symbols each op header references. Ops not
    // listed here (Shape, SqueezeUnsqueeze) run on the host and have no
    // shader to swap in.
    auto lookup_spec = [](T t) -> Spec {
        switch (t) {
        case T::ADD:
            return {"add", "add", true, true};
        case T::ATAN:
            return {"atan", "atan", false, true};
        case T::AVERAGEPOOL:
            return {"averagepool", nullptr, false, false};
        case T::BATCHNORM:
            return {"batchnorm", nullptr, false, false};
        case T::COL2IM:
            return {"col2im", nullptr, false, false};
        case T::CONCAT:
            return {"concat", "concat", false, true};
        case T::CONV2D:
            return {"conv2d", "conv2d", true, true};
        case T::COS:
            return {"cos", "cos", false, true};
        case T::DIV:
            return {"div", "div", false, true};
        case T::EMBEDDING_FORWARD:
            return {nullptr, "embedding_forward", false, false};
        case T::ERF:
            return {"erf", "erf", false, true};
        case T::EQUAL:
            return {nullptr, "equal", false, false};
        case T::EXPAND:
            return {nullptr, "expand", false, true};
        case T::FLOOR:
            return {"floor", "floor", false, false};
        case T::FUSED_ELEMWISE:
            return {nullptr, "fused_elemwise", false, true};
        case T::GATHER:
            return {nullptr, "gather", false, true};
        case T::GEMM:
            return {nullptr, "gemm", false, false};
        case T::GLOBALAVERAGEPOOL:
            return {"globalaveragepool", nullptr, false, false};
        case T::GRIDSAMPLE:
            return {"gridsample", nullptr, false, false};
        case T::LAYERNORM:
            return {"layernorm", "layernorm", false, true};
        case T::MATMUL:
            return {"matmul", "matmul", false, true};
        case T::MAXPOOL2D:
            return {"maxpool2d", nullptr, true, false};
        case T::MAX:
            return {nullptr, "max", false, true};
        case T::MIN:
            return {nullptr, "min", false, true};
        case T::MOD:
            return {nullptr, "mod", false, true};
        case T::MUL:
            return {"mul", "mul", false, true};
        case T::NEG:
            return {"neg", "neg", false, true};
        case T::NMS:
            return {"nms", nullptr, false, false};
        case T::NONZERO:
            return {nullptr, "nonzero", false, false};
        case T::POW:
            return {"pow", "pow", false, true};
        case T::PRELU:
            return {"prelu", "prelu", false, true};
        case T::QUANTIZE_LINEAR:
        case T::DEQUANTIZE_LINEAR:
            return {nullptr, "quantize_linear", false, false};
        case T::RANGE:
            return {nullptr, "range", false, false};
        case T::REDUCE:
        case T::REDUCEMEAN:
            return {"reduce", "reduce", false, true};
        case T::RELU:
            return {"relu", "relu", false, false};
        case T::RESHAPE:
            return {"reshape", "reshape", false, true};
        case T::RESIZE:
            return {"resize", "resize", false, true};
        case T::RMSNORM:
            return {nullptr, "rmsnorm", false, true};
        case T::ROTARY_EMBEDDING:
            return {nullptr, "rotary", false, true};
        case T::SCATTER_ELEMENTS:
            return {nullptr, "scatter_elements", false, true};
        case T::SCATTER_ND:
            return {nullptr, "scatternd", false, false};
        case T::SIGMOID:
            return {"sigmoid", "sigmoid", false, true};
        case T::SIN:
            return {"sin", "sin", false, true};
        case T::SLICE:
            return {"slice", "slice", false, true};
        case T::SOFTMAX:
            return {"softmax", "softmax", true, true};
        case T::SOFTPLUS:
            return {"softplus", "softplus", false, true};
        case T::SPLIT:
            return {"split", "split", false, true};
        case T::SQRT:
            return {"sqrt", "sqrt", false, true};
        case T::SUB:
            return {"sub", "sub", false, true};
        case T::TANH:
            return {"tanh", "tanh", false, true};
        case T::TOPK:
            return {nullptr, "topk", false, false};
        case T::TRANSPOSE:
            return {"transpose", "transpose", false, true};
        case T::ATTENTION:
            return {nullptr, "attention", false, true};
        case T::CAST:
            return {nullptr, "cast", false, false};
        case T::WHERE:
            return {nullptr, "where", false, false};
        // Host-only ops with no shader to swap.
        case T::SHAPE:
        case T::SQUEEZE:
        case T::UNSQUEEZE:
        case T::UNKNOWN:
        case T::TOTAL_NUM:
        default:
            return {nullptr, nullptr, false, false};
        }
    };

    Spec s = lookup_spec(type);
    const char *base = backend_buffer ? s.buffer : s.image;
    bool dual = backend_buffer ? s.buffer_fp16 : s.image_fp16;
    // Image fallback: if the requested backend has no variant but the other
    // one does (e.g. Attention is buffer-only), still emit the correct key.
    // This is only used when the runtime's backend_buffer_ and the op's
    // actual backend disagree; that mismatch is rare but keeps the archive
    // lookup aligned with what the leaf ctor already bound.
    if (!base) {
        base = backend_buffer ? s.image : s.buffer;
        dual = backend_buffer ? s.image_fp16 : s.buffer_fp16;
        if (!base) {
            return std::string();
        }
    }
    std::string key = backend_buffer ? "buffer_" : "image_";
    key += base;
    if (fp16 && dual) {
        key += "_fp16";
    }
    return key;
}

} // namespace core
} // namespace vkop

#endif // CORE_SHADER_KEYS_HPP_
