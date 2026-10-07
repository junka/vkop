// Copyright 2025 @junka
#include "core/DType.hpp"

namespace vkop {
namespace core {
namespace {

// The whitelist. The spellings are ONNX's TensorProto DataType names, since the
// converter writes exactly those (model/pypi/onnx2vkop/dag.py). A format that
// has no kernel yet is listed with supported=false so the loader can tell the
// user "recognized, not implemented" instead of silently picking a storage type
// for it.
struct NameEntry {
    const char *name;
    ElemKind kind;
};

constexpr NameEntry kNames[] = {
    {"float32", ElemKind::kFloat32},
    {"float16", ElemKind::kFloat16},
    {"int8", ElemKind::kInt8},
    {"bool", ElemKind::kBool},
    {"int32", ElemKind::kInt32},
    {"int64", ElemKind::kInt64},
    {"uint8", ElemKind::kUint8},
    {"bfloat16", ElemKind::kBFloat16},
    {"float8e4m3fn", ElemKind::kFloat8E4M3FN},
    {"float8e5m2", ElemKind::kFloat8E5M2},
    {"float4e2m1fn", ElemKind::kFloat4E2M1},
    {"int4", ElemKind::kInt4},
    {"uint4", ElemKind::kUint4},
    // Not an ONNX name (see the ElemKind comment): vkop's converter writes it,
    // and only the buffer MatMul kernel reads it.
    {"nf4", ElemKind::kNF4},
};

const char *kind_label(ElemKind kind) {
    for (const auto &e : kNames) {
        if (e.kind == kind) {
            return e.name;
        }
    }
    return "unknown";
}

} // namespace

const char *elem_name(ElemKind kind) { return kind_label(kind); }

ElemKind elem_kind_from_name(const std::string &name) {
    for (const auto &e : kNames) {
        if (name == e.name) {
            return e.kind;
        }
    }
    return ElemKind::kInvalid;
}

bool storage_matches_kind(const std::type_info &storage, ElemKind kind) {
    switch (kind) {
    case ElemKind::kFloat32:
        return storage == typeid(float);
    case ElemKind::kFloat16:
        return storage == typeid(uint16_t);
    case ElemKind::kInt8:
        return storage == typeid(int8_t);
    // A bool mask rides on int8_t storage: buffer ops read it as bytes, and
    // ONNX writes bool tensors as one byte per element. This alias is exactly
    // what elem_kind() exists to record, because the storage type alone cannot
    // tell a mask from a quantized weight.
    case ElemKind::kBool:
        return storage == typeid(int8_t);
    case ElemKind::kInt32:
        return storage == typeid(int32_t);
    case ElemKind::kInt64:
        return storage == typeid(int64_t);
    case ElemKind::kInvalid:
    case ElemKind::kUint8:
    case ElemKind::kBFloat16:
    // No C++ type means "one fp8 value", so these ride an int8_t container the
    // same way a bool mask does: the bytes are a payload only the shader
    // decodes, and elem_kind() is what records that. Kept false so that a
    // Tensor<T> never claims a float format from sizeof(T) alone.
    case ElemKind::kFloat8E4M3FN:
    case ElemKind::kFloat8E5M2:
    // The packed kinds are deliberately false here too: no storage type holds
    // "one int4 element" without reinterpreting, because two of them share a
    // byte. The loader gives them an int8_t container sized by elem_bytes()
    // instead (Tensor::set_payload_bytes), so num_elements() counts bytes, not
    // values.
    case ElemKind::kFloat4E2M1:
    case ElemKind::kInt4:
    case ElemKind::kUint4:
    case ElemKind::kNF4:
        return false;
    }
    return false;
}

void throw_unsupported_elem(ElemKind kind, const char *op, const char *what) {
    std::string msg = "vkop: ";
    msg += op;
    msg += " got ";
    msg += what;
    msg += " in element format ";
    msg += elem_name(kind);
    msg += elem_kind_supported(kind)
               ? " — this kernel does not read that format."
               : " — recognized format, no kernel implemented for it yet.";
    throw std::runtime_error(msg);
}

void require_float_elem(ElemKind kind, const char *op, const char *what) {
    if (kind != ElemKind::kFloat32 && kind != ElemKind::kFloat16) {
        throw_unsupported_elem(kind, op, what);
    }
}

ElemKind require_supported_elem(const std::string &name, const char *context) {
    const ElemKind kind = elem_kind_from_name(name);
    if (kind == ElemKind::kInvalid) {
        throw std::runtime_error("vkop: " + std::string(context) +
                                 " has unrecognized dtype '" + name + "'");
    }
    if (!elem_kind_supported(kind)) {
        throw std::runtime_error(std::string("vkop: ") + context +
                                 " is dtype '" + name +
                                 "', a recognized element format with no "
                                 "kernel implemented for it yet");
    }
    return kind;
}

} // namespace core
} // namespace vkop
