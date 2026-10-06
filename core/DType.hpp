// Copyright 2025 @junka
#ifndef CORE_DTYPE_HPP_
#define CORE_DTYPE_HPP_

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <typeinfo>

namespace vkop {
namespace core {

// What a tensor's bytes MEAN. Distinct from ITensor::dtype(), which is the C++
// type the host stores them in:
//   - numerics (is this a quantized weight? a mask? fp16 or fp32?) read
//     elem_kind();
//   - storage (bytes per element, which std::vector the shader is handed) read
//     dtype().
// The split is not cosmetic. int8_t holds BOTH the LLM's bool masks and an int8
// quantized weight, and uint16_t holds fp16 — so a kernel that infers the
// numeric format from sizeof(T) or from an input-count convention silently
// reads quantized bytes as fp32 and produces a wrong answer with no error.
// Every ElemKind a running tensor carries originates from the model file's
// dtype string, checked against the whitelist below.
enum class ElemKind : uint8_t {
    kInvalid = 0,

    // Element formats the runtime computes on today.
    kFloat32,
    kFloat16,
    kInt8, // signed byte, e.g. a weight-only quantized kernel weight
    kBool,
    kInt32,
    kInt64,

    // 4-bit weight-only payloads, two nibbles per byte, read by the buffer
    // MatMul kernel (see MatMulBuffer) with a per-K-group fp32 scale. kNF4's
    // nibble is an unsigned INDEX into the fixed 16-value NF4 codebook (the
    // quantiles of a normal distribution), dequantized as
    // codebook[q] * group_absmax, while kInt4's is a signed value.
    //
    // Neither is an ONNX TensorProto spelling that a third-party graph uses
    // here: NF4 is not an ONNX element type at all, so its name is vkop's own.
    // Both are written only by vkop's converter, and every consumer other than
    // that one kernel still fails at the loader — a format is only "supported"
    // where a kernel exists for it.
    kInt4,
    kNF4,

    // Spelled in a model file, but no kernel reads them yet. They are listed so
    // the loader can name the format and say "no kernel" instead of falling
    // into a storage guess.
    kUint8,
    kBFloat16,
    kFloat8E4M3FN,
    kFloat8E5M2,
    kFloat4E2M1,
    kUint4,
};

// Bits per element. Sub-byte kinds are the reason byte counts and element
// counts are separate quantities: 5 x kInt4 is 3 bytes, not 5.
constexpr int elem_bits(ElemKind kind) {
    switch (kind) {
    case ElemKind::kFloat32:
    case ElemKind::kInt32:
        return 32;
    case ElemKind::kFloat16:
    case ElemKind::kBFloat16:
        return 16;
    case ElemKind::kInt8:
    case ElemKind::kUint8:
    case ElemKind::kBool:
    case ElemKind::kFloat8E4M3FN:
    case ElemKind::kFloat8E5M2:
        return 8;
    case ElemKind::kFloat4E2M1:
    case ElemKind::kInt4:
    case ElemKind::kUint4:
    case ElemKind::kNF4:
        return 4;
    case ElemKind::kInt64:
        return 64;
    case ElemKind::kInvalid:
        break;
    }
    return 0;
}

// Bytes to hold n_elements of `kind`, rounding up for sub-byte packing.
constexpr size_t elem_bytes(ElemKind kind, size_t n_elements) {
    const int bits = elem_bits(kind);
    if (bits == 0) {
        return 0;
    }
    return (n_elements * static_cast<size_t>(bits) + 7) / 8;
}

// True for the formats that pack several elements into one byte, so that a
// tensor's byte count is NOT prod(dims) * sizeof(storage): the container holds
// bytes, and two elements share one. The loader sizes such a tensor's staging
// and SSBO from elem_bytes() instead of its element count (see
// Tensor::set_payload_bytes), which is why ops must read a packed weight's
// logical K and N from its dims rather than from num_elements().
constexpr bool elem_kind_packed(ElemKind kind) {
    const int bits = elem_bits(kind);
    return bits > 0 && bits < 8;
}

const char *elem_name(ElemKind kind);

// The model file's dtype spelling -> ElemKind, or kInvalid when the string is
// not a format vkop knows anything about. Names follow ONNX's TensorProto
// DataType names, which is the vocabulary the converter writes (see
// model/pypi/onnx2vkop/dag.py _DATA_TYPE_MAP).
ElemKind elem_kind_from_name(const std::string &name);

// True when a kernel can compute on this format today. False for a recognized
// but unimplemented one, which is a different error for the caller to report.
constexpr bool elem_kind_supported(ElemKind kind) {
    switch (kind) {
    case ElemKind::kFloat32:
    case ElemKind::kFloat16:
    case ElemKind::kInt8:
    // The 4-bit weight-only formats are read by the buffer MatMul kernel only
    // (see MatMulBuffer): a packed nibble with a per-group fp32 scale. Every
    // other consumer of them still fails at the loader, which is the point of
    // the whitelist — the format is only "supported" where a kernel exists.
    case ElemKind::kInt4:
    case ElemKind::kNF4:
    case ElemKind::kBool:
    case ElemKind::kInt32:
    case ElemKind::kInt64:
        return true;
    default:
        return false;
    }
}

// The format a Tensor<T>'s storage type stands for. Tensor<T> uses this as its
// default elem_kind(); the loader overrides it when the file says otherwise
// (kBool on int8_t storage) or the format has no storage type at all.
template <typename T> constexpr ElemKind elem_kind_of_storage() {
    if constexpr (std::is_same_v<T, float>) {
        return ElemKind::kFloat32;
    } else if constexpr (std::is_same_v<T, uint16_t>) {
        return ElemKind::kFloat16;
    } else if constexpr (std::is_same_v<T, int8_t>) {
        return ElemKind::kInt8;
    } else if constexpr (std::is_same_v<T, int32_t>) {
        return ElemKind::kInt32;
    } else if constexpr (std::is_same_v<T, int64_t>) {
        return ElemKind::kInt64;
    } else {
        return ElemKind::kInvalid;
    }
}

// True when `storage` is a C++ type that can hold elements of `kind` without
// reinterpreting them. kBool rides on int8_t (the buffer ops consume a mask as
// bytes), which is exactly the aliasing elem_kind() exists to make explicit.
bool storage_matches_kind(const std::type_info &storage, ElemKind kind);

[[noreturn]] void throw_unsupported_elem(ElemKind kind, const char *op,
                                         const char *what);

// Guard for kernels that only read the two float formats — Gemm, MatMul. A
// quantized weight must arrive at a kernel that dequantizes it; handing it to
// one that reads the bytes as fp32 is a silent wrong answer.
void require_float_elem(ElemKind kind, const char *op, const char *what);

// Parse a model file's dtype string and demand a format the runtime computes
// on. Throws when the string names nothing vkop knows, and when it names a
// format that is recognized but has no kernel — the second case is what a
// quantized checkpoint hits first, and storing it under a guessed element type
// is how a wrong answer gets produced quietly.
ElemKind require_supported_elem(const std::string &name, const char *context);

} // namespace core
} // namespace vkop

#endif // CORE_DTYPE_HPP_
