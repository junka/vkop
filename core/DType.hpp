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

    // Weight-only quantized payloads, one byte per value, read by the buffer
    // MatMul kernel (see MatMulBuffer) with a per-output-column fp32 scale: the
    // byte is a code to scale, not a float to use. kFloat8E5M2 keeps 5
    // exponent bits to 2 mantissa bits (wide range, coarse steps),
    // kFloat8E4M3FN the other way round (4:3, so a finer step inside the +-448
    // range it saturates at). Neither carries a zero-point: the sign, exponent
    // and mantissa are decoded in the shader.
    kFloat8E4M3FN,
    kFloat8E5M2,

    // 4-bit weight-only payloads, two nibbles per byte, read by the buffer
    // MatMul kernel (see MatMulBuffer). kNF4's nibble is an unsigned INDEX into
    // the fixed 16-value NF4 codebook (the quantiles of a normal distribution),
    // dequantized as codebook[q] * group_absmax, kInt4's is a signed value, and
    // both take one fp32 scale per (K group, column).
    //
    // kFloat4E2M1's nibble is an E2M1 code (sign, 2 exponent bits, 1 mantissa
    // bit), so its 16 values are +-0, .5, 1, 1.5, 2, 3, 4, 6 -- a *relative*
    // step like fp8, with no Inf or NaN encoding. It is the payload of the
    // NVFP4 recipe and dequantizes against a two-level scale instead: one fp8
    // E4M3 factor per 16-element K block plus one fp32 factor per tensor
    // (value * block_scale * global_scale). The e4m3 block scale is what makes
    // a group of 16 affordable: an fp32 table would cost 4 bytes where this
    // costs 1.
    //
    // INT4 and FLOAT4E2M1 are ONNX TensorProto spellings (22 and 23); NF4 is
    // not an ONNX element type at all, so its name is vkop's own. All three are
    // written only by vkop's converter, and every consumer other than that one
    // kernel still fails at the loader — a format is only "supported" where a
    // kernel exists for it.
    kInt4,
    kNF4,
    kFloat4E2M1,

    // Spelled in a model file, but no kernel reads them yet. They are listed so
    // the loader can name the format and say "no kernel" instead of falling
    // into a storage guess.
    kUint8,
    kBFloat16,
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
    // The fp8 weight-only formats are read by the buffer MatMul kernel only
    // (see MatMulBuffer): one byte per value, decoded to float in the shader
    // and scaled per output column — the same shape as an int8 weight, which is
    // why they share its kernels.
    case ElemKind::kFloat8E4M3FN:
    case ElemKind::kFloat8E5M2:
    // The 4-bit weight-only formats are read by the buffer MatMul kernel only
    // (see MatMulBuffer): a packed nibble, scaled per K group (int4, nf4) or
    // per 16-value block plus a per-tensor factor (float4e2m1). Every other
    // consumer of them still fails at the loader, which is the point of the
    // whitelist — the format is only "supported" where a kernel exists.
    case ElemKind::kInt4:
    case ElemKind::kNF4:
    case ElemKind::kFloat4E2M1:
    case ElemKind::kBool:
    case ElemKind::kInt32:
    case ElemKind::kInt64:
    // uint8 rides an int8_t container the same way bool/fp8 do (the byte is a
    // payload the QDQ shaders decode), and is read by the QuantizeLinear/
    // DequantizeLinear op for externally-quantized (QDQ) models. ORT's dynamic
    // quantizer emits uint8 weights by default, so this must clear the loader
    // or the most common QDQ graph is rejected at graph-input creation.
    case ElemKind::kUint8:
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

// True for the two float formats a kernel can compute on directly (fp32/fp16).
// Used to tell a quantize/dequantize op which side of the boundary is the
// float operand without reaching for the throwing require_float_elem.
constexpr bool is_float_elem(ElemKind kind) {
    return kind == ElemKind::kFloat32 || kind == ElemKind::kFloat16;
}

// Guard for the word-granular buffer data-movement kernels (Slice, Concat,
// Gather, Split, Transpose, Expand). They model fp32/int32 (1 value per uint
// word) or fp16 (2 per word), with int64 on a dedicated pipeline; none has a
// build for a <= 1-byte-per-element payload, which would be copied 4-per-word
// and mis-sliced silently. Throws for int8/bool/fp8/4-bit so a byte-typed cache
// tensor cannot reach one of these movers without a real byte build existing.
void require_word_movable_elem(ElemKind kind, const char *op);

// The image (texture) kernels choose their element format from the `accuracy`
// uniform: 0 = fp32, 1 = fp16, 2 = an int8 weight-only payload. Returns that
// value for the formats the calling kernel actually has a build for and throws
// otherwise, so `allow_int8` belongs to the caller, not to the format:
// conv2d.comp dequantizes an int8 weight, while globalaveragepool.comp has no
// accuracy==2 path at all — it computes the sum and stores nothing, so picking
// 2 there loses the output silently rather than wrongly.
//
// The reason this cannot come from the storage type is the alias above: an
// int8_t container holds an int8 weight, a bool mask, an fp8 payload and a
// packed 4-bit payload alike, and only elem_kind tells them apart.
int image_accuracy_elem(ElemKind kind, const char *op, const char *what,
                        bool allow_int8);

// Guard for the byte-per-element copy paths (Reshape's int8/bool branch, which
// moves one byte per value). A packed 4-bit payload shares each byte between
// two values, so copying num_elements() bytes reads past what was stored, and
// an fp8 payload moved by a kernel that never dequantizes it just relocates the
// codes. Both ride the same int8_t container as a mask does.
void require_plain_byte_elem(ElemKind kind, const char *op, const char *what);

// Parse a model file's dtype string and demand a format the runtime computes
// on. Throws when the string names nothing vkop knows, and when it names a
// format that is recognized but has no kernel — the second case is what a
// quantized checkpoint hits first, and storing it under a guessed element type
// is how a wrong answer gets produced quietly.
ElemKind require_supported_elem(const std::string &name, const char *context);

} // namespace core
} // namespace vkop

#endif // CORE_DTYPE_HPP_
