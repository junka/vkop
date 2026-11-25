// Copyright 2026 @junka
// Element-format (dtype) contract tests. Host-side only: these check what the
// runtime promises about a tensor's bytes, not what a kernel computes.
#include "setup.hpp"
#include "core/DType.hpp"
#include "core/Tensor.hpp"
#include <cmath>
#include <gtest/gtest.h>

using vkop::core::ElemKind;
using vkop::core::Tensor;
using vkop::core::elem_bits;
using vkop::core::elem_bytes;
using vkop::core::elem_kind_from_name;
using vkop::core::elem_kind_packed;
using vkop::core::elem_kind_supported;
using vkop::core::elem_name;
using vkop::core::image_accuracy_elem;
using vkop::core::require_float_elem;
using vkop::core::require_plain_byte_elem;
using vkop::core::require_supported_elem;
using vkop::core::require_word_movable_elem;
using vkop::core::storage_matches_kind;

namespace {

// The names a model file may use for a tensor the runtime computes on.
TEST(DTypeTest, RecognizesSupportedNames) {
    EXPECT_EQ(elem_kind_from_name("float32"), ElemKind::kFloat32);
    EXPECT_EQ(elem_kind_from_name("float16"), ElemKind::kFloat16);
    EXPECT_EQ(elem_kind_from_name("int8"), ElemKind::kInt8);
    EXPECT_EQ(elem_kind_from_name("bool"), ElemKind::kBool);
    EXPECT_EQ(elem_kind_from_name("int32"), ElemKind::kInt32);
    EXPECT_EQ(elem_kind_from_name("int64"), ElemKind::kInt64);
    for (const char *name :
         {"float32", "float16", "int8", "bool", "int32", "int64"}) {
        EXPECT_TRUE(elem_kind_supported(elem_kind_from_name(name))) << name;
    }
}

// bfloat16 weights: read natively by the buffer MatMul/Gemm B and Gather's
// table. It shares fp16's uint16_t container -- which is the point of
// elem_kind: storage alone says "fp16", and only the recorded kind keeps a
// kernel from reading bf16 bits with unpackHalf2x16.
TEST(DTypeTest, RecognizesBFloat16Weights) {
    const ElemKind kind = elem_kind_from_name("bfloat16");
    EXPECT_EQ(kind, ElemKind::kBFloat16);
    EXPECT_TRUE(elem_kind_supported(kind));
    EXPECT_NO_THROW(require_supported_elem("bfloat16", "initializer w"));
    EXPECT_FALSE(elem_kind_packed(kind));
    EXPECT_EQ(elem_bits(kind), 16);
    EXPECT_EQ(elem_bytes(kind, 5), 10u);
    EXPECT_STREQ(elem_name(kind), "bfloat16");
    EXPECT_TRUE(storage_matches_kind(typeid(uint16_t), kind));
    EXPECT_FALSE(storage_matches_kind(typeid(float), kind));
    EXPECT_FALSE(storage_matches_kind(typeid(int8_t), kind));
    // The container's default meaning is still fp16; bf16 must be pinned.
    Tensor<uint16_t> t(std::vector<int>{4});
    EXPECT_EQ(t.elem_kind(), ElemKind::kFloat16);
    t.set_elem_kind(ElemKind::kBFloat16);
    EXPECT_EQ(t.elem_kind(), ElemKind::kBFloat16);
    EXPECT_EQ(t.size(), 8);
    // A bf16 weight is not a float the generic float-only kernels may read.
    EXPECT_FALSE(vkop::core::is_float_elem(kind));
    EXPECT_THROW(require_float_elem(kind, "Add", "input 0"),
                 std::runtime_error);
}

// The host mirror of shaders/common/bf16.comp (core::float_to_bf16 /
// bf16_to_float). The shader's float_to_bf16 is the same three lines, so this
// is what pins its rounding: ties to even, NaN kept NaN (quiet), Inf and the
// largest-finite overflow, signed zero, and denormals, which stay exact bit
// truncations/roundings on the integer path (no flush).
TEST(DTypeTest, BFloat16RoundToNearestEven) {
    using vkop::core::bf16_to_float;
    using vkop::core::float_to_bf16;
    auto bits = [](uint32_t u) {
        float f;
        std::memcpy(&f, &u, sizeof(f));
        return f;
    };
    // Exact values pass through.
    EXPECT_EQ(float_to_bf16(1.0F), 0x3F80);
    EXPECT_EQ(float_to_bf16(-2.0F), 0xC000);
    EXPECT_EQ(float_to_bf16(0.0F), 0x0000);
    EXPECT_EQ(float_to_bf16(-0.0F), 0x8000);
    // Below / above half an ulp.
    EXPECT_EQ(float_to_bf16(bits(0x3F807FFFu)), 0x3F80);
    EXPECT_EQ(float_to_bf16(bits(0x3F808001u)), 0x3F81);
    // Exact ties: even lsb stays, odd lsb rounds up to even.
    EXPECT_EQ(float_to_bf16(bits(0x3F808000u)), 0x3F80);
    EXPECT_EQ(float_to_bf16(bits(0x3F818000u)), 0x3F82);
    EXPECT_EQ(float_to_bf16(bits(0xBF818000u)), 0xBF82);
    // A tie that carries through the mantissa into the exponent.
    EXPECT_EQ(float_to_bf16(bits(0x3FFF8000u)), 0x4000);
    // Inf stays Inf; the largest finite bf16 stays; past it rounds to Inf.
    EXPECT_EQ(float_to_bf16(bits(0x7F800000u)), 0x7F80);
    EXPECT_EQ(float_to_bf16(bits(0xFF800000u)), 0xFF80);
    EXPECT_EQ(float_to_bf16(bits(0x7F7F0000u)), 0x7F7F);
    EXPECT_EQ(float_to_bf16(bits(0x7F7FFFFFu)), 0x7F80);
    // NaN: a payload only in the low 16 bits would truncate to Inf; it must
    // come out a quiet NaN with its sign.
    EXPECT_EQ(float_to_bf16(bits(0x7F800001u)), 0x7FC0);
    EXPECT_EQ(float_to_bf16(bits(0xFF800001u)), 0xFFC0);
    EXPECT_EQ(float_to_bf16(bits(0x7FC00000u)), 0x7FC0);
    EXPECT_EQ(float_to_bf16(bits(0x7FA50000u)), 0x7FE5);
    EXPECT_TRUE(std::isnan(bf16_to_float(float_to_bf16(std::nanf("")))));
    // fp32 denormals: rounded like any other value; the smallest round to
    // (signed) zero, a tie on the denormal grid goes to even, and a carry
    // out of the denormal mantissa produces the smallest normal.
    EXPECT_EQ(float_to_bf16(bits(0x00000001u)), 0x0000);
    EXPECT_EQ(float_to_bf16(bits(0x80000001u)), 0x8000);
    EXPECT_EQ(float_to_bf16(bits(0x00018000u)), 0x0002);
    EXPECT_EQ(float_to_bf16(bits(0x00010000u)), 0x0001);
    EXPECT_EQ(float_to_bf16(bits(0x007FC000u)), 0x0080);
    // Decode is exact, denormals included.
    EXPECT_EQ(bf16_to_float(0x3F80), 1.0F);
    EXPECT_EQ(bf16_to_float(0xC040), -3.0F);
    EXPECT_EQ(bf16_to_float(0x0001), bits(0x00010000u));
    EXPECT_TRUE(std::isinf(bf16_to_float(0x7F80)));
    // Every finite bf16 round-trips through fp32 unchanged.
    for (uint32_t h = 0; h < 0x10000u; ++h) {
        const uint16_t b = static_cast<uint16_t>(h);
        const float f = bf16_to_float(b);
        if (std::isnan(f)) {
            EXPECT_TRUE(std::isnan(bf16_to_float(float_to_bf16(f))));
            continue;
        }
        ASSERT_EQ(float_to_bf16(f), b) << std::hex << h;
    }
}

// uint8 rides an int8_t container and is decoded by the QDQ shaders
// (QuantizeLinear/DequantizeLinear) for externally-quantized (QDQ) models —
// ORT's dynamic quantizer emits uint8 weights by default. It is now a
// supported format (cleared the loader's gate), unlike bfloat16 above.
TEST(DTypeTest, RecognizesUint8QDQ) {
    const ElemKind kind = elem_kind_from_name("uint8");
    EXPECT_NE(kind, ElemKind::kInvalid);
    EXPECT_TRUE(elem_kind_supported(kind));
    EXPECT_FALSE(elem_kind_packed(kind));
    EXPECT_EQ(elem_bits(kind), 8);
    EXPECT_EQ(elem_bytes(kind, 8), 8u);
    EXPECT_STREQ(elem_name(kind), "uint8");
    // No C++ type spells "one uint8 value"; it rides int8_t storage and the
    // elem_kind records the unsigned interpretation, same as bool/fp8.
    EXPECT_FALSE(storage_matches_kind(typeid(int8_t), kind));
}

// The fp8 weight-only payloads the buffer MatMul kernel decodes: one byte per
// value, so unlike int4 they need no packing and their byte count is their value
// count -- but no C++ type spells them, so a Tensor<int8_t> only holds their
// bytes and only the kernel turns them into floats.
TEST(DTypeTest, RecognizesFp8WeightOnlyNames) {
    for (const char *name : {"float8e4m3fn", "float8e5m2"}) {
        const ElemKind kind = elem_kind_from_name(name);
        EXPECT_NE(kind, ElemKind::kInvalid) << name;
        EXPECT_TRUE(elem_kind_supported(kind)) << name;
        EXPECT_FALSE(elem_kind_packed(kind)) << name;
        EXPECT_EQ(elem_bits(kind), 8);
        EXPECT_EQ(elem_bytes(kind, 8), 8u);
        EXPECT_STREQ(elem_name(kind), name);
        EXPECT_FALSE(storage_matches_kind(typeid(int8_t), kind)) << name;
    }
}

// The 4-bit weight-only formats the buffer MatMul kernel unpacks. Supported,
// but still not a storage type: their bytes are packed two to the byte, so a
// tensor holds them as raw bytes with the logical shape in its dims (see
// Tensor::set_payload_bytes). float4e2m1 is the NVFP4 payload: same packing as
// int4/nf4, a different nibble meaning (E2M1) and a two-level scale instead of
// one fp32 per group. uint4 is int4's unsigned counterpart: same packing and
// same group-scale table, a nibble read as [0, 15].
TEST(DTypeTest, RecognizesPackedWeightOnlyNames) {
    for (const char *name : {"int4", "nf4", "float4e2m1fn", "uint4"}) {
        const ElemKind kind = elem_kind_from_name(name);
        EXPECT_NE(kind, ElemKind::kInvalid) << name;
        EXPECT_TRUE(elem_kind_supported(kind)) << name;
        EXPECT_TRUE(elem_kind_packed(kind)) << name;
        EXPECT_EQ(elem_bits(kind), 4);
        EXPECT_STREQ(elem_name(kind), name);
        EXPECT_FALSE(storage_matches_kind(typeid(int8_t), kind)) << name;
        EXPECT_EQ(elem_bytes(kind, 8), 4u);
    }
    // "packed" is a statement about the bytes, not about having a kernel.
    EXPECT_TRUE(elem_kind_packed(ElemKind::kUint4));
    for (ElemKind kind : {ElemKind::kInt8, ElemKind::kBool, ElemKind::kFloat16,
                          ElemKind::kFloat32, ElemKind::kInvalid}) {
        EXPECT_FALSE(elem_kind_packed(kind)) << elem_name(kind);
    }
}

// Anything else — including a missing dtype — is not a format vkop can reason
// about, so it must never reach a tensor.
TEST(DTypeTest, RejectsUnknownNames) {
    for (const char *name : {"", "float64", "int16", "string", "fp16", "FLOAT32",
                             "float8", "int2"}) {
        EXPECT_EQ(elem_kind_from_name(name), ElemKind::kInvalid) << name;
    }
}

TEST(DTypeTest, RequireSupportedElemThrowsLoudly) {
    EXPECT_NO_THROW(require_supported_elem("int8", "initializer w"));
    EXPECT_THROW(require_supported_elem("float64", "initializer w"),
                 std::runtime_error);
    // The two failure modes read differently on purpose.
    // Every recognized format now has a reader (bfloat16, the last one, is
    // read by the MatMul/Gemm/Gather kernels), so the "recognized, no kernel"
    // branch has no name left to reach it. The unrecognized branch stays.
    try {
        require_supported_elem("weird", "initializer w");
        FAIL() << "an unrecognized dtype must not load";
    } catch (const std::runtime_error &e) {
        EXPECT_NE(std::string(e.what()).find("unrecognized"),
                  std::string::npos);
    }
}

// Byte counts are not element counts times sizeof once a format packs several
// elements into one byte, which is why the file records both.
TEST(DTypeTest, ElementWidthAndByteMath) {
    EXPECT_EQ(elem_bits(ElemKind::kFloat32), 32);
    EXPECT_EQ(elem_bits(ElemKind::kFloat16), 16);
    EXPECT_EQ(elem_bits(ElemKind::kInt64), 64);
    EXPECT_EQ(elem_bits(ElemKind::kInt8), 8);
    EXPECT_EQ(elem_bits(ElemKind::kInt4), 4);
    EXPECT_EQ(elem_bits(ElemKind::kInvalid), 0);

    EXPECT_EQ(elem_bytes(ElemKind::kFloat32, 3), 12u);
    EXPECT_EQ(elem_bytes(ElemKind::kFloat16, 3), 6u);
    EXPECT_EQ(elem_bytes(ElemKind::kInt64, 2), 16u);
    // Sub-byte: rounds up, and a 0-element tensor is 0 bytes.
    EXPECT_EQ(elem_bytes(ElemKind::kInt4, 8), 4u);
    EXPECT_EQ(elem_bytes(ElemKind::kInt4, 5), 3u);
    EXPECT_EQ(elem_bytes(ElemKind::kFloat4E2M1, 1), 1u);
    EXPECT_EQ(elem_bytes(ElemKind::kInt4, 0), 0u);
    EXPECT_EQ(elem_bytes(ElemKind::kInvalid, 16), 0u);
}

TEST(DTypeTest, StorageMatchesKind) {
    EXPECT_TRUE(storage_matches_kind(typeid(float), ElemKind::kFloat32));
    EXPECT_TRUE(storage_matches_kind(typeid(uint16_t), ElemKind::kFloat16));
    EXPECT_TRUE(storage_matches_kind(typeid(int8_t), ElemKind::kInt8));
    // A bool mask rides on int8_t storage; that alias is recorded, not guessed.
    EXPECT_TRUE(storage_matches_kind(typeid(int8_t), ElemKind::kBool));
    EXPECT_TRUE(storage_matches_kind(typeid(int32_t), ElemKind::kInt32));
    EXPECT_TRUE(storage_matches_kind(typeid(int64_t), ElemKind::kInt64));
    EXPECT_FALSE(storage_matches_kind(typeid(uint16_t), ElemKind::kFloat32));
    EXPECT_FALSE(storage_matches_kind(typeid(int8_t), ElemKind::kFloat8E4M3FN));
    EXPECT_FALSE(storage_matches_kind(typeid(int8_t), ElemKind::kInt4));
}

// A tensor's default element format comes from its storage type; the loader's
// recorded dtype overrides it.
TEST(DTypeTest, TensorElemKind) {
    Tensor<float> f32(4);
    Tensor<uint16_t> f16(4);
    Tensor<int8_t> i8(4);
    Tensor<int64_t> i64(4);
    EXPECT_EQ(f32.elem_kind(), ElemKind::kFloat32);
    EXPECT_EQ(f16.elem_kind(), ElemKind::kFloat16);
    EXPECT_EQ(i8.elem_kind(), ElemKind::kInt8);
    EXPECT_EQ(i64.elem_kind(), ElemKind::kInt64);

    i8.set_elem_kind(ElemKind::kBool);
    EXPECT_EQ(i8.elem_kind(), ElemKind::kBool);
    // The storage type is unchanged by the label: bytes are still one per
    // element, which is what dtype() reports.
    EXPECT_EQ(i8.dtype(), typeid(int8_t));
}

// A sub-byte weight: the bytes hold half as many values as the dims describe,
// so the tensor's byte count is set from the payload instead of from
// prod(dims) * sizeof(storage). The guards here are what keep a mis-sized or
// late-sized packed tensor from silently reading the wrong bytes.
TEST(DTypeTest, SetPayloadBytesShrinksOnlyPackedStorage) {
    Tensor<int8_t> w(64); // 64 logical int4 values, [8, 8]-shaped
    w.set_elem_kind(ElemKind::kInt4);
    EXPECT_EQ(w.size(), 64);
    EXPECT_NO_THROW(w.set_payload_bytes(32));
    EXPECT_EQ(w.size(), 32);
    // num_elements() stays a byte-count division; the logical K and N come from
    // the dims, which is why ops read packed tensors through getShape().
    EXPECT_EQ(w.num_elements(), 32);

    // Not a packed format, so the byte count is not the caller's to restate.
    Tensor<int8_t> i8(64);
    EXPECT_THROW(i8.set_payload_bytes(32), std::runtime_error);
    // Larger than the storage, or empty, means the dims and the payload
    // disagree.
    Tensor<int8_t> bad(64);
    bad.set_elem_kind(ElemKind::kInt4);
    EXPECT_THROW(bad.set_payload_bytes(65), std::runtime_error);
    EXPECT_THROW(bad.set_payload_bytes(0), std::runtime_error);
    // Too late: the host buffer already holds the full-size copy.
    Tensor<int8_t> uploaded(64);
    uploaded.set_elem_kind(ElemKind::kInt4);
    uploaded.fillToCPU(std::vector<int8_t>(64, 0));
    EXPECT_THROW(uploaded.set_payload_bytes(32), std::runtime_error);
    try {
        i8.set_payload_bytes(32);
        FAIL() << "a non-packed tensor must not restate its byte count";
    } catch (const std::runtime_error &e) {
        EXPECT_NE(std::string(e.what()).find("non-packed"), std::string::npos);
    }
}

// The guard float-only kernels run behind.
TEST(DTypeTest, RequireFloatElem) {
    EXPECT_NO_THROW(require_float_elem(ElemKind::kFloat32, "Gemm", "input 0"));
    EXPECT_NO_THROW(require_float_elem(ElemKind::kFloat16, "Gemm", "input 0"));
    for (ElemKind kind :
         {ElemKind::kInt8, ElemKind::kBool, ElemKind::kInt64, ElemKind::kInt4,
          ElemKind::kFloat8E4M3FN, ElemKind::kInvalid}) {
        EXPECT_THROW(require_float_elem(kind, "Gemm", "input 1"),
                     std::runtime_error)
            << elem_name(kind);
    }
    try {
        require_float_elem(ElemKind::kInt8, "MatMul", "input 1");
        FAIL() << "a quantized operand must not reach a float-only kernel";
    } catch (const std::runtime_error &e) {
        const std::string what = e.what();
        EXPECT_NE(what.find("MatMul"), std::string::npos);
        EXPECT_NE(what.find("input 1"), std::string::npos);
        EXPECT_NE(what.find("int8"), std::string::npos);
    }
}

// The guard the word-granular buffer movers (Slice/Concat/Gather/Split/
// Transpose/Expand) run behind. They move 2-byte (fp16) or 4-byte (fp32/int32)
// elements, with int64 on a dedicated path; a <= 1-byte payload has no build and
// would be copied 4-per-word and mis-sliced silently — so it must throw loudly.
// This is the precondition the fp8 KV cache clears before its cache tensors can
// reach a mover, and it also closes the same latent hole for int8/bool today.
TEST(DTypeTest, RequireWordMovableElem) {
    // Word-sized formats a mover can carry.
    for (ElemKind kind : {ElemKind::kFloat32, ElemKind::kFloat16,
                          ElemKind::kBFloat16, ElemKind::kInt32,
                          ElemKind::kInt64}) {
        EXPECT_NO_THROW(require_word_movable_elem(kind, "Slice"))
            << elem_name(kind);
    }
    // <= 1 byte per element: no mover build, must reject.
    for (ElemKind kind :
         {ElemKind::kInt8, ElemKind::kUint8, ElemKind::kBool,
          ElemKind::kFloat8E4M3FN, ElemKind::kFloat8E5M2, ElemKind::kInt4,
          ElemKind::kUint4, ElemKind::kNF4, ElemKind::kFloat4E2M1}) {
        EXPECT_THROW(require_word_movable_elem(kind, "Concat"),
                     std::runtime_error)
            << elem_name(kind);
    }
    try {
        require_word_movable_elem(ElemKind::kFloat8E4M3FN, "Gather");
        FAIL() << "an fp8 payload must not reach a word mover silently";
    } catch (const std::runtime_error &e) {
        const std::string what = e.what();
        EXPECT_NE(what.find("Gather"), std::string::npos);
        EXPECT_NE(what.find("float8e4m3fn"), std::string::npos);
    }
}

TEST(DTypeTest, ImageAccuracyElem) {
    // conv2d.comp has all three builds, keyed by what the bytes mean.
    EXPECT_EQ(image_accuracy_elem(ElemKind::kFloat32, "Conv", "input 1", true),
              0);
    EXPECT_EQ(image_accuracy_elem(ElemKind::kFloat16, "Conv", "input 1", true),
              1);
    EXPECT_EQ(image_accuracy_elem(ElemKind::kInt8, "Conv", "input 1", true), 2);

    // globalaveragepool.comp has no accuracy==2 path: it sums and stores
    // nothing, so an int8_t container there has to fail instead.
    EXPECT_THROW(image_accuracy_elem(ElemKind::kInt8, "GlobalAveragePool",
                                     "input 0", /*allow_int8=*/false),
                 std::runtime_error);

    // A byte container that is not an int8 weight is not accuracy 2 either.
    for (ElemKind kind : {ElemKind::kBool, ElemKind::kUint8,
                          ElemKind::kFloat8E4M3FN, ElemKind::kFloat8E5M2,
                          ElemKind::kInt4, ElemKind::kUint4, ElemKind::kNF4,
                          ElemKind::kFloat4E2M1}) {
        EXPECT_THROW(
            image_accuracy_elem(kind, "Conv", "input 1", /*allow_int8=*/true),
            std::runtime_error)
            << elem_name(kind);
    }

    try {
        image_accuracy_elem(ElemKind::kInt8, "GlobalAveragePool", "input 0",
                            false);
        FAIL() << "an int8 payload must not pick a missing GAP build silently";
    } catch (const std::runtime_error &e) {
        const std::string what = e.what();
        EXPECT_NE(what.find("GlobalAveragePool"), std::string::npos);
        EXPECT_NE(what.find("int8"), std::string::npos);
    }
}

TEST(DTypeTest, RequirePlainByteElem) {
    EXPECT_NO_THROW(require_plain_byte_elem(ElemKind::kInt8, "Reshape",
                                            "input 0"));
    EXPECT_NO_THROW(require_plain_byte_elem(ElemKind::kBool, "Reshape",
                                            "input 0"));

    // Packed: two values share a byte, so the copy's length is not the payload.
    for (ElemKind kind : {ElemKind::kInt4, ElemKind::kUint4, ElemKind::kNF4,
                          ElemKind::kFloat4E2M1}) {
        try {
            require_plain_byte_elem(kind, "Reshape", "input 0");
            FAIL() << elem_name(kind);
        } catch (const std::runtime_error &e) {
            const std::string what = e.what();
            EXPECT_NE(what.find("share one byte"), std::string::npos) << what;
        }
    }

    // One byte per value, but the byte is a code nothing here decodes.
    for (ElemKind kind :
         {ElemKind::kFloat8E4M3FN, ElemKind::kFloat8E5M2, ElemKind::kUint8}) {
        try {
            require_plain_byte_elem(kind, "Reshape", "input 0");
            FAIL() << elem_name(kind);
        } catch (const std::runtime_error &e) {
            const std::string what = e.what();
            EXPECT_NE(what.find("never decodes"), std::string::npos) << what;
        }
    }
}

// fp32 -> fp16 must round to nearest-even across the whole range, including the
// carry out of the mantissa: a value just under a power of two rounds up to it,
// and on an odd fp16 exponent that carry lands on an exponent bit that is
// already set. Combining the fields with OR dropped the carry there and stored
// half the value, which showed up as fp16 kernels drifting on Windows (MSVC
// takes this portable branch; x86-64 GCC/Clang use _cvtss_sh).
TEST(DTypeTest, Fp16ConversionKeepsMantissaCarry) {
    const float above_half = std::nextafter(0.5F, 0.0F);
    const float above_two = std::nextafter(2.0F, 0.0F);
    EXPECT_EQ(vkop::core::ITensor::fp32_to_fp16(above_half), 0x3800);  // 0.5
    EXPECT_EQ(vkop::core::ITensor::fp32_to_fp16(-above_half), 0xB800); // -0.5
    EXPECT_EQ(vkop::core::ITensor::fp32_to_fp16(above_two), 0x4000);   // 2.0
    EXPECT_EQ(vkop::core::ITensor::fp16_to_fp32(0x3800), 0.5F);
    EXPECT_EQ(vkop::core::ITensor::fp16_to_fp32(0xB800), -0.5F);
}

} // namespace

// The LoadModel fallback scan (core::Bf16RangeScan) must agree with the
// converter's bit test (onnx2vkop bf16_weights.TileScan): |w| > 65504, inf or
// NaN, on every one of the 65536 patterns, and on a byte stream cut at odd
// offsets (a staging chunk boundary is not promised to be even).
TEST(DTypeTest, Bf16RangeScanMatchesFp16Range) {
    using vkop::core::bf16_past_fp16;
    using vkop::core::bf16_to_float;
    for (uint32_t b = 0; b < 65536; ++b) {
        const auto h = static_cast<uint16_t>(b);
        const float v = bf16_to_float(h);
        ASSERT_EQ(bf16_past_fp16(h), !(std::fabs(v) <= 65504.0F)) << std::hex
                                                                  << b;
    }
    EXPECT_FALSE(bf16_past_fp16(0x477F));
    EXPECT_TRUE(bf16_past_fp16(0x4780));
    EXPECT_TRUE(bf16_past_fp16(0xC780));
    EXPECT_FALSE(bf16_past_fp16(0x8000));

    std::vector<uint16_t> in_range = {0x3F80, 0x477F, 0xC77F, 0x8000, 0x0001};
    const auto feed = [](const std::vector<uint16_t> &v,
                         std::initializer_list<size_t> cuts) {
        vkop::core::Bf16RangeScan s;
        const auto *p = reinterpret_cast<const uint8_t *>(v.data());
        size_t pos = 0;
        for (size_t c : cuts) {
            s.feed(p + pos, c);
            pos += c;
        }
        s.feed(p + pos, v.size() * 2 - pos);
        return s.overflow();
    };
    EXPECT_FALSE(feed(in_range, {}));
    EXPECT_FALSE(feed(in_range, {1, 2, 3}));
    auto hit = in_range;
    hit[2] = 0x4780; // bytes 4..5
    EXPECT_TRUE(feed(hit, {}));
    EXPECT_TRUE(feed(hit, {5}));      // split inside the element
    EXPECT_TRUE(feed(hit, {1, 2, 2})); // pending low byte carried twice
    hit[2] = 0x7FC0;                   // NaN
    EXPECT_TRUE(feed(hit, {3}));
}
