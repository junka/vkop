// Copyright 2026 @junka
// Element-format (dtype) contract tests. Host-side only: these check what the
// runtime promises about a tensor's bytes, not what a kernel computes.
#include "setup.hpp"
#include "core/DType.hpp"
#include "core/Tensor.hpp"
#include <gtest/gtest.h>

using vkop::core::ElemKind;
using vkop::core::Tensor;
using vkop::core::elem_bits;
using vkop::core::elem_bytes;
using vkop::core::elem_kind_from_name;
using vkop::core::elem_kind_packed;
using vkop::core::elem_kind_supported;
using vkop::core::elem_name;
using vkop::core::require_float_elem;
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

// Quantized formats are named so the loader can say "no kernel yet" instead of
// picking a storage type for them. They parse, but nothing may compute on them.
TEST(DTypeTest, RecognizesQuantizedNamesWithoutKernel) {
    for (const char *name : {"uint8", "bfloat16", "uint4"}) {
        const ElemKind kind = elem_kind_from_name(name);
        EXPECT_NE(kind, ElemKind::kInvalid) << name;
        EXPECT_FALSE(elem_kind_supported(kind)) << name;
        EXPECT_STREQ(elem_name(kind), name);
    }
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
// one fp32 per group.
TEST(DTypeTest, RecognizesPackedWeightOnlyNames) {
    for (const char *name : {"int4", "nf4", "float4e2m1fn"}) {
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
    try {
        require_supported_elem("bfloat16", "initializer w");
        FAIL() << "a format with no kernel must not load";
    } catch (const std::runtime_error &e) {
        EXPECT_NE(std::string(e.what()).find("no kernel"), std::string::npos);
    }
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

} // namespace
