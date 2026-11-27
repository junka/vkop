// Copyright 2026 @junka
// WeightPager pages a MoE slot whose resident buffer and source bytes differ in
// 16-bit float format.
//
// A --act bf16 conversion declares a slot read by bf16 nodes as bfloat16
// (onnx2vkop fp16_graph._apply_bf16_casts), while the checkpoint it pages from
// keeps its own format, recorded in the row's moe_source_dtype. A bf16
// checkpoint -- every MoE export -- is then a verbatim copy; an fp16 one has to
// be rounded on the way in, with the converter's rounding (fp16 -> fp32 ->
// bf16 RNE), or the slot would hold fp16 bits that every bf16 reader decodes as
// a plausible wrong number. The pager used to copy every source verbatim
// whatever its recorded format.
//
// Three halves: the rule table (what converts, what is refused), the host
// conversion against the converter's own reference values, and a real
// page-in into the bf16 table of bf16_gate_gather.vkopbin, read back through
// Gather (a bit copy of the table rows).

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "core/DType.hpp"
#include "core/WeightPager.hpp"
#include "core/runtime.hpp"
#include "setup.hpp"

#include <gtest/gtest.h>

namespace {

using vkop::core::bf16_to_float;
using vkop::core::ElemKind;
using vkop::core::float_to_bf16;
using vkop::core::ITensor;
using vkop::core::PagedSource;
using vkop::core::WeightPager;
using Convert = WeightPager::Convert;

uint16_t fp16_bits(float f) { return ITensor::fp32_to_fp16(f); }

std::shared_ptr<vkop::core::Runtime> load_gather() {
    if (!vkop::tests::TestEnv::is_initialized())
        vkop::tests::TestEnv::initialize();
    auto rt = std::make_shared<vkop::core::Runtime>(
        vkop::tests::TestEnv::get_command_pool(),
        std::string(BF16_GATE_DATA) + "/bf16_gate_gather.vkopbin", 0);
    rt->set_backend_buffer(true);
    rt->LoadModel();
    return rt;
}

// Rows 3 and 1 of the [4, 8] table, as raw bf16 bits.
std::vector<uint16_t> gather_rows_3_1(
    const std::shared_ptr<vkop::core::Runtime> &rt) {
    auto pool = vkop::tests::TestEnv::get_command_pool();
    auto idx = vkop::core::as_tensor<int64_t>(rt->GetInput("idx"));
    auto y = vkop::core::as_tensor<uint16_t>(rt->GetOutput("y"));
    EXPECT_TRUE(idx && y);
    if (!idx || !y) return {};
    EXPECT_EQ(y->elem_kind(), ElemKind::kBFloat16);
    std::vector<int64_t> iv{3, 1};
    idx->fillToCPU(iv);
    idx->copyToGPU(pool, iv.data());
    rt->Run();
    rt->ReadResult();
    y->copyToCPU(pool);
    std::vector<uint16_t> out(16);
    for (int i = 0; i < 16; ++i) out[i] = (*y)[i];
    return out;
}

// A source file holding `words` after `pad` bytes of junk, so the page-in has
// to honor the offset too.
std::string write_source(const std::string &tag,
                         const std::vector<uint16_t> &words, size_t pad) {
    const std::string path = ::testing::TempDir() + "wp_bf16_" + tag + ".bin";
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    std::vector<char> junk(pad, '\x5a');
    f.write(junk.data(), static_cast<std::streamsize>(junk.size()));
    f.write(reinterpret_cast<const char *>(words.data()),
            static_cast<std::streamsize>(words.size() * 2));
    return path;
}

// 32 values for the [4, 8] table: most are fp16-exact but not bf16-exact, so a
// verbatim copy of the fp16 bits and a truncation would both be caught.
std::vector<float> table_values() {
    std::vector<float> v(32);
    for (int i = 0; i < 32; ++i) v[i] = 1.0F + static_cast<float>(i) * 0x1p-10F;
    // Row 1 (read back below) carries the rounding cases.
    v[11] = 65504.0F;               // rounds up to 65536 (0x4780)
    v[13] = 1.0F + 3.0F * 0x1p-8F;  // a tie: rounds to even (0x3F82)
    v[24] = -2.5F;                  // row 3 starts here: an exact value
    return v;
}

} // namespace

TEST(WeightPagerBf16Test, ConversionRules) {
    EXPECT_EQ(WeightPager::conversion_for(ElemKind::kInvalid,
                                          ElemKind::kBFloat16),
              Convert::kNone);
    EXPECT_EQ(WeightPager::conversion_for(ElemKind::kBFloat16,
                                          ElemKind::kBFloat16),
              Convert::kNone);
    EXPECT_EQ(WeightPager::conversion_for(ElemKind::kFloat16,
                                          ElemKind::kFloat16),
              Convert::kNone);
    EXPECT_EQ(WeightPager::conversion_for(ElemKind::kFloat16,
                                          ElemKind::kBFloat16),
              Convert::kFp16ToBf16);
    EXPECT_EQ(WeightPager::conversion_for(ElemKind::kBFloat16,
                                          ElemKind::kFloat16),
              Convert::kBf16ToFp16);
    // A size-changing or non-float pair has no on-the-way-in conversion.
    EXPECT_THROW(WeightPager::conversion_for(ElemKind::kFloat32,
                                             ElemKind::kBFloat16),
                 std::runtime_error);
    EXPECT_THROW(WeightPager::conversion_for(ElemKind::kInt8,
                                             ElemKind::kFloat16),
                 std::runtime_error);
}

// The values test_bf16_fp16_parity.py pins for the converter's streamed
// fp16 -> bf16 payload: the pager must land on the same bits.
TEST(WeightPagerBf16Test, Fp16ToBf16RoundsLikeTheConverter) {
    std::vector<uint16_t> w{fp16_bits(65504.0F), fp16_bits(-65504.0F),
                            fp16_bits(1.0F + 0x1p-8F),
                            fp16_bits(1.0F + 3.0F * 0x1p-8F),
                            fp16_bits(-0.0F), 0x7C00 /* +inf */};
    WeightPager::convert_half_payload(w.data(), w.size() * 2,
                                      Convert::kFp16ToBf16);
    EXPECT_EQ(w[0], 0x4780);
    EXPECT_EQ(w[1], 0xC780);
    EXPECT_EQ(w[2], 0x3F80);
    EXPECT_EQ(w[3], 0x3F82);
    EXPECT_EQ(w[4], 0x8000);
    EXPECT_EQ(w[5], 0x7F80);

    // Every fp16 bit pattern, against the host mirror of the shader encode.
    std::vector<uint16_t> all(1u << 16);
    for (uint32_t i = 0; i < all.size(); ++i) all[i] = static_cast<uint16_t>(i);
    WeightPager::convert_half_payload(all.data(), all.size() * 2,
                                      Convert::kFp16ToBf16);
    for (uint32_t i = 0; i < all.size(); ++i) {
        const float f = ITensor::fp16_to_fp32(static_cast<uint16_t>(i));
        if (f != f) {
            EXPECT_EQ(all[i] & 0x7F80u, 0x7F80u) << "NaN stays NaN, " << i;
            EXPECT_NE(all[i] & 0x007Fu, 0u) << "NaN stays NaN, " << i;
            continue;
        }
        ASSERT_EQ(all[i], float_to_bf16(f)) << "fp16 bits " << i;
    }
}

TEST(WeightPagerBf16Test, Bf16ToFp16IsExactWhereFp16HoldsTheValue) {
    std::vector<uint16_t> w{float_to_bf16(1.5F), float_to_bf16(-2.0F),
                            float_to_bf16(1024.0F), float_to_bf16(0.0F)};
    WeightPager::convert_half_payload(w.data(), w.size() * 2,
                                      Convert::kBf16ToFp16);
    EXPECT_EQ(w[0], fp16_bits(1.5F));
    EXPECT_EQ(w[1], fp16_bits(-2.0F));
    EXPECT_EQ(w[2], fp16_bits(1024.0F));
    EXPECT_EQ(w[3], 0);
}

// The --act bf16 slot whose checkpoint is fp16: the buffer ends up holding the
// rounded bf16 bits, not the fp16 ones.
TEST(WeightPagerBf16Test, PageInRoundsAnFp16SourceIntoABf16Buffer) {
    auto rt = load_gather();
    auto pool = vkop::tests::TestEnv::get_command_pool();
    const auto vals = table_values();
    std::vector<uint16_t> src(32);
    for (int i = 0; i < 32; ++i) src[i] = fp16_bits(vals[i]);
    PagedSource s;
    s.name = "table";
    s.path = write_source("fp16", src, 24);
    s.offset = 24;
    s.size = 64;
    s.group = "0.table";
    s.src_dtype = "float16";
    WeightPager pager(rt, pool);
    pager.add(s);
    const auto st = pager.page_in("table");
    EXPECT_EQ(st.bytes, 64U);

    const auto got = gather_rows_3_1(rt);
    ASSERT_EQ(got.size(), 16U);
    for (int c = 0; c < 8; ++c) {
        EXPECT_EQ(got[c], float_to_bf16(vals[24 + c])) << "row 3 col " << c;
        EXPECT_EQ(got[8 + c], float_to_bf16(vals[8 + c])) << "row 1 col " << c;
    }
    EXPECT_EQ(got[8 + 3], 0x4780);
    EXPECT_EQ(got[8 + 5], 0x3F82);
    EXPECT_NE(got[8 + 1], src[8 + 1]) << "fp16 bits copied verbatim";
}

// The common MoE case: a bf16 checkpoint into a bf16 slot is a verbatim copy,
// recorded either as "bfloat16" or not at all.
TEST(WeightPagerBf16Test, PageInCopiesABf16SourceVerbatim) {
    for (const char *dtype : {"bfloat16", ""}) {
        auto rt = load_gather();
        auto pool = vkop::tests::TestEnv::get_command_pool();
        std::vector<uint16_t> src(32);
        // Bits no fp16 round trip preserves: past fp16's range and a NaN.
        for (int i = 0; i < 32; ++i)
            src[i] = static_cast<uint16_t>(0x4780u + 97u * i);
        src[25] = 0x7FC1;
        PagedSource s;
        s.name = "table";
        s.path = write_source(std::string("bf16_") + (*dtype ? "named" : "anon"),
                              src, 0);
        s.size = 64;
        s.group = "0.table";
        s.src_dtype = dtype;
        WeightPager pager(rt, pool);
        pager.add(s);
        pager.page_in("table");
        const auto got = gather_rows_3_1(rt);
        ASSERT_EQ(got.size(), 16U);
        for (int c = 0; c < 8; ++c) {
            EXPECT_EQ(got[c], src[24 + c]) << dtype << " row 3 col " << c;
            EXPECT_EQ(got[8 + c], src[8 + c]) << dtype << " row 1 col " << c;
        }
    }
}

// A recorded source format with no conversion is refused before any byte
// moves: the table keeps the values the model loaded with.
TEST(WeightPagerBf16Test, PageInRefusesAnUnconvertibleSource) {
    auto rt = load_gather();
    auto pool = vkop::tests::TestEnv::get_command_pool();
    std::vector<uint16_t> src(32, 0x4000);
    for (const char *dtype : {"float32", "not_a_dtype"}) {
        PagedSource s;
        s.name = "table";
        s.path = write_source("refused", src, 0);
        s.size = 64;
        s.group = "0.table";
        s.src_dtype = dtype;
        WeightPager pager(rt, pool);
        pager.add(s);
        EXPECT_THROW(pager.page_in("table"), std::runtime_error) << dtype;
    }
    const auto got = gather_rows_3_1(rt);
    ASSERT_EQ(got.size(), 16U);
    for (int c = 0; c < 8; ++c) {
        EXPECT_EQ(bf16_to_float(got[c]), float(3 * 8 + c));
        EXPECT_EQ(bf16_to_float(got[8 + c]), float(1 * 8 + c));
    }
}
