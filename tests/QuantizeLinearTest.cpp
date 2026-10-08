// Copyright 2026 @junka
#include "setup.hpp"
#include "core/Tensor.hpp"
#include "ops/OperatorFactory.hpp"
#include "ops/Ops.hpp"
#include <gtest/gtest.h>
#include <torch/torch.h>

using vkop::core::Tensor;
using vkop::core::as_tensor;
using vkop::tests::TestEnv;
namespace ops = vkop::ops;

// QuantizeLinear / DequantizeLinear GPU shader test (fp16 <-> E4M3 fp8).
//
// The op decides direction from the input's container type: a uint16_t (fp16)
// input means quantize, an int8_t (fp8) input means dequantize. The per-tensor
// fp32 scale arrives as the third input (a 1-element float tensor). We verify
// the round-trip (fp16 -> fp8 -> fp16) stays within E4M3's ~6% relative error
// and that the encoded bytes match torch's Float8_e4m3fn reference on the
// finite grid.

namespace {

template <typename T>
static void ql_upload(std::shared_ptr<Tensor<T>> t) {
    auto dev = TestEnv::get_device();
    auto cmdpool = TestEnv::get_command_pool();
    t->as_storage_buffer(dev);
    t->copyToGPU(cmdpool);
}

// Run one QuantizeLinear/DequantizeLinear pass. scale is the per-tensor fp32
// scale at ONNX inputs[1]; zero_point (if any) is the int8/uint8 byte at
// inputs[2]. fp8 has no zero_point (zp_t = nullptr).
static void run_quant(const std::shared_ptr<vkop::core::ITensor> &input,
                      const std::shared_ptr<vkop::core::ITensor> &output,
                      float scale,
                      std::shared_ptr<vkop::core::ITensor> zp_t = nullptr) {
    auto dev = TestEnv::get_device();
    auto cmdpool = TestEnv::get_command_pool();
    auto op = ops::create_from_type(ops::OpType::QUANTIZE_LINEAR, 0, 0, true);
    op->set_runtime_device(dev, cmdpool);

    // Scale input: a 1-element fp32 tensor at ONNX inputs[1].
    auto scale_t = std::make_shared<Tensor<float>>(std::vector<int>{1});
    scale_t->fillToCPU(std::vector<float>{scale});
    ql_upload(scale_t);

    // Upload the data input.
    if (input->dtype() == typeid(uint16_t)) {
        ql_upload(as_tensor<uint16_t>(input));
    } else if (input->dtype() == typeid(int8_t)) {
        ql_upload(as_tensor<int8_t>(input));
    } else if (input->dtype() == typeid(float)) {
        ql_upload(as_tensor<float>(input));
    }

    // Upload zero_point if present (int8 container; elem_kind set by caller).
    if (zp_t) {
        ql_upload(as_tensor<int8_t>(zp_t));
    }

    op->onExecute({input, scale_t, zp_t}, {output}, 0);
    auto cmd = op->get_record();
    std::vector<VkSubmitInfo> info{cmd->buildSubmitInfo()};
    vkop::VulkanCommandBuffer::submit(dev->getComputeQueue(), info);
    cmd->wait();
    dev->wait_all_done();

    if (output->dtype() == typeid(uint16_t)) {
        as_tensor<uint16_t>(output)->copyToCPU(cmdpool);
    } else if (output->dtype() == typeid(int8_t)) {
        as_tensor<int8_t>(output)->copyToCPU(cmdpool);
    } else if (output->dtype() == typeid(float)) {
        as_tensor<float>(output)->copyToCPU(cmdpool);
    }
}

static std::vector<uint16_t> ql_fp16_bits(const torch::Tensor &t) {
    auto cpu = t.cpu().contiguous().to(torch::kFloat16);
    const auto *p = reinterpret_cast<const uint16_t *>(cpu.data_ptr<at::Half>());
    return std::vector<uint16_t>(p, p + cpu.numel());
}

// A spread of values exercising normal and subnormal E4M3 ranges, plus the
// saturation boundary (448).
TEST(QuantizeLinearTest, RoundTripFp16Fp8Fp16) {
    auto vals = torch::tensor({0.0f,  1.0f,   -1.0f,   0.5f,    -0.5f,
                               2.0f,  -2.0f,  3.25f,   -3.25f,  0.001f,
                               100.0f, -100.0f, 448.0f, -448.0f, 0.015625f,
                               0.001953125f, 6.0f, -6.0f, 12.5f, -12.5f})
                    .to(torch::kFloat16);
    int n = static_cast<int>(vals.numel());
    float scale = 1.0f;  // unit scale: the bytes are the raw E4M3 codes

    // fp16 input.
    auto tin = std::make_shared<Tensor<uint16_t>>(std::vector<int>{n});
    tin->fillToCPU(ql_fp16_bits(vals));

    // fp8 output (quantize).
    auto tfp8 = std::make_shared<Tensor<int8_t>>(std::vector<int>{n}, true);
    tfp8->set_elem_kind(vkop::core::ElemKind::kFloat8E4M3FN);
    run_quant(tin, tfp8, scale);

    // Reference: torch's Float8_e4m3fn cast (unit scale = divide by 1.0).
    auto ref_f8 = vals.to(torch::kFloat32).clamp(-448.0f, 448.0f)
                      .to(torch::kFloat8_e4m3fn);
    const auto *ref_p = reinterpret_cast<const int8_t *>(ref_f8.data_ptr());
    for (int i = 0; i < n; ++i) {
        EXPECT_EQ((*tfp8)[i], ref_p[i])
            << "fp8 byte mismatch at " << i << " (val=" << vals[i].item<float>()
            << ")";
    }

    // fp8 -> fp16 (dequantize).
    auto tout = std::make_shared<Tensor<uint16_t>>(std::vector<int>{n}, true);
    run_quant(tfp8, tout, scale);

    // The round-trip recovers the E4M3 grid points; compare against torch's
    // own fp8->fp16 round-trip (its dequant grid).
    auto ref_rt = ref_f8.to(torch::kFloat16);
    const auto *ref_rt_p =
        reinterpret_cast<const uint16_t *>(ref_rt.data_ptr<at::Half>());
    for (int i = 0; i < n; ++i) {
        EXPECT_EQ((*tout)[i], ref_rt_p[i])
            << "round-trip fp16 mismatch at " << i;
    }
}

// A non-unit scale: scale=0.1 means the stored byte encodes val/0.1 (so a
// value of 10.0 -> 100.0 in fp8 units). Dequantize multiplies it back.
TEST(QuantizeLinearTest, ScaledRoundTrip) {
    float scale = 0.1f;
    auto vals = torch::tensor({0.01f, 0.1f, 1.0f, 10.0f, -10.0f, 44.8f, -44.8f})
                    .to(torch::kFloat16);
    int n = static_cast<int>(vals.numel());

    auto tin = std::make_shared<Tensor<uint16_t>>(std::vector<int>{n});
    tin->fillToCPU(ql_fp16_bits(vals));
    auto tfp8 = std::make_shared<Tensor<int8_t>>(std::vector<int>{n}, true);
    tfp8->set_elem_kind(vkop::core::ElemKind::kFloat8E4M3FN);
    run_quant(tin, tfp8, scale);

    auto tout = std::make_shared<Tensor<uint16_t>>(std::vector<int>{n}, true);
    run_quant(tfp8, tout, scale);

    // Each recovered value matches torch's own fp8 round-trip, applying the
    // SAME scale the shader does (quantize = val/scale, dequant = byte*scale).
    // The E4M3 grid is coarse: at val~10 the stored code is ~100, whose
    // nearest grid points are 96/104, an 8-unit gap, so 0.8*scale abs error is
    // expected. Comparing against torch's own round-trip isolates the shader's
    // encode/decode from fp8's inherent quantization step.
    auto scaled = (vals.to(torch::kFloat32) / scale).clamp(-448.0f, 448.0f);
    auto ref = scaled.to(torch::kFloat8_e4m3fn).to(torch::kFloat32) * scale;
    ref = ref.to(torch::kFloat16);
    const auto *ref_p =
        reinterpret_cast<const uint16_t *>(ref.data_ptr<at::Half>());
    for (int i = 0; i < n; ++i) {
        EXPECT_EQ((*tout)[i], ref_p[i])
            << "scaled round-trip byte mismatch at " << i
            << " (orig=" << vals[i].item<float>() << ")";
    }
}

// Odd element count exercises the word-packing OOB guards (4 vals/fp8 word,
// 2 vals/fp16 word; odd totals hit the boundary clauses).
TEST(QuantizeLinearTest, OddTotal) {
    auto vals = torch::tensor({1.0f, -2.0f, 3.0f, -4.0f, 5.0f}).to(torch::kFloat16);
    int n = 5;
    float scale = 1.0f;

    auto tin = std::make_shared<Tensor<uint16_t>>(std::vector<int>{n});
    tin->fillToCPU(ql_fp16_bits(vals));
    auto tfp8 = std::make_shared<Tensor<int8_t>>(std::vector<int>{n}, true);
    tfp8->set_elem_kind(vkop::core::ElemKind::kFloat8E4M3FN);
    run_quant(tin, tfp8, scale);

    auto tout = std::make_shared<Tensor<uint16_t>>(std::vector<int>{n}, true);
    run_quant(tfp8, tout, scale);

    auto ref_f8 = vals.to(torch::kFloat32).clamp(-448.0f, 448.0f)
                      .to(torch::kFloat8_e4m3fn).to(torch::kFloat16);
    const auto *ref_p =
        reinterpret_cast<const uint16_t *>(ref_f8.data_ptr<at::Half>());
    for (int i = 0; i < n; ++i) {
        EXPECT_EQ((*tout)[i], ref_p[i])
            << "odd-total round-trip mismatch at " << i;
    }
}

// ---- External QDQ (int8/uint8 + scale + zero_point) ----------------------
//
// These mirror what ORT's quantize_dynamic emits: a per-tensor scale and an
// asymmetric zero_point. The shader's encode is q = round(x/scale) + zp, clamp
// to the format range; decode is (q - zp) * scale. The reference is numpy's
// own asymmetric quantization (the exact grid ORT lands on), so a byte-exact
// match isolates the shader's encode/decode from int8's inherent step.

// Build a 1-element zero_point tensor (int8 container) carrying one byte.
// `zp_byte` is the raw byte value in [0,255] for uint8 / [-128,127] for int8.
static std::shared_ptr<vkop::core::ITensor>
ql_zp_tensor(int zp_byte, vkop::core::ElemKind kind) {
    auto t = std::make_shared<Tensor<int8_t>>(std::vector<int>{1});
    t->fillToCPU(std::vector<int8_t>{static_cast<int8_t>(zp_byte)});
    t->set_elem_kind(kind);
    return t;
}

// int8, symmetric zero_point (zp=0). Compares the recovered fp16 against
// torch's own int8 round-trip on the same grid.
TEST(QuantizeLinearTest, Int8AsymmetricRoundTrip) {
    auto vals = torch::tensor({0.0f, 1.0f, -1.0f, 0.5f, -0.5f, 2.0f, -2.0f,
                               10.0f, -10.0f, 100.0f, -100.0f, 0.25f, 7.5f,
                               -7.5f, 3.3f, -3.3f})
                    .to(torch::kFloat16);
    int n = static_cast<int>(vals.numel());
    float scale = 1.0f;     // one int8 unit per 1.0
    int zp = 0;             // symmetric (zp=0) for the int8 case

    auto tin = std::make_shared<Tensor<uint16_t>>(std::vector<int>{n});
    tin->fillToCPU(ql_fp16_bits(vals));
    auto tq = std::make_shared<Tensor<int8_t>>(std::vector<int>{n}, true);
    tq->set_elem_kind(vkop::core::ElemKind::kInt8);
    auto zp_t = ql_zp_tensor(zp, vkop::core::ElemKind::kInt8);
    run_quant(tin, tq, scale, zp_t);

    // Reference bytes: q = clamp(round(x/scale) + zp, -128, 127).
    auto x = vals.to(torch::kFloat32);
    auto q = torch::clamp(torch::round(x / scale) + float(zp), -128.0f, 127.0f)
                 .to(torch::kInt8);
    const auto *q_p = q.data_ptr<int8_t>();
    for (int i = 0; i < n; ++i) {
        EXPECT_EQ((*tq)[i], q_p[i])
            << "int8 encode byte mismatch at " << i
            << " (val=" << vals[i].item<float>() << ")";
    }

    // Dequant back to fp16 and compare against torch's own round-trip.
    auto tout = std::make_shared<Tensor<uint16_t>>(std::vector<int>{n}, true);
    run_quant(tq, tout, scale, zp_t);
    auto ref = (q.to(torch::kFloat32) - float(zp)) * scale;
    ref = ref.to(torch::kFloat16);
    const auto *ref_p =
        reinterpret_cast<const uint16_t *>(ref.data_ptr<at::Half>());
    for (int i = 0; i < n; ++i) {
        EXPECT_EQ((*tout)[i], ref_p[i])
            << "int8 dequant byte mismatch at " << i;
    }
}

// uint8, asymmetric zero_point (the common ORT dynamic-quantize output for
// weights: q in [0,255], zp typically non-zero). zp is carried as a plain int
// in [0,255] so the reference math does not overflow int8_t.
TEST(QuantizeLinearTest, UInt8AsymmetricRoundTrip) {
    auto vals = torch::tensor({0.0f, 1.0f, 2.0f, 5.0f, 10.0f, 50.0f, 100.0f,
                               200.0f, 250.0f, 0.5f, 7.25f, 13.0f, 99.0f,
                               150.0f, 3.7f, 255.0f})
                    .to(torch::kFloat16);
    int n = static_cast<int>(vals.numel());
    float scale = 1.0f;
    int zp = 128;  // asymmetric: shift so [0,255] covers signed range

    auto tin = std::make_shared<Tensor<uint16_t>>(std::vector<int>{n});
    tin->fillToCPU(ql_fp16_bits(vals));
    auto tq = std::make_shared<Tensor<int8_t>>(std::vector<int>{n}, true);
    tq->set_elem_kind(vkop::core::ElemKind::kUint8);
    auto zp_t = ql_zp_tensor(zp, vkop::core::ElemKind::kUint8);
    run_quant(tin, tq, scale, zp_t);

    // Reference: q = clamp(round(x/scale) + zp, 0, 255), stored as uint8 byte.
    auto x = vals.to(torch::kFloat32);
    auto q = torch::clamp(torch::round(x / scale) + float(zp), 0.0f, 255.0f)
                 .to(torch::kUInt8);
    const auto *q_p = q.data_ptr<uint8_t>();
    for (int i = 0; i < n; ++i) {
        EXPECT_EQ(static_cast<uint8_t>((*tq)[i]), q_p[i])
            << "uint8 encode byte mismatch at " << i
            << " (val=" << vals[i].item<float>() << ")";
    }

    auto tout = std::make_shared<Tensor<uint16_t>>(std::vector<int>{n}, true);
    run_quant(tq, tout, scale, zp_t);
    auto ref = (q.to(torch::kFloat32) - float(zp)) * scale;
    ref = ref.to(torch::kFloat16);
    const auto *ref_p =
        reinterpret_cast<const uint16_t *>(ref.data_ptr<at::Half>());
    for (int i = 0; i < n; ++i) {
        EXPECT_EQ((*tout)[i], ref_p[i])
            << "uint8 dequant byte mismatch at " << i;
    }
}

// A non-unit scale with asymmetric uint8 — the realistic ORT dynamic case:
// scale picks the grid spacing, zp shifts it. Byte-exact vs torch.
TEST(QuantizeLinearTest, UInt8ScaledAsymmetric) {
    auto vals = torch::tensor({0.1f, 0.5f, 1.0f, 2.0f, 5.0f, 10.0f, 20.0f,
                               -1.0f, -5.0f, 0.0f, 0.05f, 15.0f})
                    .to(torch::kFloat16);
    int n = static_cast<int>(vals.numel());
    float scale = 0.1f;    // grid step 0.1
    int zp = 100;          // asymmetric shift

    auto tin = std::make_shared<Tensor<uint16_t>>(std::vector<int>{n});
    tin->fillToCPU(ql_fp16_bits(vals));
    auto tq = std::make_shared<Tensor<int8_t>>(std::vector<int>{n}, true);
    tq->set_elem_kind(vkop::core::ElemKind::kUint8);
    auto zp_t = ql_zp_tensor(zp, vkop::core::ElemKind::kUint8);
    run_quant(tin, tq, scale, zp_t);

    auto x = vals.to(torch::kFloat32);
    auto q = torch::clamp(torch::round(x / scale) + float(zp), 0.0f, 255.0f)
                 .to(torch::kUInt8);
    const auto *q_p = q.data_ptr<uint8_t>();
    for (int i = 0; i < n; ++i) {
        EXPECT_EQ(static_cast<uint8_t>((*tq)[i]), q_p[i])
            << "uint8 scaled encode mismatch at " << i
            << " (val=" << vals[i].item<float>() << ")";
    }

    auto tout = std::make_shared<Tensor<uint16_t>>(std::vector<int>{n}, true);
    run_quant(tq, tout, scale, zp_t);
    auto ref = (q.to(torch::kFloat32) - float(zp)) * scale;
    ref = ref.to(torch::kFloat16);
    const auto *ref_p =
        reinterpret_cast<const uint16_t *>(ref.data_ptr<at::Half>());
    for (int i = 0; i < n; ++i) {
        EXPECT_EQ((*tout)[i], ref_p[i])
            << "uint8 scaled dequant mismatch at " << i;
    }
}

} // namespace

