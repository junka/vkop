// Copyright 2026 @junka
#ifndef OPS_QUANTIZE_LINEAR_HPP_
#define OPS_QUANTIZE_LINEAR_HPP_

#include "core/DType.hpp"
#include "ops/BufferBase.hpp"
#include <cstdio>
#include <cstdlib>

extern "C" {
extern unsigned char buffer_quantize_linear_spv[];
extern unsigned int buffer_quantize_linear_spv_len;
}

namespace vkop {
namespace ops {

namespace quant {

// Push constant for the combined QuantizeLinear/DequantizeLinear shader.
//
//   fmt        which byte format the quantized side holds:
//                0 = fp8 E4M3, 1 = fp8 E5M2, 2 = int8, 3 = uint8
//   mode       0 = quantize (float -> bytes), 1 = dequantize (bytes -> float)
//   total      element count (== byte count on the quantized side)
//   scale      per-tensor fp32 scale (read from inputs[1])
//   zp         per-tensor zero_point as a plain int (read from inputs[2]).
//              fp8 has no zero_point (zp stays 0); int8/uint8 are asymmetric in
//              general (ORT dynamic quantize emits a non-zero uint8 zp).
//   out_fp32   dequant only: 0 = fp16 output, 1 = fp32 output. The float side
//              of quantize follows inputs[0]'s dtype the same way.
//   in_fp32    quant only: 0 = fp16 input,  1 = fp32 input.
struct alignas(16) QuantPC {
    int mode;     // 0 = quantize, 1 = dequantize
    int total;    // element count
    float scale;  // per-tensor fp32 scale
    int zp;       // per-tensor zero_point (int)
    int fmt;      // 0=e4m3, 1=e5m2, 2=int8, 3=uint8
    int out_fp32; // dequant: output fp32?
    int in_fp32;  // quant: input fp32?
    int _pad;
};

} // namespace quant

// SSBO op: ONNX QuantizeLinear / DequantizeLinear.
//
// Two use sites share this one op + shader
// (shaders/buffer/quantize_linear.comp):
//
//   1. fp8 KV cache (internal). The LLM stores each layer's
//      past/present_key_values as E4M3 fp8 bytes to halve KV-cache VRAM;
//      attention computes in fp16, so the graph wraps the cache boundary with
//      QuantizeLinear (fp16 -> fp8) and DequantizeLinear (fp8 -> fp16). The fp8
//      E4M3 field layout matches matmul.comp's e4m3_val exactly.
//
//   2. External QDQ int8/uint8 quantization. An ONNX model whose graph already
//      carries QuantizeLinear/DequantizeLinear nodes (e.g. produced by ORT
//      quantize_dynamic) is imported as-is: the converter is node-passthrough,
//      so these nodes reach the runtime, which runs them here. int8/uint8 use
//      scale + zero_point (asymmetric); the dequantized float output feeds an
//      ordinary fp16/fp32 MatMul.
//
// ONNX input contract (both Q and DQ): (x, scale, [zero_point]).
//   - scale is inputs[1], a 1-element fp32 tensor (per-tensor; per-axis is a
//     later phase). zero_point is inputs[2], same shape, int8/uint8 for the
//     int8/uint8 formats and absent for fp8.
//   - NOTE: an earlier version read scale from inputs[2], which is ONNX's
//     zero_point slot. That happened to work for the fp8 KV cache (the cache
//     graph omits zero_point, so inputs[2] was empty and the op fell back to
//     scale=1.0, and the unit test fed scale at inputs[2] to mask it). The
//     int8/uint8 path cannot tolerate that: zero_point is real and non-zero,
//     so scale MUST come from inputs[1].
//
// Direction is decided from inputs[0]'s elem_kind: a float input means
// quantize, a byte input (int8/uint8/fp8) means dequantize. The byte format is
// decided from the quantized-side tensor's elem_kind (the output for quantize,
// the input for dequant).
class QuantizeLinear : public BufferFactory {
  public:
    QuantizeLinear()
        : BufferFactory(OpType::QUANTIZE_LINEAR, buffer_quantize_linear_spv,
                        buffer_quantize_linear_spv_len,
                        {DESCRIPTOR_TYPE_STORAGE, DESCRIPTOR_TYPE_STORAGE},
                        sizeof(quant::QuantPC)) {}

    void setAttribute(const std::unordered_map<std::string, std::string>
                          &attributes) override {
        // ONNX QuantizeLinear/DequantizeLinear have no attributes that affect
        // this op (per-axis `axis` is a later phase). Direction and format come
        // from input/output dtypes at execute() time.
        (void)attributes;
    }

  private:
    void execute(
        const std::vector<std::shared_ptr<core::ITensor>> &inputs,
        const std::vector<std::shared_ptr<core::ITensor>> &outputs) override {
        if (inputs.empty() || outputs.empty())
            return;

        auto shape = inputs[0]->getShape();
        int total = total_elems(shape);

        const core::ElemKind in_kind = inputs[0]->elem_kind();
        const bool dequant = !core::is_float_elem(in_kind);
        // The quantized-side elem_kind: output for quantize, input for dequant.
        const core::ElemKind q_kind =
            dequant ? in_kind : outputs[0]->elem_kind();

        int fmt = 0;
        switch (q_kind) {
        case core::ElemKind::kFloat8E4M3FN:
            fmt = 0;
            break;
        case core::ElemKind::kFloat8E5M2:
            fmt = 1;
            break;
        case core::ElemKind::kInt8:
            fmt = 2;
            break;
        case core::ElemKind::kUint8:
            fmt = 3;
            break;
        default:
            throw std::runtime_error(
                std::string("vkop: QuantizeLinear unsupported quantized format "
                            "'") +
                core::elem_name(q_kind) + "'");
        }

        // Per-tensor scale from inputs[1] (ONNX: x, scale, [zero_point]).
        float scale = 1.0f;
        if (inputs.size() >= 2 && inputs[1]) {
            auto s = core::as_tensor<float>(inputs[1]);
            if (s && s->num_elements() > 0) {
                s->copyToCPU(m_cmdpool_);
                scale = (*s)[0];
            }
        }
        if (scale == 0.0f)
            scale = 1.0f; // a zero scale is meaningless; guard

        // Per-tensor zero_point from inputs[2]. fp8 has none (stays 0);
        // int8/uint8 read one byte and reinterpret per format.
        int zp = 0;
        if (inputs.size() >= 3 && inputs[2]) {
            auto z = core::as_tensor<int8_t>(inputs[2]);
            if (z && z->num_elements() > 0) {
                z->copyToCPU(m_cmdpool_);
                const int8_t raw = (*z)[0];
                zp = (q_kind == core::ElemKind::kUint8)
                         ? int(static_cast<uint8_t>(raw))
                         : int(raw);
            }
        }

        // The float side's dtype (fp16 vs fp32). For quantize it is the input;
        // for dequant the output. uint16_t container == fp16, float == fp32.
        const bool float_fp32 = dequant ? (outputs[0]->dtype() == typeid(float))
                                        : (inputs[0]->dtype() == typeid(float));

        // Resize + bind. The quantized side is always int8_t storage (fp8 and
        // uint8 ride it the same way int8 does); the float side is uint16_t
        // (fp16) or float (fp32).
        if (dequant) {
            if (float_fp32) {
                auto out = core::as_tensor<float>(outputs[0]);
                if (out->num_elements() != total)
                    out->resize(shape);
                bind_ssbo<float>(outputs[0], /*is_output=*/true);
            } else {
                auto out = core::as_tensor<uint16_t>(outputs[0]);
                if (out->num_elements() != total)
                    out->resize(shape);
                bind_ssbo<uint16_t>(outputs[0], /*is_output=*/true);
            }
            bind_ssbo<int8_t>(inputs[0], /*is_output=*/false);
        } else {
            auto out = core::as_tensor<int8_t>(outputs[0]);
            if (out->num_elements() != total)
                out->resize(shape);
            bind_ssbo<int8_t>(outputs[0], /*is_output=*/true);
            if (float_fp32) {
                bind_ssbo<float>(inputs[0], /*is_output=*/false);
            } else {
                bind_ssbo<uint16_t>(inputs[0], /*is_output=*/false);
            }
        }

        quant::QuantPC pc{};
        pc.mode = dequant ? 1 : 0;
        pc.total = total;
        pc.scale = scale;
        pc.zp = zp;
        pc.fmt = fmt;
        pc.out_fp32 = dequant ? (float_fp32 ? 1 : 0) : 0;
        pc.in_fp32 = dequant ? 0 : (float_fp32 ? 1 : 0);
        pc.in_fp32 = dequant ? 0 : (float_fp32 ? 1 : 0);
        // One thread per 4-element word on the byte side: int8/uint8/fp8 all
        // pack 4 bytes/word. The float side is fp16 (2 vals/word, so 4 values
        // = 2 fp16 words) or fp32 (1 val/word, so 4 values = 4 fp32 words).
        // The shader handles the byte:float word mapping per fmt.
        submit(&pc, UP_DIV((total + 3) / 4, 256), 1, 1);
    }
};

} // namespace ops
} // namespace vkop
#endif // OPS_QUANTIZE_LINEAR_HPP_
