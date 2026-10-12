"""Manual weight-only QDQ injection (no ORT quantizer).

ORT's quantize_static on an fp16 model is broken for our purpose: it emits
empty-valued activation scales (calibration fails on fp16) and quantizes
activations too, which is NOT our external-QDQ design. quantize_dynamic emits
QOperator (MatMulInteger + DynamicQuantizeLinear) — also wrong (runtime
activation quant, which vkop doesn't implement).

Our external-QDQ design is strictly weight-only:
  - each MatMul/Gemm with a const-B (initializer) weight gets its weight
    replaced by  uint8_byte, fp32_scale, uint8_zp  and wrapped with a
    DequantizeLinear node that expands the uint8 back to fp16 in-graph.
  - activations stay fp16/fp32 (untouched), feeding the ordinary MatMul.
  - Phase 1 correctness path: DQ -> fp16 -> ordinary MatMul (no VRAM save
    yet; that's Phase 2, folding DQ->MatMul into weight-only MatMul).

This is exactly what the vkop QuantizeLinear/DequantizeLinear op expects
(scale at inputs[1], zp at inputs[2], fp32 scale, uint8 zp for uint8 fmt),
and it's what ORT-on-QDQ will run as ground truth.

Quantization: per-tensor asymmetric uint8.
  q   = clamp(round(x / scale) + zp, 0, 255)
  scale = (max(x) - min(x)) / 255
  zp    = round(-min(x) / scale)           # so min maps to 0
  decode = (q - zp) * scale

Output: llm.qdq.onnx + llm.qdq.weights.bin (external data).
"""

import os
import numpy as np
import onnx
from onnx import helper, TensorProto

FP16 = "llm.onnx"
QDQ = "llm.qdq.onnx"


def quantize_weight_u8(w):
    """Per-tensor asymmetric uint8 quantization of an fp16 array.
    Returns (uint8_bytes, fp32_scale, uint8_zp)."""
    x = w.astype(np.float32)
    xmin = float(x.min())
    xmax = float(x.max())
    # Guard against degenerate (constant) weights.
    if xmax == xmin:
        scale = 1.0
        zp = 0
        q = np.zeros_like(x, dtype=np.uint8)
        return q, np.float32(scale), np.uint8(zp)
    scale = (xmax - xmin) / 255.0
    zp = int(round(-xmin / scale))
    # Clamp zp into [0, 255] (it should already be, for well-behaved ranges).
    zp = max(0, min(255, zp))
    q = np.round(x / scale) + zp
    q = np.clip(q, 0, 255).astype(np.uint8)
    return q, np.float32(scale), np.uint8(zp)


def insert_weight_only_qdq(model):
    g = model.graph
    init_by_name = {t.name: t for t in g.initializer}

    # Find MatMul/Gemm ops whose B (input[1]) is a const initializer.
    targets = []
    for node in g.node:
        if node.op_type not in ("MatMul", "Gemm"):
            continue
        if len(node.input) < 2:
            continue
        b_name = node.input[1]
        if b_name not in init_by_name:
            continue
        b_init = init_by_name[b_name]
        # Only quantize float weights (fp16 or fp32).
        if b_init.data_type not in (TensorProto.FLOAT, TensorProto.FLOAT16):
            continue
        targets.append((node, b_name, b_init))

    print(f"[qdq] {len(targets)} MatMul/Gemm weight tensors to quantize")

    # For each weight: emit (uint8_q, fp32 scale, uint8 zp) initializers + a
    # DequantizeLinear node. ONNX DequantizeLinear requires scale to be fp32
    # (fp16 scale is rejected by ORT as invalid). The DQ output element type
    # follows the scale (fp32); for an fp16 consumer MatMul we insert a
    # Cast(fp32->fp16) between DQ and the MatMul so the weight dtype matches
    # what the consumer originally saw. vkop reads the fp32 scale directly
    # (as_tensor<float>) and has a Cast op, so this graph is valid for both.
    n_quant = 0
    n_cast = 0
    total_orig_bytes = 0
    total_q_bytes = 0
    for node, b_name, b_init in targets:
        w = onnx.numpy_helper.to_array(b_init).astype(np.float32)
        q, scale, zp = quantize_weight_u8(w)
        total_orig_bytes += w.nbytes
        total_q_bytes += q.nbytes + 4 + 1

        base = b_name
        q_name = f"{base}__u8"
        s_name = f"{base}__scale"
        z_name = f"{base}__zp"
        dq_out = f"{base}__dq"        # fp32
        is_fp16 = (b_init.data_type == TensorProto.FLOAT16)
        # The tensor feeding the MatMul B input (after optional Cast).
        feed = dq_out

        q_init = helper.make_tensor(q_name, TensorProto.UINT8,
                                    list(q.shape), q.tobytes(), raw=True)
        s_init = helper.make_tensor(s_name, TensorProto.FLOAT, [],
                                    [float(scale)])
        z_init = helper.make_tensor(z_name, TensorProto.UINT8, [],
                                    [int(zp)])
        g.initializer.extend([q_init, s_init, z_init])

        # DQ output is fp32 (scale type).
        dq_vi = helper.make_tensor_value_info(dq_out, TensorProto.FLOAT,
                                              list(q.shape))
        g.value_info.append(dq_vi)
        dq_node = helper.make_node(
            "DequantizeLinear",
            inputs=[q_name, s_name, z_name],
            outputs=[dq_out],
            name=f"{base}__DequantizeLinear",
        )
        g.node.append(dq_node)

        if is_fp16:
            # Cast fp32 -> fp16 to match the original weight dtype the MatMul
            # consumed.
            cast_out = f"{base}__castfp16"
            cast_vi = helper.make_tensor_value_info(
                cast_out, TensorProto.FLOAT16, list(q.shape))
            g.value_info.append(cast_vi)
            cast_node = helper.make_node(
                "Cast",
                inputs=[dq_out],
                outputs=[cast_out],
                name=f"{base}__Cast_fp16",
                to=TensorProto.FLOAT16,
            )
            g.node.append(cast_node)
            feed = cast_out
            n_cast += 1

        node.input[1] = feed
        n_quant += 1

    # Remove the now-unused original weight initializers.
    used = set()
    for node in g.node:
        used.update(node.input)
    removed = 0
    for t in list(g.initializer):
        if t.name not in used and not (t.name.endswith("__u8")
                                       or t.name.endswith("__scale")
                                       or t.name.endswith("__zp")):
            g.initializer.remove(t)
            removed += 1
    print(f"[qdq] quantized {n_quant} weights ({n_cast} Cast fp32->fp16 added), "
          f"removed {removed} now-unused original initializers")
    print(f"[qdq] approx weight storage: "
          f"{total_orig_bytes/1e6:.1f}MB fp32 -> "
          f"{total_q_bytes/1e6:.1f}MB (uint8+scales)")
    return n_quant


def main():
    print(f"[load] {FP16}")
    m = onnx.load(FP16, load_external_data=True)
    insert_weight_only_qdq(m)
    # Save with external data (uint8 weights are still ~hundreds of MB).
    for f in os.listdir("."):
        if f == QDQ or f.startswith("llm.qdq."):
            if f != "llm.weights.bin":  # keep the fp16 source weights
                os.remove(f)
    onnx.save_model(m, QDQ, save_as_external_data=True,
                    all_tensors_to_one_file=True,
                    location="llm.qdq.weights.bin", convert_attribute=True)
    print(f"[ok] saved {QDQ} ({os.path.getsize(QDQ)/1e6:.1f}MB) + "
          f"llm.qdq.weights.bin ({os.path.getsize('llm.qdq.weights.bin')/1e6:.1f}MB)")
    # Verify node counts.
    from collections import Counter
    c = Counter(n.op_type for n in m.graph.node)
    print(f"[ok] node counts: MatMul={c.get('MatMul',0)} "
          f"Gemm={c.get('Gemm',0)} DequantizeLinear={c.get('DequantizeLinear',0)} "
          f"QuantizeLinear={c.get('QuantizeLinear',0)} "
          f"Cast={c.get('Cast',0)} "
          f"DynamicQuantizeLinear={c.get('DynamicQuantizeLinear',0)} "
          f"MatMulInteger={c.get('MatMulInteger',0)}")


if __name__ == "__main__":
    main()
