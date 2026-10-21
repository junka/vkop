#!/usr/bin/env python3
"""Generate a tiny, runnable ONNX carrying the KV-cache rotation around an
attention block — the smallest graph that exercises fuse_into_attention +
fuse_kv_cache_append and can actually be *run*.

No weights: Q/K/V arrive as graph inputs (post-RoPE), so the file is a few KB
and converting it never touches the 3.4 GB llm.onnx. The point is not to model
a decoder but to reproduce the decoder's attention+cache *shape*:

    past_key_values_0 [1,2,Hkv,cap,D]
      ├ Gather(axis=1,0) -> K history ┐  Concat(axis=2, ., k_new) -> k_full
      ├ Gather(axis=1,1) -> V history ┘  Concat(axis=2, ., v_new) -> v_full
      MatMul(q, k_full, transB=1)*scale + bias -> Softmax -> MatMul(., v_full)
      Unsqueeze(k_full,1), Unsqueeze(v_full,1) -> Concat(axis=1) -> present

That is the 7-node rotation the fold deletes, so
VKOP_NO_FUSIONS=fuse_kv_cache_append on the converted binary is a true A/B of
"rotation" vs "cache form" over one dispatch schedule.

MatMul takes K at [1,Hkv,kv,D], so H == Hkv here (rep=1). The kernel's GQA path
(rep>1) is exercised by the op-level BufferRankTest case instead — putting a
repeat_kv in this graph would mean a dynamic Reshape target, which loses the
static head_dim the attention matcher requires.
"""
import os
import sys
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

H = int(sys.argv[2]) if len(sys.argv) > 2 else 16
HKV = int(sys.argv[3]) if len(sys.argv) > 3 else 16
D = int(sys.argv[4]) if len(sys.argv) > 4 else 128
# VKOP_RING=1 declares the sliding-window ring's slot-of-row-0 input. The
# converter wires it into the fused Attention (has_ring=1) when it sees it, so
# this is the switch that makes the probe actually exercise the ring.
RING = os.environ.get("VKOP_RING", "") not in ("", "0")
# VKOP_KV_FP8=1 builds the fp8 variant of the same block: past/present are
# E4M3 bytes and a Q/DQ pair wraps the boundary, exactly as the real exporter
# emits it. The converter is supposed to fold that pair into the Attention
# (cache_scale), which is the only way an fp8 cache can be appended in place —
# and therefore the only way it can take the ring.
FP8 = os.environ.get("VKOP_KV_FP8", "") not in ("", "0")
KV_SCALE = float(os.environ.get("VKOP_KV_SCALE", "0.1"))
S = "S"      # q rows
KV = "kv"    # cached rows
KVT = "kvT"  # cached + fresh

F16 = TensorProto.FLOAT16
PRES = "present_key_values_0"

init = [
    numpy_helper.from_array(np.array(0, dtype=np.int64), "idx_k"),
    numpy_helper.from_array(np.array(1, dtype=np.int64), "idx_v"),
    numpy_helper.from_array(np.array(1, dtype=np.int64), "u1"),
    numpy_helper.from_array(np.array(D ** -0.5, dtype=np.float16), "scale"),
]
if FP8:
    init.append(numpy_helper.from_array(
        np.array(KV_SCALE, dtype=np.float32), "kv_cache_fp8_scale"))


def vi(name, dims, dt=F16):
    return helper.make_tensor_value_info(name, dt, dims)


# The two Gathers read the history; with fp8 that is the DQ's fp16 output, and
# the graph input they would otherwise read is the fp8 cache itself.
past_ref = "past_fp16" if FP8 else "past_key_values_0"

n = []
if FP8:
    # The DQ has to lead: the Gathers below read its output, and the checker
    # enforces topological order.
    n.append(helper.make_node(
        "DequantizeLinear", ["past_key_values_0", "kv_cache_fp8_scale"],
        ["past_fp16"], name="DequantizeLinear_kv_0"))
n += [
    helper.make_node("Gather", [past_ref, "idx_k"], ["k_hist"], axis=1),
    helper.make_node("Gather", [past_ref, "idx_v"], ["v_hist"], axis=1),
    helper.make_node("Concat", ["k_hist", "k_new"], ["k_full"], axis=2),
    helper.make_node("Concat", ["v_hist", "v_new"], ["v_full"], axis=2),
    helper.make_node("Unsqueeze", ["k_full", "u1"], ["k_unsq"]),
    helper.make_node("Unsqueeze", ["v_full", "u1"], ["v_unsq"]),
    helper.make_node("Concat", ["k_unsq", "v_unsq"],
                     ["present_fp16" if FP8 else PRES], axis=1),
    helper.make_node("Transpose", ["k_full"], ["k_full_t"], perm=[0, 1, 3, 2]),
    helper.make_node("MatMul", ["q", "k_full_t"], ["qk"]),
    helper.make_node("Mul", ["qk", "scale"], ["qk_s"]),
    helper.make_node("Add", ["qk_s", "attention_bias"], ["qk_b"]),
    helper.make_node("Softmax", ["qk_b"], ["p"], axis=-1),
    helper.make_node("MatMul", ["p", "v_full"], ["y"]),
]
if FP8:
    n.append(helper.make_node(
        "QuantizeLinear", ["present_fp16", "kv_cache_fp8_scale"], [PRES],
        name="QuantizeLinear_kv_0"))

graph_inputs = [vi("q", [1, H, S, D]),
                vi("k_new", [1, HKV, S, D]),
                vi("v_new", [1, HKV, S, D]),
                vi("attention_bias", [1, 1, S, KVT]),
                vi("past_key_values_0", [1, 2, HKV, KV, D],
                   TensorProto.FLOAT8E4M3FN if FP8 else F16)]
if RING:
    graph_inputs.append(
        helper.make_tensor_value_info("kv_ring_head", TensorProto.INT32, [1]))

graph = helper.make_graph(
    n, "cache_block", graph_inputs,
    [vi("y", [1, H, S, D]),
     vi(PRES, [1, 2, HKV, KVT, D],
        TensorProto.FLOAT8E4M3FN if FP8 else F16)],
    initializer=init,
    # The DQ bridge must be declared fp16 explicitly, exactly as the real
    # exporter's kv_fp8_rewrite.py does: with no value_info, ONNX types a
    # DequantizeLinear output float32 (its default with no output_dtype
    # attribute) and the fp32 word-movers then reinterprets fp16 and fp8 bytes
    # as 32-bit words. In the folded graph those nodes are deleted so this
    # declaration is dead — but in the VKOP_NO_FUSIONS ablation the DQ stays,
    # and without it the fp8 bytes reach the concat as fp32, K/V no longer
    # match Q's width, and the Attention op refuses to bind (it used to
    # dereference the null buffer; see ops/Attention.hpp).
    value_info=[vi("past_fp16", [1, 2, HKV, KV, D]),
                vi("present_fp16", [1, 2, HKV, KVT, D])] if FP8 else [],
)
m = helper.make_model(graph,
                      opset_imports=[helper.make_opsetid("", 19 if FP8 else 17)])
m.ir_version = 9 if FP8 else 8
onnx.checker.check_model(m)
out = sys.argv[1] if len(sys.argv) > 1 else "/tmp/cache_block.onnx"
onnx.save(m, out)
print(f"wrote {out}  H={H} Hkv={HKV} D={D} ring={RING} fp8={FP8}")
