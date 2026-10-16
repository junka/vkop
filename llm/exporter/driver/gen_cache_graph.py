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
import sys
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

H = int(sys.argv[2]) if len(sys.argv) > 2 else 16
HKV = int(sys.argv[3]) if len(sys.argv) > 3 else 16
D = int(sys.argv[4]) if len(sys.argv) > 4 else 128
S = "S"      # q rows
KV = "kv"    # cached rows
KVT = "kvT"  # cached + fresh

F16 = TensorProto.FLOAT16

init = [
    numpy_helper.from_array(np.array(0, dtype=np.int64), "idx_k"),
    numpy_helper.from_array(np.array(1, dtype=np.int64), "idx_v"),
    numpy_helper.from_array(np.array(1, dtype=np.int64), "u1"),
    numpy_helper.from_array(np.array(D ** -0.5, dtype=np.float16), "scale"),
]


def vi(name, dims):
    return helper.make_tensor_value_info(name, F16, dims)


n = [
    helper.make_node("Gather", ["past_key_values_0", "idx_k"], ["k_hist"], axis=1),
    helper.make_node("Gather", ["past_key_values_0", "idx_v"], ["v_hist"], axis=1),
    helper.make_node("Concat", ["k_hist", "k_new"], ["k_full"], axis=2),
    helper.make_node("Concat", ["v_hist", "v_new"], ["v_full"], axis=2),
    helper.make_node("Unsqueeze", ["k_full", "u1"], ["k_unsq"]),
    helper.make_node("Unsqueeze", ["v_full", "u1"], ["v_unsq"]),
    helper.make_node("Concat", ["k_unsq", "v_unsq"], ["present_key_values_0"],
                     axis=1),
    helper.make_node("Transpose", ["k_full"], ["k_full_t"], perm=[0, 1, 3, 2]),
    helper.make_node("MatMul", ["q", "k_full_t"], ["qk"]),
    helper.make_node("Mul", ["qk", "scale"], ["qk_s"]),
    helper.make_node("Add", ["qk_s", "attention_bias"], ["qk_b"]),
    helper.make_node("Softmax", ["qk_b"], ["p"], axis=-1),
    helper.make_node("MatMul", ["p", "v_full"], ["y"]),
]

graph = helper.make_graph(
    n, "cache_block",
    [vi("q", [1, H, S, D]),
     vi("k_new", [1, HKV, S, D]),
     vi("v_new", [1, HKV, S, D]),
     vi("attention_bias", [1, 1, S, KVT]),
     vi("past_key_values_0", [1, 2, HKV, KV, D])],
    [vi("y", [1, H, S, D]),
     vi("present_key_values_0", [1, 2, HKV, KVT, D])],
    initializer=init,
)
m = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
m.ir_version = 8
onnx.checker.check_model(m)
out = sys.argv[1] if len(sys.argv) > 1 else "/tmp/cache_block.onnx"
onnx.save(m, out)
print(f"wrote {out}  H={H} Hkv={HKV} D={D}")
