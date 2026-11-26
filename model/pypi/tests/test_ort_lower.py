"""ort_lower: the fused vkop graph, lowered back to ONNX ORT can run.

Two layers of evidence:

  * per-op: a DAG holding one vkop node (the form a fusion pass emits) is
    lowered and run in ORT against a numpy model of what the kernel computes
    (ops/*.hpp). This pins the lowering's reading of each vkop attribute.
  * end to end: an ONNX graph goes through the real converter up to the end
    of fusion, is lowered, and ORT(source) must equal ORT(lowered) on every
    tensor name both expose. This is the check the tool exists for -- a pass
    that changed the math shows up here without a GPU.

onnxruntime is not a converter dependency; the module skips without it.
"""
import contextlib
import io
import os

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

ort = pytest.importorskip("onnxruntime")

from onnx2vkop.dag import DAGBasedModel, Node  # noqa: E402
from onnx2vkop.ort_lower import (LoweringError, lower_dag,  # noqa: E402
                                 upcast_model_to_fp32)

F32 = TensorProto.FLOAT
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
RNG = np.random.default_rng(1234)


def td(name, shape, dtype=F32):
    return {"name": name, "shape": list(shape), "dtype": dtype}


def make_dag(nodes, inputs, outputs, inits=None):
    dag = DAGBasedModel()
    dag.inputs = list(inputs)
    dag.outputs = list(outputs)
    for name, arr in (inits or {}).items():
        dag.initializers[name] = numpy_helper.from_array(arr, name)
    for n in nodes:
        dag.add_node(n)
    dag.onnx_opset = 17
    return dag


def run_lowered(dag, feeds):
    model = lower_dag(dag)
    so = ort.SessionOptions()
    so.log_severity_level = 3
    sess = ort.InferenceSession(model.SerializeToString(), so,
                                providers=["CPUExecutionProvider"])
    names = [o.name for o in sess.get_outputs()]
    return dict(zip(names, sess.run(None, feeds)))


def rnd(*shape):
    return RNG.standard_normal(shape).astype(np.float32)


def close(a, b, tol=1e-5):
    np.testing.assert_allclose(a, b, rtol=tol, atol=tol)


# ----------------------------------------------------------------- per-op

def test_rmsnorm():
    x, w = rnd(2, 5, 16), rnd(16)
    dag = make_dag([Node("RMSNorm", "rn", {"eps": 1e-6, "axis": -1},
                         [td("x", x.shape), td("w", w.shape)],
                         [td("y", x.shape)])],
                   [td("x", x.shape)], [td("y", x.shape)], {"w": w})
    y = run_lowered(dag, {"x": x})["y"]
    ref = x / np.sqrt((x * x).mean(-1, keepdims=True) + 1e-6) * w
    close(y, ref)


def test_reduce():
    x = rnd(3, 4, 5)
    dag = make_dag([Node("Reduce", "r", {"reduce_op": "sum_square",
                                         "axes": [1], "keepdims": 0},
                         [td("x", x.shape)], [td("y", [3, 5])])],
                   [td("x", x.shape)], [td("y", [3, 5])])
    close(run_lowered(dag, {"x": x})["y"], (x * x).sum(1))


def test_fused_elemwise_program():
    # out = silu(a) * b + 0.5  ->  words: SILU r2=a ; MUL r3=r2*b ; ADD r0=r3+s0
    a, b = rnd(2, 8), rnd(8)
    ops = [14, 2, 0, 0, 3, 3, 2, 1, 1, 0, 3, 100]
    dag = make_dag([Node("FusedElemwise", "fe",
                         {"ops": ops, "scalars": [0.5], "input_shapes": [],
                          "out_shape": [2, 8], "rank": 2},
                         [td("a", a.shape), td("b", b.shape)],
                         [td("y", a.shape)])],
                   [td("a", a.shape), td("b", b.shape)], [td("y", a.shape)])
    ref = a / (1 + np.exp(-a)) * b + 0.5
    close(run_lowered(dag, {"a": a, "b": b})["y"], ref)


def rope_ref(x, cos, sin):
    """x [..., S, D] with cos/sin broadcastable; rotate_half on the first
    cos.shape[-1] dims, the rest passes through."""
    rd = cos.shape[-1]
    xr, xp = x[..., :rd], x[..., rd:]
    rot = np.concatenate([-xr[..., rd // 2:], xr[..., :rd // 2]], -1)
    return np.concatenate([xr * cos + rot * sin, xp], -1)


@pytest.mark.parametrize("rd", [8, 6])
def test_rotary_default_and_partial(rd):
    B, H, S, D = 1, 3, 4, 8
    x = rnd(B, H, S, D)
    cos, sin = rnd(B, S, rd), rnd(B, S, rd)       # Unsqueeze folded away
    dag = make_dag([Node("RotaryEmbedding", "re", {},
                         [td("x", x.shape), td("c", cos.shape),
                          td("s", sin.shape)], [td("y", x.shape)])],
                   [td("x", x.shape), td("c", cos.shape), td("s", sin.shape)],
                   [td("y", x.shape)])
    y = run_lowered(dag, {"x": x, "c": cos, "s": sin})["y"]
    close(y, rope_ref(x, cos[:, None], sin[:, None]))


def test_rotary_untransposed():
    B, S, H, D = 1, 5, 2, 8
    x = rnd(B, S, H, D)
    cos, sin = rnd(B, 1, S, D), rnd(B, 1, S, D)
    dag = make_dag([Node("RotaryEmbedding", "re", {"input_untransposed": 1},
                         [td("x", x.shape), td("c", cos.shape),
                          td("s", sin.shape)], [td("y", [B, H, S, D])])],
                   [td("x", x.shape), td("c", cos.shape), td("s", sin.shape)],
                   [td("y", [B, H, S, D])])
    y = run_lowered(dag, {"x": x, "c": cos, "s": sin})["y"]
    close(y, rope_ref(x.transpose(0, 2, 1, 3), cos, sin))


def test_rotary_component_seq_major():
    S, C, H, D = 6, 3, 2, 8
    x = rnd(S, C, H, D)
    cos, sin = rnd(S, D), rnd(S, D)
    dag = make_dag([Node("RotaryEmbedding", "re", {"input_component": 1},
                         [td("x", x.shape), td("c", cos.shape),
                          td("s", sin.shape)], [td("y", [S, H, D])])],
                   [td("x", x.shape), td("c", cos.shape), td("s", sin.shape)],
                   [td("y", [S, H, D])])
    y = run_lowered(dag, {"x": x, "c": cos, "s": sin})["y"]
    close(y, rope_ref(x[:, 1], cos[:, None], sin[:, None]))


def softmax(x):
    e = np.exp(x - x.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


def attn_ref(q, k, v, scale, mask=None, causal=False):
    rep = q.shape[1] // k.shape[1]
    k, v = np.repeat(k, rep, 1), np.repeat(v, rep, 1)
    s = q @ k.transpose(0, 1, 3, 2) * scale
    if mask is not None:
        s = s + mask
    if causal:
        S, KV = s.shape[-2:]
        s = np.where(np.arange(KV)[None] <= (KV - S) + np.arange(S)[:, None],
                     s, -np.inf)
    return softmax(s) @ v


def test_attention_gqa_mask_scale():
    B, H, Hkv, S, KV, D = 1, 4, 2, 3, 5, 8
    q, k, v = rnd(B, H, S, D), rnd(B, Hkv, KV, D), rnd(B, Hkv, KV, D)
    mask = rnd(B, 1, S, KV)
    dag = make_dag([Node("Attention", "a",
                         {"num_heads": H, "num_kv_heads": Hkv, "causal": 0,
                          "scale": 0.3, "mask_causal": 1},
                         [td("q", q.shape), td("k", k.shape), td("v", v.shape),
                          td("m", mask.shape)], [td("y", q.shape)])],
                   [td(n, a.shape) for n, a in
                    (("q", q), ("k", k), ("v", v), ("m", mask))],
                   [td("y", q.shape)])
    y = run_lowered(dag, {"q": q, "k": k, "v": v, "m": mask})["y"]
    close(y, attn_ref(q, k, v, 0.3, mask), 1e-4)


def test_attention_causal():
    B, H, S, KV, D = 1, 2, 3, 6, 4
    q, k, v = rnd(B, H, S, D), rnd(B, H, KV, D), rnd(B, H, KV, D)
    dag = make_dag([Node("Attention", "a",
                         {"num_heads": H, "num_kv_heads": H, "causal": 1},
                         [td("q", q.shape), td("k", k.shape), td("v", v.shape)],
                         [td("y", q.shape)])],
                   [td(n, a.shape) for n, a in (("q", q), ("k", k), ("v", v))],
                   [td("y", q.shape)])
    y = run_lowered(dag, {"q": q, "k": k, "v": v})["y"]
    close(y, attn_ref(q, k, v, 1.0, causal=True), 1e-4)


def test_attention_kv_cache_present():
    H, Hkv, S, P, D = 4, 2, 2, 3, 8
    q, kn, vn = rnd(1, H, S, D), rnd(1, Hkv, S, D), rnd(1, Hkv, S, D)
    past = rnd(1, 2, Hkv, P, D)
    dag = make_dag([Node("Attention", "a",
                         {"num_heads": H, "num_kv_heads": Hkv, "causal": 0,
                          "has_cache": 1},
                         [td("q", q.shape), td("kn", kn.shape),
                          td("vn", vn.shape), td("past", past.shape)],
                         [td("y", q.shape), td("present", [1, 2, Hkv, P + S, D])])],
                   [td(n, a.shape) for n, a in
                    (("q", q), ("kn", kn), ("vn", vn), ("past", past))],
                   [td("y", q.shape), td("present", [1, 2, Hkv, P + S, D])])
    r = run_lowered(dag, {"q": q, "kn": kn, "vn": vn, "past": past})
    k = np.concatenate([past[:, 0], kn], 2)
    v = np.concatenate([past[:, 1], vn], 2)
    close(r["present"], np.stack([k, v], 1))
    close(r["y"], attn_ref(q, k, v, 1.0), 1e-4)


def test_attention_seq_major_rank3_v_component():
    S, C, H, D = 5, 3, 2, 8
    q, k = rnd(S, H, D), rnd(S, H, D)
    qkv = rnd(S, C, H, D)
    dag = make_dag([Node("Attention", "a",
                         {"num_heads": H, "num_kv_heads": H, "causal": 0,
                          "seq_major": 1, "v_component": 2, "scale": 0.25},
                         [td("q", q.shape), td("k", k.shape),
                          td("qkv", qkv.shape)], [td("y", [1, S, H, D])])],
                   [td(n, a.shape) for n, a in
                    (("q", q), ("k", k), ("qkv", qkv))],
                   [td("y", [1, S, H, D])])
    y = run_lowered(dag, {"q": q, "k": k, "qkv": qkv})["y"]
    t = lambda a: a.transpose(1, 0, 2)[None]           # [S,H,D] -> [1,H,S,D]
    ref = attn_ref(t(q), t(k), t(qkv[:, 2]), 0.25).transpose(0, 2, 1, 3)
    close(y, ref, 1e-4)


def test_matmul_transb_and_gqa_b():
    a, b = rnd(1, 4, 3, 8), rnd(1, 2, 5, 8)           # B has 2 kv heads, transB
    dag = make_dag([Node("MatMul", "mm", {"transB": 1},
                         [td("a", a.shape), td("b", b.shape)],
                         [td("y", [1, 4, 3, 5])])],
                   [td("a", a.shape), td("b", b.shape)], [td("y", [1, 4, 3, 5])])
    y = run_lowered(dag, {"a": a, "b": b})["y"]
    close(y, a @ np.repeat(b, 2, 1).transpose(0, 1, 3, 2), 1e-4)


def test_matmul_swiglu_interleaved_weight():
    K, N = 8, 12
    x, g, u = rnd(2, K), rnd(K, N), rnd(K, N)
    gu = np.concatenate([g.reshape(K, N // 4, 4), u.reshape(K, N // 4, 4)],
                        axis=2).reshape(K, 2 * N)      # fold_swiglu_gate_up
    dag = make_dag([Node("MatMul", "mm", {"swiglu": [1]},
                         [td("x", x.shape), td("w", gu.shape)],
                         [td("y", [2, N])])],
                   [td("x", x.shape)], [td("y", [2, N])], {"w": gu})
    xg = x @ g
    close(run_lowered(dag, {"x": x})["y"], xg / (1 + np.exp(-xg)) * (x @ u), 1e-4)


def test_matmul_down_act_lane_major():
    K, N, O = 8, 12, 6                                 # N_up 12 -> padded 16
    x, g, u, d = rnd(2, K), rnd(K, N), rnd(K, N), rnd(N, O)
    pad = (N + 7) // 8 * 8

    def lane(w):
        f = np.zeros((K, pad), np.float32)
        f[:, :N] = w
        return f.reshape(K, pad // 8, 8)
    dag = make_dag([Node("MatMul", "mm", {"down_act": [1]},
                         [td("x", x.shape), td("wd", d.shape),
                          td("wg", [K, pad // 8, 8]), td("wu", [K, pad // 8, 8])],
                         [td("y", [2, O])])],
                   [td("x", x.shape)], [td("y", [2, O])],
                   {"wd": d, "wg": lane(g), "wu": lane(u)})
    xg = x @ g
    ref = (xg / (1 + np.exp(-xg)) * (x @ u)) @ d
    close(run_lowered(dag, {"x": x})["y"], ref, 1e-4)


@pytest.mark.parametrize("trans_b", [0, 1])
def test_matmul_int8_weight_scale(trans_b):
    K, N = 8, 6
    q = RNG.integers(-127, 128, size=(N, K) if trans_b else (K, N)).astype(np.int8)
    s = (RNG.random(N) * 0.05).astype(np.float32)
    x = rnd(3, K)
    dag = make_dag([Node("MatMul", "mm", {"transB": trans_b},
                         [td("x", x.shape), td("q", q.shape, TensorProto.INT8),
                          td("s", s.shape)], [td("y", [3, N])])],
                   [td("x", x.shape)], [td("y", [3, N])], {"q": q, "s": s})
    w = q.astype(np.float32) * (s[:, None] if trans_b else s[None])
    ref = x @ (w.T if trans_b else w)
    close(run_lowered(dag, {"x": x})["y"], ref, 1e-4)


def test_matmul_epilogue_scale_bias_softmax():
    a, b, bias = rnd(1, 2, 3, 4), rnd(1, 2, 4, 6), rnd(1, 1, 3, 6)
    dag = make_dag([Node("MatMul", "mm",
                         {"epi_scale": 0.5, "epi_bias": 1, "attn_softmax": 1},
                         [td("a", a.shape), td("b", b.shape),
                          td("bias", bias.shape)], [td("y", [1, 2, 3, 6])])],
                   [td(n, v.shape) for n, v in
                    (("a", a), ("b", b), ("bias", bias))],
                   [td("y", [1, 2, 3, 6])])
    y = run_lowered(dag, {"a": a, "b": b, "bias": bias})["y"]
    close(y, softmax((a @ b) * 0.5 + bias), 1e-5)


def test_conv_swish_activation():
    x, w = rnd(1, 2, 5, 5), rnd(3, 2, 3, 3)
    dag = make_dag([Node("Conv", "c",
                         {"activation": "Swish", "pads": [1, 1, 1, 1],
                          "weight_pack": 1, "weight_transpose": 1},
                         [td("x", x.shape), td("w", w.shape)],
                         [td("y", [1, 3, 5, 5])])],
                   [td("x", x.shape)], [td("y", [1, 3, 5, 5])], {"w": w})
    y = run_lowered(dag, {"x": x})["y"]
    sess = ort.InferenceSession(helper.make_model(helper.make_graph(
        [helper.make_node("Conv", ["x", "w"], ["y"], pads=[1, 1, 1, 1])], "g",
        [helper.make_tensor_value_info("x", F32, x.shape)],
        [helper.make_tensor_value_info("y", F32, None)],
        [numpy_helper.from_array(w, "w")]),
        opset_imports=[helper.make_opsetid("", 17)], ir_version=10
    ).SerializeToString(),
        providers=["CPUExecutionProvider"])
    c = sess.run(None, {"x": x})[0]
    close(y, c / (1 + np.exp(-c)), 1e-5)


def test_softmax_nan_and_layernorm_skip():
    x, skip = rnd(2, 6), rnd(2, 6)
    x[0, :] = np.nan
    sc, bi = rnd(6), rnd(6)
    dag = make_dag([
        Node("Softmax", "sm", {"axis": -1, "nan_optimization": True,
                               "nan_replacement_value": 0.0},
             [td("x", x.shape)], [td("p", x.shape)]),
        Node("LayerNormalization", "ln", {"axis": -1, "epsilon": 1e-5},
             [td("x2", x.shape), td("sc", [6]), td("bi", [6]), td("skip", x.shape)],
             [td("y", x.shape)]),
    ], [td("x", x.shape), td("x2", x.shape), td("skip", x.shape)],
        [td("p", x.shape), td("y", x.shape)], {"sc": sc, "bi": bi})
    x2 = rnd(2, 6)
    r = run_lowered(dag, {"x": x, "x2": x2, "skip": skip})
    p = softmax(np.nan_to_num(x, nan=0.0))
    p[0] = 0.0
    close(r["p"], p)
    h = x2 + skip
    ref = (h - h.mean(-1, keepdims=True)) / np.sqrt(h.var(-1, keepdims=True) + 1e-5)
    close(r["y"], ref * sc + bi, 1e-4)


def test_aliased_unsqueeze_gets_its_rank_back():
    """fold_unsqueeze deletes the Unsqueeze and leaves the consumer reading the
    rank-2 tensor under its rank-3 dict; the lowering restores the rank."""
    a, b = rnd(4, 6), rnd(1, 5, 6)
    dag = make_dag([Node("Relu", "r", {}, [td("a", a.shape)], [td("ra", a.shape)]),
                    Node("Add", "add", {}, [td("ra", [4, 1, 6]), td("b", b.shape)],
                         [td("y", [4, 5, 6])])],
                   [td("a", a.shape), td("b", b.shape)], [td("y", [4, 5, 6])])
    y = run_lowered(dag, {"a": a, "b": b})["y"]
    close(y, np.maximum(a, 0)[:, None, :] + b)


def test_unknown_semantic_attribute_is_refused():
    dag = make_dag([Node("Relu", "r", {"some_vkop_flag": 1},
                         [td("a", [2])], [td("y", [2])])],
                   [td("a", [2])], [td("y", [2])])
    with pytest.raises(LoweringError, match="some_vkop_flag"):
        lower_dag(dag)


def test_ops_without_ort_counterpart_are_refused():
    dag = make_dag([Node("LinearAttention", "la", {},
                         [td("a", [2])], [td("y", [2])])],
                   [td("a", [2])], [td("y", [2])])
    with pytest.raises(LoweringError, match="LinearAttention"):
        lower_dag(dag)


# ---------------------------------------------------- end to end (fusion)

def test_kv_cache_fold_is_numerically_neutral():
    """fuse_kv_cache_append on test_kv_cache_fusion's HF fragment: the graph
    before the fold and after it, both lowered, give the same `out` and
    `present` -- the fold's soundness checked numerically, not structurally."""
    import copy

    from onnx2vkop.optimizer import FusionOptimizer
    from test_kv_cache_fusion import build

    before = build()
    after = copy.deepcopy(before)
    ms = FusionOptimizer.match_kv_cache_append(after)
    assert len(ms) == 1 and FusionOptimizer.fold_kv_cache_append(after, ms[0])
    assert after.nodes["ATT"].attributes["has_cache"] == 1
    S = 2
    feeds = {"q": rnd(1, 16, S, 128), "k_new": rnd(1, 8, S, 128),
             "v_new": rnd(1, 8, S, 128), "past_key_values_0": rnd(1, 2, 8, 1, 128),
             "past_key_values_1": rnd(1, 2, 8, 1, 128),
             "attention_bias": rnd(1, 1, 1, 1)}
    r0 = run_lowered(before, feeds)
    r1 = run_lowered(after, feeds)
    for name in ("out", "present_key_values_0"):
        close(r1[name], r0[name], 1e-5)


def fused_dag(path):
    from onnx2vkop.converter import ModelConverter
    c = ModelConverter()
    with contextlib.redirect_stdout(io.StringIO()):
        dag = c.parse_onnx_model(path, 1, quant="fp32")
        c.initializer_merger.move_input_tensor_to_attr(dag)
        c.initializer_merger.merge_initializers(dag)
        c.initializer_merger.convert_flat_to_reshape(dag)
        c.initializer_merger.remove_redundant_reshape(dag)
        pre = {n.outputs[0]["name"]: n for n in dag.nodes.values()}
        c.fusion_optimizer.optimize(dag)
        c._cleanup_orphan_inputs(dag, pre)
        dag.build_dependencies()
    return dag


def ort_all(model, feeds, names=None):
    m = onnx.ModelProto()
    m.CopyFrom(model)
    have = {o.name for o in m.graph.output}
    for n in m.graph.node:
        for o in n.output:
            if o and o not in have and (names is None or o in names):
                have.add(o)
                m.graph.output.append(onnx.ValueInfoProto(name=o))
    so = ort.SessionOptions()
    so.log_severity_level = 3
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    s = ort.InferenceSession(m.SerializeToString(), so,
                             providers=["CPUExecutionProvider"])
    ins = {i.name for i in s.get_inputs()}
    return dict(zip([o.name for o in s.get_outputs()],
                    s.run(None, {k: v for k, v in feeds.items() if k in ins})))


def assert_fusion_preserved(src_path, feeds, expect_ops=(), tol=1e-4):
    dag = fused_dag(src_path)
    ops = {n.op_type for n in dag.nodes.values()}
    for op in expect_ops:
        assert op in ops, f"fusion did not produce {op}; got {sorted(ops)}"
    low = lower_dag(dag, base_dir=os.path.dirname(src_path))
    r_low = ort_all(low, feeds)
    src = upcast_model_to_fp32(onnx.load(src_path))
    r_src = ort_all(src, feeds, names=set(r_low))
    common = [n for n in r_low if n in r_src]
    assert common, "no tensor name survived on both sides"
    for n in common:
        a, b = np.asarray(r_src[n], np.float64), np.asarray(r_low[n], np.float64)
        if a.size != b.size:
            raise AssertionError(f"{n}: shape {a.shape} vs {b.shape}")
        if not a.size:
            continue
        b = b.reshape(a.shape)
        scale = max(np.nanmax(np.abs(a)), 1e-6)
        err = np.nanmax(np.abs(a - b)) / scale
        assert err < tol, f"{n}: rel err {err:.3g}"
    return len(common)


def save(tmp_path, nodes, inputs, outputs, inits, name="m.onnx", opset=17):
    g = helper.make_graph(nodes, "g", inputs, outputs,
                          [numpy_helper.from_array(v, k) for k, v in inits.items()])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", opset)])
    m.ir_version = 10
    m = onnx.shape_inference.infer_shapes(m)
    p = str(tmp_path / name)
    onnx.save(m, p)
    return p


def vi(name, shape, et=F32):
    return helper.make_tensor_value_info(name, et, shape)


def test_e2e_rmsnorm_rope_swiglu(tmp_path):
    """The HF decoder-block shapes fuse into RMSNorm / RotaryEmbedding /
    swiglu MatMul / FusedElemwise, and the lowered graph matches the source."""
    B, H, S, D, K, N = 1, 2, 4, 8, 16, 24
    nodes = [
        # RMSNorm, HF form (fp32 island)
        helper.make_node("Pow", ["x", "two"], ["x2"]),
        helper.make_node("ReduceMean", ["x2"], ["ms"], axes=[-1]),
        helper.make_node("Add", ["ms", "eps"], ["mse"]),
        helper.make_node("Sqrt", ["mse"], ["rms"]),
        helper.make_node("Div", ["one", "rms"], ["inv"]),
        helper.make_node("Mul", ["x", "inv"], ["xn"]),
        helper.make_node("Mul", ["xn", "gamma"], ["h"]),
        # swiglu MLP
        helper.make_node("MatMul", ["h", "wg"], ["g"]),
        helper.make_node("Sigmoid", ["g"], ["sg"]),
        helper.make_node("Mul", ["g", "sg"], ["silu"]),
        helper.make_node("MatMul", ["h", "wu"], ["u"]),
        helper.make_node("Mul", ["silu", "u"], ["act"]),
        helper.make_node("MatMul", ["act", "wd"], ["mlp"]),
        # rotate_half RoPE on q [B,H,S,D]
        helper.make_node("Slice", ["q", "s0", "sh", "ax"], ["q1"]),
        helper.make_node("Slice", ["q", "sh", "send", "ax"], ["q2"]),
        helper.make_node("Neg", ["q2"], ["nq2"]),
        helper.make_node("Concat", ["nq2", "q1"], ["rot"], axis=-1),
        helper.make_node("Mul", ["q", "cos"], ["qc"]),
        helper.make_node("Mul", ["rot", "sin"], ["rs"]),
        helper.make_node("Add", ["qc", "rs"], ["qr"]),
    ]
    inits = {
        "two": np.array(2.0, np.float32), "eps": np.array(1e-6, np.float32),
        "one": np.array(1.0, np.float32), "gamma": rnd(K),
        "wg": rnd(K, N), "wu": rnd(K, N), "wd": rnd(N, K),
        "s0": np.array([0], np.int64), "sh": np.array([D // 2], np.int64),
        "send": np.array([D], np.int64), "ax": np.array([-1], np.int64),
    }
    p = save(tmp_path, nodes,
             [vi("x", [B, S, K]), vi("q", [B, H, S, D]),
              vi("cos", [B, 1, S, D]), vi("sin", [B, 1, S, D])],
             [vi("mlp", [B, S, K]), vi("qr", [B, H, S, D])], inits)
    feeds = {"x": rnd(B, S, K), "q": rnd(B, H, S, D),
             "cos": rnd(B, 1, S, D), "sin": rnd(B, 1, S, D)}
    assert_fusion_preserved(p, feeds, expect_ops=("RotaryEmbedding",))


def test_e2e_conv_swish_and_softmax_nan(tmp_path):
    nodes = [
        helper.make_node("Conv", ["x", "w", "b"], ["c"], pads=[1, 1, 1, 1]),
        helper.make_node("Sigmoid", ["c"], ["sc"]),
        helper.make_node("Mul", ["c", "sc"], ["y"]),
        helper.make_node("Softmax", ["s"], ["p"], axis=-1),
        helper.make_node("IsNaN", ["p"], ["isn"]),
        helper.make_node("Where", ["isn", "zero", "p"], ["pz"]),
    ]
    p = save(tmp_path, nodes, [vi("x", [1, 2, 6, 6]), vi("s", [2, 7])],
             [vi("y", [1, 4, 6, 6]), vi("pz", [2, 7])],
             {"w": rnd(4, 2, 3, 3), "b": rnd(4), "zero": np.array(0.0, np.float32)})
    assert_fusion_preserved(p, {"x": rnd(1, 2, 6, 6), "s": rnd(2, 7)})


@pytest.mark.parametrize("rel", [
    "onnx_models/resnet18.onnx",
    "image/exporter/dit_prefill_tiny_static.onnx",
    "image/exporter/dit_decode_tiny_static.onnx",
])
def test_e2e_local_models(rel):
    """Real exports, when the checkout has them (not in git)."""
    path = os.path.join(REPO, rel)
    if not os.path.exists(path):
        pytest.skip(f"{rel} not present")
    m = onnx.load(path, load_external_data=False)
    feeds = {}
    rng = np.random.default_rng(0)
    for i in m.graph.input:
        shp = [d.dim_value or 1 for d in i.type.tensor_type.shape.dim]
        et = i.type.tensor_type.elem_type
        feeds[i.name] = np.zeros(shp, np.int64) if et == TensorProto.INT64 \
            else (rng.standard_normal(shp) * 0.5).astype(np.float32)
    assert_fusion_preserved(path, feeds)
