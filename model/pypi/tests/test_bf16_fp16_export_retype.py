"""--act bf16 on an export that declares FLOAT16 activations.

Every LLM export in the repo declares its activations FLOAT16, and the domain
pass used to rewrite FLOAT only, so `-q bf16 --act bf16` on GLM-Edge retyped
0 tensors. Now FLOAT16 is a candidate too, and (fp16_graph.bf16_partition)
an op with no bf16 kernel keeps the width the export gave it behind a Cast:
fp16 for an fp16 export, fp32 for an fp32 one.
"""
import contextlib
import io

import numpy
import pytest
from onnx import TensorProto as T, helper as h, numpy_helper as nh

from onnx2vkop import bf16_weights, numeric_audit
from onnx2vkop.dag import DAGBasedModel, Node
from onnx2vkop.fp16_graph import GraphFp16

F, F16, BF = T.FLOAT, T.FLOAT16, T.BFLOAT16
NAME = {F: "float32", F16: "float16", BF: "bfloat16"}


def ref(name, dtype=F16, shape=(4, 64)):
    return {"name": name, "shape": list(shape), "dtype": dtype}


def node(op, name, ins, outs, dtype=F16, **attrs):
    return Node(op, name, attrs,
                [ref(n, dtype) if isinstance(n, str) else n for n in ins],
                [ref(n, dtype) if isinstance(n, str) else n for n in outs])


def build(nodes, inputs=(), outputs=(), inits=None, dtype=F16):
    m = DAGBasedModel()
    for n in nodes:
        m.nodes[n.name] = n
    m.inputs = [{"name": n, "shape": [4, 64], "dtype": NAME[dtype]}
                if isinstance(n, str) else n for n in inputs]
    m.outputs = [{"name": n, "shape": [4, 64], "dtype": NAME[dtype]}
                 if isinstance(n, str) else n for n in outputs]
    for name, arr in (inits or {}).items():
        m.initializers[name] = nh.from_array(arr, name)
    return m


def bf16(m, **kw):
    with contextlib.redirect_stdout(io.StringIO()):
        return GraphFp16.apply(m, "buffer", half="bf16", **kw)


def decl(m, name):
    """Every declaration of `name` in the file (node entries and graph I/O),
    normalized to the elem int; asserts they agree."""
    seen = set()
    for n in m.nodes.values():
        for e in list(n.inputs) + list(n.outputs):
            if e["name"] == name:
                seen.add(e["dtype"])
    for e in list(m.inputs) + list(m.outputs):
        if e["name"] == name:
            seen.add({"float32": F, "float16": F16,
                      "bfloat16": BF}.get(e["dtype"], e["dtype"]))
    assert len(seen) == 1, (name, seen)
    return seen.pop()


def casts(m):
    """{(source, output): to} of every Cast in the graph."""
    return {(n.inputs[0]["name"], n.outputs[0]["name"]): n.attributes["to"]
            for n in m.nodes.values() if n.op_type == "Cast"}


def assert_topological(m):
    have = {e["name"] for e in m.inputs} | set(m.initializers)
    for n in m.nodes.values():
        for e in n.inputs:
            assert e["name"] in have, f"{n.name} reads {e['name']} before it exists"
        have |= {e["name"] for e in n.outputs}


W = numpy.linspace(-1, 1, 64 * 64, dtype=numpy.float16).reshape(64, 64)
G = numpy.linspace(0.5, 1.5, 64, dtype=numpy.float16)
B = numpy.linspace(-0.1, 0.1, 64, dtype=numpy.float16)


def llm_block():
    """fp16 export: RMSNorm -> MatMul -> AveragePool (no bf16 kernel) -> Add,
    a KV cache through ScatterND (no bf16 kernel), an attention bias,
    logits."""
    nodes = [
        node("RMSNorm", "norm", ["x", "g"], ["n"]),
        node("MatMul", "mm", ["n", "w"], ["y"]),
        node("AveragePool", "tr", ["y"], ["t"]),
        node("Add", "add", ["t", "b"], ["z"]),
        node("Add", "bias", ["z", "attention_bias"], ["zb"]),
        node("ScatterND", "kv", ["past_key_values_0",
                                 ref("kv_idx", T.INT64, (1, 1)), "zb"],
             ["present_key_values_0"]),
        node("MatMul", "head", ["zb", "w"], ["logits"]),
    ]
    return build(nodes,
                 inputs=["x", "attention_bias", "past_key_values_0"],
                 outputs=["logits", "present_key_values_0"],
                 inits={"w": W, "g": G, "b": B,
                        "kv_idx": numpy.zeros((1, 1), numpy.int64)})


def test_fp16_export_is_retyped_with_casts_at_the_islands():
    m = llm_block()
    s = bf16(m)
    assert not s["refused"], s["conflicts"]
    # The bf16 domain: allow-listed ops, their tensors and the graph inputs.
    for t in ("x", "n", "y", "z", "zb", "attention_bias",
              "past_key_values_0"):
        assert decl(m, t) == BF, t
    for e in m.inputs:
        assert e["dtype"] == "bfloat16", e
    # AveragePool keeps the export's fp16 behind a Cast pair; so does
    # ScatterND.
    assert decl(m, "t") == F16
    assert {"tr", "kv"} <= set(s["blockers"])
    c = casts(m)
    assert c[("y", "y__fp16")] == F16
    assert c[("t", "t__bf16")] == BF
    assert c[("past_key_values_0", "past_key_values_0__fp16")] == F16
    assert c[("zb", "zb__fp16")] == F16
    assert m.nodes["tr"].inputs[0]["name"] == "y__fp16"
    assert m.nodes["add"].inputs[0]["name"] == "t__bf16"
    # A graph output keeps its name: the producer writes <name>__<width> and
    # a Cast writes the output. present_key_values is bf16 like the input it
    # will be fed back into; logits stays fp16 (TopK has no bf16 kernel).
    assert m.nodes["kv"].outputs[0]["name"] == "present_key_values_0__fp16"
    assert c[("present_key_values_0__fp16", "present_key_values_0")] == BF
    assert decl(m, "present_key_values_0") == BF
    assert m.nodes["head"].outputs[0]["name"] == "logits__bf16"
    assert c[("logits__bf16", "logits")] == F16
    assert decl(m, "logits") == F16
    assert s["casts"] == len(c) == 6
    assert_topological(m)
    # Nothing binds bf16 beside another width (Casts excepted).
    assert GraphFp16.bf16_kind_conflicts(m) == []
    assert GraphFp16.mixed_half_kinds(m) == []


def test_fp16_weights_round_to_bf16_and_are_audited():
    m = llm_block()
    with numeric_audit.collecting() as audit:
        bf16(m)
    for name, src in (("w", W), ("g", G), ("b", B)):
        init = m.initializers[name]
        assert init.data_type == BF, name
        bits = numpy.frombuffer(init.raw_data, numpy.uint16).reshape(src.shape)
        want = bf16_weights.fp32_to_bf16_bits(src.astype(numpy.float32))
        assert (bits == want).all(), name
    rows = {r.tensor: r for r in audit.rows}
    assert {"w", "g", "b"} <= set(rows)
    assert all(rows[n].fmt == "bf16" for n in ("w", "g", "b"))
    # fp16 -> bf16 is NOT exact: 11 significant bits down to 8.
    assert rows["w"].max_abs > 0


def test_mover_follows_its_input():
    """A Reshape between two islands stays at their width: no Cast pair."""
    m = build([node("AveragePool", "t0", ["x"], ["a"]),
               node("Reshape", "r", ["a", ref("shape", T.INT64, (2,))], ["b"]),
               node("AveragePool", "t1", ["b"], ["c"]),
               node("Add", "add", ["c", "c"], ["d"])],
              inputs=["x"], outputs=["d"])
    m.initializers["shape"] = nh.from_array(numpy.array([4, 64]), "shape")
    bf16(m)
    assert decl(m, "a") == decl(m, "b") == decl(m, "c") == F16
    assert decl(m, "d") == BF
    assert set(casts(m)) == {("x", "x__fp16"), ("c", "c__bf16")}
    assert_topological(m)
    # ...and between two bf16 ops it is bf16.
    m = build([node("Add", "a0", ["x", "x"], ["a"]),
               node("Reshape", "r", ["a", ref("shape", T.INT64, (2,))], ["b"]),
               node("Mul", "m", ["b", "b"], ["c"])],
              inputs=["x"], outputs=["c"])
    m.initializers["shape"] = nh.from_array(numpy.array([4, 64]), "shape")
    bf16(m)
    assert decl(m, "b") == BF and casts(m) == {}


def test_fp32_export_island_stays_fp32_behind_a_cast():
    w = numpy.ones((64, 64), numpy.float32)
    m = build([node("MatMul", "mm", ["x", "w"], ["y"], dtype=F),
               node("AveragePool", "relu", ["y"], ["r"], dtype=F),
               node("Add", "add", ["r", "y"], ["z"], dtype=F)],
              inputs=["x"], outputs=["z"], inits={"w": w}, dtype=F)
    bf16(m)
    assert decl(m, "y") == BF and decl(m, "z") == BF
    assert decl(m, "r") == F
    assert casts(m) == {("y", "y__fp32"): F, ("r", "r__bf16"): BF}
    assert m.initializers["w"].data_type == BF


def test_initializer_read_by_both_sides_is_split():
    m = build([node("Add", "add", ["x", "b"], ["y"]),
               node("AveragePool", "relu", ["b"], ["r"]),
               node("Add", "add2", ["y", "r"], ["z"])],
              inputs=["x"], outputs=["z"], inits={"b": B})
    bf16(m)
    assert m.initializers["b"].data_type == F16       # the island's bytes
    assert m.initializers["b__bf16"].data_type == BF  # the bf16 reader's copy
    assert m.nodes["add"].inputs[1]["name"] == "b__bf16"
    assert m.nodes["relu"].inputs[0]["name"] == "b"


def test_export_casts_are_retargeted_or_dropped():
    """fp16 export with an fp32 stretch: Cast(to=FLOAT) -> AveragePool ->
    Cast(to=FLOAT16) -> Add. The AveragePool island keeps fp32; the second Cast
    feeds a bf16 Add, so it converts to bf16 instead. A Cast whose both
    sides became bf16 is dropped."""
    m = build([Node("Cast", "up", {"to": F}, [ref("x")], [ref("xf", F)]),
               node("AveragePool", "relu", ["xf"], ["rf"], dtype=F),
               Node("Cast", "down", {"to": F16}, [ref("rf", F)], [ref("r")]),
               node("Add", "add", ["r", "x"], ["z"]),
               Node("Cast", "noop", {"to": F}, [ref("z")], [ref("zf", F)]),
               node("Mul", "mul", ["zf", "zf"], ["out"], dtype=F)],
              inputs=["x"], outputs=[ref("out", "float32")])
    s = bf16(m)
    assert not s["refused"], s["conflicts"]
    assert decl(m, "xf") == F and decl(m, "rf") == F
    assert m.nodes["up"].attributes["to"] == F     # bf16 -> fp32
    assert m.nodes["down"].attributes["to"] == BF  # fp32 -> bf16
    assert decl(m, "r") == BF
    assert "noop" not in m.nodes
    assert m.nodes["mul"].inputs[0]["name"] == "z"
    assert s["dropped_casts"] == 1
    assert_topological(m)


def test_blocked_matmul_is_an_island_not_a_collapse():
    """A MatMul with no bf16 route (the down_act fold, a quantized B) used to
    hold every tensor it touched -- and through them the graph -- at fp32."""
    w = numpy.ones((64, 7), numpy.float16)
    m = build([node("Add", "a", ["x", "x"], ["y"]),
               Node("MatMul", "mm", {"down_act": 1},
                    [ref("y"), ref("w", F16, (64, 7))],
                    [ref("o", F16, (4, 7))]),
               node("Add", "a2", [ref("o", F16, (4, 7)),
                                  ref("o", F16, (4, 7))],
                    [ref("p", F16, (4, 7))])],
              inputs=["x"], outputs=[ref("p", "float16", (4, 7))],
              inits={"w": w})
    s = bf16(m)
    assert "down_act" in s["blockers"]["mm"]
    assert decl(m, "y") == BF and decl(m, "p") == BF
    assert decl(m, "o") == F16
    assert m.initializers["w"].data_type == F16
    assert set(casts(m)) == {("y", "y__fp16"), ("o", "o__bf16")}


def test_mixed_kind_in_one_op_is_still_refused():
    """The per-node refusal: an fp16 operand beside bf16 in one non-Cast op
    (here, what a later -q fp16 weight pass would leave)."""
    m = llm_block()
    bf16(m)
    assert GraphFp16.bf16_kind_conflicts(m) == []
    m.nodes["add"].inputs[1]["dtype"] = F16
    m.initializers["b"] = nh.from_array(B, "b")
    bad = GraphFp16.bf16_kind_conflicts(m)
    assert bad and "add" in bad[0], bad
    assert GraphFp16.mixed_half_kinds(m) == bad
    # A bf16 WEIGHT in its weight slot beside fp16 activations is legal.
    w = numpy.zeros((64, 64), numpy.uint16)
    m2 = build([node("MatMul", "mm", ["x", ref("w", BF, (64, 64))], ["y"])],
               inputs=["x"], outputs=["y"])
    m2.initializers["w"] = nh.from_array(w, "w")
    m2.initializers["w"].data_type = BF
    assert GraphFp16.bf16_kind_conflicts(m2) == []
