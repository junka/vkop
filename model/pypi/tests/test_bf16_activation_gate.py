"""--act bf16 admits only the ops whose `_bf16` kernel really decodes bf16.

The runtime's allow-list (core/Bf16Gate.hpp) refuses a bf16 activation on any
other op at LoadModel. The converter mirrors it (fp16_graph.BF16_ACT_OPS) so a
--act bf16 conversion keeps those ops -- and every tensor they touch -- FP32
instead of shipping a graph the runtime will refuse, and it refuses outright a
graph that would mix fp16 and bf16, which no vkop kernel can run.
"""
import contextlib
import io
import os
import subprocess
import sys

import numpy
import onnx
import pytest
from onnx import TensorProto as T, helper as h, numpy_helper as nh

from onnx2vkop.dag import DAGBasedModel, Node
from onnx2vkop.fp16_graph import BF16_ACT_OPS, GraphFp16, _BFLOAT16

F, F16 = T.FLOAT, T.FLOAT16
PYPI = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def ref(name, dtype=F, shape=(-1,)):
    return {"name": name, "shape": list(shape), "dtype": dtype}


def node(op, name, ins, outs, **attrs):
    return Node(op, name, attrs,
                [ref(n) if isinstance(n, str) else n for n in ins],
                [ref(n) if isinstance(n, str) else n for n in outs])


def build(nodes, inputs=(), outputs=(), initializers=None):
    m = DAGBasedModel()
    for n in nodes:
        m.nodes[n.name] = n
    m.inputs = [{"name": n, "shape": [4, 64], "dtype": "float32"}
                for n in inputs]
    m.outputs = [{"name": n, "shape": [4, 64], "dtype": "float32"}
                 for n in outputs]
    for name, arr in (initializers or {}).items():
        m.initializers[name] = nh.from_array(arr, name)
    return m


def bf16(m, **kw):
    with contextlib.redirect_stdout(io.StringIO()):
        return GraphFp16.apply(m, "buffer", half="bf16", **kw)


def dtype_of(m, name):
    for n in m.nodes.values():
        for e in list(n.inputs) + list(n.outputs):
            if e["name"] == name:
                return e.get("dtype")
    return None


W = numpy.ones((64, 64), dtype=numpy.float32)
B = numpy.ones(64, dtype=numpy.float32)


def test_allow_list_matches_the_runtime():
    assert BF16_ACT_OPS == {
        "Add", "Sub", "Mul", "Div", "Pow", "Min", "Max", "Mod", "Split",
        "LayerNormalization", "LayerNorm", "RMSNorm", "Softmax", "MatMul",
        "Attention"}


def test_allow_listed_chain_goes_bf16():
    m = build([node("MatMul", "mm", ["x", "w"], ["y"]),
               node("Add", "add", ["y", "b"], ["z"]),
               node("Softmax", "sm", ["z"], ["p"], axis=-1)],
              inputs=["x"], outputs=["p"], initializers={"w": W, "b": B})
    s = bf16(m)
    assert not s["refused"]
    for t in ("x", "y", "z", "p"):
        assert dtype_of(m, t) == _BFLOAT16, t
    assert m.initializers["w"].data_type == _BFLOAT16
    assert m.outputs[0]["dtype"] == "bfloat16"


@pytest.mark.parametrize("op", ["Relu", "Transpose", "Expand",
                                "FusedElemwise", "Gather", "Reduce"])
def test_gated_op_keeps_its_whole_island_fp32(op):
    """A gated op holds its tensors FP32, and that propagates through every
    allow-listed op that shares one: one shader per node, no Cast to bridge."""
    m = build([node("MatMul", "mm", ["x", "w"], ["y"]),
               node(op, "gated", ["y"], ["z"])],
              inputs=["x"], outputs=["z"], initializers={"w": W})
    s = bf16(m)
    assert "gated" in s["blockers"]
    for t in ("x", "y", "z"):
        assert dtype_of(m, t) == F, (op, t, dtype_of(m, t))
    assert m.initializers["w"].data_type == F


def test_the_same_gated_op_is_fine_under_fp16():
    m = build([node("MatMul", "mm", ["x", "w"], ["y"]),
               node("Relu", "r", ["y"], ["z"])],
              inputs=["x"], outputs=["z"], initializers={"w": W})
    with contextlib.redirect_stdout(io.StringIO()):
        GraphFp16.apply(m, "buffer", half="fp16")
    assert dtype_of(m, "z") == F16


def test_cast_is_not_a_bf16_boundary():
    """cast.comp converts fp32<->fp16 only: under bf16 both sides of a Cast
    stay FP32 rather than becoming a bf16 Cast the runtime refuses."""
    m = build([node("Add", "a0", ["x", "b"], ["y"]),
               node("Cast", "c", ["y"], ["yc"], to=F),
               node("Add", "a1", ["yc", "b2"], ["z"])],
              inputs=["x"], outputs=["z"],
              initializers={"b": B, "b2": B})
    bf16(m)
    assert dtype_of(m, "y") == F
    assert dtype_of(m, "yc") == F
    assert dtype_of(m, "z") == F


@pytest.mark.parametrize("attr", ["swiglu", "down_act", "epi_bias",
                                  "attn_softmax"])
def test_matmul_folds_stay_fp32(attr):
    m = build([node("MatMul", "mm", ["x", "w"], ["y"], **{attr: 1})],
              inputs=["x"], outputs=["y"], initializers={"w": W})
    s = bf16(m)
    assert attr in s["blockers"]["mm"]
    assert dtype_of(m, "y") == F


def test_matmul_odd_n_stays_fp32():
    """Odd N lands on reduce_pack, whose pack pass writes fp16."""
    w = numpy.ones((64, 7), dtype=numpy.float32)
    m = build([Node("MatMul", "mm", {},
                    [ref("x", shape=(4, 64)), ref("w", shape=(64, 7))],
                    [ref("y", shape=(4, 7))])],
              inputs=["x"], outputs=["y"], initializers={"w": w})
    s = bf16(m)
    assert "odd N" in s["blockers"]["mm"]
    assert dtype_of(m, "y") == F


def test_matmul_with_a_quantized_weight_stays_fp32():
    m = build([node("MatMul", "mm", ["x", "w"], ["y"])],
              inputs=["x"], outputs=["y"], initializers={"w": W})
    s = bf16(m, skip_weights={"w"}, quantized={"w"})
    assert "quantized" in s["blockers"]["mm"]
    assert dtype_of(m, "y") == F


def test_fp16_beside_bf16_in_one_node_is_refused():
    """An export that already declares an fp16 operand cannot sit beside a
    bf16 domain tensor: the _bf16 module would read the fp16 one as bf16."""
    m = build([node("Add", "a", ["x", ref("c", F16)], ["y"])],
              inputs=["x"], outputs=["y"])
    m.initializers["c"] = h.make_tensor("c", F16, [64], numpy.ones(64))
    s = bf16(m)
    assert s["refused"]
    assert any("beside BF16" in c for c in s["conflicts"]), s["conflicts"]
    assert dtype_of(m, "x") == F  # nothing written


def test_mixed_half_kinds_exempts_bf16_weight_slots():
    """A bf16 MatMul B beside fp16 activations is the --keep-bf16 weight
    contract, not a mix."""
    m = build([node("MatMul", "mm", [ref("x", F16), ref("w", _BFLOAT16)],
                    [ref("y", F16)])])
    m.initializers["w"] = h.make_tensor("w", _BFLOAT16, [1], [0])
    assert GraphFp16.mixed_half_kinds(m) == []
    m.nodes["add"] = node("Add", "add", [ref("y", F16), ref("q", _BFLOAT16)],
                          [ref("z", _BFLOAT16)])
    assert GraphFp16.mixed_half_kinds(m)


# ----------------------------------------------------------------- the CLI --
def convert(src, out, *extra):
    p = subprocess.run(
        [sys.executable, "-m", "onnx2vkop.cli", "-i", src, "-o", out,
         "--target-backend", "buffer", *extra],
        capture_output=True, text=True, cwd=PYPI)
    return p.returncode, p.stdout + p.stderr


def inspect_dtypes(path):
    import json
    p = subprocess.run(
        [sys.executable, "-m", "onnx2vkop.inspect", path, "--json"],
        capture_output=True, text=True, env={**os.environ, "PYTHONPATH": PYPI})
    assert p.returncode == 0, p.stderr[-400:]
    return json.loads(p.stdout)


def save(graph, path):
    m = h.make_model(graph, opset_imports=[h.make_opsetid("", 17)])
    m.ir_version = 8
    onnx.save(m, path)


def test_cli_refuses_a_mixed_fp16_bf16_graph(tmp_path):
    """Two independent branches: an fp32 one the bf16 domain takes, and one
    the export already declares fp16, which the pass never rewrites. The
    result would hold both half formats; the conversion must refuse it."""
    rng = numpy.random.default_rng(0)
    g = h.make_graph(
        [h.make_node("Add", ["x", "b"], ["z"]),
         h.make_node("Mul", ["u", "c"], ["v"])],
        "g",
        [h.make_tensor_value_info("x", T.FLOAT, [4, 64]),
         h.make_tensor_value_info("u", T.FLOAT16, [4, 64])],
        [h.make_tensor_value_info("z", T.FLOAT, [4, 64]),
         h.make_tensor_value_info("v", T.FLOAT16, [4, 64])],
        [nh.from_array(rng.standard_normal(64).astype(numpy.float32), "b"),
         nh.from_array(rng.standard_normal(64).astype(numpy.float16), "c")])
    src, out = str(tmp_path / "m.onnx"), str(tmp_path / "m.vkopbin")
    save(g, src)
    rc, log = convert(src, out, "--act", "bf16")
    assert rc != 0, log[-1500:]
    assert "mixes fp16 and bf16" in log, log[-1500:]
    # The same graph under fp16 is fine.
    rc, log = convert(src, out)
    assert rc == 0, log[-1500:]


def test_cli_bf16_checkpoint_stays_bf16(tmp_path):
    """A bf16 checkpoint used to be normalized to fp16 BEFORE the domain
    pass, which only rewrites FLOAT -- so --act bf16 shipped its weights and
    I/O as float16 beside bf16 activations. Now it goes through fp32 (exact)
    and comes back bf16."""
    rng = numpy.random.default_rng(1)
    w = nh.from_array(rng.standard_normal((64, 64)).astype(numpy.float32), "w")
    b = nh.from_array(rng.standard_normal(64).astype(numpy.float32), "b")
    for t in (w, b):
        arr = nh.to_array(t)
        bits = (arr.view(numpy.uint32) >> 16).astype(numpy.uint16)
        t.raw_data = bits.tobytes()
        t.data_type = T.BFLOAT16
    g = h.make_graph(
        [h.make_node("MatMul", ["x", "w"], ["y"]),
         h.make_node("Add", ["y", "b"], ["z"])],
        "g",
        [h.make_tensor_value_info("x", T.BFLOAT16, [4, 64])],
        [h.make_tensor_value_info("z", T.BFLOAT16, [4, 64])],
        [w, b])
    src, out = str(tmp_path / "m.onnx"), str(tmp_path / "m.vkopbin")
    save(g, src)
    rc, log = convert(src, out, "--act", "bf16")
    assert rc == 0, log[-2000:]
    d = inspect_dtypes(out)
    text = repr(d)
    assert "float16" not in text.replace("bfloat16", ""), text[:1500]
    assert "bfloat16" in text
