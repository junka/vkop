"""--act bf16 admits only the ops whose `_bf16` kernel really decodes bf16.

The runtime's allow-list (core/Bf16Gate.cpp) refuses a bf16 activation on any
other op at LoadModel. The converter mirrors it (fp16_graph.BF16_ACT_OPS): an
op off the list is an ISLAND that keeps the width the export gave its tensors,
bounded by boundary Casts (cast.comp converts fp32/fp16/bf16), and a graph in
which one node would still bind bf16 beside another float width outside the
own-format slots (fp16_graph._BF16_OTHER_FLOAT_SLOTS) is refused.
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

from onnx2vkop import fp16_graph
from onnx2vkop.dag import DAGBasedModel, Node
from onnx2vkop.fp16_graph import BF16_ACT_OPS, GraphFp16, _BFLOAT16

F, F16 = T.FLOAT, T.FLOAT16
PYPI = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Ops with no verified bf16 kernel after the unary/elementwise and the
# data-movement streams: AveragePool/Where/TopK have no dual build at all,
# Tile no buffer kernel.
REFUSED = ["AveragePool", "Tile", "Where", "TopK"]


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


def casts(m):
    """{(source, output): to} of every Cast in the graph."""
    return {(n.inputs[0]["name"], n.outputs[0]["name"]): n.attributes["to"]
            for n in m.nodes.values() if n.op_type == "Cast"}


W = numpy.ones((64, 64), dtype=numpy.float32)
B = numpy.ones(64, dtype=numpy.float32)


def test_allow_list_matches_the_runtime():
    assert BF16_ACT_OPS == {
        "Add", "Sub", "Mul", "Div", "Pow", "Min", "Max", "Mod", "Split",
        "LayerNormalization", "LayerNorm", "RMSNorm", "Softmax", "MatMul",
        "Attention",
        # byte movers
        "Reshape", "Slice", "Squeeze", "Unsqueeze",
        # unary family, elementwise, per-operand formats
        "Sin", "Cos", "Sigmoid", "Tanh", "Erf", "Sqrt", "Relu", "Neg",
        "Floor", "Atan", "Softplus", "Clip",
        "Expand", "Concat", "FusedElemwise", "Gather", "Cast",
        # data movement / reductions
        "Transpose", "Pad", "RotaryEmbedding", "Resize", "MaxPool",
        "MaxPool2d", "GlobalAveragePool", "Reduce", "ReduceMean",
        "InstanceNormalization", "InstanceNorm", "PRelu", "CumSum",
        "Cumsum", "ScatterElements"}
    assert not set(REFUSED) & BF16_ACT_OPS


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
    assert casts(m) == {}


@pytest.mark.parametrize("op", ["Relu", "Sigmoid", "Neg",
                                "Expand", "FusedElemwise", "Transpose",
                                "Reduce", "Clip", "CumSum", "Resize",
                                "InstanceNormalization"])
def test_verified_op_joins_the_bf16_domain(op):
    """The unary body, Expand, FusedElemwise and the data-movement/reduction
    kernels decode bf16 end to end, so they run bf16 with no Cast around."""
    m = build([node("MatMul", "mm", ["x", "w"], ["y"]),
               node(op, "op", ["y"], ["z"])],
              inputs=["x"], outputs=["z"], initializers={"w": W})
    s = bf16(m)
    assert not s["refused"], s["conflicts"]
    assert "op" not in s["blockers"], s["blockers"]
    for t in ("x", "y", "z"):
        assert dtype_of(m, t) == _BFLOAT16, (op, t, dtype_of(m, t))
    assert casts(m) == {}


@pytest.mark.parametrize("op", ["Sin", "Cos"])
def test_sin_cos_on_an_activation_goes_bf16(op):
    """Sin/Cos of an activation (not a phase: its cone leaves the phase ops
    at the Softmax) run in their bf16 module."""
    m = build([node("Softmax", "sm", ["x"], ["y"], axis=-1),
               node(op, "op", ["y"], ["z"])],
              inputs=["x"], outputs=["z"])
    s = bf16(m)
    assert not s["refused"], s["conflicts"]
    assert "op" not in s["blockers"], s["blockers"]
    for t in ("x", "y", "z"):
        assert dtype_of(m, t) == _BFLOAT16, (op, t)
    assert casts(m) == {}


def test_rope_phase_is_held_at_the_export_width():
    """position * inv_freq -> Cos/Sin -> RotaryEmbedding: bf16 keeps 8
    significant bits (position 300 * freq has a 2 rad spacing, a bf16
    position id is not an integer past 256), so the phase, its Cast and
    Cos/Sin keep the export's FP32 and RotaryEmbedding reads the FP32
    tables at their own width -- q stays bf16, and the export's own Cast is
    the only one in the graph."""
    inv = numpy.linspace(1.0, 1e-4, 64, dtype=numpy.float32).reshape(64, 1)
    m = build([node("Cast", "pc", [ref("pos", T.INT64)], ["posf"], to=F),
               node("MatMul", "ph", ["inv", "posf"], ["phase"]),
               node("Transpose", "tp", ["phase"], ["phase_t"], perm=[1, 0]),
               node("Cos", "cos", ["phase_t"], ["c"]),
               node("Sin", "sin", ["phase_t"], ["s"]),
               node("MatMul", "mm", ["x", "w"], ["q"]),
               node("RotaryEmbedding", "re", ["q", "c", "s"], ["y"])],
              inputs=["x"], outputs=["y"],
              initializers={"w": W, "inv": inv})
    m.inputs.append({"name": "pos", "shape": [1, 64], "dtype": "int64"})
    s = bf16(m)
    assert not s["refused"], s["conflicts"]
    for t in ("posf", "phase", "phase_t", "c", "s"):
        assert dtype_of(m, t) == F, t
    assert m.nodes["pc"].attributes["to"] == F
    assert {"ph", "cos", "sin"} <= set(s["blockers"])
    assert "re" not in s["blockers"] and "mm" not in s["blockers"]
    assert dtype_of(m, "q") == _BFLOAT16 and dtype_of(m, "y") == _BFLOAT16
    assert m.initializers["inv"].data_type == F
    assert casts(m) == {("pos", "posf"): F}
    assert GraphFp16.bf16_kind_conflicts(m) == []


def test_gather_writes_bf16_from_its_own_table_width():
    """Gather -DBF16_OUT: an embedding lookup inside the domain writes a bf16
    activation from a table of any float width, so the FP32 table keeps its
    bytes (the slot reads its own format) and needs no Cast."""
    table = numpy.ones((16, 64), dtype=numpy.float32)
    m = build([node("Gather", "g", ["table", ref("ids", T.INT64)], ["e"]),
               node("MatMul", "mm", ["e", "w"], ["y"])],
              inputs=[], outputs=["y"],
              initializers={"table": table, "w": W})
    s = bf16(m)
    assert not s["refused"], s["conflicts"]
    assert dtype_of(m, "e") == _BFLOAT16
    assert dtype_of(m, "y") == _BFLOAT16
    assert m.initializers["table"].data_type == F
    assert casts(m) == {}


@pytest.mark.parametrize("op", REFUSED)
def test_refused_op_is_an_island_behind_casts(op):
    """An op with no bf16 kernel keeps the export's width (FP32 here) and the
    boundary Casts bridge it; the allow-listed MatMul before it stays bf16."""
    m = build([node("MatMul", "mm", ["x", "w"], ["y"]),
               node(op, "gated", ["y"], ["z"])],
              inputs=["x"], outputs=["z"], initializers={"w": W})
    s = bf16(m)
    assert not s["refused"], s["conflicts"]
    assert "gated" in s["blockers"]
    assert dtype_of(m, "y") == _BFLOAT16
    g = m.nodes["gated"]
    assert g.inputs[0]["name"] == "y__fp32" and g.inputs[0]["dtype"] == F
    assert g.outputs[0]["dtype"] == F
    assert casts(m) == {("y", "y__fp32"): F, ("z__fp32", "z"): _BFLOAT16}
    assert m.initializers["w"].data_type == _BFLOAT16


def test_the_same_gated_op_is_fine_under_fp16():
    m = build([node("MatMul", "mm", ["x", "w"], ["y"]),
               node("Relu", "r", ["y"], ["z"])],
              inputs=["x"], outputs=["z"], initializers={"w": W})
    with contextlib.redirect_stdout(io.StringIO()):
        GraphFp16.apply(m, "buffer", half="fp16")
    assert dtype_of(m, "z") == F16


def test_cast_to_fp32_into_an_island_is_a_bf16_boundary():
    """cast.comp converts fp32/fp16/bf16 among themselves, so an export
    Cast(to=FLOAT) in front of an island is the boundary: it reads the bf16
    side and writes the FP32 the island runs at -- no extra Cast."""
    m = build([node("Add", "a0", ["x", "b"], ["y"]),
               node("Cast", "c", ["y"], ["yc"], to=F),
               node("AveragePool", "pool", ["yc"], ["z"])],
              inputs=["x"], outputs=["z"],
              initializers={"b": B})
    s = bf16(m)
    assert not s["refused"], s["conflicts"]
    assert dtype_of(m, "y") == _BFLOAT16
    assert dtype_of(m, "yc") == F
    assert m.nodes["c"].attributes["to"] == F
    assert m.nodes["pool"].inputs[0]["name"] == "yc"


def test_cast_to_half_is_retargeted_to_bf16():
    """A Cast(to=FLOAT16) whose output feeds the bf16 domain must say
    BFLOAT16: the runtime allocates a Cast's output from `to`."""
    m = build([node("AveragePool", "pool", ["x"], ["xf"]),
               node("Cast", "c", ["xf"], ["xc"], to=F16),
               node("Add", "a", ["xc", "b"], ["z"])],
              inputs=["x"], outputs=["z"], initializers={"b": B})
    s = bf16(m)
    assert not s["refused"], s["conflicts"]
    assert dtype_of(m, "xf") == F
    assert dtype_of(m, "xc") == _BFLOAT16
    assert m.nodes["c"].attributes["to"] == _BFLOAT16


def test_rotary_keeps_an_fp32_table_beside_bf16_x():
    """cos/sin from an op with no bf16 kernel stay FP32 and need no Cast: the
    rotation (and the MatMul feeding it) stays bf16, because rotary.comp
    reads an fp32 table at its own width (cs_fp32) and Bf16Gate admits that
    slot."""
    m = build([node("MatMul", "mm", ["x", "w"], ["q"]),
               node("AveragePool", "phase", ["pos"], ["cos"]),
               node("AveragePool", "phase2", ["pos2"], ["sin"]),
               node("RotaryEmbedding", "re", ["q", "cos", "sin"], ["y"])],
              inputs=["x", "pos", "pos2"], outputs=["y"],
              initializers={"w": W})
    s = bf16(m)
    assert not s["refused"], s["conflicts"]
    assert "re" not in s["blockers"], s["blockers"]
    for t in ("q", "y"):
        assert dtype_of(m, t) == _BFLOAT16, t
    re = m.nodes["re"]
    assert [e["name"] for e in re.inputs] == ["q", "cos", "sin"]
    for t in ("cos", "sin"):
        assert dtype_of(m, t) == F, t
    assert GraphFp16.bf16_kind_conflicts(m) == []


def test_rotary_widens_an_fp16_table_to_fp32():
    """A 16-bit table is read in X's format, so an fp16 cos/sin beside bf16 X
    would be decoded as bf16: it is widened (exactly) to fp32 instead."""
    f16 = lambda n: ref(n, F16)  # noqa: E731
    m = build([node("MatMul", "mm", ["x", "w"], ["q"]),
               node("AveragePool", "phase", [f16("pos")], [f16("cos")]),
               node("RotaryEmbedding", "re", ["q", f16("cos"), f16("cos")],
                    ["y"])],
              inputs=["x", "pos"], outputs=["y"], initializers={"w": W})
    s = bf16(m)
    assert not s["refused"], s["conflicts"]
    assert dtype_of(m, "cos") == F16
    assert casts(m)[("cos", "cos__fp32")] == F
    assert [e["name"] for e in m.nodes["re"].inputs] == \
        ["q", "cos__fp32", "cos__fp32"]
    assert GraphFp16.bf16_kind_conflicts(m) == []
    # ...and the refusal agrees: the same node with the fp16 table is a mix.
    m.nodes["re"].inputs[1] = f16("cos")
    assert any("re" in c for c in GraphFp16.bf16_kind_conflicts(m))


def test_rotary_table_follows_a_bf16_producer():
    m = build([node("MatMul", "mm", ["x", "w"], ["q"]),
               node("Mul", "c", ["p", "b"], ["cos"]),
               node("RotaryEmbedding", "re", ["q", "cos", "cos"], ["y"])],
              inputs=["x", "p"], outputs=["y"],
              initializers={"w": W, "b": B})
    s = bf16(m)
    assert not s["refused"], s["conflicts"]
    for t in ("q", "cos", "y"):
        assert dtype_of(m, t) == _BFLOAT16, t
    assert casts(m) == {}


def test_matmul_down_fold_is_an_island():
    """matmul_down_act.comp reads fp16 only: the down fold keeps the export
    width behind Casts under bf16."""
    m = build([node("MatMul", "mm", ["x", "w"], ["y"], down_act=1)],
              inputs=["x"], outputs=["y"], initializers={"w": W})
    s = bf16(m)
    assert "down_act" in s["blockers"]["mm"]
    assert m.nodes["mm"].outputs[0]["dtype"] == F
    assert m.initializers["w"].data_type == F


@pytest.mark.parametrize("attr", ["swiglu", "epi_bias", "attn_softmax"])
def test_matmul_folds_with_a_bf16_kernel_go_bf16(attr):
    """The gate/up fold, the QK^T epilogue and the fused attn_softmax kernel
    read and write bf16 in the _bf16 module (Bf16ActivationTest), so those
    MatMuls join the bf16 domain."""
    m = build([node("MatMul", "mm", ["x", "w"], ["y"], **{attr: 1})],
              inputs=["x"], outputs=["y"], initializers={"w": W})
    s = bf16(m)
    assert "mm" not in s["blockers"]
    assert dtype_of(m, "y") == _BFLOAT16
    assert casts(m) == {}


def test_matmul_odd_n_goes_bf16():
    """Odd N lands on reduce_pack, whose pack pass encodes bf16 now."""
    w = numpy.ones((64, 7), dtype=numpy.float32)
    m = build([Node("MatMul", "mm", {},
                    [ref("x", shape=(4, 64)), ref("w", shape=(64, 7))],
                    [ref("y", shape=(4, 7))])],
              inputs=["x"], outputs=["y"], initializers={"w": w})
    s = bf16(m)
    assert "mm" not in s["blockers"]
    assert dtype_of(m, "y") == _BFLOAT16


def test_matmul_with_a_quantized_weight_is_an_island():
    m = build([node("MatMul", "mm", ["x", "w"], ["y"])],
              inputs=["x"], outputs=["y"], initializers={"w": W})
    s = bf16(m, skip_weights={"w"}, quantized={"w"})
    assert "quantized" in s["blockers"]["mm"]
    assert m.nodes["mm"].outputs[0]["dtype"] == F


@pytest.mark.parametrize("fmt", ["fp8", "nvfp4", "nf4"])
def test_matmul_with_a_non_int8_quantized_weight_is_an_island(fmt):
    m = build([node("MatMul", "mm", ["x", "w"], ["y"])],
              inputs=["x"], outputs=["y"], initializers={"w": W})
    s = bf16(m, skip_weights={"w"}, quantized={"w": fmt})
    assert "quantized" in s["blockers"]["mm"]
    assert m.nodes["mm"].outputs[0]["dtype"] == F


@pytest.mark.parametrize("fmt", ["int8", "int4"])
def test_matmul_with_an_int8_or_int4_weight_goes_bf16(fmt):
    """matmul.comp's _bf16 module decodes a byte B to fp32 and applies the
    fp32 per-column scale (Bf16ActivationTest.MatMulInt8WeightMatchesFp32),
    so a plain -q int8 MatMul stays in the bf16 domain."""
    m = build([node("MatMul", "mm", ["x", "w"], ["y"])],
              inputs=["x"], outputs=["y"], initializers={"w": W})
    s = bf16(m, skip_weights={"w"}, quantized={"w": fmt})
    assert "mm" not in s["blockers"]
    assert dtype_of(m, "y") == _BFLOAT16
    assert casts(m) == {}


@pytest.mark.parametrize("attr", ["swiglu", "epi_bias", "attn_softmax"])
def test_int8_matmul_folds_stay_islands(attr):
    """No test covers an int8 weight in these epilogues under bf16."""
    m = build([node("MatMul", "mm", ["x", "w"], ["y"], **{attr: 1})],
              inputs=["x"], outputs=["y"], initializers={"w": W})
    s = bf16(m, skip_weights={"w"}, quantized={"w": "int8"})
    assert "quantized" in s["blockers"]["mm"]


def test_mixed_half_kinds_exempts_the_int8_scale():
    """[A bf16, W int8, s fp32] is the int8 weight-only contract the runtime
    gate admits (bf16_int8_weight_slot); an fp32 scale anywhere else is not."""
    m = build([node("MatMul", "mm",
                    [ref("x", _BFLOAT16), ref("w", T.INT8), ref("s", F)],
                    [ref("y", _BFLOAT16)])])
    m.initializers["w"] = h.make_tensor("w", T.INT8, [1], [0])
    m.initializers["s"] = h.make_tensor("s", F, [1], [1.0])
    assert GraphFp16.mixed_half_kinds(m) == []
    m.nodes["add"] = node("Add", "add", [ref("y", _BFLOAT16), ref("s", F)],
                          [ref("z", _BFLOAT16)])
    assert GraphFp16.mixed_half_kinds(m)


def test_imported_nvfp4_matmul_goes_bf16():
    """The folded NVFP4 form [A, W e2m1, s e4m3, g fp32] has a bf16 kernel
    (Bf16ActivationTest.MatMulNvfp4WeightMatchesFp32): it stays in the bf16
    domain and its fp32 global is not a width conflict."""
    m = build([node("MatMul", "mm",
                    ["x", ref("w", T.FLOAT4E2M1), ref("s", T.FLOAT8E4M3FN),
                     ref("g", F)], ["y"])],
              inputs=["x"], outputs=["y"])
    m.initializers["w"] = h.make_tensor("w", T.FLOAT4E2M1, [2], [0, 0])
    m.initializers["s"] = h.make_tensor("s", T.FLOAT8E4M3FN, [1], [1.0])
    m.initializers["g"] = h.make_tensor("g", F, [1], [1.0])
    s = bf16(m)
    assert "mm" not in s["blockers"]
    assert dtype_of(m, "y") == _BFLOAT16
    assert GraphFp16.mixed_half_kinds(m) == []


def test_fp16_operand_beside_bf16_is_retyped():
    """An export that declares an fp16 operand of a bf16 node used to be
    refused (the _bf16 module would read it as bf16). FLOAT16 is a candidate
    under bf16 now, so the operand is retyped instead: its payload is
    rounded to bf16 from the fp16 value."""
    m = build([node("Add", "a", ["x", ref("c", F16)], ["y"])],
              inputs=["x"], outputs=["y"])
    m.initializers["c"] = h.make_tensor("c", F16, [64], numpy.ones(64))
    s = bf16(m)
    assert not s["refused"], s["conflicts"]
    assert dtype_of(m, "c") == _BFLOAT16
    assert m.initializers["c"].data_type == _BFLOAT16
    assert set(numpy.frombuffer(m.initializers["c"].raw_data,
                                numpy.uint16)) == {0x3F80}  # 1.0


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


def test_own_format_slots_match_the_runtime_gate():
    """fp16_graph._BF16_OTHER_FLOAT_SLOTS is the converter's copy of
    core/Bf16Gate.cpp bf16_other_float_ok (pinned by Bf16ActivationTest.
    OtherFloatSlotsAreOneRule): any width in every FusedElemwise input,
    Gather's table, Clip's bounds and Pad's constant; fp32 (not fp16) in
    RotaryEmbedding's cos/sin; bf16 only everywhere else."""
    rule = fp16_graph._bf16_slot_rule
    assert rule("FusedElemwise", 0) == rule("FusedElemwise", 5) == "any"
    assert rule("Gather", 0) == "any" and rule("Gather", 1) is None
    assert rule("Clip", 1) == rule("Clip", 2) == "any"
    assert rule("Clip", 0) is None
    assert rule("Pad", 2) == "any" and rule("Pad", 0) is None
    assert rule("RotaryEmbedding", 1) == rule("RotaryEmbedding", 2) == "fp32"
    assert rule("RotaryEmbedding", 0) is None
    assert rule("Add", 1) is None and rule("Cast", 0) is None


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


def test_cli_retypes_an_fp16_branch_to_bf16(tmp_path):
    """Two independent branches: an fp32 one and one the export already
    declares fp16. Before BF16_CAST_BOUNDARY the pass never rewrote FLOAT16
    and the result held both half formats (refused); now an fp16 export is a
    candidate like an fp32 one, so both branches come out bf16."""
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
    assert rc == 0, log[-1500:]
    text = repr(inspect_dtypes(out))
    assert "float16" not in text.replace("bfloat16", ""), text[:1500]
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
