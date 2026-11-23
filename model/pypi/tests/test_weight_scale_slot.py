"""Where a streamed weight-only scale goes, and how a built-in bias differs.

Both pieces are about the SAME slot: vkop's scale is an extra node input, and
the index it lands on depends on whether the consumer already has a bias there.

    MatMul, no bias : [A, B, scale]          -> slot 2
    Gemm,   w/ bias : [A, B, C, scale]       -> slot 3
    Conv,   w/ bias : [X, W, B, scale]       -> slot 3

Reading slot 2 unconditionally is a SILENT corruption, not a crash, for every
format whose scale is one value per output column: a Gemm's input 2 is its bias,
and a bias of N entries is exactly the shape of an int8 scale. The 4-bit table
is [K/group, N], so there it at least raised -- which is how the MiniMax-H3
transformer conversion surfaced the defect, after writing a truncated artifact.

These tests pin the slot, and pin that the Gemm lowering carries the scale
through instead of dropping it.
"""

import contextlib
import io

import numpy
import onnx
from onnx import TensorProto, helper, numpy_helper

from onnx2vkop import quant_payload
from onnx2vkop.dag import DAGBasedModel, Node
from onnx2vkop.optimizer import FusionOptimizer


def _node(op_type, name, inputs, attrs=None):
    return helper.make_node(op_type, inputs, [name + "_out"], name=name,
                            **(attrs or {}))


# ---- scale_slot ------------------------------------------------------------

def test_scale_slot_matmul_is_two():
    n = _node("MatMul", "mm", ["X", "W"])
    assert quant_payload.scale_slot(n) == 2


def test_scale_slot_gemm_with_bias_is_three():
    """C sits at 2; the scale must go past it, or C is read as the scale."""
    n = _node("Gemm", "g", ["X", "W", "C"])
    assert quant_payload.scale_slot(n) == 3


def test_scale_slot_gemm_without_bias_is_two():
    n = _node("Gemm", "g", ["X", "W"])
    assert quant_payload.scale_slot(n) == 2


def test_scale_slot_conv_with_bias_is_three():
    n = _node("Conv", "c", ["X", "W", "B"])
    assert quant_payload.scale_slot(n) == 3


def test_scale_slot_has_no_answer_for_an_unsupported_op():
    """A consumer with no scale contract must be refused, not guessed at."""
    n = _node("SomeOtherOp", "o", ["X", "W"])
    assert quant_payload.scale_slot(n) is None


# ---- finalize appends to the right slot ------------------------------------

def _deferred_entry(tmp_path, name, k, n, group):
    """A real quantizing entry, as weight_stream.prepare would record it.

    `finalize` recomputes the table from the SOURCE bands (that is the whole
    point of the streaming design), so the entry needs a real file behind it.
    """
    src = (numpy.arange(k * n, dtype=numpy.float32).reshape(k, n) + 1.0)
    path = tmp_path / "w.bin"
    src.tofile(str(path))
    return {
        "name": name, "dims": [k, n], "data_type": TensorProto.FLOAT,
        "quant": "int4", "group": group, "transposed": False,
        "location": str(path), "offset": 0,
        "length": src.nbytes, "base_dir": str(tmp_path),
    }


def _finalize_model(op_type, with_bias):
    """Build the model; read `model.graph.node[0]` afterwards.

    NOT the node passed to make_graph: make_model deep-copies the graph, so a
    reference taken before it is a detached prototype and assertions against it
    can never observe what `finalize` did.
    """
    k, n = 8, 4
    w = numpy_helper.from_array(numpy.zeros((k, n), numpy.float32), "W")
    inits = [w]
    inputs = ["X", "W"]
    if with_bias:
        inits.append(numpy_helper.from_array(numpy.zeros(n, numpy.float32), "C"))
        inputs.append("C")
    node = _node(op_type, "carrier", inputs)
    graph = helper.make_graph(
        [node], "g",
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, k])],
        [helper.make_tensor_value_info("carrier_out", TensorProto.FLOAT, [1, n])],
        inits)
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])


def test_finalize_appends_the_scale_after_a_gemm_bias(tmp_path):
    model = _finalize_model("Gemm", with_bias=True)
    deferred = {"W": _deferred_entry(tmp_path, "W", 8, 4, 4)}
    with contextlib.redirect_stdout(io.StringIO()):
        quant_payload.finalize(model, deferred)
    got = list(model.graph.node[0].input)
    assert got == ["X", "W", "C", "W_scale"], got
    assert deferred["W"]["scale"] == "W_scale"


def test_finalize_appends_the_scale_at_two_for_a_biasless_matmul(tmp_path):
    model = _finalize_model("MatMul", with_bias=False)
    deferred = {"W": _deferred_entry(tmp_path, "W", 8, 4, 4)}
    with contextlib.redirect_stdout(io.StringIO()):
        quant_payload.finalize(model, deferred)
    got = list(model.graph.node[0].input)
    assert got == ["X", "W", "W_scale"], got


def test_finalize_does_not_mistake_a_gemm_bias_for_a_scale(tmp_path):
    """The regression: a bias at slot 2 was recorded as the scale table.

    A Gemm whose bias happened to be present used to be treated as "already
    carries a scale", so no table was ever added and the writer decoded the
    weight with the bias.
    """
    model = _finalize_model("Gemm", with_bias=True)
    deferred = {"W": _deferred_entry(tmp_path, "W", 8, 4, 4)}
    with contextlib.redirect_stdout(io.StringIO()):
        quant_payload.finalize(model, deferred)
    assert deferred["W"]["scale"] != "C", "the bias was taken for the scale"
    names = {i.name for i in model.graph.initializer}
    assert "W_scale" in names, "no scale initializer was added"


# ---- the Gemm lowering keeps the scale -------------------------------------

def _dag_gemm(weight_dtype, with_bias, with_scale=True):
    k, n = 8, 4
    np_dtype = (numpy.float32 if weight_dtype == TensorProto.FLOAT
                else numpy.int8)
    m = DAGBasedModel()
    m.initializers["W"] = numpy_helper.from_array(
        numpy.zeros((k, n), np_dtype), "W")
    m.initializers["W"].data_type = weight_dtype
    ins = [{"name": "X", "shape": [1, k], "dtype": 1}, {"name": "W", "shape": [k, n], "dtype": 1}]
    if with_bias:
        ins.append({"name": "C", "shape": [n], "dtype": 1})
    if with_scale:
        ins.append({"name": "W_scale", "shape": [n], "dtype": 1})
    m.nodes["g"] = Node("Gemm", "g", {}, ins,
                        [{"name": "g_out", "shape": [1, n], "dtype": 1}])
    return m


def test_gemm_lowering_carries_a_quantized_scale_through():
    """[A, B_int8, C, scale] -> MatMul[A, B, scale] + Add(C)."""
    m = _dag_gemm(TensorProto.INT8, with_bias=True)
    with contextlib.redirect_stdout(io.StringIO()):
        matches = FusionOptimizer.match_gemm_to_matmul(m)
        assert len(matches) == 1
        FusionOptimizer.fold_gemm_to_matmul(m, matches[0])
    mm = next(x for x in m.nodes.values() if x.op_type == "MatMul")
    assert [i["name"] for i in mm.inputs] == ["X", "W", "W_scale"], \
        [i["name"] for i in mm.inputs]
    add = next(x for x in m.nodes.values() if x.op_type == "Add")
    assert [i["name"] for i in add.inputs][1] == "C"


def test_gemm_lowering_does_not_treat_a_biasless_scale_as_a_bias():
    """[A, B_int8, scale] is a scale at slot 2, not a bias -- no Add."""
    m = _dag_gemm(TensorProto.INT8, with_bias=False)
    with contextlib.redirect_stdout(io.StringIO()):
        matches = FusionOptimizer.match_gemm_to_matmul(m)
        assert len(matches) == 1
        assert matches[0]["has_bias"] is False
        FusionOptimizer.fold_gemm_to_matmul(m, matches[0])
    assert not any(x.op_type == "Add" for x in m.nodes.values()), \
        "a biasless quantized Gemm must not gain an Add"
    mm = next(x for x in m.nodes.values() if x.op_type == "MatMul")
    assert [i["name"] for i in mm.inputs] == ["X", "W", "W_scale"]


def test_gemm_lowering_still_splits_a_float_bias():
    """The unquantized path is unchanged: [A, B, C] -> MatMul[A, B] + Add(C)."""
    m = _dag_gemm(TensorProto.FLOAT, with_bias=True, with_scale=False)
    with contextlib.redirect_stdout(io.StringIO()):
        matches = FusionOptimizer.match_gemm_to_matmul(m)
        FusionOptimizer.fold_gemm_to_matmul(m, matches[0])
    mm = next(x for x in m.nodes.values() if x.op_type == "MatMul")
    assert [i["name"] for i in mm.inputs] == ["X", "W"]
    add = next(x for x in m.nodes.values() if x.op_type == "Add")
    assert [i["name"] for i in add.inputs][1] == "C"
