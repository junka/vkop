"""Conversion must refuse a graph with an op the runtime cannot execute.

Before this check, an unknown op_type was silently skipped at load
(core/runtime.cpp): the node registered no output tensor, so a downstream
consumer read a null/misaligned input and a graph output kept a previous
round's bytes -- wrong output with every test still green.
validate_supported_ops (called at the end of apply_optimizations) turns that
into a conversion-time UnsupportedOpError.
"""
from types import SimpleNamespace

import onnx
import pytest
from onnx import TensorProto, helper

from onnx2vkop.converter import ModelConverter, UnsupportedOpError


def _opts():
    return SimpleNamespace(quant=None, unify=False, rgba=False,
                           group_size=64, target_backend="buffer")


def _single_node_model(tmp_path, op_type, **attrs):
    x = helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 2])
    y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 2])
    node = helper.make_node(op_type, ["X"], ["Y"], name="n0", **attrs)
    graph = helper.make_graph([node], "g", [x], [y])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    path = tmp_path / f"{op_type}.onnx"
    onnx.save(model, str(path))
    return path


def _convert(path):
    conv = ModelConverter()
    dag = conv.parse_onnx_model(str(path), 1)
    conv.apply_optimizations(dag, _opts())


def test_unknown_op_is_rejected(tmp_path):
    with pytest.raises(UnsupportedOpError) as ei:
        _convert(_single_node_model(tmp_path, "Clip", min=0.0, max=1.0))
    assert "Clip" in str(ei.value)


def test_supported_op_converts(tmp_path):
    # Positive control: a graph of only-known ops must not raise.
    _convert(_single_node_model(tmp_path, "Relu"))


def test_control_flow_message_mentions_folding():
    """If/Loop/Scan carry their body as a GRAPH attribute vkop cannot lower;
    the error must say so rather than just 'unknown op'."""

    class _N:
        def __init__(self, op):
            self.op_type = op

    class _Dag:
        nodes = {"a": _N("If"), "b": _N("Loop")}

    with pytest.raises(UnsupportedOpError) as ei:
        ModelConverter().validate_supported_ops(_Dag())
    msg = str(ei.value)
    assert "If" in msg and "Loop" in msg
    assert "fold" in msg.lower()
