#!/usr/bin/env python3
"""Tests for the value-free onnxoptimizer view in optimizer.optimize_model.

onnxoptimizer's C++ backend serializes the whole proto, so a model carrying its
weights inline (>2 GiB) silently comes back as a 0-node graph and every
structural pass is lost. optimize_model now demotes big initializers to graph
inputs, interleaves the structural passes with constant folding, and reattaches
the bytes. These tests pin the round-trip: values, dims, dtype and the graph's
input contract must all be exactly what the caller handed in.
"""
import numpy as np
from onnx import TensorProto, helper

import onnx2vkop.optimizer as opt

WEIGHT_ELEMS = 1 << 20  # 4 MiB fp32, above _VIEW_DEMOTE_CAP


def _model(with_dead_reshape=True):
    weight = np.zeros(WEIGHT_ELEMS, dtype=np.float32)
    weight[0] = 1.5
    w = helper.make_tensor("W", TensorProto.FLOAT, [WEIGHT_ELEMS],
                           weight.tobytes(), raw=True)
    nodes = [helper.make_node("Mul", ["X", "W"], ["Y"], name="mul")]
    if with_dead_reshape:
        nodes.append(helper.make_node(
            "Constant", [], ["shape_c"], name="const_shape",
            value=helper.make_tensor("shape_c", TensorProto.INT64, [2], [1, 4])))
        nodes.append(helper.make_node("Reshape", ["Y", "shape_c"], ["R"],
                                      name="dead_reshape"))
    graph = helper.make_graph(
        nodes, "g",
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 4])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 4])],
        [w])
    model = helper.make_model(graph, producer_name="test")
    model.opset_import[0].version = 17
    model.ir_version = 8
    return model


def test_view_path_restores_weights_and_input_contract(monkeypatch, capsys):
    # The gate is 1.5 GiB of initializer bytes; drop it so a 4 MiB test model
    # takes the view branch.
    monkeypatch.setattr(opt, "_VIEW_MIN_INIT_BYTES", 1 << 20)
    out = opt.ONNXOptimizer.optimize_model(_model(), batch_size=1)
    assert "value-free view" in capsys.readouterr().out

    inits = {i.name: i for i in out.graph.initializer}
    assert "W" in inits, "demoted weight was not reattached"
    assert inits["W"].data_type == TensorProto.FLOAT
    assert list(inits["W"].dims) == [WEIGHT_ELEMS]
    assert np.frombuffer(inits["W"].raw_data, dtype=np.float32)[0] == 1.5

    # The synthetic input for W must be gone; only X remains a model input.
    assert [i.name for i in out.graph.input] == ["X"]
    # eliminate_deadend ran on the view, which the old path never got to do.
    assert [n.name for n in out.graph.node] == ["mul"]


def test_small_model_keeps_original_path(monkeypatch, capsys):
    monkeypatch.setattr(opt, "_VIEW_MIN_INIT_BYTES", 1 << 30)
    out = opt.ONNXOptimizer.optimize_model(_model(), batch_size=1)
    assert "value-free view" not in capsys.readouterr().out
    inits = {i.name: i for i in out.graph.initializer}
    assert len(inits["W"].raw_data) == WEIGHT_ELEMS * 4
    assert [i.name for i in out.graph.input] == ["X"]


def test_demote_and_reattach_round_trip():
    model = _model()
    demoted, added = opt._demote_large_initializers(model)
    assert [d[0] for d in demoted] == ["W"] and added == ["W"]
    assert list(model.graph.initializer) == []
    assert [i.name for i in model.graph.input] == ["X", "W"]
    wv = model.graph.input[1]
    assert wv.type.tensor_type.elem_type == TensorProto.FLOAT
    assert wv.type.tensor_type.shape.dim[0].dim_value == WEIGHT_ELEMS

    opt._reattach_initializers(model, demoted, added,
                               {"W": np.full(WEIGHT_ELEMS, 2.5, np.float32).tobytes()})
    assert [i.name for i in model.graph.initializer] == ["W"]
    assert model.graph.initializer[0].raw_data[:4] == np.float32(2.5).tobytes()
    assert [i.name for i in model.graph.input] == ["X"]


def test_reattach_skips_weight_no_longer_read():
    model = _model(with_dead_reshape=False)
    demoted, added = opt._demote_large_initializers(model)
    del model.graph.node[:]
    node = model.graph.node.add()
    node.op_type, node.name = "Identity", "id"
    node.input.extend(["X"])
    node.output.extend(["Y"])

    opt._reattach_initializers(model, demoted, added, {"W": b"\x01\x02\x03\x04"})
    assert list(model.graph.initializer) == []
    assert [i.name for i in model.graph.input] == ["X"]


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
