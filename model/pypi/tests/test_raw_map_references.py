"""The reference-based raw_map must preserve weights on every path.

`optimize_model` used to build
    raw_map = {init.name: init.raw_data for ...}
which copies every weight held in the proto -- free for a streamed model (the
bytes are in the external file), but a second full copy under `-q int4`, where
prepare() hydrates everything first. It now holds the TensorProto MESSAGES.

That change is only safe where the messages are out of the graph at the moment
fold() runs, because protobuf gives references, not copies (measured on b3):

    g.initializer[0].raw_data = b""   # clears the graph's message too
    for t in g.initializer: ...       # t IS the message, not a copy

So a reference held in raw_map is cleared along with the graph's. Which happens
depends on the branch:

  * demote branch (weights > _VIEW_MIN_INIT_BYTES, what the 27B takes): the
    initializers are REMOVED from the graph before fold, so fold cannot reach
    them and the references stay valid.
  * no-demote branch, onnxoptimizer succeeds: it returns a rebuilt model, so
    fold clears that copy's messages and raw_map's still hold theirs.
  * no-demote branch, onnxoptimizer RAISES or returns an empty graph: both
    fallbacks assign `optimized_model = onnx_model`, the identical object, so
    fold clears the very messages raw_map references.

That last combination is the one this file exists for: it silently produced a
0-byte weight until optimize_model started capturing those bytes before fold.
"""
import numpy as np
import pytest
from onnx import TensorProto, helper

import onnx2vkop.optimizer as opt

ELEMS = (1 << 20) + 4096          # above fold()'s 1 MiB strip cap
W0 = 1.5


def _model(dead_node=True):
    """One big fp32 weight read by a Mul, plus a foldable dead Reshape."""
    w = np.zeros(ELEMS, np.float32)
    w[0] = W0
    t = helper.make_tensor("W", TensorProto.FLOAT, [ELEMS], w.tobytes(), raw=True)
    nodes = [helper.make_node("Mul", ["X", "W"], ["Y"], name="mul")]
    if dead_node:
        nodes.append(helper.make_node(
            "Constant", [], ["shape_c"], name="const_shape",
            value=helper.make_tensor("shape_c", TensorProto.INT64, [2], [1, 4])))
    g = helper.make_graph(
        nodes, "g",
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 4])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 4])], [t])
    m = helper.make_model(g, producer_name="t")
    m.opset_import[0].version = 17
    m.ir_version = 8
    return m


def _assert_weight_intact(out, label):
    inits = {i.name: i for i in out.graph.initializer}
    assert "W" in inits, f"{label}: W was dropped from the graph"
    init = inits["W"]
    assert len(init.raw_data) == ELEMS * 4, (
        f"{label}: W came back with {len(init.raw_data)} bytes, not "
        f"{ELEMS * 4} -- its message was cleared through a shared reference")
    assert list(init.dims) == [ELEMS], f"{label}: dims {list(init.dims)}"
    assert init.data_type == TensorProto.FLOAT, f"{label}: dtype changed"
    assert np.frombuffer(init.raw_data, np.float32)[0] == W0, \
        f"{label}: the value did not survive"


@pytest.fixture
def gate(monkeypatch):
    """Set the demote gate, restoring it afterwards (it is module state)."""
    def _set(n):
        monkeypatch.setattr(opt, "_VIEW_MIN_INIT_BYTES", n)
    return _set


def test_demote_branch_preserves_the_weight(gate):
    """The 27B's path: weights leave the graph before fold runs."""
    gate(1 << 20)
    _assert_weight_intact(opt.ONNXOptimizer.optimize_model(_model(), batch_size=1),
                          "demote branch")


def test_no_demote_with_onnxoptimizer_ok(gate):
    """onnxoptimizer hands back a rebuilt model, so fold clears its copy."""
    gate(1 << 40)
    _assert_weight_intact(opt.ONNXOptimizer.optimize_model(_model(), batch_size=1),
                          "no-demote, optimiser ok")


def test_no_demote_when_onnxoptimizer_raises(gate, monkeypatch):
    """The regression: the fallback keeps the SAME object, so fold clears the
    messages raw_map references and the weight came back as 0 bytes."""
    gate(1 << 40)

    def boom(*a, **kw):
        raise RuntimeError("simulated: model too large to serialize")

    monkeypatch.setattr(opt.optimizer, "optimize", boom)
    _assert_weight_intact(opt.ONNXOptimizer.optimize_model(_model(), batch_size=1),
                          "no-demote, optimiser raises")


def test_no_demote_when_onnxoptimizer_returns_empty(gate, monkeypatch):
    """The other same-object fallback: a degenerate 0-node graph.

    Unlike the raising case, this one cannot be checked by reading the weight
    back: ConstantFolder.fold is handed the restored `onnx_model` and its
    shape-inference call returns the empty stub, so fold's result is an empty
    graph and the weight is gone for a reason that has nothing to do with
    raw_map. What this pins is that the fallback itself hands fold the REAL
    model -- otherwise the (correct) 0-node result would be reached from a
    model that had already lost its weights, and the two would be
    indistinguishable from the outside.
    """
    gate(1 << 40)
    seen = {}

    real_optimize = opt.optimizer.optimize

    def empty(*a, **kw):
        stub = helper.make_model(helper.make_graph([], "empty", [], [], []))
        stub.opset_import[0].version = 17
        stub.ir_version = 8
        return stub

    real_fold = opt.ConstantFolder.fold

    def spy(model, **kw):
        seen["name"] = model.graph.name
        seen["inits"] = len(model.graph.initializer)
        return real_fold(model, **kw)

    monkeypatch.setattr(opt.optimizer, "optimize", empty)
    monkeypatch.setattr(opt.ConstantFolder, "fold", spy)
    try:
        opt.ONNXOptimizer.optimize_model(_model(), batch_size=1)
    finally:
        opt.optimizer.optimize = real_optimize
        opt.ConstantFolder.fold = real_fold

    assert seen.get("name") == "g", (
        f"fold ran on {seen.get('name')!r}, not the restored onnx_model")
    assert seen.get("inits") == 1, (
        f"fold saw {seen.get('inits')} initializer(s); the fallback did not "
        f"restore the graph")


def test_a_typed_field_weight_is_never_demoted_or_lost(gate):
    """float_data carries the values instead of raw_data; the demote gate
    leaves those in place, so they must come through untouched."""
    gate(1 << 20)
    t = helper.make_tensor("W", TensorProto.FLOAT, [ELEMS],
                           np.full(ELEMS, W0, np.float32).tolist())
    g = helper.make_graph(
        [helper.make_node("Mul", ["X", "W"], ["Y"], name="mul")], "g",
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 4])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 4])], [t])
    m = helper.make_model(g, producer_name="t")
    m.opset_import[0].version = 17
    m.ir_version = 8

    out = opt.ONNXOptimizer.optimize_model(m, batch_size=1)
    init = {i.name: i for i in out.graph.initializer}["W"]
    assert not init.HasField("raw_data"), "typed-field storage was converted"
    assert len(init.float_data) == ELEMS
    assert init.float_data[0] == W0
