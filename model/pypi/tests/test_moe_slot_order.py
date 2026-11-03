import numpy as np
import onnx
from onnx import TensorProto, helper

from onnx2vkop.optimizer import ONNXOptimizer, _position_paging_slots


def make(expert_counts=(4, 4), slot_first=False):
    inits = []
    metas = {}
    outs = []
    for gi, n_exp in enumerate(expert_counts):
        grp = "L%d.gate_proj" % gi
        for e in range(n_exp):
            name = "moe.%s.e%d" % (grp, e)
            inits.append(helper.make_tensor(name, TensorProto.FLOAT16, [4, 4],
                                            np.zeros(16, np.float16)))
            metas[name] = (grp, e)
            outs.append(name)
        w = helper.make_tensor("L%d.qkv.weight" % gi, TensorProto.FLOAT16, [4, 4],
                               np.ones(16, np.float16))
        inits.append(w)
    if slot_first:
        inits.sort(key=lambda t: (not t.name.startswith("moe."), t.name))
    else:
        inits.sort(key=lambda t: (t.name.startswith("moe."), t.name))
    graph = helper.make_graph(
        [helper.make_node("MatMul", ["x", "L0.qkv.weight"], ["y"])], "t",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT16, [1, 4])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT16, [1, 4])],
        initializer=inits)
    for name in outs:
        graph.output.append(helper.make_tensor_value_info(
            name, TensorProto.FLOAT16, [4, 4]))
    return helper.make_model(graph), metas


def order(model, metas):
    seen = []
    for t in model.graph.initializer:
        if t.name in metas:
            seen.append(metas[t.name])
    return seen


def test_position_slots_reorders_interleaved():
    model, metas = make()
    moved = _position_paging_slots(model, metas)
    assert moved > 0
    want = [(g, e) for g in ("L0.gate_proj", "L1.gate_proj") for e in range(4)]
    assert order(model, metas) == want


def test_position_slots_is_idempotent():
    model, metas = make(slot_first=True)
    assert _position_paging_slots(model, metas) == 0
    want = [(g, e) for g in ("L0.gate_proj", "L1.gate_proj") for e in range(4)]
    assert order(model, metas) == want


def test_position_slots_rejects_expert_id_gap():
    model, metas = make(expert_counts=(4,))
    del metas["moe.L0.gate_proj.e2"]
    try:
        _position_paging_slots(model, metas)
    except ValueError as exc:
        assert "expert ids" in str(exc)
    else:
        raise AssertionError("a gap in expert ids must be refused")


def test_slot_order_survives_the_optimizer_passes():
    """The real regression: eliminate_unused_initializer rebuilds the list."""
    model, metas = make()
    _position_paging_slots(model, metas)
    neg = ONNXOptimizer()
    model = neg.optimize_model(model, keep_alive=set(metas))
    got = order(model, metas)
    assert got == [(g, e) for g in ("L0.gate_proj", "L1.gate_proj") for e in range(4)]
