"""A SwiGLU fold's weights must enter the FP16 domain like any other weight.

Both folds (fuse_swiglu_gate_up -> MatMul swiglu=1 over an interleaved Wgu,
fuse_swiglu_down_proj -> MatMul down_act over [x, Wd, Wg, Wu]) built their
weight input entries as a bare {"name": ...}. GraphFp16._decls only sees a
tensor through an entry that carries a dtype, so on an fp32 export the folded
weights never entered the domain and shipped as fp32 bytes, which the fold
kernels (fp16-only) read as halves: every output NaN on a D=256/F=512 MLP
(and on every other size and route). LLM exports already store fp16 weights,
which is why only fp32 sources showed it.
"""
import contextlib
import io

import numpy as np
import pytest
from onnx import TensorProto, numpy_helper

from onnx2vkop.dag import DAGBasedModel, Node
from onnx2vkop.fp16_graph import GraphFp16
from onnx2vkop.optimizer import FusionOptimizer, Quantizer

FLOAT, FLOAT16 = TensorProto.FLOAT, TensorProto.FLOAT16
M, D, F = 4, 32, 64


def ref(name, dims):
    return {"name": name, "shape": list(dims), "dtype": FLOAT}


def build(declared=True):
    """An fp32 SwiGLU MLP exactly as the parser hands it over.

    `declared=False` drops the weights' dtype from their entries (a reference
    the parser could not resolve); the fold then reads the payload's."""
    rng = np.random.default_rng(11)
    m = DAGBasedModel()
    for name, shape in (("Wg", (D, F)), ("Wu", (D, F)), ("Wd", (F, D))):
        m.initializers[name] = numpy_helper.from_array(
            rng.normal(size=shape).astype(np.float32), name)
    mm = lambda n, a, w, o, ws, os_: Node("MatMul", n, {},  # noqa: E731
                                          [ref(a, [M, ws[0]]), ref(w, ws)],
                                          [ref(o, os_)])
    m.nodes["gate"] = mm("gate", "x", "Wg", "g", (D, F), (M, F))
    m.nodes["up"] = mm("up", "x", "Wu", "u", (D, F), (M, F))
    m.nodes["sig"] = Node("Sigmoid", "sig", {}, [ref("g", [M, F])],
                          [ref("s", [M, F])])
    m.nodes["m1"] = Node("Mul", "m1", {}, [ref("g", [M, F]), ref("s", [M, F])],
                         [ref("a", [M, F])])
    m.nodes["m2"] = Node("Mul", "m2", {}, [ref("a", [M, F]), ref("u", [M, F])],
                         [ref("h", [M, F])])
    m.nodes["down"] = mm("down", "h", "Wd", "y", (F, D), (M, D))
    if not declared:
        for n in m.nodes.values():
            for e in n.inputs:
                if e["name"].startswith("W"):
                    del e["dtype"]
    m.inputs = [{"name": "x", "shape": [M, D], "dtype": "float32"}]
    m.outputs = [{"name": "y", "shape": [M, D], "dtype": "float32"}]
    return m


def fold(m, which):
    match = getattr(FusionOptimizer, f"match_swiglu_{which}")(m)
    assert len(match) == 1
    assert getattr(FusionOptimizer, f"fold_swiglu_{which}")(m, match[0])
    attr = "swiglu" if which == "gate_up" else "down_act"
    return [n for n in m.nodes.values() if n.attributes.get(attr)][0]


@pytest.mark.parametrize("declared", [True, False])
@pytest.mark.parametrize("which", ["gate_up", "down_proj"])
def test_fold_weights_are_downcast_by_the_fp16_domain(which, declared):
    m = build(declared)
    fused = fold(m, which)
    weights = [i["name"] for i in fused.inputs[1:]]
    before = {w: numpy_helper.to_array(m.initializers[w]) for w in weights}
    # The entry carries a dtype (declared, else the payload's), so the
    # domain pass can see it.
    for entry in fused.inputs[1:]:
        assert entry.get("dtype") == FLOAT, entry

    with contextlib.redirect_stdout(io.StringIO()):
        summary = GraphFp16.apply(m, "buffer")

    assert not summary["refused"]
    for w in weights:
        init = m.initializers[w]
        assert init.data_type == FLOAT16, (
            f"{which}: fold weight {w} stayed "
            f"{TensorProto.DataType.Name(init.data_type)}; the runtime reads "
            f"it as fp16")
        np.testing.assert_array_equal(numpy_helper.to_array(init),
                                      before[w].astype(np.float16))
    for entry in fused.inputs:
        assert entry.get("dtype") == FLOAT16, entry
    Quantizer._validate_folded_weights(m)


@pytest.mark.parametrize("which", ["gate_up", "down_proj"])
def test_validator_refuses_an_fp32_fold_weight(which):
    """-q fp32 (no domain pass) must not ship a fold the kernels misread."""
    m = build()
    fold(m, which)
    with pytest.raises(ValueError, match="fp32"):
        Quantizer._validate_folded_weights(m)
