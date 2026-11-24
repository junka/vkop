"""`-q fp16` must FAIL LOUDLY when the FP16 domain pass refuses.

The domain pass declines a graph whose nodes would mix FP32 and FP16 operands,
because the runtime picks each node's shader from its data input's dtype and
one shader cannot read two widths. `GraphFp16.apply` reports that as
`refused=True` -- and `apply_optimizations` used to discard the return value and
run the weight-only pass anyway.

Weight-only quantization halves the weights while leaving the activations
alone, which is precisely the width mix the domain pass just refused. The
artifact then loads and the runtime rejects it at execute time. Measured on the
MiniMax-H3 text encoder: the pass refused with 250 conflicts and the silent
fallback shipped a graph whose fused Attention got an fp32 mask beside fp16
Q/K/V, so every run died with

    Attention: the attention bias must have Q's element type
      (Q is float16, the bias is float32)

This pins the guard that was added at the `-q fp16` dispatch: a refusal must
abort, and the weight-only pass must not run behind it. The domain policy
itself stays in test_fp16_graph_domain.py.
"""
import inspect

import numpy
from onnx import TensorProto, numpy_helper

from onnx2vkop.converter import ModelConverter
from onnx2vkop.dag import DAGBasedModel, Node
from onnx2vkop import fp16_graph

F, F16 = TensorProto.FLOAT, TensorProto.FLOAT16


def ref(name, dtype=F):
    return {"name": name, "shape": [-1], "dtype": dtype}


def node(op, name, ins, outs, **attrs):
    return Node(op, name, attrs, [ref(n) if isinstance(n, str) else n
                                  for n in ins],
                [ref(n) if isinstance(n, str) else n for n in outs])


def mixed_graph():
    """A MatMul reading an FP16 activation beside a weight another op needs FP32.

    The weight must stay wide (BatchNormalization reads it) while the MatMul
    already reads a half, so that node would mix widths and the domain pass
    refuses. Same shape as test_fp16_graph_domain.py's shared-weight case.
    """
    m = DAGBasedModel()
    m.nodes["mm"] = node("MatMul", "mm", [ref("a16", F16), "W"], ["o"])
    m.nodes["bn"] = node("BatchNormalization", "bn", ["x", "W"], ["b"])
    m.inputs = [{"name": "x", "shape": [1, 4], "dtype": "float32"},
                {"name": "a16", "shape": [1, 4], "dtype": "float16"}]
    m.outputs = [{"name": "o", "shape": [1, 4], "dtype": "float32"}]
    m.initializers["W"] = numpy_helper.from_array(
        numpy.ones((4, 4), dtype=numpy.float32), "W")
    m.build_dependencies()
    return m


def test_the_domain_pass_reports_refused_for_a_mixed_graph():
    """The signal the guard keys on must actually be set."""
    m = mixed_graph()
    summary = fp16_graph.GraphFp16.apply(m)
    assert summary["refused"] is True
    assert summary["conflicts"], "refused without naming a conflict"


def test_the_domain_pass_does_not_refuse_a_clean_graph():
    """Guard against over-correcting: a convertible graph must convert."""
    m = DAGBasedModel()
    m.nodes["mm"] = node("MatMul", "mm", ["x", "W"], ["o"])
    m.inputs = [{"name": "x", "shape": [1, 4], "dtype": "float32"}]
    m.outputs = [{"name": "o", "shape": [1, 4], "dtype": "float32"}]
    m.initializers["W"] = numpy_helper.from_array(
        numpy.ones((4, 4), dtype=numpy.float32), "W")
    m.build_dependencies()
    summary = fp16_graph.GraphFp16.apply(m)
    assert summary["refused"] is False
    assert m.initializers["W"].data_type == F16


def test_the_fp16_dispatch_stops_on_a_refused_domain_pass():
    """The `-q fp16` branch must check `refused` before the weight pass.

    Asserted on the source rather than by executing the whole optimizer: the
    behavior under test is the ORDER of two calls, and the weight-only pass
    behind them is a large, separately-tested unit.
    """
    src = inspect.getsource(ModelConverter.apply_optimizations)
    apply_call = src.index("GraphFp16.apply")
    guard = src.index('fp16_summary.get("refused")')
    # Search past the guard: the earlier occurrences of the weight pass's name
    # are the comments above it, not a call.
    weight_pass = src.index("quantize_to_fp16_selective", guard)
    assert apply_call < guard < weight_pass, (
        "the refused check must sit between GraphFp16.apply and the "
        "weight-only fp16 pass, or a refusal silently degrades to the "
        "mixed-width artifact this guard exists to prevent")
    # And it must abort, not warn.
    segment = src[guard:weight_pass]
    assert "raise" in segment, (
        "a refused domain pass must raise; printing a warning and continuing "
        "writes the artifact the runtime then rejects")
