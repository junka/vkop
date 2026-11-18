"""Folding an int8 activation Q/DQ pair into a MatMul for the DP4A route.

ops/Matmul.hpp has no int8-A fallback: when A is a byte tensor the route must
be ksplit_w8a8_dp4a or the kernel throws. The converter therefore has to ship
a model whose MatMul shapes already satisfy every gate in core/KernelSelect.cpp
before it re-points A at the QuantizeLinear output. These cases pin each gate
individually -- a fold that fires where the runtime cannot select the route
produces a model that loads and then refuses to run.

The scale assertion is the other half. The kernel computes
acc * a_scale * weight_scale, so the value the converter records as `a_scale`
is what makes the result correct, not just plausible; a dropped attribute is
off by the activation scale (here a factor of two).
"""

import contextlib
import io

import numpy as np
import pytest
from onnx import TensorProto, numpy_helper

from onnx2vkop.dag import DAGBasedModel, Node
from onnx2vkop.optimizer import W8A8ActivationFold

A_SCALE = 0.5  # exact in float32, so the attribute text is stable


def ref(name, shape, dtype=None):
    d = {"name": name, "shape": list(shape)}
    if dtype is not None:
        d["dtype"] = dtype
    return d


def build(
    *,
    m=1,
    k=512,
    n=256,
    out_dtype=TensorProto.FLOAT16,
    zero_point=0,
    with_cast=False,
    transB=0,
    weight_dtype=np.int8,
    a_scale=A_SCALE,
    dq_scale=None,
    extra_dq_consumer=False,
):
    mx = DAGBasedModel()
    mx.initializers["W"] = numpy_helper.from_array(np.ones((k, n), weight_dtype), "W")
    mx.initializers["s_a"] = numpy_helper.from_array(np.array(a_scale, np.float32), "s_a")
    mx.initializers["zp"] = numpy_helper.from_array(np.array(zero_point, np.int8), "zp")
    mx.initializers["s_w"] = numpy_helper.from_array(np.ones(n, np.float32), "s_w")
    dq_scale_name = "s_a"
    if dq_scale is not None:
        mx.initializers["s_dq"] = numpy_helper.from_array(np.array(dq_scale, np.float32), "s_dq")
        dq_scale_name = "s_dq"

    mx.nodes["q"] = Node(
        "QuantizeLinear", "q", {},
        [ref("x", (m, k), TensorProto.FLOAT16), ref("s_a", ()), ref("zp", ())],
        [ref("q", (m, k), TensorProto.INT8)],
    )
    mx.nodes["dq"] = Node(
        "DequantizeLinear", "dq", {},
        [ref("q", (m, k)), ref(dq_scale_name, ()), ref("zp", ())],
        [ref("dq", (m, k), TensorProto.FLOAT)],
    )
    a_name = "dq"
    if with_cast:
        mx.nodes["cast"] = Node(
            "Cast", "cast", {"to": TensorProto.FLOAT16},
            [ref("dq", (m, k))],
            [ref("a", (m, k), TensorProto.FLOAT16)],
        )
        a_name = "a"
    mx.nodes["mm"] = Node(
        "MatMul", "mm", {"transB": transB},
        [ref(a_name, (m, k)), ref("W", (k, n)), ref("s_w", (n,))],
        [ref("y", (m, n), out_dtype)],
    )
    if extra_dq_consumer:
        mx.nodes["relu"] = Node(
            "Relu", "relu", {}, [ref("dq", (m, k))], [ref("r", (m, k), out_dtype)]
        )
    mx.build_dependencies()
    return mx


def fold(mx):
    with contextlib.redirect_stdout(io.StringIO()):
        return W8A8ActivationFold.fold(mx)


def test_folds_int8_a_and_records_scale():
    mx = build()
    assert fold(mx) == 1

    mm = mx.nodes["mm"]
    assert mm.inputs[0]["name"] == "q"
    assert mm.inputs[0]["dtype"] == TensorProto.INT8
    assert mm.inputs[0]["shape"] == [1, 512]
    # The kernel's epilogue multiplies by this; a missing attribute halves the
    # result for this graph.
    assert float(mm.attributes["a_scale"]) == pytest.approx(A_SCALE)
    # The weight contract is untouched: [W_int8, scale] still trailing.
    assert [i["name"] for i in mm.inputs[1:]] == ["W", "s_w"]
    # The pair's DQ only fed this MatMul, so it is gone.
    assert "dq" not in mx.nodes


def test_looks_through_fp16_cast():
    mx = build(with_cast=True)
    assert fold(mx) == 1
    assert mx.nodes["mm"].inputs[0]["name"] == "q"
    assert "cast" not in mx.nodes
    assert "dq" not in mx.nodes


def test_moves_the_pair_in_front_of_a_shape_view():
    """A Reshape between the pair and the MatMul is a storage alias, so the
    quantization moves to its input and the view copies bytes."""
    mx = build(with_cast=True)
    mx.nodes["view"] = Node(
        "Reshape", "view", {},
        [ref("a", (1, 512)), ref("newshape", (2,))],
        [ref("a_v", (1, 512), TensorProto.FLOAT16)],
    )
    mx.nodes["mm"].inputs[0] = ref("a_v", (1, 512))
    mx.build_dependencies()

    assert fold(mx) == 1
    assert mx.nodes["mm"].inputs[0]["name"] == "a_v"
    assert mx.nodes["mm"].inputs[0]["dtype"] == TensorProto.INT8
    view = mx.nodes["view"]
    assert view.inputs[0]["name"] == "q"
    assert view.inputs[0]["dtype"] == TensorProto.INT8
    assert view.outputs[0]["dtype"] == TensorProto.INT8
    assert float(mx.nodes["mm"].attributes["a_scale"]) == pytest.approx(A_SCALE)
    assert "cast" not in mx.nodes
    assert "dq" not in mx.nodes


def test_moves_the_pair_in_front_of_a_three_dim_squeeze():
    # [1,1,K] -> Squeeze -> [1,K], the row the MatMul wants.
    mx = build(with_cast=True)
    mx.nodes["q"].inputs[0] = ref("x", (1, 1, 512), TensorProto.FLOAT16)
    mx.nodes["q"].outputs[0] = ref("q", (1, 1, 512), TensorProto.INT8)
    mx.nodes["dq"].inputs[0] = ref("q", (1, 1, 512))
    mx.nodes["dq"].outputs[0] = ref("dq", (1, 1, 512), TensorProto.FLOAT)
    mx.nodes["cast"].inputs[0] = ref("dq", (1, 1, 512))
    mx.nodes["cast"].outputs[0] = ref("a", (1, 1, 512), TensorProto.FLOAT16)
    mx.nodes["view"] = Node(
        "Squeeze", "view", {},
        [ref("a", (1, 1, 512))],
        [ref("a_v", (1, 512), TensorProto.FLOAT16)],
    )
    mx.nodes["mm"].inputs[0] = ref("a_v", (1, 512))
    mx.build_dependencies()

    assert fold(mx) == 1
    assert mx.nodes["mm"].inputs[0]["name"] == "a_v"
    view = mx.nodes["view"]
    assert view.inputs[0]["name"] == "q"
    # The view keeps reading the shape the pair quantized, not the squeezed one.
    assert view.inputs[0]["shape"] == [1, 1, 512]
    assert view.outputs[0]["dtype"] == TensorProto.INT8


def test_refuses_a_view_that_another_node_reads():
    mx = build(with_cast=True)
    mx.nodes["view"] = Node(
        "Reshape", "view", {}, [ref("a", (1, 512))], [ref("a_v", (1, 512))]
    )
    mx.nodes["mm"].inputs[0] = ref("a_v", (1, 512))
    mx.nodes["relu"] = Node(
        "Relu", "relu", {}, [ref("a_v", (1, 512))], [ref("r", (1, 512))]
    )
    mx.build_dependencies()

    assert fold(mx) == 0
    # The float view still has a reader, so the pair must stay intact.
    assert "dq" in mx.nodes and "cast" in mx.nodes
    assert "a_scale" not in mx.nodes["mm"].attributes


def test_refuses_a_word_mover_between_the_pair_and_the_matmul():
    """Transpose copies elements word-wise and rejects 1-byte elements, so an
    int8 A through it would throw at execute."""
    mx = build(with_cast=True)
    mx.nodes["view"] = Node(
        "Transpose", "view", {"perm": [0, 1]},
        [ref("a", (1, 512))], [ref("a_v", (1, 512))],
    )
    mx.nodes["mm"].inputs[0] = ref("a_v", (1, 512))
    mx.build_dependencies()

    assert fold(mx) == 0
    assert "dq" in mx.nodes
    assert "a_scale" not in mx.nodes["mm"].attributes


def test_keeps_dq_with_another_consumer():
    mx = build(extra_dq_consumer=True)
    assert fold(mx) == 1
    assert mx.nodes["mm"].inputs[0]["name"] == "q"
    # relu still reads the float tensor, so the DQ (and the Q feeding it) stay.
    assert "dq" in mx.nodes
    assert mx.nodes["relu"].inputs[0]["name"] == "dq"


def assert_unchanged(mx):
    mm = mx.nodes["mm"]
    assert mm.inputs[0]["name"] in ("dq", "a")
    assert "a_scale" not in mm.attributes
    assert "dq" in mx.nodes


@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param({"zero_point": 3}, id="asymmetric-grid"),
        pytest.param({"m": 9}, id="m-above-8"),
        pytest.param({"m": -1}, id="dynamic-m"),
        pytest.param({"k": 128}, id="k-below-256"),
        pytest.param({"k": 510}, id="k-not-multiple-of-4"),
        pytest.param({"n": 250}, id="n-not-multiple-of-4"),
        pytest.param({"out_dtype": TensorProto.FLOAT}, id="fp32-output"),
        pytest.param({"transB": 1}, id="transposed-weight"),
        pytest.param({"weight_dtype": np.float16}, id="float-weight"),
        pytest.param({"a_scale": 0.0}, id="zero-scale"),
        pytest.param({"dq_scale": 0.25}, id="dq-scale-disagrees"),
    ],
)
def test_gate_refuses_and_leaves_graph_alone(kwargs):
    mx = build(**kwargs)
    assert fold(mx) == 0
    assert_unchanged(mx)


def test_refuses_fused_epi_bias():
    # ksplit_w8a8_dp4a has no bias slot in its epilogue; folding here would
    # drop the bias, not just change the route.
    mx = build()
    mx.nodes["mm"].attributes["epi_bias"] = 1
    assert fold(mx) == 0
    assert_unchanged(mx)


def test_refuses_a_swiglu_down_act_node():
    """The down_act row sits above the DP4A row in KernelSelect's table and
    its kernel reads a float A, so folding here leaves a model that loads and
    then throws."""
    mx = build()
    mm = mx.nodes["mm"]
    mm.inputs.append(ref("Wg", (256, 512)))
    mm.inputs.append(ref("Wu", (256, 512)))
    mm.attributes["down_act"] = [1]
    mx.build_dependencies()
    assert fold(mx) == 0
    assert_unchanged(mx)


def test_refuses_an_attention_softmax_node():
    """attn_softmax pre-empts every other route, DP4A included."""
    mx = build()
    mx.nodes["mm"].attributes["attn_softmax"] = 1
    assert fold(mx) == 0
    assert_unchanged(mx)


def test_three_dim_static_a_with_small_batch_times_seq_folds():
    mx = build(m=2)
    # 2 x 4 = 8 is the last admissible batch*m; shape it as [2, 4, K].
    mx.nodes["q"].inputs[0] = ref("x", (2, 4, 512), TensorProto.FLOAT16)
    mx.nodes["q"].outputs[0] = ref("q", (2, 4, 512), TensorProto.INT8)
    mx.nodes["dq"].inputs[0] = ref("q", (2, 4, 512))
    mx.nodes["dq"].outputs[0] = ref("dq", (2, 4, 512), TensorProto.FLOAT)
    mx.nodes["mm"].inputs[0] = ref("dq", (2, 4, 512))
    mx.nodes["mm"].outputs[0] = ref("y", (2, 4, 256), TensorProto.FLOAT16)
    mx.build_dependencies()

    assert fold(mx) == 1
    assert mx.nodes["mm"].inputs[0]["name"] == "q"
    assert mx.nodes["mm"].inputs[0]["shape"] == [2, 4, 512]
