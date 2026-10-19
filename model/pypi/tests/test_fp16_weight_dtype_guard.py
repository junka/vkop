"""A weight may only be stored as fp16 when its op actually runs fp16
(Quantizer.quantize_to_fp16_selective's safe-weight branch).

The runtime picks an op's shader from its DATA input, not from the weight:

    core/runtime.cpp
      node_inputs[0]->dtype() == uint16_t -> "_f16_"
      node_inputs[0]->dtype() == float    -> "_f32_"
      (no typed input)                    -> precision_ (fp16 on the LLM path)
      op_fp16 = (node_inputs[0]->dtype() == typeid(uint16_t))

and then binds the weight to that same shader. So a fp16 weight behind an fp32
shader is not a rounding error: the shader reads a 768-byte operand out of a
384-byte buffer. `-q fp16` used to do exactly that to llm.onnx's
mrope_static.M — the model's only FLOAT initializer, whose consumer
/mrope_static/MatMul reads the fp32 Cast(to=float) -> Transpose chain — and
that single tensor silently corrupted every prefill position after 0.

The verdict is read off the dtype the parser recorded on the consumer's data
input (shape_inference fills it in for every intermediate), which is only
authoritative on the buffer backend: the legacy image backend forces fp16
input images whatever the file says.
"""
import contextlib
import io
import os
import tempfile

import numpy
import onnx
from onnx import TensorProto, helper, numpy_helper

from onnx2vkop.converter import ModelConverter
from onnx2vkop.dag import DAGBasedModel, Node
from onnx2vkop.optimizer import Quantizer


def tref(name, dtype=None):
    ref = {"name": name, "shape": [-1]}
    if dtype is not None:
        ref["dtype"] = dtype
    return ref


def build(initializer_name, initializer_dtype, consumers):
    """Model with one initializer and its consumers.

    consumers: list of (op_type, node_name, inputs) where each input is a ref.
    """
    m = DAGBasedModel()
    m.initializers[initializer_name] = numpy_helper.from_array(
        numpy.ones((4, 4), dtype=numpy.float32), initializer_name
    )
    if initializer_dtype is not None:
        # The pass skips anything that is not FLOAT outright, so a fp16
        # initializer only needs the declared type — the bytes are never read.
        m.initializers[initializer_name].data_type = initializer_dtype
    for op_type, node_name, ins in consumers:
        m.nodes[node_name] = Node(
            op_type, node_name, {}, list(ins), [tref(node_name + "_out")]
        )
    return m


def quantize(m, target_backend="buffer"):
    with contextlib.redirect_stdout(io.StringIO()):
        Quantizer.quantize_to_fp16_selective(m, target_backend)
    return m


def is_fp16(m):
    return m.initializers["W"].data_type == TensorProto.FLOAT16


# ---------------------------------------------------------------- guards ----

def test_fp32_data_input_keeps_the_weight_fp32():
    """The llm.onnx rope case: MatMul(fp32 activation, M)."""
    m = build("W", None, [("MatMul", "mm", [tref("act", TensorProto.FLOAT), tref("W")])])
    assert not is_fp16(quantize(m))


def test_fp16_data_input_quantizes_the_weight():
    """Every other weight in llm.onnx: the activation is already fp16."""
    m = build("W", None, [("MatMul", "mm", [tref("act", TensorProto.FLOAT16), tref("W")])])
    assert is_fp16(quantize(m))


def test_untyped_data_input_quantizes_the_weight():
    """No recorded dtype -> the runtime follows precision_ (fp16), so quantize.

    A tensor shape inference could not type has no declaration to contradict;
    assuming fp32 here would break fp16 models instead of fixing fp32 ones.
    """
    m = build("W", None, [("MatMul", "mm", [tref("act"), tref("W")])])
    assert is_fp16(quantize(m))


def test_all_initializer_inputs_quantize():
    """An op with no data flow at all (a folded constant) has no fp32 signal."""
    m = build("W", None, [("Gemm", "g", [tref("W"), tref("W2")])])
    m.initializers["W2"] = numpy_helper.from_array(
        numpy.ones((4, 4), dtype=numpy.float32), "W2"
    )
    assert is_fp16(quantize(m))


def test_one_fp32_consumer_among_many_blocks_the_downcast():
    """A shared weight must satisfy every consumer, not the first one."""
    m = build(
        "W",
        None,
        [
            ("MatMul", "a", [tref("x16", TensorProto.FLOAT16), tref("W")]),
            ("MatMul", "b", [tref("x32", TensorProto.FLOAT), tref("W")]),
        ],
    )
    assert not is_fp16(quantize(m))


def test_gemm_bias_follows_the_data_input():
    """Gemm(x, W, B): both operands see the same verdict from x."""
    for act_dtype, expect_fp16 in ((TensorProto.FLOAT, False), (TensorProto.FLOAT16, True)):
        m = build(
            "W",
            None,
            [("Gemm", "g", [tref("x", act_dtype), tref("W"), tref("B")])],
        )
        m.initializers["B"] = numpy_helper.from_array(
            numpy.ones((4,), dtype=numpy.float32), "B"
        )
        with contextlib.redirect_stdout(io.StringIO()):
            Quantizer.quantize_to_fp16_selective(m)
        assert (m.initializers["W"].data_type == TensorProto.FLOAT16) == expect_fp16
        assert (m.initializers["B"].data_type == TensorProto.FLOAT16) == expect_fp16


def test_image_backend_skips_the_guard():
    """The image backend forces fp16 input images, so the declaration is moot."""
    m = build("W", None, [("MatMul", "mm", [tref("act", TensorProto.FLOAT), tref("W")])])
    assert is_fp16(quantize(m, target_backend="image"))


def test_already_fp16_weight_is_untouched():
    """310 of llm.onnx's 311 initializers: the guard never even runs."""
    m = build(
        "W",
        TensorProto.FLOAT16,
        [("MatMul", "mm", [tref("act", TensorProto.FLOAT), tref("W")])],
    )
    assert is_fp16(quantize(m))


# ------------------------------------------------------- end to end ---------

def test_declared_dtype_reaches_the_dag_through_shape_inference():
    """Cast(to=float) -> Transpose -> MatMul: the whole chain, as exported.

    Guards the mechanism the fix leans on, not just the branch: the parser
    only sees this dtype because optimize_model runs shape_inference over an
    ONNX graph that carries no value_info of its own (llm.onnx has none).
    """
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "rope.onnx")
        graph = helper.make_graph(
            [
                helper.make_node("Cast", ["pos"], ["c"], to=TensorProto.FLOAT),
                helper.make_node("MatMul", ["c", "M"], ["y"]),
            ],
            "rope",
            [helper.make_tensor_value_info("pos", TensorProto.INT64, [1, 4])],
            [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 64])],
            [numpy_helper.from_array(numpy.zeros((4, 64), dtype=numpy.float32), "M")],
        )
        onnx.save(
            helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)]), path
        )

        dag = ModelConverter().parse_onnx_model(path, 1)
        rope = dag.nodes["MatMul_1"]
        assert rope.inputs[0]["dtype"] == TensorProto.FLOAT, rope.inputs

        with contextlib.redirect_stdout(io.StringIO()):
            Quantizer.quantize_to_fp16_selective(dag)
        # Stored fp32 -- storing fp16 here is what corrupted the LLM prefill.
        assert dag.initializers["M"].data_type == TensorProto.FLOAT
