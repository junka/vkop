"""int8 weight-only quantization for the GEMM family
(Quantizer.quantize_to_int8_weight_only).

Both buffer kernels read exactly ONE fp32 dequant scale per output column of
Y, so which physical axis gets reduced is decided by the weight's layout:
transB=0 stores B as [K, N] (reduce axis 0), transB=1 stores it as [N, K]
(reduce axis 1). A wrong axis is not an accuracy problem -- the scale then
indexes the wrong dimension and no kernel can recover the intended values, so
every ambiguous case stays FP32 instead of emitting a graph whose only
possible answer is a wrong one. These cases pin that decision table: what gets
quantized, along which axis, and where the scale lands in the node's inputs.
"""
import contextlib
import io
import sys

sys.path.insert(0, "model/pypi")

import numpy
from onnx import TensorProto, numpy_helper

from onnx2vkop.dag import DAGBasedModel, Node
from onnx2vkop.optimizer import Quantizer


def tref(name):
    return {"name": name, "shape": [-1]}


def build(weight_name, weight, consumers, dtype=numpy.float32):
    """Model with one initializer and its consuming nodes.

    consumers: list of (op_type, node_name, attributes, input_names).
    """
    m = DAGBasedModel()
    m.initializers[weight_name] = numpy_helper.from_array(
        numpy.asarray(weight, dtype=dtype), weight_name
    )
    for op_type, node_name, attrs, ins in consumers:
        m.nodes[node_name] = Node(
            op_type, node_name, dict(attrs), [tref(i) for i in ins], [tref(node_name + "_out")]
        )
    return m


def quantize(m):
    # The pass prints a per-tensor report; keep it out of the test log.
    with contextlib.redirect_stdout(io.StringIO()):
        Quantizer.quantize_to_int8_weight_only(m)
    return m


def dtype_of(m, name):
    return m.initializers[name].data_type


def input_names(node):
    return [i["name"] for i in node.inputs]


def check_roundtrip(orig, int8_arr, scale, reduce_axis):
    """Every element must sit within half a step of its own column scale.

    This is what fails loudly if the scale was computed along the wrong axis:
    the broadcast then mixes columns, and the far-from-max elements of a column
    are nowhere near the other column's step.
    """
    assert int8_arr.dtype == numpy.int8, f"payload is {int8_arr.dtype}, not int8"
    bshape = list(orig.shape)
    bshape[reduce_axis] = 1
    try:
        scale_b = scale.reshape(bshape)
    except ValueError:
        raise AssertionError(
            f"scale of {scale.size} entries cannot describe a {orig.shape} "
            f"weight reduced over axis {reduce_axis} (wants {bshape})")
    deq = int8_arr.astype(numpy.float32) * scale_b
    err = numpy.abs(deq - orig)
    bound = scale_b / 2.0 + 1e-6
    assert numpy.all(err <= bound), (
        f"max dequant error {err.max()} exceeds half-step {bound.max()}")
    return deq


def test_gemm_transb_quantized_per_output_column():
    """The folded-Transpose / nn.Linear signature: W is [N, K], transB=1.
    Columns of Y are ROWS of W, so the reduce axis is 1 and the scale has N
    entries."""
    numpy.random.seed(0)
    n, k = 8, 5
    w = numpy.random.randn(n, k).astype(numpy.float32) * numpy.array(
        [3.0, 0.5, 1.2, 2.4, 0.7, 1.9, 0.3, 4.1], dtype=numpy.float32
    ).reshape(n, 1)
    m = quantize(build("W", w, [("Gemm", "g", {"transB": 1}, ["x", "W"])]))
    assert dtype_of(m, "W") == TensorProto.INT8, "transB=1 Gemm weight must quantize"
    scale = numpy_helper.to_array(m.initializers["W_scale"])
    assert scale.shape == (n,), f"scale must have N={n} entries, got {scale.shape}"
    assert scale.dtype == numpy.float32
    assert input_names(m.nodes["g"]) == ["x", "W", "W_scale"], \
        f"scale must be appended LAST, got {input_names(m.nodes['g'])}"
    q = numpy_helper.to_array(m.initializers["W"])
    check_roundtrip(w, q, scale, reduce_axis=1)
    # And the scale really is that axis's amax/127.
    numpy.testing.assert_allclose(scale, numpy.abs(w).max(axis=1) / 127.0, rtol=1e-6)
    print("  PASS (Gemm transB=1: [N, K] reduced over K, scale length N)")


def test_matmul_transb0_quantized_per_output_column():
    """B is [K, N] (transB=0): the reduce axis is 0. Same N entries, same slot
    for the scale."""
    numpy.random.seed(1)
    k, n = 4, 6
    w = numpy.random.randn(k, n).astype(numpy.float32) * numpy.array(
        [2.5, 0.4, 3.1, 1.0], dtype=numpy.float32
    ).reshape(k, 1)
    m = quantize(build("W", w, [("MatMul", "mm", {"transB": 0}, ["x", "W"])]))
    assert dtype_of(m, "W") == TensorProto.INT8
    scale = numpy_helper.to_array(m.initializers["W_scale"])
    assert scale.shape == (n,), f"scale must have N={n} entries, got {scale.shape}"
    assert input_names(m.nodes["mm"]) == ["x", "W", "W_scale"]
    q = numpy_helper.to_array(m.initializers["W"])
    check_roundtrip(w, q, scale, reduce_axis=0)
    numpy.testing.assert_allclose(scale, numpy.abs(w).max(axis=0) / 127.0, rtol=1e-6)
    print("  PASS (MatMul transB=0: [K, N] reduced over K, scale length N)")


def test_a_operand_preserved():
    """A quantized weight only makes sense on the B side: the kernels index the
    scale by output column, and an A operand's columns are not the output's."""
    numpy.random.seed(2)
    w = numpy.random.randn(7, 3).astype(numpy.float32)
    m = quantize(build("W", w, [("MatMul", "mm", {"transB": 0}, ["W", "x"])]))
    assert dtype_of(m, "W") == TensorProto.FLOAT, "A-side weight must stay FP32"
    assert "W_scale" not in m.initializers
    assert input_names(m.nodes["mm"]) == ["W", "x"]
    print("  PASS (A-operand weight preserved as FP32)")


def test_batched_weight_preserved():
    """A rank-3 weight would need one scale vector per batch slice; the shaders
    read exactly one, so it stays FP32."""
    numpy.random.seed(3)
    w = numpy.random.randn(2, 4, 6).astype(numpy.float32)
    m = quantize(build("W", w, [("MatMul", "mm", {"transB": 0}, ["x", "W"])]))
    assert dtype_of(m, "W") == TensorProto.FLOAT, "batched weight must stay FP32"
    assert "W_scale" not in m.initializers
    print("  PASS (rank-3 batched weight preserved as FP32)")


def test_disagreeing_transb_preserved():
    """Two readers of one weight with different transB cannot share a single
    reduce axis -> nothing is decidable -> preserve."""
    numpy.random.seed(4)
    w = numpy.random.randn(6, 4).astype(numpy.float32)
    m = quantize(build("W", w, [
        ("MatMul", "mm0", {"transB": 0}, ["x", "W"]),
        ("MatMul", "mm1", {"transB": 1}, ["y", "W"]),
    ]))
    assert dtype_of(m, "W") == TensorProto.FLOAT, \
        "consumers that disagree on transB must leave the weight FP32"
    assert "W_scale" not in m.initializers
    print("  PASS (disagreeing transB preserved as FP32)")


def test_gemm_bias_preserved():
    """A 1-D Gemm bias is the third input, not a weight. Per-tensor
    quantization of it would reach a kernel that only reads float/half bias, so
    it must stay FP32 -- the B-operand/rank-2 gate above already covers it."""
    numpy.random.seed(5)
    bias = numpy.random.randn(8).astype(numpy.float32)
    m = quantize(build("b", bias, [("Gemm", "g", {"transB": 1}, ["x", "W", "b"])]))
    assert dtype_of(m, "b") == TensorProto.FLOAT, "Gemm bias must stay FP32"
    assert "b_scale" not in m.initializers
    assert input_names(m.nodes["g"]) == ["x", "W", "b"]
    print("  PASS (1-D Gemm bias preserved as FP32)")


def test_conv_bias_still_preserved():
    """The Conv/ConvTranspose bias slot predates the GEMM family and must keep
    working: a 1-D Conv bias stays FP32 and gets no scale input."""
    numpy.random.seed(6)
    bias = numpy.random.randn(4).astype(numpy.float32)
    m = quantize(build("b", bias, [("Conv", "c", {}, ["x", "W", "b"])]))
    assert dtype_of(m, "b") == TensorProto.FLOAT, "Conv bias must stay FP32"
    print("  PASS (Conv bias still preserved)")


def test_fp16_weight_quantizes_like_its_fp32_twin():
    """Every LLM export stores weights as FLOAT16, and the weight-only kernels
    are buffer ops — i.e. the models that can use int8 are exactly the ones the
    old FP32-only gate skipped. A half weight must quantize to the same payload
    and scale as the same values read back as float32 (fp16 -> fp32 is
    lossless), so nothing in the kernels or the error bound changes."""
    numpy.random.seed(7)
    n, k = 8, 5
    w16 = (numpy.random.randn(n, k) * numpy.array(
        [3.0, 0.5, 1.2, 2.4, 0.7, 1.9, 0.3, 4.1], dtype=numpy.float32).reshape(n, 1)
        ).astype(numpy.float16)
    consumers = [("MatMul", "mm", {"transB": 1}, ["x", "W"])]
    m16 = quantize(build("W", w16, consumers, dtype=numpy.float16))
    assert dtype_of(m16, "W") == TensorProto.INT8, "FP16 weight must quantize"
    scale16 = numpy_helper.to_array(m16.initializers["W_scale"])
    q16 = numpy_helper.to_array(m16.initializers["W"])
    assert scale16.dtype == numpy.float32
    assert scale16.shape == (n,)
    assert input_names(m16.nodes["mm"]) == ["x", "W", "W_scale"]
    check_roundtrip(w16.astype(numpy.float32), q16, scale16, reduce_axis=1)

    m32 = quantize(build("W", w16.astype(numpy.float32), consumers))
    numpy.testing.assert_array_equal(q16, numpy_helper.to_array(m32.initializers["W"]))
    numpy.testing.assert_array_equal(
        scale16, numpy_helper.to_array(m32.initializers["W_scale"]))
    print("  PASS (FP16 weight quantizes to the FP32 twin's payload and scale)")


def test_non_float_source_still_preserved():
    """Widening the gate to FP16 must not widen it past the formats no kernel
    reads: an int8 initializer arrives already quantized and gets no scale."""
    w = numpy.arange(12, dtype=numpy.int8).reshape(4, 3)
    m = quantize(build("W", w, [("MatMul", "mm", {"transB": 0}, ["x", "W"])],
                      dtype=numpy.int8))
    assert dtype_of(m, "W") == TensorProto.INT8
    assert "W_scale" not in m.initializers
    assert input_names(m.nodes["mm"]) == ["x", "W"]
    print("  PASS (non-float initializer preserved, no scale appended)")


if __name__ == "__main__":
    for fn in (test_gemm_transb_quantized_per_output_column,
               test_matmul_transb0_quantized_per_output_column,
               test_a_operand_preserved,
               test_batched_weight_preserved,
               test_disagreeing_transb_preserved,
               test_gemm_bias_preserved,
               test_conv_bias_still_preserved,
               test_fp16_weight_quantizes_like_its_fp32_twin,
               test_non_float_source_still_preserved):
        print(fn.__name__ + ":")
        fn()
    print("\nALL PASS")
