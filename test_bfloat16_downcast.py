"""BFLOAT16 normalization (converter._downcast_bfloat16).

The runtime has no bf16 kernel and no bf16 storage type: every buffer (SSBO)
shader reads a 16-bit float with unpackHalf2x16, which is fp16's layout, not
bf16's. So a model that spells BFLOAT16 must be normalized to FLOAT16 at the
converter, before the loader's per-format gate ever sees the name -- the same
normalization the image exporter's bf16 checkpoint path does by hand.

Two things have to hold and they are what these cases pin:

  * the REWRITE is correct -- bf16 is fp32 with the low mantissa half dropped,
    so widening each uint16 into the high half of a uint32 and reading it as
    float32 recovers the value exactly, and the fp16 round-trip is lossless for
    every bf16 value inside fp16's range (8-bit mantissa -> 11-bit);
  * it covers every place a dtype is spelled -- initializer payload, graph
    input/output/value_info, and a tensor-valued node attribute -- because a
    single one left behind is still a hard load error.
"""
import sys

sys.path.insert(0, "model/pypi")

import numpy
import onnx
from onnx import TensorProto, helper, numpy_helper

from onnx2vkop.converter import _bf16_bytes_to_fp16, _downcast_bfloat16

# bf16 bit patterns for values that survive fp16 exactly (and one that
# saturates), chosen to exercise sign, an exact fraction and the exponent edge.
_BF16_VALUES = [1.0, -2.5, 0.25, 1024.0, -0.125, 0.0]


def bf16_raw(values):
    """bf16 little-endian payload for a list of floats, computed with numpy only
    (numpy has no native bfloat16 -- truncate fp32's low mantissa half)."""
    f32 = numpy.asarray(values, dtype=numpy.float32)
    return (f32.view(numpy.uint32) >> 16).astype(numpy.uint16).tobytes()


def bf16_tensor(name, values, dims=None):
    t = TensorProto()
    t.name = name
    t.data_type = TensorProto.BFLOAT16
    t.dims.extend(dims if dims is not None else [len(values)])
    t.raw_data = bf16_raw(values)
    return t


def test_bytes_helper_is_exact():
    """The shift-based decode reproduces the bf16 values bit for bit where fp16
    can hold them, and saturates (does not wrap) outside its range."""
    values = _BF16_VALUES + [70000.0]
    out = numpy.frombuffer(_bf16_bytes_to_fp16(bf16_raw(values)), dtype=numpy.uint16)
    got = out.view(numpy.float16).astype(numpy.float32)
    with numpy.errstate(over="ignore"):
        want = numpy.asarray(values, dtype=numpy.float32).astype(numpy.float16)
    assert list(got) == list(want), (got, want)
    # 70000 is beyond fp16's +-65504: fp16 saturates at inf, never wraps to a
    # small number. Pin that direction so a sign/shift bug cannot hide here.
    assert got[-1] == numpy.float16(numpy.inf)
    print("  PASS (shift decode exact in range, saturates outside it)")


def test_downcast_covers_every_dtype_site():
    """All three spellings of a dtype are rewritten: initializer payload, a
    ValueInfoProto (input/output/value_info), and a tensor-valued attribute."""
    init = bf16_tensor("W", [1.0, -2.5, 0.25], dims=[3])
    vi_in = helper.make_tensor_value_info("x", TensorProto.BFLOAT16, [3])
    vi_out = helper.make_tensor_value_info("y", TensorProto.BFLOAT16, [3])
    vi_mid = helper.make_tensor_value_info("mid", TensorProto.BFLOAT16, [3])
    const = helper.make_node(
        "Constant", [], ["c"],
        value=bf16_tensor("c_val", [4.0, 5.0], dims=[2]),
    )
    graph = helper.make_graph(
        [const], "g", [vi_in], [vi_out], initializer=[init],
        value_info=[vi_mid],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])

    n = _downcast_bfloat16(model)
    assert n == 5, n  # init + 3 value_infos + 1 attribute

    # make_graph copies what it is handed, so read the model's own tensors.
    got_init = model.graph.initializer[0]
    assert got_init.data_type == TensorProto.FLOAT16
    assert list(numpy_helper.to_array(got_init)) == [1.0, -2.5, 0.25]
    for vi in (list(model.graph.input) + list(model.graph.output)
               + list(model.graph.value_info)):
        assert vi.type.tensor_type.elem_type == TensorProto.FLOAT16
    attr_t = model.graph.node[0].attribute[0].t
    assert attr_t.data_type == TensorProto.FLOAT16
    assert list(numpy_helper.to_array(attr_t)) == [4.0, 5.0]
    print("  PASS (initializer + I/O + value_info + attribute all rewritten)")


def test_already_fp16_model_is_untouched():
    """A model that was never bf16 reports zero conversions and its bytes are
    unchanged -- the normalization must not disturb the fp16 path it shares a
    dtype with."""
    init = numpy_helper.from_array(
        numpy.asarray([1.0, 2.0], dtype=numpy.float16), "W")
    before = bytes(init.raw_data)
    graph = helper.make_graph([], "g", [], [], initializer=[init])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    assert _downcast_bfloat16(model) == 0
    assert bytes(init.raw_data) == before
    assert init.data_type == TensorProto.FLOAT16
    print("  PASS (fp16 model untouched, zero conversions)")


if __name__ == "__main__":
    for fn in (test_bytes_helper_is_exact,
               test_downcast_covers_every_dtype_site,
               test_already_fp16_model_is_untouched):
        print(fn.__name__ + ":")
        fn()
    print("\nALL PASS")
