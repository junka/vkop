"""int4 / nf4 weight-only group quantization for MatMul
(Quantizer.quantize_to_4bit_weight_only) and the packed payload's blob layout.

The buffer MatMul kernels read a 4-bit weight as two nibbles per byte, the even
element in the LOW nibble, in row-major [K, N] order, plus one fp32 scale per
(output column, slice of K) at binding 4 — a [K/group_size, N] table. So the
grouping axis, the nibble order and the byte count the writer records all have
to agree with the shader, and a weight the shader cannot address has to stay
float here rather than land in a graph whose only possible answer is a wrong
one:

  * MatMul only — Gemm/Conv have no nibble kernel;
  * a plain 2-D B operand with transB == 0 — [N, K] would put a column's nibbles
    along K instead of across it;
  * N % 8 == 0 — a packed row must start on a 32-bit word;
  * K % group_size == 0 — a group is a whole number of rows.

These cases pin the payload bytes, the scale table, the error bound, and every
gate above.
"""
import contextlib
import io
import sys

sys.path.insert(0, "model/pypi")

import numpy
from onnx import TensorProto, numpy_helper

from onnx2vkop.dag import (
    DAGBasedModel,
    Node,
    _init_byte_len,
    _init_bytes,
    _init_dtype_name,
)
from onnx2vkop.optimizer import Quantizer

# The shader's codebook, restated here on purpose: the converter and the kernel
# have to agree on it, and sharing one Python list would only prove that the
# same file was read twice.
NF4_CODEBOOK = numpy.array(
    [
        -1.0, -0.6961928009986877, -0.5250730514526367, -0.3949174189567566,
        -0.2844413814544678, -0.1847814998626709, -0.0910967006323811, 0.0,
        0.0795802986717224, 0.1601973110246658, 0.2447470557689667,
        0.3361703515052795, 0.4407098295211792, 0.5626170039176941,
        0.7229568369388580, 1.0,
    ],
    dtype=numpy.float32,
)


def tref(name):
    return {"name": name, "shape": [-1]}


def build(weight_name, weight, consumers, dtype=numpy.float32):
    m = DAGBasedModel()
    m.initializers[weight_name] = numpy_helper.from_array(
        numpy.asarray(weight, dtype=dtype), weight_name
    )
    for op_type, node_name, attrs, ins in consumers:
        m.nodes[node_name] = Node(
            op_type, node_name, dict(attrs), [tref(i) for i in ins],
            [tref(node_name + "_out")],
        )
    return m


def quantize(m, fmt="int4", group_size=64):
    with contextlib.redirect_stdout(io.StringIO()):
        Quantizer.quantize_to_4bit_weight_only(m, group_size, fmt)
    return m


def unpack(raw, count):
    """The kernel's nibble walk: element i is the low nibble of byte i/2 when i
    is even."""
    bytes_arr = numpy.frombuffer(raw, dtype=numpy.uint8)
    lo = bytes_arr & 0xF
    hi = bytes_arr >> 4
    out = numpy.empty(count, dtype=numpy.uint8)
    out[0::2] = lo
    out[1::2] = hi
    return out


def codes_signed(nibbles):
    """A 4-bit two's-complement nibble read back as int4."""
    vals = nibbles.astype(numpy.int32)
    return numpy.where(vals >= 8, vals - 16, vals)


def expected_scale(w, group_size, fmt):
    k, n = w.shape
    w3 = w.reshape(k // group_size, group_size, n)
    amax = numpy.max(numpy.abs(w3), axis=1)
    amax = numpy.where(amax == 0, 1.0, amax)
    return (amax / 7.0 if fmt == "int4" else amax).astype(numpy.float32)


def check_payload(m, w, group_size, fmt, name="W"):
    init = m.initializers[name]
    k, n = w.shape
    n_groups = k // group_size
    scale = numpy_helper.to_array(m.initializers[f"{name}_scale"])
    assert scale.dtype == numpy.float32
    assert scale.shape == (n_groups, n), f"scale is {scale.shape}, wants {(n_groups, n)}"
    numpy.testing.assert_allclose(scale, expected_scale(w, group_size, fmt), rtol=1e-6)

    nib = unpack(init.raw_data, k * n)
    # The grouped view is what the scale broadcasts over; a (K, N) operand times
    # an (n_groups, 1, N) scale would silently broadcast into a 3-D product.
    w3 = w.reshape(n_groups, group_size, n)
    grouped = (n_groups, group_size, n)
    if fmt == "int4":
        codes = codes_signed(nib).reshape(grouped)
        deq = codes.astype(numpy.float32) * scale.reshape(n_groups, 1, n)
        # Symmetric int4 never uses -8: the step is amax/7, so |w/scale| <= 7.
        assert codes.min() >= -7, f"int4 reached {codes.min()}, past the +7 step"
        assert codes.max() <= 7
    else:
        assert nib.min() >= 0 and nib.max() <= 15, "nf4 nibble must be a codebook index"
        deq = (NF4_CODEBOOK[nib.astype(numpy.int64).reshape(grouped)] *
               scale.reshape(n_groups, 1, n))

    err = numpy.abs(deq - w3)
    # Half the widest step a group can take: int4's step is scale, nf4's is the
    # largest codebook gap times scale. Anything above it means the scale or the
    # nibble was grouped along the wrong axis.
    step = (scale.reshape(n_groups, 1, n) if fmt == "int4" else
            numpy.diff(NF4_CODEBOOK).max() * scale.reshape(n_groups, 1, n))
    bound = 0.5 * step + 1e-6
    assert numpy.all(err <= bound), (
        f"max dequant error {err.max()} exceeds half-step {bound.max()}")

    # And the blob layout the C++ loader cross-checks against dims x dtype.
    assert _init_byte_len(init) == (k * n + 1) // 2, "packed payload is half a byte/value"
    assert len(_init_bytes(init)) == _init_byte_len(init)
    return scale, deq


def matmul_case(w, group_size=4, fmt="int4"):
    k, n = w.shape
    m = build("W", w, [("MatMul", "mm", {"transB": 0}, ["x", "W"])])
    q = quantize(m, fmt, group_size)
    assert [i["name"] for i in q.nodes["mm"].inputs] == ["x", "W", "W_scale"], \
        "the scale must be appended last"
    return q


def test_int4_payload_and_scale():
    """Row-major nibble order, low nibble first, one scale per (group, column)."""
    numpy.random.seed(0)
    k, n, g = 8, 16, 4
    w = (numpy.random.randn(k, n) * numpy.array(
        [3.0, 0.5, 1.2, 2.4, 0.7, 1.9, 0.3, 4.1], dtype=numpy.float32).reshape(k, 1)
        ).astype(numpy.float32)
    q = matmul_case(w, g)
    init = q.initializers["W"]
    assert init.data_type == TensorProto.INT4
    assert list(init.dims) == [k, n], "dims keep describing the logical [K, N]"
    assert _init_dtype_name(init) == "int4"
    assert len(init.raw_data) == k * n // 2
    check_payload(q, w, g, "int4")
    # The first byte really is element 0 in its low nibble.
    low_nibble = init.raw_data[0] & 0xF
    col0 = w[0, 0] / (expected_scale(w, g, "int4")[0, 0])
    assert low_nibble == int(numpy.round(col0)) & 0xF, (
        f"byte 0 low nibble is {low_nibble:x}, element 0 codes as "
        f"{int(numpy.round(col0)):x}")
    print("  PASS (int4: packed nibbles, per-group scale, half-step error)")


def test_nf4_payload_is_a_codebook_index():
    numpy.random.seed(1)
    k, n, g = 8, 8, 8
    w = numpy.random.randn(k, n).astype(numpy.float32)
    q = matmul_case(w, g, "nf4")
    init = q.initializers["W"]
    # UINT4 is the storage; the name that reaches the loader says what the
    # nibble means.
    assert init.data_type == TensorProto.UINT4
    assert _init_dtype_name(init) == "nf4"
    assert [(e.key, e.value) for e in init.metadata_props] == [("vkop_dtype", "nf4")]
    check_payload(q, w, g, "nf4")
    # A weight that is exactly one of its group's codebook values must round-trip
    # with zero error. Row i is codebook[i] x that column's absmax, so with one
    # group (group_size == K) the normalized value is codebook[i] itself and the
    # code has to be i.
    kk, nn = 16, 8
    amps = numpy.array([0.5, 1.0, 2.0, 0.25, 3.0, 1.5, 0.75, 4.0], dtype=numpy.float32)
    w_exact = (NF4_CODEBOOK[:, None] * amps[None, :]).astype(numpy.float32)
    q2 = matmul_case(w_exact, kk, "nf4")
    nib = unpack(q2.initializers["W"].raw_data, kk * nn)
    expected_codes = numpy.repeat(numpy.arange(kk, dtype=numpy.uint8), nn)
    assert numpy.array_equal(nib, expected_codes), (
        f"an exact codebook value coded as {nib[:kk].tolist()}, wants "
        f"{list(range(kk))}")
    scale2 = numpy_helper.to_array(q2.initializers["W_scale"])
    numpy.testing.assert_allclose(
        (NF4_CODEBOOK[nib.astype(numpy.int64).reshape(kk, nn)] * scale2.reshape(1, nn)),
        w_exact, rtol=0, atol=1e-7)
    print("  PASS (nf4: UINT4 storage named nf4, codes are codebook indices)")


def test_int4_and_nf4_share_the_byte_layout():
    """Same shape, same grouping: the two formats differ only in what the nibble
    means, so the byte count and the element order must not move."""
    numpy.random.seed(2)
    w = numpy.random.randn(8, 8).astype(numpy.float32)
    i4 = matmul_case(w, 4, "int4").initializers["W"]
    nf = matmul_case(w, 4, "nf4").initializers["W"]
    assert len(i4.raw_data) == len(nf.raw_data) == 32
    assert list(i4.dims) == list(nf.dims) == [8, 8]
    print("  PASS (int4/nf4 agree on byte count and element order)")


def test_fp16_source_quantizes_like_its_fp32_twin():
    numpy.random.seed(3)
    w16 = numpy.random.randn(8, 16).astype(numpy.float16)
    consumers = [("MatMul", "mm", {"transB": 0}, ["x", "W"])]
    m16 = quantize(build("W", w16, consumers, dtype=numpy.float16), "int4", 4)
    assert m16.initializers["W"].data_type == TensorProto.INT4
    m32 = quantize(build("W", w16.astype(numpy.float32), consumers), "int4", 4)
    assert m16.initializers["W"].raw_data == m32.initializers["W"].raw_data
    numpy.testing.assert_array_equal(
        numpy_helper.to_array(m16.initializers["W_scale"]),
        numpy_helper.to_array(m32.initializers["W_scale"]))
    print("  PASS (fp16 source lands on the fp32 twin's payload and scale)")


def test_gates_preserve_the_float_weight():
    """Every layout the nibble kernel cannot address stays float: no payload
    rewrite, no scale initializer, no third input."""
    numpy.random.seed(4)
    cases = [
        ("Gemm transB=1 has no nibble kernel",
         build("W", numpy.random.randn(8, 4).astype(numpy.float32),
               [("Gemm", "g", {"transB": 1}, ["x", "W"])]),
         {"transB": 0}, 4, "Gemm"),
        ("Conv has no nibble kernel",
         build("W", numpy.random.randn(4, 4, 8, 8).astype(numpy.float32),
               [("Conv", "c", {}, ["x", "W"])]),
         {}, 4, "Conv"),
        ("transB=1 MatMul stores [N, K]",
         build("W", numpy.random.randn(8, 4).astype(numpy.float32),
               [("MatMul", "mm", {"transB": 1}, ["x", "W"])]),
         {"transB": 1}, 4, "MatMul"),
        ("rank-3 batched weight",
         build("W", numpy.random.randn(2, 8, 8).astype(numpy.float32),
               [("MatMul", "mm", {"transB": 0}, ["x", "W"])]),
         {"transB": 0}, 4, "MatMul"),
        ("A-side weight",
         build("W", numpy.random.randn(8, 8).astype(numpy.float32),
               [("MatMul", "mm", {"transB": 0}, ["W", "x"])]),
         {"transB": 0}, 4, "MatMul"),
        ("N % 8 != 0 leaves rows mid-word",
         build("W", numpy.random.randn(8, 12).astype(numpy.float32),
               [("MatMul", "mm", {"transB": 0}, ["x", "W"])]),
         {"transB": 0}, 4, "MatMul"),
        ("K % group_size != 0 cuts a group mid-row",
         build("W", numpy.random.randn(6, 8).astype(numpy.float32),
               [("MatMul", "mm", {"transB": 0}, ["x", "W"])]),
         {"transB": 0}, 4, "MatMul"),
        ("non-float source arrives already quantized",
         build("W", numpy.arange(64, dtype=numpy.int8).reshape(8, 8),
               [("MatMul", "mm", {"transB": 0}, ["x", "W"])], dtype=numpy.int8),
         {"transB": 0}, 4, "MatMul"),
    ]
    for label, m, _attrs, group, _op in cases:
        before = m.initializers["W"].data_type
        before_inputs = {
            n.name: [i["name"] for i in n.inputs] for n in m.nodes.values()
        }
        out = quantize(m, "int4", group)
        assert out.initializers["W"].data_type == before, f"{label}: payload was rewritten"
        assert "W_scale" not in out.initializers, f"{label}: a scale appeared"
        assert {n.name: [i["name"] for i in n.inputs] for n in out.nodes.values()} == \
            before_inputs, f"{label}: a scale input was appended"
        print(f"  PASS ({label})")


def test_odd_group_size_is_rejected_before_any_work():
    """The fp16 kernels walk A two halves per word, so an odd group would leave a
    tap straddling a word — the caller has to hear about it, not get a graph."""
    numpy.random.seed(5)
    m = build("W", numpy.random.randn(8, 8).astype(numpy.float32),
              [("MatMul", "mm", {"transB": 0}, ["x", "W"])])
    for bad in (0, -4, 3):
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                Quantizer.quantize_to_4bit_weight_only(m, bad, "int4")
            raise AssertionError(f"group_size={bad} was accepted")
        except ValueError as e:
            assert "group_size" in str(e)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            Quantizer.quantize_to_4bit_weight_only(m, 64, "fp4")
        raise AssertionError("format fp4 was accepted")
    except ValueError as e:
        assert "format" in str(e)
    print("  PASS (odd/zero group_size and an unknown format raise)")


def test_writer_records_the_packed_size_and_name():
    """The C++ loader cross-checks the recorded byte count against
    elem_bytes(dtype, prod(dims)) and refuses a mismatch, so the writer's size,
    bytes and dtype string have to be the packed ones — and a raw_data whose
    length disagrees with the dims is a broken file, not something to pad."""
    from onnx2vkop import dag as dag_module

    numpy.random.seed(6)
    q = matmul_case(numpy.random.randn(8, 16).astype(numpy.float32), 4, "int4")
    init = q.initializers["W"]
    assert dag_module._init_byte_len(init) == 64
    assert _init_dtype_name(init) == "int4"
    assert _init_bytes(init) == init.raw_data

    broken = TensorProto()
    broken.data_type = TensorProto.INT4
    broken.name = "W"
    broken.dims.extend([8, 16])
    broken.raw_data = init.raw_data[:8]
    try:
        _init_bytes(broken)
        raise AssertionError("a short packed payload was written anyway")
    except ValueError as e:
        assert "bytes" in str(e)

    # A float initializer is unaffected: same size, same name, same bytes.
    f32 = numpy_helper.from_array(numpy.arange(12, dtype=numpy.float32).reshape(3, 4), "s")
    assert _init_byte_len(f32) == 48 and _init_dtype_name(f32) == "float32"
    assert _init_bytes(f32) == numpy.arange(12, dtype=numpy.float32).tobytes()
    print("  PASS (writer: packed size, dtype name, raw bytes, mismatch raises)")


if __name__ == "__main__":
    for fn in (test_int4_payload_and_scale,
               test_nf4_payload_is_a_codebook_index,
               test_int4_and_nf4_share_the_byte_layout,
               test_fp16_source_quantizes_like_its_fp32_twin,
               test_gates_preserve_the_float_weight,
               test_odd_group_size_is_rejected_before_any_work,
               test_writer_records_the_packed_size_and_name):
        print(fn.__name__ + ":")
        fn()
    print("\nALL PASS")
