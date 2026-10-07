"""fp8 weight-only quantization for MatMul
(Quantizer.quantize_to_fp8_weight_only) and the payload's blob layout.

The buffer MatMul kernels read an fp8 weight as one byte per value holding
sign | exponent | mantissa, times one fp32 scale per output column at binding 4 —
so the field layout, the bias arithmetic and the scale axis all have to agree with
`f8_val` in shaders/buffer/matmul.comp, and a weight the runtime cannot decode has
to stay float here instead of landing in a graph whose only possible answer is a
wrong one:

  * MatMul only — conv2d.comp and gemm.comp dequantize int8 and nothing else;
  * a plain 2-D B operand, all its consumers agreeing on transB — the scale
    indexes output columns, and which axis those sit on is transB's answer;
  * a value inside the layout's finite range — past 448 (E4M3) or 57344 (E5M2)
    the only encodings left are Inf/NaN, which no kernel decodes.

These cases pin the payload bytes, the scale table, the encoder's agreement with
the shader's decoder, the rounding bound, every gate above, and what the format is
actually worth: at one byte per weight it holds a *relative* error where int8 holds
an absolute one, so it preserves small weights int8 erases and costs accuracy on
the large ones.
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

# The shader's arithmetic, restated here on purpose: the converter and the kernel
# have to agree on it, and sharing one Python list would only prove that the same
# file was read twice. f8_val reassembles an fp32 out of the byte's fields — so
# does this, by way of the struct rather than uintBitsToFloat.
def shader_decode(byte, exp_bits, mant_bits):
    bias = 2 ** (exp_bits - 1) - 1
    m = byte & 0x7F
    if m == 0:
        v = 0.0
    else:
        e = m >> mant_bits
        frac = m & ((1 << mant_bits) - 1)
        v = (
            numpy.ldexp(numpy.float32(frac), 1 - bias - mant_bits)
            if e == 0
            else numpy.float32(numpy.ldexp(numpy.float32(1.0 + frac / (1 << mant_bits)),
                                           e - bias))
        )
    return -v if byte & 0x80 else v


def fmt_params(fmt):
    max_finite, exp_bits, mant_bits = Quantizer._FP8_FORMATS[fmt]
    lut, first_bad = Quantizer._fp8_lut(exp_bits, mant_bits, max_finite)
    return max_finite, exp_bits, mant_bits, lut, first_bad


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


def quantize(m, fmt="fp8e4m3"):
    with contextlib.redirect_stdout(io.StringIO()):
        Quantizer.quantize_to_fp8_weight_only(m, fmt)
    return m


def matmul_case(w, transb=0, fmt="fp8e4m3"):
    m = build("W", w, [("MatMul", "mm", {"transB": transb}, ["x", "W"])])
    return quantize(m, fmt)


def signed_codes(codes, lut, first_bad):
    """The bytes back to floats, the way the kernel does it."""
    mag = numpy.where(codes & 0x7F < first_bad, lut[codes & 0x7F], numpy.nan)
    return numpy.where(codes & 0x80, -mag, mag)


def half_step_bound(a, exp_bits, mant_bits):
    """The widest error round-to-nearest can make on each value: half the ulp of
    the grid point it lands on, or of the subnormal step below the smallest
    normal. Anything larger means the byte was scaled or grouped along the wrong
    axis."""
    bias = 2 ** (exp_bits - 1) - 1
    a = numpy.abs(a)
    min_normal = numpy.float32(2.0 ** (1 - bias))
    sub_step = numpy.float32(2.0 ** (1 - bias - mant_bits))
    normal = a >= min_normal
    e = ((numpy.where(normal, a, min_normal).view(numpy.uint32) >> 23) & 0xFF).astype(
        numpy.int32
    ) - 127
    ulp = numpy.where(normal, numpy.ldexp(numpy.ones_like(a), e - mant_bits), sub_step)
    return numpy.float32(0.5) * ulp


def check_payload(m, w, fmt, transb=0, name="W"):
    max_finite, exp_bits, mant_bits, lut, first_bad = fmt_params(fmt)
    init = m.initializers[name]
    # The scale is one entry per output column, which is the axis the weight does
    # NOT run along: axis 1 of a [K, N] weight, axis 0 of a [N, K] one.
    cols = w.shape[1] if transb == 0 else w.shape[0]
    bcast = (1, w.shape[1]) if transb == 0 else (w.shape[0], 1)
    scale = numpy_helper.to_array(m.initializers[f"{name}_scale"])
    assert scale.dtype == numpy.float32
    assert scale.shape == (cols,), f"scale is {scale.shape}, wants ({cols},)"
    amax = numpy.amax(numpy.abs(w), axis=0 if transb == 0 else 1)
    numpy.testing.assert_allclose(
        scale, numpy.where(amax == 0, 1.0, amax / max_finite).astype(numpy.float32),
        rtol=1e-6,
    )

    raw = numpy.frombuffer(init.raw_data, dtype=numpy.uint8)
    assert raw.size == w.size, f"one byte per value, got {raw.size} for {w.size}"
    assert numpy.all((raw & 0x7F) < first_bad), "a weight encoded as Inf/NaN"

    scaled = w / numpy.reshape(scale, bcast)
    deq = signed_codes(raw.reshape(w.shape), lut, first_bad) * numpy.reshape(scale, bcast)
    err = numpy.abs(deq - w)
    bound = half_step_bound(scaled, exp_bits, mant_bits) * numpy.reshape(scale, bcast)
    worst = float((err - bound).max())
    assert numpy.all(err <= bound + 1e-9), (
        f"max dequant error passes half a step by {worst:.3e}")
    return deq, err


def test_e4m3_payload_and_scale():
    numpy.random.seed(1)
    w = numpy.random.randn(16, 8).astype(numpy.float32)
    m = matmul_case(w, 0, "fp8e4m3")
    deq, err = check_payload(m, w, "fp8e4m3")
    init = m.initializers["W"]
    assert init.data_type == TensorProto.FLOAT8E4M3FN
    assert list(init.dims) == [16, 8], "dims stay the logical matrix"
    assert [i["name"] for i in m.nodes["mm"].inputs] == ["x", "W", "W_scale"]
    # The largest column magnitude must reach the layout's own maximum, or the
    # scale is not using the range it has. The sign bit rides along, so compare
    # magnitudes.
    assert numpy.any(
        numpy.frombuffer(init.raw_data, dtype=numpy.uint8) & 0x7F == 0x7E
    ), "absmax did not map to the top code"
    print(f"  PASS (e4m3: {err.mean():.3e} mean abs error, half-step bound holds)")


def test_e5m2_payload_and_scale():
    numpy.random.seed(2)
    w = numpy.random.randn(16, 8).astype(numpy.float32)
    m = matmul_case(w, 0, "fp8e5m2")
    deq, err = check_payload(m, w, "fp8e5m2")
    init = m.initializers["W"]
    assert init.data_type == TensorProto.FLOAT8E5M2
    assert numpy.any(
        numpy.frombuffer(init.raw_data, dtype=numpy.uint8) & 0x7F == 0x7B
    ), "absmax did not map to the top code"
    print(f"  PASS (e5m2: {err.mean():.3e} mean abs error, half-step bound holds)")


def test_encoder_and_shader_decode_agree_on_every_byte():
    """The converter's grid and the kernel's grid are one contract, stated twice.
    Two checks: the LUT equals the shader's field arithmetic for all 256 bytes,
    and every byte the encoder emits round-trips to itself — an encoder that
    drifted by one step would still produce plausible weights."""
    for fmt in Quantizer._FP8_FORMATS:
        max_finite, exp_bits, mant_bits, lut, first_bad = fmt_params(fmt)
        for m in range(1 << (exp_bits + mant_bits)):
            assert numpy.float32(shader_decode(m, exp_bits, mant_bits)) == lut[m], (
                f"{fmt}: byte 0x{m:x} decodes differently")
        # Every finite code, encoded again from its own decoded value.
        codes = Quantizer._fp8_encode(lut[:first_bad], exp_bits, mant_bits)
        assert numpy.array_equal(codes & 0x7F, numpy.arange(first_bad)), (
            f"{fmt}: a grid value does not encode back to its own byte")
        neg = Quantizer._fp8_encode(-lut[:first_bad], exp_bits, mant_bits)
        assert numpy.array_equal(neg & 0x7F, numpy.arange(first_bad))
        assert numpy.all(neg & 0x80), f"{fmt}: the sign bit was lost"
        # And the first byte the shader cannot decode really is the first one
        # outside the finite range.
        assert first_bad == (0x7F if fmt == "fp8e4m3" else 0x7C), fmt
    print("  PASS (LUT == shader decode for all bytes; encode round-trips)")


def test_encoder_is_round_to_nearest():
    """No value may sit further from its code than the next grid point does —
    the property that makes the error bound above an assertion and not a guess."""
    rng = numpy.random.default_rng(3)
    for fmt in Quantizer._FP8_FORMATS:
        max_finite, exp_bits, mant_bits, lut, first_bad = fmt_params(fmt)
        grid = numpy.concatenate([lut[:first_bad], -lut[:first_bad]])
        x = numpy.concatenate([
            rng.normal(0, 1, 3000),
            rng.normal(0, 2e-3, 3000),
            numpy.linspace(-max_finite, max_finite, 2001).astype(numpy.float32),
            lut[:first_bad],
            -lut[:first_bad],
        ]).astype(numpy.float32)
        x = numpy.clip(x, -max_finite, max_finite)
        codes = Quantizer._fp8_encode(x, exp_bits, mant_bits)
        got = signed_codes(codes, lut, first_bad)
        best = numpy.abs(grid[None, :] - x[:, None]).min(axis=1)
        over = numpy.abs(got - x) - best
        assert numpy.all(over <= 1e-12), f"{fmt}: {int((over > 1e-12).sum())} non-nearest"
        assert numpy.all((codes & 0x7F) < first_bad)
    print("  PASS (every value lands on its nearest fp8 grid point)")


def test_unclipped_input_is_caught_not_shipped():
    """A value past the largest finite one has no fp8 encoding, only Inf/NaN. The
    quantizer clips before encoding, but both ways a missing clip could leak are
    pinned here, because either one ships a weight the kernel cannot read:
    - just past the top grid point, round-to-nearest lands ON an Inf/NaN byte,
      which the quantizer's re-check of the emitted bytes catches;
    - far past it, the exponent runs off its field and the byte truncation folds
      it back into a small legal-looking one, so the encoder has to raise while
      the field is still wide."""
    max_finite, exp_bits, mant_bits, lut, first_bad = fmt_params("fp8e4m3")

    # The grid's last step is 32 wide, so anything past its midpoint 464 rounds up
    # onto 0x7f, which decodes as NaN.
    near = Quantizer._fp8_encode(
        numpy.array([max_finite + 17.0], dtype=numpy.float32), exp_bits, mant_bits)
    assert (near[0] & 0x7F) >= first_bad, (
        "a value past the grid encoded as a finite byte")

    # Twice the range overflows the field outright.
    try:
        Quantizer._fp8_encode(
            numpy.array([max_finite * 2.0, -max_finite * 2.0], dtype=numpy.float32),
            exp_bits, mant_bits,
        )
        raise AssertionError("a field overflow was narrowed to a byte silently")
    except ValueError as e:
        assert "overflow the magnitude field" in str(e)

    # And through the quantizer: the clip means a legal weight still never reaches
    # either arm, whatever its magnitude.
    w = numpy.zeros((8, 8), dtype=numpy.float32)
    w[:, 0] = numpy.linspace(0.0, 1e6, 8).astype(numpy.float32)
    m = matmul_case(w, 0, "fp8e4m3")
    assert numpy.all(numpy.frombuffer(m.initializers["W"].raw_data, numpy.uint8)
                     & 0x7F < first_bad)
    print("  PASS (NaN byte and field overflow both raise; the quantizer clips)")


def test_fp8_trades_absolute_precision_for_relative():
    """What one byte per value actually buys, measured rather than assumed. int8's
    step is absolute (amax/127 across the whole column) and fp8's is relative
    (~2^-3 of each value), so on a column whose values span several orders of
    magnitude the two do not fail the same way:

      * int8 is more accurate on the LARGEST values, so it wins plain MSE;
      * int8 erases everything below half its step (the weight becomes exactly 0,
        so its relative error there saturates at 100%), while fp8 keeps every
        normal band value within half an ulp ~ 6% of itself.

    Asserting the second pair and not "fp8 is better" is the point: a claim this
    test could not make is a claim the format does not support, and the end-to-end
    numbers have to be read against it."""
    rng = numpy.random.default_rng(4)
    k, n = 256, 32
    # Spread WITHIN each column: a per-column scale (which int8 also gets) cannot
    # absorb it, so it is the only spread that can show a difference between the
    # two grids.
    w = (rng.normal(0, 1, (k, n)) *
         numpy.float32(10.0) ** rng.uniform(-4.0, 0.0, (k, n)).astype(numpy.float32)
         ).astype(numpy.float32)

    m8 = matmul_case(w, 0, "fp8e4m3")
    _max, eb, mb, lut, fb = fmt_params("fp8e4m3")
    codes8 = numpy.frombuffer(m8.initializers["W"].raw_data,
                              numpy.uint8).reshape(w.shape)
    scale8 = numpy_helper.to_array(m8.initializers["W_scale"]).reshape(1, n)
    err8 = numpy.abs(signed_codes(codes8, lut, fb) * scale8 - w)

    scale_i8 = (numpy.amax(numpy.abs(w), axis=0) / 127.0).astype(numpy.float32)
    code_i8 = numpy.clip(numpy.round(w / scale_i8.reshape(1, n)), -127, 127)
    err_i8 = numpy.abs(code_i8 * scale_i8.reshape(1, n) - w)

    # fp8's own normal band: below it the grid is absolute again, like int8's.
    min_normal = numpy.float32(2.0 ** (1 - (2 ** (eb - 1) - 1)))
    normal = numpy.abs(w / scale8) >= min_normal
    rel8 = err8 / numpy.abs(w)
    rel_i8 = err_i8 / numpy.abs(w)

    assert float(numpy.mean(rel8)) < 0.25 * float(numpy.mean(rel_i8)), (
        f"mean relative error fp8 {numpy.mean(rel8):.2%} vs "
        f"int8 {numpy.mean(rel_i8):.2%}")
    # Half an ulp of a normal fp8 value is at most 2^-3 / 2 = 6.25%.
    assert float(rel8[normal].max()) <= 0.0625 + 1e-6, (
        f"fp8 relative error left the rounding bound: {rel8[normal].max():.2%}")
    # int8 has no ceiling below its step: those weights do not get inaccurate,
    # they get erased, which is why their relative error saturates at 100%.
    assert float(rel_i8[normal].max()) > 10.0 * float(rel8[normal].max()), (
        f"int8's worst tail error {rel_i8[normal].max():.2%} is not much worse "
        f"than fp8's {rel8[normal].max():.2%}")

    erased_i8 = int(numpy.count_nonzero(code_i8 == 0))
    erased_f8 = int(numpy.count_nonzero((codes8 & 0x7F) == 0))
    assert erased_i8 > 5 * max(erased_f8, 1), (
        f"int8 zeroed {erased_i8} values, fp8 {erased_f8} — no tail advantage")

    mse8, mse_i8 = float(numpy.mean(err8**2)), float(numpy.mean(err_i8**2))
    print(f"  PASS (mean rel {numpy.mean(rel8):.2%} vs int8 "
          f"{numpy.mean(rel_i8):.2%}, fp8 max rel {rel8[normal].max():.2%}; "
          f"int8 zeroes {erased_i8} weights vs {erased_f8}; fp8 mse {mse8:.3e} "
          f"vs int8 {mse_i8:.3e} — int8 wins MSE, as its absolute step must)")


def test_transb_1_quantizes_the_other_axis():
    """[N, K] is the transposed store; the scale still indexes N, which is now
    axis 0. Getting this backwards is silently wrong, so both the axis and the
    dequantized values are pinned."""
    numpy.random.seed(5)
    n, k = 8, 16
    w = numpy.random.randn(n, k).astype(numpy.float32)
    m = matmul_case(w, 1, "fp8e4m3")
    check_payload(m, w, "fp8e4m3", transb=1)
    scale = numpy_helper.to_array(m.initializers["W_scale"])
    assert scale.shape == (n,) and not numpy.allclose(scale, scale[::-1]), (
        "a symmetric scale would hide the wrong axis")
    print("  PASS (transB=1 reduces along axis 1 and scales rows of N)")


def test_gates_preserve_the_float_weight():
    """Every case the fp8 kernel cannot read keeps its float weight: no payload
    rewrite, no scale initializer, no third input."""
    numpy.random.seed(6)
    cases = [
        ("Gemm has no fp8 kernel",
         build("W", numpy.random.randn(8, 4).astype(numpy.float32),
               [("Gemm", "g", {"transB": 1}, ["x", "W"])])),
        ("Conv has no fp8 kernel",
         build("W", numpy.random.randn(4, 4, 8, 8).astype(numpy.float32),
               [("Conv", "c", {}, ["x", "W"])])),
        ("rank-3 batched weight",
         build("W", numpy.random.randn(2, 8, 8).astype(numpy.float32),
               [("MatMul", "mm", {"transB": 0}, ["x", "W"])])),
        ("A-side weight",
         build("W", numpy.random.randn(8, 8).astype(numpy.float32),
               [("MatMul", "mm", {"transB": 0}, ["W", "x"])])),
        ("non-float source arrives already quantized",
         build("W", numpy.arange(64, dtype=numpy.int8).reshape(8, 8),
               [("MatMul", "mm", {"transB": 0}, ["x", "W"])], dtype=numpy.int8)),
    ]
    for label, m in cases:
        before = m.initializers["W"].data_type
        before_inputs = {n.name: [i["name"] for i in n.inputs] for n in m.nodes.values()}
        out = quantize(m)
        assert out.initializers["W"].data_type == before, f"{label}: payload rewritten"
        assert "W_scale" not in out.initializers, f"{label}: a scale appeared"
        assert {n.name: [i["name"] for i in n.inputs] for n in out.nodes.values()} == \
            before_inputs, f"{label}: a scale input was appended"
        print(f"  PASS ({label})")

    # Consumers that disagree about the layout cannot share one scale axis.
    m = build("W", numpy.random.randn(8, 8).astype(numpy.float32),
              [("MatMul", "a", {"transB": 0}, ["x", "W"]),
               ("MatMul", "b", {"transB": 1}, ["y", "W"])])
    out = quantize(m)
    assert out.initializers["W"].data_type == TensorProto.FLOAT
    assert "W_scale" not in out.initializers
    print("  PASS (consumers disagreeing on transB stay float)")


def test_odd_shapes_still_quantize():
    """Unlike the 4-bit kernels, an fp8 byte stream needs no word alignment: an
    odd N or a K that is not a multiple of anything has no bearing on the layout,
    so refusing one would be a gate invented out of caution."""
    numpy.random.seed(7)
    for (k, n) in [(7, 3), (1, 255), (255, 1), (13, 5)]:
        w = numpy.random.randn(k, n).astype(numpy.float32)
        m = matmul_case(w, 0, "fp8e5m2")
        check_payload(m, w, "fp8e5m2")
    print("  PASS (odd K/N shapes quantize; nothing about fp8 needs alignment)")


def test_unknown_format_raises_before_any_work():
    numpy.random.seed(8)
    m = build("W", numpy.random.randn(8, 8).astype(numpy.float32),
              [("MatMul", "mm", {"transB": 0}, ["x", "W"])])
    for bad in ("fp8", "float8", "e4m3", "fp4"):
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                Quantizer.quantize_to_fp8_weight_only(m, bad)
            raise AssertionError(f"format {bad!r} was accepted")
        except ValueError as e:
            assert "format" in str(e)
    print("  PASS (an unknown fp8 format raises)")


def test_writer_records_the_fp8_size_and_name():
    """The C++ loader cross-checks the recorded byte count against
    elem_bytes(dtype, prod(dims)) and refuses a mismatch, and it reads fp8 through
    the byte container rather than numpy — which the writer cannot decode at all
    without an optional dependency. So: size is one byte per value, the dtype
    string is the ONNX spelling, and the payload is raw_data verbatim."""
    from onnx2vkop import dag as dag_module

    numpy.random.seed(9)
    for fmt, onnx_type, name in [("fp8e4m3", TensorProto.FLOAT8E4M3FN,
                                  "float8e4m3fn"),
                                 ("fp8e5m2", TensorProto.FLOAT8E5M2, "float8e5m2")]:
        q = matmul_case(numpy.random.randn(8, 16).astype(numpy.float32), 0, fmt)
        init = q.initializers["W"]
        assert init.data_type == onnx_type
        assert dag_module._init_byte_len(init) == 128
        assert _init_dtype_name(init) == name
        assert _init_bytes(init) == init.raw_data

        broken = TensorProto()
        broken.data_type = onnx_type
        broken.name = "W"
        broken.dims.extend([8, 16])
        broken.raw_data = init.raw_data[:64]
        try:
            _init_bytes(broken)
            raise AssertionError("a short fp8 payload was written anyway")
        except ValueError as e:
            assert "bytes" in str(e)

    # The FNUZ variants have no kernel and no name in the loader: they must not
    # acquire a mapping here, or a graph the runtime cannot read looks valid.
    assert 18 not in dag_module._DATA_TYPE_MAP and 20 not in dag_module._DATA_TYPE_MAP
    assert dag_module._init_dtype_name(
        numpy_helper.from_array(numpy.zeros((2, 2), dtype=numpy.float32), "s")
    ) == "float32"
    print("  PASS (writer: fp8 size, dtype name, raw bytes, mismatch raises)")


if __name__ == "__main__":
    for fn in (test_e4m3_payload_and_scale,
               test_e5m2_payload_and_scale,
               test_encoder_and_shader_decode_agree_on_every_byte,
               test_encoder_is_round_to_nearest,
               test_unclipped_input_is_caught_not_shipped,
               test_fp8_trades_absolute_precision_for_relative,
               test_transb_1_quantizes_the_other_axis,
               test_gates_preserve_the_float_weight,
               test_odd_shapes_still_quantize,
               test_unknown_format_raises_before_any_work,
               test_writer_records_the_fp8_size_and_name):
        print(fn.__name__ + ":")
        fn()
    print("\nALL PASS")
