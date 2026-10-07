"""nvfp4 weight-only quantization for MatMul
(Quantizer.quantize_to_nvfp4_weight_only) and the payload's blob layout.

The buffer MatMul kernels read NVFP4 as three things at once — an E2M1 nibble per
value (two per byte, the even element low), one fp8 E4M3 *byte* per (16-value
block of K, output column) at binding 4, and one fp32 factor per tensor at
binding 5 — so the nibble grid, the block length, the e4m3 field layout, the scale
axis and the byte count the writer records all have to agree with `e2m1_val`,
`e4m3_val` and `scale4` in shaders/buffer/matmul.comp. A weight the runtime cannot
decode has to stay float here instead of landing in a graph whose only possible
answer is a wrong one:

  * MatMul only — Gemm/Conv have no nibble kernel;
  * a plain 2-D B operand with transB == 0 — [N, K] would put a column's nibbles
    along K instead of across it;
  * N % 8 == 0 — a packed row must start on a 32-bit word;
  * K % 16 == 0 — a block is 16 rows, and the format fixes that, not the caller;
  * a block scale that stays inside e4m3's finite range.

These cases pin all three tensors, the two encoders' agreement with the shader's
two decoders, the rounding and saturation bounds, every gate above, and what the
format is worth: 0.5625 bytes per value buys a 16-value group, which no fp32-scale
format can offer at that price.
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

BLOCK = Quantizer._NVFP4_BLOCK
MAX_F4 = Quantizer._E2M1_MAX
# e4m3's largest finite value: the per-tensor factor is chosen so the largest
# block scale lands exactly on it.
MAX_F8 = 448.0

NF4_CODEBOOK = numpy.array([
    -1.0, -0.6961928009986877, -0.5250730514526367, -0.3949174189567566,
    -0.2844413814544678, -0.1847814998626709, -0.0910967006323811, 0.0,
    0.0795802986717224, 0.1601973110246658, 0.2447470557689667,
    0.3361703515052795, 0.4407098295211792, 0.5626170039176941,
    0.7229568369388580, 1.0,
], dtype=numpy.float32)


# The shader's bit assembly, restated here on purpose: the converter and the
# kernel have to agree on it, and sharing one Python table would only prove that
# the same file was read twice. e2m1_val slides a normal code's fields into an
# fp32 and keeps only exponent 0 on its own subnormal term.
def shader_e2m1(nibble):
    m = nibble & 7
    if m <= 1:
        v = numpy.float32(m) * numpy.float32(0.5)
    else:
        e, mant = (m >> 1) - 1, m & 1     # exponent bias 1
        v = numpy.float32(numpy.ldexp(numpy.float32(1.0 + mant * 0.5), e))
    return -v if nibble & 8 else v


def shader_e4m3(byte):
    m = byte & 0x7F
    if m == 0:
        v = numpy.float32(0.0)
    else:
        e, frac = m >> 3, m & 7
        v = (
            numpy.float32(numpy.ldexp(numpy.float32(frac), -9))
            if e == 0
            else numpy.float32(numpy.ldexp(numpy.float32(1.0 + frac / 8.0), e - 7))
        )
    return -v if byte & 0x80 else v


def global_scale_of(w):
    return numpy.float32(numpy.amax(numpy.abs(w)) / (MAX_F8 * MAX_F4))


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


def quantize(m, group_size=BLOCK):
    with contextlib.redirect_stdout(io.StringIO()):
        Quantizer.quantize_to_nvfp4_weight_only(m, group_size)
    return m


def matmul_case(w, transb=0, group_size=BLOCK, dtype=numpy.float32):
    m = build("W", w, [("MatMul", "mm", {"transB": transb}, ["x", "W"])], dtype)
    return quantize(m, group_size)


def unpack(raw, count):
    """The kernel's nibble walk: element i is the low nibble of byte i/2 if even."""
    b = numpy.frombuffer(raw, dtype=numpy.uint8)
    out = numpy.empty(count, dtype=numpy.uint8)
    out[0::2] = b & 0xF
    out[1::2] = b >> 4
    return out


def kernel_decode(m, k, n, name="W"):
    """Decode the three tensors the way the two kernels do, down to the nibble and
    byte arithmetic, and return the codes and the per-value scale separately: a
    disagreement between converter and shader then shows up as a wrong weight,
    not as a matching pair of tables. A 4-bit payload's raw bytes are shared, so
    int4 and nf4 initializers can be read through it too."""
    nib = unpack(m.initializers[name].raw_data, k * n)
    raw_scale = numpy.frombuffer(
        m.initializers[f"{name}_scale"].raw_data, numpy.uint8)
    g = 1.0
    if name in m.initializers and f"{name}_scale_global" in m.initializers:
        g = float(numpy_helper.to_array(m.initializers[f"{name}_scale_global"])[0])
    n_rows = max(k // BLOCK, 1)
    block = numpy.array([[shader_e4m3(int(b)) for b in row]
                         for row in raw_scale.reshape(n_rows, n)],
                        dtype=numpy.float32)
    s = numpy.repeat(block, BLOCK, axis=0) * g
    codes = numpy.array([shader_e2m1(int(c)) for c in nib],
                        dtype=numpy.float32).reshape(k, n)
    return codes, s


def check_payload(m, w, name="W"):
    k, n = w.shape
    assert k % BLOCK == 0 and n % 8 == 0

    init = m.initializers[name]
    assert init.data_type == TensorProto.FLOAT4E2M1
    assert list(init.dims) == [k, n], "dims stay the logical matrix"
    assert len(init.raw_data) == k * n // 2, "two nibbles per byte"

    scale = m.initializers[f"{name}_scale"]
    assert scale.data_type == TensorProto.FLOAT8E4M3FN
    assert list(scale.dims) == [k // BLOCK, n]
    assert len(scale.raw_data) == (k // BLOCK) * n, "one byte per block per column"
    assert numpy.all((numpy.frombuffer(scale.raw_data, numpy.uint8) & 0x7F) < 0x7F), \
        "a block scale encoded as e4m3 Inf or NaN"

    g_init = m.initializers[f"{name}_scale_global"]
    assert g_init.data_type == TensorProto.FLOAT
    assert list(g_init.dims) == [1]
    numpy.testing.assert_allclose(
        numpy_helper.to_array(g_init), [global_scale_of(w)], rtol=1e-6)

    assert [i["name"] for i in m.nodes["mm"].inputs] == \
        ["x", "W", "W_scale", "W_scale_global"]

    # The block scale reduces along K and indexes N: one entry per 16 rows of K
    # and output column, which is the axis a [K, N] weight does not run along.
    w3 = w.reshape(k // BLOCK, BLOCK, n)
    amax_b = numpy.max(numpy.abs(w3), axis=1)
    g = float(numpy_helper.to_array(g_init)[0])
    s_real = amax_b / (MAX_F4 * g)
    assert float(s_real.max()) <= MAX_F8 + 1e-3, (
        f"a block scale leaves e4m3's range: {s_real.max()}")

    codes, s = kernel_decode(m, k, n, name)
    deq = codes * s
    s3 = s.reshape(k // BLOCK, BLOCK, n)
    assert not numpy.any((s3[:, 0, :] == 0) & (amax_b > 0)), (
        "a block holding values got a scale of zero and was flattened")

    # No value may sit further from its code than the next grid point does,
    # measured on the scale the kernel actually applies. Saturation is inside the
    # same bound: the applied byte is within half an e4m3 step (1/16 relative) of
    # absmax / 6, so the block's top reaches 6.375 grid steps and pays 3/8 of one,
    # while the widest half-step is 1.
    grid = Quantizer._E2M1_GRID
    a_over = numpy.abs(w3) / numpy.where(s3 > 0, s3, numpy.float32(1.0))
    interval = numpy.clip(
        numpy.searchsorted(grid, numpy.minimum(a_over, grid[-1]), side="right") - 1,
        0, len(grid) - 2).reshape(a_over.shape)
    bound = numpy.float32(0.5) * (grid[interval + 1] - grid[interval]) * s3
    err3 = numpy.abs(deq - w).reshape(w3.shape)
    worst = float((err3 - bound).max())
    assert worst <= 1e-6, f"dequant error passes half a grid step by {worst:.3e}"
    return deq, numpy.abs(deq - w)


def test_three_tensors_and_their_layout():
    numpy.random.seed(1)
    w = numpy.random.randn(32, 8).astype(numpy.float32)
    m = matmul_case(w)
    check_payload(m, w)
    # A block's absmax has to reach the top of the e2m1 grid, or the scale is
    # wasting the range it was given. The sign bit rides along, so compare the
    # magnitude field.
    nib = unpack(m.initializers["W"].raw_data, 32 * 8)
    assert numpy.any(nib & 0x7 == 7), "no value mapped to the top code"
    print("  PASS (payload [32,8] fp4, block scale [2,8] e4m3, global fp32)")


def test_e2m1_encoder_matches_the_shader_decoder():
    """Two statements of one contract. Every nibble has to read back as the value
    the kernel's bit arithmetic produces, and a grid value re-encoded has to give
    its own nibble — a table that drifted one step would otherwise ship weights
    the runtime misreads."""
    codes = numpy.arange(16, dtype=numpy.uint8)
    from_table = Quantizer._e2m1_decode(codes)
    from_bits = numpy.array([shader_e2m1(int(c)) for c in codes], dtype=numpy.float32)
    numpy.testing.assert_array_equal(from_table, from_bits)

    vals = Quantizer._E2M1_GRID
    assert numpy.array_equal(Quantizer._e2m1_encode(vals) & 7,
                             numpy.arange(8)), "a grid value re-encoded wrongly"
    neg = Quantizer._e2m1_encode(-vals)
    assert numpy.array_equal(neg & 7, numpy.arange(8))
    assert numpy.all(neg & 8), "the sign bit was lost"
    print("  PASS (all 16 nibbles: table == bit assembly, encode round-trips)")


def test_e2m1_encoder_is_round_to_nearest_even():
    """Every tie goes to the even mantissa, and no value sits further from its
    code than its neighbor does — the two properties that make the bound in
    check_payload an assertion rather than a guess."""
    ties = [(0.25, 0.0), (0.75, 1.0), (1.25, 1.0), (1.75, 2.0), (2.5, 2.0),
            (3.5, 4.0), (5.0, 4.0)]
    x = numpy.array([t for t, _ in ties] + [-t for t, _ in ties], dtype=numpy.float32)
    want = numpy.array([v for _, v in ties] + [-v for _, v in ties], dtype=numpy.float32)
    got = Quantizer._e2m1_decode(Quantizer._e2m1_encode(x))
    numpy.testing.assert_array_equal(got, want)

    rng = numpy.random.default_rng(2)
    grid = numpy.concatenate([Quantizer._E2M1_GRID, -Quantizer._E2M1_GRID])
    x = numpy.clip(
        numpy.concatenate([
            rng.normal(0, 1, 4000),
            rng.normal(0, 0.2, 4000),
            numpy.linspace(-MAX_F4, MAX_F4, 2001).astype(numpy.float32),
        ]).astype(numpy.float32), -MAX_F4, MAX_F4)
    got = Quantizer._e2m1_decode(Quantizer._e2m1_encode(x))
    best = numpy.abs(grid[None, :] - x[:, None]).min(axis=1)
    over = numpy.abs(got - x) - best
    assert numpy.all(over <= 1e-12), f"{int((over > 1e-12).sum())} non-nearest"
    print(f"  PASS (ties go to even mantissa; all {x.size} values round to nearest)")


def test_value_past_the_grid_raises():
    """E2M1 has no Inf code, so a value past the grid's top has two ways to go: it
    can saturate to 6, which is the format's designed answer and what the
    quantizer's clip leaves, or it can carry out of the 3-bit magnitude field,
    where a byte truncation would fold it back into a small, legal-looking nibble.
    The midpoint between 6 and the step above it is 7, so that is exactly where
    the encoder starts raising."""
    # Inside the midpoint: nearest is still the top code, so no raise and no
    # wraparound.
    sat = Quantizer._e2m1_encode(numpy.array([6.1, 6.9, -6.9], dtype=numpy.float32))
    assert list(sat & 7) == [7, 7, 7], f"6.1/6.9 did not saturate to the top: {sat}"
    assert sat[2] & 8, "the sign was lost on a saturated value"

    for bad in (7.0, 12.0, 1e6):
        try:
            Quantizer._e2m1_encode(numpy.array([bad], dtype=numpy.float32))
            raise AssertionError(f"{bad} was encoded into a 3-bit field")
        except ValueError as e:
            assert "magnitude field" in str(e)

    # Through the quantizer the clip is what keeps a real weight on the legal side
    # of that line. One column's 1e6 absmax sets the tensor's factor, so the other
    # columns have nothing left to encode: a range this skewed is legal, and the
    # format's own answer is to flatten it.
    w = numpy.zeros((16, 8), dtype=numpy.float32)
    w[:, 0] = numpy.linspace(0.0, 1e6, 16).astype(numpy.float32)
    m = matmul_case(w)
    codes, s = kernel_decode(m, 16, 8)
    assert numpy.all(numpy.abs(codes) <= MAX_F4), "a payload value left the grid"
    assert float(numpy.abs((codes * s)[:, 1:]).max()) == 0.0
    print("  PASS (6.1..6.9 saturate to the top code; 7 and past raise)")


def test_block_scale_stays_inside_e4m3():
    """The per-tensor factor exists to put the largest block scale exactly on
    e4m3's top code, which is what lets one byte cover a block. Checked on a
    weight whose block absmaxes span orders of magnitude — the case a scale
    chosen per column instead of per tensor would break."""
    rng = numpy.random.default_rng(3)
    w = (rng.normal(0, 1, (64, 16)) *
         numpy.float32(10.0) ** rng.uniform(-3.0, 3.0, (64, 1))).astype(numpy.float32)
    m = matmul_case(w)
    g = float(numpy_helper.to_array(m.initializers["W_scale_global"])[0])
    assert numpy.isclose(g, float(global_scale_of(w)), rtol=1e-6)
    bytes_ = numpy.frombuffer(m.initializers["W_scale"].raw_data, numpy.uint8)
    vals = numpy.abs(numpy.array([shader_e4m3(int(b)) for b in bytes_],
                                 dtype=numpy.float32))
    assert numpy.all(numpy.isfinite(vals)), "a block scale decoded as Inf/NaN"
    assert float(vals.max()) >= 0.9 * MAX_F8, (
        f"the largest block scale only reached {vals.max()}, not e4m3's top")
    check_payload(m, w)
    print(f"  PASS (global {g:.3e} puts the top block scale on {vals.max():.1f} "
          f"of e4m3's 448)")


def test_a_zero_block_is_reported_not_silently_flattened():
    """A block whose scale rounds to zero decodes to 0 for all 16 values. That is
    the format's own behavior, so it is allowed — but only out loud."""
    w = numpy.zeros((32, 8), dtype=numpy.float32)
    w[:, :4] = numpy.linspace(-1.0, 1.0, 32).astype(numpy.float32)[:, None]
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        Quantizer.quantize_to_nvfp4_weight_only(
            build("W", w, [("MatMul", "mm", {"transB": 0}, ["x", "W"])]))
    text = buf.getvalue()
    assert "Blocks whose scale rounded to 0" in text, text
    # An all-zero weight has nothing to erase: its blocks are legitimately zero,
    # so the count of flattened nonzero blocks has to read 0.
    assert "rounded to 0 with nonzero values: 0" in text, text
    print("  PASS (zero-scale blocks counted and printed, zero here)")


def test_group_size_other_than_16_raises_before_any_work():
    """16 is not a tuning knob of this format, it IS the format: the scale byte is
    the block's absmax over 6, and no other span can be decoded with it. A caller
    asking for another block length wants a different encoding, so this raises
    rather than quietly quantizing to 16 anyway."""
    numpy.random.seed(4)
    m = build("W", numpy.random.randn(32, 8).astype(numpy.float32),
              [("MatMul", "mm", {"transB": 0}, ["x", "W"])])
    for bad in (8, 32, 64):
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                Quantizer.quantize_to_nvfp4_weight_only(m, bad)
            raise AssertionError(f"group_size={bad} was accepted")
        except ValueError as e:
            assert "16" in str(e)
    assert m.initializers["W"].data_type == TensorProto.FLOAT
    print("  PASS (group_size != 16 raises and leaves the weight float)")


def test_gates_preserve_the_float_weight():
    """Every case the nvfp4 kernel cannot read keeps its float weight: no payload
    rewrite, no scale initializers, no appended inputs."""
    numpy.random.seed(5)
    cases = [
        ("Gemm has no nvfp4 kernel",
         build("W", numpy.random.randn(16, 8).astype(numpy.float32),
               [("Gemm", "g", {"transB": 1}, ["x", "W"])])),
        ("Conv has no nvfp4 kernel",
         build("W", numpy.random.randn(8, 8, 4, 4).astype(numpy.float32),
               [("Conv", "c", {}, ["x", "W"])])),
        ("rank-3 batched weight",
         build("W", numpy.random.randn(2, 16, 8).astype(numpy.float32),
               [("MatMul", "mm", {"transB": 0}, ["x", "W"])])),
        ("A-side weight",
         build("W", numpy.random.randn(16, 16).astype(numpy.float32),
               [("MatMul", "mm", {"transB": 0}, ["W", "x"])])),
        ("transB=1 stores [N, K]",
         build("W", numpy.random.randn(16, 16).astype(numpy.float32),
               [("MatMul", "mm", {"transB": 1}, ["x", "W"])])),
        ("N=5 is not a multiple of 8",
         build("W", numpy.random.randn(16, 5).astype(numpy.float32),
               [("MatMul", "mm", {"transB": 0}, ["x", "W"])])),
        ("K=24 is not a whole number of 16-value blocks",
         build("W", numpy.random.randn(24, 8).astype(numpy.float32),
               [("MatMul", "mm", {"transB": 0}, ["x", "W"])])),
        ("a non-float source arrives already quantized",
         build("W", numpy.arange(128, dtype=numpy.int8).reshape(16, 8),
               [("MatMul", "mm", {"transB": 0}, ["x", "W"])], dtype=numpy.int8)),
    ]
    for label, m in cases:
        before = m.initializers["W"].data_type
        before_inputs = {n.name: [i["name"] for i in n.inputs] for n in m.nodes.values()}
        out = quantize(m)
        assert out.initializers["W"].data_type == before, f"{label}: payload rewritten"
        for extra in ("W_scale", "W_scale_global"):
            assert extra not in out.initializers, f"{label}: {extra} appeared"
        assert {n.name: [i["name"] for i in n.inputs] for n in out.nodes.values()} == \
            before_inputs, f"{label}: a scale input was appended"
        print(f"  PASS ({label})")

    # Consumers that disagree about the layout cannot share one scale axis.
    m = build("W", numpy.random.randn(16, 8).astype(numpy.float32),
              [("MatMul", "a", {"transB": 0}, ["x", "W"]),
               ("MatMul", "b", {"transB": 1}, ["y", "W"])])
    out = quantize(m)
    assert out.initializers["W"].data_type == TensorProto.FLOAT
    print("  PASS (consumers disagreeing on transB stay float)")


def quantize_4bit(w, group_size, fmt):
    m = build("W", w, [("MatMul", "mm", {"transB": 0}, ["x", "W"])])
    with contextlib.redirect_stdout(io.StringIO()):
        Quantizer.quantize_to_4bit_weight_only(m, group_size, fmt)
    return m


def test_a_fine_group_at_half_a_byte_per_value():
    """What the format is for, measured rather than assumed. Its cost is fixed by
    the layout: half a byte per value plus one scale byte per 16, i.e. 0.5625 --
    the same price as int4 or nf4 with an fp32 scale per 64, and no fp32-scale
    format can reach a 16-value group at that price (it would cost 0.75). So the
    claims to pin are the byte count exactly, and that e2m1's step is *relative*:
    every value it places in its normal band keeps within a third of itself, where
    a same-size fixed-step grid erases everything under half a step. Absolute MSE
    is reported, not asserted: on a flat-topped weight distribution a 15-level
    fixed grid can win it, and that is a fact about the data, not a defect."""
    rng = numpy.random.default_rng(6)
    k, n = 256, 16
    # Spread WITHIN each column, so neither a per-column nor a per-block scale can
    # absorb it: this is the spread the two grids fail differently on.
    w = (rng.normal(0, 1, (k, n)) *
         numpy.float32(10.0) ** rng.uniform(-3.0, 0.0, (k, n)).astype(numpy.float32)
         ).astype(numpy.float32)

    m4 = matmul_case(w)
    deq4, err4 = check_payload(m4, w)
    codes4, s4 = kernel_decode(m4, k, n)
    payload_bytes = (_init_byte_len(m4.initializers["W"]) +
                     _init_byte_len(m4.initializers["W_scale"]))
    per_value = payload_bytes / (k * n)
    assert abs(per_value - 0.5625) < 1e-9, f"nvfp4 costs {per_value} bytes/value"

    m_i4 = quantize_4bit(w, 64, "int4")
    nib_i4 = unpack(m_i4.initializers["W"].raw_data, k * n).astype(numpy.int32)
    nib_i4 = numpy.where(nib_i4 >= 8, nib_i4 - 16, nib_i4).reshape(k, n)
    scale_i4 = numpy_helper.to_array(m_i4.initializers["W_scale"])
    deq_i4 = (nib_i4.reshape(k // 64, 64, n) * scale_i4[:, None, :]).reshape(k, n)
    err_i4 = numpy.abs(deq_i4 - w)
    assert (k * n // 2 + 4 * (k // 64) * n) / (k * n) == per_value, (
        "int4@64 no longer costs what nvfp4 costs, so the comparison is stale")

    # e2m1's normal band: a value coded at 1.0 or above sits at least three
    # quarters of a scale step out, and half the narrowest step there is a
    # quarter, so its relative error cannot exceed a third.
    normal = (numpy.abs(codes4) >= 1.0) & (s4 > 0)
    rel4 = err4 / numpy.abs(w)
    assert float(rel4[normal].max()) <= 1.0 / 3.0 + 1e-6, (
        f"e2m1 left its rounding bound: {rel4[normal].max():.2%}")
    # int4's step is absolute (the group's absmax over 7), so below half a step a
    # weight is not inaccurate but gone, and its relative error saturates at 100%.
    rel_i4 = err_i4 / numpy.abs(w)
    erased = nib_i4 == 0
    assert numpy.any(erased), "no weight was erased, so the data does not test " \
        "the thing this comparison is for"
    assert float(rel_i4[erased].min()) > 0.9, "an erased weight's relative error " \
        "should saturate near 100%"
    assert float(numpy.mean(rel4[normal])) < 0.5 * float(numpy.mean(rel_i4)), (
        f"mean relative error nvfp4 {numpy.mean(rel4[normal]):.2%} is not clearly "
        f"better than int4@64 {numpy.mean(rel_i4):.2%}")
    print(f"  PASS ({per_value} bytes/value for both; nvfp4 rel error max "
          f"{rel4[normal].max():.2%}, mean {numpy.mean(rel4[normal]):.2%} vs "
          f"int4@64 mean {numpy.mean(rel_i4):.2%}; int4 zeroes "
          f"{int(numpy.count_nonzero(erased))} weights; mse nvfp4 "
          f"{numpy.mean(err4**2):.3e} vs int4@64 {numpy.mean(err_i4**2):.3e})")


def test_fp16_source_arrives_as_the_same_payload():
    """The LLMs vkop ships are exported in fp16, so the fp16 initializer is the
    real path; its bytes have to match what the same values produce from fp32, or
    the loader would be reading a half's bits as something else."""
    numpy.random.seed(7)
    w16 = numpy.random.randn(32, 8).astype(numpy.float16)
    m16 = matmul_case(w16, dtype=numpy.float16)
    m32 = matmul_case(w16.astype(numpy.float32))
    assert m16.initializers["W"].data_type == TensorProto.FLOAT4E2M1
    assert m16.initializers["W"].raw_data == m32.initializers["W"].raw_data
    assert m16.initializers["W_scale"].raw_data == m32.initializers["W_scale"].raw_data
    print("  PASS (an fp16 initializer gives the fp32 path's exact bytes)")


def test_writer_records_the_nvfp4_size_and_names():
    """The C++ loader cross-checks the recorded byte count against
    elem_bytes(dtype, prod(dims)) and refuses a mismatch, and it reads both the
    4-bit and the fp8 payload through a byte container numpy cannot decode at all
    without an optional dependency. So: the packed payload is raw_data verbatim at
    half the value count, the block scale table verbatim at one byte per entry,
    and both dtype strings are the spellings the loader's name table uses."""
    numpy.random.seed(8)
    w = numpy.random.randn(32, 8).astype(numpy.float32)
    q = matmul_case(w)

    for name, onnx_type, spelled, dims, want in [
        ("W", TensorProto.FLOAT4E2M1, "float4e2m1fn", [32, 8], 128),
        ("W_scale", TensorProto.FLOAT8E4M3FN, "float8e4m3fn", [2, 8], 16),
    ]:
        init = q.initializers[name]
        assert init.data_type == onnx_type
        assert _init_byte_len(init) == want, f"{name}: {_init_byte_len(init)} != {want}"
        assert _init_dtype_name(init) == spelled
        assert _init_bytes(init) == init.raw_data

        broken = TensorProto()
        broken.data_type = onnx_type
        broken.name = name
        broken.dims.extend(dims)
        broken.raw_data = init.raw_data[:-1]
        try:
            _init_bytes(broken)
            raise AssertionError(f"a short {name} payload was written anyway")
        except ValueError as e:
            assert "bytes" in str(e)

    g = q.initializers["W_scale_global"]
    assert _init_dtype_name(g) == "float32"
    assert _init_byte_len(g) == 4
    assert float(numpy_helper.to_array(g)[0]) == float(global_scale_of(w))
    # An fp4 payload with an odd element count rounds up to a whole byte, which
    # the loader's dims x elem_bits check has to agree with.
    odd = TensorProto()
    odd.data_type = TensorProto.FLOAT4E2M1
    odd.name = "odd"
    odd.dims.extend([1, 7])
    odd.raw_data = b"\x00\x00"
    assert _init_byte_len(odd) == 4, _init_byte_len(odd)
    print("  PASS (writer: 128 + 16 bytes, both dtype names, mismatch raises)")


if __name__ == "__main__":
    for fn in (test_three_tensors_and_their_layout,
               test_e2m1_encoder_matches_the_shader_decoder,
               test_e2m1_encoder_is_round_to_nearest_even,
               test_value_past_the_grid_raises,
               test_block_scale_stays_inside_e4m3,
               test_a_zero_block_is_reported_not_silently_flattened,
               test_group_size_other_than_16_raises_before_any_work,
               test_gates_preserve_the_float_weight,
               test_a_fine_group_at_half_a_byte_per_value,
               test_fp16_source_arrives_as_the_same_payload,
               test_writer_records_the_nvfp4_size_and_names):
        print(fn.__name__ + ":")
        fn()
    print("\nALL PASS")
