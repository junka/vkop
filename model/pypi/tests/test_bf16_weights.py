"""bfloat16 weights in the MatMul/Gemm B and Gather table slots (bf16_weights.py).

The runtime reads a bf16 payload ("bfloat16", raw little-endian uint16 bits, the
layout an fp16 weight has in the same slot, no scale) for exactly two kinds of
constant initializer: the B input of MatMul/Gemm and the table input of Gather.
Everything else must keep the default downcast.

What these cases pin:

  * the RNE fp32 -> bf16 rounding (ties to even, overflow to inf, NaN kept);
  * --keep-bf16 end to end: a qualifying weight lands in the .vkopbin as
    "bfloat16" with the source's bits, bit for bit; a bias / norm scale / graph
    I/O / a shared initializer still downcasts;
  * a weight whose Transpose folds into the MatMul (transB) keeps the source
    layout, and one the ConstantFolder materialized is matched by value;
  * a weight a pass rewrote (values no longer the source's) falls back to the
    default payload instead of shipping something new;
  * -q bf16 rounds fp32/fp16 sources once and keeps bf16 sources exact;
  * a streamed bf16 weight is copied verbatim under --keep-bf16;
  * without the flag the artifact is byte-identical to the old behaviour.
"""
import contextlib
import io
from types import SimpleNamespace

import numpy
import onnx
import pytest
from onnx import TensorProto, external_data_helper, helper, numpy_helper

from onnx2vkop import bf16_weights
from onnx2vkop.bf16_weights import bf16_bits_to_fp32, fp32_to_bf16_bits
from onnx2vkop.converter import ModelConverter
from onnx2vkop.dag import DAGBasedModel, Node
from onnx2vkop.generated.vkop.model import Model as _Model
from onnx2vkop.inspect import ModelView


# ---------------------------------------------------------------------------
# RNE rounding
# ---------------------------------------------------------------------------

def _f32(bits):
    return numpy.asarray(bits, dtype=numpy.uint32).view(numpy.float32)


def test_rne_exact_values_are_truncation():
    vals = numpy.asarray([0.0, -0.0, 1.0, -2.5, 0.15625, 3.0e38, -1e-30],
                         dtype=numpy.float32)
    # Values whose low 16 bits are zero convert by truncation, sign included.
    exact = _f32(vals.view(numpy.uint32) & numpy.uint32(0xFFFF0000))
    got = fp32_to_bf16_bits(exact)
    assert list(got) == list((exact.view(numpy.uint32) >> 16).astype(numpy.uint16))
    assert got[1] == 0x8000  # -0.0 keeps its sign


def test_rne_ties_go_to_even():
    # 1.0 = 0x3F800000. bf16 ulp at 1.0 is 2^-7; the halfway point is 0x8000
    # in the low half.
    cases = {
        0x3F808000: 0x3F80,  # 1 + 2^-8: tie, 0x3F80 is even -> stays
        0x3F818000: 0x3F82,  # tie, 0x3F81 is odd -> rounds up to even
        0x3F808001: 0x3F81,  # just above the tie -> up
        0x3F807FFF: 0x3F80,  # just below the tie -> down
        0xBF818000: 0xBF82,  # negative tie, odd -> away from zero to even
        0x3F80FFFF: 0x3F81,  # carry inside the mantissa
        0x3FFFFFFF: 0x4000,  # carry into the exponent: 1.999.. -> 2.0
        0x00008000: 0x0000,  # subnormal tie, even -> 0
        0x00018000: 0x0002,  # subnormal tie, odd -> up
    }
    got = fp32_to_bf16_bits(_f32(list(cases)))
    assert [hex(g) for g in got] == [hex(v) for v in cases.values()]


def test_rne_overflow_and_infinities():
    cases = {
        0x7F7FFFFF: 0x7F80,  # fp32 max rounds past bf16 max -> +inf
        0x7F7F8000: 0x7F80,  # tie at the top, odd -> even = +inf
        0x7F7F7FFF: 0x7F7F,  # below the tie -> bf16 max
        0xFF7FFFFF: 0xFF80,  # -> -inf
        0x7F800000: 0x7F80,  # +inf
        0xFF800000: 0xFF80,  # -inf
    }
    with numpy.errstate(over="ignore"):
        got = fp32_to_bf16_bits(_f32(list(cases)))
    assert [hex(g) for g in got] == [hex(v) for v in cases.values()]


def test_rne_keeps_nan_a_nan():
    # 0x7F800001's payload lives only in the low half: a plain rounding add
    # would emit 0x7F80 (= +inf). It must stay a NaN, sign preserved.
    nans = _f32([0x7FC00000, 0x7F800001, 0xFF800001, 0x7FFFFFFF, 0xFFFFFFFF])
    got = fp32_to_bf16_bits(nans)
    back = bf16_bits_to_fp32(got)
    assert numpy.isnan(back).all(), [hex(g) for g in got]
    assert list(got >> 15) == [0, 0, 1, 0, 1]
    assert got[0] == 0x7FC0


def test_rne_matches_a_float64_reference_on_random_bits():
    """Against an independent rounding: nearest bf16 by float64 distance, ties
    to the even mantissa."""
    rng = numpy.random.RandomState(1)
    bits = rng.randint(0, 2 ** 32, size=20000, dtype=numpy.uint64).astype(
        numpy.uint32)
    x = bits.view(numpy.float32)
    keep = numpy.isfinite(x) & (numpy.abs(x) < 3.0e38)
    x = x[keep]
    got = fp32_to_bf16_bits(x)
    lo = (x.view(numpy.uint32) & numpy.uint32(0xFFFF0000)).view(numpy.float32)
    lo_bits = (lo.view(numpy.uint32) >> 16).astype(numpy.uint16)
    hi_bits = lo_bits + numpy.uint16(1)
    hi = bf16_bits_to_fp32(hi_bits)
    d_lo = numpy.abs(x.astype(numpy.float64) - lo.astype(numpy.float64))
    d_hi = numpy.abs(hi.astype(numpy.float64) - x.astype(numpy.float64))
    want = numpy.where(d_lo < d_hi, lo_bits,
                       numpy.where(d_hi < d_lo, hi_bits,
                                   numpy.where(lo_bits % 2 == 0, lo_bits,
                                               hi_bits)))
    assert numpy.array_equal(got, want)


def test_fp16_source_rounds_through_its_exact_fp32_value():
    h = numpy.asarray([1.0009765625, 65504.0, -0.0001], dtype=numpy.float16)
    got = fp32_to_bf16_bits(h.astype(numpy.float32))
    # 1 + 2^-10 is below bf16's half-ulp (2^-8) -> 1.0
    assert got[0] == 0x3F80
    assert bf16_bits_to_fp32(got)[1] == 65536.0  # 65504 rounds up in bf16


# ---------------------------------------------------------------------------
# end-to-end fixtures
# ---------------------------------------------------------------------------

def bf16_bits(values):
    return fp32_to_bf16_bits(numpy.asarray(values, dtype=numpy.float32))


def bf16_init(name, arr):
    arr = numpy.asarray(arr, dtype=numpy.float32)
    return helper.make_tensor(name, TensorProto.BFLOAT16, list(arr.shape),
                              bf16_bits(arr).tobytes(), raw=True)


def _rand(rng, *shape, scale=1.0):
    return (rng.randn(*shape) * scale).astype(numpy.float32)


V, K, N = 32, 16, 24


def _bf16_model(rng, *, big=False):
    """A bf16 export: embedding Gather -> MatMul(W) -> Add(bias) -> Mul(scale)
    -> MatMul(Transpose(Wt)) -> MatMul(S), where S is ALSO read as the A
    (activation-side) operand of a second MatMul -- shared with a slot that is
    not a bf16 weight slot. Activations declared bf16 too, which the default
    normalizes to fp16.

    `big` puts a value past fp16's 65504 in W, which the default turns into inf
    and --keep-bf16 must keep exact.
    """
    E = _rand(rng, V, K)
    W = _rand(rng, K, N)
    if big:
        W[0, 0] = 1.0e6
    Wt = _rand(rng, K, N)          # stored [K, N], read through Transpose
    bias = _rand(rng, N)
    scale = _rand(rng, N)
    S = _rand(rng, K, K)
    inits = [bf16_init("E", E), bf16_init("W", W), bf16_init("bias", bias),
             bf16_init("scale", scale), bf16_init("Wt", Wt), bf16_init("S", S)]
    nodes = [
        helper.make_node("Gather", ["E", "ids"], ["emb"], name="embed"),
        helper.make_node("MatMul", ["emb", "W"], ["h0"], name="proj"),
        helper.make_node("Add", ["h0", "bias"], ["h1"], name="bias_add"),
        helper.make_node("Mul", ["h1", "scale"], ["h2"], name="norm_scale"),
        helper.make_node("Transpose", ["Wt"], ["WtT"], name="tr", perm=[1, 0]),
        helper.make_node("MatMul", ["h2", "WtT"], ["h3"], name="back"),
        helper.make_node("MatMul", ["h3", "S"], ["Y"], name="shared_mm"),
        helper.make_node("MatMul", ["S", "col"], ["Y2"], name="s_as_a"),
    ]
    graph = helper.make_graph(
        nodes, "bf16",
        [helper.make_tensor_value_info("ids", TensorProto.INT64, [1, 4]),
         helper.make_tensor_value_info("col", TensorProto.BFLOAT16, [K, 1])],
        [helper.make_tensor_value_info("Y", TensorProto.BFLOAT16, [1, 4, K]),
         helper.make_tensor_value_info("Y2", TensorProto.BFLOAT16, [K, 1])],
        inits)
    model = helper.make_model(graph, producer_name="test",
                              opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    return model, dict(E=E, W=W, Wt=Wt, bias=bias, scale=scale, S=S)


def _opts(quant="fp16", keep_bf16=False, **kw):
    base = dict(quant=quant, unify=False, rgba=False, group_size=64,
                target_backend="buffer", keep_bf16=keep_bf16)
    base.update(kw)
    return SimpleNamespace(**base)


def _convert(tmp_path, model, out_name, *, quant="fp16", keep_bf16=False,
             save_external=False):
    src = tmp_path / (out_name + ".onnx")
    if save_external:
        external_data_helper.convert_model_to_external_data(
            model, all_tensors_to_one_file=True, location=out_name + ".bin",
            size_threshold=0)
    onnx.save_model(model, str(src))
    conv = ModelConverter()
    with contextlib.redirect_stdout(io.StringIO()):
        dag = conv.parse_onnx_model(str(src), 1, quant=quant,
                                    keep_bf16=keep_bf16)
        conv.apply_optimizations(dag, _opts(quant=quant, keep_bf16=keep_bf16))
        out = tmp_path / (out_name + ".vkopbin")
        dag.save_to_binary(str(out))
    return out, dag


def _rows(path):
    """name -> (dtype, dims, payload bytes) read back from the artifact."""
    view = ModelView(str(path))
    data = path.read_bytes()
    if view.external_blob_trailer is not None:
        base = view.external_blob_trailer

        def payload(off, size):
            return data[base + off: base + off + size]
    else:
        root = _Model.Model.GetRootAs(data, 0)
        blob = root.InitializerBlobAsNumpy().tobytes()

        def payload(off, size):
            return blob[off: off + size]
    return {r["name"]: (r["dtype"], r["dims"], payload(r["offset"], r["size"]))
            for r in view.initializers()}


def _node_input_dtype(path, tensor):
    view = ModelView(str(path))
    seen = set()
    for n in view.nodes():
        for inp in n.get("inputs", []):
            if isinstance(inp, dict) and inp.get("name") == tensor:
                seen.add(inp.get("dtype"))
    return seen


# ---------------------------------------------------------------------------
# --keep-bf16
# ---------------------------------------------------------------------------

def test_keep_bf16_qualifying_weights_are_bit_identical(tmp_path):
    rng = numpy.random.RandomState(0)
    model, ref = _bf16_model(rng, big=True)
    out, _ = _convert(tmp_path, model, "keep", keep_bf16=True)
    rows = _rows(out)

    # MatMul B (direct) and the Gather table: bf16, the source bits verbatim.
    for name in ("W", "E"):
        dtype, dims, payload = rows[name]
        assert dtype == "bfloat16", (name, dtype)
        assert payload == bf16_bits(ref[name]).tobytes(), name
        assert list(dims) == list(ref[name].shape)
    # The >65504 value survives (the fp16 default makes it inf).
    w = bf16_bits_to_fp32(numpy.frombuffer(rows["W"][2], dtype=numpy.uint16))
    assert w[0] == bf16_bits_to_fp32(bf16_bits([1.0e6]))[0]


def test_keep_bf16_transpose_folded_weight_keeps_source_layout(tmp_path):
    """Transpose(Wt) -> MatMul ends up either as transB=1 on the same name or
    as a materialized constant; either way the payload is the source's bf16
    bits in whatever layout the fp16 payload in that slot has."""
    rng = numpy.random.RandomState(0)
    model, ref = _bf16_model(rng)
    out_keep, _ = _convert(tmp_path, model, "keep", keep_bf16=True)
    model2, _ = _bf16_model(numpy.random.RandomState(0))
    out_def, _ = _convert(tmp_path, model2, "def")
    keep, dflt = _rows(out_keep), _rows(out_def)
    # Find the fp16 weight the back-projection reads in the default artifact.
    cands = [n for n, (dt, dims, _) in dflt.items()
             if dt == "float16" and sorted(dims) == sorted([K, N])
             and n not in ("W",)]
    assert cands, dflt.keys()
    name = cands[0]
    fp16_default = numpy.frombuffer(dflt[name][2], dtype=numpy.float16)
    dtype, dims, payload = keep[name]
    assert dtype == "bfloat16"
    assert list(dims) == list(dflt[name][1])
    got = bf16_bits_to_fp32(numpy.frombuffer(payload, dtype=numpy.uint16))
    # Same layout: element-for-element the fp16 payload is the bf16 value
    # rounded to fp16.
    assert numpy.array_equal(got.astype(numpy.float16), fp16_default)
    src = ref["Wt"] if list(dims) == [K, N] else ref["Wt"].T
    assert payload == bf16_bits(src).tobytes()


def test_keep_bf16_non_qualifying_tensors_still_downcast(tmp_path):
    rng = numpy.random.RandomState(0)
    model, _ = _bf16_model(rng)
    out, _ = _convert(tmp_path, model, "keep", keep_bf16=True)
    model2, _ = _bf16_model(numpy.random.RandomState(0))
    out_def, _ = _convert(tmp_path, model2, "def")
    rows, dflt = _rows(out), _rows(out_def)
    # bias (Add), scale (Mul) and S (MatMul B, but ALSO MatMul A) are not
    # bf16-slot-only: each is exactly what the default conversion writes.
    for name in ("bias", "scale", "S"):
        assert rows[name] == dflt[name], name
        assert rows[name][0] == "float16", (name, rows[name][0])
    # Nothing in the artifact is bf16 except weights in a qualifying slot.
    view = ModelView(str(out))
    bf16 = sorted(r["name"] for r in view.initializers()
                  if r["dtype"] == "bfloat16")
    assert set(bf16) - {"W", "E"} and all(
        n in ("W", "E") or n.startswith("Wt") for n in bf16), bf16
    # Graph I/O stays in the activation domain.
    for io_ in view.shapes() + view.outputs():
        assert io_.get("dtype") != "bfloat16", io_


def test_keep_bf16_shared_with_a_non_qualifying_reader_downcasts():
    """A weight read as MatMul B AND as an Add operand keeps the default."""
    m = DAGBasedModel()
    W = numpy.ones((4, 4), numpy.float32) * 3.0
    m.initializers["W"] = numpy_helper.from_array(W.astype(numpy.float16), "W")
    m.initializers["G"] = numpy_helper.from_array(W.astype(numpy.float16), "G")
    t = lambda n: {"name": n, "shape": [4, 4], "dtype": 10}
    m.nodes["mm"] = Node("MatMul", "mm", {}, [t("x"), t("W")], [t("y")])
    m.nodes["add"] = Node("Add", "add", {}, [t("y"), t("W")], [t("z")])
    m.nodes["mm2"] = Node("MatMul", "mm2", {}, [t("z"), t("G")], [t("q")])
    m.nodes["ga"] = Node("Gather", "ga", {}, [t("G"), t("ix")], [t("r")])
    sources = {n: ((4, 4), bf16_bits(W).tobytes()) for n in ("W", "G")}
    with contextlib.redirect_stdout(io.StringIO()):
        s = bf16_weights.apply(m, keep=True, sources=sources)
    assert m.initializers["W"].data_type == TensorProto.FLOAT16
    # Shared between two QUALIFYING slots (MatMul B + Gather table) is fine.
    assert m.initializers["G"].data_type == TensorProto.BFLOAT16
    assert s["kept"] == ["G"]
    assert m.nodes["mm2"].inputs[1]["dtype"] == TensorProto.BFLOAT16
    assert m.nodes["add"].inputs[1]["dtype"] == 10


def test_keep_bf16_falls_back_when_a_pass_rewrote_the_weight():
    """Values no longer equal to the source's downcast -> today's payload."""
    m = DAGBasedModel()
    W = numpy.arange(12, dtype=numpy.float32).reshape(3, 4) / 7.0
    rewritten = (bf16_bits_to_fp32(bf16_bits(W)) * 2.0).astype(numpy.float16)
    m.initializers["W"] = numpy_helper.from_array(rewritten, "W")
    t = lambda n: {"name": n, "shape": [-1], "dtype": 10}
    m.nodes["mm"] = Node("MatMul", "mm", {}, [t("x"), t("W")], [t("y")])
    sources = {"W": ((3, 4), bf16_bits(W).tobytes())}
    with contextlib.redirect_stdout(io.StringIO()):
        s = bf16_weights.apply(m, keep=True, sources=sources)
    assert m.initializers["W"].data_type == TensorProto.FLOAT16
    assert numpy.array_equal(numpy_helper.to_array(m.initializers["W"]),
                             rewritten)
    assert s["fallback"] == ["W"]


def test_keep_bf16_skips_folded_matmuls():
    m = DAGBasedModel()
    W = numpy.ones((4, 8), numpy.float32)
    m.initializers["W"] = numpy_helper.from_array(W.astype(numpy.float16), "W")
    t = lambda n: {"name": n, "shape": [-1], "dtype": 10}
    m.nodes["mm"] = Node("MatMul", "mm", {"swiglu": 1}, [t("x"), t("W")],
                         [t("y")])
    with contextlib.redirect_stdout(io.StringIO()):
        bf16_weights.apply(m, keep=True,
                           sources={"W": ((4, 8), bf16_bits(W).tobytes())})
    assert m.initializers["W"].data_type == TensorProto.FLOAT16


def test_materialized_transpose_is_matched_by_value():
    m = DAGBasedModel()
    rng = numpy.random.RandomState(3)
    W = _rand(rng, 6, 5)  # source [6, 5]; the DAG holds its transpose
    cur = bf16_bits_to_fp32(bf16_bits(W)).T.astype(numpy.float16)
    m.initializers["WtT"] = numpy_helper.from_array(
        numpy.ascontiguousarray(cur), "WtT")
    t = lambda n: {"name": n, "shape": [-1], "dtype": 10}
    m.nodes["mm"] = Node("MatMul", "mm", {}, [t("x"), t("WtT")], [t("y")])
    with contextlib.redirect_stdout(io.StringIO()):
        s = bf16_weights.apply(m, keep=True,
                               sources={"Wt": ((6, 5), bf16_bits(W).tobytes())})
    assert s["kept"] == ["WtT"]
    got = m.initializers["WtT"]
    assert got.data_type == TensorProto.BFLOAT16 and list(got.dims) == [5, 6]
    assert got.raw_data == numpy.ascontiguousarray(bf16_bits(W).T).tobytes()


# ---------------------------------------------------------------------------
# default mode unchanged
# ---------------------------------------------------------------------------

def test_default_mode_has_no_bf16_and_ignores_the_new_code(tmp_path):
    rng = numpy.random.RandomState(0)
    model, _ = _bf16_model(rng)
    out_a, _ = _convert(tmp_path, model, "a")
    view = ModelView(str(out_a))
    assert all(r["dtype"] != "bfloat16" for r in view.initializers())


def test_keep_bf16_on_a_model_without_bf16_is_byte_identical(tmp_path):
    rng = numpy.random.RandomState(0)
    W = _rand(rng, K, N)
    graph = helper.make_graph(
        [helper.make_node("MatMul", ["X", "W"], ["Y"], name="mm")], "g",
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, K])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, N])],
        [numpy_helper.from_array(W, "W")])
    mk = lambda: helper.make_model(graph, producer_name="test",
                                   opset_imports=[helper.make_opsetid("", 17)])
    a, _ = _convert(tmp_path, mk(), "a")
    b, _ = _convert(tmp_path, mk(), "b", keep_bf16=True)
    assert a.read_bytes() == b.read_bytes()


# ---------------------------------------------------------------------------
# -q bf16
# ---------------------------------------------------------------------------

def test_quant_bf16_rounds_fp32_weights_once(tmp_path):
    rng = numpy.random.RandomState(5)
    W = _rand(rng, K, N)
    E = _rand(rng, V, K)
    b = _rand(rng, N)
    graph = helper.make_graph(
        [helper.make_node("Gather", ["E", "ids"], ["h"], name="embed"),
         helper.make_node("MatMul", ["h", "W"], ["m"], name="mm"),
         helper.make_node("Add", ["m", "b"], ["Y"], name="add")], "g",
        [helper.make_tensor_value_info("ids", TensorProto.INT64, [1, 4])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 4, N])],
        [numpy_helper.from_array(W, "W"), numpy_helper.from_array(E, "E"),
         numpy_helper.from_array(b, "b")])
    model = helper.make_model(graph, producer_name="test",
                              opset_imports=[helper.make_opsetid("", 17)])
    out, _ = _convert(tmp_path, model, "q", quant="bf16")
    rows = _rows(out)
    for name, src in (("W", W), ("E", E)):
        assert rows[name][0] == "bfloat16", (name, rows[name][0])
        # Straight from fp32, not through an fp16 intermediate.
        assert rows[name][2] == fp32_to_bf16_bits(src).tobytes(), name
    # The bias is not a bf16 slot: it follows the FP16 activation domain.
    assert rows["b"][0] == "float16"


def test_quant_bf16_keeps_bf16_sources_exact(tmp_path):
    rng = numpy.random.RandomState(0)
    model, ref = _bf16_model(rng, big=True)
    out, _ = _convert(tmp_path, model, "q", quant="bf16")
    rows = _rows(out)
    for name in ("W", "E"):
        assert rows[name][0] == "bfloat16"
        assert rows[name][2] == bf16_bits(ref[name]).tobytes(), name


def test_quant_bf16_honours_exclusion():
    m = DAGBasedModel()
    W = numpy.full((4, 4), 1.0 + 2.0 ** -9, numpy.float32)
    m.initializers["W"] = numpy_helper.from_array(W, "W")
    m.initializers["V"] = numpy_helper.from_array(W, "V")
    t = lambda n: {"name": n, "shape": [-1], "dtype": 1}
    m.nodes["mm"] = Node("MatMul", "mm", {}, [t("x"), t("W")], [t("y")])
    m.nodes["mm2"] = Node("MatMul", "mm2", {}, [t("y"), t("V")], [t("z")])
    with contextlib.redirect_stdout(io.StringIO()):
        s = bf16_weights.apply(m, keep=True, quantize=True, sources={},
                               exclude={"V"})
    assert s["rounded"] == ["W"]
    assert m.initializers["V"].data_type == TensorProto.FLOAT


# ---------------------------------------------------------------------------
# streamed (deferred) weights
# ---------------------------------------------------------------------------

def _streamed_bf16_model(rng):
    W = _rand(rng, 64, 64)
    graph = helper.make_graph(
        [helper.make_node("MatMul", ["X", "W"], ["Y"], name="mm")], "g",
        [helper.make_tensor_value_info("X", TensorProto.BFLOAT16, [1, 64])],
        [helper.make_tensor_value_info("Y", TensorProto.BFLOAT16, [1, 64])],
        [bf16_init("W", W)])
    model = helper.make_model(graph, producer_name="test",
                              opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    return model, W


@pytest.mark.parametrize("keep", [False, True])
def test_streamed_bf16_weight(tmp_path, monkeypatch, keep):
    monkeypatch.setenv("VKOP_STREAM_WEIGHTS", "1")
    monkeypatch.setenv("VKOP_STREAM_MIN_MIB", "0")
    monkeypatch.setenv("VKOP_EXTERNAL_BLOB_MB", "0")
    model, W = _streamed_bf16_model(numpy.random.RandomState(2))
    out, dag = _convert(tmp_path, model, "s", keep_bf16=keep,
                        save_external=True)
    assert "W" in dag.deferred_weights, "the weight must actually stream"
    dtype, dims, payload = _rows(out)["W"]
    src = bf16_bits(W)
    if keep:
        assert dtype == "bfloat16"
        assert payload == src.tobytes()
    else:
        assert dtype == "float16"
        with numpy.errstate(over="ignore"):
            want = bf16_bits_to_fp32(src).astype(numpy.float16)
        assert payload == want.tobytes()


def test_keep_bf16_gemm_transb_weight_keeps_its_stored_layout(tmp_path):
    """Gemm(transB=1) stores B as [N, K]; the fp16 payload in that slot keeps
    that layout (the MatMul it lowers to carries transB), so the bf16 payload
    must too -- the source bits verbatim, not transposed."""
    rng = numpy.random.RandomState(7)
    B = _rand(rng, N, K)
    C = _rand(rng, N)
    graph = helper.make_graph(
        [helper.make_node("Gemm", ["X", "B", "C"], ["Y"], name="gemm",
                          transB=1)], "g",
        [helper.make_tensor_value_info("X", TensorProto.BFLOAT16, [4, K])],
        [helper.make_tensor_value_info("Y", TensorProto.BFLOAT16, [4, N])],
        [bf16_init("B", B), bf16_init("C", C)])
    model = helper.make_model(graph, producer_name="test",
                              opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    out, _ = _convert(tmp_path, model, "gemm", keep_bf16=True)
    rows = _rows(out)
    model2 = onnx.load_from_string(model.SerializeToString())
    dflt = _rows(_convert(tmp_path, model2, "gemm_def")[0])
    assert rows["B"][0] == "bfloat16"
    assert list(rows["B"][1]) == list(dflt["B"][1]) == [N, K]
    assert rows["B"][2] == bf16_bits(B).tobytes()
    # The bias is an Add operand after lowering: default downcast.
    bias = [n for n in rows if n != "B"]
    assert all(rows[n] == dflt[n] for n in bias), bias


# ---------------------------------------------------------------------------
# range scan: which bf16 weights a half-staging MatMul tile would clamp
# ---------------------------------------------------------------------------

from onnx2vkop import numeric_audit  # noqa: E402


@pytest.fixture
def audits(monkeypatch):
    """The NumericAudit(s) the converter's own begin/finish produced."""
    got = []
    real = numeric_audit.finish

    def finish(mode=None):
        a = real("off")
        got.append(a)
        return a
    monkeypatch.setattr(numeric_audit, "finish", finish)
    return got


def _bits(*vals):
    return numpy.asarray(vals, dtype=numpy.uint16)


@pytest.mark.parametrize("pattern,over,sub", [
    (0x477F, 0, 0),   # 65280: largest bf16 inside fp16's range
    (0x4780, 1, 0),   # 65536: first past 65504
    (0xC77F, 0, 0),   # -65280
    (0xC780, 1, 0),   # -65536
    (0x387F, 0, 1),   # just under 2**-14
    (0x3880, 0, 0),   # 2**-14: fp16's smallest normal
    (0xB87F, 0, 1),   # negative, under fp16 normal
    (0x0001, 0, 1),   # bf16 subnormal
    (0x7F80, 1, 0),   # +inf
    (0xFF80, 1, 0),   # -inf
    (0x7FC0, 1, 0),   # NaN
    (0xFF81, 1, 0),   # negative signalling NaN
    (0x8000, 0, 0),   # -0
    (0x0000, 0, 0),   # +0
])
def test_tile_scan_boundary_bit_patterns(pattern, over, sub):
    s = bf16_weights.scan_bits(_bits(pattern))
    assert (s.elements, s.overflow, s.subnormal) == (1, over, sub)
    # The bit thresholds are exactly fp16's range.
    v = float(bf16_bits_to_fp32(_bits(pattern))[0])
    if over:
        assert not (abs(v) <= 65504.0)
    else:
        assert abs(v) <= 65504.0
    if sub:
        assert 0.0 < abs(v) < 2.0 ** -14


def test_tile_scan_matches_a_float_reference_on_every_bit_pattern():
    allbits = numpy.arange(1 << 16, dtype=numpy.uint32).astype(numpy.uint16)
    v = bf16_bits_to_fp32(allbits)
    with numpy.errstate(invalid="ignore"):
        want_over = int(numpy.count_nonzero(~(numpy.abs(v) <= 65504.0)))
        want_sub = int(numpy.count_nonzero(
            (numpy.abs(v) > 0) & (numpy.abs(v) < 2.0 ** -14)))
    s = bf16_weights.scan_bits(allbits)
    assert (s.overflow, s.subnormal) == (want_over, want_sub)


def test_tile_scan_chunking_and_odd_byte_splits(monkeypatch):
    rng = numpy.random.RandomState(3)
    bits = rng.randint(0, 1 << 16, size=10001).astype(numpy.uint16)
    whole = bf16_weights.scan_bits(bits)
    monkeypatch.setattr(bf16_weights, "_SCAN_CHUNK", 7)
    chunked = bf16_weights.scan_bits(bits)
    raw = bits.tobytes()
    streamed = bf16_weights.TileScan()
    pos = 0
    for step in (1, 3, 5, 2, 9999, 7, 11):
        streamed.add_bytes(raw[pos:pos + step])
        pos += step
    streamed.add_bytes(raw[pos:])
    for s in (chunked, streamed):
        assert (s.elements, s.overflow, s.subnormal) == (
            whole.elements, whole.overflow, whole.subnormal)


def _attrs(path):
    return {n.get("name"): n.get("attributes") or {}
            for n in ModelView(str(path)).nodes()}


def test_keep_bf16_flags_only_the_overflowing_weight(tmp_path, audits):
    rng = numpy.random.RandomState(0)
    model, _ = _bf16_model(rng, big=True)
    out, _ = _convert(tmp_path, model, "keep", keep_bf16=True)
    audit = audits[-1]
    attrs = _attrs(out)
    assert attrs["proj"].get("bf16_wide") == [1]       # W holds 1e6
    for other in ("back", "shared_mm", "s_as_a", "embed"):
        assert "bf16_wide" not in attrs[other], other
    rows = {r.tensor: r for r in audit.rows}
    assert rows["W"].tile_overflow == 1
    assert rows["W"].rel_l2 == 0.0                    # kept: lossless
    assert rows["E"].tile_overflow == 0
    assert "1 tile-overflow element(s) in 1 weight(s)" in audit.summary_line()
    assert "tile-overflow 1" in rows["W"].line()


def test_keep_bf16_in_range_emits_no_flag(tmp_path):
    rng = numpy.random.RandomState(0)
    model, _ = _bf16_model(rng, big=False)
    out, _ = _convert(tmp_path, model, "keep", keep_bf16=True)
    assert not any("bf16_wide" in a for a in _attrs(out).values())


def test_quant_bf16_rounding_path_is_scanned(tmp_path, audits):
    """-q bf16 over an fp32 weight past fp16's range: the ROUNDED bits are
    what the scan sees, and the row record_roundtrip made carries the count."""
    rng = numpy.random.RandomState(4)
    W = _rand(rng, K, N)
    W[2, 3] = -7.0e4
    W2 = _rand(rng, N, K)
    graph = helper.make_graph(
        [helper.make_node("MatMul", ["X", "W"], ["H"], name="mm0"),
         helper.make_node("MatMul", ["H", "W2"], ["Y"], name="mm1")], "g",
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, [4, K])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [4, K])],
        [numpy_helper.from_array(W, "W"), numpy_helper.from_array(W2, "W2")])
    model = helper.make_model(graph, producer_name="test",
                              opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    out, _ = _convert(tmp_path, model, "q", quant="bf16")
    audit = audits[-1]
    rows = _rows(out)
    assert rows["W"][0] == "bfloat16"
    attrs = _attrs(out)
    assert attrs["mm0"].get("bf16_wide") == [1]
    assert "bf16_wide" not in attrs["mm1"]
    by = {r.tensor: r for r in audit.rows}
    assert by["W"].tile_overflow == 1 and by["W"].rel_l2 > 0
    assert by["W2"].tile_overflow == 0
    # One row per weight: the scan joined the round-trip row.
    assert [r.tensor for r in audit.rows].count("W") == 1


@pytest.mark.parametrize("big", [False, True])
def test_streamed_bf16_weight_is_scanned(tmp_path, monkeypatch, big,
                                        audits):
    monkeypatch.setenv("VKOP_STREAM_WEIGHTS", "1")
    monkeypatch.setenv("VKOP_STREAM_MIN_MIB", "0")
    monkeypatch.setenv("VKOP_EXTERNAL_BLOB_MB", "0")
    model, W = _streamed_bf16_model(numpy.random.RandomState(2))
    if big:
        W = W.copy()
        W[5, 7] = numpy.inf
        W[6, 1] = 1.0e5
        model.graph.initializer[0].CopyFrom(bf16_init("W", W))
    out, dag = _convert(tmp_path, model, "s", keep_bf16=True,
                        save_external=True)
    audit = audits[-1]
    assert "W" in dag.deferred_weights
    dtype, _dims, payload = _rows(out)["W"]
    assert dtype == "bfloat16" and payload == bf16_bits(W).tobytes()
    flagged = _attrs(out)["mm"].get("bf16_wide") == [1]
    assert flagged == big
    row = [r for r in audit.rows if r.tensor == "W"]
    assert len(row) == 1 and row[0].tile_overflow == (2 if big else 0)


def test_streamed_source_changed_after_the_scan_is_refused(tmp_path,
                                                           monkeypatch):
    from onnx2vkop import weight_stream
    monkeypatch.setenv("VKOP_STREAM_WEIGHTS", "1")
    monkeypatch.setenv("VKOP_STREAM_MIN_MIB", "0")
    monkeypatch.setenv("VKOP_EXTERNAL_BLOB_MB", "0")
    model, _W = _streamed_bf16_model(numpy.random.RandomState(2))
    # The graph-time scan reports the weight in range; the bytes then turn
    # out to hold an overflow (simulates a source rewritten mid-conversion).
    real = bf16_weights._scan_deferred

    def lying(entry):
        s = real(entry)
        s.overflow = 0
        return s
    src = tmp_path / "s.bin"
    orig_apply = bf16_weights.apply

    def apply_then_corrupt(*a, **k):
        out = orig_apply(*a, **k)
        data = bytearray(src.read_bytes())
        data[0:2] = numpy.asarray([0x7F80], dtype=numpy.uint16).tobytes()
        src.write_bytes(bytes(data))
        return out
    monkeypatch.setattr(bf16_weights, "_scan_deferred", lying)
    monkeypatch.setattr(bf16_weights, "apply", apply_then_corrupt)
    with pytest.raises(weight_stream.DeferredWeightError, match="bf16_wide"):
        _convert(tmp_path, model, "s", keep_bf16=True, save_external=True)


def _keep_artifact(tmp_path, name, model, **kw):
    d = tmp_path / name
    d.mkdir()
    out, _ = _convert(d, onnx.load_from_string(model.SerializeToString()),
                      "m", keep_bf16=True, **kw)
    return out.read_bytes()


def test_no_overflow_artifact_is_byte_identical_without_the_scan(
        tmp_path, monkeypatch):
    """In range, the scan writes nothing: the artifact equals one built with
    the scan stubbed out entirely (in-memory and streamed)."""
    model, _ = _bf16_model(numpy.random.RandomState(0), big=False)
    smodel, _ = _streamed_bf16_model(numpy.random.RandomState(2))

    def both():
        for k in ("VKOP_STREAM_WEIGHTS", "VKOP_STREAM_MIN_MIB",
                  "VKOP_EXTERNAL_BLOB_MB"):
            monkeypatch.delenv(k, raising=False)
        mem = _keep_artifact(tmp_path, f"m{len(seen)}", model)
        monkeypatch.setenv("VKOP_STREAM_WEIGHTS", "1")
        monkeypatch.setenv("VKOP_STREAM_MIN_MIB", "0")
        monkeypatch.setenv("VKOP_EXTERNAL_BLOB_MB", "0")
        streamed = _keep_artifact(tmp_path, f"s{len(seen)}", smodel,
                                  save_external=True)
        seen.append((mem, streamed))

    seen = []
    both()
    monkeypatch.setattr(bf16_weights, "_record_scan", lambda *a, **k: None)
    monkeypatch.setattr(bf16_weights, "_scan_deferred",
                        lambda e: bf16_weights.TileScan())
    both()
    assert seen[0][0] == seen[1][0]
    assert seen[0][1] == seen[1][1]
