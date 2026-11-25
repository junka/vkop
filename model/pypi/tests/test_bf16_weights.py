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
