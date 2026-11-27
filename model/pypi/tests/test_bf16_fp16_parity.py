"""Where the converter took an fp16 input but refused or mangled its bf16 twin.

Each case builds the same tiny graph twice -- once with fp16 (or fp32) data,
once with bf16 data carrying the same values -- and asserts the bf16 one now
converts the way the fp16 one always did:

  * a bf16 initializer stored in `int32_data` (helper.make_tensor raw=False)
    instead of `raw_data`: died "cannot reshape array of size 0", while the
    fp16 twin in `int32_data` converted;
  * the usual bf16 export (fp32 I/O, Cast(to=bf16) in, Cast(to=fp32) out)
    under --act fp16 / -q fp16 / -q fp32: the Cast's `to` stayed BFLOAT16
    after the weights became FLOAT16, so the FP16 domain pass refused and the
    artifact bound bf16 activations to fp16 weights (-q fp16 / -q fp32 failed
    outright). The fp16 export converts to the identical bytes now;
  * a STREAMED weight under --act bf16: it was written float16 under a
    bfloat16 declaration (the runtime refuses that node), where the in-memory
    twin was a real bf16 payload. Plain rows and the SwiGLU gate/up interleave;
  * -q bf16 hydrated every weight (deferral refused) while -q fp16 streams;
  * AWQ refused a bf16 norm scale (the --act bf16 domain pass makes one), so a
    --act bf16 AWQ conversion scaled nothing;
  * the numeric audit had no bf16 threshold, so no bf16 weight was ever
    flagged;
  * stream_export's BF16 target truncated where its F16 target rounds.
"""
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import onnx
import pytest
from onnx import TensorProto as T
from onnx import external_data_helper
from onnx import helper as h
from onnx import numpy_helper as nh

from onnx2vkop.bf16_weights import bf16_bits_to_fp32, fp32_to_bf16_bits
from onnx2vkop.generated.vkop.model import Model as _Model
from onnx2vkop.inspect import ModelView

PYPI = str(Path(__file__).resolve().parents[1])
K, N = 64, 32


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _bf16_exact(a):
    """fp32 values a bf16 holds exactly (fp16 holds them too at this range)."""
    a = np.ascontiguousarray(a, dtype=np.float32)
    return (a.view(np.uint32) & np.uint32(0xFFFF0000)).view(np.float32)


def _bits(a):
    return (np.ascontiguousarray(a, np.float32).view(np.uint32)
            >> 16).astype(np.uint16)


def _tensor(name, values, kind, raw=True):
    """`values` (fp32) as an fp16/bf16/fp32 initializer, raw or int32_data."""
    if kind == "fp32":
        return nh.from_array(np.asarray(values, np.float32), name)
    if kind == "fp16":
        bits = np.asarray(values, np.float16).view(np.uint16)
        dt = T.FLOAT16
    else:
        bits = _bits(values)
        dt = T.BFLOAT16
    if raw:
        return h.make_tensor(name, dt, list(values.shape),
                             bits.astype("<u2").tobytes(), raw=True)
    t = onnx.TensorProto(name=name, data_type=dt, dims=list(values.shape))
    t.int32_data.extend(int(v) for v in bits.reshape(-1))
    return t


def _values(seed=0):
    rng = np.random.default_rng(seed)
    w = _bf16_exact(rng.standard_normal((K, N)) / 8)
    s = _bf16_exact(1 + rng.standard_normal(K) / 10)
    return w, s


def mul_matmul(path, kind, raw=True, external=False):
    """y = (x * S) @ W, fp32 I/O; S and W in `kind`."""
    w, s = _values()
    nodes = [h.make_node("Mul", ["x", "S"], ["xs"]),
             h.make_node("MatMul", ["xs", "W"], ["y"])]
    g = h.make_graph(nodes, "g", [h.make_tensor_value_info("x", T.FLOAT,
                                                           [1, 4, K])],
                     [h.make_tensor_value_info("y", T.FLOAT, [1, 4, N])],
                     [_tensor("W", w, kind, raw), _tensor("S", s, kind, raw)])
    _save(g, path, external)
    return w, s


def cast_export(path, kind):
    """The usual low-precision export: Cast in, MatMul, Cast back to fp32."""
    w, _ = _values()
    lo = T.BFLOAT16 if kind == "bf16" else T.FLOAT16
    nodes = [h.make_node("Cast", ["x"], ["xc"], to=lo),
             h.make_node("MatMul", ["xc", "W"], ["yl"]),
             h.make_node("Cast", ["yl"], ["y"], to=T.FLOAT)]
    g = h.make_graph(nodes, "g", [h.make_tensor_value_info("x", T.FLOAT,
                                                           [1, 4, K])],
                     [h.make_tensor_value_info("y", T.FLOAT, [1, 4, N])],
                     [_tensor("W", w, kind)])
    _save(g, path, False)


def swiglu_mlp(path, kind, external, d=256, f=512):
    rng = np.random.default_rng(3)
    wg = _bf16_exact(rng.standard_normal((d, f)) / 16)
    wu = _bf16_exact(rng.standard_normal((d, f)) / 16)
    wd = _bf16_exact(rng.standard_normal((f, d)) / 22)
    nodes = [h.make_node("MatMul", ["x", "Wg"], ["g"]),
             h.make_node("Sigmoid", ["g"], ["sg"]),
             h.make_node("Mul", ["g", "sg"], ["silu"]),
             h.make_node("MatMul", ["x", "Wu"], ["u"]),
             h.make_node("Mul", ["silu", "u"], ["a"]),
             h.make_node("MatMul", ["a", "Wd"], ["out"])]
    g = h.make_graph(nodes, "mlp",
                     [h.make_tensor_value_info("x", T.FLOAT, [1, 4, d])],
                     [h.make_tensor_value_info("out", T.FLOAT, [1, 4, d])],
                     [_tensor("Wg", wg, kind), _tensor("Wu", wu, kind),
                      _tensor("Wd", wd, kind)])
    _save(g, path, external)


def _save(graph, path, external):
    m = h.make_model(graph, opset_imports=[h.make_opsetid("", 17)])
    m.ir_version = 8
    if external:
        external_data_helper.convert_model_to_external_data(
            m, all_tensors_to_one_file=True,
            location=os.path.basename(str(path)) + ".data", size_threshold=0)
    onnx.save(m, str(path))


def convert(src, out, *flags, ok=True):
    env = dict(os.environ)
    env.pop("VKOP_STREAM_WEIGHTS", None)
    r = subprocess.run(
        [sys.executable, "-m", "onnx2vkop.cli", "-i", str(src), "-o",
         str(out), *flags], cwd=PYPI, capture_output=True, text=True, env=env)
    log = r.stdout + r.stderr
    if ok:
        assert r.returncode == 0, log
        assert "Error" not in log, log
    return r.returncode, log


# Streaming knobs: defer every eligible tensor (these are tiny) and force the
# trailer, which is the only layout a streamed payload can be written to.
STREAM = ("--stream-min-mib", "0", "--external-blob-mb", "0")
NO_STREAM = ("--stream-weights", "off", "--external-blob-mb", "0")


def rows(path):
    """name -> (dtype, dims, payload bytes) read back from the artifact."""
    view = ModelView(str(path))
    data = Path(path).read_bytes()
    if view.external_blob_trailer is not None:
        base = view.external_blob_trailer

        def payload(off, size):
            return data[base + off: base + off + size]
    else:
        blob = _Model.Model.GetRootAs(data, 0).InitializerBlobAsNumpy() \
            .tobytes()

        def payload(off, size):
            return blob[off: off + size]
    return {r["name"]: (r["dtype"], r["dims"], payload(r["offset"], r["size"]))
            for r in view.initializers()}


def node_inputs(path):
    """[(op, [(input, dtype)])] in node order."""
    out = []
    for n in ModelView(str(path)).nodes():
        out.append((n["op_type"], [(i.get("name"), i.get("dtype"))
                                   for i in n.get("inputs", [])
                                   if isinstance(i, dict)]))
    return out


def dtypes_in(path):
    view = ModelView(str(path))
    seen = {r["dtype"] for r in view.initializers()}
    for io_ in view.shapes() + view.outputs():
        seen.add(io_.get("dtype"))
    for _op, ins in node_inputs(path):
        seen.update(d for _n, d in ins)
    return seen


# ---------------------------------------------------------------------------
# bf16 in int32_data
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("flags", [("--act", "fp16"), ("--act", "bf16"),
                                   ("--keep-bf16",)])
def test_bf16_int32_data_converts_like_raw_data(tmp_path, flags):
    mul_matmul(tmp_path / "raw.onnx", "bf16", raw=True)
    mul_matmul(tmp_path / "i32.onnx", "bf16", raw=False)
    convert(tmp_path / "raw.onnx", tmp_path / "raw.vkopbin", *flags)
    convert(tmp_path / "i32.onnx", tmp_path / "i32.vkopbin", *flags)
    assert (tmp_path / "i32.vkopbin").read_bytes() == \
        (tmp_path / "raw.vkopbin").read_bytes()


def test_fp16_int32_data_is_unchanged(tmp_path):
    """The fp16 twin always worked; it still matches its raw_data form."""
    mul_matmul(tmp_path / "raw.onnx", "fp16", raw=True)
    mul_matmul(tmp_path / "i32.onnx", "fp16", raw=False)
    convert(tmp_path / "raw.onnx", tmp_path / "raw.vkopbin")
    convert(tmp_path / "i32.onnx", tmp_path / "i32.vkopbin")
    assert (tmp_path / "i32.vkopbin").read_bytes() == \
        (tmp_path / "raw.vkopbin").read_bytes()


# ---------------------------------------------------------------------------
# Cast(to=bf16) exports
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("flags", [("--act", "fp16"), ("-q", "fp16"),
                                   ("-q", "fp32")])
def test_cast_to_bf16_export_converts_like_its_fp16_twin(tmp_path, flags):
    cast_export(tmp_path / "bf.onnx", "bf16")
    cast_export(tmp_path / "hf.onnx", "fp16")
    _, log = convert(tmp_path / "bf.onnx", tmp_path / "bf.vkopbin", *flags)
    convert(tmp_path / "hf.onnx", tmp_path / "hf.vkopbin", *flags)
    assert "REFUSED" not in log, log
    assert "bfloat16" not in dtypes_in(tmp_path / "bf.vkopbin")
    # The weight values are exact in both formats, so the artifacts agree to
    # the byte.
    assert (tmp_path / "bf.vkopbin").read_bytes() == \
        (tmp_path / "hf.vkopbin").read_bytes()


@pytest.mark.parametrize("act", ["fp16", "bf16"])
def test_dequantize_linear_bf16_scale_widens_exactly(act):
    """A bf16 DequantizeLinear scale is widened to fp32 (exact) rather than
    narrowed to fp16, where a per-column int8 scale (< 6.1e-5) loses bits;
    a bf16 output_dtype follows the normalization target."""
    from onnx2vkop.converter import _downcast_bfloat16

    scale = _bf16_exact(np.array([3.0e-6, 1.7e-5, 4.2e-7, 2.5e-3]))
    q = nh.from_array(np.arange(16, dtype=np.int8).reshape(4, 4), "Q")
    dq = h.make_node("DequantizeLinear", ["Q", "s"], ["W"], axis=1,
                     output_dtype=T.BFLOAT16)
    mm = h.make_node("MatMul", ["x", "W"], ["y"])
    g = h.make_graph([dq, mm], "g",
                     [h.make_tensor_value_info("x", T.BFLOAT16, [1, 4])],
                     [h.make_tensor_value_info("y", T.BFLOAT16, [1, 4])],
                     [q, _tensor("s", scale, "bf16")])
    m = h.make_model(g, opset_imports=[h.make_opsetid("", 21)])
    _downcast_bfloat16(m, act)
    s = next(t for t in m.graph.initializer if t.name == "s")
    assert s.data_type == T.FLOAT
    assert np.array_equal(nh.to_array(s), scale)
    want = T.FLOAT if act == "bf16" else T.FLOAT16
    out_dtype = [a.i for a in m.graph.node[0].attribute
                 if a.name == "output_dtype"]
    assert out_dtype == [want]


def test_cast_to_bf16_under_act_bf16_still_runs_bf16(tmp_path):
    cast_export(tmp_path / "bf.onnx", "bf16")
    convert(tmp_path / "bf.onnx", tmp_path / "bf.vkopbin", "--act", "bf16")
    ins = dict(node_inputs(tmp_path / "bf.vkopbin"))
    assert ins["MatMul"] == [("xc", "bfloat16"), ("W", "bfloat16")]


# ---------------------------------------------------------------------------
# streamed weights under --act bf16
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kind", ["bf16", "fp16"])
def test_streamed_weight_under_act_bf16_is_a_bf16_payload(tmp_path, kind):
    mul_matmul(tmp_path / "ext.onnx", kind, external=True)
    mul_matmul(tmp_path / "mem.onnx", kind, external=False)
    _, log = convert(tmp_path / "ext.onnx", tmp_path / "ext.vkopbin",
                     "--act", "bf16", *STREAM)
    assert "streaming 1 initializer" in log, log
    convert(tmp_path / "mem.onnx", tmp_path / "mem.vkopbin", "--act", "bf16",
            *NO_STREAM)
    ext, mem = rows(tmp_path / "ext.vkopbin"), rows(tmp_path / "mem.vkopbin")
    assert ext["W"][0] == "bfloat16", ext["W"][:2]
    # Byte for byte what the in-memory domain pass writes: the source bits
    # for bf16, fp16 -> bf16 RNE for fp16.
    assert ext["W"] == mem["W"]
    w, _ = _values()
    assert ext["W"][2] == fp32_to_bf16_bits(w).astype("<u2").tobytes()
    assert ("W", "bfloat16") in dict(node_inputs(tmp_path / "ext.vkopbin"))[
        "MatMul"]


def test_streamed_fp16_rounds_to_nearest_even(tmp_path):
    """An fp16 value bf16 cannot hold is rounded, not truncated."""
    vals = np.full((K, N), 1.0 + 3 * 2.0 ** -10, np.float32)  # fp16-exact
    vals[0, :4] = [65504.0, -65504.0, 1.0 + 2.0 ** -8, 1.0 + 3 * 2.0 ** -8]
    w = nh.from_array(vals.astype(np.float16), "W")
    g = h.make_graph([h.make_node("MatMul", ["x", "W"], ["y"])], "g",
                     [h.make_tensor_value_info("x", T.FLOAT, [1, 4, K])],
                     [h.make_tensor_value_info("y", T.FLOAT, [1, 4, N])], [w])
    _save(g, tmp_path / "ext.onnx", True)
    convert(tmp_path / "ext.onnx", tmp_path / "ext.vkopbin", "--act", "bf16",
            *STREAM)
    got = np.frombuffer(rows(tmp_path / "ext.vkopbin")["W"][2], "<u2")
    want = fp32_to_bf16_bits(vals.astype(np.float16).astype(np.float32))
    assert np.array_equal(got, want.reshape(-1))
    # 65504 rounds up to 65536 (0x4780), 1 + 3*2^-8 is a tie -> even.
    assert [hex(v) for v in got[:4]] == ["0x4780", "0xc780", "0x3f80",
                                         "0x3f82"]


@pytest.mark.parametrize("kind", ["bf16", "fp16"])
def test_streamed_swiglu_gate_up_under_act_bf16(tmp_path, kind):
    swiglu_mlp(tmp_path / "ext.onnx", kind, True)
    swiglu_mlp(tmp_path / "mem.onnx", kind, False)
    convert(tmp_path / "ext.onnx", tmp_path / "ext.vkopbin", "--act", "bf16",
            *STREAM)
    convert(tmp_path / "mem.onnx", tmp_path / "mem.vkopbin", "--act", "bf16",
            *NO_STREAM)
    ext, mem = rows(tmp_path / "ext.vkopbin"), rows(tmp_path / "mem.vkopbin")
    fused = [n for n in ext if n.endswith("__swiglu_gu")]
    assert fused, sorted(ext)
    for name in fused + ["Wd"]:
        assert ext[name][0] == "bfloat16", (name, ext[name][:2])
        assert ext[name] == mem[name], name
    assert "float16" not in dtypes_in(tmp_path / "ext.vkopbin")


def test_streamed_act_fp16_is_unchanged(tmp_path):
    """--act fp16 still writes the fp16 payload it always did."""
    mul_matmul(tmp_path / "ext.onnx", "bf16", external=True)
    mul_matmul(tmp_path / "mem.onnx", "bf16", external=False)
    convert(tmp_path / "ext.onnx", tmp_path / "ext.vkopbin", *STREAM)
    convert(tmp_path / "mem.onnx", tmp_path / "mem.vkopbin", *NO_STREAM)
    ext, mem = rows(tmp_path / "ext.vkopbin"), rows(tmp_path / "mem.vkopbin")
    assert ext["W"][0] == "float16"
    assert ext["W"] == mem["W"]


# ---------------------------------------------------------------------------
# -q bf16 streams
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kind", ["bf16", "fp16"])
def test_q_bf16_streams_like_q_fp16(tmp_path, kind):
    mul_matmul(tmp_path / "ext.onnx", kind, external=True)
    mul_matmul(tmp_path / "mem.onnx", kind, external=False)
    _, log = convert(tmp_path / "ext.onnx", tmp_path / "ext.vkopbin",
                     "-q", "bf16", *STREAM)
    assert "weights not streamed" not in log, log
    assert "streaming 1 initializer" in log, log
    assert "1 streamed as bf16" in log, log
    convert(tmp_path / "mem.onnx", tmp_path / "mem.vkopbin", "-q", "bf16",
            *NO_STREAM)
    ext, mem = rows(tmp_path / "ext.vkopbin"), rows(tmp_path / "mem.vkopbin")
    assert ext["W"][0] == "bfloat16"
    assert ext["W"] == mem["W"]
    assert ext["S"] == mem["S"]


def test_q_bf16_streamed_exclusion_is_honoured(tmp_path):
    mul_matmul(tmp_path / "ext.onnx", "fp16", external=True)
    convert(tmp_path / "ext.onnx", tmp_path / "ext.vkopbin", "-q", "bf16",
            "--exclude-node", "*MatMul*", *STREAM)
    assert rows(tmp_path / "ext.vkopbin")["W"][0] == "float16"


# ---------------------------------------------------------------------------
# AWQ on a bf16 norm scale
# ---------------------------------------------------------------------------

def test_awq_folds_into_a_bf16_norm_scale():
    import contextlib
    import io

    from onnx2vkop import awq
    from onnx2vkop.dag import DAGBasedModel, Node

    def tref(name, shape=(-1,)):
        return {"name": name, "shape": list(shape)}

    k = n = 8
    m = DAGBasedModel()
    w = np.ones((k, n), np.float32)
    m.initializers["W"] = nh.from_array(w, "W")
    norm_w = _bf16_exact(np.linspace(0.5, 2.0, k))
    m.initializers["norm_w"] = _tensor("norm_w", norm_w, "bf16")
    m.nodes["norm"] = Node("RMSNorm", "norm", {"eps": 1e-6},
                           [tref("x"), tref("norm_w", (k,))],
                           [tref("h", (1, -1, k))])
    m.nodes["mm"] = Node("MatMul", "mm", {},
                         [tref("h", (1, -1, k)), tref("W", (k, n))],
                         [tref("y", (1, -1, n))])
    m.inputs = [tref("x", (1, -1, k))]
    with contextlib.redirect_stdout(io.StringIO()) as buf:
        scaled = awq.apply(m, 8, fmt="int4")
    assert scaled == {"W"}, buf.getvalue()
    out = m.initializers["norm_w"]
    assert out.data_type == T.BFLOAT16
    ratio = nh.to_array(m.initializers["W"]).astype(np.float32) / w
    k_ = float(ratio.flat[0])
    got = bf16_bits_to_fp32(np.frombuffer(out.raw_data, "<u2"))
    want = bf16_bits_to_fp32(fp32_to_bf16_bits(norm_w * (1.0 / k_)))
    assert np.array_equal(got, want)


# ---------------------------------------------------------------------------
# numeric audit
# ---------------------------------------------------------------------------

def test_numeric_audit_flags_a_bad_bf16_weight():
    from onnx2vkop import numeric_audit

    rng = np.random.default_rng(0)
    src = rng.standard_normal((256, 256)).astype(np.float32)
    good = bf16_bits_to_fp32(fp32_to_bf16_bits(src)).reshape(src.shape)
    bad = good * np.float32(1.01)
    with numeric_audit.collecting() as audit:
        numeric_audit.record_roundtrip("good", "bf16", src, good)
        numeric_audit.record_roundtrip("bad", "bf16", src, bad)
    assert [r.tensor for r in audit.flagged()] == ["bad"]


# ---------------------------------------------------------------------------
# stream_export's BF16 target
# ---------------------------------------------------------------------------

def test_stream_export_bf16_target_rounds_to_nearest_even(tmp_path):
    """stream_export imports torch at module level; stub it in a child so the
    re-encode can be checked on a machine without torch."""
    vals = np.array([1.0 + 2.0 ** -8, 1.0 + 3 * 2.0 ** -8,
                     1.0 + 2.0 ** -7 + 2.0 ** -8 + 2.0 ** -10, -2.5],
                    np.float32)
    shard = tmp_path / "w.bin"
    shard.write_bytes(vals.tobytes())
    code = textwrap.dedent(f"""
        import sys, types
        from unittest import mock
        for name in ("torch", "torch.nn", "torch.onnx", "torch.onnx._internal",
                     "torch.onnx._internal.torchscript_exporter"):
            sys.modules[name] = mock.MagicMock(name=name)
        from onnx2vkop import stream_export
        e = {{"dtype": "F32", "shard": {str(shard)!r}, "offset": 0,
              "nbytes": {vals.nbytes}, "onnx_name": "w"}}
        sys.stdout.write(stream_export._convert_tensor(e, "BF16").hex())
    """)
    r = subprocess.run([sys.executable, "-c", code], cwd=PYPI,
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    got = np.frombuffer(bytes.fromhex(r.stdout.strip()), "<u2")
    assert list(got) == list(fp32_to_bf16_bits(vals))
    # tie -> even (stays), tie -> even (up), above the tie -> up.
    assert [hex(v) for v in got] == ["0x3f80", "0x3f82", "0x3f82", "0xc020"]


def test_mark_bf16_payload_twin_for_a_shared_streamed_weight(tmp_path):
    """The bf16 domain pass gives a weight shared with an island its own bf16
    copy (`name__bf16`). Streamed, that copy is a second row over the same
    source bytes; the original row is left as it was."""
    import io

    from onnx2vkop import weight_stream as ws

    w, _ = _values()
    src = tmp_path / "w.data"
    src.write_bytes(np.asarray(w, np.float16).tobytes())
    init = onnx.TensorProto(name="W", data_type=T.FLOAT16, dims=[K, N],
                            data_location=T.EXTERNAL)
    for key, val in (("location", "w.data"), ("offset", "0"),
                     ("length", str(K * N * 2))):
        e = init.external_data.add()
        e.key, e.value = key, val
    deferred = {"W": {"name": "W", "location": "w.data",
                      "base_dir": str(tmp_path), "offset": 0,
                      "length": K * N * 2, "data_type": T.FLOAT16,
                      "dims": (K, N), "consumers": 2}}
    inits = {"W": init}
    assert ws.mark_bf16_payload(deferred, "W", init, "W__bf16", inits)
    assert deferred["W"]["data_type"] == T.FLOAT16
    assert "out_bf16" not in deferred["W"]
    assert init.data_type == T.FLOAT16
    twin = inits["W__bf16"]
    assert twin.data_type == T.BFLOAT16 and ws.is_deferred(twin)
    assert deferred["W__bf16"]["name"] == "W__bf16"

    buf = io.BytesIO()
    assert ws.write_payload(deferred["W__bf16"], buf, str(tmp_path)) \
        == K * N * 2
    want = fp32_to_bf16_bits(np.asarray(w, np.float16).astype(np.float32))
    assert buf.getvalue() == want.astype("<u2").tobytes()
    buf = io.BytesIO()
    ws.write_payload(deferred["W"], buf, str(tmp_path))
    assert buf.getvalue() == src.read_bytes()
    # A byte-format row is never re-marked.
    deferred["W"]["quant"] = "int8"
    assert not ws.mark_bf16_payload(deferred, "W", init)
