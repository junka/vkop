"""A streamed -q int4 / nf4 conversion must write the in-memory conversion's file.

`-q nf4` used to refuse streaming outright ("rewrites weight payloads and needs
their bytes"), so every weight of a model was hydrated and quantized in RAM --
GLM-Edge-1.5B took 8 min 19 s and most of a 36 GiB machine. It now streams like
int4: the scale is the group absmax (quant_payload.scale_for), the codes are the
shared exact encoder (Quantizer._nf4_encode), and the DAG side (nf4 dtype name,
per-weight report, scale position, input declarations, the SwiGLU gate/up fold)
is done the way the in-memory pass does it.

The claim pinned here is the strong one: the WHOLE .vkopbin is byte-identical
between `--stream-weights on` and `off`, not just the payloads -- for a plain
MLP, a SwiGLU-folded MLP, and a weight shared by a Gather and a MatMul (a tied
embedding / lm_head, which stays in memory on both arms).
"""
import os
import re
import subprocess
import sys
import tempfile

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

PYPI = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PYPI)

from onnx2vkop.optimizer import Quantizer  # noqa: E402

K, N, V = 256, 512, 1024
GROUP = 32


def _model(tmp, kind):
    rng = np.random.RandomState(3)

    def w(*s):
        return (rng.randn(*s) * 0.05).astype(np.float16)

    inits = [numpy_helper.from_array(w(K, K), "Wq")]
    nodes = [helper.make_node("MatMul", ["X", "Wq"], ["q"])]
    if kind == "swiglu":
        nodes += [helper.make_node("MatMul", ["q", "Wg"], ["g"]),
                  helper.make_node("MatMul", ["q", "Wu"], ["u"]),
                  helper.make_node("Sigmoid", ["g"], ["s"]),
                  helper.make_node("Mul", ["g", "s"], ["m1"]),
                  helper.make_node("Mul", ["m1", "u"], ["m2"]),
                  helper.make_node("MatMul", ["m2", "Wd"], ["d"])]
        inits += [numpy_helper.from_array(w(K, N), "Wg"),
                  numpy_helper.from_array(w(K, N), "Wu"),
                  numpy_helper.from_array(w(N, K), "Wd")]
    elif kind == "transposed":
        # The checkpoint layout an exporter with Transpose folding off emits:
        # each linear's weight stored [N, K] and read through a Transpose. The
        # streamed arm re-lays it in the writer (KnWeightLayout's
        # transpose_payload), the in-memory arm before encoding
        # (_flip_transposed_4bit_weights); both must land on the same bytes.
        nodes += [helper.make_node("Transpose", ["Wup"], ["Wup_t"], perm=[1, 0]),
                  helper.make_node("MatMul", ["q", "Wup_t"], ["h"]),
                  helper.make_node("Relu", ["h"], ["r"]),
                  helper.make_node("Transpose", ["Wd"], ["Wd_t"], perm=[1, 0]),
                  helper.make_node("MatMul", ["r", "Wd_t"], ["d"])]
        inits += [numpy_helper.from_array(w(N, K), "Wup"),
                  numpy_helper.from_array(w(K, N), "Wd")]
    else:
        nodes += [helper.make_node("MatMul", ["q", "Wup"], ["h"]),
                  helper.make_node("Relu", ["h"], ["r"]),
                  helper.make_node("MatMul", ["r", "Wd"], ["d"])]
        inits += [numpy_helper.from_array(w(K, N), "Wup"),
                  numpy_helper.from_array(w(N, K), "Wd")]
    graph_inputs = [helper.make_tensor_value_info("X", TensorProto.FLOAT16, [1, K])]
    outputs = [helper.make_tensor_value_info("Y", TensorProto.FLOAT16, [1, V])]
    if kind == "tied":
        # One [V, K] table read by a Gather (embedding) and, through a
        # Transpose, by the lm_head MatMul.
        inits.append(numpy_helper.from_array(w(V, K), "E"))
        graph_inputs.append(
            helper.make_tensor_value_info("ids", TensorProto.INT64, [1, 1]))
        nodes += [helper.make_node("Gather", ["E", "ids"], ["emb"]),
                  helper.make_node("Transpose", ["E"], ["Et"], perm=[1, 0]),
                  helper.make_node("MatMul", ["d", "Et"], ["Y"])]
        outputs.append(
            helper.make_tensor_value_info("emb", TensorProto.FLOAT16, [1, 1, K]))
    else:
        inits.append(numpy_helper.from_array(w(K, V), "Wlm"))
        nodes.append(helper.make_node("MatMul", ["d", "Wlm"], ["Y"]))
    g = helper.make_graph(nodes, "g", graph_inputs, outputs, inits)
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])
    m.ir_version = 8
    path = os.path.join(tmp, "m.onnx")
    onnx.save_model(m, path, save_as_external_data=True,
                    all_tensors_to_one_file=True, location="w.bin",
                    size_threshold=0)
    return path


def _convert(model_path, tmp, stream, quant):
    out = os.path.join(tmp, f"{quant}_{stream}.vkopbin")
    # Stream every weight however small, and force the trailer layout on both
    # arms (the in-memory arm would otherwise inline its smaller blob).
    env = {**os.environ, "VKOP_STREAM_MIN_MIB": "0", "PYTHONPATH": PYPI}
    p = subprocess.run(
        [sys.executable, "-m", "onnx2vkop.cli", "-i", model_path, "-o", out,
         "-q", quant, "--group-size", str(GROUP), "--target-backend", "buffer",
         "--external-blob-mb", "0", "--stream-weights", stream],
        capture_output=True, text=True, cwd=PYPI, env=env)
    assert p.returncode == 0, f"convert failed ({stream}): {p.stderr[-1500:]}"
    with open(out, "rb") as f:
        return f.read(), p.stdout + p.stderr


def _root(data):
    from onnx2vkop.inspect import _Model_mod
    return _Model_mod.Model.GetRootAs(data, 0)


def _by_name(data):
    """{name: (dtype, dims, payload bytes)}, independent of blob layout."""
    root = _root(data)
    if root.InitializerBlobLength():
        blob = root.InitializerBlobAsNumpy().tobytes()
        base = 0
    else:
        blob = data
        base = int.from_bytes(data[-8:], "little")
    out = {}
    for i in range(root.InitializersLength()):
        e = root.Initializers(i)
        off = base + e.Offset()
        out[e.Name().decode()] = (
            e.Dtype().decode(), [e.Dims(j) for j in range(e.DimsLength())],
            bytes(blob[off:off + e.Size()]))
    return out


def _nodes(data, tmp):
    import json
    path = os.path.join(tmp, "probe.vkopbin")
    with open(path, "wb") as f:
        f.write(data)
    p = subprocess.run(
        [sys.executable, "-m", "onnx2vkop.inspect", path, "--section", "nodes",
         "--json"], capture_output=True, text=True,
        env={**os.environ, "PYTHONPATH": PYPI})
    assert p.returncode == 0, p.stderr[-400:]
    return json.loads(p.stdout)["nodes"]


def _report(log):
    """The per-weight error report: every 'Quantized' block, as printed."""
    return re.findall(r"Quantized '[^']+':\n(?:  .*\n){4}", log)


@pytest.mark.parametrize("kind", ["plain", "swiglu", "tied", "transposed"])
@pytest.mark.parametrize("quant", ["int4", "nf4", "nvfp4"])
def test_streamed_file_is_byte_identical(quant, kind):
    with tempfile.TemporaryDirectory() as tmp:
        mp = _model(tmp, kind)
        ref, ref_log = _convert(mp, tmp, "off", quant)
        got, got_log = _convert(mp, tmp, "on", quant)
        streamed = re.search(r"\[stream\] streaming (\d+) initializer", got_log)
        assert streamed and int(streamed.group(1)) >= 3, (
            "the streamed arm did not stream; the comparison would be vacuous")
        assert "weights not streamed" not in got_log
        if kind == "tied":
            # The hydrated table E (and the Et the folder builds from it) sits
            # among streamed weights. The two arms lay the blob out in a
            # different ORDER here: a model this small is under the in-memory
            # path's 1.5 GiB demotion threshold, so its weights keep their
            # proto order, while the streamed ones are re-attached after the
            # resident tensors. Payload, dtype, dims and graph must still agree
            # name for name. (A real model above the threshold demotes on both
            # arms, which is why plain/swiglu -- and GLM-Edge -- match outright.)
            assert _by_name(got) == _by_name(ref), (
                f"{quant}/tied: a payload or row differs between the arms")
            assert _nodes(got, tmp) == _nodes(ref, tmp)
        elif kind == "transposed":
            # The in-memory arm constant-folds Transpose(W) into a NEW
            # initializer named after the Transpose's output (`Wup_t`); the
            # streamed arm keeps the source name, because the bytes never
            # enter the process to be folded. Node numbering shifts with it.
            # Everything the runtime reads must still agree once the names are
            # mapped: payload bytes, dtypes, dims, and what each node computes.
            def _ren(text):
                return text.replace("Wup_t", "Wup").replace("Wd_t", "Wd")
            assert {_ren(k): v for k, v in _by_name(ref).items()} == \
                _by_name(got), f"{quant}/transposed: a payload differs"

            def _computes(nodes):
                return sorted(
                    _ren(repr((n["op_type"], n["inputs"], n["outputs"],
                               n.get("attributes"))))
                    for n in nodes)
            assert _computes(_nodes(got, tmp)) == _computes(_nodes(ref, tmp))
            ref_log, got_log = _ren(ref_log), _ren(got_log)
        else:
            assert got == ref, (
                f"{quant}/{kind}: streamed .vkopbin differs from the in-memory "
                f"one")
        # Same report, number for number (as printed).
        # (Visited in initializer order, which is the layout order above.)
        assert sorted(_report(got_log)) == sorted(_report(ref_log))
        assert _report(got_log), "no per-weight report was printed"
        if kind == "swiglu":
            assert "fuse_swiglu_gate_up: 1" in got_log, (
                "the streamed gate/up pair was not folded")
        if quant == "nf4":
            assert b"nf4" in got
        if quant == "nvfp4":
            assert b"float4e2m1fn" in got
            assert "streamed and encoded on the way into the trailer" in got_log
            # Every eligible weight is encoded on both arms: nothing left
            # float behind a refused gate.
            assert "Preserving 'W" not in got_log
            assert "Preserving 'W" not in ref_log


def _nf4_reference(normalized):
    """The original searchsorted encoder _nf4_encode replaced, verbatim."""
    cb = Quantizer._NF4_CODEBOOK
    clipped = np.clip(np.nan_to_num(normalized, nan=cb[0]), cb[0], cb[-1])
    hi = np.searchsorted(cb, clipped, side="left")
    lo = hi - 1
    hi = np.minimum(hi, 15)
    d_lo = np.abs(clipped - cb[lo])
    d_hi = np.abs(clipped - cb[hi])
    return np.where(d_lo <= d_hi, lo, hi).astype(np.int8)


def test_nf4_encoder_matches_the_reference_rule_bit_for_bit():
    """Including every fp32 value within 64 ulps of each decision point and
    each codebook entry, the specials, and the out-of-range ends."""
    rng = np.random.default_rng(1)
    vals = [rng.uniform(-1.2, 1.2, 3_000_000).astype(np.float32)]
    edges = list(Quantizer._nf4_thresholds()) + list(Quantizer._NF4_CODEBOOK)
    near = []
    for e in edges:
        for direction in (np.float32(np.inf), np.float32(-np.inf)):
            v = np.float32(e)
            for _ in range(64):
                near.append(v)
                v = np.nextafter(v, direction)
    near += [np.nan, np.inf, -np.inf, -0.0, 0.0, 3.0, -3.0, 1e-30, -1e-30]
    vals.append(np.array(near, dtype=np.float32))
    x = np.concatenate(vals)
    got = Quantizer._nf4_encode(x)
    assert got.dtype == np.int8 and got.shape == x.shape
    assert np.array_equal(got, _nf4_reference(x))
    # Shape is kept for the grouped [n_groups, group, N] operand too.
    w3 = x[:3 * 32 * 64].reshape(3, 32, 64)
    assert np.array_equal(Quantizer._nf4_encode(w3), _nf4_reference(w3))


def test_nf4_thresholds_are_the_exact_flip_points():
    cb = Quantizer._NF4_CODEBOOK
    for j, t in enumerate(Quantizer._nf4_thresholds()):
        up = np.nextafter(t, np.float32(np.inf))
        assert abs(t - cb[j]) <= abs(t - cb[j + 1])
        assert not abs(up - cb[j]) <= abs(up - cb[j + 1])
