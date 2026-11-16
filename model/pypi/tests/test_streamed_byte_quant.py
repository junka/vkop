"""A streamed byte-format weight must convert to the same bytes as an in-memory one.

`-q int8` (and fp8) has two paths to the same artifact:

  * `--stream-weights off` -- prepare hydrates, fuse_transpose_into_matmul
    rewrites `MatMul(X, Transpose(W))` to `MatMul(X, W, transB=1)`, and the
    ordinary quantizer sees [N, K] and appends its per-output-column scale.
  * `--stream-weights on` -- the weight stays in its file and quant_payload
    quantizes it on the way into the trailer.

The two are supposed to agree byte for byte. They did not, in two independent
ways, and nothing caught it because the exported Qwen3.5 graph is the first
thing to combine "weights behind a Transpose" with "a byte-format quantization
that defers":

  1. finalize appended the scale to whoever READS the weight -- the Transpose --
     and fold_transpose_into_matmul then rebuilt the MatMul as [A, B], stranding
     every scale. The runtime rejected all 401 weights:
         MatMul with a int8 weight needs exactly [A, B, scale], got 2 inputs
  2. the scale reduced the wrong axis. A scale is per OUTPUT COLUMN, and in the
     stored checkpoint layout [N, K] those columns are axis 0, not axis 1. The
     payload still had the right size and the artifact still loaded, so the only
     symptom would have been wrong numbers.

Both are pinned here against the in-memory path, which is the reference: same
payload bytes, same scale values, same MatMul arity. A test of the streamed path
alone would have agreed with itself.

  3. `finalize` appended the scale to the node that reads the weight, which on
     this graph is the Transpose -- making a 2-input unary op. onnxoptimizer's
     IR asserts on that ("ir.h:554: input: Assertion inputs_.size() == 1
     failed"), and `optimize_model` catches it and skips its whole structural
     pass set. ConstantFolder alone then folds a different Shape/Slice/Gather
     chain, leaving attention's Q declared as [-1,-1,-1,256] instead of
     [-1,24,-1,256]; the runtime derived num_heads = -1 and rejected the 27B
     artifact with the misleading
         Attention: head_dim must be even and <= 256

     Like (2) it is invisible in the numbers -- the payload and scale are both
     right -- so it is pinned structurally, in test_finalize_* below.
"""
import json
import os
import subprocess
import sys
import tempfile

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

PYPI = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
K, N = 512, 2048          # N % 8 == 0, K % 8 == 0; 2 MiB fp16


def _model(tmp, perm=(1, 0)):
    rng = np.random.RandomState(7)
    w = (rng.randn(N, K) * 0.05).astype(np.float16)
    w[0, 0] = 2.0                       # the global max, as a landmark
    if perm is None:
        # Direct: the weight is already [K, N].
        t = numpy_helper.from_array(w.T.copy(), "W")
        nodes = [helper.make_node("MatMul", ["X", "W"], ["Y"])]
    else:
        t = numpy_helper.from_array(w, "W")
        nodes = [helper.make_node("Transpose", ["W"], ["Wt"], perm=list(perm)),
                 helper.make_node("MatMul", ["X", "Wt"], ["Y"])]
    g = helper.make_graph(
        nodes, "g",
        [helper.make_tensor_value_info("X", TensorProto.FLOAT16, [1, K])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT16, [1, N])], [t])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])
    m.ir_version = 8
    path = os.path.join(tmp, "m.onnx")
    onnx.save_model(m, path, save_as_external_data=True,
                    all_tensors_to_one_file=True, location="w.bin",
                    size_threshold=0)
    return path


def _convert(model_path, tmp, stream, quant="int8"):
    out = os.path.join(tmp, "o_%s_%s.vkopbin" % (quant, stream))
    p = subprocess.run(
        [sys.executable, "-m", "onnx2vkop.cli", "-i", model_path, "-o", out,
         "-q", quant, "--target-backend", "buffer",
         # Force the trailer: a 2 MiB blob otherwise trips dag.py's
         # "fits a FlatBuffer vector" gate and never defers at all.
         "--external-blob-mb", "1",
         "--stream-weights", stream],
        capture_output=True, text=True, cwd=PYPI)
    assert p.returncode == 0, f"convert failed ({stream}): {p.stderr[-800:]}"
    return out


def _section(binpath, sec):
    p = subprocess.run(
        [sys.executable, "-m", "onnx2vkop.inspect", binpath,
         "--section", sec, "--json"],
        capture_output=True, text=True,
        env={**os.environ, "PYTHONPATH": PYPI})
    assert p.returncode == 0, p.stderr[-400:]
    return json.loads(p.stdout)


def _trailer_pieces(binpath):
    """(quantized payload bytes, fp32 scale values) out of the trailer.

    The payload dtype is the format's own: int8 names itself, the fp8 flavours
    are float8e4m3fn / float8e5m2. Anything one byte, one value.
    """
    summ = _section(binpath, "summary")["summary"]
    inits = _section(binpath, "initializers")["initializers"]
    trailer = summ["external_blob_trailer_offset"]
    with open(binpath, "rb") as f:
        blob = f.read()
    payload = scale = None
    for e in inits:
        raw = blob[trailer + e["offset"]:trailer + e["offset"] + e["size"]]
        if e.get("dtype") in ("int8", "float8e4m3fn", "float8e5m2"):
            payload = raw
        elif e.get("dtype") == "float32" and e["name"].endswith("_scale"):
            scale = np.frombuffer(raw, dtype="<f4").copy()
    return payload, scale


def _matmul_inputs(binpath):
    for n in _section(binpath, "nodes")["nodes"]:
        if n.get("op_type") in ("MatMul", "Gemm"):
            return [i if isinstance(i, str) else i.get("name")
                    for i in n.get("inputs", [])]
    return []


@pytest.mark.parametrize("quant", ["int8", "fp8e4m3"])
def test_streamed_matches_in_memory(quant):
    """The whole point: same artifact either way, for a Transpose-fed weight."""
    with tempfile.TemporaryDirectory() as tmp:
        mp = _model(tmp)
        ref = _convert(mp, tmp, "off", quant)
        st = _convert(mp, tmp, "on", quant)

        ref_pay, ref_scale = _trailer_pieces(ref)
        st_pay, st_scale = _trailer_pieces(st)
        assert st_pay is not None, "streamed artifact has no quantized payload"
        assert st_scale is not None, (
            "streamed artifact lost its scale -- this is the bug that made the "
            "runtime reject every weight with 'needs exactly [A, B, scale]'")

        # The axis: one scale per OUTPUT COLUMN, which for the stored [N, K]
        # layout is dims[0].
        assert len(st_scale) == N, (
            f"streamed scale has {len(st_scale)} entries; a per-output-column "
            f"scale of this weight needs {N} (K is {K})")
        assert np.allclose(st_scale, ref_scale), (
            "streamed and in-memory scales disagree -- wrong reduction axis")

        # The codes, which depend on the scale above.
        assert st_pay == ref_pay, (
            "streamed and in-memory payloads differ byte for byte")

        # And the MatMul still carries the scale as its third input.
        assert _matmul_inputs(st) == ["X", "W", "W_scale"], _matmul_inputs(st)


def test_the_marker_weight_keeps_the_largest_scale():
    """W[0,0] is the global max, and it is output column 0, so scale[0] must be
    the largest entry. This is what distinguishes the two axes outright: on the
    wrong one the largest scale lands at index 0 of a K-length table."""
    with tempfile.TemporaryDirectory() as tmp:
        mp = _model(tmp)
        st = _convert(mp, tmp, "on")
        _, scale = _trailer_pieces(st)
        assert len(scale) == N
        assert int(np.argmax(scale)) == 0, (
            f"largest scale is at index {int(np.argmax(scale))}, not 0 -- the "
            f"scale was reduced along the wrong axis")


def test_a_direct_weight_still_reduces_the_other_axis():
    """The non-transposed layout must keep working: there the columns ARE axis
    1 of [K, N], so the same fix must not flip it the other way."""
    with tempfile.TemporaryDirectory() as tmp:
        mp = _model(tmp, perm=None)
        ref = _convert(mp, tmp, "off")
        st = _convert(mp, tmp, "on")
        ref_pay, ref_scale = _trailer_pieces(ref)
        st_pay, st_scale = _trailer_pieces(st)
        assert len(st_scale) == N
        assert np.allclose(st_scale, ref_scale)
        assert st_pay == ref_pay


def test_the_fold_keeps_a_scale_that_sits_on_the_transpose():
    """The narrow unit: fold_transpose_into_matmul must carry the Transpose's
    extra inputs onto the node it builds. Scale on the Transpose is exactly what
    finalize produces on a streamed graph."""
    sys.path.insert(0, PYPI)
    from onnx2vkop.dag import DAGBasedModel, Node
    from onnx2vkop.optimizer import FusionOptimizer

    m = DAGBasedModel()
    m.initializers["W"] = numpy_helper.from_array(
        np.zeros((8, 4), np.float16), "W")
    m.initializers["S"] = numpy_helper.from_array(
        np.ones(8, np.float32), "S")
    # The scale rides on the Transpose, which is where finalize put it.
    m.nodes["t"] = Node("Transpose", "t", {"perm": [1, 0]},
                        [{"name": "W", "shape": [8, 4]},
                         {"name": "S", "shape": [8]}],
                        [{"name": "T", "shape": [4, 8]}])
    m.nodes["mm"] = Node("MatMul", "mm", {},
                         [{"name": "X", "shape": [-1, 4]},
                          {"name": "T", "shape": [4, 8]}],
                         [{"name": "Y", "shape": [-1, 8]}])
    matches = FusionOptimizer.match_transpose_into_matmul(m)
    assert len(matches) == 1
    FusionOptimizer.fold_transpose_into_matmul(m, matches[0])
    names = [i["name"] for i in m.nodes["mm"].inputs]
    assert names == ["X", "W", "S"], names
    assert m.nodes["mm"].attributes["transB"] == 1


def _finalize_probe(tmp):
    """Run prepare + finalize on a Transpose-fed model; return the proto.

    Small enough to inspect node by node, shaped like the 27B: the weight sits
    behind a Transpose, so finalize has a choice of which node to give the scale
    to -- and only one of them keeps the graph legal.
    """
    sys.path.insert(0, PYPI)
    from onnx2vkop import quant_payload, weight_stream

    mp = _model(tmp)
    m = onnx.load(mp, load_external_data=False)
    deferred = weight_stream.prepare(m, tmp, quant="int8", unify=False,
                                     rgba=False, verbose=False)
    assert any(e.get("quant") for e in deferred.values()), "nothing deferred"
    n = quant_payload.finalize(m, deferred, verbose=False)
    assert n == 1, f"finalize attached {n} scale(s), expected 1"
    return m


def test_finalize_never_makes_a_unary_view_multi_input():
    """The scale must not land on the Transpose.

    Transpose is unary in the ONNX schema. A 2-input Transpose is an illegal
    node that onnxoptimizer's IR asserts on, which silently skips every
    structural pass -- and that is what left the 27B's Q head axis symbolic.
    """
    with tempfile.TemporaryDirectory() as tmp:
        m = _finalize_probe(tmp)
        bad = [(n.op_type, n.name, list(n.input)) for n in m.graph.node
               if n.op_type in ("Transpose", "Reshape", "Squeeze", "Unsqueeze",
                                "Identity", "Flatten")
               and len(n.input) != 1]
        assert not bad, (
            "finalize appended an operand to a unary view: " + repr(bad))


def test_finalize_puts_the_scale_on_the_weight_consumer():
    """Same artifact, and the MatMul is where the in-memory path puts it.

    The proto still spells B as `Wt` -- folding the Transpose into the MatMul is
    a DAG-stage rewrite (fuse_transpose_into_matmul) that has not run yet. What
    matters here is only WHICH node carries the scale.
    """
    with tempfile.TemporaryDirectory() as tmp:
        m = _finalize_probe(tmp)
        mm = [n for n in m.graph.node if n.op_type == "MatMul"]
        assert len(mm) == 1
        assert list(mm[0].input) == ["X", "Wt", "W_scale"], list(mm[0].input)
        tr = [n for n in m.graph.node if n.op_type == "Transpose"][0]
        assert list(tr.input) == ["W"], list(tr.input)


def test_the_finalized_proto_survives_the_converter_s_pass_list():
    """The decisive one: onnxoptimizer must not raise on the proto finalize
    produced. When it raises, optimize_model falls back to ConstantFolder alone,
    which is the graph drift that broke attention on the 27B."""
    onnxoptimizer = pytest.importorskip("onnxoptimizer")
    sys.path.insert(0, PYPI)
    from onnx2vkop.optimizer import ONNXOPTIMIZER_PASSES

    with tempfile.TemporaryDirectory() as tmp:
        m = _finalize_probe(tmp)
        try:
            out = onnxoptimizer.optimize(m, ONNXOPTIMIZER_PASSES)
        except Exception as e:      # noqa: BLE001 -- the assertion is the point
            raise AssertionError(
                "onnxoptimizer rejects the finalized graph, so optimize_model "
                f"would skip every structural pass: {e}") from e
        assert len(out.graph.node) > 0
