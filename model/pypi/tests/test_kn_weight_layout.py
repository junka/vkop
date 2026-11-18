"""Byte weights must end up as [K, N] with transB=0.

The runtime's split-K byte GEMV has two forms for one kernel family. Under
transB=1 it reads four columns per lane with consecutive lanes a whole K row
apart; under transB=0 the WIDE form reads 16 *consecutive* columns per lane.
Measured on the 27B at K=5120/N=10240 the two reach 33 GB/s and 213 GB/s on the
same card, so which one a converted weight lands on is a 6.5x on every decode
token.

`KnWeightLayout` is the pass that decides, and what these cases pin is that the
decision is about the PRODUCT rather than about how the source happens to look:

  * a [N, K] weight behind a folded Transpose is flipped, and the payload is
    exactly the transpose of what the old path wrote -- same codes, same scale
    per output column, byte for byte;
  * a weight that is ALREADY [K, N] is left alone -- the 27B exports two such
    groups ((5120, 6144) and (5120, 17408)) and a rule keyed on "first dim looks
    big" transposes them into garbage;
  * the scale keeps its per-output-column meaning, which is the failure mode
    that stays silent: the wrong axis leaves a correctly-sized payload that
    decodes with a scale from the wrong direction;
  * the streamed and in-memory paths agree, because a deferred weight's bytes
    are produced by the writer and a hydrated one's here;
  * a weight shared by two consumers, or read from the A side, or with a K the
    graph does not pin, keeps its exported layout.
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
K, N = 512, 2048          # K % 4 == 0, N % 16 == 0; 2 MiB fp16 source


def _weight(seed=7):
    rng = np.random.RandomState(seed)
    w = (rng.randn(N, K) * 0.05).astype(np.float16)
    w[0, 0] = 2.0         # the global max, in output column 0 / k 0
    return w


def _model(tmp, *, transposed=True, perm=(1, 0)):
    """MatMul over a weight in the checkpoint layout, as an exporter writes it.

    `transposed=False` hands the MatMul the already-flipped [K, N] weight, which
    is the shape the 27B's two linear-attention groups export.
    """
    w = _weight()
    if transposed:
        t = numpy_helper.from_array(w, "W")
        nodes = [helper.make_node("Transpose", ["W"], ["Wt"], perm=list(perm)),
                 helper.make_node("MatMul", ["X", "Wt"], ["Y"])]
    else:
        t = numpy_helper.from_array(np.ascontiguousarray(w.T.copy()), "W")
        nodes = [helper.make_node("MatMul", ["X", "W"], ["Y"])]
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


def _convert(model_path, tmp, *, stream="on", quant="int8", tag=""):
    out = os.path.join(tmp, "o_%s_%s%s.vkopbin" % (quant, stream, tag))
    p = subprocess.run(
        [sys.executable, "-m", "onnx2vkop.cli", "-i", model_path, "-o", out,
         "-q", quant, "--target-backend", "buffer",
         "--external-blob-mb", "1", "--stream-weights", stream],
        capture_output=True, text=True, cwd=PYPI)
    assert p.returncode == 0, f"convert failed ({stream}): {p.stderr[-900:]}"
    return out, p.stdout + p.stderr


def _section(binpath, sec):
    p = subprocess.run(
        [sys.executable, "-m", "onnx2vkop.inspect", binpath,
         "--section", sec, "--json"],
        capture_output=True, text=True,
        env={**os.environ, "PYTHONPATH": PYPI})
    assert p.returncode == 0, p.stderr[-400:]
    return json.loads(p.stdout)


def _tensor(binpath, name):
    """(side-table row, raw payload bytes) for one initializer."""
    summ = _section(binpath, "summary")["summary"]
    inits = {e["name"]: e for e in _section(binpath, "initializers")["initializers"]}
    row = inits[name]
    trailer = summ["external_blob_trailer_offset"]
    with open(binpath, "rb") as f:
        f.seek(trailer + row["offset"])
        raw = f.read(row["size"])
    return row, raw


def _matmul(binpath):
    for n in _section(binpath, "nodes")["nodes"]:
        if n.get("op_type") == "MatMul":
            return n
    raise AssertionError("no MatMul in the artifact")


def _payload(binpath, name="W"):
    row, raw = _tensor(binpath, name)
    return row, np.frombuffer(raw, dtype=np.int8)


def _scale(binpath, name="W_scale"):
    row, raw = _tensor(binpath, name)
    return row, np.frombuffer(raw, dtype=np.float32)


def _expected_scale(w):
    """Per-output-column absmax / 127 for a [N, K] source.

    The transpose this pass performs keeps every entry on the column it already
    belonged to, so this table is what the artifact must carry whether or not a
    flip happened.
    """
    amax = np.max(np.abs(w.astype(np.float32)), axis=1)
    return np.where(amax == 0, 1.0, amax / 127.0).astype(np.float32)


def _expected_codes(w):
    """The int8 codes of a [N, K] source, for comparison against a [K, N] payload."""
    s = _expected_scale(w)
    return np.round(w.astype(np.float32) / s[:, None]).astype(np.int8)


# --------------------------------------------------------------------------
# the decision
# --------------------------------------------------------------------------

def test_a_transpose_fed_weight_is_flipped_to_kn():
    """[N, K] + transB=1 becomes [K, N] + transB=0, and the numbers are the same.

    The reference is the in-memory quantization of the SOURCE, transposed: that
    is the arithmetic the MatMul did before this pass existed, so anything else
    is a wrong answer however well-formed the artifact looks.
    """
    w = _weight()
    with tempfile.TemporaryDirectory() as tmp:
        out, log = _convert(_model(tmp), tmp)
        assert "[kn] 1 byte weight(s) rewritten" in log, log[-1200:]

        row, pay = _payload(out)
        assert tuple(row["dims"]) == (K, N), row
        assert row["dtype"] == "int8"
        assert np.array_equal(pay.reshape(K, N),
                              np.ascontiguousarray(_expected_codes(w).T))

        assert _matmul(out).get("attributes", {}).get("transB") in (None, 0), \
            "a [K, N] weight must not carry transB"

        _, sc = _scale(out)
        assert len(sc) == N, "one scale entry per output column"
        assert np.allclose(sc, _expected_scale(w))


def test_an_already_kn_weight_is_left_alone():
    """The (5120, 6144)/(5120, 17408) shape: K is the LEADING dim.

    A rule that keyed on "first dim is the bigger one" would transpose this and
    every downstream answer would be wrong while the artifact still loaded.
    """
    # The direct export's weight, and the scale/codes the transB=0 kernel
    # expects for it: [K, N], so the output columns are axis 1.
    kn = np.ascontiguousarray(_weight().T)
    with tempfile.TemporaryDirectory() as tmp:
        out, log = _convert(_model(tmp, transposed=False), tmp)
        assert "[kn] 0 byte weight(s) rewritten" in log, log[-1200:]
        assert "already [K, N]" in log

        row, pay = _payload(out)
        assert tuple(row["dims"]) == (K, N), row
        amax = np.max(np.abs(kn.astype(np.float32)), axis=0)
        want_scale = np.where(amax == 0, 1.0, amax / 127.0).astype(np.float32)
        want = np.round(kn.astype(np.float32) / want_scale[None, :]).astype(np.int8)
        assert np.array_equal(pay.reshape(K, N), want)
        _, sc = _scale(out)
        assert np.allclose(sc, want_scale)


def test_the_streamed_and_hydrated_payloads_agree():
    """Same artifact either way. The streamed path reorders in the writer, the
    hydrated one in memory, and only one of them can be right."""
    with tempfile.TemporaryDirectory() as tmp:
        mp = _model(tmp)
        on, _ = _convert(mp, tmp, stream="on")
        off, _ = _convert(mp, tmp, stream="off")
        row_on, pay_on = _payload(on)
        row_off, pay_off = _payload(off)
        assert tuple(row_on["dims"]) == tuple(row_off["dims"]) == (K, N)
        assert pay_on.tobytes() == pay_off.tobytes()
        assert np.allclose(_scale(on)[1], _scale(off)[1])


@pytest.mark.parametrize("quant", ["int8", "fp8e4m3"])
def test_every_byte_format_is_flipped(quant):
    """fp8 has the same WIDE kernel and the same [N, K] handicap."""
    with tempfile.TemporaryDirectory() as tmp:
        out, log = _convert(_model(tmp), tmp, quant=quant)
        assert "[kn] 1 byte weight(s) rewritten" in log, log[-1200:]
        row, _ = _tensor(out, "W")
        assert tuple(row["dims"]) == (K, N)
        assert row["dtype"] in ("int8", "float8e4m3fn")


def test_the_scale_axis_is_the_column_axis_not_k():
    """The silent failure this pass could introduce.

    If the transpose ran BEFORE the quantization, the absmax would reduce over K
    and the table would have K entries. Both artifacts load; only one decodes
    correctly. The marker weight (a 2.0 spike at column 0, k 0) is the sharpest
    witness: it is the largest value in output column 0, so scale[0] must be the
    table's maximum.
    """
    w = _weight()
    with tempfile.TemporaryDirectory() as tmp:
        out, _ = _convert(_model(tmp), tmp)
        _, sc = _scale(out)
        assert len(sc) == N, f"scale has {len(sc)} entries; K is {K}, N is {N}"
        assert int(np.argmax(sc)) == 0
        assert abs(float(sc[0]) - float(np.abs(w[0]).max()) / 127.0) < 1e-7


# --------------------------------------------------------------------------
# the refusals
# --------------------------------------------------------------------------

def _dag_with(weight_dims, consumers, activation_k):
    """A hand-built DAG for the decision rule alone."""
    from onnx2vkop.dag import DAGBasedModel, Node

    m = DAGBasedModel()
    m.initializers["W"] = numpy_helper.from_array(
        np.zeros(weight_dims, dtype=np.int8), "W")
    m.initializers["W_scale"] = numpy_helper.from_array(
        np.ones(weight_dims[0], dtype=np.float32), "W_scale")
    m.inputs.append({"name": "X", "shape": [1, activation_k],
                     "dtype": "float16"})
    for i, (transb, b_position) in enumerate(consumers):
        ins = ([{"name": "X", "shape": [1, activation_k]},
                {"name": "W", "shape": list(weight_dims)},
                {"name": "W_scale", "shape": [weight_dims[0]]}]
               if b_position else
               [{"name": "W", "shape": list(weight_dims)},
                {"name": "X", "shape": [1, activation_k]}])
        attrs = {} if transb == 0 else {"transB": transb}
        m.nodes["mm%d" % i] = Node(
            "MatMul", "mm%d" % i, attrs, ins,
            [{"name": "Y%d" % i, "shape": [1, weight_dims[0]]}])
    return m


def test_a_weight_read_by_two_consumers_keeps_its_layout():
    from onnx2vkop.optimizer import KnWeightLayout

    m = _dag_with([N, K], [(1, True), (1, True)], K)
    assert KnWeightLayout.apply(m, verbose=False) == 0
    assert tuple(m.initializers["W"].dims) == (N, K)
    assert m.nodes["mm0"].attributes.get("transB") == 1


def test_a_weight_in_the_activation_slot_is_not_touched():
    from onnx2vkop.optimizer import KnWeightLayout

    m = _dag_with([N, K], [(1, False)], N)
    assert KnWeightLayout.apply(m, verbose=False) == 0
    assert tuple(m.initializers["W"].dims) == (N, K)


def test_an_unpinned_k_keeps_the_exported_layout():
    """A symbolic K has nothing to check the orientation against."""
    from onnx2vkop.dag import DAGBasedModel, Node
    from onnx2vkop.optimizer import KnWeightLayout

    m = DAGBasedModel()
    m.initializers["W"] = numpy_helper.from_array(
        np.zeros((N, K), dtype=np.int8), "W")
    m.nodes["mm"] = Node(
        "MatMul", "mm", {"transB": 1},
        [{"name": "X", "shape": [-1, -1]},
         {"name": "W", "shape": [N, K]}],
        [{"name": "Y", "shape": [-1, N]}])
    assert KnWeightLayout.apply(m, verbose=False) == 0
    assert tuple(m.initializers["W"].dims) == (N, K)


def test_a_transb_zero_weight_whose_dims_contradict_k_is_left_alone():
    """transB=0 promises [K, N]. A weight whose leading dim is not A's K is one
    the graph describes differently than this pass assumes -- not something to
    silently re-lay."""
    from onnx2vkop.optimizer import KnWeightLayout

    m = _dag_with([N, K], [(0, True)], K)   # leading dim N != K
    assert KnWeightLayout.apply(m, verbose=False) == 0
    assert tuple(m.initializers["W"].dims) == (N, K)


def test_fp16_weights_are_never_flipped():
    """The fp16 tile stages B once per output block, so its layout is not the
    bottleneck -- and its kernels have their own gates."""
    from onnx2vkop.dag import DAGBasedModel, Node
    from onnx2vkop.optimizer import KnWeightLayout

    m = DAGBasedModel()
    m.initializers["W"] = numpy_helper.from_array(
        np.zeros((N, K), dtype=np.float16), "W")
    m.nodes["mm"] = Node(
        "MatMul", "mm", {"transB": 1},
        [{"name": "X", "shape": [1, K]}, {"name": "W", "shape": [N, K]}],
        [{"name": "Y", "shape": [1, N]}])
    assert KnWeightLayout.apply(m, verbose=False) == 0
    assert tuple(m.initializers["W"].dims) == (N, K)
    assert m.nodes["mm"].attributes.get("transB") == 1


def test_the_node_records_the_payload_shape_it_will_read():
    """The runtime reshapes each input to its consumer's recorded view, so a
    flipped payload declared as [N, K] on the node is a wrong answer, not a
    cosmetic mismatch."""
    from onnx2vkop.optimizer import KnWeightLayout

    m = _dag_with([N, K], [(1, True)], K)
    assert KnWeightLayout.apply(m, verbose=False) == 1
    assert tuple(m.nodes["mm0"].inputs[1]["shape"]) == (K, N)
    assert "transB" not in m.nodes["mm0"].attributes
    # The re-laid payload really is the transpose (all zeros here, so compare
    # the element count and shape rather than the values).
    assert tuple(m.initializers["W"].dims) == (K, N)
