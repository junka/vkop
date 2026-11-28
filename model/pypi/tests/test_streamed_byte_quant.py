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


def _convert_forced_trailer(model_path, tmp, stream, quant, group,
                            tag=""):
    """Convert with the blob forced into the trailer.

    `--external-blob-mb 1` is enough for the streamed arm (a deferred weight
    costs the SOURCE's bytes) but not for an in-memory one: the hydrated
    int4 payload is half the fp16 source, which lands under the gate and
    inlines into the FlatBuffer, where there is no trailer offset to read.
    """
    out = os.path.join(tmp, "f_%s_%s%s.vkopbin" % (quant, stream, tag))
    p = subprocess.run(
        [sys.executable, "-m", "onnx2vkop.cli", "-i", model_path, "-o", out,
         "-q", quant, "--group-size", str(group),
         "--target-backend", "buffer", "--external-blob-mb", "0",
         "--stream-weights", stream],
        capture_output=True, text=True, cwd=PYPI)
    assert p.returncode == 0, f"convert failed ({stream}): {p.stderr[-800:]}"
    return out


def _model_weight_nk():
    """The exact [N, K] fp16 array `_model` writes, so a test can compute the
    reference from the same numbers the converter read."""
    rng = np.random.RandomState(7)
    w = (rng.randn(N, K) * 0.05).astype(np.float16)
    w[0, 0] = 2.0
    return w


def _convert(model_path, tmp, stream, quant="int8", extra=()):
    out = os.path.join(tmp, "o_%s_%s%s.vkopbin" % (quant, stream,
                                                   "".join(extra)))
    p = subprocess.run(
        [sys.executable, "-m", "onnx2vkop.cli", "-i", model_path, "-o", out,
         "-q", quant, "--target-backend", "buffer",
         # Force the trailer: a 2 MiB blob otherwise trips dag.py's
         # "fits a FlatBuffer vector" gate and never defers at all.
         "--external-blob-mb", "1",
         "--stream-weights", stream, *extra],
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


# --------------------------------------------------------------------------
# int4: a grouped scale that is local to a band, so it streams too
# --------------------------------------------------------------------------
#
# The scale of a 4-bit weight is [K/group, N] — one absmax per (group of K,
# output column). That table LOOKS like a whole-matrix reduction, and the
# deferral gate refused it for as long as it was read that way. It is local:
# each entry depends only on its own `group` rows, so a band-at-a-time sweep
# adds to the same accumulator and lands on the same table. These cases pin the
# three things that make the streamed payload usable at all:
#
#   * the payload and the table are bit-identical to the in-memory quantizer's;
#   * the band boundary may fall INSIDE a group (ROW_BLOCK is 2048 and `group`
#     is a CLI knob, so nothing guarantees they line up);
#   * a [N, K] source behind a Transpose is re-laid as [K, N] and packed two
#     columns per byte, which is the only layout `unpack_w4` can read.

import numpy as np  # noqa: E402  (already imported above; kept explicit here)

GROUP = 32


def _int4_reference(kn, group=GROUP):
    """(packed nibbles, [n_groups, N] scale) the in-memory quantizer writes.

    Uses Quantizer._quantize_w3 directly -- the one encoder the -q path, the
    AWQ path and this streaming path are all supposed to agree with -- rather
    than restating the arithmetic here, which would only prove the test can
    copy itself.
    """
    sys.path.insert(0, PYPI)
    from onnx2vkop.optimizer import Quantizer

    k, n = kn.shape
    codes, scale_keep = Quantizer._quantize_w3(
        kn.reshape(k // group, group, n), "int4")
    nib = codes.reshape(-1).astype(np.uint8) & 0xF
    packed = (nib[0::2] | (nib[1::2] << 4)).astype(np.uint8)
    return packed, scale_keep.reshape(k // group, n).astype(np.float32)


def _int4_row_and_bytes(binpath, name):
    summ = _section(binpath, "summary")["summary"]
    rows = {e["name"]: e for e in _section(binpath, "initializers")["initializers"]}
    row = rows[name]
    with open(binpath, "rb") as f:
        f.seek(summ["external_blob_trailer_offset"] + row["offset"])
        return row, f.read(row["size"])


def test_int4_streams_the_transpose_fed_weight_byte_for_byte():
    """The 27B's shape: [N, K] behind a Transpose, packed as [K, N].

    A source band is a run of output COLUMNS, so the writer has to transpose
    and pack on the way out. Two nibbles per byte means the unit it can write
    per k is half the band -- which is why an odd band start would silently
    corrupt a byte, and why the writer checks for it.
    """
    w_nk = _model_weight_nk()
    with tempfile.TemporaryDirectory() as tmp:
        st = _convert(_model(tmp), tmp, "on", quant="int4",
                      extra=("--group-size", str(GROUP)))
        row, payload = _int4_row_and_bytes(st, "W")
        srow, scale_raw = _int4_row_and_bytes(st, "W_scale")
        assert row["dtype"] == "int4", row
        assert tuple(row["dims"]) == (K, N), (
            f"payload dims {row['dims']}; the nibble kernel reads [K, N]")
        assert srow["dims"] == [K // GROUP, N], (
            f"scale table is {srow['dims']}, wants [K/group, N]")

        want_pay, want_scale = _int4_reference(
            np.ascontiguousarray(w_nk.T).astype(np.float32))
        assert payload == want_pay.tobytes(), (
            "streamed nibbles differ from the in-memory quantizer's")
        got_scale = np.frombuffer(scale_raw, dtype=np.float32)
        assert np.array_equal(got_scale, want_scale.reshape(-1)), (
            "streamed group scales differ from the in-memory quantizer's")


def test_int4_streams_a_direct_kn_weight_byte_for_byte():
    """The already-[K, N] layout: bands are K rows and no reordering happens."""
    w_kn = np.ascontiguousarray(_model_weight_nk().T)
    with tempfile.TemporaryDirectory() as tmp:
        st = _convert(_model(tmp, perm=None), tmp, "on", quant="int4",
                      extra=("--group-size", str(GROUP)))
        row, payload = _int4_row_and_bytes(st, "W")
        assert row["dtype"] == "int4" and tuple(row["dims"]) == (K, N)
        want_pay, _ = _int4_reference(w_kn.astype(np.float32))
        assert payload == want_pay.tobytes()


def test_int4_streamed_and_hydrated_payloads_agree():
    """Same artifact either way, for BOTH source layouts.

    The streamed path reorders packed nibbles in the writer
    (`weight_stream._write_int4`); the hydrated one transposes the VALUES first
    (`Quantizer._flip_transposed_4bit_weights`) and then packs. Those are
    different code paths reaching the same bytes, and only one of them can be
    right -- on the 27B a disagreement here would have been silently wrong
    weights in whichever arm took the other branch.

    Byte equality is the claim, not "close enough": the group absmax covers the
    same set of values in either layout, so the codes and the [K/group, N]
    table have to come out identical.
    """
    for perm in ((1, 0), None):
        tag = "transposed" if perm else "direct"
        with tempfile.TemporaryDirectory() as tmp:
            mp = _model(tmp, perm=perm)
            # `--external-blob-mb 0` on BOTH arms: the hydrated int4 artifact
            # inlines its blob (a 2 MiB fp16 source becomes a 1 MiB payload,
            # under the 1 MiB gate), and the inline layout has no trailer to
            # read a payload back from. The streamed arm always needs the
            # trailer. Forcing it keeps the two comparable.
            st = _convert(mp, tmp, "on", quant="int4",
                          extra=("--group-size", str(GROUP)))
            hy = _convert_forced_trailer(mp, tmp, "off", "int4", GROUP)
            srow, spay = _int4_row_and_bytes(st, "W")
            hrow, hpay = _int4_row_and_bytes(hy, "W")
            assert srow["dtype"] == hrow["dtype"] == "int4", (tag, srow, hrow)
            assert tuple(srow["dims"]) == tuple(hrow["dims"]) == (K, N), tag
            assert spay == hpay, f"{tag}: streamed payload differs from hydrated"

            _, sscale = _int4_row_and_bytes(st, "W_scale")
            _, hscale = _int4_row_and_bytes(hy, "W_scale")
            assert sscale == hscale, f"{tag}: scale tables differ"


def test_int4_group_scale_is_local_to_a_band(monkeypatch):
    """A band boundary inside a group must not lose or gain an absmax.

    ROW_BLOCK is 2048 and `group` is a CLI knob, so the two are not required to
    line up — and a band that starts mid-group is exactly the case a
    "ROW_BLOCK % group_size == 0" assumption would get wrong. Sweeping with
    ROW_BLOCK=7 (a prime) over a K of 512 puts a boundary inside nearly every
    group.
    """
    sys.path.insert(0, PYPI)
    from onnx2vkop import quant_payload
    from onnx2vkop.weight_stream import DeferredWeightError

    rng = np.random.RandomState(13)
    w_kn = (rng.randn(K, N) * 0.05).astype(np.float16)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "src.bin")
        w_kn.tofile(path)
        entry = {"name": "W", "location": path, "base_dir": tmp, "offset": 0,
                 "length": w_kn.nbytes, "data_type": TensorProto.FLOAT16,
                 "src_dtype": TensorProto.FLOAT16,
                 "dims": tuple(w_kn.shape), "transposed": False}
        reader = quant_payload.WeightReader(entry, "W")
        want = _int4_reference(w_kn.astype(np.float32))[1]

        seen = set()

        orig = quant_payload.WeightReader.bands
        try:
            for block in (7, 31, 64, 2048):
                def patched(self, _block=block):
                    return orig(self, block=_block)
                quant_payload.WeightReader.bands = patched
                got = quant_payload.grouped_amax(
                    quant_payload.WeightReader(entry, "W"), GROUP, False)
                assert np.array_equal(got / 7.0, want) or np.array_equal(
                    got, want * 7.0), (
                    f"block={block} disagrees with the whole-matrix grouping")
                seen.add(block)
        finally:
            quant_payload.WeightReader.bands = orig
        assert seen == {7, 31, 64, 2048}


def test_int4_refuses_a_group_that_does_not_divide_k():
    """A partial last group is a payload no kernel can walk, so it is a skip."""
    rng = np.random.RandomState(14)
    with tempfile.TemporaryDirectory() as tmp:
        # K=512 with group 48: 512 % 48 == 32.
        out = _convert(_model(tmp), tmp, "on", quant="int4",
                       extra=("--group-size", "48"))
        row, _ = _int4_row_and_bytes(out, "W")
        assert row["dtype"] == "float16", (
            "an ungroupable K must leave the weight wide, not ship nibbles")


@pytest.mark.parametrize("quant", ["int8", "int4"])
def test_exclusion_keeps_streaming_the_other_weights(quant):
    """--exclude-node used to force --stream-weights off (the trailer-time
    quantizer cannot see an exclusion), which loaded the whole model: a 40 GB
    DiT was OOM-killed for excluding one 2 MB head. Now the excluded weight
    alone is read into memory and stays float; every other one still streams
    and quantizes."""
    rng = np.random.RandomState(3)
    wq = (rng.randn(K, N) * 0.05).astype(np.float16)
    wk = (rng.randn(K, N) * 0.05).astype(np.float16)
    with tempfile.TemporaryDirectory() as tmp:
        g = helper.make_graph(
            [helper.make_node("MatMul", ["X", "Wq"], ["Yq"], name="quantized"),
             helper.make_node("MatMul", ["X", "Wk"], ["Yk"], name="kept")],
            "g",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT16, [1, K])],
            [helper.make_tensor_value_info("Yq", TensorProto.FLOAT16, [1, N]),
             helper.make_tensor_value_info("Yk", TensorProto.FLOAT16, [1, N])],
            [numpy_helper.from_array(wq, "Wq"), numpy_helper.from_array(wk, "Wk")])
        m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])
        m.ir_version = 8
        mp = os.path.join(tmp, "m.onnx")
        onnx.save_model(m, mp, save_as_external_data=True,
                        all_tensors_to_one_file=True, location="w.bin",
                        size_threshold=0)
        out = os.path.join(tmp, "o.vkopbin")
        p = subprocess.run(
            [sys.executable, "-m", "onnx2vkop.cli", "-i", mp, "-o", out,
             "-q", quant, "--group-size", "64", "--target-backend", "buffer",
             "--external-blob-mb", "1", "--stream-weights", "on",
             "--exclude-node", "kept"],
            capture_output=True, text=True, cwd=PYPI)
        assert p.returncode == 0, p.stderr[-800:]
        assert "[stream] streaming 1 initializer(s)" in p.stdout, p.stdout[-1500:]
        dt = {e["name"]: e.get("dtype")
              for e in _section(out, "initializers")["initializers"]}
        assert dt["Wk"] == "float16", dt
        assert dt["Wq"] == quant, dt
