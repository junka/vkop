"""Importing bitsandbytes NF4 weights: com.microsoft::MatMulBnb4 -> nf4 MatMul.

ONNX has no NF4 element type, so a pre-quantized NF4 checkpoint reaches the
converter as onnxruntime's contrib op (what MatMulBnb4Quantizer and the
bitsandbytes exporters emit). PreQuantizedWeights._fold_bnb4 rewrites it into
the exact [A, W_nf4, W_scale] MatMul `-q nf4` emits, copying the codes -- a
lossless import, the 4-bit analog of --keep-bf16.

What is pinned here, each against onnxruntime itself where it can be:

  * the contract: B packs the TRANSPOSED [N, K] weight, first element in the
    HIGH nibble; absmax is one entry per block_size run of that stream; the
    codebook is bitsandbytes' NF4 table -- which vkop's _NF4_CODEBOOK (and the
    shader copies) must equal bit-for-bit. ORT is run on A = identity, so its
    output IS the decoded weight, and the import must reproduce it exactly;
  * bit identity with the -q nf4 twin (same payload bytes, same scale bytes,
    same input list) for several block sizes;
  * every gate: FP4, a block spanning two output columns, N % 8, an odd block,
    transB=0, byte/absmax count mismatches, non-finite absmax, a shared or
    runtime-computed payload;
  * fp16/bf16 absmax widened exactly; a payload shared by two nodes;
  * the whole converter (in-memory and external-data sources);
  * opt-in GPU end to end (VKOP_BENCH): the converted vkopbin against the
    original MatMulBnb4 model in onnxruntime via tools/vkop_vs_ort.py, for
    decode (M=1) and tile (M>=16) shapes.
"""
import contextlib
import io
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy
import onnx
import pytest
from onnx import TensorProto, external_data_helper, helper, numpy_helper

from onnx2vkop.converter import ModelConverter
from onnx2vkop.dag import DAGBasedModel, Node
from onnx2vkop.optimizer import PreQuantizedWeights, Quantizer

REPO_ROOT = Path(__file__).resolve().parents[3]

# bitsandbytes' NF4 table (functional.py, get_4bit_type("nf4")), restated: the
# converter's copy has to equal this, and onnxruntime has to decode with it.
BNB_NF4 = numpy.array(
    [-1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
     -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
     0.07958029955625534, 0.16093020141124725, 0.24611230194568634,
     0.33791524171829224, 0.44070982933044434, 0.5626170039176941,
     0.7229568362236023, 1.0],
    dtype=numpy.float32)


# --------------------------------------------------------------------------
# builders
# --------------------------------------------------------------------------

def pack_bnb(codes_nk):
    """[N, K] code grid -> MatMulBnb4's uint8 B: row-major, first element in
    the HIGH nibble."""
    flat = numpy.asarray(codes_nk, dtype=numpy.uint8).reshape(-1)
    if flat.size % 2:
        flat = numpy.concatenate([flat, numpy.zeros(1, numpy.uint8)])
    return ((flat[0::2] << 4) | flat[1::2]).astype(numpy.uint8)


def random_bnb(k, n, block, seed=0, absmax_dtype=numpy.float32):
    rng = numpy.random.default_rng(seed)
    codes = rng.integers(0, 16, size=(n, k), dtype=numpy.uint8)
    absmax = (rng.random(n * k // block) * 0.2 + 0.01).astype(absmax_dtype)
    return codes, absmax


def bnb_dag(k, n, block, codes_nk=None, absmax=None, *, qt=1, attrs=None,
            b_init=None, absmax_init=None, seed=0):
    if codes_nk is None:
        codes_nk, absmax = random_bnb(k, n, block, seed)
    m = DAGBasedModel()
    m.initializers["B"] = b_init if b_init is not None else \
        numpy_helper.from_array(pack_bnb(codes_nk), "B")
    m.initializers["absmax"] = absmax_init if absmax_init is not None else \
        numpy_helper.from_array(absmax, "absmax")
    a = dict(K=k, N=n, block_size=block, quant_type=qt)
    a.update(attrs or {})
    m.nodes["mm"] = Node("MatMulBnb4", "mm", a,
                         [{"name": "A", "shape": [1, k]},
                          {"name": "B", "shape": [-1]},
                          {"name": "absmax", "shape": [-1]}],
                         [{"name": "Y", "shape": [1, n]}])
    m.build_dependencies()
    return m


def quiet(fn, *args, **kw):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*args, **kw)


def fold(m):
    return quiet(PreQuantizedWeights.fold, m)


def nibbles(init):
    """UINT4 payload -> flat code array (even element in the LOW nibble)."""
    raw = numpy.frombuffer(init.raw_data, dtype=numpy.uint8)
    out = numpy.empty(raw.size * 2, numpy.uint8)
    out[0::2] = raw & 0xF
    out[1::2] = raw >> 4
    return out


def imported_weight(m, k, n, name="B"):
    codes = nibbles(m.initializers[name])[: k * n].reshape(k, n)
    scale = numpy_helper.to_array(m.initializers[name + "_scale"])
    group = k // scale.shape[0]
    return BNB_NF4[codes.astype(numpy.int64)] * numpy.repeat(scale, group, 0)


def bnb_onnx(k, n, block, codes_nk, absmax, *, a_shape=None, half=False,
             qt=1):
    tt = TensorProto.FLOAT16 if half else TensorProto.FLOAT
    node = helper.make_node("MatMulBnb4", ["A", "B", "absmax"], ["Y"],
                            domain="com.microsoft", name="mm", K=k, N=n,
                            block_size=block, quant_type=qt)
    a_shape = a_shape or [1, k]
    graph = helper.make_graph(
        [node], "bnb",
        [helper.make_tensor_value_info("A", tt, a_shape)],
        [helper.make_tensor_value_info("Y", tt, a_shape[:-1] + [n])],
        [numpy_helper.from_array(pack_bnb(codes_nk), "B"),
         numpy_helper.from_array(
             absmax.astype(numpy.float16 if half else numpy.float32),
             "absmax")])
    return helper.make_model(
        graph, ir_version=8,
        opset_imports=[helper.make_opsetid("", 17),
                       helper.make_opsetid("com.microsoft", 1)])


def ort_run(model, a):
    ort = pytest.importorskip("onnxruntime")
    sess = ort.InferenceSession(model.SerializeToString(),
                                providers=["CPUExecutionProvider"])
    return sess.run(None, {"A": a})[0]


# --------------------------------------------------------------------------
# the codebook and the contract, against onnxruntime
# --------------------------------------------------------------------------

def test_codebook_is_bitsandbytes_nf4_bit_for_bit():
    assert numpy.array_equal(Quantizer._NF4_CODEBOOK.view(numpy.uint32),
                             BNB_NF4.view(numpy.uint32))


def test_shader_codebooks_match_converter():
    """matmul.comp and quantize_linear.comp restate the table; a drift there is
    a model that decodes every nf4 weight with a different grid."""
    import re
    for rel in ("shaders/buffer/matmul.comp",
                "shaders/buffer/quantize_linear.comp"):
        src = (REPO_ROOT / rel).read_text()
        body = re.search(r"float nf4_code\(int q\)\s*\{.*?float\[16\]\((.*?)\);",
                         src, re.S).group(1)
        vals = numpy.array([float(v) for v in body.replace("\n", " ").split(",")],
                           dtype=numpy.float32)
        assert numpy.array_equal(vals.view(numpy.uint32),
                                 BNB_NF4.view(numpy.uint32)), rel


def test_ort_decodes_with_the_same_codebook_and_high_nibble_first():
    """Codes 0..15 down column 0 and 15..0 down column 1 (absmax 1 and 2): with
    A = I, onnxruntime's output is the decoded [K, N] weight."""
    k, n, block = 16, 2, 16
    codes = numpy.stack([numpy.arange(16), 15 - numpy.arange(16)])
    absmax = numpy.array([1.0, 2.0], numpy.float32)
    y = ort_run(bnb_onnx(k, n, block, codes, absmax, a_shape=[k, k]),
                numpy.eye(k, dtype=numpy.float32))
    assert numpy.array_equal(y[:, 0], BNB_NF4)
    assert numpy.array_equal(y[:, 1], BNB_NF4[::-1] * 2)


@pytest.mark.parametrize("k,n,block", [(64, 8, 16), (128, 24, 32),
                                       (256, 64, 64), (512, 16, 128),
                                       (256, 40, 256)])
def test_import_reproduces_ort_weight_exactly(k, n, block):
    codes, absmax = random_bnb(k, n, block, seed=k + n)
    y = ort_run(bnb_onnx(k, n, block, codes, absmax, a_shape=[k, k]),
                numpy.eye(k, dtype=numpy.float32))
    m = bnb_dag(k, n, block, codes, absmax)
    assert fold(m) == 1
    w = imported_weight(m, k, n)
    assert numpy.array_equal(w.astype(numpy.float32), y)


def test_ort_quantizer_output_imports_exactly():
    """Weights encoded by onnxruntime's own MatMulBnb4Quantizer (not this
    file's packer) decode to what ORT computes."""
    pytest.importorskip("onnxruntime")
    from onnxruntime.quantization.matmul_bnb4_quantizer import \
        MatMulBnb4Quantizer
    k, n, block = 256, 48, 64
    rng = numpy.random.default_rng(3)
    w = (rng.standard_normal((k, n)) * 0.05).astype(numpy.float32)
    g = helper.make_graph(
        [helper.make_node("MatMul", ["A", "W"], ["Y"], name="mm")], "g",
        [helper.make_tensor_value_info("A", TensorProto.FLOAT, [k, k])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [k, n])],
        [numpy_helper.from_array(w, "W")])
    model = helper.make_model(g, ir_version=8,
                              opset_imports=[helper.make_opsetid("", 17)])
    q = MatMulBnb4Quantizer(model, MatMulBnb4Quantizer.NF4, block)
    q.process()
    model = q.model.model
    y = ort_run(model, numpy.eye(k, dtype=numpy.float32))

    inits = {t.name: t for t in model.graph.initializer}
    node = model.graph.node[0]
    m = bnb_dag(k, n, block, b_init=inits[node.input[1]],
                absmax_init=inits[node.input[2]])
    m.initializers = {"B": inits[node.input[1]],
                      "absmax": inits[node.input[2]]}
    m.initializers["B"].name = "B"
    assert fold(m) == 1
    assert numpy.array_equal(imported_weight(m, k, n).astype(numpy.float32), y)
    # and it is the weight the quantizer was handed, to NF4 resolution
    assert numpy.abs(y - w).max() < 0.5 * numpy.diff(BNB_NF4).max() * \
        numpy.abs(w).max()


# --------------------------------------------------------------------------
# the -q nf4 twin: same bytes, same input list
# --------------------------------------------------------------------------

@pytest.mark.parametrize("k,n,block", [(64, 8, 16), (128, 16, 32),
                                       (256, 64, 64), (512, 32, 128)])
def test_bit_identical_to_q_nf4_twin(k, n, block):
    codes, absmax = random_bnb(k, n, block, seed=7)
    # bitsandbytes normalizes each block by its own absmax, so every block holds
    # a +-1 code; a decoded weight then re-quantizes onto the same codes.
    for b in range(absmax.size):
        row, g = divmod(b, k // block)
        codes[row, g * block + (b % block)] = 15 if b % 2 else 0
    m = bnb_dag(k, n, block, codes, absmax)
    fold(m)

    w_kn = (BNB_NF4[codes.astype(numpy.int64)] *
            numpy.repeat(absmax.reshape(n, k // block), block, 1)).T
    twin = DAGBasedModel()
    twin.initializers["B"] = numpy_helper.from_array(
        numpy.ascontiguousarray(w_kn, dtype=numpy.float32), "B")
    twin.nodes["mm"] = Node("MatMul", "mm", {},
                            [{"name": "A", "shape": [1, k]},
                             {"name": "B", "shape": [k, n]}],
                            [{"name": "Y", "shape": [1, n]}])
    quiet(Quantizer.quantize_to_4bit_weight_only, twin, block, "nf4")

    for name in ("B", "B_scale"):
        got, want = m.initializers[name], twin.initializers[name]
        assert got.data_type == want.data_type
        assert list(got.dims) == list(want.dims)
        assert got.raw_data == want.raw_data, name
    assert [(e.key, e.value) for e in m.initializers["B"].metadata_props] == \
        [("vkop_dtype", "nf4")]
    got_node, want_node = m.nodes["mm"], twin.nodes["mm"]
    assert got_node.op_type == "MatMul"
    assert [i["name"] for i in got_node.inputs] == \
        [i["name"] for i in want_node.inputs] == ["A", "B", "B_scale"]
    assert int(got_node.attributes.get("transB", 0)) == 0
    assert "absmax" not in m.initializers


# --------------------------------------------------------------------------
# scale widening, sharing
# --------------------------------------------------------------------------

def test_fp16_absmax_widens_exactly():
    k, n, block = 128, 16, 64
    codes, absmax = random_bnb(k, n, block, absmax_dtype=numpy.float16)
    m = bnb_dag(k, n, block, codes, absmax)
    fold(m)
    scale = numpy_helper.to_array(m.initializers["B_scale"])
    assert scale.dtype == numpy.float32
    assert numpy.array_equal(scale, absmax.astype(numpy.float32)
                             .reshape(n, k // block).T)


def test_bf16_absmax_widens_exactly():
    k, n, block = 128, 16, 64
    codes, absmax = random_bnb(k, n, block)
    bits = (absmax.view(numpy.uint32) >> 16).astype(numpy.uint16)
    init = helper.make_tensor("absmax", TensorProto.BFLOAT16, [absmax.size],
                              bits.tobytes(), raw=True)
    m = bnb_dag(k, n, block, codes, absmax, absmax_init=init)
    fold(m)
    want = (bits.astype(numpy.uint32) << 16).view(numpy.float32)
    assert numpy.array_equal(numpy_helper.to_array(m.initializers["B_scale"]),
                             want.reshape(n, k // block).T)


def test_payload_shared_by_two_nodes_is_relaid_once():
    k, n, block = 64, 8, 32
    m = bnb_dag(k, n, block)
    m.nodes["mm2"] = Node("MatMulBnb4", "mm2", dict(m.nodes["mm"].attributes),
                          [{"name": "A2", "shape": [1, k]},
                           {"name": "B", "shape": [-1]},
                           {"name": "absmax", "shape": [-1]}],
                          [{"name": "Y2", "shape": [1, n]}])
    assert fold(m) == 2
    for node in m.nodes.values():
        assert [i["name"] for i in node.inputs][1:] == ["B", "B_scale"]
    assert "absmax" not in m.initializers


# --------------------------------------------------------------------------
# gates
# --------------------------------------------------------------------------

def _raises(m, match):
    with pytest.raises(ValueError, match=match):
        fold(m)


def test_fp4_refused_and_says_why():
    _raises(bnb_dag(64, 8, 16, qt=0), r"FP4.*NOT the E2M1")


def test_unknown_quant_type_refused():
    _raises(bnb_dag(64, 8, 16, qt=2), "quant_type=2")


def test_block_spanning_two_columns_refused():
    # K=48, block 32: the second block covers k 32..47 of column 0 and 0..15 of
    # column 1, which no per-(group, column) table can express.
    codes = numpy.zeros((8, 48), numpy.uint8)
    absmax = numpy.ones(8 * 48 // 32, numpy.float32)
    _raises(bnb_dag(48, 8, 32, codes, absmax), "spans the end of one output")


def test_n_not_multiple_of_8_refused():
    _raises(bnb_dag(64, 12, 16), r"N=12 is not a multiple of 8")


def test_odd_block_refused():
    codes = numpy.zeros((8, 63), numpy.uint8)
    _raises(bnb_dag(63, 8, 21, codes, numpy.ones(24, numpy.float32)),
            "EVEN")


def test_transb_zero_refused():
    _raises(bnb_dag(64, 8, 16, attrs={"transB": 0}), "transB=0")


def test_byte_count_mismatch_refused():
    codes, absmax = random_bnb(64, 8, 16)
    b = numpy_helper.from_array(pack_bnb(codes)[:-1], "B")
    _raises(bnb_dag(64, 8, 16, codes, absmax, b_init=b), "holds 255 bytes")


def test_absmax_count_mismatch_refused():
    codes, absmax = random_bnb(64, 8, 16)
    _raises(bnb_dag(64, 8, 16, codes, absmax[:-1]), "absmax has 31 entries")


def test_non_finite_absmax_refused():
    codes, absmax = random_bnb(64, 8, 16)
    absmax[3] = numpy.inf
    _raises(bnb_dag(64, 8, 16, codes, absmax), "non-finite")


def test_runtime_payload_refused():
    m = bnb_dag(64, 8, 16)
    del m.initializers["B"]
    _raises(m, "computed at run time")


def test_payload_read_by_another_op_refused():
    m = bnb_dag(64, 8, 16)
    m.nodes["peek"] = Node("Cast", "peek", {"to": 1},
                           [{"name": "B", "shape": [-1]}],
                           [{"name": "Bf", "shape": [-1]}])
    _raises(m, r"also read by \['Cast'\]")


def test_a_inner_dim_must_be_k():
    m = bnb_dag(64, 8, 16)
    m.nodes["mm"].inputs[0]["shape"] = [1, 32]
    _raises(m, "A's last dim is 32 but K=64")


def test_plain_uint4_dq_points_at_matmulbnb4():
    payload = helper.make_tensor("W", TensorProto.UINT4, [16, 8],
                                 bytes(64), raw=True)
    m = DAGBasedModel()
    m.initializers["W"] = payload
    m.initializers["s"] = numpy_helper.from_array(
        numpy.ones((1, 8), numpy.float32), "s")
    m.nodes["dq"] = Node("DequantizeLinear", "dq", {"axis": 0,
                                                    "block_size": 16},
                         [{"name": "W"}, {"name": "s"}],
                         [{"name": "Wd"}])
    m.nodes["mm"] = Node("MatMul", "mm", {}, [{"name": "A"}, {"name": "Wd"}],
                         [{"name": "Y"}])
    _raises(m, "MatMulBnb4")


# --------------------------------------------------------------------------
# the whole converter
# --------------------------------------------------------------------------

def _convert(tmp_path, model, stem, external=False, quant=None,
             group_size=64):
    src = tmp_path / f"{stem}.onnx"
    if external:
        external_data_helper.convert_model_to_external_data(
            model, all_tensors_to_one_file=True, location=f"{stem}.bin",
            size_threshold=0)
    onnx.save_model(model, str(src))
    conv = ModelConverter()
    opts = SimpleNamespace(quant=quant, unify=False, rgba=False,
                           group_size=group_size, target_backend="buffer",
                           keep_bf16=False)
    with contextlib.redirect_stdout(io.StringIO()):
        dag = conv.parse_onnx_model(str(src), 1, quant=quant,
                                    group_size=group_size)
        conv.apply_optimizations(dag, opts)
    return src, dag


@pytest.mark.parametrize("external", [False, True])
def test_converter_end_to_end(tmp_path, external):
    k, n, block = 256, 64, 64
    codes, absmax = random_bnb(k, n, block, seed=11)
    _, dag = _convert(tmp_path, bnb_onnx(k, n, block, codes, absmax), "bnb",
                      external=external)
    (node,) = dag.nodes.values()
    assert node.op_type == "MatMul"
    b, s = node.inputs[1]["name"], node.inputs[2]["name"]
    assert dag.initializers[b].data_type == TensorProto.UINT4
    assert dag.initializers[s].data_type == TensorProto.FLOAT
    want = BNB_NF4[codes.astype(numpy.int64)] * \
        numpy.repeat(absmax.reshape(n, k // block), block, 1)
    assert numpy.array_equal(imported_weight(dag, k, n, b), want.T)


@pytest.mark.parametrize("k,n,block", [(256, 64, 64), (512, 136, 32),
                                       (1024, 64, 256)])
def test_ort_bnb_export_equals_q_nf4_of_the_float_model(tmp_path, k, n, block):
    """The whole loop on real data: onnxruntime's MatMulBnb4Quantizer applied
    to a float model, imported, lands on the SAME bytes as `-q nf4
    --group-size block` applied to that float model. Both pick the nearest of
    the same 16 codes under the same per-(block, column) absmax -- which is
    only true because the codebooks are the same table."""
    pytest.importorskip("onnxruntime")
    from onnxruntime.quantization.matmul_bnb4_quantizer import \
        MatMulBnb4Quantizer
    rng = numpy.random.default_rng(k + n)
    w = (rng.standard_normal((k, n)) * 0.05).astype(numpy.float32)
    g = helper.make_graph(
        [helper.make_node("MatMul", ["A", "W"], ["Y"], name="mm")], "g",
        [helper.make_tensor_value_info("A", TensorProto.FLOAT, [1, k])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, n])],
        [numpy_helper.from_array(w, "W")])
    float_model = helper.make_model(
        g, ir_version=8, opset_imports=[helper.make_opsetid("", 17)])
    bnb = onnx.ModelProto()
    bnb.CopyFrom(float_model)
    q = MatMulBnb4Quantizer(bnb, MatMulBnb4Quantizer.NF4, block)
    q.process()

    tables = []
    for stem, model, quant in (("f", float_model, "nf4"),
                               ("b", q.model.model, None)):
        _, dag = _convert(tmp_path, model, stem, quant=quant, group_size=block)
        (node,) = dag.nodes.values()
        tables.append([dag.initializers[i["name"]] for i in node.inputs[1:]])
    for got, want in zip(*tables):
        assert got.data_type == want.data_type
        assert list(got.dims) == list(want.dims)
        assert got.raw_data == want.raw_data


def test_converter_bf16_absmax_is_not_rounded_through_fp16(tmp_path):
    k, n, block = 128, 16, 64
    codes, absmax = random_bnb(k, n, block)
    absmax = absmax * 1e-3 + numpy.float32(3e-7)  # fp16 would lose these bits
    model = bnb_onnx(k, n, block, codes, absmax)
    bits = (absmax.view(numpy.uint32) >> 16).astype(numpy.uint16)
    for i, t in enumerate(model.graph.initializer):
        if t.name == "absmax":
            model.graph.initializer[i].CopyFrom(helper.make_tensor(
                "absmax", TensorProto.BFLOAT16, [absmax.size], bits.tobytes(),
                raw=True))
    _, dag = _convert(tmp_path, model, "bf16")
    (node,) = dag.nodes.values()
    got = numpy_helper.to_array(dag.initializers[node.inputs[2]["name"]])
    want = (bits.astype(numpy.uint32) << 16).view(numpy.float32)
    assert numpy.array_equal(got, want.reshape(n, k // block).T)


# --------------------------------------------------------------------------
# GPU end to end (opt-in): vkopbin vs onnxruntime on the original model
# --------------------------------------------------------------------------

BENCH = os.environ.get("VKOP_BENCH")


@pytest.mark.skipif(not BENCH, reason="set VKOP_BENCH=<vkop-bench or wrapper> "
                    "to run the converted model on a GPU")
@pytest.mark.parametrize("k,n,block,a_shape", [
    (1024, 256, 64, [1, 1024]),        # decode GEMV
    (1024, 256, 32, [32, 1024]),       # tile
    (768, 136, 128, [17, 768]),        # N not a multiple of 16, odd M
    (512, 128, 64, [2, 5, 512]),       # batched A
])
def test_vkop_matches_ort_on_gpu(tmp_path, k, n, block, a_shape):
    pytest.importorskip("onnxruntime")
    from onnx2vkop import cli
    master, _ = cli.resolve_master_archive()
    if master is None:
        pytest.skip("no vkop_shaders.bin to embed")
    codes, absmax = random_bnb(k, n, block, seed=k ^ n)
    model = bnb_onnx(k, n, block, codes, absmax, a_shape=a_shape)
    src = tmp_path / "bnb.onnx"
    onnx.save_model(model, str(src))
    out = tmp_path / "bnb.vkopbin"
    subprocess.run([sys.executable, "-m", "onnx2vkop.cli", "-i", str(src),
                    "-o", str(out), "-q", "fp32"],
                   check=True, capture_output=True,
                   cwd=str(Path(__file__).resolve().parents[1]))
    proc = subprocess.run(
        [sys.executable, str(REPO_ROOT / "tools" / "vkop_vs_ort.py"),
         str(out), "--onnx", str(src), "--random-fill", "--precision", "fp32",
         "--bench", BENCH, "--cos-min", "0.999999", "--rtol", "1e-5",
         "--atol", "1e-5", "--bad-frac-max", "0"],
        capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-2000:]
