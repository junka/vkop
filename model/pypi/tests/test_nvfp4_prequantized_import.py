"""Importing pre-quantized NVFP4 weights (PreQuantizedWeights, the e2m1 arm).

An opset-23 DequantizeLinear takes a float scale, so NVFP4's two scale levels
are written as a chain -- this is what NVIDIA ModelOpt's NVFP4QuantExporter
(modelopt/onnx/export/nvfp4_exporter.py, `_replace_fp4qdq_with_2dq`) emits:

    s_f32 = DequantizeLinear(s_e4m3 [N, K/16] FLOAT8E4M3FN, g fp32 [1])
    W_f   = DequantizeLinear(W [N, K] FLOAT4E2M1, s_f32, axis=-1, block_size=16)
    MatMul(Cast(X, fp16), Cast(Transpose(W_f), fp16))      (or W_f -> Cast -> MatMul)

The import has to land on EXACTLY what `-q nvfp4` emits -- [A, W e2m1 [K, N],
block scale e4m3 [K/16, N], global fp32 [1]], transB 0 -- with the source's own
bits, only permuted when the source is [N, K]. The cases below pin that against
the -q twin byte for byte, check the chain's meaning against ONNX's reference
evaluator through an independent decoder, recover the bytes from a graph whose
first level was already evaluated (only when the split is exact), refuse every
shape the kernel cannot read, and -- when a vkop-bench is available -- run the
converted model against ONNX Runtime.
"""
import contextlib
import io
import os
import subprocess
import sys
import json

import numpy
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

from onnx2vkop.dag import DAGBasedModel, Node, _init_bytes, _init_dtype_name
from onnx2vkop.optimizer import PreQuantizedWeights, Quantizer

PYPI = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.dirname(os.path.dirname(PYPI))
BLOCK = 16

# Independent decoders (ONNX's FLOAT4E2M1 / FLOAT8E4M3FN definitions), written
# out from the bit layout rather than shared with the converter.
E2M1 = numpy.array([0, 0.5, 1, 1.5, 2, 3, 4, 6,
                    -0.0, -0.5, -1, -1.5, -2, -3, -4, -6], dtype=numpy.float32)


def e4m3(byte):
    s = -1.0 if byte & 0x80 else 1.0
    e, m = (byte >> 3) & 0xF, byte & 7
    if e == 15 and m == 7:
        return numpy.nan
    return s * (m / 8 * 2.0 ** -6 if e == 0 else (1 + m / 8) * 2.0 ** (e - 7))


E4M3 = numpy.array([e4m3(b) for b in range(256)], dtype=numpy.float32)


# --------------------------------------------------------------------------
# builders
# --------------------------------------------------------------------------

def tref(name):
    return {"name": name, "shape": [-1]}


def run(fn, *args):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*args)


def pack(codes):
    flat = numpy.asarray(codes, dtype=numpy.uint8).reshape(-1) & 0x0F
    return (flat[0::2] | (flat[1::2] << 4)).astype(numpy.uint8).tobytes()


def unpack(raw, shape):
    b = numpy.frombuffer(raw, dtype=numpy.uint8)
    out = numpy.empty(b.size * 2, dtype=numpy.uint8)
    out[0::2], out[1::2] = b & 0x0F, b >> 4
    return out.reshape(shape)


def fp4(name, codes):
    return helper.make_tensor(name, TensorProto.FLOAT4E2M1, list(codes.shape),
                              pack(codes), raw=True)


def fp8(name, codes, dtype=TensorProto.FLOAT8E4M3FN):
    codes = numpy.asarray(codes, dtype=numpy.uint8)
    return helper.make_tensor(name, dtype, list(codes.shape), codes.tobytes(),
                              raw=True)


def randw(k, n, seed):
    rng = numpy.random.RandomState(seed)
    amps = numpy.abs(rng.randn(k).astype(numpy.float32)) + 0.2
    return rng.randn(k, n).astype(numpy.float32) * amps[:, None]


def q_twin(k=64, n=16, seed=0):
    """What `-q nvfp4` emits for a float [K, N] weight: the reference bytes."""
    m = DAGBasedModel()
    m.initializers["W"] = numpy_helper.from_array(randw(k, n, seed), "W")
    m.nodes["mm"] = Node("MatMul", "mm", {"transB": 0},
                         [tref("x"), tref("W")], [tref("y")])
    run(Quantizer.quantize_to_nvfp4_weight_only, m)
    return m


def twin_codes(twin, k, n):
    """(nibbles [K, N], e4m3 bytes [K/16, N], global) of a -q twin."""
    w = twin.initializers["W"]
    s = twin.initializers["W_scale"]
    g = numpy_helper.to_array(twin.initializers["W_scale_global"])
    return (unpack(w.raw_data, (k, n)),
            numpy.frombuffer(s.raw_data, dtype=numpy.uint8).reshape(k // BLOCK, n),
            numpy.float32(g.reshape(-1)[0]))


def chain_graph(codes, scale_bytes, g, *, axis, views=(), transb=0,
                block=BLOCK, global_dims=(1,), extra_nodes=(), dq2_attrs=None,
                dq1_attrs=None, n_matmuls=1):
    """The ModelOpt-style chain as a DAG: DQ1(e4m3, g) -> DQ2(e2m1, .) ->
    views -> MatMul. `views` is a sequence of ("Transpose"|"Cast", attrs)."""
    m = DAGBasedModel()
    m.initializers["W"] = fp4("W", codes)
    m.initializers["W_f8_scale"] = fp8("W_f8_scale", scale_bytes)
    m.initializers["W_f8_scale_f32_scale"] = numpy_helper.from_array(
        numpy.full(global_dims, g, dtype=numpy.float32), "W_f8_scale_f32_scale")
    m.nodes["dq1"] = Node("DequantizeLinear", "dq1", dict(dq1_attrs or {}),
                          [tref("W_f8_scale"), tref("W_f8_scale_f32_scale")],
                          [tref("W_f32_scale")])
    attrs = {"axis": axis, "block_size": block}
    attrs.update(dq2_attrs or {})
    m.nodes["dq2"] = Node("DequantizeLinear", "dq2", attrs,
                          [tref("W"), tref("W_f32_scale")], [tref("W_deq")])
    b = "W_deq"
    for i, (op, a) in enumerate(views):
        out = f"v{i}"
        m.nodes[out] = Node(op, out, dict(a), [tref(b)], [tref(out)])
        b = out
    for j in range(n_matmuls):
        name = "mm" if j == 0 else f"mm{j}"
        m.nodes[name] = Node("MatMul", name, {"transB": transb},
                             [tref("x"), tref(b)], [tref(name + "_y")])
    for node in extra_nodes:
        m.nodes[node.name] = node
    return m


def fold(m):
    assert run(PreQuantizedWeights.fold, m) >= 1
    return m


def assert_matches_twin(m, twin, mm="mm"):
    node = m.nodes[mm]
    assert [i["name"] for i in node.inputs] == [
        "x", "W", "W_scale", "W_scale_global"]
    assert int(node.attributes.get("transB", 0) or 0) == 0
    for name in ("W", "W_scale", "W_scale_global"):
        got, want = m.initializers[name], twin.initializers[name]
        assert got.data_type == want.data_type, name
        assert list(got.dims) == list(want.dims), name
        assert _init_bytes(got) == _init_bytes(want), name
    assert _init_dtype_name(m.initializers["W"]) == "float4e2m1fn"
    assert not any(n.op_type in ("DequantizeLinear", "Transpose", "Cast")
                   for n in m.nodes.values())
    # The source tables were copied into the MatMul's own inputs.
    assert "W_f8_scale" not in m.initializers
    assert "W_f8_scale_f32_scale" not in m.initializers


# --------------------------------------------------------------------------
# positive: the -q nvfp4 twin, byte for byte
# --------------------------------------------------------------------------

K, N = 64, 16


def test_kn_chain_reproduces_the_q_nvfp4_twin():
    twin = q_twin(K, N, 1)
    codes, sb, g = twin_codes(twin, K, N)
    m = fold(chain_graph(codes, sb, g, axis=0))
    assert_matches_twin(m, twin)


# Each way the [N, K] ModelOpt layout reaches its MatMul. parity: the MatMul's
# effective B is the transpose of the stored payload.
NK_ROUTES = [
    pytest.param((), 1, id="folded-transpose-transB"),
    pytest.param((("Transpose", {"perm": [1, 0]}),), 0, id="transpose"),
    pytest.param((("Transpose", {"perm": [1, 0]}),
                  ("Cast", {"to": TensorProto.FLOAT16})), 0,
                 id="modelopt-transpose-then-cast"),
    pytest.param((("Cast", {"to": TensorProto.FLOAT16}),
                  ("Transpose", {})), 0, id="cast-then-default-perm"),
]


@pytest.mark.parametrize("views,transb", NK_ROUTES)
def test_nk_chain_is_permuted_onto_the_twin(views, transb):
    """ModelOpt stores the Linear weight [N, K] with axis=-1 blocks. The
    import transposes the nibble grid and the block-scale table -- a pure
    permutation -- and must land on the [K, N] twin's exact bytes."""
    twin = q_twin(K, N, 2)
    codes, sb, g = twin_codes(twin, K, N)
    m = fold(chain_graph(numpy.ascontiguousarray(codes.T),
                         numpy.ascontiguousarray(sb.T), g, axis=-1,
                         views=views, transb=transb))
    assert_matches_twin(m, twin)
    assert m.nodes["mm"].inputs[1]["shape"] == [K, N]


def test_kn_payload_behind_transpose_and_transB_cancels():
    """Two transposes are none: a [K, N] payload, Transpose, transB=1."""
    twin = q_twin(K, N, 3)
    codes, sb, g = twin_codes(twin, K, N)
    m = fold(chain_graph(codes, sb, g, axis=0,
                         views=(("Transpose", {"perm": [1, 0]}),), transb=1))
    assert_matches_twin(m, twin)


def test_scalar_global_and_zero_zero_points_are_accepted():
    twin = q_twin(K, N, 4)
    codes, sb, g = twin_codes(twin, K, N)
    m = chain_graph(codes, sb, g, axis=0, global_dims=())
    m.initializers["zp4"] = fp4("zp4", numpy.zeros((K, N), numpy.uint8))
    m.nodes["dq2"].inputs.append(tref("zp4"))
    m.initializers["zp8"] = fp8("zp8", numpy.zeros((K // BLOCK, N)))
    m.nodes["dq1"].inputs.append(tref("zp8"))
    assert_matches_twin(fold(m), twin)
    assert "zp4" not in m.initializers and "zp8" not in m.initializers


def test_one_dq_output_feeding_two_matmuls():
    twin = q_twin(K, N, 5)
    codes, sb, g = twin_codes(twin, K, N)
    m = fold(chain_graph(codes, sb, g, axis=0, n_matmuls=2))
    assert_matches_twin(m, twin, "mm")
    assert_matches_twin(m, twin, "mm1")


def test_a_shared_first_level_serves_two_weights():
    """One e4m3 table + global feeding two e2m1 payloads: each weight gets its
    own copy and the shared first DQ goes once both are folded."""
    twin = q_twin(K, N, 6)
    codes, sb, g = twin_codes(twin, K, N)
    m = chain_graph(codes, sb, g, axis=0)
    m.initializers["V"] = fp4("V", codes ^ 0x1)
    m.nodes["dqv"] = Node("DequantizeLinear", "dqv",
                          {"axis": 0, "block_size": BLOCK},
                          [tref("V"), tref("W_f32_scale")], [tref("V_deq")])
    m.nodes["mmv"] = Node("MatMul", "mmv", {"transB": 0},
                          [tref("x"), tref("V_deq")], [tref("yv")])
    assert run(PreQuantizedWeights.fold, m) == 2
    assert_matches_twin(m, twin)
    assert [i["name"] for i in m.nodes["mmv"].inputs] == [
        "x", "V", "V_scale", "V_scale_global"]
    assert _init_bytes(m.initializers["V_scale"]) == sb.tobytes()


def test_the_chain_means_what_the_twin_decodes_to():
    """The graph semantics this import claims, checked on ONNX's own reference
    evaluator: DQ2(DQ1(...)) of the ModelOpt [N, K] chain equals the -q twin's
    weight decoded by an independent e2m1/e4m3 table."""
    from onnx.reference import ReferenceEvaluator
    twin = q_twin(K, N, 7)
    codes, sb, g = twin_codes(twin, K, N)
    nk = numpy.ascontiguousarray(codes.T)
    nodes = [
        helper.make_node("DequantizeLinear", ["S8", "G"], ["S"]),
        helper.make_node("DequantizeLinear", ["W", "S"], ["Wf"], axis=-1,
                         block_size=BLOCK),
    ]
    graph = helper.make_graph(
        nodes, "g", [],
        [helper.make_tensor_value_info("Wf", TensorProto.FLOAT, [N, K])],
        [fp4("W", nk), fp8("S8", numpy.ascontiguousarray(sb.T)),
         numpy_helper.from_array(numpy.array([g], numpy.float32), "G")])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 23)])
    onnx.checker.check_model(model, full_check=True)
    got = ReferenceEvaluator(model).run(None, {})[0]
    want = E2M1[codes] * numpy.repeat(E4M3[sb] * g, BLOCK, axis=0)
    numpy.testing.assert_array_equal(got.T, want.astype(numpy.float32))


# --------------------------------------------------------------------------
# a first level that was already evaluated
# --------------------------------------------------------------------------

def folded_scale_graph(table, codes, axis=0):
    m = DAGBasedModel()
    m.initializers["W"] = fp4("W", codes)
    m.initializers["S"] = numpy_helper.from_array(table, "S")
    m.nodes["dq2"] = Node("DequantizeLinear", "dq2",
                          {"axis": axis, "block_size": BLOCK},
                          [tref("W"), tref("S")], [tref("W_deq")])
    m.nodes["mm"] = Node("MatMul", "mm", {"transB": 0},
                         [tref("x"), tref("W_deq")], [tref("y")])
    return m


def test_an_evaluated_first_level_splits_back_exactly():
    """onnxsim-style: DQ1 already evaluated to fp32. The e4m3 bytes come back
    because every entry is exactly e4m3 * g for one g; the kernel's product is
    then the graph's product, bit for bit."""
    twin = q_twin(K, N, 8)
    codes, sb, g = twin_codes(twin, K, N)
    table = (E4M3[sb] * g).astype(numpy.float32)
    m = fold(folded_scale_graph(table, codes))
    node = m.nodes["mm"]
    assert [i["name"] for i in node.inputs] == [
        "x", "W", "W_scale", "W_scale_global"]
    got_sb = numpy.frombuffer(m.initializers["W_scale"].raw_data, numpy.uint8)
    got_g = numpy_helper.to_array(m.initializers["W_scale_global"])
    assert got_g.dtype == numpy.float32 and got_g.shape == (1,)
    numpy.testing.assert_array_equal(
        (E4M3[got_sb] * got_g[0]).astype(numpy.float32), table.reshape(-1))
    assert m.initializers["W_scale"].data_type == TensorProto.FLOAT8E4M3FN
    assert _init_bytes(m.initializers["W"]) == _init_bytes(twin.initializers["W"])
    # Here the search lands on the twin's own split.
    assert got_sb.tobytes() == sb.tobytes() and got_g[0] == g


# --------------------------------------------------------------------------
# gates: everything the kernel cannot read raises
# --------------------------------------------------------------------------

def good():
    twin = q_twin(K, N, 9)
    codes, sb, g = twin_codes(twin, K, N)
    return codes, sb, g


def expect_error(m, needle):
    with pytest.raises(ValueError) as exc:
        run(PreQuantizedWeights.fold, m)
    assert needle in str(exc.value), str(exc.value)


def g_block32():
    c, s, g = good()
    return chain_graph(c, s[: K // 32], g, axis=0, block=32)


def g_wrong_axis():
    c, s, g = good()
    return chain_graph(c, s, g, axis=1)


def g_vector_global():
    c, s, g = good()
    return chain_graph(c, s, g, axis=0, global_dims=(2,))


def g_nonzero_zp():
    c, s, g = good()
    m = chain_graph(c, s, g, axis=0)
    zp = numpy.zeros((K, N), numpy.uint8)
    zp[0, 0] = 2
    m.initializers["zp4"] = fp4("zp4", zp)
    m.nodes["dq2"].inputs.append(tref("zp4"))
    return m


def g_scale_read_elsewhere():
    c, s, g = good()
    return chain_graph(c, s, g, axis=0, extra_nodes=[
        Node("Add", "spy", {}, [tref("W_f32_scale"), tref("x")], [tref("z")])])


def g_gemm():
    c, s, g = good()
    m = chain_graph(c, s, g, axis=0)
    m.nodes["mm"].op_type = "Gemm"
    return m


def g_bf16_cast():
    c, s, g = good()
    return chain_graph(c, s, g, axis=0,
                       views=(("Cast", {"to": TensorProto.BFLOAT16}),))


def g_n12():
    rng = numpy.random.RandomState(0)
    return chain_graph(rng.randint(0, 16, (K, 12)).astype(numpy.uint8),
                       numpy.full((K // BLOCK, 12), 0x38, numpy.uint8),
                       numpy.float32(0.01), axis=0)


def g_unsplittable_single_dq():
    c, _, _ = good()
    table = numpy.random.RandomState(1).rand(K // BLOCK, N).astype(
        numpy.float32) + 0.01
    return folded_scale_graph(table, c)


def g_per_tensor_single_dq():
    c, _, _ = good()
    return folded_scale_graph(numpy.array(0.5, numpy.float32), c)


def g_nan_scale_byte():
    c, s, g = good()
    s = s.copy()
    s[0, 0] = 0x7F
    return chain_graph(c, s, g, axis=0)


def g_fp16_block_scale():
    c, s, g = good()
    m = chain_graph(c, s, g, axis=0)
    m.initializers["W_f8_scale"] = numpy_helper.from_array(
        numpy.ones((K // BLOCK, N), numpy.float16), "W_f8_scale")
    return m


def g_blocked_first_level():
    c, s, g = good()
    return chain_graph(c, s, g, axis=0, dq1_attrs={"block_size": 4})


def g_payload_shared():
    c, s, g = good()
    return chain_graph(c, s, g, axis=0, extra_nodes=[
        Node("Identity", "peek", {}, [tref("W")], [tref("w2")])])


def g_graph_output():
    c, s, g = good()
    m = chain_graph(c, s, g, axis=0)
    m.outputs.append({"name": "W_deq"})
    return m


def g_a_operand():
    c, s, g = good()
    m = chain_graph(c, s, g, axis=0)
    m.nodes["mm"].inputs = [tref("W_deq"), tref("x")]
    return m


def g_layouts_disagree():
    c, s, g = good()
    m = chain_graph(c, s, g, axis=0, n_matmuls=2)
    m.nodes["mm1"].attributes["transB"] = 1
    return m


def g_runtime_scale():
    c, s, g = good()
    m = chain_graph(c, s, g, axis=0)
    m.nodes["dq2"].inputs[1] = tref("x")
    del m.nodes["dq1"]
    return m


def g_bf16_output():
    c, s, g = good()
    return chain_graph(c, s, g, axis=0,
                       dq2_attrs={"output_dtype": TensorProto.BFLOAT16})


GATES = [
    pytest.param(g_block32, "block_size=32", id="block-not-16"),
    pytest.param(g_wrong_axis, "blocks cut dim 1", id="blocks-cut-n"),
    pytest.param(g_vector_global, "2 entries", id="global-not-one-value"),
    pytest.param(g_nonzero_zp, "non-zero zero_point", id="nonzero-zero-point"),
    pytest.param(g_scale_read_elsewhere, "also reads", id="block-scale-shared"),
    pytest.param(g_gemm, "MatMul only", id="gemm-consumer"),
    pytest.param(g_bf16_cast, "cannot reproduce", id="bf16-cast-view"),
    pytest.param(g_n12, "multiple of 8", id="n-not-multiple-of-8"),
    pytest.param(g_unsplittable_single_dq, "two-level form",
                 id="single-dq-table-not-e4m3-times-g"),
    pytest.param(g_per_tensor_single_dq, "per-tensor", id="single-dq-per-tensor"),
    pytest.param(g_nan_scale_byte, "NaN", id="nan-block-scale"),
    pytest.param(g_fp16_block_scale, "FLOAT8E4M3FN", id="block-scale-not-e4m3"),
    pytest.param(g_blocked_first_level, "ONE fp32", id="blocked-first-level"),
    pytest.param(g_payload_shared, "also read by", id="payload-shared"),
    pytest.param(g_graph_output, "graph output", id="dequantized-is-output"),
    pytest.param(g_a_operand, "single B operand", id="a-operand"),
    pytest.param(g_layouts_disagree, "both layouts", id="layouts-disagree"),
    pytest.param(g_runtime_scale, "computed at run time", id="runtime-scale"),
    pytest.param(g_bf16_output, "output_dtype", id="bf16-output-dtype"),
]


@pytest.mark.parametrize("build,needle", GATES)
def test_nvfp4_gate_raises(build, needle):
    expect_error(build(), needle)


# --------------------------------------------------------------------------
# through the CLI: an external-data ModelOpt-style file
# --------------------------------------------------------------------------

def modelopt_onnx(path, k, n, m_rows, seed, *, external=False, orient="nk",
                  cast=True):
    """An opset-23 file in the shape ModelOpt's exporter writes, with random
    nibbles and block bytes (no float weight ever existed)."""
    rng = numpy.random.RandomState(seed)
    shape = (n, k) if orient == "nk" else (k, n)
    sshape = (n, k // BLOCK) if orient == "nk" else (k // BLOCK, n)
    codes = rng.randint(0, 16, size=shape).astype(numpy.uint8)
    sb = rng.randint(0x20, 0x60, size=sshape).astype(numpy.uint8)
    g = numpy.float32(0.0123)
    inits = [fp4("W_f4", codes), fp8("W_f8_scale", sb),
             numpy_helper.from_array(numpy.array([g], numpy.float32),
                                     "W_f8_scale_f32_scale")]
    nodes = [
        helper.make_node("DequantizeLinear", ["W_f8_scale", "W_f8_scale_f32_scale"],
                         ["W_f32_scale"], name="W_DequantizeLinear"),
        helper.make_node("DequantizeLinear", ["W_f4", "W_f32_scale"], ["W_deq"],
                         name="W_DequantizeLinear_1",
                         axis=-1 if orient == "nk" else 0, block_size=BLOCK),
    ]
    b = "W_deq"
    if orient == "nk":
        nodes.append(helper.make_node("Transpose", [b], ["W_t"], perm=[1, 0]))
        b = "W_t"
    x = "X"
    y = "Y"
    if cast:
        nodes.append(helper.make_node("Cast", [b], ["W_f16"],
                                      to=TensorProto.FLOAT16))
        nodes.append(helper.make_node("Cast", ["X"], ["X_f16"],
                                      to=TensorProto.FLOAT16))
        b, x, y = "W_f16", "X_f16", "Y16"
    nodes.append(helper.make_node("MatMul", [x, b], [y]))
    if cast:
        nodes.append(helper.make_node("Cast", ["Y16"], ["Y"], to=TensorProto.FLOAT))
    graph = helper.make_graph(
        nodes, "nvfp4",
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, [m_rows, k])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [m_rows, n])],
        inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 23)])
    model.ir_version = 10
    onnx.checker.check_model(model, full_check=True)
    if external:
        onnx.save_model(model, path, save_as_external_data=True,
                        all_tensors_to_one_file=True, location="w.bin",
                        size_threshold=0)
    else:
        onnx.save(model, path)
    kn_codes = codes.T if orient == "nk" else codes
    kn_sb = sb.T if orient == "nk" else sb
    weight = E2M1[kn_codes] * numpy.repeat(E4M3[kn_sb] * g, BLOCK, axis=0)
    return numpy.ascontiguousarray(kn_codes), numpy.ascontiguousarray(kn_sb), g, \
        weight.astype(numpy.float32)


def convert(src, out, *extra):
    p = subprocess.run(
        [sys.executable, "-m", "onnx2vkop.cli", "-i", src, "-o", out,
         "--target-backend", "buffer", "--external-blob-mb", "0", *extra],
        capture_output=True, text=True, cwd=PYPI)
    assert p.returncode == 0, p.stdout[-1500:] + p.stderr[-1500:]
    # The e2m1/e4m3 initializers are types onnxoptimizer cannot hold; they
    # used to make it skip every structural pass on an imported checkpoint.
    assert "onnxoptimizer skipped" not in p.stdout + p.stderr, (
        "onnxoptimizer did not run on the imported NVFP4 model")
    return out


def section(path, sec):
    p = subprocess.run(
        [sys.executable, "-m", "onnx2vkop.inspect", path, "--section", sec,
         "--json"], capture_output=True, text=True,
        env={**os.environ, "PYTHONPATH": PYPI})
    assert p.returncode == 0, p.stderr[-400:]
    return json.loads(p.stdout)[sec]


@pytest.mark.parametrize("stream", ["on", "off"])
def test_external_data_modelopt_file_converts_to_the_exact_bits(tmp_path, stream):
    """The payload of a ModelOpt file lives in external data. weight_stream
    hydrates it (an e2m1/e4m3 payload read by DequantizeLinear is not a
    streamable pass-through), so the fold sees the bytes either way and the
    trailer must hold exactly the permuted source bits."""
    k, n = 128, 64
    src = str(tmp_path / "m.onnx")
    codes, sb, g, _ = modelopt_onnx(src, k, n, 1, 11, external=True)
    out = convert(src, str(tmp_path / "m.vkopbin"), "--stream-weights", stream)

    nodes = [nd for nd in section(out, "nodes") if nd["op_type"] == "MatMul"]
    assert len(nodes) == 1
    ins = [i if isinstance(i, str) else i.get("name") for i in nodes[0]["inputs"]]
    assert ins[1:] == ["W_f4", "W_f4_scale", "W_f4_scale_global"], ins
    summ = section(out, "summary")
    trailer = summ["external_blob_trailer_offset"]
    blob = open(out, "rb").read()
    got = {}
    for e in section(out, "initializers"):
        got[e["name"]] = (e["dtype"], e["dims"],
                          blob[trailer + e["offset"]:trailer + e["offset"] + e["size"]])
    assert got["W_f4"][:2] == ("float4e2m1fn", [k, n])
    assert got["W_f4"][2] == pack(codes)
    assert got["W_f4_scale"][:2] == ("float8e4m3fn", [k // BLOCK, n])
    assert got["W_f4_scale"][2] == sb.tobytes()
    assert got["W_f4_scale_global"][:2] == ("float32", [1])
    assert numpy.frombuffer(got["W_f4_scale_global"][2], "<f4")[0] == g


# --------------------------------------------------------------------------
# end to end on the GPU, against ONNX Runtime (opt-in: needs vkop-bench)
# --------------------------------------------------------------------------

def fp32_reference(src, dst, weight_kn):
    """ONNX Runtime has no FLOAT4E2M1 DequantizeLinear kernel (1.19 stops at
    opset 21; 1.32's CPU EP has no DQ(23) for e2m1), so its reference runs the
    same graph with the chain replaced by the dequantized weight -- decoded by
    this file's independent tables, and asserted equal to ONNX's reference
    evaluator on the original file first."""
    from onnx.reference import ReferenceEvaluator
    model = onnx.load(src)
    keep = [nd for nd in model.graph.node if nd.op_type != "DequantizeLinear"]
    w_deq = weight_kn.T  # the chain's output is [N, K]; Transpose follows
    ref_x = numpy.random.RandomState(0).randn(
        *[d.dim_value for d in model.graph.input[0].type.tensor_type.shape.dim]
    ).astype(numpy.float32)
    want = ReferenceEvaluator(model).run(None, {"X": ref_x})[0]
    graph = helper.make_graph(
        keep, "ref", list(model.graph.input), list(model.graph.output),
        [numpy_helper.from_array(numpy.ascontiguousarray(w_deq), "W_deq")])
    ref = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 21)])
    ref.ir_version = 10
    onnx.save(ref, dst)
    got = ReferenceEvaluator(ref).run(None, {"X": ref_x})[0]
    numpy.testing.assert_array_equal(got, want)


BENCH = os.environ.get("VKOP_BENCH")
ARCHIVE = os.environ.get("VKOP_MASTER_ARCHIVE",
                         os.path.join(REPO, "build", "vkop_shaders.bin"))


@pytest.mark.skipif(not BENCH or not os.path.exists(ARCHIVE),
                    reason="set VKOP_BENCH (a vkop-bench, or a wrapper that "
                           "exports the Vulkan loader env) and a shader archive")
@pytest.mark.parametrize("m_rows", [1, 32], ids=["decode-gemv", "tile"])
@pytest.mark.parametrize("cast", [True, False], ids=["cast", "nocast"])
def test_vkop_matches_onnxruntime(tmp_path, m_rows, cast):
    pytest.importorskip("onnxruntime")
    k, n = 512, 256
    src = str(tmp_path / "q.onnx")
    _, _, _, weight = modelopt_onnx(src, k, n, m_rows, 21, cast=cast)
    ref = str(tmp_path / "ref.onnx")
    fp32_reference(src, ref, weight)
    out = convert(src, str(tmp_path / "q.vkopbin"), "--master-archive", ARCHIVE)
    p = subprocess.run(
        [sys.executable, os.path.join(REPO, "tools", "vkop_vs_ort.py"), out,
         "--onnx", ref, "--bench", BENCH, "--random-fill"],
        capture_output=True, text=True)
    report = p.stdout + p.stderr
    print("\n".join(line for line in report.splitlines()
                    if line.startswith(("Y ", "output", "gates", "RESULT"))))
    assert p.returncode == 0, report[-2000:]
    assert "RESULT: 1/1" in report, report[-2000:]
