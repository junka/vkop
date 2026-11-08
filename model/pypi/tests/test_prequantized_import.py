"""Importing weights that arrive already quantized (PreQuantizedWeights.fold).

A third-party quantized ONNX graph spells a Linear weight as
`MatMul(X, DequantizeLinear(W_q, s[, zp]))`. vkop's buffer MatMul/Gemm kernels
are the same dequantizer, fed from the payload initializer plus a scale appended
as the last input — so the fold has to delete the node and re-point the consumer
into EXACTLY the input list the -q quantizers emit, or the shader binds the wrong
buffers. These cases pin that against the -q twin (byte-for-byte), the layout
normalization a [N, K] export needs, and every shape the spec does not resolve
onto a kernel, which has to raise instead of shipping a graph whose only possible
answer is a wrong one.

The fold has two other jobs a QDQ graph hands it, and they are the opposite of
importing. A DequantizeLinear whose payload a QuantizeLinear in the same graph
produced is ACTIVATION quantization: both halves are runtime ops
(ops/QuantizeLinear.hpp), so the correct action is to leave the pair alone and
count no weight -- and to still fold the real weights sitting next to it. A
DequantizeLinear over an integer constant no kernel decodes -- ORT's QDQ exporter
writes the Conv bias as QInt32 with a per-channel scale, in a slot that takes a
float -- has no storage to bind, so the fold evaluates the node's own formula on
the host. A 4-D int8 Conv weight is the fourth contract: the kernel indexes the
table by output channel, not by output column.

The gate cases are parametrized rather than run in sequence on purpose: the fold
mutates the graph it is given, so each case builds its own, and one gate firing
must not hide the rest. That is how the four dead gates in this file got dead --
a first-failure-aborts runner stopped the suite at an impossible assertion, and
nothing behind it had ever executed.
"""
import contextlib
import io
from types import SimpleNamespace

import numpy
import pytest
from onnx import TensorProto, helper, numpy_helper

from onnx2vkop.dag import (
    DAGBasedModel,
    Node,
    _init_byte_len,
    _init_bytes,
    _init_dtype_name,
)
from onnx2vkop.optimizer import PreQuantizedWeights, Quantizer

DQ_OUT = "W_deq"

NF4_CODEBOOK = numpy.array(
    [
        -1.0, -0.6961928009986877, -0.5250730514526367, -0.3949174189567566,
        -0.2844413814544678, -0.1847814998626709, -0.0910967006323811, 0.0,
        0.0795802986717224, 0.1601973110246658, 0.2447470557689667,
        0.3361703515052795, 0.4407098295211792, 0.5626170039176941,
        0.7229568369388580, 1.0,
    ],
    dtype=numpy.float32,
)


# --------------------------------------------------------------------------
# graph builders -- one per case, because the fold rewrites what it is handed
# --------------------------------------------------------------------------

def tref(name):
    return {"name": name, "shape": [-1]}


def add_consumers(m, consumers):
    for op_type, name, attrs, ins in consumers:
        m.nodes[name] = Node(op_type, name, dict(attrs), [tref(i) for i in ins],
                             [tref(name + "_out")])
    return m


def float_weight_graph(weight, consumers, extras=None, name="W"):
    """The un-quantized graph the -q quantizers are given."""
    m = DAGBasedModel()
    m.initializers[name] = numpy_helper.from_array(weight, name)
    for ename, arr in (extras or {}).items():
        m.initializers[ename] = numpy_helper.from_array(arr, ename)
    return add_consumers(m, consumers)


def dq_graph(payload, scale, consumers, attrs=None, zero_point=None, extras=None):
    """The graph a third-party exporter writes: initializers plus DQ plus reader."""
    m = DAGBasedModel()
    m.initializers[payload.name] = payload
    m.initializers[scale.name] = scale
    if zero_point is not None:
        m.initializers[zero_point.name] = zero_point
    for ename, arr in (extras or {}).items():
        m.initializers[ename] = numpy_helper.from_array(arr, ename)
    ins = [payload.name, scale.name]
    if zero_point is not None:
        ins.append(zero_point.name)
    m.nodes["dq"] = Node("DequantizeLinear", "dq", dict(attrs or {}),
                         [tref(i) for i in ins], [tref(DQ_OUT)])
    return add_consumers(m, consumers)


def run(fn, *args):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*args)


def fold(m):
    assert run(PreQuantizedWeights.fold, m) == 1
    return m


def quantize_int8(m):
    run(Quantizer.quantize_to_int8_weight_only, m)
    return m


def quantize_fp8(m, fmt="fp8e4m3"):
    run(Quantizer.quantize_to_fp8_weight_only, m, fmt)
    return m


def quantize_4bit(m, fmt="int4", group=4):
    run(Quantizer.quantize_to_4bit_weight_only, m, group, fmt)
    return m


def matmul_consumer(attrs=None, name="mm", ins=None):
    return [("MatMul", name, {"transB": 0} if attrs is None else attrs,
             ["x", DQ_OUT] if ins is None else ins)]


def packed(codes, dims, dtype, name="W", nf4=False):
    """A nibble grid as the file stores it: element i in byte i/2, even element
    in the LOW nibble, row-major over dims."""
    flat = numpy.asarray(codes).reshape(-1).astype(numpy.uint8) & 0x0F
    assert flat.size % 2 == 0
    raw = (flat[0::2] | (flat[1::2] << 4)).astype(numpy.uint8).tobytes()
    init = helper.make_tensor(name, dtype, dims, raw, raw=True)
    if nf4:
        init.metadata_props.add(key="vkop_dtype", value="nf4")
    return init


def unpack(raw, count):
    byte_arr = numpy.frombuffer(raw, dtype=numpy.uint8)
    out = numpy.empty(byte_arr.size * 2, dtype=numpy.uint8)
    out[0::2] = byte_arr & 0x0F
    out[1::2] = byte_arr >> 4
    return out[:count]


def transpose_packed(raw, dims):
    """The exporter's [N, K] view of the same weights, restated independently of
    the importer so a byte-identity test proves both agree on what a transpose
    of a nibble grid is."""
    n0, n1 = dims
    grid = unpack(raw, n0 * n1).reshape(dims)
    return numpy.ascontiguousarray(grid.T)


def arr(init):
    return numpy_helper.to_array(init)


def randn(k, n, seed):
    numpy.random.seed(seed)
    amps = numpy.abs(numpy.random.randn(k).astype(numpy.float32)) + 0.2
    return (numpy.random.randn(k, n).astype(numpy.float32) * amps[:, None])


def conv_weights(oc, ic, kh, kw, seed):
    """A 4-D kernel with a different magnitude per OUTPUT CHANNEL, so a scale
    table that indexes the wrong axis is a different vector, not a permutation
    of the same numbers."""
    flat = randn(oc, ic * kh * kw, seed)
    return flat.reshape(oc, ic, kh, kw).astype(numpy.float32)


def add_activation_pair(m, scale_name, zp_name=None, out=DQ_OUT, q="aq", dq="adq",
                        producer="a_in"):
    """`producer -> QuantizeLinear -> DequantizeLinear -> consumers`: what an
    exporter writes for an ACTIVATION. The payload is a graph value, so there are
    no quantized bytes for the fold to bind, and both halves run on the GPU."""
    q_ins = [producer, scale_name] + ([zp_name] if zp_name else [])
    m.nodes[q] = Node("QuantizeLinear", q, {}, [tref(i) for i in q_ins],
                      [tref(q + "_val")])
    dq_ins = [q + "_val", scale_name] + ([zp_name] if zp_name else [])
    m.nodes[dq] = Node("DequantizeLinear", dq, {}, [tref(i) for i in dq_ins],
                       [tref(out)])
    return m


def activation_pair(scale, zp=None, consumers=None):
    m = DAGBasedModel()
    m.initializers["a_scale"] = scale
    if zp is not None:
        m.initializers["a_zp"] = zp
    add_activation_pair(m, "a_scale", "a_zp" if zp is not None else None)
    return add_consumers(m, consumers if consumers is not None
                         else matmul_consumer())


# --------------------------------------------------------------------------
# positive: the fold must land on the -q path's bytes
# --------------------------------------------------------------------------

def test_int8_matmul_reproduces_the_q_int8_twin():
    """Per-axis INT8 -> [A, W_int8, scale], byte-identical to what -q int8 emits
    for the same weight, and with the DQ node gone."""
    w = randn(8, 16, 0)
    twin = quantize_int8(float_weight_graph(
        w, [("MatMul", "mm", {"transB": 0}, ["x", "W"])]))
    payload, scale = twin.initializers["W"], twin.initializers["W_scale"]
    assert payload.data_type == TensorProto.INT8

    m = fold(dq_graph(payload, scale, matmul_consumer(), attrs={"axis": 1}))

    assert "dq" not in m.nodes, "the DequantizeLinear node survived"
    assert [i["name"] for i in m.nodes["mm"].inputs] == ["x", "W", "W_scale"]
    assert m.nodes["mm"].inputs[1]["shape"] == [8, 16]
    assert _init_bytes(m.initializers["W"]) == _init_bytes(payload)
    assert _init_dtype_name(m.initializers["W"]) == "int8"
    numpy.testing.assert_array_equal(arr(m.initializers["W_scale"]), arr(scale))


def test_int8_gemm_with_bias_keeps_the_scale_last():
    """Gemm's weight-only slots are [A, B_int8, C_bias, scale] — keying the bias
    off the input COUNT would read the scale as a bias, so the fold has to
    append rather than insert, on a [N, K] weight with transB=1."""
    n, k = 16, 8
    w = randn(n, k, 1)
    bias = numpy.random.randn(n).astype(numpy.float32)
    twin = quantize_int8(float_weight_graph(
        w, [("Gemm", "g", {"transB": 1}, ["x", "W", "b"])], extras={"b": bias}))
    payload, scale = twin.initializers["W"], twin.initializers["W_scale"]
    assert arr(scale).shape == (n,), "the int8 twin indexes output columns"

    m = fold(dq_graph(payload, scale,
                      [("Gemm", "g", {"transB": 1}, ["x", DQ_OUT, "b"])],
                      attrs={"axis": 0}, extras={"b": bias}))

    assert [i["name"] for i in m.nodes["g"].inputs] == ["x", "W", "b", "W_scale"]
    assert m.nodes["g"].attributes["transB"] == 1, "Gemm keeps its [N, K] layout"
    assert _init_bytes(m.initializers["W"]) == _init_bytes(payload)
    numpy.testing.assert_array_equal(arr(m.initializers["W_scale"]), arr(scale))


def test_shared_weight_folds_for_every_consumer():
    """One DQ read by two MatMuls: both consumers get the payload and the scale,
    exactly as the quantizer handles a weight with several readers."""
    w = randn(8, 16, 2)
    twin = quantize_int8(float_weight_graph(
        w, [("MatMul", "mm", {"transB": 0}, ["x", "W"]),
            ("MatMul", "mm2", {"transB": 0}, ["y", "W"])]))
    payload, scale = twin.initializers["W"], twin.initializers["W_scale"]

    m = fold(dq_graph(payload, scale,
                      [("MatMul", "mm", {"transB": 0}, ["x", DQ_OUT]),
                       ("MatMul", "mm2", {"transB": 0}, ["y", DQ_OUT])],
                      attrs={"axis": 1}))

    for name in ("mm", "mm2"):
        assert "dq" not in m.nodes
    assert [i["name"] for i in m.nodes["mm"].inputs] == ["x", "W", "W_scale"]
    assert [i["name"] for i in m.nodes["mm2"].inputs] == ["y", "W", "W_scale"]


def test_fp8_matmul_reproduces_the_q_fp8_twin():
    w = randn(8, 16, 3)
    twin = quantize_fp8(float_weight_graph(
        w, [("MatMul", "mm", {"transB": 0}, ["x", "W"])]), "fp8e4m3")
    payload, scale = twin.initializers["W"], twin.initializers["W_scale"]
    assert payload.data_type == TensorProto.FLOAT8E4M3FN

    m = fold(dq_graph(payload, scale, matmul_consumer(), attrs={"axis": 1}))

    assert _init_bytes(m.initializers["W"]) == _init_bytes(payload)
    assert _init_dtype_name(m.initializers["W"]) == "float8e4m3fn"
    assert _init_byte_len(m.initializers["W"]) == 8 * 16, "one byte per value"
    numpy.testing.assert_array_equal(arr(m.initializers["W_scale"]), arr(scale))


def test_fp16_scale_is_widened_exactly():
    """The kernels bind a float32 table, and fp16 -> fp32 is exact, so an fp16
    scale imports as the same numbers in the dtype the shader reads."""
    w = randn(8, 16, 4)
    twin = quantize_int8(float_weight_graph(
        w, [("MatMul", "mm", {"transB": 0}, ["x", "W"])]))
    scale16 = numpy_helper.from_array(
        arr(twin.initializers["W_scale"]).astype(numpy.float16), "W_scale")

    m = fold(dq_graph(twin.initializers["W"], scale16, matmul_consumer(),
                      attrs={"axis": 1}))

    imported = m.initializers["W_scale"]
    assert imported.data_type == TensorProto.FLOAT
    assert arr(imported).dtype == numpy.float32
    # Widening must not move a value, and the values it must not move are the
    # ones that came IN -- `scale16`, already rounded to the 11-bit grid. The
    # twin's pre-rounding fp32 scale is NOT reachable from an fp16 input, so
    # comparing against it would make this case unsatisfiable by construction.
    numpy.testing.assert_array_equal(arr(imported),
                                     arr(scale16).astype(numpy.float32))


def test_zero_zero_point_is_dropped():
    """A zero point of all-zeros is the identity the formula claims, so the fold
    takes it and the payload's own offset stays 0 — and the now-unread
    initializer must not ride along in the blob."""
    w = randn(8, 16, 5)
    twin = quantize_int8(float_weight_graph(
        w, [("MatMul", "mm", {"transB": 0}, ["x", "W"])]))
    zp = numpy_helper.from_array(numpy.zeros((16,), dtype=numpy.int8), "W_zp")

    m = fold(dq_graph(twin.initializers["W"], twin.initializers["W_scale"],
                      matmul_consumer(), attrs={"axis": 1}, zero_point=zp))

    assert "W_zp" not in m.initializers, "a dead zero point was still written"
    assert [i["name"] for i in m.nodes["mm"].inputs] == ["x", "W", "W_scale"]


def test_int4_blocked_reproduces_the_q_int4_twin():
    """Blocked INT4 with axis=0 on a [K, N] weight is already the kernel's
    convention: [K/g, N] fp32 rows, one scale per (group, column)."""
    w = randn(8, 16, 6)
    group = 4
    twin = quantize_4bit(float_weight_graph(
        w, [("MatMul", "mm", {"transB": 0}, ["x", "W"])]), "int4", group)
    payload, scale = twin.initializers["W"], twin.initializers["W_scale"]
    assert arr(scale).shape == (8 // group, 16)

    m = fold(dq_graph(payload, scale, matmul_consumer(),
                      attrs={"axis": 0, "block_size": [group]}))

    assert [i["name"] for i in m.nodes["mm"].inputs] == ["x", "W", "W_scale"]
    assert _init_bytes(m.initializers["W"]) == _init_bytes(payload)
    assert _init_dtype_name(m.initializers["W"]) == "int4"
    numpy.testing.assert_array_equal(arr(m.initializers["W_scale"]), arr(scale))


def test_int4_transposed_export_is_normalized_to_K_N():
    """A third-party Linear arrives [N, K] and the fusion pass has already folded
    its Transpose into the MatMul as transB=1 — a layout the nibble kernel
    refuses. The importer transposes the packed grid and the table and flips
    transB, and must land on the same bytes as the [K, N] twin."""
    k, n, group = 8, 16, 4
    w = randn(k, n, 7)
    twin = quantize_4bit(float_weight_graph(
        w, [("MatMul", "mm", {"transB": 0}, ["x", "W"])]), "int4", group)
    twin_payload = twin.initializers["W"]

    # The same weights as an exporter that stores [N, K]: nibble grid transposed,
    # scale table transposed, blocked along dim 1.
    nk_grid = transpose_packed(twin_payload.raw_data, [k, n])
    payload = packed(nk_grid, [n, k], TensorProto.INT4)
    scale = numpy_helper.from_array(
        numpy.ascontiguousarray(arr(twin.initializers["W_scale"]).T), "W_scale")
    assert scale.dims == [n, k // group]

    m = fold(dq_graph(payload, scale,
                      matmul_consumer(attrs={"transB": 1}),
                      attrs={"axis": 1, "block_size": [group]}))

    assert m.nodes["mm"].attributes["transB"] == 0, "the layout flip is the fold"
    assert m.nodes["mm"].inputs[1]["shape"] == [k, n]
    assert list(m.initializers["W"].dims) == [k, n]
    assert m.initializers["W"].raw_data == twin_payload.raw_data, (
        "transposing the nibble grid twice must return the twin's bytes")
    numpy.testing.assert_array_equal(
        arr(m.initializers["W_scale"]), arr(twin.initializers["W_scale"]))


def test_nf4_rides_on_the_dtype_metadata_through_the_transpose():
    """UINT4 is storage; the nf4 name is what makes the nibble a codebook index,
    so a payload rewrite must carry the metadata entry along with the bytes."""
    k, n, group = 16, 8, 8
    w = randn(k, n, 8)
    twin = quantize_4bit(float_weight_graph(
        w, [("MatMul", "mm", {"transB": 0}, ["x", "W"])]), "nf4", group)
    assert _init_dtype_name(twin.initializers["W"]) == "nf4"

    nk_grid = transpose_packed(twin.initializers["W"].raw_data, [k, n])
    payload = packed(nk_grid, [n, k], TensorProto.UINT4, nf4=True)
    scale = numpy_helper.from_array(
        numpy.ascontiguousarray(arr(twin.initializers["W_scale"]).T), "W_scale")

    m = fold(dq_graph(payload, scale, matmul_consumer(attrs={"transB": 1}),
                      attrs={"axis": 1, "block_size": [group]}))

    assert _init_dtype_name(m.initializers["W"]) == "nf4"
    assert m.initializers["W"].data_type == TensorProto.UINT4
    assert m.initializers["W"].raw_data == twin.initializers["W"].raw_data
    # And the values still decode through the codebook the shader holds.
    codes = unpack(m.initializers["W"].raw_data, k * n).reshape(k, n)
    table = arr(m.initializers["W_scale"])
    deq = NF4_CODEBOOK[codes.astype(numpy.int64)] * table.repeat(group, axis=0)
    assert numpy.abs(deq - w).max() <= 0.5 * numpy.diff(NF4_CODEBOOK).max() * table.max()


def test_int8_matmul_accepts_the_transposed_layout_the_kernel_reads():
    """A folded-Transpose MatMul (transB=1) is the [N, K] layout int8/fp8 DOes
    support, so the import keeps the layout and only checks the scale's axis."""
    n, k = 16, 8
    w = randn(n, k, 13)
    twin = quantize_int8(float_weight_graph(
        w, [("MatMul", "mm", {"transB": 1}, ["x", "W"])]))
    payload, scale = twin.initializers["W"], twin.initializers["W_scale"]

    m = fold(dq_graph(payload, scale, matmul_consumer(attrs={"transB": 1}),
                      attrs={"axis": 0}))

    assert m.nodes["mm"].attributes["transB"] == 1
    assert m.nodes["mm"].inputs[1]["shape"] == [n, k]
    assert _init_bytes(m.initializers["W"]) == _init_bytes(payload)


def test_graph_without_dequantize_is_untouched():
    w = randn(8, 16, 9)
    m = float_weight_graph(w, [("MatMul", "mm", {"transB": 0}, ["x", "W"])])
    before_nodes = dict((n, [i["name"] for i in m.nodes[n].inputs]) for n in m.nodes)
    assert run(PreQuantizedWeights.fold, m) == 0
    assert {n: [i["name"] for i in nn.inputs] for n, nn in m.nodes.items()} == before_nodes
    assert m.initializers["W"].data_type == TensorProto.FLOAT


def test_dependencies_are_rebuilt_after_the_nodes_go():
    """Deleting a node without rebuilding the dependency sets leaves the removed
    DQ named in every dependents set, which the level scheduler then dispatches
    against a node that no longer exists."""
    payload, scale = good_int8()
    m = fold(dq_graph(payload, scale, matmul_consumer(), attrs={"axis": 1}))
    names = set(m.nodes)
    for node in m.nodes.values():
        assert node.dependencies <= names and node.dependents <= names


def test_fp8_on_a_matmul_is_the_pairing_that_exists():
    """The kernel table gives fp8 a MatMul and nothing else, so the legal
    pairing has to fold. Asserting the success side is what keeps a narrowed
    table from passing the gate suite by rejecting everything."""
    payload, scale = good_int8()
    fp8 = helper.make_tensor("W", TensorProto.FLOAT8E4M3FN, [8, 16],
                             bytes(8 * 16), raw=True)

    m = fold(dq_graph(fp8, scale, matmul_consumer(), attrs={"axis": 1}))

    assert "dq" not in m.nodes
    assert [i["name"] for i in m.nodes["mm"].inputs] == ["x", "W", "W_scale"]


def test_q_int4_itself_refuses_a_non_multiple_of_8_output():
    """The import's N % 8 gate mirrors the quantizer's: if -q int4 would not
    produce this shape, no kernel reads it either."""
    bad_n = quantize_4bit(float_weight_graph(
        randn(8, 12, 12), [("MatMul", "mm", {"transB": 0}, ["x", "W"])]), "int4", 4)
    assert bad_n.initializers["W"].data_type == TensorProto.FLOAT, (
        "-q int4 itself refuses N % 8 != 0")


# --------------------------------------------------------------------------
# positive: an activation QDQ pair is not a weight, it is two runtime ops
# --------------------------------------------------------------------------

def test_activation_qdq_pair_survives_the_fold():
    """`value -> QuantizeLinear -> DequantizeLinear -> MatMul` is what an
    exporter writes to quantize an ACTIVATION. ops/QuantizeLinear.hpp runs both
    halves on the buffer backend, so the fold's only correct action is to do
    nothing -- and to count no weight, because a graph that is all activation
    pairs must not report an import it did not perform."""
    m = activation_pair(numpy_helper.from_array(numpy.float32(0.02), "a_scale"),
                        zp=numpy_helper.from_array(numpy.int8(0), "a_zp"))
    before = {n: [i["name"] for i in nn.inputs] for n, nn in m.nodes.items()}

    assert run(PreQuantizedWeights.fold, m) == 0
    assert {n: [i["name"] for i in nn.inputs] for n, nn in m.nodes.items()} == before
    assert "a_scale" in m.initializers and "a_zp" in m.initializers, (
        "the fold popped the tables of a pair it never touched")


def test_activation_pair_is_kept_while_weights_fold_around_it():
    """The classification is per node, not per graph: a QDQ pair sitting next to
    a quantized weight must not stop the weight from reaching its kernel, and
    the weight's fold must not take the pair with it."""
    w = randn(8, 16, 23)
    twin = quantize_int8(float_weight_graph(
        w, [("MatMul", "mm", {"transB": 0}, ["x", "W"])]))
    m = dq_graph(twin.initializers["W"], twin.initializers["W_scale"],
                 matmul_consumer(), attrs={"axis": 1})
    m.initializers["a_scale"] = numpy_helper.from_array(numpy.float32(0.05),
                                                        "a_scale")
    add_activation_pair(m, "a_scale", out="act_out", producer="act_in")
    m.nodes["mm2"] = Node("MatMul", "mm2", {"transB": 0},
                          [tref("act_out"), tref("h")], [tref("mm2_out")])
    pair_before = {n: [i["name"] for i in nn.inputs] for n, nn in m.nodes.items()
                   if n in ("aq", "adq")}

    assert run(PreQuantizedWeights.fold, m) == 1, "only the weight is an import"
    assert "dq" not in m.nodes
    assert [i["name"] for i in m.nodes["mm"].inputs] == ["x", "W", "W_scale"]
    assert {n: [i["name"] for i in nn.inputs] for n, nn in m.nodes.items()
            if n in ("aq", "adq")} == pair_before
    assert "a_scale" in m.initializers


# --------------------------------------------------------------------------
# positive: an int8 Conv kernel folds onto the [X, W, (b), scale] -q int8 emits
# --------------------------------------------------------------------------

def test_int8_conv_weight_reproduces_the_q_int8_twin():
    """conv2d.comp is a per-output-channel dequantizer, so a QDQ Conv weight
    lands on the same input list and the same bytes as -q int8 emits for it:
    [X, W, scale], table last."""
    oc, ic, kh, kw = 16, 8, 3, 3
    twin = quantize_int8(float_weight_graph(
        conv_weights(oc, ic, kh, kw, 24), [("Conv", "c", {}, ["x", "W"])]))
    payload, scale = twin.initializers["W"], twin.initializers["W_scale"]
    assert arr(scale).shape == (oc,), "the -q Conv twin indexes output channels"

    m = fold(dq_graph(payload, scale, [("Conv", "c", {}, ["x", DQ_OUT])],
                      attrs={"axis": 0}))

    assert "dq" not in m.nodes
    assert [i["name"] for i in m.nodes["c"].inputs] == ["x", "W", "W_scale"]
    assert m.nodes["c"].inputs[1]["shape"] == [oc, ic, kh, kw]
    assert _init_bytes(m.initializers["W"]) == _init_bytes(payload)
    numpy.testing.assert_array_equal(arr(m.initializers["W_scale"]), arr(scale))


def test_int8_conv_with_bias_keeps_the_scale_last():
    """The runtime finds the Conv scale by counting past the optional bias
    (ops/Conv2d.hpp has_bias ? 3 : 2), so an imported graph has to put it in the
    slot the -q pass puts it in -- after the bias, not before it."""
    oc, ic, kh, kw = 16, 8, 3, 3
    bias = numpy.random.randn(oc).astype(numpy.float32)
    twin = quantize_int8(float_weight_graph(
        conv_weights(oc, ic, kh, kw, 25), [("Conv", "c", {}, ["x", "W", "b"])],
        extras={"b": bias}))
    payload, scale = twin.initializers["W"], twin.initializers["W_scale"]
    assert [i["name"] for i in twin.nodes["c"].inputs] == ["x", "W", "b", "W_scale"]

    m = fold(dq_graph(payload, scale, [("Conv", "c", {}, ["x", DQ_OUT, "b"])],
                      attrs={"axis": 0}, extras={"b": bias}))

    assert [i["name"] for i in m.nodes["c"].inputs] == ["x", "W", "b", "W_scale"]
    assert _init_bytes(m.initializers["W"]) == _init_bytes(payload)
    numpy.testing.assert_array_equal(arr(m.initializers["W_scale"]), arr(scale))


def test_int8_conv_scale_arriving_as_fp16_is_widened():
    """An exporter that quantized an fp16 model writes fp16 per-channel scales;
    the Conv kernel binds fp32 and half -> float is exact, so the imported table
    is the same numbers in the dtype the shader reads."""
    oc, ic, kh, kw = 8, 4, 3, 3
    twin = quantize_int8(float_weight_graph(
        conv_weights(oc, ic, kh, kw, 26), [("Conv", "c", {}, ["x", "W"])]))
    scale16 = numpy_helper.from_array(
        arr(twin.initializers["W_scale"]).astype(numpy.float16), "W_scale")

    m = fold(dq_graph(twin.initializers["W"], scale16,
                      [("Conv", "c", {}, ["x", DQ_OUT])], attrs={"axis": 0}))

    imported = m.initializers["W_scale"]
    assert imported.data_type == TensorProto.FLOAT
    numpy.testing.assert_array_equal(arr(imported),
                                     arr(scale16).astype(numpy.float32))


# --------------------------------------------------------------------------
# positive: an integer constant no kernel decodes is evaluated, not bound
# --------------------------------------------------------------------------

def test_int32_conv_bias_dequantizes_on_the_host():
    """ORT's QDQ exporter writes the Conv bias as QInt32 with a per-channel
    scale. No vkop kernel decodes int32 and the slot takes a float anyway, so the
    honest import is the node's own definition, evaluated here -- and the tables
    the deleted node was reading must go, or they ride along in the blob."""
    q = numpy.array([100, -200, 50, -25, 0, 127, -128, 4], dtype=numpy.int32)
    s = numpy.array([0.01, 0.02, 0.005, 0.1, 1.0, 0.03, 0.02, 0.001],
                    dtype=numpy.float32)
    m = fold(dq_graph(numpy_helper.from_array(q, "b_q"),
                      numpy_helper.from_array(s, "b_scale"),
                      [("Conv", "c", {}, ["x", "W", DQ_OUT])], attrs={"axis": 0}))

    assert "dq" not in m.nodes
    assert [i["name"] for i in m.nodes["c"].inputs] == ["x", "W", "b_q_dequant"]
    out = m.initializers["b_q_dequant"]
    assert out.data_type == TensorProto.FLOAT
    assert list(out.dims) == [8]
    numpy.testing.assert_allclose(arr(out), q.astype(numpy.float32) * s, rtol=1e-6)
    for dead in ("b_q", "b_scale"):
        assert dead not in m.initializers, f"{dead} only ever fed this node"


def test_int32_constant_subtracts_its_zero_point_before_scaling():
    """(q - zero_point) * scale, each table broadcast along the `axis` attribute
    -- which ONNX defaults to 1 and is not given here. Scaling first would put
    the offset in the wrong units, and one scalar offset for a per-channel table
    would apply one channel's zero point to every channel."""
    q = numpy.arange(6, dtype=numpy.int32).reshape(2, 3)
    zp = numpy.array([1, -2, 0], dtype=numpy.int32)
    s = numpy.array([0.5, 0.25, 2.0], dtype=numpy.float32)
    m = fold(dq_graph(numpy_helper.from_array(q, "c_q"),
                      numpy_helper.from_array(s, "c_scale"),
                      [("Add", "a", {}, ["x", DQ_OUT])],
                      zero_point=numpy_helper.from_array(zp, "c_zp")))

    got = arr(m.initializers["c_q_dequant"])
    numpy.testing.assert_allclose(got, (q - zp[None, :]).astype(numpy.float32)
                                  * s[None, :], rtol=1e-6)
    assert m.nodes["a"].inputs[1] == {"name": "c_q_dequant", "shape": [2, 3],
                                      "dtype": TensorProto.FLOAT,
                                      "value_dynamic": False}
    assert "c_zp" not in m.initializers


def test_a_full_qdq_graph_folds_every_weight_and_keeps_every_pair():
    """An ORT-style block in one graph: quantized activations around a Conv with
    an int8 kernel and a QInt32 bias, and a Linear with an int8 weight. Each
    DequantizeLinear is classified on its own evidence, so the pass has to get
    all five answers right at once -- three imports, two pairs left in place, and
    dependency sets that name only nodes that still exist."""
    oc, ic, kh, kw = 16, 8, 3, 3
    conv_twin = quantize_int8(float_weight_graph(
        conv_weights(oc, ic, kh, kw, 27), [("Conv", "c", {}, ["x", "W"])]))
    lin_twin = quantize_int8(float_weight_graph(
        randn(oc, 32, 28), [("MatMul", "mm", {"transB": 0}, ["x", "H"])],
        name="H"))
    b_q = numpy_helper.from_array(numpy.arange(oc, dtype=numpy.int32) - 8, "b_q")
    b_s = numpy_helper.from_array(
        (numpy.abs(numpy.random.randn(oc).astype(numpy.float32)) + 0.1) / 127.0,
        "b_scale")

    m = DAGBasedModel()
    for init in (conv_twin.initializers["W"], conv_twin.initializers["W_scale"],
                 lin_twin.initializers["H"], lin_twin.initializers["H_scale"],
                 b_q, b_s,
                 numpy_helper.from_array(numpy.float32(0.02), "a_scale")):
        m.initializers[init.name] = init
    m.nodes["wq"] = Node("DequantizeLinear", "wq", {"axis": 0},
                         [tref("W"), tref("W_scale")], [tref("W_deq")])
    m.nodes["bq"] = Node("DequantizeLinear", "bq", {"axis": 0},
                         [tref("b_q"), tref("b_scale")], [tref("b_deq")])
    add_activation_pair(m, "a_scale", out="x_qdq", q="aq1", dq="adq1",
                        producer="x")
    m.nodes["c"] = Node("Conv", "c", {}, [tref("x_qdq"), tref("W_deq"),
                                          tref("b_deq")], [tref("c_out")])
    m.nodes["flat"] = Node("Reshape", "flat", {}, [tref("c_out"), tref("shape")],
                           [tref("flat_out")])
    add_activation_pair(m, "a_scale", out="flat_qdq", q="aq2", dq="adq2",
                        producer="flat_out")
    m.nodes["hq"] = Node("DequantizeLinear", "hq", {"axis": 1},
                         [tref("H"), tref("H_scale")], [tref("H_deq")])
    m.nodes["mm"] = Node("MatMul", "mm", {"transB": 0},
                         [tref("flat_qdq"), tref("H_deq")], [tref("mm_out")])

    assert run(PreQuantizedWeights.fold, m) == 3
    assert [i["name"] for i in m.nodes["c"].inputs] == \
        ["x_qdq", "W", "b_q_dequant", "W_scale"]
    assert [i["name"] for i in m.nodes["mm"].inputs] == \
        ["flat_qdq", "H", "H_scale"]
    for pair in ("aq1", "adq1", "aq2", "adq2"):
        assert pair in m.nodes, "an activation pair was folded as a weight"
    for gone in ("wq", "bq", "hq"):
        assert gone not in m.nodes
    assert "b_q" not in m.initializers and "b_scale" not in m.initializers
    assert m.initializers["a_scale"].data_type == TensorProto.FLOAT, (
        "a pair the runtime still has to execute lost its table")
    names = set(m.nodes)
    for node in m.nodes.values():
        assert node.dependencies <= names and node.dependents <= names


# --------------------------------------------------------------------------
# gates: everything the spec does not resolve onto a kernel has to raise
# --------------------------------------------------------------------------

def good_int8():
    """A valid per-axis INT8 payload + scale, for the gates to break one by one."""
    w = randn(8, 16, 10)
    twin = quantize_int8(float_weight_graph(
        w, [("MatMul", "mm", {"transB": 0}, ["x", "W"])]))
    return twin.initializers["W"], twin.initializers["W_scale"]


def expect_error(m, needle):
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            PreQuantizedWeights.fold(m)
    except ValueError as e:
        assert needle in str(e), f"{needle!r} missing from {str(e)!r}"
        return str(e)
    raise AssertionError(f"no error raised, expected one containing {needle!r}")


@pytest.fixture(scope="module")
def byte_pieces():
    """One valid INT8 weight plus a broken variant of every input the fold
    reads. The graphs reference these initializers by name; the fold rewrites
    the DAG, never these."""
    payload, scale = good_int8()
    return SimpleNamespace(
        payload=payload,
        scale=scale,
        scalar=numpy_helper.from_array(numpy.float32(0.01), "W_scale"),
        blocked=numpy_helper.from_array(
            numpy.full((2, 16), 0.01, dtype=numpy.float32), "W_scale"),
        short=numpy_helper.from_array(numpy.full((8,), 0.01, dtype=numpy.float32),
                                      "W_scale"),
        zp=numpy_helper.from_array(numpy.full((16,), 3, dtype=numpy.int8), "W_zp"),
        zp_u8=numpy_helper.from_array(numpy.zeros((16,), dtype=numpy.uint8), "W_zp"),
        nan=numpy_helper.from_array(
            numpy.full((16,), numpy.nan, dtype=numpy.float32), "W_scale"),
        int_scale=numpy_helper.from_array(numpy.arange(16, dtype=numpy.int64),
                                          "W_scale"),
        fp8=helper.make_tensor("W", TensorProto.FLOAT8E4M3FN, [8, 16],
                               bytes(8 * 16), raw=True),
        u8=helper.make_tensor("W", TensorProto.UINT8, [8, 16], bytes(8 * 16),
                              raw=True),
        batched=helper.make_tensor("W", TensorProto.INT8, [2, 8, 16],
                                   bytes(2 * 8 * 16), raw=True),
    )


def byte_graph(pieces, consumers=None, attrs=None, zp_init=None, p=None, s=None):
    m = dq_graph(p if p is not None else pieces.payload,
                 s if s is not None else pieces.scale,
                 consumers if consumers is not None else matmul_consumer(),
                 attrs if attrs is not None else {"axis": 1},
                 zero_point=zp_init)
    return m


def byte_graph_with_dead_input(pieces, index, name):
    """A DQ whose input slot no longer names an initializer: the exporter wrote
    an activation-side QDQ, or computes the scale at run time."""
    m = byte_graph(pieces)
    m.nodes["dq"].inputs[index]["name"] = name
    return m


BYTE_GATES = [
    pytest.param(lambda px: byte_graph_with_dead_input(px, 0, "activation"),
                 "graph value", id="payload-is-a-graph-value"),
    pytest.param(lambda px: byte_graph_with_dead_input(px, 1, "computed_scale"),
                 "computed at run time", id="scale-computed-at-runtime"),
    pytest.param(lambda px: byte_graph(px, zp_init=px.zp),
                 "non-zero zero_point", id="non-zero-zero-point"),
    pytest.param(lambda px: byte_graph(px, zp_init=px.zp_u8),
                 "requires them to match", id="zero-point-dtype-disagrees"),
    pytest.param(lambda px: byte_graph(px, s=px.scalar),
                 "rank 0", id="per-tensor-scale"),
    pytest.param(lambda px: byte_graph(px, s=px.blocked),
                 "rank 2", id="blocked-scale-on-a-byte-weight"),
    pytest.param(lambda px: byte_graph(px, attrs={"axis": 0}),
                 "output columns", id="scale-indexes-the-wrong-axis"),
    pytest.param(lambda px: byte_graph(px, s=px.short),
                 "dim 1 (length 8)", id="scale-length-is-not-n"),
    pytest.param(lambda px: byte_graph(px, s=px.nan),
                 "non-finite", id="non-finite-scale"),
    pytest.param(lambda px: byte_graph(px, s=px.int_scale),
                 "float32 table", id="integer-scale"),
    pytest.param(lambda px: byte_graph(px, p=px.fp8,
                                       consumers=[("Gemm", "g", {"transB": 0},
                                                  ["x", DQ_OUT])]),
                 "MatMul only", id="fp8-on-a-gemm"),
    pytest.param(lambda px: byte_graph(px, consumers=[("Add", "a", {},
                                                       ["x", DQ_OUT])]),
                 "only readable by MatMul", id="consumer-is-not-a-gemm-or-matmul"),
    pytest.param(lambda px: byte_graph(
        px, consumers=[("MatMul", "mm", {"transB": 0}, [DQ_OUT, "x"])]),
        "single B operand", id="weight-on-the-a-side"),
    pytest.param(lambda px: byte_graph(px, p=px.batched),
                 "one scale table per batch", id="rank-3-batched-weight"),
    pytest.param(lambda px: byte_graph(
        px, consumers=[("MatMul", "mm", {"transB": 0}, ["x", DQ_OUT]),
                       ("MatMul", "mm2", {"transB": 1}, ["y", DQ_OUT])]),
        "disagree on transB", id="consumers-disagree-on-transb"),
    pytest.param(lambda px: byte_graph(px, consumers=[]),
                 "nothing reads", id="no-consumers"),
    pytest.param(lambda px: byte_graph(
        px, attrs={"axis": 1, "output_dtype": TensorProto.BFLOAT16}),
        "output_dtype", id="output-dtype-the-kernels-cannot-reproduce"),
    pytest.param(lambda px: byte_graph(px, p=px.u8),
                 "no vkop dequantization kernel", id="uint8-payload-is-asymmetric"),
]


@pytest.mark.parametrize("build,needle", BYTE_GATES)
def test_byte_weight_gate_raises(byte_pieces, build, needle):
    """Each case breaks exactly one input of an otherwise valid INT8 import."""
    expect_error(build(byte_pieces), needle)


@pytest.fixture(scope="module")
def nibble_pieces():
    group = 4
    twin = quantize_4bit(float_weight_graph(
        randn(8, 16, 11), [("MatMul", "mm", {"transB": 0}, ["x", "W"])]),
        "int4", group)
    payload, scale = twin.initializers["W"], twin.initializers["W_scale"]
    return SimpleNamespace(
        group=group,
        payload=payload,
        scale=scale,
        plain_u4=packed(numpy.zeros((8, 16), dtype=numpy.uint8), [8, 16],
                        TensorProto.UINT4),
        zero_int4=packed(numpy.zeros((8, 16), dtype=numpy.int8), [8, 16],
                         TensorProto.INT4),
        int_scale=numpy_helper.from_array(numpy.zeros((1,), numpy.int32),
                                          "W_scale"),
        # NVFP4's storage type. ONNX validates raw_data against dims in
        # make_tensor, so 8x16 nibbles have to arrive as 64 packed bytes.
        e2m1=helper.make_tensor("W", TensorProto.FLOAT4E2M1, [8, 16], bytes(64),
                                raw=True),
        percol=numpy_helper.from_array(numpy.full((16,), 0.01, dtype=numpy.float32),
                                       "W_scale"),
        n12_payload=packed(numpy.zeros((8, 12), dtype=numpy.int8), [8, 12],
                           TensorProto.INT4),
        n12_scale=numpy_helper.from_array(numpy.zeros((2, 12), dtype=numpy.float32),
                                          "W_scale"),
    )


def nibble_graph(pieces, p=None, s=None, attrs=None, consumers=None):
    return dq_graph(p if p is not None else pieces.payload,
                    s if s is not None else pieces.scale,
                    consumers if consumers is not None else matmul_consumer(),
                    {"axis": 0, "block_size": [pieces.group]}
                    if attrs is None else attrs)


def truncated_payload(pieces):
    """A packed grid one byte short of the value count. Built by editing the
    protobuf field after make_tensor, because onnx itself refuses a length that
    disagrees with dims -- the gate under test is the fold's, not the
    constructor's."""
    t = helper.make_tensor("W", TensorProto.INT4, [8, 16], pieces.payload.raw_data,
                           raw=True)
    t.raw_data = t.raw_data[:8]
    return t


NIBBLE_GATES = [
    pytest.param(lambda px: nibble_graph(px, p=px.e2m1),
                 "NVFP4", id="e2m1-has-no-room-for-the-global-scale"),
    pytest.param(lambda px: nibble_graph(px, p=px.plain_u4),
                 "names itself nf4", id="uint4-without-the-nf4-metadata"),
    pytest.param(lambda px: nibble_graph(px, p=px.zero_int4, s=px.int_scale),
                 "float32 table", id="integer-scale"),
    pytest.param(lambda px: nibble_graph(px, s=px.percol),
                 "blocked", id="per-column-scale-on-a-nibble"),
    pytest.param(lambda px: nibble_graph(
        px, attrs={"axis": 1, "block_size": [px.group]}),
        "blocked along dim 1", id="groups-must-cut-k"),
    # 6 is EVEN, so it clears the parity gate (covered by the -4 and 0 cases)
    # and still fails K % group: 8 values cannot split into whole 6-value blocks.
    pytest.param(lambda px: nibble_graph(px, attrs={"axis": 0, "block_size": [6]}),
                 "multiple of block_size", id="k-not-a-multiple-of-the-group"),
    pytest.param(lambda px: nibble_graph(px, attrs={"axis": 0, "block_size": [3, 3]}),
                 "exactly one block_size", id="more-than-one-blocked-axis"),
    pytest.param(lambda px: nibble_graph(px, attrs={"axis": 0}),
                 "exactly one block_size", id="blocked-nibble-without-a-size"),
    pytest.param(lambda px: nibble_graph(px, attrs={"axis": 0, "block_size": [-4]}),
                 "positive EVEN", id="negative-group-size"),
    pytest.param(lambda px: nibble_graph(px, attrs={"axis": 0, "block_size": [0]}),
                 "positive EVEN", id="zero-group-size"),
    pytest.param(lambda px: nibble_graph(px, p=px.n12_payload, s=px.n12_scale),
                 "multiple of 8", id="n-not-a-multiple-of-8"),
    pytest.param(lambda px: nibble_graph(px, p=truncated_payload(px)),
                 "packed payload carries", id="short-packed-payload"),
    pytest.param(lambda px: nibble_graph(
        px, consumers=[("Gemm", "g", {"transB": 0}, ["x", DQ_OUT])]),
        "MatMul only", id="no-4-bit-gemm-kernel"),
]


@pytest.mark.parametrize("build,needle", NIBBLE_GATES)
def test_4bit_gate_raises(nibble_pieces, build, needle):
    """Each case breaks exactly one input of an otherwise valid INT4 import."""
    expect_error(build(nibble_pieces), needle)


# --------------------------------------------------------------------------
# gates: an activation pair the runtime op would throw on, at conversion
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def act_pieces():
    """A valid per-tensor activation pair plus the tables that break it."""
    return SimpleNamespace(
        scalar=numpy_helper.from_array(numpy.float32(0.02), "a_scale"),
        per_axis=numpy_helper.from_array(
            numpy.full((8,), 0.02, dtype=numpy.float32), "a_scale"),
        zp_scalar=numpy_helper.from_array(numpy.int8(0), "a_zp"),
        zp_axis=numpy_helper.from_array(numpy.zeros((8,), dtype=numpy.int8),
                                        "a_zp"),
    )


def act_graph(pieces, scale=None, zp=None):
    return activation_pair(pieces.scalar if scale is None else scale,
                           pieces.zp_scalar if zp is None else zp)


def act_graph_with_non_q_producer(pieces):
    """The payload IS a graph value, but nothing named QuantizeLinear made it:
    the pair test keys on the producer's op type, or `Relu -> DequantizeLinear`
    would be passed to a runtime that has no quantized bytes to read."""
    m = act_graph(pieces)
    del m.nodes["aq"]
    m.nodes["relu"] = Node("Relu", "relu", {}, [tref("a_in")], [tref("aq_val")])
    return m


def act_graph_with_per_axis_scale_on_the_q_side(pieces):
    """The exporter put the table on the QuantizeLinear and a scalar on the
    DequantizeLinear. Both halves execute, so either side alone is a model that
    throws on its first inference."""
    m = act_graph(pieces)
    m.initializers["q_axis_scale"] = numpy_helper.from_array(
        numpy.full((8,), 0.02, dtype=numpy.float32), "q_axis_scale")
    m.nodes["aq"].inputs[1]["name"] = "q_axis_scale"
    return m


ACTIVATION_GATES = [
    pytest.param(lambda px: act_graph(px, scale=px.per_axis),
                 "per-axis activation scale", id="pair-scale-is-per-axis"),
    pytest.param(lambda px: act_graph(px, zp=px.zp_axis),
                 "per-axis activation zero_point", id="pair-zero-point-is-per-axis"),
    pytest.param(act_graph_with_per_axis_scale_on_the_q_side,
                 "per-axis activation scale", id="q-side-scale-is-per-axis"),
    pytest.param(act_graph_with_non_q_producer,
                 "graph value", id="dq-over-a-value-no-q-produced"),
]


@pytest.mark.parametrize("build,needle", ACTIVATION_GATES)
def test_activation_pair_gate_raises(act_pieces, build, needle):
    """Each case breaks one input of an otherwise valid activation pair."""
    expect_error(build(act_pieces), needle)


# --------------------------------------------------------------------------
# gates: a Conv weight the Conv kernel cannot index
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def conv_pieces():
    oc, ic, kh, kw = 16, 8, 3, 3
    twin = quantize_int8(float_weight_graph(
        conv_weights(oc, ic, kh, kw, 29), [("Conv", "c", {}, ["x", "W"])]))
    payload, scale = twin.initializers["W"], twin.initializers["W_scale"]
    return SimpleNamespace(
        dims=[oc, ic, kh, kw], oc=oc, payload=payload, scale=scale,
        scalar=numpy_helper.from_array(numpy.float32(0.01), "W_scale"),
        short=numpy_helper.from_array(
            numpy.full((oc - 1,), 0.01, dtype=numpy.float32), "W_scale"),
        nan=numpy_helper.from_array(
            numpy.full((oc,), numpy.nan, dtype=numpy.float32), "W_scale"),
        rank3=helper.make_tensor("W", TensorProto.INT8, [2, ic, kh],
                                 bytes(2 * ic * kh), raw=True),
        fp8=helper.make_tensor("W", TensorProto.FLOAT8E4M3FN, [oc, ic, kh, kw],
                               bytes(oc * ic * kh * kw), raw=True),
    )


def conv_graph(pieces, consumers=None, attrs=None, p=None, s=None):
    return dq_graph(p if p is not None else pieces.payload,
                    s if s is not None else pieces.scale,
                    [("Conv", "c", {}, ["x", DQ_OUT])] if consumers is None
                    else consumers,
                    {"axis": 0} if attrs is None else attrs)


CONV_GATES = [
    pytest.param(lambda px: conv_graph(px, attrs={"axis": 1}),
                 "indexes dim 1", id="scale-indexes-a-spatial-axis"),
    pytest.param(lambda px: conv_graph(px, s=px.short),
                 "per output channel", id="scale-length-is-not-oc"),
    pytest.param(lambda px: conv_graph(px, s=px.scalar),
                 "Conv scale has rank 0", id="per-tensor-conv-scale"),
    pytest.param(lambda px: conv_graph(px, p=px.rank3),
                 "4-D", id="rank-3-conv-weight"),
    pytest.param(lambda px: conv_graph(px, s=px.nan),
                 "non-finite", id="non-finite-conv-scale"),
    pytest.param(lambda px: conv_graph(
        px, consumers=[("Conv", "c", {}, ["x", DQ_OUT]),
                       ("MatMul", "mm", {"transB": 0}, ["y", DQ_OUT])]),
        "cannot be shared with a GEMM-family reader",
        id="conv-weight-shared-with-a-gemm"),
    pytest.param(lambda px: conv_graph(
        px, consumers=[("ConvTranspose", "ct", {}, ["x", DQ_OUT])]),
        "only readable by MatMul", id="no-convtranspose-kernel"),
    pytest.param(lambda px: conv_graph(px, p=px.fp8),
                 "only int8 has a Conv kernel", id="fp8-on-a-conv"),
]


@pytest.mark.parametrize("build,needle", CONV_GATES)
def test_conv_weight_gate_raises(conv_pieces, build, needle):
    """Each case breaks exactly one input of an otherwise valid INT8 Conv
    import: the 4-D shape, the per-output-channel table, or the reader."""
    expect_error(build(conv_pieces), needle)


# --------------------------------------------------------------------------
# gates: a constant whose dequantization this host cannot evaluate
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def const_pieces():
    """A valid [2, 3] QInt32 constant plus a broken variant of each table the
    host-side dequantization reads."""
    return SimpleNamespace(
        payload=numpy_helper.from_array(
            numpy.arange(6, dtype=numpy.int32).reshape(2, 3), "c_q"),
        scale=numpy_helper.from_array(
            numpy.array([0.5, 0.25, 2.0], dtype=numpy.float32), "c_scale"),
        rank0=numpy_helper.from_array(numpy.int32(5), "c_q"),
        rows=numpy_helper.from_array(
            numpy.array([0.5, 0.25], dtype=numpy.float32), "c_scale"),
        grid=numpy_helper.from_array(
            numpy.full((2, 3), 0.5, dtype=numpy.float32), "c_scale"),
        int_scale=numpy_helper.from_array(
            numpy.arange(6, dtype=numpy.int64).reshape(2, 3), "c_scale"),
        short_zp=numpy_helper.from_array(
            numpy.array([1, -2], dtype=numpy.int32), "c_zp"),
    )


def const_graph(pieces, consumers=None, attrs=None, p=None, s=None, zp=None):
    return dq_graph(p if p is not None else pieces.payload,
                    s if s is not None else pieces.scale,
                    [("Add", "a", {}, ["x", DQ_OUT])] if consumers is None
                    else consumers,
                    {"axis": 1} if attrs is None else attrs, zero_point=zp)


def const_graph_with_computed(name):
    def build(px):
        m = const_graph(px)
        if name == "zero_point":
            m.nodes["dq"].inputs.append(tref("computed_zero_point"))
        else:
            m.nodes["dq"].inputs[1]["name"] = "computed_scale"
        return m

    return build


CONSTANT_GATES = [
    pytest.param(lambda px: const_graph(px, p=px.rank0),
                 "rank-0 integer constant", id="scalar-constant-has-no-axis"),
    pytest.param(lambda px: const_graph(px, s=px.rows),
                 "scale table is 2 long", id="scale-length-is-not-the-axis"),
    pytest.param(lambda px: const_graph(px, s=px.grid),
                 "scale table has rank 2", id="scale-is-not-1-d"),
    pytest.param(lambda px: const_graph(px, s=px.int_scale),
                 "scale is float or half", id="integer-scale"),
    pytest.param(lambda px: const_graph(px, zp=px.short_zp),
                 "zero_point table is 2 long", id="zero-point-length-disagrees"),
    pytest.param(const_graph_with_computed("scale"),
                 "weight-only kernel binds a constant scale table",
                 id="scale-computed-at-runtime"),
    pytest.param(const_graph_with_computed("zero_point"),
                 "host-side dequantization needs the constant",
                 id="zero-point-computed-at-runtime"),
    pytest.param(lambda px: const_graph(
        px, attrs={"axis": 1, "output_dtype": TensorProto.INT64}),
        "a host-evaluated constant is float or half",
        id="output-dtype-is-not-float"),
]


@pytest.mark.parametrize("build,needle", CONSTANT_GATES)
def test_constant_gate_raises(const_pieces, build, needle):
    """Each case is a dequantization the host cannot evaluate faithfully, so the
    fold has to raise rather than write numbers the graph did not ask for."""
    expect_error(build(const_pieces), needle)
