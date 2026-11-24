"""fuse_swiglu_gate_up: gate/up GEMMs + silu(g)*u -> one MatMul swiglu=1.

The runtime (ops/Matmul.hpp, swiglu) reads B as Wg and Wu interleaved four
columns at a time and applies silu(g) * u in the epilogue, so these cases pin
the converter half of that ABI:

  * the five nodes become one MatMul [x, Wgu] producing the Mul's output, and
    the down projection stays a plain MatMul;
  * Wgu is exactly g0..g3 u0..u3 g4.. for in-memory and streamed weights;
  * shapes or readers the fold cannot keep exact are left alone.
"""
import io
import os
import tempfile

import numpy as np
from onnx import TensorProto, numpy_helper

from onnx2vkop import weight_stream
from onnx2vkop.dag import DAGBasedModel, Node
from onnx2vkop.optimizer import FusionOptimizer

FLOAT16 = TensorProto.FLOAT16
K, N_UP, N = 16, 32, 8


def tdict(name, dims=None, dtype=FLOAT16):
    return {"name": name, "shape": list(dims or []), "dtype": dtype}


def build(n_up=N_UP, extra_gate_reader=False):
    rng = np.random.default_rng(3)
    wg = rng.normal(size=(K, n_up)).astype(np.float16)
    wu = rng.normal(size=(K, n_up)).astype(np.float16)
    wd = rng.normal(size=(n_up, N)).astype(np.float16)
    m = DAGBasedModel()
    for name, arr in (("Wg", wg), ("Wu", wu), ("Wd", wd)):
        m.initializers[name] = numpy_helper.from_array(arr, name)
    m.nodes["gate"] = Node("MatMul", "gate", {},
                           [tdict("x", [1, K]), tdict("Wg", [K, n_up])],
                           [tdict("g", [1, n_up])])
    m.nodes["up"] = Node("MatMul", "up", {},
                         [tdict("x", [1, K]), tdict("Wu", [K, n_up])],
                         [tdict("u", [1, n_up])])
    m.nodes["sig"] = Node("Sigmoid", "sig", {}, [tdict("g", [1, n_up])],
                          [tdict("s", [1, n_up])])
    m.nodes["m1"] = Node("Mul", "m1", {},
                         [tdict("g", [1, n_up]), tdict("s", [1, n_up])],
                         [tdict("a", [1, n_up])])
    m.nodes["m2"] = Node("Mul", "m2", {},
                         [tdict("a", [1, n_up]), tdict("u", [1, n_up])],
                         [tdict("h", [1, n_up])])
    m.nodes["down"] = Node("MatMul", "down", {},
                           [tdict("h", [1, n_up]), tdict("Wd", [n_up, N])],
                           [tdict("y", [1, N])])
    if extra_gate_reader:
        m.nodes["tap"] = Node("Relu", "tap", {}, [tdict("g", [1, n_up])],
                              [tdict("t", [1, n_up])])
    return m, wg, wu


def interleave(wg, wu):
    k, n = wg.shape
    return np.concatenate([wg.reshape(k, n // 4, 4), wu.reshape(k, n // 4, 4)],
                          axis=2).reshape(k, 2 * n)


def fold_all(m):
    matches = FusionOptimizer.match_swiglu_gate_up(m)
    for mt in matches:
        FusionOptimizer.fold_swiglu_gate_up(m, mt)
    return len(matches)


def silu_mul(x, wg, wu):
    g = x @ wg.astype(np.float32)
    u = x @ wu.astype(np.float32)
    return g / (1.0 + np.exp(-g)) * u


def test_fold_in_memory():
    m, wg, wu = build()
    assert fold_all(m) == 1
    ops = sorted(n.op_type for n in m.nodes.values())
    assert ops == ["MatMul", "MatMul"]
    fused = [n for n in m.nodes.values() if n.attributes.get("swiglu")][0]
    assert fused.outputs[0]["name"] == "h"
    assert fused.inputs[0]["name"] == "x"
    assert m.nodes["down"].inputs[0]["name"] == "h"
    gu = numpy_helper.to_array(m.initializers[fused.inputs[1]["name"]])
    assert gu.shape == (K, 2 * N_UP)
    np.testing.assert_array_equal(gu, interleave(wg, wu))
    assert "Wg" not in m.initializers and "Wu" not in m.initializers
    # What the runtime computes from it: each octet = gate quartet, up quartet.
    x = np.random.default_rng(5).normal(size=(3, K)).astype(np.float32)
    p = (x @ gu.astype(np.float32)).reshape(3, N_UP // 4, 2, 4)
    g, u = p[:, :, 0, :].reshape(3, N_UP), p[:, :, 1, :].reshape(3, N_UP)
    np.testing.assert_allclose(g / (1.0 + np.exp(-g)) * u,
                               silu_mul(x, wg, wu), rtol=1e-5, atol=1e-5)


def test_fold_streamed_weights():
    m, wg, wu = build()
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "w.bin")
        with open(path, "wb") as f:
            f.write(b"\0" * 64)
            off_g = f.tell()
            f.write(wg.tobytes())
            off_u = f.tell()
            f.write(wu.tobytes())
        deferred = {}
        for name, off, arr in (("Wg", off_g, wg), ("Wu", off_u, wu)):
            t = TensorProto()
            t.name = name
            t.data_type = FLOAT16
            t.dims.extend(arr.shape)
            t.data_location = TensorProto.EXTERNAL
            for key, val in (("location", "w.bin"), ("offset", str(off)),
                             ("length", str(arr.nbytes))):
                kv = t.external_data.add()
                kv.key, kv.value = key, val
            m.initializers[name] = t
            deferred[name] = {"name": name, "location": "w.bin",
                              "base_dir": d, "offset": off,
                              "length": arr.nbytes, "data_type": FLOAT16,
                              "dims": arr.shape, "consumers": 1}
        m.deferred_weights = deferred
        assert fold_all(m) == 1
        fused = [n for n in m.nodes.values() if n.attributes.get("swiglu")][0]
        name = fused.inputs[1]["name"]
        assert set(deferred) == {name}
        weight_stream.verify_deferred(deferred, m)
        out = io.BytesIO()
        written = weight_stream.write_payload(deferred[name], out, d, m)
        assert written == 2 * wg.nbytes
        got = np.frombuffer(out.getvalue(), dtype=np.float16).reshape(K, 2 * N_UP)
        np.testing.assert_array_equal(got, interleave(wg, wu))


def test_no_fold_when_gate_has_another_reader():
    m, _, _ = build(extra_gate_reader=True)
    assert fold_all(m) == 0


def test_no_fold_when_n_up_not_multiple_of_four():
    m, _, _ = build(n_up=30)
    assert fold_all(m) == 0


# ---------------------------------------------------------------------------
# A quantized gate/up weight must never reach the fold.
#
# The swiglu epilogue applies silu(g) * u at store time and owns the dequant
# scale slot a quantized weight would need, so ops/Matmul.hpp admits only
# [x, W] with a float B. The quantizers reach the same folded MatMul through
# `_match_weight_only`; before that gate grew a swiglu arm, `-q fp8e4m3` (and
# int4/nf4/nvfp4) turned the fused node into a three- or four-input MatMul with
# a byte/nibble B, which converted cleanly and threw at the first execute.
# ---------------------------------------------------------------------------
import pytest  # noqa: E402
import contextlib  # noqa: E402

from onnx2vkop.optimizer import Quantizer  # noqa: E402

# The formats whose swiglu epilogue has no dequant scale slot. int8/fp8 store
# through ks_store_wide_w8, whose epilogue applies a per-column scale and no
# activation, so a byte weight would silently write the plain [M, 2*N] product.
_BYTE_QUANTIZERS = [
    ("int8", lambda m: Quantizer.quantize_to_int8_weight_only(m)),
    ("fp8e4m3", lambda m: Quantizer.quantize_to_fp8_weight_only(m, "fp8e4m3")),
    ("fp8e5m2", lambda m: Quantizer.quantize_to_fp8_weight_only(m, "fp8e5m2")),
]

# The 4-bit formats DO convert: the nibble WIDE GEMV stores through
# ks_store_wide, which carries the swiglu arm. The scale is applied per K group
# before the activation, the same order a float kernel applies it in.
_4BIT_QUANTIZERS = [
    ("int4", lambda m: Quantizer.quantize_to_4bit_weight_only(m, 16, "int4")),
    ("nf4", lambda m: Quantizer.quantize_to_4bit_weight_only(m, 16, "nf4")),
    ("nvfp4", lambda m: Quantizer.quantize_to_nvfp4_weight_only(m)),
]


@pytest.mark.parametrize("fmt,quantize", _BYTE_QUANTIZERS,
                         ids=[q[0] for q in _BYTE_QUANTIZERS])
def test_quantizer_leaves_a_folded_gate_up_weight_float(fmt, quantize):
    """A byte format must skip the interleaved [K, 2N] weight, so the node stays
    loadable."""
    m, _, _ = build()
    assert fold_all(m) == 1
    fused = [n for n in m.nodes.values() if n.attributes.get("swiglu")][0]
    name = fused.inputs[1]["name"]
    before = m.initializers[name].data_type

    with contextlib.redirect_stdout(io.StringIO()):
        quantize(m)

    assert m.initializers[name].data_type == before, (
        f"{fmt}: the swiglu weight was rewritten to "
        f"{TensorProto.DataType.Name(m.initializers[name].data_type)}")
    assert [i["name"] for i in fused.inputs] == ["x", name], (
        f"{fmt}: a dequant scale input was appended to the swiglu MatMul")
    Quantizer._validate_folded_weights(m)


@pytest.mark.parametrize("fmt,quantize", _4BIT_QUANTIZERS,
                         ids=[q[0] for q in _4BIT_QUANTIZERS])
def test_4bit_converts_a_folded_gate_up_weight(fmt, quantize):
    """The nibble form IS supported: the weight becomes a packed payload plus
    its group scale, and the validator accepts the node.

    The scale table keeps the interleaved [K, 2*N_up] width -- the kernel
    indexes one entry per (K group, output column) of the SAME payload, so the
    gate and up halves of an octet take their own group scales.
    """
    m, _, _ = build()
    assert fold_all(m) == 1
    fused = [n for n in m.nodes.values() if n.attributes.get("swiglu")][0]
    name = fused.inputs[1]["name"]
    wide = 2 * N_UP

    with contextlib.redirect_stdout(io.StringIO()):
        quantize(m)

    payload = m.initializers[name]
    assert payload.data_type in (TensorProto.INT4, TensorProto.UINT4,
                                 TensorProto.FLOAT4E2M1), (
        f"{fmt}: weight stayed {TensorProto.DataType.Name(payload.data_type)}")
    assert list(payload.dims) == [K, wide], (
        f"{fmt}: payload dims {list(payload.dims)} lost the interleaved width")
    # The scale rides the same node. int4/nf4 append one fp32 [K/g, N] table;
    # nvfp4 appends an fp8 E4M3 block table plus its per-tensor fp32 factor.
    scale_names = [i["name"] for i in fused.inputs[2:]]
    assert scale_names, f"{fmt}: no scale input was appended"
    scale = m.initializers[scale_names[0]]
    expect_dt = (TensorProto.FLOAT8E4M3FN if fmt == "nvfp4"
                 else TensorProto.FLOAT)
    assert scale.data_type == expect_dt, (
        f"{fmt}: scale is {TensorProto.DataType.Name(scale.data_type)}")
    # One entry per (K group, output column): n_groups * 2*N_up. nvfp4's block
    # is 16 values of K, the same divisor the other two use in this test.
    if fmt == "nvfp4":
        assert len(scale_names) == 2, "nvfp4 needs its global factor too"
        glob = m.initializers[scale_names[1]]
        assert glob.data_type == TensorProto.FLOAT
        assert int(np.prod(glob.dims)) == 1
    assert int(np.prod(scale.dims)) == (K // 16) * wide, (
        f"{fmt}: scale has {list(scale.dims)}, expected [{K // 16}, {wide}]")
    Quantizer._validate_folded_weights(m)


def test_a_byte_quantized_folded_weight_is_refused_by_the_validator():
    """If a byte payload ever reached the fold, the conversion fails instead of
    shipping.

    This is the fail-closed half: PreQuantizedWeights can import somebody
    else's payload and weight_stream can encode a streamed one, so the gate is
    not the only route to a quantized folded weight. The validator runs at the
    end of the quantization stage in converter.py.
    """
    m, _, _ = build()
    assert fold_all(m) == 1
    fused = [n for n in m.nodes.values() if n.attributes.get("swiglu")][0]
    name = fused.inputs[1]["name"]

    fake = numpy_helper.from_array(np.zeros((K, 2 * N_UP), dtype=np.int8), name)
    fake.data_type = TensorProto.INT8
    m.initializers[name] = fake
    m.initializers[name + "_scale"] = numpy_helper.from_array(
        np.ones(2 * N_UP, dtype=np.float32), name + "_scale")
    fused.inputs.append({"name": name + "_scale", "shape": [2 * N_UP]})

    with pytest.raises(ValueError, match="swiglu"):
        Quantizer._validate_folded_weights(m)
