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
