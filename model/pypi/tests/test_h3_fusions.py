#!/usr/bin/env python3
"""MiniMax-H3 graph folds: tag_hadamard_matmul and fuse_swiglu_interleaved.

tag_hadamard_matmul tags MatMul(x[.., 256], H) hadamard=256 only when B holds
exactly the normalized regular Hadamard of order 256 (ConvRot's kron(h4)^4/16);
the runtime then runs hadamard256.comp instead of a GEMM.

fuse_swiglu_interleaved folds MatMul -> Reshape[-1, F/4, 2, 4] ->
Gather(axis 2, 0 | 1) -> silu(g) * u -> Reshape[-1, F] (the exporter's form
for an fc1 whose rows are interleaved gate/up four at a time) into one
MatMul swiglu=1.
"""
import json
import os
import subprocess
import sys
import tempfile

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

PYPI = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _hadamard256():
    h4 = np.array([[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]],
                  np.float32)
    h = h4
    while h.shape[0] < 256:
        h = np.kron(h, h4)
    return h / np.float32(16.0)


def _convert(model, tmp, *extra):
    src = os.path.join(tmp, "m.onnx")
    onnx.save(model, src)
    out = os.path.join(tmp, "m.vkopbin")
    p = subprocess.run(
        [sys.executable, "-m", "onnx2vkop.cli", "-i", src, "-o", out,
         "--target-backend", "buffer", *extra],
        capture_output=True, text=True, cwd=PYPI)
    assert p.returncode == 0, p.stdout[-1500:] + p.stderr[-1500:]
    q = subprocess.run(
        [sys.executable, "-m", "onnx2vkop.inspect", out, "--section", "nodes",
         "--json"], capture_output=True, text=True,
        env={**os.environ, "PYTHONPATH": PYPI})
    assert q.returncode == 0, q.stderr[-400:]
    return json.loads(q.stdout)["nodes"]


def _model(nodes, inits, x_shape, y_name, y_shape):
    g = helper.make_graph(
        nodes, "g",
        [helper.make_tensor_value_info("X", TensorProto.FLOAT16, x_shape)],
        [helper.make_tensor_value_info(y_name, TensorProto.FLOAT16, y_shape)],
        inits)
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])
    m.ir_version = 8
    return m


def _hadamard_graph(h):
    return _model([helper.make_node("MatMul", ["X", "H"], ["Y"], name="rot")],
                  [numpy_helper.from_array(h.astype(np.float16), "H")],
                  ["n", 256], "Y", ["n", 256])


def test_exact_hadamard_is_tagged():
    with tempfile.TemporaryDirectory() as tmp:
        nodes = _convert(_hadamard_graph(_hadamard256()), tmp)
    mm = [n for n in nodes if n["op_type"] == "MatMul"]
    assert len(mm) == 1
    assert mm[0]["attributes"].get("hadamard") in (256, [256])


def test_a_perturbed_matrix_is_left_a_gemm():
    h = _hadamard256().copy()
    h[3, 7] = -h[3, 7]
    with tempfile.TemporaryDirectory() as tmp:
        nodes = _convert(_hadamard_graph(h), tmp)
    mm = [n for n in nodes if n["op_type"] == "MatMul"]
    assert len(mm) == 1 and "hadamard" not in mm[0]["attributes"]


def _swiglu_graph(f=64, k=32, last_shape=(2, 4)):
    rng = np.random.default_rng(0)
    w = (rng.standard_normal((k, 2 * f)) * 0.1).astype(np.float16)
    inits = [numpy_helper.from_array(w, "W"),
             numpy_helper.from_array(
                 np.array([-1, f // 4, *last_shape], np.int64), "s1"),
             numpy_helper.from_array(np.array([-1, f], np.int64), "s2"),
             numpy_helper.from_array(np.array(0, np.int64), "i0"),
             numpy_helper.from_array(np.array(1, np.int64), "i1")]
    nodes = [
        helper.make_node("MatMul", ["X", "W"], ["h"], name="fc1"),
        helper.make_node("Reshape", ["h", "s1"], ["h4"], name="r1"),
        helper.make_node("Gather", ["h4", "i0"], ["g"], axis=2, name="gg"),
        helper.make_node("Gather", ["h4", "i1"], ["u"], axis=2, name="gu"),
        helper.make_node("Sigmoid", ["g"], ["sg"], name="sig"),
        helper.make_node("Mul", ["g", "sg"], ["silu"], name="m1"),
        helper.make_node("Mul", ["silu", "u"], ["y4"], name="m2"),
        helper.make_node("Reshape", ["y4", "s2"], ["Y"], name="r2"),
    ]
    return _model(nodes, inits, ["n", k], "Y", ["n", f])


def test_interleaved_swiglu_folds_into_one_matmul():
    with tempfile.TemporaryDirectory() as tmp:
        nodes = _convert(_swiglu_graph(), tmp)
    ops = [n["op_type"] for n in nodes]
    assert ops.count("MatMul") == 1, ops
    mm = next(n for n in nodes if n["op_type"] == "MatMul")
    assert mm["attributes"].get("swiglu") in (1, [1]), mm
    assert mm["outputs"][0]["dims"][-1] == 64
    assert not {"Gather", "Sigmoid", "Reshape"} & set(ops), ops


def test_a_different_reshape_is_not_folded():
    # [-1, F/4, 4, 2] is not the epilogue's interleave
    with tempfile.TemporaryDirectory() as tmp:
        nodes = _convert(_swiglu_graph(last_shape=(4, 2)), tmp)
    mm = [n for n in nodes if n["op_type"] == "MatMul"]
    assert all("swiglu" not in n["attributes"] for n in mm)
