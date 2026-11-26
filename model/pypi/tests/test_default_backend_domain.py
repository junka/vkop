"""The default CLI conversion (--target-backend auto) must get the half domain.

1114754 started forwarding the CLI's `target_backend` into the width passes,
and the CLI default is "auto". GraphFp16.apply only declares for "buffer", so a
plain `onnx2vkop -i m.onnx` printed "auto backend: ... nothing to declare" and
shipped fp32 activations -- for fp16 and bf16 alike -- while every CLI test
passed `--target-backend buffer` explicitly and never saw it. "auto" carries
both backends' shaders and the image backend ignores declarations, so the width
passes must treat it as buffer (converter._declared_widths_backend).
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import onnx
import pytest
from onnx import TensorProto as T
from onnx import helper as h
from onnx import numpy_helper as nh

PYPI = str(Path(__file__).resolve().parents[1])


def tiny_fp32_model(path):
    w = nh.from_array(
        np.random.default_rng(0).standard_normal((64, 64)).astype(np.float32),
        "w")
    g = h.make_graph(
        [h.make_node("MatMul", ["x", "w"], ["y"]),
         h.make_node("Relu", ["y"], ["z"])],
        "g",
        [h.make_tensor_value_info("x", T.FLOAT, [4, 64])],
        [h.make_tensor_value_info("z", T.FLOAT, [4, 64])],
        [w])
    m = h.make_model(g, opset_imports=[h.make_opsetid("", 17)])
    m.ir_version = 8
    onnx.save(m, path)


def convert(src, out, *extra):
    p = subprocess.run(
        [sys.executable, "-m", "onnx2vkop.cli", "-i", src, "-o", out, *extra],
        capture_output=True, text=True, cwd=PYPI)
    log = p.stdout + p.stderr
    assert p.returncode == 0, log[-2000:]
    return log


def dtypes_of(path):
    p = subprocess.run(
        [sys.executable, "-m", "onnx2vkop.inspect", path, "--json"],
        capture_output=True, text=True, env={**os.environ, "PYTHONPATH": PYPI})
    assert p.returncode == 0, p.stderr[-400:]
    d = json.loads(p.stdout)
    return ({t["name"]: t["dtype"] for t in d["inputs"] + d["outputs"]},
            {t["name"]: t["dtype"] for t in d["initializers"]})


@pytest.mark.parametrize("backend", [[], ["--target-backend", "auto"],
                                     ["--target-backend", "buffer"]],
                         ids=["default", "auto", "buffer"])
def test_default_and_auto_get_the_fp16_domain(tmp_path, backend):
    src, out = str(tmp_path / "m.onnx"), str(tmp_path / "m.vkopbin")
    tiny_fp32_model(src)
    log = convert(src, out, *backend)
    assert "nothing to declare" not in log, log[-2000:]
    io, inits = dtypes_of(out)
    assert io == {"x": "float16", "z": "float16"}
    assert inits == {"w": "float16"}


def test_image_backend_still_declares_nothing(tmp_path):
    src, out = str(tmp_path / "m.onnx"), str(tmp_path / "m.vkopbin")
    tiny_fp32_model(src)
    log = convert(src, out, "--target-backend", "image")
    assert "nothing to declare" in log
