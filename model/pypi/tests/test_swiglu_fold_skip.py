"""The SwiGLU folds are skipped when the artifact could not run them.

fuse_swiglu_gate_up (MatMul swiglu=1) and fuse_swiglu_down_proj (down_act)
only have half-activation buffer kernels. Before, --act fp32 (= -q fp32) on an
fp32-weight SwiGLU export either shipped fp32 fold weights the kernels read as
halves (all NaN) or, after 58562ef, failed the conversion at
_validate_folded_weights. The converter now drops those two passes for
--act fp32 and the image backend (converter._fusions_the_target_cannot_run),
so the unfolded chain ships and runs.
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
REPO = str(Path(PYPI).parents[1])
D, F = 256, 512


def swiglu_mlp(path, m=4, seed=0):
    rng = np.random.default_rng(seed)
    wg = (rng.standard_normal((D, F)) / 16).astype(np.float32)
    wu = (rng.standard_normal((D, F)) / 16).astype(np.float32)
    wd = (rng.standard_normal((F, D)) / 22).astype(np.float32)
    nodes = [h.make_node("MatMul", ["x", "Wg"], ["g"]),
             h.make_node("Sigmoid", ["g"], ["sg"]),
             h.make_node("Mul", ["g", "sg"], ["silu"]),
             h.make_node("MatMul", ["x", "Wu"], ["u"]),
             h.make_node("Mul", ["silu", "u"], ["a"]),
             h.make_node("MatMul", ["a", "Wd"], ["out"])]
    g = h.make_graph(nodes, "mlp",
                     [h.make_tensor_value_info("x", T.FLOAT, [m, D])],
                     [h.make_tensor_value_info("out", T.FLOAT, [m, D])],
                     [nh.from_array(wg, "Wg"), nh.from_array(wu, "Wu"),
                      nh.from_array(wd, "Wd")])
    model = h.make_model(g, opset_imports=[h.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.save(model, path)


def convert(src, out, *extra):
    p = subprocess.run(
        [sys.executable, "-m", "onnx2vkop.cli", "-i", src, "-o", out, *extra],
        capture_output=True, text=True, cwd=PYPI)
    log = p.stdout + p.stderr
    assert p.returncode == 0, log[-2000:]
    return log


def fold_attrs(path):
    p = subprocess.run(
        [sys.executable, "-m", "onnx2vkop.inspect", path, "--json"],
        capture_output=True, text=True, env={**os.environ, "PYTHONPATH": PYPI})
    assert p.returncode == 0, p.stderr[-400:]
    found = set()
    for n in json.loads(p.stdout)["nodes"]:
        for a in ("swiglu", "down_act"):
            if a in n.get("attributes", {}):
                found.add(a)
    return found


@pytest.mark.parametrize("flags", [["-q", "fp32"], ["--act", "fp32"],
                                   ["--act", "fp32", "-q", "int8"],
                                   ["--target-backend", "image"]],
                         ids=["q-fp32", "act-fp32", "int8-act-fp32", "image"])
def test_folds_are_skipped_where_they_cannot_run(tmp_path, flags):
    src, out = str(tmp_path / "m.onnx"), str(tmp_path / "m.vkopbin")
    swiglu_mlp(src)
    log = convert(src, out, *flags)
    assert "[fusions] fuse_swiglu_gate_up skipped" in log
    assert "[fusions] fuse_swiglu_down_proj skipped" in log
    assert fold_attrs(out) == set()


@pytest.mark.parametrize("flags", [[], ["--target-backend", "buffer"]],
                         ids=["default", "buffer"])
def test_folds_still_run_on_half_activations(tmp_path, flags):
    src, out = str(tmp_path / "m.onnx"), str(tmp_path / "m.vkopbin")
    swiglu_mlp(src)
    log = convert(src, out, *flags)
    assert "[fusions] fuse_swiglu" not in log
    assert "swiglu" in fold_attrs(out)


def test_bf16_keeps_the_gate_up_fold_and_drops_the_down_fold(tmp_path):
    """--act bf16: the gate/up fold has bf16 kernels, the down fold's
    matmul_down_act.comp does not (its node would be held FP32 by the bf16
    gate, and no fold kernel takes fp32)."""
    src, out = str(tmp_path / "m.onnx"), str(tmp_path / "m.vkopbin")
    swiglu_mlp(src)
    log = convert(src, out, "--act", "bf16")
    assert "[fusions] fuse_swiglu_down_proj skipped" in log
    assert "[fusions] fuse_swiglu_gate_up skipped" not in log
    assert fold_attrs(out) == {"swiglu"}


BENCH = os.environ.get("VKOP_BENCH")
ARCHIVE = os.environ.get("VKOP_MASTER_ARCHIVE",
                         os.path.join(REPO, "build", "vkop_shaders.bin"))


@pytest.mark.skipif(not BENCH or not os.path.exists(ARCHIVE),
                    reason="set VKOP_BENCH (a vkop-bench, or a wrapper that "
                           "exports the Vulkan loader env) and a shader archive")
@pytest.mark.parametrize("m", [1, 16], ids=["gemv", "tile"])
def test_fp32_act_unfolded_mlp_matches_onnxruntime(tmp_path, m):
    pytest.importorskip("onnxruntime")
    src, out = str(tmp_path / "m.onnx"), str(tmp_path / "m.vkopbin")
    swiglu_mlp(src, m=m)
    convert(src, out, "-q", "fp32", "--master-archive", ARCHIVE)
    p = subprocess.run(
        [sys.executable, os.path.join(REPO, "tools", "vkop_vs_ort.py"), out,
         "--onnx", src, "--bench", BENCH, "--random-fill",
         "--precision", "fp32", "--cos-min", "0.9999"],
        capture_output=True, text=True)
    report = p.stdout + p.stderr
    assert p.returncode == 0, report[-2000:]
    assert "RESULT: 1/1" in report, report[-2000:]
