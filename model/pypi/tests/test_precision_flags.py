"""-q is the WEIGHT format, --act the ACTIVATION dtype.

--half-precision used to pick the activation width while -q fp32 also meant
"keep the export's activations", so one axis lived in two flags. --act is now
the single activation knob (converter.resolve_precision); --half-precision is
a deprecated alias, and -q fp32 still implies --act fp32.
"""
import subprocess
import sys

import pytest

from test_default_backend_domain import PYPI, dtypes_of, tiny_fp32_model

sys.path.insert(0, PYPI)
from onnx2vkop.converter import activation_dtype, resolve_precision  # noqa: E402


@pytest.mark.parametrize("quant,act,legacy,want", [
    (None, None, None, (None, "fp16")),
    ("int4", None, None, ("int4", "fp16")),
    ("fp32", None, None, ("fp32", "fp32")),
    (None, "fp32", None, ("fp32", "fp32")),
    ("int8", "fp32", None, ("int8", "fp32")),
    ("int4", "bf16", None, ("int4", "bf16")),
    (None, None, "bf16", (None, "bf16")),
    (None, "bf16", "bf16", (None, "bf16")),
])
def test_resolve(quant, act, legacy, want):
    assert resolve_precision(quant, act, legacy) == want


@pytest.mark.parametrize("quant,act,legacy", [
    ("fp16", "fp32", None),   # an fp16 artifact has no fp32 activations
    ("bf16", "fp32", None),   # never had an fp32-activation build
    ("fp32", "fp16", None),   # keep-the-bytes vs. a domain that rewrites them
    (None, "fp16", "bf16"),   # the alias contradicts the new flag
])
def test_resolve_rejects(quant, act, legacy):
    with pytest.raises(ValueError):
        resolve_precision(quant, act, legacy)


def test_activation_dtype_reads_legacy_args():
    class A:
        pass
    a = A(); a.quant = "fp32"
    assert activation_dtype(a) == "fp32"
    a = A(); a.quant = "int4"; a.half_precision = "bf16"
    assert activation_dtype(a) == "bf16"
    a = A(); a.quant = "int4"; a.act = "fp32"
    assert activation_dtype(a) == "fp32"


def run(src, out, *extra):
    p = subprocess.run(
        [sys.executable, "-m", "onnx2vkop.cli", "-i", src, "-o", out, *extra],
        capture_output=True, text=True, cwd=PYPI)
    return p.returncode, p.stdout + p.stderr


@pytest.mark.parametrize("flags,io_dt,w_dt", [
    ([], "float16", "float16"),
    (["--act", "bf16"], "bfloat16", "bfloat16"),
    (["--half-precision", "bf16"], "bfloat16", "bfloat16"),
    (["--act", "fp32"], "float32", "float32"),
    (["-q", "fp32"], "float32", "float32"),
    (["-q", "int8"], "float16", "int8"),
    (["-q", "int8", "--act", "fp32"], "float32", "int8"),
], ids=["default", "act-bf16", "legacy-half", "act-fp32", "q-fp32",
        "int8", "int8-act-fp32"])
def test_cli_axes(tmp_path, flags, io_dt, w_dt):
    src, out = str(tmp_path / "m.onnx"), str(tmp_path / "m.vkopbin")
    tiny_fp32_model(src)
    rc, log = run(src, out, *flags)
    assert rc == 0, log[-2000:]
    io, inits = dtypes_of(out)
    assert io == {"x": io_dt, "z": io_dt}, (io, log[-1500:])
    assert inits["w"] == w_dt, (inits, log[-1500:])
    if "--half-precision" in flags:
        assert "--half-precision is deprecated" in log


def test_cli_rejects_contradiction(tmp_path):
    src, out = str(tmp_path / "m.onnx"), str(tmp_path / "m.vkopbin")
    tiny_fp32_model(src)
    rc, log = run(src, out, "-q", "fp16", "--act", "fp32")
    assert rc != 0 and "-q fp16 --act fp32" in log, log[-800:]
