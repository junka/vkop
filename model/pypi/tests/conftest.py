"""Put the repo-root `tools/` on sys.path for the converter test suite.

`test_shader_map_agrees.py` imports `filter_shaders`, which lives in the repo's
top-level `tools/` -- outside the `onnx2vkop` package and outside the wheel, so
the only thing that ever made it importable was `onnx2vkop/cli.py:12-17`, which
appends that directory as an import side effect. Under plain pytest nothing
imports the CLI first, so the test died with `ModuleNotFoundError: No module
named 'filter_shaders'` before checking anything, and 2026-10-18's "151 python
tests pass" was only reproducible by whoever remembered to export
`PYTHONPATH=<repo>/tools`.

Doing the path hop here means `pytest model/pypi/tests` works from any cwd,
with or without an editable install, and with or without `tools/` present (the
test itself skips cleanly if the master archive has not been built).
"""
import sys
from pathlib import Path

# tests/ -> pypi/ -> model/ -> repo root
_REPO_ROOT = Path(__file__).resolve().parents[3]
_TOOLS = _REPO_ROOT / "tools"

if _TOOLS.is_dir() and str(_TOOLS) not in sys.path:
    sys.path.append(str(_TOOLS))
