"""shader_map.py and core/ShaderKeys.hpp must describe the same op → shader table.

The Python side drives onnx2vkop's `--emit-shader-archive` per-model filtering;
the C++ side drives Runtime::LoadModel's post-create_from_type `override_spv()`
lookup. A divergence between them is silent at conversion time and shows up
later as either:

  * a stripped build whose model archive lacks a shader the graph dispatches on
    (Runtime falls through to a stubbed array and Operator::set_runtime_device
    throws with "this libvkop was built with VKOP_STRIP_BUILTIN_SHADERS=ON"), or
  * a full-archive run that ships bytes for a shader no graph ever uses, which
    just wastes space.

Both cases pass tests that only exercise the "everything works" path (llm_chat
with the embedded build), so the drift has to be caught here, at the point of
truth. This test parses:

  1. core/ShaderKeys.hpp's `lookup_spec` switch, giving {OpType enum → (image,
     buffer, img_fp16, buf_fp16)}.
  2. ops/Ops.hpp's `convert_opstring_to_enum`, giving {ONNX op_type string →
     OpType enum}. Multiple strings map to one enum (Conv and Conv2d both →
     OpType::CONV2D); we require the Python table to have an entry for every
     string the C++ side recognises.

Then we cross-check the two against shader_map._PRIMARY, keyed by op_type
string. Any drift in either direction fails loudly.
"""
import os
import re
from typing import Dict, Tuple

import pytest

from onnx2vkop import shader_map

REPO_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)
SHADER_KEYS_CPP = os.path.join(REPO_ROOT, "core", "ShaderKeys.hpp")
OPS_HPP = os.path.join(REPO_ROOT, "ops", "Ops.hpp")


def _parse_enum_to_shader_spec(path: str) -> Dict[str, Tuple[str, str, bool, bool]]:
    with open(path) as f:
        text = f.read()
    m = re.search(r"auto lookup_spec = \[\]\(T t\)[^{]*\{(.*?)\n    \};", text, re.DOTALL)
    assert m, "could not locate lookup_spec lambda in ShaderKeys.hpp"
    body = m.group(1)
    # Multiple `case T::NAME:` clauses may fall through to one return, so
    # collect all case labels leading up to each return.
    spec_by_enum: Dict[str, Tuple[str, str, bool, bool]] = {}
    pending_cases: list = []
    for line in body.splitlines():
        line = line.strip()
        cm = re.match(r"^case T::([A-Z_]+):$", line)
        if cm:
            pending_cases.append(cm.group(1))
            continue
        rm = re.match(r"^return \{([^}]+)\};", line)
        if rm and pending_cases:
            parts = [p.strip() for p in rm.group(1).split(",")]
            if len(parts) != 4:
                continue
            image = _parse_str_or_null(parts[0])
            buffer_ = _parse_str_or_null(parts[1])
            img_fp16 = parts[2] == "true"
            buf_fp16 = parts[3] == "true"
            for e in pending_cases:
                spec_by_enum[e] = (image, buffer_, img_fp16, buf_fp16)
            pending_cases = []
    return spec_by_enum


def _parse_str_or_null(raw: str):
    if raw == "nullptr":
        return None
    m = re.match(r'^"([^"]+)"$', raw)
    if m:
        return m.group(1)
    # Fall through: unknown shapes (e.g. "default:") skip silently.
    return None


def _parse_opstring_to_enum(path: str) -> Dict[str, str]:
    """{ONNX-style op_type string: OpType enum member name}."""
    with open(path) as f:
        text = f.read()
    m = re.search(
        r"inline OpType convert_opstring_to_enum\([^)]*\)\s*\{(.*?)\n\}",
        text,
        re.DOTALL,
    )
    assert m, "could not locate convert_opstring_to_enum in Ops.hpp"
    body = m.group(1)
    result: Dict[str, str] = {}
    for name, enum in re.findall(
        r'name == "([^"]+)"[^"]*return vkop::ops::OpType::([A-Z_]+)', body
    ):
        result[name] = enum
    return result


@pytest.mark.skipif(
    not (os.path.exists(SHADER_KEYS_CPP) and os.path.exists(OPS_HPP)),
    reason="core/ShaderKeys.hpp or ops/Ops.hpp not present",
)
def test_python_table_matches_cpp_table():
    enum_to_spec = _parse_enum_to_shader_spec(SHADER_KEYS_CPP)
    name_to_enum = _parse_opstring_to_enum(OPS_HPP)
    # Every op_type string C++ accepts must be in shader_map._PRIMARY or
    # explicitly host-only (Shape/Squeeze/Unsqueeze/Identity).
    HOST_ONLY = {"Shape", "Squeeze", "Unsqueeze"}
    missing = sorted(
        n for n, e in name_to_enum.items()
        if n not in shader_map._PRIMARY and n not in HOST_ONLY
    )
    assert not missing, (
        f"convert_opstring_to_enum accepts {missing} that shader_map._PRIMARY "
        "does not know"
    )
    # For every alias group (Conv/Conv2d, MaxPool/MaxPool2d/Maxpool2d, ...),
    # the shader spec C++ reports must match what shader_map reports for the
    # same string.
    for name, enum in name_to_enum.items():
        if name in HOST_ONLY:
            continue
        if enum not in enum_to_spec:
            continue  # host-only (SHAPE, SQUEEZE, UNSQUEEZE) or unknown
        cpp = enum_to_spec[enum]
        py = shader_map._PRIMARY[name]
        # Spec is (image_basename, buffer_basename, img_fp16, buf_fp16).
        # Python may store None for a missing side; C++ uses nullptr → None
        # after _parse_str_or_null.
        assert tuple(py) == cpp, (
            f"op_type {name!r} (enum {enum}) disagrees: "
            f"python {tuple(py)} vs cpp {cpp}"
        )


def test_secondary_keys_exist_in_master_archive():
    """Every (op, backend) secondary key must exist in the master archive or
    a filtered per-model archive will silently drop it. Filtered archive
    misses at LoadModel time fall through to embedded, so on a stripped
    build the op hits the stubbed-array throw and a whole model becomes
    unrunnable — this catches it at build/CI time instead.
    """
    from filter_shaders import read_master  # tools/ on sys.path in the CLI

    master_path = os.path.join(REPO_ROOT, "build", "vkop_shaders.bin")
    if not os.path.exists(master_path):
        pytest.skip("master archive not built yet (no build/vkop_shaders.bin)")
    master = read_master(master_path)
    for (op, backend), keys in shader_map._SECONDARY.items():
        for k in keys:
            assert k in master, (
                f"secondary key {k!r} declared for ({op}, {backend}) is not "
                "in the master archive -- either the shader was never "
                "compiled or the table lists a stale name"
            )
