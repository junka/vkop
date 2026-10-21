"""runtime_op_names() must equal the op_type strings Ops.hpp resolves.

shader_map.runtime_op_names() is the one set converter-side validation and the
coverage tools use to decide whether a graph is convertible. The runtime's
actual accept list is `convert_opstring_to_enum` in ops/Ops.hpp. If the two
drift, a graph the converter accepts still maps to OpType::UNKNOWN at load --
which runtime.cpp now turns into a hard error. So this test parses Ops.hpp and
fails on any divergence, the same way test_shader_map_agrees.py parses
ShaderKeys.hpp.
"""
import re
from pathlib import Path

from onnx2vkop import shader_map

_REPO_ROOT = Path(__file__).resolve().parents[3]
_OPS_HPP = _REPO_ROOT / "ops" / "Ops.hpp"


def _cpp_accepted_names():
    src = _OPS_HPP.read_text()
    start = src.index("inline OpType convert_opstring_to_enum")
    end = src.index("return vkop::ops::OpType::UNKNOWN;", start)
    body = src[start:end]
    return set(re.findall(r'name\s*==\s*"([A-Za-z0-9_]+)"', body))


def test_runtime_op_names_match_ops_hpp():
    cpp = _cpp_accepted_names()
    py = shader_map.runtime_op_names()
    assert not (py - cpp), (
        f"runtime_op_names has names Ops.hpp rejects: {sorted(py - cpp)}")
    assert not (cpp - py), (
        f"Ops.hpp accepts names runtime_op_names lacks: {sorted(cpp - py)}")


def test_archive_only_spellings_are_excluded():
    """_PRIMARY's alias spellings the runtime does NOT resolve must not be
    offered as accepted, or validation would pass a node the runtime rejects."""
    accepted = shader_map.runtime_op_names()
    for spelling in ("Topk", "Maxpool2d"):
        assert spelling not in accepted
    for spelling in ("TopK", "MaxPool2d", "MaxPool"):
        assert spelling in accepted


def test_host_only_ops_are_included():
    accepted = shader_map.runtime_op_names()
    for op in ("Shape", "Squeeze", "Unsqueeze"):
        assert op in accepted


def test_known_ops_has_no_tuples():
    """known_ops() used to union _SECONDARY.keys(), which are (op, backend)
    tuples -- a mixed str/tuple set that was unusable as a whitelist."""
    assert all(isinstance(op, str) for op in shader_map.known_ops())
