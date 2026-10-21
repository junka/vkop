"""shader_map.py and core/ShaderKeys.hpp must describe the same op → shader table.

The Python side drives onnx2vkop's per-model archive filtering;
the C++ side drives Runtime::LoadModel's post-create_from_type `override_spv()`
lookup. A divergence between them is silent at conversion time and shows up
later as either:

  * a model whose archive lacks a shader the graph dispatches on
    (Runtime falls through to a stubbed array and Operator::set_runtime_device
    throws with "this libvkop carries no built-in shaders"), or
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
    misses at LoadModel time fall through to embedded, so the op hits the
    stubbed-array throw and a whole model becomes unrunnable — this catches
    it at build/CI time instead.
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


def test_every_literal_resolve_spv_key_is_declared():
    """The other direction: every key an op hard-codes into
    Operator::resolve_spv() must appear in shader_map._SECONDARY.

    Those are the pipelines Runtime never sees (they are built lazily inside
    set_runtime_device from a device-feature or dtype test), so nothing derives
    their archive key from the op table -- the C++ literal is the only place it
    exists. Omit one and a converted model's subset simply lacks it: the op
    resolves to its stub array and loses that kernel, with
    only a runtime warning to say so. This is the silent half of the drift the
    master-archive test above cannot see, because the key does exist in the
    master (CMake compiled it) -- it just never reaches the model.
    """
    ops_dir = os.path.join(REPO_ROOT, "ops")
    literals = set()
    for name in sorted(os.listdir(ops_dir)):
        if not name.endswith((".hpp", ".cpp")):
            continue
        with open(os.path.join(ops_dir, name)) as f:
            literals.update(
                re.findall(r'resolve_spv\(\s*"([^"]+)"', f.read()))
    shipped = {k for keys in shader_map._SECONDARY.values() for k in keys}
    missing = sorted(literals - shipped)
    assert not missing, (
        f"ops/ builds pipelines with {missing}, which shader_map._SECONDARY "
        "does not list, so onnx2vkop never embeds them into a model's archive"
    )


# ------------------------------------------------------------- kernel plan
# Model.kernel_plans is written into every .vkopbin and read by
# Runtime::kernel_key_for_node in place of the C++ table, and the per-model
# archive is filtered from the same rows. So the plan is now a third copy of
# the op -> key decision; these tests pin it to the two that came before.
def _legacy_key_selection(op_types, backend, emit_fp16=True):
    """The formula shader_keys_for_model used before the plan existed:
    this backend's primary keys + this backend's secondary pipelines."""
    keys = set()
    for op in set(op_types):
        keys.update(shader_map._primary_keys(op, backend, emit_fp16))
        keys.update(shader_map._SECONDARY.get((op, backend), ()))
    return keys


def test_plan_reproduces_the_legacy_key_selection():
    """The plan-derived selection must equal the formula it replaced, for
    every op and backend. A difference means an already-converted model (or
    one reconverted today) would lose a shader its graph dispatches."""
    ops = shader_map.known_ops()
    for op in sorted(ops, key=str):
        for backend in ("buffer", "image"):
            for emit_fp16 in (True, False):
                got = shader_map.shader_keys_for_model(
                    [op], backend=backend, emit_fp16=emit_fp16)
                want = _legacy_key_selection([op], backend, emit_fp16)
                assert got == want, (op, backend, emit_fp16, got, want)
    for backend in ("buffer", "image"):
        assert (shader_map.shader_keys_for_model(ops, backend=backend)
                == _legacy_key_selection(ops, backend))
    assert (shader_map.shader_keys_for_model(ops, backend="auto")
            == (_legacy_key_selection(ops, "buffer")
                | _legacy_key_selection(ops, "image")))


def test_plan_variants_carry_their_backend_prefix():
    """A pinned backend keeps only the variants matching its prefix (the
    selection _SECONDARY used to make), so a variant spelled with the other
    backend's prefix would be dropped from that archive."""
    for (op, backend), keys in shader_map._SECONDARY.items():
        for k in keys:
            assert k.startswith(backend + "_"), (op, backend, k)


def test_plan_cells_are_post_fallback():
    """Each cell is the key the runtime will ask for, not a raw basename:
    an op without a variant on a backend resolves to the other backend's
    spelling (Attention is buffer-only, MaxPool is image-only), and an op
    that is not in DUAL_FP16_SHADERS repeats its fp32 key in the fp16 cell
    rather than leaving the cell empty and letting the runtime guess."""
    attention = shader_map.plan_for_op("Attention")
    assert attention["image"] == "buffer_attention"
    assert attention["image_fp16"] == "buffer_attention_fp16"
    assert attention["buffer"] == "buffer_attention"
    assert shader_map.plan_for_op("MaxPool")["buffer"] == "image_maxpool2d"
    assert shader_map.plan_for_op("MaxPool")["buffer_fp16"] == "image_maxpool2d_fp16"
    # dual: Conv has both an fp32 and an fp16 shader per backend
    assert shader_map.plan_for_op("Conv")["buffer_fp16"] == "buffer_conv2d_fp16"
    # not dual: Relu's fp16 request resolves to the same shader
    assert shader_map.plan_for_op("Relu")["buffer_fp16"] == "buffer_relu"
    assert shader_map.plan_for_op("Relu")["image_fp16"] == "image_relu"
    # host-only ops have no key under any spelling, hence no row
    for host_only in ("Shape", "Squeeze", "Unsqueeze", "Identity", "NotAnOp"):
        assert shader_map.plan_for_op(host_only) is None
