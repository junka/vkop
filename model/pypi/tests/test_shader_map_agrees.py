"""shader_map.py and core/ShaderKeys.hpp must describe the same op → shader table.

The Python side drives onnx2vkop's per-model archive filtering;
the C++ side drives Runtime::LoadModel's post-create_from_type `override_spv()`
lookup. A divergence between them is silent at conversion time and shows up
later as either:

  * a model whose archive lacks a shader the graph dispatches on
    (Runtime finds no key and Operator::resolve_primary_spv throws with "this
    libvkop carries no built-in SPIR-V"), or
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
        cm = re.match(r"^case T::([A-Z][A-Z0-9_]*):$", line)
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
    """{ONNX-style op_type string: OpType enum member name}.

    A clause may accept several spellings for one enum (`Conv2d` and `Conv`
    both return CONV2D), and enum members contain digits, so this walks the
    body collecting `name == "…"` conditions until the `return` that consumes
    them rather than pairing one string with one enum.
    """
    with open(path) as f:
        text = f.read()
    m = re.search(
        r"inline OpType convert_opstring_to_enum\([^)]*\)\s*\{(.*?)\n\}",
        text,
        re.DOTALL,
    )
    assert m, "could not locate convert_opstring_to_enum in Ops.hpp"
    result: Dict[str, str] = {}
    pending: list = []
    for raw in m.group(1).splitlines():
        line = raw.strip()
        pending.extend(re.findall(r'name == "([^"]+)"', line))
        rm = re.search(r"return vkop::ops::OpType::([A-Z][A-Z0-9_]*);", line)
        if rm:
            for name in pending:
                result[name] = rm.group(1)
            pending = []
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
    a filtered per-model archive will silently drop it. A filtered archive
    miss at LoadModel time means the op resolves no shader and the whole model
    becomes unrunnable — this catches it at build/CI time instead.

    The fp8 hardware-decode group is the one exception, and only as a group:
    CMake builds those modules only when the configure-time glslc accepts
    GL_EXT_float_e4m3, so a master built with the system 1.3.275 shaderc has
    none of them. All present or all absent — a partial group is an
    inconsistent archive (partial rebuild, hand-edited file) and fails here.
    """
    from filter_shaders import read_master  # tools/ on sys.path in the CLI

    master_path = os.path.join(REPO_ROOT, "build", "vkop_shaders.bin")
    if not os.path.exists(master_path):
        pytest.skip("master archive not built yet (no build/vkop_shaders.bin)")
    master = read_master(master_path)
    f8_present = [k for k in shader_map.FP8_HW_KEYS if k in master]
    assert len(f8_present) in (0, len(shader_map.FP8_HW_KEYS)), (
        "the fp8 hardware-decode group is partly present in "
        f"{master_path}: found {f8_present}. CMake emits those keys "
        "under one VKOP_GLSLC_HAS_FLOAT8 gate, so a partial group means a "
        "stale or hand-edited archive — rebuild it."
    )
    f8_absent_ok = not f8_present
    for (op, backend), keys in shader_map._SECONDARY.items():
        for k in keys:
            if f8_absent_ok and k in shader_map.FP8_HW_KEYS:
                continue
            assert k in master, (
                f"secondary key {k!r} declared for ({op}, {backend}) is not "
                "in the master archive -- either the shader was never "
                "compiled or the table lists a stale name"
            )


def test_fp16_buffer_ops_list_both_bda_variants():
    """An op whose buffer shader has a dual fp16 build asks the archive for
    <base>_bda_fp16 when the tensor is fp16 (Operator::try_build_bda_pipeline
    appends _fp16). If that key is not also listed as a secondary variant,
    onnx2vkop never embeds it and the op silently falls back to descriptors.
    The master-archive test cannot see this: the key exists in the master,
    it just never reaches the model. MaxPool was exactly this gap."""
    missing = []
    for op, (_img, buf, _img16, buf16) in shader_map._PRIMARY.items():
        if not buf or not buf16:
            continue
        row = set(shader_map._SECONDARY.get((op, "buffer"), []))
        base = f"buffer_{buf}_bda"
        if base in row and f"{base}_fp16" not in row:
            missing.append((op, f"{base}_fp16"))
    assert not missing, (
        "these fp16 buffer ops ship the fp32 BDA key but not its fp16 twin: "
        f"{missing}; add <base>_bda_fp16 to shader_map._SECONDARY"
    )


def test_every_literal_resolve_spv_key_is_declared():
    """The other direction: every key an op hard-codes into
    Operator::resolve_spv() must appear in shader_map._SECONDARY.

    Those are the pipelines Runtime never sees (they are built lazily inside
    set_runtime_device from a device-feature or dtype test), so nothing derives
    their archive key from the op table -- the C++ literal is the only place it
    exists. Omit one and a converted model's subset simply lacks it: the op
    resolves no shader and loses that kernel, with
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
    spelling (Attention is buffer-only, AveragePool is image-only), and an op
    that is not in DUAL_FP16_SHADERS repeats its fp32 key in the fp16 cell
    rather than leaving the cell empty and letting the runtime guess."""
    attention = shader_map.plan_for_op("Attention")
    assert attention["image"] == "buffer_attention"
    assert attention["image_fp16"] == "buffer_attention_fp16"
    assert attention["buffer"] == "buffer_attention"
    averagepool = shader_map.plan_for_op("AveragePool")
    assert averagepool["buffer"] == "image_averagepool"
    assert averagepool["buffer_fp16"] == "image_averagepool"
    # MaxPool gained a real buffer variant; its cells must not fall back.
    maxpool = shader_map.plan_for_op("MaxPool")
    assert maxpool["buffer"] == "buffer_maxpool2d"
    assert maxpool["buffer_fp16"] == "buffer_maxpool2d_fp16"
    # dual: Conv has both an fp32 and an fp16 shader per backend
    assert shader_map.plan_for_op("Conv")["buffer_fp16"] == "buffer_conv2d_fp16"
    # not dual: Gemm's fp16 request resolves to the same shader
    assert shader_map.plan_for_op("Gemm")["buffer_fp16"] == "buffer_gemm"
    # Relu gained an fp16 buffer build; its image path is still single-build
    assert shader_map.plan_for_op("Relu")["buffer_fp16"] == "buffer_relu_fp16"
    assert shader_map.plan_for_op("Relu")["image_fp16"] == "image_relu"
    # host-only ops have no key under any spelling, hence no row
    for host_only in ("Shape", "Squeeze", "Unsqueeze", "Identity", "NotAnOp"):
        assert shader_map.plan_for_op(host_only) is None


RUNTIME_CPP = os.path.join(REPO_ROOT, "core", "runtime.cpp")


def _parse_fp16_dispatch_switch(path: str) -> set:
    """{OpType enum} whose fp16 build is chosen from the tensor dtype.

    The switch's fall-through groups each end in a return; a group is
    dtype-driven when the statement that produced it reads
    `typeid(uint16_t)` off a tensor and not when it hands back `precision`.
    Comment lines are skipped because the explanations in that switch talk
    about fp16 without deciding anything.
    """
    with open(path) as f:
        text = f.read()
    m = re.search(r"int op_shader_fp16\([^)]*\)\s*\{(.*?)\n\}", text, re.DOTALL)
    assert m, "could not locate op_shader_fp16 in core/runtime.cpp"
    driven, pending, chunk = set(), [], []

    def flush():
        if pending and any("typeid(uint16_t)" in line for line in chunk):
            driven.update(pending)
        del pending[:]
        del chunk[:]

    for raw in m.group(1).splitlines():
        line = raw.strip()
        if not line or line.startswith("//"):
            continue
        # Enum members carry digits (CONV2D, MAXPOOL2D); a label that does not
        # match one is a statement, not a case.
        cm = re.match(r"^case ops::OpType::([A-Z][A-Z0-9_]*):$", line)
        if cm:
            if chunk:
                flush()
            pending.append(cm.group(1))
            continue
        if line.startswith("default:"):
            flush()
            continue
        # Statements before the first case (the `data_input == nullptr` guard,
        # the `switch` line) also read a dtype-adjacent expression, but they
        # belong to no group; ignoring them while `pending` is empty keeps the
        # prologue from being mistaken for a fall-through body.
        if pending:
            chunk.append(line)
    flush()
    return driven


@pytest.mark.skipif(not (os.path.exists(RUNTIME_CPP) and os.path.exists(OPS_HPP)),
                    reason="core/runtime.cpp or ops/Ops.hpp not present")
def test_fp16_dispatch_table_matches_the_runtime_switch():
    """shader_map._FP16_DTYPE_DRIVEN is the converter's copy of the runtime's
    `op_shader_fp16` case list, in ONNX op_type spellings.

    The domain pass (onnx2vkop/fp16_graph.py) decides an op's precision from
    this list, and the shipped converter cannot read C++ -- so a name that
    entered the switch on one side only would hand an FP16 artifact a node
    whose build comes from --precision instead of from its tensors. That is
    not a slow model, it is a wrong one, and nothing at run time reports it.
    """
    driven_enums = _parse_fp16_dispatch_switch(RUNTIME_CPP)
    name_to_enum = _parse_opstring_to_enum(OPS_HPP)
    by_enum: Dict[str, set] = {}
    for name, enum in name_to_enum.items():
        by_enum.setdefault(enum, set()).add(name)
    expected = {n for e in driven_enums for n in by_enum.get(e, ())}
    # A vkop-internal spelling C++ never accepts joins the group through the
    # shader it shares with an accepted alias ("Maxpool2d" vs "MaxPool2d").
    for name, spec in shader_map._PRIMARY.items():
        if name in name_to_enum or name in expected:
            continue
        if any(o in expected and shader_map._PRIMARY[o][1] == spec[1]
               for o in shader_map._PRIMARY):
            expected.add(name)
    assert expected == shader_map._FP16_DTYPE_DRIVEN, (
        f"runtime switch says {sorted(expected - shader_map._FP16_DTYPE_DRIVEN)} "
        f"are dtype-driven and the table misses them; "
        f"{sorted(shader_map._FP16_DTYPE_DRIVEN - expected)} are in the table "
        "but the switch dispatches them from precision_"
    )


def test_dtype_driven_also_requires_a_real_fp16_build():
    """An op in the switch with no dual build asks for a key that resolves to
    the fp32 shader, so `fp16_dtype_driven` has to say no anyway.

    TopK is exactly that: in the switch, keyed on para_.fp16 internally, but
    with a single buffer build. It must not be able to pull a graph into the
    FP16 domain.
    """
    assert "TopK" in shader_map._FP16_DTYPE_DRIVEN
    assert not shader_map.fp16_dtype_driven("TopK")
    assert shader_map.plan_for_op("TopK")["buffer_fp16"] == \
        shader_map.plan_for_op("TopK")["buffer"]
    for op in ("Conv", "MatMul", "GlobalAveragePool", "MaxPool", "Add"):
        assert shader_map.fp16_dtype_driven(op), op
    # An op the switch does not even list cannot be in the domain either.
    for op in ("Gemm", "BatchNormalization", "AveragePool", "Where", "NotAnOp"):
        assert not shader_map.fp16_dtype_driven(op), op


def test_host_only_ops_have_no_shader_at_all():
    """fp16_graph treats them as dtype-neutral, which is only safe because
    they copy data without computing on it -- no key, no build, no choice."""
    for op in shader_map.HOST_ONLY_OPS:
        assert shader_map.plan_for_op(op) is None, op
        assert not shader_map.fp16_dtype_driven(op), op
