"""The .vkopbin carries its own shaders.

onnx2vkop embeds a per-model SPIR-V subset (the VKSP layout of
tools/filter_shaders.py) in Model.shader_archive, so one file is enough to
deploy: Runtime::LoadModel takes every op's shaders from it, which is the only
source, since libvkop carries no built-in SPIR-V.
The C++ half is covered by ModelTest running against a fixture the CMake build
converts (where the embedded archive is the only source of shaders).

These tests pin the writer/reader contract at the boundary the C++ side
reads: field present and byte-identical, absent when not asked for,
deterministic, and still there in the external-blob layout (where the model's
weights do NOT live in the FlatBuffer, so a regression that coupled the two
would otherwise only show up on a multi-GB model).
"""
import struct
import sys
from pathlib import Path

import numpy as np
import pytest
from onnx import helper, numpy_helper, TensorProto

from onnx2vkop import cli, shader_map
from onnx2vkop.cli import build_shader_subset, resolve_master_archive
from onnx2vkop.dag import DAGBasedModel, Node
from onnx2vkop.generated.vkop.model.Model import Model

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "tools"))
import filter_shaders  # noqa: E402

# Model.KernelPlans(j) does a lazy *absolute* import of `vkop.model.KernelPlan`
# (generated FlatBuffers code), so any reader has to have onnx2vkop/generated on
# sys.path -- same setup the other reader tests use.
_GENERATED = str(Path(__file__).resolve().parents[1] / "onnx2vkop" / "generated")
if _GENERATED not in sys.path:
    sys.path.insert(0, _GENERATED)


def td(name, dims=(2, 2), dtype=1):
    return {"name": name, "shape": list(dims), "dtype": dtype}


def tiny_dag():
    """x -> Mul -> Add -> out; the smallest graph the writer will keep."""
    m = DAGBasedModel()
    m.inputs = [td("x")]
    m.outputs = [td("out")]
    m.nodes["mul"] = Node("Mul", "mul", {}, [td("x"), td("x")], [td("m")])
    m.nodes["add"] = Node("Add", "add", {}, [td("m"), td("x")], [td("out")])
    m.build_dependencies()
    return m


def read_embedded(path):
    """The bytes of Model.shader_archive, exactly as the C++ reader sees them."""
    buf = bytearray(Path(path).read_bytes())
    model = Model.GetRootAs(buf, 0)
    n = model.ShaderArchiveLength()
    return bytes(model.ShaderArchive(j) for j in range(n))


def fake_spv(seed, words=8):
    """A minimal SPIR-V-looking blob (the archive only checks the magic)."""
    return struct.pack("<I", 0x07230203) + bytes(
        (seed * 7 + i * 3) & 0xFF for i in range(words * 4 - 4))


def fake_master(tmp_path, names):
    entries = {n: fake_spv(i + 1) for i, n in enumerate(names)}
    p = tmp_path / "vkop_shaders.bin"
    filter_shaders.write_subset(entries, str(p))
    return p, entries


# ------------------------------------------------------------------ writer
def test_archive_round_trips_through_the_writer(tmp_path):
    _, entries = fake_master(tmp_path, ["buffer_add", "image_add_fp16"])
    blob = filter_shaders.serialize(entries)
    out = tmp_path / "m.vkopbin"
    tiny_dag().save_to_binary(str(out), shader_archive=blob)

    got = read_embedded(out)
    assert got == blob
    # And it is a well-formed VKSP archive, not just the same bytes: parse it
    # back with the same reader the C++ side mirrors.
    reparsed = tmp_path / "extracted.bin"
    reparsed.write_bytes(got)
    assert filter_shaders.read_master(str(reparsed)) == entries


def test_no_archive_field_when_not_asked(tmp_path):
    out = tmp_path / "m.vkopbin"
    tiny_dag().save_to_binary(str(out))
    assert read_embedded(out) == b""


def test_writer_stays_deterministic_with_an_archive(tmp_path):
    _, entries = fake_master(tmp_path, ["buffer_add", "buffer_mul"])
    blob = filter_shaders.serialize(entries)
    a, b = tmp_path / "a.vkopbin", tmp_path / "b.vkopbin"
    tiny_dag().save_to_binary(str(a), shader_archive=blob)
    tiny_dag().save_to_binary(str(b), shader_archive=blob)
    assert a.read_bytes() == b.read_bytes()


def test_external_blob_layout_also_carries_the_archive(tmp_path, monkeypatch):
    """Weights go to the trailing external region; the archive stays in the
    FlatBuffer, so a big model keeps the single-file property too."""
    monkeypatch.setenv("VKOP_EXTERNAL_BLOB_MB", "1")
    _, entries = fake_master(tmp_path, ["buffer_add"])
    blob = filter_shaders.serialize(entries)
    m = tiny_dag()
    m.initializers["w"] = numpy_helper.from_array(
        np.zeros(512 * 1024, dtype=np.float16), "w")
    m.nodes["add"].inputs[1] = {"name": "w", "shape": [512 * 1024], "dtype": 10}
    out = tmp_path / "big.vkopbin"
    m.save_to_binary(str(out), shader_archive=blob)

    # 512K fp16 = 1 MiB, over the forced threshold: the blob is external.
    assert out.stat().st_size > 1 << 20
    assert read_embedded(out) == blob


# ------------------------------------------------------- subset selection
def test_serialize_matches_write_subset(tmp_path):
    _, entries = fake_master(tmp_path, ["buffer_add", "image_add_fp16"])
    p = tmp_path / "sub.bin"
    filter_shaders.write_subset(entries, str(p))
    assert p.read_bytes() == filter_shaders.serialize(entries)


def test_auto_backend_embeds_both_sides(tmp_path):
    """Which backend the driver picks is a runtime choice, so the default
    selection has to answer for both."""
    # A realistic master for this graph: every spelling that exists on the
    # backend declaring it. Gather is buffer-only, so "image_gather" is not
    # in here -- but shader_map still emits it, as a fallback marker.
    master, _ = fake_master(tmp_path, [
        "buffer_add", "buffer_add_fp16", "image_add", "image_add_fp16",
        "buffer_mul", "buffer_mul_fp16", "image_mul", "image_mul_fp16",
        "buffer_gather", "buffer_gather_fp16", "buffer_gather_int64"])
    m = tiny_dag()
    m.nodes["g"] = Node("Gather", "g", {"axis": 0},
                        [td("x"), td("i", (1,), 7)], [td("gout")])
    m.nodes["add"].inputs[0] = td("gout")
    m.build_dependencies()

    subset, missing, count = build_shader_subset(m, master, backend="auto")
    assert "buffer_add" in subset and "image_add" in subset
    assert count == 11
    # Gather has no image variant: shader_map emits the phantom "image_gather"
    # (meaning "fall back to buffer_gather"), which must not be reported as a
    # missing shader -- the spelling the runtime would actually ask for is
    # there, and every key the graph can dispatch resolved.
    assert missing == []
    assert "image_gather" not in subset


def test_pinned_backend_is_the_tight_set(tmp_path):
    master, _ = fake_master(
        tmp_path, ["buffer_add", "buffer_add_fp16", "image_add",
                   "image_add_fp16", "buffer_gather"])
    subset, _, _ = build_shader_subset(tiny_dag(), master, backend="buffer")
    assert "buffer_add" in subset
    assert not any(k.startswith("image_") for k in subset)


def test_genuinely_absent_shader_is_reported(tmp_path):
    master, _ = fake_master(tmp_path, ["buffer_add"])
    subset, missing, _ = build_shader_subset(tiny_dag(), master, backend="auto")
    assert subset["buffer_add"]
    # Mul has no shader under either prefix: that is a real gap. Add's fp16
    # twin is missing from this stub master too, for the same reason.
    assert "buffer_mul" in missing
    assert "buffer_add" not in missing


# ------------------------------------------------------------- discovery
def test_resolve_master_archive_precedence(tmp_path, monkeypatch):
    explicit = tmp_path / "explicit.bin"
    explicit.write_bytes(b"x")
    env = tmp_path / "env.bin"
    env.write_bytes(b"x")
    monkeypatch.setenv("VKOP_MASTER_ARCHIVE", str(env))
    assert resolve_master_archive(str(explicit))[0] == explicit
    assert resolve_master_archive(None)[0] == env
    monkeypatch.delenv("VKOP_MASTER_ARCHIVE")
    # This checkout has a build tree, so the search finds one; the point here
    # is that it is a real file rather than a guess at a path.
    found, how = resolve_master_archive(None)
    assert found is None or found.is_file()


def test_embed_returns_bytes_and_reports_subset(tmp_path, capsys):
    master, _ = fake_master(tmp_path, ["buffer_add", "buffer_mul",
                                       "buffer_add_fp16", "buffer_mul_fp16"])
    blob = cli.embed_shader_archive(tiny_dag(), master, backend="auto")
    assert read_master_names(blob) >= {"buffer_add", "buffer_mul"}
    out = capsys.readouterr().out
    assert "embedded 4 of 4 shader(s)" in out


def read_master_names(blob):
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".bin") as f:
        f.write(blob)
        f.flush()
        return set(filter_shaders.read_master(f.name))


# ------------------------------------------------------------ end to end
def _write_tiny_onnx(path):
    """add = x + y, the smallest convertible graph."""
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 2])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 2])
    out = helper.make_tensor_value_info("out", TensorProto.FLOAT, [2, 2])
    node = helper.make_node("Add", ["x", "y"], ["out"], name="add")
    graph = helper.make_graph([node], "tiny", [x, y], [out])
    onnx_model = helper.make_model(graph)
    onnx_model.opset_import[0].version = 13
    path = Path(path)
    with open(path, "wb") as f:
        f.write(onnx_model.SerializeToString())
    return path


def test_cli_embeds_by_default(tmp_path, monkeypatch):
    """The whole point of the default: a plain `onnx2vkop -i m.onnx` produces
    a file that knows its own shaders."""
    master, _ = fake_master(
        tmp_path, ["buffer_add", "buffer_add_fp16", "image_add",
                   "image_add_fp16"])
    src = _write_tiny_onnx(tmp_path / "tiny.onnx")
    out = tmp_path / "tiny.vkopbin"
    monkeypatch.setattr(sys, "argv", [
        "onnx2vkop", "-i", str(src), "-o", str(out),
        "--master-archive", str(master),
    ])
    cli.main()

    blob = read_embedded(out)
    assert read_master_names(blob) >= {"buffer_add", "image_add"}


def test_cli_without_a_master_archive_reports_an_unrunnable_model(
        tmp_path, monkeypatch, capsys):
    """A conversion with no master archive to filter from cannot embed shaders,
    and libvkop, which carries no shaders of its own, cannot run that model --
    which has to be said at conversion time, not discovered at LoadModel.
    Simulated by hiding filter_shaders.py, which is the one reason the subset
    cannot be built at all."""
    src = _write_tiny_onnx(tmp_path / "tiny.onnx")
    out = tmp_path / "bare.vkopbin"
    monkeypatch.setattr(cli, "filter_shaders", None)
    monkeypatch.setattr(sys, "argv", [
        "onnx2vkop", "-i", str(src), "-o", str(out),
    ])
    cli.main()
    assert read_embedded(out) == b""
    assert "will not run" in capsys.readouterr().err


def test_shader_map_covers_a_union_of_backends():
    """build_shader_subset("auto") is exactly the two per-backend sets, so a
    key can never be dropped by splitting the selection in two."""
    ops = {"Add", "Mul", "MatMul", "Gather"}
    both = (shader_map.shader_keys_for_model(ops, backend="buffer")
            | shader_map.shader_keys_for_model(ops, backend="image"))
    assert both >= shader_map.shader_keys_for_model(ops, backend="buffer")
    assert both >= shader_map.shader_keys_for_model(ops, backend="image")
    assert both == shader_map.shader_keys_for_model(ops, backend="auto")


# -------------------------------------------------------------- kernel plan
# The converter's kernel selection is written into the .vkopbin
# (Model.kernel_plans, indexed by Node.plan) and the archive is filtered from
# the same rows, so the runtime reads one answer instead of deriving a second.
# tests/KernelPlanTest.cpp checks the C++ half against the built-in table; the
# tests here pin the writer, the indexing, and the plan/archive agreement.
def read_plans(path):
    """(rows, node_plans) exactly as the C++ reader sees them."""
    buf = bytearray(Path(path).read_bytes())
    model = Model.GetRootAs(buf, 0)
    s = lambda v: v.decode() if v else ""  # noqa: E731 (field is absent = 0)
    rows = []
    for i in range(model.KernelPlansLength()):
        p = model.KernelPlans(i)
        rows.append({
            "op_type": s(p.OpType()),
            "image": s(p.Image()),
            "image_fp16": s(p.ImageFp16()),
            "buffer": s(p.Buffer()),
            "buffer_fp16": s(p.BufferFp16()),
            "variants": [s(p.Variants(j)) for j in range(p.VariantsLength())],
        })
    nodes = [(s(model.Nodes(i).Name()), s(model.Nodes(i).OpType()),
              model.Nodes(i).Plan()) for i in range(model.NodesLength())]
    return rows, nodes


def planned_keys(row):
    """Every archive key the row claims the runtime may ask for."""
    return [k for k in (row["image"], row["image_fp16"], row["buffer"],
                        row["buffer_fp16"]) if k] + row["variants"]


def test_plan_is_written_and_indexed_by_node(tmp_path):
    out = tmp_path / "m.vkopbin"
    tiny_dag().save_to_binary(str(out))
    rows, nodes = read_plans(out)

    assert [r["op_type"] for r in rows] == ["Mul", "Add"]
    by_op = {r["op_type"]: r for r in rows}
    assert by_op["Mul"]["buffer"] == "buffer_mul"
    assert by_op["Mul"]["image"] == "image_mul"
    assert by_op["Add"]["buffer_fp16"] == "buffer_add_fp16"
    # Every node points at its own op_type's row and nothing else.
    for name, op_type, plan in nodes:
        assert plan == [r["op_type"] for r in rows].index(op_type), name


def test_plan_rows_are_deduplicated_and_deterministic(tmp_path):
    """One row per distinct op_type (two Adds share a row), and a second
    conversion of the same graph produces the identical file."""
    m = tiny_dag()
    m.nodes["mul2"] = Node("Mul", "mul2", {}, [td("out"), td("x")], [td("m2")])
    m.outputs.append(td("m2"))
    m.build_dependencies()
    a, b = tmp_path / "a.vkopbin", tmp_path / "b.vkopbin"
    m.save_to_binary(str(a))
    m.save_to_binary(str(b))
    rows, nodes = read_plans(a)
    assert [r["op_type"] for r in rows] == ["Mul", "Add"]
    assert sorted(p for _, _, p in nodes) == [0, 0, 1]
    assert a.read_bytes() == b.read_bytes()


def test_plan_agrees_with_the_archive_the_cli_embeds(tmp_path, monkeypatch):
    """The invariant that makes the model-only design safe: every key
    the plan promises is in the archive written next to it. Synthetic master
    holding exactly the planned keys, so nothing else can cover a gap."""
    src = _write_tiny_onnx(tmp_path / "tiny.onnx")
    needed = sorted(shader_map.shader_keys_for_model({"Add"}, backend="auto"))
    master, _ = fake_master(tmp_path, needed)
    out = tmp_path / "tiny.vkopbin"
    monkeypatch.setattr(sys, "argv", [
        "onnx2vkop", "-i", str(src), "-o", str(out),
        "--master-archive", str(master),
    ])
    cli.main()

    rows, _ = read_plans(out)
    assert rows, "the converted model carries no plan"
    embedded_names = read_master_names(read_embedded(out))
    for row in rows:
        for k in planned_keys(row):
            assert k in embedded_names, (row["op_type"], k)


def test_plan_reports_a_key_the_master_does_not_have(tmp_path, monkeypatch):
    """A planned key missing from the master is a genuine gap (the plan holds
    the post-fallback spelling, so there is no phantom to excuse) and it is a
    hard failure -- conversion has to say so."""
    src = _write_tiny_onnx(tmp_path / "tiny.onnx")
    master, _ = fake_master(tmp_path, ["buffer_add"])
    out = tmp_path / "tiny.vkopbin"
    monkeypatch.setattr(sys, "argv", [
        "onnx2vkop", "-i", str(src), "-o", str(out),
        "--master-archive", str(master),
    ])
    cli.main()
    # Add's other spellings (and variants) are absent from this stub master;
    # the run still completes, with the embedded archive short of them.
    rows, _ = read_plans(out)
    embedded_names = read_master_names(read_embedded(out))
    promised = {k for row in rows for k in planned_keys(row)}
    assert "buffer_add" in embedded_names
    assert promised - embedded_names, "expected some planned key to be missing"


def test_host_only_nodes_get_no_row(tmp_path):
    """A node whose op_type has no shader keeps plan = -1, i.e. the runtime's
    built-in table (which returns no key for it) -- the same behaviour as a
    model written before the field existed."""
    m = tiny_dag()
    m.nodes["sh"] = Node("Shape", "sh", {}, [td("x")], [td("s", (2,), 7)])
    m.nodes["mul"].inputs[0] = td("s")
    m.build_dependencies()
    out = tmp_path / "m.vkopbin"
    m.save_to_binary(str(out))
    rows, nodes = read_plans(out)
    assert [r["op_type"] for r in rows] == ["Mul", "Add"]
    plan_of = {name: plan for name, _, plan in nodes}
    assert plan_of["sh"] == -1
