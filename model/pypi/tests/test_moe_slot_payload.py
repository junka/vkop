"""A paging slot's bytes stay in the checkpoint, never in the trailer.

The measured failure this pins down: writing the Qwen1.5-MoE model transcribed
all 4320 slots into the vkopbin trailer -- 23.6 GiB of expert bytes -- and the
conversion died with ENOSPC while the runtime would have read the checkpoint
anyway. A slot row is a *declaration* (name, dtype, dims, and the
moe_source/moe_offset/moe_size slice to pread from), not a payload.
"""
import numpy as np
import onnx
import os
import pytest
from onnx import TensorProto, helper
import sys

from onnx2vkop import weight_stream
from onnx2vkop.dag import DAGBasedModel, Node

# The generated readers import each other as `vkop.model.*`, so the generated
# package dir is the import root for anything that parses a file back.
sys.path.insert(0, str(__import__("pathlib").Path(
    __import__("onnx2vkop").__file__).parent / "generated"))
from vkop.model.Model import Model  # noqa: E402


def td(name, shape):
    return {"name": name, "dims": list(shape), "dtype": ""}


def build(experts=4, group="L0.gate_proj", dense=2):
    """A graph whose single MatMul consumes slot 0, plus `dense` ordinary
    weights read by the same node so the blob is never empty."""
    dm = DAGBasedModel()
    a = {"transA": 0, "transB": 0, "alpha": 1.0, "beta": 0.0}
    slot0 = "moe.%s.e0" % group
    inputs = [td("x", [1, 4]), td(slot0, [4, 4])]
    inputs += [td("d%d" % i, [4, 4]) for i in range(dense)]
    outs = [td("y", [1, 4])] + [td("y%d" % i, [1, 4]) for i in range(dense)]
    dm.nodes["mm0"] = Node("MatMul", "mm0", a, inputs[:2], [td("t0", [1, 4])])
    for i in range(dense):
        dm.nodes["mm%d" % (i + 1)] = Node(
            "MatMul", "mm%d" % (i + 1), a,
            [inputs[0], td("d%d" % i, [4, 4])], [outs[i + 1]])
    dm.nodes["add"] = Node("Add", "add", {}, [td("t0", [1, 4]), td("y1", [1, 4])],
                           outs[:1])
    dm.inputs = [td("x", [1, 4])]
    dm.outputs = outs[:1]

    slots, srcs = {}, {}
    for e in range(experts):
        name = "moe.%s.e%d" % (group, e)
        dm.initializers[name] = helper.make_tensor(
            name, TensorProto.FLOAT16, [4, 4], np.full(16, e, np.float16))
        slots[name] = (group, e)
        # Deliberately not contiguous: a 40-byte stride for a 32-byte tensor is
        # exactly the checkpoint's shape (name-sorted projections), so the
        # writer must not assume arithmetic addressing.
        srcs[name] = ("ckpt/model-00001.safetensors", 100 + 40 * e, 32,
                      TensorProto.FLOAT16)
    for i in range(dense):
        dm.initializers["d%d" % i] = helper.make_tensor(
            "d%d" % i, TensorProto.FLOAT16, [4, 4], np.full(16, i + 2, np.float16))
    dm.moe_slots = slots
    dm.moe_sources = srcs
    dm.moe_keep_alive = set(slots)
    dm.deferred_weights = {}
    dm.weight_base_dir = "."
    return dm, slots, srcs


def write(dm, path):
    # Small test models inline the blob into the FlatBuffer by default, and the
    # trailer is where a slot would have been transcribed. Force it, since the
    # 800 MiB default is the multi-GB case this contract exists for.
    os.environ["VKOP_EXTERNAL_BLOB_MB"] = "0"
    try:
        dm.save_to_binary(path)
    finally:
        del os.environ["VKOP_EXTERNAL_BLOB_MB"]
    return path


def read(path):
    buf = open(path, "rb").read()
    model = Model.GetRootAs(buf, 0)
    rows = {}
    for i in range(model.InitializersLength()):
        e = model.Initializers(i)
        rows[e.Name().decode()] = {
            "dtype": e.Dtype().decode(), "offset": e.Offset(), "size": e.Size(),
            "group": e.MoeGroup().decode() if e.MoeGroup() else "",
            "expert": e.MoeExpertId(),
            "source": e.MoeSource().decode() if e.MoeSource() else "",
            "src_off": e.MoeOffset(), "src_size": e.MoeSize(),
        }
    base = int.from_bytes(buf[-8:], "little")
    return rows, base, len(buf)


def test_slot_lays_down_no_bytes_and_names_its_slice(tmp_path):
    dm, slots, srcs = build()
    path = write(dm, str(tmp_path / "m.vkopbin"))
    rows, base, total = read(path)

    for name, (grp, eid) in slots.items():
        row = rows[name]
        assert row["group"] == grp and row["expert"] == eid
        assert row["size"] == 0, "a slot must not declare trailer bytes"
        assert row["offset"] == 0
        assert row["source"] == srcs[name][0]
        assert row["src_off"] == srcs[name][1]
        assert row["src_size"] == 32
        assert row["dtype"] == "float16"

    # A slot's extent is 0, so the two dense weights sit at the blob's start:
    # both are in the trailer and the four slots add nothing to it.
    dense = [n for n in ("d0", "d1") if n in rows]
    assert len(dense) == 1, list(rows)
    assert rows[dense[0]]["offset"] == 0


def test_all_slot_model_leaves_an_empty_blob(tmp_path):
    dm, slots, srcs = build(dense=0)
    del dm.nodes["add"]
    dm.nodes["mm0"].outputs = [td("y", [1, 4])]
    dm.outputs = [td("y", [1, 4])]
    path = write(dm, str(tmp_path / "m.vkopbin"))
    rows, base, total = read(path)
    assert total - base - 8 == 0, "no dense weight, so the blob must be empty"
    assert len([r for r in rows.values() if r["group"]]) == 4


def test_dense_weight_keeps_its_layout_beside_slots(tmp_path):
    # The slots have to be invisible in the layout, not merely small: each still
    # has an entry in the ordered blob list, and the writer that streams the
    # trailer must let those entries pass without moving the cursor the dense
    # offsets were computed against. If it did not, every dense weight after
    # the first slot would shift by a slot's width.
    dm, slots, srcs = build()
    path = write(dm, str(tmp_path / "m.vkopbin"))
    rows, base, total = read(path)
    buf = open(path, "rb").read()
    names = ["d0", "d1"]
    # One dense weight is unreferenced and pruned, so the other is the blob.
    dense = [n for n in names if n in rows]
    assert len(dense) == 1, list(rows)
    n = dense[0]
    assert total - base - 8 == 32, "the trailer holds the dense weight only"
    assert rows[n]["offset"] == 0
    want = np.full(16, int(n[1:]) + 2, np.float16).tobytes()
    assert buf[base:base + 32] == want


def test_write_refuses_a_gap_in_expert_ids(tmp_path):
    # A missing expert leaves the group's rows at ids 0..2 with 3 absent, which
    # the runtime would read as a shift: expert 3's buffer would hold the bytes
    # the exporter wrote for a different expert, silently.
    dm, slots, srcs = build()
    drop = "moe.L0.gate_proj.e2"
    del dm.moe_slots[drop]
    del dm.moe_sources[drop]
    dm.moe_keep_alive.discard(drop)
    del dm.initializers[drop]
    with pytest.raises(weight_stream.DeferredWeightError) as exc:
        write(dm, str(tmp_path / "m.vkopbin"))
    assert "expected 0.." in str(exc.value)


def test_write_refuses_a_slice_shorter_than_the_slot(tmp_path):
    # The runtime preads `moe_size` bytes into the slot's buffer, so a short
    # slice reads whatever follows the expert in the checkpoint shard.
    dm, slots, srcs = build()
    name = "moe.L0.gate_proj.e1"
    loc, off, _, dt = dm.moe_sources[name]
    dm.moe_sources[name] = (loc, off, 16, dt)
    with pytest.raises(weight_stream.DeferredWeightError) as exc:
        write(dm, str(tmp_path / "m.vkopbin"))
    assert "byte(s) but the tensor is" in str(exc.value)
