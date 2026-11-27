"""MoE paging slots under --act bf16.

The measured failure: a --act bf16 conversion of a MoE model either died or
left the slot a MatMul reads declared float16 beside bf16 activations, which
the runtime's bf16 gate (core/Bf16Gate.hpp) refuses at load:

  * a bf16 checkpoint (every MoE export: make_moe_variant.py writes the slots
    BFLOAT16) died in restore_paging_slots -- _downcast_bfloat16 widens every
    bf16 tensor to float32 under --act bf16, including the still-external slot,
    whose dtype then described 4-byte elements of a 2-byte slice;
  * an fp16 checkpoint converted, but the bf16 domain pass skipped the slot
    ("cannot be rewritten to bf16 at write time") and wrote it float16.

A slot lays no bytes in the trailer -- the runtime pages it from the
checkpoint slice its row names -- so "a bf16 slot" is a bfloat16 *declaration*
plus the checkpoint's own format in moe_source_dtype. A bf16 checkpoint is then
paged verbatim (the bits a streamed non-MoE bf16 weight gets in the trailer)
and an fp16 one is rounded by core/WeightPager.hpp on the way in.

The graph is the make_moe_variant.py shape in miniature: one group of three
experts in an external checkpoint shard with a non-contiguous stride, slot 0
wired to a MatMul, slots 1-2 held by keep-alive chains whose outputs are graph
outputs, declarations on value_info with vkop_moe_* metadata.
"""
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

from onnx2vkop.generated.vkop.model import Model as _Model
from onnx2vkop.inspect import ModelView

PYPI = str(Path(__file__).resolve().parents[1])
K, N, EXPERTS = 64, 32, 3
SHARD = "ckpt/shard-00001.bin"
SLOT_BYTES = K * N * 2
HEAD = 64       # bytes before the first expert in the shard
STRIDE = SLOT_BYTES + 16


def slot(e):
    return f"moe.L0.gate.e{e}"


def _bf16_exact(a):
    a = np.ascontiguousarray(a, np.float32)
    return (a.view(np.uint32) & np.uint32(0xFFFF0000)).view(np.float32)


def build(d, kind):
    """m.onnx + ckpt shard under `d`; slot payloads are `kind` (bf16/fp16)."""
    rng = np.random.default_rng(7)
    os.makedirs(os.path.join(d, "ckpt"), exist_ok=True)
    shard = bytearray(b"\0" * HEAD)
    dt = T.BFLOAT16 if kind == "bf16" else T.FLOAT16
    slots = []
    for e in range(EXPERTS):
        w = _bf16_exact(rng.standard_normal((K, N)) / 8)
        if kind == "bf16":
            raw = (w.view(np.uint32) >> 16).astype("<u2").tobytes()
        else:
            raw = w.astype("<f2").tobytes()
        assert len(shard) == HEAD + STRIDE * e
        shard += raw + b"\0" * (STRIDE - SLOT_BYTES)
        t = onnx.TensorProto(name=slot(e), data_type=dt, dims=[K, N])
        t.data_location = T.EXTERNAL
        for k, v in (("location", SHARD), ("offset", str(HEAD + STRIDE * e)),
                     ("length", str(SLOT_BYTES))):
            kv = t.external_data.add()
            kv.key, kv.value = k, v
        slots.append(t)
    Path(d, SHARD).write_bytes(bytes(shard))

    wd = rng.standard_normal((K, N)).astype(np.float32) / 8
    nodes = [h.make_node("MatMul", ["x", slot(0)], ["a"]),
             h.make_node("MatMul", ["x", "Wd"], ["b"]),
             h.make_node("Add", ["a", "b"], ["y"])]
    outs = [h.make_tensor_value_info("y", T.FLOAT, [1, 4, N])]
    inits = slots + [nh.from_array(wd, "Wd"),
                     nh.from_array(np.array([1], np.int64), "idx1"),
                     nh.from_array(np.array([0], np.int64), "ax0")]
    # make_moe_variant.py's keep-alive chain: Shape -> Gather(1) -> Reshape
    # [1, in] -> ReduceSum, its output a graph output so the node stays live.
    for e in range(1, EXPERTS):
        s, w = slot(e), f"keep{e}"
        nodes += [h.make_node("Shape", [s], [w + ".shp"]),
                  h.make_node("Gather", [w + ".shp", "idx1"], [w + ".in1"],
                              axis=0),
                  h.make_node("Reshape", [s, w + ".in1"], [w + ".row"]),
                  h.make_node("ReduceSum", [w + ".row", "ax0"], [w],
                              keepdims=0)]
        outs.append(h.make_tensor_value_info(w, T.FLOAT16, None))
    g = h.make_graph(nodes, "moe", [h.make_tensor_value_info(
        "x", T.FLOAT, [1, 4, K])], outs, inits)
    for e in range(EXPERTS):
        vi = h.make_tensor_value_info(slot(e), dt, [K, N])
        for k, v in (("vkop_moe_group", "0.gate"),
                     ("vkop_moe_expert_id", str(e))):
            kv = vi.metadata_props.add()
            kv.key, kv.value = k, v
        g.value_info.append(vi)
    m = h.make_model(g, opset_imports=[h.make_opsetid("", 17)])
    m.ir_version = 8
    onnx.save(m, os.path.join(d, "m.onnx"))
    return os.path.join(d, "m.onnx")


def convert(src, out, *flags):
    env = dict(os.environ)
    env.pop("VKOP_STREAM_WEIGHTS", None)
    r = subprocess.run(
        [sys.executable, "-m", "onnx2vkop.cli", "-i", str(src), "-o",
         str(out), "--stream-min-mib", "0", "--external-blob-mb", "0",
         *flags], cwd=PYPI, capture_output=True, text=True, env=env)
    log = r.stdout + r.stderr
    assert r.returncode == 0, log
    assert "Error" not in log, log
    return log


def rows(path):
    m = _Model.Model.GetRootAs(Path(path).read_bytes(), 0)
    out = {}
    for i in range(m.InitializersLength()):
        e = m.Initializers(i)
        out[e.Name().decode()] = {
            "dtype": e.Dtype().decode(), "size": e.Size(),
            "group": (e.MoeGroup() or b"").decode(),
            "expert": e.MoeExpertId(),
            "source": (e.MoeSource() or b"").decode(),
            "src_off": e.MoeOffset(), "src_size": e.MoeSize(),
            "src_dtype": (e.MoeSourceDtype() or b"").decode()}
    return out


def matmul_inputs(path):
    return [{i["name"]: i.get("dtype") for i in n.get("inputs", [])
             if isinstance(i, dict)}
            for n in ModelView(str(path)).nodes() if n["op_type"] == "MatMul"]


@pytest.mark.parametrize("kind", ["bf16", "fp16"])
def test_wired_slot_is_a_bf16_declaration_under_act_bf16(tmp_path, kind):
    src = build(str(tmp_path), kind)
    log = convert(src, tmp_path / "m.vkopbin", "--act", "bf16")
    assert "cannot be rewritten to bf16" not in log, log
    r = rows(tmp_path / "m.vkopbin")

    # The wired slot is read by a bf16 MatMul, so its buffer is bf16 -- and
    # the row says what the checkpoint holds, which is what the pager keys
    # its verbatim copy (bf16) or its fp16 -> bf16 rounding on.
    row = r[slot(0)]
    assert row["dtype"] == "bfloat16", row
    assert row["src_dtype"] == ("bfloat16" if kind == "bf16" else "float16")
    slot_mm = [m for m in matmul_inputs(tmp_path / "m.vkopbin")
               if slot(0) in m]
    assert slot_mm == [{"x": "bfloat16", slot(0): "bfloat16"}], slot_mm

    # Still a slot: a group/expert row naming its checkpoint slice, with no
    # trailer bytes -- the bf16 declaration must not transcribe the pool.
    for e in range(EXPERTS):
        row = r[slot(e)]
        assert (row["group"], row["expert"]) == ("0.gate", e), row
        assert row["size"] == 0, row
        assert row["source"] == SHARD
        assert (row["src_off"], row["src_size"]) == (HEAD + STRIDE * e,
                                                     SLOT_BYTES)
        assert row["src_dtype"] == ("bfloat16" if kind == "bf16"
                                    else "float16"), row
        # A keep-alive slot is read by fp16 nodes only, so it keeps fp16.
        assert row["dtype"] == ("bfloat16" if e == 0 else "float16"), row
    view = ModelView(str(tmp_path / "m.vkopbin"))
    blob = (Path(tmp_path / "m.vkopbin").stat().st_size
            - view.external_blob_trailer - 8)
    assert blob == K * N * 2, "the trailer holds Wd only, no expert bytes"


def test_act_fp16_slot_rows_are_unchanged(tmp_path):
    """--act fp16 keeps the float16 slot rows it always wrote."""
    src = build(str(tmp_path), "fp16")
    log = convert(src, tmp_path / "m.vkopbin")
    r = rows(tmp_path / "m.vkopbin")
    for e in range(EXPERTS):
        assert r[slot(e)]["dtype"] == "float16", r[slot(e)]
        assert r[slot(e)]["src_dtype"] == "float16"
        assert r[slot(e)]["size"] == 0
    assert "bfloat16" not in {m.get(slot(0)) for m in
                              matmul_inputs(tmp_path / "m.vkopbin")}, log


def test_act_fp16_bf16_checkpoint_names_its_real_dtype(tmp_path):
    """--act fp16 on a bf16 checkpoint: the slot is declared float16 but its
    bytes on disk are bf16. moe_source_dtype must say so, or the pager copies
    bf16 bits into the fp16 slot verbatim (it converts bf16 -> fp16 when told)."""
    src = build(str(tmp_path), "bf16")
    convert(src, tmp_path / "m.vkopbin")
    r = rows(tmp_path / "m.vkopbin")
    for e in range(EXPERTS):
        assert r[slot(e)]["dtype"] == "float16", r[slot(e)]
        assert r[slot(e)]["src_dtype"] == "bfloat16", r[slot(e)]
        assert r[slot(e)]["size"] == 0
