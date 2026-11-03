"""Write the two MoE slot fixtures MoeSlotLoadTest reads back.

`moe_paged_slots.vkopbin` goes through the real writer: two groups of three
experts declared as paging slots (no trailer bytes, each naming a slice of a
checkpoint shard), plus two ordinary dense weights so the model is a model.
`moe_slots_absent.vkopbin` is the same graph with no slot declarations at all --
what a file written before the MoE fields existed looks like.

The slices are deliberately *not* contiguous (a 40-byte stride for a 32-byte
tensor, as the safetensors shards do), because that is what makes arithmetic
addressing `first + expert_id * size` wrong and the lookup table necessary.
"""
import os
import sys

import numpy as np
from onnx import TensorProto, helper

from onnx2vkop import weight_stream
from onnx2vkop.dag import DAGBasedModel, Node

SRC = "ckpt/shard-00001.safetensors"
GROUPS = ("L0.gate_proj", "L0.up_proj")
EXPERTS = 3
TENSOR_BYTES = 32          # fp16 [4, 4]
STRIDE = 40                # what the checkpoint's name-sorted layout gives


def td(name, shape):
    return {"name": name, "dims": list(shape), "dtype": ""}


def slot_name(group, e):
    return "moe.%s.e%d" % (group, e)


def build(paged):
    dm = DAGBasedModel()
    a = {"transA": 0, "transB": 0, "alpha": 1.0, "beta": 0.0}
    first = slot_name(GROUPS[0], 0)
    # d1 is an ordinary weight read by a live node. Every *other* expert matrix
    # is read by nothing -- a paging slot exists so any expert can be paged into
    # it later, not because this one is live -- so only `d1` is here for
    # liveness; the slot names are kept by the keep-alive set below.
    dm.nodes["mm0"] = Node("MatMul", "mm0", a,
                           [td("x", [1, 4]), td("d1", [4, 4])],
                           [td("t0", [1, 4])])
    dm.nodes["mm1"] = Node("MatMul", "mm1", a,
                           [td("t0", [1, 4]), td(first, [4, 4])],
                           [td("y", [1, 4])])
    dm.inputs = [td("x", [1, 4])]
    dm.outputs = [td("y", [1, 4])]

    dm.initializers["d0"] = helper.make_tensor(
        "d0", TensorProto.FLOAT16, [4, 4], np.full(16, 7, np.float16))
    dm.initializers["d1"] = helper.make_tensor(
        "d1", TensorProto.FLOAT16, [4, 4], np.full(16, 9, np.float16))
    for group in GROUPS:
        for e in range(EXPERTS):
            name = slot_name(group, e)
            dm.initializers[name] = helper.make_tensor(
                name, TensorProto.FLOAT16, [4, 4], np.full(16, e, np.float16))

    dm.moe_slots, dm.moe_sources = {}, {}
    if paged:
        for group in GROUPS:
            for e in range(EXPERTS):
                name = slot_name(group, e)
                dm.moe_slots[name] = (group, e)
                # bf16 in the checkpoint, fp16 in the slot: the runtime converts
                # on the way in, which is why the row carries both dtypes.
                dm.moe_sources[name] = (SRC, 100 + STRIDE * e, TENSOR_BYTES,
                                        TensorProto.BFLOAT16)
    # Same graph, no slot declarations: no moe_keep_alive, no sources, and the
    # writer's ordinary liveness prune applies to every initializer.
    dm.moe_keep_alive = set(dm.moe_slots)
    dm.deferred_weights = {}
    dm.weight_base_dir = "."
    return dm


def write(dm, path):
    # These models are tiny, so the writer would inline the blob into the
    # FlatBuffer; the contract under test lives in the external trailer.
    os.environ["VKOP_EXTERNAL_BLOB_MB"] = "0"
    try:
        dm.save_to_binary(path)
    finally:
        del os.environ["VKOP_EXTERNAL_BLOB_MB"]


def main(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    write(build(paged=True), os.path.join(out_dir, "moe_paged_slots.vkopbin"))
    write(build(paged=False), os.path.join(out_dir, "moe_slots_absent.vkopbin"))
    print("[fixture] moe_paged_slots.vkopbin + moe_slots_absent.vkopbin in "
          + out_dir)
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: make_moe_slots_fixture.py <output-dir>", file=sys.stderr)
        sys.exit(2)
    sys.exit(main(sys.argv[1]))
