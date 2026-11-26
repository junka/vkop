"""Write the bf16 activation-gate fixtures Bf16ActivationTest loads.

Every file is a one-node graph built straight on the DAG and saved with the
real writer (plus the shaders it needs), NOT through the converter: the
converter now keeps exactly these shapes out of the bf16 domain, and the point
here is what Runtime::LoadModel does when a file carries them anyway (an older
converter, a hand-built graph).

  bf16_gate_add.vkopbin     x(bf16) + b(bf16 init) -> y(bf16)    loads and runs
  bf16_gate_relu.vkopbin    Relu(x bf16) -> y bf16                refused
  bf16_gate_cast.vkopbin    Cast(x bf16, to=FLOAT) -> y fp32      refused
  bf16_gate_gather.vkopbin  Gather(table bf16, idx) -> y bf16     refused
  bf16_gate_mixed.vkopbin   x(bf16) + b(fp16 init) -> y(bf16)     refused

usage: make_bf16_gate_fixture.py <output-dir> <master vkop_shaders.bin>
"""
import os
import sys
from pathlib import Path

import numpy as np
from onnx import TensorProto, helper, numpy_helper

from onnx2vkop import bf16_weights
from onnx2vkop.cli import embed_shader_archive
from onnx2vkop.dag import DAGBasedModel, Node

BF16 = TensorProto.BFLOAT16
FP16 = TensorProto.FLOAT16
FP32 = TensorProto.FLOAT
INT64 = TensorProto.INT64
SHAPE = [2, 8]


def ref(name, shape, dtype):
    return {"name": name, "shape": list(shape), "dtype": dtype}


def bf16_init(name, values):
    arr = np.asarray(values, np.float32)
    bits = np.ascontiguousarray(
        bf16_weights.fp32_to_bf16_bits(arr).astype(np.uint16))
    t = numpy_helper.from_array(bits, name)
    t.data_type = BF16
    return t


def bias_values():
    # Exact in bf16 (multiples of 1/8 below 16), so the reference is exact too.
    return (np.arange(16, dtype=np.float32).reshape(SHAPE) - 7.0) / 8.0


def model(node, inputs, outputs, inits):
    dm = DAGBasedModel()
    dm.nodes[node.name] = node
    dm.inputs = inputs
    dm.outputs = outputs
    for t in inits:
        dm.initializers[t.name] = t
    dm.deferred_weights = {}
    dm.weight_base_dir = "."
    return dm


def build_all():
    out = {}
    x = ref("x", SHAPE, BF16)
    y = ref("y", SHAPE, BF16)
    out["bf16_gate_add"] = model(
        Node("Add", "add0", {}, [x, ref("b", SHAPE, BF16)], [y]),
        [ref("x", SHAPE, "bfloat16")], [ref("y", SHAPE, "bfloat16")],
        [bf16_init("b", bias_values())])
    out["bf16_gate_relu"] = model(
        Node("Relu", "relu0", {}, [x], [y]),
        [ref("x", SHAPE, "bfloat16")], [ref("y", SHAPE, "bfloat16")], [])
    out["bf16_gate_cast"] = model(
        Node("Cast", "cast0", {"to": FP32}, [x], [ref("y", SHAPE, FP32)]),
        [ref("x", SHAPE, "bfloat16")], [ref("y", SHAPE, "float32")], [])
    out["bf16_gate_gather"] = model(
        Node("Gather", "gather0", {"axis": 0},
             [ref("table", [4, 8], BF16), ref("idx", [2], INT64)],
             [ref("y", SHAPE, BF16)]),
        [ref("idx", [2], "int64")], [ref("y", SHAPE, "bfloat16")],
        [bf16_init("table", np.arange(32, dtype=np.float32).reshape(4, 8))])
    out["bf16_gate_mixed"] = model(
        Node("Add", "add_mixed", {}, [x, ref("b", SHAPE, FP16)], [y]),
        [ref("x", SHAPE, "bfloat16")], [ref("y", SHAPE, "bfloat16")],
        [helper.make_tensor("b", FP16, SHAPE,
                            bias_values().astype(np.float16).flatten())])
    return out


def main(out_dir, master):
    os.makedirs(out_dir, exist_ok=True)
    for name, dm in build_all().items():
        keys = set()
        blob = embed_shader_archive(dm, Path(master), backend="buffer",
                                    keys_out=keys)
        # The gate is the loader's, so the fixtures must carry the _bf16
        # modules an allow-listed op would resolve (embed_shader_archive adds
        # them beside every buffer _fp16 key).
        dm.save_to_binary(os.path.join(out_dir, name + ".vkopbin"),
                          shader_archive=blob, available_keys=keys or None)
    print("[fixture] bf16 gate fixtures in " + out_dir)
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__, file=sys.stderr)
        sys.exit(2)
    sys.exit(main(sys.argv[1], sys.argv[2]))
