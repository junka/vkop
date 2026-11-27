"""Create a dynamic branched graph for ActivationReuseCheck.

Usage: PYTHONPATH=model/pypi python tests/make_activation_reuse_fixture.py \
       <output.vkopbin> <vkop_shaders.bin>
"""

import sys
from pathlib import Path

import numpy as np
from onnx import TensorProto, numpy_helper

from onnx2vkop.cli import embed_shader_archive
from onnx2vkop.dag import DAGBasedModel, Node


def ref(name, shape, dtype=TensorProto.FLOAT):
    return {"name": name, "shape": list(shape), "dtype": dtype}


def main(output, shaders):
    shape = [-1, 16]
    r = lambda name: ref(name, shape)
    model = DAGBasedModel()
    nodes = [
        Node("Add", "add0", {}, [r("x"), r("x")], [r("a")]),
        Node("Relu", "relu0", {}, [r("a")], [r("b")]),
        Node("Neg", "neg0", {}, [r("a")], [r("c")]),
        Node("Add", "add1", {}, [r("b"), r("c")], [r("d")]),
        Node("Neg", "neg1", {}, [r("d")], [r("e")]),
        Node("Neg", "neg2", {}, [r("e")], [r("f")]),
        Node("Add", "add2", {}, [r("a"), r("f")], [r("g")]),
        Node("Reshape", "reshape0", {},
             [r("g"), ref("target", [2], TensorProto.INT64)], [r("view")]),
        Node("Add", "add3", {}, [r("view"), r("x")], [r("y")]),
    ]
    for node in nodes:
        model.add_node(node)
    model.inputs = [ref("x", shape, "float32")]
    model.outputs = [ref("y", shape, "float32")]
    target = numpy_helper.from_array(np.asarray([-1, 16], np.int64), "target")
    model.initializers[target.name] = target
    model.build_dependencies()
    keys = set()
    blob = embed_shader_archive(model, Path(shaders), backend="buffer",
                                keys_out=keys)
    model.save_to_binary(output, shader_archive=blob, available_keys=keys or None)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    main(sys.argv[1], sys.argv[2])
