#!/usr/bin/env python3
"""Parse llm.vkopbin and classify the surviving runtime nodes.

The ConstantFolder inside the converter already folded the 4350 shape/pos-embed
chain nodes, so most of what remains are true runtime ops. This script reads the
FlatBuffer directly (no re-conversion) and prints:
  - the runtime node histogram
  - every Cast node's input/output shape + dtype (the mixed fp16/fp32 case)
  - a classification: ops vkop's runtime supports today vs. ones it must gain
"""
import sys
from collections import Counter

import flatbuffers  # noqa

from model.pypi.onnx2vkop.generated.vkop.model import Model
from model.pypi.onnx2vkop.generated.vkop.model import Node as FBNode

# ops the runtime supports (from ops/Ops.hpp + OperatorFactory.hpp)
SUPPORTED = {
    "Add", "Atan", "AveragePool", "BatchNormalization", "BatchNorm", "Col2Im",
    "Concat", "Conv", "Conv2d", "Div", "EmbeddingForward", "Erf", "Floor",
    "Gemm", "GlobalAveragePool", "GridSample", "LayerNormalization", "LayerNorm",
    "MatMul", "MaxPool", "MaxPool2d", "Mul", "Pow", "PRelu", "Reduce", "Relu",
    "Reshape", "Resize", "Sigmoid", "Slice", "Softmax", "Softplus", "Split",
    "Sub", "TopK", "Transpose", "Nms", "Gather", "Range", "Expand", "Sqrt",
    "Sin", "Cos", "Neg", "Where", "Tanh", "Equal", "NonZero", "ScatterElements",
}

# ATTR_TYPE -> name (mirror fbs AttrType enum)
_ATTR_NAMES = {0: "Int64", 1: "Float32", 2: "Bool", 3: "String",
               4: "Ints", 5: "Floats", 6: "Tensor"}


def load_model(path):
    with open(path, "rb") as f:
        data = f.read()
    return Model.Model.GetRootAsModel(data, 0)


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "llm/exporter/llm.vkopbin"
    model = load_model(path)

    nodes = []
    for i in range(model.NodesLength()):
        n = model.Nodes(i)
        nodes.append(n)

    counts = Counter(n.OpType().decode() for n in nodes)
    missing = Counter()
    print(f"{'count':<7}{'op':<24}{'status'}")
    print("-" * 46)
    for op, c in counts.most_common():
        status = "SUPPORTED" if op in SUPPORTED else "MISSING"
        if status == "MISSING":
            missing[op] = c
        print(f"{c:<7}{op:<24}{status}")

    print(f"\nTotal runtime nodes: {len(nodes)}")
    print(f"Distinct ops:        {len(counts)}")
    print(f"Missing ops:         {sum(missing.values())} nodes / {len(missing)} types")

    # --- Cast detail ---
    print("\n=== Cast nodes (shape / dtype) ===")
    for n in nodes:
        if n.OpType().decode() != "Cast":
            continue
        to_attr = None
        for a in range(n.AttributesLength()):
            attr = n.Attributes(a)
            if attr.Key().decode() == "to":
                to_attr = attr
        to = None
        if to_attr is not None:
            if to_attr.Type() == 0:  # Int64
                to = to_attr.Ival()
        def shape_of(t):
            dims = [t.Dims(j) for j in range(t.DimsLength())]
            return dims
        in_s = shape_of(n.Inputs(0)) if n.InputsLength() > 0 else []
        in_dt = None
        out_s = shape_of(n.Outputs(0)) if n.OutputsLength() > 0 else []
        out_dt = None
        # dtype from the node's attribute only (runtime infers dtype from shape/precision)
        dt = {1: "fp32", 6: "i32", 7: "i64", 9: "bool", 10: "fp16", 11: "fp64"}
        print(f"  {n.Name().decode():<58} in={in_s} -> out={out_s} to={to}")

    # --- MatMul shapes (for the KV cache / attention blocks) ---
    print("\n=== MatMul node shapes ===")
    for n in nodes:
        if n.OpType().decode() != "MatMul":
            continue
        def shp(t):
            return [t.Dims(j) for j in range(t.DimsLength())]
        in0 = shp(n.Inputs(0)) if n.InputsLength() > 0 else []
        out = shp(n.Outputs(0)) if n.OutputsLength() > 0 else []
        print(f"  {n.Name().decode():<58} {in0} -> {out}")


if __name__ == "__main__":
    main()
