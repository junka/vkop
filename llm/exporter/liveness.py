#!/usr/bin/env python3
"""Backward liveness analysis on llm.vkopbin.

The runtime skips UNKNOWN ops (Constant, Shape, Unsqueeze, Cast, Squeeze,
ScatterND) but never registers their outputs — so a *live* consumer of one of
those outputs asserts. This script traces from the model outputs backward
through the graph and reports, for each unsupported op type, how many nodes are
actually live (must be handled) vs dead leftover from constant folding (can be
pruned).

Supported op names mirror the runtime's convert_opstring_to_enum + factory.
"""
import sys
from collections import Counter, defaultdict, deque

import sys
sys.path.insert(0, "/tmp/vkmod")
from vkop.model import Model

SUPPORTED = {
    "Add", "Atan", "AveragePool", "BatchNormalization", "BatchNorm", "Col2Im",
    "Concat", "Conv", "Conv2d", "Div", "EmbeddingForward", "Erf", "Floor",
    "Gemm", "GlobalAveragePool", "GridSample", "LayerNormalization", "LayerNorm",
    "MatMul", "MaxPool", "MaxPool2d", "Mul", "Pow", "PRelu", "Reduce", "Relu",
    "Reshape", "Resize", "Sigmoid", "Slice", "Softmax", "Softplus", "Split",
    "Sub", "TopK", "Transpose", "Nms", "Gather", "Range", "Expand", "Sqrt",
    "Sin", "Cos", "Neg", "Where", "Tanh", "Equal", "NonZero", "ScatterElements",
}


def load_model(path):
    with open(path, "rb") as f:
        data = f.read()
    return Model.Model.GetRootAsModel(data, 0)


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "llm/exporter/llm.vkopbin"
    model = load_model(path)

    nodes = [model.Nodes(i) for i in range(model.NodesLength())]

    # producer: tensor name -> node index
    # consumers: tensor name -> list of node indices
    producer = {}
    consumers = defaultdict(list)
    for i, n in enumerate(nodes):
        for j in range(n.OutputsLength()):
            producer[n.Outputs(j).Name().decode()] = i
        for j in range(n.InputsLength()):
            consumers[n.Inputs(j).Name().decode()].append(i)

    # seed with model outputs
    worklist = deque()
    for i in range(model.OutputsLength()):
        worklist.append(model.Outputs(i).Name().decode())

    live_nodes = [False] * len(nodes)
    seen_tensors = set()
    while worklist:
        tname = worklist.popleft()
        if tname in seen_tensors:
            continue
        seen_tensors.add(tname)
        i = producer.get(tname)
        if i is None or live_nodes[i]:
            continue
        live_nodes[i] = True
        n = nodes[i]
        for j in range(n.InputsLength()):
            worklist.append(n.Inputs(j).Name().decode())

    total = Counter()
    live = Counter()
    for i, n in enumerate(nodes):
        op = n.OpType().decode()
        total[op] += 1
        if live_nodes[i]:
            live[op] += 1

    print(f"{'op':<20}{'total':>7}{'live':>7}")
    print("-" * 34)
    for op, c in total.most_common():
        mark = "  <-- MISSING, live!" if op not in SUPPORTED and live[op] else ""
        print(f"{op:<20}{c:>7}{live[op]:>7}{mark}")

    unsupported_live = {op: live[op] for op in total if op not in SUPPORTED and live[op]}
    unsupported_dead = {op: total[op] - live[op] for op in total if op not in SUPPORTED}
    print(f"\nLive unsupported nodes: {sum(unsupported_live.values())} "
          f"/ {sum(total[op] for op in total if op not in SUPPORTED)} total")
    print(f"Dead unsupported (prunable): {sum(unsupported_dead.values())}")

    # detail: live unsupported nodes by name
    print("\n=== Live unsupported node names ===")
    for i, n in enumerate(nodes):
        op = n.OpType().decode()
        if op not in SUPPORTED and live_nodes[i]:
            nm = n.Name().decode()
            outs = [n.Outputs(j).Name().decode() for j in range(n.OutputsLength())]
            print(f"  [{op}] {nm} -> {outs}")
    print(f"\ntotal live unsupported listed above: {sum(unsupported_live.values())}")


if __name__ == "__main__":
    main()
