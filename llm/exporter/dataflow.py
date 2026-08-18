#!/usr/bin/env python3
"""Classify int64/int32 tensor dataflow in llm.vkopbin.

For every node, inspect each input slot:
  - if the tensor is produced by a node, what op produced it (dtype of the DATA)
  - if it's an initializer, what dtype string it has
  - whether it's empty-dims (metadata/scalar, e.g. Unsqueeze axes)

Answers the critical question for Stage 2: do existing GPU SSBO ops
(Concat/Mul/Slice/Expand/Gather/Add/Div/Pow/Where/Equal...) ever receive
int64/int32 tensor DATA (not just small int64 initializer scalars)? If so,
their 4-byte-word shaders would misread values.
"""
import sys
from collections import Counter, defaultdict

sys.path.insert(0, "/tmp/vkmod")
from vkop.model import Model


def load_model(path):
    with open(path, "rb") as f:
        data = f.read()
    return Model.Model.GetRootAsModel(data, 0)


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "llm/exporter/llm.vkopbin"
    model = load_model(path)

    nodes = [model.Nodes(i) for i in range(model.NodesLength())]

    # producer: tensor -> node index
    producer = {}
    for i, n in enumerate(nodes):
        for j in range(n.OutputsLength()):
            producer[n.Outputs(j).Name().decode()] = i

    # initializer dtype map
    init_dtype = {}
    for i in range(model.InitializersLength()):
        e = model.Initializers(i)
        init_dtype[e.Name().decode()] = e.Dtype().decode()

    # node op string + rank
    op_of = {i: n.OpType().decode() for i, n in enumerate(nodes)}

    def rank_of(t):
        return t.DimsLength()

    # For each (op, input_slot) where the input is int64/int32 producer or initializer:
    stats = defaultdict(Counter)  # (op, slot) -> Counter(dtype_source)
    examples = defaultdict(list)

    for i, n in enumerate(nodes):
        op = op_of[i]
        for j in range(n.InputsLength()):
            in_t = n.Inputs(j)
            name = in_t.Name().decode()
            r = rank_of(in_t)
            # source classification
            if r == 0:
                continue  # empty dims -> metadata/nullptr path
            if name in init_dtype:
                src = ("init:" + init_dtype[name], name)
            elif name in producer:
                src = ("node:" + op_of[producer[name]], name)
            else:
                continue
            dt, tname = src
            if dt.startswith("init:int") or dt.startswith("node:int"):
                stats[(op, j)][dt] += 1
                if len(examples[(op, j)]) < 2:
                    examples[(op, j)].append(f"{tname} (rank {r})")

    print(f"{'op':<12}{'slot':<5}{'count':<6} int64/int32 sources")
    print("-" * 80)
    for (op, slot), c in sorted(stats.items()):
        parts = ", ".join(f"{k}: {v}" for k, v in c.most_common())
        ex = " | ".join(examples[(op, slot)])
        print(f"{op:<12}{slot:<5}{sum(c.values()):<6}{parts}  e.g. {ex}")

    # Also: any node input whose rank >= 1 is a producer but the producer is one
    # of the unsupported meta ops (Shape/Unsqueeze/Cast) -> downstream dtype.
    print("\n=== int64-producing op breakdown ===")
    prod_count = Counter()
    for i, n in enumerate(nodes):
        op = op_of[i]
        if op in ("Shape",):
            prod_count["Shape"] += 1
        elif op == "Unsqueeze":
            prod_count["Unsqueeze"] += 1
        elif op == "Cast":
            # dtype from attr 'to'
            for a in range(n.AttributesLength()):
                attr = n.Attributes(a)
                if attr.Key().decode() == "to":
                    prod_count["Cast->to" + str(attr.Ival())] += 1
        elif op == "Squeeze":
            prod_count["Squeeze"] += 1
        elif op == "ScatterND":
            prod_count["ScatterND"] += 1
    for k, v in prod_count.most_common():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
