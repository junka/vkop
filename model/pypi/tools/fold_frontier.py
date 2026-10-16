#!/usr/bin/env python
"""Which op types sit on all-constant inputs but are NOT in ConstantFolder._FOLDABLE_OPS?

Constant folding pays off only at the frontier: a node whose inputs are all
known constants and whose op has no numpy implementation yet. If that set is
empty, adding ops to _FOLDABLE_OPS folds exactly zero additional nodes on this
model -- so measure before writing the evaluator.

    cd model/pypi
    python tools/fold_frontier.py -i ../../llm/exporter/llm.onnx

Reaches the same fixed point ConstantFolder.fold() does (small initializers +
Constant nodes as seeds, Shape unlocked by fully-known shapes), but only tracks
name availability instead of computing values, so it runs on multi-GB graphs.
"""
import argparse
import os
import sys
from collections import Counter

import onnx
from onnx import shape_inference

PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PKG_ROOT)

from onnx2vkop.optimizer import ConstantFolder  # noqa: E402

# fold() strips raw_data above this cap to keep the proto under the 2 GB
# serialization limit; those initializers are not usable as fold inputs.
_SMALL_INIT = 1 << 20


def census(model):
    """Return (foldable_reached, blocked) where blocked maps op_type -> count."""
    known = {i.name for i in model.graph.initializer
             if len(i.raw_data) <= _SMALL_INIT}
    for n in model.graph.node:
        if n.op_type == "Constant":
            known.add(n.output[0])

    shape_known = set()
    for v in list(model.graph.value_info) + list(model.graph.input) \
             + list(model.graph.output):
        dims = v.type.tensor_type.shape.dim
        if dims and all(d.HasField("dim_value") and d.dim_value > 0 for d in dims):
            shape_known.add(v.name)

    foldable = ConstantFolder._FOLDABLE_OPS
    reached = 0
    blocked = Counter()
    example = {}
    for _ in range(30):
        changed = False
        for node in model.graph.node:
            if not node.output or node.output[0] in known:
                continue
            if node.op_type == "Shape":
                ready = bool(node.input[0]) and node.input[0] in shape_known
            else:
                ready = all(i in known for i in node.input if i)
            if not ready:
                continue
            # Unblock downstream either way: a folded node yields its value, a
            # blocked node is treated as available so its consumers are judged
            # on their own op_type rather than being masked by this one.
            known.add(node.output[0])
            if node.op_type in foldable:
                reached += 1
            else:
                blocked[node.op_type] += 1
                example.setdefault(node.op_type, node.name)
            changed = True
        if not changed:
            break
    return reached, blocked, example


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True)
    args = ap.parse_args()

    model = onnx.load(args.input, load_external_data=False)
    try:
        model = shape_inference.infer_shapes(model, data_prop=True)
    except Exception as e:
        print("shape_inference failed (%s): Shape-op folding will under-report" % e)

    reached, blocked, example = census(model)
    print("nodes=%d  foldable_now=%d  blocked_at_frontier=%d" % (
        len(model.graph.node), reached, sum(blocked.values())))
    if not blocked:
        print("frontier empty: extending _FOLDABLE_OPS folds 0 more nodes here")
    else:
        print("extending _FOLDABLE_OPS with these would fold more:")
        for op, cnt in blocked.most_common():
            print("  %-22s %4d  e.g. %s" % (op, cnt, example[op]))


if __name__ == "__main__":
    main()
