#!/usr/bin/env python3
"""Cut an HF-style decoder ONNX down to its first N layers, by backward cone.

For per-node accuracy checks at LLM scale without paying for the whole model
(onnx2vkop --emit-ort-onnx needs every weight in RAM, and the ORT reference
runs in fp32):

    python3 tools/trim_llm_prefix.py --src llm/exporter/llm.onnx --keep 2 --out /tmp/l2/llm2.onnx
    onnx2vkop -i /tmp/l2/llm2.onnx -o /tmp/l2/llm2.vkopbin --emit-ort-onnx /tmp/l2/llm2.ort.onnx ...
    python3 tools/vkop_vs_ort.py /tmp/l2/llm2.vkopbin --onnx /tmp/l2/llm2.ort.onnx --inputs DIR --node-dump

The new graph outputs are `present_key_values_0..N-1` plus the normalized
hidden state entering layer N (the Mul applying `*layers.N.input_layernorm.
weight`, which depends on all of layer N-1's residual). Everything else is the
backward cone of those tensors: nodes, initializers and graph inputs nothing
reaches are dropped. Unlike trim_llm_layers.py (Qwen3.5-specific: it rebuilds
the final norm + lm_head), this needs no knowledge of the block structure, so
it works on any export whose layers are named `...layers.<i>.input_layernorm.
weight` (Qwen3-VL, GLM-Edge, Phi-4, ...).

Weights are not copied: the output keeps each initializer's external-data
offset, and a symlink to the source's weights file is placed next to it.
"""
import argparse
import os
import sys

import onnx
from onnx import helper


def trim(src: str, out: str, keep: int) -> dict:
    m = onnx.load(src, load_external_data=False)
    g = m.graph
    suffix = f"layers.{keep}.input_layernorm.weight"
    norm_w = [t.name for t in g.initializer if t.name.endswith(suffix)]
    if len(norm_w) != 1:
        raise SystemExit(f"expected one initializer ending in {suffix!r}, "
                         f"found {norm_w} (is --keep past the last layer?)")
    hid = [n.output[0] for n in g.node
           if n.op_type == "Mul" and norm_w[0] in n.input]
    if len(hid) != 1:
        raise SystemExit(f"expected one Mul applying {norm_w[0]}, found {hid}")
    have = {o.name for o in g.output}
    outs = [f"present_key_values_{i}" for i in range(keep)]
    missing = [o for o in outs if o not in have]
    if missing:
        raise SystemExit(f"graph has no outputs {missing}")
    outs += hid

    prod = {o: n for n in g.node for o in n.output}
    need, stack, kept = set(), list(outs), set()
    while stack:
        t = stack.pop()
        if t in need:
            continue
        need.add(t)
        n = prod.get(t)
        if n is not None and id(n) not in kept:
            kept.add(id(n))
            stack.extend(i for i in n.input if i)
    nodes = [n for n in g.node if id(n) in kept]
    inits = [t for t in g.initializer if t.name in need]
    inputs = [i for i in g.input if i.name in need]
    vinfo = {v.name: v for v in list(g.value_info) + list(g.output)}
    outputs = [vinfo.get(o) or helper.make_tensor_value_info(
        o, onnx.TensorProto.FLOAT16, None) for o in outs]
    ng = helper.make_graph(nodes, g.name, inputs, outputs, initializer=inits,
                           value_info=[v for v in g.value_info if v.name in need])
    nm = helper.make_model(ng, opset_imports=m.opset_import)
    nm.ir_version = m.ir_version
    for p in m.metadata_props:
        nm.metadata_props.add(key=p.key, value=p.value)

    out_dir = os.path.dirname(os.path.abspath(out))
    os.makedirs(out_dir, exist_ok=True)
    src_dir = os.path.dirname(os.path.abspath(src))
    for loc in {kv.value for t in inits for kv in t.external_data
                if kv.key == "location"}:
        link = os.path.join(out_dir, loc)
        target = os.path.join(src_dir, loc)
        if os.path.abspath(link) != os.path.abspath(target) and \
                not os.path.lexists(link):
            os.symlink(target, link)
    onnx.save(nm, out)
    weight_bytes = sum(int(kv.value) for t in inits for kv in t.external_data
                       if kv.key == "length")
    return {"nodes": len(nodes), "initializers": len(inits),
            "weight_bytes": weight_bytes, "inputs": [i.name for i in inputs],
            "outputs": outs}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--keep", type=int, required=True,
                    help="decoder layers to keep, counted from 0")
    a = ap.parse_args(argv)
    s = trim(a.src, a.out, a.keep)
    print(f"{s['nodes']} nodes, {s['initializers']} initializers "
          f"({s['weight_bytes'] / 1e6:.0f} MB of weights)")
    print(f"inputs:  {s['inputs']}")
    print(f"outputs: {s['outputs']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
