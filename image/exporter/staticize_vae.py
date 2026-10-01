#!/usr/bin/env python3
"""Staticize + shape-chain constant folding for the VAE decoder ONNX.

Stage 1: pin graph input/output dims (1,64,1,32,32 → 1,4,1,512,512) and drop
dynamic_axes symbols, so shape inference resolves everything concrete.
Stage 2: run ORT once, capture every small int64/bool intermediate produced by
shape-plumbing ops (Shape/Gather/Concat/Unsqueeze/Slice/ConstantOfShape/Equal/
Tile/Cast/Expand/Squeeze…), replace those nodes with initializers, delete the
now-dead nodes, iterate to fixpoint.

This is the minimum onnx-simplifier equivalent needed to make the graph
convertible to vkop (no dynamic shapes, no If left after simplify_vae_if.py).

    /Users/doudou/qi21-env/bin/python staticize_vae.py \
        [--in vae_decoder_512_folded.onnx] [--out vae_decoder_512_static.onnx]
"""

import argparse
import collections
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

HERE = Path(__file__).resolve().parent

FOLD_OPS = {
    "Shape", "Gather", "Concat", "Unsqueeze", "Slice", "ConstantOfShape",
    "Equal", "Tile", "Expand", "Size", "Cast", "Squeeze", "Reshape",
    "Constant",
}
MAX_FOLD_ELEMS = 64


def pin_static(m, in_shape, out_shape):
    g = m.graph
    inp = g.input[0]
    del inp.type.tensor_type.shape.dim[:]
    for d in in_shape:
        inp.type.tensor_type.shape.dim.add().dim_value = d
    out = g.output[0]
    del out.type.tensor_type.shape.dim[:]
    for d in out_shape:
        out.type.tensor_type.shape.dim.add().dim_value = d
    return m


def fold_round(m):
    """One folding pass: replace small shape-chain node outputs with
    initializers computed by ORT. Returns count of folded nodes."""
    g = m.graph
    # value_info for elem types
    m2 = onnx.shape_inference.infer_shapes(m, strict_mode=False)
    etype = {v.name: v.type.tensor_type.elem_type
             for v in list(m2.graph.value_info) + list(m2.graph.output)}
    cand = [n for n in g.node
            if n.op_type in FOLD_OPS and n.output[0] in etype
            and etype[n.output[0]] in (TensorProto.INT64, TensorProto.BOOL,
                                       TensorProto.INT32)]
    if not cand:
        return 0
    cand_names = [n.output[0] for n in cand]
    existing = {o.name for o in g.output}
    for c in cand_names:
        if c not in existing:
            g.output.append(helper.make_tensor_value_info(
                c, etype[c], None))
    tmp = HERE / ".staticize_run.onnx"
    onnx.save(m, str(tmp))
    so = ort.SessionOptions()
    so.log_severity_level = 3
    sess = ort.InferenceSession(str(tmp), so, providers=["CPUExecutionProvider"])
    x = np.zeros((1, 64, 1, 32, 32), dtype=np.float32)
    names = [o.name for o in sess.get_outputs()]
    outs = sess.run(None, {sess.get_inputs()[0].name: x})
    vals = dict(zip(names, outs))
    tmp.unlink(missing_ok=True)
    tmp.with_suffix(".onnx.data").unlink(missing_ok=True)

    graph_out_names = {o.name for o in g.output if o.name == "decoded"}
    folded = 0
    remove = set()
    for n in cand:
        v = vals.get(n.output[0])
        if v is None or v.size > MAX_FOLD_ELEMS or v.size == 0:
            continue
        if any(o in graph_out_names for o in n.output):
            continue
        g.initializer.append(numpy_helper.from_array(v, n.output[0]))
        remove.add(id(n))
        folded += 1
    keep = [nd for nd in g.node if id(nd) not in remove]
    del g.node[:]
    g.node.extend(keep)
    del g.output[:]
    g.output.append(helper.make_tensor_value_info(
        "decoded", TensorProto.FLOAT, [1, 4, 1, 512, 512]))
    return folded


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default=str(HERE / "vae_decoder_512_folded.onnx"))
    ap.add_argument("--out", default=str(HERE / "vae_decoder_512_static.onnx"))
    args = ap.parse_args()

    m = onnx.load(args.inp, load_external_data=False)
    m = pin_static(m, (1, 64, 1, 32, 32), (1, 4, 1, 512, 512))
    total = 0
    for i in range(10):
        n = fold_round(m)
        print(f"[round {i}] folded {n} shape-chain nodes")
        total += n
        if n == 0:
            break
    onnx.save(m, args.out)

    c = collections.Counter(nd.op_type for nd in m.graph.node)
    print("[hist]", sorted(c.items(), key=lambda kv: -kv[1]))

    # equivalence check
    so = ort.SessionOptions()
    so.log_severity_level = 3
    a = ort.InferenceSession(args.inp, so, providers=["CPUExecutionProvider"])
    b = ort.InferenceSession(args.out, so, providers=["CPUExecutionProvider"])
    x = np.random.randn(1, 64, 1, 32, 32).astype(np.float32)
    ya = a.run(None, {a.get_inputs()[0].name: x})[0]
    yb = b.run(None, {b.get_inputs()[0].name: x})[0]
    print(f"[verify] maxabs(folded, static) = {np.max(np.abs(ya - yb)):.3g}")


if __name__ == "__main__":
    main()
