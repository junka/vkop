#!/usr/bin/env python3
"""Fold the 39 trivial If nodes (Squeeze vs Identity) in vae_decoder_512.onnx.

torch's exporter lowered the single-frame temporal squeeze to If(cond) where
cond is shape-derived. With a fixed input shape the condition is a constant:
evaluate every cond with ORT once, splice the taken branch into the main
graph, then verify the folded model is output-identical.

    /Users/doudou/qi21-env/bin/python simplify_vae_if.py [--in f.onnx] [--out g.onnx]
"""

import argparse
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper

HERE = Path(__file__).resolve().parent


def cond_values(model_path, input_shape):
    m = onnx.load(model_path, load_external_data=False)
    g = m.graph
    ifs = [n for n in g.node if n.op_type == "If"]
    cond_names = [n.input[0] for n in ifs]
    existing = {o.name for o in g.output}
    for c in cond_names:
        if c not in existing:
            g.output.append(helper.make_tensor_value_info(c, onnx.TensorProto.BOOL, None))
    tmp = str(model_path) + ".condout.onnx"
    onnx.save(m, tmp)
    so = ort.SessionOptions()
    so.log_severity_level = 3
    sess = ort.InferenceSession(tmp, so, providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    dummy = np.zeros(input_shape, dtype=np.float32)
    outs = sess.run(None, {in_name: dummy})
    names = [o.name for o in sess.get_outputs()]
    Path(tmp).unlink(missing_ok=True)
    Path(tmp + ".data").unlink(missing_ok=True)
    vals = []
    for c in cond_names:
        vals.append(bool(outs[names.index(c)].reshape(-1)[0]))
    return vals


def fold(model_path, out_path, conds):
    m = onnx.load(model_path, load_external_data=False)
    g = m.graph
    ifs = [n for n in g.node if n.op_type == "If"]
    cond_by_name = {n.name: c for n, c in zip(ifs, conds)}
    repl = {n.name: None for n in ifs}
    for n in ifs:
        branch = n.attribute[0].g if bool(cond_by_name[n.name]) else n.attribute[1].g
        nodes = []
        for bn in branch.node:
            new = onnx.NodeProto()
            new.CopyFrom(bn)
            if new.output[0] == branch.output[0].name:
                new.output[0] = n.output[0]
            new.name = n.name + "_fold" + str(len(nodes))
            nodes.append(new)
        repl[n.name] = nodes
    final = []
    for nd in g.node:
        if nd.op_type == "If":
            final.extend(repl[nd.name])
        else:
            final.append(nd)
    del g.node[:]
    g.node.extend(final)
    onnx.save(m, out_path)
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default=str(HERE / "vae_decoder_512.onnx"))
    ap.add_argument("--out", default=str(HERE / "vae_decoder_512_folded.onnx"))
    ap.add_argument("--shape", default="1,64,1,32,32")
    args = ap.parse_args()
    shape = tuple(int(x) for x in args.shape.split(","))

    conds = cond_values(args.inp, shape)
    print(f"[fold] {sum(conds)}/{len(conds)} If take the Squeeze branch")
    fold(args.inp, args.out, conds)

    # verify equivalence
    so = ort.SessionOptions()
    so.log_severity_level = 3
    a = ort.InferenceSession(args.inp, so, providers=["CPUExecutionProvider"])
    b = ort.InferenceSession(args.out, so, providers=["CPUExecutionProvider"])
    x = np.random.randn(*shape).astype(np.float32)
    ya = a.run(None, {a.get_inputs()[0].name: x})[0]
    yb = b.run(None, {b.get_inputs()[0].name: x})[0]
    d = np.max(np.abs(ya - yb))
    print(f"[verify] maxabs(orig, folded) = {d}")
    assert d == 0.0, "folded graph is not equivalent!"
    import collections
    m = onnx.load(args.out, load_external_data=False)
    c = collections.Counter(n.op_type for n in m.graph.node)
    print("[hist]", sorted(c.items(), key=lambda kv: -kv[1]))


if __name__ == "__main__":
    main()
