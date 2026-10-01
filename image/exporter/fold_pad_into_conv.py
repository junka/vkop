#!/usr/bin/env python3
"""Stage 3 cleanup for the VAE decoder graph: fold every constant-0 Pad into
the pads attribute of the Conv it feeds (vkop has no Pad op).

    /Users/doudou/qi21-env/bin/python fold_pad_into_conv.py
"""

from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper

HERE = Path(__file__).resolve().parent


def get_init(g, name):
    for init in g.initializer:
        if init.name == name:
            return numpy_helper.to_array(init)
    return None


def main():
    inp = HERE / "vae_decoder_512_static.onnx"
    out = HERE / "vae_decoder_512_clean.onnx"
    m = onnx.load(str(inp), load_external_data=False)
    g = m.graph

    consumers = {}
    for n in g.node:
        for i in n.input:
            consumers.setdefault(i, []).append(n)

    pads = [n for n in g.node if n.op_type == "Pad"]
    merged = 0
    remove = set()
    for p in pads:
        cons = consumers.get(p.output[0], [])
        if len(cons) != 1 or cons[0].op_type != "Conv":
            print(f"[skip] {p.name}: consumers {[c.op_type for c in cons]}")
            continue
        conv = cons[0]
        padv = get_init(g, p.input[1]) if len(p.input) > 1 else None
        if padv is None:
            print(f"[skip] {p.name}: pads not an initializer")
            continue
        val = get_init(g, p.input[2]) if len(p.input) > 2 else None
        if val is not None and np.any(val != 0):
            print(f"[skip] {p.name}: nonzero pad value")
            continue
        rank = len(padv) // 2
        if len(padv) != 2 * 4 or rank != 4:
            print(f"[skip] {p.name}: pad rank {rank} (only 4D supported)")
            continue
        # onnx Pad pads (4D input): [n_beg,c_beg,h_beg,w_beg, n_end,c_end,h_end,w_end]
        # Conv pads: [h_beg, w_beg, h_end, w_end]
        new_pads = np.array([padv[2], padv[3], padv[2 + rank], padv[3 + rank]])
        for a in conv.attribute:
            if a.name == "pads":
                new_pads = new_pads + np.array(list(a.ints))
                del a.ints[:]
                a.ints.extend(int(x) for x in new_pads)
                break
        else:
            from onnx import helper
            a = helper.make_attribute("pads", [int(x) for x in new_pads])
            conv.attribute.append(a)
        conv.input[0] = p.input[0]
        remove.add(id(p))
        merged += 1

    keep = [n for n in g.node if id(n) not in remove]
    del g.node[:]
    g.node.extend(keep)
    # drop now-dead pad initializer chains? leave them, converter prunes.
    onnx.save(m, str(out))
    print(f"[done] merged {merged}/{len(pads)} Pads into Conv -> {out.name}")

    # equivalence
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.log_severity_level = 3
    a = ort.InferenceSession(str(inp), so, providers=["CPUExecutionProvider"])
    b = ort.InferenceSession(str(out), so, providers=["CPUExecutionProvider"])
    x = np.random.randn(1, 64, 1, 32, 32).astype(np.float32)
    ya = a.run(None, {a.get_inputs()[0].name: x})[0]
    yb = b.run(None, {b.get_inputs()[0].name: x})[0]
    print(f"[verify] maxabs(static, clean) = {np.max(np.abs(ya - yb)):.3g}")


if __name__ == "__main__":
    main()
