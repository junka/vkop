#!/usr/bin/env python3
"""Dump ORT visual.onnx intermediate stats for every node output, to compare
against vkop's VKOP_DUMP_TENSORS='*' output and find the first diverging op.

Reads /tmp/vref/pv.bin (from dump_visual_ref.py) as pixel_values, adds every
node output as a graph output, runs ORT, prints per-tensor stats in the same
format as the C++ driver's [name] line.

visual.onnx is a dynamic-grid graph, so it also needs grid_thw: taken from
meta.txt next to pv.bin when present, else "H,W" via argv[2], else 14x14 (the
export's trace example).

Usage: python3 export/dump_visual_intermediates.py [pv.bin] [grid_h,grid_w]
"""
import os, sys, ast
import numpy as np
import onnx
import onnxruntime as ort

_PKG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VISUAL_ONNX = os.path.join(_PKG_DIR, "visual.onnx")


def fp16_stats(arr):
    f = arr.astype(np.float32)
    finite = f[np.isfinite(f)]
    mn = float(finite.min()) if finite.size else 0.0
    mx = float(finite.max()) if finite.size else 0.0
    mean = float(finite.mean()) if finite.size else 0.0
    print(f"ne={f.size} min={mn:.4g} max={mx:.4g} mean={mean:.4g}")


def read_grid(pv_path, arg):
    """(grid_h, grid_w) for this pv.bin: argv > meta.txt > 14x14."""
    if arg:
        h, w = arg.split(",")
        return int(h), int(w)
    meta = os.path.join(os.path.dirname(os.path.abspath(pv_path)), "meta.txt")
    if os.path.exists(meta):
        for line in open(meta):
            if line.startswith("grid_thw="):
                g = ast.literal_eval(line.split("=", 1)[1].strip())[0]
                return int(g[1]), int(g[2])
    return 14, 14


def main():
    pv_path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/vref/pv.bin"
    flat = np.fromfile(pv_path, dtype=np.float16)
    row = 1536
    seq_len = flat.size // row
    pv = flat.reshape(seq_len, row)
    gh, gw = read_grid(pv_path, sys.argv[2] if len(sys.argv) > 2 else None)
    assert gh * gw == seq_len, \
        f"pv has {seq_len} rows but grid {gh}x{gw} = {gh*gw} patches"
    grid = np.array([[1, gh, gw]], dtype=np.int64)
    print(f"pv {pv.shape} grid_thw={grid.tolist()}")
    print(f"Loading {VISUAL_ONNX}")
    m = onnx.load(VISUAL_ONNX)
    want = set()
    for n in m.graph.node:
        for o in n.output:
            if o:
                want.add(o)
    existing = {o.name for o in m.graph.output}
    added = 0
    for name in sorted(want):
        if name in existing:
            continue
        vi = m.graph.output.add()
        vi.name = name
        added += 1
    print(f"Added {added} intermediate outputs ({len(m.graph.output)} total)")
    import tempfile
    tmpdir = tempfile.mkdtemp()
    aug = os.path.join(tmpdir, "aug.onnx")
    onnx.save_model(m, aug, save_as_external_data=True, size_threshold=0,
                    convert_attribute=False, location="aug.weights.bin")
    sess = ort.InferenceSession(aug, providers=["CPUExecutionProvider"])
    out_names = [o.name for o in sess.get_outputs()]
    in_names = {i.name for i in sess.get_inputs()}
    feed = {}
    for n in in_names:
        if n == "pixel_values":
            feed[n] = pv
        elif n == "grid_thw":
            feed[n] = grid
    print(f"Running ORT with {len(feed)} inputs, {len(out_names)} outputs...")
    res = sess.run(out_names, feed)
    print(f"Got {len(res)} results\n")
    for name, arr in zip(out_names, res):
        if arr.dtype == np.float16:
            print(f"[{name}] ", end="")
            fp16_stats(arr)
        elif arr.dtype == np.float32:
            print(f"[{name}] (f32) ", end="")
            fp16_stats(arr.astype(np.float16))


if __name__ == "__main__":
    main()
