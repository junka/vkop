#!/usr/bin/env python3
"""Export ONNX Runtime intermediate stats for one decode round, to compare
against vkop's VKOP_DUMP_TENSORS output.

Adds every graph value_info output as a session output, runs round1's feed,
and prints per-tensor min/max/mean/first4 stats — same format as the C++
driver's [dump] line. Compare line-by-line to find the first diverging op.

Usage: python3 dump_ort_intermediates.py <round_dir>
"""
import os
import sys
import numpy as np
import onnx
import onnxruntime as ort

_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
LLM_ONNX = os.path.join(_PKG_DIR, "llm.onnx")


def fp16_stats(arr):
    """arr: fp16 numpy array. Print stats matching the C++ driver format."""
    f = arr.astype(np.float32)
    nan = int(np.isnan(f).sum())
    inf = int(np.isinf(f).sum())
    zero = int((f == 0).sum())
    finite = f[np.isfinite(f)]
    mn = float(finite.min()) if finite.size else 0.0
    mx = float(finite.max()) if finite.size else 0.0
    first = arr.reshape(-1)[:4].astype(np.uint16)
    first_hex = ",".join(f"{x:04x}" for x in first)
    print(f"ne={f.size} nan={nan} inf={inf} zero={zero} min={mn:.4g} max={mx:.4g} first=[{first_hex}]")


def main():
    round_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(_PKG_DIR, "decode_dump/round1")
    print(f"Loading ONNX + adding all intermediates as outputs: {LLM_ONNX}")
    m = onnx.load(LLM_ONNX)
    # Collect every named value (node outputs + graph outputs + inputs).
    want = set()
    for n in m.graph.node:
        for o in n.output:
            if o:
                want.add(o)
    # Build value_info for each (ONNX needs shape/type to mark as output; ORT
    # can also infer). Use onnx.shape_inference to get types.
    try:
        m_inf = onnx.shape_inference.infer_shapes(m)
        vi = {v.name: v for v in m_inf.graph.value_info}
    except Exception as e:
        print(f"shape inference failed: {e}")
        vi = {}
    # Append all wanted names as graph outputs.
    existing = {o.name for o in m.graph.output}
    for name in sorted(want):
        if name in existing:
            continue
        if name in vi:
            m.graph.output.append(vi[name])
        # If no value_info, skip — ORT needs type info to expose it.
    onnx.save(m, "/tmp/llm_all_outs.onnx")
    print(f"Saved augmented model with {len(m.graph.output)} outputs")

    so = ort.SessionOptions()
    sess = ort.InferenceSession("/tmp/llm_all_outs.onnx", sess_options=so,
                                providers=["CPUExecutionProvider"])
    out_names = [o.name for o in sess.get_outputs()]
    in_name_set = {i.name for i in sess.get_inputs()}

    # Load round feed.
    feed = {}
    with open(os.path.join(round_dir, "input_names.txt")) as f:
        names = [l.strip() for l in f if l.strip()]
    for name in names:
        arr = np.load(os.path.join(round_dir, name + ".npy"))
        if name in in_name_set:
            feed[name] = arr
    print(f"Running ORT with {len(feed)} inputs, requesting {len(out_names)} outputs...")
    res = sess.run(out_names, feed)
    print(f"Got {len(res)} results\n")

    # Print fp16 tensors in the same format as the C++ driver, in graph order.
    for name, arr in zip(out_names, res):
        if arr.dtype == np.float16:
            print(f"[{name}] ", end="")
            fp16_stats(arr)
        # skip non-fp16 (int64/bool/float32) to keep output comparable


if __name__ == "__main__":
    main()
