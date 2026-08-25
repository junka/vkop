#!/usr/bin/env python3
"""Dump ORT stats for the model's existing graph outputs (logits +
present_key_values_*) for one decode round, in the same format as the C++
driver's [name] line, so they can be diffed against VKOP_DUMP_TENSORS output.

Usage: python3 dump_ort_outputs.py <round_dir>
"""
import os
import sys
import numpy as np
import onnxruntime as ort

_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
LLM_ONNX = os.path.join(_PKG_DIR, "llm.onnx")


def fp16_stats(arr):
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
    print(f"Loading {LLM_ONNX}")
    sess = ort.InferenceSession(LLM_ONNX, providers=["CPUExecutionProvider"])
    out_names = [o.name for o in sess.get_outputs()]
    in_name_set = {i.name for i in sess.get_inputs()}

    feed = {}
    with open(os.path.join(round_dir, "input_names.txt")) as f:
        names = [l.strip() for l in f if l.strip()]
    for name in names:
        arr = np.load(os.path.join(round_dir, name + ".npy"))
        if name in in_name_set:
            feed[name] = arr
    print(f"Running ORT round={round_dir} with {len(feed)} inputs, {len(out_names)} outputs...")
    res = sess.run(out_names, feed)
    print()
    for name, arr in zip(out_names, res):
        if arr.dtype == np.float16:
            print(f"[{name}] ", end="")
            fp16_stats(arr)
        else:
            print(f"[{name}] dtype={arr.dtype} shape={arr.shape}")


if __name__ == "__main__":
    main()
