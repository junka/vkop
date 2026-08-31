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
    # Append all wanted names as graph outputs. ORT can expose an intermediate
    # value as an output even without upfront type/shape info — it infers from
    # the producer node — so append an empty ValueInfoProto (just the name) for
    # every node output not already a graph output. shape_inference on this
    # large dynamic-shape model returns 0 value_info, so relying on it (the old
    # `if name in vi` guard) skipped every intermediate.
    existing = {o.name for o in m.graph.output}
    added = 0
    for name in sorted(want):
        if name in existing:
            continue
        vi_proto = m.graph.output.add()
        vi_proto.name = name
        added += 1
    print(f"Added {added} intermediate outputs ({len(m.graph.output)} total)")
    # Save with external data into a temp dir so the 3.4GB weights do NOT get
    # inlined into a single .onnx (which corrupts the file / blows memory) and
    # so the real llm.weights.bin is never touched. location is a relative
    # filename inside tmpdir.
    import tempfile
    _tmpdir = tempfile.mkdtemp()
    _aug_path = os.path.join(_tmpdir, "aug.onnx")
    onnx.save_model(m, _aug_path, save_as_external_data=True,
                    size_threshold=0, convert_attribute=False,
                    location="aug.weights.bin")
    print(f"Saved augmented model with {len(m.graph.output)} outputs")

    so = ort.SessionOptions()
    sess = ort.InferenceSession(_aug_path, sess_options=so,
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
