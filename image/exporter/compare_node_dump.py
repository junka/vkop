#!/usr/bin/env python3
"""Diff vkop VKOP_NODE dumps against ORT intermediates (report-only).

vkop files: build/node_dump/NNNNNN_lvlL_nN_NODE_outK.raw (fp16=uint16 LE).
Finds the first execution-order node whose output diverges from ORT beyond
a relative tolerance, which pinpoints the broken op without touching code.

    /Users/doudou/qi21-env/bin/python compare_node_dump.py \
        --dump ../build/node_dump --onnx dit_prefill_tiny_si.onnx
"""

import argparse
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper

HERE = Path(__file__).resolve().parent


def ort_intermediates(onnx_path, feeds):
    m = onnx.load(onnx_path, load_external_data=False)
    m = onnx.shape_inference.infer_shapes(m, strict_mode=False)
    etype = {v.name: v.type.tensor_type.elem_type for v in m.graph.value_info}
    names = set()
    for n in m.graph.node:
        for o in n.output:
            names.add(o)
    existing = {v.name for v in m.graph.output}
    for name in sorted(names - existing):
        et = etype.get(name, TensorProto.FLOAT)
        m.graph.output.append(helper.make_tensor_value_info(name, et, None))
    tmp = onnx_path + ".allout.onnx"
    onnx.save(m, tmp)
    so = ort.SessionOptions()
    so.log_severity_level = 3
    sess = ort.InferenceSession(tmp, so, providers=["CPUExecutionProvider"])
    outs = sess.run(None, feeds)
    Path(tmp).unlink(missing_ok=True)
    Path(tmp + ".data").unlink(missing_ok=True)
    return {o: v for o, v in zip([x.name for x in sess.get_outputs()], outs)}


def load_vkop(path, ort_arr):
    raw = path.read_bytes()
    if ort_arr.dtype == np.int64 and len(raw) == 8 * ort_arr.size:
        return raw and np.frombuffer(raw, dtype=np.int64).astype(np.float32)
    if len(raw) == 2 * ort_arr.size:
        return np.frombuffer(raw, dtype=np.float16).astype(np.float32)
    if len(raw) == 4 * ort_arr.size:
        return np.frombuffer(raw, dtype=np.float32)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", default="../build/node_dump")
    ap.add_argument("--onnx", default="dit_prefill_tiny_si.onnx")
    ap.add_argument("--refdir", default="ref")
    ap.add_argument("--graph", default="prefill")
    args = ap.parse_args()

    ref = HERE / args.refdir
    feeds = {
        "prompt_embeds": np.fromfile(ref / "prompt_embeds.raw", dtype=np.float16).reshape(1, 8, 96),
        "cos": np.fromfile(ref / "cos_prefill.raw", dtype=np.float16).reshape(8, 128),
        "sin": np.fromfile(ref / "sin_prefill.raw", dtype=np.float16).reshape(8, 128),
        "timestep_zero": np.fromfile(ref / "timestep_zero.raw", dtype=np.float32) if (ref / "timestep_zero.raw").exists() else np.zeros(1, dtype=np.float32),
        "attention_bias": np.fromfile(ref / "bias_prefill.raw", dtype=np.float16).reshape(1, 1, 8, 8),
    }
    ort_map = ort_intermediates(str(HERE / args.onnx), feeds)

    files = sorted(Path(args.dump).glob("*_out0.raw"))
    bad = 0
    for f in files:
        # NNNNNN_lvlL_nN_<node>_outK.raw — node may itself contain no '_'? names
        # are like "_MatMul_14" after slash replacement.
        stem = f.name.split("_", 3)[3][:-len("_out0.raw")]
        node = "/" + stem.lstrip("_")
        for cand in (node + "_output_0", node):
            if cand in ort_map:
                node = cand
                break
        else:
            continue
        r = ort_map[node].astype(np.float32).reshape(-1) if ort_map[node].dtype != np.int64 else ort_map[node].reshape(-1).astype(np.float32)
        v = load_vkop(f, ort_map[node])
        if v is None or v.size == 0:
            continue
        if r.size != v.size:
            print(f"  SHAPE {node}: vkop n={v.size} ort n={r.size}")
            bad += 1
            if bad > 8:
                break
            continue
        d = np.max(np.abs(v - r))
        scale = max(np.max(np.abs(r)), 1e-6)
        flag = "OK " if d <= 2e-3 * scale else "BAD"
        print(f"  {flag} {node:26s} maxabs={d:.5g} scale={scale:.5g} n={v.size}")
        if flag == "BAD":
            bad += 1


if __name__ == "__main__":
    main()
