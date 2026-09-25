#!/usr/bin/env python3
"""Dump HF-preprocessed pixel_values + ORT visual.onnx outputs as ground truth
for visual_probe's --ref-pv / --ref-out diff.

Replaces dump_visual_ref.py, which dies on `Qwen3VLVideoProcessor requires the
Torchvision library` (AutoProcessor pulls the video processor). AutoImageProcessor
alone needs no torchvision.
"""
import os, sys, argparse
import numpy as np
import onnxruntime as ort
from PIL import Image
from transformers import AutoImageProcessor

_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
VISUAL_ONNX = os.path.join(_PKG_DIR, "visual.onnx")
MODEL = os.path.expanduser("~/.cache/modelscope/hub/models/Qwen/Qwen3-VL-2B-Instruct")
NAMES = ["image_features", "deepstack_features_0", "deepstack_features_1",
         "deepstack_features_2"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image")
    ap.add_argument("--synth", type=int, default=224)
    ap.add_argument("--out", default="/tmp/vref")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    if args.image:
        img = Image.open(args.image).convert("RGB")
    else:
        n = args.synth
        arr = np.zeros((n, n, 3), dtype=np.uint8)
        for y in range(n):
            for x in range(n):
                arr[y, x] = [(y * 255 // n) & 0xFF, (x * 255 // n) & 0xFF,
                             ((y + x) * 255 // (2 * n)) & 0xFF]
        img = Image.fromarray(arr, "RGB")

    proc = AutoImageProcessor.from_pretrained(MODEL)
    inputs = proc(images=[img], do_resize=False, return_tensors="np")
    pv = np.asarray(inputs.pixel_values)
    grid = np.asarray(inputs.image_grid_thw)
    print(f"image={img.size} pixel_values={pv.shape} dtype={pv.dtype} grid_thw={grid.tolist()}")

    sess = ort.InferenceSession(VISUAL_ONNX, providers=["CPUExecutionProvider"])
    info = sess.get_inputs()[0]
    want = {"tensor(float)": np.float32, "tensor(float16)": np.float16}.get(info.type)
    print(f"ort input {info.name} type={info.type} -> cast to {want}")
    pv_in = pv.astype(want)
    outs = sess.run(None, {info.name: pv_in})

    for name, a in zip(NAMES, outs):
        a16 = a.astype(np.float16)
        a16.tofile(os.path.join(args.out, name + ".bin"))
        f = a16.astype(np.float32)
        print(f"  {name} shape={a.shape} min={f.min():.4f} max={f.max():.4f} "
              f"mean={f.mean():.5f}")
    pv.astype(np.float16).tofile(os.path.join(args.out, "pv.bin"))
    with open(os.path.join(args.out, "meta.txt"), "w") as f:
        f.write(f"grid_thw={grid.tolist()}\nseq_len={pv.shape[0]}\nrow={pv.shape[1]}\n")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
