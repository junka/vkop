#!/usr/bin/env python3
"""Dump HF-preprocessed pixel_values + ORT visual.onnx outputs as ground truth
for the C++ visual_probe `--ref-pv` / `--ref-out` diff.

do_resize=False path (matches the 224x224 exported visual.onnx): the image must
already be 224x224 (or whatever EXPORT_IMG_SIZE was). No smart_resize.

Usage:
  python3 dump_visual_ref.py --image <img>            # real image (must be 224x224)
  python3 dump_visual_ref.py --synth 224              # synthetic 224x224 RGB
  python3 dump_visual_ref.py --out /tmp/vref

Outputs into <out>:
  pv.bin                 raw fp16 [seq_len, 1536]  (C++ --ref-pv)
  image_features.bin     raw fp16 [seq_len, 2048]  (C++ --ref-out)
  deepstack_features_0.bin
  deepstack_features_1.bin
  deepstack_features_2.bin
  meta.txt               grid_thw, seq_len, row, shapes
"""
import os, sys, argparse
import numpy as np
import torch
import onnxruntime as ort
from PIL import Image
from transformers import AutoProcessor

_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
VISUAL_ONNX = os.path.join(_PKG_DIR, "visual.onnx")
MODEL = os.path.expanduser("~/.cache/modelscope/hub/models/Qwen/Qwen3-VL-2B-Instruct")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", help="image path (must be 224x224 for do_resize=False)")
    ap.add_argument("--synth", type=int, help="synthetic NxN RGB image (deterministic)")
    ap.add_argument("--out", default=os.path.join(_PKG_DIR, "vref"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    if args.image:
        img = Image.open(args.image).convert("RGB")
    elif args.synth:
        # deterministic synthetic image: pixel = (r,g,b) pattern
        n = args.synth
        arr = np.zeros((n, n, 3), dtype=np.uint8)
        for y in range(n):
            for x in range(n):
                arr[y, x] = [(y * 255 // n) & 0xFF, (x * 255 // n) & 0xFF,
                             ((y + x) * 255 // (2 * n)) & 0xFF]
        img = Image.fromarray(arr, "RGB")
    else:
        ap.error("need --image or --synth")

    print(f"image size = {img.size}")
    proc = AutoProcessor.from_pretrained(MODEL)
    inputs = proc(text=[""], images=[img], do_resize=False, padding=True, return_tensors="pt")
    pv = inputs.pixel_values.cpu().numpy().astype(np.float16)  # (seq_len, 1536)
    grid_thw = inputs.image_grid_thw.cpu().numpy()              # (1,3)
    seq_len, row = pv.shape
    print(f"pixel_values shape={pv.shape} grid_thw={grid_thw.tolist()}")

    # Run ORT visual.
    print(f"Loading {VISUAL_ONNX}")
    sess = ort.InferenceSession(VISUAL_ONNX, providers=["CPUExecutionProvider"])
    out = sess.run(None, {"pixel_values": pv})
    names = ["image_features", "deepstack_features_0", "deepstack_features_1",
             "deepstack_features_2"]
    for name, arr in zip(names, out):
        arr = arr.astype(np.float16)
        path = os.path.join(args.out, name + ".bin")
        arr.tofile(path)
        f = arr.astype(np.float32)
        print(f"  {name} shape={arr.shape} min={f.min():.4f} max={f.max():.4f} mean={f.mean():.4f} -> {path}")

    pv_path = os.path.join(args.out, "pv.bin")
    pv.tofile(pv_path)
    with open(os.path.join(args.out, "meta.txt"), "w") as f:
        f.write(f"grid_thw={grid_thw.tolist()}\nseq_len={seq_len}\nrow={row}\n")
        for name, arr in zip(names, out):
            f.write(f"{name}={list(arr.shape)}\n")
    print(f"pv -> {pv_path}")
    print(f"meta + 4 outputs -> {args.out}")


if __name__ == "__main__":
    main()
