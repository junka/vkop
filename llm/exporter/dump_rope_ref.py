#!/usr/bin/env python3
"""Dump HF get_rope_index reference position_ids for the C++ rope_index diff.

Builds a multimodal (text+image) prompt via AutoProcessor, calls the HF
Qwen3-VL get_rope_index, and writes:
  pos_ids.bin      int64, shape (3, B, L), row-major  -> C++ diff
  meta.txt         L, n_img, grid_thw, mm_token_type run-length encoding
  input_ids.bin    int64 (1, L)
  mtt.bin          int32 (1, L)   mm_token_type_ids
  am.bin           int8  (1, L)   attention_mask

So the C++ side can reconstruct the exact same inputs and diff its pos_ids
byte-for-byte.

Usage:
  python3 dump_rope_ref.py --image <img> --text "<prompt>" --out <dir>
  python3 dump_rope_ref.py --synth 224 --text "<prompt>" --out <dir>
"""
import os, sys, argparse
import numpy as np
import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.expanduser("~/.cache/modelscope/hub/models/Qwen/Qwen3-VL-2B-Instruct")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", help="image path (must match visual.onnx export size)")
    ap.add_argument("--synth", type=int, help="synthetic NxN RGB image")
    ap.add_argument("--text", default="What is in this image?")
    ap.add_argument("--out", default=os.path.join(_PKG_DIR, "rope_ref"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    if args.image:
        from PIL import Image
        img = Image.open(args.image).convert("RGB")
    elif args.synth:
        from PIL import Image
        n = args.synth
        arr = np.zeros((n, n, 3), dtype=np.uint8)
        for y in range(n):
            for x in range(n):
                arr[y, x] = [(y * 255 // n) & 0xFF, (x * 255 // n) & 0xFF,
                             ((y + x) * 255 // (2 * n)) & 0xFF]
        img = Image.fromarray(arr, "RGB")
    else:
        ap.error("need --image or --synth")

    print("Loading processor + reference model...")
    proc = AutoProcessor.from_pretrained(MODEL)
    ref = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL, attn_implementation="eager", torch_dtype=torch.float16).eval()

    # Build the prompt via the chat template so <|image_pad|> is expanded to
    # the right number of image tokens (grid_thw prod / merge^2). A plain text
    # string would leave mtt all-text (rope_delta=0) and not exercise the
    # image branch of get_rope_index.
    messages = [{"role": "user", "content": [
        {"type": "image", "image": img},
        {"type": "text", "text": args.text},
    ]}]
    text = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = proc(text=[text], images=[img], do_resize=False,
                  padding=True, return_tensors="pt")
    input_ids = inputs.input_ids           # (1, L)
    am = inputs.attention_mask             # (1, L)
    mtt = getattr(inputs, "mm_token_type_ids",
                  torch.zeros_like(input_ids, dtype=torch.int32))
    grid_thw = getattr(inputs, "image_grid_thw", None)

    L = input_ids.shape[1]
    print(f"L={L}  grid_thw={grid_thw.tolist() if grid_thw is not None else None}")

    with torch.no_grad():
        pos_ids, rope_delta = ref.model.get_rope_index(
            input_ids=input_ids, mm_token_type_ids=mtt,
            image_grid_thw=grid_thw, attention_mask=am)
    # pos_ids: (3, B, L) int64; rope_delta: (B, 1)
    print(f"position_ids shape={tuple(pos_ids.shape)} dtype={pos_ids.dtype}")
    print(f"rope_delta={rope_delta.tolist()}")

    pos = pos_ids.cpu().numpy().astype(np.int64)
    pos.tofile(os.path.join(args.out, "pos_ids.bin"))
    input_ids.cpu().numpy().astype(np.int64).tofile(os.path.join(args.out, "input_ids.bin"))
    mtt.cpu().numpy().astype(np.int32).tofile(os.path.join(args.out, "mtt.bin"))
    am.cpu().numpy().astype(np.int8).tofile(os.path.join(args.out, "am.bin"))
    with open(os.path.join(args.out, "meta.txt"), "w") as f:
        f.write(f"L={L}\n")
        f.write(f"B={input_ids.shape[0]}\n")
        f.write(f"n_img={grid_thw.shape[0] if grid_thw is not None else 0}\n")
        if grid_thw is not None:
            for i in range(grid_thw.shape[0]):
                t, h, w = grid_thw[i].tolist()
                f.write(f"grid_thw_{i}={t},{h},{w}\n")
        f.write(f"rope_delta={rope_delta.cpu().numpy().tolist()}\n")
        # mm_token_type run-length encoding for debugging
        mtt_list = mtt[0].tolist()
        runs = []
        if mtt_list:
            cur = mtt_list[0]; start = 0
            for i in range(1, len(mtt_list)):
                if mtt_list[i] != cur:
                    runs.append((cur, start, i))
                    cur = mtt_list[i]; start = i
            runs.append((cur, start, len(mtt_list)))
        f.write(f"mtt_runs={runs}\n")
    print(f"dumped to {args.out}")


if __name__ == "__main__":
    main()
