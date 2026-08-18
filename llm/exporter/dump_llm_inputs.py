#!/usr/bin/env python3
"""Dump llm.onnx prefill inputs + reference logits for the C++ driver.

Text-only (no image) prefill, mirroring llm/exporter/infer.py::run_llm with
past=None. Writes one .npy per graph input and one reference_logits.npy under
the output dir, so the C++ vkop driver can load+upload them and compare.

Usage:
  python3 dump_llm_inputs.py <text> <out_dir>
"""
import os
import sys

import numpy as np
import torch
import onnxruntime as ort
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
LLM_ONNX = os.path.join(_PKG_DIR, "llm.onnx")
MODEL = os.path.expanduser("~/.cache/modelscope/hub/models/Qwen/Qwen3-VL-2B-Instruct")

NLAYERS = 28
HIDDEN = 2048
NKV = 8
HD = 128
IMG_TOK = 151655


def n16(x):
    return x.cpu().numpy().astype(np.float16)


def main():
    text = sys.argv[1] if len(sys.argv) > 1 else "Hello"
    out_dir = sys.argv[2] if len(sys.argv) > 2 else os.path.join(_PKG_DIR, "inputs_dump")
    os.makedirs(out_dir, exist_ok=True)

    print("Loading processor + reference model...")
    proc = AutoProcessor.from_pretrained(MODEL)
    ref = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL, attn_implementation="eager", torch_dtype=torch.float16).eval()

    inputs = proc(text=[text], images=None, do_resize=False, padding=True, return_tensors="pt")
    input_ids = inputs.input_ids  # (1, L)
    am = inputs.attention_mask   # (1, L)
    mtt = getattr(inputs, "mm_token_type_ids", torch.zeros_like(input_ids, dtype=torch.int32))
    grid_thw = getattr(inputs, "image_grid_thw", None)

    L = input_ids.shape[1]
    print(f"prompt: {text!r}  L={L}")

    # image_pad_mask: no image -> all false
    image_pad_mask = torch.zeros_like(input_ids, dtype=torch.bool)

    # embeds
    with torch.no_grad():
        emb = ref.get_input_embeddings()(input_ids)  # (1, L, H) fp16
    inputs_embeds = emb  # no image scatter needed (no image tokens)

    # position_ids (prefill 3D MRoPE)
    with torch.no_grad():
        pos_ids, _ = ref.model.get_rope_index(
            input_ids=input_ids, mm_token_type_ids=mtt,
            image_grid_thw=grid_thw, attention_mask=am)
    print(f"position_ids shape={tuple(pos_ids.shape)} dtype={pos_ids.dtype}")

    # causal attention_bias (1,1,L,L) upper-triangular = finfo.min
    ab = torch.triu(
        torch.full((L, L), torch.finfo(torch.float16).min, dtype=torch.float16),
        diagonal=1).unsqueeze(0).unsqueeze(0)

    # deepstack: zeros (no image)
    ds = [np.zeros((1, HIDDEN), dtype=np.float16) for _ in range(3)]

    # past_key_values: zeros with kv_len=0
    past = [np.zeros((1, 2, NKV, 0, HD), dtype=np.float16) for _ in range(NLAYERS)]

    feed = {
        "inputs_embeds": n16(inputs_embeds),
        "position_ids": pos_ids.cpu().numpy().astype(np.int64),
        "attention_bias": n16(ab),
        "deepstack_embeds_0": ds[0],
        "deepstack_embeds_1": ds[1],
        "deepstack_embeds_2": ds[2],
        "image_pad_mask": image_pad_mask.cpu().numpy().astype(bool),
    }
    for i in range(NLAYERS):
        feed[f"past_key_values_{i}"] = past[i]

    # --- ORT reference ---
    print("Running ONNX Runtime reference...")
    so = ort.SessionOptions()
    lsess = ort.InferenceSession(LLM_ONNX, sess_options=so, providers=["CPUExecutionProvider"])
    # input names may include extra graph inputs not in feed; match by name.
    ort_inputs = {k: v for k, v in feed.items() if k in {i.name for i in lsess.get_inputs()}}
    res = lsess.run(None, ort_inputs)
    logits = res[0]  # (1, L, vocab) fp16
    print(f"reference logits shape={logits.shape} dtype={logits.dtype}")
    next_id = int(np.argmax(logits[0, -1]))
    print(f"reference next token id = {next_id}")

    # --- dump ---
    for name, arr in feed.items():
        path = os.path.join(out_dir, name + ".npy")
        np.save(path, arr)
        print(f"  saved {name} shape={arr.shape} dtype={arr.dtype}")
    np.save(os.path.join(out_dir, "reference_logits.npy"), logits)
    # also record input names order for the C++ driver
    with open(os.path.join(out_dir, "input_names.txt"), "w") as f:
        for name in feed:
            f.write(name + "\n")
    print(f"\nDumped {len(feed)} inputs + reference logits to {out_dir}")


if __name__ == "__main__":
    main()
