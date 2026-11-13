#!/usr/bin/env python3
"""Extract the language model's embedding table to embed_tokens.bin (raw fp16).

llm_chat's decode loop looks up token id -> embedding row from a standalone
[vocab, hidden] fp16 file; the table is not part of the vkop graph. This reads
just that one tensor out of the checkpoint instead of building the whole model
(see llm/exporter/export/dump_embed_tokens.py, which does the same by loading
the full model).

Usage:
    python3 llm/server/embed_table.py [--model DIR] [--out FILE]
"""
import argparse
import json
import os
import struct
import sys

DEFAULT_MODEL = os.path.expanduser(
    "~/.cache/modelscope/hub/models/Qwen/Qwen3-VL-2B-Instruct")
DEFAULT_OUT = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "exporter", "embed_tokens.bin")

# Candidates in preference order: language tower only, then any embed table.
KEY_CANDIDATES = [
    "model.language_model.embed_tokens.weight",
    "model.embed_tokens.weight",
    "language_model.model.embed_tokens.weight",
]


def _pick_key(header):
    for k in KEY_CANDIDATES:
        if k in header:
            return k
    for k in header:
        if k.endswith("embed_tokens.weight"):
            return k
    raise SystemExit("no embed_tokens.weight tensor in checkpoint")


def extract(model_dir, out_path):
    st = os.path.join(model_dir, "model.safetensors")
    if not os.path.exists(st):
        # Sharded checkpoints: the embed table lives in exactly one shard.
        index = os.path.join(model_dir, "model.safetensors.index.json")
        if not os.path.exists(index):
            raise SystemExit(f"neither model.safetensors nor an index in {model_dir}")
        with open(index) as f:
            weight_map = json.load(f)["weight_map"]
        shard = next((v for k, v in weight_map.items()
                      if k.endswith("embed_tokens.weight")), None)
        if shard is None:
            raise SystemExit("index has no embed_tokens.weight")
        st = os.path.join(model_dir, shard)

    with open(st, "rb") as f:
        hlen = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(hlen))
        key = _pick_key(header)
        meta = header[key]
        shape = meta["shape"]
        dtype = meta["dtype"]
        begin, end = meta["data_offsets"]
        if len(shape) != 2:
            raise SystemExit(f"{key} is {len(shape)}-D, expected [vocab, hidden]")
        vocab, hidden = shape
        f.seek(8 + hlen + begin)
        raw = f.read(end - begin)

    if dtype == "F16":
        blob = raw
    elif dtype in ("BF16", "F32"):
        # Widen through float32 and let numpy round to nearest even. Truncating
        # bf16 -> fp16 directly would drop 3 more mantissa bits; embedding
        # weights are O(1) so fp16 range is never the limiting factor.
        import numpy as np
        if dtype == "BF16":
            u16 = np.frombuffer(raw, dtype="<u2").astype(np.uint32)
            f32 = (u16 << 16).view(np.float32)
        else:
            f32 = np.frombuffer(raw, dtype="<f4")
        blob = f32.astype(np.float16).astype("<f2").tobytes()
    else:
        raise SystemExit(f"unsupported embed dtype {dtype}")

    expected = vocab * hidden * 2
    if len(blob) != expected:
        raise SystemExit(f"size mismatch: {len(blob)} vs {expected}")

    tmp = out_path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(blob)
    os.replace(tmp, out_path)
    print(f"[ok] {out_path}  vocab={vocab} hidden={hidden} dtype=fp16 "
          f"{len(blob) / 1e6:.1f}MB (from {dtype} {key})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--out", default=DEFAULT_OUT)
    a = ap.parse_args()
    extract(a.model, a.out)


if __name__ == "__main__":
    main()
