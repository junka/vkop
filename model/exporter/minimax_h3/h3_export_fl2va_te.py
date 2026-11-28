#!/usr/bin/env python3
"""MiniMax-H3 FL2VA text tower (ComfyUI repack) -> ONNX.

`text_encoders/qwen3vl_32b_minimax_h3_bf16.safetensors` is Qwen3-VL-32B's
language model truncated to its first 50 layers (plus the vision tower, unused
for t2va). ComfyUI (comfy/text_encoders/minimax.py) conditions the DiT on the
UNNORMALIZED hidden state after layer 50: no final norm, no lm_head. For a
text-only prompt Qwen3-VL's interleaved MRoPE has equal (t, h, w) positions and
reduces to plain rotate-half RoPE (theta 5e6, head_dim 128).

  graph inputs
    inputs_embeds   [L, 5120] fp32 embed_tokens rows of the raw prompt tokens
                                   (no template, no specials). The 1.5 GB
                                   table is looked up on the host from
                                   embed_tokens.bf16 (raw [151936, 5120] bf16,
                                   written next to the graph), as image_gen
                                   does: a Gather over it would be hydrated
                                   into the protobuf and overflow 2 GB.
    attention_bias  [L, L]  fp32   causal additive mask (0 / large negative)
    rope_cos        [L, 128] fp32  cos(cat(f, f)), f = pos * inv_freq
    rope_sin        [L, 128] fp32
  graph output
    hidden          [L, 5120]      layer-50 hidden state

Usage: python h3_export_fl2va_te.py --ckpt .../qwen3vl_32b_minimax_h3_bf16.safetensors \
           --out ~/models/h3_fl2va_te/te.onnx [--num-layers 2 --reference ref.npz]
"""
from __future__ import annotations

import argparse
import math
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

CFG = dict(vocab=151936, hidden=5120, heads=64, kv_heads=8, head_dim=128,
           ffn=25600, layers=50, eps=1e-6, theta=5.0e6)


def rms_norm(x, w, eps):
    xf = x.float()
    y = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return y.to(x.dtype) * w.to(x.dtype)


class RMSNorm(nn.Module):
    def __init__(self, dim, eps, dtype, device):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.empty(dim, dtype=dtype, device=device),
                                   requires_grad=False)

    def forward(self, x):
        return rms_norm(x, self.weight, self.eps)


def lin(i, o, dtype, device):
    m = nn.Linear(i, o, bias=False, dtype=dtype, device=device)
    m.requires_grad_(False)
    return m


def rotate_half(x):
    h = x.shape[-1] // 2
    return torch.cat((-x[..., h:], x[..., :h]), dim=-1)


class Attention(nn.Module):
    def __init__(self, c, dtype, device):
        super().__init__()
        self.h, self.kv, self.d = c["heads"], c["kv_heads"], c["head_dim"]
        self.q_proj = lin(c["hidden"], self.h * self.d, dtype, device)
        self.k_proj = lin(c["hidden"], self.kv * self.d, dtype, device)
        self.v_proj = lin(c["hidden"], self.kv * self.d, dtype, device)
        self.o_proj = lin(self.h * self.d, c["hidden"], dtype, device)
        self.q_norm = RMSNorm(self.d, c["eps"], dtype, device)
        self.k_norm = RMSNorm(self.d, c["eps"], dtype, device)

    def forward(self, x, bias, cos, sin):
        L = x.shape[0]
        q = self.q_norm(self.q_proj(x).reshape(L, self.h, self.d))
        k = self.k_norm(self.k_proj(x).reshape(L, self.kv, self.d))
        v = self.v_proj(x).reshape(L, self.kv, self.d)
        q = q * cos + rotate_half(q) * sin
        k = k * cos + rotate_half(k) * sin
        rep = self.h // self.kv
        # HF repeat_kv: query head j reads kv head j // rep
        k = k.unsqueeze(2).expand(L, self.kv, rep, self.d).reshape(L, self.h, self.d)
        v = v.unsqueeze(2).expand(L, self.kv, rep, self.d).reshape(L, self.h, self.d)
        q, k, v = (t.transpose(0, 1).unsqueeze(0) for t in (q, k, v))  # [1,H,L,D]
        s = torch.matmul(q, k.transpose(-1, -2)) * (1.0 / math.sqrt(self.d)) + bias
        p = torch.softmax(s, dim=-1)
        o = torch.matmul(p, v).squeeze(0).transpose(0, 1).reshape(L, self.h * self.d)
        return self.o_proj(o)


class MLP(nn.Module):
    def __init__(self, c, dtype, device):
        super().__init__()
        self.gate_proj = lin(c["hidden"], c["ffn"], dtype, device)
        self.up_proj = lin(c["hidden"], c["ffn"], dtype, device)
        self.down_proj = lin(c["ffn"], c["hidden"], dtype, device)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Layer(nn.Module):
    def __init__(self, c, dtype, device):
        super().__init__()
        self.input_layernorm = RMSNorm(c["hidden"], c["eps"], dtype, device)
        self.post_attention_layernorm = RMSNorm(c["hidden"], c["eps"], dtype, device)
        self.self_attn = Attention(c, dtype, device)
        self.mlp = MLP(c, dtype, device)

    def forward(self, x, bias, cos, sin):
        x = x + self.self_attn(self.input_layernorm(x), bias, cos, sin)
        return x + self.mlp(self.post_attention_layernorm(x))


class Tower(nn.Module):
    def __init__(self, c, dtype, device, layer_ids):
        super().__init__()
        # Keyed by the checkpoint's layer index, so a slice (layers 25..49)
        # keeps its tensor names (model.layers.25.*) for the splice.
        self.layers = nn.ModuleDict({str(i): Layer(c, dtype, device)
                                     for i in layer_ids})


class TextTower(nn.Module):
    def __init__(self, c=CFG, dtype=torch.bfloat16, device="meta",
                 layer_ids=None):
        super().__init__()
        self.dtype = dtype
        self.model = Tower(c, dtype, device,
                           layer_ids if layer_ids is not None
                           else range(c["layers"]))

    def forward(self, inputs_embeds, attention_bias, rope_cos, rope_sin):
        dt = self.dtype
        x = inputs_embeds.to(dt)
        bias = attention_bias.to(dt).unsqueeze(0).unsqueeze(0)       # [1,1,L,L]
        cos = rope_cos.to(dt).unsqueeze(1)                          # [L,1,D]
        sin = rope_sin.to(dt).unsqueeze(1)
        for layer in self.model.layers.values():
            x = layer(x, bias, cos, sin)
        return x


INPUT_NAMES = ["inputs_embeds", "attention_bias", "rope_cos", "rope_sin"]
OUTPUT_NAMES = ["hidden"]
DYNAMIC_AXES = {"inputs_embeds": {0: "L"}, "attention_bias": {0: "L", 1: "L2"},
                "rope_cos": {0: "L"}, "rope_sin": {0: "L"}, "hidden": {0: "L"}}


def host_inputs(ids, table, c=CFG):
    """Every graph input for one prompt (the C++ driver builds the same).
    `table` is the bf16 embedding table as uint16 [vocab, hidden]."""
    L = len(ids)
    rows = (table[np.asarray(ids)].astype(np.uint32) << 16).view(np.float32)
    inv = 1.0 / (c["theta"] ** (np.arange(0, c["head_dim"], 2, dtype=np.float32)
                                / c["head_dim"]))
    f = np.arange(L, dtype=np.float32)[:, None] * inv[None, :].astype(np.float32)
    emb = np.concatenate([f, f], -1)
    bias = np.triu(np.full((L, L), -3.0e38, np.float32), 1)
    return {"inputs_embeds": np.ascontiguousarray(rows),
            "attention_bias": bias,
            "rope_cos": np.cos(emb).astype(np.float32),
            "rope_sin": np.sin(emb).astype(np.float32)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--num-layers", type=int, default=CFG["layers"])
    ap.add_argument("--layers", default=None,
                    help="START,END: export only layers [START, END) -- the "
                         "50-layer tower is 24 GB at int8, more than a 36 GB "
                         "box can load beside the OS; two halves run in "
                         "sequence (inputs_embeds of the second = hidden of "
                         "the first)")
    ap.add_argument("--reference", default=None,
                    help="run the truncated tower on CPU (fp32) and save "
                         "inputs/output to this .npz")
    ap.add_argument("--text", default="A red fox runs through a snowy forest at dawn.")
    args = ap.parse_args(argv)
    c = dict(CFG, layers=args.num_layers)
    layer_ids = range(c["layers"])
    if args.layers:
        lo, hi = (int(v) for v in args.layers.split(","))
        layer_ids = range(lo, hi)
    out_dir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(out_dir, exist_ok=True)

    from transformers import Qwen2Tokenizer
    tok_dir = os.path.join(out_dir, "tokenizer")
    if not os.path.isdir(tok_dir):
        tok_dir = os.path.join(os.path.dirname(out_dir), "h3_fl2va_te", "tokenizer")
    tok = Qwen2Tokenizer.from_pretrained(tok_dir)
    ids = tok(args.text, add_special_tokens=False)["input_ids"]

    # The embedding table as a raw bf16 file the driver mmaps.
    import json, struct
    with open(args.ckpt, "rb") as f:
        hlen = struct.unpack("<Q", f.read(8))[0]
        e = json.loads(f.read(hlen))["model.embed_tokens.weight"]
    assert e["dtype"] == "BF16" and e["shape"] == [c["vocab"], c["hidden"]], e
    table = np.memmap(args.ckpt, np.uint16, "r", offset=8 + hlen + e["data_offsets"][0],
                      shape=(c["vocab"], c["hidden"]))
    emb_path = os.path.join(out_dir, "embed_tokens.bf16")
    if not os.path.exists(emb_path) or os.path.getsize(emb_path) != table.nbytes:
        with open(emb_path, "wb") as dst:
            for r0 in range(0, c["vocab"], 8192):
                dst.write(np.ascontiguousarray(table[r0:r0 + 8192]).tobytes())
        print(f"[embed] wrote {emb_path} ({table.nbytes / 2**30:.2f} GiB)")
    inputs = host_inputs(ids, table, c)

    if args.reference:
        from safetensors import safe_open
        model = TextTower(c, dtype=torch.float32, device="cpu",
                          layer_ids=layer_ids).eval()
        sd = {}
        with safe_open(args.ckpt, "pt") as f:
            for name in model.state_dict():
                sd[name] = f.get_tensor(name).float()
        model.load_state_dict(sd, assign=True)
        with torch.no_grad():
            h = model(*[torch.from_numpy(inputs[n]) for n in INPUT_NAMES])
        np.savez(args.reference, **inputs, hidden=h.numpy())
        print(f"[ref] {args.reference}: L={len(ids)} hidden absmax "
              f"{h.abs().max():.1f} rms {h.pow(2).mean().sqrt():.3f}")
        del model, sd

    import h3_export_common as common
    ck_dir = os.path.join(out_dir, "ckpt", "te")
    os.makedirs(ck_dir, exist_ok=True)
    link = os.path.join(ck_dir, os.path.basename(args.ckpt))
    for f in os.listdir(ck_dir):
        if os.path.join(ck_dir, f) != link:
            os.unlink(os.path.join(ck_dir, f))
    if not os.path.lexists(link):
        os.symlink(os.path.abspath(args.ckpt), link)
    wrapper = TextTower(c, device="meta", layer_ids=layer_ids).eval()
    dummies = tuple(torch.from_numpy(inputs[n]).to("meta") for n in INPUT_NAMES)
    with torch.no_grad():
        common.export_streaming_real(
            wrapper, dummies, INPUT_NAMES, OUTPUT_NAMES, DYNAMIC_AXES,
            model_dir=os.path.join(out_dir, "ckpt"), component="te",
            out_path=args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
