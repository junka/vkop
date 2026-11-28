#!/usr/bin/env python3
"""MiniMax-H3 video VAE decoder (ComfyUI repack, ViT3D) -> ONNX: one clip.

`vae/minimax_h3_video_vae_fp16.safetensors` = causal CNN encoder + a 36-layer
ViT decoder (comfy/ldm/minimax/vae.py ViT3DDecoder). ComfyUI decodes a video
as overlapping temporal clips (7 latent frames, 2 shared) and 256-px spatial
tiles (64-px overlap), blending in pixel space; that bookkeeping is the host's
(example/minimax_h3/fl2va.cpp). This graph is ONE decoder call:

  graph inputs
    z        [N, 24]     fp32  denormalized latent tokens of one clip/tile,
                               (t, h, w) row-major: z * std + mean, then
                               x.flatten(2).transpose(1, 2)
    rope_cos [N + 5, 24] fp32  cos(2*pi * id * inv_freq) per (axis, freq),
    rope_sin [N + 5, 24] fp32  ids from create_token_ids, 0 for the 5 suffix
                               rows (4 register tokens + 1 zero token)
  graph output
    pixels   [N, 3072]   fp32  per token: [3, 4 frames, 16, 16] raw decoder
                               output (host: unpatchify, * pixel_std + mean,
                               clamp to [0, 1])

Usage: python h3_export_fl2va_vae.py --ckpt .../minimax_h3_video_vae_fp16.safetensors \
           --out ~/models/h3_fl2va_vae/vae.onnx [--num-layers 2 --reference ref.npz]

Pre-quantized checkpoint: `minimax_h3_video_vae_int8_convrot.safetensors` is
read as-is. Its 144 decoder linears (to_qkv/to_out/w1/w2) are comfy-kitchen
`int8_tensorwise` with ConvRot: I8 codes [N, K] with one fp32 scale per output
row, absmax/127 (every row reaches |q| = 127), of the weight rotated as W @ H
(block Hadamard, group 256, the same H as the DiT's `--convrot`). The export
streams q * scale as fp16 and puts `x @ H` before each of those linears; the
converter's `-q int8` is the same per-output-channel absmax/127, so it
recovers the repack's codes exactly. `--convrot G` applies the same recipe to
the fp16 checkpoint (rotate, let the converter round).
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

from h3_export_fl2va import hadamard, rotate_in

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

CFG = dict(dim=2048, heads=32, dim_head=64, layers=36, z=24, eps=1e-5,
           n_reg=4, rope_pairs=24, out=3 * 4 * 16 * 16, convrot=0)
ROT_LINEARS = ("attn.to_qkv", "attn.to_out", "ff.w1", "ff.w2")


def rms(x, w, eps):
    xf = x.float()
    y = (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)).to(x.dtype)
    return y if w is None else y * w.to(x.dtype)


def P(shape, dtype, device):
    return nn.Parameter(torch.empty(shape, dtype=dtype, device=device),
                        requires_grad=False)


def lin(i, o, dtype, device):
    m = nn.Linear(i, o, bias=True, dtype=dtype, device=device)
    m.requires_grad_(False)
    return m


class Norm(nn.Module):
    def __init__(self, d, dtype, device):
        super().__init__()
        self.weight = P(d, dtype, device)


class Attn(nn.Module):
    def __init__(self, c, dtype, device):
        super().__init__()
        inner = c["heads"] * c["dim_head"]
        self.to_qkv = lin(inner, inner * 3, dtype, device)
        self.to_out = lin(inner, inner, dtype, device)


class FF(nn.Module):
    def __init__(self, c, dtype, device):
        super().__init__()
        self.w1 = lin(c["dim"], c["dim"] * 4 * 2, dtype, device)
        self.w2 = lin(c["dim"] * 4, c["dim"], dtype, device)


class Block(nn.Module):
    def __init__(self, c, dtype, device):
        super().__init__()
        self.c = c
        self.norm1 = Norm(c["dim"], dtype, device)
        self.attn = Attn(c, dtype, device)
        self.scale1 = P(c["dim"], dtype, device)
        self.norm2 = Norm(c["dim"], dtype, device)
        self.ff = FF(c, dtype, device)
        self.scale2 = P(c["dim"], dtype, device)

    def forward(self, x, cos, sin, rot):
        c = self.c
        L, H, D, R = x.shape[0], c["heads"], c["dim_head"], c["rope_pairs"]
        qkv = self.attn.to_qkv(rot(rms(x, self.norm1.weight, c["eps"])))
        qkv = qkv.reshape(L, H, 3 * D)                 # per head: [q | k | v]
        q, k, v = qkv[..., :D], qkv[..., D:2 * D], qkv[..., 2 * D:]
        q, k = rms(q, None, c["eps"]), rms(k, None, c["eps"])

        def rope(t):  # split-half over the first 2R dims (kitchen rms_rope)
            a, b, rest = t[..., :R], t[..., R:2 * R], t[..., 2 * R:]
            return torch.cat((a * cos - b * sin, b * cos + a * sin, rest), -1)

        q, k = rope(q), rope(k)
        q, k, v = (t.transpose(0, 1).unsqueeze(0) for t in (q, k, v))
        s = torch.matmul(q, k.transpose(-1, -2)) * (1.0 / math.sqrt(D))
        o = torch.matmul(torch.softmax(s, -1), v).squeeze(0).transpose(0, 1)
        x = x + self.attn.to_out(rot(o.reshape(L, H * D))) * self.scale1.to(x.dtype)
        h = self.ff.w1(rot(rms(x, self.norm2.weight, c["eps"])))
        g, u = h[:, :4 * c["dim"]], h[:, 4 * c["dim"]:]
        return x + self.ff.w2(rot(F.silu(g) * u)) * self.scale2.to(x.dtype)


class Decoder(nn.Module):
    def __init__(self, c, dtype, device):
        super().__init__()
        self.x_embedder = lin(c["z"], c["dim"], dtype, device)
        self.register_tokens = P((1, c["n_reg"], c["dim"]), dtype, device)
        self.transformer_blocks = nn.ModuleList(Block(c, dtype, device)
                                                for _ in range(c["layers"]))
        self.norm_out = nn.LayerNorm(c["dim"], eps=c["eps"], dtype=dtype,
                                     device=device)
        self.norm_out.requires_grad_(False)
        self.proj_out = lin(c["dim"], c["out"], dtype, device)


class PostQuantConv(nn.Module):
    def __init__(self, c, dtype, device):
        super().__init__()
        self.weight = P((c["z"], c["z"], 1, 1, 1), dtype, device)
        self.bias = P(c["z"], dtype, device)


class VAEDecoder(nn.Module):
    def __init__(self, c=CFG, dtype=torch.float16, device="meta"):
        super().__init__()
        self.c, self.dtype = c, dtype
        self.post_quant_conv = PostQuantConv(c, dtype, device)
        self.decoder = Decoder(c, dtype, device)
        g = c["convrot"]
        if g:
            # ConvRot: every block linear sees x @ blockdiag(H) and its weight
            # is W @ H (+-1/sqrt(g) is exact in fp16 and bf16)
            self.register_buffer("convrot_h", torch.empty(g, g, dtype=dtype,
                                                          device=device))

    def rot(self, x):
        g = self.c["convrot"]
        if not g:
            return x
        return torch.matmul(x.reshape(-1, g), self.convrot_h).reshape(x.shape)

    def convrot_names(self):
        if not self.c["convrot"]:
            return []
        return [f"decoder.transformer_blocks.{i}.{m}.weight"
                for i in range(self.c["layers"]) for m in ROT_LINEARS]

    def forward(self, z, rope_cos, rope_sin):
        dt, c = self.dtype, self.c
        n = z.shape[0]
        w = self.post_quant_conv.weight.reshape(c["z"], c["z"])
        x = torch.matmul(z.to(dt), w.t()) + self.post_quant_conv.bias
        h = self.decoder.x_embedder(x)
        # the zero suffix token; h[:1] * 0 rather than zeros_like, which
        # traces to ConstantOfShape (no runtime kernel)
        h = torch.cat((h, self.decoder.register_tokens[0].to(dt),
                       h[:1] * 0.0), 0)
        # comfy builds the rotation table in the model dtype
        cos, sin = rope_cos.to(dt).unsqueeze(1), rope_sin.to(dt).unsqueeze(1)
        for blk in self.decoder.transformer_blocks:
            h = blk(h, cos, sin, self.rot)
        out = self.decoder.proj_out(self.decoder.norm_out(h))
        return out[:n].float()


INPUT_NAMES = ["z", "rope_cos", "rope_sin"]
OUTPUT_NAMES = ["pixels"]
DYNAMIC_AXES = {"z": {0: "n"}, "rope_cos": {0: "n5"}, "rope_sin": {0: "n5"},
                "pixels": {0: "n"}}


def host_rope(T, H, W, c=CFG):
    """cos/sin [T*H*W + 5, 24] (comfy RotaryEmbeddingND over create_token_ids)."""
    axes = [(np.arange(0.5, n, dtype=np.float32) / n) * 2.0 - 1.0 for n in (T, H, W)]
    g = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
    g = np.concatenate([g, np.zeros((c["n_reg"] + 1, 3), np.float32)])
    inv = (1.0 / 100.0 ** np.arange(0, 1, 2 * 3 / 48, dtype=np.float32)).astype(np.float32)
    ang = (2.0 * math.pi * g[:, :, None] * inv[None, None, :]).reshape(len(g), -1)
    return np.cos(ang).astype(np.float32), np.sin(ang).astype(np.float32)


def ckpt_quant(path):
    """{weight name: comfy_quant dict} for the decoder's pre-quantized linears
    (empty for the fp16 checkpoint)."""
    import json
    from safetensors import safe_open
    out = {}
    with safe_open(path, "np") as f:
        for k in f.keys():
            if k.startswith("decoder.") and k.endswith(".comfy_quant"):
                out[k[:-len("comfy_quant")] + "weight"] = json.loads(
                    bytes(f.get_tensor(k)).decode())
    return out


def weight_transforms(path, quant, names, g):
    """Checkpoint name -> fp32 rewrite: q * scale for the int8_convrot file
    (already in the rotated basis), W @ H for a plain one."""
    if quant:
        from safetensors import safe_open
        f = safe_open(path, "np")
        bad = [n for n, q in quant.items()
               if q.get("format") != "int8_tensorwise" or
               not q.get("convrot") or q.get("convrot_groupsize") != g]
        if bad or not set(names) <= set(quant):
            raise SystemExit(f"[int8_convrot] unexpected layout: {bad[:3]}, "
                             f"{len(quant)} quantized vs {len(names)} rotated")
        return {n: (lambda w, n=n: w * f.get_tensor(n + "_scale").astype(np.float32))
                for n in names}
    h = hadamard(g)
    return {n: (lambda w, h=h: rotate_in(w, h)) for n in names}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--num-layers", type=int, default=CFG["layers"])
    ap.add_argument("--reference", default=None)
    ap.add_argument("--clip", default="7,8,8", help="T,H,W of the trace/reference clip")
    ap.add_argument("--convrot", type=int, default=0,
                    help="ConvRot group for a plain checkpoint (256 = the "
                         "repack's recipe); set from the file for int8_convrot")
    args = ap.parse_args(argv)
    quant = ckpt_quant(args.ckpt)
    g = args.convrot
    if quant:
        g = next(iter(quant.values()))["convrot_groupsize"]
        if args.convrot and args.convrot != g:
            raise SystemExit(f"--convrot {args.convrot} but the checkpoint uses {g}")
        print(f"[int8_convrot] {len(quant)} pre-quantized linears, group {g}")
    c = dict(CFG, layers=args.num_layers, convrot=g)
    out_dir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(out_dir, exist_ok=True)

    T, H, W = (int(v) for v in args.clip.split(","))
    rng = np.random.default_rng(0)
    z = rng.standard_normal((T * H * W, c["z"])).astype(np.float32)
    cos, sin = host_rope(T, H, W, c)
    inputs = {"z": z, "rope_cos": cos, "rope_sin": sin}

    if args.reference:
        from safetensors import safe_open
        model = VAEDecoder(c, dtype=torch.float32, device="cpu").eval()
        tf = weight_transforms(args.ckpt, quant, model.convrot_names(), g) if g else {}
        sd = {}
        with safe_open(args.ckpt, "pt") as f:
            for name in model.state_dict():
                if name == "convrot_h":
                    sd[name] = torch.from_numpy(hadamard(g))
                    continue
                t = f.get_tensor(name).float()
                if name in tf:
                    t = torch.from_numpy(np.asarray(tf[name](t.numpy()), np.float32))
                sd[name] = t
        model.load_state_dict(sd, assign=True)
        with torch.no_grad():
            px = model(*[torch.from_numpy(inputs[n]) for n in INPUT_NAMES])
        np.savez(args.reference, **inputs, pixels=px.numpy())
        print(f"[ref] {args.reference}: pixels {tuple(px.shape)} absmax "
              f"{px.abs().max():.3f}")
        del model, sd

    import h3_export_common as common
    ck_dir = os.path.join(out_dir, "ckpt", "vae")
    os.makedirs(ck_dir, exist_ok=True)
    link = os.path.join(ck_dir, os.path.basename(args.ckpt))
    for f in os.listdir(ck_dir):
        if os.path.join(ck_dir, f) != link:
            os.unlink(os.path.join(ck_dir, f))
    if not os.path.lexists(link):
        os.symlink(os.path.abspath(args.ckpt), link)
    wrapper = VAEDecoder(c, device="meta").eval()
    dummies = tuple(torch.from_numpy(inputs[n]).to("meta") for n in INPUT_NAMES)
    transforms, computed = {}, {}
    if g:
        import onnx
        transforms = weight_transforms(args.ckpt, quant, wrapper.convrot_names(), g)
        computed = {"convrot_h": onnx.numpy_helper.from_array(
            hadamard(g).astype(np.float16), "convrot_h")}
    with torch.no_grad():
        common.export_streaming_real(
            wrapper, dummies, INPUT_NAMES, OUTPUT_NAMES, DYNAMIC_AXES,
            model_dir=os.path.join(out_dir, "ckpt"), component="vae",
            out_path=args.out,
            # the int8 file keeps everything else fp32; the graph is fp16
            target_dtype="F16" if (g or quant) else None,
            transforms=transforms, computed=computed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
