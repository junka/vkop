#!/usr/bin/env python3
"""MiniMax-H3 audio VAE decoder (ComfyUI repack, BigVGAN) -> ONNX.

`vae/minimax_h3_audio_vae_fp32.safetensors` = DAC encoder + attention
posterior head + `dec_in_proj` + a BigVGAN decoder (comfy/ldm/minimax/
audio_vae.py, MiniMaxH3AudioVAE.decode). Only the decode path is exported:

  z [S, 32, 1, T]  normalized latent, one row per stereo channel (S = 2),
                   T latent frames at 40 per second
    -> (z * std + mean)            folded into dec_in_proj's weight/bias
    -> dec_in_proj  1x1 conv 32 -> 2048
    -> BigVGAN: conv_pre, 7 x (ConvTranspose upsample, 3 AMP blocks averaged),
       anti-aliased SnakeBeta, conv_post (no bias), clamp [-1, 1]
  wave [S, 1, 1, T * 800]          32 kHz

Rewrites against the reference, all exact in fp32 up to rounding:

* Every Conv1d is a Conv2d over [N, C, 1, L] (the runtime's Conv is 4-D).
* SnakeBeta x + sin(e^a x)^2 / (e^b + 1e-9): e^a and 1/(e^b + 1e-9) are
  computed here and stored as [1, C, 1, 1] constants (no Exp/Reciprocal).
* The alias-free resampler's shared 12-tap filters become per-channel
  depthwise weights [C, 1, 1, 12] (the upsample ratio 2 folded in), so the
  graph has no Expand. Replicate padding is ONNX Pad mode=edge.
* The latent denormalization is folded into dec_in_proj:
  W' = W * std, b' = b + W @ mean.

The weights (decoder ~250 MB fp32) are small enough to build the model on CPU
and export with the TorchScript exporter directly; no meta/streaming path.

Usage:
  python h3_export_fl2va_audio_vae.py --ckpt .../minimax_h3_audio_vae_fp32.safetensors \
      --out ~/models/h3_fl2va_avae/avae.onnx [--reference ref.npz --frames 16]

`--reference` also runs a literal fp32 port of ComfyUI's decode (1-D convs,
Exp, replicate pad, expanded filters) and checks the rewritten graph against it.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

UP_RATES = (5, 5, 2, 2, 2, 2, 2)
UP_KERNELS = (9, 9, 4, 4, 4, 4, 4)
RES_KERNELS = (3, 7, 11)
RES_DILATIONS = (1, 3, 5)
CH0 = 1024
HOP = 800


def load_decoder_sd(path):
    from safetensors import safe_open
    sd = {}
    with safe_open(path, "pt") as f:
        meta = json.loads(f.metadata()["minimax_h3_audio_vae"])
        for k in f.keys():
            if k.startswith(("decoder.", "dec_in_proj.", "latents_")):
                sd[k] = f.get_tensor(k).float()
    return sd, meta


# ---------------------------------------------------------------- reference
# A literal port of comfy/ldm/minimax/audio_vae.py's decode path.

def ref_snake(x, a, b):
    a = torch.exp(a).view(1, -1, 1)
    b = torch.exp(b).view(1, -1, 1)
    t = torch.sin(a * x)
    return t * t * (b + 1e-9).reciprocal() + x


def ref_act1d(x, sd, p):
    C = x.shape[1]
    fu = sd[p + "upsample.filter"]
    x = F.pad(x, (5, 5), mode="replicate")
    x = F.conv_transpose1d(x, fu.expand(C, -1, -1), stride=2, groups=C) * 2
    x = x[..., 15:-15]
    x = ref_snake(x, sd[p + "act.alpha"], sd[p + "act.beta"])
    fd = sd[p + "downsample.lowpass.filter"]
    x = F.pad(x, (5, 6), mode="replicate")
    return F.conv1d(x, fd.expand(C, -1, -1), stride=2, groups=C)


def ref_decode(sd, z):
    """z [B, 32, 2, T] normalized -> [B, 2, T*800] (comfy decode)."""
    b, c, s, t = z.shape
    z = z.permute(0, 2, 1, 3).reshape(b * s, c, t)
    z = z * sd["latents_std"].view(1, -1, 1) + sd["latents_mean"].view(1, -1, 1)
    x = F.conv1d(z, sd["dec_in_proj.weight"], sd["dec_in_proj.bias"])
    x = F.conv1d(x, sd["decoder.conv_pre.weight"], sd["decoder.conv_pre.bias"],
                 padding=3)
    for i, (u, k) in enumerate(zip(UP_RATES, UP_KERNELS)):
        p = f"decoder.ups.{i}.0."
        x = F.conv_transpose1d(x, sd[p + "weight"], sd[p + "bias"], stride=u,
                               padding=(k - u) // 2)
        xs = None
        for j, rk in enumerate(RES_KERNELS):
            r = f"decoder.resblocks.{i * 3 + j}."
            h = x
            for m, d in enumerate(RES_DILATIONS):
                xt = ref_act1d(h, sd, f"{r}activations.{2 * m}.")
                xt = F.conv1d(xt, sd[f"{r}convs1.{m}.weight"],
                              sd[f"{r}convs1.{m}.bias"], dilation=d,
                              padding=(rk * d - d) // 2)
                xt = ref_act1d(xt, sd, f"{r}activations.{2 * m + 1}.")
                xt = F.conv1d(xt, sd[f"{r}convs2.{m}.weight"],
                              sd[f"{r}convs2.{m}.bias"], padding=(rk - 1) // 2)
                h = xt + h
            xs = h if xs is None else xs + h
        x = xs / 3
    x = ref_act1d(x, sd, "decoder.activation_post.")
    x = F.conv1d(x, sd["decoder.conv_post.weight"], padding=3).clamp(-1, 1)
    return x.reshape(b, s, -1)


# ---------------------------------------------------------------- graph

def P(t):
    return nn.Parameter(t.contiguous(), requires_grad=False)


class EdgePad(torch.autograd.Function):
    """Replicate pad of the last axis with STATIC pads: F.pad(mode=replicate)
    traces to a dynamic pads computation (ConstantOfShape/Concat/...)."""

    @staticmethod
    def forward(ctx, x, left, right):
        return F.pad(x, (left, right, 0, 0), mode="replicate")

    @staticmethod
    def symbolic(g, x, left, right):
        pads = g.op("Constant", value_t=torch.tensor([0, 0, 0, left, 0, 0, 0, right],
                                                     dtype=torch.int64))
        return g.op("Pad", x, pads, mode_s="edge")


class Act(nn.Module):
    """Anti-aliased SnakeBeta on [N, C, 1, L]."""

    def __init__(self, sd, p):
        super().__init__()
        C = sd[p + "act.alpha"].numel()
        self.a = P(torch.exp(sd[p + "act.alpha"]).view(1, C, 1, 1))
        self.ib = P((torch.exp(sd[p + "act.beta"]) + 1e-9).reciprocal().view(1, C, 1, 1))
        self.up = P((sd[p + "upsample.filter"] * 2).view(1, 1, 1, 12).expand(C, 1, 1, 12).clone())
        self.dn = P(sd[p + "downsample.lowpass.filter"].view(1, 1, 1, 12).expand(C, 1, 1, 12).clone())
        self.C = C

    def forward(self, x):
        x = EdgePad.apply(x, 5, 5)
        x = F.conv_transpose2d(x, self.up, stride=(1, 2), groups=self.C)
        x = x[..., 15:-15]
        t = torch.sin(x * self.a)
        x = x + t * t * self.ib
        x = EdgePad.apply(x, 5, 6)
        return F.conv2d(x, self.dn, stride=(1, 2), groups=self.C)


class Conv(nn.Module):
    def __init__(self, w, b, dilation=1, padding=0):
        super().__init__()
        self.w = P(w.unsqueeze(2))
        self.b = P(b) if b is not None else None
        self.d, self.p = dilation, padding

    def forward(self, x):
        return F.conv2d(x, self.w, self.b, dilation=(1, self.d),
                        padding=(0, self.p))


class AMP(nn.Module):
    def __init__(self, sd, r, k):
        super().__init__()
        self.c1 = nn.ModuleList(Conv(sd[f"{r}convs1.{m}.weight"], sd[f"{r}convs1.{m}.bias"],
                                     d, (k * d - d) // 2)
                                for m, d in enumerate(RES_DILATIONS))
        self.c2 = nn.ModuleList(Conv(sd[f"{r}convs2.{m}.weight"], sd[f"{r}convs2.{m}.bias"],
                                     1, (k - 1) // 2)
                                for m in range(len(RES_DILATIONS)))
        self.acts = nn.ModuleList(Act(sd, f"{r}activations.{i}.") for i in range(6))

    def forward(self, x):
        for m in range(len(RES_DILATIONS)):
            xt = self.c2[m](self.acts[2 * m + 1](self.c1[m](self.acts[2 * m](x))))
            x = xt + x
        return x


class AudioDecoder(nn.Module):
    def __init__(self, sd, num_ups=len(UP_RATES)):
        super().__init__()
        w = sd["dec_in_proj.weight"][:, :, 0]               # [2048, 32]
        std, mean = sd["latents_std"], sd["latents_mean"]
        self.inp = Conv((w * std[None, :]).unsqueeze(-1),
                        sd["dec_in_proj.bias"] + w @ mean)
        self.pre = Conv(sd["decoder.conv_pre.weight"], sd["decoder.conv_pre.bias"], 1, 3)
        self.ups = nn.ModuleList()
        self.cfg = []
        for i in range(num_ups):
            p = f"decoder.ups.{i}.0."
            self.ups.append(nn.ParameterList([P(sd[p + "weight"].unsqueeze(2)),
                                              P(sd[p + "bias"])]))
            self.cfg.append((UP_RATES[i], (UP_KERNELS[i] - UP_RATES[i]) // 2))
        self.res = nn.ModuleList(AMP(sd, f"decoder.resblocks.{i * 3 + j}.", k)
                                 for i in range(num_ups)
                                 for j, k in enumerate(RES_KERNELS))
        self.full = num_ups == len(UP_RATES)
        if self.full:
            self.post_act = Act(sd, "decoder.activation_post.")
            self.post = Conv(sd["decoder.conv_post.weight"], None, 1, 3)

    def forward(self, z):
        x = self.pre(self.inp(z))
        for i, (u, pad) in enumerate(self.cfg):
            w, b = self.ups[i]
            x = F.conv_transpose2d(x, w, b, stride=(1, u), padding=(0, pad))
            r = self.res[3 * i:3 * i + 3]
            x = (r[0](x) + r[1](x) + r[2](x)) * (1.0 / 3.0)
        if not self.full:
            return x
        return torch.clamp(self.post(self.post_act(x)), -1.0, 1.0)


def cos(a, b):
    a, b = a.double().flatten(), b.double().flatten()
    return float((a @ b) / (a.norm() * b.norm() + 1e-30))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--frames", type=int, default=16, help="T of the trace/reference")
    ap.add_argument("--num-ups", type=int, default=len(UP_RATES),
                    help="truncate after this many upsample stages (debug)")
    ap.add_argument("--reference", default=None,
                    help="write inputs/outputs of the fp32 torch graph here and "
                         "check it against the literal ComfyUI port")
    ap.add_argument("--latent", default=None,
                    help="reference input: a driver audio_latent.f32 [32,2,T] "
                         "(else seeded noise)")
    args = ap.parse_args(argv)
    torch.set_grad_enabled(False)
    sd, meta = load_decoder_sd(args.ckpt)
    assert meta["sample_rate"] == 32000 and meta["latent_channels"] == 32
    model = AudioDecoder(sd, args.num_ups).eval()

    if args.latent:
        zr = np.fromfile(args.latent, np.float32).reshape(32, 2, -1)
        zr = zr[:, :, :args.frames]
    else:
        zr = np.random.default_rng(0).standard_normal((32, 2, args.frames)).astype(np.float32)
    T = zr.shape[2]
    z = torch.from_numpy(np.ascontiguousarray(zr.transpose(1, 0, 2))).unsqueeze(2)  # [2,32,1,T]

    if args.reference:
        y = model(z)
        if model.full:
            r = ref_decode(sd, torch.from_numpy(zr)[None])[0]
            c = cos(y.reshape(2, -1), r)
            print(f"[ref] graph port vs literal comfy port: cos {c:.7f}, "
                  f"max|d| {float((y.reshape(2, -1) - r).abs().max()):.2e}")
        np.savez(args.reference, z=z.numpy(), wave=y.numpy())
        print(f"[ref] {args.reference}: z {tuple(z.shape)} -> {tuple(y.shape)}, "
              f"rms {float(y.pow(2).mean().sqrt()):.4f} absmax {float(y.abs().max()):.4f}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.onnx.export(model, (z,), args.out, input_names=["z"], output_names=["wave"],
                      dynamic_axes={"z": {3: "t"}, "wave": {3: "samples"}},
                      opset_version=17, dynamo=False, do_constant_folding=True)
    import onnx
    m = onnx.load(args.out, load_external_data=False)
    ops = {}
    for n in m.graph.node:
        ops[n.op_type] = ops.get(n.op_type, 0) + 1
    print(f"[export] {args.out}: {len(m.graph.node)} nodes {dict(sorted(ops.items()))}; "
          f"{os.path.getsize(args.out) / 2**20:.0f} MiB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
