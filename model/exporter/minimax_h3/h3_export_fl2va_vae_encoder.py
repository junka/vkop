#!/usr/bin/env python3
"""MiniMax-H3 video VAE encoder, single image -> ONNX (one 256-px tile).

`vae/minimax_h3_video_vae_fp16.safetensors` = causal 3-D CNN encoder + ViT
decoder (comfy/ldm/minimax/vae.py). Ref2VA encodes each reference image with
`MiniMaxH3VideoVAE.encode` on a ONE-frame clip [1, 3, 1, th, tw]:

  pixels [0, 1] --comfy VAE.process_input--> [-1, 1]
    --_normalize_pixels--> (x - imagenet_mean) / imagenet_std
    --tiled_encode (256-px tiles, >= 64-px overlap, latent-space linear blend)
       per tile: EncoderFCN3D -> quant_conv          (moments, 48 ch)
    --> moments[:, :, -1:], mean = first 24 ch       (posterior MEAN, no sample)
    --> (mean - latents_mean) / latents_std          (normalized: what the DiT sees)

With one frame every CausalConv3d takes comfy's `autopad="causal_zero"` path
(no temporal padding, weight[:, :, -1:]), and the time-stride-2 downsamplers
keep T = 1, so the encoder is EXACTLY a 2-D CNN on the last temporal tap of
each kernel; TemporalIsolatedGroupNorm is a plain 2-D GroupNorm(32, eps 1e-6).
This graph is that 2-D CNN for ONE tile, with exact folds:

* imagenet normalization -> conv_in (reflect padding commutes with a
  per-channel affine): W' = W / std, b' = b - sum(W * mean / std).
* quant_conv (1x1) mean half + latent normalization -> conv_out:
  W'' = (Wq[:24] / ls) @ Wout, b'' = (Wq[:24] @ bout + bq[:24] - lm) / ls.
  latents_mean/std are the checkpoint's buffers (stored fp16, which is what
  comfy's load_state_dict puts in the module), not the fp32 metadata copy.

  graph input   pixels [N, 3, H, W]  fp32 RGB in [0, 1], one tile per batch row
                                     (H, W = 256, or the whole side if <= 256;
                                     multiples of 16; dynamic)
  graph output  latent [N, 24, H/16, W/16] fp32, normalized posterior mean

Spatial tiling and the blend are the host's (`tile_plan` / `tiled_encode`
in h3_ref2va_prep.py are the reference; the latent blend is linear with weights summing to
1, so blending normalized latents equals comfy's blend-then-normalize).

Usage:
  python h3_export_fl2va_vae_encoder.py --ckpt .../minimax_h3_video_vae_fp16.safetensors \\
      --out ~/models/h3_ref2va/vae_enc/vae_enc.onnx [--reference ref.npz [--pixels P.f32 --size H,W]]

`--reference` runs the fp32 graph module AND a literal 5-D port of comfy's
encode (Conv3d with causal_zero, temporal-isolated GroupNorm, quant_conv,
latent normalization) on the same tile, and reports their cosine.
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

CH, CH_MULT = 128, (1, 2, 2, 4, 4, 8)
SPACE_DOWN = (2, 2, 2, 2, 1, 1)
TIME_DOWN = (1, 2, 2, 1, 1, 1)
NUM_RES = 2
Z = 24
GN_GROUPS, GN_EPS = 32, 1e-6
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def load_encoder_sd(path):
    from safetensors import safe_open
    sd = {}
    with safe_open(path, "pt") as f:
        for k in f.keys():
            if k.startswith(("encoder.", "quant_conv.", "latents_")):
                sd[k] = f.get_tensor(k).float()
    return sd


def levels():
    mid = [CH * m for m in CH_MULT]
    return [mid[0]] + mid[:-1], mid


# ---------------------------------------------------------------- reference
# Literal port of comfy's MiniMaxH3VideoVAE encode path for a [B, 3, T, H, W]
# clip in [-1, 1] (only T == 1 is exercised by Ref2VA).

def _ref_conv(sd, p, x, pad=(0, 0, 0), stride=1, spatial_pad=None):
    w, b = sd[p + ".weight"], sd[p + ".bias"]
    pad_t, pad_h, pad_w = pad
    if spatial_pad is None:
        spatial_pad = (pad_w, pad_w, pad_h, pad_h)
    front = 0 if x.shape[2] == 1 else pad_t * 2
    if any(spatial_pad):
        x = F.pad(x, (*spatial_pad, 0, 0), mode="reflect")
    if front:
        x = F.pad(x, (0, 0, 0, 0, front, 0))
    if x.shape[2] == 1 and pad_t:
        w = w[:, :, -x.shape[2]:]                    # autopad="causal_zero"
    return F.conv3d(x, w, b, stride)


def _ref_gn_silu(sd, p, x):
    b, c, t, h, w = x.shape
    y = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, 1, h, w)
    y = F.group_norm(y, GN_GROUPS, sd[p + ".weight"], sd[p + ".bias"], GN_EPS)
    return F.silu(y.view(b, t, c, h, w).permute(0, 2, 1, 3, 4))


def ref_moments(sd, x):
    """normalized pixels [B, 3, T, H, W] -> moments [B, 48, T', H/16, W/16]."""
    bin_, bmid = levels()
    h = _ref_conv(sd, "encoder.conv_in", x, (1, 1, 1))
    for i in range(len(CH_MULT)):
        for j in range(NUM_RES):
            p = f"encoder.down.{i}.block.{j}"
            cin = bin_[i] if j == 0 else bmid[i]
            r = _ref_conv(sd, p + ".conv1", _ref_gn_silu(sd, p + ".norm1", h), (1, 1, 1))
            s = _ref_conv(sd, p + ".nin_shortcut", h) if cin != bmid[i] else h
            h = _ref_conv(sd, p + ".conv2", _ref_gn_silu(sd, p + ".norm2", r), (1, 1, 1)) + s
        if SPACE_DOWN[i] * TIME_DOWN[i] > 1:
            p = f"encoder.down.{i}.downsample.conv"
            st = (TIME_DOWN[i], SPACE_DOWN[i], SPACE_DOWN[i])
            sp = (0, 1, 0, 1) if SPACE_DOWN[i] == 2 else None
            h = _ref_conv(sd, p, h, (1, 0, 0), st, sp)
    h = _ref_conv(sd, "encoder.conv_out", _ref_gn_silu(sd, "encoder.norm_out", h), (1, 1, 1))
    return F.conv3d(h, sd["quant_conv.weight"], sd["quant_conv.bias"])


def ref_encode_tile(sd, px01):
    """[B, 3, H, W] pixels in [0, 1] -> normalized latent [B, 24, H/16, W/16]."""
    x = (px01 * 2.0 - 1.0).unsqueeze(2)                       # process_input
    m = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1, 1)
    s = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1, 1)
    x = (x + 1.0) * 0.5
    x = (x - m) / s                                           # _normalize_pixels
    mom = ref_moments(sd, x)[:, :, -1:]
    mean = mom[:, :Z]
    lm = sd["latents_mean"].view(1, -1, 1, 1, 1)
    ls = sd["latents_std"].view(1, -1, 1, 1, 1)
    return ((mean - lm) / ls)[:, :, 0]


# ---------------------------------------------------------------- tiling
# comfy's split_tiles / blend / tiled_encode are host-side: h3_ref2va_prep.py
# (tile_plan, tiled_encode).

# ---------------------------------------------------------------- graph

def P(t):
    return nn.Parameter(t.contiguous(), requires_grad=False)


class ReflectPad(torch.autograd.Function):
    """Reflect pad of H/W with STATIC pads (F.pad traces a dynamic pads chain)."""

    @staticmethod
    def forward(ctx, x, t, b, l, r):
        return F.pad(x, (l, r, t, b), mode="reflect")

    @staticmethod
    def symbolic(g, x, t, b, l, r):
        pads = g.op("Constant", value_t=torch.tensor([0, 0, t, l, 0, 0, b, r],
                                                     dtype=torch.int64))
        return g.op("Pad", x, pads, mode_s="reflect")


class Conv(nn.Module):
    """3x3 (reflect pad 1) or 1x1 conv on the last temporal tap."""

    def __init__(self, w, b, stride=1, pad=(1, 1, 1, 1)):
        super().__init__()
        self.w, self.b = P(w), P(b)
        self.stride, self.pad = stride, pad

    def forward(self, x):
        if any(self.pad):
            x = ReflectPad.apply(x, *self.pad)
        return F.conv2d(x, self.w, self.b, self.stride)


def tap(sd, p):
    return sd[p + ".weight"][:, :, -1], sd[p + ".bias"]


class GNSiLU(nn.Module):
    def __init__(self, sd, p):
        super().__init__()
        self.w, self.b = P(sd[p + ".weight"]), P(sd[p + ".bias"])

    def forward(self, x):
        return F.silu(F.group_norm(x, GN_GROUPS, self.w, self.b, GN_EPS))


class Res(nn.Module):
    def __init__(self, sd, p, cin, cout):
        super().__init__()
        self.n1, self.n2 = GNSiLU(sd, p + ".norm1"), GNSiLU(sd, p + ".norm2")
        self.c1, self.c2 = Conv(*tap(sd, p + ".conv1")), Conv(*tap(sd, p + ".conv2"))
        self.sc = Conv(*tap(sd, p + ".nin_shortcut"), pad=(0, 0, 0, 0)) if cin != cout else None

    def forward(self, x):
        h = self.c1(self.n1(x))
        s = self.sc(x) if self.sc is not None else x
        return self.c2(self.n2(h)) + s


class Encoder2D(nn.Module):
    def __init__(self, sd):
        super().__init__()
        bin_, bmid = levels()
        w, b = tap(sd, "encoder.conv_in")
        m = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
        s = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
        self.conv_in = Conv(w / s, b - (w * m / s).sum((1, 2, 3)))
        self.blocks = nn.ModuleList()
        for i in range(len(CH_MULT)):
            for j in range(NUM_RES):
                cin = bin_[i] if j == 0 else bmid[i]
                self.blocks.append(Res(sd, f"encoder.down.{i}.block.{j}", cin, bmid[i]))
            if SPACE_DOWN[i] * TIME_DOWN[i] > 1:
                # time stride is moot at T = 1; spatial: reflect pad right/bottom 1
                assert SPACE_DOWN[i] == 2
                self.blocks.append(Conv(*tap(sd, f"encoder.down.{i}.downsample.conv"),
                                        stride=2, pad=(0, 1, 0, 1)))
        self.norm_out = GNSiLU(sd, "encoder.norm_out")
        wo, bo = tap(sd, "encoder.conv_out")                 # [48, 1024, 3, 3]
        wq = sd["quant_conv.weight"].reshape(2 * Z, 2 * Z)[:Z]
        bq = sd["quant_conv.bias"][:Z]
        lm, ls = sd["latents_mean"], sd["latents_std"]
        w2 = torch.einsum("om,mikl->oikl", wq.double(), wo.double()) / ls.double().view(-1, 1, 1, 1)
        b2 = (wq.double() @ bo.double() + bq.double() - lm.double()) / ls.double()
        self.conv_out = Conv(w2.float(), b2.float())

    def forward(self, pixels):
        h = self.conv_in(pixels)
        for blk in self.blocks:
            h = blk(h)
        return self.conv_out(self.norm_out(h))


def cos(a, b):
    a, b = a.double().flatten(), b.double().flatten()
    return float((a @ b) / (a.norm() * b.norm() + 1e-30))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--size", default="256,256", help="H,W of the trace/reference tile")
    ap.add_argument("--reference", default=None,
                    help="write pixels/latent of the fp32 graph module here and "
                         "check it against the literal comfy port")
    ap.add_argument("--pixels", default=None,
                    help="reference input: raw f32 [3, H, W] in [0, 1] (else seeded "
                         "smooth noise); with --size it is cropped to the top-left tile")
    args = ap.parse_args(argv)
    torch.set_grad_enabled(False)
    sd = load_encoder_sd(args.ckpt)
    model = Encoder2D(sd).eval()
    H, W = (int(v) for v in args.size.split(","))

    if args.pixels:
        raw = np.fromfile(args.pixels, np.float32)
        side = int(round(math.sqrt(raw.size / 3)))
        import json
        meta = os.path.splitext(args.pixels)[0] + ".json"
        if os.path.exists(meta):
            j = json.load(open(meta))
            ph, pw = j["th"], j["tw"]
        else:
            ph = pw = side
        px = raw.reshape(3, ph, pw)[:, :H, :W]
    else:
        g = torch.Generator().manual_seed(0)
        px = torch.rand(1, 3, H // 16, W // 16, generator=g)
        px = F.interpolate(px, size=(H, W), mode="bilinear", align_corners=False)
        px = (px + 0.05 * torch.rand(1, 3, H, W, generator=g)).clamp(0, 1)[0].numpy()
    x = torch.from_numpy(np.ascontiguousarray(px, np.float32))[None]

    if args.reference:
        y = model(x)
        r = ref_encode_tile(sd, x)
        print(f"[ref] graph module vs literal comfy port: cos {cos(y, r):.7f}, "
              f"max|d| {float((y - r).abs().max()):.2e}, latent rms {float(r.std()):.3f}")
        np.savez(args.reference, pixels=x.numpy(), latent=y.numpy())

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.onnx.export(model, (x,), args.out, input_names=["pixels"], output_names=["latent"],
                      dynamic_axes={"pixels": {0: "n", 2: "h", 3: "w"},
                                    "latent": {0: "n", 2: "h16", 3: "w16"}},
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
