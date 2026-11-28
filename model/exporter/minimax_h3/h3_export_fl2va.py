#!/usr/bin/env python3
"""MiniMax-H3 FL2VA / Ref2VA denoiser (ComfyUI single-file checkpoint) -> ONNX.

The FL2VA and Ref2VA pipelines use the older single-stream DiT
(`blocks.N.attn.qkv_proj`, `token_refiner`, `final_layer.video_out`) that no
released diffusers ships. ComfyUI implements it in `comfy/ldm/minimax/model.py`
(class MiniMaxH3Model); this file is a pure-torch port of that forward, written
so it traces to plain ONNX ops on `meta` and streams the real weights from the
Comfy-Org repack (`diffusion_models/minimax_h3_{fl2va,ref2va}[_pruned]_bf16`).

One graph = one denoising step. The host owns everything that indexes or
loops (see example/minimax_h3/README.md "FL2VA"):

  graph inputs
    video_rows   [Nv_all, 96]  fp32  patchified video latents (cond rows first,
                                     in the order PackedLayout packs them)
    audio_rows   [Na_all, 32]  fp32  channel-major stereo audio rows
    text_states  [L, 5120]     fp32  Qwen3-VL layer-50 hidden states
    pack_index   [S]           int64 row of concat([text, video, audio]) that
                                     lands at packed position s
    t_emb        [U, 8]        fp32  adaln_t_table lerp at each distinct t
    mod_index    [S]           int64 adaln mod row (t_row * 3 + modality tag)
    rope_cos     [S, 48]       fp32  cos of the split-half angles
    rope_sin     [S, 48]       fp32  sin of the split-half angles
    video_index  [Nv]          int64 packed rows of the target video segment
    video_t      [Nv]          int64 t_emb row of each of them
    audio_index  [Na]          int64 packed rows of the target audio segment
    audio_t      [Na]          int64 t_emb row of each of them
  graph outputs
    video_v      [Nv, 96]      fp32  head output (the model returns -v)
    audio_v      [Na, 32]      fp32

Numerics follow ComfyUI's bf16 path: residual stream, attention and MLP in the
model dtype (bf16), adaLN projections in fp32 (curve-form checkpoints keep
them fp32 and do not apply SiLU), patch projections and output heads in fp32.

Usage (venv with torch + onnx, this checkout's model/pypi on the path):

  python h3_export_fl2va.py --ckpt .../minimax_h3_fl2va_pruned_bf16.safetensors \
      --out ~/models/h3_fl2va/dit.onnx [--num-layers 2] [--reference ref.npz]
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

CFG = dict(hidden=5376, heads=56, head_dim=128, ffn=14336, text_dim=5120,
           t_dim=8, video_dim=96, audio_dim=32, num_layers=50,
           refiner_layers=2, eps=1e-5, rope_half=48)


def rms_norm(x, w, eps):
    # comfy: F.rms_norm in the input dtype; computed in fp32 and rounded back,
    # which is what the fused kernels do internally.
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


def hadamard(g: int) -> np.ndarray:
    """comfy-kitchen ConvRot's regular Hadamard: kron powers of a symmetric 4x4,
    normalized. Symmetric and orthogonal, so H @ H = I and
    x @ W.T == (x @ H) @ (W @ H).T for a block-diagonal H over the K dim."""
    h4 = np.array([[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]],
                  np.float32)
    h = h4
    while h.shape[0] < g:
        h = np.kron(h, h4)
    if h.shape[0] != g:
        raise ValueError(f"convrot group must be a power of 4, got {g}")
    return h / np.float32(math.sqrt(g))


def rotate_in(w: np.ndarray, h: np.ndarray) -> np.ndarray:
    """[N, K] weight -> W @ blockdiag(H) along K (H symmetric)."""
    n, k = w.shape
    g = h.shape[0]
    return (w.reshape(n, k // g, g) @ h).reshape(n, k)


class LoraMerge:
    """ComfyUI generic LoRA (`diffusion_model.<module>.lora_{A,B}.weight`,
    scalar `.alpha`) merged into the base weight while it streams:
    W += strength * alpha / rank * B @ A (rank = A.shape[0]; the repack's
    fused qkv carries a concatenated A and 3x alpha, so the ratio holds)."""

    def __init__(self, path: str, strength: float = 1.0):
        from safetensors import safe_open
        self.f = safe_open(path, "pt")
        self.strength = strength
        keys = set(self.f.keys())
        self.targets = sorted(k[len("diffusion_model."):-len(".lora_A.weight")]
                              for k in keys if k.endswith(".lora_A.weight"))
        self.has_alpha = all(f"diffusion_model.{t}.alpha" in keys
                             for t in self.targets)

    def delta(self, module: str) -> np.ndarray:
        pre = f"diffusion_model.{module}"
        a = self.f.get_tensor(pre + ".lora_A.weight").float()
        b = self.f.get_tensor(pre + ".lora_B.weight").float()
        # no alpha: already folded into B (the Pika adapters)
        scale = (self.f.get_tensor(pre + ".alpha").item() / a.shape[0]
                 if self.has_alpha else 1.0)
        return (b @ a).numpy() * np.float32(self.strength * scale)


class _Rot:
    """Online activation rotation x -> x @ blockdiag(H), H shared by the whole
    model (`FL2VADiT.convrot_h`). A plain object, so assigning it to a module
    does not register the owner as a submodule."""

    def __init__(self, owner):
        self.owner = owner

    def __call__(self, x):
        h = self.owner.convrot_h
        g = h.shape[0]
        return torch.matmul(x.reshape(-1, g), h).reshape(x.shape)


def _ident(x):
    return x


def linear(dim_in, dim_out, bias, dtype, device):
    m = nn.Linear(dim_in, dim_out, bias=bias, dtype=dtype, device=device)
    m.requires_grad_(False)
    return m


class Attention(nn.Module):
    def __init__(self, c, dtype, device):
        super().__init__()
        self.heads, self.head_dim = c["heads"], c["head_dim"]
        inner = self.heads * self.head_dim
        self.qkv_proj = linear(c["hidden"], inner * 3, False, dtype, device)
        self.q_norm = RMSNorm(self.head_dim, c["eps"], dtype, device)
        self.k_norm = RMSNorm(self.head_dim, c["eps"], dtype, device)
        self.out_proj = linear(inner, c["hidden"], False, dtype, device)
        self.half = c["rope_half"]
        self.rot = _ident

    def rope(self, x, cos, sin):
        # comfy-kitchen rms_rope_split_half, rot_dim = 2*half: pair i is
        # (x[i], x[i+half]) over the first 2*half dims; the rest pass through.
        h = self.half
        a, b, rest = x[..., :h], x[..., h:2 * h], x[..., 2 * h:]
        return torch.cat((a * cos - b * sin, b * cos + a * sin, rest), dim=-1)

    def forward(self, x, cos=None, sin=None):
        s = x.shape[0]
        inner = self.heads * self.head_dim
        qkv = self.qkv_proj(self.rot(x))
        q, k, v = qkv[:, :inner], qkv[:, inner:2 * inner], qkv[:, 2 * inner:]
        q = self.q_norm(q.reshape(s, self.heads, self.head_dim))
        k = self.k_norm(k.reshape(s, self.heads, self.head_dim))
        v = v.reshape(s, self.heads, self.head_dim)
        if cos is not None:
            q = self.rope(q, cos, sin)
            k = self.rope(k, cos, sin)
        q = q.transpose(0, 1).unsqueeze(0)          # [1, H, S, D]
        k = k.transpose(0, 1).unsqueeze(0)
        v = v.transpose(0, 1).unsqueeze(0)
        scores = torch.matmul(q, k.transpose(-1, -2)) * (1.0 / math.sqrt(self.head_dim))
        prob = torch.softmax(scores, dim=-1)
        out = torch.matmul(prob, v)                  # [1, H, S, D]
        out = out.squeeze(0).transpose(0, 1).reshape(s, inner)
        return self.out_proj(self.rot(out))


class MLP(nn.Module):
    def __init__(self, c, dtype, device):
        super().__init__()
        self.ffn = c["ffn"]
        self.fc1 = linear(c["hidden"], c["ffn"] * 2, False, dtype, device)
        self.fc2 = linear(c["ffn"], c["hidden"], False, dtype, device)
        self.rot = _ident

    def forward(self, x):
        h = self.fc1(self.rot(x))
        gate, up = h[:, :self.ffn], h[:, self.ffn:]
        return self.fc2(self.rot(F.silu(gate) * up))


class AdalnProj(nn.Module):
    """Curve-form adaLN: fp32 linear over the t_emb basis, no SiLU."""

    def __init__(self, c, expand, modalities, device, wide=False):
        super().__init__()
        self.expand, self.modalities, self.hidden = expand, modalities, c["hidden"]
        # stored fp16 in the checkpoint; comfy runs the projection in fp32
        self.linear = linear(c["t_dim"], expand * c["hidden"] * modalities, True,
                             torch.float32 if wide else torch.float16, device)

    def forward(self, t_emb):
        w = self.linear.weight.float()
        b = self.linear.bias.float()
        x = torch.matmul(t_emb, w.t()) + b
        x = x.reshape(-1, self.expand * self.hidden)  # [U*modalities, expand*H]
        return [x[:, i * self.hidden:(i + 1) * self.hidden]
                for i in range(self.expand)]


class RefinerBlock(nn.Module):
    def __init__(self, c, dtype, device):
        super().__init__()
        self.norm1 = RMSNorm(c["hidden"], c["eps"], dtype, device)
        self.norm2 = RMSNorm(c["hidden"], c["eps"], dtype, device)
        self.attn = Attention(c, dtype, device)
        self.mlp = MLP(c, dtype, device)

    def forward(self, x):
        x = self.attn(self.norm1(x)) + x
        return self.mlp(self.norm2(x)) + x


class TokenRefiner(nn.Module):
    def __init__(self, c, dtype, device):
        super().__init__()
        self.blocks = nn.ModuleList(RefinerBlock(c, dtype, device)
                                    for _ in range(c["refiner_layers"]))
        self.final_norm = RMSNorm(c["hidden"], c["eps"], dtype, device)

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return self.final_norm(x)


class DiTBlock(nn.Module):
    def __init__(self, c, dtype, device):
        super().__init__()
        self.norm1 = RMSNorm(c["hidden"], c["eps"], dtype, device)
        self.norm2 = RMSNorm(c["hidden"], c["eps"], dtype, device)
        self.attn = Attention(c, dtype, device)
        self.mlp = MLP(c, dtype, device)
        self.adaln_proj = AdalnProj(c, 6, 3, device, dtype == torch.float32)

    def forward(self, x, t_emb, mod_index, cos, sin):
        dt = x.dtype
        mods = [torch.index_select(m.to(dt), 0, mod_index)
                for m in self.adaln_proj(t_emb)]
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mods
        h = self.norm1(x) * (1.0 + scale_msa) + shift_msa
        x = x + self.attn(h, cos, sin) * gate_msa
        h = self.norm2(x) * (1.0 + scale_mlp) + shift_mlp
        return x + self.mlp(h) * gate_mlp


class FinalLayer(nn.Module):
    def __init__(self, c, dtype, device):
        super().__init__()
        self.norm = RMSNorm(c["hidden"], c["eps"], dtype, device)
        self.adaln_proj = AdalnProj(c, 2, 1, device, dtype == torch.float32)
        self.video_out = linear(c["hidden"], c["video_dim"], True, torch.float32, device)
        self.audio_out = linear(c["hidden"], c["audio_dim"], True, torch.float32, device)

    def head(self, x, rows, t_rows, shift, scale, out):
        h = self.norm(torch.index_select(x, 0, rows))
        sc = torch.index_select(scale, 0, t_rows)
        sh = torch.index_select(shift, 0, t_rows)
        return out(h.float() * (1.0 + sc) + sh)

    def forward(self, x, t_emb, video_index, video_t, audio_index, audio_t):
        shift, scale = self.adaln_proj(t_emb)
        return (self.head(x, video_index, video_t, shift, scale, self.video_out),
                self.head(x, audio_index, audio_t, shift, scale, self.audio_out))


class FL2VADiT(nn.Module):
    def __init__(self, c=CFG, dtype=torch.bfloat16, device="meta"):
        super().__init__()
        self.c = c
        self.dtype = dtype
        self.video_patch_proj = linear(c["video_dim"], c["hidden"], True, torch.float32, device)
        self.audio_patch_proj = linear(c["audio_dim"], c["hidden"], True, torch.float32, device)
        self.condition_proj = linear(c["text_dim"], c["hidden"], True, dtype, device)
        self.token_refiner = TokenRefiner(c, dtype, device)
        self.blocks = nn.ModuleList(DiTBlock(c, dtype, device)
                                    for _ in range(c["num_layers"]))
        self.final_layer = FinalLayer(c, dtype, device)
        g = c.get("convrot", 0)
        if g:
            # ConvRot-style: every attention/MLP linear sees x @ H; its weight
            # is rotated to W @ H while streaming (`convrot_transforms`).
            self.register_buffer("convrot_h", torch.empty(g, g, dtype=dtype,
                                                          device=device))
            for m in self.modules():
                if isinstance(m, (Attention, MLP)):
                    m.rot = _Rot(self)

    def convrot_names(self):
        """Checkpoint names of the weights the online rotation applies to."""
        out = []
        for name, m in self.named_modules():
            if isinstance(m, Attention):
                out += [f"{name}.qkv_proj.weight", f"{name}.out_proj.weight"]
            elif isinstance(m, MLP):
                out += [f"{name}.fc1.weight", f"{name}.fc2.weight"]
        return out

    def forward(self, video_rows, audio_rows, text_states, pack_index, t_emb,
                mod_index, rope_cos, rope_sin, video_index, video_t,
                audio_index, audio_t):
        dt = self.dtype
        video_e = self.video_patch_proj(video_rows).to(dt)
        audio_e = self.audio_patch_proj(audio_rows).to(dt)
        text_h = self.token_refiner(self.condition_proj(text_states.to(dt)))
        h = torch.index_select(torch.cat((text_h, video_e, audio_e), 0), 0,
                               pack_index)
        # comfy builds the rotation table in the model dtype
        cos = rope_cos.to(dt).unsqueeze(1)          # [S, 1, 48]
        sin = rope_sin.to(dt).unsqueeze(1)
        for blk in self.blocks:
            h = blk(h, t_emb, mod_index, cos, sin)
        return self.final_layer(h, t_emb, video_index, video_t, audio_index,
                                audio_t)


INPUT_NAMES = ["video_rows", "audio_rows", "text_states", "pack_index",
               "t_emb", "mod_index", "rope_cos", "rope_sin", "video_index",
               "video_t", "audio_index", "audio_t"]
OUTPUT_NAMES = ["video_v", "audio_v"]
DYNAMIC_AXES = {
    "video_rows": {0: "nv_all"}, "audio_rows": {0: "na_all"},
    "text_states": {0: "text_len"}, "pack_index": {0: "seq"},
    "t_emb": {0: "n_t"}, "mod_index": {0: "seq"}, "rope_cos": {0: "seq"},
    "rope_sin": {0: "seq"}, "video_index": {0: "nv"}, "video_t": {0: "nv"},
    "audio_index": {0: "na"}, "audio_t": {0: "na"},
    "video_v": {0: "nv"}, "audio_v": {0: "na"},
}


# ---------------------------------------------------------------------------
# host side (mirrors comfy PackedLayout / _forward for plain t2va: no keyframes,
# no refs). Used for the dummy trace inputs and the CPU reference; the C++
# driver implements the same thing.
# ---------------------------------------------------------------------------

FRAME_PER_TOKEN = (1, 4, 4, 4, 4)
FRAME_RESCALE = 5.0 / 3.0
SEG_TAG = {"text": 1, "video": 0, "audio": 2}


def _axis_from_sqrt_area(dim, patch, sqrt_area):
    ratio = dim / sqrt_area
    n = dim // patch
    return (np.arange(n, dtype=np.float64) * (ratio / n) + (1.0 - ratio) / 2.0) * 32.0


def time_shift_sigma(sigma, from_shift, to_shift):
    base = sigma / (from_shift + sigma * (1.0 - from_shift))
    return to_shift * base / (1.0 + (to_shift - 1.0) * base)


def host_inputs(text_len, latent_t, lat_h, lat_w, audio_t, sigma_v,
                inv_freq, t_table, rng=None, text_states=None,
                video_latent=None, audio_latent=None):
    """Build every graph input for a t2va step (layout [text | audio | video])."""
    rng = rng or np.random.default_rng(0)
    hh, ww = lat_h // 2, lat_w // 2
    area = math.sqrt(lat_h * lat_w)
    ah, aw = _axis_from_sqrt_area(lat_h, 2, area), _axis_from_sqrt_area(lat_w, 2, area)
    gh, gw = np.meshgrid(ah, aw, indexing="ij")
    frame = np.stack([gh.reshape(-1), gw.reshape(-1)], -1)          # [hh*ww, 2]
    n_frame = frame.shape[0]
    pos = [np.stack([np.arange(text_len, dtype=np.float64),
                     np.zeros(text_len), np.zeros(text_len)], -1)]
    cursor = float(text_len)
    ag = np.zeros((audio_t * 2, 3))
    ag[:, 0] = np.tile(cursor + np.arange(audio_t, dtype=np.float64), 2)
    ag[:audio_t, 2] = aw[0]
    ag[audio_t:, 2] = aw[-1]
    pos.append(ag)
    spans = np.array([FRAME_RESCALE * FRAME_PER_TOKEN[k % 5] for k in range(latent_t)])
    tg = cursor + np.concatenate([[0.0], np.cumsum(spans)[:-1]])
    vg = np.empty((latent_t, n_frame, 3))
    vg[:, :, 0] = tg[:, None]
    vg[:, :, 1:] = frame[None]
    pos.append(vg.reshape(-1, 3))
    position = np.concatenate(pos)                                  # [S, 3]
    S = position.shape[0]
    n_audio, n_video = audio_t * 2, latent_t * n_frame
    seg = [("text", 0, text_len), ("audio", text_len, text_len + n_audio),
           ("video", text_len + n_audio, S)]

    # rope: per axis pos * inv_freq, concatenated (t, h, w) -> 48 angles
    ang = (position.astype(np.float32)[:, :, None] * inv_freq[None, None, :]).reshape(S, -1)
    rope_cos, rope_sin = np.cos(ang).astype(np.float32), np.sin(ang).astype(np.float32)

    t_v = 1.0 - sigma_v
    t_a = 1.0 - time_shift_sigma(sigma_v, 12.0, 3.0)
    seg_t = {"text": t_v, "video": t_v, "audio": t_a}
    unique_t = sorted({t_v, t_a})
    t_row = {t: i for i, t in enumerate(unique_t)}
    mod_index = np.empty(S, np.int64)
    for kind, a, b in seg:
        mod_index[a:b] = t_row[seg_t[kind]] * 3 + SEG_TAG[kind]
    tv = np.asarray(unique_t, np.float32)
    pos_t = np.clip(tv, 0.0, 1.0) * (t_table.shape[0] - 1)
    i0 = np.minimum(np.floor(pos_t).astype(np.int64), t_table.shape[0] - 2)
    frac = (pos_t - i0)[:, None]
    t_emb = (t_table[i0] + (t_table[i0 + 1] - t_table[i0]) * frac).astype(np.float32)

    if video_latent is None:
        video_latent = rng.standard_normal((24, latent_t, lat_h, lat_w)).astype(np.float32)
    # patchify [C, T, H, W] -> [T*hh*ww, C*1*2*2]
    v = video_latent.reshape(24, latent_t, 1, hh, 2, ww, 2)
    video_rows = np.einsum("ctrhpwq->thwcrpq", v).reshape(latent_t * hh * ww, 96)
    if audio_latent is None:
        audio_latent = rng.standard_normal((32, 2, audio_t)).astype(np.float32)
    audio_rows = audio_latent.transpose(1, 2, 0).reshape(2 * audio_t, 32)
    if text_states is None:
        text_states = rng.standard_normal((text_len, CFG["text_dim"])).astype(np.float32)

    # concat order in the graph is [text, video, audio]; packed is [text, audio, video]
    pack = np.concatenate([np.arange(text_len),
                           text_len + n_video + np.arange(n_audio),
                           text_len + np.arange(n_video)]).astype(np.int64)
    return {
        "video_rows": np.ascontiguousarray(video_rows, np.float32),
        "audio_rows": np.ascontiguousarray(audio_rows, np.float32),
        "text_states": np.ascontiguousarray(text_states, np.float32),
        "pack_index": pack,
        "t_emb": t_emb,
        "mod_index": mod_index,
        "rope_cos": rope_cos, "rope_sin": rope_sin,
        "video_index": np.arange(text_len + n_audio, S, dtype=np.int64),
        "video_t": np.full(n_video, t_row[t_v], np.int64),
        "audio_index": np.arange(text_len, text_len + n_audio, dtype=np.int64),
        "audio_t": np.full(n_audio, t_row[t_a], np.int64),
    }


def read_tensor(path, name):
    from safetensors import safe_open
    with safe_open(path, "pt") as f:
        return f.get_tensor(name)


def load_real(model, ckpt):
    """Materialize `model` on CPU with the checkpoint's tensors (reference only;
    only use on a truncated model -- the full one is 40 GB)."""
    from safetensors import safe_open
    sd = {}
    g = model.c.get("convrot", 0)
    rot = set(model.convrot_names()) if g else set()
    with safe_open(ckpt, "pt") as f:
        for name, p in model.state_dict().items():
            if name == "convrot_h":
                sd[name] = torch.from_numpy(hadamard(g)).to(p.dtype)
                continue
            t = f.get_tensor(name)
            if name in rot:
                t = torch.from_numpy(rotate_in(t.float().numpy(), hadamard(g)))
            sd[name] = t.to(p.dtype)
    model.load_state_dict(sd, assign=True)
    return model


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True, help="Comfy-Org bf16 single-file DiT")
    ap.add_argument("--out", required=True, help="output .onnx (weights beside it)")
    ap.add_argument("--num-layers", type=int, default=CFG["num_layers"],
                    help="truncate the DiT stack (numerics tests)")
    ap.add_argument("--reference", default=None,
                    help="also run the torch port on CPU with real weights and "
                         "save inputs/outputs to this .npz (truncated models only)")
    ap.add_argument("--geometry", default="16,2,16,16,8",
                    help="text_len,latent_t,latent_h,latent_w,audio_t for the "
                         "dummy trace / reference")
    ap.add_argument("--sigma", type=float, default=0.7)
    ap.add_argument("--dtype", choices=["bf16", "fp32"], default="fp32",
                    help="compute dtype of the graph. fp32 widens every "
                         "checkpoint tensor while streaming (the runtime has no "
                         "bf16-activation kernel beside quantized weights, and "
                         "H3 overflows fp16); bf16 mirrors ComfyUI exactly")
    ap.add_argument("--convrot", type=int, default=0,
                    help="ConvRot group (power of 4, e.g. 256): rotate every "
                         "attention/MLP linear's input by a block Hadamard, in "
                         "the graph for x and while streaming for W, so a "
                         "later -q int4 quantizes outlier-free weights")
    ap.add_argument("--lora", default=None,
                    help="ComfyUI LoRA (e.g. loras/minimax_h3_fl2v_turbo_8step_*) "
                         "merged into the weights while they stream, before "
                         "--convrot and before the converter quantizes them")
    ap.add_argument("--lora-strength", type=float, default=1.0)
    args = ap.parse_args(argv)

    c = dict(CFG, num_layers=args.num_layers, convrot=args.convrot)
    dtype = torch.float32 if args.dtype == "fp32" else torch.bfloat16
    out_dir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(out_dir, exist_ok=True)

    # host tables the driver needs (tiny, fp32)
    inv_freq = read_tensor(args.ckpt, "rope.inv_freq").float().numpy()
    t_table = read_tensor(args.ckpt, "adaln_t_table").float().numpy()
    inv_freq.tofile(os.path.join(out_dir, "rope_inv_freq.f32"))
    t_table.tofile(os.path.join(out_dir, "adaln_t_table.f32"))
    print(f"[host] rope_inv_freq {inv_freq.shape} adaln_t_table {t_table.shape}")

    geo = [int(v) for v in args.geometry.split(",")]
    inputs = host_inputs(*geo, args.sigma, inv_freq, t_table)

    if args.reference:
        model = load_real(FL2VADiT(c, dtype=dtype, device="cpu"), args.ckpt).eval()
        with torch.no_grad():
            v, a = model(*[torch.from_numpy(inputs[n]) for n in INPUT_NAMES])
        np.savez(args.reference, **inputs, video_v=v.numpy(), audio_v=a.numpy())
        print(f"[ref] wrote {args.reference}: video_v {tuple(v.shape)} "
              f"absmax {v.abs().max():.4f}, audio_v {tuple(a.shape)} "
              f"absmax {a.abs().max():.4f}")
        del model

    import h3_export_common as common
    # SafetensorsIndex reads every *.safetensors in a directory; the Comfy repack
    # keeps a dozen variants with identical names side by side, so expose the
    # one checkpoint through a private directory.
    ck_dir = os.path.join(out_dir, "ckpt", "dit")
    os.makedirs(ck_dir, exist_ok=True)
    link = os.path.join(ck_dir, os.path.basename(args.ckpt))
    for f in os.listdir(ck_dir):
        if os.path.join(ck_dir, f) != link:
            os.unlink(os.path.join(ck_dir, f))
    if not os.path.lexists(link):
        os.symlink(os.path.abspath(args.ckpt), link)

    wrapper = FL2VADiT(c, dtype=dtype, device="meta").eval()
    # every tensor of a weightless trace lives on meta, inputs included
    dummies = tuple(torch.from_numpy(inputs[n]).to("meta") for n in INPUT_NAMES)
    transforms, computed = {}, {}
    lora = LoraMerge(args.lora, args.lora_strength) if args.lora else None
    if lora:
        print(f"[lora] {os.path.basename(args.lora)}: {len(lora.targets)} "
              f"linears, strength {args.lora_strength}")
    if args.convrot:
        import onnx
        h = hadamard(args.convrot)
        transforms = {n: (lambda w, h=h: rotate_in(w, h))
                      for n in wrapper.convrot_names()}
        if dtype == torch.float32:
            hp = onnx.numpy_helper.from_array(h, "convrot_h")
        else:
            # +-1/sqrt(g) is exact in bf16 for a power-of-4 g
            bits = (h.view(np.uint32) >> 16).astype("<u2")
            hp = onnx.helper.make_tensor("convrot_h", onnx.TensorProto.BFLOAT16,
                                         h.shape, bits.tobytes(), raw=True)
        computed = {"convrot_h": hp}
        print(f"[convrot] group {args.convrot}: {len(transforms)} weights rotated")
    if lora:
        # merge first, then rotate: the LoRA delta lives in the original basis
        for t in lora.targets:
            n = t + ".weight"
            then = transforms.get(n, lambda w: w)
            transforms[n] = (lambda w, t=t, then=then:
                             then(w + lora.delta(t)))
    with torch.no_grad():
        common.export_streaming_real(
            wrapper, dummies, INPUT_NAMES, OUTPUT_NAMES, DYNAMIC_AXES,
            model_dir=os.path.join(out_dir, "ckpt"), component="dit",
            out_path=args.out,
            target_dtype="F32" if dtype == torch.float32 else None,
            transforms=transforms, computed=computed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
