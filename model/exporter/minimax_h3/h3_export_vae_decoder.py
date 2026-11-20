"""Export the MiniMax-H3 video VAE decoder to ONNX.

`AutoencoderKLMiniMaxH3` is not a convolutional autoencoder in the usual sense:
the encoder is a causal 3-D CNN, but the **decoder is a ViT**
(`MiniMaxH3VideoViTDecoder3d`) -- every latent voxel becomes a token, 36
transformer blocks attend over them, and a linear patch projection expands each
token into a `4 x 16 x 16` pixel block. That is why this decoder exports
cleanly where a `ConvTranspose` stack would not (the runtime has no
`ConvTranspose` kernel at all).

Only the decoder is exported. The encoder exists in the checkpoint but a
T2VA-style pipeline only needs `decode`; a conditioning-image path that needs
`encode` is a separate component and would need its own op audit (it is the
causal Conv3d + GroupNorm stack).

Host vs device
--------------
* `post_quant_conv`, the ViT decoder and the `(t,h,w) -> pixel` reshuffle are
  in the graph.
* The `latents_mean/std` denormalization (`latents * std + mean`) is one
  elementwise line and stays on the host, as does the ImageNet
  un-normalization of the RGB output and the spatial tiling loop
  (`use_tiling` splits the canvas into 256-px tiles and blends the overlaps --
  Python control flow over a variable tile count).
* `AutoencoderKLMiniMaxH3._decode` also chunks over *time*: latents are decoded
  in groups of `tokens_chunk_size` frames with a cross-fade over
  `frame_overlap`. That chunking is host-side too; this graph decodes one
  chunk.

Two changes to the reference decoder, both forced by the exporter
------------------------------------------------------------------
1. **The rotary position grid is a graph input.** The reference builds it
   in-graph from `torch.arange(0.5, size)` per axis. Tracing `.tile(2)` on a
   dynamically-shaped tensor makes the TorchScript exporter emit an `If`
   (control flow) plus a `Tile`, and the converter rejects control flow
   outright. The grid depends only on the latent's `(F, H, W)`, which the host
   knows, and the graph must not re-derive it anyway. `build_position_grid`
   below is the exact reference formula, hoisted out.
2. **The register tokens are broadcast-added instead of `.expand`ed.**
   `register_tokens.expand(B, -1, -1)` traces to `Expand` on a target shape
   built from `Shape`+`ConstantOfShape`, and the converter has no
   `ConstantOfShape` kernel. `register_tokens + zeros_like(x[:, :1])` is the
   same broadcast without the shape chain; the `zeros_like` is itself written
   as `x[:, :1] * 0` so it does not reintroduce one.

Both are bit-exact (they only change how the same values are computed).
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn

from h3_export_common import (
    export_graph,
    meta_safe_buffers,
    patch_rmsnorm,
    try_convert,
)

TINY = dict(
    in_channels=24,
    out_channels=3,
    patch_size=16,
    patch_size_t=4,
    num_layers=2,
    num_attention_heads=2,
    attention_head_dim=16,
    num_register_tokens=4,
    ffn_mult=4,
    rope_theta=100.0,
    rope_dim_ratio=0.75,
    norm_eps=1e-5,
)


# ---------------------------------------------------------------------------
# hoisted pieces of the reference forward
# ---------------------------------------------------------------------------

def build_position_grid(num_frames: int, height: int, width: int,
                        num_suffix: int) -> torch.Tensor:
    """The reference `MiniMaxH3VideoViTDecoder3d.forward` position grid.

    Each axis is length-normalized to `[-1, 1)` and scaled by `2*pi` inside the
    rotary module. The `num_suffix` register + CLS tokens all sit at position 0.
    Hoisted verbatim from `autoencoder_kl_minimax_h3.py`; the host computes it
    because tracing the arange-based construction produces `If` nodes.
    """
    grids = [
        2.0 * (torch.arange(0.5, size, dtype=torch.float32) / size) - 1.0
        for size in (num_frames, height, width)
    ]
    position_ids = torch.stack(torch.meshgrid(*grids, indexing="ij"),
                               dim=-1).flatten(0, 2)
    position_ids = position_ids.unsqueeze(0)
    suffix_ids = position_ids.new_zeros((1, num_suffix, 3))
    return torch.cat([position_ids, suffix_ids], dim=1)


class H3VideoRopeOnnx(nn.Module):
    """`MiniMaxH3VideoRotaryPosEmbed` with `.tile(2)` written as a concat.

    `.tile(2)` on a `(B, S, D)` tensor doubles the LAST axis, i.e.
    `cat([x, x], -1)`; the exporter's `aten::tile` symbolic wraps it in an
    `If`/`Tile` pair that the converter refuses.
    """

    def __init__(self, rope):
        super().__init__()
        import math

        self._two_pi = 2.0 * math.pi
        self.register_buffer("inv_freq", rope.inv_freq, persistent=False)

    def forward(self, position_ids):
        angles = self._two_pi * position_ids[:, :, :, None] * self.inv_freq[None, None, None, :]
        angles = angles.flatten(2, 3)
        angles = torch.cat([angles, angles], dim=-1).unsqueeze(2)
        return angles.cos(), angles.sin()


# ---------------------------------------------------------------------------
# the export wrapper
# ---------------------------------------------------------------------------

class H3VideoDecoderOnnx(nn.Module):
    """`MiniMaxH3VideoViTDecoder3d` with a host-supplied position grid.

    Inputs:
      latents        (B, latent_channels, F, H, W)   denormalized latents
      position_ids   (B, F*H*W + R + 1, 3)           host-built rotary grid

    Output:
      video          (B, out_channels, F*pt, H*ps, W*ps)
    """

    INPUT_NAMES = ["latents", "position_ids"]

    def __init__(self, decoder, *, onesided_shuffle: bool = False):
        super().__init__()
        self.d = decoder
        self.rope = H3VideoRopeOnnx(decoder.rope)

    def forward(self, latents, position_ids):
        d = self.d
        batch, channels, frames, height, width = latents.shape

        x = latents.permute(0, 2, 3, 4, 1).reshape(
            batch, frames * height * width, channels)
        x = d.proj_in(x)
        num_patches = x.shape[1]

        # Register + CLS tokens. Broadcast-add instead of `.expand`: the Expand
        # target shape is built from Shape/ConstantOfShape, which the converter
        # does not have. `* 0` gives zeros with no shape chain either.
        zero_row = x[:, :1, :] * 0
        registers = d.register_tokens.to(x.dtype) + zero_row
        x = torch.cat([x, registers, zero_row], dim=1)

        rotary = self.rope(position_ids)
        for block in d.transformer_blocks:
            x = block(x, rotary)

        x = d.norm_out(x).to(d.proj_out.weight.dtype)
        x = d.proj_out(x)[:, :num_patches, :]

        ps, pst = d.patch_size, d.patch_size_t
        x = x.view(batch, frames, height, width, d.out_channels, pst, ps, ps)
        x = x.permute(0, 4, 1, 5, 2, 6, 3, 7).contiguous()
        return x.reshape(batch, d.out_channels, frames * pst,
                         height * ps, width * ps)


# ---------------------------------------------------------------------------
# build + dummies
# ---------------------------------------------------------------------------

def build_decoder(cfg: dict, *, device: torch.device, tiny: bool):
    from diffusers.models.autoencoders.autoencoder_kl_minimax_h3 import (
        MiniMaxH3VideoViTDecoder3d,
    )

    if tiny:
        kwargs = dict(TINY)
    else:
        # The decoder's constructor takes the *derived* geometry; the top-level
        # VAE config states the compression factors and `AutoencoderKLMiniMaxH3`
        # multiplies them out. Reproduce that derivation here so this script
        # reads the shipped config and not a hand-copied number.
        import math

        kwargs = dict(
            in_channels=int(cfg["latent_channels"]),
            out_channels=int(cfg["out_channels"]),
            patch_size=math.prod(int(v) for v in cfg["spatial_downsample_factors"]),
            patch_size_t=math.prod(int(v) for v in cfg["temporal_downsample_factors"]),
            num_layers=int(cfg["decoder_num_layers"]),
            num_attention_heads=int(cfg["decoder_num_attention_heads"]),
            attention_head_dim=int(cfg["decoder_attention_head_dim"]),
            num_register_tokens=int(cfg["decoder_num_register_tokens"]),
            ffn_mult=int(cfg["decoder_ffn_mult"]),
            rope_theta=float(cfg["decoder_rope_theta"]),
            rope_dim_ratio=float(cfg["decoder_rope_dim_ratio"]),
            norm_eps=float(cfg["decoder_norm_eps"]),
        )
    if device.type == "meta":
        with torch.device("meta"):
            decoder = MiniMaxH3VideoViTDecoder3d(**kwargs)
    else:
        decoder = MiniMaxH3VideoViTDecoder3d(**kwargs)
    return decoder.eval(), kwargs


def dummy_inputs(decoder, *, device, frames=2, height=4, width=4):
    c = int(decoder.proj_in.weight.shape[1])
    latents = torch.randn(1, c, frames, height, width)
    grid = build_position_grid(frames, height, width,
                               decoder.num_register_tokens + 1)
    return (latents.to(device), grid.to(device))


def dynamic_axes() -> dict:
    """Honest axes for the ViT decoder.

    Symbolic: the latent frame/height/width axes and the derived pixel-frame /
    pixel-height / pixel-width outputs, plus the token count on `position_ids`.
    A different clip length or canvas changes all of them.

    Static, deliberately:
      * the batch axis. The decoder's `register_tokens` broadcast is the only
        place batch appears, and leaving it static is what keeps the graph free
        of the `Shape`/`ConstantOfShape` chain (see the module docstring). The
        reference decodes one clip per call and the host's tiling/time-chunk
        loop is what would otherwise vary it. Making it dynamic is possible but
        needs a converter-side `ConstantOfShape`, so it is NOT claimed here.
      * `latent_channels` -- a weight shape;
      * `position_ids`' last axis (3).
    """
    return {
        "latents": {2: "frames", 3: "height", 4: "width"},
        "position_ids": {1: "num_tokens"},
        "video": {2: "pixel_frames", 3: "pixel_height", 4: "pixel_width"},
    }


def load_config(model_path: str) -> dict:
    import json

    with open(os.path.join(model_path, "vae", "config.json")) as f:
        return json.load(f)


def run(args) -> int:
    dry = bool(args.dry_run)
    if args.tiny:
        cfg = dict(TINY)
    elif args.model is not None:
        cfg = load_config(args.model)
    else:
        raise SystemExit(
            "error: --dry-run needs either --model <dir> (for vae/config.json) "
            "or --tiny"
        )

    device = torch.device("meta") if dry else torch.device("cpu")
    decoder, kwargs = build_decoder(cfg, device=device, tiny=args.tiny)
    print(f"[build] {'TINY ' if args.tiny else ''}MiniMaxH3VideoViTDecoder3d "
          f"device={device}")
    print(f"[build] " + " ".join(f"{k}={v}" for k, v in kwargs.items()))

    patch_rmsnorm(decoder)   # the QK norms are RMSNorm over head_dim
    if dry:
        n = meta_safe_buffers(decoder)
        print(f"[meta] {n} computed buffer(s) promoted to graph inputs")

    wrapper = H3VideoDecoderOnnx(decoder).eval()
    dummies = dummy_inputs(decoder, device=device)
    out = args.out or "vae_decoder.onnx"

    export_graph(
        wrapper, dummies, out,
        input_names=H3VideoDecoderOnnx.INPUT_NAMES,
        output_names=["video"],
        dynamic_axes=dynamic_axes(),
        dry_run=dry,
    )
    if dry:
        print("[dry-run] no weight bytes were read; the graph above is the "
              "inventory the real export produces.")
        if args.no_convert:
            print("[dry-run] --no-convert: skipping onnx2vkop")
            return 0
        return 0 if try_convert(out, os.path.dirname(os.path.abspath(out))) == 0 else 5
    return 0
