"""Export the MiniMax-H3 audio VAE (BigVGAN) decoder -- and fail loudly.

Superseded for the ComfyUI repack: `h3_export_fl2va_audio_vae.py` exports the
same BigVGAN decoder with Exp/Reciprocal folded on the host, the replicate pads
as static Pad(edge) and the 1-D convs as 4-D ones. It converts and runs
(FL2VA.md section 3d). This diffusers-layout script is kept as it was.

**This component cannot be converted today.** The script exists so the gap is
recorded and testable rather than rediscovered, and so `--dry-run` prints the
precise op set that blocks it.

What was measured
-----------------
The audio decoder is BigVGAN (`MiniMaxH3AudioBigVGANDecoder`): seven
`ConvTranspose1d` upsamplers, alias-free SnakeBeta activations (each wrapped in
an upsample/act/downsample triple), and AMP residual blocks. Traced with the
dynamo exporter at the checkpoint's real size it is ~2500 nodes.

The gap is measured on every run, not frozen (`NEEDED_OPS` is the set this
component needs; `check_runtime_ops` decides which are actually missing). When
this file was written the missing set was
`Exp, ReduceL2, Reciprocal, CastLike`; `ConvTranspose`, `Pad` and `Clip` were
added to the runtime by another workstream while the exporter was being
written, and a hard-coded list would have gone stale. The CONVERTER is still
the authority, so run the final `onnx2vkop` on the artifact before believing
any of this.

Why it is still written
-----------------------
* `--dry-run` is the acceptance test for the day the kernels land: it prints
  the inventory and the missing set, and it needs no weights.
* The wrapper is the part that is genuinely this project's work (linear
  `dec_in_proj`, then the decoder, denormalization on the host), and it will be
  correct once the ops exist.

Host vs device
--------------
The graph is `dec_in_proj -> decoder`. `latents * std + mean` and the final
`clamp` stay where the reference has them, except the clamp becomes part of the
graph only if `Clip` is ever added -- today the host applies it.
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn

from h3_export_common import export_graph, check_runtime_ops

# Ops this component is known to need. Whether each is a *gap* is decided at
# run time against `shader_map.runtime_op_names()`, never frozen here: while
# this file was being written, another workstream added ConvTranspose, Pad and
# Clip to the runtime, and a hard-coded list would have kept reporting them.
# (`Exp`, `ReduceL2`, `Reciprocal` and `CastLike` are still missing as of the
# last measurement; see the module docstring.)
NEEDED_OPS = ("ConvTranspose", "Pad", "Exp", "ReduceL2", "Reciprocal",
              "CastLike", "Clip")

TINY = dict(
    latent_channels=4,
    latent_dim=16,
    encoder_dim=8,
    decoder_dim=16,
    num_attention_heads=2,
    encoder_rates=(2, 4),
    decoder_rates=(2, 4),
    decoder_kernel_sizes=(4, 4),
    resblock_kernel_sizes=(3,),
    resblock_dilation_sizes=((1,),),
)


class H3AudioDecoderOnnx(nn.Module):
    """`dec_in_proj` + BigVGAN decoder.

    Input:  latents  (B, latent_channels, T)
    Output: waveform (B, 1, T * prod(decoder_rates))
    """

    INPUT_NAMES = ["latents"]

    def __init__(self, vae):
        super().__init__()
        self.vae = vae

    def forward(self, latents):
        v = self.vae
        h = v.dec_in_proj(latents.to(v.dec_in_proj.weight.dtype))
        return v.decoder(h)


def build_vae(cfg: dict, *, device: torch.device, tiny: bool):
    from diffusers.models.autoencoders.autoencoder_kl_minimax_h3_audio import (
        AutoencoderKLMiniMaxH3Audio,
    )

    if tiny:
        kwargs = dict(TINY)
    else:
        kwargs = dict(
            latent_channels=int(cfg["latent_channels"]),
            latent_dim=int(cfg["latent_dim"]),
            encoder_dim=int(cfg["encoder_dim"]),
            encoder_rates=tuple(int(v) for v in cfg["encoder_rates"]),
            decoder_dim=int(cfg["decoder_dim"]),
            decoder_rates=tuple(int(v) for v in cfg["decoder_rates"]),
            decoder_kernel_sizes=tuple(int(v) for v in cfg["decoder_kernel_sizes"]),
            num_attention_heads=int(cfg["num_attention_heads"]),
            resblock_kernel_sizes=tuple(int(v) for v in cfg["resblock_kernel_sizes"]),
            resblock_dilation_sizes=tuple(
                tuple(int(d) for d in ds)
                for ds in cfg["resblock_dilation_sizes"]),
            sampling_rate=int(cfg["sampling_rate"]),
        )
    if device.type == "meta":
        with torch.device("meta"):
            vae = AutoencoderKLMiniMaxH3Audio(**kwargs)
    else:
        vae = AutoencoderKLMiniMaxH3Audio(**kwargs)
    return vae.eval(), kwargs


def dynamic_axes() -> dict:
    """`T` is the only thing that moves; the hop ratio is a weight constant."""
    return {
        "latents": {0: "batch", 2: "latent_frames"},
        "waveform": {0: "batch", 2: "samples"},
    }


def load_config(model_path: str) -> dict:
    import json

    with open(os.path.join(model_path, "audio_vae", "config.json")) as f:
        return json.load(f)


def run(args) -> int:
    dry = bool(args.dry_run)
    if args.tiny:
        cfg = dict(TINY)
    elif args.model is not None:
        cfg = load_config(args.model)
    else:
        raise SystemExit(
            "error: --dry-run needs either --model <dir> (for "
            "audio_vae/config.json) or --tiny"
        )

    print("[audio-vae] NOTE: this component is known to need runtime kernels "
          f"it does not have ({', '.join(NEEDED_OPS)}; re-checked against the "
          "live allow-list below). The dry run prints the real inventory and "
          "the converter's refusal; see the module docstring.")

    # This is the one component whose dry run does NOT use the meta device.
    # Its exporter is dynamo (see below), and the dynamo save path writes the
    # weights to external data with `tensor.cpu()`, which raises on a meta
    # tensor. The audio VAE is ~0.2 GiB, so building it for real on CPU is
    # cheap -- and unlike the other three components, "does it trace" is not
    # the open question here; the missing kernels are.
    device = torch.device("cpu")
    if dry:
        print("[audio-vae] dry run builds on CPU, not meta: the dynamo "
              "exporter cannot save meta weights. The component is small "
              "(~0.2 GiB).")
    vae, kwargs = build_vae(cfg, device=device, tiny=args.tiny)
    print(f"[build] {'TINY ' if args.tiny else ''}AutoencoderKLMiniMaxH3Audio "
          f"device={device}")
    print(f"[build] " + " ".join(f"{k}={v}" for k, v in kwargs.items()
                                 if k != "latents_mean"
                                 and k != "latents_std"))

    wrapper = H3AudioDecoderOnnx(vae).eval()
    latents = torch.randn(2, kwargs["latent_channels"], 8).to(device)
    out = args.out or "audio_vae_decoder.onnx"

    # The TorchScript exporter cannot trace the alias-free blocks at all
    # ("convolution for kernel of unknown shape"), so this component uses the
    # dynamo exporter, which produces a complete graph.
    #
    # `optimize=False` is required on a meta build: torch's dynamo path runs
    # `onnxscript`'s IR optimizer afterward, and that pass manager raises
    # `PassError` on a graph whose weights have no storage. Leaving it on would
    # make the dry run fail for a reason unrelated to this component's actual
    # problem (the missing kernels).
    import onnx

    torch.onnx.export(wrapper, (latents,), out,
                      input_names=H3AudioDecoderOnnx.INPUT_NAMES,
                      output_names=["waveform"],
                      opset_version=17, dynamo=True, optimize=False,
                      dynamic_axes=dynamic_axes())
    proto = onnx.load(out)
    counts: dict[str, int] = {}
    for node in proto.graph.node:
        counts[node.op_type] = counts.get(node.op_type, 0) + 1
    from h3_export_common import runtime_op_names
    missing_ops = set(counts) - runtime_op_names()
    print(f"[export] {out}: {len(proto.graph.node)} node(s)")
    print("[export] op inventory:")
    for op, n in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
        marker = ("   <- NO RUNTIME KERNEL"
                  if op in NEEDED_OPS and op in missing_ops else "")
        print(f"    {n:5d}  {op}{marker}")

    missing = check_runtime_ops(counts, component=os.path.basename(out))
    blocking = [op for op in NEEDED_OPS if op in counts and op in missing]
    cleared = [op for op in NEEDED_OPS if op in counts and op not in missing]
    print(f"[audio-vae] still missing a kernel: {blocking or 'none'}")
    if cleared:
        print(f"[audio-vae] newly covered since the gap was written: {cleared}")
    print("[audio-vae] onnx2vkop will refuse this graph until the runtime "
          "gains those kernels. Nothing was downloaded and no vkopbin was "
          "written.")
    # Deliberately a non-zero exit: this component is known-unconvertible, and
    # a script that exits 0 would let a pipeline driver treat it as done.
    return 4 if missing else 0
