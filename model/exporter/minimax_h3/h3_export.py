#!/usr/bin/env python3
"""Export one MiniMax-H3 component to ONNX.

MiniMax-H3 is three pipelines (base / T2VA, FL2VA, Ref2VA) over one
architecture, and a "component" of one of them is the unit that can be
exported: the full model is ~498 GiB and no single process can hold three
copies of it. So this tool exports **one component at a time**:

    text_encoder       50-layer truncated Qwen3-VL text tower  -> prompt_embeds
    transformer        the denoiser, video only or video+audio
    vae_decoder        the video ViT decoder
    audio_vae_decoder  the BigVGAN waveform decoder

The diffusion loop, both schedulers, latent packing/unpacking, the RoPE
position grids, the `latents_mean/std` denormalization and the video
post-processing all stay on the host. There is no export of them here and
there cannot be: they are Python control flow over a variable number of
sampling steps, which is the sampler's business, not a kernel's.

`--dry-run` builds the graph with no weights (meta device) and prints the op
inventory it would produce. It is the only validation that runs without the
checkpoint, and every exporter supports it.

Examples
--------
    # what would the denoiser produce, with no weights on disk?
    python3 h3_export.py --component transformer --dry-run

    # the real thing, one component at a time
    python3 h3_export.py --component transformer --model /models/MiniMax-H3 \
        --pipeline base --with-audio --out out/transformer.onnx
"""

from __future__ import annotations

import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from h3_export_common import (  # noqa: E402
    MIN_DIFFUSERS,
    TESTED_DIFFUSERS,
    check_diffusers,
    check_transformers,
)

COMPONENTS = {
    "text_encoder": "h3_export_text_encoder",
    "transformer": "h3_export_transformer",
    "vae_decoder": "h3_export_vae_decoder",
    "audio_vae_decoder": "h3_export_audio_vae_decoder",
}

PIPELINES = {
    # base/model_index.json: `MiniMaxH3ModularPipeline`, diffusers 0.36.0.dev0
    "base": "transformer",
    # FL2VA/ and Ref2VA/: self-contained pipelines with their own transformer,
    # text encoder and video VAE, and an *older* 0.32.2 API (MiniMaxH3DiTModel
    # / MiniMaxH3VideoVAE / MiniMaxH3AudioVAE) that no released diffusers has.
    "fl2va": "FL2VA",
    "ref2va": "Ref2VA",
}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Export one MiniMax-H3 component to ONNX.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Components: " + ", ".join(COMPONENTS) + "\n"
            "Pipelines:  " + ", ".join(PIPELINES) + "\n"
            "Schedulers and the sampling loop are host-side and are never "
            "exported."
        ),
    )
    p.add_argument("--component", required=True, choices=sorted(COMPONENTS),
                   help="which graph to build")
    p.add_argument("--pipeline", default="base", choices=sorted(PIPELINES),
                   help="which pipeline variant the component comes from "
                        "(default: base). fl2va/ref2va use an API no released "
                        "diffusers carries -- see README.md.")
    p.add_argument("--model", default=None,
                   help="path to the MiniMax-H3 checkout (the directory "
                        "holding model_index.json). Not needed with "
                        "--dry-run, which builds from the config files only.")
    p.add_argument("--out", default=None,
                   help="output .onnx path (default: "
                        "<component>[.audio].onnx in the current directory)")
    p.add_argument("--dry-run", action="store_true",
                   help="build the graph on `meta` (no weight bytes), print "
                        "the op inventory, write the weightless graph if "
                        "--out is given, and exit.")
    p.add_argument("--no-convert", action="store_true",
                   help="with --dry-run, stop after the graph: do not hand it "
                        "to onnx2vkop. The conversion is the authoritative op "
                        "check but takes a few minutes on the full-size "
                        "transformer/text encoder.")
    p.add_argument("--with-audio", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="transformer only: include the audio stream (default "
                        "on; --no-with-audio gives the video-only graph)")
    p.add_argument("--attn-heads", type=int, default=None,
                   help="SMOKE TEST ONLY: build the transformer at this head "
                        "count instead of the checkpoint's. Requires "
                        "--dry-run. Used to make a tiny graph that traces in "
                        "seconds; the shape is not the checkpoint's.")
    p.add_argument("--tiny", action="store_true",
                   help="SMOKE TEST ONLY: build a structurally complete but "
                        "tiny model (requires --dry-run). Verifies that the "
                        "wrappers trace and convert; says nothing about the "
                        "real shapes.")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    # Prerequisites first: a missing class must not look like a missing model.
    check_diffusers()
    if args.component == "text_encoder":
        check_transformers()

    if args.pipeline != "base":
        sub = PIPELINES[args.pipeline]
        print(
            f"[pipeline] {args.pipeline!r} lives in {sub}/ and its "
            "model_index.json names MiniMaxH3DiTModel / MiniMaxH3VideoVAE / "
            "MiniMaxH3AudioVAE at diffusers 0.32.2. No released diffusers "
            "provides those classes, so this component cannot be built from "
            "them yet. See README.md, 'The FL2VA / Ref2VA problem'.",
            file=sys.stderr,
        )
        return 3

    if args.model is None and not args.dry_run:
        print(
            "error: --model is required unless --dry-run is given",
            file=sys.stderr,
        )
        return 2
    if args.model is not None and not os.path.isdir(args.model):
        print(f"error: --model {args.model!r} is not a directory",
              file=sys.stderr)
        return 2

    if (args.attn_heads or args.tiny) and not args.dry_run:
        print("error: --attn-heads/--tiny are smoke-test knobs and require "
              "--dry-run", file=sys.stderr)
        return 2

    module = __import__(COMPONENTS[args.component])
    return module.run(args)


if __name__ == "__main__":
    raise SystemExit(main())
