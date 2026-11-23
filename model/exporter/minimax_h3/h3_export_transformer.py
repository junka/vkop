"""Export the MiniMax-H3 denoiser (`MiniMaxH3Transformer3DModel`) to ONNX.

One forward of this graph is one denoising step: it takes the current noisy
video/audio rows plus the text condition and returns the predicted velocity for
the video rows (and, when the audio stream is enabled, the audio rows). The
sampler that calls it, and the `MiniMaxH3Scheduler` that consumes the velocity,
stay in Python.

The reference forward
---------------------
`MiniMaxH3Transformer3DModel.forward` builds one packed 1-D sequence out of
text, conditioning-video, audio and target-video rows, runs 50 blocks of full
self-attention over it, and reads the two modalities back out by index::

    x = zeros(B, S, 5376)
    x = index_copy(x, 1, text_indices,  refiner(context_embedder(text)))
    x = index_copy(x, 1, video_indices, proj_in(video_rows))
    x = index_copy(x, 1, audio_indices, audio_proj_in(audio_rows))
    for block in blocks:            # 50, no cross-attention anywhere
        x = block(x, temb, adaln_indices, rope(position_ids))
    x = norm_out(x, temb, timestep_indices)
    return proj_out(x)[video_indices], audio_proj_out(x)[audio_indices]

The wrappers here are that forward with three changes, each forced by the
converter and each verified numerically identical to the reference:

1. **The timestep MLP runs on the host.** `diffusers.get_timestep_embedding`
   calls `torch.arange(..., device=timesteps.device)`. On a `meta` build that
   arange is a meta constant and the exporter's constant-fold tries to copy it
   out ("Cannot copy out of meta tensor; no data!"), which is what made a
   weights-free dry run impossible. It is also work that belongs to the host:
   it depends only on the sampler's timestep, and it is one 256->5376->2688 MLP
   per step against 50 blocks of attention. The caller passes `temb`
   `(num_timesteps, 2688)` in. **This is the one place the graph boundary
   differs from the reference forward**, and README.md says so.

2. **Packing is a permutation, not three `index_copy`s.** `index_copy` traces
   to `ScatterElements` whose target shape comes from a
   `Shape`/`ConstantOfShape`/`Where` chain, and the converter has no
   `ConstantOfShape` kernel. Concatenating `[text | video | audio]` and applying
   ONE host-supplied `pack_order` gather produces the identical buffer
   (verified maxdiff 0.0) with a single `Gather`.

3. **RMSNorm is the `RMSNormShim` chain** (see `h3_export_common`): the class
   builds it from `F.rms_norm`, which has no opset-17 symbolic.

With the audio stream off (`--no-with-audio`) the audio rows, the audio
projection and the audio head are dropped, and the packed sequence is
`[text | video]` in order, so the permutation is the identity and no
`pack_order` input exists.
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn

from h3_export_common import (
    export_graph,
    export_streaming_real,
    meta_safe_buffers,
    patch_rmsnorm,
    try_convert,
)

MODALITY_NUM = 3  # video=0, text=1, audio=2 (MINIMAX_H3_MODALITY_NUM)


# ---------------------------------------------------------------------------
# tiny smoke-test geometry
# ---------------------------------------------------------------------------

# Structurally identical to the checkpoint's transformer, sized so a trace takes
# seconds. Note it does NOT reproduce the checkpoint's
# `inner_dim > hidden_size` (5120 vs 5376): that is a property of the released
# weights, not a requirement of the code, so nothing here should be read as a
# performance statement.
TINY = dict(
    num_attention_heads=2,
    attention_head_dim=8,
    hidden_size=32,
    num_layers=2,
    num_refiner_layers=1,
    ffn_dim=64,
    in_channels=4,
    audio_in_channels=4,
    patch_size=(1, 2, 2),
    text_dim=16,
    freq_dim=8,
    time_embed_hidden_dim=32,
    time_embed_dim=16,
    rope_freq_dim=1,
)


def build_model(cfg: dict, *, device: torch.device, tiny: bool):
    from diffusers.models.transformers.transformer_minimax_h3 import (
        MiniMaxH3Transformer3DModel,
    )

    if tiny:
        kwargs = dict(TINY)
    else:
        kwargs = dict(
            num_attention_heads=int(cfg["num_attention_heads"]),
            attention_head_dim=int(cfg["attention_head_dim"]),
            hidden_size=int(cfg["hidden_size"]),
            num_layers=int(cfg["num_layers"]),
            num_refiner_layers=int(cfg["num_refiner_layers"]),
            ffn_dim=int(cfg["ffn_dim"]),
            in_channels=int(cfg["in_channels"]),
            audio_in_channels=int(cfg["audio_in_channels"]),
            patch_size=tuple(cfg["patch_size"]),
            text_dim=int(cfg["text_dim"]),
            freq_dim=int(cfg["freq_dim"]),
            time_embed_hidden_dim=int(cfg["time_embed_hidden_dim"]),
            time_embed_dim=int(cfg["time_embed_dim"]),
            rope_freq_dim=int(cfg["rope_freq_dim"]),
            rope_theta=float(cfg.get("rope_theta", 10000.0)),
            norm_eps=float(cfg.get("norm_eps", 1e-5)),
            qk_norm_eps=float(cfg.get("qk_norm_eps", 1e-5)),
            final_norm_eps=float(cfg.get("final_norm_eps", 1e-5)),
        )
    if device.type == "meta":
        with torch.device("meta"):
            model = MiniMaxH3Transformer3DModel(**kwargs)
    else:
        model = MiniMaxH3Transformer3DModel(**kwargs)
    return model.eval()


# ---------------------------------------------------------------------------
# the export wrappers
# ---------------------------------------------------------------------------

class H3DenoiserOnnx(nn.Module):
    """One denoising step, joint video+audio.

    Inputs:
      hidden_states          (B, N_vid, C_v*pt*ph*pw)  patchified video rows
      audio_hidden_states    (B, N_aud, C_a)           audio latent rows
      encoder_hidden_states  (B, N_txt, text_dim)      text-encoder output
      temb                   (T, time_embed_dim)       host-side timestep MLP
      timestep_indices       (S,)   int64  row -> temb row
      token_tags             (S,)   int64  0 video / 1 text / 2 audio
      position_ids           (S,3)  f32    (t,h,w) rotary coordinates
      video_indices          (N_vid,) int64
      audio_indices          (N_aud,) int64
      text_indices           (N_txt,) int64
      pack_order             (S,)   int64  packed row -> [text|video|audio] row

    Outputs:
      sample                 (B, N_vid, C_v*pt*ph*pw)  video velocity
      audio_sample           (B, N_aud, C_a)           audio velocity
    """

    INPUT_NAMES = [
        "hidden_states",
        "audio_hidden_states",
        "encoder_hidden_states",
        "temb",
        "timestep_indices",
        "token_tags",
        "position_ids",
        "video_indices",
        "audio_indices",
        "text_indices",
        "pack_order",
    ]

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(
        self,
        hidden_states,
        audio_hidden_states,
        encoder_hidden_states,
        temb,
        timestep_indices,
        token_tags,
        position_ids,
        video_indices,
        audio_indices,
        text_indices,
        pack_order,
    ):
        m = self.model
        # The checkpoint is mixed precision (patch projections fp32, block
        # stack bf16) and the text stream sets the packed sequence's dtype, so
        # every projection is aligned with its own parameter dtype exactly as
        # the reference does.
        text_embeds = m.context_embedder(
            encoder_hidden_states.to(m.context_embedder.weight.dtype))
        text_embeds = m.token_refiner(text_embeds)
        video_embeds = m.proj_in(
            hidden_states.to(m.proj_in.weight.dtype)).to(text_embeds.dtype)
        audio_embeds = m.audio_proj_in(
            audio_hidden_states.to(m.audio_proj_in.weight.dtype)
        ).to(text_embeds.dtype)

        packed = torch.cat([text_embeds, video_embeds, audio_embeds], dim=1)
        x = packed.index_select(1, pack_order)

        adaln_indices = timestep_indices * MODALITY_NUM + token_tags
        rotary_emb = m.rope(position_ids)
        for block in m.transformer_blocks:
            x = block(x, temb, adaln_indices, rotary_emb)

        x = m.norm_out(x, temb, timestep_indices).to(m.proj_out.weight.dtype)
        return (
            m.proj_out(x).index_select(1, video_indices),
            m.audio_proj_out(x).index_select(1, audio_indices),
        )


class H3DenoiserVideoOnlyOnnx(nn.Module):
    """One denoising step, video stream only.

    Same block stack and the same `proj_in` / `proj_out`, with the audio rows,
    the audio input projection and the audio head removed. The packed sequence
    is `[text | video]` in order, so no `pack_order` is needed -- the reference
    would apply the identity permutation, and dropping it removes a Gather.

    Inputs: hidden_states, encoder_hidden_states, temb, timestep_indices,
            token_tags, position_ids, video_indices, text_indices
    Output: sample (B, N_vid, C_v*pt*ph*pw)
    """

    INPUT_NAMES = [
        "hidden_states",
        "encoder_hidden_states",
        "temb",
        "timestep_indices",
        "token_tags",
        "position_ids",
        "video_indices",
        "text_indices",
    ]

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(
        self,
        hidden_states,
        encoder_hidden_states,
        temb,
        timestep_indices,
        token_tags,
        position_ids,
        video_indices,
        text_indices,
    ):
        m = self.model
        text_embeds = m.context_embedder(
            encoder_hidden_states.to(m.context_embedder.weight.dtype))
        text_embeds = m.token_refiner(text_embeds)
        video_embeds = m.proj_in(
            hidden_states.to(m.proj_in.weight.dtype)).to(text_embeds.dtype)
        x = torch.cat([text_embeds, video_embeds], dim=1)

        adaln_indices = timestep_indices * MODALITY_NUM + token_tags
        rotary_emb = m.rope(position_ids)
        for block in m.transformer_blocks:
            x = block(x, temb, adaln_indices, rotary_emb)

        x = m.norm_out(x, temb, timestep_indices).to(m.proj_out.weight.dtype)
        return m.proj_out(x).index_select(1, video_indices)


# ---------------------------------------------------------------------------
# dummies + dynamic axes
# ---------------------------------------------------------------------------

def dummy_inputs(model, *, with_audio: bool, device: torch.device,
                 n_text=8, n_video=12, n_audio=6, num_timesteps=1):
    """Representative shapes; for tracing only the row counts matter.

    The packed layout used is the one the pipeline's
    `build_packed_sequence` produces: text rows first, then the audio rows,
    then the video rows.
    """
    m = model
    c_v = int(m.proj_in.weight.shape[1])
    c_a = int(m.audio_proj_in.weight.shape[1])
    c_t = int(m.context_embedder.weight.shape[1])
    # `norm_out.linear` is time_embed_dim -> 2*hidden, so its INPUT width is
    # time_embed_dim -- the width every `adaln_proj.linear` also consumes.
    temb_dim = int(m.norm_out.linear.weight.shape[1])
    s = n_text + n_video + n_audio

    video = torch.randn(1, n_video, c_v)
    audio = torch.randn(1, n_audio, c_a)
    text = torch.randn(1, n_text, c_t)
    temb = torch.randn(num_timesteps, temb_dim)
    timestep_indices = torch.zeros(s, dtype=torch.long)
    tags = torch.zeros(s, dtype=torch.long)
    tags[n_text:n_text + n_audio] = 2   # audio
    tags[n_text + n_audio:] = 0         # video
    tags[:n_text] = 1                   # text
    position_ids = torch.randn(s, 3)
    video_indices = torch.arange(n_text + n_audio, s, dtype=torch.long)
    audio_indices = torch.arange(n_text, n_text + n_audio, dtype=torch.long)
    text_indices = torch.arange(0, n_text, dtype=torch.long)

    if with_audio:
        # concat order is [text | video | audio]; we want the packed order
        # [text | audio | video]. pack_order[packed_row] = concat_row.
        concat_of = list(range(n_text))                          # text
        concat_of += list(range(n_text + n_audio, s))            # video
        concat_of += list(range(n_text, n_text + n_audio))       # audio
        pack_of_concat = [0] * s
        for packed_row, concat_row in enumerate(concat_of):
            pack_of_concat[concat_row] = packed_row
        pack_order = torch.tensor(pack_of_concat, dtype=torch.long)
        args = (video, audio, text, temb, timestep_indices, tags, position_ids,
                video_indices, audio_indices, text_indices, pack_order)
        names = list(H3DenoiserOnnx.INPUT_NAMES)
    else:
        args = (video, text, temb, timestep_indices, tags, position_ids,
                video_indices, text_indices)
        names = list(H3DenoiserVideoOnlyOnnx.INPUT_NAMES)

    return names, tuple(a.to(device) for a in args)


def dynamic_axes(*, with_audio: bool) -> dict:
    """What genuinely varies between calls.

    Symbolic: the row counts (`num_video_tokens`, `num_audio_tokens`,
    `num_text_tokens`) and the packed length `seq_len` -- these change with
    clip length, latent resolution and prompt length. The batch axis is
    symbolic as well, but note the reference calls it "a pure replication
    axis": the index tensors describe ONE packed layout, which every batch item
    repeats, so a batch > 1 does not let the graph re-derive a second layout.

    Static, and deliberately NOT marked:
      * the per-row feature counts (`C_v*pt*ph*pw`, `C_a`, `text_dim`) -- they
        are weight shapes;
      * `temb`'s feature axis (`time_embed_dim`);
      * `position_ids`' last axis (3: the (t,h,w) axes).
    """
    axes = {
        "hidden_states": {0: "batch", 1: "num_video_tokens"},
        "encoder_hidden_states": {0: "batch", 1: "num_text_tokens"},
        "temb": {0: "num_timesteps"},
        "timestep_indices": {0: "seq_len"},
        "token_tags": {0: "seq_len"},
        "position_ids": {0: "seq_len"},
        "video_indices": {0: "num_video_tokens"},
        "text_indices": {0: "num_text_tokens"},
        "sample": {0: "batch", 1: "num_video_tokens"},
    }
    if with_audio:
        axes["audio_hidden_states"] = {0: "batch", 1: "num_audio_tokens"}
        axes["audio_indices"] = {0: "num_audio_tokens"}
        axes["audio_sample"] = {0: "batch", 1: "num_audio_tokens"}
        axes["pack_order"] = {0: "seq_len"}
    return axes


def load_config(model_path: str) -> dict:
    import json

    with open(os.path.join(model_path, "transformer", "config.json")) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------

def run(args) -> int:
    with_audio = bool(args.with_audio)
    dry = bool(args.dry_run)

    if args.tiny:
        cfg = dict(TINY)
    elif args.model is not None:
        cfg = load_config(args.model)
    else:
        raise SystemExit(
            "error: --dry-run needs either --model <dir> (for "
            "transformer/config.json) or --tiny"
        )

    if args.attn_heads:
        cfg = dict(cfg)
        cfg["num_attention_heads"] = int(args.attn_heads)

    # `--stream` builds on meta from the start: the splice supplies the bytes,
    # and a CPU build would materialize 61.7 GiB of initializers for nothing --
    # measured: the process is OOM-killed at 63.7 GB peak before the trace even
    # begins.
    device = (torch.device("meta")
              if (dry or getattr(args, "stream", False))
              else torch.device("cpu"))
    tag = "TINY" if args.tiny else "checkpoint config"
    print(f"[build] MiniMaxH3Transformer3DModel ({tag}) device={device}")
    print(f"[build] layers={cfg['num_layers']} hidden={cfg['hidden_size']} "
          f"heads={cfg['num_attention_heads']}x{cfg['attention_head_dim']} "
          f"ffn={cfg['ffn_dim']} text_dim={cfg['text_dim']} "
          f"in={cfg['in_channels']} audio_in={cfg['audio_in_channels']} "
          f"patch={tuple(cfg['patch_size'])} with_audio={with_audio}")

    model = build_model(cfg, device=device, tiny=args.tiny)
    patch_rmsnorm(model)
    if device.type == "meta":
        n = meta_safe_buffers(model)   # rope.inv_freq -> a splicable parameter
        print(f"[meta] {n} computed buffer(s) promoted to graph inputs")

    wrapper_cls = H3DenoiserOnnx if with_audio else H3DenoiserVideoOnlyOnnx
    wrapper = wrapper_cls(model).eval()
    names, dummies = dummy_inputs(model, with_audio=with_audio, device=device)

    out = args.out or ("transformer.onnx" if with_audio
                       else "transformer.video_only.onnx")
    outs = ["sample", "audio_sample"] if with_audio else ["sample"]

    if getattr(args, "stream", False):
        # Real weights, bounded memory: the model is already on meta, so the
        # splice supplies every byte and nothing is materialized here.
        export_streaming_real(
            wrapper, dummies, names, outs,
            dynamic_axes(with_audio=with_audio),
            model_dir=args.model, component="transformer", out_path=out,
            ckpt_prefix="model.", rope_theta=float(cfg["rope_theta"]),
        )
        if args.no_convert:
            print("[stream] --no-convert: skipping onnx2vkop")
            return 0
        return 0 if try_convert(out, os.path.dirname(os.path.abspath(out))) == 0 else 5

    export_graph(
        wrapper, dummies, out,
        input_names=names, output_names=outs,
        dynamic_axes=dynamic_axes(with_audio=with_audio),
        dry_run=dry,
    )
    if dry:
        print("[dry-run] no weight bytes were read. The op inventory above is "
              "what the real export produces; the initializer table is empty "
              "because the weights are graph inputs.")
        # The op-name pre-filter is not the authority. Hand the weightless
        # graph to the converter: it runs the fusion passes and the real
        # `validate_supported_ops`, which is the answer the user needs.
        if args.no_convert:
            print("[dry-run] --no-convert: skipping onnx2vkop")
            return 0
        return 0 if try_convert(out, os.path.dirname(os.path.abspath(out))) == 0 else 5
    return 0
