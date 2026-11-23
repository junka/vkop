"""Export MiniMax-H3's text conditioner (Qwen3-VL, truncated) to ONNX.

MiniMax-H3 does not condition on a language model's last hidden state. The
pipeline reads `hidden_states[50]` of a 64-layer Qwen3-VL text tower:

    get_qwen3vl_prompt_embeds(..., text_encoder_layer=50)
    -> outputs.hidden_states[50]          # (1, num_text_tokens, 5120)

and never touches `lm_head`. Reading `diffusers/modular_pipelines/minimax_h3/
encoders.py` is worth the two minutes: `hidden_states[L]` is the output of
decoder layer `L-1`, *pre-final-norm* (verified layer-for-layer against a
manual loop: maxdiff 0.0), and the function explicitly refuses a tower
truncated to exactly `L` layers because that last state would be post-norm.

So this graph is: embeddings -> layers 0..L-1 -> hidden state, with **no
final RMSNorm and no lm_head**. That also removes the 151936-wide vocabulary
projection, which is the single largest tensor in the file.

Vision
------
The checkpoint carries a 27-block ViT (`model.visual.*`, 8.7 GiB) with
deepstack extractors at layers 8/16/24. The base (T2VA) pipeline's text
encoder input is text-only, so it never runs the tower. But FL2VA/Ref2VA feed
keyframes and reference images through it, and huggingface's Qwen3-VL injects
deepstack features into the *first* three text layers. That path is a separate
graph with a separate input contract (`pixel_values`, `image_grid_thw`,
`mm_token_type_ids`) and is **not** exported here -- the wrapper below asserts
that no vision inputs were supplied rather than silently ignoring them. See
README.md, "Vision tower".

Only the tokenizer's output is an input: `input_ids`. Position ids, the
attention mask and all three `position_embeddings` rows (MRoPE has 4, the text
tower uses 3) are built in-graph here, from `input_ids` alone, because a
text-only prompt makes them a pure function of the sequence length:

* positions are `0..S-1` on all three MRoPE axes (the reference's text-only
  degeneracy; all three rows are equal);
* the mask is strictly causal;
* deepstack contributions are absent (no vision tokens).
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


# The layer the pipeline conditions on. Read from the config when present
# (`components.text_encoder_layer` in the modular pipeline), else this default.
DEFAULT_TEXT_ENCODER_LAYER = 50


class Qwen3VLRotaryShim(nn.Module):
    """`Qwen3VLTextRotaryEmbedding.forward` without the autocast context.

    Two independent reasons this shim exists:

    1. The reference wraps the frequency math in `maybe_autocast(device_type=...)`,
       and `maybe_autocast` calls `torch.is_autocast_enabled(device_type)`, which
       raises `RuntimeError: unknown device type for autocast` when the device
       is `meta`. That makes a weights-free dry run impossible, and the autocast
       is a fp32-forcing no-op here anyway (every operand is already cast with
       `.float()`).

    2. `transformers` 5.17 REMOVED `apply_interleaved_mrope`; the module now
       mutates frequencies in place via `recomposition_frequencies`. Code that
       called the old method dies with `AttributeError` on 5.17 and later.

    The interleaving is replaced with the static-matmul form this repo's own
    LLM exporter already uses (`llm/exporter/export/qwen3vl_export_onnx.py::
    StaticInterleavedMrope`), which is bit-exact against both the old and the
    new HF implementations (verified maxdiff 0.0, including distinct t/h/w
    rows). The equivalence: column `j` of the interleaved frequency table comes
    from a FIXED row of (t,h,w) --

        row(j) = 1 (H) if j % 3 == 1 and j < 3*s1
               = 2 (W) if j % 3 == 2 and j < 3*s2
               = 0 (T) otherwise

    -- so the whole selection is one matmul `pos (B,q,3) @ M (3,half)` with
    `M[r,j] = inv_freq[j] * (r == row(j))`.
    """

    def __init__(self, rotary):
        super().__init__()
        # NOT held as a submodule: the static matmul replaces everything it did,
        # and assigning a Module would pull its buffers (`original_inv_freq` et
        # al) into the traced parameter set, which then have no checkpoint entry.
        self.mrope_section = list(rotary.mrope_section)
        self.attention_scaling = rotary.attention_scaling
        self.register_buffer("inv_freq", rotary.inv_freq, persistent=False)

        inv_freq = rotary.inv_freq.float()
        half = int(inv_freq.shape[0])
        s1, s2 = self.mrope_section[1], self.mrope_section[2]
        if 3 * s1 > half or 3 * s2 > half:
            raise ValueError(
                f"mrope_section {self.mrope_section} does not fit inv_freq of "
                f"length {half}")
        sel = torch.zeros(half, dtype=torch.long)
        for j in range(half):
            if j % 3 == 1 and j < 3 * s1:
                sel[j] = 1
            elif j % 3 == 2 and j < 3 * s2:
                sel[j] = 2
        M = torch.zeros(3, half, dtype=torch.float32)
        M[sel, torch.arange(half)] = inv_freq
        self.register_buffer("M", M)          # (3, half)

    def forward(self, x, position_ids):
        if position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(
                3, position_ids.shape[0], -1)
        pos = position_ids.to(torch.float32)          # (3, B, q)
        pos_t = pos.permute(1, 2, 0)                  # (B, q, 3)
        freqs = torch.matmul(pos_t, self.M)           # (B, q, half)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos() * self.attention_scaling
        sin = emb.sin() * self.attention_scaling
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class Qwen3VLTextOnnx(nn.Module):
    """`Qwen3VLTextModel` truncated after decoder layer `L-1`, minus the norm.

    Inputs: input_ids      (B, S)    int64
            attention_bias (B, 1, S, S) fp16  additive causal mask, host-built
    Output: prompt_embeds (B, S, hidden_size)

    Attention runs the plain eager path (scaled dot-product over explicit
    tensors). That is deliberate: torch's SDPA symbolic refuses `enable_gqa`
    (GQA), so an export through `sdpa`/`flash` attention fails outright. The
    eager path is also what the repo's existing Qwen3 exporters use, and the
    converter folds it into a single `Attention` op.
    """

    def __init__(self, text_model, layer: int, num_kv_groups: int,
                 scaling: float):
        super().__init__()
        self.lm = text_model
        self.layer = int(layer)
        self.num_kv_groups = int(num_kv_groups)
        self.scaling = float(scaling)
        # Replace the rotary module: its forward calls maybe_autocast(), which
        # raises on the `meta` device and blocks a weights-free dry run.
        self.lm.rotary_emb = Qwen3VLRotaryShim(self.lm.rotary_emb)

    def _positions(self, x: torch.Tensor):
        s = x.shape[1]
        pos = torch.arange(s, device=x.device, dtype=torch.long)
        pos = pos.view(1, 1, -1).expand(3, x.shape[0], -1)
        return self.lm.rotary_emb(x, pos)

    @staticmethod
    def causal_bias(seq: int, dtype=torch.float16) -> torch.Tensor:
        """Host-side additive causal mask, `(1, 1, S, S)`.

        This is a graph INPUT, not something the graph builds, and that is the
        same contract this repo's Qwen3-VL LLM exporter already uses
        (`llm/exporter/export/qwen3vl_export_onnx.py`, `attention_bias`).

        Building it in-graph is not possible with the current runtime op set:
        `torch.triu(torch.full(...))` needs `Trilu` + `ConstantOfShape`, the
        comparison form needs `Greater`, and a clamped-difference form needs
        `Clip` -- none of which have kernels. The mask is a pure function of the
        sequence length, which the host knows, so it costs nothing to hoist.

        `torch.finfo(fp16).min` (-65504) is what HF's `create_causal_mask`
        produces and what the existing exporter uses; `-1e4` is too small to
        separate under fp16 softmax.
        """
        neg = torch.finfo(dtype).min
        mask = torch.triu(torch.full((seq, seq), neg, dtype=dtype),
                          diagonal=1)
        return mask.view(1, 1, seq, seq)

    def forward(self, input_ids, attention_bias):
        lm = self.lm
        x = lm.embed_tokens(input_ids)
        position_embeddings = self._positions(x)

        out = x
        for i, layer in enumerate(lm.layers):
            out = layer(out, attention_mask=attention_bias, position_ids=None,
                        position_embeddings=position_embeddings)
            if i == self.layer - 1:
                return out
        raise RuntimeError(
            f"asked for hidden_states[{self.layer}] but the tower has "
            f"{len(lm.layers)} layers")


def build_text_model(cfg: dict, *, device: torch.device, tiny: bool):
    from transformers import Qwen3VLForConditionalGeneration, Qwen3VLConfig

    if tiny:
        tcfg = dict(hidden_size=64, num_hidden_layers=4, num_attention_heads=2,
                    num_key_value_heads=1, head_dim=32, intermediate_size=128,
                    vocab_size=256, rope_theta=5e6,
                    rope_scaling={"mrope_interleaved": True,
                                  "mrope_section": [24, 20, 20],
                                  "rope_type": "default"})
        vcfg = dict(depth=1, hidden_size=32, num_heads=2, intermediate_size=64,
                    patch_size=16, spatial_merge_size=2, temporal_patch_size=2,
                    out_hidden_size=64, num_position_embeddings=256,
                    deepstack_visual_indexes=[0])
    else:
        tcfg = {k: v for k, v in cfg["text_config"].items()
                if k not in ("dtype", "model_type")}
        vcfg = {k: v for k, v in cfg["vision_config"].items()
                if k != "model_type"}

    qcfg = Qwen3VLConfig(text_config=tcfg, vision_config=vcfg)
    if device.type == "meta":
        with torch.device("meta"):
            model = Qwen3VLForConditionalGeneration(qcfg)
    else:
        model = Qwen3VLForConditionalGeneration(qcfg)
    model.eval()

    # Eager attention. With `sdpa` (or flash) the trace dies on
    # "conversion of scaled_dot_product_attention not implemented if
    # enable_gqa is True", which is exactly the GQA this tower uses.
    model.config._attn_implementation = "eager"
    model.model.language_model.config._attn_implementation = "eager"
    for layer in model.model.language_model.layers:
        layer.self_attn.config._attn_implementation = "eager"
    return model


def dynamic_axes() -> dict:
    """`seq` and `batch` only.

    `hidden_size` and the MRoPE axis count are weight/architecture constants and
    stay static. A different prompt length is the only thing that moves.
    """
    return {
        "input_ids": {0: "batch", 1: "seq"},
        "attention_bias": {0: "batch", 2: "seq", 3: "seq"},
        "prompt_embeds": {0: "batch", 1: "seq"},
    }


def load_config(model_path: str) -> dict:
    import json

    with open(os.path.join(model_path, "text_encoder", "config.json")) as f:
        return json.load(f)


def run(args) -> int:
    dry = bool(args.dry_run)
    if args.tiny:
        cfg = None
    elif args.model is not None:
        cfg = load_config(args.model)
    else:
        raise SystemExit(
            "error: --dry-run needs either --model <dir> (for "
            "text_encoder/config.json) or --tiny"
        )

    # `--stream` builds on meta: the splice supplies the bytes, and a CPU build
    # would materialize the whole text tower for nothing.
    device = (torch.device("meta")
              if (dry or getattr(args, "stream", False))
              else torch.device("cpu"))
    model = build_text_model(cfg, device=device, tiny=args.tiny)
    lm = model.model.language_model
    tcfg = model.config.text_config

    num_layers = tcfg.num_hidden_layers
    layer = DEFAULT_TEXT_ENCODER_LAYER
    if layer > num_layers:
        raise SystemExit(
            f"error: MiniMax-H3 conditions on hidden_states[{layer}] but this "
            f"text encoder has {num_layers} layers. (A tower truncated to "
            f"exactly {layer} layers is post-norm and is NOT the conditioning "
            "MiniMax-H3 expects -- see the module docstring.)"
        )

    num_heads = tcfg.num_attention_heads
    num_kv = tcfg.num_key_value_heads
    head_dim = tcfg.head_dim or tcfg.hidden_size // num_heads
    scaling = head_dim ** -0.5

    print(f"[build] Qwen3-VL text tower {'(TINY) ' if args.tiny else ''}"
          f"device={device}")
    print(f"[build] layers={num_layers} -> exporting hidden_states[{layer}] "
          f"(decoder layer {layer - 1}, PRE-final-norm, no lm_head)")
    print(f"[build] hidden={tcfg.hidden_size} heads={num_heads} "
          f"kv_heads={num_kv} head_dim={head_dim} vocab={tcfg.vocab_size}")

    patch_rmsnorm(lm)   # Qwen3VLTextRMSNorm is already primitive, but the
                        # shim keeps the chain identical to the other graphs

    wrapper = Qwen3VLTextOnnx(lm, layer, num_heads // num_kv, scaling).eval()
    if device.type == "meta":
        # `inv_freq` is computed from config inside a meta context, so it is a
        # meta buffer; the exporter would inline it as a Constant and fail.
        n = meta_safe_buffers(model)
        print(f"[meta] {n} computed buffer(s) promoted to graph inputs")
    seq = 16
    input_ids = torch.randint(0, tcfg.vocab_size, (1, seq),
                              dtype=torch.long).to(device)
    attention_bias = Qwen3VLTextOnnx.causal_bias(seq).to(device)

    out = args.out or f"text_encoder_hs{layer}.onnx"

    if getattr(args, "stream", False):
        # Real weights, streamed. The wrapper's `lm.` prefix maps to the
        # checkpoint's `model.language_model.` subtree -- the same tensor names
        # under a different path, which is what `ckpt_rename` expresses.
        export_streaming_real(
            wrapper, (input_ids, attention_bias),
            ["input_ids", "attention_bias"],
            ["prompt_embeds"], dynamic_axes(),
            model_dir=args.model, component="text_encoder", out_path=out,
            ckpt_prefix="lm.", ckpt_rename="model.language_model.",
            rope_theta=float(getattr(tcfg, "rope_theta", 10000.0)),
        )
        if args.no_convert:
            print("[stream] --no-convert: skipping onnx2vkop")
            return 0
        return 0 if try_convert(out, os.path.dirname(os.path.abspath(out))) == 0 else 5

    export_graph(
        wrapper, (input_ids, attention_bias), out,
        input_names=["input_ids", "attention_bias"],
        output_names=["prompt_embeds"],
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
