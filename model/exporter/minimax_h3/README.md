# MiniMax-H3 → ONNX → vkop

Export scripts for ModelScope's `MiniMax/MiniMax-H3`, one **component** at a
time, plus the host/device split the runtime expects.

MiniMax-H3 is a **joint video + audio diffusion** model, not an LLM. A pipeline
runs a host-side sampler loop that calls a denoiser many times; the denoiser is
a 50-block transformer over one packed sequence of text, video and audio rows.
Nothing here exports a sampling loop, a scheduler, or a monolithic graph.

> **Do not export all three pipelines in one process.** The checkpoint is
> ~498 GiB across three self-contained pipeline variants; no process can hold
> more than one component at a time. One invocation = one graph.

---

## Prerequisites

| need | version | why |
|---|---|---|
| Python | 3.10+ | |
| `torch` | 2.4+ (tested 2.10.0) | `F.rms_norm` symbolic absence is worked around, see below |
| `transformers` | with Qwen3-VL (tested 5.3.0) | the text conditioner is `Qwen3VLForConditionalGeneration` |
| `diffusers` | **>= 0.40.0**, tested **0.43.0** | the H3 classes first ship in 0.40.0 |
| `onnx` | tested 1.19.0 | |
| `onnx2vkop` | this checkout, `model/pypi/` | conversion + the runtime op allow-list |

### The diffusers situation (read this before anything else)

`model_index.json` says `"_diffusers_version": "0.36.0.dev0"`. **That string
does not identify a usable release.**

| | |
|---|---|
| `diffusers` installed on this box | **no** (`pip show diffusers` → not found) |
| 0.36.0 wheel (satisfies the version string) | contains **0** `minimax*` files |
| 0.40.0 / 0.41.0 / 0.42.0 wheel | 13 `minimax_h3` files — classes present |
| 0.43.0 wheel | 23 `minimax_h3` files — what everything here was tested against |

So `MiniMaxH3Transformer3DModel` is **not importable on this box as it stands**.
Get it from either:

```bash
pip install "diffusers==0.43.0"        # or any >= 0.40.0
# or a checkout of diffusers `main`:
pip install "git+https://github.com/huggingface/diffusers@main"
```

Everything in this directory was exercised against **0.43.0**. A test prefix
that avoids touching the main venv:

```bash
pip download --no-deps diffusers==0.43.0 -d /tmp/dtyp
python3 -c "import zipfile; zipfile.ZipFile('/tmp/dtyp/diffusers-0.43.0-py3-none-any.whl').extractall('/tmp/dtyp/site')"
pip install --target /tmp/dtyp/deps "safetensors>=0.8.0" "huggingface-hub>=1.32,<2.0" \
    httpx2 httpcore2 truststore anyio h11 certifi idna sniffio typing_extensions \
    tqdm fsspec pyyaml requests filelock tomli shellingham rich markdown-it-py mdurl pygments
PYTHONPATH=/tmp/dtyp/deps:/tmp/dtyp/site python3 h3_export.py --component transformer --dry-run --model <model>
```

`h3_export.common.check_diffusers()` enforces this and says the same thing when
it fails, so a missing prerequisite can never look like a missing model.

### The FL2VA / Ref2VA problem

`FL2VA/` and `Ref2VA/` are complete self-contained pipelines with their own
transformer, text encoder and video VAE. But their `model_index.json` names
`MiniMaxH3DiTModel`, `MiniMaxH3VideoVAE`, `MiniMaxH3AudioVAE` and
`MiniMaxH3Qwen3VLHFEncoder` at `"_diffusers_version": "0.32.2"`, with
`auto_map` entries pointing at files that are **not in the downloaded tree**.

**No released diffusers contains those classes** (0.43.0 has none of them).
Their weights also differ: the FL2VA transformer's tensor names
(`blocks.N.attn.qkv_proj`, `video_patch_proj`, `rope.inv_freq`, `final_layer.*`)
share almost nothing with the base transformer's
(`transformer_blocks.N.attn.to_q`, `proj_in`, `norm_out.*`). That is an older
architecture, not a re-spelled one.

`--pipeline fl2va|ref2va` therefore exits 3 with this explanation. Exporting
them needs either the pipeline's own modeling code (the `auto_map` targets,
which were not published in the config bundle we have) or a from-scratch
port. **The base pipeline is the one that can be built today.**

---

## Components

| component | class | graph in | graph out | weights | status |
|---|---|---|---|---|---|
| `text_encoder` | `Qwen3VLForConditionalGeneration` text tower, truncated | `input_ids` | `prompt_embeds` | 62.1 GiB | dry run + convert ✅ |
| `transformer` | `MiniMaxH3Transformer3DModel` | 11 row/index tensors | `sample`, `audio_sample` | 61.7 GiB | dry run + convert ✅ |
| `transformer --no-with-audio` | same, video stream only | 8 tensors | `sample` | 61.7 GiB | dry run + convert ✅ |
| `vae_decoder` | `AutoencoderKLMiniMaxH3` decoder (ViT) | `latents`, `position_ids` | `video` | 9.7 GiB | dry run + convert ✅ |
| `audio_vae_decoder` | `AutoencoderKLMiniMaxH3Audio` BigVGAN decoder | `latents` | `waveform` | ~0.2 GiB | **blocked on runtime ops** ❌ |
| `transformer_ref` | duplicate of `transformer` | — | — | 61.7 GiB | same graph; byte-identical config **and** tensor-name set |
| `scheduler`, `audio_scheduler` | `MiniMaxH3Scheduler` | — | — | 0 | **host only, never exported** |

`transformer_ref` is confirmed a duplicate: `transformer/config.json` and
`transformer_ref/config.json` are byte-identical, and their
`*.safetensors.index.json` weight maps have the same 638 names. Export it once
and point both slots at the same artifact.

Total for the base pipeline: text encoder 62.1 + transformer 61.7 + transformer_ref
61.7 + video VAE 9.7 + audio VAE ≈ **195 GiB** of the ~498 GiB; the rest is the
FL2VA and Ref2VA copies.

---

## Verified status (measured on this box, no weights)

Every wrapper was traced at the checkpoint's real config and, separately, run
numerically against the unmodified diffusers module at a tiny geometry. The
numerical check is the one that matters for correctness — it needs no
checkpoint, only the classes.

| check | result |
|---|---|
| `RMSNormShim` vs `nn.RMSNorm` | **maxdiff 0.0** |
| transformer wrapper vs `MiniMaxH3Transformer3DModel.forward` (video head) | **maxdiff 0.0** |
| transformer wrapper vs reference (audio head) | **maxdiff 0.0** |
| video-only wrapper vs reference over a layout with no audio rows | **maxdiff 0.0** |
| video VAE decoder wrapper vs `MiniMaxH3VideoViTDecoder3d.forward` | **maxdiff 0.0** |
| text encoder wrapper vs `hidden_states[L]` from `model(...)` | **maxdiff 0.0** (also confirmed `hidden_states[i]` == the output of decoder layer `i-1`, pre-final-norm) |

Note the joint transformer comparison is only exact when the reference is given
the same `temb` the wrapper receives; comparing against a reference that
recomputes `temb` from a *different* `timestep` gives ~0.3, which is the
timestep changing, not a bug.

Graph-level, with the initializer table empty:

| component | nodes (50-layer / full config) | onnx2vkop |
|---|---|---|
| transformer | 15358 | **converged** — 90 s, 13.8 MiB vkopbin |
| vae_decoder | ~9k | **converged** |
| text_encoder | 13226 | **converged** |
| audio_vae_decoder | ~2500 | **refused** — 4 missing ops |

Reproduce:

```bash
# fast (graph only, 7-17 s)
python3 h3_export.py --component transformer --dry-run --no-convert --model /tmp/h3cfg/h3
# full (also runs the converter, ~90 s)
python3 h3_export.py --component transformer --dry-run --model /tmp/h3cfg/h3
```

`--dry-run` needs a diffusers that has the classes; `--model` can point at a
config-only tree like `/tmp/h3cfg/h3` (it reads `*/config.json` only).

---

## Run order

```bash
# 0. what would each graph contain, with no weights on disk? (do this first)
python3 h3_export.py --component transformer  --dry-run --model /path/to/MiniMax-H3
python3 h3_export.py --component vae_decoder  --dry-run --model /path/to/MiniMax-H3
python3 h3_export.py --component text_encoder --dry-run --model /path/to/MiniMax-H3
python3 h3_export.py --component audio_vae_decoder --dry-run --model /path/to/MiniMax-H3

# 1. text encoder (needed once per prompt length family; see dynamic axes)
python3 h3_export.py --component text_encoder --model /path/to/MiniMax-H3 \
    --out out/text_encoder_hs50.onnx
python3 -m onnx2vkop.cli -i out/text_encoder_hs50.onnx -o out/text_encoder_hs50.vkopbin -q fp16

# 2. denoiser
python3 h3_export.py --component transformer --model /path/to/MiniMax-H3 \
    --out out/transformer.onnx
python3 -m onnx2vkop.cli -i out/transformer.onnx -o out/transformer.vkopbin -q fp16

# 3. video VAE decoder
python3 h3_export.py --component vae_decoder --model /path/to/MiniMax-H3 \
    --out out/vae_decoder.onnx
python3 -m onnx2vkop.cli -i out/vae_decoder.onnx -o out/vae_decoder.vkopbin -q fp16

# 4. audio VAE — currently refuses; see "Audio VAE" below
python3 h3_export.py --component audio_vae_decoder --model /path/to/MiniMax-H3 \
    --out out/audio_vae_decoder.onnx
```

`-q fp16` is not a free choice: the checkpoint is bf16/fp32 mixed, bf16 is
normalized to fp16 by the converter, and the fp16 shader domain is what the
existing LLM artifacts use. Use `-q int8` only after the correctness path is
green.

---

## Host vs device split

The dividing line is: **anything that indexes or loops is the host's, anything
that is a fixed stack of tensor ops is the graph's.**

| stays in Python (host) | runs on the GPU (graph) |
|---|---|
| the denoising loop (`MiniMaxH3LoopDenoiser`), step count, `sigmas` | one transformer forward = one step |
| `MiniMaxH3Scheduler` / `MiniMaxH3SchedulerStep` (video shift 12.0, audio 3.0) | — |
| **the timestep MLP** (`time_proj` + `time_embedder`), passed in as `temb` | everything the `temb` feeds (AdaLN modulation) |
| packed-layout construction: `token_tags`, `timestep_indices`, `position_ids`, `video_indices`, `audio_indices`, `text_indices`, `pack_order` | reading those tensors by index |
| patchify / unpatchify of video latents, audio row reshaping | — |
| `latents * std + mean` (both VAEs) and the ImageNet un-normalization | decoder forward |
| VAE spatial tiling and the temporal chunk/cross-fade loop | one decoder forward per tile/chunk |
| the RoPE **position grid** for the VAE decoder | RoPE cos/sin + the rotation |
| tokenization, `Qwen3VLProcessor`, chat template | text tower forward |
| the output `clamp(0,1)`, PIL/np conversion | — |

Two of these are deliberate departures from the reference `forward` and are
called out because they change the graph boundary:

1. **`temb` is a graph input.** The reference computes it in-graph, but
   `get_timestep_embedding` calls `torch.arange(device=timesteps.device)`, which
   on a `meta` build becomes a meta constant that the exporter cannot fold
   ("Cannot copy out of meta tensor; no data!"). Hoisting it also matches the
   split above: it depends only on the sampler's scalar timestep.
2. **`position_ids` for the VAE decoder is a graph input.** The reference builds
   it with `torch.arange(0.5, size)`; tracing `.tile(2)` on a dynamic tensor
   emits `If` (control flow), which the converter rejects outright. The grid is
   a pure function of `(F, H, W)`, which the host knows.

Both are numerically identical to the reference (verified), not approximations.

---

## Op-coverage findings (measured, not guessed)

`shader_map.runtime_op_names()` is the runtime's allow-list and it is small.
Three gaps bite this model, and only one is fatal:

| op | where | consequence |
|---|---|---|
| `aten::rms_norm` (not ONNX) | every diffusers 0.40+ RMSNorm | no opset-17 symbolic → **all four components fail to trace**. Fixed by `RMSNormShim`. |
| `ConstantOfShape` | `index_copy` packing, `zeros_like`, `register_tokens.expand` | converter rejects. Fixed by permutation packing and broadcast-add. |
| `If` / `Tile` | VAE decoder `.tile(2)` | converter rejects control flow. Fixed by host position grid + concat. |
| **`Exp`, `ReduceL2`, `Reciprocal`, `CastLike`** | **audio VAE (BigVGAN)** | **no workaround from the export side** |

Everything else the four graphs need is covered: `MatMul`, `Gemm`, `Mul`, `Add`,
`Slice`, `Concat`, `Reshape`, `Transpose`, `Softmax`, `RotaryEmbedding`,
`FusedElemwise`, `Reduce`, `Gather`, `Split`, `Sin`, `Cos`, `Pow`, `Sqrt`,
`Div`, `Sigmoid`, `LayerNormalization`.

`Conv3d` is **not** an issue: `Conv2dBuffer::execute` handles 5-D input (a 2-D
conv is the `D=1` degenerate case). That matters for the VAE *encoder*, which is
a causal Conv3d + GroupNorm stack — but the encoder is not exported here, and it
would additionally need `GroupNorm`, which the runtime does not have.

### Audio VAE: why it is a stub

Traced with the dynamo exporter (the TorchScript path cannot trace the
alias-free blocks at all — `F.pad` makes the conv's kernel shape unknown), the
full-size BigVGAN decoder is ~2500 nodes:

```
256 Conv, 254 Pad, 254 Exp, 135 ReduceL2, 134 ConvTranspose, 130 CastLike,
128 Shape, 127 Pow/Reciprocal/Sin/Slice, 1 Clip, ...
```

**The gap moves.** When this was first measured, `ConvTranspose`, `Pad` and
`Clip` also had no kernel. During this work another workstream added them to
`shader_map` → `runtime_op_names()`, and a re-run against the live tree reports:

```
[ops] audio_vae_decoder.onnx: 4 op type(s) have no vkop kernel:
      CastLike x1, Exp x1, Reciprocal x1, ReduceL2 x1
[audio-vae] newly covered since the gap was written: ['ConvTranspose', 'Pad', 'Clip']
```

So the still-missing set is `Exp`, `ReduceL2`, `Reciprocal`, `CastLike`:

* `Exp` — SnakeBeta's `exp(alpha)` / `exp(beta)`. Worth attention because the
  fused-elemwise program already has an `OP_EXP` code, so the kernel work may be
  mostly plumbing `Exp` into `runtime_op_names()` and the shader map.
* `ReduceL2` — weight-norm's `g * v / ||v||`.
* `Reciprocal`, `CastLike` — small, but each needs a kernel to exist.

`h3_export_audio_vae_decoder.py` re-checks this on every run rather than
trusting a frozen list, prints the inventory, and exits **4** deliberately: a
script that exits 0 would let a driver treat the component as done.

Note the audio VAE dry run builds on **CPU, not meta**: the dynamo exporter's
save path calls `tensor.cpu()` and cannot save meta weights. That is acceptable
here only because the component is small (~0.2 GiB) and because "does it trace"
is not this component's open question — the missing kernels are.

---

## Dynamic axes: what is actually dynamic

Only the things a different clip/prompt actually changes are symbolic. The
per-row feature counts are weight shapes and stay static everywhere.

**transformer**

| input | dynamic | static |
|---|---|---|
| `hidden_states` | batch, `num_video_tokens` | feature = `in_channels * pt*ph*pw` = 96 |
| `audio_hidden_states` | batch, `num_audio_tokens` | 32 |
| `encoder_hidden_states` | batch, `num_text_tokens` | 5120 |
| `temb` | `num_timesteps` | 2688 |
| `timestep_indices`, `token_tags`, `pack_order` | `seq_len` | |
| `position_ids` | `seq_len` | last axis = 3 |
| `video_indices` / `audio_indices` / `text_indices` | row count | |

The batch axis is symbolic but is **not** a second packed layout: the reference
calls it "a pure replication axis" — the index tensors describe one layout that
every batch item repeats.

**vae_decoder**: `frames`, `height`, `width` (latent) and the derived
`pixel_frames`/`pixel_height`/`pixel_width` are symbolic. The **batch axis is
deliberately static** — making it dynamic reintroduces the
`Shape`/`ConstantOfShape` chain the converter lacks. The reference decodes one
clip per call; batch is the host's chunk loop.

**text_encoder**: `batch`, `seq` only.

---

## Artifact sizes

`safetensors` `total_size` from the index files, and what that becomes:

| artifact | weights | ONNX fp16 (est.) | note |
|---|---|---|---|
| `text_encoder_hs50.onnx` + `.bin` | 62.1 GiB (1058 tensors, 14 shards) | ~62 GiB | **no `lm_head`** (151936×5120 = 1.6 GiB bf16 saved) and no visual tower run; the file still contains the tower's weights unless excluded |
| `transformer.onnx` + `.bin` | 61.7 GiB (638 tensors, 14 shards) | ~62 GiB | |
| `transformer_ref` | 61.7 GiB | — | same graph, export once |
| `vae_decoder.onnx` + `.bin` | 9.7 GiB (703 tensors, 3 shards) | ~10 GiB | encoder weights included unless excluded |
| `audio_vae_decoder.onnx` | ~0.2 GiB | — | blocked |

**Both `vae_decoder` and `text_encoder` currently carry weights the graph does
not use** (the video *encoder*; the vision tower and `lm_head`). `onnx2vkop`'s
`terminal-prune` drops unreferenced initializers, so the `.vkopbin` is smaller
than the `.onnx` suggests — but the ONNX itself is not pruned, and
`torch.onnx.export` still has to load them. On the real checkpoint that means
the export of `text_encoder` needs its component's full 62 GiB resident unless a
weight-pruning pass runs first. This is the first thing to fix before a real
run (see "What breaks first").

---

## Dry run: what it does and does not prove

`--dry-run` builds the model on `torch.device("meta")` (shapes, no storage),
traces it with `export_params=False` (weights become graph inputs, the
initializer table stays empty), prints the node inventory, then — unless
`--no-convert` — hands the weightless graph to `onnx2vkop`. Conversion of a
weightless graph is legitimate: the initializer table is empty, so it exercises
the fusion passes, `validate_supported_ops` and shader selection without a byte
of weights.

It **does** prove: the wrappers trace, the op set survives the converter's
fusion passes and `validate_supported_ops`, the shader subset resolves, and the
graph's dynamic axes are consistent.

It **does not** prove: that the weight-splicing path
(`onnx2vkop.stream_export`) resolves every parameter name against the real
shards (it does not — see the `model.` prefix below), or anything about peak
memory. Numerical correctness is covered separately, by the tiny-geometry
comparison against the unmodified diffusers modules (see "Verified status").

---

## What breaks first on a real run

In order:

1. **`text_encoder` / `vae_decoder` will try to hold their full component in
   RAM.** The trace needs every parameter, and the component is 62 GiB / 9.7 GiB.
   The fix is a pruning pass that drops `model.visual.*` + `lm_head` (text
   encoder) and the video `encoder.*` (VAE) *before* the trace, or the
   `onnx2vkop.stream_export` meta-trace + splice path. Neither is wired up here:
   `stream_export.build_graph_without_weights` requires filling in
   `Graph._export_onnx`'s positional arguments, whose count changed in recent
   torch versions (`torch/onnx/_internal/torchscript_exporter/utils.py:1553`
   vs `:1572` in 2.10), so it needs a version guard.
2. **`transformer` parameter names carry a `model.` prefix.** The wrapper holds
   the model as a submodule, so the traced parameter names are
   `model.transformer_blocks.0.attn.to_q.weight` while the checkpoint says
   `transformer_blocks.0.attn.to_q.weight`. `SafetensorsIndex.nearest` matches by
   *suffix* and requires the checkpoint name to end with `"." + traced_name`, so
   it returns `None` for every prefixed name. Measured:

   ```
   model.proj_in.weight                        -> None      # would be spliced
   proj_in.weight                              -> proj_in.weight
   ```

   Two fixes, either is fine: strip the `model.` prefix before splicing, or
   export the model *itself* with a swapped `forward`
   (`wrapper.forward = types.MethodType(fn, model)`) so the names match by
   construction. The second is better and is what I would do first — it also
   removes the prefix from the `.onnx` itself.
3. **The other two pipelines.** They cannot be built at all (see the FL2VA /
   Ref2VA section).

Beyond that, the first *numerical* thing to check is the transformer's
`pack_order`: it is the only piece of the graph whose correctness depends on the
host and the graph agreeing on a convention. The sampler must build it as
`pack_order[packed_row] = index into cat([text | video | audio])`. A transposed
permutation is a silent wrong answer, not a crash — the dry run cannot catch it.
