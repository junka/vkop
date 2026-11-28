# MiniMax-H3 FL2VA on vkop: prompt → video

This is how MiniMax-H3's text-to-audio-video pipeline runs on vkop when it comes
from the ComfyUI single-file repack (`Comfy-Org/MiniMax-H3`). It covers the
source files, the three exporters, the conversions, the driver, and the numbers
measured on an Apple M5 Max (36 GB). The `README.md` next to this file covers
the diffusers-layout base pipeline, which is a different architecture.

## 1. Pipeline

One prompt runs four models in sequence. Only one model is resident at a time.

```
prompt ──tokenize──► token ids ──[1] text tower──► layer-50 hidden [L, 5120]
                                                         │
noise ──[2] DiT ×N steps (Euler, per-stream sigma)───────┘──► video latent [24, T, H/16, W/16]
                                                              + audio latent [32, 2, Ta]
video latent ──[3] video VAE decoder──► frames [3, F, H, W] ──ffmpeg──► mp4
audio latent ──[4] audio VAE decoder──► stereo 32 kHz wav ──ffmpeg (mux)──┘
```

| # | model | role |
|---|---|---|
| 1 | Qwen3-VL-32B, first 50 of 64 LM layers | prompt → conditioning |
| 2 | FL2VA DiT, 50 blocks + 2-block token refiner | one denoising step per call |
| 3 | video VAE, ViT3D decoder, 36 layers | latent → pixels |
| 4 | audio VAE, BigVGAN decoder (7 upsamplers, 21 AMP blocks) | latent → 32 kHz stereo waveform |

The behaviour follows ComfyUI (`comfy/ldm/minimax/model.py`,
`comfy/text_encoders/minimax.py`, `comfy/ldm/minimax/vae.py`,
`blueprints/Image to Video (MiniMax H3).json`):

- **Prompt.** Raw text tokens with no chat template and no special tokens,
  tokenized with ComfyUI's Qwen2 tokenizer files. The conditioning is the
  unnormalized hidden state after layer 50: no final norm, no lm_head.
- **Sampler.** No CFG and no negative prompt (the workflow uses
  `BasicGuider`), with the `simple` scheduler.
  - Video sigma is on the shift-12 grid; audio runs on its own shift-3
    schedule.
  - The head predicts `x0 − noise`, so each stream takes the Euler step
    `x ← x + (σ − σ_next)·out`.
- **Packed sequence.** `[text | audio (stereo, channel-major) | video (2×2
  patches)]`, with per-row adaLN modulation (curve table `adaln_t_table` [1025,
  8], no SiLU).
  - RoPE: split-half over the first 96 of 128 head dims, using (t, h, w)
    positions.
- **VAE.** The latent is denormalized (`z·std + mean`). It is decoded as clips
  of 7 latent frames that overlap by 2 (a 5-frame pixel blend), each clip as
  256-px spatial tiles with an overlap of at least 64 px, then mapped as
  `pixel·imagenet_std + mean` and clamped to [0, 1].
- **Geometry.**
  - Frames snap up to 17k+5, with `latent_t = (frames−5)/17·5 + 2`.
  - The latent is 1/16 of the pixels in each spatial dimension.
  - `audio_t = round(frames/24·40)`.
  - `width` and `height` must be multiples of 32.

## 2. Source files

Everything comes from `~/.cache/modelscope/hub/models/Comfy-Org/MiniMax-H3/`.

| model | file | size | dtype |
|---|---|---|---|
| text tower | `text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors` (default) | 15.7 GB | NVFP4 + AWQ, imported as-is |
| text tower (alternative, and the embedding table) | `text_encoders/qwen3vl_32b_minimax_h3_bf16.safetensors` | 51.5 GB | bf16 (it also carries the vision tower, which t2va does not use) |
| DiT | `diffusion_models/minimax_h3_fl2va_pruned_bf16.safetensors` | 40.2 GB | bf16 (adaLN fp16, patch projections and heads fp32) |
| video VAE | `vae/minimax_h3_video_vae_fp16.safetensors` | 5.2 GB | fp16 (encoder + decoder; only the decoder is used) |
| tokenizer | ComfyUI's `comfy/text_encoders/qwen25_tokenizer/` (vocab.json, merges.txt) | – | – |

The text tower is imported from the repack's NVFP4 file as-is (§3a); the
DiT and VAE are quantized from their full-precision files by vkop's streaming
quantizer. The DiT takes the `int8_convrot` recipe (§3b) without the file
itself: rotating the bf16 weights and requantizing gives the same int4 as
requantizing the rotated int8 file (weight error 0.1172 vs 0.1176). The other
pre-quantized files (`*_fp8_scaled`, `*_w6a8`) are not used (the video
VAE's `int8_convrot` can be imported as-is, §3c, but is not the default): fp8 weights save
nothing over int8, Apple GPUs have no fp8 matmul (activations cannot go fp8),
and w6a8 has no kernel.

## 3. Exporters (`model/exporter/minimax_h3/`)

Each exporter is a pure-torch port of the ComfyUI module, built and traced on
`meta`. The real weights are spliced in by checkpoint name and streamed
(`h3_export_common.export_streaming_real`), so the export never holds a model
in RAM. Each one writes `<name>.onnx` (graph only) and `<name>.weights.bin`
(the tensors, copied as stored).

Each exporter also has `--num-layers N --reference ref.npz`, which runs the
truncated model in fp32 on CPU and saves its inputs and outputs. That is how
every graph below was verified, at 1–2 layers.

| script | graph | inputs → outputs | notes |
|---|---|---|---|
| `h3_export_fl2va_te.py` | text tower, or the slice `--layers START,END` | `inputs_embeds [L,5120]`, `attention_bias [L,L]`, `rope_cos/sin [L,128]` → `hidden [L,5120]` | writes `embed_tokens.bf16` (raw [151936,5120] bf16) for host lookup; a Gather over it would be hydrated into the protobuf and overflow 2 GB |
| `h3_export_fl2va.py` | one DiT step | `video_rows, audio_rows, text_states, pack_index, t_emb, mod_index, rope_cos/sin, video/audio_index, video/audio_t` → `video_v, audio_v` | writes `rope_inv_freq.f32` and `adaln_t_table.f32` for the host; `--dtype bf16` (default `fp32` only for the legacy fp32-activation artifact) |
| `h3_export_fl2va_audio_vae.py` | audio decode, whole clip in one call | `z [2,32,1,Ta]` (normalized; one row per stereo channel) → `wave [2,1,1,Ta·800]` | see §3d; builds on CPU (decoder 250 MB fp32), no streaming |
| `h3_export_fl2va_vae.py` | one decoder call (one clip/tile) | `z [N,24]` (denormalized), `rope_cos/sin [N+5,24]` → `pixels [N,3072]` | the zero suffix token is `h[:1]*0`, because `zeros_like` traces to ConstantOfShape, which has no kernel |
| `h3_export_fl2va_vae_encoder.py` | video VAE encoder, one 256-px tile of one image (Ref2VA refs) | `pixels [N,3,h,w]` RGB in [0,1] → `latent [N,24,h/16,w/16]` (normalized posterior mean) | see §3f; host prep and tiling in `h3_ref2va_prep.py`; builds on CPU (60 M params), no streaming |

### 3a. NVFP4 text tower (`h3_export_fl2va_te.py --nvfp4`)

How the comfy-kitchen layout of `qwen3vl_32b_minimax_h3_nvfp4_awq` is
stored:

- **Weights:** U8 `[N, K/2]` with the *even* element in the high nibble
  (`hi_first`).
- **Block scales:** e4m3 `[N, K/16]` in cuBLAS's blocked tile order.
- **Global scale:** `weight_scale_2`, one fp32 value.
- **Dequant:** `value = E2M1 × block × global`.
- **AWQ:** `pre_quant_scale`, present only on `o_proj` and `down_proj`,
  multiplies the input. For q/k/v and gate/up the smoothing is folded into
  the norm weights, so the norms must come from this file.
- **Embeddings:** stored as int8 here; the host reads rows from the bf16
  file's table instead (same weights, unquantized).

The exporter traces each Linear as `[x·pre_quant_scale] @ Wᵀ`, then replaces
the W input with ModelOpt's chain:
`DQ(W e2m1 [N,K], DQ(s e4m3 [N,K/16], g), axis=1, block_size=16)`. Each
weight's nibbles are swapped to ONNX's low-first order and its scales are
un-swizzled (`from_blocked`), one tensor at a time.
`PreQuantizedWeights` folds the chain into vkop's NVFP4 MatMul, which
runs beside bf16 activations (`Bf16ActivationTest.MatMulNvfp4WeightMatchesFp32`).

Checked against the bf16 checkpoint (layer 3): o_proj comes out at rel 0.096
of the bf16 weights and down_proj at 0.117, which is the normal FP4 level.
The other nibble order gives > 1.3. The graph matches an fp32 run of the same
dequantized weights at cos 0.99999. For the real prompt, the conditioning
agrees with the int8 tower at cos 0.9998. The NVFP4 payload is hydrated during
conversion, which peaks at 26.4 GB for the whole tower.

End to end, with the same prompt, seed 42, 50 steps and the int4 coop8 DiT, the
video looks the same as the int8-tower run (`h3_fl2va_run_dog/v2/dog_nvfp4.mp4`
next to `dog.mp4`). That run took 13.7 s/step.

### 3b. ConvRot rotation for the int4 DiT (`h3_export_fl2va.py --convrot 256`)

The repack's `int8_convrot` DiT rotates every attention/MLP linear's input
by a block-diagonal regular Hadamard H (kron powers of a symmetric 4×4,
group 256, H·H = I), so `x·Wᵀ = (x·H)·(W·H)ᵀ`. `--convrot 256` does the same:
`W·H` is applied to qkv/out/fc1/fc2 (main blocks and token refiner) while the
weights stream (`export_streaming_real(transforms=)`), and the graph gets
`reshape(x, [-1, 256]) @ H` before each of them, with H one shared bf16
initializer (±1/16 is exact in bf16 and in int4).

Measured on real weights, the rotation brings int4 group-128 error from
0.117–0.176 down to 0.117 (the Gaussian floor); the gain is largest on the
late-block fc2/qkv weights. One full-model step at 256×160×56f against
the int8 DiT (rms of the `out` difference; `out` rms ≈ 3.0 video, 1.0 audio):

| int4 DiT | video | audio | s/step |
|---|---|---|---|
| everything int4, output head included | 0.564 | 0.234 | – |
| video head + condition_proj kept wide | 0.481 | 0.234 | 4.42 |
| … plus `--convrot 256` | 0.478 | 0.161 | 4.52 |

So the head exclusion is what helps video, and the rotation is what helps
audio, for +2–3 % time. Both are in the default recipe.

Keeping the head wide uses `--exclude-node`. That flag used to switch weight
streaming off for the whole conversion: the trailer-time quantizer cannot see
an exclusion, so the converter loaded every weight into memory, and excluding
the 2 MB head made the 40 GB DiT conversion run out of memory. Now the
exclusion is resolved on the raw graph first (`exclusion.resolve_proto`):
only the excluded weights are read into memory and skipped by the in-memory
quantizer, and everything else still streams (`weight_stream.prepare(hold=)`).
A pattern that only matches names given after optimization (unnamed nodes)
falls back to the old whole-model load rather than being ignored. Test:
`test_streamed_byte_quant.py::test_exclusion_keeps_streaming_the_other_weights`.

Environment: venv `~/.venvs/h3`, Python 3.12, with torch, onnx 1.19.1,
onnxoptimizer, flatbuffers, safetensors, transformers and pytest. Put this
checkout's `model/pypi` on `PYTHONPATH`.

### 3c. Pre-quantized video VAE (`minimax_h3_video_vae_int8_convrot`)

`h3_export_fl2va_vae.py` recognizes this file from its `comfy_quant` entries
and imports it as-is. The format of its 144 decoder linears (to_qkv, to_out,
w1, w2 × 36):

- `{"format": "int8_tensorwise", "convrot": true, "convrot_groupsize": 256}`.
- I8 codes `[N, K]` of `W·H`, where H is the DiT's group-256 Hadamard
  (matches the fp16 weights rotated with it at rel 0.008).
- One fp32 `weight_scale [N, 1]` per output row, absmax/127: every row
  reaches |q| = 127.
- `x_embedder`, `proj_out`, the norms and the biases are fp32.

The exporter streams `q·scale` as fp16 and puts `x @ H` before each of these
linears. `-q int8` in the converter is the same per-output-channel absmax/127
scheme, so it gets the repack's codes back exactly. H itself is quantized to
int8 without error, because ±1/16 is exact. `--convrot 256` on the fp16 file
applies the same recipe with round-to-nearest. At 2 layers it matches the
unrotated graph at rel 1e-6.

Measured on the dog latent (512×288×56f, 18 decoder calls), against a
bf16-weight VAE, with no other GPU load:

| VAE | GPU time | total | peak | PSNR vs bf16 |
|---|---|---|---|---|
| bf16 weights | 27.9 s | 43.0 s | 5.6 GB | – |
| int8 from fp16 (default) | 26.2 s | 36.2 s | 3.4 GB | 46.31 dB (min 45.08) |
| int8_convrot as-is | 29.2 s | 39.2 s | 3.4 GB | 46.38 dB (min 45.18) |

The weight error per linear is 0.0095 → 0.0083, but that is invisible next to
the bf16 activations. The 144 online rotations cost +11 % GPU time. ConvRot
pays off in ComfyUI because it also quantizes activations to int8 (W8A8);
here they stay bf16. So the fp16 → int8 path stays the default, and the
int8_convrot file is the option when only the 2.8 GB download is wanted
(`~/models/h3_fl2va_vae_i8cr/vae.vkopbin`):

```bash
$PY h3_export_fl2va_vae.py --ckpt $C/vae/minimax_h3_video_vae_int8_convrot.safetensors \
    --out $M/h3_fl2va_vae_i8cr/vae.onnx
$CONV -i $M/h3_fl2va_vae_i8cr/vae.onnx -o $M/h3_fl2va_vae_i8cr/vae.vkopbin -q int8 --external-blob-mb 1
```

### 3d. Audio VAE (`h3_export_fl2va_audio_vae.py`)

`vae/minimax_h3_audio_vae_fp32.safetensors` (605 MB) is ComfyUI's
`MiniMaxH3AudioVAE` (`comfy/ldm/minimax/audio_vae.py`): a DAC encoder, an
attention posterior head, then `dec_in_proj` (1×1 conv 32 → 2048) and a
BigVGAN decoder (conv_pre 2048 → 1024; 7 ConvTranspose upsamplers ×5 ×5 ×2⁵ =
800 samples per latent frame; after each, 3 AMP blocks with kernels 3/7/11
and dilations 1/3/5 are averaged; anti-aliased SnakeBeta; conv_post without
bias; clamp to [-1, 1]). The stereo channels are decoded independently as a
batch of 2. Only the decode path is exported, and it needs no new runtime op.
These rewrites make it fit the existing kernels:

- Each Conv1d becomes a Conv2d over `[N, C, 1, L]`, because the runtime's Conv
  is 4-D.
- SnakeBeta `x + sin(eᵃx)² / (eᵇ + 1e-9)` gets `eᵃ` and `1/(eᵇ + 1e-9)` as
  `[1, C, 1, 1]` constants computed at export, so the graph needs no Exp or
  Reciprocal. The converter fuses the tail into FusedElemwise.
- The alias-free 2× resampler, `pad(replicate) → depthwise ConvTranspose
  stride 2 → slice → snake → pad(replicate) → depthwise Conv stride 2`, uses
  per-channel `[C, 1, 1, 12]` filter weights (ratio 2 folded in) instead of
  `filter.expand`. The replicate pads are emitted as `Pad mode=edge` with
  constant pads, through an autograd.Function symbolic: `F.pad` traces its
  pads to a ConstantOfShape/Concat chain.
- The latent denormalization `z·std + mean` is folded into `dec_in_proj`
  (`W·std`, `b + W·mean`).

The graph has 2502 ONNX nodes (1160 after conversion: 256 Conv, 120
ConvTranspose, 240 Pad, 113 Slice, 127 Sin, 134 FusedElemwise). `Ta` is
dynamic, and one call decodes the whole clip: the largest activation is
2 × 8 ch × 2·Ta·800 floats.

The activations stay fp32. Against fp32 torch on the dog latent, fp16
reaches only cos 0.9987 and bf16 0.9937, while fp32 is exact, and fp16 saves
only ~25 % of the GPU time.

### 3e. Turbo LoRAs (`h3_export_fl2va.py --lora`)

The repack ships lightx2v's step-distilled LoRAs (ComfyUI generic format:
`diffusion_model.<module>.lora_{A,B}.weight` plus a scalar `.alpha`, on
qkv/out/fc1/fc2 of all 50 blocks and the token refiner). `--lora FILE
[--lora-strength 1]` merges `W += strength·alpha/rank·B·A` while the weight
streams, before the ConvRot rotation, so the converter quantizes the merged
weight. The rest of the pipeline is unchanged. lightx2v samples them exactly as
the base driver does: Euler, no CFG, `q = (N−i)/N` grid shifted per stream
(audio shift 3).

| LoRA | steps | video shift | ΔW/W | bf16 values changed by the merge |
|---|---|---|---|---|
| `minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16` | 8 | 12 (6 looks the same) | 0.5–2e-4 | 1–4 % |
| `minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16` | 4 | 6 (`--shift-video 6`) | ~1e-3 | 20–26 % |

The 8-step delta is far below bf16 and int4 rounding, but it survives
anyway. It is low-rank and structured, and rounding of weights spread
randomly over the grid keeps it on average. The result is clearly visible:
8 steps of the base model give a flat, washed-out frame, while 8 turbo steps
look like the 50-step run. Spot check of a merged, rotated weight against
numpy: rel 0.0017, which is bf16 rounding. One step costs the same as the
base DiT (13.7–13.9 s at 512×288×56f), so the denoising loop takes
110 s (8 steps) or 55 s (4 steps) instead of 705 s.

### 3f. Video VAE encoder for Ref2VA reference images (`h3_export_fl2va_vae_encoder.py`)

Ref2VA (`MiniMaxH3ReferenceToVideo`) encodes every reference image with the
video VAE's causal CNN encoder (`EncoderFCN3D`, ch 128, mult 1/2/2/4/4/8, 2
res blocks per level) and puts the latent into the DiT's ref rows
(`latent_h = th/16`, `latent_w = tw/16`, one latent frame). What comfy does
for one image (`VAE.encode` → `MiniMaxH3VideoVAE.encode`, T = 1):

1. Resize (node): aspect kept, `match` = down only to the generation's pixel
   area, `max` = short edge ≤ 2048 (never up); each side rounded to a multiple
   of 32; `comfy.utils.lanczos` (PIL LANCZOS through uint8).
2. `process_input` x·2−1, then `_normalize_pixels` ((x+1)/2 − imagenet mean) /
   imagenet std: net, imagenet normalization of the [0, 1] pixels.
3. `tiled_encode`: 256-px tiles with ≥ 64-px overlap (`split_tiles`), each
   encoded alone (GroupNorm statistics are per tile), blended linearly in
   latent space against the unblended neighbour above, then to the left.
4. `moments[:, :, -1:]`, the posterior **mean** (first 24 of 48 channels; no
   sampling; only the H3 *image* VAEs sample), then `(mean − latents_mean) /
   latents_std` with the checkpoint's fp16-stored buffers. That normalized
   latent is what the DiT gets, the same space as its own video latent.

With one frame every `CausalConv3d` takes comfy's `autopad="causal_zero"`
path (no temporal padding, only the kernel's last temporal tap) and the
time-stride-2 downsamplers keep T = 1, so the encoder is exactly a 2-D CNN:
reflect-padded 3×3 convs on `weight[:, :, -1]`, 2-D GroupNorm(32, eps 1e-6) +
SiLU, the downsamplers as reflect pad right/bottom 1 + stride-2 conv. The
graph is that 2-D CNN for one tile, with exact folds: imagenet normalization
into `conv_in` (reflect padding commutes with a per-channel affine), and
`quant_conv`'s mean half plus the latent normalization into `conv_out`.
It needs no new runtime op (33 Conv, 30 Pad reflect, 25 InstanceNorm via the
GroupNorm reshape chain, SiLU fused into FusedElemwise); H, W and the batch
are dynamic.

Host contract (`h3_ref2va_prep.py vae IMAGE --width W --height H [--length F]
[--ref-image-size match|max] --out PREFIX`):

- `PREFIX.f32` holds `[3, th, tw]` float32 planar RGB in [0, 1], the resized
  ref; `PREFIX.json` holds `th`, `tw`, `latent_h/w` and the tile plan
  (`vae_tiles`: `y`, `x` tile starts, `y_overlap`, `x_overlap` in pixels;
  tiles are 256 px, or the whole side when it is ≤ 256).
- Per tile, run the graph on `pixels[:, y:y+h, x:x+w]` as `[1, 3, h, w]`;
  it gives `latent [1, 24, h/16, w/16]`, already normalized.
- Blend as `tiled_encode` in the prep script does: latent overlap = pixel
  overlap/16 = e; tile (i, j) first blends its top e rows with the
  *unblended* tile (i−1, j)'s bottom e rows using weights `k/e` for itself
  and `1 − k/e` for the neighbour (k = 0..e−1), then its left e columns with
  the unblended tile (i, j−1) the same way, and then drops its last e
  rows/cols unless it is the last row/column. Concatenate. Normalizing
  before blending is exact (the weights sum to 1).
- Result: `[24, th/16, tw/16]`, the DiT's ref-image latent, row-major
  (c, h, w).

Activations stay fp32. On a 256² tile against fp32 torch, fp16 gives NaN,
bf16 reaches only cos 0.99969 (max |d| 0.57), and int8 weights cos 0.99987 at
2.5× the time, while bf16 is no faster than fp32 (68 vs 69 ms).

| check (fp32 graph, red_car_beach 1024×768) | result |
|---|---|
| graph module vs literal 5-D port of comfy's encode, one 256² tile | cos 1.0000000, max \|d\| 4e-5 |
| ComfyUI's own `MiniMaxH3VideoVAE.encode` (vae.py, stubbed `comfy.*`, CPU fp32) vs vkop tiles + host blend, `match` 448×320 (4 tiles) | cos 1.0000000, max \|d\| 1.6e-4 (latent std 1.03) |
| same, `max` 1024×768 (20 tiles) | cos 1.0000000, max \|d\| 5.1e-4 |
| same, 256×192 (one non-square tile) | cos 1.0000000, max \|d\| 3e-5 |
| GPU time per 256² tile (N = 1) | 68 ms, peak 0.7 GB; 20 tiles batched: 1.07 s but 6.4 GB peak, so run tiles one at a time |

## 4. Commands

```bash
C=~/.cache/modelscope/hub/models/Comfy-Org/MiniMax-H3
M=~/models
PY="env PYTHONPATH=$PWD/model/pypi $HOME/.venvs/h3/bin/python"
CONV="$HOME/.venvs/h3/bin/python -m onnx2vkop.cli --act bf16 --target-backend buffer \
      --master-archive $PWD/build/vkop_shaders.bin"   # run from model/pypi

# text tower, NVFP4 (default): one 13.7 GB graph. The bf16 table and the
# tokenizer come from the bf16 export below (or any earlier one).
$PY h3_export_fl2va_te.py --nvfp4 --ckpt $C/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors \
    --embed $M/h3_fl2va_te/embed_tokens.bf16 --out $M/h3_fl2va_te4/te.onnx
$CONV -i $M/h3_fl2va_te4/te.onnx -o $M/h3_fl2va_te4/te.vkopbin -q bf16

# text tower, int8 from bf16 (alternative): two halves of ~12 GB each (the
# 24 GB whole does not load on 36 GB); also writes embed_tokens.bf16
mkdir -p $M/h3_fl2va_te/tokenizer && cp <ComfyUI>/comfy/text_encoders/qwen25_tokenizer/* $M/h3_fl2va_te/tokenizer/
$PY h3_export_fl2va_te.py --ckpt $C/text_encoders/qwen3vl_32b_minimax_h3_bf16.safetensors \
    --out $M/h3_fl2va_te_a/te.onnx --layers 0,25
$PY h3_export_fl2va_te.py --ckpt ... --out $M/h3_fl2va_te_b/te.onnx --layers 25,50
$CONV -i $M/h3_fl2va_te_a/te.onnx -o $M/h3_fl2va_te_a/te.vkopbin -q int8   # same for _b

# DiT, int4 + ConvRot (default; 37 GiB streamed, a few minutes per command). The
# excluded nodes are the video output head and condition_proj (look the names
# up in dit.onnx: the Gemm reading final_layer.video_out.weight / condition_proj.weight).
$PY h3_export_fl2va.py --ckpt $C/diffusion_models/minimax_h3_fl2va_pruned_bf16.safetensors \
    --out $M/h3_fl2va_cr/dit.onnx --dtype bf16 --convrot 256
$CONV -i $M/h3_fl2va_cr/dit.onnx -o $M/h3_fl2va_cr/dit_int4.vkopbin -q int4 --group-size 128 \
    --exclude-node Gemm_26143,Gemm_6339

# turbo DiTs: same export with the LoRA merged, same conversion (the exclusion
# node names come out identical; turbo_build.sh in ~/models/tools greps them)
$PY h3_export_fl2va.py --ckpt $C/diffusion_models/minimax_h3_fl2va_pruned_bf16.safetensors \
    --out $M/h3_fl2va_turbo8/dit.onnx --dtype bf16 --convrot 256 \
    --lora $C/loras/minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors
$CONV -i $M/h3_fl2va_turbo8/dit.onnx -o $M/h3_fl2va_turbo8/dit_int4.vkopbin -q int4 --group-size 128 \
    --exclude-node Gemm_26143,Gemm_6339
# (4-step: loras/minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16 -> h3_fl2va_turbo4)

# DiT without rotation (int8 reference, 20 GB; does not fit at 2687 rows)
$PY h3_export_fl2va.py --ckpt ... --out $M/h3_fl2va_bf16/dit.onnx --dtype bf16
$CONV -i $M/h3_fl2va_bf16/dit.onnx -o $M/h3_fl2va_bf16/dit_int8.vkopbin -q int8

# VAE decoder
$PY h3_export_fl2va_vae.py --ckpt $C/vae/minimax_h3_video_vae_fp16.safetensors \
    --out $M/h3_fl2va_vae/vae.onnx
$CONV -i $M/h3_fl2va_vae/vae.onnx -o $M/h3_fl2va_vae/vae.vkopbin -q int8 --external-blob-mb 1

# audio VAE decoder (fp32, 268 MB; seconds). --reference also checks the
# graph against a literal fp32 port of comfy's decode.
$PY h3_export_fl2va_audio_vae.py --ckpt $C/vae/minimax_h3_audio_vae_fp32.safetensors \
    --out $M/h3_fl2va_avae/avae.onnx [--reference ref.npz --frames 16]
$CONV --act fp32 -q fp32 -i $M/h3_fl2va_avae/avae.onnx -o $M/h3_fl2va_avae/avae_fp32.vkopbin

# video VAE encoder for Ref2VA refs (fp32, 248 MB; seconds). --reference
# checks the graph against a literal 5-D port of comfy's encode.
$PY h3_export_fl2va_vae_encoder.py --ckpt $C/vae/minimax_h3_video_vae_fp16.safetensors \
    --out $M/h3_ref2va/vae_enc/vae_enc.onnx [--reference ref.npz]
$CONV --act fp32 -q fp32 -i $M/h3_ref2va/vae_enc/vae_enc.onnx -o $M/h3_ref2va/vae_enc/vae_enc_fp32.vkopbin
python3 h3_ref2va_prep.py vae ref.png --width 512 --height 288 --out ref0   # ref0.f32 + ref0.json

# driver
make -C build -j16 minimax_h3_fl2va
```

Run (two processes, so the system reclaims the text tower before the DiT
loads):

```bash
# tokens (raw, no template)
$HOME/.venvs/h3/bin/python -c "from transformers import Qwen2Tokenizer; import numpy as np; \
  t=Qwen2Tokenizer.from_pretrained('$M/h3_fl2va_te/tokenizer'); \
  np.asarray(t('<prompt>', add_special_tokens=False)['input_ids'], np.int64).tofile('ids.i64')"

export VKOP_ACTIVATION_REUSE=1     # required: dynamic shapes, see §6
build/minimax_h3_fl2va --te $M/h3_fl2va_te4/te.vkopbin \
    --embed $M/h3_fl2va_te/embed_tokens.bf16 --prompt-ids ids.i64 --text-out text_states.f32 \
    --dit $M/h3_fl2va_cr/dit_int4.vkopbin --width 512 --height 288 --length 56
build/minimax_h3_fl2va --text-states text_states.f32 \
    --dit $M/h3_fl2va_cr/dit_int4.vkopbin \
    --width 512 --height 288 --length 56 --steps 50 --seed 42 --save run/ \
    --vae $M/h3_fl2va_vae/vae.vkopbin --frames run/frames \
    --audio-vae $M/h3_fl2va_avae/avae_fp32.vkopbin --wav run/audio.wav
# turbo: --dit $M/h3_fl2va_turbo8/dit_int4.vkopbin --steps 8
#    or  --dit $M/h3_fl2va_turbo4/dit_int4.vkopbin --steps 4 --shift-video 6
ffmpeg -framerate 24 -i run/frames/frame_%04d.ppm -i run/audio.wav \
    -c:v libx264 -pix_fmt yuv420p -c:a aac -b:a 192k out.mp4
```

`--latent FILE` decodes a saved latent with only the VAE loaded.
`--audio-latent FILE` does the same for a saved audio latent
(`audio_latent.f32`, [32, 2, Ta]), with only the audio VAE loaded. The audio
is decoded after the DiT is released and before the video VAE loads. The
`.wav` holds 32-bit float stereo at 32 kHz.
`$M/tools/run_guard.sh LOG cmd...` runs a command under a memory guard: it
kills the command when free memory falls under 3% or pressure stays critical
for 6 s, and records the peak footprint.

## 5. Precision

| model | weights | activations | artifact |
|---|---|---|---|
| text tower | NVFP4 + AWQ, imported (int8 alternative: 2 × 12.2 GB) | bf16 | 13.7 GB, plus a 1.56 GB host embedding table |
| DiT | int4 group 128 on ConvRot-rotated weights; video head and condition_proj bf16 (int8: 20.2 GB) | bf16 | 10.8 GB (after a `.comp` change: `tools/patch_model_shaders.py`) |
| VAE decoder | int8 | bf16 | 2.45 GB |
| audio VAE decoder | fp32 | fp32 | 268 MB |
| video VAE encoder (Ref2VA) | fp32 | fp32 | 248 MB |

Norms and biases stay bf16. The sampler state lives on the host in fp32.

Why every model runs bf16 activations:

- **DiT:** H3's hidden states overflow fp16.
- **VAE:** the decoder gives all-NaN output in fp16.
- **Text tower:** its hidden states reach |x| ≈ 15k.

Quantized weights beside bf16 activations needed new support in this
checkout:

- `core/Bf16Gate.cpp` admits `[A bf16, W int8|int4, scale fp32]` MatMuls.
- `fp16_graph.py` keeps them in the bf16 domain.
- `matmul_coop8.comp` gained an int8 B stage (codes as exact halves, column
  scale in the epilogue) and an int4 B stage (nibbles dequantized with their
  [K/group, N] scale while staging), both bf16-only.
- Tests: `Bf16ActivationTest.MatMulInt8WeightMatchesFp32` and
  `MatMulInt4WeightMatchesFp32`, plus `test_bf16_activation_gate.py`.

## 6. Verification and measurements (M5 Max)

| check | result |
|---|---|
| each graph at 1–2 layers vs fp32 torch port | DiT cos 0.99996, text tower cos 0.99998, VAE cos 0.99995, split text tower chained cos 0.99998 |
| full 50-layer DiT, one step, bf16+int8 coop8 vs fp32 | video cos 0.99982, audio cos 0.99857 |
| VAE decode of a sampled latent | coherent image |
| torch port vs ComfyUI's own code | **not run** (ported by reading the source); the audio VAE is the exception, see below |
| audio VAE graph (fp32) vs literal fp32 port of comfy decode, dog latent Ta = 93 | cos 1.0000000, max abs diff 8e-6 (vkop run) |
| audio VAE: ComfyUI's own `MiniMaxH3AudioVAE` (stubbed `comfy.ops`) encodes a 1 s test signal (440 + 3000 Hz L, gated 660 Hz R); vkop decodes the latent | vs comfy's decode cos 1.0000000 on both channels; spectral peaks at 440/3000 and 660 Hz |

| config | s/step | peak |
|---|---|---|
| DiT int8, fp32 activations, 554 rows (reduce route) | 46.8 | 23.3 GB |
| DiT int8, bf16, 554 rows (SIMT tile) | 4.2 | 21.3 GB |
| DiT int8, bf16, 554 rows (coop8) | 2.3 | 21.4 GB |
| DiT int4, bf16, 512×288×56f = 2687 rows (SIMT tile) | 23.1 | 19.8 GB |
| DiT int4, bf16, 2687 rows (coop8 int4 stage) | 12.8 | 19.8 GB |
| DiT int4 + ConvRot 256, head wide, 2687 rows (coop8) | 14.1 | 19.9 GB |
| same + turbo 8-step LoRA, 8 steps | 13.7 (110 s total) | 19.9 GB |
| same + turbo 4-step LoRA, 4 steps, shift 6 | 13.8 (55 s total) | 19.9 GB |
| VAE, 56 frames 512×288 (3 clips × 6 tiles) | 29 s total | – |
| DiT int8, bf16, 2687 rows | – | 26.3 GB, killed: does not fit beside the other processes on 36 GB |
| text tower int8, 53 tokens, two halves | 4.3 s total (run 0.7 s) | ~12 GB per half |
| text tower NVFP4, 53 tokens, one graph | 2.5 s total (run 0.7 s) | 13.2 GB |
| audio VAE fp32, Ta = 93 (2.3 s of audio) | 0.48 s GPU, 0.75 s with load (first run on a cold pipeline cache: 2–3 s) | 1.1 GB |
| audio VAE fp32, Ta = 202 (5 s) | 0.99 s GPU, 1.2 s total | 1.8 GB |

The decoded dog-prompt audio is finite and quiet. 50-step int4 runs give
rms 0.007–0.009 (≈ −42 dBFS) with most energy at 4–12 kHz, a hiss-like spectrum
(flatness 0.35–0.5). The 8-step turbo run (`t8_s12`) gives rms 0.048 with
most energy at 100–1000 Hz. The sampled audio latents have rms 0.47–0.56,
where an in-distribution latent is ≈ 1 (the test tone encodes to 0.65), so
the quiet output most likely comes from the DiT/sampler, not from the
decoder. The decoder itself matches comfy exactly.

`VKOP_ACTIVATION_REUSE=1` is required. The DiT graph has dynamic shapes, so
the load-time shape pool does not apply, and without reuse every intermediate
keeps its own buffer, which fills memory on the first run. With reuse on, the
peak is weights plus a few GB.

## 7. Artifacts (`~/models`)

| path | used | note |
|---|---|---|
| `h3_fl2va_te4/te.vkopbin` | yes | NVFP4 text tower |
| `h3_fl2va_te_a/te.vkopbin`, `h3_fl2va_te_b/te.vkopbin` | alternative | int8 text tower layers 0–24 / 25–49 |
| `h3_fl2va_te/embed_tokens.bf16`, `h3_fl2va_te/tokenizer/` | yes | |
| `h3_fl2va_cr/dit_int4.vkopbin` (+ its host tables) | yes | DiT, int4 + ConvRot, 50 steps |
| `h3_fl2va_turbo8/dit_int4.vkopbin`, `h3_fl2va_turbo4/dit_int4.vkopbin` | yes | same with the 8-step / 4-step turbo LoRA merged (10.8 GB each) |
| `h3_fl2va_bf16/dit_int4.vkopbin`, `dit_int4h.vkopbin`, `dit_int8.vkopbin` | comparison | DiT without rotation (all int4 / head wide / int8) |
| `h3_fl2va_bf16/rope_inv_freq.f32`, `adaln_t_table.f32` | yes | DiT host tables (the driver reads them from the DiT's directory) |
| `h3_fl2va_vae/vae.vkopbin` | yes | |
| `h3_fl2va_avae/avae_fp32.vkopbin` | yes | audio VAE decoder (`avae_fp16`/`avae_bf16.vkopbin`: less accurate comparisons) |
| `h3_ref2va/vae_enc/vae_enc_fp32.vkopbin` | Ref2VA | video VAE encoder, one tile (§3f) |
| `*/…weights.bin` | no | export intermediates, only needed to reconvert |

The fp32-activation DiT, the unsplit tower and the 1–2-layer test models were
deleted on 2026-11-28. The exporters rebuild any of them.

## 8. Open items

- The 4-step LoRA is trained for 768p; try it at 1344×768 (needs the
  activation budget at ~3× the rows).
- The out_proj rotation could be folded into the V projection (per-head,
  group 64 or 128) instead of running online.
- Audio VAE speed: ~60 % of its GPU time goes to the 127 alias-free
  activations (6 ops each). One fused `pad → up → snake → down` kernel would
  remove ~760 dispatches. The audio VAE is < 1 s of a multi-minute run, so
  this has not been done.
- Audio loudness: the sampled audio latents are under-scaled (rms ≈ 0.5,
  see §6). Check the audio stream of the DiT/sampler against ComfyUI.
- A numeric check of the torch ports against ComfyUI itself.
- FL2VA keyframes (first/last frame conditioning): the layout supports them
  in ComfyUI; the driver only does plain t2va.
