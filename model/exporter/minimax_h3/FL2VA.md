# MiniMax-H3 FL2VA on vkop: prompt → video

This is how MiniMax-H3's text-to-audio-video pipeline runs on vkop when it comes
from the ComfyUI single-file repack (`Comfy-Org/MiniMax-H3`). It covers the
source files, the three exporters, the conversions, the driver, and the numbers
measured on an Apple M5 Max (36 GB). The `README.md` next to this file covers
the diffusers-layout base pipeline, which is a different architecture.

## 1. Pipeline

One prompt runs three models in sequence. Only one model is resident at a time.

```
prompt ──tokenize──► token ids ──[1] text tower──► layer-50 hidden [L, 5120]
                                                         │
noise ──[2] DiT ×N steps (Euler, per-stream sigma)───────┘──► video latent [24, T, H/16, W/16]
                                                              (+ audio latent, not decoded)
video latent ──[3] video VAE decoder──► frames [3, F, H, W] ──ffmpeg──► mp4
```

| # | model | role |
|---|---|---|
| 1 | Qwen3-VL-32B, first 50 of 64 LM layers | prompt → conditioning |
| 2 | FL2VA DiT, 50 blocks + 2-block token refiner | one denoising step per call |
| 3 | video VAE, ViT3D decoder, 36 layers | latent → pixels |
| – | audio VAE (BigVGAN) | not run: missing runtime ops; the audio latent is generated but not decoded |

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
| text tower | `text_encoders/qwen3vl_32b_minimax_h3_bf16.safetensors` | 51.5 GB | bf16 (it also carries the vision tower, which t2va does not use) |
| DiT | `diffusion_models/minimax_h3_fl2va_pruned_bf16.safetensors` | 40.2 GB | bf16 (adaLN fp16, patch projections and heads fp32) |
| video VAE | `vae/minimax_h3_video_vae_fp16.safetensors` | 5.2 GB | fp16 (encoder + decoder; only the decoder is used) |
| tokenizer | ComfyUI's `comfy/text_encoders/qwen25_tokenizer/` (vocab.json, merges.txt) | – | – |

The repack's pre-quantized files (`*_int8_convrot`, `*_fp8_scaled`,
`*_w6a8`, `*_nvfp4_awq`) are not used. Every model is quantized from its
full-precision file by vkop's own streaming quantizer. Importing
`int8_convrot` directly is still open: it needs a grouped Hadamard rotation of
the activations and streaming of int8 DequantizeLinear payloads.

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
| `h3_export_fl2va_vae.py` | one decoder call (one clip/tile) | `z [N,24]` (denormalized), `rope_cos/sin [N+5,24]` → `pixels [N,3072]` | the zero suffix token is `h[:1]*0`, because `zeros_like` traces to ConstantOfShape, which has no kernel |

Environment: venv `~/.venvs/h3`, Python 3.12, with torch, onnx 1.19.1,
onnxoptimizer, flatbuffers, safetensors, transformers and pytest. Put this
checkout's `model/pypi` on `PYTHONPATH`.

## 4. Commands

```bash
C=~/.cache/modelscope/hub/models/Comfy-Org/MiniMax-H3
M=~/models
PY="env PYTHONPATH=$PWD/model/pypi $HOME/.venvs/h3/bin/python"
CONV="$HOME/.venvs/h3/bin/python -m onnx2vkop.cli --act bf16 --target-backend buffer \
      --master-archive $PWD/build/vkop_shaders.bin"   # run from model/pypi

# text tower: two halves of ~12 GB each (the 24 GB whole does not load on 36 GB)
mkdir -p $M/h3_fl2va_te/tokenizer && cp <ComfyUI>/comfy/text_encoders/qwen25_tokenizer/* $M/h3_fl2va_te/tokenizer/
$PY h3_export_fl2va_te.py --ckpt $C/text_encoders/qwen3vl_32b_minimax_h3_bf16.safetensors \
    --out $M/h3_fl2va_te_a/te.onnx --layers 0,25
$PY h3_export_fl2va_te.py --ckpt ... --out $M/h3_fl2va_te_b/te.onnx --layers 25,50
$CONV -i $M/h3_fl2va_te_a/te.onnx -o $M/h3_fl2va_te_a/te.vkopbin -q int8   # same for _b

# DiT (37 GiB streamed, ~1 min)
$PY h3_export_fl2va.py --ckpt $C/diffusion_models/minimax_h3_fl2va_pruned_bf16.safetensors \
    --out $M/h3_fl2va_bf16/dit.onnx --dtype bf16
$CONV -i $M/h3_fl2va_bf16/dit.onnx -o $M/h3_fl2va_bf16/dit_int4.vkopbin -q int4 --group-size 128
$CONV -i $M/h3_fl2va_bf16/dit.onnx -o $M/h3_fl2va_bf16/dit_int8.vkopbin -q int8   # faster, 20 GB

# VAE decoder
$PY h3_export_fl2va_vae.py --ckpt $C/vae/minimax_h3_video_vae_fp16.safetensors \
    --out $M/h3_fl2va_vae/vae.onnx
$CONV -i $M/h3_fl2va_vae/vae.onnx -o $M/h3_fl2va_vae/vae.vkopbin -q int8 --external-blob-mb 1

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
build/minimax_h3_fl2va --te $M/h3_fl2va_te_a/te.vkopbin,$M/h3_fl2va_te_b/te.vkopbin \
    --embed $M/h3_fl2va_te/embed_tokens.bf16 --prompt-ids ids.i64 --text-out text_states.f32 \
    --dit $M/h3_fl2va_bf16/dit_int4.vkopbin --width 512 --height 288 --length 56
build/minimax_h3_fl2va --text-states text_states.f32 \
    --dit $M/h3_fl2va_bf16/dit_int4.vkopbin \
    --width 512 --height 288 --length 56 --steps 50 --seed 42 --save run/ \
    --vae $M/h3_fl2va_vae/vae.vkopbin --frames run/frames
ffmpeg -framerate 24 -i run/frames/frame_%04d.ppm -c:v libx264 -pix_fmt yuv420p out.mp4
```

`--latent FILE` decodes a saved latent with only the VAE loaded.
`$M/tools/run_guard.sh LOG cmd...` runs a command under a memory guard: it
kills the command when free memory falls under 3% or pressure stays critical
for 6 s, and records the peak footprint.

## 5. Precision

| model | weights | activations | artifact |
|---|---|---|---|
| text tower | int8, per output channel | bf16 | 2 × 12.2 GB, plus a 1.56 GB host embedding table |
| DiT | int4, group 128 (int8: 20.2 GB) | bf16 | 10.8 GB (after a `.comp` change: `tools/patch_model_shaders.py`) |
| VAE decoder | int8 | bf16 | 2.45 GB |

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
| torch port vs ComfyUI's own code | **not run** (ported by reading the source) |

| config | s/step | peak |
|---|---|---|
| DiT int8, fp32 activations, 554 rows (reduce route) | 46.8 | 23.3 GB |
| DiT int8, bf16, 554 rows (SIMT tile) | 4.2 | 21.3 GB |
| DiT int8, bf16, 554 rows (coop8) | 2.3 | 21.4 GB |
| DiT int4, bf16, 512×288×56f = 2687 rows (SIMT tile) | 23.1 | 19.8 GB |
| DiT int4, bf16, 2687 rows (coop8 int4 stage) | 12.8 | 19.8 GB |
| VAE, 56 frames 512×288 (3 clips × 6 tiles) | 29 s total | – |
| DiT int8, bf16, 2687 rows | – | 26.3 GB, killed: does not fit beside the other processes on 36 GB |
| text tower, 53 tokens, two halves | 4.3 s total (run 0.7 s) | ~12 GB per half |

`VKOP_ACTIVATION_REUSE=1` is required. The DiT graph has dynamic shapes, so
the load-time shape pool does not apply, and without reuse every intermediate
keeps its own buffer, which fills memory on the first run. With reuse on, the
peak is weights plus a few GB.

## 7. Artifacts (`~/models`)

| path | used | note |
|---|---|---|
| `h3_fl2va_te_a/te.vkopbin`, `h3_fl2va_te_b/te.vkopbin` | yes | text tower layers 0–24 / 25–49 |
| `h3_fl2va_te/embed_tokens.bf16`, `h3_fl2va_te/tokenizer/` | yes | |
| `h3_fl2va_bf16/dit_int4.vkopbin` (`dit_int8.vkopbin`) | yes | DiT |
| `h3_fl2va_bf16/rope_inv_freq.f32`, `adaln_t_table.f32` | yes | DiT host tables (the driver reads them from the DiT's directory) |
| `h3_fl2va_vae/vae.vkopbin` | yes | |
| `*/…weights.bin` | no | export intermediates, only needed to reconvert |

The fp32-activation DiT, the unsplit tower and the 1–2-layer test models were
deleted on 2026-11-28. The exporters rebuild any of them.

## 8. Open items

- Import `int8_convrot` directly (Hadamard rotation in the graph, streamed
  int8 DQ payloads).
- Audio VAE (BigVGAN ops).
- A numeric check of the torch ports against ComfyUI itself.
- FL2VA keyframes (first/last frame conditioning): the layout supports them
  in ComfyUI; the driver only does plain t2va.
