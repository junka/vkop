# MiniMax-H3 on vkop

A driver for the `.vkopbin` artifacts the H3 exporters produce, plus the
host-side contract they expect.

MiniMax-H3 is a **joint video + audio diffusion model**, not an LLM. The
sampler, both schedulers, the packed-layout construction, the timestep MLP and
the VAE's rotary grid all live on the host; the graphs carry one denoising step,
one decode, and one text-tower forward. `model/exporter/minimax_h3/README.md`
is the authority on that split — read it first.

## Components and what this driver covers

| component | graph | this driver |
|---|---|---|
| `transformer` | one denoising step, `[text \| video \| audio]` packed | ✅ full |
| `transformer --no-with-audio` | same, video only | ✅ full |
| `vae_decoder` | ViT latent → pixels | ✅ full |
| `text_encoder` | Qwen3-VL tower → `prompt_embeds` | ✅ full |
| `audio_vae_decoder` | BigVGAN waveform | ❌ not converted (missing ops) |

`text_encoder` converts and runs. The driver feeds in-vocabulary ids and a
causal mask (it has no tokenizer), runs the tower, and reports the
`prompt_embeds` statistics — a smoke test of the graph contract, not a real
prompt. Wiring it to the denoiser (tokenize → embed → pack → denoise) is the
production driver's job, so the denoiser here still gets synthetic
`encoder_hidden_states`; the output is a **finiteness/shape smoke test**, not a
claim that the video looks right.

### The text encoder's dtype (a real bug this fixed)

The tower was traced in torch's **fp32 default** while `attention_bias` — a
graph *input*, made fp16 to match HF's `create_causal_mask` — stayed fp16. Every
other tensor being fp32, HF's `attn_weights.float()` softmax path inserted a
`Cast(to=fp32)` between the mask and the attention. That Cast seeded an FP32
island; the `-q fp16` domain pass refused the graph with **250 nodes mixing
FP32 and FP16 operands**, and conversion used to discard that refusal and run
weight-only quantization anyway — shipping an artifact whose fused `Attention`
got an fp32 mask beside fp16 Q/K/V, which the runtime rejects
(`ops/Attention.hpp`: "the attention bias must have Q's element type").

Two fixes:

* `h3_export_text_encoder.build_text_model` moves the tower to fp16 before the
  trace (the LLM exporter already builds its tower in fp16). `Cast(attention_bias)`
  goes 2 → 0 and the graph is coherent. bf16 does **not** help — the mask is
  fp16 either way.
* `converter.apply_optimizations` now **fails loudly** when the fp16 domain pass
  refuses, instead of silently falling back. Weight-only quantization behind a
  refused pass is exactly the width mix the pass exists to prevent.

## Build

```bash
cmake -DENABLE_EXAMPLE=ON .. && make minimax_h3
```

Needs a Vulkan device. It skips `llvmpipe` on purpose — timing a software
rasterizer proves nothing about the target.

## Run

```bash
# the denoiser, 2 sampler steps, default tiny geometry
./minimax_h3 --transformer transformer.vkopbin

# video-only variant
./minimax_h3 --transformer transformer_novid.vkopbin --no-audio

# the VAE decoder (latent 2x16x16 -> 8x256x256 pixels)
./minimax_h3 --vae vae_decoder.vkopbin

# both, and keep the outputs
./minimax_h3 --transformer tf.vkopbin --vae vae.vkopbin \
             --steps 4 --latent-h 16 --latent-w 16 --save out/
```

Knobs: `--steps`, `--text-tokens`, `--latent-frames`, `--latent-h`,
`--latent-w`, `--audio-latents`, `--no-audio`, `--save DIR`.

### Shape rules the driver encodes

These are weight shapes in the graph, so the driver treats them as constants and
only the **row counts** vary:

* `hidden_states` is `[B, n_video_rows, 96]` — `in_channels(24) * 1*2*2`.
* `audio_hidden_states` is `[B, n_audio_rows, 32]`.
* `n_video_rows = latent_frames * (latent_h / 2) * (latent_w / 2)`.
* `n_audio_rows = audio_latents * 2` (channel-major, 2 channels).
* `temb` is `[num_timesteps, 2688]`; the graph wants **one row per distinct
  timestep**, and `timestep_indices` maps every packed row to its row.

### VAE decoder

`latents` is `[1, 24, frames, h, w]`; the ViT makes **one token per latent
voxel**, appends `num_register_tokens + 1 = 5` suffix rows, and
`position_ids` must be `[1, frames*h*w + 5, 3]`. Getting that `+5` wrong is the
easy mistake: a short grid does not fail loudly upstream, it shows up as a
`Softmax` shape error deep in the first attention block.

## What "OK" means

The driver prints, per output, `n` / `finite` / `min` / `max` / `mean`, writes
the tensors when asked, and exits non-zero only on a real problem. A run that
prints `OK` has proven: the artifact loads on this device, every graph input's
recorded dtype and shape line up, the plan executes, and no intermediate
produced NaN/Inf. It has **not** proven numerical agreement with the reference
model — that needs real conditioning (text tower + checkpoint timestep MLP +
scheduler), which is the production driver's job, not an example's.

### Measured on b3 (RTX 4000 Ada, 2026-11-23/24)

`tf_fixed_i4.vkopbin` (int4, 18.6 GB), `vae_fixed.vkopbin` (fp16, 4.6 GB), and
`text_encoder_fp16.vkopbin` (fp16, 50.3 GB):

```
[transformer] layout: text=8 audio=8 video=128 seq=144 steps=2
  step 1/2  run 13979.1 ms   sample finite=yes   audio_sample finite=yes
  step 2/2  run 12485.4 ms   sample finite=yes   audio_sample finite=yes
[vae] latents 1x2x16x16 -> 512 tokens + 5 suffix
  run 2053.4 ms              video  n=1572864 finite=yes
[text_encoder] seq=16
  run 3566.1 ms              prompt_embeds n=81920 finite=yes
OK
```

The tiny geometry (`--latent-h 4 --latent-w 4`, 24-row sequence) runs in
2.8-4.0 s/step; the 144-row sequence above is ~13 s/step. These are the
*converted-artifact* costs, not a statement about real sampling settings.

**Prerequisite:** an artifact converted before the `_shape_known` fix (see
`model/pypi/onnx2vkop/optimizer.py`) carries a spurious `0` dim and dies with a
SIGFPE in `Softmax`. If a run crashes there, reconvert; `onnx2vkop.inspect` on
the `.vkopbin` shows the `0`.


On b3 (RTX 4000 Ada) the process prints `OK` and then dies with SIGSEGV in the
NVIDIA EGL `atexit` handler. **Judge these runs by the printed result, not the
exit code** — see `memory/b3-bench-harness-crosscompile.md` in the vault.
