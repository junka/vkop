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

`text_encoder` converts and runs. With `--prompt`, the driver tokenizes the
provided text, builds the causal mask, runs the tower, and passes
`prompt_embeds` into the denoiser before loading the denoiser Runtime. Without
`--prompt`, the denoiser uses synthetic `encoder_hidden_states`; a standalone
`--text-encoder` run uses in-vocabulary ids only as a graph smoke test.

Prompt preprocessing is not yet verified against a known-good MiniMax-H3
reference: the driver currently calls `Tokenizer::encode` on the supplied text
as-is, with no automatic chat template or BOS/EOS insertion. Do not treat a
finite output as proof of semantic prompt correctness until that input contract
is confirmed.

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

### Experimental activation reuse

By default, compatible buffer-backend runs use a conservative, Runtime-local
activation buffer pool. Set `VKOP_ACTIVATION_REUSE=0` to disable it.
The buffer backend records, submits and waits for each execution level before retiring
dead floating-point intermediates. It uses exact-size buffer matches and keeps
graph inputs, outputs, weights, KV caches, and shared/view buffers out of the
pool. MatMul scratch uses separate per-level leases. Set
`VKOP_ACTIVATION_REUSE_STATS=1` to report allocation and reuse counts.

This mode trades throughput for lower peak memory. The default automatically
falls back to the normal path for non-buffer backends, replay, level hooks,
level statistics, NaN scans, or intermediate tensor dumps. Explicitly setting
`VKOP_ACTIVATION_REUSE=1` instead requires compatibility and fails if any of
those modes are requested. Retired intermediate tensors remain in the graph,
but reading their old GPU values after `Run()` is an error.
Free buffers remain cached until the Runtime is destroyed or the switch is
disabled for a subsequent run; varying shapes can therefore grow the cache.
The normal graph-submit path remains unchanged when the switch is off.

The optional `ActivationReuseCheck` build target exercises dynamic shapes,
branching, aliases and switch transitions. Generate its input with
`tests/make_activation_reuse_fixture.py` and the build's shader archive.
On b3, set `VKOP_TEST_SKIP_ATEXIT=1` for this test to bypass the known NVIDIA
EGL exit-handler crash after the test has already torn down its Runtimes.

### Split artifacts: `--prelude`

The released canvas does not fit beside the denoiser's full resident set: the
int4 denoiser is 18.6 GB, leaving ~1.8 GiB of a 20 GB card, so the packed
sequence dies around 180 rows while the real canvas needs 2016.
`model/exporter/minimax_h3/h3_split_adaln.py` splits the problem in two, because
`adaln_proj.linear` — 6.5 GB, 35% of the artifact — is a **pure function of the
timestep**:

* `<stem>_prelude.onnx` — `temb` → `adaln_table_<b>` for every block.
* `<stem>_body.onnx` — the same graph with each block's adaln `Gemm` replaced by
  a graph input `adaln_table_<b>` of shape `[T, 6, 5376]`.

Peak is then `max(prelude, body)` instead of their sum: body is 10.7 GB, leaving
9.6 GiB for activations. `temb_times` already enumerates every distinct
timestep the sampler visits, so **one** prelude pass builds the tables for all
steps and the per-step `timestep_indices` selects rows:

```bash
./minimax_h3 \
  --prompt "a cat walking on the beach at sunset, cinematic" \
  --tokenizer    h3_qwen2.bin \
  --text-encoder text_encoder_fixed.vkopbin \
  --checkpoint   /data2/wjjdata/minimax-h3/transformer \
  --prelude      prelude_fix.vkopbin \
  --transformer  split/body_bf16r.vkopbin \
  --vae          vae_bf16c.vkopbin \
  --vae-config   /data2/wjjdata/minimax-h3/vae/config.json \
  --steps 50 --latent-frames 1 --latent-h 16 --latent-w 16 --save out/
```

The prelude Runtime is destroyed before the body loads, and the tables travel
through host memory like `prompt_embeds`.

**A sliced body without `--prelude` is refused, by design.** An unfilled
`adaln_table_*` buffer reads back as **all zeros**, which zeroes AdaLN's
shift/scale/gate and silently removes both the timestep and the conditioning
from every block — no crash, no NaN, just a picture that ignores the prompt and
barely moves with `--steps`. That was the state that made 4-step and 50-step
output look identical. The driver checks both directions: it throws if a sliced
body is given without `--prelude`, and it throws if the prelude's table count or
row count disagrees with what the body/sampler expects.

### Prompt execution path: prompt in, PNG frames out

Without `--prompt` the driver feeds noise and the frames are noise. The full
path takes five inputs and produces `frame_NNN.png`:

```bash
./minimax_h3 \
  --prompt "a cat walking on the beach at sunset, cinematic" \
  --tokenizer    h3_qwen2.bin \
  --text-encoder text_encoder_fixed.vkopbin \
  --checkpoint   /data2/wjjdata/minimax-h3/transformer \
  --transformer  tf_fixed_i4.vkopbin \
  --vae          vae_fixed.vkopbin \
  --vae-config   /data2/wjjdata/minimax-h3/vae/config.json \
  --steps 50 --latent-frames 2 --latent-h 16 --latent-w 16 --save out/
```

What each piece is, and why it is host-side:

* `--tokenizer` — Qwen2 BPE. MiniMax-H3's `tokenizer/` produces a `.bin`
  **byte-identical** to the LLM's `qwen3_vl.bin`, so either works.
* `--text-encoder` — the Qwen3-VL tower truncated after `hidden_states[50]`
  (pre-final-norm, no `lm_head`). Graph inputs are `input_ids` and
  `attention_bias` (a host-built additive causal mask, `-65504` for masked).
* `--checkpoint` — the denoiser's own shards, read for the timestep MLP only:
  `time_embedder.linear_1` (`5376x256`), `linear_2` (`2688x5376`), plus biases,
  all F32. `get_timestep_embedding` (256-half sinusoid, **cos first**) →
  `linear_1` → SiLU → `linear_2`. `h3_host.hpp` has a minimal safetensors
  reader for exactly these four tensors.
* `--vae-config` — the VAE's `config.json`, for `latents_mean`/`latents_std`.
  The graph starts at `decode`, so `latents * std + mean` is host-side; the
  decoded RGB is ImageNet-normalized and is reverted by the PNG writer.

The driver runs both schedules from the checkpoint's own configs: **video
shift 12, audio shift 3**. Because they differ, one step has two distinct
timesteps, so `temb` is uploaded per step together with a matching
`timestep_indices` — that is `build_row_timesteps`'s
`unique(sorted=True, return_inverse=True)`.

The initial noise is standard normal (`randn_tensor`), not uniform.

After sampling, the driver **un-patchifies** the rows back into
`[24, F, H, W]`, de-normalizes, runs the VAE, and writes one PNG per latent
frame times `patch_t = 4`.

**Step count matters.** With shift 12 a coarse grid barely moves: at 4 steps
the first three sigmas are `1.0, 0.973, 0.923` and the run is still nearly
pure noise. The checkpoint's reference pipeline defaults to **50**.

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
produced NaN/Inf.

With the five conditioning inputs, the driver executes the host-side path:
tokenizer → tower → packed layout → per-step `temb`/indices → sampler →
un-patchify → de-normalize → decoder → PNG. Prompt tokenization/wrapping still
needs comparison with the reference pipeline, and this is not a numerical
agreement test — compare `prompt_embeds` and `sample` against a torch run on the
same ids after confirming the prompt input contract.

Note `prompt_embeds` can show a large `absmax` (~1.5e4): Qwen3-VL's
`hidden_states[50]` is pre-final-norm and carries outlier channels. That is
expected and is why the conditioner width is 5120.

### bf16 artifacts: shader changes need a reconversion

`.vkopbin` carries **its own** SPIR-V archive (`model.shader_archive_size`), and
the runtime reads shaders only from there — `libvkop` links no SPIR-V of its
own. So editing a shader and rebuilding `vkop_shaders.bin` changes nothing for
an artifact that is already on disk: the model must be **reconverted**.

While auditing a bf16 build, `cmp`-ing the `-DFP16` and `-DFP16 -DBF16` SPIR-V
is the right first move, but the verdict depends on how the shader dispatches:

| how the shader picks its width | identical SPIR-V means |
|---|---|
| runtime push-constant bit (`uPC.bf16`, as in `matmul.comp`) | **correct** |
| pure word mover, no decode (`slice`, `concat`, `reshape`) | **correct** |
| `#ifdef BF16` at compile time (`rotary`, `fused_elemwise`) | **the bf16 path is missing** |

`grep -c BF16 <shader>` returning 0 is not proof of a defect — check which row
the shader is in. Two real defects found this way:

* `fused_elemwise.comp` — 49 unpack + 51 pack sites hardcoded to fp16, so its
  `_bf16` module was byte-identical to the fp16 one. 363 nodes in the body.
* `rotary.comp` — the output pack and the cos/sin decode were both hardcoded to
  fp16, so bf16 cos/sin were decoded as fp16 and the rotation output landed at
  1e5…3e6 instead of ~1e1, overflowing the next MatMul to inf. These two are
  why any `seq` except a lucky few produced NaN.

When a bf16 run goes non-finite, `VKOP_NAN_SCAN=1` only works on a **small**
geometry (`--latent-h 4 --latent-w 4`): a full-size graph has too many levels
and dies in `vkCreateSemaphore`. Then `VKOP_DUMP_NODE=<op>` on the reported
node, and walk the ONNX graph backwards from there.

One caution on that walk: before `rotary` was fixed, only some `seq` values
produced NaN and the passing ones clustered on `seq % 8 == 0`, which read like an
alignment or shared-memory race. It was not — different `seq` simply made the
already-wrong rotation overflow to inf with different probability. A statistical
pattern in a numeric failure is not evidence about the mechanism.

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
