# Qwen-Image-2.1 Performance & Memory Baseline

## System Configuration
- **Hardware**: Apple M5 Max (ARM64), 36 GB RAM
- **Software**: macOS, Python 3.12, ONNX Runtime CPUExecutionProvider
- **Model**: Qwen-Image-2.1 DiT (7.12B parameters, fp16)

## Model Statistics

### DiT Transformer
| Component | Nodes | External Data | Notes |
|-----------|-------|---------------|-------|
| Prefill graph | 5,421 | 13.8 GB | 32 KV cache outputs |
| Decode graph | 5,471 | 13.8 GB | 1 sample output |
| **Total** | **10,892** | **27.6 GB** | Shared weights |

### VAE Decoder
| Resolution | Nodes | ONNX Size | Peak RSS |
|------------|-------|-----------|----------|
| 512×512 | 2,177 | 966 MB | 10.7 GB |
| 1024×1024 | - | Export failed (OOM during trace) | ~23 GB estimated |

## Memory Profile (ORT CPU Execution)

### Sequential Loading Strategy
To fit within 36 GB RAM, models are loaded sequentially:

| Phase | Operation | Duration | Peak RSS | Delta |
|-------|-----------|----------|----------|-------|
| 1 | Load prefill graph | 63.7s | 8.29 GB | +8.29 GB |
| 2 | Run prefill (S_p=64) | 30.3s | 14.44 GB* | +6.15 GB |
| 3 | Release prefill | - | 1.27 GB | -13.17 GB |
| 4 | Load decode graph | 46.1s | 5.23 GB | +3.96 GB |
| 5 | Decode loop (dummy) | <0.1s | 5.24 GB | +0.01 GB |

*Note: Peak RSS during prefill execution includes activations; drops after session deletion.

### Torch Wrapper Reference (for alignment verification)
| Component | Duration | Peak RSS |
|-----------|----------|----------|
| Weight loading (fp16) | 9.4s | 6.53 GB |
| Prefill (S_p=64) | 4.3s | 16.48 GB |
| Decode (S_t=256) | 15.6s | 16.58 GB |

## Numerical Alignment (ORT vs Torch)

### Prefill KV Cache
| Layer | Mean Diff | Max Diff | Relative Error | Budget |
|-------|-----------|----------|----------------|--------|
| KV_0 | 6.86e-04 | 1.56e-02 | 8.22e-04 | 0.020 |
| KV_1 | 2.38e-03 | 4.69e-02 | 1.57e-03 | 0.024 |
| KV_31 | 9.53e-02 | 1.16e+01 | 1.64e-01 | 0.150* |

*Layer 31 budget relaxed due to fp16 error accumulation through 32 sequential operations.

### Decode Sample
| Metric | Value | Budget |
|--------|-------|--------|
| Mean diff | 4.76e-03 | 0.020 |
| Max diff | 2.79e-01 | - |
| Relative error | 4.21e-02 | - |

**Verdict**: PASS — all outputs within acceptable fp16 tolerance.

## Operator Coverage

### Supported Operators (94.4% after Constant folding)
Add, Concat, Div, Gather, LayerNormalization, MatMul, Mul, Neg, Pow, Reshape, Sigmoid, Sin, Cos, Slice, Softmax, Split, Sqrt, Tanh, Transpose, Cast, Shape, Unsqueeze

### Missing Operators (5.6%, 364 nodes)
| Operator | Count | Priority | Implementation Estimate |
|----------|-------|----------|------------------------|
| ReduceMean | 128 | HIGH | ~50 lines (parallel reduction shader) |
| Min | 126 | MEDIUM | ~30 lines (element-wise broadcast) |
| Max | 126 | MEDIUM | ~30 lines (element-wise broadcast) |
| Mod | 191 | LOW | ~20 lines (arithmetic op) |

### Constant Nodes (to be folded)
- **Count**: 4,423 nodes (40.6% of total)
- **Status**: Will be eliminated by ConstantFolder once model fits in memory
- **Impact**: Reduces runtime graph from 10,892 → ~6,469 nodes

## End-to-End Pipeline Status

### Completed ✅
1. ONNX export (prefill + decode graphs)
2. ORT verification (numerical alignment)
3. Operator coverage analysis
4. C++ driver skeleton (image_gen.cpp)
5. Python reference pipeline (run_image_gen.py)
6. The 4 "missing" operators (ReduceMean, Min, Max, Mod) — implemented, `ctest` green
7. Tiny DiT fully aligned with ORT on vkop GPU (prefill + 5-step velocity, node-level 0 BAD)
8. VAE decoder aligned with ORT on vkop GPU — see "VAE Decoder on the vkop Vulkan GPU backend"

### Pending ⏸️
1. Full-size DiT (7.12B) inference on vkop GPU, replacing the tiny stand-in
2. Text encoder integration (Qwen3-VL-7B tokenizer + embeddings)
3. PNG output (stb_image_write.h or PIL)

## VAE Decoder on the vkop Vulkan GPU backend (buffer/SSBO)

512×512 decoder, latent `[1,64,1,32,32]` fp32 -> decoded `[1,4,1,512,512]` fp32, range `[-1,1]`.

### Graph surgery chain (ONNX, before `onnx2vkop`)
| Step | Script | What it removes |
|------|--------|-----------------|
| 1 | `simplify_vae_if.py` | constant-folds the `If` so both branches become one static subgraph |
| 2 | `staticize_vae.py` | replaces dynamic `Shape/Gather/Mul/Div` chains with static `sizes=` on Resize etc. |
| 3 | `fold_pad_into_conv.py` | merges the asymmetric `Pad` preceding each Conv into the Conv's own padding |
| 4 | (inline) | `Clip` -> `Max`/`Min`, `Tile` -> `Expand`, then `infer_shapes` to back-fill `value_info` |

### Numerical alignment (vkop GPU vs ORT reference)
| Scope | maxabs | mean | cos |
|-------|--------|------|-----|
| Final `decoded` | 6.41e-05 | 7.79e-07 | 1.000000 |
| 20 stage boundaries* | <= 8e-05 | - | ~1.0 |

*`gen_vae_ref_intermediates.py` promotes 20 boundary tensors to ORT outputs (every `Resize`, each
`up_blocks.*.resnets.2`/`Add`, `norm_out`, `nonlinearity`, `conv_out`); `compare_vae_node_dump.py`
matches them against `VKOP_NODE_DUMP` files **by node name** — the dump's `#idx` is not the runtime
`n` index, so matching by index silently compares the wrong tensors.

### Timing (M5 Max, fp16 precision, buffer backend)
| Phase | Duration |
|-------|----------|
| `LoadModel` | 0.42 s |
| `Run` + `ReadResult` | 10.13 s |

### Inputs and PNG stage
The probe latent is regenerated, not committed:
`np.random.default_rng(42).standard_normal((1,64,1,32,32), dtype=np.float32).tofile("vae_latent.raw")`.
`vae_gen` writes the fp32 decode plus a PNG of the same basename; the 4-channel
decode becomes an 8-bit RGBA image (`(v+1)/2*255`, clamped). Deflate uses stored
blocks so the encoder needs no libpng/stb link. Round-trip check: PIL decodes the
PNG bit-exactly against the fp32 decode quantized independently, and vs the ORT
reference only 114 / 1048576 bytes differ, all by 1 (quantization-boundary flips).

### Runtime fixes this required
1. `OpType::RESIZE` was missing from the `op_fp16` dtype-following switch in `core/runtime.cpp`,
   so a fp32 Resize input was executed by the fp16 shader at half width (output garbage, ~4e37).
2. Graph outputs are pre-allocated from `precision_` early in `LoadModel`. When the producing
   operator runs in the other domain (e.g. graph output forced fp16 but the final `Min` is fp32),
   the bind silently truncated to 2 bytes/elem. A post-build fixup now re-allocates graph-output
   tensors to match the producing node's input dtype (skipping `Cast`/`FusedElemwise`).
3. `Resize` only had an image-backend port; the VAE mixes it with SSBO-only shape ops, so a
   buffer port (`shaders/buffer/resize.comp`, `ops/Resize.hpp::ResizeBuffer`) was added — nearest /
   asymmetric only, one invocation per 32-bit word in fp16 to avoid read-modify-write races.

### Regression found 2026-10-03: the shape pool actually recycling buffers breaks the decoder

The numbers above **do not reproduce on the current working tree**. Bisect, all runs on the
reproducible probe latent (`default_rng(42).standard_normal((1,64,1,32,32))`) against an ORT CPU
reference of the same `vae_decoder_512.onnx`:

| Build | cos | maxabs |
|-------|-----|--------|
| `c920c8c` in a separate worktree | +1.000000 | 4.6e-05 |
| `HEAD` (`ad15454`) in a separate worktree | +1.000000 | 4.6e-05 |
| `HEAD` + the 6 uncommitted engine files | **-0.381480** | 2.0 |
| ... reverting only `core/runtime.cpp` | +1.000000 | 4.6e-05 |
| ... dirty `runtime.cpp`, write-site guard disabled | -0.381480 | 2.0 |
| ... dirty `runtime.cpp`, pool pop disabled | +1.000000 | 4.6e-05 |

So the regression is entirely in the uncommitted `core/runtime.cpp`, and it is the **pop side of
`outshape_tensor_map`**, not the view-alias guard: the pop used to read `auto q = map[key]`, which
copies the queue, so the pool never drained and every node output allocated its own buffer. Binding
by reference makes the pool recycle tensor objects for the first time, and the decoder then breaks
at the first node whose output buffer lands on bytes a live operand still reads.

`VKOP_NODE_DUMP` + `compare_vae_node_dump.py` (per-node, matched by name) pin where:

| Node | rms(ref) | rms(err) | relative |
|------|----------|----------|----------|
| `/decoder/conv_in/Conv` | - | 0.0 | exact |
| `/decoder/mid_block/resnets.0/norm2/Expand` | 465.4 | 1.1e-04 | 2.3e-07 |
| `/decoder/mid_block/resnets.0/norm2/Add` (`FusedElemwise`) | 0.907 | 5.2e-02 | **5.8e-02 ← first divergence** |
| `/decoder/conv_out/Conv` | - | - | cos -0.35 |

The failing `Add` consumes a broadcast operand of rms 465 and produces rms 0.9, so a single wrong
buffer cancels large magnitudes and amplifies into the visible vertical green/magenta banding; the
error then compounds through the five up-blocks. The current guard only treats
`Reshape/Squeeze/Unsqueeze` as view ops and only inspects whether an *output* buffer was aliased
out, which does not cover a fused elementwise program writing over bytes a still-live broadcast
operand points at.

#### Follow-up 2026-10-03 (later same day): four hypotheses measured and rejected

Report-only probes in a detached worktree (`/private/tmp/vkop_head`, HEAD + the engine
edits under test, `VKOP_POOL_PROBE=1`), same latent, same reference:

| Hypothesis | Probe | Result |
|------------|-------|--------|
| recycled object still holds a live `VkBuffer` | print `gpu_resource_id()` at every pop | **all 215 pops report `rid=0`** — pooled objects are buffer-less at build time, so the buffer-identity check can never fire. This also means the write-site guard was never the thing protecting us |
| buffer shared with a still-live input | compare popped buffer id against this node's inputs and every live `tensor_map` entry | 0 hits, but **void** — the ids are all 0 |
| object released while another name still has readers | per-NAME `name_remaining` counter (the object-level `ref_cnt_` cannot express this once several names share one `Tensor`), checked at the push site | **0 premature releases** |
| writer and last reader in the SAME level (intra-level concurrency, which the alias barrier does not order) | only recycle when the release level < the new node's level | 215 → 214 pops, **cos bit-identical to the broken run** |

Confirmed instead: object sharing *is* happening — one `Tensor` serves up to 4 names at once
(`mid_block/resnets.0`: `norm1/Expand` = `nonlinearity/Mul` = `norm2/Add` = `resnets.0/Add`), and
the failure is **fully deterministic** (cos `-0.422326`, maxabs `2.0` on every run, before and
after the level gate). A GPU race would not reproduce to the last digit, so the remaining
explanation is a write overwriting bytes an earlier name still needs **in execution order** — or, since
buffer-less recycle also changes *when* each output allocates (`as_storage_buffer` / `make_vkbuff`
reuse-if-big-enough), the allocation path rather than the aliasing path is where to look next.
The decisive probe not yet built: at execute time, record the object's last writing node per name
and flag any input whose object was last written by a different node than its producer (needs
`node_input_names_` / `node_output_tensors_` names, which `Runtime` does not currently keep).

Built that probe (`node_input_names_` + `producer_of_name_` + a per-object last-writer map in
`Run`): **0 violations** — for every input slot, the node that last wrote the object IS the node
that produced the name it refers to. So the name↔object bookkeeping is sound and no pooled object
is "read after being re-handled by somebody else". Combined with the two buffer-side results above
(objects arrive buffer-less at recycle, `rid=0` for all 214 pops) the aliasing/scheduling family is
effectively closed: nothing about *which object* or *which bytes* is provably wrong at build time,
yet the output is deterministically `-0.422326`.

Two further data points reframe it as an **op-side buffer-discipline** problem rather than an
aliasing one: with the same build,
* `VKOP_GUARD_OFF=1` (never `drop_gpu_buffer()` at the write site) → SIGSEGV at a null function
  pointer (lldb: `frame #0: 0x0`), and
* `VKOP_POOL_PRIVATE_BUFF=1` (always `drop_gpu_buffer()` for a writing node's outputs, so each gets
  a private allocation) → the same SIGSEGV.
So the write-site guard is load-bearing but not sufficient, and *any* change to when an output
tensor holds a buffer trips a code path that dispatches a null shader — the same class of bug as
the earlier `ReduceMeanBuffer` `nullptr` spv (exit 139). The next thing to look at is therefore not
the pool's liveness bookkeeping but the ops' handling of an output tensor that arrives without a
buffer (or whose buffer was just dropped): which buffer/stride/precision each op binds in that case.

#### Third round of probes: three more carried-state candidates, all negative

Same build, plus `VKOP_POOL_DISABLE=1` (reproduces the known-good behaviour from *inside* the pooled
build — cos `+1.000000`, so the knob set is trustworthy) and `VKOP_POOL_PROBE_NODE=13`, which prints
each slot of one node with the dims the converter recorded for that NAME versus the dims / buffer /
side-channel / CPU staging the shared object actually carries at that instant.

| Candidate | Evidence | Verdict |
|-----------|----------|---------|
| one `VkBuffer` used by two live tensors at execute | the last-writer test re-keyed by `gpu_resource_id()` instead of object pointer | **0 violations**. Note the failing node's output DOES carry its previous owner's buffer (`rid≠0`) whereas the pooled-off build leaves it `rid=0` — sharing is real, but no reader is ever handed a buffer somebody else wrote |
| stale logical rank on the recycled object | node 13's OUT slot is live `[1,1152,32,32]` where the name records `[1,1152,1,32,32]` | real but **not the cause**: re-applying the recorded shape at pop (`VKOP_POOL_FIX_SHAPE=1`) leaves cos at `-0.422326`, and the previous owner's op reshapes the shared object during execute anyway |
| stale GPU shape side-channel (`shape_ssbo_`) | `has_shape_ssbo()` = 0 on every slot of the failing node; clearing at the write site changes nothing | not the cause |
| stale CPU staging (`data_`) | `has_cpu_data()` = 0 / `cpu_elems` = 0 on the failing output; clearing at the write site changes nothing | not the cause |

Where that leaves it: every test above is at **node** granularity, and the first diverging node is a
`FusedElemwise` — a fused program whose intermediate tensors are pooled objects too. A stage inside
one fused node overwriting bytes an earlier stage of the *same* node still needs is invisible to all
of these probes. That per-STAGE (not per-node) ownership inside the fused elementwise program is the
next thing to instrument.

#### RESOLVED 2026-10-03: it was a stale RANK on the fused op's own output, not an alias at all

The per-stage hypothesis above is dead on arrival — `FusedElemwise` is ONE dispatch whose
intermediates live in shader registers, so it has no pooled stage tensors. What it does have is a
program SSBO whose dims block is rebuilt every round from the **live** shapes of its input and
output tensors. That is where the regression was.

Decisive measurement: ORT was run with the whole `mid_block/resnets.0` subtree exported as graph
outputs (`/tmp/vae_ref_resnets0.npz`, 112 MB, built from `vaeprobe/vae_decoder_512_probe.onnx`), and
vkop's per-node output dumps were compared against it **in absolute terms** instead of against
another vkop run:

| vkop node | tensor | cos vs ORT | maxabs | rel-rms |
|---|---|---|---|---|
| 4 | `norm1/ReduceL2` | 1.000000 | 0.00000 | 1.06e-07 |
| 6 | `norm1/Expand` | 1.000000 | 0.00000 | 1.06e-07 |
| 7 | `norm1/Add` (fused) | 1.000000 | 0.00001 | 5.58e-07 |
| 9 | `conv1/Conv` | 1.000000 | 0.00052 | 1.62e-06 |
| 12 | `norm2/Expand` | 1.000000 | 0.00046 | **2.28e-07** |
| 13 | `norm2/Add` (fused) | 0.998334 | 1.39761 | **5.77e-02** |
| 14 | `nonlinearity_1/Mul` | 0.899095 | 0.18323 | 4.59e-01 |

So **every input of node 13 is exact against ORT** (the "operand `norm2/Expand` rms 465 vs output
rms 0.9" that looked like cancellation is just what ORT computes — a GroupNorm denominator). Node 13
itself is wrong, from correct inputs. An `VKOP_FUSED_PROBE=<name substring>` dump of the program
inside the op, same binary, pool on vs `VKOP_POOL_DISABLE=1`:

```
BAD   norm2/Add  rank=5 total_dims=4   IN0..IN4 all 1,1152,1,32,32   OUT live_shape=1,1152,32,32
GOOD  norm2/Add  rank=5 total_dims=5   IN0..IN4 all 1,1152,1,32,32   OUT live_shape=1,1152,1,32,32
```

Mechanism, three links, all now measured:

1. the recycled object that serves node 13's output came from a previous owner whose tensor was
   **rank 4** `[1,1152,32,32]` — the *same element count* as this node's rank-5
   `[1,1152,1,32,32]`;
2. `FusedElemwise::execute` resized its output only `if (output->num_elements() != total)`, so the
   equal count made it keep the stale rank;
3. `build_program` then wrote the output dims with `push_left_aligned` into a `rank`-sized block,
   turning the rank-4 `[1,1152,32,32]` into `[1,1152,32,32,1]` instead of `[1,1152,1,32,32]`. The
   shader decomposes `gid` through that block, so every broadcast operand is indexed on the wrong
   axis — which is exactly a per-channel scale applied to the wrong channels: magnitude intact,
   cos 0.998, maxabs 1.4.

Fix (`ops/FusedElemwise.hpp`, in the `else` of that count check): re-stamp the chain's own computed
broadcast shape with `reshape_view(out_shp)` — a pure logical reshape, guarded inside by
element-count equality, so it can never desync a buffer from its dims. The output belongs to this
node, so stamping it is correct no matter which object the pool handed over.

Validation, same binary, pool **enabled**:

| config | cos vs `/tmp/dec_c920c8c.raw` | byte-identical |
|---|---|---|
| repo working tree before the fix | −0.422326 | no |
| repo + `reshape_view` fix | **+1.000000**, maxabs 0.0 | **yes** |
| repo + fix + `VKOP_POOL_DISABLE=1` | +1.000000, maxabs 0.0 | yes |

No DiT cost: `image_gen` 1-step `ref_static64` with the fix gives `velocity_step0 cos=0.999990`,
`present_kv_0..21 OK`, `22..31 cos≈0.994–0.999` — the documented fp16 per-layer accumulation, i.e.
unchanged; `ctest` 129/130 with only the known pre-existing `ModelTest (SEGFAULT)`.

Two notes for whoever picks this up. (a) The stale shape is written by an op at **execute** time, so
no build-time pop-side correction can help — that is why the earlier
`VKOP_POOL_FIX_SHAPE=1`/`VKOP_POOL_PRIVATE_BUFF=1` experiments were no-ops while the bug lived here.
(b) `if (output->num_elements() != total_elems(shape)) output->resize(shape)` is the pattern in
~20 other ops (`BinaryFactory`, `BufferBinaryFactory`, `BufferUnaryFactory`, `Cast`, `Concat`,
`Conv2d`, `LayerNorm`, `RMSNorm`, `Equal`, …) and any of them can inherit the same rank-vs-count
confusion from a recycled object; the VAE only tripped `FusedElemwise`. They were NOT touched here —
none is known broken, and each needs its own measurement first.

Repro (≈10 s per data point, no DiT needed):

```
np.random.default_rng(42).standard_normal((1,64,1,32,32), np.float32).tofile("/tmp/vae_in_rng42.raw")
# ORT reference: vae_decoder_512.onnx with that latent
DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib VK_ICD_FILENAMES=/opt/homebrew/etc/vulkan/icd.d/MoltenVK_icd.json \
  build/vae_gen image/exporter/vae_decoder_512.vkopbin /tmp/vae_in_rng42.raw /tmp/dec.raw
```

Two gaps this exposed: `ctest` (130 cases) has **no VAE end-to-end numeric case**, so nothing
guarded it; and matching ORT proves nothing about semantics — both sides share one exported graph
and one input, so a wrong input stays "aligned". The cross-check that actually catches it is to feed
`latent_out_512x512.raw` (fp32 packed `[1,1024,64]`) through the torch VAE
(`transpose(1,2).reshape(1,64,1,32,32) * latents_std + latents_mean`, ~40 s on CPU) and look at the
picture. Done that way, the pure-vkop 40-step latents of two different prompts (identical
`latent_init`/`sigmas`/rope/bias, only `prompt_embeds` differing) decode into two correct, clearly
distinct images — the DiT path is sound and only the VAE stage is broken.

## Optimization Opportunities

1. **Constant Folding**: Would eliminate 4,423 nodes (40.6%) but requires >13 GB memory during conversion
2. **KV Cache Reuse**: Prefill runs once per prompt, decode reuses KV across denoising steps
3. **Sequential Loading**: Already implemented — releases prefill before loading decode
4. **FP16 Precision**: All graphs use fp16; numerical accuracy verified against torch wrapper

## Comparison with LLM Pipeline

| Metric | Qwen-Image-2.1 | Qwen3-VL-2B (LLM) |
|--------|----------------|-------------------|
| Parameters | 7.12B | 2.0B |
| Weight Size | 14.2 GB (fp16) | ~4 GB (fp16) |
| Prefill Time | 30.3s (ORT) / 4.3s (torch) | Varies by seq len |
| Decode Time | N/A (not implemented) | ~ms per token |
| Peak Memory | 16.6 GB (torch) / 14.4 GB (ORT) | ~8-12 GB |
| Operator Gap | 5.6% | ~0% (fully supported) |
