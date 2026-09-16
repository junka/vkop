# Phase 2-4: GPU-driven shape-meta (eliminate 134 readbacks/round)

Branch: fix/llm-buffer-op-correctness
Status: Phase 1 infra ALREADY EXISTS (dispatch_indirect, submit_indirect).

## Current state (verified 2026-09-16)

Steady-state readbacks (rbprof, 134/round, ~677ms = the floor):
- Expand 59, Reshape 56, Slice 8, Range 6, ScatterElements 3, Cast 1, NonZero 1

Each readback reads an int64 shape-value SSBO (GPU) to resolve output dims on
CPU, so output->resize(dim) + downstream getShape() work. The GPU data copy
doesn't need it. Gather/Concat/Shape are CLEAN (no readback in their GPU paths).

Shape values ORIGINATE on CPU (Shape op: inputs[0]->getShape() → fillToCPU →
copyToGPUDeferred). They get transformed by GPU shaders (gather_int64.comp,
concat int64). Then Reshape/Expand read them BACK to resolve dims.

## Core design: shape_ssbo_ side-channel

Add to ITensor (core/Tensor.hpp, protected, line ~263):
```cpp
// Authoritative producing-shape for GPU-driven consumers. An int64 SSBO
// (rank dims) carrying this tensor's REAL shape, populated by the op that
// produced the tensor. When non-null, downstream GPU shaders read broadcast/
// dispatch dims from here instead of push_constant dims (which may be a
// placeholder during migration). Cleared on recycle.
std::shared_ptr<VulkanBuffer> shape_ssbo_;
int shape_ssbo_rank_ = 0;
```
Accessors: `has_shape_ssbo()`, `get_shape_ssbo()`, `set_shape_ssbo(buf, rank)`.

Migration invariant: carry BOTH dims_ (best-effort, may be placeholder with
correct element-count) AND shape_ssbo_ (authoritative). Downstream ops check
shape_ssbo_ first; if absent, fall back to getShape() (compat with unconverted
ops). This avoids the atomic-conversion risk in the original plan.

## Phase 2: shape-value producers populate shape_ssbo_ (NO readback change yet)

Goal: every tensor in the dynamic shape-meta chain carries shape_ssbo_, so
Phase 3 consumers can read it. This phase is NON-BREAKING (readbacks still
happen; we just ALSO populate the side-channel).

1. **Shape.hpp** (already CPU-knows shape): after copyToGPUDeferred, also set
   output->set_shape_ssbo(its own SSBO, rank). The Shape output IS the shape
   value SSBO. Trivial.

2. **Gather.hpp gpuGatherInt64**: the output's shape IS the gathered shape
   value. After gpuGatherInt64, alias the output's own int64 SSBO as its
   shape_ssbo_ (the gathered data IS shape values). set_shape_ssbo(output SSBO,
   out_shape.size()). No readback.

3. **Concat.hpp int64 path**: output is concatenated shape values. Alias
   output SSBO as shape_ssbo_. No readback.

4. **Reshape.hpp int64 path** (line 327-338, already aliases data): the output
   IS the reshaped shape-value tensor. Alias output SSBO as shape_ssbo_ with
   rank=dim.size(). BUT dim still needs readback (line 272) for now — keep it.
   The shape_ssbo_ carries the reshaped int64 values; rank = dim.size() (which
   we DO know from the shape input's element count, CPU-known via getShape of
   the shape input... wait, the shape input's ELEMENT COUNT is its rank, which
   is getShape()[0], CPU-known). So shape_ssbo_rank_ is CPU-known even without
   reading the VALUES.

5. **Expand.hpp int64 path**: output is broadcast shape values. Alias output
   SSBO as shape_ssbo_, rank=out_shape.size() (CPU-known from broadcast math
   that uses getShape, not values).

VERIFY Phase 2: 6/6 MATCH (no behavior change, just side-channel populated).
Add a VKOP_SHAPE_SSBO_DBG to count populated side-channels.

## Phase 3: downstream consumers read shape from SSBO + indirect dispatch

1. **buffer_common.comp**: add `load_dims_from_ssbo` — reads IArr8 from a bound
   int64 SSBO (as ivec2, low 32 bits) instead of PC. `broadcast_index` takes
   dims loaded from SSBO when uPC.broadcast==2.

2. **Binary shaders** (mul/add/equal/div/pow): add optional shape-SSBO bindings
   (out/in0/in1 shapes at bindings 3,4,5). When broadcast==2, load dims from
   SSBO. Dispatch via indirect (total from a shape->dispatch pre-pass).

3. **dispatch_from_shape.comp** (new tiny shader): given a shape SSBO + PC
   {rank, thread_block, mode (total/n/m/etc)}, writes
   VkDispatchIndirectCommand{w,h,z} into an indirect buffer. Used by Binary
   (mode=total, w=UP_DIV(total,256)) and MatMul (mode=matmul, separate).

4. **BufferBinaryFactory.hpp**: when inputs carry shape_ssbo_, bind them, set
   broadcast=2, dispatch indirect. Else fall back to PC dims (compat).

5. **Matmul.hpp**: M/N/K/batch from shape SSBOs; indirect dispatch.

VERIFY Phase 3: 6/6 MATCH + a broadcast test case in BufferRankTest.

## Phase 4: retire readbacks

Once downstream consumes GPU shapes:
- Reshape int64 path: drop the line-272 copyToCPU (output->resize uses a
  placeholder element-count shape; downstream reads shape_ssbo_ for real dims).
- Expand int64 path: drop read_target_shape readback.
- Slice/Range/ScatterElements: shape-value outputs flow GPU->GPU; their own
  dispatch moves to indirect or is eliminated where total is CPU-known.
- Remove option-C LearnState caches (no longer needed).

VERIFY Phase 4: VKOP_OPPROF Reshape~0ms, rbprof TOTAL~0, decode ~250ms. 6/6.

## Execution order (this session)

Do Phase 2 first (non-breaking, low risk, verifiable). Then assess Phase 3
scope before committing to it. Commit after each verified phase.

### DONE — Phase 2 (committed cc0ec96):
1. Tensor.hpp: add shape_ssbo_ member + accessors (ITensor base). ✅
2. Shape.hpp: set_shape_ssbo on output. ✅
3. Gather.hpp gpuGatherInt64: set_shape_ssbo on output (conservative: only if
   data input has shape_ssbo_). ✅
4. Concat.hpp int64: set_shape_ssbo on output. ✅
5. Reshape.hpp int64 GPU-alias path: set_shape_ssbo on output. ✅
6. Expand.hpp int64 path: set_shape_ssbo on output. ✅
7. Build + 6/6 MATCH + rbprof unchanged (134 steady). ✅
8. VKOP_SHAPE_SSBO_DBG confirms 834 tensors carry side-channel. ✅
9. Committed. ✅

### DONE — Phase 3 shader infra (committed ec58353):
- shaders/buffer/dispatch_from_shape.comp (mode 0 BINARY). ✅
- Symbol buffer_dispatch_from_shape_spv registered. ✅
- Not yet wired to any op.

### NEXT — Phase 3+4 atomic conversion (NOT YET STARTED):
KEY INSIGHT discovered: in Phase 3, getShape() is STILL CPU-correct (Reshape/
Expand still readback). So the SSBO path yields NOTHING until Phase 4 retires
the readbacks. Phase 3 (Binary reads SSBO) + Phase 4 (retire readback) MUST
land atomically — otherwise either half is a no-op or breaks correctness.

BLOCKERS to atomic retire:
1. Reshape int64 readback (line 272) resolves dim VALUES including -1/0
   wildcards (lines 294-310: dim[i]==0 copies from input, dim[i]==-1 infers).
   Skipping readback requires a GPU shader to resolve wildcards. Must check
   whether the 56 dynamic LLM Reshapes actually use -1/0 (if not, simpler).
2. Binary getShape() (BufferBinaryFactory.hpp:163-164) returns placeholder
   {total} once Reshape retires → broadcast breaks. Must convert Binary to
   read shape_ssbo_ (buffer_common load_dims + shader bindings + indirect).
3. Must convert ALL binary shaders (add/sub/mul/div/pow/equal) + MatMul, not
   just one, or the chain half-converts and corrupts.

PROPOSED incremental path (future session):
- Step A: instrument which of the 56 dynamic Reshapes use -1/0 wildcards.
  If none → retire is simpler (no GPU wildcard resolution needed).
- Step B: convert ONE binary shader (mul) to broadcast==2 SSBO path, behind
  VKOP_GPU_SHAPE flag, validate output matches PC path on a broadcast test.
- Step C: gate Reshape/Expand readback-skip on VKOP_GPU_SHAPE + downstream
  consumer converted. Atomic per-chain rollout.
- Step D: extend to all binary shaders + MatMul.
- Step E: remove flag, remove option-C cache.

## Risk notes
- Phase 2 is additive (side-channel only, no behavior change) → lowest risk. ✅
- Phase 3 shader infra is additive (unused) → lowest risk. ✅
- The real risk is the atomic Phase 3+4 retire: output->resize without CPU
  dim → downstream getShape() returns placeholder. Must convert the full
  dynamic chain atomically, or carry both dims_ + shape_ssbo_ during migration
  (the Phase 2 invariant already does this).
- Indirect dispatch adds a pre-pass per dynamic op — extra GPU work but no
  CPU sync. Net win if it replaces a ~0.43ms readback-forced submit.
