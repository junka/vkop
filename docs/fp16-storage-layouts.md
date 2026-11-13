# fp16 storage layouts: native half arrays vs packed half2 words

The three qwen3_5 linear-attention ops (`CumSum`, `Conv1d`, `LinearAttention`)
ship **two fp16 builds from one source**, and the runtime picks one per op:

| shader key | storage | per-element access |
|---|---|---|
| `<key>_fp16` | `float16_t data[]` | one 16-bit load/store |
| `<key>_fp16_packed` | `uint data[]` | `unpackHalf2x16` / `packHalf2x16` on a shared word |

`VKOP_FP16_PACKED=<Op,Op,...>` selects the packed arm at run time (the op
appends `_packed` to its resolved key). Both produce **identical bytes on
disk** — N halves and N/2 words are the same buffer, so no converter change and
no model re-conversion is involved.

## Why two layouts exist

The packed form is what most of this repository does: `buffer_common.comp`'s
`BUF_LOAD`/`BUF_STORE` macros read-modify-write a `uint` word holding two
halves. It works everywhere, including on a device without
`storageBuffer16BitAccess`, and it is why `DTYPE`/`DTYPE2` exist.

The native form is what `GL_EXT_shader_16bit_storage` +
`GL_EXT_shader_explicit_arithmetic_types_float16` give you. `gather.comp` and
`conv2d.comp` already use it. It costs no pack/unpack instructions, which is
the obvious expectation.

## Measured: no difference

Interleaved A/B on the local Intel ARL iGPU, replayed submits only (record once,
then submit the same kept command buffer):

```
[bench] 200 replayed submits per sample, 7 interleaved samples per arm
Conv1d             native min=  0.1260 med=  0.1359 | packed min=  0.1302 med=  0.1361 | packed/native min=1.034 med=1.002
LinearAttention    native min=  0.4009 med=  0.4678 | packed min=  0.3820 med=  0.4688 | packed/native min=0.953 med=1.002
CumSum             native min=  0.0914 med=  0.0989 | packed min=  0.0860 med=  0.1036 | packed/native min=0.941 med=1.047
```

Repeated with 300 rounds / 9 samples:

```
Conv1d             packed/native min=1.048 med=1.036
LinearAttention    packed/native min=1.015 med=1.015
CumSum             packed/native min=0.983 med=1.027
```

**The ratio is 0.98–1.05 in both runs, i.e. inside the noise.**

The shaders really are different code — `buffer_cumsum_fp16.spv` carries
`OpCapability StorageBuffer16BitAccess` and 3 `OpFConvert`; the packed build
has neither and instead carries extra `OpBitcast`/`OpShiftRightLogical` — so
this is not a case of the two builds collapsing to the same module.

## Why they cost the same

All three kernels are **memory bound, not instruction bound**:

* `Conv1d` at the 27B geometry reads/writes ~10K channels of state and taps a
  4-wide window — the arithmetic is a handful of FMAs per byte moved.
* `LinearAttention` moves the [Hv, Dk, Dv] state (48 × 128 × 128 floats = 3 MB
  per layer) every call; its inner loops are 128-long dot products over that
  state.
* `CumSum` walks a line serially with one accumulated float.

On an iGPU the memory subsystem is the bottleneck and the few dozen
pack/unpack ALU ops hide entirely behind it. A discrete GPU with more compute
headroom per byte, or a compute-bound variant of these kernels, is where the
native form would start to matter.

## So which one is the default?

Native (`<key>_fp16`), because:

* it is the simpler indexing — one element per array slot, so the thread
  mapping is the plain one;
* it removes the packed layout's extra correctness hazard. In the packed form
  two neighbouring elements share a word, which forces the thread-per-word
  discipline: `CumSum` had to pair adjacent *lines* (giving each line its own
  thread silently lost updates — the correctness probe caught element 5 holding
  raw input instead of a prefix sum), and `Conv1d`/`LinearAttention` compute
  two or pack from LDS for the same reason;
* it costs nothing measurable here.

The packed arm stays in the tree as the A/B control, and as the fallback for a
device without `storageBuffer16BitAccess`.

## Reproducing

```
./build/benchmark/fp16_layout_bench 200 7    # <submits per sample> <samples>
```

Correctness for both arms:

```
./build/tests/LinearAttnOpsProbe                                   # native
VKOP_FP16_PACKED=CumSum,Conv1d,LinearAttention \
  ./build/tests/LinearAttnOpsProbe                                 # packed
```

The probe compares against a CPU reference at both the small shape and the
Qwen3.5-27B geometry (Hk=16, Hv=48, Dk=Dv=128), and checks the in-place state
contract (`present` aliasing `past`).
