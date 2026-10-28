"""Weights that never enter RAM: conversion streams them from their file.

Re-converting the 13.9 GB text encoder cost 33 GiB of RSS and took the machine
down with a Jetsam kill. The arithmetic is two full copies: hydration pulls
every weight into a TensorProto's `raw_data`, and protobuf returns a *copy* per
`raw_data` read -- so `raw_map = {init.name: init.raw_data}` in
`ONNXOptimizer.optimize_model` is already a second complete set before the
first pass runs, and the pass rounds stack more.

Pass-through weights do not need either copy. A weight that

  * lives in an external data file,
  * is already the element type the blob stores (so no quantizer touches it),
  * and is consumed only by MatMul/Gemm/Conv,

cannot be looked at by the converter: those ops have an activation input, so
`ConstantFolder` will not fold them, and no fusion pass reads an operand's bytes
to decide anything. Such a tensor keeps its `external_data` reference and is
copied from its source file straight into the .vkopbin trailer, at the offset
the side table records.

Anything that is not pass-through is hydrated one tensor at a time before the
passes, so a model whose weights the pipeline does inspect behaves exactly as
before -- this only ever declines to hold bytes nobody reads. Deferral is
refused outright (not half-applied) when `-q` is a real quantization, or `-u`/
`-r` is set: those rewrite weight payloads and must have them in memory.

Fail-closed points:
  * `guarded_to_array` raises for a deferred tensor instead of letting numpy
    turn an empty payload into "cannot reshape array of size 0" somewhere
    downstream;
  * `verify_deferred` runs before writing: a deferred tensor still read by a
    live node but missing from the side table, or whose dtype/dims/length
    changed, aborts the conversion instead of writing a hole;
  * the writer checks that the source file really holds `length` bytes.
"""
from __future__ import annotations

import os
from collections import Counter
from typing import Dict, List, Optional, Tuple

import onnx
from onnx import TensorProto, numpy_helper
import numpy as np
from onnx.external_data_helper import uses_external_data

# Chunk size for the file-to-file copy.
_CHUNK_BYTES = 1 << 23

# Element types whose bytes are already what the blob stores under -q fp16.
# bfloat16 IS pass-through: the runtime has no bf16 storage, so a bf16
# initializer's bytes cannot ride as-is -- but write_payload() converts them
# chunk-wise to fp16 on the way into the trailer (MoE expert pool: 23.2 GB of
# per-expert checkpoint tensors; hydrating them all to convert would be the
# OOM this module exists to prevent). prepare() flips the recorded dtype to
# FLOAT16 at deferral time so _downcast_bfloat16 leaves the file bytes alone.
_PASS_THROUGH_DATA_TYPES = frozenset(
    {TensorProto.FLOAT16, TensorProto.BFLOAT16})
_PASS_THROUGH_ITEMSIZE = {TensorProto.FLOAT16: 2, TensorProto.BFLOAT16: 2}

# Ops that take a weight as an operand and are never evaluated during
# conversion, because one of their inputs is an activation.
_WEIGHT_ONLY_CONSUMERS = frozenset({"MatMul", "Gemm", "Conv"})

_ENV_ENABLE = "VKOP_STREAM_WEIGHTS"
_ENV_MIN_MIB = "VKOP_STREAM_MIN_MIB"


class DeferredWeightError(RuntimeError):
    """A deferred weight was asked for by value, or vanished from the graph."""


def _env_flag(name: str, default: bool) -> bool:
    v = os.environ.get(name, "").strip().lower()
    if not v:
        return default
    return v not in ("0", "off", "no", "false")


def stream_min_bytes() -> int:
    """Weights at or above this many bytes stay in their file (default 1 MiB).

    The same cap the value-free view already demotes at, so the two agree.
    """
    v = os.environ.get(_ENV_MIN_MIB, "").strip()
    if v.isdigit() and int(v) >= 0:
        return int(v) << 20
    return 1 << 20


def deferral_refusal(quant: Optional[str], unify: bool,
                     rgba: bool) -> Optional[str]:
    """Why deferral must not run, or None when it may."""
    if not _env_flag(_ENV_ENABLE, True):
        return f"{_ENV_ENABLE}=0"
    if quant not in (None, "fp16"):
        return f"-q {quant} rewrites weight payloads and needs their bytes"
    if unify:
        return "--unify concatenates weight payloads and needs their bytes"
    if rgba:
        return "--rgba rewrites weight payloads and needs their bytes"
    return None


def payload_bytes(tensor) -> int:
    """Bytes this initializer occupies, from dtype and dims only."""
    n = 1
    for d in tensor.dims:
        n *= int(d)
    size = _PASS_THROUGH_ITEMSIZE.get(tensor.data_type)
    if size is None:
        return -1
    return n * size


def is_deferred(tensor) -> bool:
    """An initializer whose payload we deliberately left in its file."""
    return (tensor.data_location == TensorProto.EXTERNAL
            and not tensor.raw_data
            and uses_external_data(tensor))


def _external_fields(tensor) -> Tuple[str, int, int]:
    """(location, offset, length), with a missing length filled in from dims.

    A `length` of 0 is *not* "the rest of the file" here. onnx's own writer
    records a length of 0 for any tensor whose `raw_data` is an empty field that
    is nonetheless *present* -- `load_external_data_for_tensor` leaves the field
    set after reading a zero-length span -- so a writer that re-saves a
    round-tripped model with `onnx.save` turns every true length into 0. That
    happens to `llm_moe.linweights.bin`'s 290 small fp32 tensors (qkv biases,
    norms). Handing such a tensor to `load_external_data_for_tensor` makes it
    read from `offset` to EOF of a 3 GB file: measured 2.4 GiB of transient RSS
    for ONE 4096-byte bias, i.e. the whole model's conversion OOMs. The payload
    size is fully determined by dtype and dims, so that is what we read.
    """
    meta = {e.key: e.value for e in tensor.external_data}
    location = meta.get("location")
    if not location:
        raise DeferredWeightError(
            f"initializer '{tensor.name}' declares EXTERNAL storage without a "
            f"'location' entry -- its payload is nowhere to read from")
    offset, length = int(meta.get("offset", 0)), int(meta.get("length", 0))
    if length <= 0:
        length = payload_bytes(tensor)
        if length < 0:
            raise DeferredWeightError(
                f"initializer '{tensor.name}' has no 'length' entry and its "
                f"dtype {tensor.data_type} has no fixed element size, so its "
                f"payload extent cannot be derived")
    return location, offset, length


def _hydrate(tensor, base_dir: str) -> None:
    """Read one external tensor's bytes, bounded by its own payload extent.

    Not `load_external_data_for_tensor`: that trusts the external_data
    `length`, and a length of 0 there means "read to EOF" (see
    `_external_fields` for how a model acquires such an entry and what it costs
    to honor it). The extent is derived instead, and the read is bounded, so a
    model whose side file is 3 GB cannot turn a 4 KB bias into a 3 GB read.
    """
    location, offset, length = _external_fields(tensor)
    path = location
    if not os.path.isabs(path):
        path = os.path.join(base_dir, path)
    with open(path, "rb") as f:
        f.seek(offset)
        data = f.read(length)
    if len(data) != length:
        raise DeferredWeightError(
            f"initializer '{tensor.name}' wants {length} byte(s) at offset "
            f"{offset} of {path}, which holds only {len(data)}")
    tensor.raw_data = data
    tensor.data_location = TensorProto.DEFAULT
    del tensor.external_data[:]


def _all_tensors(model):
    """Every TensorProto `load_external_data_for_model` would have loaded.

    Delegated to onnx's own scan so the coverage cannot drift from the
    behaviour this replaces -- initializers, `input.default_value`, sparse
    initializers and node attribute tensors, including inside local functions.
    `prepare` re-checks against this set that nothing was missed rather than
    trusting either walk.
    """
    try:
        from onnx.external_data_helper import _get_all_tensors
        return list(_get_all_tensors(model))
    except ImportError:
        pass

    def walk(graph):
        for init in graph.initializer:
            yield init
        for node in graph.node:
            for attr in node.attribute:
                if attr.type == onnx.AttributeProto.TENSOR:
                    yield attr.t
                elif attr.type == onnx.AttributeProto.TENSORS:
                    for t in attr.tensors:
                        yield t
                elif attr.type == onnx.AttributeProto.GRAPH:
                    yield from walk(attr.g)
                elif attr.type == onnx.AttributeProto.GRAPHS:
                    for g in attr.graphs:
                        yield from walk(g)
    tensors = list(walk(model.graph))
    for function in getattr(model, "functions", []):
        tensors.extend(walk(function))
    return tensors


def _stream_entry(tensor, base_dir, location, offset, length,
                  consumers: int) -> dict:
    """The deferred-table row for one tensor whose bytes stay in its file."""
    entry = {
        "name": tensor.name,
        "location": location,
        "base_dir": base_dir,
        "offset": offset,
        "length": length,
        "data_type": tensor.data_type,
        "dims": tuple(int(d) for d in tensor.dims),
        "consumers": consumers,
    }
    if tensor.data_type == TensorProto.BFLOAT16:
        # Bytes stay in the source file; the recorded dtype flips to fp16 NOW so
        # _downcast_bfloat16 (which runs after prepare) does not "convert" the
        # empty raw_data of an external tensor and leave bf16 bytes labeled fp16
        # in the trailer. The real conversion happens chunk-wise in
        # write_payload.
        entry["src_dtype"] = TensorProto.BFLOAT16
        entry["data_type"] = TensorProto.FLOAT16
        tensor.data_type = TensorProto.FLOAT16
    return entry


def prepare(model, base_dir: str, *, quant: Optional[str] = None,
            unify: bool = False, rgba: bool = False,
            keep_in_file=(), verbose: bool = True) -> Dict[str, dict]:
    """Hydrate what the pipeline may read; return the deferred-weight table.

    Called right after `onnx.load(..., load_external_data=False)`: the whole
    model is value-free at that point, so every tensor this function decides to
    keep in memory is read from the external file individually and the rest
    never is.

    `keep_in_file` names tensors whose bytes must stay on disk even when the
    gates below would hydrate them: the MoE paging slots. It has to be an input
    rather than a post-hoc correction, because hydration is irreversible at this
    scale -- a 4320-slot expert pool is 23 GiB, and reading it once is the OOM
    this whole path exists to prevent. Slots fail the dtype gate for a reason
    that says nothing about whether their bytes may be streamed (a producer may
    leave the pool float32; the runtime derives each expert's stride from dtype
    and dims, so the extent is checked instead), and they fail the consumer gate
    by design (an unwired slot is read by nobody). See `restore_paging_slots`
    for the contract each named slot is then held to.
    """
    declared = {name for name in keep_in_file}
    graph = model.graph
    consumers: Dict[str, Counter] = {}
    for node in graph.node:
        for name in node.input:
            consumers.setdefault(name, Counter())[node.op_type] += 1

    refusal = deferral_refusal(quant, unify, rgba)
    cap = stream_min_bytes() if refusal is None else 0

    deferred: Dict[str, dict] = {}
    # The writer finds a payload by name, so a name repeated in the graph cannot
    # be tracked by `deferred` alone -- those tensors are hydrated instead of
    # deferred, unless they are declared slots (see below).
    seen_names = Counter(init.name for init in graph.initializer)
    by_name = {}
    for init in graph.initializer:
        if not uses_external_data(init):
            continue  # inline already: the bytes are in the proto
        by_name.setdefault(init.name, init)
        location, offset, length = _external_fields(init)
        keep_in_file_gates = (refusal is None
                              and seen_names[init.name] == 1
                              and length >= cap)
        if keep_in_file_gates and init.name not in declared:
            keep_in_file_gates = (
                init.data_type in _PASS_THROUGH_DATA_TYPES
                and length == payload_bytes(init)
                and _only_weight_consumers(consumers.get(init.name)))
        if keep_in_file_gates:
            deferred[init.name] = _stream_entry(
                init, base_dir, location, offset, length,
                sum(consumers.get(init.name, Counter()).values()))

    # A declared slot the gates above turned down anyway -- a name the graph
    # repeats is the one case left -- must still not be hydrated: its bytes are
    # the expert pool. It is streamed as a slot instead, and
    # `restore_paging_slots` holds it to the slot contract.
    for name in sorted(declared - set(deferred)):
        init = by_name.get(name)
        if init is None or refusal is not None:
            continue
        location, offset, length = _external_fields(init)
        deferred[name] = _stream_entry(init, base_dir, location, offset, length, 1)

    # Everything else that points at an external file gets its bytes, with the
    # same state transition onnx's own whole-model loader applies, so no
    # downstream code can tell the two paths apart.
    streams = set(deferred) | declared
    hydrated = 0
    for tensor in _all_tensors(model):
        if tensor.name in streams or not uses_external_data(tensor):
            continue
        _hydrate(tensor, base_dir)
        hydrated += 1

    # Fail closed on anything this walk did not reach: an external reference
    # nobody loaded becomes a silently missing weight, and the count is the
    # name-independent check that only the planned tensors stayed in the file.
    still_external = [t.name for t in _all_tensors(model) if uses_external_data(t)]
    streamed_external = [n for n in still_external if n in deferred]
    unexpected = [n for n in still_external if n not in deferred]
    seen_declared = sum(1 for n in declared if n in deferred)
    if unexpected or len(streamed_external) + len(declared) - seen_declared != len(streams):
        raise DeferredWeightError(
            f"{len(still_external)} tensor(s) still point at external data "
            f"after hydration but only {len(deferred)} were left there "
            f"(unexpected: {unexpected[:5]}); _all_tensors does not reach where "
            f"they live, so streaming would lose them, or reaches a second copy "
            f"of a name that streaming can only carry once")

    if verbose:
        if refusal is not None:
            print(f"[stream] weights not streamed ({refusal}); "
                  f"{len(graph.initializer)} initializer(s) fully in memory")
        else:
            kept_bytes = sum(e["length"] for e in deferred.values())
            print(f"[stream] streaming {len(deferred)} initializer(s) "
                  f"({kept_bytes >> 20} MiB) straight from "
                  f"{sorted({e['location'] for e in deferred.values()})}; "
                  f"{hydrated} read into memory "
                  f"[cap {cap >> 20} MiB, pass-through dtypes "
                  f"{sorted(_PASS_THROUGH_DATA_TYPES)}, consumers "
                  f"{sorted(_WEIGHT_ONLY_CONSUMERS)}]")
    return deferred


def _only_weight_consumers(ops: Optional[Counter]) -> bool:
    """True when every node reading this tensor takes it as a weight.

    A tensor nobody consumes is a dead initializer; the writer's liveness prune
    drops it, so holding it in a file is as good as holding it in RAM -- but it
    also gives nothing back, so it is not deferred. See prepare's `keep_in_file`
    for the one case that overrides this.
    """
    if not ops:
        return False
    return set(ops) <= _WEIGHT_ONLY_CONSUMERS


def restore_paging_slots(model, base_dir: str, deferred: Dict[str, dict],
                         slot_names) -> Tuple[int, int]:
    """Re-defer, after the fact, the external tensors of declared paging slots.

    `prepare` is told the slot names up front and skips them, so in the normal
    case this function only has to validate what `prepare` recorded and clear
    the `external_data` pointer that would otherwise make a later walk hydrate
    the pool. It still builds an entry when one is missing, because `prepare`
    can decline a name it was told to keep for reasons of its own -- a tensor
    whose name appears twice in the graph cannot be tracked by name -- and a
    declared slot must never be hydrated regardless.

    The entries are built from the same helpers `prepare` uses. Every field is
    by construction what `prepare` would have recorded; what is *checked* is
    the contract rather than the convenience: the tensor is an initializer of
    this graph, still external (a slot whose bytes were already read cannot be
    re-deferred without losing the mutations the passes applied to the resident
    copy), and has a dtype with a fixed element size whose derived payload
    matches the declared extent -- the runtime computes an expert's stride from
    dtype and dims, so a mismatch here would page the wrong bytes.

    Every slot ends up carrying *no* external_data. That is not cosmetic: a
    declared slot must never be hydrated again by a later walk
    (`_downcast_bfloat16`, the optimizer's value-view, the writer's own scans),
    and hydration is triggered by `uses_external_data`, so clearing the pointer
    is what makes "the bytes stay on disk" a property of the model rather than
    of which walks happen to run. The deferred table is now the only place the
    payload's location lives.

    Returns (entries added, pointers cleared).
    """
    by_name = {t.name: t for t in _all_tensors(model)}
    still_external = {t.name for t in _all_tensors(model)
                      if uses_external_data(t)}
    added = 0
    freed = 0
    for name in slot_names:
        tensor = by_name.get(name)
        if tensor is None:
            raise DeferredWeightError(
                f"MoE paging slot '{name}' is not a tensor of this graph; the "
                f"exporter's slot declarations name something the model does "
                f"not have")
        entry = deferred.get(name)
        if entry is not None:
            # prepare() already kept the bytes in the file, but the *views* of
            # them are a different question: hydration is driven by residency
            # (`uses_external_data`), and a tensor whose bytes were read into
            # raw_data by some other walk -- or left external by a producer that
            # never converted -- is not what the writer streams. Say so rather
            # than letting the trailer be assembled from a stale entry.
            if tensor.raw_data:
                raise DeferredWeightError(
                    f"MoE paging slot '{name}' is deferred but already has "
                    f"{len(tensor.raw_data)} byte(s) of raw_data; the bytes are "
                    f"resident and the passes' view of them is what the graph "
                    f"was built from, so streaming the file's copy would "
                    f"convert a different set of bytes than the model runs")
            # An external slot carries the file's extent in `external_data`,
            # which is 20 bytes per tensor of proto. Dropping it once the
            # payload lives in the deferred table is what makes a 4320-slot
            # model convertible at all; the writer finds the payload by name.
            if uses_external_data(tensor):
                del tensor.external_data[:]
                tensor.data_location = TensorProto.DEFAULT
                freed += 1
            continue
        if name not in still_external:
            raise DeferredWeightError(
                f"MoE paging slot '{name}' is not an external tensor, so its "
                f"bytes are in the protobuf and cannot be streamed; the "
                f"exporter must write slot payloads to an external file")
        location, offset, length = _external_fields(tensor)
        item = _PASS_THROUGH_ITEMSIZE.get(tensor.data_type)
        if item is None:
            raise DeferredWeightError(
                f"MoE paging slot '{name}' has dtype {tensor.data_type}, which "
                f"has no fixed element size; the streaming path carries slot "
                f"payloads byte-for-byte and only fp16/bf16 qualify")
        want = payload_bytes(tensor)
        if length != want:
            raise DeferredWeightError(
                f"MoE paging slot '{name}' declares {length} byte(s) at offset "
                f"{offset} but its dtype and dims describe {want}; the runtime "
                f"derives each expert's stride from dtype and dims, so the "
                f"offset table would point past or short of the slot")
        if tensor.data_type == TensorProto.BFLOAT16:
            # Same state transition prepare() applies: the file keeps bf16 bytes
            # and write_payload converts chunk-wise, so the recorded dtype has
            # to flip now or _downcast_bfloat16 sees an empty raw_data.
            entry = {"name": name, "location": location, "base_dir": base_dir,
                     "offset": offset, "length": length,
                     "data_type": TensorProto.FLOAT16,
                     "src_dtype": TensorProto.BFLOAT16,
                     "dims": tuple(int(d) for d in tensor.dims), "consumers": 1}
            tensor.data_type = TensorProto.FLOAT16
        else:
            entry = {"name": name, "location": location, "base_dir": base_dir,
                     "offset": offset, "length": length,
                     "data_type": tensor.data_type,
                     "dims": tuple(int(d) for d in tensor.dims), "consumers": 1}
        deferred[name] = entry
        added += 1
    return added, freed


def write_payload(entry: dict, out_f, base_dir: str) -> int:
    """Copy one deferred tensor from its source file into the output.

    Returns the bytes written. A short read is a hard error: the side table
    already promised `length` bytes at that offset, and a trailer that is short
    would make the runtime read the neighbouring tensor. A bf16 source is
    converted chunk-wise to fp16 (the runtime's only 16-bit float storage);
    bf16 -> fp16 is the fp32 value rounded once, the same recipe as
    converter._bf16_bytes_to_fp16, so both paths agree bit-for-bit.
    """
    path = entry["location"]
    if not os.path.isabs(path):
        path = os.path.join(base_dir, path)
    remaining = entry["length"]
    to_fp16 = entry.get("src_dtype") == TensorProto.BFLOAT16
    with open(path, "rb") as src:
        src.seek(entry["offset"])
        while remaining:
            block = src.read(min(_CHUNK_BYTES, remaining))
            if not block:
                raise DeferredWeightError(
                    f"deferred weight '{entry['name']}' asks for {entry['length']} "
                    f"bytes at offset {entry['offset']} of {path}, but the file "
                    f"runs out {remaining} byte(s) early")
            if to_fp16:
                u16 = np.frombuffer(block, dtype=np.uint16)
                f32 = (u16.astype(np.uint32) << 16).view(np.float32)
                with np.errstate(over="ignore"):
                    # Past-fp16-range magnitudes saturate to inf; that is the
                    # defined outcome, matching the converter's own upfront
                    # bf16 normalization.
                    block = np.ascontiguousarray(
                        f32.astype(np.float16)).tobytes()
            out_f.write(block)
            remaining -= len(block)
    return entry["length"] - remaining


def verify_not_sidebar_located(slots, trailer_path: str) -> List[str]:
    """A slot's bytes must be in the output file, not left in a side file.

    Left alone, a weight whose external `location` is not the trailer file is
    still written to the right offset by write_payload -- but its bytes live in
    the checkpoint. The runtime only knows the offset, so a paged model would
    load, look correct, and pop every non-resident expert back to whatever the
    (unrelated) trailer file holds at that offset: silent zeros.

    The check is on *where the bytes are read from at write time*. A slot whose
    `location` is the trailer is already in the file the loader pages, so its
    copy is trivially the file's own bytes. A slot read from anywhere else --
    the exporter records each of the 4320 slots' `location` as the safetensors
    shard it streams from -- still lands in the trailer, because the writer
    copies it through `write_payload`; what is unverified is that the
    transcription matches the expert the slot names. A warning says the second
    kind exists; the caller decides whether that is acceptable, since only it
    knows if the checkpoint is still on disk to page from in the first place.
    """
    if not slots:
        return []
    trailer = os.path.realpath(trailer_path)
    return [n for n, e in slots.items()
            if os.path.realpath(os.path.join(e.get("base_dir", ""),
                                             e["location"])) != trailer]


def verify_deferred(deferred: Dict[str, dict], dag_model) -> None:
    """Every deferred tensor must still be the tensor we promised to stream.

    A deferred weight that a pass *read* instead of merely rewiring shows up
    here as a changed dtype/dims/length or a lost external reference; one that
    vanished while a live node still names it would otherwise be written as no
    bytes at all. Both abort. A weight no live node reads anymore is a
    legitimate prune and only gets reported.
    """
    if not deferred:
        return
    refs = Counter()
    for node in dag_model.nodes.values():
        for name in node.inputs:
            refs[name if isinstance(name, str) else name.get("name")] += 1

    problems: List[str] = []
    pruned = 0
    for name, e in deferred.items():
        init = dag_model.initializers.get(name)
        if init is None:
            if refs.get(name):
                problems.append(
                    f"'{name}' is read by {refs[name]} live input(s) but is no "
                    f"longer an initializer")
            else:
                pruned += 1
            continue
        # Two legal shapes of "still deferred": the tensor parses as deferred
        # outright, or its external_data pointer was cleared on purpose by
        # `restore_paging_slots`, which is what keeps later walks from
        # hydrating it. A pass that ate the payload instead leaves the same
        # visible state (no pointer, no bytes), so that case is only caught by
        # the extent checks below -- which the cleared slots still pass.
        if not is_deferred(init):
            cleared = (init.data_location == TensorProto.DEFAULT
                       and not init.raw_data
                       and not init.external_data)
            if not cleared:
                problems.append(
                    f"'{name}' lost its external-data reference (raw_data is "
                    f"{len(init.raw_data)} byte(s)); nothing was going to "
                    f"stream it")
                continue
        got = payload_bytes(init)
        if (init.data_type != e["data_type"] or tuple(int(d) for d in init.dims) != e["dims"]
                or got != e["length"]):
            problems.append(
                f"'{name}' changed while deferred: dtype {e['data_type']}->{init.data_type}, "
                f"dims {list(e['dims'])}->{list(init.dims)}, "
                f"bytes {e['length']}->{got if got >= 0 else 'n/a'}")
    if problems:
        raise DeferredWeightError(
            f"{len(problems)} deferred weight(s) are no longer streamable "
            f"(a pass consumed their bytes instead of their name; rerun with "
            f"{_ENV_ENABLE}=0 to keep them in memory):\n  " + "\n  ".join(problems))
    if pruned:
        print(f"[stream] {pruned} deferred weight(s) pruned as unreferenced")


_ORIGINAL_TO_ARRAY = numpy_helper.to_array


def guarded_to_array(tensor, *args, **kwargs):
    """numpy_helper.to_array that refuses to pretend a deferred payload exists."""
    if isinstance(tensor, TensorProto) and is_deferred(tensor):
        raise DeferredWeightError(
            f"initializer '{tensor.name}' was left in its external file "
            f"({tensor.external_data[0].value if tensor.external_data else '?'}"
            f", {payload_bytes(tensor) >> 20} MiB) and has no bytes to decode. "
            f"Either this tensor should not have been deferred, or the caller "
            f"needs {_ENV_ENABLE}=0.")
    return _ORIGINAL_TO_ARRAY(tensor, *args, **kwargs)


_GUARD_ATTR = "_vkop_deferral_guard"


def install_guard() -> None:
    """Route every numpy_helper.to_array in the pipeline through the refusal.

    Patched on the module attribute (not the names already imported), because
    the ~30 call sites in converter/optimizer/dag all look it up at call time;
    rewriting each one would leave the next new site unguarded.
    """
    if getattr(numpy_helper.to_array, _GUARD_ATTR, False):
        return
    numpy_helper.to_array = guarded_to_array
    setattr(guarded_to_array, _GUARD_ATTR, True)
