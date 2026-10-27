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
from onnx.external_data_helper import (
    load_external_data_for_tensor,
    uses_external_data,
)

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
    meta = {e.key: e.value for e in tensor.external_data}
    location = meta.get("location")
    if not location:
        raise DeferredWeightError(
            f"initializer '{tensor.name}' declares EXTERNAL storage without a "
            f"'location' entry -- its payload is nowhere to read from")
    return location, int(meta.get("offset", 0)), int(meta.get("length", 0))


def _hydrate(tensor, base_dir: str) -> None:
    """Read one external tensor's bytes, exactly as onnx's own loader does."""
    load_external_data_for_tensor(tensor, base_dir)
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


def prepare(model, base_dir: str, *, quant: Optional[str] = None,
            unify: bool = False, rgba: bool = False,
            verbose: bool = True) -> Dict[str, dict]:
    """Hydrate what the pipeline may read; return the deferred-weight table.

    Called right after `onnx.load(..., load_external_data=False)`: the whole
    model is value-free at that point, so every tensor this function decides to
    keep in memory is read from the external file individually and the rest
    never is.
    """
    graph = model.graph
    consumers: Dict[str, Counter] = {}
    for node in graph.node:
        for name in node.input:
            consumers.setdefault(name, Counter())[node.op_type] += 1

    refusal = deferral_refusal(quant, unify, rgba)
    cap = stream_min_bytes() if refusal is None else 0

    deferred: Dict[str, dict] = {}
    # The writer finds a payload by name, so a name repeated in the graph cannot
    # be tracked -- those tensors are hydrated instead of deferred.
    seen_names = Counter(init.name for init in graph.initializer)
    for init in graph.initializer:
        if not uses_external_data(init):
            continue  # inline already: the bytes are in the proto
        location, offset, length = _external_fields(init)
        keep_in_file = (
            refusal is None
            and seen_names[init.name] == 1
            and init.data_type in _PASS_THROUGH_DATA_TYPES
            and length >= cap
            and length == payload_bytes(init)
            and _only_weight_consumers(consumers.get(init.name)))
        if keep_in_file:
            entry = {
                "name": init.name,
                "location": location,
                "offset": offset,
                "length": length,
                "data_type": init.data_type,
                "dims": tuple(int(d) for d in init.dims),
                "consumers": sum(consumers.get(init.name, Counter()).values()),
            }
            if init.data_type == TensorProto.BFLOAT16:
                # Bytes stay in the source file; the recorded dtype flips to
                # fp16 NOW so _downcast_bfloat16 (which runs after prepare)
                # does not "convert" the empty raw_data of an external tensor
                # and leave bf16 bytes labeled fp16 in the trailer. The real
                # conversion happens chunk-wise in write_payload.
                entry["src_dtype"] = TensorProto.BFLOAT16
                entry["data_type"] = TensorProto.FLOAT16
                init.data_type = TensorProto.FLOAT16
            deferred[init.name] = entry

    # Everything else that points at an external file gets its bytes, with the
    # same state transition onnx's own whole-model loader applies, so no
    # downstream code can tell the two paths apart.
    hydrated = 0
    for tensor in _all_tensors(model):
        if tensor.name in deferred or not uses_external_data(tensor):
            continue
        _hydrate(tensor, base_dir)
        hydrated += 1

    # Fail closed on anything this walk did not reach: an external reference
    # nobody loaded becomes a silently missing weight, and the count is the
    # name-independent check that only the planned tensors stayed in the file.
    still_external = [t.name for t in _all_tensors(model) if uses_external_data(t)]
    if sorted(still_external) != sorted(deferred):
        extra = [n for n in still_external if n not in deferred]
        raise DeferredWeightError(
            f"{len(still_external)} tensor(s) still point at external data "
            f"after hydration but only {len(deferred)} were deferred "
            f"(unexpected: {extra[:5]}); _all_tensors does not reach where they "
            f"live, so streaming would lose them")

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
    also gives nothing back, so it is not deferred.
    """
    if not ops:
        return False
    return set(ops) <= _WEIGHT_ONLY_CONSUMERS


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
        if not is_deferred(init):
            problems.append(
                f"'{name}' lost its external-data reference (raw_data is "
                f"{len(init.raw_data)} byte(s)); nothing was going to stream it")
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
