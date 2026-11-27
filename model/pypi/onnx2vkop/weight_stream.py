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
import shutil
import tempfile
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
# Where the payload is bf16 after all (--act bf16's domain pass, -q bf16,
# --keep-bf16) the row is re-marked later and written as bf16 instead -- see
# mark_bf16_payload.
_PASS_THROUGH_DATA_TYPES = frozenset(
    {TensorProto.FLOAT16, TensorProto.BFLOAT16})
_PASS_THROUGH_ITEMSIZE = {TensorProto.FLOAT16: 2, TensorProto.BFLOAT16: 2,
                          TensorProto.FLOAT: 4,
                          TensorProto.INT8: 1,
                          TensorProto.FLOAT8E4M3FN: 1,
                          TensorProto.FLOAT8E5M2: 1}

# Formats whose payload can be built from a band of the source at a time, so
# the weight's bytes never have to be in RAM: their scale is re-derived from the
# weight's own magnitudes and the value is fully overwritten by the new payload.
# Gates the payload adds on top are the same gates `Quantizer._match_weight_only`
# applies, so the two stay one set by construction (a drift here is an unscaled s
# with nowhere to fold its 1/s -- a silently wrong model).
#
# int4 belongs here for a reason that is easy to miss. Its scale is grouped along
# K -- one entry per (group_size values of K, output column) -- which reads like
# "the scale spans K, so the matrix must be in hand". It does not: a group's
# scale is an absmax over its OWN group_size rows and nothing else, so a
# band-at-a-time sweep accumulates exactly the same table (quant_payload's
# _grouped_amax). Only the grouping boundary matters, and a band that starts
# mid-group is handled by slicing on `offset % group_size` rather than assuming
# the two line up. nf4 is here for the same reason: its scale is the same
# grouped absmax (without the /7) and its code is a per-element nearest-codebook
# lookup, the shared `Quantizer._nf4_encode`, so a band encodes exactly as the
# in-memory pass would (pinned byte-for-byte by test_streamed_4bit). nvfp4 is
# here too, although its per-TENSOR factor (every block scale is divided by it)
# reads like a whole-matrix reduction: that factor is amax_tensor / (448 * 6),
# and amax_tensor is the max of the [K/16, N] block-absmax table the same
# grouped sweep builds -- so one pass still yields both levels
# (quant_payload.nvfp4_tables).
_VALUE_FREE_FORMATS = ("int8", "fp8e4m3", "fp8e5m2", "int4", "nf4", "nvfp4")

# Values per band for the [K, N] 4-bit writer (see _write_int4).
_BAND_VALUES = 1 << 22

# Rows per band for the writers that re-lay an [N, K] source as [K, N]. Each
# band costs one positioned write per k, so these keep the historical
# ROW_BLOCK bands rather than the value-bounded default (quant_payload
# .default_block), which would multiply the write count on a wide K.
_TRANSPOSED_BAND_ROWS = 2048

# The grouped 4-bit members of that set: packed two codes per byte, a
# [K/group, N] scale table, and the [K, N]-only layout gate (_int4_layout_ok).
# nvfp4's group is fixed at 16 by the format, whatever --group-size says.
_FOUR_BIT_FORMATS = ("int4", "nf4", "nvfp4")
_NVFP4_BLOCK = 16

# The payload element type each of those produces. The deferred row
# records it, which is what lets the extent checks and the writer treat
# a 1-byte (or half-byte) payload as the same number of elements as the
# 4-byte source.
_PAYLOAD_DTYPES = {
    "int8": TensorProto.INT8,
    "fp8e4m3": TensorProto.FLOAT8E4M3FN,
    "fp8e5m2": TensorProto.FLOAT8E5M2,
    "int4": TensorProto.INT4,
    # ONNX has no NF4 element type: the nibble's storage is UINT4 and its
    # meaning rides the initializer's `vkop_dtype` metadata, which the
    # quantizer stamps on the deferred tensor (Quantizer._stream_4bit_rows).
    "nf4": TensorProto.UINT4,
    # NVFP4's E2M1 nibble; its two scale levels are separate initializers.
    "nvfp4": TensorProto.FLOAT4E2M1,
}


def exports_weight_values(quant: Optional[str], unify: bool,
                          rgba: bool) -> bool:
    """Does any pass read a weight's *values*, not just its byte count?

    AWQ's real-valued rescale, --unify and --rgba all need the values: AWQ
    multiplies the weight, and the other two concatenate or repack payloads. The
    value-free formats (int8, fp8, int4) only need the count plus a per-column or
    per-group absmax that a band-at-a-time pass computes, so a model under those
    can keep a 9 GB fp16 weight in its file instead of inflating it to 18 GB
    first. A plain float export copies what is there.
    """
    if unify or rgba:
        return True
    return quant not in _VALUE_FREE_FORMATS


# Ops that take a weight as an operand and are never evaluated during
# conversion, because one of their inputs is an activation.
_WEIGHT_ONLY_CONSUMERS = frozenset({"MatMul", "Gemm", "Conv"})

# Pure VIEWS: they move a tensor's indices around and nothing else, so a
# weight's bytes mean the same thing on either side of one -- the consumer can
# do the same arithmetic off the source layout instead of a materialized copy.
# `Transpose` is the one that matters in practice: an exporter that runs with
# constant folding off emits `Transpose(w)` into MatMul's B position for every
# linear, which on the 27B is 497 weights and 48.9 GiB of the 49.3 GiB total.
#
# A weight reachable from its consumer through ONLY these is still never read
# by the converter. What must not qualify is anything that reads a value:
# `Cast` changes the element type (a byte-format payload would be the wrong
# width), and `Mul`/`Add`/`Exp`/`Div` compute something the pipeline may fold.
_VIEW_OPS = frozenset({"Transpose", "Reshape", "Squeeze", "Unsqueeze",
                       "Identity", "Flatten"})

# How far to follow a view chain before giving up. The graphs that need this
# are one or two hops (`Transpose -> MatMul`, the shape every folding-off
# exporter emits); a bound keeps a pathological chain from turning the gate
# into a graph walk.
_VIEW_CHAIN_MAX = 8

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
    """Why deferral must not run, or None when it may.

    A value-free format (int8, fp8, int4, nf4) is allowed: it derives its scale from
    the weight's magnitudes and overwrites the values, so the bytes can be
    quantized on the way into the trailer (`quant_payload`) instead of being
    inflated into an array first. Their scales are LOCAL to a slice of the
    stored rows -- a per-output-column absmax for the byte formats, a
    per-(K group, output column) absmax for int4 -- so a band-at-a-time sweep
    computes exactly the table the in-memory quantizer would.

    nf4 streams like int4: the same grouped absmax (unscaled) and a
    per-element codebook lookup shared with the in-memory pass.

    nvfp4 streams too: its per-tensor factor is the max of the same grouped
    (16-value) absmax table, so the one sweep yields both scale levels.

    AWQ still refuses: it is a real-valued rescale of the whole weight
    (`W * s` needs every element before any of it can be written).
    """
    if not _env_flag(_ENV_ENABLE, True):
        return f"{_ENV_ENABLE}=0"
    # -q bf16 streams like -q fp16: its payload is the source's bits (a bf16
    # source) or each fp16 value rounded once, both element for element, so
    # write_payload converts a band at a time (mark_bf16_payload). It used to
    # hydrate the whole model for that.
    if quant is not None and quant not in ("fp16", "bf16") + _VALUE_FREE_FORMATS:
        return f"-q {quant} rewrites weight payloads and needs their bytes"
    if unify:
        return "--unify concatenates weight payloads and needs their bytes"
    if rgba:
        return "--rgba rewrites weight payloads and needs their bytes"
    return None


def payload_bytes(tensor, dtype=None) -> int:
    """Bytes this initializer occupies, from dtype and dims only.

    `dtype` overrides the tensor's own element type: a deferred weight under a
    byte format is about to become a 1-byte payload with the same element
    count, and every extent check has to ask for that length, not the source's.
    """
    n = 1
    for d in tensor.dims:
        n *= int(d)
    size = _PASS_THROUGH_ITEMSIZE.get(
        tensor.data_type if dtype is None else dtype)
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


def mark_bf16_payload(deferred: Dict[str, dict], name: str, init,
                      target: Optional[str] = None,
                      initializers: Optional[dict] = None) -> bool:
    """Make a deferred 16-bit float row write a bfloat16 payload.

    The streamed counterpart of a bf16 payload built in memory: a bf16 source
    is copied verbatim (its bits ARE the value), an fp16 one is rounded once,
    fp16 -> fp32 (exact) -> bf16 RNE, which is what fp16_graph's
    `_bf16_payload` / bf16_weights' `-q bf16` do to the same values in memory.
    `write_payload` does the conversion band by band, so the weight still never
    enters RAM.

    Only rows the writer copies element for element qualify: a plain
    pass-through or the SwiGLU gate/up interleave. A row that encodes (`quant`)
    or re-lays (`lane_major`, `transpose_payload`) its payload is refused, and
    so is any source dtype other than fp16/bf16. Returns whether it was marked.

    `target` other than `name` is a second, bf16 copy of a shared weight (the
    bf16 domain pass gives the bf16 readers their own tensor while an island
    keeps the export's): it gets its own row over the same source bytes and an
    external-data twin of `init` in `initializers`, so neither tensor is ever
    hydrated.
    """
    entry = deferred.get(name)
    if entry is None:
        return False
    for key in ("quant", "lane_major", "transpose_payload", "src_dims"):
        if entry.get(key) is not None:
            return False
    src = entry.get("src_dtype", entry.get("data_type"))
    if src not in (TensorProto.FLOAT16, TensorProto.BFLOAT16):
        return False
    if entry.get("interleave") is not None:
        spec = entry["interleave"]
        if any(spec[k].get("src_dtype", spec[k].get("data_type"))
               not in (TensorProto.FLOAT16, TensorProto.BFLOAT16)
               for k in ("g", "u")):
            return False
    if target is not None and target != name:
        if initializers is None:
            return False
        twin = TensorProto()
        twin.CopyFrom(init)
        twin.name = target
        twin.data_type = TensorProto.BFLOAT16
        new = dict(entry)
        new["name"] = target
        new["src_dtype"] = src
        new["data_type"] = TensorProto.BFLOAT16
        new["out_bf16"] = True
        new["consumers"] = 1
        deferred[target] = new
        initializers[target] = twin
        return True
    entry["src_dtype"] = src
    entry["data_type"] = TensorProto.BFLOAT16
    entry["out_bf16"] = True
    init.data_type = TensorProto.BFLOAT16
    return True


def _fp16_block_to_bf16(block: bytes) -> bytes:
    """fp16 bytes -> bf16 bytes: widened exactly, rounded once (RNE)."""
    try:
        from .bf16_weights import fp32_to_bf16_bits
    except ImportError:  # flat-module layout
        from bf16_weights import fp32_to_bf16_bits
    f32 = np.frombuffer(block, dtype="<f2").astype(np.float32)
    return np.ascontiguousarray(fp32_to_bf16_bits(f32)).astype("<u2").tobytes()


def add_fp16_to_bf16_roundtrip(stats, block: bytes) -> bytes:
    """Fold one fp16 source band's fp16 -> bf16 rounding into `stats` (a
    numeric_audit.RoundTrip) and return the bf16 bytes it becomes -- the same
    bytes write_payload will write for it."""
    out = _fp16_block_to_bf16(block)
    src = np.frombuffer(block, dtype="<f2").astype(np.float32)
    got = (np.frombuffer(out, dtype="<u2").astype(np.uint32)
           << 16).view(np.float32)
    stats.add(src, got)
    return out


def bf16_payload_roundtrip(entry: dict, base_dir: str = ""):
    """What writing a bf16-marked row (mark_bf16_payload) costs its weight, as
    a numeric_audit.RoundTrip -- the streamed counterpart of the
    record_roundtrip the in-memory bf16 conversions make.

    The source is read one band at a time (the conversion's own chunk size),
    so the weight never enters RAM: the statistics are sums and maxima that
    accumulate band by band. A bf16 source is copied verbatim and costs
    exactly nothing, so its elements are counted without reading it. A
    SwiGLU gate/up interleave row is the union of its two sources (the
    interleave only permutes them).
    """
    try:
        from . import numeric_audit
    except ImportError:  # flat-module layout
        import numeric_audit
    stats = numeric_audit.RoundTrip("bf16")
    spec = entry.get("interleave")
    sources = [spec["g"], spec["u"]] if spec is not None else [entry]
    for src in sources:
        dtype = src.get("src_dtype", src.get("data_type"))
        length = int(src["length"]) if spec is None else (
            int(spec["k"]) * int(spec["n"]) * 2)
        if dtype == TensorProto.BFLOAT16:
            stats.elements += length // 2
            continue
        if dtype != TensorProto.FLOAT16:
            raise DeferredWeightError(
                f"deferred weight '{entry['name']}': a bf16 payload from "
                f"source dtype {dtype}")
        path = source_read_path(src, None, base_dir)
        with open(path, "rb") as f:
            f.seek(int(src["offset"]))
            remaining = length
            while remaining:
                block = f.read(min(_CHUNK_BYTES, remaining))
                if not block:
                    raise DeferredWeightError(
                        f"deferred weight '{src['name']}' runs out of source "
                        f"bytes {remaining} byte(s) early in {path}")
                add_fp16_to_bf16_roundtrip(stats, block)
                remaining -= len(block)
    return stats


def _int4_layout_ok(init, uses: Dict[str, list], group_size: int,
                    matmul_only: bool = False) -> bool:
    """Can this weight's bytes become a packed 4-bit payload as they stream?

    The same gates `Quantizer._match_weight_only` applies to an in-memory
    weight, asked here because the decision to defer is made before the DAG
    exists. Everything is re-checked there; this only keeps a weight that can
    never be 4-bit out of the deferred set, where it would otherwise be written
    as a quantized payload the quantizer then refuses -- an artifact whose
    weight nobody decodes.

      * a plain 2-D matrix;
      * N % 8 -- a packed row has to start on a 32-bit word;
      * K % group_size, with group_size positive and even -- a group is a whole
        number of rows and the fp16 tap reads two halves per word;
      * a chain that reaches exactly one MatMul or Gemm in its B position. The
        runtime's 4-bit kernel hangs off MatMul, but `lower_gemm_to_matmul`
        rewrites every supported Gemm into MatMul + Add before the DAG exists,
        so a Gemm-declared weight ends up on that same kernel. A shared weight
        has no single layout, and a weight already carrying a third input is a
        graph this pass did not build;
      * the [K, N] orientation, which for a source behind a Transpose -- or a
        Gemm with transB=1, which means the same thing -- is the flip
        KnWeightLayout performs. The check is on the STORED dims, since the
        source's [N, K] is the payload's [K, N] the other way round.
    """
    dims = [int(d) for d in init.dims]
    if len(dims) != 2:
        return False
    chain = view_chain_to_weight(uses, init.name)
    if chain is None:
        return False
    node, operand = chain[-1]
    if node.op_type not in ("MatMul", "Gemm") or operand != 1:
        return False
    if matmul_only and node.op_type != "MatMul":
        # nvfp4 appends TWO inputs ([A, W, s, g]); lower_gemm_to_matmul keys
        # "has a bias" off the input count for a quantized weight, so a
        # four-input Gemm would lose its global scale to a bias Add. A Gemm's
        # weight is hydrated instead and quantized in memory after lowering.
        return False
    transposed = is_transposed_weight(uses, init.name)
    n, k = (dims[0], dims[1]) if transposed else (dims[1], dims[0])
    if n % 8:
        return False
    if group_size <= 0 or group_size % 2:
        return False
    return k % group_size == 0


def prepare(model, base_dir: str, *, quant: Optional[str] = None,
            unify: bool = False, rgba: bool = False, group_size: int = 0,
            keep_in_file=(), transcribe_slots: bool = False,
            verbose: bool = True) -> Dict[str, dict]:
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
    leave the pool float32, and `moe_source_dtype` on the row records that the
    source is bf16 while the buffer is fp16 -- the runtime notes it and converts
    on the way in, and the slice's extent is taken from the source file rather
    than derived), and they fail the consumer gate by design (an unwired slot is
    read by nobody). See `restore_paging_slots` for the contract each named slot
    is then held to.

    A declared slot is *not* transcribed into the trailer by default: it gets no
    deferred entry, so the writer lays no bytes down for it, and its extent is
    recorded as 0. The runtime pages every expert -- resident ones included --
    from the file the exporter named (`moe_source`/`moe_offset`/`moe_size`,
    vkop_model.fbs InitializerEntry), so a second copy in the trailer would be
    23 GiB of dead weight that also hid a missing source slice behind plausible
    bytes. Pass `transcribe_slots=True` to get that second copy anyway, which is
    what a model whose checkpoint has been deleted needs.
    """
    declared = {name for name in keep_in_file}
    graph = model.graph
    consumers: Dict[str, Counter] = {}
    # name -> [(node, operand_index)] for _reachable_as_weight. Built once: the
    # 27B asks about 850 weights, and rebuilding it per weight would walk the
    # 12.6k-node graph 850 times.
    uses: Dict[str, list] = {}
    for node in graph.node:
        for i, name in enumerate(node.input):
            consumers.setdefault(name, Counter())[node.op_type] += 1
            uses.setdefault(name, []).append((node, i))

    refusal = deferral_refusal(quant, unify, rgba)
    cap = stream_min_bytes() if refusal is None else 0

    deferred: Dict[str, dict] = {}
    # The writer finds a payload by name, so a name repeated in the graph cannot
    # be tracked by `deferred` alone -- those tensors are hydrated instead of
    # deferred, unless they are declared slots (see below).
    seen_names = Counter(init.name for init in graph.initializer)
    for init in graph.initializer:
        if not uses_external_data(init):
            continue  # inline already: the bytes are in the proto
        location, offset, length = _external_fields(init)
        # Captured before `_stream_entry` runs: for a bfloat16 tensor that
        # helper flips `init.data_type` to FLOAT16 in place (so the tensor's
        # empty raw_data is not handed to `_downcast_bfloat16` as if it held
        # bytes), and the file's own element type is what the payload reader
        # must seek by.
        src_dtype = init.data_type
        is_slot = init.name in declared
        if is_slot and not transcribe_slots:
            # A slot is paged from the file the exporter named, never from the
            # trailer, so it gets no bytes here. Recording it in `deferred`
            # would make the writer copy 23 GiB into the vkopbin only for the
            # runtime to read a different file anyway (see `_slot_entries`).
            continue
        keep_in_file_gates = (refusal is None
                              and seen_names[init.name] == 1
                              and length >= cap)
        # A value-free format quantizes every float weight it can see, whatever
        # float the source file holds. The fp16 case used to fall through as a
        # plain pass-through, which was wrong twice over: the payload shipped as
        # fp16 (silently unquantized) and the down_act fold, which reads its
        # three weights as one concatenated tensor, died demanding bytes a
        # deferred row does not have. Both are fixed by recording such a weight
        # as a *quantizing* deferred entry, exactly like the fp32 case below.
        #
        # BFLOAT16 belongs here for the same reason and was the bug that made a
        # bf16 checkpoint convert to an int4 artifact byte-identical to its fp16
        # twin: every MiniMax-H3 tensor is bf16, so `quantizes` was False for
        # all 706 of them, each took the plain pass-through branch, and the
        # quantizer then skipped every deferred weight (optimizer's "left in its
        # source file, weight_stream builds the nibbles") -- both sides assuming
        # the other would encode it. The source dtype is recorded separately, so
        # the reader still seeks the file in 2-byte bf16 rows and the encoder
        # converts each band to fp32 before the absmax/round.
        quantizes = (init.data_type in (TensorProto.FLOAT,
                                        TensorProto.FLOAT16,
                                        TensorProto.BFLOAT16)
                     and not exports_weight_values(quant, unify, rgba))
        if keep_in_file_gates and not is_slot:
            keep_in_file_gates = (
                (init.data_type in _PASS_THROUGH_DATA_TYPES
                 or quantizes)
                and length == payload_bytes(init)
                # Direct MatMul/Gemm/Conv, or the same thing one view deeper.
                # The second form is what an exporter with constant folding off
                # produces, and on the 27B it is 48.9 GiB of the 49.3 GiB.
                and (_only_weight_consumers(consumers.get(init.name))
                     or _reachable_as_weight(uses, init.name)))
            if (keep_in_file_gates and quantizes
                    and quant in _FOUR_BIT_FORMATS):
                # A 4-bit payload has a layout the graph has to agree with
                # before it can be deferred at all: the nibble kernels unpack
                # only a [K, N] payload with N % 8, and this pass -- not the
                # quantizer -- is the last one that still sees the Transpose
                # that says which axis holds the columns. The quantizer's gate
                # re-checks all of it against the finished DAG; what is decided
                # here is only what the ROW records: the group size (the scale
                # table's row length, which the kernel reads back as K /
                # n_groups) and the stored layout the source reader seeks by.
                keep_in_file_gates = _int4_layout_ok(
                    init, uses,
                    _NVFP4_BLOCK if quant == "nvfp4" else group_size,
                    matmul_only=quant == "nvfp4")
        if keep_in_file_gates:
            entry = _stream_entry(
                init, base_dir, location, offset, length,
                sum(consumers.get(init.name, Counter()).values()))
            if quantizes:
                # A value-free format quantizes this weight's bytes on the way
                # into the trailer, so the row says both what the file holds
                # (`src_dtype`) and what the payload will be (`data_type`):
                # the writer, verify_deferred and the length checks key on the
                # latter, while the tensor keeps the float type so no pass
                # believes 1-byte values sit in a 2- or 4-byte payload.
                # quant_payload fills in the scale.
                #
                # `src_dtype` is the dtype captured before `_stream_entry` ran,
                # not `init.data_type` (which that helper has just flipped to
                # FLOAT16 for a bf16 source). The reader seeks the file by this
                # and decodes with it, so overwriting bf16 with fp16 would make
                # every band decode as a half-typed bit pattern.
                entry["src_dtype"] = src_dtype
                entry["data_type"] = _PAYLOAD_DTYPES[quant]
                entry["quant"] = quant
                # Which stored axis holds the output columns, decided by the
                # consumer chain -- see is_transposed_weight. Read here because
                # this is the only point that still has the graph: by the time
                # the scale is computed the Transpose may have been folded away.
                entry["transposed"] = is_transposed_weight(uses, init.name)
                if quant in _FOUR_BIT_FORMATS:
                    # The group length, recorded because the scale table's row
                    # count is derived from it (n_groups = K / group) and a
                    # reader has only the payload's dims to go on.
                    entry["group"] = int(_NVFP4_BLOCK if quant == "nvfp4"
                                         else group_size)
            deferred[init.name] = entry

    # A declared slot the gates above turned down anyway -- a name the graph
    # repeats is the one case left -- must still not be hydrated: its bytes are
    # the expert pool. It is streamed as a slot instead, and
    # `restore_paging_slots` holds it to the slot contract. Walked in graph
    # order, not sorted, so the order the slot entries come out in is the order
    # the exporter declared them in.
    if transcribe_slots:
        for init in graph.initializer:
            if init.name in deferred or init.name not in declared:
                continue
            if refusal is not None:
                continue
            location, offset, length = _external_fields(init)
            deferred[init.name] = _stream_entry(
                init, base_dir, location, offset, length, 1)

    # Everything else that points at an external file gets its bytes, with the
    # same state transition onnx's own whole-model loader applies, so no
    # downstream code can tell the two paths apart.
    streams = set(deferred) | declared
    hydrated = 0
    for tensor in _all_tensors(model):
        if not uses_external_data(tensor) or tensor.name in streams:
            continue
        _hydrate(tensor, base_dir)
        hydrated += 1

    # Fail closed on anything this walk did not reach: an external reference
    # nobody loaded becomes a silently missing weight. Compared by name, not by
    # object: `_all_tensors` builds fresh TensorProto wrappers on every scan (it
    # reads 6382 objects for 6308 distinct tensors on llm_moe_paged.onnx), so
    # object identity does not survive from one scan to the next, while the
    # name set is exactly what the writer keys payloads on.
    still_external = {t.name for t in _all_tensors(model) if uses_external_data(t)}
    # A declared slot is *meant* to stay external -- prepare skips it and
    # restore_paging_slots clears its pointer later -- so it is not a leak.
    unexpected = sorted(still_external - streams)
    if unexpected:
        raise DeferredWeightError(
            f"{len(still_external)} name(s) still point at external data after "
            f"hydration, and {len(unexpected)} of them were not meant to stay "
            f"there (unexpected: {unexpected[:5]}); _all_tensors does not reach "
            f"where they live, so streaming would lose them")

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


def view_chain_to_weight(uses: Dict[str, list], name):
    """The pure-view chain from `name` to a weight position, or None.

    Shared by `_reachable_as_weight` (does such a chain exist?) and
    `is_transposed_weight` (does it flip the matrix?), because the two ask
    about the same walk and computing it twice invites them to disagree.

    Returns the list of (node, operand_index) hops, ending at the weight
    consumer, or None when no chain qualifies -- see `_reachable_as_weight`
    for the rule and why each part of it is there.
    """
    chain = []
    cur = name
    for _ in range(_VIEW_CHAIN_MAX):
        consumers = uses.get(cur, [])
        if len(consumers) != 1:
            # Nobody reads it (a dead initializer: the writer prunes it anyway,
            # so deferring gives nothing back), or several do (a branch).
            return None
        node, operand = consumers[0]
        op = node.op_type
        if op in _WEIGHT_ONLY_CONSUMERS:
            # Reached a consumer. It must be reading this in a weight position;
            # a weight in an activation slot would mean the op computes on it.
            if operand < 1:
                return None
            chain.append((node, operand))
            return chain
        if op not in _VIEW_OPS:
            return None
        # A view with several outputs (Split) would not be a view of the same
        # tensor; `_VIEW_OPS` holds none, but the check keeps that honest if
        # the set grows.
        if len(node.output) != 1:
            return None
        chain.append((node, operand))
        cur = node.output[0]
    return None


def _last_two_swap(node) -> bool:
    """Does this node make its weight read as the transpose of its storage?

    Two nodes do, and both matter for the same reason -- they decide which
    STORED axis holds the output columns, and therefore which axis a 4-bit
    scale table is reduced over:

      * `Transpose` with a perm that swaps the last two dims (any other perm is
        not a matrix transpose of a 2-D weight);
      * `Gemm` with transB=1. Gemm's B is the checkpoint's [N, K] and transB=1
        says "read it as [K, N]", which is exactly what the Transpose spelling
        means. This is not a hypothetical: MiniMax-H3's adaLN projections are
        Gemm(transB=1), 496 MiB each across 50 layers (24.31 GiB), and every
        one of them was hydrated into RAM because this function said "not
        transposed" and `_int4_layout_ok` therefore refused to defer it.

    `lower_gemm_to_matmul` later rewrites Gemm into MatMul + Add (carrying
    transB onto the MatMul), so the runtime never sees a Gemm either way.
    """
    if node.op_type == "Gemm":
        return bool(next((a.i for a in node.attribute if a.name == "transB"), 0))
    if node.op_type != "Transpose":
        return False
    perm = next((list(a.ints) for a in node.attribute if a.name == "perm"), None)
    if not perm or len(perm) < 2:
        return False
    r = len(perm)
    return (all(perm[i] == i for i in range(r - 2))
            and perm[r - 2] == r - 1 and perm[r - 1] == r - 2)


def is_transposed_weight(uses: Dict[str, list], name) -> bool:
    """Does this weight reach its consumer through a matrix transpose?

    Decides the AXIS of the per-output-column scale, and it is not cosmetic:
    the stored bytes of a checkpoint-layout weight are [N, K], because that is
    how nn.Linear keeps them and the exporter emits the transpose as a node.
    A scale is always per OUTPUT COLUMN (N entries), but which STORED axis
    holds the columns depends on whether that Transpose is still in the chain:

        direct   [K, N]  -> columns are axis 1 -> reduce axis 0
        Transpose[N, K]  -> columns are axis 0 -> reduce axis 1

    Getting this wrong is silent in the worst way: the payload still has the
    right size and the artifact still loads, so the only symptom is a model
    that computes wrong answers.
    """
    chain = view_chain_to_weight(uses, name)
    if chain is None:
        return False
    return any(_last_two_swap(node) for node, _ in chain)


def _reachable_as_weight(uses: Dict[str, list], name) -> bool:
    """Is every path from `name` to a live consumer a pure view chain?

    `_only_weight_consumers` answers this only for a weight wired *directly*
    into MatMul/Gemm/Conv. An exporter run with constant folding off wires it
    through a `Transpose` instead (see `_VIEW_OPS`), and that one hop is the
    difference between a 27B conversion fitting in RAM and OOMing: on
    qwen3_8_27b, 497 of 850 weights arrive as `Transpose(w) -> MatMul`, and
    hydrating them is 48.9 GiB.

    A view moves indices, not values, so a weight that reaches its consumer
    through only views is still never READ by the converter -- the same
    argument `_only_weight_consumers` makes, one hop longer. Anything that
    inspects or rewrites a value disqualifies the chain, `Cast` included (a
    byte-format payload has to be the width it ships as).

    Everything on the chain must also be single-consumer. A view feeding two
    nodes is fine in principle, but the gate's promise is about THE tensor whose
    bytes are being left on disk, and a branch means one of those paths is
    somewhere the reasoning has not looked.

    `uses` is name -> [(node, operand_index)], built once by `prepare` rather
    than rebuilt per weight. Returns True only for a chain that ENDS at a
    weight position.
    """
    return view_chain_to_weight(uses, name) is not None


def slot_sources(model, base_dir: str, slot_names) -> Dict[str, tuple]:
    """Locate each paging slot's bytes in the file the exporter streamed it from.

    Returns name -> (location, offset, length, data_type) where `location` is
    the file relative to `base_dir` and data_type is the dtype *of the bytes
    there* (bfloat16 stays bfloat16: the runtime converts a paged expert on the
    way in, exactly as the loader converts any bf16 initializer).

    The extent is taken from `external_data` rather than derived from dtype and
    dims, which is why it is read here, before `restore_paging_slots` clears the
    pointer. The two agree for every slot the exporter emits (checked, not
    assumed) -- but only the file knows how many bytes the expert actually has,
    and a mismatch is a runtime pread of the wrong length.

    This is the addressing unit the runtime pages by anyway: a slot's
    (location, offset, length) *is* its entry in the source file. vkopbin has no
    way to keep it that way (the trailer is one file, the checkpoint's shards
    are not), so the slice is recorded in the slot's InitializerEntry instead.
    """
    by_name = {t.name: t for t in _all_tensors(model)}
    sources: Dict[str, tuple] = {}
    for name in slot_names:
        tensor = by_name.get(name)
        if tensor is None:
            raise DeferredWeightError(
                f"MoE paging slot '{name}' is not a tensor of this graph; the "
                f"exporter's slot declarations name something the model does "
                f"not have (see make_moe_variant.py).")
        if not uses_external_data(tensor):
            # A slot whose declaration lost its external pointer is one
            # `restore_paging_slots` already handled; its extent is known from
            # the deferred entry it built, but the *location* is gone. Nothing
            # can page from a file name that is no longer written down.
            raise DeferredWeightError(
                f"MoE paging slot '{name}' has no external_data: its source "
                f"file is unknown, so the runtime has nothing to page from. "
                f"slot_sources() must run before restore_paging_slots().")
        location, offset, length = _external_fields(tensor)
        sources[name] = (location, int(offset), int(length), tensor.data_type)
    return sources


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
                f"which the runtime preads straight into the slot's buffer, so the "
                f"slice would point past or short of the slot")
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


class InPlaceSourceGuard:
    """Keeps a deferred weight's source readable when the output is that file.

    `onnx2vkop -i m.onnx -o m.vkopbin` whose deferred weights already point at
    that .vkopbin is a real re-conversion: the writer opens the destination with
    "wb", which truncates the very file every payload is read from.

    Keeping a read fd open does NOT help -- `open(..., "wb")` truncates the
    inode itself, so every descriptor on it sees zero bytes. The source is
    copied aside instead, and reads are redirected there. The copy is only made
    when the source and the destination really are the same file, so an ordinary
    conversion pays nothing.

    This has to be established BEFORE the destination is opened for writing --
    by the time a payload writer runs, the truncation has already happened.

    Usage::

        with InPlaceSourceGuard.open(deferred, out_path) as guard:
            with open(out_path, "wb") as f:
                ... write_payload(entry, f, base_dir) ...
    """

    def __init__(self):
        self._tmpdir = None
        self.paths = {}     # resolved source path -> the private copy

    @classmethod
    def open(cls, deferred: dict, out_path: str) -> "InPlaceSourceGuard":
        """Copy aside every deferred source that names `out_path` itself."""
        guard = cls()
        try:
            out_real = os.path.realpath(out_path)
        except OSError:
            return guard
        base = os.path.dirname(out_real)
        for entry in deferred.values():
            src = entry.get("location")
            if not src:
                continue
            if not os.path.isabs(src):
                src = os.path.join(entry.get("base_dir") or base, src)
            try:
                src_real = os.path.realpath(src)
            except OSError:
                continue
            if src_real != out_real or src_real in guard.paths:
                continue
            if guard._tmpdir is None:
                guard._tmpdir = tempfile.mkdtemp(prefix="vkop-inplace-")
            # Same directory is not required: the copy is private and short
            # lived. A multi-GB checkpoint costs one streamed copy here, which
            # is the price of not destroying the input.
            dst = os.path.join(guard._tmpdir, "src%d" % len(guard.paths))
            try:
                shutil.copyfile(src_real, dst)
            except OSError:
                continue
            guard.paths[src_real] = dst
        return guard

    @property
    def active(self) -> bool:
        return bool(self.paths)

    def __enter__(self):
        global _active_guard
        _active_guard = self
        return self

    def __exit__(self, *exc):
        global _active_guard
        _active_guard = None
        if self._tmpdir is not None:
            shutil.rmtree(self._tmpdir, ignore_errors=True)
            self._tmpdir = None
        self.paths = {}
        return False


# The guard in force while save_to_binary writes its blob. Module-level because
# every payload writer (write_payload and the helper readers it dispatches to)
# resolves its source through source_read_path, and threading a parameter
# through all of them would only add a way to forget it.
_active_guard = None


def source_read_path(entry: dict, out_f, base_dir: str,
                     guard: "InPlaceSourceGuard" = None) -> str:
    """The path a payload writer should read `entry`'s source bytes from.

    Normally that is just entry["location"] resolved against base_dir. The
    exception is an in-place reconversion, handled by InPlaceSourceGuard: this
    returns the guard's private copy for a source it took aside, so the read
    lands on the old bytes rather than the truncated destination. A guard that
    copied nothing (the common case) leaves the plain path alone.
    """
    path = entry["location"]
    if not os.path.isabs(path):
        path = os.path.join(entry.get("base_dir") or base_dir, path)
    g = guard if guard is not None else _active_guard
    if g is not None and g.paths:
        try:
            real = os.path.realpath(path)
            if real in g.paths:
                return g.paths[real]
        except OSError:
            pass
    return path


def write_payload(entry: dict, out_f, base_dir: str, dag_model=None) -> int:
    """Copy one deferred tensor from its source file into the output.

    Returns the bytes written. A short read is a hard error: the side table
    already promised `length` bytes at that offset, and a trailer that is short
    would make the runtime read the neighbouring tensor. A bf16 source is
    converted chunk-wise to fp16 (the runtime's only 16-bit float storage);
    bf16 -> fp16 is the fp32 value rounded once, the same recipe as
    converter._bf16_bytes_to_fp16, so both paths agree bit-for-bit.

    An entry carrying `quant` is the same idea for a byte format: its scale is
    the source's per-column absmax, computed and stashed by quant_payload at
    prepare time, and every band is quantized as it goes past. The output is
    one byte per element where the source had four, which is why the entry --
    not the tensor -- is what says how many bytes this writes.
    """
    if entry.get("quant") is not None:
        return _write_quantized(entry, out_f, base_dir, dag_model)
    if entry.get("lane_major") is not None:
        return _write_lane_major(entry, out_f, base_dir)
    if entry.get("interleave") is not None:
        return _write_interleave(entry, out_f, base_dir)
    path = source_read_path(entry, out_f, base_dir)
    remaining = entry["length"]
    # A row bf16_weights flipped (--keep-bf16: the weight lands in a slot the
    # runtime reads bf16 from) copies the source bits verbatim instead, and
    # so does a bf16 source whose payload is bf16 (`out_bf16`, see
    # mark_bf16_payload). An fp16 source with a bf16 payload is rounded.
    out_bf16 = bool(entry.get("out_bf16") or entry.get("keep_bf16"))
    to_fp16 = (entry.get("src_dtype") == TensorProto.BFLOAT16
               and not out_bf16)
    fp16_to_bf16 = (out_bf16
                    and entry.get("src_dtype") == TensorProto.FLOAT16)
    # A kept bf16 row was range-scanned from this same source when the graph
    # was built (bf16_weights.apply), because its bf16_wide attribute had to be
    # decided before the payload is written. The bytes pass through here
    # anyway, so the scan is repeated on them: a source that changed in
    # between would otherwise ship a weight the attribute misdescribes.
    tile_scan = None
    if out_bf16 and "bf16_tile_overflow" in entry:
        try:
            from .bf16_weights import TileScan
        except ImportError:  # flat-module layout
            from bf16_weights import TileScan
        tile_scan = TileScan()
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
            elif fp16_to_bf16:
                # Same byte count out as in (2 -> 2), so `length` holds.
                block = _fp16_block_to_bf16(block)
            if tile_scan is not None:
                tile_scan.add_bytes(block)
            out_f.write(block)
            remaining -= len(block)
    if tile_scan is not None and (bool(tile_scan.overflow)
                                  != bool(entry["bf16_tile_overflow"])):
        raise DeferredWeightError(
            f"deferred bf16 weight '{entry['name']}': {path} now holds "
            f"{tile_scan.overflow} element(s) past fp16's range where the "
            f"graph was built from {entry['bf16_tile_overflow']}; the source "
            f"changed during the conversion, so its bf16_wide attribute is "
            f"wrong. Re-run the conversion.")
    return entry["length"] - remaining


def _read_rows(src_entry: dict, base_dir: str, row0: int, rows: int,
               n_dim: int, out_f=None, out_bf16: bool = False) -> np.ndarray:
    """Rows [row0, row0+rows) of a [K, N] 16-bit deferred source, as fp16 bits
    (bf16 bits with `out_bf16`: a bf16 source verbatim, an fp16 one rounded
    once -- see mark_bf16_payload).

    `out_f` is the file being written; when the source and that file are the
    same file (an in-place reconversion) it is what redirects this read to the
    guard's copy -- see source_read_path.
    """
    path = source_read_path(src_entry, out_f, base_dir)
    want = rows * n_dim * 2
    with open(path, "rb") as f:
        f.seek(int(src_entry["offset"]) + row0 * n_dim * 2)
        block = f.read(want)
    if len(block) != want:
        raise DeferredWeightError(
            f"deferred weight '{src_entry['name']}' runs out of source bytes "
            f"while interleaving for the SwiGLU gate/up fold")
    u16 = np.frombuffer(block, dtype=np.uint16)
    src_bf16 = src_entry.get("src_dtype") == TensorProto.BFLOAT16
    if out_bf16:
        if not src_bf16:
            u16 = np.frombuffer(_fp16_block_to_bf16(block), dtype=np.uint16)
    elif src_bf16:
        f32 = (u16.astype(np.uint32) << 16).view(np.float32)
        with np.errstate(over="ignore"):
            u16 = f32.astype(np.float16).view(np.uint16)
    return u16.reshape(rows, n_dim)


def _write_interleave(entry: dict, out_f, base_dir: str) -> int:
    """Stream the SwiGLU gate/up fold's weight: Wg and Wu ([K, N] each) as one
    [K, 2N] payload interleaved four columns at a time (g0..g3 u0..u3 g4 ..),
    see FusionOptimizer.fold_swiglu_gate_up. Read in row bands from both
    sources, so neither is ever held whole."""
    spec = entry["interleave"]
    k_dim, n_dim = int(spec["k"]), int(spec["n"])
    band = max(1, _CHUNK_BYTES // max(1, n_dim * 2))
    written = 0
    row = 0
    while row < k_dim:
        take = min(band, k_dim - row)
        out_bf16 = bool(entry.get("out_bf16"))
        g = _read_rows(spec["g"], base_dir, row, take, n_dim, out_f, out_bf16)
        u = _read_rows(spec["u"], base_dir, row, take, n_dim, out_f, out_bf16)
        gu = np.concatenate([g.reshape(take, n_dim // 4, 4),
                             u.reshape(take, n_dim // 4, 4)], axis=2)
        out_f.write(np.ascontiguousarray(gu).tobytes())
        written += take * 2 * n_dim * 2
        row += take
    return written


def _write_lane_major(entry: dict, out_f, base_dir: str) -> int:
    """Stream a lane-major entry's source, writing the row-major reorder.

    The SwiGLU fold's gate/up weights are read by ONE lane per output column for
    every k, so in the source's [K, N] layout a lane's consecutive k's sit N
    elements apart: every load is a 2-byte access into a fresh 128-byte line.
    The fold wants eight consecutive columns of a fixed k, i.e. [K][N] read in
    16-byte groups, so the payload is reordered to

        [K][ceil(N/8)][8]        (rows straight from the source, columns grouped)

    which is a pure permutation of the same numel elements: no value changes, no
    precision is touched. A trailing group whose column count is not 8 is
    zero-padded to keep every group a whole 16-byte word (the kernel's row stride
    is then the padded width, which the host passes as K_pad).

    The source is read in row bands so a wide weight never materializes more
    than a few hundred MiB.
    """
    path = source_read_path(entry, out_f, base_dir)
    k_dim, n_dim, padded = entry["lane_major"]
    rows = entry["length"] // max(1, n_dim * 2)   # fp16 source: 2 bytes/element
    written = 0
    with open(path, "rb") as src:
        src.seek(entry["offset"])
        band = max(1, _CHUNK_BYTES // max(1, n_dim * 2))
        left = rows
        while left:
            take = min(band, left)
            block = src.read(take * n_dim * 2)
            if len(block) != take * n_dim * 2:
                raise DeferredWeightError(
                    f"deferred weight '{entry['name']}' runs out of source "
                    f"bytes while reordering to lane-major")
            a = np.frombuffer(block, dtype=np.uint16).reshape(take, n_dim)
            pad = np.zeros((take, padded - n_dim), dtype=np.uint16)
            out_f.write(
                np.ascontiguousarray(np.concatenate([a, pad], axis=1)).tobytes())
            written += take * padded * 2
            left -= take
    # K is implied by the payload size; keep the entry honest about it.
    del k_dim
    return written


def scale_values_for(entry: dict, dag_model):
    """The scale table a quantizing entry decodes with, as an ndarray.

    `quant_payload.finalize` put the numbers into the graph as an initializer
    named `entry["scale"]`; they are read back from there rather than kept on
    the entry, because a 4-bit conversion's tables are 1.5 GiB on the 27B and
    holding a second numpy copy is pure peak RSS (see finalize's comment).

    `entry["scale_values"]` is still honoured for callers that set it -- tests
    build entries by hand -- but the normal path is the initializer.
    """
    cached = entry.get("scale_values")
    if cached is not None:
        return np.asarray(cached, dtype=np.float32)
    if entry.get("quant") == "nvfp4":
        return _nvfp4_scale_values(entry, dag_model)
    name = entry.get("scale")
    if not name:
        raise DeferredWeightError(
            f"deferred weight '{entry['name']}' is quantized but carries no "
            f"scale initializer name; quant_payload.finalize did not run")
    init = dag_model.initializers.get(name)
    if init is None:
        raise DeferredWeightError(
            f"deferred weight '{entry['name']}': scale initializer '{name}' is "
            f"not in the model, so the payload cannot be decoded")
    from onnx import numpy_helper
    return np.ascontiguousarray(
        np.asarray(numpy_helper.to_array(init), dtype=np.float32))


def _scale_init(entry: dict, dag_model, key: str):
    name = entry.get(key)
    if not name:
        raise DeferredWeightError(
            f"deferred weight '{entry['name']}' is quantized but carries no "
            f"{key} initializer name; quant_payload.finalize did not run")
    init = dag_model.initializers.get(name)
    if init is None:
        raise DeferredWeightError(
            f"deferred weight '{entry['name']}': {key} initializer '{name}' is "
            f"not in the model, so the payload cannot be decoded")
    return init


def nvfp4_global_for(entry: dict, dag_model) -> np.float32:
    """The per-tensor fp32 factor of a streamed nvfp4 weight, from the graph."""
    init = _scale_init(entry, dag_model, "scale_global")
    from onnx import numpy_helper
    return np.float32(np.asarray(numpy_helper.to_array(init),
                                 dtype=np.float32).reshape(-1)[0])


def _nvfp4_scale_values(entry: dict, dag_model):
    """The fp32 [K/16, N] factor table a streamed nvfp4 weight decodes with.

    finalize stored the two levels the kernel reads (e4m3 bytes and the fp32
    global); the encoder wants their product, which is rebuilt here with the
    in-memory pass's own arithmetic (Quantizer._nvfp4_dequant_table). The e4m3
    table is read from raw_data rather than through numpy_helper, which needs
    ml_dtypes for fp8 and the converter does not depend on it.
    """
    from .optimizer import Quantizer

    block = _scale_init(entry, dag_model, "scale")
    codes = np.frombuffer(block.raw_data, dtype=np.uint8).reshape(
        [int(d) for d in block.dims])
    return Quantizer._nvfp4_dequant_table(
        codes, nvfp4_global_for(entry, dag_model))


def _write_quantized(entry: dict, out_f, base_dir: str, dag_model=None) -> int:
    """Stream a quantizing entry's source, write the payload the row names.

    The scale was computed at prepare time and lives on the entry, so this is a
    straight pass: read a band, encode it, write it. The band length is
    quant_payload.ROW_BLOCK rows, so a wide weight never materializes more than
    a few hundred MiB.

    An entry carrying `transpose_payload` writes the same codes in the other
    order: the source is the checkpoint's [N, K], the side table promises [K, N]
    (see KnWeightLayout), and every code keeps its value and its scale index --
    only the byte it lands on moves. The scale is per OUTPUT COLUMN (or per
    output column per K group) either way, which is why it is computed from the
    source (where those columns are rows) and needs no second pass here.
    """
    from . import quant_payload

    if entry.get("quant") in _FOUR_BIT_FORMATS:
        return _write_int4(entry, out_f, dag_model, base_dir)

    if entry.get("transpose_payload"):
        return _write_quantized_transposed(entry, out_f, base_dir, dag_model)

    reader = quant_payload.WeightReader(
        entry, entry["name"],
        path_for=lambda e: source_read_path(e, out_f, base_dir))
    # Which stored axis holds the output columns -- recorded by prepare while
    # the Transpose was still in the graph, because it decides both the scale's
    # axis and how the scale broadcasts over a band.
    transposed = bool(entry.get("transposed"))
    # `entry["scale"]` is the initializer's NAME; the values live in the graph
    # (scale_values_for). Recomputing them here would mean a second full pass
    # over the source.
    scale = (scale_values_for(entry, dag_model) if dag_model is not None
             else entry.get("scale_values"))
    if scale is None:
        scale = quant_payload.scale_for(entry["quant"], reader, transposed)
    codes = quant_payload.code_byte(entry["quant"], reader, scale, transposed)
    written = 0
    for offset, band in reader.bands():
        blob = codes(offset, band)
        out_f.write(blob.tobytes())
        written += blob.size
    return written


def _pack_nibbles(codes) -> np.ndarray:
    """Two int4 codes per byte along the LAST axis, the even one low.

    That is ONNX's INT4 packing and the kernel's: `unpack_w4(word, base)` reads
    element `e` from nibble `e` of the word, so a byte holds elements 2c and
    2c+1 of the row. A signed code's nibble is its two's complement, which
    `& 0xF` produces; the shader sign-extends it back with bitfieldExtract.

    The last axis is the output-column axis of a payload row (`uPC.N` columns
    per K), and the gate has already made it a multiple of 8, so a row always
    ends on a byte boundary and never shares one with the next row.
    """
    if codes.shape[-1] % 2:
        raise DeferredWeightError(
            f"cannot pack {codes.shape[-1]} codes per row into nibbles; a "
            f"payload row must be an even number of columns")
    nib = codes.astype(np.uint8) & 0xF
    return np.ascontiguousarray(nib[..., 0::2] | (nib[..., 1::2] << 4))


def _write_int4(entry: dict, out_f, dag_model=None,
                base_dir: str = "") -> int:
    """Stream a 4-bit (int4 / nf4) entry's source, write packed nibbles.

    Two codes per byte, the even element low. The format only changes the
    per-element encode (quant_payload.code_nibble); everything about layout
    below is shared.

    Two layouts, one payload:

      * the source is already [K, N] -- a direct MatMul weight. Bands are
        consecutive K rows, each row packs its own column pairs, and the stream
        is exactly the payload in order.
      * the source is [N, K] -- the checkpoint layout behind a folded Transpose,
        which KnWeightLayout re-lays as [K, N]. A band is a run of output
        COLUMNS, so each of its quantized rows is a slice of every payload row:
        after the transpose each k gets `rows / 2` bytes at
        `(k * N + offset) / 2`. `offset` is even for every band the reader
        produces (ROW_BLOCK is even and the gate makes N a multiple of 8), so
        a nibble pair never straddles two bands; that is checked rather than
        assumed.

    The scale table is [K / group, N] fp32 either way, and `code_nibble` picks
    each row's group from its K index, so a band that starts mid-group is
    divided by the right scale for every row it contains.
    """
    from . import quant_payload

    fmt = entry["quant"]
    reader = quant_payload.reader_for(
        entry, entry["name"],
        path_for=lambda e: source_read_path(e, out_f, base_dir))
    transposed = bool(entry.get("transposed"))
    group = int(entry.get("group") or 0)
    if group <= 0:
        raise DeferredWeightError(
            f"deferred weight '{entry['name']}' is {fmt} but its row records no "
            f"group size; the scale table's row length cannot be derived")
    scale = (scale_values_for(entry, dag_model) if dag_model is not None
             else entry.get("scale_values"))
    if scale is None:
        scale = quant_payload.scale_for(fmt, reader, transposed, group)
    codes = quant_payload.code_nibble(fmt, reader, scale, group, transposed)

    if entry.get("transpose_payload"):
        return _write_int4_transposed(entry, out_f, reader, codes)
    if transposed:
        # The source is the checkpoint's [N, K] and KnWeightLayout did not flip
        # it -- so the consumer still says transB=1, which the nibble kernel
        # refuses. Writing the stored layout here would produce a payload the
        # side table describes as [N, K] that the runtime would then index as
        # [K, N]: right size, garbage values, no error. The weight can only be
        # 4-bit if the flip happened, so this aborts rather than guesses.
        # (KnWeightLayout can decline for reasons prepare cannot see: a K the
        # graph does not pin, or a consumer set fusion changed.)
        raise DeferredWeightError(
            f"deferred weight '{entry['name']}' was recorded as a [N, K] "
            f"source but no pass re-laid it as [K, N]; a 4-bit payload is only "
            f"readable in that orientation, so encoding the stored order would "
            f"ship silently transposed weights. Reconvert with "
            f"{_ENV_ENABLE}=0, or exclude this weight from -q {fmt}.")

    # A [K, N] band is consecutive payload rows, so its size only bounds memory:
    # ~4M values (16 MiB of fp32 plus the encoder's temporaries) instead of
    # ROW_BLOCK whole rows, which for a 59264-column lm_head was 485 MiB of
    # fp32 band and as much again in temporaries.
    block = max(1, _BAND_VALUES // max(1, reader.n))
    written = 0
    for offset, band in reader.bands(block=block):
        packed = _pack_nibbles(codes(offset, band))
        out_f.write(packed.tobytes())
        written += packed.nbytes
    return written


def _write_int4_transposed(entry: dict, out_f, reader, codes) -> int:
    """Write an int4 payload as [K, N] from an [N, K] source, packed nibbles.

    Same strategy as `_write_quantized_transposed`, with one extra constraint:
    a byte holds TWO adjacent columns of the same k, so the unit written per k
    is `rows / 2` bytes at `(k * dst_n + offset) / 2`. Bands are whole runs of
    output columns and always start even (asserted), which is what keeps a
    nibble pair inside one band.
    """
    src_n, src_k = reader.k, reader.n          # file layout: [N, K]
    dst_k, dst_n = (int(d) for d in entry["dims"])
    if (dst_k, dst_n) != (src_k, src_n):
        raise DeferredWeightError(
            f"deferred weight '{entry['name']}': the side table promises "
            f"[{dst_k}, {dst_n}] but the source is [{src_n}, {src_k}]; a "
            f"transposed payload must be the same matrix the other way round")
    if dst_n % 2:
        raise DeferredWeightError(
            f"deferred weight '{entry['name']}': N={dst_n} is odd, so a packed "
            f"row would share a byte with the next one")

    out_f.flush()
    base = out_f.tell()
    pwrite = _pwrite_for(out_f)
    done = 0
    for offset, band in reader.bands(block=_TRANSPOSED_BAND_ROWS):
        rows = band.shape[0]
        if offset % 2:
            raise DeferredWeightError(
                f"deferred weight '{entry['name']}': band at column {offset} "
                f"starts on an odd column, so a nibble pair would straddle two "
                f"bands")
        q = codes(offset, band)
        if q.shape != (rows, src_k):
            raise DeferredWeightError(
                f"deferred weight '{entry['name']}': band {offset} encoded to "
                f"{q.shape}, expected {(rows, src_k)}")
        # [K, rows] -> [K, rows / 2] packed bytes, pairs inside the band.
        packed = _pack_nibbles(np.ascontiguousarray(q.T))
        for k in range(src_k):
            pwrite(out_f, packed[k].tobytes(), base + (k * dst_n + offset) // 2)
        done += rows
    if done != dst_n:
        raise DeferredWeightError(
            f"deferred weight '{entry['name']}': source bands covered {done} of "
            f"{dst_n} output columns")
    total = (dst_n * dst_k + 1) // 2
    out_f.seek(base + total)
    return total


def _write_quantized_transposed(entry: dict, out_f, base_dir: str,
                               dag_model=None) -> int:
    """Write a quantized [N, K] source as the [K, N] payload the side table says.

    The source band is [rows, K] with `rows` consecutive output columns, so one
    band fills a `rows`-byte slice of EVERY output row: for k in [0, K) the
    codes of column n at row k land at `k * N + n`. Reading stays sequential
    (one contiguous band), and each write is `rows` bytes -- 2 KiB at the
    default ROW_BLOCK -- instead of the one byte a naive element write would
    move.

    The file is positioned by absolute offset rather than by seeking the
    buffered writer per row: a 248320-column weight is ~10^6 row writes, and
    alternating `seek`/`write` on a buffered stream flushes the buffer each
    time. `os.pwrite` on the same descriptor leaves the stream's position where
    this function puts it at the end, so the writer's next payload still lands
    where its offsets were computed.
    """
    from . import quant_payload

    reader = quant_payload.WeightReader(
        entry, entry["name"],
        path_for=lambda e: source_read_path(e, out_f, base_dir))
    transposed = bool(entry.get("transposed"))
    if not transposed:
        raise DeferredWeightError(
            f"deferred weight '{entry['name']}' asks for a transposed payload "
            f"but its source is already [K, N]; only a [N, K] source can be "
            f"reordered into the other layout")

    scale = (scale_values_for(entry, dag_model) if dag_model is not None
             else entry.get("scale_values"))
    if scale is None:
        scale = quant_payload.scale_for(entry["quant"], reader, transposed)
    codes = quant_payload.code_byte(entry["quant"], reader, scale, transposed)

    src_n, src_k = reader.k, reader.n          # file layout: [N, K]
    dst_k, dst_n = (int(d) for d in entry["dims"])
    if (dst_k, dst_n) != (src_k, src_n):
        raise DeferredWeightError(
            f"deferred weight '{entry['name']}': the side table promises "
            f"[{dst_k}, {dst_n}] but the source is [{src_n}, {src_k}]; a "
            f"transposed payload must be the same matrix the other way round")

    out_f.flush()
    base = out_f.tell()
    pwrite = _pwrite_for(out_f)
    left = dst_n                       # output columns still to be written
    done = 0
    for offset, band in reader.bands(block=_TRANSPOSED_BAND_ROWS):
        rows = band.shape[0]
        q = codes(offset, band)
        if q.shape != (rows, src_k):
            raise DeferredWeightError(
                f"deferred weight '{entry['name']}': band {offset} quantized to "
                f"{q.shape}, expected {(rows, src_k)}")
        t = np.ascontiguousarray(q.T)  # [K, rows]
        for k in range(src_k):
            pwrite(out_f, t[k].tobytes(), base + k * dst_n + offset)
        done += rows
        left -= rows
    if left:
        raise DeferredWeightError(
            f"deferred weight '{entry['name']}': source bands covered {done} of "
            f"{dst_n} output columns")
    out_f.seek(base + dst_n * dst_k)
    return dst_n * dst_k


def _pwrite_for(stream):
    """A write-at-offset callable for `stream`, falling back to seek+write."""
    fileno = getattr(stream, "fileno", None)
    if fileno is not None and hasattr(os, "pwrite"):
        try:
            fd = fileno()
        except (OSError, ValueError):
            fd = None
        if fd is not None:
            def _at(stream, data, offset):
                view = memoryview(data)
                while view:
                    n = os.pwrite(fd, view, offset)
                    view = view[n:]
                    offset += n
            return _at

    def _seek_write(stream, data, offset):
        stream.seek(offset)
        stream.write(data)
    return _seek_write


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
        # `length` is the extent in the SOURCE file, in source bytes -- that is
        # what write_payload and WeightReader.bands seek and read by. A byte
        # format writes one byte per element out of a four-byte source, so its
        # payload is a quarter the size and the two must never be compared.
        # What is checked here is that the source extent still matches what the
        # entry promised; the payload size is derived at write time, not stored.
        #
        # A transposed payload (KnWeightLayout) is the one pass allowed to
        # change an entry's dims: it flips the LOGICAL shape to [K, N] while the
        # source stays [N, K], and records the file's layout in `src_dims`.
        # Checking against that is what keeps the promise ("the same bytes,
        # read in the same order") exact rather than waived -- the element
        # count it multiplies out to is still the numel `length` was sized for.
        quantized = e.get("quant") is not None
        got = payload_bytes(init, e.get("src_dtype") if quantized else None)
        want_dims = tuple(e.get("src_dims") or e["dims"])
        if ((not quantized and init.data_type != e["data_type"])
                or tuple(int(d) for d in init.dims) != want_dims
                or got != e["length"]):
            problems.append(
                f"'{name}' changed while deferred: dtype {e['data_type']}->{init.data_type}, "
                f"dims {list(want_dims)}->{list(init.dims)}, "
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
