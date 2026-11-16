#!/usr/bin/env python3
"""Stream a Qwen3.5 checkpoint into ONNX without ever holding the weights.

`torch.onnx.export` needs every parameter resident: it copies each one into the
protobuf's initializer table. On the 27B model that is ~50 GB peak on a 62 GB
box, and the two-copy path (materialize the checkpoint, then copy into the
module tree) pushed 5 GB into swap. This module removes that requirement.

How it works -- three stages, only the middle one touches weight bytes:

  1. BUILD (no weights). The model is instantiated on the meta device, so every
     parameter carries a shape and a dtype but no storage. `_model_to_graph`
     traces it normally; `Graph._export_onnx` is then called with
     `export_params=False`, which emits the graph with the weights as *graph
     inputs* and an empty initializer table. Peak: a few hundred MB of graph
     structure.

  2. SPLICE (no weights). Each parameter input is turned back into an
     initializer that points at an external file, with the byte offset and
     length looked up in the checkpoint's own safetensors headers. Still no
     weight bytes are read.

  3. COPY (one tensor at a time). For each initializer, a bounded chunk is read
     from its shard and appended to the external data file. Peak: one chunk.

The result is byte-for-byte the ONNX the eager exporter would have produced,
which the caller verifies against a reference export when one is available.

Why the weights become graph inputs in stage 1: the exporter's parameter
handling is all-or-nothing. `defer_weight_export` still copies (it defers only
the *writing*), and a meta parameter fails at the first `tensor.cpu().numpy()`.
`export_params=False` is the one path that never looks at a payload.
"""
from __future__ import annotations

import json
import os

import numpy as np
import struct
from typing import Dict, Iterable, List, Tuple

import torch
import torch.nn as nn
from torch.onnx import OperatorExportTypes
from torch.onnx._internal.torchscript_exporter import utils as exporter_utils

import onnx
from onnx import numpy_helper

# Read this many bytes per copy step. Bounded so a single large tensor never
# materializes: the peak is this, not the tensor.
CHUNK = 8 << 20

# safetensors dtype name -> (onnx TensorProto dtype, bytes per element)
_ST_DTYPES = {
    "F16": (onnx.TensorProto.FLOAT16, 2),
    "BF16": (onnx.TensorProto.BFLOAT16, 2),
    "F32": (onnx.TensorProto.FLOAT, 4),
    "F64": (onnx.TensorProto.DOUBLE, 8),
    "I8": (onnx.TensorProto.INT8, 1),
    "U8": (onnx.TensorProto.UINT8, 1),
    "I32": (onnx.TensorProto.INT32, 4),
    "I64": (onnx.TensorProto.INT64, 8),
    "BOOL": (onnx.TensorProto.BOOL, 1),
}


_ONNX_NAME_BY_ST = {"F32": "F32", "F16": "F16", "BF16": "BF16",
                    "I64": "I64", "I32": "I32", "I8": "I8", "U8": "U8"}
_ONNX_BY_NAME = {"F32": onnx.TensorProto.FLOAT,
                 "F16": onnx.TensorProto.FLOAT16,
                 "BF16": onnx.TensorProto.BFLOAT16,
                 "I64": onnx.TensorProto.INT64,
                 "I32": onnx.TensorProto.INT32,
                 "I8": onnx.TensorProto.INT8,
                 "U8": onnx.TensorProto.UINT8}
_ONNX_ELEM_BYTES = {"F32": 4, "F16": 2, "BF16": 2, "I64": 8, "I32": 4,
                    "I8": 1, "U8": 1}


class SafetensorsIndex:
    """name -> (shard path, byte offset, nbytes, dtype, shape) from the headers.

    A safetensors file starts with a little-endian u64 header length, then a
    JSON header mapping each tensor name to its `data_offsets` (relative to the
    end of the header) and its dtype/shape. Reading that is a seek and a few
    hundred KB per shard -- the weights themselves are never touched.
    """

    def __init__(self, model_dir: str):
        self.model_dir = model_dir
        self.entries: Dict[str, dict] = {}
        shards = sorted(
            f for f in os.listdir(model_dir) if f.endswith(".safetensors"))
        if not shards:
            raise FileNotFoundError(f"no *.safetensors in {model_dir}")
        for shard in shards:
            path = os.path.join(model_dir, shard)
            with open(path, "rb") as f:
                raw = f.read(8)
                if len(raw) != 8:
                    raise ValueError(f"{shard}: truncated header length")
                hlen = struct.unpack("<Q", raw)[0]
                header = json.loads(f.read(hlen))
            base = 8 + hlen
            for name, meta in header.items():
                if name == "__metadata__":
                    continue
                begin, end = meta["data_offsets"]
                self.entries[name] = {
                    "shard": path,
                    "offset": base + begin,
                    "nbytes": end - begin,
                    "dtype": meta["dtype"],
                    "shape": list(meta["shape"]),
                }
        self.shards = shards

    def __contains__(self, name: str) -> bool:
        return name in self.entries

    def get(self, name: str) -> dict:
        return self.entries[name]

    def nearest(self, name: str, shape: list | None = None) -> str | None:
        """Resolve an exporter parameter name against the checkpoint's names.

        The exporter names a parameter after the module attribute
        (`layers.0.self_attn.q_proj.weight`); the checkpoint prefixes it
        (`model.language_model.`). Exact match first, then a SUFFIX match on a
        token boundary -- `norm.weight` must not match
        `layers.0.linear_attn.norm.weight`, which it would if the comparison
        ignored the dot before the suffix.
        """
        if name in self.entries:
            return name
        # Boundary-anchored suffix: ".layers.0.q_proj.weight" or the name
        # itself, never a trailing fragment of a longer attribute path.
        want = "." + name
        hits = [k for k in self.entries
                if k == name or k.endswith(want)]
        if len(hits) == 1:
            return hits[0]
        if len(hits) > 1 and shape is not None:
            # The traced parameter's SHAPE disambiguates a common leaf name:
            # `norm.weight` is the final [hidden] norm, while
            # `layers.N.linear_attn.norm.weight` is [head_v_dim]. Keeping only
            # the shape-compatible candidates is a real distinction, not a
            # guess.
            want_shape = [int(d) for d in shape]
            hits = [k for k in hits
                    if [int(d) for d in self.entries[k]["shape"]] == want_shape]
        if len(hits) == 1:
            return hits[0]
        if len(hits) > 1:
            raise ValueError(
                f"'{name}' (shape {shape}) is ambiguous in the checkpoint: "
                f"{sorted(hits)[:4]}{' ...' if len(hits) > 4 else ''}. "
                f"Rename the parameter or convert with the eager exporter.")
        return None


def build_graph_without_weights(model: nn.Module, args: Tuple, *,
                                input_names: List[str],
                                output_names: List[str],
                                opset: int = 17):
    """Stages 1: trace on meta and emit the proto with no initializer bytes.

    Returns (onnx.ModelProto, {param name: shape}).
    """
    graph, params, _out = exporter_utils._model_to_graph(
        model, args, verbose=False,
        input_names=input_names, output_names=output_names,
        operator_export_type=OperatorExportTypes.ONNX,
        do_constant_folding=True)

    # export_params=False is the only path that never reads a payload: it hands
    # `_export_onnx` an empty parameter dict, so the weights come out as graph
    # inputs instead of initializers.
    raw, _export_map, _ext, _names = graph._export_onnx(
        {}, opset, {}, False, OperatorExportTypes.ONNX, True,
        False, {}, True, "", {})
    proto = onnx.ModelProto()
    proto.ParseFromString(raw)

    shapes = {k: list(v.shape) for k, v in params.items()}
    return proto, shapes


def splice_initializers(proto: onnx.ModelProto, shapes: Dict[str, list],
                        index: SafetensorsIndex, *,
                        weights_file: str = "llm.weights.bin",
                        target_dtype: str | None = None,
                        allow_missing: Iterable[str] = ()) -> Dict[str, dict]:
    """Stages 2: turn the weight graph-inputs into external initializers.

    Changes no bytes. Returns {initializer name: index entry} for stage 3.
    """
    allowed = set(allow_missing)
    # The exporter names a parameter after its module attribute
    # (`layers.0.input_layernorm.weight`); the checkpoint prefixes it
    # (`model.language_model.`), which `nearest()` resolves by suffix. A
    # constant the tracer inlined (an anonymous `onnx::Conv1d_904`) is NOT in
    # `shapes` -- those are the ones this pass must skip rather than guess at.
    by_name = {vi.name: vi for vi in proto.graph.input}
    resolved: Dict[str, dict] = {}
    keep_inputs = []
    for vi in proto.graph.input:
        if vi.name not in shapes:
            keep_inputs.append(vi)
            continue
        hit = index.nearest(vi.name, shapes.get(vi.name))
        if hit is None:
            if vi.name in allowed:
                continue
            keep_inputs.append(vi)
            continue
        entry = dict(index.get(hit))
        entry["onnx_name"] = vi.name
        # The exporter's shape comes from the module, the header's from the
        # checkpoint. They must agree, or the splice would wire a tensor with
        # one shape into a node that expects another.
        declared = list(shapes.get(vi.name, []))
        if declared and list(entry["shape"]) != declared:
            raise ValueError(
                f"{vi.name}: checkpoint shape {entry['shape']} != traced "
                f"{declared}")
        # The declaration follows what the payload WILL be, after the optional
        # conversion in copy_weights. `target_dtype` is an ONNX dtype name.
        src_dtype_name = _ONNX_NAME_BY_ST[entry["dtype"]]
        onnx_dtype_name = target_dtype or src_dtype_name
        onnx_dtype = _ONNX_BY_NAME[onnx_dtype_name]
        elem_size = _ONNX_ELEM_BYTES[onnx_dtype_name]
        n_elems = 1
        for d in entry["shape"]:
            n_elems *= int(d)
        entry["payload_nbytes"] = n_elems * elem_size
        init = proto.graph.initializer.add()
        init.name = vi.name
        init.data_type = onnx_dtype
        init.dims.extend(entry["shape"])
        init.data_location = onnx.TensorProto.EXTERNAL
        kv = init.external_data.add()
        kv.key = "location"
        kv.value = weights_file
        kv = init.external_data.add()
        kv.key = "offset"
        kv.value = "0"          # filled in stage 3
        kv = init.external_data.add()
        kv.key = "length"
        kv.value = str(entry["payload_nbytes"])
        resolved[vi.name] = entry

    del proto.graph.input[:]
    proto.graph.input.extend(keep_inputs)
    return resolved


def copy_weights(resolved: Dict[str, dict], out_path: str, *,
                 target_dtype: str | None = None) -> Tuple[int, Dict[str, int]]:
    """Stage 3: append each tensor's bytes, one bounded chunk at a time.

    `target_dtype` names an ONNX dtype the payload must be CONVERTED to -- the
    case that matters is an fp32 checkpoint exported into an fp16 graph, which
    is what `from_pretrained(torch_dtype=torch.float16)` does implicitly on the
    eager path. Without it the spliced weights stay fp32 while every activation
    around them is fp16, and the model computes garbage (measured: all-NaN
    logits).

    A conversion has to see the whole tensor at once for the elementwise cast,
    so it reads that tensor in full; the streaming property still holds for the
    (common) no-conversion case and for the output file, which is written one
    chunk at a time either way.

    Returns (bytes written, {name: offset}).
    """
    offsets: Dict[str, int] = {}
    written = 0
    with open(out_path, "wb") as dst:
        for name, entry in resolved.items():
            offsets[name] = written
            want_conv = (target_dtype is not None
                         and entry["dtype"] != target_dtype)
            if want_conv:
                blob = _convert_tensor(entry, target_dtype)
                dst.write(blob)
                written += len(blob)
                entry["target_dtype"] = target_dtype
                continue
            with open(entry["shard"], "rb") as src:
                src.seek(entry["offset"])
                remaining = entry["nbytes"]
                while remaining:
                    block = src.read(min(CHUNK, remaining))
                    if not block:
                        raise IOError(
                            f"{name}: shard ended {remaining} byte(s) early")
                    dst.write(block)
                    remaining -= len(block)
                    written += len(block)
    return written, offsets


_ST_NP = {"F32": np.float32, "F16": np.float16, "BF16": None,
          "I64": np.int64, "I32": np.int32, "I8": np.int8, "U8": np.uint8}


def _convert_tensor(entry: dict, target: str) -> bytes:
    """Read one tensor and re-encode it as `target` (an ONNX dtype name)."""
    src_np = _ST_NP.get(entry["dtype"])
    if src_np is None:
        if entry["dtype"] == "BF16":
            # bf16 is fp32 with the low 16 mantissa bits dropped.
            raw = _read_all(entry)
            u16 = np.frombuffer(raw, dtype=np.uint16)
            src_np = np.float32
            arr = (u16.astype(np.uint32) << 16).view(np.float32)
        else:
            raise ValueError(f"{entry['onnx_name']}: unsupported dtype "
                             f"{entry['dtype']}")
    else:
        arr = np.frombuffer(_read_all(entry), dtype=src_np)
    if target == "F16":
        with np.errstate(over="ignore"):
            return np.ascontiguousarray(arr.astype(np.float16)).tobytes()
    if target == "F32":
        return np.ascontiguousarray(arr.astype(np.float32)).tobytes()
    if target == "BF16":
        f32 = arr.astype(np.float32)
        u32 = f32.view(np.uint32)
        return ((u32 >> 16).astype(np.uint16)).tobytes()
    raise ValueError(f"no conversion to {target}")


def _read_all(entry: dict) -> bytes:
    with open(entry["shard"], "rb") as f:
        f.seek(entry["offset"])
        return f.read(entry["nbytes"])


def patch_offsets(proto: onnx.ModelProto, offsets: Dict[str, int]) -> None:
    for init in proto.graph.initializer:
        if init.data_location != onnx.TensorProto.EXTERNAL:
            continue
        for kv in init.external_data:
            if kv.key == "offset":
                kv.value = str(offsets[init.name])


def export_streaming(model_dir: str, build_fn, out_dir: str, *,
                     model_file: str = "llm.onnx",
                     weights_file: str = "llm.weights.bin",
                     opset: int = 17) -> Tuple[str, str]:
    """Run all three stages.

    `build_fn()` must return (meta_model, args, input_names, output_names); it
    is a callback because the model construction is architecture-specific and
    belongs to the caller's exporter.
    """
    os.makedirs(out_dir, exist_ok=True)
    index = SafetensorsIndex(model_dir)
    print(f"[stream] {len(index.entries)} tensor(s) across "
          f"{len(index.shards)} shard(s), headers only", flush=True)

    meta_model, args, in_names, out_names = build_fn()
    print("[stream] tracing on meta (no weight bytes)", flush=True)
    proto, shapes = build_graph_without_weights(
        meta_model, args, input_names=in_names, output_names=out_names,
        opset=opset)
    print(f"[stream] graph: {len(proto.graph.node)} node(s)", flush=True)

    resolved = splice_initializers(proto, shapes, index,
                                   weights_file=weights_file)
    print(f"[stream] {len(resolved)} weight(s) spliced as external refs",
          flush=True)

    wpath = os.path.join(out_dir, weights_file)
    nbytes, offsets = copy_weights(resolved, wpath)
    patch_offsets(proto, offsets)

    mpath = os.path.join(out_dir, model_file)
    onnx.save(proto, mpath)
    print(f"[stream] wrote {mpath} ({os.path.getsize(mpath)/2**20:.1f} MiB) "
          f"and {wpath} ({nbytes/2**30:.1f} GiB)", flush=True)
    return mpath, wpath


# ---------------------------------------------------------------------------
# KNOWN LIMITATION: intermediate tensors lose their symbolic dims
#
# The three stages above are verified for a plain nn.Module: a small two-layer
# MLP exported both ways is byte-identical, and onnxruntime agrees to 0.0.
#
# On qwen3_5 the graph comes out structurally right -- 56 initializers, the 10
# real inputs, weights spliced and converted to fp16 -- but the SEQUENCE axis
# is frozen at the export-time dummy length on INTERMEDIATE tensors:
#
#     eager  : K/V dims = [-1, 2, -1, 32]     seq is symbolic
#     stream : K/V dims = [-1, 2,  8, 32]     seq is the dummy length (8)
#
# The graph inputs and outputs keep their symbols; it is only the tensors in
# between that lose them. At run time the Attention op then sizes its additive
# bias from that frozen 8 (q=8, kv=9 -> 72 elements instead of 2), and refuses
# the real input.
#
# Cause: `torch.onnx.export` propagates `dynamic_axes` while it BUILDS the
# graph, inside `_optimize_graph(..., dynamic_axes=..., module=...)`. Calling
# `_model_to_graph` and then `graph._export_onnx(...)` separately passes the
# same dynamic_axes to the second half, but by then the shapes are already
# baked into the TorchScript IR.
#
# Next step, in order of preference:
#   1. copy the eager path's full call sequence (_jit_pass_onnx_deduplicate_
#      initializers, _jit_pass_onnx_assign_scoped_names_for_node_and_value,
#      and the module= argument to _optimize_graph), swapping only the final
#      payload write for export_params=False;
#   2. or run the ONNX shape-inference pass with symbolic inputs afterwards,
#      which cannot restore a dim that is already a literal;
#   3. or accept it for models whose prefill and decode lengths are fixed.
#
# Until then the eager exporter (with low_cpu_mem_usage) is what ships.
# ---------------------------------------------------------------------------
