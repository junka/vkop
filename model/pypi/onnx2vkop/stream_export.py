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

Two things a meta trace cannot do for itself, and the caller has to make good:

  * a dim that lives in a WEIGHT's shape resolves as a generated placeholder
    instead of a literal, because the exporter's shape inference reads the
    parameter tensors and a meta tensor has none. `resolve_shape_dims` recovers
    it after the splice, when the weights are real.
  * a parameter the exporter REWROTE before tracing (the zero-centred RMSNorm
    fold) still reads as its checkpoint value, because `add_` on a meta tensor
    writes nowhere. `splice_initializers(addend=...)` reapplies it to the bytes.

Both are silent failures -- the graph loads and runs, it just computes the
wrong thing -- so neither is caught by a byte-identity check on an ordinary
model. The regression tests at the end of tests/test_stream_export.py cover
them, and the note at the end of this file records how they were found.
"""
from __future__ import annotations

import json
import os

import numpy as np
import struct
from typing import Dict, Iterable, List, Mapping, Tuple

import torch
import torch.nn as nn
from torch.onnx import OperatorExportTypes
from torch.onnx._internal.torchscript_exporter import utils as exporter_utils

import onnx
import onnx.shape_inference
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

    def nearest(self, name: str, shape: list | None = None,
                prefix: str | None = None) -> str | None:
        """Resolve an exporter parameter name against the checkpoint's names.

        The exporter names a parameter after the module attribute
        (`layers.0.self_attn.q_proj.weight`); the checkpoint prefixes it
        (`model.language_model.`). Exact match first, then a SUFFIX match on a
        token boundary -- `norm.weight` must not match
        `layers.0.linear_attn.norm.weight`, which it would if the comparison
        ignored the dot before the suffix.

        `prefix` names the checkpoint subtree the traced module came from, and
        is consulted only when a suffix is ambiguous. Qwen3.5-27B ships an
        `mtp.*` speculative-decoding head whose leaf names AND shapes repeat the
        text tower's exactly, so `layers.0.mlp.gate_proj.weight` matches both
        `model.language_model.layers.0.mlp.gate_proj.weight` and
        `mtp.layers.0.mlp.gate_proj.weight` with nothing in the shape to
        separate them. The caller knows which subtree it traced; the shape does
        not.
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
        if len(hits) > 1 and prefix:
            # Keep the candidates inside the subtree the caller traced. A
            # namesake elsewhere in the checkpoint (an MTP head, a vision
            # tower) is a different module, not an alternative spelling.
            in_prefix = [k for k in hits if k.startswith(prefix)]
            if len(in_prefix) == 1:
                return in_prefix[0]
            if in_prefix:
                hits = in_prefix
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
                f"Pass a `prefix` naming the subtree that was traced.")
        return None


def build_graph_without_weights(model: nn.Module, args: Tuple, *,
                                input_names: List[str],
                                output_names: List[str],
                                opset: int = 17,
                                dynamic_axes: Mapping[str, Mapping[int, str]]
                                | None = None):
    """Stages 1: trace on meta and emit the proto with no initializer bytes.

    Not enough on its own for a real export -- it emits correct GRAPH STRUCTURE
    but its parameter inputs keep the exporter's generated names (`onnx::MatMul_43`),
    so nothing can be resolved against a checkpoint. A caller that needs the
    weights uses the per-component exporters' `trace_weightless`, which drives
    the same primitives and then renames each parameter input after its module
    attribute. This entry remains for structural/inventory use.

    `_trigger_symbolic_function_registration()` is called first, and it is not
    optional: without it the trace has no symbolic for `aten::unflatten` and
    dies with `UnsupportedOperatorError ... to ONNX opset version 20`. That
    "opset 20" is a red herring -- the message prints torch's default opset
    because the trace never got as far as the requested one; with the
    registration in place, opset 17 traces fine.

    `dynamic_axes` must be handed to the TRACE as well as to the write. torch
    consumes it in two places: `_model_to_graph` marks the traced inputs
    symbolic before the ONNX conversion pass, and `_export_onnx` records the
    names on the boundary. Passing it only to the second leaves the trace with
    literal shapes, and the two protos then differ STRUCTURALLY, not just in
    their dim declarations: measured on a 4 layer Qwen3.5, the meta trace came
    out with 14 extra `Cast` nodes and 13 missing `Identity` nodes against the
    eager export (827 vs 831 nodes). Those extra Casts sit on the attention
    mask's path, and a converter that fuses the mask-add/Softmax pattern into
    a single attention op then binds the wrong tensor for the bias.

    Returns (onnx.ModelProto, {param name: shape}).
    """
    exporter_utils._trigger_symbolic_function_registration()

    graph, params, _out = exporter_utils._model_to_graph(
        model, args, verbose=False,
        input_names=input_names, output_names=output_names,
        operator_export_type=OperatorExportTypes.ONNX,
        do_constant_folding=True,
        dynamic_axes=dynamic_axes)

    # export_params=False is the only path that never reads a payload: it hands
    # `_export_onnx` an empty parameter dict, so the weights come out as graph
    # inputs instead of initializers.
    raw, _export_map, _ext, _names = graph._export_onnx(
        {}, opset, dynamic_axes or {}, False, OperatorExportTypes.ONNX, True,
        False, {}, True, "", {})
    proto = onnx.ModelProto()
    proto.ParseFromString(raw)

    shapes = {k: list(v.shape) for k, v in params.items()}
    return proto, shapes


def splice_initializers(proto: onnx.ModelProto, shapes: Dict[str, list],
                        index: SafetensorsIndex, *,
                        weights_file: str = "llm.weights.bin",
                        target_dtype: str | None = None,
                        allow_missing: Iterable[str] = (),
                        addend: Mapping[str, float] | None = None,
                        prefix: str | None = None
                        ) -> Dict[str, dict]:
    """Stages 2: turn the weight graph-inputs into external initializers.

    Changes no bytes. Returns {initializer name: index entry} for stage 3.

    `addend` maps a parameter name (as the exporter names it, i.e. the graph
    input) to a constant that must be added to its stored value. It exists
    because the eager exporter rewrites some parameters on the LIVE model
    before tracing -- `Qwen3_5RMSNorm`'s zero-centred fold does
    `weight.data.add_(1.0)` so the graph can use a plain `out * w` -- and a
    meta parameter has no storage, so that write silently does nothing while
    the splice copies the checkpoint's raw bytes. The graph is therefore built
    for `1 + w` and fed `w`: on a checkpoint whose norms sit near zero the
    model normalises by 0 and every logit comes out 0.0. Passing the fold here
    reproduces it on the payload instead.

    `prefix` names the checkpoint subtree that was traced (`model.language_model.`
    for a text tower inside a multimodal checkpoint). It only breaks ties whose
    shape cannot: see `SafetensorsIndex.nearest`.
    """
    adds = dict(addend or {})
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
        hit = index.nearest(vi.name, shapes.get(vi.name), prefix)
        if hit is None:
            if vi.name in allowed:
                continue
            keep_inputs.append(vi)
            continue
        entry = dict(index.get(hit))
        entry["onnx_name"] = vi.name
        # A parameter the caller rewrote before tracing carries its offset into
        # the payload (see the `addend` note above).
        if vi.name in adds:
            entry["addend"] = float(adds[vi.name])
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

    An entry may also carry `addend` (see `splice_initializers`), a constant
    added to every element on the way out. Both a conversion and an addend have
    to see the whole tensor at once for the elementwise arithmetic, so those
    read that tensor in full; the streaming property still holds for the
    (common) untouched case and for the output file, which is written one
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
            if want_conv or entry.get("addend") or entry.get("transform"):
                blob = _convert_tensor(entry, target_dtype or
                                       _ONNX_NAME_BY_ST[entry["dtype"]],
                                       addend=entry.get("addend", 0.0),
                                       transform=entry.get("transform"))
                dst.write(blob)
                written += len(blob)
                entry["target_dtype"] = target_dtype or entry["dtype"]
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


def _convert_tensor(entry: dict, target: str,
                    addend: float = 0.0, transform=None) -> bytes:
    """Read one tensor, optionally offset it, and re-encode it as `target`.

    `addend` is added in fp32 before the re-encode. It exists because a few
    parameters are not used as stored: the exporter rewrites them on the live
    model before tracing (see `splice_initializers` for the case that matters),
    and a meta model cannot hold that rewrite.

    `transform` is the general form of the same thing: a callable taking the
    fp32 tensor (checkpoint shape) and returning one of the same shape, e.g.
    a Hadamard rotation of a linear's input dim (MiniMax-H3 `--convrot`).
    """
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
    if addend:
        arr = arr.astype(np.float32) + addend
    if transform is not None:
        shape = [int(d) for d in entry["shape"]]
        arr = np.asarray(transform(arr.astype(np.float32).reshape(shape)),
                         dtype=np.float32)
        if list(arr.shape) != shape:
            raise ValueError(f"{entry['onnx_name']}: transform changed shape "
                             f"{shape} -> {list(arr.shape)}")
        arr = arr.reshape(-1)
    if target == "F16":
        with np.errstate(over="ignore"):
            return np.ascontiguousarray(arr.astype(np.float16)).tobytes()
    if target == "F32":
        return np.ascontiguousarray(arr.astype(np.float32)).tobytes()
    if target == "BF16":
        # Round to nearest even, like the F16 branch above (numpy's cast) and
        # every bf16 encode in the converter. `u32 >> 16` truncated instead:
        # a biased, up-to-1-ulp error on every fp32/fp16 -> bf16 conversion.
        # A bf16 source is unaffected (its low half is zero, so RNE is exact).
        from .bf16_weights import fp32_to_bf16_bits
        return np.ascontiguousarray(
            fp32_to_bf16_bits(arr.astype(np.float32))).astype("<u2").tobytes()
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


def resolve_shape_dims(proto: onnx.ModelProto,
                       dynamic_axes: Mapping[str, Mapping[int, str]] | None = None
                       ) -> int:
    """Recover the dims the meta trace could not prove, keeping symbols symbolic.

    A dim that comes from a weight's SHAPE -- the vocab width off lm_head, the
    key/value head counts off the qkv projections -- is resolved by the
    exporter inside its own shape-inference pass, which reads the parameter
    tensors. On the meta device those tensors have no storage, so the pass has
    nothing to propagate and the dim surfaces as a generated placeholder
    (`MatMullogits_dim_2`) where the eager export has `512`. Measured on a 4
    layer Qwen3.5: 317 of 831 traced values carry a known shape under meta
    against 505 under real weights, while the two protos are otherwise
    node-for-node identical (same ops, same inline constants, same parameter
    declarations).

    By the time this runs the weights are spliced in as real initializers, so
    the information is present again and a standard ONNX inference pass can
    resolve it. That pass also collapses every genuinely symbolic axis to the
    literal length it can prove from the dummy inputs -- the freeze this module
    exists to avoid -- so each axis the caller declared dynamic is written back
    afterwards. `dynamic_axes` takes the same shape as `torch.onnx.export`'s.

    Call this LAST, after any pass that declares shapes for the custom ops: a
    dim behind a `Conv1d`/`LinearAttention` output cannot be inferred until
    that output's shape is declared, and those declarations are architecture
    knowledge the caller adds after the splice.

    `data_prop` is left OFF on purpose. The dims this recovers come from a
    weight's declared shape, which is metadata -- value propagation is not
    needed to reach them, and it is the part that would materialize a
    payload. On a 52 GB checkpoint that difference is the whole budget.

    Only a literal (or a declared symbol) is ever accepted; a placeholder is
    left alone if inference cannot do better, so an unfixable dim is no worse
    off than before. Returns the number of axes changed.
    """
    intended: Dict[Tuple[str, int], str] = {}
    for name, axes in (dynamic_axes or {}).items():
        for axis, symbol in axes.items():
            intended[(name, int(axis))] = symbol

    inferred = onnx.shape_inference.infer_shapes(
        proto, strict_mode=False, data_prop=False)
    inferred_by_name = {v.name: v for v in list(inferred.graph.input)
                        + list(inferred.graph.output)}

    changed = 0
    for vi in list(proto.graph.input) + list(proto.graph.output):
        src = inferred_by_name.get(vi.name)
        if src is None:
            continue
        dims = vi.type.tensor_type.shape.dim
        if len(dims) != len(src.type.tensor_type.shape.dim):
            continue
        for axis, dim in enumerate(dims):
            want = intended.get((vi.name, axis))
            if want is not None:
                # A declared axis stays symbolic whatever inference proved.
                if dim.dim_param != want or dim.dim_value:
                    dim.ClearField("dim_value")
                    dim.dim_param = want
                    changed += 1
                continue
            got = src.type.tensor_type.shape.dim[axis]
            if got.dim_value and not dim.dim_value:
                # Prefer a literal inference can prove over a placeholder.
                dim.ClearField("dim_param")
                dim.dim_value = got.dim_value
                changed += 1
    return changed


def export_streaming(model_dir: str, build_fn, out_dir: str, *,
                     model_file: str = "llm.onnx",
                     weights_file: str = "llm.weights.bin",
                     opset: int = 17,
                     dynamic_axes: Mapping[str, Mapping[int, str]] | None = None,
                     addend: Mapping[str, float] | None = None,
                     prefix: str | None = None
                     ) -> Tuple[str, str]:
    """Run all three stages.

    `build_fn()` must return (meta_model, args, input_names, output_names); it
    is a callback because the model construction is architecture-specific and
    belongs to the caller's exporter.

    `dynamic_axes` names the axes that must stay symbolic (the same mapping
    `torch.onnx.export` takes). It is applied twice, and both matter: on the
    trace, so the meta graph is structurally the eager one, and again after the
    splice, where the weights are real initializers and shape inference can
    finally prove the dims the meta trace could not -- see `resolve_shape_dims`.

    `addend` maps a graph-input name to a constant the payload must carry.
    Pass `{name: 1.0}` for any parameter the caller rewrote before tracing; see
    `splice_initializers` for why a meta model cannot hold that rewrite.

    `prefix` names the checkpoint subtree that was traced, for a checkpoint
    that carries more than one copy of the module (an MTP head, a vision
    tower). See `SafetensorsIndex.nearest`.
    """
    os.makedirs(out_dir, exist_ok=True)
    index = SafetensorsIndex(model_dir)
    print(f"[stream] {len(index.entries)} tensor(s) across "
          f"{len(index.shards)} shard(s), headers only", flush=True)

    meta_model, args, in_names, out_names = build_fn()
    print("[stream] tracing on meta (no weight bytes)", flush=True)
    proto, shapes = build_graph_without_weights(
        meta_model, args, input_names=in_names, output_names=out_names,
        opset=opset, dynamic_axes=dynamic_axes)
    print(f"[stream] graph: {len(proto.graph.node)} node(s)", flush=True)

    resolved = splice_initializers(proto, shapes, index,
                                   weights_file=weights_file,
                                   addend=addend, prefix=prefix)
    print(f"[stream] {len(resolved)} weight(s) spliced as external refs",
          flush=True)

    wpath = os.path.join(out_dir, weights_file)
    nbytes, offsets = copy_weights(resolved, wpath)
    patch_offsets(proto, offsets)

    # Do this AFTER the offsets exist: inference has to read the weights to
    # resolve the shape-derived dims this pass is here to recover.
    fixed = resolve_shape_dims(proto, dynamic_axes)
    print(f"[stream] resolved {fixed} dim(s) from the spliced weights",
          flush=True)

    mpath = os.path.join(out_dir, model_file)
    onnx.save(proto, mpath)
    print(f"[stream] wrote {mpath} ({os.path.getsize(mpath)/2**20:.1f} MiB) "
          f"and {wpath} ({nbytes/2**30:.1f} GiB)", flush=True)
    return mpath, wpath


# ---------------------------------------------------------------------------
# HOW THE META PATH DIFFERS FROM THE EAGER ONE (three findings)
#
# 1. A dim that lives in a WEIGHT's SHAPE does not resolve.
#
#    eager  : logits dim 2 = 512                    (lm_head's out_features)
#    stream : logits dim 2 = 'MatMullogits_dim_2'   (a generated placeholder)
#    eager  : present_key_values_3 = [B, 2, 2, kv, 32]
#    stream : present_key_values_3 = [B, Concat..., Concat..., kv, Concat...]
#
#    Note the SEQUENCE axis is NOT what breaks -- `seq` and `kv_len` come
#    through on both sides. What breaks is every dim whose value comes from a
#    parameter. Measured on a 4 layer Qwen3.5, holding everything else equal:
#
#      * real weights vs meta weights: identical node sequences (831 each),
#        identical inline constants (236 each), identical parameter
#        declarations. Only the resolved dims differ.
#      * the traced values: 505 of 831 carry a known shape under real weights,
#        317 under meta.
#      * ruled out by experiment, one at a time: do_constant_folding (818 vs
#        818), an empty vs populated initializer dict, passing dynamic_axes to
#        `_model_to_graph` (changes node count 827 -> 831, resolves nothing),
#        and the GLOBALS settings (export_onnx_opset_version, in_onnx_export,
#        _trigger_symbolic_function_registration).
#
#    The mechanism: the exporter resolves such a dim inside
#    `_jit_pass_onnx_graph_shape_type_inference`, which propagates from the
#    PARAMETER TENSORS. A meta parameter has a shape but no storage, so the
#    propagation stalls. No calling convention fixes that -- the value was
#    never computed. It IS recoverable after the splice, when the weights are
#    real initializers, which is what `resolve_shape_dims` does. A blanket
#    inference pass would also collapse every genuinely symbolic axis to the
#    dummy length, so it is told which axes are dynamic and writes them back.
#
# 2. `dynamic_axes` has to reach the TRACE, not just the write.
#
#    torch consumes it twice: `_model_to_graph` marks the traced inputs
#    symbolic before the ONNX conversion pass, and `_export_onnx` records the
#    names on the boundary. Skipping the first leaves the meta graph
#    STRUCTURALLY different from eager -- 827 vs 831 nodes, 14 extra `Cast`
#    and 13 missing `Identity` -- and those extra Casts sit on the attention
#    mask's path, so the converter's mask-add/Softmax fusion binds the wrong
#    tensor and the runtime rejects the bias ("got 2 elements, expected 182").
#
# 3. A parameter the exporter REWROTE before tracing still reads as its
#    checkpoint value, and a 27B checkpoint carries namesakes.
#
#    `fold_zero_centered_norms` does `weight.data.add_(1.0)` on the live model
#    so the graph can use a plain `out * w`. On a meta parameter that write is
#    a silent no-op (no storage), so the graph is built for `1 + w` and fed the
#    raw `w`: on the tiny model, whose norms are all 0.0, every logit came out
#    exactly 0.0. `splice_initializers(addend=...)` reapplies the fold to the
#    bytes. Separately, the real checkpoint ships an `mtp.*` speculative head
#    whose parameter names AND shapes repeat the text tower's, so suffix
#    matching alone is ambiguous and the export aborts; `prefix` names the
#    subtree that was traced.
#
# All three are silent or late failures -- 1 and 3 produce a graph that loads
# and runs while computing the wrong thing -- so none is caught by a
# byte-identity check on an ordinary model. See the regression tests at the end
# of tests/test_stream_export.py.
# ---------------------------------------------------------------------------
