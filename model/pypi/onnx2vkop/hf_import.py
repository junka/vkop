"""Convert a Hugging Face safetensors checkpoint without an ONNX copy of it.

`onnx2vkop --hf-checkpoint DIR` lands here. The converter's passes, quantizers
and audits all speak ONNX, so this module does not replace that input; it makes
the ONNX it hands them *graph-only*:

  1. BUILD + TRACE on `meta`. The architecture is instantiated from
     config.json with no storage, wrapped in a builder (see `GraphSpec`), and
     traced with `export_params=False`. Every parameter comes out as a NAMED
     graph input (`layers.0.self_attn.q_proj.weight`), never as bytes.

  2. SPLICE TO SHARDS. Each parameter input becomes an initializer whose
     `external_data` points straight into the checkpoint's own *.safetensors
     shard: `location` = the shard (relative to the ONNX file), `offset` = the
     tensor's byte offset from the shard's header, `length` = its size. No
     weight byte is read or written. The result is a few MB of ONNX next to an
     untouched checkpoint.

  3. CONVERT. The usual `ModelConverter` pipeline loads that ONNX with
     `load_external_data=False`; `weight_stream` then either streams each
     pass-through weight from its shard into the .vkopbin trailer or hydrates
     it one tensor at a time -- exactly as it does for an exporter's
     `llm.weights.bin`, because to it a shard is just another external file.

Compared to `stream_export.export_streaming` (which this reuses) the one
difference is stage 3 of that module, `copy_weights`, which is gone: there is
no intermediate `*.weights.bin`, so a 50 GB checkpoint needs 50 GB of disk, not
100.

Dtypes. A tensor is referenced in place when the checkpoint stores it at the
width the graph was traced at, or when it is BF16 -- the converter's
`_downcast_bfloat16` / `weight_stream` already turn a bf16 initializer into
whatever the graph needs (fp16, or fp32 for an fp32 consumer), chunk-wise for
a streamed one. Anything else (an fp32 checkpoint under an fp16 trace, a
parameter that needs an `addend`) is converted one tensor at a time into a
small side file next to the ONNX and referenced there. The trace width
defaults to fp16 (what every in-tree LLM exporter emits), or fp32 under
`--act bf16/fp32` or for an fp32 checkpoint; see `pick_trace_dtype`.

Builders. A checkpoint carries weights, not a graph, so something has to say
what the graph is. That is a builder: `fn(config, torch_dtype, index) ->
GraphSpec`. The in-tree ones live in `onnx2vkop/hf_builders/`, one module per
architecture, and are picked by config.json's `model_type` (see
`hf_builders.REGISTRY`); each module's `make_wrapper(model)` is ALSO what the
matching llm/exporter/export/*_export_onnx.py script exports from, so the
real-weight export and the meta trace are one implementation.
`--hf-builder NAME` forces a registered one, `--hf-builder pkg.mod:fn` (or
`file.py:fn`) plugs in another.

torch and transformers are imported lazily: `-i model.onnx` conversions keep
working without them.
"""
from __future__ import annotations

import dataclasses
import importlib
import json
import os
import sys
from typing import Callable, Dict, List, Mapping, Optional, Tuple

import numpy as np
import onnx
from onnx import TensorProto, numpy_helper

OPSET = 17

# torch dtype name -> safetensors/stream_export dtype name
_TORCH_TO_ST = {"float16": "F16", "bfloat16": "BF16", "float32": "F32",
                "int64": "I64", "int32": "I32", "int8": "I8", "uint8": "U8"}
_ST_TO_ONNX = {"F16": TensorProto.FLOAT16, "BF16": TensorProto.BFLOAT16,
               "F32": TensorProto.FLOAT, "F64": TensorProto.DOUBLE,
               "I64": TensorProto.INT64, "I32": TensorProto.INT32,
               "I8": TensorProto.INT8, "U8": TensorProto.UINT8,
               "BOOL": TensorProto.BOOL}


class HFImportError(RuntimeError):
    """The checkpoint cannot be turned into a graph by the chosen builder."""


@dataclasses.dataclass
class GraphSpec:
    """What a builder hands the tracer.

    `wrapper` holds its parameters on the meta device; its state-dict names
    are what the splice resolves against the checkpoint (exact name first,
    then a dot-anchored suffix match, see `SafetensorsIndex.nearest`).

    `constants` maps a wrapper parameter/buffer name to the value it must
    carry when that value is COMPUTED rather than stored (a RoPE `inv_freq`).
    It becomes a small inline initializer instead of a checkpoint reference.

    `aliases` maps a wrapper name to the checkpoint name to read it from when
    the checkpoint spells it differently (a tied `lm_head.weight` that only
    exists as `embed_tokens.weight`).

    `ckpt_prefix` names the checkpoint subtree that was traced; it only breaks
    ties a suffix and shape cannot (an `mtp.*` head repeating the text tower).

    `addend` maps a wrapper name to a constant the payload must carry on top
    of the stored value (a zero-centred norm folded to `1 + w`). Such a tensor
    is rewritten into the side file.
    """
    wrapper: object
    dummies: Tuple
    input_names: List[str]
    output_names: List[str]
    dynamic_axes: Dict[str, Dict[int, str]]
    constants: Dict[str, np.ndarray] = dataclasses.field(default_factory=dict)
    aliases: Dict[str, str] = dataclasses.field(default_factory=dict)
    ckpt_prefix: Optional[str] = None
    addend: Dict[str, float] = dataclasses.field(default_factory=dict)
    # Called with the ModelProto after the splice and before the shape repair
    # and save: architecture knowledge the trace cannot carry (value_info for
    # a custom op's outputs, metadata_props the runtime reads).
    post_splice: Optional[Callable] = None
    # Free-form facts the caller prints (layers, heads, ...).
    summary: Dict[str, object] = dataclasses.field(default_factory=dict)


# ---------------------------------------------------------------------------
# checkpoint inspection (headers only)
# ---------------------------------------------------------------------------

def load_config(ckpt_dir: str) -> dict:
    path = os.path.join(ckpt_dir, "config.json")
    if not os.path.isfile(path):
        raise HFImportError(f"{ckpt_dir} has no config.json; --hf-checkpoint "
                            "expects a Hugging Face model directory")
    with open(path) as f:
        return json.load(f)


def float_dtype_census(index) -> Dict[str, int]:
    """Bytes per floating dtype across the checkpoint (headers only)."""
    out: Dict[str, int] = {}
    for e in index.entries.values():
        if e["dtype"] in ("F16", "BF16", "F32", "F64"):
            out[e["dtype"]] = out.get(e["dtype"], 0) + e["nbytes"]
    return out


def pick_trace_dtype(requested: str, act: Optional[str], index) -> str:
    """The width the graph is traced at: "fp16" or "fp32".

    fp16 is what the in-tree LLM exporters emit and what the converter's FP16
    domain expects to find. fp32 is required under `--act bf16` (the bf16
    domain pass only rewrites FLOAT -- an fp16-traced graph would come out
    mixed fp16/bf16, see `converter._downcast_bfloat16`) and is the
    conventional input under `--act fp32`. An fp32 checkpoint is also traced
    fp32, so its weights can be referenced in place and the converter does the
    downcast, instead of this module copying the whole model into a side file.
    """
    if requested in ("fp16", "fp32"):
        return requested
    if act in ("bf16", "fp32"):
        return "fp32"
    census = float_dtype_census(index)
    if census and max(census, key=census.get) in ("F32", "F64"):
        return "fp32"
    return "fp16"


# ---------------------------------------------------------------------------
# meta trace
# ---------------------------------------------------------------------------

def buffers_to_params(module) -> List[str]:
    """Turn every buffer into a non-trainable parameter, in place.

    A buffer on meta is inlined by the TorchScript exporter as an
    `onnx::Constant` and the export dies copying it out; a parameter goes
    through `export_params=False` and becomes a named graph input instead,
    which the splice then fills from `GraphSpec.constants`. Returns the dotted
    names converted.
    """
    import torch.nn as nn
    names: List[str] = []

    def walk(mod, prefix):
        for name, buf in list(mod._buffers.items()):
            if buf is None:
                continue
            del mod._buffers[name]
            mod.register_parameter(name, nn.Parameter(buf, requires_grad=False))
            names.append(prefix + name)
        for cname, child in mod.named_children():
            walk(child, prefix + cname + ".")

    walk(module, "")
    return names


def _rename_value(proto, old: str, new: str) -> None:
    for vi in list(proto.graph.input) + list(proto.graph.output):
        if vi.name == old:
            vi.name = new
    for node in proto.graph.node:
        for i, name in enumerate(node.input):
            if name == old:
                node.input[i] = new
        for i, name in enumerate(node.output):
            if name == old:
                node.output[i] = new


def _value_names(proto) -> set:
    names = {vi.name for vi in list(proto.graph.input)
             + list(proto.graph.output) + list(proto.graph.value_info)}
    for node in proto.graph.node:
        names.update(node.input)
        names.update(node.output)
    return names


def trace_weightless(spec: GraphSpec, opset: int = OPSET):
    """Emit the graph with every parameter a DISTINCT, NAMED graph input.

    The same call sequence as `minimax_h3/h3_export_common.trace_weightless`
    (where the reasons were measured), generalized: no H3-specific naming.

      * `_trigger_symbolic_function_registration()` is required; without it
        the trace has no `aten::unflatten` symbolic and reports a misleading
        "opset 20" error.
      * `_jit_pass_onnx_deduplicate_initializers` must NOT run: it keys on
        the storage pointer, every meta tensor has `data_ptr() == 0`, and the
        parameter set collapses to a handful of inputs plus Identity nodes.
      * the parameter inputs are renamed after their state-dict name BY
        POSITION (inputs keep their order through the export; user inputs come
        first), because `input_names` is not reliably applied by
        `_optimize_graph`.

    Returns (proto, {param: shape}, {param: st dtype name}).
    """
    import torch
    from torch.onnx import OperatorExportTypes
    from torch.onnx._internal.torchscript_exporter import utils as ex

    ex._trigger_symbolic_function_registration()
    ex.GLOBALS.export_onnx_opset_version = opset
    ex.GLOBALS.onnx_shape_inference = True

    wrapper = spec.wrapper
    with torch.no_grad():
        graph, params, _out, module = ex._create_jit_graph(wrapper,
                                                           spec.dummies)
        params_dict = ex._get_named_param_dict(graph, params)
        graph = ex._optimize_graph(
            graph, OperatorExportTypes.ONNX, params_dict=params_dict,
            dynamic_axes=spec.dynamic_axes,
            input_names=list(spec.input_names), module=module)
        ex._C._jit_pass_onnx_assign_scoped_names_for_node_and_value(graph)
        raw = graph._export_onnx(
            {}, opset, spec.dynamic_axes or {}, False,
            OperatorExportTypes.ONNX, True, False, {}, True, "", {})[0]
    proto = onnx.ModelProto()
    proto.ParseFromString(raw)

    state = torch.jit._unique_state_dict(wrapper)
    sd_names = list(state.keys())
    n_user = len(spec.dummies)
    proto_inputs = [vi.name for vi in proto.graph.input]
    param_inputs = proto_inputs[n_user:]
    if len(param_inputs) != len(sd_names):
        raise HFImportError(
            f"trace: {len(param_inputs)} parameter inputs for "
            f"{len(sd_names)} parameters -- the exporter aliased or dropped "
            f"some (first inputs: {proto_inputs[:8]})")
    # Rename in two hops through unique temporaries, and free each target name
    # first: `_optimize_graph` applies `input_names` to only SOME values, and
    # can hand a user input's name to an internal one instead (measured: a
    # Cast whose output was called `inputs_embeds` while the real input was
    # not), which a plain rename would turn into a duplicate definition.
    tmp = [f"__hf_import_{i}" for i in range(len(proto_inputs))]
    for old, t in zip(proto_inputs, tmp):
        _rename_value(proto, old, t)
    outs = [vi.name for vi in proto.graph.output]
    out_tmp = [f"__hf_import_out_{i}" for i in range(len(outs))]
    for old, t in zip(outs, out_tmp):
        _rename_value(proto, old, t)
    wanted = (list(zip(tmp, list(spec.input_names) + sd_names))
              + list(zip(out_tmp, spec.output_names)))
    for t, new in wanted:
        if _value_names(proto) & {new}:
            _rename_value(proto, new, f"{new}__internal")
        _rename_value(proto, t, new)

    # Apply the declared symbolic axes on the finished proto (see the H3 note
    # in `_apply_io_names`: `_optimize_graph` leaves some as `<x>_dim_<i>`).
    by_name = {vi.name: vi for vi in list(proto.graph.input)
               + list(proto.graph.output)}
    for tname, axes in (spec.dynamic_axes or {}).items():
        vi = by_name.get(tname)
        if vi is None:
            continue
        dims = vi.type.tensor_type.shape.dim
        for axis, label in axes.items():
            if 0 <= axis < len(dims):
                dims[axis].ClearField("dim_value")
                dims[axis].dim_param = label

    shapes = {n: list(v.shape) for n, v in state.items()}
    dtypes = {}
    for n, v in state.items():
        key = str(v.dtype).replace("torch.", "")
        if key not in _TORCH_TO_ST:
            raise HFImportError(f"{n}: traced dtype {v.dtype} unsupported")
        dtypes[n] = _TORCH_TO_ST[key]
    return proto, shapes, dtypes


# ---------------------------------------------------------------------------
# splice
# ---------------------------------------------------------------------------

def _external(init, location: str, offset: int, length: int) -> None:
    init.data_location = TensorProto.EXTERNAL
    del init.external_data[:]
    for k, v in (("location", location), ("offset", str(offset)),
                 ("length", str(length))):
        kv = init.external_data.add()
        kv.key = k
        kv.value = v


def _location(path: str, onnx_dir: str) -> str:
    """Shard path as the ONNX file should spell it.

    Relative when possible (the pair can move together), absolute otherwise
    (different volume). `weight_stream._hydrate` and the trailer writer both
    accept either.
    """
    try:
        return os.path.relpath(path, onnx_dir)
    except ValueError:
        return os.path.abspath(path)


def splice_to_shards(proto, shapes: Mapping[str, list],
                     dtypes: Mapping[str, str], index, spec: GraphSpec,
                     onnx_dir: str, side_file: str) -> Dict[str, int]:
    """Turn the parameter graph-inputs into initializers that point at shards.

    Returns counts: {"shard": n, "side": n, "const": n, "side_bytes": n}.
    Fails closed: a parameter that is neither a constant nor found in the
    checkpoint, or found with a different shape, aborts the export.
    """
    from . import stream_export as SE

    counts = {"shard": 0, "side": 0, "const": 0, "side_bytes": 0}
    missing: List[str] = []
    keep: List = []
    side_path = os.path.join(onnx_dir, side_file)
    side = None
    side_off = 0
    try:
        for vi in proto.graph.input:
            name = vi.name
            if name not in shapes:
                keep.append(vi)          # a real graph input
                continue
            traced = dtypes[name]
            if name in spec.constants:
                arr = np.asarray(spec.constants[name])
                want = list(shapes[name])
                if list(arr.shape) != want:
                    raise HFImportError(
                        f"{name}: constant has shape {list(arr.shape)}, the "
                        f"trace declares {want}")
                np_dtype = {"F16": np.float16, "F32": np.float32,
                            "I64": np.int64, "I32": np.int32}.get(traced)
                if np_dtype is None:
                    raise HFImportError(f"{name}: constant dtype {traced}")
                proto.graph.initializer.append(numpy_helper.from_array(
                    np.ascontiguousarray(arr.astype(np_dtype)), name))
                counts["const"] += 1
                continue
            ck = spec.aliases.get(name)
            if ck is None:
                ck = index.nearest(name, shapes[name], spec.ckpt_prefix)
            if ck is None or ck not in index:
                missing.append(name)
                continue
            entry = dict(index.get(ck))
            entry["onnx_name"] = name
            if [int(d) for d in entry["shape"]] != [int(d) for d in shapes[name]]:
                raise HFImportError(
                    f"{name} -> {ck}: checkpoint shape {entry['shape']} != "
                    f"traced {shapes[name]}")
            src = entry["dtype"]
            if src not in _ST_TO_ONNX:
                raise HFImportError(f"{ck}: checkpoint dtype {src} unsupported")
            add = float(spec.addend.get(name, 0.0))
            in_place = not add and (
                src == traced or (src == "BF16" and traced in ("F16", "F32")))
            init = proto.graph.initializer.add()
            init.name = name
            init.dims.extend(int(d) for d in entry["shape"])
            if in_place:
                init.data_type = _ST_TO_ONNX[src]
                _external(init, _location(entry["shard"], onnx_dir),
                          entry["offset"], entry["nbytes"])
                counts["shard"] += 1
                continue
            # Converted (or offset) payload: one tensor in RAM at a time.
            blob = SE._convert_tensor(entry, traced, addend=add)
            if side is None:
                side = open(side_path, "wb")
            init.data_type = _ST_TO_ONNX[traced]
            _external(init, side_file, side_off, len(blob))
            side.write(blob)
            side_off += len(blob)
            counts["side"] += 1
    finally:
        if side is not None:
            side.close()
    if missing:
        raise HFImportError(
            f"{len(missing)} parameter(s) not in the checkpoint and not "
            f"computed constants: {missing[:6]}"
            f"{' ...' if len(missing) > 6 else ''}")
    counts["side_bytes"] = side_off
    if side is None and os.path.exists(side_path):
        os.remove(side_path)             # stale from an earlier run
    del proto.graph.input[:]
    proto.graph.input.extend(keep)
    return counts


# ---------------------------------------------------------------------------
# builders
# ---------------------------------------------------------------------------

def build_llama_decoder(config, torch_dtype, index) -> GraphSpec:
    """The Llama-family builder (kept here for callers of the old name)."""
    from .hf_builders import llama
    return llama.build(config, torch_dtype, index)


def resolve_builder(spec: Optional[str], config=None) -> Callable:
    """`--hf-builder` if given, else the registered builder for model_type."""
    if not spec:
        from . import hf_builders
        return hf_builders.for_config(config)
    if ":" not in spec:
        from . import hf_builders
        return hf_builders.by_name(spec)
    mod_name, fn_name = spec.split(":", 1)
    if mod_name.endswith(".py") and os.path.isfile(mod_name):
        import importlib.util
        s = importlib.util.spec_from_file_location("_hf_builder", mod_name)
        mod = importlib.util.module_from_spec(s)
        s.loader.exec_module(mod)
    else:
        mod = importlib.import_module(mod_name)
    fn = getattr(mod, fn_name, None)
    if not callable(fn):
        raise HFImportError(f"--hf-builder {spec!r}: {fn_name} not callable")
    return fn
# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------

def export_graph_onnx(ckpt_dir: str, onnx_path: str, *,
                      builder: Optional[str] = None,
                      trace_dtype: str = "auto",
                      act: Optional[str] = None) -> str:
    """Write a graph-only ONNX whose initializers reference `ckpt_dir`'s shards.

    Returns `onnx_path`. Peak memory is the graph plus, at most, one tensor
    being converted into the side file.
    """
    import torch
    from transformers import AutoConfig
    from . import stream_export as SE

    ckpt_dir = os.path.abspath(ckpt_dir)
    load_config(ckpt_dir)                    # a clear error for a wrong dir
    index = SE.SafetensorsIndex(ckpt_dir)
    census = float_dtype_census(index)
    print(f"[hf] {ckpt_dir}: {len(index.entries)} tensor(s) in "
          f"{len(index.shards)} shard(s); "
          + ", ".join(f"{k} {v / 2**30:.2f} GiB"
                      for k, v in sorted(census.items())), flush=True)

    tdt = pick_trace_dtype(trace_dtype, act, index)
    torch_dtype = torch.float16 if tdt == "fp16" else torch.float32
    config = AutoConfig.from_pretrained(ckpt_dir, local_files_only=True)
    spec = resolve_builder(builder, config)(config, torch_dtype, index)
    if not isinstance(spec, GraphSpec):
        raise HFImportError("the builder must return an onnx2vkop.hf_import."
                            "GraphSpec")
    if spec.summary:
        print("[hf] " + " ".join(f"{k}={v}" for k, v in spec.summary.items()),
              flush=True)
    print(f"[hf] tracing on meta at {tdt} (no weight bytes)", flush=True)
    proto, shapes, dtypes = trace_weightless(spec)
    print(f"[hf] graph: {len(proto.graph.node)} node(s), "
          f"{len(shapes)} parameter(s)", flush=True)

    onnx_dir = os.path.dirname(os.path.abspath(onnx_path)) or "."
    os.makedirs(onnx_dir, exist_ok=True)
    side = os.path.splitext(os.path.basename(onnx_path))[0] + ".side.bin"
    counts = splice_to_shards(proto, shapes, dtypes, index, spec, onnx_dir,
                              side)
    print(f"[hf] {counts['shard']} weight(s) referenced in place in the "
          f"shards, {counts['side']} converted into {side} "
          f"({counts['side_bytes'] / 2**20:.1f} MiB), {counts['const']} "
          f"computed constant(s)", flush=True)

    if spec.post_splice is not None:
        spec.post_splice(proto)
    # Last: the dims recovered here may sit behind outputs post_splice
    # declared (see stream_export.resolve_shape_dims).
    fixed = SE.resolve_shape_dims(proto, spec.dynamic_axes)
    onnx.save(proto, onnx_path)
    print(f"[hf] wrote {onnx_path} "
          f"({os.path.getsize(onnx_path) / 2**20:.1f} MiB); {fixed} dim(s) "
          f"recovered", flush=True)
    return onnx_path
