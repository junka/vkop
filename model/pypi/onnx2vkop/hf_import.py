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
what the graph is. `build_llama_decoder` (the default) reproduces the in-tree
`qwen3_export_onnx.py` wrapper -- explicit KV I/O, `attention_bias`,
`inputs_embeds` -- for any checkpoint with the Llama/Qwen2/Qwen3/Mistral block
structure, so the result runs in llm_chat unchanged. `--hf-builder pkg.mod:fn`
plugs in another one; it receives `(config, torch_dtype, index)` and returns a
`GraphSpec`.

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
# default builder: Llama-family decoder with the in-tree llm I/O contract
# ---------------------------------------------------------------------------

def _rotate_half(x):
    import torch
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def _make_decoder_wrapper(lm, lm_head, num_heads, num_kv_heads, head_dim,
                          scaling, attention_scaling, inv_freq_len):
    """Same graph as llm/exporter/export/qwen3_export_onnx.py's wrapper.

    Kept op-for-op identical on purpose: the converter's attention / RMSNorm
    / SwiGLU fusions were written against that export, and the runtime's LLM
    driver reads its I/O names. q_norm/k_norm are applied only when the
    checkpoint has them (Qwen3), which is the one place Qwen2/Llama/Mistral
    differ in this block.

    RoPE is computed here instead of through the HF rotary module: that module
    moves its CPU `inv_freq` to the input's device and runs under autocast,
    neither of which a meta trace survives. The arithmetic is the HF default
    one -- `freqs = inv_freq x pos`, `cat(freqs, freqs)`, cos/sin scaled by
    `attention_scaling` -- with `inv_freq` taken from the HF module itself, so
    rope-scaling variants that only change `inv_freq`/`attention_scaling`
    (linear, llama3, yarn, ...) come out right. Dynamic NTK, which changes
    `inv_freq` with the sequence length at run time, is refused by the caller.
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class DecoderLM(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = lm.layers
            self.norm = lm.norm
            self.lm_head = lm_head
            self.rope_inv_freq = nn.Parameter(
                torch.empty(inv_freq_len, dtype=torch.float32, device="meta"),
                requires_grad=False)

        def rope(self, position_ids, dtype):
            inv = self.rope_inv_freq[None, :, None].float()
            inv = inv.expand(position_ids.shape[0], -1, 1)
            freqs = (inv @ position_ids[:, None, :].float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * attention_scaling
            sin = emb.sin() * attention_scaling
            return cos.to(dtype), sin.to(dtype)

        def block(self, layer, hidden, cos, sin, attention_bias, past_kv):
            B, q_len, _ = hidden.shape
            residual = hidden
            h = layer.input_layernorm(hidden)
            attn = layer.self_attn
            q = attn.q_proj(h).view(B, q_len, num_heads, head_dim)
            if getattr(attn, "q_norm", None) is not None:
                q = attn.q_norm(q)
            q = q.transpose(1, 2)
            k = attn.k_proj(h).view(B, q_len, num_kv_heads, head_dim)
            if getattr(attn, "k_norm", None) is not None:
                k = attn.k_norm(k)
            k = k.transpose(1, 2)
            v = attn.v_proj(h).view(B, q_len, num_kv_heads,
                                    head_dim).transpose(1, 2)
            c = cos.unsqueeze(1)
            s = sin.unsqueeze(1)
            q = (q * c) + (_rotate_half(q) * s)
            k = (k * c) + (_rotate_half(k) * s)
            k_new = torch.cat([past_kv[:, 0], k], dim=2)
            v_new = torch.cat([past_kv[:, 1], v], dim=2)
            groups = num_heads // num_kv_heads
            if groups > 1:
                b, nkv, slen, hd = k_new.shape
                k_r = k_new[:, :, None, :, :].expand(
                    b, nkv, groups, slen, hd).reshape(b, nkv * groups, slen, hd)
                v_r = v_new[:, :, None, :, :].expand(
                    b, nkv, groups, slen, hd).reshape(b, nkv * groups, slen, hd)
            else:
                k_r, v_r = k_new, v_new
            w = torch.matmul(q, k_r.transpose(2, 3)) * scaling
            w = w + attention_bias
            w = F.softmax(w, dim=-1, dtype=torch.float32).to(q.dtype)
            out = torch.matmul(w, v_r)
            out = out.transpose(1, 2).reshape(B, q_len, num_heads * head_dim)
            hidden = residual + attn.o_proj(out)
            residual = hidden
            h = layer.post_attention_layernorm(hidden)
            mlp = layer.mlp
            h = mlp.down_proj(mlp.act_fn(mlp.gate_proj(h)) * mlp.up_proj(h))
            hidden = residual + h
            return hidden, torch.stack([k_new, v_new], dim=1)

        def forward(self, inputs_embeds, position_ids, attention_bias,
                    *past_kvs):
            hidden = inputs_embeds
            cos, sin = self.rope(position_ids, hidden.dtype)
            presents = []
            for i, layer in enumerate(self.layers):
                hidden, pk = self.block(layer, hidden, cos, sin,
                                        attention_bias, past_kvs[i])
                presents.append(pk)
            hidden = self.norm(hidden)
            return (self.lm_head(hidden), *presents)

    return DecoderLM()


def _text_prefix(index) -> Optional[str]:
    """Checkpoint prefix of the text tower (`model.`, `model.language_model.`).

    Found from the first layer's q_proj; a namesake under a vision tower or an
    MTP head is ignored, since those are what the prefix exists to exclude.
    """
    tail = "layers.0.self_attn.q_proj.weight"
    cands = [k[: -len(tail)] for k in index.entries if k.endswith(tail)]
    skip = ("visual", "vision", "mtp", "audio", "speech")
    cands = [c for c in cands if not any(s in c for s in skip)] or cands
    if not cands:
        return None
    for pref in ("model.language_model.", "model.", ""):
        if pref in cands:
            return pref
    return sorted(cands, key=len)[0]


_LLAMA_FAMILY = {"llama", "qwen2", "qwen3", "mistral", "qwen3_text",
                 "qwen2_text"}


def build_llama_decoder(config, torch_dtype, index) -> GraphSpec:
    """Default builder: the in-tree LLM I/O contract on a Llama-style block.

    Inputs:  inputs_embeds [B,S,H], position_ids [B,S] int64,
             attention_bias [B,1,q,kv], past_key_values_i [B,2,nkv,kv,hd]
    Outputs: logits [B,S,V], present_key_values_i [B,2,nkv,kv,hd]

    The embedding table is not in the graph (llm_chat reads it from its own
    file; see llm/server/embed_table.py, which also reads safetensors).
    """
    import torch
    from transformers import AutoModelForCausalLM

    text_cfg = config.get_text_config() if hasattr(
        config, "get_text_config") else config
    mtype = getattr(text_cfg, "model_type", "?")
    try:
        text_cfg._attn_implementation = "eager"
    except Exception:
        pass
    rope_params = (getattr(text_cfg, "rope_parameters", None)
                   or getattr(text_cfg, "rope_scaling", None) or {})
    rope_type = (rope_params.get("rope_type") or rope_params.get("type")
                 or "default") if isinstance(rope_params, dict) else "default"
    if rope_type == "dynamic":
        raise HFImportError(
            "dynamic-NTK RoPE recomputes inv_freq from the sequence length at "
            "run time; the static graph cannot carry that. Use --hf-builder.")

    with torch.device("meta"):
        try:
            model = AutoModelForCausalLM.from_config(text_cfg,
                                                     dtype=torch_dtype)
        except TypeError:
            model = AutoModelForCausalLM.from_config(text_cfg,
                                                     torch_dtype=torch_dtype)
    model = model.to(torch_dtype).eval()
    lm = getattr(model, "model", None)
    lm_head = getattr(model, "lm_head", None)
    need = []
    if lm is None or not hasattr(lm, "layers") or not hasattr(lm, "norm"):
        need.append("model.layers / model.norm")
    if lm_head is None:
        need.append("lm_head")
    if not need:
        l0 = lm.layers[0]
        for path in ("input_layernorm", "post_attention_layernorm",
                     "self_attn.q_proj", "self_attn.k_proj",
                     "self_attn.v_proj", "self_attn.o_proj", "mlp.gate_proj",
                     "mlp.up_proj", "mlp.down_proj", "mlp.act_fn"):
            obj = l0
            for p in path.split("."):
                obj = getattr(obj, p, None)
            if obj is None:
                need.append(f"layers.N.{path}")
    if need:
        raise HFImportError(
            f"model_type {mtype!r} is not a Llama-style decoder (missing "
            f"{', '.join(need)}); pass --hf-builder pkg.module:function")
    if mtype not in _LLAMA_FAMILY:
        print(f"[hf] model_type {mtype!r} has the Llama block structure but "
              f"is not one of the tested families ({sorted(_LLAMA_FAMILY)}); "
              f"verify its output against HF before trusting it",
              file=sys.stderr)
    if "gemma" in mtype:
        raise HFImportError("Gemma's (1 + w) RMSNorm is not handled by the "
                            "default builder; pass --hf-builder")

    attn0 = lm.layers[0].self_attn
    hidden = int(text_cfg.hidden_size)
    num_heads = int(text_cfg.num_attention_heads)
    num_kv = int(getattr(text_cfg, "num_key_value_heads", None) or num_heads)
    head_dim = int(getattr(text_cfg, "head_dim", None)
                   or hidden // num_heads)
    scaling = float(getattr(attn0, "scaling", head_dim ** -0.5))
    nlayers = len(lm.layers)

    # The rotary module is cheap; build it for real (CPU) to read inv_freq
    # and attention_scaling exactly as HF computes them for this config.
    rot_cls = type(lm.rotary_emb)
    rot = rot_cls(config=text_cfg)
    inv_freq = rot.inv_freq.detach().float().cpu().numpy()
    attention_scaling = float(getattr(rot, "attention_scaling", 1.0))

    wrapper = _make_decoder_wrapper(lm, lm_head, num_heads, num_kv, head_dim,
                                    scaling, attention_scaling,
                                    inv_freq.shape[0]).eval()
    # The wrapper's own blocks should carry no buffers (the rotary module is
    # deliberately not part of it). One that appears has a value this builder
    # cannot know, so refuse rather than leave a dangling graph input.
    converted = buffers_to_params(wrapper)
    if converted:
        raise HFImportError(f"unexpected buffer(s) {converted[:4]} in the "
                            f"decoder wrapper; the default builder does not "
                            f"know their values. Pass --hf-builder.")
    constants = {"rope_inv_freq": inv_freq}

    prefix = _text_prefix(index)
    aliases = {}
    if "lm_head.weight" not in index.entries and not any(
            k.endswith(".lm_head.weight") for k in index.entries):
        tied = (prefix or "model.") + "embed_tokens.weight"
        if tied in index.entries:
            aliases["lm_head.weight"] = tied
        else:
            raise HFImportError("checkpoint has no lm_head.weight and no "
                                f"{tied} to tie it to")

    S = 8
    dt = torch_dtype
    dev = "meta"
    dummies = (torch.empty(1, S, hidden, dtype=dt, device=dev),
               torch.empty(1, S, dtype=torch.long, device=dev),
               torch.empty(1, 1, S, S, dtype=dt, device=dev),
               *[torch.empty(1, 2, num_kv, 0, head_dim, dtype=dt, device=dev)
                 for _ in range(nlayers)])
    input_names = (["inputs_embeds", "position_ids", "attention_bias"]
                   + [f"past_key_values_{i}" for i in range(nlayers)])
    output_names = (["logits"]
                    + [f"present_key_values_{i}" for i in range(nlayers)])
    dyn = {"inputs_embeds": {0: "batch", 1: "seq"},
           "position_ids": {0: "batch", 1: "seq"},
           "attention_bias": {0: "batch", 2: "q_len", 3: "kv_len"},
           "logits": {0: "batch", 1: "seq"}}
    for i in range(nlayers):
        dyn[f"past_key_values_{i}"] = {0: "batch", 3: "kv_len"}
        dyn[f"present_key_values_{i}"] = {0: "batch", 3: "kv_len"}

    return GraphSpec(
        wrapper=wrapper, dummies=dummies, input_names=input_names,
        output_names=output_names, dynamic_axes=dyn, constants=constants,
        aliases=aliases, ckpt_prefix=prefix,
        summary={"model_type": mtype, "layers": nlayers, "hidden": hidden,
                 "heads": num_heads, "kv_heads": num_kv,
                 "head_dim": head_dim, "rope": rope_type,
                 "ckpt_prefix": prefix, "tied_lm_head": bool(aliases)})


def resolve_builder(spec: Optional[str]) -> Callable:
    if not spec:
        return build_llama_decoder
    if ":" not in spec:
        raise HFImportError(f"--hf-builder {spec!r}: expected pkg.module:function")
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
    spec = resolve_builder(builder)(config, torch_dtype, index)
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

    fixed = SE.resolve_shape_dims(proto, spec.dynamic_axes)
    onnx.save(proto, onnx_path)
    print(f"[hf] wrote {onnx_path} "
          f"({os.path.getsize(onnx_path) / 2**20:.1f} MiB); {fixed} dim(s) "
          f"recovered", flush=True)
    return onnx_path
