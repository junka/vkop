"""Main model converter implementation."""

from collections import defaultdict
import os
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper

try:
    from .dag import (
        DAGBasedModel,
        Node,
        _DATA_TYPE_MAP,
        _VKOP_CAUSAL_MASK_META_PREFIX,
        _VKOP_LOGITS_LAST_ROW_META_PREFIX,
        _init_byte_len,
    )
    from . import shader_map
    from . import weight_stream
    from . import layout_audit
    from . import numeric_audit
    from . import exclusion
    from . import quant_payload
    from . import bf16_weights
    from .fp16_graph import GraphFp16
    from .optimizer import (
        FusionOptimizer,
        InitializerMerger,
        ONNXOptimizer,
        KnWeightLayout,
        PreQuantizedWeights,
        Quantizer,
        RGBAConverter,
        Unifier,
        W8A8ActivationFold,
        _disabled_passes,
        _position_paging_slots,
    )
except ImportError:
    from dag import (
        DAGBasedModel,
        Node,
        _DATA_TYPE_MAP,
        _VKOP_CAUSAL_MASK_META_PREFIX,
        _VKOP_LOGITS_LAST_ROW_META_PREFIX,
        _init_byte_len,
    )
    import shader_map
    import weight_stream
    import layout_audit
    import numeric_audit
    import exclusion
    import quant_payload
    import bf16_weights
    from fp16_graph import GraphFp16
    from optimizer import (
        FusionOptimizer,
        InitializerMerger,
        ONNXOptimizer,
        KnWeightLayout,
        PreQuantizedWeights,
        Quantizer,
        RGBAConverter,
        Unifier,
        W8A8ActivationFold,
        _disabled_passes,
        _position_paging_slots,
    )



# The graph output the LM head writes in every exporter in this tree.
DEFAULT_LOGITS_LAST_ROW = frozenset({"logits"})


def resolve_logits_last_row(metadata_props, env_value):
    """Graph outputs whose LM head keeps only the last sequence position.

    Default: {"logits"}. A greedy or sampling decoder reads logits[0, L-1, :]
    and nothing else, and every driver here (llm_chat, llm_serve, llm_driver's
    ORT diff) takes the last row by shape, so a one-row output is read the same
    way. Only the name `logits` is assumed -- not "*" -- because other models
    end in a MatMul too (the DiT's proj_out is rank 3) and need every row.

    Keeping every position has to be said explicitly (perplexity, prompt
    logprobs, a per-position ORT diff):
      * VKOP_LOGITS_LAST_ROW=0 (or the CLI's --all-logits-rows): no output is
        cut, whatever the metadata says;
      * metadata `vkop_logits_last_row:<out>` = "0": that output keeps all rows.
    Adding outputs: metadata `vkop_logits_last_row:<out>` with any other value,
    or VKOP_LOGITS_LAST_ROW=<a,b> (comma-separated names, "*" for every graph
    output a MatMul produces). A graph that already slices its own last row
    folds regardless (optimizer.match_logits_last_row).
    """
    env = (env_value or "").strip()
    if env == "0":
        return set()
    wanted = set(DEFAULT_LOGITS_LAST_ROW)
    for entry in metadata_props or []:
        if not entry.key.startswith(_VKOP_LOGITS_LAST_ROW_META_PREFIX):
            continue
        name = entry.key[len(_VKOP_LOGITS_LAST_ROW_META_PREFIX):]
        if (entry.value or "").strip() in ("", "0"):
            wanted.discard(name)
        else:
            wanted.add(name)
    for name in env.split(","):
        name = name.strip()
        if name:
            wanted.add(name)
    return wanted



def _declared_widths_backend(args) -> str:
    """The backend whose dtype rules the width passes (FP16/bf16 domain, the
    fp16 weight pass, the -q fp32 mixed-width check) should follow.

    "auto" ships both backends' shaders and lets the runtime pick, so it must
    get the buffer backend's declarations: the buffer backend creates every
    tensor with the dtype the file declares, while the image backend ignores
    declarations (it builds every graph input fp16), so declaring for buffer
    costs the image side nothing. Treating "auto" as "not buffer" silently
    skipped the whole half-domain pass for a default conversion -- fp32
    activations, fp16 or bf16 alike.
    """
    backend = getattr(args, "target_backend", "buffer") or "buffer"
    return "image" if backend == "image" else "buffer"


# Activation dtypes (--act). The two precision axes are independent knobs:
#   -q     WEIGHT format: how initializer payloads are stored (fp16, bf16,
#          int8, fp8*, int4, nf4, nvfp4, awq; fp32 = keep the export's bytes).
#   --act  ACTIVATION dtype: the width the domain pass (fp16_graph.py)
#          declares every half-capable tensor at. fp32 = keep the export's
#          declarations and skip the pass.
# A new activation format goes here first, then into fp16_graph.GraphFp16
# (its element type and range rules) and shader_map (its shader suffix).
ACTIVATION_DTYPES = ("fp16", "bf16", "fp32")

# Weight formats whose kernels have run with an fp32 activation domain. That
# was the only domain for weight-only formats before the FP16 domain became
# the default (2ebbbae). -q bf16 and -q fp16 never had one.
_FP32_ACT_WEIGHT_FORMATS = ("fp32", "int8", "fp8e4m3", "fp8e5m2", "int4",
                            "nf4", "nvfp4", "awq")


def resolve_precision(quant, act=None, legacy_half=None):
    """Settle (-q, --act) into one consistent pair, or raise ValueError.

    `legacy_half` is the deprecated --half-precision spelling of --act.
    Defaults keep every existing command line meaning what it meant:
    no --act gives fp16, except -q fp32, which has always also meant "keep
    the export's activation widths". --act fp32 without -q keeps the weights
    too, i.e. it is -q fp32.
    """
    if legacy_half is not None:
        if act is not None and act != legacy_half:
            raise ValueError(
                f"--half-precision {legacy_half} contradicts --act {act}; "
                f"--half-precision is a deprecated spelling of --act")
        act = legacy_half
    if act is None:
        act = "fp32" if quant == "fp32" else "fp16"
    if act not in ACTIVATION_DTYPES:
        raise ValueError(f"--act {act!r}: expected one of {ACTIVATION_DTYPES}")
    if act == "fp32":
        if quant is None:
            quant = "fp32"
        elif quant not in _FP32_ACT_WEIGHT_FORMATS:
            raise ValueError(
                f"-q {quant} --act fp32: -q {quant} has no fp32-activation "
                f"build (it IS a half-width artifact). Use --act fp16/bf16, "
                f"or -q fp32 to keep the export's widths")
    elif quant == "fp32":
        raise ValueError(
            f"-q fp32 --act {act}: -q fp32 keeps the export's bytes, and a "
            f"half activation domain rewrites the weights it covers. Drop -q "
            f"(weights follow the activation width) or pick a weight format")
    return quant, act


def activation_dtype(args) -> str:
    """The activation dtype an args object asks for.

    Reads --act when the caller resolved it; otherwise derives it the way
    resolve_precision would, so programmatic callers that only set `quant`
    (and maybe the legacy `half_precision`) keep their old meaning.
    """
    act = getattr(args, "act", None)
    if act is not None:
        return act
    if getattr(args, "quant", None) == "fp32":
        return "fp32"
    return getattr(args, "half_precision", None) or "fp16"


def _fusions_the_target_cannot_run(args) -> dict:
    """Fusion passes whose output this conversion's artifact could not run,
    as {pass name: reason}.

    The SwiGLU folds (fuse_swiglu_gate_up -> MatMul swiglu=1,
    fuse_swiglu_down_proj -> down_act) only have half-activation kernels on
    the buffer backend: ops/Matmul.hpp throws "swiglu needs fp16" for an fp32
    activation, and the fold kernels read their weights as halves. Under
    --act fp32 (which -q fp32 implies) or for the image backend, folding
    would either ship an artifact that throws at load or -- with an fp32
    export's weights -- be refused by Quantizer._validate_folded_weights.
    The unfolded gate/up/SiLU/Mul/down chain runs everywhere, so drop the
    folds instead.
    """
    if activation_dtype(args) == "fp32":
        why = ("--act fp32 keeps fp32 activations and the SwiGLU fold kernels "
               "only run on fp16/bf16 activations")
    elif _declared_widths_backend(args) == "image":
        why = "the image backend has no SwiGLU fold kernels"
    elif activation_dtype(args) == "bf16":
        # The gate/up fold runs on bf16 (coop8 / WIDE epilogues and the
        # swiglu_pack pass read and write bf16); the down fold's module
        # (matmul_down_act.comp) reads fp16 only, so its node would be held
        # FP32 by the bf16 gate -- and no fold kernel takes fp32.
        return {"fuse_swiglu_down_proj":
                "--act bf16: the down-projection fold kernel reads fp16 "
                "only (the gate/up fold still runs)"}
    else:
        return {}
    return {"fuse_swiglu_gate_up": why, "fuse_swiglu_down_proj": why}

class UnsupportedOpError(RuntimeError):
    """A graph contains an op_type the runtime cannot execute.

    Raised at conversion time so a missing operator is a loud failure rather
    than a node the runtime silently skips (core/runtime.cpp used to `continue`
    on OpType::UNKNOWN, leaving downstream consumers reading a null/misaligned
    input — wrong output with every test still green).
    """


# Control-flow op_types carry their body as a GRAPH attribute (then_branch /
# body), which vkop has no way to lower. They also resolve to UNKNOWN in the
# runtime, so validate_supported_ops catches them anyway; naming them here lets
# the error say *why* instead of just "unknown op".
_CONTROL_FLOW_OPS = {"If", "Loop", "Scan"}


# Graph inputs and outputs are the model's I/O contract, so their dtype is
# resolved against the same table initializers use (writer and reader must agree
# on the spelling — see dag._DATA_TYPE_MAP) and an element format missing from
# it is a HARD error, not the old silent "". The C++ loader reads an empty
# ShapeRef.dtype as "historical fp16" (core/runtime.cpp), which would store a
# quantized fp8 KV cache at I/O as half-precision and read every byte wrong with
# nothing to trace it to at load or run time. Raising by default is what keeps a
# format we have no name for from laundering into fp16; a format that has a name
# but no kernel yet still has to clear the loader's own per-format gate.
def _io_dtype_name(elem_type, what):
    name = _DATA_TYPE_MAP.get(int(elem_type))
    if name is None:
        raise ValueError(
            f"vkop: {what} has ONNX elem_type {int(elem_type)}, which has no "
            f"dtype name in the converter's table; refusing to emit it so the "
            f"loader cannot silently read it as fp16."
        )
    return name


# ONNX has no NF4 element type, so a 4-bit KV cache that stores NF4 nibbles
# spells them UINT4 (its storage) and names the meaning in a `vkop_dtype`
# metadata prop on the ValueInfoProto -- the same extension channel a packed
# *initializer* uses (dag._VKOP_DTYPE_META_KEY). Graph I/O therefore has to be
# resolved through it before the elem_type table, or the nf4 cache would be
# loaded as uint4 and every nibble decoded through the wrong codebook.
def _io_dtype_override(model, vi):
    # Model-level FIRST: onnxoptimizer.optimize() rebuilds the graph and drops
    # every ValueInfoProto metadata_props, so by the time the converter sees the
    # graph only the model-level copy survives (verified against onnxoptimizer).
    # The ValueInfoProto prop is still accepted so a hand-built or un-optimized
    # model works too.
    for entry in getattr(model, "metadata_props", []):
        if entry.key == f"vkop_io_dtype:{vi.name}":
            return entry.value
    for entry in getattr(vi, "metadata_props", []):
        if entry.key == "vkop_dtype":
            return entry.value
    return None


# ONNX BFLOAT16 is not a format the runtime can compute on: every buffer (SSBO)
# shader unpacks fp16 with unpackHalf2x16, so a bf16 tensor has no storage type
# it could ride and no kernel that could read it. Rather than grow a second
# 16-bit float format through every op, the converter normalizes it away. bf16's
# 8-bit mantissa is narrower than fp16's 11-bit significand, so a weight that
# fits fp16's exponent range converts with NO rounding at all -- the same
# argument the image exporter's bf16 checkpoint path already relies on
# (image/exporter/qi21_export_text_encoder_onnx.py:46-48). Values outside fp16's
# +-65504 saturate, which is why this is a weight/IO normalization and not an
# arithmetic path a mid-graph bf16 activation could depend on.
_BFLOAT16 = onnx.TensorProto.BFLOAT16
_FLOAT16 = onnx.TensorProto.FLOAT16
_FLOAT32 = onnx.TensorProto.FLOAT


def _bf16_bytes_to_fp16(raw: bytes) -> bytes:
    """Reinterpret a buffer of bf16 bit patterns as fp16.

    bf16 is fp32 with the low 16 mantissa bits dropped, so the conversion is a
    shift: widen each uint16 into the high half of a uint32 and read it as
    float32, then let numpy round once to fp16. Pure numpy on purpose -- numpy
    has no native bfloat16, and taking on ml_dtypes just to move a tensor
    between two float formats would be a new converter dependency."""
    u16 = np.frombuffer(raw, dtype=np.uint16)
    f32 = (u16.astype(np.uint32) << 16).view(np.float32)
    with np.errstate(over="ignore"):
        # A magnitude past fp16's +-65504 saturates to inf; numpy would warn,
        # but that is the defined outcome here, not an error.
        return np.ascontiguousarray(f32.astype(np.float16)).tobytes()


_FP32_ELEM_TYPES = {_FLOAT32, onnx.TensorProto.DOUBLE}


def _consumer_runs_fp32(graph, init_names: set) -> set:
    """Names of initializers that at least one consumer reads through an
    fp32 data operand.

    This is the same rule `Quantizer._match_weight_only` applies to a weight it
    is about to convert (`consumer_runs_fp32` there): the shader dtype of an op
    is decided by its DATA input (core/runtime.cpp's op_shader_fp16 reads
    node_inputs[0]), never by the operand the weight happens to be. A 16-bit
    buffer handed to such an op is not an accuracy loss -- the fp32 shader
    reads it with uintBitsToFloat, so two neighbouring halves become one float
    and the buffer is walked at half density. That is silent corruption, not
    rounding, which is why the operand has to match the shader.

    The data dtype has to be resolved here rather than read from
    value_info: this runs BEFORE optimize_model, which is what runs
    shape_inference and fills the value_info in. So follow the producer chain
    back to something whose type is declared -- a graph input, another
    initializer, or a Cast (whose `to` attribute is authoritative). An
    unresolvable chain is treated as "not fp32", i.e. it downcasts, which is
    the pre-existing behaviour for anything this cannot prove.

    Only the FIRST non-initializer input is examined: that is the data operand
    by definition (MatMul(x, W), Gemm(x, W, B), Add(x, bias), Mul(x, scale)).
    """

    producers = {}
    for node in graph.node:
        for out in node.output:
            producers[out] = node
    declared = {vi.name: vi.type.tensor_type.elem_type
                for vi in graph.input}
    for init in graph.initializer:
        declared.setdefault(init.name, init.data_type)

    def resolves_fp32(name, depth=0):
        if depth > 16:
            return False
        elem = declared.get(name)
        if elem is not None:
            return elem in _FP32_ELEM_TYPES
        node = producers.get(name)
        if node is None:
            return False
        if node.op_type == "Cast":
            to = next((a.i for a in node.attribute if a.name == "to"), None)
            return to in _FP32_ELEM_TYPES
        if node.op_type == "Constant":
            val = next((a.t for a in node.attribute if a.name == "value"), None)
            return val is not None and val.data_type in _FP32_ELEM_TYPES
        # Any other producer is dtype-preserving from its own data operand.
        for inp in node.input:
            if inp not in init_names:
                return resolves_fp32(inp, depth + 1)
        return False

    forced = set()
    for node in graph.node:
        data_operands = [i for i in node.input if i and i not in init_names]
        if not data_operands:
            continue
        if not resolves_fp32(data_operands[0]):
            continue
        for inp in node.input:
            if inp in init_names:
                forced.add(inp)
    return forced


def _bf16_bytes_to_fp32(raw: bytes) -> bytes:
    """Reinterpret a buffer of bf16 bit patterns as float32.

    bf16 IS fp32 with the low 16 mantissa bits dropped, so widening each
    uint16 into the high half of a uint32 recovers the exact value -- no
    rounding at all. Used for a weight whose consumer computes in fp32 (see
    _downcast_bfloat16)."""
    u16 = np.frombuffer(raw, dtype=np.uint16)
    return np.ascontiguousarray(u16.astype(np.uint32) << 16).tobytes()


# Set by _downcast_bfloat16 so the caller can report the two outcomes apart:
# `returned` counts the fp16 conversions, and this counts the fp32 widenings.
_downcast_stats = {"widened": 0}


def _downcast_bfloat16(model, act: str = "fp16") -> int:
    """Rewrite every bfloat16 tensor in the model to float16, in place.

    Under `act == "bf16"` the target is float32 instead, for every tensor: the
    bf16 domain pass (fp16_graph.py) re-declares what it owns as bfloat16 from
    FLOAT, and bf16 -> fp32 -> bf16 is exact. Normalizing to float16 there
    left a bf16 checkpoint's weights and I/O as FLOAT16, which the domain pass
    never rewrites (it only downgrades FLOAT) -- i.e. a mixed fp16/bf16 graph
    from a plain `--act bf16` conversion.

    Covers the three places a dtype is spelled: an initializer's payload, a
    ValueInfoProto's element type (graph input/output/value_info), and a node
    attribute holding a tensor (Constant). Returns how many were converted.

    An initializer whose consumer reads it through an fp32 data operand is the
    one exception: it is WIDENED to float32 instead. fp16 is not a storage type
    the runtime can hand an fp32 shader -- see _consumer_runs_fp32. That is the
    MiniMax-H3 case: the checkpoint stores adaln_proj.bias and every
    LayerNorm/Mul scale as bf16, while the graph's arithmetic is fp32 (the
    transformer is exported fp32 and only the weights are bf16), so converting
    them to fp16 produced an artifact that computed garbage -- the entire
    +96808-wide adaln bias read at half density, worth a 5144 absolute error on
    a table whose own magnitudes are ~50.
    """
    count = 0
    init_names = {init.name for init in model.graph.initializer}
    forced_fp32 = _consumer_runs_fp32(model.graph, init_names)
    # A bf16 MatMulBnb4 absmax becomes the fp32 nf4 scale table
    # (PreQuantizedWeights._fold_bnb4); rounding it through fp16 first would
    # make the import lossy, while bf16 -> fp32 is exact.
    forced_fp32 |= {node.input[2] for node in model.graph.node
                    if node.op_type == "MatMulBnb4" and len(node.input) > 2}
    widened = 0
    to_fp32 = act == "bf16"
    for init in model.graph.initializer:
        if init.data_type != _BFLOAT16:
            continue
        if to_fp32 or init.name in forced_fp32:
            init.raw_data = _bf16_bytes_to_fp32(init.raw_data)
            init.data_type = _FLOAT32
            widened += 1
            continue
        init.raw_data = _bf16_bytes_to_fp16(init.raw_data)
        init.data_type = _FLOAT16
        count += 1
    _downcast_stats["widened"] = widened
    for vi in (list(model.graph.input) + list(model.graph.output)
               + list(model.graph.value_info)):
        if vi.type.tensor_type.elem_type == _BFLOAT16:
            vi.type.tensor_type.elem_type = _FLOAT32 if to_fp32 else _FLOAT16
            count += 1
    for node in model.graph.node:
        for attr in node.attribute:
            if (attr.type == onnx.AttributeProto.TENSOR
                    and attr.t.data_type == _BFLOAT16):
                if to_fp32:
                    attr.t.raw_data = _bf16_bytes_to_fp32(attr.t.raw_data)
                    attr.t.data_type = _FLOAT32
                else:
                    attr.t.raw_data = _bf16_bytes_to_fp16(attr.t.raw_data)
                    attr.t.data_type = _FLOAT16
                count += 1
    return count


class ModelConverter:
    """Main class for converting ONNX models to DAG-based format."""

    def __init__(self):
        self.optimizer = ONNXOptimizer()
        self.fusion_optimizer = FusionOptimizer()
        self.initializer_merger = InitializerMerger()
        self.quantizer = Quantizer()

    def parse_onnx_model(self, onnx_path: str, batch_size: int = 1, *,
                         quant: str = None, unify: bool = False,
                         rgba: bool = False,
                         group_size: int = 0,
                         keep_bf16: bool = False,
                         act: str = "fp16") -> DAGBasedModel:
        """Parse and convert ONNX model to DAG format.

        `group_size` is the 4-bit K group (`-q int4`'s knob). It is needed
        before any quantization runs, because a deferred int4 weight's scale
        table is `[K / group, N]` and the row records the grouping so the
        writer and the runtime can derive the same table length. The other
        formats ignore it.

        `keep_bf16` (--keep-bf16) stashes the original bits of every bf16
        initializer that could land in a MatMul/Gemm B or Gather table slot,
        before the fp16 normalization below rewrites them, so
        apply_optimizations can emit them as bf16 (bf16_weights.py). `-q bf16`
        stashes them too, so a bf16 source is not re-rounded off its fp16
        downcast.
        """
        model = onnx.load(onnx_path, load_external_data=False)
        # Load external weights (llm.weights.bin) so the optimized model can be
        # processed without a 2GB+ single-file protobuf save -- but not
        # necessarily *all* of them: a pass-through weight is one the pipeline
        # cannot read by value (see weight_stream), and holding a 13.9 GB set of
        # those in raw_data plus a second copy in raw_map is what took the
        # machine down. Deferred tensors keep their external_data reference and
        # are streamed into the .vkopbin trailer at save time.
        base_dir = os.path.dirname(os.path.abspath(onnx_path))
        weight_stream.install_guard()
        # Called on both sides of optimize_model, so bind it once here.
        restore_paging_slots = weight_stream.restore_paging_slots

        # MoE paging slots: vkop_moe_group / vkop_moe_expert_id on the
        # graph.input / graph.value_info declarations of slot initializers
        # (make_moe_variant.py). Both are scanned: a slot wired to a MatMul is
        # a graph *input* (initializer twin), while an unwired slot is only a
        # value_info declaration. Reading just graph.input converted a model
        # whose slots all live in value_info into a slot-less dense one -- the
        # 4320 expert matrices then load as ordinary weights, no error.
        # Read BEFORE optimize_model — onnxoptimizer rebuilds the graph and
        # drops ValueInfoProto metadata_props (the same reason the causal
        # mask rides on model-level metadata), so this is the last point the
        # declaration is readable. Also read before `prepare`, which has to be
        # told which names not to read: the pool is 23 GiB and hydration is not
        # undoable at that scale. The names feed the liveness prune's keep-alive
        # set (a slot's bytes exist so the runtime can page any expert into it,
        # not because a live node reads it); the (group, expert_id) mapping
        # rides along for the runtime's slot table.
        moe_slots = {}
        for source in (list(model.graph.input) + list(model.graph.value_info)):
            grp = eid = None
            for e in source.metadata_props:
                if e.key == "vkop_moe_group":
                    grp = e.value
                elif e.key == "vkop_moe_expert_id":
                    eid = e.value
            if grp is not None:
                moe_slots[source.name] = (grp, eid)
        if moe_slots:
            print(f"[moe] {len(moe_slots)} paging slot(s) declared "
                  f"({len({g for g, _ in moe_slots.values()})} group(s))")

        deferred_weights = weight_stream.prepare(
            model, base_dir, quant=quant, unify=unify, rgba=rgba,
            group_size=group_size, keep_in_file=moe_slots)

        # A float32 weight left in its file under a byte format is quantized on
        # the way into the trailer, so its scale has to exist in the graph: this
        # computes it from the source, adds it as an initializer and appends it
        # to each consumer, exactly the three-input shape -q int8 builds for an
        # in-memory weight. Runs on the proto because no DAG exists yet.
        if deferred_weights:
            quant_payload.finalize(model, deferred_weights)

        # Normalize bf16 away before anything reads a dtype: the runtime has no
        # bf16 kernel and no bf16 storage type, so a model that spells it has to
        # become a type the loader accepts. Runs before optimize_model so the
        # ConstantFolder and every fusion pass only ever see the normalized
        # form.
        #
        # fp16 is the default target, but NOT for a tensor an fp32 shader reads:
        # core/runtime.cpp decides an op's precision from its data input, so a
        # 16-bit operand there is read at half density instead of rounded.
        # _downcast_bfloat16 widens those to fp32 (see its docstring).
        bf16_sources = {}
        if keep_bf16 or quant == "bf16":
            bf16_sources = bf16_weights.stash_sources(model)
            if bf16_sources:
                print(f"[bf16] {len(bf16_sources)} bf16 weight source(s) "
                      f"stashed for the bf16 weight slots")
        # `act` (--act) picks the normalization target: under --act bf16 a
        # bf16 source goes to float32, which the bf16 domain pass turns back
        # into bf16 exactly (see _downcast_bfloat16).
        bf16_count = _downcast_bfloat16(model, act)
        if bf16_count or _downcast_stats["widened"]:
            target = "float32" if act == "bf16" else "float16"
            print(f"[bf16] {bf16_count} bfloat16 tensor(s) -> {target}, "
                  f"{_downcast_stats['widened']} -> float32 (fp32 consumers)")

        # Slots have to be declared in expert order *before* the passes run, and
        # the passes are not order-preserving: onnxoptimizer's
        # eliminate_unused_initializer rebuilds `graph.initializer`, and the
        # order it keeps is that of the surviving graph inputs. Measured: the
        # 4320 slots came out fully interleaved with the dense weights, so the
        # row order stopped matching the exporter's and every diagnostic that
        # compares "expert e of layer l" against a sorted list went wrong.
        # Addressing no longer depends on it (each slot carries its own
        # moe_source/moe_offset/moe_size), but sorted output is still what makes
        # a dump readable.
        if moe_slots:
            moved = _position_paging_slots(model, moe_slots)
            print(f"[moe] slot declaration order: {moved} initializer(s) "
                  f"moved ahead of the dense weights")

        # Where each slot's bytes are in the checkpoint. Read before the
        # pointer is cleared, and kept because the writer records it: a paged
        # expert is pread from this slice at run time, and the trailer does not
        # hold a copy (weight_stream.prepare's `transcribe_slots`).
        #
        # `restore_paging_slots` then drops the external_data pointer so every
        # later walk of the graph skips hydrating the pool (the bytes were kept
        # out of RAM by `prepare`), and checks each slot's (dtype, dims, length)
        # extent. With transcription off, a slot gets no deferred entry, so that
        # extent check is the only signal left that a pass ate a payload -- the
        # two states otherwise look identical (no pointer, no bytes).
        moe_sources = {}
        if moe_slots:
            moe_sources = weight_stream.slot_sources(model, base_dir, moe_slots)
            restore_paging_slots(model, base_dir, deferred_weights, moe_slots)
        PreQuantizedWeights.strip_fp4_operand_casts(model)
        print("Optimizing ONNX model...")
        # The scale initializers quant_payload.finalize added for streamed
        # weights are found again by name (the entry records it), so no pass may
        # merge one into an identical twin: an nvfp4 global is a single fp32
        # value, and equal ones across weights are common.
        quant_scales = {n for e in (deferred_weights or {}).values()
                        for n in (e.get("scale"), e.get("scale_global")) if n}
        model = self.optimizer.optimize_model(model, batch_size,
                                              keep_alive=moe_slots,
                                              pinned=quant_scales)

        # Do NOT onnx.save here — llm.onnx weights exceed protobuf's 2GB
        # single-file limit. The DAG converter reads model.graph directly
        # from the in-memory object.
        dag_model = DAGBasedModel()
        # The default-domain opset, for ort_lower: the DAG keeps each node in
        # its source opset's form (axes as attribute vs input, ...).
        dag_model.onnx_opset = next(
            (o.version for o in model.opset_import if o.domain in ("", "ai.onnx")),
            None)
        dag_model.moe_keep_alive = set(moe_slots)
        dag_model.moe_slots = moe_slots
        dag_model.moe_sources = moe_sources
        # Slots dropped out of the model entirely while the passes ran (a
        # keep-alive node the optimizer removed takes its slot with it). Caught
        # here, while the declaration is still in hand; the write-time
        # keep-alive check in dag.save_to_binary is the backstop.
        if moe_slots:
            added, cleared = restore_paging_slots(
                model, base_dir, deferred_weights, moe_slots)
            held = sum(e["length"] for n, e in deferred_weights.items()
                       if n in moe_slots)
            if held:
                print(f"[moe] {len(moe_slots) - added} slot(s) streamed by "
                      f"prepare, {added} built here, {cleared} external_data "
                      f"pointer(s) cleared; {held >> 20} MiB of experts stays "
                      f"on disk")
            else:
                # The normal case now: slots are not transcribed into the
                # trailer, so the only copy of the pool is the checkpoint.
                print(f"[moe] {len(moe_slots)} slot(s) declared, {cleared} "
                      f"external_data pointer(s) cleared, 0 slot bytes in the "
                      f"trailer; the pool stays in the checkpoint")
        # The streaming contract the parse step negotiated: names whose payload
        # never entered RAM, and where to read them from when the blob is
        # written. save_to_binary verifies it before writing anything.
        dag_model.deferred_weights = deferred_weights
        dag_model.weight_base_dir = base_dir
        # Original bf16 bits by name (empty unless --keep-bf16 / -q bf16);
        # consumed and freed by bf16_weights.apply in apply_optimizations.
        dag_model.bf16_sources = bf16_sources
        # Graph-level declarations that no amount of staring at the graph can
        # recover — today just "this tensor is the standard causal attention
        # bias". Model-level metadata_props is the only channel that survives
        # onnxoptimizer.optimize() (see _io_dtype_override), so it is read from
        # `model` and not from the value infos.
        for entry in getattr(model, "metadata_props", []):
            if not entry.key.startswith(_VKOP_CAUSAL_MASK_META_PREFIX):
                continue
            if entry.value in ("", "0"):
                continue
            dag_model.causal_mask_inputs.add(
                entry.key[len(_VKOP_CAUSAL_MASK_META_PREFIX) :]
            )
        # The same declaration from the environment, for a model already
        # exported without it: VKOP_CAUSAL_MASK=attention_bias (comma
        # separated). Conversion happens once per model while the metadata
        # has to be re-emitted by whoever built the ONNX, so this is the
        # bridge that lets an existing .onnx be reconverted with the
        # declaration rather than re-exported.
        for name in os.environ.get("VKOP_CAUSAL_MASK", "").split(","):
            if name.strip():
                dag_model.causal_mask_inputs.add(name.strip())
        if dag_model.causal_mask_inputs:
            print(
                "Declared causal attention mask(s):",
                ", ".join(sorted(dag_model.causal_mask_inputs)),
            )

        # Which logits rows to keep is a policy the graph cannot show (`logits`
        # has no in-graph consumer), so it is decided here; see
        # resolve_logits_last_row for the rules. The default keeps only the
        # last position of a graph output named `logits`: every driver in this
        # tree reads that row alone, and the other L-1 rows cost 100-200 ms of
        # LM head per long prefill (Qwen3-VL-2B, L~850-3200 on an M5 Max).
        dag_model.logits_last_row = resolve_logits_last_row(
            getattr(model, "metadata_props", []),
            os.environ.get("VKOP_LOGITS_LAST_ROW"))
        if dag_model.logits_last_row:
            print(
                "Logits last row only:",
                ", ".join(sorted(dag_model.logits_last_row)),
                "(--all-logits-rows keeps every position)",
            )
        else:
            print("Logits: every position kept")

        graph = model.graph
        # `full_check` serializes the whole model to a byte string before the
        # checker sees it, and that copy is sized by the RESIDENT initializers.
        # A deferred weight has no bytes here, but a quantizing conversion's
        # SCALE TABLES do: `-q int4` on the 27B puts 1.5 GiB of group scales in
        # the proto, so the serialization added 3 GiB of transient RSS --
        # measured, and it then raised anyway ("model too large"). The size is
        # known from dtype and dims without touching a payload, so the doomed
        # call is skipped rather than paid for. Under the proto's 2 GB limit the
        # copy is bounded and the full check still runs.
        resident = sum(_init_byte_len(init) for init in graph.initializer)
        if resident > (1 << 30):
            print("ONNX model full validation skipped (model too large for "
                  "full_check).")
        else:
            try:
                onnx.checker.check_model(model, full_check=True)
                print("ONNX model passed full validation.")
            except Exception:
                print("ONNX model full validation skipped (model too large for "
                      "full_check).")

        if not self.optimizer.is_topologically_sortable(graph):
            print("Graph is not topologically sorted. Please sort it before proceeding.")
            return None

        # Add initializers
        for initializer in graph.initializer:
            dag_model.initializers[initializer.name] = initializer
        # Deferred weights have no bytes but they do have a payload dtype; the
        # proto still spells the *source* type, and every later check reads the
        # DAG. Repaired here rather than at the earlier spot because the fills
        # above replace the dict wholesale.
        for _name, _entry in deferred_weights.items():
            _qe = _entry.get("data_type")
            _init = dag_model.initializers.get(_name)
            if _qe is not None and _init is not None:
                _init.data_type = _qe

        initializer_names = {init.name for init in graph.initializer}

        # Add inputs
        for inp in graph.input:
            if inp.name in initializer_names:
                continue
            tensor_type = inp.type.tensor_type
            shape_dims = [
                dim.dim_value if dim.HasField("dim_value") and not dim.dim_param
                else -1 for dim in tensor_type.shape.dim
            ]
            dtype_str = _io_dtype_override(model, inp) or _io_dtype_name(
                tensor_type.elem_type, f"graph input {inp.name}")
            print("Graph input:", inp.name, "of shape:", shape_dims, "tensor type:", tensor_type.elem_type)
            dag_model.inputs.append({"name": inp.name, "shape": shape_dims, "dtype": dtype_str})

        # Add outputs
        for out in graph.output:
            tensor_type = out.type.tensor_type
            shape_dims = [
                dim.dim_value if dim.HasField("dim_value") and not dim.dim_param
                else -1 for dim in tensor_type.shape.dim
            ]
            dtype_str = _io_dtype_override(model, out) or _io_dtype_name(
                tensor_type.elem_type, f"graph output {out.name}")
            print("Graph output:", out.name, "of shape:", shape_dims)
            dag_model.outputs.append({"name": out.name, "shape": shape_dims, "dtype": dtype_str})

        modified_shapes = defaultdict(list)
        ELEMWISE_OPS = {
            "Add",
            "And",
            "Div",
            "Equal",
            "Greater",
            "Less",
            "Max",
            "Mean",
            "Min",
            "Mul",
            "Or",
            "Pow",
            "Sub",
            "Sum",
            "Where",
            "Xor",
        }
        for i, node in enumerate(graph.node):
            if not node.name:
                # 使用 op_type + index 生成唯一名，避免冲突
                node.name = f"{node.op_type}_{i}"

        # --- value_dynamic annotation pass ---
        # A shape-meta tensor's element VALUES are "dynamic" (vary across decode
        # rounds) when its producer chain reaches a graph-input dim that is
        # symbolic (dim_param set, i.e. kv_len/batch/seq). The runtime uses this
        # to cache readback results across rounds for STATIC shape tensors
        # (Reshape/Unsqueeze/Slice inputs[1]), skipping the per-round copyToCPU
        # stall. Default is static (cacheable); uncertain tensors MUST be marked
        # dynamic to stay correct.
        producer_of = {}  # tensor_name -> (node, output_index)
        for node in graph.node:
            for oi, out_name in enumerate(node.output):
                if out_name:
                    producer_of[out_name] = (node, oi)

        # Which graph inputs carry a symbolic dim (dim_param)? Those names are
        # the root of dynamic shape values.
        graph_input_shape = {}  # name -> list of dim (concrete int or -1)
        for inp in graph.input:
            tt = inp.type.tensor_type
            dims = [
                dim.dim_value if dim.HasField("dim_value") and not dim.dim_param
                else -1
                for dim in tt.shape.dim
            ]
            graph_input_shape[inp.name] = dims
        initializer_names = {init.name for init in graph.initializer}

        # value_dynamic cache: tensor_name -> bool. Memoized recursion.
        value_dynamic_cache = {}

        def tensor_value_dynamic(tname, visiting=None):
            """True if tname's element values depend on a dynamic graph dim."""
            if tname in value_dynamic_cache:
                return value_dynamic_cache[tname]
            if visiting is None:
                visiting = set()
            if tname in visiting:
                return True  # cycle -> be conservative (dynamic)
            visiting = visiting | {tname}
            result = False
            # Root: graph input. Dynamic iff it has a symbolic dim.
            if tname in graph_input_shape:
                result = any(d == -1 for d in graph_input_shape[tname])
            elif tname in initializer_names:
                result = False  # initializer: static
            elif tname in producer_of:
                node, _oi = producer_of[tname]
                op = node.op_type
                # Shape(x): output value is x's shape. Dynamic iff x has a
                # symbolic dim. (Shape of an initializer -> static.)
                if op == "Shape":
                    if node.input:
                        iname = node.input[0]
                        # x's shape: look up value_info/graph_input
                        xdims = None
                        for vi in list(graph.value_info) + list(graph.input) + list(graph.output):
                            if vi.name == iname:
                                tt = vi.type.tensor_type
                                xdims = [
                                    dim.dim_value if dim.HasField("dim_value") and not dim.dim_param
                                    else -1
                                    for dim in tt.shape.dim
                                ]
                                break
                        if xdims is None and iname in graph_input_shape:
                            xdims = graph_input_shape[iname]
                        if xdims is None and iname in initializer_names:
                            xdims = []  # initializer shape is concrete; static
                        result = bool(xdims) and any(d == -1 for d in xdims)
                    else:
                        result = False
                else:
                    # Any other op: conservative — dynamic if ANY input's
                    # values are dynamic. (Gather/Concat/Slice/Where/Equal/
                    # Mul/Expand/Cast/Range all propagate dynamic-ness through
                    # their data inputs.) ConstantOfShape with a dynamic shape
                    # input is also dynamic, but its output VALUES are a fill
                    # constant — however its SHAPE changes, so downstream
                    # consumers treating it as shape metadata must re-read.
                    # Mark dynamic to be safe.
                    for iname in node.input:
                        if iname and tensor_value_dynamic(iname, visiting):
                            result = True
                            break
            # else: unknown (e.g. node output not in value_info) -> conservative dynamic
            elif tname not in graph_input_shape and tname not in initializer_names:
                result = True
            value_dynamic_cache[tname] = result
            return result

        for node in graph.node:
            print("Processing node:", node.name, "of type:", node.op_type)
            attributes = {}
            if node.op_type == "TopK" and len(node.input) >= 2:
                # ONNX TopK carries k as a scalar int64 INPUT (node.input[1]),
                # not an attribute. ops/Topk.hpp reads it from the "K"
                # attribute via setAttribute(); without this fold the op runs
                # on its ctor default (k=1, fp16-rounded to 2) and writes a
                # [rows, 2] result over the [rows, k] outputs -- the router
                # weights then carry only the first two slots (measured on
                # qwen2_moe: TopK_output_0 ne=64 instead of 128, logits
                # correlated with the reference but wrong).
                # Fail closed when the constant cannot be resolved: a wrong K
                # silently reads the wrong bytes, which is worse than a
                # conversion that refuses.
                k_name = node.input[1]
                k_arr = None
                for init in graph.initializer:
                    if init.name == k_name:
                        k_arr = numpy_helper.to_array(init)
                        break
                if k_arr is None:
                    for n2 in graph.node:
                        if n2.op_type == "Constant" and n2.output and \
                                n2.output[0] == k_name:
                            for a in n2.attribute:
                                if a.name == "value":
                                    k_arr = numpy_helper.to_array(a.t)
                            break
                if k_arr is None or k_arr.size != 1:
                    raise UnsupportedOpError(
                        f"vkop: TopK node '{node.name}': k input '{k_name}' "
                        f"is not a resolvable scalar constant (the runtime "
                        f"reads K from an attribute, not an input)")
                attributes["K"] = int(k_arr.reshape(-1)[0])
            for attr in node.attribute:
                # https://github.com/onnx/onnx/blob/main/onnx/onnx.proto
                # message AttributeProto
                # enum AttributeType {
                #     UNDEFINED = 0;
                #     FLOAT = 1;
                #     INT = 2;
                #     STRING = 3;
                #     TENSOR = 4;
                #     GRAPH = 5;
                #     SPARSE_TENSOR = 11;
                #     TYPE_PROTO = 13;

                #     FLOATS = 6;
                #     INTS = 7;
                #     STRINGS = 8;
                #     TENSORS = 9;
                #     GRAPHS = 10;
                #     SPARSE_TENSORS = 12;
                #     TYPE_PROTOS = 14;
                # }

                if attr.type == onnx.AttributeProto.INT:
                    attributes[attr.name] = attr.i
                elif attr.type == onnx.AttributeProto.FLOAT:
                    attributes[attr.name] = attr.f
                elif attr.type == onnx.AttributeProto.STRING:
                    attributes[attr.name] = attr.s.decode("utf-8")
                elif attr.type == onnx.AttributeProto.TENSOR:
                    attributes[attr.name] = numpy_helper.to_array(attr.t)
                elif attr.type == onnx.AttributeProto.INTS:
                    attributes[attr.name] = list(attr.ints)
                elif attr.type == onnx.AttributeProto.FLOATS:
                    attributes[attr.name] = list(attr.floats)
                elif attr.type in (
                    onnx.AttributeProto.GRAPH,
                    onnx.AttributeProto.GRAPHS,
                ):
                    # A subgraph attribute means control flow (If/Loop/Scan):
                    # its body is a nested ONNX graph vkop has no representation
                    # for. Dropping it here (the old behaviour: a warning) left
                    # the node with no branches and passed it through to be
                    # silently skipped at load. Fail loudly instead.
                    raise UnsupportedOpError(
                        f"vkop: node '{node.name}' ({node.op_type}) carries a "
                        f"subgraph attribute '{attr.name}'. Control flow "
                        f"(If/Loop/Scan) is not supported; fold it to constants "
                        f"at export time (see image/exporter/simplify_vae_if.py)."
                    )
                else:
                    print(
                        f"Warning: Unsupported attribute type {attr.type} for attribute {attr.name}"
                    )

            outputs_with_shape = []
            for output_name in node.output:
                print("  Output:", output_name)
                # Find the output tensor to get its shape
                output_tensor = None
                for value_info in graph.value_info:
                    if value_info.name == output_name:
                        output_tensor = value_info
                        break
                if output_tensor is None:
                    for out in graph.output:
                        if out.name == output_name:
                            output_tensor = out
                            break
                if output_tensor is None:
                    print(f"Warning: Output tensor {output_name} not found in graph.")
                    outputs_with_shape.append({"name": output_name, "shape": []})
                    continue
                tensor_type = output_tensor.type.tensor_type
                shape_dims = [
                    dim.dim_value if dim.HasField("dim_value") and not dim.dim_param
                    else -1
                    for dim in tensor_type.shape.dim
                ]
                # Dynamic dims: a -1 here means the dim is symbolic (dim_param
                # set, no concrete value) and its concrete value is only known
                # at runtime (e.g. kv_len). Keep -1 as the "dynamic" sentinel —
                # the runtime recomputes the actual shape from input data for
                # dynamic-shape ops (Concat, Reshape, ...). A concrete 0 (no
                # dim_param) means genuinely empty (e.g. kv_len=0 at round0
                # prefill) and stays 0. Do NOT normalize -1->1: that would freeze
                # growing dims (KV-cache kv_len) at 1 and break decode rounds.
                #  GlobalAveragePool compression to 2d, so we can use storage instead of image
                if node.op_type == "GlobalAveragePool" and len(shape_dims) >= 2:
                    original_shape = shape_dims[:]
                    shape_dims = [original_shape[0], original_shape[1]]
                    print(f"Modified GlobalAveragePool shape from {original_shape} to {shape_dims}")
                    modified_shapes[output_name] = shape_dims

                # 记录输出 dtype（ONNX TensorProto elem_type int）。Cast 折叠 pass
                # 需要它判断 Cast 输入/输出类型是否相同（避免靠 out.get("dtype") 拿到 None
                # 误判，原 FIXME 的根源）。1=float32, 10=float16, 6=int32, 7=int64 等。
                out_dtype = tensor_type.elem_type
                outputs_with_shape.append(
                    {"name": output_name, "shape": shape_dims, "dtype": out_dtype,
                     "value_dynamic": tensor_value_dynamic(output_name)})

            inputs_with_shape = []
            for input_name in node.input:
                print("  Input:", input_name)
                # Find the input tensor to get its shape
                input_tensor = None
                for value_info in graph.value_info:
                    if value_info.name == input_name:
                        input_tensor = value_info
                        break
                if input_tensor is None:
                    for inp in graph.input:
                        if inp.name == input_name:
                            input_tensor = inp
                            break
                if input_tensor is None:
                    for out in graph.output:
                        if out.name == input_name:
                            input_tensor = out
                            break
                if input_tensor is None:
                    for initializer in graph.initializer:
                        if initializer.name == input_name:
                            data_type_map = {
                                1: onnx.TensorProto.FLOAT,
                                2: onnx.TensorProto.UINT8,
                                3: onnx.TensorProto.INT8,
                                4: onnx.TensorProto.UINT16,
                                5: onnx.TensorProto.INT16,
                                6: onnx.TensorProto.INT32,
                                7: onnx.TensorProto.INT64,
                                8: onnx.TensorProto.STRING,
                                9: onnx.TensorProto.BOOL,
                                10: onnx.TensorProto.FLOAT16,
                                11: onnx.TensorProto.DOUBLE,
                                12: onnx.TensorProto.UINT32,
                                13: onnx.TensorProto.UINT64,
                                14: onnx.TensorProto.COMPLEX64,
                                15: onnx.TensorProto.COMPLEX128,
                                16: onnx.TensorProto.BFLOAT16,
                            }
                            data_type = data_type_map.get(
                                initializer.data_type, onnx.TensorProto.UNDEFINED
                            )
                            input_tensor = onnx.helper.make_tensor_value_info(
                                initializer.name, data_type, initializer.dims
                            )
                            break

                if input_tensor is None:
                    print(f'Warning: Input tensor "{input_name}" not found in graph.')
                    inputs_with_shape.append({"name": input_name, "shape": []})
                    continue
                tensor_type = input_tensor.type.tensor_type
                shape_dims = [
                    dim.dim_value if dim.HasField("dim_value") and not dim.dim_param
                    else -1
                    for dim in tensor_type.shape.dim
                ]
                # Dynamic dims: a -1 here means the dim is symbolic (dim_param)
                # and its concrete value is only known at runtime (e.g. kv_len).
                # Keep -1 as the "dynamic" sentinel — the runtime recomputes the
                # actual shape from input data for dynamic-shape ops (Concat,
                # Reshape, ...). A concrete 0 (no dim_param) means genuinely
                # empty. Do NOT normalize -1->1: that would freeze growing dims
                # (KV-cache kv_len) at 1 and break decode rounds.
                if input_name in modified_shapes and len(modified_shapes[input_name]) > 0:
                    shape_dims = modified_shapes[input_name]
                    print(f"Modified shape of input {input_name} to {shape_dims}")
                inputs_with_shape.append(
                    {"name": input_name, "shape": shape_dims,
                     "dtype": tensor_type.elem_type,
                     "value_dynamic": tensor_value_dynamic(input_name)})

            # if node.op_type in ELEMWISE_OPS and len(node.input) == 2:
            #     for i in range(len(inputs_with_shape)):
            #         if inputs_with_shape[i]["name"] == node.input[0]:
            #             id0 = i
            #         if inputs_with_shape[i]["name"] == node.input[1]:
            #             id1 = i
            #     if len(inputs_with_shape[id0]["shape"]) != len(inputs_with_shape[id1]["shape"]):
            #         shape = self.broadcast_shapes(
            #             inputs_with_shape[id0]["shape"], inputs_with_shape[id1]["shape"]
            #         )
            #         print(
            #             "  Elemwise operation with multiple inputs. Broadcasting shapes...", shape
            #         )
            #         if len(inputs_with_shape[id0]["shape"]) != len(shape):
            #             inputs_with_shape[id0]["shape"] = shape
            #             for initializer in graph.initializer:
            #                 if initializer.name == inputs_with_shape[id0]["name"]:
            #                     orig_init = dag_model.initializers[initializer.name]
            #                     orig_array = numpy_helper.to_array(orig_init)
            #                     if orig_array.size == 1:
            #                         scalar_val = orig_array.item()
            #                         expanded_data = np.full(
            #                             shape, scalar_val, dtype=orig_array.dtype
            #                         )
            #                     else:
            #                         expanded_data = np.broadcast_to(orig_array, shape).copy()
            #                     new_init = numpy_helper.from_array(
            #                         expanded_data, name=initializer.name
            #                     )
            #                     dag_model.initializers[initializer.name] = new_init
            #         if len(inputs_with_shape[id1]["shape"]) != len(shape):
            #             inputs_with_shape[id1]["shape"] = shape
            #             for initializer in graph.initializer:
            #                 if initializer.name == inputs_with_shape[id1]["name"]:
            #                     orig_init = dag_model.initializers[initializer.name]
            #                     orig_array = numpy_helper.to_array(orig_init)
            #                     if orig_array.size == 1:
            #                         scalar_val = orig_array.item()
            #                         expanded_data = np.full(
            #                             shape, scalar_val, dtype=orig_array.dtype
            #                         )
            #                     else:
            #                         expanded_data = np.broadcast_to(orig_array, shape).copy()
            #                     new_init = numpy_helper.from_array(
            #                         expanded_data, name=initializer.name
            #                     )
            #                     dag_model.initializers[initializer.name] = new_init

            new_node = Node(
                op_type=node.op_type,
                name=node.name,
                attributes=attributes,
                inputs=inputs_with_shape,
                outputs=outputs_with_shape,
            )
            dag_model.add_node(new_node)

        dag_model.build_dependencies()
        return dag_model

    @staticmethod
    def broadcast_shapes(shape_a, shape_b):
        """Compute broadcasted shape per ONNX rules."""
        if not shape_a and not shape_b:
            return []
        if not shape_a:
            return shape_b
        if not shape_b:
            return shape_a

        # Make copies and align from right
        a, b = list(shape_a), list(shape_b)
        len_a, len_b = len(a), len(b)
        max_len = max(len_a, len_b)

        # Left-pad with 1s
        a = [1] * (max_len - len_a) + a
        b = [1] * (max_len - len_b) + b

        result = []
        for da, db in zip(a, b):
            if da == db:
                result.append(da)
            elif da == 1:
                result.append(db)
            elif db == 1:
                result.append(da)
            else:
                raise ValueError(f"Shapes {shape_a} and {shape_b} are not broadcastable")
        return result

    def validate_supported_ops(self, dag_model: DAGBasedModel) -> None:
        """Fail loudly if any node's op_type has no runtime implementation.

        The runtime used to `continue` past an unknown op (core/runtime.cpp),
        registering no output tensor for it — so a missing operator produced
        wrong output (a downstream consumer reads a null/misaligned input, or a
        graph output keeps a previous round's bytes) with every test still
        green. Catching it here, at conversion time, is what makes an operator
        gap attributable instead of silent.

        `shader_map.runtime_op_names()` is the accepted set: exactly the strings
        Ops.hpp's convert_opstring_to_enum resolves to a real OpType, host-only
        ops included, archive-only aliases (Topk/Maxpool2d) excluded.
        """
        allowed = shader_map.runtime_op_names()
        counts = {}
        for node in dag_model.nodes.values():
            if node.op_type not in allowed:
                counts[node.op_type] = counts.get(node.op_type, 0) + 1
        if not counts:
            return
        detail = ", ".join(f"{op} x{n}" for op, n in sorted(counts.items()))
        hint = ""
        control_flow = sorted(op for op in counts if op in _CONTROL_FLOW_OPS)
        if control_flow:
            hint = (
                " Control flow (" + ", ".join(control_flow) + ") has no runtime "
                "support: its body is a GRAPH attribute vkop cannot lower — fold "
                "it to constants at export time."
            )
        raise UnsupportedOpError(
            f"vkop: graph uses {len(counts)} op type(s) with no runtime kernel: "
            f"{detail}. These used to be silently skipped at load, producing "
            f"wrong output; conversion refuses them instead." + hint
        )

    def apply_optimizations(self, dag_model: DAGBasedModel, args):
        """Apply all optimizations to the model."""
        self.initializer_merger.move_input_tensor_to_attr(dag_model)
        self.initializer_merger.merge_initializers(dag_model)
        self.initializer_merger.convert_flat_to_reshape(dag_model)
        self.initializer_merger.remove_redundant_reshape(dag_model)

        # Save pre-optimize producer map for orphan cleanup after fusion
        # passes may delete nodes without updating all consumers.
        pre_producer = {}
        for n in dag_model.nodes.values():
            out_name = n.outputs[0]["name"] if isinstance(n.outputs[0], dict) else n.outputs[0]
            pre_producer[out_name] = n

        # The --exclude-* patterns are the exporter's node names, and fusion
        # rewrites those (fuse_into_attention replaces several MatMuls with one
        # synthesized node), so the resolution has to happen while the names a
        # user can actually read still exist. The weights it returns keep those
        # names through fusion -- a fused node re-points at the same
        # initializers -- which is what lets the set be computed here and used
        # at the -q dispatch below.
        excluded_weights = exclusion.resolve(
            dag_model,
            getattr(args, "exclude_nodes", None),
            getattr(args, "exclude_scopes", None),
            getattr(args, "exclude_downstream_of", None),
        )

        self.fusion_optimizer.optimize(
            dag_model, skip=_fusions_the_target_cannot_run(args))

        # The Conv weight-image layout is a converter decision now, so assert it
        # here rather than reading it back off the runtime's guess.
        self._validate_conv_weight_layout(dag_model)

        # Post-fusion orphan cleanup: some fusion passes delete nodes but miss
        # updating all consumers (e.g. Unsqueeze eliminated by pass A but still
        # consumed by Concat which pass B forgot). Walk all inputs, find orphans,
        # and if their former producer was a passthrough (Unsqueeze/Identity/
        # Squeeze), rewrite the consumer input to skip the eliminated node.
        self._cleanup_orphan_inputs(dag_model, pre_producer)

        dag_model.build_dependencies()

        # The ORT reference graph is cut here: every fusion has run, and
        # nothing below has touched a weight's value or an activation's width
        # yet (FP16 domain, quantization, KnWeightLayout, RGBA/unify all
        # follow). See ort_lower.py.
        ort_path = getattr(args, "emit_ort_onnx", None)
        if ort_path:
            try:
                from .ort_lower import lower_dag, save_lowered
            except ImportError:
                from ort_lower import lower_dag, save_lowered
            lowered = lower_dag(
                dag_model, fp32=not getattr(args, "ort_keep_dtype", False))
            save_lowered(lowered, ort_path)
            print(f"[ort] wrote {ort_path}: {len(lowered.graph.node)} nodes, "
                  f"{len(lowered.graph.output)} outputs")
            del lowered

        # A graph from outside can arrive with its Linear weights already on a
        # quantization grid (DequantizeLinear over a payload initializer). Fold
        # those into the same weight-only inputs the -q path emits, BEFORE -q
        # runs: otherwise a `-q int8` conversion of such a graph re-quantizes the
        # dequantized float (a second round of error) and a plain conversion
        # fails on an operator the runtime does not have.
        PreQuantizedWeights.fold(dag_model)

        # What each -q format costs the weights it is applied to, gathered while
        # the quantizers hold both sides of the round trip and printed once when
        # they are done (numeric_audit.py). Report-only: a weight the format is
        # bad at still computes, and refusing the conversion would set the
        # model's accuracy budget without being asked to.
        numeric_audit.begin()

        # Activations default to FP16, whatever the weight format.
        #
        # The FP16 domain is a property of the GRAPH, not of the weight
        # encoding: which ops can run half-typed and where a Cast is needed is
        # decided by the data flow alone. Weight-only formats (-q int4/int8/
        # fp8/nf4/...) leave the activations exactly as the export declared
        # them, so an fp32-traced export spends 2x the activation VRAM it needs
        # -- and on a card that cannot hold it, does not run at all. Measured on
        # MiniMax-H3's denoiser: ~58 MiB of activation per sequence row, i.e.
        # ~60 GB at the real 1024-row geometry, of which half was pure width.
        #
        # So the domain pass runs for every float weight format; only an
        # explicit request to keep the export's width (--act fp32, which
        # -q fp32 implies) skips it. The activation width is its own knob
        # (activation_dtype / resolve_precision), not a -q value. A
        # REFUSED pass is reported and the conversion continues: for a
        # weight-only format, leaving the graph at its exported width is the
        # artifact the user already had. -q fp16 turns the refusal into an
        # error below, because there the flip IS the request.
        act = activation_dtype(args)
        want_fp16_domain = act != "fp32"
        bf16_only = None
        fp16_summary = {"converted": 0, "declared": 0, "skipped_backend": False,
                        "refused": False, "out_of_range": [], "domain": set(),
                        "forced": set(), "blockers": {}, "conflicts": []}
        if want_fp16_domain:
            # The weight-only passes compute their scale and codes from the
            # stored values, so the domain pass must not re-round those bytes
            # first: one code on a boundary moves (33 -> 34 measured in
            # test_kn_weight_layout) and the streamed and in-memory payloads
            # stop agreeing. The DECLARATION still flips -- the op runs fp16 --
            # and the quantizer replaces the payload outright.
            skip = set()
            if args.quant == "bf16":
                # Same contract as the byte formats: the domain pass flips the
                # declaration, the bf16 pass below rounds the payload ONCE
                # from its source width.
                skip = bf16_weights.collect_names(dag_model,
                                                  exclude=excluded_weights)
                bf16_only = set(skip)
            elif args.quant not in ("fp16", "fp32", None):
                skip = self.quantizer.collect_quantizable_names(
                    dag_model, args.quant,
                    group_size=getattr(args, "group_size", 64),
                    exclude=excluded_weights)
            fp16_summary = GraphFp16.apply(
                dag_model, _declared_widths_backend(args),
                skip_weights=skip,
                half=act,
                # Under --act bf16 a MatMul whose B a byte/4-bit quantizer
                # will replace has no bf16-activation kernel (the runtime
                # refuses a quantized operand beside bf16), so the domain
                # pass keeps it FP32. -q bf16's own weights are bf16 and fine.
                quantized=skip if args.quant != "bf16" else set())
        elif (act == "fp32"
              and _declared_widths_backend(args) == "buffer"):
            # --act fp32 keeps the export's widths, so a node the export already
            # declares with FP16 and FP32 operands side by side reaches the
            # runtime as-is -- and runs one shader picked from its data input
            # over both. Measured on a ModelOpt NVFP4 export (Cast(to=fp16) on
            # the GEMM activation, FLOAT output): converted, then all-NaN.
            # The image backend builds everything fp16 and ignores the
            # declarations, so only the buffer backend is checked.
            conflicts = GraphFp16.mixed_width_conflicts(dag_model)
            if conflicts:
                for line in conflicts[:10]:
                    print(f"[fp32-widths]   {line}", file=sys.stderr)
                raise RuntimeError(
                    f"onnx2vkop: --act fp32 / -q fp32 keeps the export's widths, and "
                    f"{len(conflicts)} node(s) declare FP16 and FP32 operands "
                    f"in one op, which the runtime cannot execute (one shader "
                    f"per node, picked from the data input's dtype). First: "
                    f"{conflicts[0]}. Convert with --act fp16/bf16 (the half domain "
                    f"pass reconciles the widths) or fix the export's dtype "
                    f"coherence.")

        if args.quant == "fp16":
            if excluded_weights:
                print("[exclude] WARNING: -q fp16 ignores the exclusion "
                      "flags. A per-node 'stay fp32' verdict needs the same "
                      "fp32-domain propagation quantize_to_fp16_selective runs "
                      "for the ops that can only be right at full width -- one "
                      "wide weight beside fp16 activations is a Cast node, not "
                      "a precision knob, and fp16_graph.py refuses the rewrite "
                      "rather than shipping one.", file=sys.stderr)
            # The domain pass already ran above (it runs for every float
            # format); here only the refusal is fatal, because for -q fp16 the
            # flip is the request rather than an optimization beside it.
            #
            # A REFUSED domain pass must not fall through to the weight pass
            # below. The domain pass declines when some node would end up with
            # an FP32 operand beside an FP16 one -- a width mix one shader
            # cannot execute, because the runtime picks each node's build from
            # its data input. Weight-only quantization then halves the weights
            # anyway, which is exactly that mix, and the artifact ships with a
            # dtype the runtime rejects at execute time. Measured on the
            # MiniMax-H3 text encoder: the fp16 `attention_bias` reaches its
            # first op through a Cast(to=fp32), that Cast seeds an FP32 island,
            # the pass refused with 250 conflicts, and the old silent fallback
            # produced `Attention: the attention bias must have Q's element
            # type (Q is float16, the bias is float32)` on every run.
            if fp16_summary.get("refused"):
                raise RuntimeError(
                    "onnx2vkop: -q fp16 refused this graph -- the FP16 domain "
                    "pass found nodes that would mix FP32 and FP16 operands, "
                    "which the runtime cannot execute (one shader per node, "
                    "picked from the data input's dtype). Converting anyway "
                    "would ship exactly that mix. The conflict lines above "
                    "name the nodes; the usual cause is an activation chain "
                    "traced in one width while its inputs or weights are "
                    "declared in the other, so fix the export's dtype "
                    "coherence (or convert --act fp32) rather than forcing it.")
            # Then the weight pass, for what the domain pass cannot own: an
            # export that already declares FP16 activations but ships its
            # weights FP32. On a graph the domain pass just flipped it finds
            # every weight FP16 already and says so. The fp16 weight pass needs
            # the target backend too: only the buffer backend creates tensors
            # with the dtype the file declares, so only there can a "this op
            # runs fp32" verdict be read off the graph.
            self.quantizer.quantize_to_fp16_selective(
                dag_model, _declared_widths_backend(args)
            )
        elif args.quant == "int8":
            self.quantizer.quantize_to_int8_weight_only(
                dag_model, exclude=excluded_weights
            )
        elif args.quant in ("int4", "nf4"):
            self.quantizer.quantize_to_4bit_weight_only(
                dag_model, getattr(args, "group_size", 64), args.quant,
                exclude=excluded_weights,
            )
        elif args.quant in ("fp8e4m3", "fp8e5m2"):
            self.quantizer.quantize_to_fp8_weight_only(
                dag_model, args.quant, exclude=excluded_weights
            )
        elif args.quant == "nvfp4":
            self.quantizer.quantize_to_nvfp4_weight_only(
                dag_model, exclude=excluded_weights
            )
        elif args.quant == "awq":
            # AWQ is a real-valued weight transform that pairs with any low-bit
            # format; --awq-format picks which. The pass itself is in
            # quantize_to_awq_weight_only, which runs the plain quantizer for
            # that format restricted to the weights it scaled.
            self.quantizer.quantize_to_awq_weight_only(
                dag_model,
                getattr(args, "group_size", 64),
                getattr(args, "awq_format", "int4"),
                getattr(args, "awq_stats", None),
                getattr(args, "awq_alpha", 0.5),
                getattr(args, "awq_grid", 20),
                exclude=excluded_weights,
            )
        elif args.quant == "bf16":
            # Weight-only bf16 for the MatMul/Gemm B and Gather table slots;
            # see bf16_weights.py. Run below together with --keep-bf16.
            pass
        elif excluded_weights:
            # No -q at all (or a value the byte passes do not answer to): the
            # exclusion resolved to weights that no quantizer was going to
            # touch, so the flags changed nothing about this model.
            print(
                f"[exclude] WARNING: -q {args.quant!r} runs no weight "
                f"quantizer, so the {len(excluded_weights)} excluded weight(s) "
                "were already staying at their source dtype. The flags only "
                "select anything for -q int8/fp8e4m3/fp8e5m2/int4/nf4/nvfp4/"
                "awq/bf16.", file=sys.stderr)

        # bf16 weights (opt-in). After every weight quantizer, so a weight a
        # byte format took is no longer a float and is left alone; before
        # KnWeightLayout and the layout audit, which only key on byte dtypes.
        # --keep-bf16 restores the source bits of an unmodified bf16 weight in
        # a qualifying slot (lossless: anything else keeps today's payload);
        # -q bf16 additionally rounds every other qualifying float weight.
        keep_bf16 = bool(getattr(args, "keep_bf16", False))
        if keep_bf16 or args.quant == "bf16":
            if getattr(args, "unify", False) or getattr(args, "rgba", False):
                raise ValueError(
                    "onnx2vkop: --keep-bf16 / -q bf16 cannot be combined with "
                    "--unify or --rgba: both repack weight payloads, and the "
                    "bf16 weight contract covers only a plain MatMul/Gemm B or "
                    "Gather table initializer.")
            bf16_weights.apply(
                dag_model, keep=True, quantize=args.quant == "bf16",
                exclude=excluded_weights, only=bf16_only)

        if os.environ.get("VKOP_H3_FUSED_ATTENTION") == "1":
            if act != "bf16" or _declared_widths_backend(args) != "buffer":
                raise ValueError(
                    "VKOP_H3_FUSED_ATTENTION=1 requires --act bf16 and "
                    "the buffer backend")
            if getattr(args, "unify", False) or getattr(args, "rgba", False):
                raise ValueError(
                    "VKOP_H3_FUSED_ATTENTION=1 does not support --unify "
                    "or --rgba")
            from .h3_attention import apply as apply_h3_attention
            count = apply_h3_attention(dag_model)
            print(f"[h3-attention] fused {count} BF16 attention patterns "
                  "(approximate; validate output before use)")

        if act == "bf16" and _declared_widths_backend(args) == "buffer":
            # The runtime refuses a node binding bf16 beside fp16/fp32 (its
            # _bf16 module decodes every 16-bit operand as bf16). Weight
            # passes run after the domain pass (-q fp16's selective weight
            # pass writes float16), so the check is made here, on the
            # finished graph, rather than left for LoadModel to find. An fp16
            # island behind a Cast is legal (cast.comp converts all three
            # widths); only a per-node mix outside the own-format slots
            # (fp16_graph._BF16_OTHER_FLOAT_SLOTS) is refused.
            mixed = GraphFp16.mixed_half_kinds(dag_model)
            if mixed:
                raise RuntimeError(
                    "onnx2vkop: --act bf16 refused: the converted graph mixes "
                    "fp16 and bf16 (" + mixed[0] + "). vkop refuses a node "
                    "that binds bf16 beside another float width. Use "
                    "--act fp16, or a weight format other than -q fp16.")
        numeric_audit.finish()

        # The folded-weight ABI, checked once for every format.
        #
        # `_match_weight_only` refuses to quantize a weight that belongs to a
        # fold, but it is only one of the ways a weight can arrive on a
        # quantization grid: PreQuantizedWeights.fold imports somebody else's
        # payload, and a streamed weight is encoded by weight_stream. Each of
        # those ends up in a MatMul whose kernel predicts a float B with a
        # fixed input count, so a disagreement has to be caught here rather
        # than at the first execute on the target box.
        Quantizer._validate_folded_weights(dag_model)

        # Byte weights are re-laid as [K, N] with transB=0 here, so the runtime
        # reaches the WIDE form of the split-K byte GEMV: 16 consecutive columns
        # per lane instead of four columns a whole K row apart. Quantization has
        # already fixed each weight's element type (this pass keys on the
        # payload dtype), and nothing below reads a weight's values, so the
        # payload can still be a deferred one whose bytes are produced at write
        # time. KnWeightLayout explains the decision rule and why the transpose
        # follows the quantize rather than preceding it.
        KnWeightLayout.apply(dag_model)

        # Every consumer-driven weight layout is settled by now: the fusion
        # passes that fold a Transpose or a Q/DQ pair have re-pointed each
        # consumer at its real weight, quantization has fixed that weight's
        # element size, and KnWeightLayout has chosen the byte weights' physical
        # orientation. Run the layout audit in this window -- RGBA below is the
        # one remaining pass that rewrites a 4-D initializer's dims (into its
        # packed form), and an audit needs the logical rank.
        self._audit_weight_layouts(dag_model)

        # int8 x int8 DP4A (opt-in): re-point eligible MatMul A inputs from
        # their DequantizeLinear to the QuantizeLinear that produced the int8
        # tensor, and record the activation scale. Runs after the weight
        # quantization so "int8 weight" is a settled fact, and before RGBA /
        # Unifier, which renumber initializers. See W8A8ActivationFold for the
        # gates; a graph that matches nothing is left exactly as it was.
        if getattr(args, "w8a8", False):
            W8A8ActivationFold.fold(dag_model)

        if getattr(args, "rgba", False):
            RGBAConverter.convert(dag_model)

        if getattr(args, "unify", False):
            Unifier.unify(dag_model)

        # Phase 4-r2 形状链 host 化：把 Reshape target shape 的 int64 链
        # （Shape->Gather->Concat）提取成 host 表达式，写进 vkop_host_shape
        # 属性。转换失败的 Reshape 保持原两输入形态回退旧路径。死链节点由
        # save_to_binary 的 liveness 剪枝删除。
        from .host_shape import extract_host_shape_expressions
        hs = extract_host_shape_expressions(dag_model)
        if hs["converted"]:
            detail = " ".join(f"{k}={v}" for k, v in
                              sorted(hs.get("per_op", {}).items()))
            print(f"[host_shape] {hs['converted']}/{hs['candidates']} target "
                  f"shapes extracted to host expressions ({detail})")
            # 重跑本次提取前 optimizer 循环已消耗过一次 prune
            # （_prune_materialize_done 守卫会把 save_to_binary 里的再调用
            # 变成空操作）——重置守卫，让孤儿化的形状链在写盘前被剪掉。
            dag_model._prune_materialize_done = False

        # Final gate: any op the runtime cannot execute fails the conversion
        # here (see validate_supported_ops) instead of being skipped at load.
        # Runs last so it sees the post-fusion, post-prune graph.
        self.validate_supported_ops(dag_model)

    @staticmethod
    def _validate_conv_weight_layout(dag_model):
        """Assert the Conv weight-image layout the converter decided.

        ConvWeightLayoutPass writes weight_transpose / weight_pack on every Conv
        whose weight is a constant; the runtime only reads them (core/runtime.cpp
        builds a name -> (transpose, pack) map at load). Two ways that contract
        can break, both of which the runtime would trust blindly:

          - a Conv with a constant weight that carries neither attribute: the
            pass was skipped, or a future pass created the Conv after it. The
            runtime still has the old rank-gated geometry fallback, so this is
            recoverable, but it means the file's layout was not decided by the
            converter and the invariant is no longer testable.
          - an attribute that contradicts the weight's rank: a non-4-D weight has
            no folded addressing and no correct batch/chan transpose, and
            transpose=1 there makes Tensor::getGPUShape invent a batch/chan.

        Both fail the conversion. A 5th-dimensional Conv weight is also
        unaddressable by Conv2dImage::execute, which indexes weight_shape[0..3].

        Skipped entirely when the pass was ablated (VKOP_NO_FUSIONS): then no
        Conv carries an attribute and the runtime's fallback is the only layout.
        """
        if "annotate_conv_weight_layout" in _disabled_passes("VKOP_NO_FUSIONS"):
            return
        for node in dag_model.nodes.values():
            if node.op_type not in ("Conv", "Conv2d") or len(node.inputs) < 2:
                continue
            init = dag_model.initializers.get(node.inputs[1]["name"])
            if init is None:
                continue  # non-constant weight: no image, no attribute
            rank = len(init.dims)
            expected = 1 if rank == 4 else 0
            attrs = node.attributes or {}
            for key in ("weight_transpose", "weight_pack"):
                if key not in attrs:
                    raise UnsupportedOpError(
                        f"Conv '{node.name}' has a constant {rank}-D weight but "
                        f"no '{key}' attribute -- ConvWeightLayoutPass did not "
                        f"annotate it (was it ablated, or did a pass create this "
                        f"Conv after the annotate step?)"
                    )
                if int(attrs[key]) != expected:
                    raise UnsupportedOpError(
                        f"Conv '{node.name}': {key}={attrs[key]} contradicts its "
                        f"{rank}-D weight (expected {expected}); the image conv "
                        f"cannot address that layout"
                    )

    @staticmethod
    def _audit_weight_layouts(dag_model):
        """Report any weight whose stored layout costs its kernel speed.

        Called once, from apply_optimizations, in the window after the fusion
        passes and quantization have settled every weight's consumer and element
        size but before RGBA rewrites a 4-D initializer's dims. The audit itself
        is report-only (layout_audit.py explains the three checks and why each
        one is a line of a kernel); VKOP_LAYOUT_AUDIT=strict is the CI mode that
        turns a hazard into a conversion failure, which is opt-in on purpose --
        a weight in a slow layout still computes correctly, and a performance
        hazard must not decide whether a model converts.

        The structural half of the contract is a different question and stays
        fail-closed in _validate_conv_weight_layout above.

        Not an OptimizationPass: the MultiRoundOptimizer calls every pass's
        apply() once per round and swallows the exceptions it raises
        (optimizer.py:1007-1016), so a check placed there would both repeat and
        be silently skipped.
        """
        audit = layout_audit.report_layout_audit(dag_model)
        if audit.hazards and layout_audit.layout_audit_mode() == "strict":
            detail = "\n  ".join(audit.hazard_lines())
            raise UnsupportedOpError(
                f"layout audit (VKOP_LAYOUT_AUDIT=strict): {len(audit.hazards)} "
                f"weight(s) sit in a layout their kernel's fast path cannot "
                f"read:\n  {detail}"
            )
        return audit

    @staticmethod
    def _cleanup_orphan_inputs(dag_model, pre_producer):
        """Fix consumer nodes that reference tensors whose producer was deleted.

        Iterates until convergence since one round of replacements can expose
        additional orphans (e.g. replacing Expand_output with Expand_input
        where Expand_input is also an orphan pointing to a deleted Unsqueeze).
        """
        passthrough_ops = {"Unsqueeze", "Identity", "Squeeze", "Transpose",
                           "Cast", "Reshape", "Expand"}
        total_fixed = 0
        for _ in range(5):  # max 5 rounds
            live = set()
            for n in dag_model.nodes.values():
                for o in n.outputs:
                    live.add(o["name"] if isinstance(o, dict) else o)
            for i in dag_model.inputs:
                live.add(i["name"])
            for k in dag_model.initializers.keys():
                live.add(k)

            fixed = 0
            for n in list(dag_model.nodes.values()):
                new_inputs = []
                changed = False
                for inp in n.inputs:
                    if isinstance(inp, dict):
                        iname = inp.get("name", "")
                    else:
                        iname = inp
                    if iname and iname not in live and iname in pre_producer:
                        prod = pre_producer[iname]
                        if prod.op_type in passthrough_ops and len(prod.inputs) >= 1:
                            new_inputs.append(prod.inputs[0])
                            changed = True
                            fixed += 1
                            continue
                    new_inputs.append(inp)
                if changed:
                    dag_model.nodes[n.name] = Node(
                        op_type=n.op_type, name=n.name,
                        attributes=dict(n.attributes),
                        inputs=new_inputs, outputs=list(n.outputs),
                    )
            total_fixed += fixed
            if fixed == 0:
                break
        if total_fixed:
            print(f"Cleaned up {total_fixed} orphan inputs (passthrough producer elimination)")
