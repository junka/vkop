"""Main model converter implementation."""

from collections import defaultdict
import os
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
    )
    from . import shader_map
    from . import weight_stream
    from . import layout_audit
    from .optimizer import (
        FusionOptimizer,
        InitializerMerger,
        ONNXOptimizer,
        PreQuantizedWeights,
        Quantizer,
        RGBAConverter,
        Unifier,
        _disabled_passes,
    )
except ImportError:
    from dag import (
        DAGBasedModel,
        Node,
        _DATA_TYPE_MAP,
        _VKOP_CAUSAL_MASK_META_PREFIX,
        _VKOP_LOGITS_LAST_ROW_META_PREFIX,
    )
    import shader_map
    import weight_stream
    import layout_audit
    from optimizer import (
        FusionOptimizer,
        InitializerMerger,
        ONNXOptimizer,
        PreQuantizedWeights,
        Quantizer,
        RGBAConverter,
        Unifier,
        _disabled_passes,
    )


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


def _downcast_bfloat16(model) -> int:
    """Rewrite every bfloat16 tensor in the model to float16, in place.

    Covers the three places a dtype is spelled: an initializer's payload, a
    ValueInfoProto's element type (graph input/output/value_info), and a node
    attribute holding a tensor (Constant). Returns how many were converted."""
    count = 0
    for init in model.graph.initializer:
        if init.data_type == _BFLOAT16:
            init.raw_data = _bf16_bytes_to_fp16(init.raw_data)
            init.data_type = _FLOAT16
            count += 1
    for vi in (list(model.graph.input) + list(model.graph.output)
               + list(model.graph.value_info)):
        if vi.type.tensor_type.elem_type == _BFLOAT16:
            vi.type.tensor_type.elem_type = _FLOAT16
            count += 1
    for node in model.graph.node:
        for attr in node.attribute:
            if (attr.type == onnx.AttributeProto.TENSOR
                    and attr.t.data_type == _BFLOAT16):
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
                         rgba: bool = False) -> DAGBasedModel:
        """Parse and convert ONNX model to DAG format."""
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
            keep_in_file=moe_slots)

        # Normalize bf16 away before anything reads a dtype: the runtime has no
        # bf16 kernel and no bf16 storage type, so a model that spells it has to
        # become fp16 here or the loader rejects it at the format gate. Runs
        # before optimize_model so the ConstantFolder and every fusion pass only
        # ever see the fp16 form.
        bf16_count = _downcast_bfloat16(model)
        if bf16_count:
            print(f"[bf16] downcast {bf16_count} bfloat16 tensor(s) to float16")

        print("Optimizing ONNX model...")
        model = self.optimizer.optimize_model(model, batch_size,
                                              keep_alive=moe_slots)

        # Do NOT onnx.save here — llm.onnx weights exceed protobuf's 2GB
        # single-file limit. The DAG converter reads model.graph directly
        # from the in-memory object.
        dag_model = DAGBasedModel()
        dag_model.moe_keep_alive = set(moe_slots)
        dag_model.moe_slots = moe_slots
        # The slot bytes were kept out of RAM by `prepare`; this validates what
        # it recorded and drops the external_data pointer the optimizer may have
        # re-attached, so no later walk hydrates the pool.
        if moe_slots:
            added, cleared = weight_stream.restore_paging_slots(
                model, base_dir, deferred_weights, moe_slots)
            held = sum(e["length"] for n, e in deferred_weights.items()
                       if n in moe_slots)
            print(f"[moe] {len(moe_slots) - added} slot(s) streamed by prepare, "
                  f"{added} built here, {cleared} external_data pointer(s) "
                  f"cleared; {held >> 20} MiB of experts stays on disk")
        # The streaming contract the parse step negotiated: names whose payload
        # never entered RAM, and where to read them from when the blob is
        # written. save_to_binary verifies it before writing anything.
        dag_model.deferred_weights = deferred_weights
        dag_model.weight_base_dir = base_dir
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

        # VKOP_LOGITS_LAST_ROW=<graph output names>: keep only the last position
        # of those outputs. A greedy decode reads logits[0, L-1, :] and nothing
        # else, while the LM head computes and stores all L rows — at L=1860
        # that is a 2505 ms MatMul (14% of the prefill, linear in M) writing a
        # 1.13 GB fp16 buffer nobody reads. Which rows a caller wants is a
        # *policy* the graph cannot show (`logits` has no in-graph consumer),
        # so it is declared, not inferred — same shape as the causal mask two
        # blocks up: a model-level metadata prop is the carrier, the env var
        # the bridge for an .onnx that cannot be re-exported. "*" means every
        # graph output a MatMul produces. A graph whose export already contains
        # the last-row Slice needs neither (optimizer.match_logits_last_row
        # pushes it through the MatMul on its own).
        for entry in getattr(model, "metadata_props", []):
            if not entry.key.startswith(_VKOP_LOGITS_LAST_ROW_META_PREFIX):
                continue
            if entry.value in ("", "0"):
                continue
            dag_model.logits_last_row.add(
                entry.key[len(_VKOP_LOGITS_LAST_ROW_META_PREFIX) :]
            )
        for name in os.environ.get("VKOP_LOGITS_LAST_ROW", "").split(","):
            name = name.strip()
            if name:
                dag_model.logits_last_row.add(name)
        if dag_model.logits_last_row:
            print(
                "Logits last row only:",
                ", ".join(sorted(dag_model.logits_last_row)),
            )

        graph = model.graph
        try:
            # 大模型（>2GB 权重）full_check 内部会 SerializeToString 整个 model
            # 触发 proto 上限，只能退化为逐节点形状检查（结构性校验放后面 DAG 构建做）。
            onnx.checker.check_model(model, full_check=True)
            print("ONNX model passed full validation.")
        except Exception:
            print("ONNX model full validation skipped (model too large for full_check).")

        if not self.optimizer.is_topologically_sortable(graph):
            print("Graph is not topologically sorted. Please sort it before proceeding.")
            return None

        # Add initializers
        for initializer in graph.initializer:
            dag_model.initializers[initializer.name] = initializer

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

        self.fusion_optimizer.optimize(dag_model)

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

        # A graph from outside can arrive with its Linear weights already on a
        # quantization grid (DequantizeLinear over a payload initializer). Fold
        # those into the same weight-only inputs the -q path emits, BEFORE -q
        # runs: otherwise a `-q int8` conversion of such a graph re-quantizes the
        # dequantized float (a second round of error) and a plain conversion
        # fails on an operator the runtime does not have.
        PreQuantizedWeights.fold(dag_model)

        if args.quant == "fp16":
            # The fp16 weight pass needs the target backend: only the buffer
            # backend creates tensors with the dtype the file declares, so only
            # there can a "this op runs fp32" verdict be read off the graph.
            self.quantizer.quantize_to_fp16_selective(
                dag_model, getattr(args, "target_backend", "buffer")
            )
        elif args.quant == "int8":
            self.quantizer.quantize_to_int8_weight_only(dag_model)
        elif args.quant in ("int4", "nf4"):
            self.quantizer.quantize_to_4bit_weight_only(
                dag_model, getattr(args, "group_size", 64), args.quant
            )
        elif args.quant in ("fp8e4m3", "fp8e5m2"):
            self.quantizer.quantize_to_fp8_weight_only(dag_model, args.quant)
        elif args.quant == "nvfp4":
            self.quantizer.quantize_to_nvfp4_weight_only(dag_model)

        # Every consumer-driven weight layout is settled by now: the fusion
        # passes that fold a Transpose or a Q/DQ pair have re-pointed each
        # consumer at its real weight, and quantization has fixed that weight's
        # element size. Run the layout audit in this window -- RGBA below is the
        # one remaining pass that rewrites a 4-D initializer's dims (into its
        # packed form), and an audit needs the logical rank.
        self._audit_weight_layouts(dag_model)

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
            print(f"[host_shape] {hs['converted']}/{hs['reshapes']} Reshape "
                  f"target shapes extracted to host expressions")
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
