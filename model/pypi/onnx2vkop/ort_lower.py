"""Lower a fused vkop DAG back to a plain ONNX graph ONNX Runtime can run.

Why this exists
---------------
The fusion passes rewrite the exporter's graph into vkop's own vocabulary:
ops ORT has never heard of (RMSNorm, FusedElemwise, the rotate-half
RotaryEmbedding, the vkop Attention), and standard ops carrying attributes that
change what they compute (Conv/Add `activation`, MatMul `transB` / `swiglu` /
`down_act` / `epi_*`, Softmax `nan_optimization`, LayerNormalization's 4th
"skip" input, MatMul B with fewer KV heads than A has heads, ...). Both kinds
make the converted graph unrunnable outside vkop, so the only per-node
reference used to be the *original* graph, whose node names stop existing the
moment a pass fuses them.

This module emits each vkop node as a standard-ONNX subgraph whose LAST node
writes the vkop node's own output tensor name. The result is a model in which
every tensor the runtime materializes exists under the same name, so

  * ORT(original graph) vs ORT(lowered graph) checks the *fusion passes*
    (no GPU involved): any difference is a pass that changed the math;
  * ORT(lowered graph) vs a vkop per-node dump checks the *kernels* node by
    node, including the fused ones.

Scope
-----
It runs on the DAG right after fusion (see ModelConverter.apply_optimizations):
before the FP16-domain pass, weight quantization and the weight re-layout
passes, so the weights are still the export's floats (or the pre-quantized
int8 a DequantizeLinear fold produced, which is dequantized here). By default
everything float is computed in fp32 (`fp32=True`): ORT's CPU EP lacks fp16
kernels for many ops, and an fp32 run is the ideal reference vkop's fp16
result is measured against.

Ops with no ORT counterpart (LinearAttention / GatedDeltaNet, the Qwen3.5
causal Conv1d, EmbeddingForward, Nms) and the ring / fp8 KV cache are not
lowered yet; they raise LoweringError naming the node.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Sequence

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

__all__ = ["LoweringError", "lower_dag", "save_lowered", "upcast_model_to_fp32",
           "NODE_OUTPUTS_KEY"]

# metadata_props key of the lowered model: JSON {vkop node name: [output
# tensor names]}, which pairs a VKOP_NODE_DUMP file with its ORT tensor.
NODE_OUTPUTS_KEY = "vkop_node_outputs"


class LoweringError(RuntimeError):
    """A node the lowering cannot express in standard ONNX (yet)."""


_FLOAT_LOW = (TensorProto.FLOAT16, TensorProto.BFLOAT16)

_DTYPE_BY_NAME = {
    "float32": TensorProto.FLOAT, "uint8": TensorProto.UINT8,
    "int8": TensorProto.INT8, "uint16": TensorProto.UINT16,
    "int16": TensorProto.INT16, "int32": TensorProto.INT32,
    "int64": TensorProto.INT64, "string": TensorProto.STRING,
    "bool": TensorProto.BOOL, "float16": TensorProto.FLOAT16,
    "float64": TensorProto.DOUBLE, "uint32": TensorProto.UINT32,
    "uint64": TensorProto.UINT64, "bfloat16": TensorProto.BFLOAT16,
}

# Attributes that only steer how vkop executes a node, never what it computes.
# Dropped silently; any other attribute the ONNX schema does not know is an
# error, so a pass that adds a semantic attribute cannot be lowered wrong
# without someone noticing.
_EXEC_ONLY_ATTRS = {
    "weight_pack", "weight_transpose",           # ConvWeightLayoutPass
    "mask_causal", "flash", "flash_coop8", "kv_split",  # Attention speed hints
    "vkop_host_shape",                           # host_shape (post-fusion)
    "value_dynamic",
}

# Op names the runtime accepts as aliases of a standard op.
_ALIASES = {
    "GatedDeltaNet": "LinearAttention", "Conv1D": "Conv1d",
    "Conv2d": "Conv", "MaxPool2d": "MaxPool", "BatchNorm": "BatchNormalization",
    "InstanceNorm": "InstanceNormalization", "LayerNorm": "LayerNormalization",
    "Cumsum": "CumSum",
}

_NOT_LOWERABLE = {"EmbeddingForward", "Nms"}

_REDUCE_OPS = {
    "sum": "ReduceSum", "mean": "ReduceMean", "max": "ReduceMax",
    "min": "ReduceMin", "prod": "ReduceProd", "sum_square": "ReduceSumSquare",
    "l1_norm": "ReduceL1", "l2_norm": "ReduceL2", "log_sum": "ReduceLogSum",
    "log_sum_exp": "ReduceLogSumExp",
}

# FusedElemwise op codes (optimizer._FUSED_OP_CODES, shaders/buffer/
# fused_elemwise.comp).
_FE_BINARY = {1: "Add", 2: "Sub", 3: "Mul", 4: "Div", 5: "Pow", 15: "Min",
              16: "Max"}
_FE_UNARY = {6: "Sqrt", 7: "Sigmoid", 8: "Neg", 9: "Exp", 10: "Tanh",
             13: "Erf"}
_FE_CAST, _FE_SILU = 11, 14


def _tname(td) -> str:
    return td["name"] if isinstance(td, dict) else str(td)


def _tshape(td) -> Optional[List[int]]:
    s = td.get("shape") if isinstance(td, dict) else None
    if not isinstance(s, (list, tuple)) or len(s) == 0:
        return None
    return [int(d) if isinstance(d, (int, np.integer)) else -1 for d in s]


def _elem_type(dt) -> Optional[int]:
    if isinstance(dt, (int, np.integer)):
        return int(dt) or None
    if isinstance(dt, str):
        return _DTYPE_BY_NAME.get(dt)
    return None


def _to_float32_array(t: TensorProto) -> np.ndarray:
    if t.data_type == TensorProto.BFLOAT16:
        # numpy_helper returns bf16 as an ml_dtypes array (or raw uint16 bits on
        # older onnx); go through the bits, which both spell the same way.
        raw = np.frombuffer(t.raw_data, dtype=np.uint16) if t.raw_data else \
            np.asarray(t.int32_data, dtype=np.uint32).astype(np.uint16)
        return (raw.astype(np.uint32) << 16).view(np.float32).reshape(
            tuple(t.dims))
    return numpy_helper.to_array(t).astype(np.float32)


def _attr_value(v):
    """DAG attribute value -> something helper.make_attribute accepts."""
    if isinstance(v, np.ndarray):
        return numpy_helper.from_array(v)
    if isinstance(v, (bool, np.bool_)):
        return int(v)
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        return float(v)
    if isinstance(v, (list, tuple)):
        out = [_attr_value(x) for x in v]
        if out and all(isinstance(x, int) for x in out):
            return out
        if out and all(isinstance(x, (int, float)) for x in out):
            return [float(x) for x in out]
        return out
    return v


class _Lowerer:
    def __init__(self, dag, opset: int, fp32: bool, base_dir: Optional[str]):
        self.dag = dag
        self.opset = opset
        self.fp32 = fp32
        self.base_dir = base_dir
        self.nodes: List[onnx.NodeProto] = []
        self.group: List[int] = []      # vkop node index per emitted node
        self.cur_group = -1
        self.inits: Dict[str, TensorProto] = {}
        self.taken = set(dag.initializers)
        for n in dag.nodes.values():
            self.taken.update(_tname(o) for o in n.outputs)
            self.taken.update(_tname(i) for i in n.inputs)
        self.taken.update(_tname(i) for i in dag.inputs)
        self._uid = 0
        # (vkop node index, input dict) per vkop node input, for the rank/shape
        # reconciliation after lowering.
        self.expect: List[tuple] = []
        self.deferred = getattr(dag, "deferred_weights", None) or {}

    # ------------------------------------------------------------ utilities
    def fresh(self, hint: str) -> str:
        while True:
            self._uid += 1
            name = f"{hint}__ort{self._uid}"
            if name not in self.taken:
                self.taken.add(name)
                return name

    def emit(self, op, inputs, outputs=None, name=None, **attrs) -> str:
        if outputs is None:
            outputs = [self.fresh(op)]
        elif isinstance(outputs, str):
            outputs = [outputs]
        attrs = {k: v for k, v in attrs.items() if v is not None}
        self.nodes.append(helper.make_node(
            op, list(inputs), list(outputs), name=name or self.fresh(op),
            **attrs))
        self.group.append(self.cur_group)
        return outputs[0]

    def const(self, value, dtype=None, hint="c") -> str:
        arr = np.asarray(value, dtype=dtype)
        name = self.fresh(hint)
        self.inits[name] = numpy_helper.from_array(arr, name)
        return name

    def i64(self, values, hint="i") -> str:
        return self.const(np.asarray(values, dtype=np.int64), hint=hint)

    def f32(self, value, hint="f") -> str:
        return self.const(np.asarray(value, dtype=np.float32), hint=hint)

    def init_array(self, name: str) -> Optional[np.ndarray]:
        if name in self.deferred:
            raise LoweringError(
                f"weight '{name}' is streamed (its bytes never entered RAM); "
                f"convert with --stream-weights off to emit an ORT graph")
        t = self.dag.initializers.get(name)
        if t is None:
            return None
        if t.data_location == TensorProto.EXTERNAL:
            t2 = TensorProto()
            t2.CopyFrom(t)
            onnx.external_data_helper.load_external_data_for_tensor(
                t2, self.base_dir or ".")
            t = t2
        if t.data_type in _FLOAT_LOW or t.data_type == TensorProto.FLOAT:
            return _to_float32_array(t) if self.fp32 or \
                t.data_type == TensorProto.BFLOAT16 else numpy_helper.to_array(t)
        return numpy_helper.to_array(t)

    def use_init(self, name: str) -> None:
        """Copy a DAG initializer into the output model (fp32 if asked)."""
        if name in self.inits or name not in self.dag.initializers:
            return
        t = self.dag.initializers[name]
        if name in self.deferred:
            raise LoweringError(
                f"weight '{name}' is streamed (its bytes never entered RAM); "
                f"convert with --stream-weights off to emit an ORT graph")
        if self.fp32 and t.data_type in _FLOAT_LOW:
            self.inits[name] = numpy_helper.from_array(
                self.init_array(name), name)
        elif t.name != name:
            # The DAG is keyed by name; a pass may store a proto whose own
            # name field it never set.
            t2 = TensorProto()
            t2.CopyFrom(t)
            t2.name = name
            self.inits[name] = t2
        else:
            self.inits[name] = t

    # opset-dependent forms
    def unsqueeze(self, x, axes, out=None):
        if self.opset >= 13:
            return self.emit("Unsqueeze", [x, self.i64(axes)], out)
        return self.emit("Unsqueeze", [x], out, axes=list(axes))

    def squeeze(self, x, axes, out=None):
        if self.opset >= 13:
            return self.emit("Squeeze", [x, self.i64(axes)], out)
        return self.emit("Squeeze", [x], out, axes=list(axes))

    def reduce(self, op, x, axes, keepdims=1, out=None):
        axes_as_input = self.opset >= (13 if op == "ReduceSum" else 18)
        if axes is None or len(axes) == 0:
            return self.emit(op, [x], out, keepdims=keepdims)
        if axes_as_input:
            return self.emit(op, [x, self.i64(axes)], out, keepdims=keepdims)
        return self.emit(op, [x], out, axes=list(axes), keepdims=keepdims)

    def slice_(self, x, start, end, axis, out=None):
        return self.emit("Slice", [x, self.i64([start]), self.i64([end]),
                                   self.i64([axis])], out)

    def gather_scalar(self, x, index, axis, out=None):
        return self.emit("Gather", [x, self.const(np.int64(index))], out,
                         axis=axis)

    def shape_dim(self, x, axis):
        """1-D int64 tensor [x.shape[axis]]."""
        shp = self.emit("Shape", [x])
        return self.emit("Gather", [shp, self.i64([axis])], axis=0)

    def activation(self, x, act, out=None):
        if act in ("Relu", "Sigmoid", "Tanh"):
            return self.emit(act, [x], out)
        if act == "HardSwish":
            if self.opset >= 14:
                return self.emit("HardSwish", [x], out)
            hs = self.emit("HardSigmoid", [x], alpha=1.0 / 6, beta=0.5)
            return self.emit("Mul", [x, hs], out)
        if act == "Mish":
            sp = self.emit("Softplus", [x])
            return self.emit("Mul", [x, self.emit("Tanh", [sp])], out)
        if act == "Swish":
            return self.emit("Mul", [x, self.emit("Sigmoid", [x])], out)
        raise LoweringError(f"unknown fused activation '{act}'")

    def silu(self, x, out=None):
        return self.emit("Mul", [x, self.emit("Sigmoid", [x])], out)

    # ------------------------------------------------------------- driver
    def run(self, expose_all: bool):
        dag = self.dag
        dag.build_dependencies()
        order = dag.topological_sort()
        for gi, node in enumerate(order):
            self.cur_group = gi
            op = _ALIASES.get(node.op_type, node.op_type)
            for td in node.inputs:
                if _tname(td) in dag.initializers:
                    self.use_init(_tname(td))
            handler = getattr(self, f"_lower_{op}", None)
            try:
                if op in _NOT_LOWERABLE:
                    raise LoweringError(
                        f"{node.op_type} has no ONNX Runtime counterpart yet")
                if handler is not None:
                    handler(node, dict(node.attributes or {}))
                else:
                    self._lower_generic(node, op, dict(node.attributes or {}))
            except LoweringError as e:
                raise LoweringError(f"node '{node.name}' ({node.op_type}): {e}") \
                    from None
            for td in node.inputs:
                if _tname(td):
                    self.expect.append((gi, td))
        self.cur_group = -1

        inputs = []
        for td in dag.inputs:
            et = _elem_type(td.get("dtype")) or TensorProto.FLOAT
            if self.fp32 and et in _FLOAT_LOW:
                et = TensorProto.FLOAT
            shape = _tshape(td)
            dims = None if shape is None else [
                d if d >= 0 else f"{_tname(td)}_d{i}" for i, d in enumerate(shape)]
            inputs.append(helper.make_tensor_value_info(_tname(td), et, dims))

        # Element type the converter recorded per output tensor, for declaring
        # graph outputs shape inference cannot type.
        self.out_types = {}
        for node in order:
            for o in node.outputs:
                et = _elem_type(o.get("dtype")) if isinstance(o, dict) else None
                if et:
                    if self.fp32 and et in _FLOAT_LOW:
                        et = TensorProto.FLOAT
                    self.out_types[_tname(o)] = et
        # vkop node name -> its output tensor names: the runtime's per-node
        # dump (VKOP_NODE_DUMP) names files by node, ORT names tensors.
        self.node_outputs = {node.name: [_tname(o) for o in node.outputs]
                             for node in order}
        out_names = [_tname(o) for o in dag.outputs]
        if expose_all:
            seen = set(out_names)
            for node in order:
                for o in node.outputs:
                    n = _tname(o)
                    if n and n not in seen and n not in dag.initializers:
                        seen.add(n)
                        out_names.append(n)
        return inputs, out_names

    def finish(self, inputs, out_names, producer_name="onnx2vkop.ort_lower"):
        graph_inputs = {vi.name for vi in inputs}
        self._reconcile_shapes(inputs)
        produced = {o for n in self.nodes for o in n.output}
        names = [n for n in out_names if n in produced or n in graph_inputs]
        model = self._build(inputs, producer_name)
        _set_output_types(model, names, getattr(self, "out_types", {}))
        helper.set_model_props(model, {
            NODE_OUTPUTS_KEY: json.dumps(self.node_outputs,
                                         separators=(",", ":"))})
        return model

    def _build(self, inputs, producer_name="onnx2vkop.ort_lower"):
        used = {i for n in self.nodes for i in n.input}
        graph = helper.make_graph(self.nodes, "vkop_lowered", inputs, [],
                                  initializer=[t for k, t in self.inits.items()
                                               if k in used])
        model = helper.make_model(
            graph, opset_imports=[helper.make_opsetid("", self.opset)],
            producer_name=producer_name)
        model.ir_version = min(model.ir_version, 10)
        return model

    # ---------------------------------------------- rank/shape reconciliation
    def _reconcile_shapes(self, inputs, rounds: int = 64):
        """Make each vkop node's operands carry the shape it was converted with.

        Several passes alias a Squeeze/Unsqueeze/Reshape output to its input
        and let the consumer read the other rank (the runtime reads buffers
        flat), and the loader records GlobalAveragePool's output as [N, C].
        ONNX is strict about rank, so where the inferred producer shape and the
        shape the consumer's input dict declares disagree, insert a Reshape
        to the declared one in front of exactly that consumer's nodes.
        """
        done = set()
        for _ in range(rounds):
            inferred = _infer_shapes(self._build(inputs))
            if not inferred:
                return
            producer_group = {o: g for n, g in zip(self.nodes, self.group)
                              for o in n.output}
            cands = []
            for gi, td in self.expect:
                name, want = _tname(td), _tshape(td)
                have = inferred.get(name)
                if (gi, name) in done or want is None or have is None or \
                        not _mismatch(have, want):
                    continue
                cands.append((gi, name, have, want))
            if not cands:
                return
            # A fix changes shapes only at and after its own node, so in one
            # round take just the fixes whose tensor is produced upstream of
            # every node fixed in it; the rest wait for a fresh inference.
            # (Fixing all at once read the conv state's post-fix rank from the
            # pre-fix inference: Qwen3.5's state_out reshaped to [3,3].)
            g0 = min(c[0] for c in cands)
            fixes = {}                       # group -> [(name, nodes, new)]
            for gi, name, have, want in cands:
                if producer_group.get(name, -1) >= g0:
                    continue
                made = self._reshape_to(name, have, want)
                if made is None:
                    done.add((gi, name))
                    continue
                done.add((gi, name))
                fixes.setdefault(gi, []).append((name,) + made)
            if not fixes:
                continue
            new_nodes, new_groups, placed = [], [], set()
            for n, gi in zip(self.nodes, self.group):
                if gi in fixes:
                    if gi not in placed:
                        placed.add(gi)
                        for _, fnodes, _ in fixes[gi]:
                            new_nodes.extend(fnodes)
                            new_groups.extend([-1] * len(fnodes))
                    for name, _, new in fixes[gi]:
                        for j, i in enumerate(n.input):
                            if i == name:
                                n.input[j] = new
                new_nodes.append(n)
                new_groups.append(gi)
            self.nodes, self.group = new_nodes, new_groups

    def _reshape_to(self, name, have, want):
        """Nodes reshaping `name` (shape `have`, None = unknown dim) to `want`
        (-1 = dynamic). Returns (nodes, new_name) or None if not derivable."""
        before = len(self.nodes)
        new_name = self.fresh(name + "__as")
        if all(d >= 0 for d in want):
            shape = self.i64(want)
        else:
            src_nu = [i for i, d in enumerate(have) if d != 1]
            dst_nu = [i for i, d in enumerate(want) if d != 1]
            if len(src_nu) != len(dst_nu):
                return None
            pieces = []
            shp = None
            for i, d in enumerate(want):
                if d >= 0:
                    pieces.append(self.i64([d]))
                else:
                    if shp is None:
                        shp = self.emit("Shape", [name])
                    src_axis = src_nu[dst_nu.index(i)]
                    pieces.append(self.emit("Gather", [shp, self.i64([src_axis])],
                                            axis=0))
            shape = self.emit("Concat", pieces, axis=0)
        self.emit("Reshape", [name, shape], new_name)
        made = self.nodes[before:]
        del self.nodes[before:]
        del self.group[before:]
        return made, new_name

    # ------------------------------------------------------------ generic
    def _lower_generic(self, node, op, attrs):
        act = attrs.pop("activation", None)
        try:
            schema = onnx.defs.get_schema(op, self.opset, "")
        except Exception:
            raise LoweringError(f"'{op}' is not an ONNX op at opset {self.opset}")
        known = set(schema.attributes)
        out_attrs = {}
        for k, v in attrs.items():
            if k in known:
                out_attrs[k] = _attr_value(v)
            elif k not in _EXEC_ONLY_ATTRS:
                raise LoweringError(
                    f"attribute '{k}' is not part of ONNX {op} and the lowering "
                    f"does not know what it computes")
        inputs = [_tname(i) for i in node.inputs]
        outputs = [_tname(o) for o in node.outputs]
        if self.fp32:
            if op == "Cast" and out_attrs.get("to") in _FLOAT_LOW:
                out_attrs["to"] = TensorProto.FLOAT
            if op in ("Constant", "ConstantOfShape") and \
                    isinstance(out_attrs.get("value"), TensorProto) and \
                    out_attrs["value"].data_type in _FLOAT_LOW:
                out_attrs["value"] = numpy_helper.from_array(
                    _to_float32_array(out_attrs["value"]))
        if act is None:
            self.emit(op, inputs, outputs, name=node.name, **out_attrs)
        else:
            mid = self.fresh(outputs[0])
            self.emit(op, inputs, [mid] + outputs[1:], name=node.name,
                      **out_attrs)
            self.activation(mid, act, outputs[0])

    # standard ops whose inputs a converter pass moved into attributes
    def _lower_Resize(self, node, attrs):
        ins = [_tname(i) for i in node.inputs]
        scales = attrs.pop("scales", None)
        sizes = attrs.pop("sizes", None)
        x = ins[0]
        roi = ins[1] if len(ins) > 1 else ""
        rest = ins[2:]
        s_name = self.f32(scales) if scales is not None else \
            (rest.pop(0) if rest and sizes is not None and len(rest) > 1 else "")
        if scales is None and sizes is None and rest:
            s_name = rest.pop(0)
        z_name = self.i64(sizes) if sizes is not None else (rest.pop(0) if rest else "")
        node2 = _shallow(node, inputs=[{"name": x}, {"name": roi},
                                       {"name": s_name}, {"name": z_name}])
        self._lower_generic(node2, "Resize", attrs)

    def _lower_Pad(self, node, attrs):
        pads = attrs.pop("pads", None)
        ins = [dict(i) for i in node.inputs]
        if pads is not None and self.opset >= 11:
            ins = [ins[0], {"name": self.i64(pads)}] + ins[1:]
        elif pads is not None:
            attrs["pads"] = pads
        self._lower_generic(_shallow(node, inputs=ins), "Pad", attrs)

    def _lower_TopK(self, node, attrs):
        k = attrs.pop("K", None)
        ins = [dict(i) for i in node.inputs]
        if len(ins) < 2 and k is not None:
            ins.append({"name": self.i64([k])})
        self._lower_generic(_shallow(node, inputs=ins), "TopK", attrs)

    def _lower_Split(self, node, attrs):
        # The converter keeps the split sizes as an attribute (the opset < 13
        # spelling); from 13 on ONNX reads them from input 1.
        split = attrs.pop("split", None)
        ins = [dict(i) for i in node.inputs]
        if split is not None:
            if self.opset >= 13 and len(ins) < 2:
                ins.append({"name": self.i64(list(split))})
            elif self.opset < 13:
                attrs["split"] = split
        self._lower_generic(_shallow(node, inputs=ins), "Split", attrs)

    def _lower_GlobalAveragePool(self, node, attrs):
        self._lower_generic(node, "GlobalAveragePool", attrs)

    # --------------------------------------------------------- vkop ops
    def _lower_Reduce(self, node, attrs):
        op = _REDUCE_OPS.get(attrs.get("reduce_op"))
        if op is None:
            raise LoweringError(f"unknown reduce_op {attrs.get('reduce_op')!r}")
        axes = attrs.get("axes")
        if isinstance(axes, (int, np.integer)):
            axes = [int(axes)]
        self.reduce(op, _tname(node.inputs[0]), axes,
                    int(attrs.get("keepdims", 1)), _tname(node.outputs[0]))

    def _lower_RMSNorm(self, node, attrs):
        x, w = _tname(node.inputs[0]), _tname(node.inputs[1])
        eps = float(attrs.get("eps", 1e-6))
        axis = attrs.get("axis", -1)
        axes = list(axis) if isinstance(axis, (list, tuple)) else [int(axis)]
        sq = self.emit("Mul", [x, x])
        ms = self.reduce("ReduceMean", sq, axes, 1)
        rstd = self.emit("Reciprocal", [self.emit("Sqrt", [
            self.emit("Add", [ms, self.f32(eps)])])])
        y = self.emit("Mul", [x, rstd])
        self.emit("Mul", [y, w], _tname(node.outputs[0]), name=node.name)

    def _lower_LayerNormalization(self, node, attrs):
        ins = [dict(i) for i in node.inputs]
        if len(ins) >= 4:
            s = self.emit("Add", [_tname(ins[0]), _tname(ins[3])])
            ins = [{"name": s}] + ins[1:3]
        self._lower_generic(_shallow(node, inputs=ins), "LayerNormalization",
                            attrs)

    def _lower_Softmax(self, node, attrs):
        nan = attrs.pop("nan_optimization", None)
        repl = float(attrs.pop("nan_replacement_value", 0.0))
        if not nan:
            return self._lower_generic(node, "Softmax", attrs)
        mid = self.fresh(_tname(node.outputs[0]))
        self._lower_generic(_shallow(node, outputs=[{"name": mid}]), "Softmax",
                            attrs)
        self.emit("Where", [self.emit("IsNaN", [mid]), self.f32(repl), mid],
                  _tname(node.outputs[0]))

    def _lower_FusedElemwise(self, node, attrs):
        ops = list(attrs.get("ops", []))
        scalars = list(attrs.get("scalars", []))
        regs = {i: _tname(td) for i, td in enumerate(node.inputs)}
        consts = {}

        def operand(code):
            if code >= 100:
                idx = code - 100
                if idx not in consts:
                    consts[idx] = self.f32(np.float32(scalars[idx]), "fe_s")
                return consts[idx]
            if code not in regs:
                raise LoweringError(f"FusedElemwise reads unset register {code}")
            return regs[code]

        words = [ops[i:i + 4] for i in range(0, len(ops), 4)]
        if not words:
            raise LoweringError("FusedElemwise with an empty program")
        for wi, (oc, dst, a, b) in enumerate(words):
            last = wi == len(words) - 1
            out = _tname(node.outputs[0]) if last else None
            if oc in _FE_BINARY:
                r = self.emit(_FE_BINARY[oc], [operand(a), operand(b)], out)
            elif oc in _FE_UNARY:
                r = self.emit(_FE_UNARY[oc], [operand(a)], out)
            elif oc == _FE_SILU:
                r = self.silu(operand(a), out)
            elif oc == _FE_CAST:
                r = self.emit("Identity", [operand(a)], out)
            else:
                raise LoweringError(f"FusedElemwise op code {oc} not lowered")
            regs[dst] = r

    def _lower_RotaryEmbedding(self, node, attrs):
        x_td, cos_td, sin_td = node.inputs[0], node.inputs[1], node.inputs[2]
        x = _tname(x_td)
        xs = _tshape(x_td)
        comp = int(attrs.get("input_component", -1))
        if comp >= 0:
            x = self.gather_scalar(x, comp, axis=1)        # [S,C,H,D] -> [S,H,D]
            rank = 3
            hd = xs[-1] if xs else -1
        else:
            if xs is None:
                raise LoweringError("RotaryEmbedding needs X's rank")
            rank = len(xs)
            hd = xs[-1]
            if attrs.get("input_untransposed"):
                x = self.emit("Transpose", [x], perm=[0, 2, 1, 3])
        cs = _tshape(cos_td)
        rd = cs[-1] if cs and cs[-1] > 0 else hd
        if rd is None or rd <= 0:
            raise LoweringError("RotaryEmbedding needs a static rotary dim")
        if rank == 4:      # Y [B,H,S,D]; cos rows indexed (b*S + s)*rd + d
            tgt = self.emit("Concat", [self.i64([-1, 1]), self.shape_dim(x, 2),
                                       self.i64([rd])], axis=0)
        elif rank == 3:    # Y [S,H,D] (seq-major vision tower)
            tgt = self.emit("Concat", [self.shape_dim(x, 0), self.i64([1, rd])],
                            axis=0)
        else:
            raise LoweringError(f"RotaryEmbedding on a rank-{rank} X")
        cos = self.emit("Reshape", [_tname(cos_td), tgt])
        sin = self.emit("Reshape", [_tname(sin_td), tgt])
        partial = hd > 0 and rd < hd
        xr = self.slice_(x, 0, rd, -1) if partial else x
        x1 = self.slice_(xr, 0, rd // 2, -1)
        x2 = self.slice_(xr, rd // 2, rd, -1)
        rot = self.emit("Concat", [self.emit("Neg", [x2]), x1], axis=-1)
        out = _tname(node.outputs[0])
        y = self.emit("Add", [self.emit("Mul", [xr, cos]),
                              self.emit("Mul", [rot, sin])],
                      None if partial else out)
        if partial:
            self.emit("Concat", [y, self.slice_(x, rd, hd, -1)], out, axis=-1)

    def _repeat_kv(self, t, heads, kv_heads):
        """[B,Hkv,S,D] -> [B,H,S,D], head h reading kv head h // rep."""
        if heads == kv_heads:
            return t
        rep = heads // kv_heads
        u = self.unsqueeze(t, [2])
        e = self.emit("Expand", [u, self.i64([1, 1, rep, 1, 1])])
        shp = self.emit("Shape", [t])
        tgt = self.emit("Concat", [
            self.emit("Gather", [shp, self.i64([0])], axis=0),
            self.i64([heads, -1]),
            self.emit("Gather", [shp, self.i64([3])], axis=0)], axis=0)
        return self.emit("Reshape", [e, tgt])

    def _repeat_kv_like(self, t, q):
        """[B,Hkv,S,D] -> [B,H,S,D] with H = q.shape[1], all from runtime
        shapes; head h reads kv head h // (H / Hkv)."""
        ts, qs = self.emit("Shape", [t]), self.emit("Shape", [q])
        dim = lambda shp, i: self.emit("Gather", [shp, self.i64([i])], axis=0)
        h, hkv = dim(qs, 1), dim(ts, 1)
        rep = self.emit("Div", [h, hkv])
        u = self.unsqueeze(t, [2])
        e = self.emit("Expand", [u, self.emit("Concat", [
            self.i64([1, 1]), rep, self.i64([1, 1])], axis=0)])
        tgt = self.emit("Concat", [dim(ts, 0), h, self.i64([-1]), dim(ts, 3)],
                        axis=0)
        return self.emit("Reshape", [e, tgt])

    def _lower_Attention(self, node, attrs):
        if attrs.get("has_ring"):
            raise LoweringError("the sliding-window (ring) KV cache is not "
                                "lowered yet")
        if attrs.get("cache_scale") is not None:
            raise LoweringError("the fp8 KV cache is not lowered yet")
        heads = int(attrs.get("num_heads", -1))
        kv_heads = int(attrs.get("num_kv_heads", heads))
        scale = float(attrs.get("scale", 1.0))
        seq_major = bool(attrs.get("seq_major", 0))
        v_comp = int(attrs.get("v_component", -1))
        has_cache = bool(attrs.get("has_cache", 0))
        ins = list(node.inputs)
        q, k, v = (_tname(t) for t in ins[:3])
        q_rank = len(_tshape(ins[0]) or [0, 0, 0, 0])
        mask = None
        if has_cache:
            past = _tname(ins[3])
            extra = [t for t in ins[4:] if _tname(t) != "kv_ring_head"]
            mask = _tname(extra[0]) if extra else None
        else:
            mask = _tname(ins[3]) if len(ins) > 3 and _tname(ins[3]) else None

        def to_bhsd(t, rank):
            if not seq_major:
                return t
            if rank == 3:   # [S,H,D] -> [1,H,S,D]
                return self.unsqueeze(self.emit("Transpose", [t], perm=[1, 0, 2]),
                                      [0])
            return self.emit("Transpose", [t], perm=[0, 2, 1, 3])

        if v_comp >= 0:
            v = self.gather_scalar(v, v_comp, axis=1)     # [S,C,H,D] -> [S,H,D]
            v_rank = 3
        else:
            v_rank = len(_tshape(ins[2]) or [0] * q_rank)
        q = to_bhsd(q, q_rank)
        k = to_bhsd(k, len(_tshape(ins[1]) or [0] * q_rank))
        v = to_bhsd(v, v_rank)

        if has_cache:
            pk = self.gather_scalar(past, 0, axis=1)       # [1,Hkv,P,D]
            pv = self.gather_scalar(past, 1, axis=1)
            k = self.emit("Concat", [pk, k], axis=2)
            v = self.emit("Concat", [pv, v], axis=2)
            if len(node.outputs) > 1:
                self.emit("Concat", [self.unsqueeze(k, [1]),
                                     self.unsqueeze(v, [1])],
                          _tname(node.outputs[1]), axis=1)

        if heads > 0 and kv_heads > 0:
            k = self._repeat_kv(k, heads, kv_heads)
            v = self._repeat_kv(v, heads, kv_heads)
        else:
            # The fold records -1 when Q's head axis was dynamic at conversion
            # (Qwen3.5's gated q_proj); the runtime reads heads off Q's shape
            # at execute time, so the lowering does the same, in-graph.
            k = self._repeat_kv_like(k, q)
            v = self._repeat_kv_like(v, q)
        s = self.emit("MatMul", [q, self.emit("Transpose", [k],
                                              perm=[0, 1, 3, 2])])
        if scale != 1.0:
            s = self.emit("Mul", [s, self.f32(scale)])
        if mask is not None:
            s = self.emit("Add", [s, mask])
        if attrs.get("causal"):
            # keep j <= (kv - S) + i
            sq, kvl = self.shape_dim(q, 2), self.shape_dim(k, 2)
            ones = self.emit("ConstantOfShape", [self.emit("Concat", [sq, kvl],
                                                           axis=0)],
                             value=numpy_helper.from_array(np.ones(1, np.float32)))
            off = self.emit("Squeeze", [self.emit("Sub", [kvl, sq]),
                                        self.i64([0])])
            tri = self.emit("Trilu", [ones, off], upper=0)
            keep = self.emit("Cast", [tri], to=TensorProto.BOOL)
            s = self.emit("Where", [keep, s, self.f32(-np.inf)])
        p = self.emit("Softmax", [s], axis=-1)
        out = _tname(node.outputs[0])
        if seq_major:
            y = self.emit("MatMul", [p, v])
            self.emit("Transpose", [y], out, perm=[0, 2, 1, 3])
        else:
            self.emit("MatMul", [p, v], out, name=node.name)

    def _lower_Conv1d(self, node, attrs):
        """Qwen3.5's causal depthwise conv (ops/Conv1d.hpp): x [C,S] (or
        [1,C,S]), weight [C,K], state [C,K-1] holding the previous round's last
        K-1 samples. y[c,t] = sum_k w[c,k] * hist[c, t+k] over
        hist = concat(state, x); state_out = hist's last K-1 columns."""
        x, w, st = (_tname(t) for t in node.inputs[:3])
        k = int(attrs.get("kernel", 0))
        w_arr = self.init_array(w)
        if w_arr is not None:
            c = int(w_arr.shape[0])
            k = k or int(w_arr.shape[-1])
            w3 = self.const(np.ascontiguousarray(
                w_arr.reshape(c, 1, -1)).astype(np.float32), hint=w)
        else:
            ws = _tshape(node.inputs[1])
            if not ws or ws[0] <= 0:
                raise LoweringError("Conv1d needs a static channel count")
            c = ws[0]
            k = k or ws[-1]
            w3 = self.emit("Reshape", [w, self.i64([c, 1, -1])])
        if k <= 0:
            raise LoweringError("Conv1d needs its kernel size")
        x3 = self.emit("Reshape", [x, self.i64([1, c, -1])])
        s3 = self.emit("Reshape", [st, self.i64([1, c, k - 1])])
        hist = self.emit("Concat", [s3, x3], axis=-1)
        y = self.emit("Conv", [hist, w3], group=c, kernel_shape=[k])
        self.emit("Reshape", [y, self.emit("Shape", [x])],
                  _tname(node.outputs[0]))
        if len(node.outputs) > 1 and _tname(node.outputs[1]):
            tail = self.slice_(hist, -(k - 1), 2 ** 62, -1)
            self.emit("Reshape", [tail, self.emit("Shape", [st])],
                      _tname(node.outputs[1]))

    def _lower_LinearAttention(self, node, attrs):
        """The gated delta rule (ops/LinearAttention.hpp) as an ONNX Scan over
        the sequence, the recurrent state [H,Dk,Dv] carried in fp32.

        Layouts: rank 4 is the exporter's [1,S,H,D] (g/beta [1,S,H]); rank 3
        is [H,S,D] (g/beta [H,S]). q/k have Hk heads; value head h reads key
        head h // (H/Hk) -- repeat_interleave, done here before the loop."""
        if self.opset < 13:
            raise LoweringError("the LinearAttention Scan body needs opset >= 13")
        q, k, v, g, b, st = (_tname(t) for t in node.inputs[:6])
        qs = _tshape(node.inputs[0])
        vs = _tshape(node.inputs[2])
        if not qs or len(qs) not in (3, 4) or not vs:
            raise LoweringError("LinearAttention needs q/v ranks")
        rank4 = len(qs) == 4
        hk = qs[2] if rank4 else qs[0]
        h = int(attrs.get("heads", 0)) or (vs[2] if rank4 else vs[0])
        dk = int(attrs.get("dk", 0)) or qs[-1]
        dv = int(attrs.get("dv", 0)) or vs[-1]
        if min(hk, h, dk, dv) <= 0 or h % hk:
            raise LoweringError(f"LinearAttention: heads {h}/{hk}, dk {dk}, "
                                f"dv {dv} must be static and Hk | H")

        def seq_major(t, d):        # -> [S, heads, d] (or [S, heads] if d=0)
            if rank4:
                return self.squeeze(t, [0])
            return self.emit("Transpose", [t], perm=[1, 0, 2] if d else [1, 0])

        q3, k3 = seq_major(q, dk), seq_major(k, dk)
        v3 = seq_major(v, dv)
        g2, b2 = seq_major(g, 0), seq_major(b, 0)
        eps = self.f32(1e-6)

        def l2n(t):
            ss = self.reduce("ReduceSum", self.emit("Mul", [t, t]), [-1], 1)
            return self.emit("Mul", [t, self.emit("Reciprocal", [
                self.emit("Sqrt", [self.emit("Add", [ss, eps])])])])

        qn = self.emit("Mul", [l2n(q3), self.f32(1.0 / np.sqrt(dk))])
        kn = l2n(k3)
        if h != hk:
            rep = h // hk

            def expand(t):
                u = self.unsqueeze(t, [2])
                e = self.emit("Expand", [u, self.i64([1, 1, rep, 1])])
                return self.emit("Reshape", [e, self.i64([-1, h, dk])])
            qn, kn = expand(qn), expand(kn)
        st3 = self.emit("Reshape", [st, self.i64([h, dk, dv])])

        # Scan body: (state; q_t, k_t, v_t, g_t, b_t) -> (state'; out_t)
        n = self.fresh("la")
        S0, Q, K, V, G, B = (f"{n}_{x}" for x in ("s", "q", "k", "v", "g", "b"))
        c_ax1 = f"{n}_ax1"
        c_ax12 = f"{n}_ax12"
        body = [
            helper.make_node("Constant", [], [c_ax1], value=numpy_helper.from_array(
                np.array([1], np.int64))),
            helper.make_node("Constant", [], [c_ax12], value=numpy_helper.from_array(
                np.array([1, 2], np.int64))),
            helper.make_node("Exp", [G], [f"{n}_eg"]),
            helper.make_node("Unsqueeze", [f"{n}_eg", c_ax12], [f"{n}_eg3"]),
            helper.make_node("Mul", [S0, f"{n}_eg3"], [f"{n}_sd"]),
            helper.make_node("Unsqueeze", [K, f"{n}_ax2"], [f"{n}_k3"]),
            helper.make_node("Mul", [f"{n}_sd", f"{n}_k3"], [f"{n}_sk"]),
            helper.make_node("ReduceSum", [f"{n}_sk", c_ax1], [f"{n}_kv"],
                             keepdims=0),
            helper.make_node("Sub", [V, f"{n}_kv"], [f"{n}_vd"]),
            helper.make_node("Unsqueeze", [B, c_ax1], [f"{n}_b2"]),
            helper.make_node("Mul", [f"{n}_vd", f"{n}_b2"], [f"{n}_delta"]),
            helper.make_node("Unsqueeze", [f"{n}_delta", c_ax1], [f"{n}_d3"]),
            helper.make_node("Mul", [f"{n}_k3", f"{n}_d3"], [f"{n}_upd"]),
            helper.make_node("Add", [f"{n}_sd", f"{n}_upd"], [f"{n}_s1"]),
            helper.make_node("Unsqueeze", [Q, f"{n}_ax2"], [f"{n}_q3"]),
            helper.make_node("Mul", [f"{n}_s1", f"{n}_q3"], [f"{n}_sq"]),
            helper.make_node("ReduceSum", [f"{n}_sq", c_ax1], [f"{n}_o"],
                             keepdims=0),
        ]
        body.insert(0, helper.make_node(
            "Constant", [], [f"{n}_ax2"],
            value=numpy_helper.from_array(np.array([2], np.int64))))
        f32 = TensorProto.FLOAT
        graph = helper.make_graph(
            body, f"{n}_body",
            [helper.make_tensor_value_info(S0, f32, [h, dk, dv]),
             helper.make_tensor_value_info(Q, f32, [h, dk]),
             helper.make_tensor_value_info(K, f32, [h, dk]),
             helper.make_tensor_value_info(V, f32, [h, dv]),
             helper.make_tensor_value_info(G, f32, [h]),
             helper.make_tensor_value_info(B, f32, [h])],
            [helper.make_tensor_value_info(f"{n}_s1", f32, [h, dk, dv]),
             helper.make_tensor_value_info(f"{n}_o", f32, [h, dv])])
        st_last, outs = self.fresh("la_state"), self.fresh("la_out")
        self.emit("Scan", [st3, qn, kn, v3, g2, b2], [st_last, outs],
                  body=graph, num_scan_inputs=5)
        # outs [S, H, Dv]
        if rank4:
            self.unsqueeze(outs, [0], _tname(node.outputs[0]))
        else:
            self.emit("Transpose", [outs], _tname(node.outputs[0]),
                      perm=[1, 0, 2])
        if len(node.outputs) > 1 and _tname(node.outputs[1]):
            self.emit("Reshape", [st_last, self.emit("Shape", [st])],
                      _tname(node.outputs[1]))

    def _lower_MatMul(self, node, attrs):
        ins = list(node.inputs)
        out = _tname(node.outputs[0])
        a = _tname(ins[0])
        trans_b = int(attrs.pop("transB", 0) or 0)
        swiglu = attrs.pop("swiglu", None)
        down_act = attrs.pop("down_act", None)
        epi_scale = attrs.pop("epi_scale", None)
        epi_bias = attrs.pop("epi_bias", None)
        attn_softmax = attrs.pop("attn_softmax", None)
        for k in list(attrs):
            if k not in _EXEC_ONLY_ATTRS:
                raise LoweringError(f"MatMul attribute '{k}' is not lowered")

        if down_act:                       # [x, Wd, Wg, Wu]
            wd = _tname(ins[1])
            n_up = int(self.dag.initializers[wd].dims[0])
            w = {}
            for key, td in (("g", ins[2]), ("u", ins[3])):
                arr = self.init_array(_tname(td))
                if arr is None:
                    raise LoweringError("down_act needs initializer gate/up weights")
                arr = arr.reshape(arr.shape[0], -1)[:, :n_up]   # undo lane-major
                w[key] = self.const(np.ascontiguousarray(arr), hint=_tname(td))
            g = self.emit("MatMul", [a, w["g"]])
            u = self.emit("MatMul", [a, w["u"]])
            h = self.emit("Mul", [self.silu(g), u])
            self.emit("MatMul", [h, wd], out, name=node.name)
            return

        b = _tname(ins[1])
        rest = ins[2:]
        bias = _tname(rest.pop()) if epi_bias else None
        b_arr = None
        if b in self.dag.initializers and (
                self.dag.initializers[b].data_type in (TensorProto.INT8,
                                                       TensorProto.UINT8)):
            if not rest:
                raise LoweringError("int8 MatMul weight without a scale input")
            q = self.init_array(b).astype(np.float32)
            sc = self.init_array(_tname(rest.pop(0))).astype(np.float32).reshape(-1)
            b_arr = q * (sc[:, None] if trans_b else sc[None, :])
        if rest:
            raise LoweringError(f"MatMul with {len(ins)} inputs is not lowered")

        if swiglu:
            arr = b_arr if b_arr is not None else self.init_array(b)
            if arr is None:
                raise LoweringError("swiglu needs an initializer weight")
            k_dim, n2 = arr.shape
            grp = arr.reshape(k_dim, n2 // 8, 8)
            g = self.const(np.ascontiguousarray(grp[:, :, :4].reshape(k_dim, -1)),
                           hint=b + "_g")
            u = self.const(np.ascontiguousarray(grp[:, :, 4:].reshape(k_dim, -1)),
                           hint=b + "_u")
            gm = self.emit("MatMul", [a, g])
            um = self.emit("MatMul", [a, u])
            y = self.emit("Mul", [self.silu(gm), um],
                          None if (epi_scale or bias or attn_softmax) else out)
        else:
            if b_arr is not None:
                if trans_b:
                    b_arr, trans_b = np.ascontiguousarray(b_arr.T), 0
                b = self.const(b_arr.astype(np.float32), hint=b)
            if trans_b:
                b_shape = _tshape(ins[1])
                if b_shape is None and b in self.dag.initializers:
                    b_shape = list(self.dag.initializers[b].dims)
                b_rank = len(b_shape) if b_shape else 2
                perm = list(range(b_rank - 2)) + [b_rank - 1, b_rank - 2]
                b = self.emit("Transpose", [b], perm=perm)
            b = self._gqa_operand(ins[0], ins[1], b, node.outputs[0])
            last = not (epi_scale is not None or bias or attn_softmax)
            y = self.emit("MatMul", [a, b], out if last else None,
                          name=node.name if last else None)
        if epi_scale is not None or bias or attn_softmax:
            if epi_scale is not None and float(epi_scale) != 1.0:
                y = self.emit("Mul", [y, self.f32(float(epi_scale))])
            if bias:
                y = self.emit("Add", [y, bias],
                              None if attn_softmax else out)
            if attn_softmax:
                self.emit("Softmax", [y], out, axis=-1)
            elif not bias:
                self.emit("Identity", [y], out)

    def _gqa_operand(self, a_td, b_td, b, out_td):
        """MatMul B with fewer heads than A (fold_repeat_kv_expand removed the
        repeat_kv Expand; the runtime broadcasts h -> h // rep). A's head count
        comes from A's shape, or from the output's when A's is dynamic (Phi-4:
        A [-1,-1,-1,-1], B [-1,8,-1,128], Y [-1,24,-1,-1])."""
        sa, sb, so = _tshape(a_td), _tshape(b_td), _tshape(out_td)
        if not sb or len(sb) != 4:
            return b
        ha = sa[-3] if sa and len(sa) == 4 else -1
        if ha <= 0 and so and len(so) == 4:
            ha = so[-3]
        hb = sb[-3]
        if ha > 0 and hb > 1 and ha != hb and ha % hb == 0 and len(sb) == 4:
            return self._repeat_kv(b, ha, hb)
        return b


def _shallow(node, inputs=None, outputs=None):
    class _N:
        pass
    n = _N()
    n.op_type, n.name, n.attributes = node.op_type, node.name, node.attributes
    n.inputs = node.inputs if inputs is None else inputs
    n.outputs = node.outputs if outputs is None else outputs
    return n


def _weightless(model, keep_bytes: int = 1 << 16) -> onnx.ModelProto:
    """The model with every large initializer turned into a typed graph input.
    Shape inference needs the small ones' values (Reshape targets, axes) but
    not the weights, and a model with them can exceed protobuf's 2 GB."""
    g = model.graph
    small, big = [], []
    for t in g.initializer:
        (small if len(t.raw_data) <= keep_bytes and
         t.data_location != TensorProto.EXTERNAL else big).append(t)
    inputs = list(g.input) + [helper.make_tensor_value_info(
        t.name, t.data_type, list(t.dims)) for t in big]
    graph = helper.make_graph(list(g.node), g.name, inputs, list(g.output),
                              initializer=small)
    m = helper.make_model(graph, opset_imports=list(model.opset_import))
    m.ir_version = model.ir_version
    return m


def _infer_shapes(model) -> Dict[str, List[Optional[int]]]:
    try:
        m = onnx.shape_inference.infer_shapes(_weightless(model),
                                              strict_mode=False, data_prop=True)
    except Exception:
        return {}
    out = {}
    for vi in list(m.graph.value_info) + list(m.graph.input) + \
            list(m.graph.output):
        tt = vi.type.tensor_type
        if not tt.HasField("shape"):
            continue
        out[vi.name] = [d.dim_value if d.HasField("dim_value") else None
                        for d in tt.shape.dim]
    for t in m.graph.initializer:
        out[t.name] = list(t.dims)
    return out


def _mismatch(have, want) -> bool:
    if len(have) != len(want):
        return True
    return any(h is not None and w >= 0 and h != w for h, w in zip(have, want))


def _set_output_types(model, names, declared=None):
    """Declare graph outputs with an elem type (ORT rejects UNDEFINED), leaving
    the shape open: shape inference's when it has one, else the converter's.
    An output neither knows is dropped rather than guessed -- a wrong type is
    a load error, a missing intermediate only a gap in the per-node report."""
    try:
        inferred = onnx.shape_inference.infer_shapes(_weightless(model),
                                                     strict_mode=False)
        types = {vi.name: vi.type.tensor_type.elem_type
                 for vi in list(inferred.graph.value_info) +
                 list(inferred.graph.output)}
    except Exception:
        types = {}
    del model.graph.output[:]
    graph_inputs = {vi.name: vi.type.tensor_type.elem_type
                    for vi in model.graph.input}
    for name in names:
        et = types.get(name) or graph_inputs.get(name) or \
            (declared or {}).get(name)
        if not et:
            continue
        model.graph.output.append(helper.make_tensor_value_info(name, et, None))


def lower_dag(dag, *, opset: Optional[int] = None, fp32: bool = True,
              expose_all: bool = True, base_dir: Optional[str] = None
              ) -> onnx.ModelProto:
    """Lower a (fused) DAGBasedModel to a standard ONNX model.

    `expose_all` adds every vkop node output as a graph output, which is what a
    per-node comparison needs. `opset` defaults to the source model's
    (recorded on the DAG by the converter), else 17.
    """
    if opset is None:
        opset = int(getattr(dag, "onnx_opset", 0) or 17)
    low = _Lowerer(dag, opset, fp32, base_dir or getattr(dag, "weight_base_dir",
                                                          None))
    inputs, outs = low.run(expose_all)
    return low.finish(inputs, outs)


def save_lowered(model: onnx.ModelProto, path: str) -> None:
    """Save, spilling weights to `<path>.data` when the model would exceed
    protobuf's 2 GB limit."""
    size = sum(len(t.raw_data) for t in model.graph.initializer)
    if size > (1 << 31) - (64 << 20):
        onnx.save(model, path, save_as_external_data=True,
                  all_tensors_to_one_file=True,
                  location=os.path.basename(path) + ".data", size_threshold=1024)
    else:
        onnx.save(model, path)


def upcast_model_to_fp32(model: onnx.ModelProto) -> onnx.ModelProto:
    """fp16/bf16 -> fp32 copy of a plain ONNX model (initializers, graph I/O,
    Cast targets, Constant values), so the original graph can be run as the
    same fp32 reference the lowered one is."""
    m = onnx.ModelProto()
    m.CopyFrom(model)
    g = m.graph
    for i, t in enumerate(g.initializer):
        if t.data_type in _FLOAT_LOW:
            g.initializer[i].CopyFrom(numpy_helper.from_array(
                _to_float32_array(t), t.name))
    for vi in list(g.input) + list(g.output) + list(g.value_info):
        if vi.type.tensor_type.elem_type in _FLOAT_LOW:
            vi.type.tensor_type.elem_type = TensorProto.FLOAT
    for n in g.node:
        for a in n.attribute:
            if n.op_type == "Cast" and a.name == "to" and a.i in _FLOAT_LOW:
                a.i = TensorProto.FLOAT
            if a.type == onnx.AttributeProto.TENSOR and a.t.data_type in _FLOAT_LOW:
                a.t.CopyFrom(numpy_helper.from_array(_to_float32_array(a.t),
                                                     a.t.name))
    return m
