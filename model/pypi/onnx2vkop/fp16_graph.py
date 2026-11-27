"""Whole-graph FP16: what makes -q fp16 produce an FP16 artifact.

`quantize_to_fp16_selective` is a WEIGHT pass. It asks, per initializer,
whether the op that reads it already runs FP16, and downcasts the weight only
then -- which is the right question for a graph an exporter delivered in FP16,
and the reason an FP32-exported convnet converted with -q fp16 stayed FP32
end to end: every Conv's data input is declared FLOAT, so every weight is
refused, and the artifact is the FP32 model under a different name.

Refusing was correct. The runtime takes an op's shader build from the dtype of
the TENSORS it is handed (core/runtime.cpp `op_shader_fp16`, and the
output-allocation marker at `node_inputs[0]->dtype()`), so a lone FP16 weight
behind an FP32 Conv makes that Conv walk a 384-byte table as a 768-byte
operand. Precision is a property of the whole node, not of one input slot, and
one node's dtype is decided by what flows through it.

So this module does not ask "may this weight go FP16". It computes the FP16
DOMAIN of the graph:

  * start assuming every tensor the file declares FLOAT goes FP16;
  * a node that cannot follow the declared dtype (no dual FP16 buffer build,
    or a build the runtime only picks from the --precision flag) has to stay
    FP32, and then every FLOAT tensor it touches has to stay FP32 too;
  * "has to stay FP32" propagates through every node except Cast, in both
    directions, to a fixed point -- the node beside a forced node would run
    FP32 storage under an FP16 build, which is the same garbage in the other
    direction;
  * Cast is the only node allowed to sit between the two domains, because its
    output dtype is the `to` attribute and `buffer_cast` is dtype-generic. An
    FP32 island with no Cast around it is therefore not a boundary, it is a
    contradiction, and the audit at the end refuses the rewrite rather than
    shipping half a graph.

What is left is the domain: every name in it is declared FLOAT16 in all four
places the file spells a dtype (node inputs, node outputs, graph inputs,
graph outputs -- rewriting by NAME means a tensor two nodes declare
inconsistently comes out consistent), and its FP32 initializers get payload
bytes that match the declaration.

Two deliberate limits:

  * The buffer backend only. The legacy image backend builds every graph input
    as `Tensor<uint16_t>` regardless of the declaration, so there the file's
    dtype says nothing about the run and rewriting it changes nothing.
  * A weight whose magnitude exceeds FP16's largest finite half is reported,
    not blocked: saturating it is what an FP16 export does, and deciding
    between saturating and keeping the tensor wide is the scale-for-kernel
    gate's job, not this pass's.
"""
from typing import Dict, List, Set

import numpy as np
import onnx
from onnx import numpy_helper

try:
    from . import shader_map
    from . import numeric_audit
    from . import bf16_weights
    from .dag import Node
except ImportError:  # run as a flat script from inside onnx2vkop/, as converter.py supports
    import shader_map
    import numeric_audit
    import bf16_weights
    from dag import Node

_FLOAT = onnx.TensorProto.FLOAT
_FLOAT16 = onnx.TensorProto.FLOAT16
# The other half-width float. Same 16-bit storage and 32-bit arithmetic as
# fp16, but 8 exponent bits instead of 5: fp32's range, ~3 significant decimal
# digits. MiniMax-H3 needs it because its activations legitimately exceed
# fp16's 65504 (hidden states reach ~4200 and a 14336-wide FFN accumulation
# crosses the ceiling).
_BFLOAT16 = onnx.TensorProto.BFLOAT16
_DOUBLE = onnx.TensorProto.DOUBLE
# BFLOAT16 is a float like the others: leaving it out made a bf16 declaration
# invisible to _decls/_audit, so an fp16 tensor beside a bf16 one (or a bf16
# one beside an FP32 one) was never reported.
_FLOAT_FAMILY = (_FLOAT, _FLOAT16, _BFLOAT16, _DOUBLE)
_INT32 = onnx.TensorProto.INT32
_INT64 = onnx.TensorProto.INT64

# The ops whose `_bf16` build decodes AND encodes bf16 end to end -- the
# converter mirror of core/Bf16Gate.cpp's bf16_activation_op, which the
# runtime enforces at LoadModel. Under --act bf16 these run their _bf16
# module; every other op is an ISLAND that keeps the width the export gave its
# tensors, bounded by boundary Casts (bf16_partition). Spellings are the
# runtime's (ops/Ops.hpp convert_opstring_to_optype), aliases included. Ops
# with a `_bf16` archive key but no verified kernel (Conv, AveragePool, TopK,
# Where, ...) stay off the list.
BF16_ACT_OPS = frozenset({
    # shaders/common/binary_elementwise_body.comp (bdec2/benc2)
    "Add", "Sub", "Mul", "Div", "Pow", "Min", "Max", "Mod",
    # BUF_LOAD / BUF_PACK2 under -DBF16
    "Split", "LayerNormalization", "LayerNorm", "RMSNorm", "Softmax",
    # route-gated inside the op (KernelSelect matmul_route_takes_bf16_act,
    # Attention's half_q routing)
    "MatMul", "Attention",
    # Byte movers: they copy or alias 16-bit words and never decode one
    # (Reshape aliases the input buffer, Slice's _bf16 module is the fp16
    # word mover, Squeeze/Unsqueeze are host aliases).
    # Bf16ActivationTest.LoadModelMovesBf16BitsThroughByteMovers.
    "Reshape", "Slice", "Squeeze", "Unsqueeze",
    # shaders/common/unary_elementwise_body.comp (UDEC2/UENC2) + clip.comp
    "Sin", "Cos", "Sigmoid", "Tanh", "Erf", "Sqrt", "Relu", "Neg", "Floor",
    "Atan", "Softplus", "Clip",
    # movers: expand.comp (BUF_LOAD/BUF_PACK2), concat.comp (16-bit words),
    # transpose.comp (BUF_BITS16), pad.comp (+ a host-encoded bf16 constant)
    "Expand", "Concat", "Transpose", "Pad",
    # Per-operand formats (_BF16_OTHER_FLOAT_SLOTS): FusedElemwise's per-input
    # dtype codes, Gather's -DBF16_OUT from a bf16/fp16/fp32 table. Cast is
    # the boundary itself (cast.comp modes 3-6 convert fp32/fp16/bf16); it is
    # listed so the two tables stay one-for-one with Bf16Gate.
    "FusedElemwise", "Gather", "Cast",
    # BUF_LOAD / BUF_UNPACK2 / BUF_PACK2 under -DBF16 (tests/Bf16DataMoveTest)
    "RotaryEmbedding", "Resize", "MaxPool", "MaxPool2d", "GlobalAveragePool",
    "Reduce", "ReduceMean", "InstanceNormalization", "InstanceNorm", "PRelu",
    # own bf16 codec (cumsum.comp uint16_t storage, scatter_elements.comp CAS)
    "CumSum", "Cumsum", "ScatterElements",
})

# The ops that carry 16-bit words through unchanged. Under --act bf16 they
# take the width of their DATA input (input 0) instead of having one of their
# own: a Reshape between two fp16 islands stays fp16 (no Cast pair around
# it), one between two bf16 ops is bf16. Each has fp32, fp16 and bf16 builds,
# so whichever width arrives runs. Flatten and Identity never reach the
# runtime (the optimizer lowers/removes them) but are movers all the same.
_BF16_MOVERS = frozenset({"Reshape", "Slice", "Squeeze", "Unsqueeze", "Split",
                          "Transpose", "Flatten", "Identity"})

# Operand slots of a bf16 node that may hold ANOTHER float width, because the
# kernel (or the host) reads that operand at its own format -- the converter
# mirror of core/Bf16Gate.cpp's bf16_other_float_ok. One table serves both the
# partition (no Cast where the reader takes the width as it is) and the
# finished-graph refusal (bf16_kind_conflicts):
#   "any"   fp32, fp16 or bf16: every FusedElemwise input (per-input dtype
#           code), Gather's table (-DBF16_OUT), Clip's min/max and Pad's
#           constant_value (scalars the host reads by their own kind);
#   "fp32"  fp32 or bf16 only: RotaryEmbedding's cos/sin. rotary.comp reads
#           an fp32 table at fp32 stride (RotaryPC::cs_fp32) and a 16-bit one
#           in X's own format, so an fp16 table beside bf16 X is NOT legal.
# Cast is not listed: every operand of a Cast is any width (its kind "cast").
# op -> {input slot: rule}; slot None means every input.
_BF16_OTHER_FLOAT_SLOTS = {
    "FusedElemwise": {None: "any"},
    "Gather": {0: "any"},
    "Clip": {1: "any", 2: "any"},
    "Pad": {2: "any"},
    "RotaryEmbedding": {1: "fp32", 2: "fp32"},
}


def _bf16_slot_rule(op_type, index):
    """None (bf16 only), "any" or "fp32" -- see _BF16_OTHER_FLOAT_SLOTS."""
    rules = _BF16_OTHER_FLOAT_SLOTS.get(op_type)
    if not rules:
        return None
    return rules.get(index, rules.get(None))

# Graph outputs that keep the export's width under --act bf16 even when a bf16
# op produces them: the LLM drivers read `logits` through the GPU TopK
# post-process (llm_common.hpp register_gpu_argmax) and TopK has no bf16
# kernel, so a Cast bf16 -> export width is put in front of them instead.
_BF16_KEEP_OUTPUTS = frozenset({"logits"})

# Slots in which a bf16 tensor is a WEIGHT, read beside activations of any
# width (core/Bf16Gate.hpp bf16_weight_slot).
_BF16_WEIGHT_SLOTS = {"MatMul": (1,), "Gemm": (1,), "Gather": (0,)}

# MatMul attributes whose kernel has no bf16 read: the SwiGLU down fold
# (matmul_down_act.comp, fp16-only). The runtime refuses a bf16 A on it. The
# gate/up fold (swiglu), the QK^T epilogue (epi_bias, bf16 mask decode) and the
# fused attn_softmax row kernel all run in the _bf16 module, as does odd N
# (reduce + a bf16-encoding pack pass) -- Bf16ActivationTest covers each.
_BF16_MATMUL_FOLDS = ("down_act",)

# The qwen3_5 linear-attention ops take mixed widths by design: their DATA
# tensors follow the artifact's dtype while their recurrent state is fp32
# (HF runs the whole gated-delta recurrence in fp32, and a half state drifts
# over a long decode). Their kernels bind each tensor at its own dtype, so a
# mixed operand set is the intended form rather than the corruption the dtype
# guard exists for -- hence the exemption in _audit below.
#
# op_type -> (state input indices, state output indices)
_STATE_SLOTS = {
    "Conv1d": ((2,), (1,)),
    "Conv1D": ((2,), (1,)),
    "LinearAttention": ((5,), (1,)),
    "GatedDeltaNet": ((5,), (1,)),
}

# Name prefixes of the state tensors those ops carry. Matching by NAME as well
# as by slot is what covers the view nodes the exporter puts around a state (a
# batch-dim Unsqueeze, a Reshape): the op-level slots stop at the custom node,
# but the view's output is still a state and must stay FP32.
_STATE_NAME_PREFIXES = ("conv_state", "recurrent_state")

# Ops whose SHADER reads each operand at that operand's own width, so a mixed
# operand set is supported rather than a corruption. FusedElemwise exists
# precisely to straddle a Cast: its program carries a per-input fp16 flag
# (shaders/buffer/fused_elemwise.comp's load helpers) and the build dtype
# follows the terminal OUTPUT, not input[0] (core/runtime.cpp's op_shader_fp16).
_MIXED_WIDTH_OPS = {"FusedElemwise"}

# The RMSNorm variance chain, whose square has to run in FP32:
#
#     Pow(x, 2) -> Reduce -> FusedElemwise      (Add(eps)+Sqrt, fused)
#
# `x**2` on a hidden state of magnitude ~890 is ~7.9e5, past fp16's 65504, so
# the packed store saturates to +Inf and everything downstream is NaN.
# Measured on MiniMax-H3's denoiser at seq=160 with VKOP_NAN_SCAN=1:
#   FIRST NaN/Inf at Pow_5416, in[0] min=-889.5 max=462.5, out=7c00
#
# The shape is the LOWERED spelling, not the exported one: the optimizer turns
# ReduceMean into Reduce and folds Add(eps)+Sqrt into one FusedElemwise.
# Matching the export's Pow -> ReduceMean -> Add -> Sqrt found zero chains.
_VARIANCE_CHAIN = ("Pow", "Reduce", "FusedElemwise")

# What a Sin/Cos phase is computed from (see GraphFp16._hold_phase_chains).
_PHASE_OPS = {"FusedElemwise", "Mul", "Add", "Sub", "Div", "Neg", "Unsqueeze",
              "Squeeze", "Reshape", "Expand", "Concat", "Transpose", "MatMul",
              "Gather", "Slice", "Cast", "Where"}
_PHASE_CONE_MAX = 32

# Largest finite half. Past this a value rounds to Inf, and the fp16 kernels
# have no saturation path -- the reference exporter does the same and calls it
# a lossy conversion.
_FP16_MAX = 65504.0


class GraphFp16:
    """Compute and apply the FP16 domain of a DAG."""

    @staticmethod
    def _decls(dag_model) -> Dict[str, dict]:
        """{tensor name: {"elem", "float", "slots"}} over the whole file.

        Reads the same four places the writer does. Node input/output entries
        carry the ONNX elem_type int the parser recorded from shape inference;
        graph inputs/outputs carry the dtype NAME `_io_dtype_name` resolved.
        An entry with no dtype at all -- a tensor reference the parser could
        not resolve -- is left out: nothing here may invent a declaration, and
        the runtime takes such a tensor from its producer anyway.

        `elem` keeps the widest declaration seen for the name, because two
        nodes disagreeing about one tensor is a bug in the graph and the
        conservative reading (FLOAT beats DOUBLE beats FLOAT16) is the one that
        can still be rewritten into something coherent.
    """
        decls: Dict[str, dict] = {}
        rank = {_FLOAT: 4, _DOUBLE: 3, _BFLOAT16: 2, _FLOAT16: 1}

        def note(name, elem, slot=None):
            d = decls.get(name)
            if d is None:
                decls[name] = d = {"elem": elem, "float": False, "slots": []}
            if slot is not None:
                d["slots"].append(slot)
            if elem in _FLOAT_FAMILY:
                d["float"] = True
                if rank[elem] > rank.get(d["elem"], 0):
                    d["elem"] = elem
            elif not d["float"]:
                d["elem"] = elem

        for node in dag_model.nodes.values():
            for entry in list(node.inputs) + list(node.outputs):
                name = entry.get("name")
                if name is None or "dtype" not in entry:
                    continue
                elem = int(entry["dtype"])
                note(name, elem,
                     (entry, "dtype") if elem in _FLOAT_FAMILY else None)

        for entry in list(dag_model.inputs) + list(dag_model.outputs):
            name = entry.get("name")
            dtype = entry.get("dtype")
            if name is None or dtype is None:
                continue
            elem = {"float32": _FLOAT, "float16": _FLOAT16,
                    "bfloat16": _BFLOAT16}.get(dtype)
            if elem is None and isinstance(dtype, int):
                elem = dtype
            note(name, elem if elem is not None else 0,
                 (entry, "dtype") if elem in _FLOAT_FAMILY else None)

        for name, init in dag_model.initializers.items():
            if name in decls:
                # Merge the payload's own type into the widest view. An
                # initializer NO node references is not in the domain either:
                # it has no declaration to be consistent with and no consumer
                # to satisfy, and rewriting the bytes of a tensor the graph
                # does not read is not what a precision pass is for -- the
                # weight pass below reaches the same verdict ("no consumers").
                note(name, int(init.data_type))
        return decls

    @staticmethod
    def _bf16_node_blocker(node, decls, quantized):
        """Why an allow-listed op still cannot run this NODE in bf16."""
        if node.op_type == "MatMul":
            for a in _BF16_MATMUL_FOLDS:
                v = node.attributes.get(a)
                if v not in (None, 0, "0", [0], []):
                    return f"MatMul {a}: no bf16 kernel for the fold"
            if len(node.inputs) > 1 and \
                    node.inputs[1].get("name") in quantized:
                return ("MatMul: a quantized weight has no bf16-"
                        "activation kernel")
        for entry in list(node.inputs) + list(node.outputs):
            d = decls.get(entry.get("name"))
            if d is None or d["float"] or d["elem"] in (0, _INT32,
                                                        _INT64):
                continue
            return (f"{node.op_type}: operand '{entry.get('name')}' is "
                    f"elem type {d['elem']}, which no bf16 kernel binds "
                    "beside bf16")
        return None

    @staticmethod
    def _candidates(dag_model, decls, half: str) -> Set[str]:
        """The names the domain pass may retype.

        FLOAT always. Under bf16 also FLOAT16: every LLM export in the repo
        declares its activations FLOAT16, and bf16 is a different 16-bit
        format, not "the half that is already there" -- leaving them out made
        `-q bf16 --act bf16` on GLM-Edge retype 0 tensors.
        """
        kinds = (_FLOAT, _FLOAT16) if half == "bf16" else (_FLOAT,)
        candidates = {n for n, d in decls.items() if d["elem"] in kinds}
        # The qwen3_5 linear-attention ops follow the dtype for their DATA
        # tensors but keep their recurrent STATE in fp32: HF runs the whole
        # gated-delta recurrence in fp32, and a half state drifts over a long
        # decode (see ops/LinearAttention.hpp and ops/Conv1d.hpp). Their state
        # slots are therefore never candidates, so the propagation below cannot
        # pull them into the domain -- and because a group is one shader over
        # all its tensors, dropping the state from the group is what lets the
        # rest of the node be fp16.
        for n in [n for n in candidates if n.startswith(_STATE_NAME_PREFIXES)]:
            candidates.discard(n)
        for node in dag_model.nodes.values():
            slots = _STATE_SLOTS.get(node.op_type)
            if slots is None:
                continue
            in_idx, out_idx = slots
            for i in in_idx:
                if i < len(node.inputs):
                    candidates.discard(node.inputs[i].get("name"))
            for i in out_idx:
                if i < len(node.outputs):
                    candidates.discard(node.outputs[i].get("name"))
        return candidates

    @staticmethod
    def plan(dag_model, half: str = "fp16", quantized=None) -> dict:
        """{domain, forced, reasons, blockers, conflicts} for this DAG.

        Computes, writes nothing. `blockers` maps a node name to why it cannot
        enter the domain, which is what makes a partially-rewritten graph
        explainable: the FP32 island is always reachable from a named op.

        half="bf16" is not a propagation: an op with no bf16 kernel is an
        island behind boundary Casts (cast.comp converts fp32/fp16/bf16), so
        the answer is bf16_partition's, reported in this method's shape.
        """
        if half == "bf16":
            part = GraphFp16.bf16_partition(dag_model, quantized=quantized)
            return {"decls": part["decls"], "domain": part["domain"],
                    "forced": part["forced"], "reasons": {},
                    "blockers": part["blockers"], "conflicts": [],
                    "partition": part}
        decls = GraphFp16._decls(dag_model)
        candidates = GraphFp16._candidates(dag_model, decls, half)

        # Propagation unit: the FLOAT-declared names of one node. An op runs a
        # single shader over all of them, so one that must stay FP32 keeps them
        # all FP32.
        group_of = {
            node.name: {e["name"] for e in list(node.inputs) + list(node.outputs)
                        if e.get("name") in candidates}
            for node in dag_model.nodes.values()}
        # A linear-attention op's state tensors are fp32 by design, so they must
        # not be in any propagation group: a group is "one shader over all of
        # these", and a state in it would drag the data tensors to FP32 (or the
        # state to FP16) whichever way the queue went.
        for node in dag_model.nodes.values():
            slots = _STATE_SLOTS.get(node.op_type)
            if slots is None:
                continue
            in_idx, out_idx = slots
            names = {node.inputs[i].get("name") for i in in_idx
                     if i < len(node.inputs)}
            names |= {node.outputs[i].get("name") for i in out_idx
                      if i < len(node.outputs)}
            group_of[node.name] -= names
        touched: Dict[str, List[str]] = {}
        for node_name, group in group_of.items():
            for name in group:
                touched.setdefault(name, []).append(node_name)

        forced: Set[str] = set()
        reasons: Dict[str, str] = {}
        blockers: Dict[str, str] = {}
        queue: List[str] = []

        def force(names, why):
            for name in names:
                if name in candidates and name not in forced:
                    forced.add(name)
                    reasons.setdefault(name, why)
                    queue.append(name)

        def classify(op):
            if op == "Cast":
                return "boundary"
            if op in shader_map.HOST_ONLY_OPS:
                # No shader at all: a host-side copy whose output dtype is its
                # input's by construction. It can live in either domain.
                return "transparent"
            if op not in shader_map._PRIMARY:
                return "blocked"
            if not shader_map.fp16_dtype_driven(op):
                return "blocked"
            return "domain"

        for node in dag_model.nodes.values():
            kind = classify(node.op_type)
            group = group_of[node.name]
            if kind == "boundary":
                # Only the OUTPUT side of a Cast is decided by the attribute.
                # Forcing its input too would drag the whole graph back to
                # FP32 at the first Cast(to=1), which is exactly the boundary
                # this node exists to provide.
                to = node.attributes.get("to")
                if to is not None and int(to) != _FLOAT16:
                    force({e["name"] for e in node.outputs
                           if e.get("name") in candidates},
                          f"Cast(to={int(to)}) emits that dtype")
                continue
            if kind == "blocked" and group:
                blockers[node.name] = (
                    f"{node.op_type}: its build does not follow the tensor "
                    "dtype" if node.op_type in shader_map._FP16_DTYPE_DRIVEN
                    else f"{node.op_type}: no dual FP16 buffer build")
                force(group, blockers[node.name])

        # A Sin/Cos PHASE (t*freq) does not fit fp16 -- it grows with the
        # position and saturates at 65504, after which the rotation is wrong.
        # bf16 has fp32's exponent range, so the same argument does not apply
        # and holding the cone wide would drag the whole attention block (and,
        # through the propagation, most of the graph) back to FP32: measured on
        # the VAE, where this single rule left 1835 of 2417 tensors at FP32.
        if half != "bf16":
            GraphFp16._hold_phase_chains(dag_model, candidates, force,
                                         blockers)

        # Ops whose operands are independent widths: a forced input says
        # nothing about the others, so it must not propagate.
        #   FusedElemwise  -- exists to straddle a Cast; the shader reads each
        #                     operand at that operand's own width and the build
        #                     follows the terminal output.
        #   the two qwen3_5 ops -- data follows the dtype, state is always fp32.
        # Without this, one fp32 Cast anywhere upstream swept the whole
        # downstream chain to FP32 and the fp16 domain collapsed (measured:
        # 10 of 66 tensors, 0 Cast-to-fp16 in the graph).
        independent = _MIXED_WIDTH_OPS | set(_STATE_SLOTS)
        while queue:
            name = queue.pop()
            for node_name in touched.get(name, ()):
                node = dag_model.nodes[node_name]
                if node.op_type == "Cast":
                    # The boundary node: it may read either domain, and its
                    # output was already decided by `to` above.
                    continue
                if node.op_type in independent:
                    blockers.setdefault(
                        node_name,
                        f"{node.op_type}: operands are independent widths")
                    # ...but the state slots of the qwen3_5 ops are fp32 by
                    # contract, so they are held there explicitly rather than
                    # left to the (now stopped) propagation.
                    slots = _STATE_SLOTS.get(node.op_type)
                    if slots is not None:
                        in_idx, out_idx = slots
                        force({node.inputs[i].get("name") for i in in_idx
                               if i < len(node.inputs)},
                              f"{node.op_type}: state is fp32 by design")
                        force({node.outputs[i].get("name") for i in out_idx
                               if i < len(node.outputs)},
                              f"{node.op_type}: state is fp32 by design")
                    continue
                blockers.setdefault(
                    node_name, f"{node.op_type} runs FP32: it touches '{name}'")
                force(group_of[node_name], blockers[node_name])

        domain = candidates - forced
        # Same argument as the rope phase above: x**2 overflows fp16's 65504
        # and needs the chain held wide. bf16's exponent range holds it with no
        # special handling -- 890**2 = 7.9e5 is an ordinary bf16 -- so leaving
        # this rule on under bf16 costs a Cast per norm AND, through the
        # propagation, drags each block's consumers back to FP32.
        if half != "bf16":
            GraphFp16._widen_variance_chains(dag_model, candidates, forced,
                                             reasons, blockers, half)
        domain = candidates - forced
        return {"decls": decls, "domain": domain, "forced": forced,
                "reasons": reasons, "blockers": blockers,
                "conflicts": GraphFp16._audit(dag_model, decls, domain, half)}

    @staticmethod
    def _hold_phase_chains(dag_model, candidates, force, blockers) -> int:
        """Keep the phase a Sin/Cos reads -- and everything it is computed
        from -- in FP32.

        A sinusoidal embedding's phase is `t * 1000 * freq` (DiT timestep) or
        `position * inv_freq` (RoPE): magnitudes in the hundreds to thousands,
        where fp16's spacing is 0.25..1 rad, so the cos/sin of an fp16 phase is
        mostly rounding. Measured on the tiny Qwen-Image DiT decode with
        `vkop_vs_ort.py --node-dump`: /Cos at cos 0.9996 with 20% of elements
        outside the band, carried through the whole timestep MLP.

        Nothing else holds it wide. The export's own `Cast(timestep, to=fp32)`
        is folded away as redundant once the input is fp32, and the phase is
        usually one FusedElemwise, whose operands are independent widths, so no
        forced tensor upstream or downstream reaches its inputs. So the
        backward cone of each Sin/Cos input is forced here, through the
        elementwise/view/MatMul ops a phase is built from, stopping at a Cast
        (its output is forced, nothing behind it), a graph input or an
        initializer. The propagation below then carries the Sin/Cos node itself
        to the next Cast; the exports put a Cast(to=fp16) one or two nodes after
        it. A cone that grows past _PHASE_CONE_MAX nodes is not a phase
        computation (a Sin activation on a hidden state) and is left alone.
        """
        producer_of = {}
        for node in dag_model.nodes.values():
            for out in node.outputs:
                producer_of[out.get("name")] = node
        held = 0
        for node in dag_model.nodes.values():
            if node.op_type not in ("Sin", "Cos") or not node.inputs:
                continue
            cone, seen_nodes = set(), set()
            stack = [node.inputs[0].get("name")]
            ok = True
            while stack:
                name = stack.pop()
                if name in cone or name not in candidates:
                    continue
                cone.add(name)
                src = producer_of.get(name)
                if src is None or src.name in seen_nodes:
                    continue
                seen_nodes.add(src.name)
                if len(seen_nodes) > _PHASE_CONE_MAX or \
                        src.op_type not in _PHASE_OPS:
                    ok = False
                    break
                if src.op_type == "Cast":
                    continue
                stack.extend(i.get("name") for i in src.inputs)
            if not ok or not cone:
                continue
            # (Not reached under --act bf16: the caller skips this
            # rule there, because bf16's 8 exponent bits hold any phase.)
            force(cone, f"phase of {node.op_type} '{node.name}': fp16 cannot "
                        "hold t*freq")
            blockers.setdefault(node.name, f"{node.op_type}: phase held in FP32")
            held += 1
        return held

    @staticmethod
    def _widen_variance_chains(dag_model, candidates, forced, reasons,
                               blockers, half: str = "fp16") -> int:
        """Run each RMSNorm variance chain in FP32 behind a Cast, in place.

            x(fp16) -> Cast(to=fp32) -> Pow -> Reduce -> FusedElemwise -> ...

        `Pow(x, 2)` overflows fp16 on its own and BOTH ends of it are decided
        by its input's dtype (core/runtime.cpp's op_shader_fp16 reads
        node_inputs[0], and the shader's fp16 build is a compile-time `#ifdef`
        with no mixed form), so the only way to have the square computed wide
        is to hand Pow a wide input.

        Forcing that input wide directly is not an option -- it IS the hidden
        state, shared by half the graph, so one forced activation collapses the
        whole domain back to FP32 (measured: 110 conflicts, REFUSED). A Cast is
        the boundary the pass is allowed to create, and after it the chain's
        consumers are free to be half again because Sqrt has already brought
        the value back into range.

        Returns the number of chains widened.
        """
        half_elem = _BFLOAT16 if half == "bf16" else _FLOAT16
        consumers_of: Dict[str, List[str]] = {}
        producer_of: Dict[str, str] = {}
        for node in dag_model.nodes.values():
            for inp in node.inputs:
                consumers_of.setdefault(inp.get("name"), []).append(node.name)
            for out in node.outputs:
                producer_of[out.get("name")] = node.name

        widened = 0
        for node in list(dag_model.nodes.values()):
            if node.op_type != "Pow":
                continue
            chain = [node]
            while len(chain) < len(_VARIANCE_CHAIN):
                outs = [o.get("name") for o in chain[-1].outputs]
                nxt = [dag_model.nodes[nm]
                       for nm in consumers_of.get(outs[0] if outs else "", [])]
                if len(nxt) != 1:
                    break
                chain.append(nxt[0])
            if tuple(n.op_type for n in chain) != _VARIANCE_CHAIN:
                continue
            src = node.inputs[0].get("name")
            if src not in candidates or src in forced:
                # Already wide (untyped, or held FP32 by the propagation -- an
                # export's own Cast(to=fp32) feeding the norm): nothing to
                # insert. Inserting anyway declared the fp32 tensor half, and
                # the new Cast read its fp32 bytes as halves (tiny Qwen-Image
                # DiT prefill: /Pow at cos 0.68, present_kv at 0.91).
                continue
            wide = src + "__fp32"
            producer = dag_model.nodes.get(producer_of.get(src, ""))
            cast = type(node)(
                "Cast", src + "__to_fp32",
                {"to": _FLOAT},
                [dict(node.inputs[0], dtype=half_elem)],
                [{"name": wide, "shape": node.inputs[0].get("shape"),
                  "dtype": _FLOAT}])
            dag_model.nodes[cast.name] = cast
            node.inputs[0]["name"] = wide
            node.inputs[0]["dtype"] = _FLOAT
            # The Cast's input must be declared at the graph's own half width
            # so the boundary is real -- bf16 under --act bf16.
            # Writing FLOAT16 here unconditionally gave one tensor two
            # declarations (the producer said bfloat16, this Cast said
            # float16), and the loader took the Cast's side: measured as an Add
            # whose inputs were bf16 and whose declared output was float16,
            # resolved to buffer_add_fp16 while its bytes were bfloat16.
            if producer is not None:
                for out in producer.outputs:
                    if out.get("name") == src:
                        out["dtype"] = half_elem
            # Everything Pow and Reduce write is wide: they are the two ops
            # that carry the squared magnitude, and a node cannot straddle the
            # two widths. The chain's last node (the fused Add+Sqrt) writes the
            # normalised result, which Sqrt has already brought back into the
            # hidden state's range -- leaving IT half is what keeps the fp32
            # boundary from spreading to every consumer downstream.
            for n in chain[:-1]:
                for out in n.outputs:
                    reasons.setdefault(
                        out.get("name"),
                        "RMSNorm variance chain: x**2 overflows fp16")
                    if out.get("name") in candidates:
                        forced.add(out.get("name"))
            # Pow's exponent operand rides the same node, so it must be wide
            # too (a one-element constant, so it costs nothing).
            for extra in node.inputs[1:]:
                nm = extra.get("name")
                if nm in candidates:
                    forced.add(nm)
                    reasons.setdefault(nm, "operand of the fp32 x**2")
            blockers.setdefault(
                node.name,
                "RMSNorm variance chain: x**2 overflows fp16 (Cast to FP32)")
            widened += 1
        return widened

    @staticmethod
    def _audit(dag_model, decls: Dict[str, dict], domain: Set[str],
               half: str = "fp16") -> List[str]:
        """Nodes that would end up with a mixed operand set, described.

        The rewrite only downgrades FLOAT, so a tensor the file already
        declares FLOAT16 is in the domain by definition. If such a name shares
        a non-Cast node with a tensor that has to stay FP32, the graph mixes
        the two widths inside one op -- the corruption the dtype guard exists
        for, and not something rewriting can fix, since the fix is a Cast the
        exporter did not emit. Refusing is the honest answer.
        """
        out = []
        for node in dag_model.nodes.values():
            if node.op_type == "Cast" or node.op_type in _MIXED_WIDTH_OPS:
                continue
            # The state slots of the qwen3_5 linear-attention ops are SUPPOSED
            # to be the other width from their data tensors; see _STATE_SLOTS.
            slots = _STATE_SLOTS.get(node.op_type)
            exempt = set()
            if slots is not None:
                in_idx, out_idx = slots
                for i in in_idx:
                    if i < len(node.inputs):
                        exempt.add(node.inputs[i].get("name"))
                for i in out_idx:
                    if i < len(node.outputs):
                        exempt.add(node.outputs[i].get("name"))
            wide, narrow = [], []
            halves16, halvesbf = [], []
            for entry in list(node.inputs) + list(node.outputs):
                name = entry.get("name")
                if name in exempt or (name or "").startswith(
                        _STATE_NAME_PREFIXES):
                    continue
                d = decls.get(entry.get("name"))
                if d is None or not d["float"]:
                    continue
                # A domain member is about to be declared at `half`.
                kind = (_BFLOAT16 if half == "bf16" else _FLOAT16) \
                    if entry["name"] in domain else d["elem"]
                if kind == _FLOAT16:
                    halves16.append(entry["name"])
                elif kind == _BFLOAT16:
                    halvesbf.append(entry["name"])
                if d["elem"] in (_FLOAT16, _BFLOAT16) or \
                        entry["name"] in domain:
                    narrow.append(entry["name"])
                elif slots is not None:
                    # A linear-attention op's DATA slots have no usable
                    # declaration: the exporter emits the custom node with the
                    # graph's declared (FLOAT) output type, while the runtime
                    # decides the width from the data INPUT at execute time. So
                    # for these nodes the declaration says nothing about the
                    # data path -- count it as following the domain instead of
                    # dragging the node back to FP32.
                    narrow.append(entry["name"])
                else:
                    wide.append(entry["name"])
            if wide and narrow:
                out.append(f"{node.op_type} {node.name}: FP32 "
                           f"{sorted(set(wide))[:3]} beside FP16 "
                           f"{sorted(set(narrow))[:3]}")
            elif halves16 and halvesbf:
                out.append(f"{node.op_type} {node.name}: FP16 "
                           f"{sorted(set(halves16))[:3]} beside BF16 "
                           f"{sorted(set(halvesbf))[:3]}")
        return out

    @staticmethod
    def mixed_half_kinds(dag_model) -> List[str]:
        """Nodes that bind bf16 beside another float width, or [] if none.

        cast.comp converts fp32, fp16 and bf16 among themselves, so an fp16
        island behind a Cast is a legal graph; what the runtime refuses is ONE
        node binding bf16 beside another width outside the slots that read
        their own format (bf16_kind_conflicts, core/Bf16Gate.cpp). A bf16
        WEIGHT in its weight slot (MatMul/Gemm B, Gather table) beside fp16 or
        fp32 activations is the --keep-bf16 / -q bf16 contract, not a mix.
        """
        return GraphFp16.bf16_kind_conflicts(dag_model)

    @staticmethod
    def mixed_width_conflicts(dag_model) -> List[str]:
        """The _audit verdict for the graph AS DECLARED (an empty domain).

        Under -q fp32 the domain pass does not run, so nothing would reconcile
        a node that already mixes FP16 and FP32 operands in the export -- e.g.
        ModelOpt's NVFP4 exporter casts a GEMM's activation to fp16 while the
        GEMM's output stays FLOAT. The runtime builds one shader per node from
        its data input, so such a node converts fine and computes garbage (the
        ModelOpt case: all-NaN output). Same exemptions as _audit: Cast,
        the mixed-width-by-design ops and the linear-attention state slots."""
        return GraphFp16._audit(dag_model, GraphFp16._decls(dag_model), set())

    # ------------------------------------------------------------------
    # --act bf16: per-op partition with boundary Casts
    # ------------------------------------------------------------------

    @staticmethod
    def _bf16_payload(name, arr):
        """`arr` (fp32, or fp16 widened exactly) as a BFLOAT16 initializer.

        REAL bf16 payload bits, not float16 bytes under a bf16 declaration:
        the loader keys on the initializer's own dtype, and bf16-declared fp16
        bytes decode as noise. Round to nearest even from the source VALUE --
        for an fp16 source that is fp16 -> fp32 (exact) -> bf16, which is NOT
        exact (bf16 keeps 8 significant bits to fp16's 11), so the caller
        records the round trip in the numeric audit.
        """
        bits = np.ascontiguousarray(
            bf16_weights.fp32_to_bf16_bits(
                np.asarray(arr, dtype=np.float32)).astype(np.uint16))
        packed = numpy_helper.from_array(bits, name)
        packed.data_type = _BFLOAT16
        return packed

    @staticmethod
    def _bf16_kind(node, decls, quantized) -> tuple:
        """(kind, why) of one node under --act bf16 with Cast boundaries.

          "bf16"   runs its _bf16 module: every float operand is bf16;
          "mover"  copies words: its float tensors take its data input's width;
          "cast"   an export Cast: reads any width, writes its `to`;
          "any"    reads metadata only (Shape), never the bytes;
          "island" no bf16 kernel: keeps the export's widths, `why` says why.
        """
        op = node.op_type
        if op == "Cast":
            return "cast", None
        if op == "Shape":
            return "any", None
        if op in _BF16_MOVERS:
            return "mover", None
        if op not in BF16_ACT_OPS:
            return "island", f"{op}: no bf16 activation kernel (core/Bf16Gate.hpp)"
        if op not in shader_map.HOST_ONLY_OPS and (
                op not in shader_map._PRIMARY
                or not shader_map.fp16_dtype_driven(op)):
            return "island", f"{op}: its build does not follow the tensor dtype"
        why = GraphFp16._bf16_node_blocker(node, decls, quantized)
        if why is not None:
            return "island", why
        return "bf16", None

    @staticmethod
    def _bf16_phase_nodes(dag_model, cand) -> Dict[str, str]:
        """{node: why} for the Sin/Cos nodes and the backward cone of their
        phase, which keep the export's width under --act bf16.

        Range is not the problem there -- precision is: bf16 keeps 8
        significant bits, so `position * inv_freq` at position 300 has a
        spacing of 2 rad and a bf16 position id is not even an integer past
        256. HF computes RoPE in fp32 and the exports declare that stretch
        FLOAT; holding it keeps the cos/sin tables FP32, which RotaryEmbedding
        reads at their own width (_BF16_OTHER_FLOAT_SLOTS) -- no Cast at all.
        The walk is _hold_phase_chains': elementwise/view/MatMul producers,
        stopping at a Cast (it keeps its kind and converts to what its
        readers need), a graph input or an initializer; a cone past
        _PHASE_CONE_MAX nodes is not a phase and is left alone.
        """
        producer_of = {}
        for node in dag_model.nodes.values():
            for out in node.outputs:
                producer_of[out.get("name")] = node
        held: Dict[str, str] = {}
        for node in dag_model.nodes.values():
            if node.op_type not in ("Sin", "Cos") or not node.inputs:
                continue
            seen_nodes, cone = {}, set()
            stack = [node.inputs[0].get("name")]
            ok = True
            while stack:
                name = stack.pop()
                if name in cone or name not in cand:
                    continue
                cone.add(name)
                src = producer_of.get(name)
                if src is None or src.name in seen_nodes:
                    continue
                seen_nodes[src.name] = src
                if len(seen_nodes) > _PHASE_CONE_MAX or \
                        src.op_type not in _PHASE_OPS:
                    ok = False
                    break
                if src.op_type == "Cast":
                    continue
                stack.extend(i.get("name") for i in src.inputs)
            if not ok:
                continue
            why = (f"{node.op_type} '{node.name}': RoPE phase held at the "
                   "export width (bf16 has 8 significant bits)")
            held[node.name] = why
            for nm, src in seen_nodes.items():
                if src.op_type != "Cast":
                    held.setdefault(nm, why)
        return held

    @staticmethod
    def bf16_partition(dag_model, quantized=None, keep_outputs=None) -> dict:
        """Who runs bf16, who keeps the export's width, and where the Casts go.

        Computes, writes nothing. Every FLOAT/FLOAT16 tensor (the candidates)
        gets a WIDTH, decided by its producer:

          * a graph input:                bf16 (the host contract of a
                                          --act bf16 artifact: every float
                                          input, KV cache included, is bf16);
          * a "bf16" node's output:       bf16;
          * an island node's output:      the width the export declared it;
          * a mover's output:             its data input's width (a Reshape
                                          between two fp16 islands stays fp16);
          * an export Cast's output:      bf16 if a bf16 node reads it (its
                                          `to` is rewritten), else `to`;
          * an initializer:               per consumer (below).

        Then every consumer edge whose reader needs another width gets a Cast
        (one per tensor and target width, shared by all such readers): a bf16
        node needs bf16; an island needs the width the export declared for
        that tensor -- fp16 for an fp16 export, fp32 for an fp32 one -- so an
        island runs exactly the kernel it ran before; movers, Casts and Shape
        read whatever arrives. A graph output is bf16 too, except the names in
        `keep_outputs` (logits: TopK has no bf16 kernel), and gets a Cast in
        front of it when its producer leaves it at another width.

        An initializer read only by bf16 nodes becomes a bf16 payload; one read
        only by islands/movers keeps its bytes; one read by both is split into
        a bf16 copy (`<name>__bf16`) for the bf16 readers.
        """
        quantized = set(quantized or ())
        if keep_outputs is None:
            keep_outputs = _BF16_KEEP_OUTPUTS if "TopK" not in BF16_ACT_OPS \
                else frozenset()
        decls = GraphFp16._decls(dag_model)
        cand = GraphFp16._candidates(dag_model, decls, "bf16")

        def orig(name):
            return decls[name]["elem"]

        kinds, blockers = {}, {}
        producer, consumers = {}, {}
        held = GraphFp16._bf16_phase_nodes(dag_model, cand)
        for node in dag_model.nodes.values():
            kind, why = GraphFp16._bf16_kind(node, decls, quantized)
            if node.name in held and kind != "island":
                kind, why = "island", held[node.name]
            kinds[node.name] = kind
            touches = any(e.get("name") in cand
                          for e in list(node.inputs) + list(node.outputs))
            if kind == "island" and touches:
                blockers[node.name] = why
            for i, e in enumerate(node.inputs):
                if e.get("name"):
                    consumers.setdefault(e["name"], []).append((node, i))
            for e in node.outputs:
                if e.get("name"):
                    producer[e["name"]] = node

        graph_inputs = {e.get("name") for e in dag_model.inputs}
        inits = set(dag_model.initializers)
        width: Dict[str, int] = {}

        def width_of(name):
            """The width `name` is produced at (iterative over mover chains)."""
            chain = []
            cur = name
            while True:
                if cur in width:
                    w = width[cur]
                    break
                if cur not in cand:
                    w = None
                    break
                if cur in graph_inputs:
                    w = _BFLOAT16
                    break
                if cur in inits:
                    w = orig(cur)      # a mover/island reading a constant
                    break
                node = producer.get(cur)
                if node is None:
                    w = orig(cur)
                    break
                kind = kinds[node.name]
                if kind == "bf16":
                    w = _BFLOAT16
                    break
                if kind == "cast":
                    # bf16 only when some reader NEEDS bf16; a reader in a
                    # slot that takes its own format (a FusedElemwise leaf)
                    # keeps the export's `to`, and with it the precision the
                    # export asked for.
                    readers = consumers.get(cur, ())
                    w = _BFLOAT16 if any(
                        kinds[c.name] == "bf16" and
                        _bf16_slot_rule(c.op_type, i) is None
                        for c, i in readers) else orig(cur)
                    break
                if kind == "mover" and node.inputs and \
                        node.inputs[0].get("name") in cand:
                    chain.append(cur)
                    cur = node.inputs[0]["name"]
                    continue
                w = orig(cur)
                break
            for n in chain + [cur]:
                if n in cand and n not in width:
                    width[n] = w if w is not None else orig(n)
            return width.get(name, w)

        for name in sorted(cand):
            width_of(name)

        # Initializer uses, and the casts consumer edges need.
        init_uses: Dict[str, Set[int]] = {}
        # Constants read only in a slot that takes their own format: they
        # keep their bytes unless another reader already needs a bf16 copy
        # (then that copy serves this reader too -- no second payload).
        init_any: Set[str] = set()
        casts: Dict[tuple, list] = {}      # (tensor, target) -> [(node, i)]
        for node in dag_model.nodes.values():
            kind = kinds[node.name]
            for i, e in enumerate(node.inputs):
                name = e.get("name")
                if name not in cand:
                    continue
                rule = _bf16_slot_rule(node.op_type, i) \
                    if kind == "bf16" else None
                if rule is not None:
                    # A slot that reads its operand at its own format
                    # (_BF16_OTHER_FLOAT_SLOTS): "any" takes every width,
                    # "fp32" takes fp32 or bf16 -- an fp16 tensor there is
                    # widened (exactly) to fp32, an fp16 constant rounded to
                    # a bf16 payload.
                    if name in inits and name not in graph_inputs:
                        if rule == "any" or orig(name) == _FLOAT:
                            init_any.add(name)
                        else:
                            init_uses.setdefault(name, set()).add(_BFLOAT16)
                        continue
                    if rule == "fp32" and width[name] == _FLOAT16:
                        casts.setdefault((name, _FLOAT), []).append((node, i))
                    continue
                if kind == "bf16":
                    need = _BFLOAT16
                elif kind == "island":
                    need = orig(name)
                else:
                    # mover / cast / any: read whatever arrives. A mover over
                    # a constant keeps the constant's bytes.
                    if name in inits:
                        init_uses.setdefault(name, set()).add(orig(name))
                    continue
                if name in inits and name not in graph_inputs:
                    init_uses.setdefault(name, set()).add(need)
                    continue
                if width[name] != need:
                    casts.setdefault((name, need), []).append((node, i))
        for name in init_any - set(init_uses):
            init_uses[name] = {orig(name)}

        out_casts = {}
        for e in dag_model.outputs:
            name = e.get("name")
            if name not in cand:
                continue
            want = orig(name) if name in keep_outputs else _BFLOAT16
            if width[name] != want:
                out_casts[name] = want

        domain = {n for n, w in width.items() if w == _BFLOAT16}
        for name, uses in init_uses.items():
            if _BFLOAT16 in uses:
                domain.add(name)
        return {"decls": decls, "cand": cand, "width": width, "kinds": kinds,
                "blockers": blockers, "casts": casts, "out_casts": out_casts,
                "init_uses": init_uses, "domain": domain,
                "forced": cand - domain, "keep_outputs": set(keep_outputs)}

    @staticmethod
    def bf16_kind_conflicts(dag_model) -> List[str]:
        """Nodes of the FINISHED graph that bind bf16 beside fp16/fp32.

        The runtime's _bf16 module decodes every 16-bit operand as bf16 and
        reads no other float width (core/Bf16Gate.cpp check_bf16_activation),
        so such a node computes on noise. Exempt: Cast (converting is its
        job), the ops whose operands are independent widths by design, a
        bf16 WEIGHT in its weight slot (MatMul/Gemm B, Gather table) beside
        fp16/fp32 activations, and the slots _BF16_OTHER_FLOAT_SLOTS lets
        read their own format (the same table the runtime gate applies).
        """
        decls = GraphFp16._decls(dag_model)
        out = []
        for node in dag_model.nodes.values():
            if node.op_type == "Cast" or node.op_type in _MIXED_WIDTH_OPS \
                    or node.op_type in _STATE_SLOTS:
                continue
            wslots = _BF16_WEIGHT_SLOTS.get(node.op_type, ())
            bf, other = [], []
            for i, e in enumerate(node.inputs):
                d = decls.get(e.get("name"))
                if d is None or not d["float"]:
                    continue
                if i in wslots and e["name"] in dag_model.initializers:
                    continue
                rule = _bf16_slot_rule(node.op_type, i)
                if rule == "any" or (rule == "fp32" and
                                     d["elem"] in (_FLOAT, _BFLOAT16)):
                    continue
                (bf if d["elem"] == _BFLOAT16 else other).append(e["name"])
            for e in node.outputs:
                d = decls.get(e.get("name"))
                if d is None or not d["float"]:
                    continue
                (bf if d["elem"] == _BFLOAT16 else other).append(e["name"])
            if bf and other:
                out.append(f"{node.op_type} {node.name}: bfloat16 "
                           f"{sorted(set(bf))[:3]} beside "
                           f"{sorted(set(other))[:3]}")
        return out

    @staticmethod
    def _apply_bf16_casts(dag_model, summary, verbose=True, skip_weights=None,
                          quantized=None, keep_outputs=None) -> dict:
        """Write bf16_partition into the graph: declarations, payloads, Casts."""
        skip_weights = skip_weights or set()
        tag = "[half-domain:bf16]"
        part = GraphFp16.bf16_partition(dag_model, quantized=quantized,
                                        keep_outputs=keep_outputs)
        decls, width = part["decls"], part["width"]
        summary.update(domain=part["domain"], forced=part["forced"],
                       blockers=part["blockers"], conflicts=[], casts=0,
                       cast_pairs={})
        name_of = {_FLOAT: "float32", _FLOAT16: "float16",
                   _BFLOAT16: "bfloat16"}
        short = {_FLOAT: "fp32", _FLOAT16: "fp16", _BFLOAT16: "bf16"}

        # 1. Declarations: each candidate at its width, everywhere it is spelled.
        for name, w in width.items():
            for slot, key in decls[name]["slots"]:
                cur = slot.get(key)
                if isinstance(cur, str):
                    if cur in ("float32", "float16") and cur != name_of[w]:
                        slot[key] = name_of[w]
                        summary["declared"] += 1
                elif cur in (_FLOAT, _FLOAT16) and cur != w:
                    slot[key] = w
                    summary["declared"] += 1
        # Graph inputs are bf16 even when the dtype was spelled as an int.
        for e in dag_model.inputs:
            if e.get("name") in width and width[e["name"]] == _BFLOAT16:
                e["dtype"] = "bfloat16"
        # An export Cast whose output became bf16 now converts to bf16.
        for node in dag_model.nodes.values():
            if part["kinds"][node.name] == "cast" and node.outputs:
                out = node.outputs[0].get("name")
                if width.get(out) == _BFLOAT16:
                    node.attributes["to"] = _BFLOAT16

        # 2. Initializer payloads.
        deferred = getattr(dag_model, "deferred_weights", {}) or {}
        renamed_init = {}
        for name in sorted(part["init_uses"]):
            uses = part["init_uses"][name]
            if _BFLOAT16 not in uses:
                continue
            init = dag_model.initializers.get(name)
            if init is None:
                continue
            target = name
            if len(uses) > 1:
                # Read by a bf16 node AND by an island/mover: the island keeps
                # the export's bytes, the bf16 readers get their own copy.
                target = name + "__bf16"
                renamed_init[name] = target
            else:
                for slot, key in decls[name]["slots"]:
                    cur = slot.get(key)
                    if cur in (_FLOAT, _FLOAT16):
                        slot[key] = _BFLOAT16
                        summary["declared"] += 1
            if name in skip_weights and target == name:
                summary.setdefault("deferred_to_quantizer", []).append(name)
                continue
            if name in deferred:
                if verbose:
                    print(f"{tag} '{name}' is streamed; its payload dtype is "
                          "decided at write time")
                continue
            if int(init.data_type) not in (_FLOAT, _FLOAT16):
                continue
            raw = numpy_helper.to_array(init)
            src = raw.astype(np.float32)
            packed = GraphFp16._bf16_payload(target, src)
            dag_model.initializers[target] = packed
            summary["bytes_before"] += raw.nbytes
            summary["bytes_after"] += len(packed.raw_data)
            summary["converted"] += 1
            numeric_audit.record_roundtrip(
                target, "bf16", src,
                bf16_weights.bf16_bits_to_fp32(
                    np.frombuffer(packed.raw_data, dtype=np.uint16)
                ).reshape(src.shape))
        if renamed_init:
            for node in dag_model.nodes.values():
                if part["kinds"][node.name] != "bf16":
                    continue
                for e in node.inputs:
                    if e.get("name") in renamed_init:
                        e["name"] = renamed_init[e["name"]]
                        e["dtype"] = _BFLOAT16

        # 3. Boundary Casts, each placed right after its source's producer
        #    (graph inputs: at the front), so the node order stays topological.
        shape_of = {}
        for node in dag_model.nodes.values():
            for e in list(node.inputs) + list(node.outputs):
                if e.get("name") and "shape" in e:
                    shape_of.setdefault(e["name"], e.get("shape"))
        for e in dag_model.inputs:
            shape_of.setdefault(e.get("name"), e.get("shape"))

        producer = {}
        for node in dag_model.nodes.values():
            for e in node.outputs:
                producer[e.get("name")] = node.name
        after: Dict[str, list] = {}     # producer node (or None) -> new casts
        used = set(dag_model.nodes)

        def new_cast(src, src_w, dst, dst_w):
            base = f"vkop_cast_{src}_to_{short[dst_w]}"
            nm, k = base, 1
            while nm in used:
                nm, k = f"{base}_{k}", k + 1
            used.add(nm)
            cast = Node("Cast", nm, {"to": dst_w},
                        [{"name": src, "shape": shape_of.get(src) or [],
                          "dtype": src_w}],
                        [{"name": dst, "shape": shape_of.get(src) or [],
                          "dtype": dst_w}])
            after.setdefault(producer.get(src), []).append(cast)
            summary["casts"] += 1
            pair = f"{short[src_w]}->{short[dst_w]}"
            summary["cast_pairs"][pair] = summary["cast_pairs"].get(pair, 0) + 1

        for (src, need), readers in sorted(part["casts"].items(),
                                           key=lambda kv: (kv[0][0], kv[0][1])):
            dst = f"{src}__{short[need]}"
            new_cast(src, width[src], dst, need)
            for node, i in readers:
                node.inputs[i]["name"] = dst
                node.inputs[i]["dtype"] = need

        # A graph output its producer leaves at the wrong width: the producer
        # writes `<name>__<width>`, a Cast writes the output name itself.
        for name, want in sorted(part["out_casts"].items()):
            src = f"{name}__{short[width[name]]}"
            for node in dag_model.nodes.values():
                for e in list(node.inputs) + list(node.outputs):
                    if e.get("name") == name:
                        e["name"] = src
            for casts in after.values():
                for c in casts:
                    for e in c.inputs:
                        if e.get("name") == name:
                            e["name"] = src
            producer[src] = producer.get(name)
            shape_of[src] = shape_of.get(name)
            new_cast(src, width[name], name, want)
            for e in dag_model.outputs:
                if e.get("name") == name:
                    e["dtype"] = name_of[want]
        for e in dag_model.outputs:
            name = e.get("name")
            if name in width and name not in part["out_casts"] and \
                    isinstance(e.get("dtype"), str) and \
                    e["dtype"] in ("float32", "float16"):
                e["dtype"] = name_of[width[name]]

        # An export Cast left converting bf16 to bf16 (its source became bf16
        # as well) is a copy: its readers take the source directly.
        drop = {}
        outs = {e.get("name") for e in dag_model.outputs}
        for node in dag_model.nodes.values():
            if part["kinds"][node.name] != "cast" or not node.inputs or \
                    not node.outputs:
                continue
            src, dst = node.inputs[0].get("name"), node.outputs[0].get("name")
            if width.get(src) == _BFLOAT16 and width.get(dst) == _BFLOAT16 \
                    and dst not in outs:
                drop[node.name] = (dst, src)
        if drop:
            alias = {dst: src for dst, src in drop.values()}
            for nm in drop:
                del dag_model.nodes[nm]
            for node in list(dag_model.nodes.values()) + [
                    c for cs in after.values() for c in cs]:
                for e in node.inputs:
                    if e.get("name") in alias:
                        e["name"] = alias[e["name"]]
            for nm in drop:
                if nm in after:
                    # Casts placed after a dropped node read its (now
                    # aliased) output: place them after the source's producer.
                    moved = after.pop(nm)
                    src = drop[nm][1]
                    after.setdefault(producer.get(src), []).extend(moved)
            summary["dropped_casts"] = len(drop)

        if after or drop:
            ordered = {}
            for c in after.get(None, ()):
                ordered[c.name] = c
            for nm, node in dag_model.nodes.items():
                ordered[nm] = node
                for c in after.get(nm, ()):
                    ordered[c.name] = c
            dag_model.nodes.clear()
            dag_model.nodes.update(ordered)
            dag_model.build_dependencies()

        # 4. The mixed-kind refusal, on the finished graph.
        conflicts = GraphFp16.bf16_kind_conflicts(dag_model)
        summary["conflicts"] = conflicts
        if conflicts:
            summary["refused"] = True
            if verbose:
                print(f"{tag} REFUSED: {len(conflicts)} node(s) would bind "
                      "bf16 beside another float width:")
                for line in conflicts[:10]:
                    print(f"{tag}   {line}")
            return summary

        if verbose:
            for node_name, why in sorted(part["blockers"].items())[:20]:
                print(f"{tag} keeps the export width: {node_name}: {why}")
            if len(part["blockers"]) > 20:
                print(f"{tag} ... and {len(part['blockers']) - 20} more")
            pairs = ", ".join(f"{k} x{v}" for k, v in
                              sorted(summary["cast_pairs"].items()))
            print(f"{tag} {len(part['domain'])} tensor(s) bf16, "
                  f"{len(part['forced'])} keep the export width, "
                  f"{summary['casts']} boundary Cast(s)"
                  + (f" ({pairs})" if pairs else "") + ", "
                  f"{summary['converted']} weight(s) to bf16 "
                  f"({summary['bytes_before']} -> {summary['bytes_after']} "
                  "bytes)")
        return summary

    @staticmethod
    def apply(dag_model, target_backend: str = "buffer",
              verbose: bool = True, skip_weights=None,
              half: str = "fp16", quantized=None,
              keep_outputs=None) -> dict:
        """Declare and store the FP16 domain; returns what it did.

        Under half="bf16" the work is done by _apply_bf16_casts (per-op
        partition plus boundary Casts); there `keep_outputs` names graph
        outputs that keep the export's width (default _BF16_KEEP_OUTPUTS).

        `skip_weights` names initializers whose BYTES must not be rewritten
        even when their tensor is in the domain. A weight-only quantization
        pass (-q int8/int4/fp8/nf4/nvfp4/awq) computes its scale and codes from
        the stored values, so downcasting it here first would move every code
        that sits on a rounding boundary: measured as a single int8 code
        flipping 33 -> 34 in test_kn_weight_layout, and it would equally change
        what the streamed and in-memory payloads agree on. The DECLARATION
        still flips -- the op runs fp16 -- and the quantizer then replaces the
        payload outright, so the width of the intermediate never reaches the
        artifact.

        Nothing is written when the domain is empty or the audit finds a
        conflict, so a graph that cannot be flipped stays exactly the model it
        arrived as.
        """
        skip_weights = skip_weights or set()
        summary = {"converted": 0, "declared": 0, "skipped_backend": False,
                   "refused": False, "out_of_range": [], "bytes_before": 0,
                   "bytes_after": 0, "domain": set(), "forced": set(),
                   "blockers": {}, "conflicts": []}
        if target_backend != "buffer":
            summary["skipped_backend"] = True
            if verbose:
                print(f"[half-domain:{half}] {target_backend} backend: this pass only "
                      "rewrites declarations, and the image backend ignores "
                      "them (it builds every graph input as fp16), so there "
                      "is nothing to declare here.")
            return summary

        if half == "bf16":
            return GraphFp16._apply_bf16_casts(
                dag_model, summary, verbose=verbose,
                skip_weights=skip_weights, quantized=quantized,
                keep_outputs=keep_outputs)

        plan = GraphFp16.plan(dag_model, half, quantized=quantized)
        (summary["domain"], summary["forced"], summary["blockers"],
         summary["conflicts"]) = (plan["domain"], plan["forced"],
                                  plan["blockers"], plan["conflicts"])
        if plan["conflicts"]:
            summary["refused"] = True
            if verbose:
                print(f"[half-domain:{half}] REFUSED: {len(plan['conflicts'])} node(s) "
                      "would mix FP32 and FP16 operands, which the runtime "
                      "cannot execute -- it picks one shader per node from its "
                      "data input. A boundary between the two domains needs a "
                      "Cast node, and this graph does not have one:")
                for line in plan["conflicts"][:10]:
                    print(f"[half-domain:{half}]   {line}")
            return summary

        if verbose:
            for node_name, why in sorted(plan["blockers"].items()):
                print(f"[half-domain:{half}] stays FP32: {node_name}: {why}")

        # The half-width target (fp16 here; bf16 returned above). Everything
        # downstream -- the archive keys, the loader's ElemKind, the panel's
        # shader builds -- follows from the dtype this writes.
        # A name in the domain is rewritten FROM the exported width, which is
        # always FLOAT -- a slot already spelled bfloat16 or float16 was put
        # there by an earlier decision (an op that must stay wide, a Cast's
        # `to` attribute) and is not this pass's to change. Widening every
        # _FLOAT here would have rewritten a Cast(to=fp32) output to bf16,
        # i.e. silently inverted the boundary it exists to provide.
        half_elem = _FLOAT16
        half_name = "float16"
        for name in plan["domain"]:
            for slot, key in plan["decls"][name]["slots"]:
                if slot.get(key) == _FLOAT:
                    slot[key] = half_elem
                    summary["declared"] += 1
                elif slot.get(key) == "float32":
                    slot[key] = half_name
                    summary["declared"] += 1

        deferred = getattr(dag_model, "deferred_weights", {}) or {}
        for name in sorted(plan["domain"]):
            if name in skip_weights:
                # A weight-only pass owns this payload; see the docstring.
                summary.setdefault("deferred_to_quantizer", []).append(name)
                continue
            init = dag_model.initializers.get(name)
            if init is None or int(init.data_type) != _FLOAT:
                continue
            if name in deferred:
                # No bytes in the DAG to rewrite; weight_stream lays the
                # payload down and records its dtype at trailer time.
                if verbose:
                    print(f"[half-domain:{half}] '{name}' is streamed; its payload "
                          "dtype is decided at write time")
                continue
            arr = numpy_helper.to_array(init)
            summary["bytes_before"] += arr.nbytes
            if arr.size:
                peak = float(np.abs(arr).max())
                if peak > _FP16_MAX:
                    summary["out_of_range"].append((name, peak))
            with np.errstate(over="ignore"):
                # Past 65504 a value saturates to Inf. numpy would warn, and
                # the report above already names the tensor: this is the
                # defined outcome, not an error the pass can avert.
                halves = np.ascontiguousarray(arr.astype(np.float16))
            fp16 = numpy_helper.from_array(halves, name)
            dag_model.initializers[name] = fp16
            summary["bytes_after"] += len(fp16.raw_data)
            summary["converted"] += 1
            # The kernel reads `halves` widened back to fp32, so that is the
            # pair the numeric audit compares (numeric_audit.py). A weight the
            # pass left FP32 has no row: nothing was costed against a format.
            numeric_audit.record_roundtrip(name, "fp16", arr,
                                           halves.astype(np.float32))

        if verbose:
            # The tag says the DOMAIN pass, not the width: under
            # --act bf16 the tensors are BFLOAT16 and a "FP16
            # domain" line reads as if the flag had been ignored (it did
            # exactly that in a debugging session). Report the actual target.
            print(f"[half-domain:{half}] {len(plan['domain'])} tensor(s) half, "
                  f"{len(plan['forced'])} stay FP32, "
                  f"{summary['converted']} weight(s) downcast "
                  f"({summary['bytes_before']} -> {summary['bytes_after']} bytes)")
            if summary["out_of_range"]:
                print(f"[half-domain:{half}] WARNING: {len(summary['out_of_range'])} "
                      "tensor(s) hold a magnitude past FP16's 65504 and become "
                      "Inf: " + ", ".join(
                          f"{n} (max {v:.6g})"
                          for n, v in summary["out_of_range"][:5]))
        return summary
