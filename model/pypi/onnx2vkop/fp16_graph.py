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
except ImportError:  # run as a flat script from inside onnx2vkop/, as converter.py supports
    import shader_map
    import numeric_audit
    import bf16_weights

_FLOAT = onnx.TensorProto.FLOAT
_FLOAT16 = onnx.TensorProto.FLOAT16
# The other half-width float. Same 16-bit storage and 32-bit arithmetic as
# fp16, but 8 exponent bits instead of 5: fp32's range, ~3 significant decimal
# digits. MiniMax-H3 needs it because its activations legitimately exceed
# fp16's 65504 (hidden states reach ~4200 and a 14336-wide FFN accumulation
# crosses the ceiling).
_BFLOAT16 = onnx.TensorProto.BFLOAT16
_DOUBLE = onnx.TensorProto.DOUBLE
_FLOAT_FAMILY = (_FLOAT, _FLOAT16, _DOUBLE)

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
        rank = {_FLOAT: 3, _DOUBLE: 2, _FLOAT16: 1}

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
    def plan(dag_model, half: str = "fp16") -> dict:
        """{domain, forced, reasons, blockers, conflicts} for this DAG.

        Computes, writes nothing. `blockers` maps a node name to why it cannot
        enter the domain, which is what makes a partially-rewritten graph
        explainable: the FP32 island is always reachable from a named op.
        """
        decls = GraphFp16._decls(dag_model)
        candidates = {n for n, d in decls.items() if d["elem"] == _FLOAT}
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
                "conflicts": GraphFp16._audit(dag_model, decls, domain)}

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
            # (Not reached under --half-precision bf16: the caller skips this
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
            # so the boundary is real -- bf16 under --half-precision bf16.
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
    def _audit(dag_model, decls: Dict[str, dict], domain: Set[str]) -> List[str]:
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
            for entry in list(node.inputs) + list(node.outputs):
                name = entry.get("name")
                if name in exempt or (name or "").startswith(
                        _STATE_NAME_PREFIXES):
                    continue
                d = decls.get(entry.get("name"))
                if d is None or not d["float"]:
                    continue
                if d["elem"] == _FLOAT16 or entry["name"] in domain:
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
        return out

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

    @staticmethod
    def apply(dag_model, target_backend: str = "buffer",
              verbose: bool = True, skip_weights=None,
              half: str = "fp16") -> dict:
        """Declare and store the FP16 domain; returns what it did.

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

        plan = GraphFp16.plan(dag_model, half)
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

        # The half-width target: fp16 by default, bf16 when the caller asked
        # for the wider exponent (see _BFLOAT16). Everything downstream --
        # the archive keys, the loader's ElemKind, the panel's shader builds --
        # follows from the dtype this writes.
        # A name in the domain is rewritten FROM the exported width, which is
        # always FLOAT -- a slot already spelled bfloat16 or float16 was put
        # there by an earlier decision (an op that must stay wide, a Cast's
        # `to` attribute) and is not this pass's to change. Widening every
        # _FLOAT here would have rewritten a Cast(to=fp32) output to bf16,
        # i.e. silently inverted the boundary it exists to provide.
        half_elem = _BFLOAT16 if half == "bf16" else _FLOAT16
        half_name = "bfloat16" if half == "bf16" else "float16"
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
            if half == "bf16":
                # REAL bf16 payload bits, not float16 bytes under a bf16
                # declaration. The declaration alone is not enough: the loader
                # keys on the initializer's own dtype string, and a bf16 tensor
                # declared bf16 but holding fp16 bytes decodes as noise. (The
                # merge left exactly that shape behind -- declared BFLOAT16,
                # still wrote float16 -- which is why this branch exists.)
                bits = np.ascontiguousarray(
                    bf16_weights.fp32_to_bf16_bits(arr).astype(np.uint16))
                packed = numpy_helper.from_array(bits, name)
                packed.data_type = _BFLOAT16
                dag_model.initializers[name] = packed
                summary["bytes_after"] += len(packed.raw_data)
                summary["converted"] += 1
                continue
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
            # --half-precision bf16 the tensors are BFLOAT16 and a "FP16
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
