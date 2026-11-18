#!/usr/bin/env python3
"""Trim a Qwen3.5-style hybrid decoder ONNX to its first N layers.

Diagnostic one-shot tool for a bandwidth experiment: the 27B int8 model does
not fit in 20.9 GB of VRAM, so this produces a prefix model that does, in order
to measure decode ``Run()`` at a known working-set size.  The output is NOT a
semantically usable model -- it is 40 layers where the checkpoint has 64.

What it does
------------
* Keeps every node that belongs to ``layers.0 .. layers.N-1`` plus the prologue
  (RoPE + input RMSNorm) and the original final RMSNorm + ``lm_head`` nodes.
* Replaces the final RMSNorm's weight input: the source graph applies
  ``layers.N.input_layernorm.weight`` to the residual stream after layer N-1;
  the trimmed graph applies ``norm.weight`` to that same tensor instead, exactly
  the code path the source uses after layer 63.
* Drops the layer-N-onward nodes and initializers, and the per-layer graph
  inputs/outputs (``past_key_values_*``, ``conv_state_*``,
  ``recurrent_state_*`` / ``present_*``, ``*_state_out_*``) whose layer index is
  >= N.
* Leaves the external-data file untouched: every kept initializer keeps its
  original ``offset``/``length`` into the same ``.weights.bin``.  Only the graph
  changes, so the streaming converter reads exactly the kept weights.

Structural facts this relies on (asserted, not assumed)
------------------------------------------------------
* Nodes are emitted layer-block-major and blocks never interleave: the highest
  node index consuming a ``layers.L.*`` initializer is below the first node
  index consuming any ``layers.{L+1}.*`` initializer.
* The first consumer of ``layers.L.*`` is the ``Mul`` that applies
  ``layers.L.input_layernorm.weight``; its other input is the normalized
  residual stream produced after layer L-1 (or by the prologue, for L = 0).
* After each block comes a fixed residual+RMSNorm chain whose last node is a
  ``Cast`` to fp16 -- the tensor the next layer's norm-apply ``Mul`` consumes.
* The graph's final ``Mul(norm.weight, <normalized>)`` feeds a single MatMul
  with a transposed ``lm_head.weight`` that produces ``logits``.

If any assumption fails the tool aborts instead of writing a wrong graph.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Dict, List, Tuple

import onnx
from onnx import TensorProto, helper

LAYER_INIT_RE = re.compile(r"^layers\.(\d+)\.")
STATE_IN_RE = re.compile(r"^(past_key_values|conv_state|recurrent_state)_(\d+)$")
STATE_OUT_RE = re.compile(
    r"^(present_key_values|conv_state_out|recurrent_state_out)_(\d+)$")

DTYPE_SIZE = {
    TensorProto.FLOAT: 4, TensorProto.UINT8: 1, TensorProto.INT8: 1,
    TensorProto.UINT16: 2, TensorProto.INT16: 2, TensorProto.INT32: 4,
    TensorProto.INT64: 8, TensorProto.BOOL: 1, TensorProto.FLOAT16: 2,
    TensorProto.DOUBLE: 8, TensorProto.BFLOAT16: 2,
    TensorProto.FLOAT8E4M3FN: 1, TensorProto.FLOAT8E5M2: 1,
}


class TrimError(RuntimeError):
    pass


def die(msg: str) -> "NoReturn":  # noqa: F821
    raise TrimError(msg)


def payload_bytes(t: onnx.TensorProto) -> int:
    """Byte extent of an initializer, from its external length or its dims."""
    if t.data_location == TensorProto.EXTERNAL:
        meta = {e.key: e.value for e in t.external_data}
        if "length" in meta and int(meta["length"]) > 0:
            return int(meta["length"])
        # Fall through to dims: a 0 length means "to EOF" to some loaders.
    n = 1
    for d in t.dims:
        n *= d
    return n * DTYPE_SIZE[t.data_type]


def node_sig(n: onnx.NodeProto) -> Tuple[str, Tuple[str, ...]]:
    """Structural signature, ignoring tensor names and Constant payloads.

    Constants carry their values in attributes; those differ numerically across
    layers only where a layer index leaks into a shape, so comparing them would
    make the tail check report a false mismatch.  Op type plus attribute *names*
    plus scalar/list attribute values is what the residual-norm chain is: the
    same ops in the same order.
    """
    attrs = []
    for a in n.attribute:
        if a.name == "value":
            attrs.append(("value", a.t.data_type, tuple(a.t.dims)))
        else:
            attrs.append((a.name, helper.get_attribute_value(a)))
    return n.op_type, tuple(attrs)


def analyze(model: onnx.ModelProto, src_path: str = "") -> Dict:
    g = model.graph
    nodes = list(g.node)
    prod: Dict[str, int] = {}
    for i, n in enumerate(nodes):
        for o in n.output:
            if o in prod:
                die("tensor %r has two producers (%d and %d)" % (o, prod[o], i))
            prod[o] = i

    init_names = {t.name for t in g.initializer}
    layer_inits: Dict[int, List[str]] = {}
    for nm in init_names:
        m = LAYER_INIT_RE.match(nm)
        if m:
            layer_inits.setdefault(int(m.group(1)), []).append(nm)
    if not layer_inits:
        die("no initializer is named layers.<L>.* -- not the expected export")
    n_layers = max(layer_inits) + 1
    missing = [L for L in range(n_layers) if L not in layer_inits]
    if missing:
        die("layer initializers are not contiguous; missing layers %s" % missing)

    consumers: Dict[str, List[int]] = {}
    for i, n in enumerate(nodes):
        for x in n.input:
            consumers.setdefault(x, []).append(i)

    start = {}
    end = {}
    for L, names in layer_inits.items():
        idxs: List[int] = []
        for nm in names:
            idxs.extend(consumers.get(nm, []))
        if not idxs:
            die("layer %d has %d initializer(s) no node reads" % (L, len(names)))
        start[L], end[L] = min(idxs), max(idxs)
    for L in range(n_layers - 1):
        if end[L] >= start[L + 1]:
            die("layer blocks interleave: end[%d]=%d >= start[%d]=%d"
                % (L, end[L], L + 1, start[L + 1]))

    # The norm-apply Mul that opens each layer block.
    norm_apply = {}
    for L in range(n_layers):
        i = start[L]
        n = nodes[i]
        want = "layers.%d.input_layernorm.weight" % L
        if n.op_type != "Mul" or want not in n.input or len(n.input) != 2:
            die("layer %d does not start with Mul(%s, <normalized>): got %s %r"
                % (L, want, n.op_type, list(n.input)))
        other = [x for x in n.input if x != want][0]
        norm_apply[L] = (i, other)
        if other not in prod:
            die("layer %d norm-apply input %r has no producer" % (L, other))

    # Final RMSNorm + lm_head, identified by the initializers they read.
    if "norm.weight" not in init_names:
        die("no 'norm.weight' initializer")
    if "lm_head.weight" not in init_names:
        die("no 'lm_head.weight' initializer")
    norm_mul_i = consumers["norm.weight"]
    if len(norm_mul_i) != 1:
        die("norm.weight is read by %d nodes, expected 1" % len(norm_mul_i))
    norm_mul_i = norm_mul_i[0]
    lmh_use = consumers["lm_head.weight"]
    if len(lmh_use) != 1:
        die("lm_head.weight is read by %d nodes, expected 1" % len(lmh_use))
    lmh_i = lmh_use[0]
    lmh_node = nodes[lmh_i]
    if lmh_node.op_type != "Transpose":
        die("lm_head.weight consumer is %r, expected a Transpose"
            % lmh_node.op_type)
    logits_outs = [o.name for o in g.output if o.name == "logits"]
    if not logits_outs:
        die("graph output 'logits' not found")
    logits_i = prod["logits"]
    logits_node = nodes[logits_i]
    if logits_node.op_type != "MatMul" or norm_mul_i not in [
            prod[x] for x in logits_node.input if x in prod]:
        die("logits is not produced from the final RMSNorm Mul")
    norm_mul = nodes[norm_mul_i]
    if norm_mul.op_type != "Mul" or len(norm_mul.input) != 2:
        die("norm.weight consumer is not a 2-input Mul")
    normed_src = [x for x in norm_mul.input if x != "norm.weight"][0]

    # Tail region between layer N-1's block and layer N's norm-apply.
    return {
        "nodes": nodes, "prod": prod, "consumers": consumers,
        "init_names": init_names, "layer_inits": layer_inits,
        "n_layers": n_layers, "start": start, "end": end,
        "norm_apply": norm_apply,
        "norm_mul_i": norm_mul_i, "logits_i": logits_i,
        "lmh_use": lmh_i, "normed_src": normed_src,
        "logits_node": logits_node, "src_path": src_path,
    }


def check_tail_region(a: Dict, keep: int, g_output_names) -> Tuple[int, str]:
    """Validate the region between layer keep-1 and layer keep.

    Returns (index of the normalized-residual producer, that tensor's name).
    The source emits: MatMul(down_proj), Add(residual), Cast, [RMSNorm chain],
    Cast(fp16), then the dropped norm-apply Mul.
    """
    nodes = a["nodes"]
    prod = a["prod"]
    lo = a["end"][keep - 1] + 1
    hi = a["start"][keep]  # exclusive
    if lo >= hi:
        die("no residual/norm region between layer %d and %d" % (keep - 1, keep))
    normed = a["norm_apply"][keep][1]
    src_i = prod[normed]
    if not (lo <= src_i < hi):
        die("layer %d's norm input is produced at %d, outside the tail region "
            "[%d, %d)" % (keep, src_i, lo, hi))

    # What the region must *not* do is leak a value the trimmed graph needs.
    # A residual tensor produced here (the fp32 pre-norm Add) is legitimately
    # read again by the next block's attention-output Add -- that block is being
    # dropped, so the reference goes with it. What would be wrong is a graph
    # output produced here, or the normalized tensor's consumer being anything
    # other than the dropped norm-apply Mul (a second reader would be lost).
    out_names = set(g_output_names)
    for i in range(lo, hi):
        for t in nodes[i].output:
            if t in out_names:
                die("tail node %d (%s) produces graph output %r, which the "
                    "trimmed graph would have to keep" % (i, nodes[i].op_type, t))
    n_consumers = a["consumers"].get(normed, [])
    if n_consumers != [a["start"][keep]]:
        die("normalized tensor %r is consumed at %s, expected only the "
            "dropped norm-apply at %d"
            % (normed, n_consumers, a["start"][keep]))
    last = nodes[src_i]
    last_attrs = {x.name: helper.get_attribute_value(x) for x in last.attribute}
    if last.op_type != "Cast" or last_attrs.get("to") != TensorProto.FLOAT16:
        die("residual/norm chain does not end in Cast(to=fp16): %r %s"
            % (last.op_type, last_attrs))
    return src_i, normed


def tail_signature(a: Dict, layer: int, src_i: int) -> List[Tuple[str, tuple]]:
    nodes = a["nodes"]
    lo = a["end"][layer] + 1
    # Skip a leading MatMul(down_proj): it is a weight node without an
    # initializer name and identical in every block type.
    return [node_sig(nodes[i]) for i in range(lo, src_i + 1)]


def reachable_nodes(candidates, out_names) -> List[int]:
    """Positions in `candidates` of nodes reachable (backwards) from out_names.

    `candidates` is a list of (source_index, NodeProto) pairs; positions matter
    because the same source node can appear only once, but the caller needs the
    surviving candidate order to rebuild the node list.
    """
    prod = {}
    for pos, (_, n) in enumerate(candidates):
        for o in n.output:
            prod[o] = pos
    seen = set()
    stack = [prod[o] for o in out_names if o in prod]
    while stack:
        pos = stack.pop()
        if pos in seen:
            continue
        seen.add(pos)
        for x in candidates[pos][1].input:
            if x in prod:
                stack.append(prod[x])
    return sorted(seen)


def retarget_external(initializers, src_onnx: str, out_path: str,
                      mode: str) -> Tuple[str, int]:
    """Make the kept initializers' external `location` loadable from out_path.

    The source records a relative ``llm27.weights.bin``, which every loader
    resolves against the directory of the .onnx it was handed.  That is fine as
    long as the trimmed model is written next to the weights; written elsewhere
    (the /data1 output directory, say) the same relative name resolves to a
    missing file.  The 51 GB side file is never copied or rewritten, so the
    three sanctioned outcomes are:

    * ``auto``      -- keep the relative name when it resolves next to the
                       output, otherwise rewrite ``location`` to a relative
                       path from the output directory to the source file (same
                       bytes, same offsets);
    * ``keep``      -- never touch the field (the caller promises the weights
                       are reachable from the output directory);
    * ``relative``  -- always rewrite to a relative path from the output dir.

    The ONNX spec requires a *relative* location (checker and ORT both reject
    an absolute one), so the rewrite is a relpath, never an absolute path.

    Returns (location_used, number_of_initializers_rewritten).
    """
    src_dir = os.path.dirname(os.path.abspath(src_onnx))
    rel = None
    for t in initializers:
        for e in t.external_data:
            if e.key == "location":
                rel = e.value
                break
        if rel is not None:
            break
    if rel is None:
        return "", 0

    src_abs = rel if os.path.isabs(rel) else os.path.join(src_dir, rel)
    out_dir = os.path.dirname(os.path.abspath(out_path))
    out_rel_ok = os.path.exists(
        rel if os.path.isabs(rel) else os.path.join(out_dir, rel))

    if mode == "keep" or (mode == "auto" and out_rel_ok):
        return rel, 0
    if not os.path.exists(src_abs):
        die("external weights %r resolve to %r from the source model, and that "
            "file does not exist" % (rel, src_abs))
    new_loc = os.path.relpath(os.path.abspath(src_abs), out_dir)
    rewritten = 0
    for t in initializers:
        for e in t.external_data:
            if e.key == "location":
                e.value = new_loc
                rewritten += 1
                break
    return new_loc, rewritten


def trim(model: onnx.ModelProto, keep: int, out_path: str,
         relocate: str, src_path: str) -> Tuple[onnx.ModelProto, Dict]:
    a = analyze(model, src_path)
    n_layers = a["n_layers"]
    if not 1 <= keep <= n_layers:
        die("--keep must be in [1, %d], got %d" % (n_layers, keep))
    if keep == n_layers:
        die("--keep %d is the whole model; nothing to trim" % keep)

    nodes = a["nodes"]
    src_i, normed = check_tail_region(
        a, keep, [o.name for o in model.graph.output])

    # The kept tail must be the same residual+norm chain the source uses after
    # its last layer, or the substitution changes semantics in an unintended way.
    ref_sig = tail_signature(a, n_layers - 1, a["prod"][a["normed_src"]])
    got_sig = tail_signature(a, keep - 1, src_i)
    if ref_sig != got_sig:
        die("residual/norm chain after layer %d differs from the one after the "
            "last layer:\n  ref=%s\n  got=%s"
            % (keep - 1, [s[0] for s in ref_sig], [s[0] for s in got_sig]))

    # --- assemble node list -------------------------------------------------
    # (source index, node) so the prefix identity check stays exact. The tail
    # is the contiguous span from the final norm Mul through the logits node,
    # so any Cast/Transpose the exporter leaves in between comes along; the
    # reachability pass then drops whatever logits does not need.
    tail_lo = min(a["norm_mul_i"], a["lmh_use"], a["logits_i"])
    tail_hi = max(a["norm_mul_i"], a["lmh_use"], a["logits_i"])
    candidates = [(i, nodes[i]) for i in range(src_i + 1)]
    candidates += [(i, nodes[i]) for i in range(tail_lo, tail_hi + 1)]

    # Rewire the final RMSNorm onto the trimmed residual stream BEFORE the
    # reachability pass: with the original input the norm chain would still
    # depend on the dropped layers 40..62 and every walk would pull them in.
    rewired = []
    for src_index, n in candidates:
        if src_index == a["norm_mul_i"]:
            n2 = onnx.NodeProto()
            n2.CopyFrom(nodes[a["norm_mul_i"]])
            ins = list(n2.input)
            ins[ins.index(a["normed_src"])] = normed
            del n2.input[:]
            n2.input.extend(ins)
            n = n2
        rewired.append((src_index, n))

    out_names = [o.name for o in model.graph.output]
    keep_idx = reachable_nodes(rewired, out_names)
    kept_pairs = [rewired[i] for i in keep_idx]
    dropped_unreachable = len(rewired) - len(kept_pairs)
    new_nodes = [n for _, n in kept_pairs]

    # --- prune initializers to those the kept nodes read --------------------
    used = {x for n in new_nodes for x in n.input}
    kept_inits = [t for t in model.graph.initializer if t.name in used]
    dropped_inits = len(model.graph.initializer) - len(kept_inits)

    # The side file is never copied: only the pointer is adjusted, and only
    # when the original relative name would not resolve from the output dir.
    loc, rewritten = retarget_external(kept_inits, a["src_path"], out_path,
                                       relocate)

    # --- graph inputs / outputs by layer index ------------------------------
    new_inputs = []
    dropped_in = []
    for v in model.graph.input:
        m = STATE_IN_RE.match(v.name)
        if m and int(m.group(2)) >= keep:
            dropped_in.append(v.name)
            continue
        new_inputs.append(v)
    new_outputs = []
    dropped_out = []
    for v in model.graph.output:
        m = STATE_OUT_RE.match(v.name)
        if m and int(m.group(2)) >= keep:
            dropped_out.append(v.name)
            continue
        new_outputs.append(v)

    produced = {o for n in new_nodes for o in n.output}
    for v in new_outputs:
        if v.name not in produced:
            die("graph output %r is not produced by the kept nodes" % v.name)

    # --- value_info: keep only tensors the trimmed graph still produces -----
    new_vi = [v for v in model.graph.value_info if v.name in produced]

    # --- build and validate -------------------------------------------------
    g = model.graph
    new_g = helper.make_graph(
        new_nodes, g.name,
        new_inputs, new_outputs, initializer=kept_inits)
    new_g.value_info.extend(new_vi)
    new_g.doc_string = (
        "TRIM DIAGNOSTIC: %d of %d decoder layers (source: %s). "
        "Outputs are NOT semantically meaningful; this graph exists to measure "
        "decode bandwidth at a working set that fits in VRAM."
        % (keep, n_layers, os.path.basename(out_path)))
    m = helper.make_model(new_g, opset_imports=list(model.opset_import),
                          producer_name="vkop-trim_llm_layers")
    m.ir_version = model.ir_version
    for e in model.metadata_props:
        m.metadata_props.add(key=e.key, value=e.value)

    dangling = check_dangling(m)
    if dangling:
        die("%d dangling node input(s), first: %s" % (len(dangling), dangling[:5]))

    # kept prefixes remain byte-identical to the source nodes, so the trimmed
    # graph computes exactly what the source computes there; the rewired norm
    # Mul is the only edited node and it is outside the prefix.
    prefix_pairs = [p for p in kept_pairs if p[0] < src_i + 1]
    identical = sum(1 for src_index, n in prefix_pairs
                    if n.SerializeToString() == nodes[src_index].SerializeToString())
    if identical != len(prefix_pairs):
        die("kept prefix is not byte-identical to the source nodes (%d/%d)"
            % (identical, len(prefix_pairs)))

    # Absolute-vs-relative external location is resolved by every loader
    # relative to the .onnx file, so a relative name needs the side file next
    # to the output. Report what will happen rather than assuming.
    ext_locs = {e.value for t in kept_inits for e in t.external_data
                if e.key == "location"}
    resolved = (loc if os.path.isabs(loc)
                else os.path.join(os.path.dirname(os.path.abspath(out_path)), loc))
    if loc and not os.path.exists(resolved):
        die("kept initializers point at external data %r which resolves to %r "
            "from the output path and does not exist" % (loc, resolved))

    stats = {
        "keep": keep, "n_layers": n_layers,
        "nodes_in": len(nodes), "nodes_out": len(new_nodes),
        "dropped_unreachable": dropped_unreachable,
        "inits_in": len(model.graph.initializer), "inits_out": len(kept_inits),
        "dropped_inits": dropped_inits,
        "inputs_in": len(model.graph.input), "inputs_out": len(new_inputs),
        "outputs_in": len(model.graph.output), "outputs_out": len(new_outputs),
        "dropped_inputs": dropped_in, "dropped_outputs": dropped_out,
        "value_info_in": len(model.graph.value_info),
        "value_info_out": len(new_vi),
        "init_bytes": sum(payload_bytes(t) for t in kept_inits),
        "init_bytes_source": sum(payload_bytes(t)
                                 for t in model.graph.initializer),
        "layer_init_counts": {},
        "external_locations": sorted(ext_locs),
        "external_rewritten": rewritten,
        "external_resolved": resolved,
        "external_size": (os.path.getsize(resolved)
                          if loc and os.path.exists(resolved) else 0),
        "prefix_nodes_identical": identical,
        "tail_signature": [s[0] for s in got_sig],
        "attention_layers": [
            L for L in range(keep)
            if any(t.name.startswith("layers.%d.self_attn." % L)
                   for t in kept_inits)],
        "rewired": {"norm_mul": a["norm_mul_i"],
                    "from": a["normed_src"], "to": normed},
    }
    for L in range(keep):
        stats["layer_init_counts"][L] = len(a["layer_inits"][L])
    # The checker resolves external-data locations against the *model path*
    # when it is handed a path and against the cwd when handed a proto, so the
    # real check runs in main() right after the file is written.
    return m, stats


def register_custom_ops() -> List[str]:
    """Register minimal schemas so onnx.checker can vet this graph.

    The export carries two custom ops with no ONNX schema (``Conv1d`` and
    ``LinearAttention``) and uses the opset-18 two-input ``ReduceMean`` form
    while declaring opset 17, so the *source* model fails onnx.checker as
    exported. Neither quirk is something the converter rejects, but without
    schemas the checker cannot walk the graph at all, and the structural checks
    it does perform (input resolution, initializer extents, attribute types)
    are exactly what this tool wants confirmed. Declaring the real attributes
    keeps the check honest rather than blanket-exempting them.

    Returns the list of custom ops registered.
    """
    from onnx import defs

    fp = defs.OpSchema.FormalParameter
    opt = defs.OpSchema.FormalParameterOption.Optional
    specs = (
        ("Conv1d", 3, 2, ("kernel",)),
        ("LinearAttention", 6, 2, ("dk", "dv", "heads")),
    )
    for op, n_in, n_out, attrs in specs:
        schema = defs.OpSchema(
            op, "", 1,
            inputs=[fp("in%d" % i, "T", "", param_option=opt)
                    for i in range(n_in)],
            outputs=[fp("out%d" % i, "T", "", param_option=opt)
                     for i in range(n_out)],
            type_constraints=[("T", ["tensor(float)", "tensor(float16)",
                                     "tensor(double)", "tensor(int64)",
                                     "tensor(int32)"], "")],
            attributes=[
                defs.OpSchema.Attribute(
                    a, defs.OpSchema.AttrType.INT, "INT attribute")
                for a in attrs
            ],
        )
        defs.register_schema(schema)
    return [s[0] for s in specs]


def _normalize_for_checker(graph) -> Dict[str, int]:
    """Rewrite pre-opset-18 node forms so a single-version checker run works.

    The export is opset-version-inconsistent: ``Split`` carries its sizes in the
    ``split`` *attribute* (opset <= 17) while ``ReduceMean`` already takes the
    axes as a second *input* (opset 18). No single opset_import makes the
    checker accept both, so on a throwaway copy the attribute form is rewritten
    into the input form (a Constant holding the same int64 values). Semantics
    are unchanged, and the real file is never touched.
    """
    counts = {"split_attr_to_input": 0}
    rebuilt: List[onnx.NodeProto] = []
    for n in graph.node:
        split_attr = [a for a in n.attribute if a.name == "split"]
        if n.op_type == "Split" and split_attr:
            vals = list(split_attr[0].ints)
            cname = (n.name or n.output[0]) + "__split_sizes"
            rebuilt.append(helper.make_node(
                "Constant", [], [cname], name=cname,
                value=helper.make_tensor(
                    cname + "_v", TensorProto.INT64, [len(vals)], vals)))
            n2 = onnx.NodeProto()
            n2.CopyFrom(n)
            del n2.attribute[:]
            n2.input.append(cname)
            counts["split_attr_to_input"] += 1
            n = n2
        rebuilt.append(n)
    del graph.node[:]
    graph.node.extend(rebuilt)
    return counts


def checker_validate(path: str, weights_path: str, tmp_path: str) -> Tuple[str, str]:
    """Run onnx.checker on the written model, working around export quirks.

    Returns ("ok" | "ok-with-shim" | "fail", detail).

    Two accommodations, both of which the source model needs as well:

    * opset: the nodes already use the opset-18 form (two-input ReduceMean) but
      declare 17; Split still uses the pre-18 attribute form. The temp copy is
      bumped to 18 and its Split attributes rewritten to inputs.
    * layout: onnx.checker requires external data to be a regular file inside
      the model's own directory, so a model written to another volume cannot be
      checked where it lands. The check therefore runs on a temp copy next to
      the weights file, with locations normalised to the bare basename. That is
      the same graph; only the path spelling differs.

    Neither the deliverable nor the weights file is modified.
    """
    registered = register_custom_ops()
    m = onnx.load(path, load_external_data=False)
    norm = _normalize_for_checker(m.graph)
    bump = False
    for op in m.opset_import:
        if op.domain in ("", "ai.onnx") and op.version < 18:
            op.version = 18
            bump = True
    base_name = os.path.basename(weights_path)
    for t in m.graph.initializer:
        if t.data_location != TensorProto.EXTERNAL:
            continue
        for e in t.external_data:
            if e.key == "location":
                e.value = base_name
    onnx.save(m, tmp_path)
    try:
        onnx.checker.check_model(tmp_path, full_check=False)
    except Exception as exc:  # noqa: BLE001
        return "fail", str(exc)[:400]
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
    notes = []
    if bump:
        notes.append("opset_import bumped to 18")
    if norm["split_attr_to_input"]:
        notes.append("%d Split attribute(s) rewritten to the input form"
                     % norm["split_attr_to_input"])
    notes.append("checked on a temp copy beside %s" % base_name)
    notes.append("custom-op schemas registered for %s" % ", ".join(registered))
    return "ok-with-shim", "; ".join(notes)


def validate_external_extents(initializers, weights_path: str) -> List[str]:
    """No bytes are read: check each external reference lands inside the file.

    The trimmed graph reuses the source file untouched, so a wrong offset would
    only surface at conversion time (or, worse, as silently wrong weights).
    Length is compared against dtype x dims as well, which catches a pointer
    taken from the wrong tensor.
    """
    errors: List[str] = []
    size = os.path.getsize(weights_path)
    for t in initializers:
        meta = {e.key: e.value for e in t.external_data}
        if t.data_location != TensorProto.EXTERNAL:
            continue
        off = int(meta.get("offset", 0))
        ln = int(meta.get("length", 0))
        want = 1
        for d in t.dims:
            want *= d
        want *= DTYPE_SIZE[t.data_type]
        if ln != want:
            errors.append("%s: external length %d != dtype*dims %d"
                          % (t.name, ln, want))
        if off < 0 or off + ln > size:
            errors.append("%s: extent [%d, %d) outside file of %d bytes"
                          % (t.name, off, off + ln, size))
        if int(meta.get("length", -1)) < 0:
            errors.append("%s: no 'length' entry" % t.name)
    return errors


def self_check_trimmed(model: onnx.ModelProto, keep: int,
                       attention_layers=None) -> Dict:
    """Independent verification that does not rely on onnx.checker.

    Checks:
      * no dangling node input (every name is an initializer, a graph input,
        or the output of an earlier node);
      * every graph output is produced;
      * state I/O counts match the architecture's layer cadence;
      * every layer initializer is < keep and no layer >= keep survives;
      * the residual stream feeding the final norm is the one produced after
        layer keep-1 (i.e. the rewiring really happened).
    """
    errors: List[str] = []
    g = model.graph
    have = {t.name for t in g.initializer} | {v.name for v in g.input}
    produced: Dict[str, str] = {}
    for n in g.node:
        for x in n.input:
            if x and x not in have:
                errors.append("dangling input %r of %s" % (x, n.name or n.op_type))
        for o in n.output:
            have.add(o)
            produced[o] = n.name or n.op_type
    for v in g.output:
        if v.name not in produced:
            errors.append("graph output %r is not produced" % v.name)

    inits = {t.name for t in g.initializer}
    bad_layer = [nm for nm in inits
                 if LAYER_INIT_RE.match(nm)
                 and int(LAYER_INIT_RE.match(nm).group(1)) >= keep]
    if bad_layer:
        errors.append("%d initializer(s) from layers >= %d survive, e.g. %s"
                      % (len(bad_layer), keep, sorted(bad_layer)[:3]))
    kept_layers = sorted({int(LAYER_INIT_RE.match(nm).group(1))
                          for nm in inits if LAYER_INIT_RE.match(nm)})
    if kept_layers != list(range(keep)):
        errors.append("kept layer indices are not 0..%d" % (keep - 1))

    def state_names(views, rx):
        out = []
        for v in views:
            m = rx.match(v.name)
            if m:
                out.append((m.group(1), int(m.group(2))))
        return out

    ins = state_names(g.input, STATE_IN_RE)
    # Non-state inputs (embeds, position_ids, attention_bias) must survive.
    other_ins = [v.name for v in g.input if not STATE_IN_RE.match(v.name)]
    if len(other_ins) != 3:
        errors.append("expected 3 non-state graph inputs, got %d: %s"
                      % (len(other_ins), other_ins))
    outs = state_names(g.output, STATE_OUT_RE)
    other_outs = [v.name for v in g.output if not STATE_OUT_RE.match(v.name)]
    if other_outs != ["logits"]:
        errors.append("expected graph output 'logits' only, got %s" % other_outs)
    if any(L >= keep for _, L in ins):
        errors.append("graph input from a dropped layer survives")
    if any(L >= keep for _, L in outs):
        errors.append("graph output from a dropped layer survives")

    # Per-class counts, derived (not hardcoded for 64 layers): an all-attention
    # layer index is one the source exposed as past_key_values_L, everything
    # else in 0..n_layers-1 is a linear-attention layer.
    kinds = {"kv": 0, "conv": 0, "recurrent": 0}
    for prefix, L in ins:
        kinds[{"past_key_values": "kv", "conv_state": "conv",
               "recurrent_state": "recurrent"}[prefix]] += 1
    out_kinds = {"kv": 0, "conv": 0, "recurrent": 0}
    for prefix, L in outs:
        out_kinds[{"present_key_values": "kv", "conv_state_out": "conv",
                   "recurrent_state_out": "recurrent"}[prefix]] += 1
    if kinds != out_kinds:
        errors.append("state input/output class counts differ: %s vs %s"
                      % (kinds, out_kinds))

    unused = check_unused_inputs(model)
    if unused:
        # position_ids / attention_bias only feed full-attention blocks (RoPE
        # and the causal bias); a prefix that happens to hold no such layer
        # legitimately leaves them unread. Any other name is a real leak.
        explains = {"position_ids", "attention_bias"}
        real = [x for x in unused if x not in explains]
        if real:
            errors.append("graph input(s) nobody reads: %s" % real)
        elif not attention_layers:
            print("  note: no full-attention layer in 0..%d, so position_ids "
                  "and attention_bias are unused by the graph itself; the "
                  "runtime still feeds them, which is harmless."
                  % (keep - 1))

    return {
        "errors": errors,
        "state_inputs": kinds, "state_outputs": out_kinds,
        "non_state_inputs": other_ins, "non_state_outputs": other_outs,
        "kept_layers": [kept_layers[0], kept_layers[-1]] if kept_layers else [],
        "n_initializers": len(g.initializer),
        "unused_inputs": unused,
    }


def check_dangling(model: onnx.ModelProto) -> List[str]:
    g = model.graph
    have = {t.name for t in g.initializer} | {v.name for v in g.input}
    missing = []
    for n in g.node:
        for x in n.input:
            if x == "":
                continue
            if x not in have:
                missing.append("%s <- %s" % (n.name or n.op_type, x))
        for o in n.output:
            have.add(o)
    return missing


def check_unused_inputs(model: onnx.ModelProto) -> List[str]:
    """Graph inputs nothing reads would be fed by the runtime for no reason."""
    g = model.graph
    read = {x for n in g.node for x in n.input}
    return [v.name for v in g.input if v.name not in read]


def fmt_gib(n: int) -> str:
    return "%.3f GiB (%.3f GB)" % (n / 2**30, n / 1e9)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--src", default="/home/wjj/q35_model/llm27.onnx")
    ap.add_argument("--out", default=None,
                    help="default: <src-dir>/llm27_L<keep>.onnx")
    ap.add_argument("--keep", type=int, required=True,
                    help="number of decoder layers to keep (counted from 0)")
    ap.add_argument("--json", default=None, help="write stats as JSON here")
    ap.add_argument("--relocate-external",
                    choices=("auto", "keep", "relative"), default="auto",
                    help="how to make the external weights findable from the "
                         "output directory; 'auto' keeps the relative name when "
                         "the file is next to the output, else writes a relative "
                         "path from the output dir to the source weights "
                         "(never copies the file)")
    ap.add_argument("--allow-keep-all", action="store_true")
    args = ap.parse_args(argv)

    src = os.path.abspath(args.src)
    out = args.out or os.path.join(
        os.path.dirname(src), "llm27_L%d.onnx" % args.keep)
    out = os.path.abspath(out)

    model = onnx.load(src, load_external_data=False)
    try:
        if args.allow_keep_all:
            stem = analyze(model, src)["n_layers"]
            if args.keep == stem:
                onnx.save(model, out)
                print("kept all %d layers -> %s" % (stem, out))
                return 0
        trimmed, stats = trim(model, args.keep, out,
                              args.relocate_external, src)
    except TrimError as exc:
        print("TRIM FAILED: %s" % exc, file=sys.stderr)
        return 2

    onnx.save(trimmed, out)
    stats["out_path"] = out
    stats["out_size"] = os.path.getsize(out)

    # Full check with the file on disk: the external-data reference is resolved
    # relative to the model path, which only works once it has been written.
    verdict, detail = checker_validate(
        out, stats["external_resolved"] or stats["src_path"],
        os.path.join(os.path.dirname(stats["external_resolved"]) if
                     stats["external_resolved"] else os.path.dirname(out),
                     "_trim_check_tmp.onnx"))
    stats["checker"] = {"verdict": verdict, "detail": detail}
    if verdict == "fail":
        os.unlink(out)
        print("TRIM FAILED: onnx.checker rejected the written model: %s" % detail,
              file=sys.stderr)
        return 2

    # Cheap independent checks that do not depend on the checker's own path
    # handling: every node input resolves, and the state I/O counts track the
    # layer count.
    self_check = self_check_trimmed(trimmed, args.keep, stats["attention_layers"])
    stats["self_check"] = self_check
    if self_check["errors"]:
        os.unlink(out)
        print("TRIM FAILED: self-check: %s" % self_check["errors"],
              file=sys.stderr)
        return 2

    # Every kept external pointer must land inside the untouched weights file,
    # with a length that matches dtype x dims -- a wrong offset here becomes
    # silently wrong weights at run time.
    if stats["external_size"]:
        ext_errors = validate_external_extents(
            trimmed.graph.initializer, stats["external_resolved"])
        stats["external_extent_errors"] = ext_errors
        if ext_errors:
            os.unlink(out)
            print("TRIM FAILED: external extents: %s" % ext_errors[:5],
                  file=sys.stderr)
            return 2

    print("=" * 72)
    print("trimmed %d/%d layers -> %s" % (stats["keep"], stats["n_layers"], out))
    print("  nodes          %d -> %d (unreachable pruned: %d)"
          % (stats["nodes_in"], stats["nodes_out"], stats["dropped_unreachable"]))
    print("  initializers   %d -> %d"
          % (stats["inits_in"], stats["inits_out"]))
    print("  init bytes     %s" % fmt_gib(stats["init_bytes"]))
    print("                 (source %s)" % fmt_gib(stats["init_bytes_source"]))
    print("  graph inputs   %d -> %d" % (stats["inputs_in"], stats["inputs_out"]))
    print("  graph outputs  %d -> %d"
          % (stats["outputs_in"], stats["outputs_out"]))
    print("  value_info     %d -> %d"
          % (stats["value_info_in"], stats["value_info_out"]))
    print("  prefix nodes byte-identical to source: %d"
          % stats["prefix_nodes_identical"])
    print("  tail op chain: %s" % ",".join(stats["tail_signature"]))
    r = stats["rewired"]
    print("  rewired node %d: norm Mul input %r -> %r"
          % (r["norm_mul"], r["from"], r["to"]))
    print("  external data  %s (%s, %d initializer pointer(s) rewritten)"
          % (", ".join(stats["external_locations"]) or "<inline only>",
             fmt_gib(stats["external_size"]) if stats["external_size"] else "missing",
             stats["external_rewritten"]))
    print("  file size      %d bytes" % stats["out_size"])
    chk = stats["checker"]
    print("  onnx.checker   %s (%s)" % (chk["verdict"], chk["detail"]))
    sc = stats["self_check"]
    print("  self-check     %s" % ("ok" if not sc["errors"] else sc["errors"]))
    print("                 state I/O    %s / %s"
          % (sc["state_inputs"], sc["state_outputs"]))
    print("                 other I/O    %s / %s"
          % (sc["non_state_inputs"], sc["non_state_outputs"]))
    print("                 layers kept  %s of %d"
          % (sc["kept_layers"], stats["n_layers"]))
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(stats, fh, indent=2)
        print("  stats -> %s" % args.json)
    print("NOTE: this graph is a bandwidth-measurement artifact only; its "
          "logits are not meaningful.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
