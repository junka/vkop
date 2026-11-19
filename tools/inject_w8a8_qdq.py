#!/usr/bin/env python3
"""Inject a per-tensor int8 activation Q/DQ pair in front of every MatMul A that
the vkop `--w8a8` DP4A fold can consume, plus the ValueInfoProto the fold needs
to see the shape at all.

WHY THIS IS A SEPARATE PRE-STEP
===============================
`-q int8` is WEIGHT-ONLY: `Quantizer.quantize_to_int8_weight_only` rewrites an
initializer's dtype and appends its per-column scale, and never creates an
activation QuantizeLinear. A stock export therefore has nothing for
`W8A8ActivationFold` to find (measured on llm27_L40.onnx: 0 Q, 0 DQ), and
`--w8a8` prints "no MatMul satisfied the DP4A gates" and exits 0 with the graph
unchanged.

Two more preconditions, both absent from a stock export:

  1. SHAPE. The fold reads A's dims from the node's input dict, which the
     converter fills from graph.input / value_info / output. The TorchScript
     export records 0 of 332 MatMul A tensors in value_info, so `a_shape` is
     `[]` and every candidate dies at `len(a_shape) < 2`.
     onnx.shape_inference recovers all 332 (this script runs it).

  2. STATIC M. `W8A8ActivationFold.MAX_M` requires `batch*m <= 8` decided at
     CONVERSION time (a proxy for the runtime gate, which re-checks the live
     shape each round). Every MatMul A in the LLM export is `[-1, -1, K]`, so m
     is symbolic and the fold refuses. `--specialize-decode` fixes batch and
     seq to 1, which is the decode shape and the only shape this route is for.

     THIS MAKES THE ARTIFACT DECODE-ONLY. The graph inputs are declared [1,1,H]
     and the runtime allocates each input buffer from that declaration, so a
     prompt of q>1 tokens is not runnable on it. More importantly, ops/Matmul.hpp
     throws outright for `a_int8 && !dp4a`, and prefill's m is far above the
     DP4A gate's batch*m<=8, so a prefill on this artifact is a hard error, not
     a slow path. Keep a separate fp16-A artifact for prefill.

THE SCALE IS STATIC AND PER-TENSOR -- A REAL ACCURACY CONSTRAINT
===============================================================
DP4A applies one scalar to every element of every decode step and has no
zero-point correction (`int_encode` clamps to [-127,127] irreversibly). A scale
that fits one activation distribution saturates once the live distribution
drifts past it, and the failure is a cliff, not a gentle degradation: with the
scale calibrated at absmax=3.0, a 2x drift clips 10.7% of elements (relL2 0.20)
and a 4x drift clips 44% (relL2 0.45). This repo has NO per-tensor activation
calibration data today (`--awq-stats` is per-channel and only ever feeds AWQ's
weight scaling), so `--scale-mode heuristic` is the only mode usable without a
new calibration pass, and it is a guess. `--scale-mode absmax --stats` is the
honest one when a dump exists.

Usage:
    python tools/inject_w8a8_qdq.py -i in.onnx -o out.onnx --specialize-decode
    python tools/inject_w8a8_qdq.py -i in.onnx -o out.onnx --dry-run
"""
import argparse
import math
import sys

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

MIN_K = 256
MAX_M = 8


def _declared(vi_map, name):
    """(dims, dtype) from a ValueInfoProto, or (None, None)."""
    vi = vi_map.get(name)
    if vi is None:
        return None, None
    tt = vi.type.tensor_type
    if not tt.HasField("shape"):
        return None, (tt.elem_type or None)
    dims = []
    for d in tt.shape.dim:
        dims.append(d.dim_value if d.HasField("dim_value") and not d.dim_param else -1)
    return dims, (tt.elem_type or None)


def _vi(name, dtype, dims):
    """A ValueInfoProto that spells a dynamic dim as `-1` for the converter."""
    tt = onnx.TypeProto()
    tt.tensor_type.elem_type = dtype
    for d in dims:
        dim = tt.tensor_type.shape.dim.add()
        if d is None or d < 0:
            dim.dim_param = "?"
        else:
            dim.dim_value = d
    vi = onnx.ValueInfoProto()
    vi.name = name
    vi.type.CopyFrom(tt)
    return vi


def specialize_decode(model):
    """Fix batch and sequence to 1 so the fold can see a static m.

    Only the two dims that make the graph decode-shaped are touched; kv_len and
    the past/present caches stay dynamic, since they grow every step.
    """
    n = 0
    for vi in list(model.graph.input) + list(model.graph.output):
        tt = vi.type.tensor_type
        if not tt.HasField("shape"):
            continue
        dims = tt.shape.dim
        if vi.name in ("inputs_embeds", "logits"):
            # [batch, seq, hidden]
            if len(dims) >= 2:
                for idx in (0, 1):
                    if dims[idx].dim_param or dims[idx].dim_value != 1:
                        dims[idx].Clear()
                        dims[idx].dim_value = 1
                        n += 1
        elif vi.name == "position_ids":
            # [3, batch, seq]
            if len(dims) >= 3:
                for idx in (1, 2):
                    if dims[idx].dim_param or dims[idx].dim_value != 1:
                        dims[idx].Clear()
                        dims[idx].dim_value = 1
                        n += 1
        elif vi.name == "attention_bias":
            # [batch, 1, q_len, kv_len] -- q_len is seq, kv_len stays dynamic
            if len(dims) >= 4:
                for idx in (0, 2):
                    if dims[idx].dim_param or dims[idx].dim_value != 1:
                        dims[idx].Clear()
                        dims[idx].dim_value = 1
                        n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True)
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument("--specialize-decode", action="store_true",
                    help="DEPRECATED and harmful on a graph with non-MatMul "
                         "consumers of the sequence dim: writing a concrete "
                         "batch/seq into the graph inputs makes ONNX shape "
                         "inference propagate it into unrelated intermediates "
                         "(measured: a Conv1d operand became rank-2 [10240,1] "
                         "and every prefill then threw 'Conv1d: x must be "
                         "[C, S]'). The A-shape repair below already writes "
                         "[1,..,K] straight onto the MatMul inputs, which is "
                         "all the fold reads, so this is no longer needed.")
    ap.add_argument("--scale", type=float, default=None)
    ap.add_argument("--scale-mode", choices=["heuristic", "absmax", "constant"],
                    default="heuristic")
    ap.add_argument("--stats", default=None)
    ap.add_argument("--heuristic-absmax", type=float, default=8.0)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    model = onnx.load(args.input, load_external_data=False)

    if args.specialize_decode:
        n = specialize_decode(model)
        print(f"[inject] specialized {n} dim(s) to 1 (decode shape)")

    # data_prop=False: 532 external initializers totalling 33 GB; data
    # propagation would try to read them.
    print("[inject] running shape inference ...", flush=True)
    model = onnx.shape_inference.infer_shapes(
        model, strict_mode=False, data_prop=False)
    # Shape inference writes INTERMEDIATE tensors to value_info; a tensor that is
    # a graph input or a graph output is declared there instead. The converter's
    # own lookup checks all three, so this map has to as well.
    vi_map = {v.name: v for v in
              list(model.graph.value_info) + list(model.graph.input)
              + list(model.graph.output)}
    print(f"[inject] visible tensors: {len(vi_map)}")

    # ---- 1b. the shape each MatMul's A will have at DECODE ----------------
    # ONNX inference drops the concrete leading dims as soon as the chain passes
    # a broadcast Mul (the layernorm weight is an initializer, not an input, so
    # its dims come back unset and the multiply's output becomes unk__22), so
    # every MatMul A in this export reads back as [-1,-1,K]. The fold needs a
    # static m, and decode's is 1 by construction, so the DECODE shape is
    # [1,..,K] -- K being the weight's inner dim, which inference does know.
    #
    # This map is deliberately NOT written into the shared value_info. An
    # intermediate like `h` feeds LinearAttention, Attention and RMSNorm as well
    # as MatMuls, and stamping [1,1,5120] on it makes the converter record that
    # for every one of those consumers: measured, prefill then died in
    # LinearAttention (q/k came back [1,1,16,128]) and in Attention ("additive
    # bias ... got 256 elements, expected 1"). Only the three NEW tensors this
    # script creates are declared with it, and they have exactly one consumer
    # each, so nothing else can observe it.
    decode_shape = {}
    for node in model.graph.node:
        if node.op_type != "MatMul" or len(node.input) < 2:
            continue
        a_dims, _ = _declared(vi_map, node.input[0])
        b_dims, _ = _declared(vi_map, node.input[1])
        if not a_dims or not b_dims or b_dims[0] <= 0:
            continue
        want = [1] * (len(a_dims) - 1) + [b_dims[0]]
        if a_dims != want:
            decode_shape[node.input[0]] = want
    if decode_shape:
        print(f"[inject] {len(decode_shape)} MatMul A tensor(s) declared at "
              f"their decode shape [1,..,K]")

    chosen, skipped = [], []
    for node in model.graph.node:
        if node.op_type != "MatMul":
            continue
        # 2 inputs in the raw export; the converter appends the weight's
        # per-column scale to make the 3-input form the fold wants.
        if len(node.input) < 2:
            continue
        a_name = node.input[0]
        dims, dt = _declared(vi_map, a_name)
        # Use the decode shape computed above when inference could not supply a
        # static m; that is the value the fold will read once this script has
        # declared it on the node's input.
        dims = decode_shape.get(a_name, dims)
        o_dims, o_dt = _declared(vi_map, node.output[0])
        if not dims or not o_dims:
            skipped.append((node.name, "no shape for A or output"))
            continue
        k, nn = dims[-1], o_dims[-1]
        m = 1
        for d in dims[:-1]:
            if d <= 0:
                m = -1
                break
            m *= d
        if m <= 0:
            skipped.append((node.name, f"dynamic m (A dims {dims})"))
            continue
        if m > MAX_M:
            skipped.append((node.name, f"m={m} > {MAX_M}"))
            continue
        if k < MIN_K or k % 4 or nn % 4:
            skipped.append((node.name, f"K={k} N={nn} fails K>=256/K%4/N%4"))
            continue
        if o_dt != TensorProto.FLOAT16:
            skipped.append((node.name, f"output dtype {o_dt} != fp16"))
            continue
        chosen.append((node, a_name, dims))

    print(f"[inject] fold candidates: {len(chosen)}  skipped: {len(skipped)}")
    for name, why in skipped[:8]:
        print(f"[inject]   skip {name or '<unnamed>'}: {why}")
    if len(skipped) > 8:
        print(f"[inject]   ... and {len(skipped) - 8} more")

    if not chosen:
        print("[inject] nothing to do", file=sys.stderr)
        return 1

    if args.scale_mode == "constant":
        if args.scale is None:
            print("[inject] --scale-mode constant needs --scale", file=sys.stderr)
            return 2
        scale = float(args.scale)
    elif args.scale_mode == "absmax":
        if not args.stats:
            print("[inject] --scale-mode absmax needs --stats", file=sys.stderr)
            return 2
        data = np.load(args.stats)
        scale = max(float(np.max(np.abs(np.asarray(data[k], np.float32))))
                    for k in data.files) / 127.0
    else:
        scale = float(args.heuristic_absmax) / 127.0
    if not (scale > 0.0 and math.isfinite(scale)):
        print(f"[inject] bad scale {scale}", file=sys.stderr)
        return 2
    print(f"[inject] per-tensor activation scale = {scale:.9g} "
          f"(covers |x| <= {127 * scale:.4g})")

    if args.dry_run:
        return 0

    scale_name = "w8a8_act_scale"
    zp_name = "w8a8_act_zero_point"
    model.graph.initializer.append(
        numpy_helper.from_array(np.array(scale, np.float32), scale_name))
    model.graph.initializer.append(
        numpy_helper.from_array(np.array(0, np.int8), zp_name))

    # Every MatMul gets its OWN DequantizeLinear, while the QuantizeLinear and
    # the Cast are shared by all consumer of the same A.
    #
    # This shape is forced by the fold, not chosen. The fold walks A's producer
    # chain upward and absorbs a node only while it has exactly ONE dependent
    # (`len(upstream.dependents) == 1`) -- that condition is what guarantees
    # re-pointing the MatMul at the Q cannot strip a float tensor another
    # consumer still reads. So for a Cast shared by k MatMuls:
    #
    #   * the Cast has k dependents, so it is NOT absorbed and the walk stops
    #     there. That is fine and REQUIRED: the Cast keeps feeding the other
    #     MatMuls' float A.
    #   * each MatMul's A must therefore be a DQ of its own (one dependent) so
    #     the walk absorbs the DQ and lands on the int8 Q.
    #
    # Measured on the L40 export: 161 distinct A tensors, of which only the 82
    # single-consumer ones folded when the DQ was shared (81 folded / 311
    # candidates). With a per-MatMul DQ all of them are reachable.
    q_out_for = {}
    q_triplets = {}     # a_name -> (q_node, q_out, dq_node, cast_node, cast_out)
    per_matmul = []     # (node_name, dq_out)

    def q_for(a_name, dims):
        if a_name in q_out_for:
            return q_out_for[a_name]
        idx = len(q_out_for)
        q = f"w8a8_q_{idx}_i8"
        dq = f"w8a8_q_{idx}_shared_f"
        cast = f"w8a8_q_{idx}_f16"
        q_triplets[a_name] = (
            helper.make_node("QuantizeLinear", [a_name, scale_name, zp_name],
                             [q], name=f"w8a8_q_{idx}"),
            helper.make_node("DequantizeLinear", [q, scale_name, zp_name],
                             [dq], name=f"w8a8_q_{idx}_shared_dq"),
            helper.make_node("Cast", [dq], [cast], to=TensorProto.FLOAT16,
                             name=f"w8a8_q_{idx}_cast"),
        )
        model.graph.value_info.append(_vi(q, TensorProto.INT8, dims))
        model.graph.value_info.append(_vi(dq, TensorProto.FLOAT, dims))
        model.graph.value_info.append(_vi(cast, TensorProto.FLOAT16, dims))
        q_out_for[a_name] = (q, cast)
        return q_out_for[a_name]

    # Rebuild the node list so each Q/DQ/Cast group sits directly before the first
    # MatMul that consumes it and each per-MatMul DQ directly before its own
    # MatMul; the checker verifies topological order and the export interleaves
    # MatMuls with everything else.
    emitted_q = set()
    rebuilt = []
    for node in model.graph.node:
        if node.op_type == "MatMul" and len(node.input) >= 2:
            for cand, a_name, dims in chosen:
                if cand is node:
                    q_out, _cast = q_for(a_name, dims)
                    if a_name not in emitted_q:
                        rebuilt.extend(q_triplets[a_name])
                        emitted_q.add(a_name)
                    idx = len(per_matmul)
                    dq = f"w8a8_dq_{idx}_f"
                    rebuilt.append(helper.make_node(
                        "DequantizeLinear", [q_out, scale_name, zp_name], [dq],
                        name=f"w8a8_dq_{idx}"))
                    model.graph.value_info.append(
                        _vi(dq, TensorProto.FLOAT, dims))
                    node.input[0] = dq
                    per_matmul.append((node.name, dq))
                    break
        rebuilt.append(node)
    model.graph.ClearField("node")
    model.graph.node.extend(rebuilt)

    print(f"[inject] inserted {len(q_out_for)} QuantizeLinear node(s) and "
          f"{len(per_matmul)} DequantizeLinear/Cast pair(s) "
          f"for {len(chosen)} MatMul(s)")

    # Every initializer's bytes live in one 51 GB file referenced by a relative
    # path. onnx.checker resolves that path against the OUTPUT file's directory
    # and rejects a reference that climbs out of it, so the check has to run
    # against the INPUT's location -- which is where the reference was written
    # for -- with the model temporarily saved there under its own name.
    # check_model accepts a path, and a path is what makes the resolution work;
    # skip the check rather than write 33 GB of tensors into memory.
    print("[inject] skipping full checker (external data lives outside the "
          "output directory by design)")

    onnx.save(model, args.output)
    print(f"[inject] wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
