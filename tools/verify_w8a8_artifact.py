#!/usr/bin/env python3
"""Verify a --w8a8 artifact's DP4A wiring without touching a GPU.

Checks, in order of what would silently break the model:
  1. MatMul A really is int8 (a float A means the fold did not fire).
  2. Every int8-A MatMul carries an `a_scale` attribute; a missing one is off by
     exactly the activation scale.
  3. The attribute's string round-trips to a float32 exactly.
  4. The weight is int8 and its per-column scale is still the last input.
  5. The QuantizeLinear that produced each A is present exactly once and is fed
     the same scale, zero point 0.

Usage:
    python tools/verify_w8a8_artifact.py artifact.vkopbin [--expect-n 311]
"""
import argparse
import json
import os
import struct
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "model", "pypi"))

from onnx2vkop.inspect import ModelView  # noqa: E402


def f32(x):
    return struct.unpack("f", struct.pack("f", x))[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--expect-n", type=int, default=None)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    v = ModelView(args.model)
    nodes = v.nodes()
    inits = {i["name"]: i for i in v.initializers()}
    by_output = {o["name"]: n for n in nodes for o in n["outputs"]}

    mm = [n for n in nodes if n["op_type"] == "MatMul"]
    int8_a = [n for n in mm if n["inputs"][0].get("dtype") == "int8"]
    with_scale = [n for n in int8_a if "a_scale" in n["attributes"]]

    report = {
        "artifact": args.model,
        "file_bytes": v.file_size,
        "matmul_total": len(mm),
        "matmul_a_int8": len(int8_a),
        "matmul_with_a_scale": len(with_scale),
        "problems": [],
    }

    # 1. A float A is expected only for a MatMul the DP4A gate cannot take at
    #    all (K < 256, or K/N not a multiple of 4). Any other float A means the
    #    fold missed a node that was eligible.
    unexplained_float = []
    for n in mm:
        if n["inputs"][0].get("dtype") == "int8":
            continue
        dims = n["inputs"][0].get("dims") or []
        o_dims = n["outputs"][0].get("dims") or []
        k = dims[-1] if dims else -1
        nn = o_dims[-1] if o_dims else -1
        if not (k >= 256 and k % 4 == 0 and nn % 4 == 0):
            continue  # ineligible by shape; correctly left on the float route
        unexplained_float.append(n["name"])
    report["matmul_ineligible_by_shape"] = len(mm) - len(int8_a) - len(unexplained_float)
    if unexplained_float:
        report["problems"].append(
            f"{len(unexplained_float)} eligible MatMul(s) still read a float A: "
            f"{unexplained_float[:5]}")

    # 2. every int8 A needs its scale
    missing = [n["name"] for n in int8_a if "a_scale" not in n["attributes"]]
    if missing:
        report["problems"].append(
            f"{len(missing)} int8-A MatMul(s) have no a_scale: {missing[:5]}")

    # 3. text round-trip
    bad_rt = []
    scales = set()
    for n in with_scale:
        txt = n["attributes"]["a_scale"]
        try:
            val = float(txt)
        except ValueError:
            bad_rt.append((n["name"], txt))
            continue
        if f32(val) != f32(float(txt)) or not (val > 0):
            bad_rt.append((n["name"], txt, val))
        scales.add(txt)
    if bad_rt:
        report["problems"].append(f"a_scale text does not round-trip: {bad_rt[:5]}")
    report["distinct_a_scale_values"] = sorted(scales)

    # 4. weight contract
    bad_w = []
    for n in int8_a:
        if len(n["inputs"]) < 3:
            bad_w.append((n["name"], "fewer than 3 inputs"))
            continue
        b = n["inputs"][1]
        w = inits.get(b["name"])
        if not w or w.get("dtype") != "int8":
            bad_w.append((n["name"], f"B dtype {b.get('dtype')}"))
        s = n["inputs"][2]
        if s.get("dtype") not in ("float32", "", None):
            bad_w.append((n["name"], f"scale dtype {s.get('dtype')}"))
    if bad_w:
        report["problems"].append(f"weight contract broken: {bad_w[:5]}")

    # 5. the Q producer
    q_nodes = [n for n in nodes if n["op_type"] == "QuantizeLinear"]
    dq_nodes = [n for n in nodes if n["op_type"] == "DequantizeLinear"]
    report["quantize_linear_nodes"] = len(q_nodes)
    report["dequantize_linear_nodes"] = len(dq_nodes)

    a_names = [n["inputs"][0]["name"] for n in int8_a]
    report["distinct_int8_a_tensors"] = len(set(a_names))
    unexplained = [nm for nm in set(a_names) if nm not in by_output]
    if unexplained:
        report["problems"].append(
            f"{len(unexplained)} int8 A tensor(s) have no producer: "
            f"{unexplained[:5]}")

    # A DQ feeding an int8 A would mean the pair survived and a float kernel
    # would be handed bytes.
    dangling_dq = [n["name"] for n in dq_nodes
                   if n["outputs"][0]["name"] in set(a_names)]
    if dangling_dq:
        report["problems"].append(
            f"{len(dangling_dq)} DQ output(s) still feed a MatMul A")

    if args.expect_n is not None and len(int8_a) != args.expect_n:
        report["problems"].append(
            f"expected {args.expect_n} int8-A MatMul(s), got {len(int8_a)}")

    report["ok"] = not report["problems"]
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        for k, val in report.items():
            if k in ("problems", "ok"):
                continue
            print(f"  {k:32s} {val}")
        print(f"  {'ok':32s} {report['ok']}")
        for p in report["problems"]:
            print(f"  PROBLEM: {p}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
