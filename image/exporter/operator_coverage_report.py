#!/usr/bin/env python3
"""Report operator coverage gaps for Qwen-Image-2.1 DiT graphs.

Compares ONNX graph operators against vkop's supported runtime ops to identify
which operators need implementation before the image pipeline can run.

    /Users/doudou/qi21-env/bin/python operator_coverage_report.py [prefill|decode|both]
"""

import sys
from collections import Counter
from pathlib import Path

import onnx

# Ops the vkop runtime supports today (from llm/exporter/vkop_report.py)
SUPPORTED = {
    "Add", "Atan", "AveragePool", "BatchNormalization", "BatchNorm", "Col2Im",
    "Concat", "Conv", "Conv2d", "Div", "EmbeddingForward", "Erf", "Floor",
    "Gemm", "GlobalAveragePool", "GridSample", "LayerNormalization", "LayerNorm",
    "MatMul", "MaxPool", "MaxPool2d", "Mul", "Pow", "PRelu", "Reduce", "Relu",
    "Reshape", "Resize", "Sigmoid", "Slice", "Softmax", "Softplus", "Split",
    "Sub", "TopK", "Transpose", "Nms", "Gather", "Range", "Expand", "Sqrt",
    "Sin", "Cos", "Neg", "Where", "Tanh", "Equal", "NonZero", "ScatterElements",
}


def analyze_graph(mode):
    """Analyze operator coverage for a single graph."""
    path = Path(__file__).resolve().parent / f"dit_{mode}.onnx"
    if not path.exists():
        print(f"[{mode}] SKIP: {path.name} not found")
        return None

    model = onnx.load(str(path), load_external_data=False)
    onnx.load_external_data_for_model(model, str(path.parent))

    counts = Counter(node.op_type for node in model.graph.node)
    missing = Counter()
    supported_count = 0

    print(f"\n{'='*70}")
    print(f"[{mode.upper()}] Graph Analysis ({len(model.graph.node)} nodes)")
    print(f"{'='*70}")
    print(f"{'count':<7}{'op':<24}{'status':<12}{'note'}")
    print("-" * 70)

    for op, c in counts.most_common():
        status = "SUPPORTED" if op in SUPPORTED else "MISSING"
        note = ""
        if status == "MISSING":
            missing[op] = c
            # Provide hints for common missing ops
            if op == "Constant":
                note = "(should be folded by ConstantFolder)"
            elif op == "Cast":
                note = "(dtype conversion — may need fp16↔fp32 support)"
            elif op == "Shape":
                note = "(shape introspection — data-dependent)"
            elif op == "Unsqueeze":
                note = "(dimension insertion — trivial reshape)"
            elif op == "Mod":
                note = "(modulo — arithmetic op)"
            elif op == "Min":
                note = "(element-wise min — broadcast math)"
            elif op == "Max":
                note = "(element-wise max — broadcast math)"
            elif op == "ReduceMean":
                note = "(reduction — similar to existing Reduce)"
        else:
            supported_count += c
        print(f"{c:<7}{op:<24}{status:<12}{note}")

    total_nodes = len(model.graph.node)
    missing_nodes = sum(missing.values())
    covered_pct = (total_nodes - missing_nodes) / total_nodes * 100

    print(f"\nSummary:")
    print(f"  Total nodes:          {total_nodes}")
    print(f"  Supported nodes:      {supported_count} ({supported_count/total_nodes*100:.1f}%)")
    print(f"  Missing nodes:        {missing_nodes} ({missing_nodes/total_nodes*100:.1f}%)")
    print(f"  Distinct missing ops: {len(missing)} types")

    if missing:
        print(f"\nMissing operators (need implementation):")
        for op, c in missing.most_common():
            print(f"  - {op:<24} {c:>4} nodes")

    return {
        "mode": mode,
        "total": total_nodes,
        "supported": supported_count,
        "missing": missing_nodes,
        "missing_ops": dict(missing),
    }


def main():
    modes = sys.argv[1:] if len(sys.argv) > 1 else ["prefill", "decode"]
    if "both" in modes:
        modes = ["prefill", "decode"]

    results = []
    for mode in modes:
        result = analyze_graph(mode)
        if result:
            results.append(result)

    if len(results) == 2:
        prefill, decode = results
        print(f"\n{'='*70}")
        print("COMBINED SUMMARY (prefill + decode)")
        print(f"{'='*70}")
        total_all = prefill["total"] + decode["total"]
        missing_all = prefill["missing"] + decode["missing"]
        all_missing_ops = set(prefill["missing_ops"]) | set(decode["missing_ops"])

        print(f"Total nodes:          {total_all}")
        print(f"Missing nodes:        {missing_all} ({missing_all/total_all*100:.1f}%)")
        print(f"Distinct missing ops: {len(all_missing_ops)} types")

        print(f"\nUnion of missing operators:")
        union_missing = Counter()
        for m in [prefill["missing_ops"], decode["missing_ops"]]:
            union_missing.update(m)
        for op, c in union_missing.most_common():
            print(f"  - {op:<24} {c:>4} nodes total")


if __name__ == "__main__":
    main()
