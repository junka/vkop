#!/usr/bin/env python3
"""Compare vkop VKOP_DUMP_TENSORS='*' output against ORT intermediate stats,
in ONNX topological order, to find the first diverging tensor.

Usage: python3 compare_visual_intermediates.py <vkop_dump.txt> <ort_dump.txt>
"""
import os, sys, re
import onnx

_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
VISUAL_ONNX = os.path.join(_PKG_DIR, "visual.onnx")


def parse_dump(path):
    """name -> (ne, min, max, mean)."""
    out = {}
    pat = re.compile(r"^\[([^\]]+)\]\s*(?:\(f32\)\s*)?ne=(\S+)\s+min=(\S+)\s+max=(\S+)(?:\s+mean=(\S+))?")
    with open(path) as f:
        for line in f:
            m = pat.match(line.strip())
            if not m:
                continue
            name = m.group(1)
            # vkop prints shape=[...] before stats for outputs; the regex still
            # captures ne. Strip any trailing non-numeric in mean.
            try:
                ne = int(m.group(2))
                mn = float(m.group(3))
                mx = float(m.group(4))
                mean = float(m.group(5)) if m.group(5) else 0.0
            except ValueError:
                continue
            out[name] = (ne, mn, mx, mean)
    return out


def main():
    vkop = parse_dump(sys.argv[1])
    ort = parse_dump(sys.argv[2])
    m = onnx.load(VISUAL_ONNX)
    # Topological order of node outputs.
    topo = []
    for n in m.graph.node:
        for o in n.output:
            if o:
                topo.append(o)
    print(f"vkop tensors: {len(vkop)}, ort tensors: {len(ort)}, topo nodes: {len(topo)}")
    first_diff = None
    matched = 0
    for name in topo:
        if name not in ort or name not in vkop:
            continue
        matched += 1
        vne, vmn, vmx, vmean = vkop[name]
        one, omn, omx, omean = ort[name]
        if vne != one:
            print(f"[NE DIFF] {name}: vkop ne={vne} ort ne={one}")
            if first_diff is None:
                first_diff = name
            continue
        # compare stats with tolerance
        dmin = abs(vmn - omn)
        dmax = abs(vmx - omx)
        dmean = abs(vmean - omean)
        rel_min = dmin / (abs(omn) + 1e-6)
        rel_max = dmax / (abs(omx) + 1e-6)
        # flag if any stat differs by >5% or >0.5 absolute
        if dmin > 0.5 or dmax > 0.5 or dmean > 0.1 or rel_min > 0.05 or rel_max > 0.05:
            print(f"[DIFF] {name}: vkop(min={vmn:.4g} max={vmx:.4g} mean={vmean:.4g}) "
                  f"ort(min={omn:.4g} max={omx:.4g} mean={omean:.4g})")
            if first_diff is None:
                first_diff = name
    print(f"\nmatched {matched} tensors")
    print(f"FIRST DIVERGING: {first_diff}")


if __name__ == "__main__":
    main()
