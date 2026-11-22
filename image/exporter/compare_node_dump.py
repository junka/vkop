#!/usr/bin/env python3
"""Diff vkop VKOP_NODE_DUMP files against ORT intermediates (report-only).

The reference .npz comes from gen_ort_intermediates.py. Dump files are
NNNNNN_lvlL_nN_<node, '/' -> '_'>_outK.raw (fp16 or fp32, int64 for index
tensors); they are matched to ORT tensors **by node name** (the dump's index is
not the runtime `n`, so matching by index compares the wrong tensors), then
listed in runtime order with the first divergence marked.

    python compare_node_dump.py --dump ../../build/node_dump --ref vae_ref_intermediates.npz
"""
import argparse
import re
from pathlib import Path

import numpy as np

DUMP = re.compile(r"\d+_lvl\d+_n(\d+)_(.+)_out(\d+)\.raw$")


def load_dump(path, n_elems):
    raw = path.read_bytes()
    if len(raw) == 8 * n_elems:
        return np.frombuffer(raw, dtype=np.int64).astype(np.float32), "i64"
    if len(raw) == 4 * n_elems:
        return np.frombuffer(raw, dtype=np.float32), "f32"
    if len(raw) == 2 * n_elems:
        return np.frombuffer(raw, dtype=np.float16).astype(np.float32), "f16"
    return None, f"size {len(raw)}"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dump", default="../../build/node_dump")
    ap.add_argument("--ref", required=True, help=".npz from gen_ort_intermediates.py")
    ap.add_argument("--rtol", type=float, default=2e-3,
                    help="BAD when maxabs > rtol * max|ref| (default 2e-3)")
    ap.add_argument("--all", action="store_true", help="also list tensors with no dump")
    args = ap.parse_args()

    index = {}
    for f in Path(args.dump).glob("*.raw"):
        m = DUMP.match(f.name)
        if m:
            index.setdefault((m.group(2), int(m.group(3))), (int(m.group(1)), f))

    ref = np.load(args.ref)
    rows = []
    for name in ref.files:
        if name.endswith("@shape"):
            continue
        m = re.match(r"(.+)_output_(\d+)$", name)
        node, k = (m.group(1), int(m.group(2))) if m else (name, 0)
        key = node.replace("/", "_")
        hit = next((index[(c, k)] for c in (key, "FusedElemwise_fused_" + key)
                    if (c, k) in index), None)
        if hit is None:
            if args.all:
                print(f"{'-':>6} {name:68}   (no dump)")
            continue
        n, path = hit
        want = ref[name]
        got, dt = load_dump(path, want.size)
        if got is None:
            rows.append((n, name, want.size, dt, None, None, 0))
            continue
        nan = int(np.isnan(got).sum())
        mx = float(np.nanmax(np.abs(got - want))) if want.size else 0.0
        denom = float(np.linalg.norm(got) * np.linalg.norm(want)) or 1.0
        cos = float(np.nansum(got * want) / denom)
        scale = max(float(np.max(np.abs(want))) if want.size else 0.0, 1e-6)
        rows.append((n, name, want.size, dt, mx, cos, nan, mx > args.rtol * scale))

    print(f"{'n':>6} {'tensor':68} {'elems':>9} {'dt':4} {'maxabs':>10} {'cos':>10}")
    first = True
    for n, name, sz, dt, mx, cos, nan, *bad in sorted(rows):
        if mx is None:
            print(f"{n:>6} {name:68} {sz:>9}   {dt} mismatch")
            continue
        mark = ""
        if bad[0]:
            mark = "  <-- FIRST DIVERGENCE" if first else "  BAD"
            first = False
        print(f"{n:>6} {name:68} {sz:>9} {dt:4} {mx:10.5g} {cos:10.6f}"
              + (f" nan={nan}" if nan else "") + mark)
    print(f"[{len(rows)} matched, {'no divergence' if first else 'diverged'}]")


if __name__ == "__main__":
    main()
