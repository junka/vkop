#!/usr/bin/env python
"""Compare vkop node_dump outputs against ORT reference intermediates."""
import argparse
import glob
import os
import re

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dumpdir", default="../../build/node_dump")
    ap.add_argument("--ref", default="vae_ref_intermediates.npz")
    args = ap.parse_args()

    ref = np.load(args.ref)
    names = [k for k in ref.files if not k.endswith("@shape")]
    dumps = sorted(glob.glob(os.path.join(args.dumpdir, "*_out0.raw")))
    # filename: NNNNNN_lvlL_nN_<node name with / -> _>_out0.raw
    index = {}
    for f in dumps:
        m = re.match(r"\d+_lvl\d+_n(\d+)_(.+)_out0\.raw$", os.path.basename(f))
        if m:
            index.setdefault(m.group(2), (int(m.group(1)), f))

    print(f"{'n':>5} {'tensor':68} {'elems':>9} {'dt':3} {'maxabs':>10} {'cos':>10}")
    rows = []
    for name in sorted(names):
        producer = name[: -len("_output_0")].replace("/", "_") if name.endswith(
            "_output_0"
        ) else None
        hit = None
        if producer:
            for key in (producer, "FusedElemwise_fused_" + producer):
                if key in index:
                    hit = index[key]
                    break
        if not hit:
            print(f"{'-':>5} {name:68} {'-':>9}   (no dump for {producer})")
            continue
        n, path = hit
        raw = np.fromfile(path, dtype=np.uint8)
        want = ref[name]
        if raw.size == want.size * 4:
            got = raw.view(np.float32)
            dt = "f32"
        elif raw.size == want.size * 2:
            got = raw.view(np.uint16).astype(np.float16).astype(np.float32)
            dt = "f16"
        else:
            print(f"{n:>5} {name:68} {want.size:>9}   size {raw.size} mismatch")
            continue
        d = np.abs(got - want)
        nan = int(np.isnan(got).sum())
        denom = float(np.linalg.norm(got) * np.linalg.norm(want)) or 1.0
        cos = float(np.nansum(got * want) / denom)
        mx = float(np.nanmax(d)) if d.size else 0.0
        rows.append((n, name, want.size, dt, mx, cos, nan))
    rows.sort()
    bad_seen = False
    for n, name, sz, dt, mx, cos, nan in rows:
        mark = "  <-- FIRST DIVERGENCE" if (not bad_seen and mx > 0.05) else ""
        if mx > 0.05:
            bad_seen = True
        print(f"{n:>5} {name:68} {sz:>9} {dt:3} {mx:10.5f} {cos:10.6f}"
              + (f" nan={nan}" if nan else "") + mark)


if __name__ == "__main__":
    main()
