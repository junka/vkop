#!/usr/bin/env python3
"""Compare vkop GPU outputs (ref_out/) against ORT references (ref/).

Reports, per tensor: max abs diff, mean abs diff, cosine similarity, and the
count of elements differing beyond fp16 rounding (rel > 1e-2).

    /Users/doudou/qi21-env/bin/python compare_vkop_ort.py [--steps 5]
"""

import argparse
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REF = HERE / "ref"
OUT = HERE / "ref_out"


def load_ref(name):
    return np.fromfile(REF / name, dtype=np.float16)


def load_vk(name):
    return np.fromfile(OUT / name, dtype=np.float16).astype(np.float16)


def report(tag, a, b):
    a32, b32 = a.astype(np.float32), b.astype(np.float32)
    if a32.shape != b32.shape:
        print(f"{tag:32s} SHAPE MISMATCH vkop={b32.shape} ort={a32.shape}")
        return False
    diff = np.abs(a32 - b32)
    denom = np.maximum(np.abs(a32), 1e-3)
    rel = diff / denom
    cos = float(a32 @ b32 / (np.linalg.norm(a32) * np.linalg.norm(b32) + 1e-12))
    bad = int((rel > 1e-2).sum())
    nan = int(np.isnan(b32).sum())
    ok = cos > 0.999 and nan == 0
    print(f"{tag:32s} maxabs={diff.max():8.4f} meanabs={diff.mean():8.5f} "
          f"cos={cos:.6f} rel>1e-2:{bad}/{a32.size} nan:{nan} "
          f"{'OK' if ok else 'MISMATCH'}")
    return ok


def main():
    global OUT, REF
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--out", default=None, help="vkop ref_out dir (cwd of image_gen run)")
    ap.add_argument("--refdir", default=None, help="ORT reference dir, e.g. ref_full")
    args = ap.parse_args()
    if args.out:
        OUT = Path(args.out)
    if args.refdir:
        REF = HERE / args.refdir

    if not OUT.exists():
        raise SystemExit(f"{OUT} missing — run image_gen with --ref first")

    all_ok = True
    print("== prefill KV (vkop prefill vs ORT prefill, same inputs) ==")
    i = 0
    while (REF / f"ort_present_kv_{i}.raw").exists() and (OUT / f"vkop_present_kv_{i}.raw").exists():
        all_ok &= report(f"present_kv_{i}", load_ref(f"ort_present_kv_{i}.raw"),
                         load_vk(f"vkop_present_kv_{i}.raw"))
        i += 1

    print("\n== decode velocity per step ==")
    for s in range(args.steps):
        r = load_ref(f"ort_velocity_step{s}.raw")
        v = load_vk(f"vkop_velocity_step{s}.raw")
        all_ok &= report(f"velocity_step{s}", r, v)

    print("\nRESULT:", "ALIGNED" if all_ok else "DIVERGED")


if __name__ == "__main__":
    main()
