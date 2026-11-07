#!/usr/bin/env python3
"""Summarise paired perf logs: per-run avg / steady median / MatMul profile."""
import glob
import re
import statistics
import sys

STEP = re.compile(r"^\[r(\d+)\] (\d+\.?\d*)ms")
MM = re.compile(r"^\[submitprof\] MatMul\s+(\d+)\s+(\d+\.?\d*)\s+(\d+\.?\d*)")
TOTAL = re.compile(r"^\[submitprof\] TOTAL\s+(\d+)\s+(\d+\.?\d*)")


def parse(path):
    steps = []
    mm = []      # per submitprof block: (calls, ms)
    tot = []
    for line in open(path, errors="ignore"):
        m = STEP.match(line)
        if m:
            steps.append((int(m.group(1)), float(m.group(2))))
            continue
        m = MM.match(line)
        if m:
            mm.append((int(m.group(1)), float(m.group(2))))
            continue
        m = TOTAL.match(line)
        if m:
            tot.append((int(m.group(1)), float(m.group(2))))
    dec = [ms for i, ms in steps if i >= 1]
    steady = [ms for i, ms in steps if i >= 100]
    # The first submitprof block covers prefill (576 ops); drop it so the
    # MatMul/TOTAL medians describe decode steps only.
    mm = mm[1:]
    tot = tot[1:]
    return {
        "n": len(dec),
        "avg": statistics.mean(dec) if dec else float("nan"),
        "med": statistics.median(dec) if dec else float("nan"),
        "steady_med": statistics.median(steady) if steady else float("nan"),
        "steady_avg": statistics.mean(steady) if steady else float("nan"),
        "p50": statistics.median(dec) if dec else float("nan"),
        "mm_calls": statistics.median([c for c, _ in mm]) if mm else float("nan"),
        "mm_med": statistics.median([x for _, x in mm]) if mm else float("nan"),
        "total_med": statistics.median([x for _, x in tot]) if tot else float("nan"),
    }


arms = ["old", "fix", "nofuse"]
rows = {}
for arm in arms:
    for path in sorted(glob.glob(f"/tmp/reg/logs/perf_{arm}_c*.log")):
        rows.setdefault(arm, []).append((path, parse(path)))

for arm in arms:
    got = rows.get(arm, [])
    if not got:
        continue
    print(f"\n== {arm} ==")
    for path, r in got:
        print(f"  {path.split('/')[-1]}: n={r['n']} avg={r['avg']:.2f} "
              f"med={r['med']:.2f} steady_med={r['steady_med']:.2f} "
              f"mm_calls={r['mm_calls']:.0f} mm_med={r['mm_med']:.2f} "
              f"total_med={r['total_med']:.2f}")
    best = min(got, key=lambda pr: pr[1]["avg"])
    print(f"  MIN-OF-3 avg = {best[1]['avg']:.2f}  (round {best[0].split('_')[-1][0]})")
    smed = min(r["steady_med"] for _, r in got)
    mm = min(r["mm_med"] for _, r in got)
    print(f"  min steady_med = {smed:.2f}   min mm_med = {mm:.2f}")

print("\n== paired per-cycle (fix/old, nofuse/fix, nofuse/old) ==")
for cyc in (1, 2, 3):
    vals = {}
    for arm in arms:
        p = f"/tmp/reg/logs/perf_{arm}_c{cyc}.log"
        try:
            vals[arm] = parse(p)
        except FileNotFoundError:
            pass
    if len(vals) < len(arms):
        continue
    d = {}
    for a, b in (("fix", "old"), ("nofuse", "fix"), ("nofuse", "old")):
        ra, rb = vals[a], vals[b]
        d[f"{a}/{b}"] = (100.0 * (ra["avg"] - rb["avg"]) / rb["avg"],
                        100.0 * (ra["steady_med"] - rb["steady_med"]) / rb["steady_med"],
                        100.0 * (ra["mm_med"] - rb["mm_med"]) / rb["mm_med"])
    print(f"  c{cyc}: " + "  ".join(
        f"{k} avg={v[0]:+.1f}% steady={v[1]:+.1f}% mm={v[2]:+.1f}%"
        for k, v in d.items()))
