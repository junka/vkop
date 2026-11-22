#!/usr/bin/env python3
"""Summarise llm_chat decode A/B logs (from tools/perf/ab_decode.sh).

Logs are grouped into arms by file name: <arm>_c<cycle>.log. Per run it reports
decode step avg / median / steady median, and -- when the run had
VKOP_SUBMIT_PROF=1 -- the per-step MatMul and TOTAL GPU medians. Per arm it
reports the min over cycles, then per cycle the change of every arm against the
first arm given (or --base).

--op REGEX (repeatable) adds a column: the per-step median of the summed GPU ms
of every per-op line whose node name matches, e.g.
    --op 'down_proj(_\\d+)?/MatMul' --op 'gate_proj' --op 'up_proj'
Per-op lines need VKOP_SUBMIT_PROF=1 (VKOP_SUBMITPROF_TOPN=0 for all ops).
The first profile block covers prefill and is dropped.

    tools/perf/parse_perf.py tools/perf/logs/*.log
    tools/perf/parse_perf.py --base graph --op down_proj tools/perf/logs/*.log
"""
import argparse
import re
import statistics
from collections import OrderedDict
from pathlib import Path

STEP = re.compile(r"^\[r(\d+)\] (\d+\.?\d*)ms")
MM = re.compile(r"^\[submitprof\] MatMul\s+(\d+)\s+(\d+\.?\d*)\s+(\d+\.?\d*)")
TOTAL = re.compile(r"^\[submitprof\] TOTAL\s+(\d+)\s+(\d+\.?\d*)")
OPLINE = re.compile(r"^\[submitprof\]\s+(\d+\.?\d*)ms\s+(\w+)\s+(\S+)")
ARM = re.compile(r"^(.+)_c(\d+)$")

nan = float("nan")


def med(xs):
    return statistics.median(xs) if xs else nan


def parse(path, ops, steady_from):
    steps, mm, tot = [], [], []
    blocks, cur = [], []          # per profile block: [(node, ms)]
    for line in open(path, errors="ignore"):
        m = STEP.match(line)
        if m:
            steps.append((int(m.group(1)), float(m.group(2))))
            continue
        m = OPLINE.match(line)
        if m:
            cur.append((m.group(3), float(m.group(1))))
            continue
        if line.startswith("[submitprof] op-type") or TOTAL.match(line):
            if cur:
                blocks.append(cur)
                cur = []
        m = MM.match(line)
        if m:
            mm.append(float(m.group(2)))
            continue
        m = TOTAL.match(line)
        if m:
            tot.append(float(m.group(2)))
    if cur:
        blocks.append(cur)
    dec = [ms for i, ms in steps if i >= 1]
    steady = [ms for i, ms in steps if i >= steady_from]
    r = {
        "n": len(dec),
        "avg": statistics.mean(dec) if dec else nan,
        "med": med(dec),
        "steady": med(steady),
        "mm": med(mm[1:]),
        "total": med(tot[1:]),
    }
    for rx in ops:
        r[rx.pattern] = med([sum(ms for node, ms in b if rx.search(node))
                             for b in blocks[1:]])
    return r


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("logs", nargs="+")
    ap.add_argument("--op", action="append", default=[], help="node-name regex")
    ap.add_argument("--base", help="arm the others are compared against")
    ap.add_argument("--steady-from", type=int, default=100)
    args = ap.parse_args()
    ops = [re.compile(o) for o in args.op]
    cols = ["avg", "med", "steady", "mm", "total"] + [o.pattern for o in ops]

    runs = OrderedDict()          # arm -> {cycle: result}
    for p in args.logs:
        m = ARM.match(Path(p).stem)
        arm, cyc = (m.group(1), int(m.group(2))) if m else (Path(p).stem, 1)
        runs.setdefault(arm, {})[cyc] = parse(p, ops, args.steady_from)

    fmt = lambda r: " ".join(f"{c}={r[c]:.2f}" for c in cols if r[c] == r[c])
    for arm, cyc in runs.items():
        print(f"== {arm} ==")
        for c in sorted(cyc):
            print(f"  c{c}: n={cyc[c]['n']} {fmt(cyc[c])}")
        best = {c: min(r[c] for r in cyc.values()) for c in cols}
        print(f"  min: {fmt(best)}")

    base = args.base or next(iter(runs))
    if base not in runs or len(runs) < 2:
        return
    print(f"\n== per cycle vs {base} ==")
    for c in sorted(runs[base]):
        rb = runs[base][c]
        parts = []
        for arm, cyc in runs.items():
            if arm == base or c not in cyc:
                continue
            ra = cyc[c]
            d = " ".join(f"{k}={100 * (ra[k] - rb[k]) / rb[k]:+.1f}%"
                         for k in cols if rb[k] == rb[k] and ra[k] == ra[k] and rb[k])
            parts.append(f"{arm}: {d}")
        print(f"  c{c}: " + " | ".join(parts))


if __name__ == "__main__":
    main()
