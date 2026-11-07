#!/usr/bin/env python3
"""Per-step GPU time of the down_proj fold ops (28 folded nodes), median over
decode steps. Isolates the fold kernel from CPU feed gaps."""
import glob
import re
import statistics

OP = re.compile(r"^\[submitprof\]\s+(\d+\.?\d*)ms\s+MatMul\s+/(down_proj_\d+)/MatMul")


def fold_ms(path):
    per_step = []
    cur = []
    for line in open(path, errors="ignore"):
        if line.startswith("[submitprof] op-type"):
            if cur:
                per_step.append(sum(cur))
            cur = []
            continue
        m = OP.match(line)
        if m:
            cur.append(float(m.group(1)))
    if cur:
        per_step.append(sum(cur))
    per_step = per_step[1:]  # drop prefill block
    return statistics.median(per_step), len(per_step)


for path in sorted(glob.glob("/tmp/reg/logs/perf_*_c*.log")):
    med, n = fold_ms(path)
    print(f"{path.split('/')[-1]:24s} fold_sum_med={med:.3f}ms  steps={n}")
