#!/usr/bin/env python3
"""Parse VKOP_SUBMIT_PROF=1 logs (TOPN=0) into per-step op tables.

Log shape, per timed step:
    [submitprof] --- top <N> slowest ops ---
    [submitprof]   74.945ms  MatMul       /down_proj/MatMul
    ...
    [submitprof] TOTAL  1152  2426.20

Modes:
  decode  ratio/absolute metrics for the decode A/B. The criterion is the
          per-step down_proj/lm_head ratio (frequency-invariant); absolute ms
          is reported for reference only.
  blocks  per-block down_proj stats, for the two-prompt prefill run (block 5 =
          the warm second prefill).

Usage:
  parse_blocks.py decode logs/ab3_*.log
  parse_blocks.py blocks logs/pre2_*.log
"""
import re, sys, statistics

OP = re.compile(r'([0-9.]+)ms\s+(\w+)\s+(\S+)')
TOT = re.compile(r'\[submitprof\] TOTAL\s+(\d+)\s+([0-9.]+)')


def blocks(path):
    out, cur = [], None
    for ln in open(path, errors='replace'):
        if '[submitprof] --- top' in ln:
            cur = {}
            continue
        m = OP.search(ln)
        if m and cur is not None:
            cur[m.group(3)] = (m.group(2), float(m.group(1)))
        t = TOT.search(ln)
        if t and cur is not None:
            cur['__total__'] = float(t.group(2))
            out.append(cur)
            cur = None
    return out


def pct(v, q):
    v = sorted(v)
    return v[int(q * (len(v) - 1))]


def decode(files, warmup=4):
    down, mm, rat, tot = [], [], [], []
    for f in files:
        for b in blocks(f)[warmup:]:
            lm = [t for nm, (fa, t) in b.items()
                  if fa == 'MatMul' and nm == '/lm_head/MatMul']
            dn = [t for nm, (fa, t) in b.items()
                  if fa == 'MatMul' and 'down_proj' in nm]
            if not lm or not dn:
                continue
            down.append(sum(dn))
            mm.append(sum(t for nm, (fa, t) in b.items() if fa == 'MatMul'))
            rat.append(sum(dn) / lm[0])
            tot.append(b.get('__total__', 0.0))
    return down, mm, rat, tot


def show_decode(files, label):
    d, m, r, t = decode(files)
    if not d:
        print(f'{label}: no steady steps')
        return
    print('%-10s n=%3d  down p10=%6.2f med=%6.2f  mm med=%6.2f  '
          'total med=%6.2f  ratio p10=%.4f med=%.4f'
          % (label, len(d), pct(d, .1), statistics.median(d),
             statistics.median(m), statistics.median(t),
             pct(r, .1), statistics.median(r)))


def show_blocks(files):
    for f in files:
        print(f)
        for i, b in enumerate(blocks(f), 1):
            dn = [v for nm, (fa, v) in b.items()
                  if fa == 'MatMul' and 'down_proj' in nm]
            if not dn:
                continue
            print('  block %d: ops=%2d med=%6.2f min=%6.2f max=%6.2f sum=%7.1f'
                  % (i, len(dn), statistics.median(dn), min(dn), max(dn),
                     sum(dn)))


if __name__ == '__main__':
    mode = sys.argv[1]
    files = sys.argv[2:]
    if mode == 'decode':
        show_decode(files, 'merged')
    elif mode == 'blocks':
        show_blocks(files)
    else:
        sys.exit('mode must be decode|blocks')
