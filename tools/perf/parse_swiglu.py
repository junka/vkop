import glob, re, statistics
OP = re.compile(r"^\[submitprof\]\s+(\d+\.?\d*)ms\s+MatMul\s+/(down_proj|gate_proj|up_proj)(_\d+)?/MatMul")
def per_step(path):
    rows=[];cur=[]
    for line in open(path, errors='ignore'):
        if line.startswith('[submitprof] op-type'):
            if cur: rows.append(cur); cur=[]
            continue
        m=OP.match(line)
        if m: cur.append((m.group(2), float(m.group(1))))
    if cur: rows.append(cur)
    rows=rows[1:]
    s={k: statistics.median([sum(t for n,t in r if n==k) for r in rows]) for k in ('gate_proj','up_proj','down_proj')}
    s['total']=s['gate_proj']+s['up_proj']+s['down_proj']
    return s
for path in sorted(glob.glob('/tmp/reg/logs/perf_*_c*.log')):
    s=per_step(path)
    print(f"{path.split('/')[-1]:22s} gate={s['gate_proj']:.2f} up={s['up_proj']:.2f} down={s['down_proj']:.2f} total={s['total']:.2f}")
