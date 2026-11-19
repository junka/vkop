#!/usr/bin/env python3
""".vkopbin 静态分析，三个子命令：

  report    runtime 节点直方图 + SUPPORTED/MISSING 分类、Cast dtype 细节、MatMul 形状
  dataflow  int64/int32 张量数据流分类：哪些 GPU op 会收到 int DATA（而不只是
            小 int64 initializer 标量）——4 字节字 shader 会不会误读
  liveness  反向活性分析：unsupported op 里哪些是 live（必须实现）vs 常量折叠
            剩下的 dead（可剪）

SUPPORTED 统一取 onnx2vkop.shader_map.runtime_op_names()（单一事实源，与
Ops.hpp 的 convert_opstring_to_enum 对应）。手抄 op 列表漂移过：缺 RMSNorm、
RotaryEmbedding、FusedElemwise、ReduceMean、Min、Max、Mod、QuantizeLinear、
DequantizeLinear、Attention，把当时的新 op 误报成 unsupported。

FlatBuffers 模块用仓库内生成物（model/pypi/onnx2vkop/generated），不依赖
/tmp/vkmod 之类机器本地路径。

用法:
    python3 check/vkop_inspect.py report    [model.vkopbin]
    python3 check/vkop_inspect.py dataflow  [model.vkopbin]
    python3 check/vkop_inspect.py liveness  [model.vkopbin]
"""
import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
_PKG = _REPO_ROOT / "model" / "pypi"
sys.path.insert(0, str(_PKG))
# 生成绑定内部按顶层绝对导入引用 vkop.model.*（vkop_model.fbs 的 namespace），
# 所以 generated 目录本身也要可导入——与 onnx2vkop/inspect.py 同一处理。
sys.path.insert(0, str(_PKG / "onnx2vkop" / "generated"))
from onnx2vkop import shader_map  # noqa: E402
from onnx2vkop.generated.vkop.model import Model  # noqa: E402

SUPPORTED = shader_map.runtime_op_names()


def load(path):
    with open(path, "rb") as f:
        model = Model.Model.GetRootAsModel(f.read(), 0)
    nodes = [model.Nodes(i) for i in range(model.NodesLength())]
    return model, nodes


def cmd_report(model, nodes):
    counts = Counter(n.OpType().decode() for n in nodes)
    missing = Counter()
    print(f"{'count':<7}{'op':<24}{'status'}")
    print("-" * 46)
    for op, c in counts.most_common():
        status = "SUPPORTED" if op in SUPPORTED else "MISSING"
        if status == "MISSING":
            missing[op] = c
        print(f"{c:<7}{op:<24}{status}")

    print(f"\nTotal runtime nodes: {len(nodes)}")
    print(f"Distinct ops:        {len(counts)}")
    print(f"Missing ops:         {sum(missing.values())} nodes / {len(missing)} types")

    print("\n=== Cast nodes (shape / dtype) ===")
    for n in nodes:
        if n.OpType().decode() != "Cast":
            continue
        to = None
        for a in range(n.AttributesLength()):
            attr = n.Attributes(a)
            if attr.Key().decode() == "to" and attr.Type() == 0:  # Int64
                to = attr.Ival()

        def shape_of(t):
            return [t.Dims(j) for j in range(t.DimsLength())]
        in_s = shape_of(n.Inputs(0)) if n.InputsLength() > 0 else []
        out_s = shape_of(n.Outputs(0)) if n.OutputsLength() > 0 else []
        print(f"  {n.Name().decode():<58} in={in_s} -> out={out_s} to={to}")

    print("\n=== MatMul node shapes ===")
    for n in nodes:
        if n.OpType().decode() != "MatMul":
            continue

        def shp(t):
            return [t.Dims(j) for j in range(t.DimsLength())]
        in0 = shp(n.Inputs(0)) if n.InputsLength() > 0 else []
        out = shp(n.Outputs(0)) if n.OutputsLength() > 0 else []
        print(f"  {n.Name().decode():<58} {in0} -> {out}")


def cmd_dataflow(model, nodes):
    producer = {}
    for i, n in enumerate(nodes):
        for j in range(n.OutputsLength()):
            producer[n.Outputs(j).Name().decode()] = i

    init_dtype = {}
    for i in range(model.InitializersLength()):
        e = model.Initializers(i)
        init_dtype[e.Name().decode()] = e.Dtype().decode()

    op_of = {i: n.OpType().decode() for i, n in enumerate(nodes)}

    stats = defaultdict(Counter)  # (op, slot) -> Counter(dtype_source)
    examples = defaultdict(list)
    for i, n in enumerate(nodes):
        op = op_of[i]
        for j in range(n.InputsLength()):
            in_t = n.Inputs(j)
            name = in_t.Name().decode()
            r = in_t.DimsLength()
            if r == 0:
                continue  # empty dims -> metadata/nullptr path
            if name in init_dtype:
                src = ("init:" + init_dtype[name], name)
            elif name in producer:
                src = ("node:" + op_of[producer[name]], name)
            else:
                continue
            dt, tname = src
            if dt.startswith("init:int") or dt.startswith("node:int"):
                stats[(op, j)][dt] += 1
                if len(examples[(op, j)]) < 2:
                    examples[(op, j)].append(f"{tname} (rank {r})")

    print(f"{'op':<12}{'slot':<5}{'count':<6} int64/int32 sources")
    print("-" * 80)
    for (op, slot), c in sorted(stats.items()):
        parts = ", ".join(f"{k}: {v}" for k, v in c.most_common())
        ex = " | ".join(examples[(op, slot)])
        print(f"{op:<12}{slot:<5}{sum(c.values()):<6}{parts}  e.g. {ex}")

    print("\n=== int64-producing op breakdown ===")
    prod_count = Counter()
    for n in nodes:
        op = n.OpType().decode()
        if op in ("Shape", "Unsqueeze", "Squeeze", "ScatterND"):
            prod_count[op] += 1
        elif op == "Cast":
            for a in range(n.AttributesLength()):
                attr = n.Attributes(a)
                if attr.Key().decode() == "to":
                    prod_count["Cast->to" + str(attr.Ival())] += 1
    for k, v in prod_count.most_common():
        print(f"  {k}: {v}")


def cmd_liveness(model, nodes):
    from collections import deque

    producer = {}
    for i, n in enumerate(nodes):
        for j in range(n.OutputsLength()):
            producer[n.Outputs(j).Name().decode()] = i

    worklist = deque()
    for i in range(model.OutputsLength()):
        worklist.append(model.Outputs(i).Name().decode())

    live_nodes = [False] * len(nodes)
    seen_tensors = set()
    while worklist:
        tname = worklist.popleft()
        if tname in seen_tensors:
            continue
        seen_tensors.add(tname)
        i = producer.get(tname)
        if i is None or live_nodes[i]:
            continue
        live_nodes[i] = True
        n = nodes[i]
        for j in range(n.InputsLength()):
            worklist.append(n.Inputs(j).Name().decode())

    total = Counter()
    live = Counter()
    for i, n in enumerate(nodes):
        op = n.OpType().decode()
        total[op] += 1
        if live_nodes[i]:
            live[op] += 1

    print(f"{'op':<20}{'total':>7}{'live':>7}")
    print("-" * 34)
    for op, c in total.most_common():
        mark = "  <-- MISSING, live!" if op not in SUPPORTED and live[op] else ""
        print(f"{op:<20}{c:>7}{live[op]:>7}{mark}")

    unsupported_live = {op: live[op] for op in total if op not in SUPPORTED and live[op]}
    unsupported_dead = {op: total[op] - live[op] for op in total if op not in SUPPORTED}
    print(f"\nLive unsupported nodes: {sum(unsupported_live.values())} "
          f"/ {sum(total[op] for op in total if op not in SUPPORTED)} total")
    print(f"Dead unsupported (prunable): {sum(unsupported_dead.values())}")

    print("\n=== Live unsupported node names ===")
    for i, n in enumerate(nodes):
        op = n.OpType().decode()
        if op not in SUPPORTED and live_nodes[i]:
            nm = n.Name().decode()
            outs = [n.Outputs(j).Name().decode() for j in range(n.OutputsLength())]
            print(f"  [{op}] {nm} -> {outs}")
    print(f"\ntotal live unsupported listed above: {sum(unsupported_live.values())}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("command", choices=["report", "dataflow", "liveness"])
    ap.add_argument("model", nargs="?", default="llm/exporter/llm.vkopbin")
    args = ap.parse_args()
    model, nodes = load(args.model)
    {"report": cmd_report, "dataflow": cmd_dataflow,
     "liveness": cmd_liveness}[args.command](model, nodes)


if __name__ == "__main__":
    main()