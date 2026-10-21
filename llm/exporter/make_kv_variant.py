#!/usr/bin/env python3
"""从 canonical llm.onnx 派生出 KV cache 变体（fp8 / 滑窗环 / 两者），
**不重跑 3.4GB 的权重导出**。

为什么需要这个脚本：这两个变体都是纯粹的**图**改写 —— 只动图输入/输出的
ValueInfoProto、dtype 和节点接线，一个 initializer 的值都不碰。而
qwen3vl_export_onnx.py 每跑一次都要重新 trace 一遍模型并写出 3.4GB 的新
权重文件；本机根分区已经 100%（剩 6.7GB），一个变体的峰值需求（散文件
3.3GB + weights.bin 3.3GB + vkopbin 3.3GB）根本放不下。

所以改成：以 load_external_data=False 读出 llm.onnx —— 每个 initializer 的
external_data（location/offset/length 指向 llm.weights.bin）本来就是 protobuf
的一部分，原样带过去 —— 套改写，把几百 KB 的 protobuf 序列化出去。所有变体
共用同一份 llm.weights.bin，那个文件从头到尾没有被打开过，更不会被写。

用法（在 llm/exporter/ 下跑，llm.weights.bin 就在那里）：
    python3 make_kv_variant.py -o llm_fp8.onnx      --fp8
    python3 make_kv_variant.py -o llm_ring.onnx     --ring
    python3 make_kv_variant.py -o llm_fp8_ring.onnx --ring --fp8 --scale 0.1

然后逐个转换（转换器才是真正要读 3.3GB 权重的那一步，且要离开 VSCode 跑）：
    /home/gwm/.venv/bin/onnx2vkop -i llm_fp8_ring.onnx -o llm_fp8_ring.vkopbin
"""

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "export"))

import onnx  # noqa: E402
from onnx import TensorProto, helper  # noqa: E402
from kv_fp8_rewrite import insert_fp8_kv_cache  # noqa: E402


def add_ring_input(model):
    """声明滑窗环的那条通道：1 元素 int32 图输入 kv_ring_head = 「逻辑行 0 落在
    环里的哪个物理槽」。

    它不被任何 HF 节点消费 —— 它存在的唯一理由是当一条没有任何张量形状能表达、
    又是每轮都会变的会话状态传给融合后的 Attention（同 kv_len 之于形状）：
    converter 见到这个名字就把输入接到算子上并打 has_ring=1（见
    onnx2vkop/optimizer.py 的 fold_kv_cache_append）。不声明时 has_ring=0，
    转换路径与从前完全一致。
    """
    if any(vi.name == "kv_ring_head" for vi in model.graph.input):
        print("[ring] kv_ring_head already present, leaving it alone")
        return False
    model.graph.input.append(helper.make_tensor_value_info(
        "kv_ring_head", TensorProto.INT32, [1]))
    print("[ring] declared graph input kv_ring_head (int32 [1])")
    return True


def external_locations(model):
    """图里所有 external initializer 指向的文件名集合。"""
    locs = set()
    n_ext = 0
    for t in model.graph.initializer:
        if t.HasField("data_location") and t.data_location == onnx.TensorProto.EXTERNAL:
            n_ext += 1
            for kv in t.external_data:
                if kv.key == "location":
                    locs.add(kv.value)
    return n_ext, locs


def check_kv_pattern(model, fp8):
    """离线核对 converter 的 match_kv_cache_append 在**真图**上的前提条件。

    转换器先跑 fuse_into_attention 把 MatMul/Softmax 变成 Attention，再由
    match_kv_cache_append 顺着 cache 的旋转子图把它折进算子。这段逻辑只在探针图
    （gen_cache_graph.py，1 层）上验证过；真图有 28 层 + deepstack +
    image_pad_mask，任何一层长得不一样都会让那一层的匹配**静默失败** —— 那一层
    于是退化成每轮复制整段历史的七节点旋转：慢，而且滑窗环根本不生效。在这里逐层
    核对前提条件，就不用等 3.4GB 转换完才发现。

    只核对**旋转子图本身**，不管 Attention：canonical llm.onnx 历史上转过，
    28 层全中，那是 fuse_into_attention 的事。所以这里不要求 k_full 的消费者里
    有 MatMul —— 原始图里 k_full 的消费者是一堆 Shape（动态 shape 链，前几个 pass
    会清掉）加两个 Unsqueeze，只有转换后才长成 Attention。

    节点一率按**名字**认：protobuf 的 repeated 字段每次迭代产出新的 Python 包装
    对象，所以 `is` 比较（转换器在它自己的 dag_model 上用的那个）在这里不可靠，
    而实测这张图的 7349 个节点全都有唯一非空名字。
    """
    g = model.graph
    by_name = {n.name: n for n in g.node}
    consumers = {}  # tensor -> [(node_name, op_type, input_slot)]
    producer = {}   # tensor -> node_name
    for node in g.node:
        for k, i in enumerate(node.input):
            if i:
                consumers.setdefault(i, []).append((node.name, node.op_type, k))
        for o in node.output:
            if o:
                producer[o] = node.name
    outputs = {o.name for o in g.output}
    graph_inputs = {vi.name for vi in g.input}

    def cons(t):
        return consumers.get(t, [])

    def scalar_int(name):
        """标量整数值，读不到就 None。

        Gather 的 index 是 Constant 节点输出的 int64 标量（raw_data 8 字节，
        经 numpy_helper.to_array 拿到）。小张量不外置（save_model 的
        size_threshold=1024），所以这里读得到；万一读不到就返回 None，调用方
        按「未知」处理而不是当成错误。"""
        pn = producer.get(name)
        if pn is None:
            t = next((t for t in g.initializer if t.name == name), None)
            if t is None or (t.HasField("data_location") and
                             t.data_location == onnx.TensorProto.EXTERNAL):
                return None
            v = t
        elif by_name[pn].op_type == "Constant":
            v = next((onnx.helper.get_attribute_value(a) for a in by_name[pn].attribute
                      if a.name == "value"), None)
            if v is None:
                return None
        else:
            return None
        try:
            arr = onnx.numpy_helper.to_array(v)
        except Exception:
            return None
        return int(arr.reshape(-1)[0]) if arr.size == 1 else None

    def axis_of(node_name, default=0):
        v = next((onnx.helper.get_attribute_value(a) for a in by_name[node_name].attribute
                  if a.name == "axis"), default)
        try:
            return int(v)
        except Exception:
            return default

    problems = []
    nlayers = 0
    while f"past_key_values_{nlayers}" in graph_inputs:
        nlayers += 1
    if nlayers == 0:
        raise SystemExit("[precheck] no past_key_values_* graph inputs found")

    n_ok = 0
    for i in range(nlayers):
        past_name = f"past_key_values_{i}"
        pres_name = f"present_key_values_{i}"
        where = f"layer {i}"

        # ---- past 侧：谁在读这段历史，读的是什么 ------------------------
        cs = cons(past_name)
        if fp8:
            # fp8 的 past 是 E4M3 字节，Gather 读的是 DQ 之后的 fp16。
            if len(cs) != 1 or cs[0][1] != "DequantizeLinear":
                problems.append(f"{where}: {past_name} (fp8) must have exactly one consumer "
                                f"(the DequantizeLinear), got {[(c[1], c[0]) for c in cs]}")
                continue
            dq = by_name[cs[0][0]]
            dcons = cons(dq.output[0])
            if len(dcons) != 2 or any(c[1] != "Gather" for c in dcons):
                problems.append(f"{where}: the DequantizeLinear must feed exactly the two "
                                f"Gathers, got {[(c[1], c[0]) for c in dcons]} -> deleting it "
                                f"would strand a live consumer")
                continue
            gathers = [by_name[c[0]] for c in dcons]
        else:
            gathers = [by_name[c[0]] for c in cs]
            if len(cs) != 2 or any(c[1] != "Gather" for c in cs):
                problems.append(f"{where}: {past_name} must feed exactly the two Gathers, got "
                                f"{[(c[1], c[0]) for c in cs]}")
                continue

        # ---- 两个 Gather：axis=1、index {0,1}、同一份 past --------------
        want_src = gathers[0].input[0]
        if any(len(gn.input) != 2 or gn.input[0] != want_src for gn in gathers):
            problems.append(f"{where}: both Gathers must read the same cache tensor, got "
                            f"{[gn.input[0] for gn in gathers]}")
            continue
        if any(axis_of(gn.name, 1) != 1 for gn in gathers):
            problems.append(f"{where}: Gather axis must be 1, got "
                            f"{[axis_of(gn.name, 1) for gn in gathers]}")
            continue
        idx = [scalar_int(gn.input[1]) for gn in gathers]
        known = sorted(x for x in idx if x is not None)
        if known and known != [0, 1]:
            problems.append(f"{where}: Gather indices must be {{0,1}} (K on dim-1 0, V on 1 — "
                            f"the op's layout), got {idx}")
            continue

        # ---- Gather -> Concat(axis=2)[history, fresh] -------------------
        cats = []
        bad = False
        for gn in gathers:
            gc = cons(gn.output[0])
            if len(gc) != 1 or gc[0][1] != "Concat":
                problems.append(f"{where}: Gather output must feed exactly one Concat, got "
                                f"{[(c[1], c[0]) for c in gc]}")
                bad = True
                break
            cat = by_name[gc[0][0]]
            if len(cat.input) != 2 or cat.input[0] != gn.output[0]:
                problems.append(f"{where}: Concat operands must be [history, fresh] with the "
                                f"history first — the kernel indexes the fresh rows from "
                                f"kv_past; got input[0]={cat.input[0]!r}")
                bad = True
                break
            if axis_of(cat.name) not in (2, -2):
                problems.append(f"{where}: Concat axis must be 2, got {axis_of(cat.name)}")
                bad = True
                break
            cats.append(cat)
        if bad:
            continue

        # ---- 两个 Unsqueeze 汇到一个 Concat(axis=1) = present -----------
        # 原始图里 k_full 的消费者不止一个 Unsqueeze（还有 Shape 链），所以不能
        # 按「唯一消费者」找，得按「哪对 Unsqueeze 共享一个 axis=1 的 Concat」找。
        found = None
        for uk in [by_name[c[0]] for c in cons(cats[0].output[0]) if c[1] == "Unsqueeze"]:
            for uv in [by_name[c[0]] for c in cons(cats[1].output[0]) if c[1] == "Unsqueeze"]:
                shared = {c[0] for c in cons(uk.output[0])} & {c[0] for c in cons(uv.output[0])}
                shared = [by_name[s] for s in shared if by_name[s].op_type == "Concat" and
                          len(by_name[s].input) == 2 and axis_of(s) in (1, -4)]
                if shared:
                    found = (uk, uv, shared[0])
                    break
            if found:
                break
        if found is None:
            problems.append(f"{where}: no pair of Unsqueezes over K_full/V_full feeds a shared "
                            f"axis=1 Concat -> this layer's present tensor is not built the way "
                            f"the fold expects")
            continue
        uk, uv, present_node = found

        # present Concat 的第 0 个操作数必须是 K（present 是 [1,2,...]）。
        k_is_first = present_node.input[0] == uk.output[0]
        if idx[0] is not None and k_is_first != (idx[0] == 0):
            problems.append(f"{where}: present = Concat([input0, input1]) with the K Unsqueeze at "
                            f"position {0 if k_is_first else 1}, but the Gather indices are "
                            f"{idx} -> K/V would be swapped in the [1,2,Hkv,kv,D] layout")

        # ---- present 侧：驱动预分配的那个图输出 -------------------------
        pres_out = present_node.output[0]
        if pres_out not in outputs:
            if not fp8:
                problems.append(f"{where}: the present Concat writes {pres_out!r}, not the graph "
                                f"output {pres_name!r}")
                continue
            qc = cons(pres_out)
            if len(qc) != 1 or qc[0][1] != "QuantizeLinear":
                problems.append(f"{where}: the fp8 present Concat must feed exactly one "
                                f"QuantizeLinear, got {[(c[1], c[0]) for c in qc]}")
                continue
            q = by_name[qc[0][0]]
            if len(q.input) < 2 or q.output[0] != pres_name:
                problems.append(f"{where}: the QuantizeLinear must write the graph output "
                                f"{pres_name!r}, got {q.output[0]!r}")
                continue
            if q.input[1] != dq.input[1]:
                problems.append(f"{where}: the Q and DQ scales differ ({q.input[1]!r} vs "
                                f"{dq.input[1]!r}) — the fold requires one shared per-tensor "
                                f"scale")
                continue
        elif fp8:
            problems.append(f"{where}: the present Concat writes the graph output directly, but "
                            f"the fp8 rewrite should have put a QuantizeLinear in between")
            continue
        n_ok += 1

    print(f"[precheck] {n_ok}/{nlayers} layer(s) match the KV-cache rotation the converter folds "
          f"(fp8={fp8})")
    if problems:
        for p in problems[:10]:
            print(f"[precheck] MISMATCH {p}")
        raise SystemExit(f"[precheck] {len(problems)} problem(s): the converter would silently miss "
                         f"those layers' cache fold — they would keep the O(kv_len) rotation and "
                         f"the ring would not apply")


def check_wiring(model):
    """结构自检：名字引用完整性 + 拓扑序。

    实测 onnx.checker.check_model(full_check=False) 在 load_external_data=False
    下是能跑的（0.1s，四张图一起 0.2s）—— 它本来就不需要权重值：小张量不外置
    （save_model 的 size_threshold=1024），而它要用值的那些（Reshape 的 shape
    常量）都在阈值以下。所以这里先用它；它一旦因为拓扑序挂掉，下面这段按名字
    定位到底是谁读了谁，比 checker 的报错好读。

    唯一不能用的是 full_check=True：它跑到 fp8 的 DequantizeLinear 会报
    "has unsupported type: tensor(float8e4m3fn)" —— 那是 onnx 自己的 shape
    inference 还没支持 fp8，不是图的问题。也正因此，fp8 那条 DQ→fp16 的桥
    **没法**用 checker 的 shape inference 来验证，只能靠 op 级的探针（
    gen_cache_graph.py + cache_probe）和 kv_cache.hpp 的构造路径。
    """
    try:
        onnx.checker.check_model(model, full_check=False)
        print("[check] onnx.checker PASS (full_check=False)")
    except Exception as e:
        print(f"[check] onnx.checker FAIL: {type(e).__name__}: {str(e)[:200]}")

    produced_by = {}  # name -> 产出它的节点下标
    for i, node in enumerate(model.graph.node):
        for o in node.output:
            if o:
                produced_by[o] = i
    inputs_and_inits = {vi.name for vi in model.graph.input}
    inputs_and_inits |= {t.name for t in model.graph.initializer}

    hard_errors = []
    out_of_order = 0
    for i, node in enumerate(model.graph.node):
        for inp in node.input:
            if not inp:  # 可选输入
                continue
            if inp in produced_by:
                if produced_by[inp] > i:
                    out_of_order += 1
            elif inp not in inputs_and_inits:
                hard_errors.append(f"node '{node.name}' ({node.op_type}) reads undefined '{inp}'")
    if hard_errors:
        raise SystemExit("[check] " + "\n[check] ".join(hard_errors[:10]) +
                         (f"\n[check] (+{len(hard_errors) - 10} more)" if len(hard_errors) > 10 else ""))

    # 图输出必须真的被某个节点（或图输入）产出。
    for o in model.graph.output:
        if o.name not in produced_by and o.name not in inputs_and_inits:
            raise SystemExit(f"[check] graph output '{o.name}' is never produced")

    all_names = set(produced_by) | inputs_and_inits
    print(f"[check] {len(model.graph.node)} nodes, {len(model.graph.input)} inputs, "
          f"{len(model.graph.output)} outputs, {len(all_names)} tensor names, "
          f"no undefined references")
    if out_of_order:
        raise SystemExit(f"[check] {out_of_order} reference(s) to a producer that appears later "
                         f"in the node list: the graph is not topologically sorted")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-i", "--input", default=os.path.join(HERE, "llm.onnx"),
                    help="canonical model (default: <script dir>/llm.onnx)")
    ap.add_argument("-o", "--output", required=True,
                    help="variant model to write, e.g. llm_fp8_ring.onnx")
    ap.add_argument("--ring", action="store_true",
                    help="declare the kv_ring_head graph input (sliding window)")
    ap.add_argument("--fp8", action="store_true",
                    help="rewrite the KV cache I/O to E4M3 fp8")
    ap.add_argument("--scale", type=float, default=0.1,
                    help="per-tensor fp8 KV scale (default 0.1)")
    ap.add_argument("--no-causal-mask", action="store_true",
                    help="do NOT add the vkop_causal_mask:attention_bias declaration "
                         "when the canonical model lacks it")
    args = ap.parse_args()

    if not args.ring and not args.fp8:
        raise SystemExit("nothing to do: pass --ring and/or --fp8")

    print(f"[load] {args.input} (external data NOT loaded)")
    model = onnx.load(args.input, load_external_data=False)

    n_ext, locs = external_locations(model)
    print(f"[weights] {n_ext} external initializers -> {sorted(locs)}")
    if len(locs) != 1:
        raise SystemExit(f"[weights] expected exactly one external file, got {sorted(locs)}")
    weights = os.path.join(os.path.dirname(os.path.abspath(args.input)), next(iter(locs)))
    if not os.path.exists(weights):
        raise SystemExit(f"[weights] {weights} does not exist")
    print(f"[weights] reusing {weights} ({os.path.getsize(weights) / 1e9:.2f} GB) — never opened")

    # 因果 mask 的声明挂在 model 级 metadata_props 上（converter 见到才给
    # Attention 打 mask_causal=1，kernel 就能把一行的 KV 扫描在自己的对角线上
    # 截断）。必须挂 model 级：onnxoptimizer.optimize() 重建计算图时会丢掉
    # ValueInfoProto 上的 metadata_props（同 vkop_io_dtype）。
    meta = {p.key: p.value for p in model.metadata_props}
    print(f"[meta] metadata_props = {meta or '{}'}")
    if "vkop_causal_mask:attention_bias" not in meta:
        if args.no_causal_mask:
            print("[meta] causal-mask declaration missing and --no-causal-mask given: "
                  "the converter falls back to a full-row KV scan (correct, slower)")
        else:
            # 语义是已知的：本导出的 attention_bias 就是标准加法因果 mask（对角线上侧
            # 全 fp16 min）—— canonical llm.onnx 只是导出早于这行声明，所以派生文件
            # 里补上，省掉为这个声明重跑一次 3.4GB 权重导出。声明错了的后果是扫描
            # 被截断得太多（结果错误），所以这里明确打出来。
            model.metadata_props.add(key="vkop_causal_mask:attention_bias", value="1")
            print("[meta] canonical model predates the declaration -> added "
                  "vkop_causal_mask:attention_bias=1 (same semantics this exporter "
                  "declares; --no-causal-mask to skip)")
    else:
        print("[meta] causal-mask declaration carried over from the canonical model")

    if args.ring:
        add_ring_input(model)
    if args.fp8:
        insert_fp8_kv_cache(model, scale=args.scale)

    # 语义预检放在结构自检前面：它先按转换器的前提条件逐层核对，报错信息是「哪一层
    # 长得不对」，比拓扑序报错有用。
    check_kv_pattern(model, fp8=args.fp8)
    check_wiring(model)

    # 关键：不用 onnx.save_model（它按 save_as_external_data 走，会把权重又写
    # 一份），直接序列化 protobuf —— external_data 的 location/offset 原样保留。
    with open(args.output, "wb") as f:
        f.write(model.SerializeToString())
    size = os.path.getsize(args.output)
    print(f"[✓] wrote {args.output} ({size / 1e6:.1f} MB)")

    # 读回来确认指向没变。
    back = onnx.load(args.output, load_external_data=False)
    n_ext2, locs2 = external_locations(back)
    assert locs2 == locs and n_ext2 == n_ext, f"round-trip mismatch: {n_ext2} {locs2}"
    print(f"[verify] round-trip OK: {n_ext2} external tensors -> {sorted(locs2)}")
    print(f"[next]  /home/gwm/.venv/bin/onnx2vkop -i {args.output} "
          f"-o {os.path.splitext(args.output)[0]}.vkopbin")


if __name__ == "__main__":
    main()
