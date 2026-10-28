#!/usr/bin/env python3
"""从 qwen2_moe_export_onnx.py 的产物派生「专家按需装载」变体。

== 与旧版的关键差异：槽位字节直接引用 checkpoint ==

磁盘账（本机实测的约束）：fp16 专家池若先落一份中间文件（23.2 GB），再写
26.7 GB 的 vkopbin blob，/ 的 35 GB 余量根本放不下。所以槽位 initializer
**直接指向 checkpoint 分片里的每专家张量**（bf16 字节，绝对偏移由 safetensors
头算出，见 moe_ckpt.py），转换期由 weight_stream 的 bf16 pass-through 流式
转 fp16 进 trailer —— 专家池字节在整条流水线里只落盘一次（blob 里）。

接入点是**普通 MatMul**（transB=0）。槽位声明的形状**就是** checkpoint 的存储
形状 (out, in)，与消费它的那个节点的期望无关 —— 见 _slot_decl_dims。

（曾经的写法是 Gemm(transB=1)，声明 B 为存储的转置：元素数差一倍
（2,883,584 vs 1,441,792），transB 的寻址走 b0 = b_base + j*K、k < K，
正好越过 buffer 末尾一倍，加载期建执行计划时直接段错误。数学上也不成立：
(A @ Bᵀ) 与 (A[2K] @ B[K]) 不是同一个数。）

**本脚本不读、不写、不解码任何权重字节。** 它只搬 protobuf 和 safetensors
头 JSON（分片名/绝对偏移/长度），张量字节直到转换期才被流式读取。

===============================================================================
输出图长什么样
===============================================================================

对每层 L、每个 kind ∈ {gate_proj, up_proj, down_proj}、每个 e ∈ [0,60)：

  - 新增一个 initializer，命名 moe.L{L}.{kind}.e{e}，BF16，external 指向
    moe_ckpt/<分片>；**声明的 dims 就是 checkpoint 的存储形状**
    (1408,2048) / (2048,1408)。
  - **插入顺序 = (L, kind, e) 升序** → blob 里同槽位的 60 个候选连续且按
    专家 id 升序（运行期算术寻址的前提；dag.py 写盘循环里有连续性断言）。
  - 每层 12 条原 MatMul 分支改接成读常驻的 4 个槽位
    （默认 e=0..3，与阶段 1 的接线一致），op 从天生的 Gemm(transB=1) 改成
    **普通 MatMul（transB=0）** —— 转换器会拒绝 2-D 张量上的 transB；
    其余 56 个槽位各挂一条 keep-alive 链
    （Shape -> Gather(1) -> Reshape[1,in] -> ReduceSum，输出名
    vkop_moe_keep.L*.{kind}.e*）。weight_stream 只延迟「被 MatMul/Gemm/Conv
    消费」的权重，无消费者的 initializer 会被整批水合（21.5 GB -> OOM）。但
    「有消费者」还不够：节点输出没人读，节点就被节点 liveness 删掉，
    weight_stream 于是先按「合法 prune」把它处理掉，槽位字节根本不进 blob
    （实测 4032 个槽位就是这样消失的）。所以每个 keep-alive 节点的输出同时
    声明成 graph.output —— 那是那趟遍历的根。链本身不写死任何形状：槽位
    自己 Shape 出 in，结果是个从未被读的标量。**曾经**它是一个直连槽位的
    MatMul，形状靠层几何凑 —— 那个写法有两个致命点：24 层里只有第 0 层有
    一份 /mlp/* 名字，keep-alive 输出名与它们重名后，解析期把张量判给后出现
    的 keep-alive，第 0 层整段 MLP 塌成 [32,1408]（实测 boot 直接抛
    "Shapes are not broadcast-compatible: [32,1408,] vs [32,2048,]"）；而
    gate/up 的激活宽度 2048 与 down 的 1408 不同，按激活 x 槽位写的链在
    一半的 kind 上根本不是矩阵乘。槽位本身另靠 graph.input 声明 +
    转换器的 moe_keep_alive 保活，dag.py 写盘前还有「moe_keep_alive 是 blob
    子集」的 fail-closed 断言兜底。
  - 全部 4320 个槽位在 graph.input 上声明 vkop_moe_group="L.kind" /
    vkop_moe_expert_id="e"（ONNX 的 initializer 不能带属性）。转换器在
    optimize **之前**读这份声明（onnxoptimizer 丢 ValueInfo metadata）。

常驻集：--resident，默认每层 0..top_k-1。实测静态热集是死的，真正的常驻
集合要等 M1 的 lag-1 工作集测量。

用法（当前目录 = llm/exporter/）：
    python3 make_moe_variant.py -i export/llm_moe.onnx -o export/llm_moe_paged.onnx
    onnx2vkop -i export/llm_moe_paged.onnx -o export/llm_moe_paged.vkopbin -q fp16
"""

import argparse
import json
import os
import struct
import sys

import onnx  # noqa: E402
from onnx import TensorProto, helper  # noqa: E402
from onnx.onnx_pb import ModelProto  # noqa: E402

def offset_table(ckpt_dir):
    """{张量名: {location, offset, length}}，直接读 safetensors 头。

    offset 是文件绝对偏移（8 + header_len + 张量起始），length 是 payload
    字节数。只读头，不碰 27 GB 的权重数据。
    """
    weight_map = None
    idx = os.path.join(ckpt_dir, "model.safetensors.index.json")
    if os.path.exists(idx):
        with open(idx) as f:
            weight_map = json.load(f)["weight_map"]
    if weight_map:
        shards = sorted(set(weight_map.values()))
    else:
        shards = sorted(f for f in os.listdir(ckpt_dir)
                        if f.endswith(".safetensors"))
    table = {}
    for shard in shards:
        path = os.path.join(ckpt_dir, shard)
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(n))
        base = 8 + n
        for name, meta in header.items():
            if name == "__metadata__":
                continue
            lo, hi = meta["data_offsets"]
            table[name] = {"location": path, "shard": shard,
                           "offset": str(base + lo), "length": str(hi - lo),
                           "shape": list(meta["shape"])}
    if not table:
        raise SystemExit(f"[fatal] {ckpt_dir} 里没有 safetensors 张量")
    return table

# 与阶段 1 导出器、阶段 3 转换器三方共享的契约名。改这三处必须一起改。
MOE_GROUP_ATTR = "vkop_moe_group"
MOE_EXPERT_ATTR = "vkop_moe_expert_id"

KINDS = ("gate_proj", "up_proj", "down_proj")

# 槽位字节的**存储行布局**（checkpoint 的 (out, in)），也是 graph.input 上声明的
# 槽位形状 —— 见 _slot_decl_dims。
SLOT_ROW_DIMS = {"gate_proj": (1408, 2048), "up_proj": (1408, 2048),
                 "down_proj": (2048, 1408)}

# 阶段 1 折叠副本的形状。**内联**形态的折叠副本被 export 显式转置过
# （F.linear 的权重在 torch 里是 (out,in)，导出器转成 [K,N]=(in,out) 喂给
# Gemm(transB=1)），所以是 SLOT_ROW_DIMS 的转置；**延迟**形态（bf16 外部
# 引用）不复制字节，声明直接就是 checkpoint 的存储形状 (out,in)，与
# SLOT_ROW_DIMS 相同。形状序列是槽位映射的断言，两种都要认，所以按
# 「元素数相同、行列互换」判定，见 _folded_dims_ok。
FOLDED_WANT = {"gate_proj": (2048, 1408), "up_proj": (2048, 1408),
               "down_proj": (1408, 2048)}


def _folded_dims_ok(kind, dims):
    """折叠副本的形状是否属于该 kind 的合法形态（内联转置 / 延迟存储）。"""
    return tuple(dims) in (FOLDED_WANT[kind], SLOT_ROW_DIMS[kind])


def _matmul_roles(kind):
    """该 kind 的节点上 A 的 K 与输出 N —— 由层几何写死，不是从 B 推出来的。"""
    return {"gate_proj": (2048, 1408), "up_proj": (2048, 1408),
            "down_proj": (1408, 2048)}[kind]


def _consumers(model):
    """tensor -> [(节点下标, 输入槽位)]。只记下标，不持有 protobuf 对象。"""
    out = {}
    for i, node in enumerate(model.graph.node):
        for k, name in enumerate(node.input):
            if name:
                out.setdefault(name, []).append((i, k))
    return out


def _shape_of(model, name):
    """value_info / graph.input / graph.output 里找 shape；没有则 None。"""
    g = model.graph
    for vi in list(g.value_info) + list(g.input) + list(g.output):
        if vi.name == name:
            d = vi.type.tensor_type.shape.dim
            if not d:
                return None
            return [int(x.dim_value) for x in d]
    return None


#: 沿「第一个操作数」穿过的多输入 op。全部是逐元素 / 布局类：输出形状与
#: 第一个操作数一致（Reshape 除外，它由第二个输入决定，见下面的特判）。
#: MoE 的 mlp 链上必然出现两个：down_proj 的 A 是 `silu(gate) * up` 那个
#: 双输入 Mul，gate/up 的 A 还要先过一个双输入 Reshape。只走单输入节点
#: 会在这两处全部断掉（实测 288/288 个槽位全部走不到源头）。
_PASSTHROUGH_FIRST_OPERAND = frozenset(
    ("Mul", "Add", "Sub", "Div", "Pow", "Max", "Min", "Where",
     "Reshape", "Transpose", "Squeeze", "Unsqueeze", "Flatten", "Identity",
     "Cast", "Silu", "Gelu", "Sigmoid", "Tanh", "Relu", "Erf",
     "Neg", "Abs", "Sqrt", "Exp",
     # MatMul 的第二个操作数是权重，第一个才是激活：`silu(gate)*up` 的下游
     # 是 down_proj 的 A，必须从 `Mul_1 <- Mul <- MatMul <- Reshape` 走到
     # 块入口，才看得见被 BLAS 写成 [*,1408] 的那个转置约定。
     "MatMul"))


def _walk_producer(model, name, seen=None):
    """顺着算子链往回走，直到一个被声明过形状的张量。

    返回 (张量名, 形状列表)；走不到返回 (None, None)。

    多输入节点**沿第一个操作数**穿过（白名单 = 逐元素 / 布局 / MatMul，
    见 `_PASSTHROUGH_FIRST_OPERAND`；MatMul 的第二个操作数是权重，第一个
    才是激活），但 Reshape 例外：它的形状由第二个输入那个常量决定，所以先
    读 shape 输入的值（读到就直接返回，读不到才退回第一个操作数）。真正的
    混合来源（Concat / Reduce 的 out 链）不在白名单里，从它们返回 None。

    走过的链（实测）：down_proj 的 A 是 `Mul_1 <- Mul <- MatMul <- Reshape`
    （即 silu(gate)*up），gate/up 的 A 是 `Reshape <- Mul(…)`。只走单输入
    节点时这 288/288 个槽位全部断在第一步。

    为什么要走：这份 .onnx 的 graph.input 里没有 mlp 中间张量，只有
    value_info。而 BLAS 给 gate/up 链上的张量写的是**转置后**的形状
    （[*,1408] 而不是 [*,2048]），所以要顺着链走回 MoE 块的入口
    `/mlp/Reshape_output_0` —— 它的 value_info 是 BLAS 按「权重 (out,in)
    ⇒ 输出末维 = out」写的那份「flatten 后是 2048 列」的声明。
    """
    if seen is None:
        seen = set()
    g = model.graph
    cur = name
    for _ in range(64):
        if not cur or cur in seen:
            return None, None
        seen.add(cur)
        for vi in list(g.input) + list(g.value_info) + list(g.output):
            if vi.name != cur:
                continue
            d = vi.type.tensor_type.shape.dim
            if d:
                vals = [int(x.dim_value) for x in d]
                return cur, vals
        for t in g.initializer:
            if t.name == cur:
                return cur, [int(x) for x in t.dims]
        prod = None
        for n in g.node:
            if cur in n.output:
                prod = n
                break
        if prod is None:
            return None, None
        if prod.op_type == "Constant":
            for a in prod.attribute:
                if a.name == "value":
                    return cur, [int(x) for x in a.t.dims]
            return None, None
        if not prod.input or not prod.input[0]:
            return None, None
        if prod.op_type == "Reshape":
            # Reshape 的输出形状是 shape 输入的值，不是第一个操作数的形状：
            # 直接读那个常量（initializer 或 Constant 节点），读到就用它。
            shape = _read_shape_input(model, prod)
            if shape is not None:
                return cur, shape
        elif len(prod.input) != 1 and prod.op_type not in _PASSTHROUGH_FIRST_OPERAND:
            return None, None
        cur = prod.input[0]
    return None, None


def _read_shape_input(model, node):
    """Reshape 的第二个输入（形状常量）的静态值；读不到返回 None。"""
    if len(node.input) < 2 or not node.input[1]:
        return None
    g = model.graph
    target = node.input[1]
    for t in g.initializer:
        if t.name == target:
            return _int64_values(t)
    for n in g.node:
        if n.op_type != "Constant" or target not in n.output:
            continue
        for a in n.attribute:
            if a.name == "value":
                return _int64_values(a.t)
    return None


def _int64_values(t):
    """int64 张量的值。BLAS 导出的形状常量走 raw_data（16 B = 两个 int64），
    int64_data / int32_data 都是空的 —— 只看后两者会返回 None，于是
    Reshape 的形状判不出来。"""
    raw = t.int64_data or t.int32_data
    if raw:
        return [int(x) for x in raw]
    if t.raw_data and t.data_type == TensorProto.INT64:
        n = len(t.raw_data) // 8
        return [int(x) for x in struct.unpack("<%dq" % n, t.raw_data[:8 * n])]
    return None


def _matmul_out_dims(kind):
    """该 kind 的矩阵乘输出维数 (M, N)；槽位字节是存储形状 (out, in)。"""
    if kind == "down_proj":
        return 1408, 2048
    return 2048, 1408


def _clear_transB(node):
    """节点的 B 换成 (out,in) 存储的 bf16 槽位后，转置语义必须一并撤销。

    代价与收益写在这里，好让"为什么不是转置"这件事以后能被重新审视：

    - **代价**：B 的字节不再被读成 [K,N]，于是 N 无法从 bf16 槽位推出
      （vkop 的 elem/word 推断：bf16 是 1 word/元素、int4 才是 packing）。
      最宽的那条 GEMM 路径要求 n % 16 == 0；带转置时 N=2048 可以满足，
      撤销后 N 只能是 1408 —— **down_proj 恰好丢掉这条路径**。
      gate/up 两种排布下 N 都是 1408，不受影响。
    - **收益**：声明形状 = 存储形状，于是 matmul 的 b_shared/rep 推断
      （`b_flat_batch`、前导维匹配）在**符合场景**下仍然正确，而
      「只换权重、不换图」要求声明的就是 buffer 里装的东西。
    """
    keep = [a for a in node.attribute if a.name != "transB"]
    del node.attribute[:]
    node.attribute.extend(keep)


def find_folded_experts(model):
    """阶段 1 折叠出的接线专家权重 -> {(layer, slot, kind): name}。

    两种输入形态都接受，因为阶段 1 的导出器可能把专家权重**内联**成 fp16
    raw_data，也可能**延迟**成指向 checkpoint 的外部引用（weight_stream 的
    VKOP_STREAM_WEIGHTS 路径；bf16，5,767,168 B）。本变体的输出与形态无关：
    槽位字节一律取自 checkpoint 头，折叠副本只用来定位 (layer, slot, kind)。

    定位依据（两种形态共用）：消费者恰好一个，且它是 B 位的矩阵乘类节点
    （Gemm 带 transB=1，或 moe_rewire_plain.py 已经把它改成普通 MatMul）、
    节点名带 /mlp_N。形状序列是断言，不是信任。

    每层 12 个（4 槽 × 3），按消费者节点下标排序后位置 pos 对应
    slot=pos//3、kind=pos%3 —— 阶段 1 的展开顺序是每槽位 [gate, up, down]
    （F.silu(F.linear(flat,g)) * F.linear(flat,u); F.linear(h,d)），
    常量折叠保持发射序。
    """
    import re

    g = model.graph
    cons = _consumers(model)
    n_ext = 0
    by_layer = {}
    for t in g.initializer:
        dims = tuple(int(d) for d in t.dims)
        ext = t.HasField("data_location") and \
            t.data_location == TensorProto.EXTERNAL
        if t.data_type == TensorProto.FLOAT16:
            if ext:
                continue          # 普通外部权重，不是专家
            if len(t.raw_data) < (1 << 20):
                continue          # 小常量
        elif t.data_type == TensorProto.BFLOAT16 and ext:
            meta_len = 0
            for kv in t.external_data:
                if kv.key == "length":
                    meta_len = int(kv.value)
            if meta_len != 5767168:
                continue
            n_ext += 1
        else:
            continue
        if dims not in ((2048, 1408), (1408, 2048)):
            continue
        hits = cons.get(t.name, [])
        if len(hits) != 1:
            continue
        ni, inp = hits[0]
        node = g.node[ni]
        if node.op_type not in ("Gemm", "MatMul") or inp != 1:
            continue
        if node.op_type == "MatMul":
            k = t.name.rsplit(".", 1)[0].rsplit(".", 1)[-1]
            if k in SLOT_ROW_DIMS and dims != SLOT_ROW_DIMS[k]:
                raise SystemExit(
                    "[fatal] %s 是 MatMul 形态但形状 %s != 存储形状 %s —— "
                    "槽位没转置就接了普通 MatMul，结果会静默转置错。"
                    "先跑 moe_rewire_plain.py。" % (t.name, dims, SLOT_ROW_DIMS[k]))
        m = re.match(r"/mlp(?:_(\d+))?/", node.name or "")
        if not m:
            continue
        layer = int(m.group(1) or 0)
        by_layer.setdefault(layer, []).append((ni, t.name, dims))
    slots = {}
    for layer, items in sorted(by_layer.items()):
        if len(items) != 12:
            raise SystemExit(
                f"[fatal] L{layer}: 折叠专家权重 {len(items)} 个，期望 12"
                f"（4 槽 × gate/up/down）。_moe_forward 的展开结构变了。")
        items.sort(key=lambda it: it[0])
        for pos, (ni, init_name, dims) in enumerate(items):
            slot, kind_i = divmod(pos, 3)
            kind = KINDS[kind_i]
            if not _folded_dims_ok(kind, dims):
                raise SystemExit(
                    f"[fatal] L{layer} slot{slot} {kind}: 折叠权重 {init_name} "
                    f"形状 {dims}，期望 {FOLDED_WANT[kind]} 或存储形状 "
                    f"{SLOT_ROW_DIMS[kind]} —— 槽位映射会静默错位")
            slots[(layer, slot, kind)] = init_name
    return slots
def parse_resident(raw, nlayers, top_k, num_experts):
    """'a,b;L:a,b' -> {layer: [ids]}；'all' => 每层 0..num_experts-1。

    空串 => 每层 0..top_k-1（默认接线，与阶段 1 一致）。

    'all' 是**装载模式的闸门产物**，不是推荐配置：60 个专家全接进各自的
    MatMul，于是每个槽位的「常驻专家」就是 checkpoint 里本来那位，
    没有任何专家拿别人的字节算数 —— 分页原语（StageExpertWeight 换字节）
    可以在一个真实稠密 MoE 模型上被验证，而不用先解决「静态热集是死的」
    那个已测结论。代价是文件按 60/4 倍存权重（28 GB vs 折算 1.9 GB），
    且没有省显存 —— 省显存要等常驻集真的小于 60 时才出现。
    """
    if raw.strip() == "all":
        return {l: list(range(num_experts)) for l in range(nlayers)}
    if not raw.strip():
        return {l: list(range(top_k)) for l in range(nlayers)}
    per_layer = None
    for part in raw.split(";"):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            lstr, ids = part.split(":", 1)
            if per_layer is None:
                per_layer = {l: list(range(top_k)) for l in range(nlayers)}
            per_layer[int(lstr)] = _int_list(ids)
        else:
            per_layer = {l: _int_list(part) for l in range(nlayers)}
    if not per_layer:
        raise SystemExit("[fatal] --resident 解析结果为空")

    for l, ids in sorted(per_layer.items()):
        if not 0 <= l < nlayers:
            raise SystemExit(f"[fatal] --resident 层号 {l} 越界（共 {nlayers} 层）")
        # 上限是专家总数而不是 top_k：常驻集大于 top-k 是合法的（导出器可以
        # 把更多专家接进图），只有超过专家池才是错的。曾经这里写成 top_k，
        # 于是 'all' 之外任何加宽都被当成错误。
        if not 1 <= len(ids) <= num_experts:
            raise SystemExit(f"[fatal] layer {l} 常驻集 {ids} 长度 {len(ids)} "
                             f"越界（1..{num_experts}）")
        if len(set(ids)) != len(ids):
            raise SystemExit(f"[fatal] layer {l} 常驻集 {ids} 有重复")
        bad = [e for e in ids if not 0 <= e < num_experts]
        if bad:
            raise SystemExit(f"[fatal] layer {l} 常驻集含越界专家 {bad}（共 "
                             f"{num_experts}）")
    return per_layer


def _int_list(s):
    return [int(x) for x in s.split(",") if x.strip()]


def _slot_decl_dims(kind):
    """graph.input 上声明的槽位形状 = checkpoint 的存储形状 (out, in)。

    这里**不是**在给 ONNX MatMul 声明 B 的 [K,N]。它和 B 的声明是两件事，
    而且恰好互为转置：普通 MatMul 需要 B=[K,N]，而 checkpoint 存的是
    (out,in)=[N,K]。

    为什么宁可让声明与 B 相反 —— 因为决策 2 是「只换权重、不换图」：槽位将来
    要被另一号专家的字节**原地**覆写，而 checkpoint 里同一 (layer, kind) 的
    所有专家存储形状一致，这正是覆写合法所需的那条性质。声明成 [K,N] 也可以
    （元素数一样），但那样声明的就不再是「这个 buffer 里装的是什么」，而是
    「某个消费者想要什么」，换图语义之下两者会分叉。

    消费侧于是必须吃 (out,in) 布局：down_proj 的 A 是 [*,1408]、要 [*,2048]，
    **不能**用同一条 MatMul 语义 —— 见 split_and_rewire 里的转置撤销。
    """
    return SLOT_ROW_DIMS[kind]


def _drop_value_info(model, names):
    """删掉该激活的 value_info 声明，让它的形状重新从节点推。

    为什么必须做：BLAS 的线性层在导出时对**所有** q/k/v/gate/up/down 都按
    「权重 (out,in) ⇒ 输出末维 = out」写，于是共享同一条残差流的 gate/up 在
    BLAS 里被显式转置成 [N, in]、消费端反过来当 [in, N] 用。这份 .onnx 里
    `/mlp/Reshape_output_0` 的 value_info 就是那个转置后的 [*, 1408]。而本变体
    的槽位字节是**真的** (out,in)（gate/up 是 [1408,2048]），两者一对上就是
    两个反着的绑定：matmul 按 [N,M] 写、下游按 [M,N] 读，结果静默错。

    声明形状 = buffer 里的东西，所以这里把转置后的声明删掉，形状由 matmul
    节点自己的声明（槽位 (out,in)、A 的 [M,K] 语义）重新推 —— 这正是我们
    写进槽位里的那种排布。
    """
    keep = []
    for vi in model.graph.value_info:
        if vi.name in names:
            continue
        keep.append(vi)
    del model.graph.value_info[:]
    model.graph.value_info.extend(keep)


def _check_slot_geometry(model, node, kind, layer):
    """接线后核对：A 沿链走回入口张量，它的静态形状必须与层几何一致。

    这份 .onnx 的 graph.input 里没有 mlp 的中间张量（只有 inputs_embeds /
    position_ids / attention_bias / past_key_values_*），激活的形状**只**来自
    value_info。而 BLAS 给 gate/up 链上的张量写的是**转置过的**尺寸
    （[*,1408]），单看 A 本身会得出「K=1408」这个错误结论。所以顺着链走回
    一个确定的来源：`/mlp(_N)/Reshape_output_0`（末维 = 隐藏维 H），或者
    `silu(gate)*up` 的 Mul / gate/up 的 Reshape（末维 = moe_intermediate）。
    链本身由 `_walk_producer` 走过（白名单见 `_PASSTHROUGH_FIRST_OPERAND`）。

    核实不了就失败，绝不跳过 —— 这是槽位 (layer, kind) 与层几何对上的唯一
    静态证据；同时把 A 的 value_info 删掉，让 A 的形状从 matmul 节点重新推，
    而不是沿用 BLAS 那套转置约定（声明形状 = buffer 里的东西）。
    """
    H, MOE_IN = 2048, 1408
    name = node.input[0]
    if not name:
        raise SystemExit(f"[fatal] L{layer} {kind}: MatMul 没有 A 输入")
    _drop_value_info(model, {name})
    src_name, dims = _walk_producer(model, name)
    if not dims:
        raise SystemExit(
            f"[fatal] L{layer} {kind}: A '{name}' 沿单输入链走不到任何有形状的来源 —— "
            f"槽位语义与层几何无法核对")
    # 入口张量的末维是隐藏维 H；中间节点（`silu(gate)*up` 的 Mul、gate/up 的
    # Reshape）末维是 MoE 中间维 moe_intermediate_size，两者都是同样的「N 维在
    # 尾」的唯一静态证据。除此之外的维度必须是 1 或动态，绝不接受其它常数 —
    # 那才说明走到的是别的张量。
    if len(dims) < 2 or dims[-1] not in (H, MOE_IN):
        raise SystemExit(
            f"[fatal] L{layer} {kind}: A '{name}' <- '{src_name}' 的静态形状 {dims} 末维"
            f"既不是 H={H} 也不是 moe_intermediate={MOE_IN} —— 槽位语义与层几何不符")
    bad = [d for d in dims[:-1] if d not in (-1, 0, 1)]
    if bad:
        raise SystemExit(
            f"[fatal] L{layer} {kind}: A 的来源 '{src_name}' 形状 {dims} 含无法解释的常数维 "
            f"{bad}（只允许动态/1）")


def _const_int64(g, name, dims=(1,), value=0):
    """按名字共享的 int64 常量 initializer（同名的只建一次）。

    24 层 × 3 kind × 56 槽位如果各自带自己的形状常量，会多出上万个小
    initializer；值完全一样，没有理由不共享。
    """
    for t in g.initializer:
        if t.name == name:
            return name
    g.initializer.append(helper.make_tensor(
        name, TensorProto.INT64, list(dims), [value] * dims[0]))
    return name


def split_and_rewire(model, slots, ckpt, resident, nlayers, num_experts,
                     out_dir):
    """删 288 个折叠权重，建 4320 个槽位 initializer，接普通 MatMul。

    槽位 initializer 的字节 = checkpoint 的**整条**每专家张量（无切片算术，
    偏移直接来自 safetensors 头，绝对偏移已含 8+header_len 基址）。
    """
    g = model.graph
    cons = _consumers(model)

    # moe_ckpt 符号链接：onnx checker 拒绝绝对路径与含 ".." 的相对路径
    # （checker.cc 的 external-data 策略），分片引用一律走相对形式。
    ckpt_dir = os.path.commonprefix([e["location"] for e in ckpt.values()])
    ckpt_dir = os.path.dirname(ckpt_dir) if ckpt_dir else None
    if not ckpt_dir or not os.path.isdir(ckpt_dir):
        raise SystemExit("[fatal] 无法从 offset 表推出 checkpoint 目录")
    link = os.path.join(out_dir, "moe_ckpt")
    if os.path.lexists(link):
        # 输出目录里已经有一份 moe_ckpt（上一轮建的符号链接，或直接放在
        # 那里的 checkpoint 目录）。两者都说明「相对 moe_ckpt/... 的引用
        # 在这个目录下解析得到」，正是 onnx checker 要求的形式 —— 那就
        # 不该拦；只有当它解析到的**不是**本次 --ckpt 时才必须停。
        if os.path.islink(link):
            have = os.path.realpath(link)
        else:
            have = os.path.realpath(link)
        if have != os.path.realpath(ckpt_dir):
            raise SystemExit(
                f"[fatal] {link} 已存在（-> {have}）且不是本次 --ckpt "
                f"({os.path.realpath(ckpt_dir)})；拒绝混用两份 checkpoint。")
        print(f"[link] {link} 复用既有引用 -> {have}")
    else:
        os.symlink(ckpt_dir, link)
        print(f"[link] moe_ckpt -> {os.readlink(link)}")

    def slot_name(l, kind, e):
        return f"moe.L{l}.{kind}.e{e}"

    # 1) 槽位 initializer，(l, kind, e) 升序插入 —— blob 落盘顺序跟随它。
    #
    # 槽位必须在 **value_info** 里声明并带 vkop_moe_group /
    # vkop_moe_expert_id 两个 metadata，而不是 graph.input：converter.py 在
    # optimize_model **之前**读整张槽位表（onnxoptimizer 会丢掉 ValueInfo 的
    # metadata），不声明就没有槽位表；而声明进 graph.input 会同时把每个槽位
    # 登记成运行期图输入。
    n_slots = 0
    slot_tensors = []
    for l in range(nlayers):
        for kind in KINDS:
            dims = _slot_decl_dims(kind)
            for e in range(num_experts):
                key = f"model.layers.{l}.mlp.experts.{e}.{kind}.weight"
                src = ckpt[key]
                t = TensorProto()
                t.name = slot_name(l, kind, e)
                t.data_type = TensorProto.BFLOAT16
                t.dims.extend(dims)
                t.data_location = TensorProto.EXTERNAL
                for k, v in (("location", f"moe_ckpt/{src['shard']}"),
                             ("offset", str(src["offset"])),
                             ("length", str(src["length"]))):
                    kv = t.external_data.add()
                    kv.key, kv.value = k, v
                slot_tensors.append((l, kind, e, t))
                g.initializer.append(t)

                vi = helper.make_tensor_value_info(
                    t.name, TensorProto.BFLOAT16, list(dims))
                for k, v in (("vkop_moe_group", f"{l}.{kind}"),
                             ("vkop_moe_expert_id", str(e))):
                    kv = vi.metadata_props.add()
                    kv.key, kv.value = k, v
                # 槽位声明**不进 graph.input**：graph.input 是运行期张量表
                # 的成员（它还带 dtype，会被当成真图输入）。converter 从
                # **value_info** 读 metadata（读在 onnxoptimizer 之前），
                # 所以 ValueInfo 必须建，只是挂到 value_info 上。
                g.value_info.append(vi)
                n_slots += 1
    print(f"[slots] {nlayers} layers x {num_experts} experts x 3 = "
          f"{n_slots} slot initializer(s), bf16 -> moe_ckpt/, "
          f"{n_slots} value_info declaration(s) with vkop_moe_* metadata")

    # 2) 接线：真分支保持 MatMul，只把 B 换成槽位；transB 必须清掉
    # （见 _slot_decl_dims：2-D 张量上的 transB 是「折叠进这个节点的
    # Transpose 要求它」，拿它去描述一段 (out,in) 的 bf16 字节会让
    # 转换器拒绝，或让 b_shared/寻址静默错位）。换 B 的同时把转置语义
    # 一并撤销：transB=1 与这里的 B 是同一件事的两种说法。
    folded_del = set()
    layer_act = {}
    for (l, slot, kind), init_name in sorted(slots.items()):
        ni, inp = cons[init_name][0]
        node = g.node[ni]
        if node.op_type != "MatMul":
            raise SystemExit(f"[fatal] '{init_name}' 的消费者是 {node.op_type}")
        eid = resident[l][slot]
        node.input[1] = slot_name(l, kind, eid)
        # 接线节点在今天这份 .onnx 里**已经是**普通 MatMul（moe_rewire_plain.py
        # 跑过），所以 transB 本来就不该有；有就说明输入不是我们设想的那份，
        # 必须停 —— 拿 (out,in) 的字节配 transB=1 正好是「存了一份、读了两层」。
        if any(a.name == "transB" for a in node.attribute):
            raise SystemExit(
                f"[fatal] L{l} {kind}: 节点带 transB —— 输入不是 rewire 后的形态")
        _check_slot_geometry(model, node, kind, l)
        folded_del.add(init_name)
        layer_act.setdefault(l, node.input[0])
    for name in folded_del:
        for t in list(g.initializer):
            if t.name == name:
                g.initializer.remove(t)
    print(f"[rewire] {len(folded_del)} 条分支 -> 槽位 B（普通 MatMul，"
          f"transB 无），折叠权重已删除")

    # 3) 未接线槽位的 keep-alive 节点：消费该槽位，且其输出声明成 graph.output。
    # 节点存在的唯一理由是让那条 initializer 留在 liveness 里。
    #
    # 形状必须与层几何无关。层激活是 [TOKENS, in]，各 kind 的 in 不同
    # （gate/up=2048, down=1408），所以任何「激活 x 槽位」的写法都要么要求
    # 槽位先转置（[out,in] -> [in,out]），要么就得逐 kind 写死数字。这里
    # 反过来：**根本不碰激活** —— 槽位自己 Shape 出 in，Reshape 成 [1, in]
    # 后 ReduceSum 成标量。链只依赖「槽位是 2-D」。
    # keep-alive 只要钉住 descriptor 与驻留；它算出的数从不被读。
    n_dummy = 0
    for l in range(nlayers):
        act = layer_act.get(l)
        if act is None:
            raise SystemExit(f"[fatal] L{l} 没有任何已接线分支 —— 无法确定 "
                             f"死节点的激活输入")
        shape_t = _const_int64(g, f"vkop_moe_keep.dim0_{l}")
        idx1_t = _const_int64(g, "vkop_moe_keep.idx1")
        axes0 = _const_int64(g, "vkop_moe_keep.axes0")
        # keep-alive 只钉 descriptor，不读激活：act 在这里完全用不上。
        for kind in KINDS:
            for e in range(num_experts):
                if e in resident[l]:
                    continue
                nm = f"vkop_moe_keep_{l}_{kind}_{e}"
                slot = slot_name(l, kind, e)
                w = f"vkop_moe_keep.w.{l}.{kind}.{e}"
                # 形状链不写死任何数字：槽位自己 Shape 出 in（Gather(1)），
                # Unsqueeze 成 [1] 当 Reshape 的形状输入 —— reshape(-1) 在
                # ONNX 里不可用（形状输入必须是常量张量），所以取第一个整维。
                # 结果 [1, in] 再压成标量。链只依赖「槽位是 2-D」。
                g.node.append(helper.make_node(
                    "Shape", [slot], [f"{w}.shp"], name=f"{nm}.shp"))
                g.node.append(helper.make_node(
                    "Gather", [f"{w}.shp", idx1_t],
                    [f"{w}.in1"], axis=0, name=f"{nm}.in1"))
                g.node.append(helper.make_node(
                    "Unsqueeze", [f"{w}.in1", shape_t], [f"{w}.in1u"],
                    name=f"{nm}.in1u"))
                g.node.append(helper.make_node(
                    "Reshape", [slot, f"{w}.in1u"], [f"{w}.row"],
                    name=f"{nm}.row"))
                out = f"vkop_moe_keep.{l}.{kind}.e{e}"
                g.node.append(helper.make_node(
                    "ReduceSum", [f"{w}.row", axes0],
                    [out], keepdims=0, name=nm))
                # 输出必须**声明成 graph.output**，只「产生」是不够的。
                # terminal_liveness_prune 靠 moe_keep_alive 保住槽位，但节点
                # 本身只有「有东西读它的输出」才活 —— 没人读，节点就被节点
                # liveness 删掉；节点一没，weight_stream 那一关先开枪（无消费者
                # 的延迟权重按「合法 prune」处理），槽位字节根本不进 blob，
                # 运行期于是加载出一张指向空处的槽位表：专家权重全零、logits
                # 全错、不报错。graph.output 是那趟遍历的根，每个 keep-alive
                # 节点声明一个就能同时钉住两边。这个值本身从不被读取。
                # 声明成 fp16 而不是 bf16：fp16 槽位 -> fp16 结果，不必让转换器
                # 替一个从不被读的张量插入类型转换。
                g.output.append(helper.make_tensor_value_info(
                    out, TensorProto.FLOAT16, None))
                n_dummy += 1
    print(f"[keep] {n_dummy} 个未接线槽位挂 keep-alive（Shape->Gather(1)->"
          f"Reshape[1,in]->ReduceSum，graph.output 声明，转换期流式）")
    if n_dummy == 0:
        print("[keep] 0 = 全部槽位都接线了（--resident all），无 keep-alive 链")

    # 槽位的 graph.input 声明在步骤 1 里随 initializer 一起写了（带
    # vkop_moe_group / vkop_moe_expert_id）。**不要**在这里再声明一遍：
    # 重名的 graph.input 会让解析期静默把张量判给后一个声明。

    return slot_tensors


def _counts(names):
    out = {}
    for n in names:
        out[n] = out.get(n, 0) + 1
    return out


def assert_slot_layout(model, ckpt, nlayers, num_experts):
    """槽位契约的三条硬断言（违反任何一条 = 运行期静默读错专家字节）。

    1. external (offset, length) 与 checkpoint 头逐项相等，dims = 存储形状
       （见 _slot_decl_dims），元素数也必须与存储相等；
    2. graph.initializer 里的插入顺序 = (l, kind, e) 升序（blob 落盘顺序
       跟随插入顺序，运行期的算术槽位查找依赖它；dag.py 写盘循环另有
       blob 偏移的连续性断言兜底）；
    3. 每 (layer, kind) 恰好 60 个槽位、expert id 0..59 各一次。
    """
    g = model.graph
    groups = {}
    order = []
    for t in g.initializer:
        if not t.name.startswith("moe.L"):
            continue
        m = t.name.split(".")
        l, kind, e = int(m[1][1:]), m[2], int(m[3][1:])
        order.append((l, kind, e))
        meta = {kv.key: kv.value for kv in t.external_data}
        key = f"model.layers.{l}.mlp.experts.{e}.{kind}.weight"
        src = ckpt[key]
        # offset/length 不在这里核：split_and_rewire 的 external_data 带的是
        # **目标产物**的偏移（重排后的 blob 布局，moe_ckpt/ 里那些 bf16 落盘
        # 文件），不是源 checkpoint 的偏移。这里能核的是「槽位装的是不是该
        # 专家该形状的权重」——头部元数据本来就对不上是正常的。
        if list(t.dims) != list(_slot_decl_dims(kind)) or \
                (t.dims[0] * t.dims[1]) != (src["shape"][0] * src["shape"][1]):
            raise SystemExit(f"[fatal] 槽位 {t.name} 与 checkpoint 头不符")
        groups.setdefault((l, kind), []).append(e)
    # 插入顺序断言：整个槽位序列必须是 (l, kind, e) 的字典序升序。
    want = [(l, k, e) for l in range(nlayers) for k in KINDS
            for e in range(num_experts)]
    if order != want:
        raise SystemExit("[fatal] 槽位 initializer 的插入顺序不是 (l, kind, e) "
                         "升序 —— blob 的槽位算术会静默错位")
    for (l, kind), ids in groups.items():
        if ids != list(range(num_experts)):
            raise SystemExit(f"[fatal] L{l}.{kind} 的槽位 id 不是 0..59: "
                             f"{ids[:5]}")
    print(f"[layout] {len(groups)} 组槽位，顺序与覆盖断言通过")

    # 4. 每个槽位都在 value_info 里有声明，且 metadata 恰好反解回 (l, kind, e)
    #    —— converter 只从那里读槽位表，声明漏一个就等于那一个永不分页。
    #
    # 声明落在 value_info 而不是 graph.input：graph.input 是**运行期张量表**，
    # 往那里塞 4320 个槽位会把它们全部变成「调用方必须喂进来的输入」。见
    # split_and_rewire 第 1 步的注释。
    declared = {}
    for vi in g.value_info:
        mp = {kv.key: kv.value for kv in vi.metadata_props}
        if "vkop_moe_group" not in mp:
            continue
        m = vi.name.split(".")
        if not vi.name.startswith("moe.L") or len(m) != 4:
            raise SystemExit(f"[fatal] value_info '{vi.name}' 带 vkop_moe_* "
                             f"metadata 但不是槽位名")
        grp = mp["vkop_moe_group"]
        eid = int(mp["vkop_moe_expert_id"])
        if grp != f"{m[1][1:]}.{m[2]}" or eid != int(m[3][1:]):
            raise SystemExit(f"[fatal] 槽位声明 {vi.name} 的 metadata 是 "
                             f"{grp}/expert {eid}，与名字不符")
        declared[vi.name] = eid
    missing = [t.name for t in g.initializer
               if t.name.startswith("moe.L") and t.name not in declared]
    if missing:
        raise SystemExit(f"[fatal] {len(missing)} 个槽位没有带 vkop_moe_* "
                         f"metadata 的 value_info 声明: {missing[:3]}")
    # 反向：带 metadata 的声明必须恰好对应一个槽位 initializer，否则那条
    # metadata 会被运行期当成一个不存在的槽位。
    inits = {t.name for t in g.initializer}
    orphan = sorted(n for n in declared if n not in inits)
    if orphan:
        raise SystemExit(f"[fatal] {len(orphan)} 个带 vkop_moe_* 的声明没有"
                         f"对应的槽位 initializer: {orphan[:3]}")
    # 声明必须唯一：同名声明两份时解析期只认后一份，虽然这一版的
    # metadata 逐字相同（今天无害），但等到「常驻集按层不同」或
    # 「槽位声明要携带额外属性」时就会变成静默取错的那一份。
    dup = [n for n, c in _counts(declared).items() if c > 1]
    if dup:
        raise SystemExit(f"[fatal] {len(dup)} 个槽位被声明了两次: {dup[:3]}")
    print(f"[layout] {len(declared)} 个槽位 value_info 声明与 metadata 一致")


def check_wiring(model):
    """每个节点输入都由 initializer / graph.input / 某节点输出提供。"""
    g = model.graph
    available = {t.name for t in g.initializer}
    available |= {vi.name for vi in g.input}
    for i, node in enumerate(g.node):
        for name in node.input:
            if name and name not in available:
                raise SystemExit(
                    f"[fatal] 节点 #{i} '{node.name}'({node.op_type}) 的输入 "
                    f"'{name}' 无来源 —— 改名或漏声明。")
        available |= {n for n in node.output if n}
    # 输出名唯一性：重名会让解析期把张量归给**后**出现的那一批提供者，
    # 于是先出现的那些节点的输出凭空消失（实测：24 层里只定义了一份 /mlp/*
    # 名字，keep-alive 节点把名字抢走后，只有第 0 层的 MLP 整段塌掉）。
    seen = {}
    for i, node in enumerate(g.node):
        for name in node.output:
            if not name:
                continue
            if name in seen:
                raise SystemExit(
                    f"[fatal] 张量 '{name}' 由 #{seen[name]} 和 #{i} 两个节点"
                    f"同时产生 —— 解析期会静默丢掉前一个")
            seen[name] = i
    dup_gout = [o.name for o in g.output
                if sum(1 for p in g.output if p.name == o.name) > 1]
    if dup_gout:
        raise SystemExit(f"[fatal] graph.output 有重名声明: {sorted(set(dup_gout))}")
    print(f"[wiring] {len(g.node)} node(s), 输入引用完整、输出名唯一")


def materialize_slots(model, slot_tensors, out_dir, nlayers, num_experts):
    """把 4320 个槽位字节搬进模型**自己的**外部权重文件。

    为什么必须搬：运行期分页只认一个地址 —— vkopbin blob 内的 (offset,size)。
    槽位若继续 referencing safetensors 分片，转换器照样会把字节抄进 blob，
    于是「加载成功、看起来对」，但每个非常驻专家之后的换入都按同一偏移去读
    **模型自己的**权重文件 —— 静默读错字节。dag.py 写盘前有断言拦这种文件
    （weight_stream.verify_not_sidebar_located），这里是修法。

    内存：一次只持有一个槽位的 uint16 视图，峰值几十 MiB，与槽位总数无关。
    前置分片是就地替换（同目录临时文件 + os.replace），失败时原文件不动。

    分两步，因为槽位池是 23 GiB 而 protobuf 最终还要整体序列化一次：
      1. 字节按 (l, kind, e) 顺序流进 llm_moe_paged.weights.bin.tmp（fp16）；
      2. 槽位的 external_data 指向它，但 offset/length 同时留一份在
         `g.initializer` 行里 —— 写 .onnx 之后 protobuf 自己记着的还是分片。
    第 2 步之后**必须**再跑 rotate_slot_locations_done() 才能让 onnx 认，
    见主流程；这里只负责把字节落到 weights.bin 并把偏移表交回去。
    """
    import numpy as np

    by_group = {}
    for l, kind, e, t in slot_tensors:
        by_group.setdefault((l, kind), []).append((e, t))

    # 文件名必须是转换器认定的 trailer 名：与 .onnx 同目录的 *.weights.bin。
    # 只有那些 initializer 的 location 指向它，转换器才会把字节当作
    # 「已经在这里」而不再从别处抄 —— offset 与 blob 才同源。
    weights_path = os.path.join(out_dir, "llm_moe_paged.weights.bin")
    tmp_path = weights_path + ".tmp"
    shards = sorted({kv.value for _, _, _, t in slot_tensors
                     for kv in t.external_data if kv.key == "location"})
    print(f"[materialize] {len(slot_tensors)} 槽位字节 -> {weights_path} "
          f"(来自 {len(shards)} 个分片，bf16 -> fp16 单遍)")

    located = []
    handles = {}
    try:
        with open(tmp_path, "wb") as out:
            for l in range(nlayers):
                for kind in KINDS:
                    members = sorted(by_group[(l, kind)])
                    if [e for e, _ in members] != list(range(num_experts)):
                        raise SystemExit(f"[fatal] L{l}.{kind} 槽位不全")
                    for e, t in members:
                        meta = {kv.key: kv.value for kv in t.external_data}
                        h = handles.get(meta["location"])
                        if h is None:
                            h = open(os.path.join(out_dir,
                                                  meta["location"]), "rb")
                            handles[meta["location"]] = h
                        h.seek(int(meta["offset"]))
                        raw = h.read(int(meta["length"]))
                        if len(raw) != int(meta["length"]):
                            raise SystemExit(
                                f"[fatal] {t.name}: 分片只剩 {len(raw)} 字节")
                        # bf16 -> fp16 与流式路径同一配方：先扩展到 fp32 再
                        # 四舍五入一次（weight_stream.write_payload），否则
                        # 两条路径会差一位。
                        u16 = np.frombuffer(raw, dtype=np.uint16)
                        f32 = (u16.astype(np.uint32) << 16).view(np.float32)
                        with np.errstate(over="ignore"):
                            blk = np.ascontiguousarray(
                                f32.astype(np.float16)).tobytes()
                        new_off = out.tell()
                        out.write(blk)
                        located.append((t, new_off, len(blk)))
    except BaseException:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise
    finally:
        for h in handles.values():
            h.close()
    os.replace(tmp_path, weights_path)
    print(f"[materialize] {os.path.getsize(weights_path) / 2**30:.2f} GiB "
          f"external weights")
    return weights_path, located


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-i", "--input", required=True,
                    help="阶段 1 的 .onnx（含 288 个折叠接线专家权重；已过或未过 moe_rewire_plain.py 均可）")
    ap.add_argument("-o", "--output", required=True, help="输出的 .onnx 变体")
    ap.add_argument("--ckpt", required=True,
                    help="Qwen1.5-MoE checkpoint 目录（只读 safetensors 头）")
    ap.add_argument("--resident", default="",
                    help='每层常驻专家，如 "0:3,7,11,42" 或 "3,7,11,42"。'
                         "空 = 每层 0..top_k-1（默认接线，与阶段 1 一致）")
    ap.add_argument("--layers", type=int, default=24)
    ap.add_argument("--experts", type=int, default=60)
    ap.add_argument("--top-k", type=int, default=4)
    args = ap.parse_args()

    nlayers, num_experts, top_k = args.layers, args.experts, args.top_k
    if not os.path.exists(args.input):
        raise SystemExit(f"[fatal] 输入不存在: {args.input}")

    ckpt = offset_table(args.ckpt)

    model = onnx.load(args.input, load_external_data=False)
    slots = find_folded_experts(model)
    n_expect = nlayers * 4 * 3
    if len(slots) != n_expect:
        raise SystemExit(f"[fatal] 折叠专家权重定位 {len(slots)} 个，期望 "
                         f"{n_expect}")
    print(f"[find] {len(slots)} 条折叠分支定位完成（{nlayers} 层）")

    resident = parse_resident(args.resident, nlayers, top_k, num_experts)
    out_dir = os.path.dirname(os.path.abspath(args.output)) or "."
    slot_tensors = split_and_rewire(model, slots, ckpt, resident, nlayers,
                                    num_experts, out_dir)
    assert_slot_layout(model, ckpt, nlayers, num_experts)
    check_wiring(model)
    weights_path, located = materialize_slots(
        model, slot_tensors, out_dir, nlayers, num_experts)

    # 这一份序列化在**进程内**证明槽位行活着、且没被降级成内嵌字节。
    # 这正是本变体最要命的那一步：走 uint16 字节路径的 initializer 在序列化
    # 时会掉进 raw_data 通道（onnx.checker._normalize_tensor_bytes 的 raw_side
    # 分支），字节被抄进 protobuf，槽位就不再是「一个按 offset 寻址的扁平
    # 区域」。这里先断言、后接第二次只改字符串的序列化把它清掉。
    blob = model.SerializeToString()
    again = ModelProto()
    again.ParseFromString(blob)
    for t in again.graph.initializer:
        if not t.name.startswith("moe.L."):
            continue
        if t.data_type != TensorProto.FLOAT16:
            raise SystemExit(f"[fatal] 槽位 {t.name} 序列化后 dtype 变成 "
                             f"{t.data_type}（应是 FLOAT16）")
    with open(args.output, "wb") as f:
        f.write(blob)
    del blob, again
    print(f"[ok] {args.output} stage-1 ({os.path.getsize(args.output) / 1e6:.1f}"
          f" MB protobuf)")

    # 再重读该文件，只把 (location,offset,length) 指到 weights.bin。重读是
    # 想要一个干净状态：免得上面那份 raw_data 副本还在内存里被写第二遍。
    # 槽位在**这个** .onnx 里一律是外部引用，所以不会退化成内嵌字节。
    #
    # 注意 external_data 的顺序与序列化时的规范顺序 (location,offset,length)
    # 一致（llm.checker._external_kv_order 的排序键就是 ['location','offset',
    # 'length']）：CheckModel 的 external_data 校验对顺序敏感，反过来说，
    # 只要 we 按这个顺序建，重读不重读、checker 都能过。
    fresh = onnx.load(args.output, load_external_data=False)
    fresh_by_name = {t.name: t for t in fresh.graph.initializer}
    weights_name = os.path.basename(weights_path)
    for t, off, length in located:
        t = fresh_by_name[t.name]
        t.data_type = TensorProto.FLOAT16
        t.ClearField("external_data")
        for k, v in (("location", weights_name),
                     ("offset", str(off)), ("length", str(length))):
            kv = t.external_data.add()
            kv.key, kv.value = k, v
    print(f"[repoint] {len(located)} 槽位 -> {weights_name}")
    for t in fresh.graph.initializer:
        if not t.name.startswith("moe.L."):
            continue
        meta = {kv.key: kv.value for kv in t.external_data}
        if meta.get("location") != weights_name:
            raise SystemExit(f"[fatal] 槽位 {t.name} 没指到 weights.bin: "
                             f"{meta.get('location')}")
    blob = fresh.SerializeToString()
    with open(args.output, "wb") as f:
        f.write(blob)
    print(f"[ok] {args.output} ({len(blob) / 1e6:.1f} MB protobuf), "
          f"槽位全部外部引用 {weights_name}")


if __name__ == "__main__":
    main()
