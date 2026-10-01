"""把 qi21_dit 的 prefill/decode 两张图导出成 onnx2vkop 友好的 ONNX。

和 llm/exporter/qwen3vl_export_onnx.py 同一个套路：legacy TorchScript 导出器
（`dynamo=False`，避开 dynamo 的 guard）、opset 17、权重合进单个 external-data 文件。

    # 不需要 checkpoint：2 层随机权重，只看图结构和 fold 模式是否成形
    /Users/doudou/qi21-env/bin/python qi21_export_onnx.py --tiny
    # 真权重（qi21_weights.load_dit 的 fp16 堆叠权重）
    /Users/doudou/qi21-env/bin/python qi21_export_onnx.py

产物：`dit_prefill.onnx` / `dit_decode.onnx` + 各自的 weights.bin。两张图共用 96% 的
权重，所以 external data 是**两份**（fp16 下各 ~14.2 GB / 合计 ~28.5 GB），磁盘不是问题
（本机 1.4 TB 空余）。内存侧的账见 `export_pair`：合并是流式的，峰值只有 torch 里那一份
权重。C++ 驱动则必须按"载 prefill → 跑一次 → 释放 → 载 decode → 跑 40 步"的顺序，峰值
才是一份权重 + 激活。

导出后会打一份 op 直方图并断几条 onnx2vkop 依赖的结构不变量（见 `structural_check`）。
"""

import argparse
import collections
import math
import os
from pathlib import Path

import onnx
import torch

import qi21_dit as M
from qi21_dit import DecodeGraph, PrefillGraph

HERE = Path(__file__).resolve().parent
OPSET = 17


def tiny_weights(dtype):
    """2 层 / 2 head / 真 head_dim 的随机权重，只为看图结构。seed 固定，跨次可复现。"""
    cfg = M.DiTConfig(num_layers=2, heads=2, head_dim=128, mlp_ratio=3,
                      axes_dims_rope=(16, 56, 56), eps=1e-6, context_in_dim=96,
                      in_channels=16)
    w = M.DiTWeights(cfg).eval()
    torch.manual_seed(0)
    with torch.no_grad():
        for _, p in w.named_parameters():
            if p.ndim == 1 and p.numel() == cfg.head_dim:
                p.normal_(1.0, 0.05)
            elif p.ndim == 1:
                p.normal_(0.0, 0.05)
            else:
                p.normal_(0.0, 0.02)
    return cfg, w.to(dtype)


def real_weights(dtype: str):
    """`dtype` 是 **字符串**（"fp16"/"bf16"），因为 `load_dit` 按名字选 cast 目标；
    传 torch.dtype 进去会在它的 assert 上炸。"""
    from qi21_weights import load_dit
    w = load_dit(dtype)
    return w.cfg, w


# ---------------------------------------------------------------------------
# 导出
# ---------------------------------------------------------------------------
def export_pair(cfg, w, out_dir, S_p=128, S_t=256, suffix=""):
    """trace 两张图、各自把散权重合进单个 external-data 文件，返回 [(mode, path, loc, ModelProto)]。

    `S_p/S_t` 只是 **trace 用的样例尺寸**（两轴的 `prefix_len`/`target_len` 都标成了
    dynamic），故意取小：真尺寸 1024x1024 是 S_t=4096，legacy 导出器是"真跑一遍再抓图"，
    拿真尺寸 trace 一层就是 1.2 GB 的分数矩阵。

    **必须逐图交替**（trace 一张 -> 合并一张），不能"两张都 trace 完再一起合并"：legacy 导出器
    把大常量写成散文件，文件名里只有节点编号（`_Constant_150_attr__value`）、不带图名，两张图
    落在同一个扁平命名空间里 —— 后 trace 的那张会覆盖前一张的同名文件，而合并只按文件名取数据，
    覆盖了也看不出来，结果是 prefill 的某个张量里躺着 decode 的权重（数值静默错误）。交替做之后，
    每张图的散文件在下一张图开始 trace 之前就已搬进它自己的单文件并删掉了。

    合并是流式的（一次 8 MB），proto 用 `load_external_data=False` 读，所以那 14.2 GB 从没进过
    protobuf —— 峰值就是 torch 里那一份权重，不需要"先 `del w` 再合并"。
    """
    n, hd = cfg.num_layers, cfg.head_dim
    # 输入样例用 fp16 存（真权重即 fp16），bias/rope 表跟权重同 dtype。
    dtype = w.wq.dtype if torch.is_tensor(w.wq) else torch.float16
    assert int(S_t ** 0.5) ** 2 == S_t, f"S_t={S_t} 不是完全平方，拼不出 lh*lw 网格"
    lh = lw = int(S_t ** 0.5)

    # 一张表覆盖整条联合序列：前 S_p 行是文本段，后 S_t 行是目标图块。
    cos, sin = M.joint_rope_positions(S_p, lh, lw, cfg.axes_dims_rope, hd)
    cos_p, sin_p = (t[:S_p].contiguous().to(dtype) for t in (cos, sin))
    cos_t, sin_t = (t[S_p:].contiguous().to(dtype) for t in (cos, sin))
    bias_p = torch.zeros(1, 1, S_p, S_p, dtype=dtype).masked_fill(
        torch.triu(torch.ones(S_p, S_p, dtype=torch.bool), 1), M.FP16_MIN)
    bias_t = torch.zeros(1, 1, S_t, S_p + S_t, dtype=dtype)

    w.prepare_for_export()

    kv_shape = (1, 2, cfg.heads, S_p, hd)
    pref_in = [torch.randn(1, S_p, cfg.context_in_dim, dtype=dtype),
               cos_p, sin_p, torch.zeros(1, dtype=torch.float32), bias_p]
    pref_in_names = ["prompt_embeds", "cos", "sin", "timestep_zero", "attention_bias"]
    pref_out_names = [f"present_kv_{i}" for i in range(n)]
    pref_dynamic = {
        "prompt_embeds": {1: "prefix_len"},
        "cos": {0: "prefix_len"}, "sin": {0: "prefix_len"},
        "attention_bias": {2: "prefix_len", 3: "prefix_len"},
    }

    dec_in = ([torch.randn(1, S_t, cfg.in_channels, dtype=dtype),
               torch.tensor([0.5], dtype=torch.float32), cos_t, sin_t] +
              [torch.zeros(kv_shape, dtype=dtype) for _ in range(n)] + [bias_t])
    dec_in_names = (["target_latents", "timestep", "cos", "sin"] +
                    [f"past_kv_{i}" for i in range(n)] + ["attention_bias"])
    dec_out_names = ["sample"]
    dec_dynamic = {
        "target_latents": {1: "target_len"},
        "cos": {0: "target_len"}, "sin": {0: "target_len"},
        "sample": {1: "target_len"},
        "attention_bias": {2: "target_len", 3: "kv_len"},
    }
    for i in range(n):
        # KV 是 (1, 2, H, S_p, head_dim) —— 序列轴在第 3 轴，不是第 2 轴。
        pref_dynamic[f"present_kv_{i}"] = {3: "prefix_len"}
        dec_dynamic[f"past_kv_{i}"] = {3: "prefix_len"}

    made = []
    # legacy 导出器写散权重时用**相对 CWD** 的名字，读的时候也是 —— 所以整段必须在 out_dir 里跑，
    # 否则 out_dir 是临时目录时会去看调用方 CWD 里的同名旧文件而报 FileExistsError。
    prev_cwd = os.getcwd()
    os.chdir(out_dir)
    try:
        for mode, graph, inputs, in_names, out_names, dynamic in (
                ("prefill", PrefillGraph(cfg, w), pref_in, pref_in_names, pref_out_names,
                 pref_dynamic),
                ("decode", DecodeGraph(cfg, w), dec_in, dec_in_names, dec_out_names,
                 dec_dynamic)):
            path = Path(f"dit_{mode}{suffix}.onnx")
            loc = f"dit_{mode}{suffix}.weights.bin"
            print(f"[trace] {mode}: {len(in_names)} 输入 / {len(out_names)} 输出, "
                  f"{n} 层, S_p={S_p} S_t={S_t}")
            with torch.no_grad():
                torch.onnx.export(graph, tuple(inputs), str(path),
                                  input_names=in_names, output_names=out_names,
                                  dynamic_axes=dynamic, opset_version=OPSET,
                                  do_constant_folding=True, dynamo=False)
            print(f"[merge] {mode} -> {loc}")
            m = onnx.load(str(path), load_external_data=False)
            merge_scattered(m, loc)
            onnx.save(m, str(path))
            made.append((mode, out_dir / path, loc, m))
    finally:
        os.chdir(prev_cwd)
    return made


ALIGN = 64  # ORT 按 mmap 读 external data，offset 对齐到 64B
CHUNK = 8 << 20
INLINE_KEEP = 1024  # 小于这个的 inline 常量留在 proto 里，进文件不划算


def merge_scattered(m, loc):
    """把 `m` 里所有权重搬进单个 `loc` 文件，并把 proto 改成引用它（原地改）。

    调用方负责随后 `onnx.save`。`m` 必须是以 `load_external_data=False` 读进来的：torch 的
    legacy 导出器把够大的常量写成**散文件**（`_Constant_150_attr__value` 这种，一个张量一个
    文件），proto 里只有 location/offset/length；够小的还 inline 在 `raw_data`。两类都要搬，
    因为 onnx2vkop 只认一个权重文件，而散文件留在目录里会让"哪个是最终产物"变得含糊。

    不用 `onnx.save_model(save_as_external_data=True, all_tensors_to_one_file=True)`：真权重
    （32 层）实测会写出**坏账** —— 部分张量的 `external_data.length` 和它 dims 算出来的字节数
    不一致（`dit_prefill.onnx` 里 35 个张量如此，ORT 直接拒载：
    `GetExternalDataInfo TensorProto: /Constant_150_output_0 external data size mismatch.
    Computed size: 33554432, external_data.length: 100663296`），而且它要求先把整份 initializer
    读进内存（`load_external_data=True`），和 torch 那份叠在一起就是 28.5 GB。
    这里自己按快照顺序流式追加，每个张量写之前先断言"散文件大小 == dims×itemsize"，
    一次只缓冲 8 MB，写完的 length 就是**实际搬走的字节数**，坏账没有产生的余地。
    """
    import numpy as np
    from onnx.external_data_helper import _get_all_tensors

    # 快照成 list：写入顺序就是这里的遍历顺序（offset 是累加的），物化一遍让这条账可复现、
    # 也好在断言失败时直接打印出是第几个张量。
    tensors = list(_get_all_tensors(m))
    Path(loc).unlink(missing_ok=True)
    scattered, written, offset, nbytes, pad_bytes = set(), 0, 0, 0, 0
    with open(loc, "wb") as out:
        for t in tensors:
            ext = {kv.key: kv.value for kv in t.external_data}
            itemsize = np.dtype(onnx.helper.tensor_dtype_to_np_dtype(t.data_type)).itemsize
            want = math.prod(t.dims) * itemsize if t.dims else itemsize
            src = ext.get("location")
            if src:
                # 已经是散文件的：不管多小都搬进来（不搬的话它指向的文件就得一直留着，
                # 而且 proto 里会剩下一堆指向散文件的记录，"最终产物只有一个权重文件"就不成立了）。
                got = os.path.getsize(src)
                assert got == want, f"{t.name}: 散文件 {src} 有 {got}B，dims 说该有 {want}B"
                with open(src, "rb") as f:
                    read = _copy(f, out, got)
                scattered.add(src)
            elif len(t.raw_data) == want >= INLINE_KEEP:
                # 用 `raw_data` 而不是 int64_data/float_data 当判据：后两者是"解包"存储，
                # 搬进按字节寻址的 external data 就得自己编码，不值得 —— 真图上权重全是
                # raw_data 形态，非 raw_data 的只有几个 int64 常量轴。
                read = _copy_bytes(t.raw_data, out)
            else:
                continue
            pad = (-read) % ALIGN
            out.write(b"\0" * pad)
            _point_at(t, loc, offset, read)
            offset += read + pad
            nbytes += read
            pad_bytes += pad
            written += 1
    for f in scattered:
        Path(f).unlink(missing_ok=True)
    print(f"  {written} 个张量 / {nbytes:,}B 数据 + {pad_bytes:,}B 对齐填充")
    return written, nbytes


def _copy(src, dst, n):
    """流式搬 n 字节，返回**实际写入的字节数**（一次只缓冲 CHUNK）。

    用 `read`/`write` 而不是 `readinto`：`BufferedReader.readinto()` 的 length 参数是
    positional-only（真权重第一次走到这条路径时就炸在 `TypeError: BufferedReader.readinto()
    takes no keyword arguments`），而 `read(k)` 本来就限长，还省掉一个预先分配的 buffer。
    """
    copied = 0
    while copied < n:
        chunk = src.read(min(CHUNK, n - copied))
        if not chunk:
            raise EOFError(f"读到 {copied}B 就 EOF，该有 {n}B")
        dst.write(chunk)
        copied += len(chunk)
    return copied


def _copy_bytes(data, dst):
    for i in range(0, len(data), CHUNK):
        dst.write(data[i:i + CHUNK])
    return len(data)


def _point_at(t, loc, offset, length):
    """把张量 proto 改成"数据在 `loc` 的 [offset, offset+length) 里"。

    没用 `external_data_helper.set_external_data`：它在 `not raw_data` 时**直接抛**
    （"Tensor ... does not have raw_data field. Cannot set external data"），而散文件形态的
    张量恰恰就没有 raw_data —— 那正是本函数最常见的一条路径。也不带它那个 length 校验
    （它只比 offset+length 和文件大小，不看 dims），这里要的是 dims 那条账。
    """
    del t.external_data[:]
    # raw_data 必须清掉：EXTERNAL 张量同时带数据的话 proto 就存了两份，
    # 14.2 GB 的 .onnx 单文件就是这么来的。
    t.ClearField("raw_data")
    t.data_location = onnx.TensorProto.EXTERNAL
    for k, v in (("location", loc), ("offset", offset), ("length", length)):
        entry = t.external_data.add()
        entry.key, entry.value = k, str(v)


# ---------------------------------------------------------------------------
# 结构检查：onnx2vkop 的 fold 依赖具体节点形态，导出器换了写法就会静默失配
# ---------------------------------------------------------------------------
def structural_check(made, cfg):
    """按"死代码消除后的真实节点数"断言 —— onnx2vkop 的 fold 认具体节点形态，导出器
    换写法（或 PyTorch 升级换了 decomposition）就会在这里先响，而不是等到转换期静默失配。

    prefill 只返回 32 个 `present_kv_i`，**最后一层的 x 更新链没人消费**，会被
    `_jit_pass_onnx_eliminate_unused_items` 整段剪掉（该层只剩 k/v 投影 + rope）。于是
    每层的活节点数不是常数：
      MatMul  = 5(前置) + 9*(n-1) + 2(末层只剩 wk/wv)
      Softmax = n-1，Neg(rope) = 2n-1（末层的 q 也是死的），
      LayerNorm = 2(n-1)+1，Min/Max = 2(n-1)，MLP = 3(n-1)
    decode 的 `sample` 要吃满最后一层，所以全是 n 倍。实测 n=2 时 prefill 的
    MatMul=16 / Softmax=1 / Neg=3 / LN=3 / Min=Max=2，和公式逐项吻合。
    """
    n = cfg.num_layers
    expect = {
        "prefill": {"MatMul": 5 + 9 * (n - 1) + 2, "Softmax": n - 1, "Neg": 2 * n - 1,
                    "LayerNormalization": 2 * (n - 1) + 1, "Min": 2 * (n - 1),
                    "Max": 2 * (n - 1), "Sqrt": 2 * n},
        "decode": {"MatMul": 6 + 9 * n, "Softmax": n, "Neg": 2 * n,
                   "LayerNormalization": 2 * n + 1, "Min": 2 * n, "Max": 2 * n,
                   "Sqrt": 2 * n},
    }
    for mode, path, loc, m in made:
        ops = collections.Counter(node.op_type for node in m.graph.node)
        print(f"\n[{path.name}] nodes={sum(ops.values())} "
              f"inputs={len(m.graph.input)} outputs={len(m.graph.output)}")
        for op, cnt in sorted(ops.items(), key=lambda kv: (-kv[1], kv[0])):
            print(f"  {op:<20} {cnt}")

        for op, want in expect[mode].items():
            assert ops[op] == want, (path.name, op, ops[op], want)
        for banned in ("Clip", "If", "Loop", "NonZero", "ScatterND", "GatherND", "Compress"):
            assert banned not in ops, (path.name, banned)
        # 权重全在单个 external-data 文件里。**必须扫全部张量**（graph initializer + node
        # attribute）：这张图只有 6 个 initializer，其余 300+ 个常量是 node attribute 形态，
        # 只看 `m.graph.initializer` 会以为"权重不见了"。
        from onnx.external_data_helper import _get_all_tensors
        locs = {kv.value for t in _get_all_tensors(m)
                if t.data_location == onnx.TensorProto.EXTERNAL
                for kv in t.external_data if kv.key == "location"}
        assert locs == {loc}, (path.name, locs)
        # rope 的 7 节点融合模式（optimizer.py:fold_rotary_embedding）要求
        # Neg <- Slice <- Concat <- Mul <- Mul <- Add 这条链在图里出现。
        assert ops["Concat"] and ops["Slice"] and ops["Tanh"], path.name
        ext_check(path, loc, m)
        print(f"  [ok] {mode}: 节点计数与死代码模型一致，external data 单一 {loc}")


def ext_check(path, loc, m):
    """合并后的 external data 必须**逐字节严丝合缝**：每个张量的 length == 元素数×itemsize、
    offset 对齐、区间互不重叠、文件末尾不多不少正好是最后一个张量 + 它的对齐填充。

    为什么单独查：真权重导出后权重是以 **node attribute** 形态被 externalize 的（图里只有
    6 个 graph-level initializer，其余 300+ 个是 `convert_attribute=True` 挪出去的），所以
    只数 `graph.initializer` 会得出"权重没了"的假结论；而 ORT 拒载 prefill 那次报的正是
    offset/length 的账对不上 dims —— 只有按区间账才查得出来。
    """
    import numpy as np
    from onnx.external_data_helper import _get_all_tensors

    ext = [t for t in _get_all_tensors(m) if t.data_location == onnx.TensorProto.EXTERNAL]
    assert ext, path
    recs = []
    for t in ext:
        d = {kv.key: kv.value for kv in t.external_data}
        assert d["location"] == loc, (path.name, t.name, d.get("location"))
        itemsize = np.dtype(onnx.helper.tensor_dtype_to_np_dtype(t.data_type)).itemsize
        want = math.prod(t.dims) * itemsize
        got = int(d.get("length", -1))
        assert got == want, (path.name, tuple(t.dims), got, want)
        offset = int(d.get("offset", -1))
        assert offset >= 0 and offset % ALIGN == 0, (path.name, t.name, offset)
        recs.append((offset, got))
    assert len({(o, l) for o, l in recs}) == len(recs), f"{path.name}: 有张量指向同一段"
    spans = sorted(recs)
    assert all(a[0] + a[1] <= b[0] for a, b in zip(spans, spans[1:])), f"{path.name}: 区间重叠"
    size = (path.parent / loc).stat().st_size
    end = max(o + l for o, l in spans)
    # 填充只可能出现在每个张量**后面**，所以：文件末尾 = 最后一个张量的 end + 不足一个 ALIGN；
    # 总填充 = 文件大小 - 数据量，上界是 ALIGN × 张量数。
    assert end <= size < end + ALIGN, f"{path.name}: 文件尾 {size:,}B 不在 {end:,}B 之后一个 ALIGN 内"
    slack = size - sum(l for _, l in recs)
    assert 0 <= slack < ALIGN * len(recs), f"{path.name}: 填充 {slack:,}B 超出 ALIGN×{len(recs)}"
    print(f"  [ext] {len(recs)} 个 external 张量 / 数据 {size - slack:,}B + 填充 {slack:,}B = 文件 {size:,}B")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiny", action="store_true", help="2 层随机权重，只看图结构")
    ap.add_argument("--bf16", action="store_true", help="真权重用 bf16（默认 fp16）")
    ap.add_argument("--prefix-len", type=int, default=128, help="trace 样例的文本段长度")
    ap.add_argument("--target-len", type=int, default=256, help="trace 样例的目标图块数")
    ap.add_argument("--out", default=str(HERE), help="输出目录")
    args = ap.parse_args()

    torch.set_grad_enabled(False)
    if args.tiny:
        cfg, w = tiny_weights(torch.float16)
    else:
        # 字符串而不是 torch.dtype：见 `real_weights`。
        cfg, w = real_weights("bf16" if args.bf16 else "fp16")
    out_dir = Path(args.out).resolve()  # resolve 必须在 chdir 之前，否则相对 --out 会失效
    made = export_pair(cfg, w, out_dir, args.prefix_len, args.target_len,
                       suffix="_tiny" if args.tiny else "")
    structural_check(made, cfg)
    print("\n[done] 下一步：onnx2vkop -i dit_prefill.onnx -q fp16 -u")


if __name__ == "__main__":
    main()
