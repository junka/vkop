"""Qwen1.5-MoE checkpoint 的只读访问：键 → (分片, 绝对偏移, 长度, 形状) 表。

safetensors 文件布局是 [u64 header_len LE][header JSON][数据...]。header 里每个
张量的 data_offsets 相对**数据区起点**（即 8 + header_len），而 ONNX external_data
的 offset 是**绝对文件偏移** —— 本模块一次算好绝对偏移，导出器（torch 侧按张量
填充）和 make_moe_variant.py（protobuf 侧按字节区间声明槽位）共用同一张表，
谁都不必重复解析 header。

只读 header JSON（每分片 ~1.7 MB），从不打开张量字节；真正的字节读取走
`safe_open`（mmap、按键、逐张量），见 qwen2_moe_export_onnx.py 的填充循环。

这个 checkpoint 的所有 4659 个张量都是 BF16（config.json 的 torch_dtype 与
safetensors 头一致）。bf16 → fp16 的位型转换只有一个正确定义（低 16 位丢掉的
fp32 再舍入到 fp16），与 onnx2vkop 转换器 `_bf16_bytes_to_fp16` 的逐位结果一致；
这里在 torch 侧转（`tensor.to(torch.float16)`），转换器侧对 external 张量走
weight_stream 的流式转换 —— 两处对同一份字节必须给出同样的 fp16。
"""

import json
import os

# 专家张量的键形如 model.layers.7.mlp.experts.42.gate_proj.weight。shared expert
# 是 model.layers.7.mlp.shared_expert.gate_proj.weight —— 中间多了 "shared"，不含
# ".mlp.experts."，所以这个子串判断不会误伤 shared expert。
_EXPERT_MARK = ".mlp.experts."


def shard_headers(ckpt_dir):
    """{分片文件名: (header dict, 数据区绝对基址)}。基址 = 8 + header_len。"""
    out = {}
    for fn in sorted(os.listdir(ckpt_dir)):
        if not fn.endswith(".safetensors"):
            continue
        path = os.path.join(ckpt_dir, fn)
        with open(path, "rb") as f:
            n = int.from_bytes(f.read(8), "little")
            header = json.loads(f.read(n))
        out[fn] = (header, 8 + n)
    if not out:
        raise SystemExit(f"[fatal] {ckpt_dir} 下没有 .safetensors 分片")
    return out


def offset_table(ckpt_dir):
    """{ckpt_key: dict(shard, location, offset, length, shape, dtype)}。

    location 是**绝对路径**（本模块的读取方 safe_open 直接用）；写进 ONNX 的
    external location 另行处理 —— onnx checker 拒绝绝对路径，也拒绝含 ".." 的
    相对路径（checker.cc:990 附近），所以产物里引用 checkpoint 的规范形式是
    「.onnx 旁的符号链接 moe_ckpt/<分片名>」，由导出器创建并写相对引用。
    """
    table = {}
    for fn, (header, base) in shard_headers(ckpt_dir).items():
        loc = os.path.abspath(os.path.join(ckpt_dir, fn))
        for key, meta in header.items():
            if key == "__metadata__":
                continue
            off, end = meta["data_offsets"]
            table[key] = {
                "shard": fn,
                "location": loc,
                "offset": base + off,
                "length": end - off,
                "shape": list(meta["shape"]),
                "dtype": meta["dtype"],
            }
    return table


def is_expert_key(key):
    return _EXPERT_MARK in key


def non_expert_keys(table):
    return [k for k in table if not is_expert_key(k)]
