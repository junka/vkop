"""把 Qwen-Image-2.1 的 transformer 分片加载成 qi21_dit.DiTWeights（逐层堆叠 + 目标精度）。

分片名 -> 层号 的映射取自 `diffusion_pytorch_model.safetensors.index.json`；张量用
`safe_open` **逐个** 读、`copy_` 直接写进它自己的目标行，所以全程不会同时持有"整份
fp32 占位 + 整份 fp16 副本 + 整 shard 字典"这三份（fp16 权重 14.2 GB，多任何一份都会
把这台 36 GB 的机器推进交换区）。

    python -c "from qi21_weights import load_dit; m = load_dit('fp16')"
"""

import json
import struct
from pathlib import Path

import torch
from safetensors import safe_open

from qi21_dit import DiTConfig, DiTWeights

MS_ROOT = Path("/Users/doudou/qi21-env/ms/models/Qwen--Qwen-Image-2.1/snapshots/master")
SCALARS = {
    "img_in.weight": "img_in_w",
    "txt_in.text_norm.weight": "txt_norm_w",
    "txt_in.in_layer.weight": "txt_in_w",
    "txt_in.out_layer.weight": "txt_out_w",
    "time_text_embed.timestep_embedder.linear_1.weight": "tl1_w",
    "time_text_embed.timestep_embedder.linear_2.weight": "tl2_w",
    "modulation.1.weight": "mod_w",
    "norm_out.linear.weight": "out_lin_w",
    "proj_out.weight": "proj_out_w",
}
LAYERS = {
    "attn.to_q.weight": "wq",
    "attn.to_k.weight": "wk",
    "attn.to_v.weight": "wv",
    "attn.to_out.0.weight": "wo",
    "attn.norm_q.weight": "nq",
    "attn.norm_k.weight": "nk",
    "img_mlp.gate_layer.weight": "gate",
    "img_mlp.proj.weight": "up",
    "img_mlp.out.weight": "down",
}
PREFIX = "transformer_blocks."


def read_config(root: Path = MS_ROOT) -> DiTConfig:
    raw = json.loads((root / "transformer" / "config.json").read_text())
    for k in ("patch_size", "out_channels", "causal_condition"):
        assert raw[k] in (1, 64, True), (k, raw[k])
    return DiTConfig(num_layers=raw["num_layers"], heads=raw["num_attention_heads"],
                     head_dim=raw["attention_head_dim"], mlp_ratio=raw["mlp_ratio"],
                     axes_dims_rope=tuple(raw["axes_dims_rope"]), eps=raw["eps"],
                     context_in_dim=raw["context_in_dim"], in_channels=raw["in_channels"])


def safetensors_header(path: Path) -> dict:
    with path.open("rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        h = json.loads(f.read(n))
    h.pop("__metadata__", None)
    return h


def _wanted_in_shard(header: dict) -> set[str]:
    """这个 shard 里哪些张量要读 —— 用 header 决定，不靠逐个试读，省 IO。"""
    out = set()
    for name in header:
        if name in SCALARS:
            out.add(name)
            continue
        if name.startswith(PREFIX):
            _, _, tail = name[len(PREFIX):].partition(".")
            if tail in LAYERS:
                out.add(name)
    return out


def load_dit(dtype: str = "fp16", root: Path = MS_ROOT, quiet: bool = False) -> DiTWeights:
    assert dtype in ("fp16", "bf16", "fp32"), dtype
    cast = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[dtype]
    cfg = read_config(root)
    model = DiTWeights(cfg, cast)
    own = dict(model.named_parameters())

    index = json.loads((root / "transformer" /
                        "diffusion_pytorch_model.safetensors.index.json").read_text())["weight_map"]
    shards = sorted({index[k] for k in index})
    remaining = set(index)
    # 逐层权重的"到哪一层了"用位图记，缺层时在最后一步统一报；不缓存任何张量。
    filled = {a: 0 for a in set(LAYERS.values())}
    for path in (root / "transformer" / s for s in shards):
        want = {k for k, v in index.items() if v == path.name}
        pick = _wanted_in_shard(safetensors_header(path)) & want
        # **逐张量**读，不 `load_file` 整份：目标缓冲已经 14.2 GB（fp16），再叠一个 10 GB
        # 的整 shard 字典就冲到 ~25 GB，36 GB 统一内存要开始交换。`safe_open` 走 mmap，
        # `copy_` 把单个张量边转 dtype 边写进它那一行 —— 瞬时只有单个张量（最大 2.1 GB）
        # 是活的。
        with safe_open(path, framework="pt") as f:
            for name in pick:
                remaining.discard(name)
                if name in SCALARS:
                    own[SCALARS[name]].data.copy_(f.get_tensor(name))
                    continue
                idx, _, tail = name[len(PREFIX):].partition(".")
                attr = LAYERS[tail]
                own[attr].data[int(idx)].copy_(f.get_tensor(name))
                filled[attr] |= 1 << int(idx)

    for attr, mask in filled.items():
        missing = [i for i in range(cfg.num_layers) if not mask >> i & 1]
        assert not missing, f"{attr} 缺层 {missing[:5]}（共 {len(missing)}）"
    assert not remaining, f"checkpoint 里没找到: {sorted(remaining)[:5]}"
    if not quiet:
        # 生成器表达式里的 `p` 不会泄漏到函数作用域（py3），element_size 单独取一次。
        n = sum(t.numel() for t in own.values())
        print(f"[load_dit] {dtype} params={n / 1e9:.2f}B "
              f"bytes={n * next(iter(own.values())).element_size() / 1e9:.2f}GB "
              f"cfg: layers={cfg.num_layers} inner={cfg.inner_dim} mlp={cfg.inner_dim * cfg.mlp_ratio}")
    return model


def build_graphs(cfg: DiTConfig, weights: DiTWeights, mode: str):
    """构造推理/导出用的图。`prepare_for_export`（折 rope 置换 + 拆逐层 list）是幂等的，
    在这里统一调，调用方不需要知道 load 之后还差哪两步。"""
    from qi21_dit import DecodeGraph, PrefillGraph
    weights.prepare_for_export()
    return (PrefillGraph if mode == "prefill" else DecodeGraph)(cfg, weights)
