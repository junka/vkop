"""校验 qi21_dit.DiTWeights 的参数名/形状与 checkpoint 一致（不依赖 diffusers）。

safetensors 的 header 是文件开头的 JSON（前 8 字节是 u64 长度），所以只读几百字节就能
拿到全部张量的 name/shape/dtype —— 10GB 的权重还没下完就能做这个校验。

    python3 check_weights.py [shard ...]     # 不给参数则自动找 ModelScope 缓存

wrapper 把 32 层的同类权重堆成一个 (L, out, in) 参数，所以这里逐层比对每张 shard 的
第 i 层形状，再和堆叠后的期望形状对齐。
"""

import json
import struct
import sys
from collections import defaultdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from qi21_dit import DiTConfig, DiTWeights  # noqa: E402

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
DTYPES = {"F32": torch.float32, "F16": torch.float16, "BF16": torch.bfloat16}


def read_header(path: Path) -> dict:
    with path.open("rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
    header.pop("__metadata__", None)
    return header


def main() -> int:
    shards = [Path(p) for p in sys.argv[1:]] or sorted((MS_ROOT / "transformer").glob("*.safetensors"))
    if not shards:
        print("no safetensors found; 权重下载还没到位", file=sys.stderr)
        return 1

    raw = json.loads((MS_ROOT / "transformer" / "config.json").read_text())
    cfg = DiTConfig(num_layers=raw["num_layers"], heads=raw["num_attention_heads"],
                    head_dim=raw["attention_head_dim"], mlp_ratio=raw["mlp_ratio"],
                    axes_dims_rope=tuple(raw["axes_dims_rope"]), eps=raw["eps"],
                    context_in_dim=raw["context_in_dim"], in_channels=raw["in_channels"])
    print(f"[cfg] layers={cfg.num_layers} inner={cfg.inner_dim} "
          f"mlp={cfg.inner_dim * cfg.mlp_ratio} axes={cfg.axes_dims_rope} eps={cfg.eps}")

    per_layer = defaultdict(set)  # attr -> {shape}
    per_layer_dt = defaultdict(set)
    scalars = {}
    unmapped, count = [], defaultdict(int)
    for shard in shards:
        header = read_header(shard)
        print(f"[header] {shard.name}: {len(header)} tensors")
        for name, spec in header.items():
            count[name] = count[name] + 1
            if name in SCALARS:
                scalars[SCALARS[name]] = (tuple(spec["shape"]), DTYPES[spec["dtype"]])
                continue
            prefix = "transformer_blocks."
            if name.startswith(prefix):
                rest = name[len(prefix):]
                idx, _, tail = rest.partition(".")
                if tail in LAYERS:
                    per_layer[LAYERS[tail]].add(tuple(spec["shape"]))
                    per_layer_dt[LAYERS[tail]].add(DTYPES[spec["dtype"]])
                    continue
            unmapped.append(name)

    dup = [n for n, c in count.items() if c > 1]
    print(f"[tensors] total={len(count)} duplicated_across_shards={dup or 'none'}")
    print(f"[unmapped] {unmapped or 'none'}")

    own = dict(DiTWeights(cfg).named_parameters())
    bad = 0

    def check(label, got, want):
        nonlocal bad
        ok = tuple(got) == tuple(want)
        bad += 0 if ok else 1
        print(f"  {'OK  ' if ok else 'BAD '}{label}: wrapper={tuple(got)} ckpt={tuple(want)}")

    print("[scalar 权重]")
    for attr, (shape, dtype) in sorted(scalars.items()):
        check(f"{attr} {dtype}", own[attr].shape, shape)
        if own[attr].dtype != torch.float32:
            print(f"  BAD {attr}: wrapper dtype={own[attr].dtype}（应 fp32，加载时 cast）")
            bad += 1

    print("[逐层堆叠权重]")
    for attr, shapes in sorted(per_layer.items()):
        dtype = sorted(str(d) for d in per_layer_dt[attr])[0]
        if len(shapes) != 1:
            print(f"  BAD {attr}: checkpoint 里层间形状不一致 {shapes}")
            bad += 1
            continue
        if len(per_layer_dt[attr]) != 1:
            print(f"  BAD {attr}: 层间 dtype 不一致 {per_layer_dt[attr]}")
            bad += 1
        one = next(iter(shapes))
        check(f"{attr} ({dtype})", own[attr].shape, (cfg.num_layers, *one))

    ckpt_attrs = set(scalars) | set(per_layer)
    missing = ckpt_attrs - set(own)
    extra = set(own) - ckpt_attrs
    print(f"[coverage] ckpt attrs={len(scalars) + len(per_layer)} wrapper attrs={len(own)}")
    print(f"  missing in wrapper: {sorted(missing) or 'none'}")
    print(f"  extra  in wrapper : {sorted(extra) or 'none'}")
    print(f"\n[verdict] {'PASS' if bad == 0 else f'{bad} MISMATCH'}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
