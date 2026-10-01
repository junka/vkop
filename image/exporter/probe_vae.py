"""Qwen-Image-2.1 VAE decoder 的内存/耗时探针（跑上游 reference 实现，不带 wrapper）。

为什么先测再决定：DiT 那边写 wrapper 是因为上游实现里有四处**根本导不出去**的东西（python
KV cache 对象、flex_attention 的 BlockMask、`.tolist()`/复数 rope 表、布尔 scatter 赋值）。
VAE 上游这边看着温和得多：`CausalConv3d` 已经是图像特化（`nn.Conv2d` + squeeze 掉单帧的 T，
喂 cache 直接 raise），`feat_cache=None` 是一条干净的路径，`nearest-exact` 上采样和
`DupUp3D` 都是 reshape。所以**能不能直接把上游模块 trace 成 ONNX**是个真问题，而它的答案
取决于两件事：图里会不会被 trace 烘死分辨率，以及 1024x1024 的激活装不装得下 —— 后者就是
本探针的账。

口径和 `probe_dit.py` 一致：`ru_maxrss` 是进程级高水位、只增不减，逐 size 的差没有分辨率，
所以用后台线程采 current RSS 取峰值（`RssSampler`）。权重只有一份（fp32 1.35 GB），各 size
共享它，所以报"加载后 baseline"和"每个 size 的 peak 增量"。

    python3 probe_vae.py --sizes 512 1024 --dtype fp32
    python3 probe_vae.py --sizes 1024 --dtype fp16 --json /tmp/qi21_vae_ledger.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "reference"))

from probe_dit import RssSampler, cur_rss_gb  # noqa: E402

MS_ROOT = Path("/Users/doudou/qi21-env/ms/models/Qwen--Qwen-Image-2.1/snapshots/master")
SCALE = 16  # vae.config.scale_factor_spatial：latent 每边 1 格 = 图像 16x16 像素


def load_vae(dtype):
    from qi21_vae import AutoencoderKLQwenImage21
    vae = AutoencoderKLQwenImage21.from_pretrained(MS_ROOT, subfolder="vae")
    vae.eval()
    if dtype != "fp32":
        vae.to(torch.float16 if dtype == "fp16" else torch.bfloat16)
    return vae


def ledger(cfg, px):
    """按 decoder 的通道/分辨率表算"最宽一级有几个活张量"的解析账（单位：字节/元素）。

    dims = [1152, 1152, 1152, 576, 288, 144]（`decoder_base_dim=144` × `dim_mult`），
    分辨率从 px/16 逐级翻倍到 px。一个 `ResidualBlock` 里同时活着的是
    `h`（shortcut 输入）+ norm/silu 中间量 + conv1 输出 + conv2 输出，取 4 个同一尺寸的张量；
    `DupUp3D` 的 shortcut 会先在**时间轴上翻一倍**再切回单帧，所以它那份瞬时是 2 倍。
    这里只算最宽的一级（144/288 通道 @px），是个下界 —— 真实峰值还包含 allocator 的碎片。
    """
    ch = 288  # 最宽一级 up_blocks.4 的输入通道
    per_tensor = ch * px * px
    return {"widest_stage_bytes": 4 * per_tensor, "dupup_transient_bytes": 2 * ch * px * px}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[512, 1024])
    ap.add_argument("--dtype", default="fp32", choices=["fp32", "fp16", "bf16"])
    ap.add_argument("--threads", type=int, default=0, help="0 = 用 torch 默认")
    ap.add_argument("--json", type=str, default="")
    args = ap.parse_args()

    torch.set_grad_enabled(False)
    if args.threads:
        torch.set_num_threads(args.threads)

    t0 = time.time()
    vae = load_vae(args.dtype)
    el = cur_rss_gb()
    print(f"[load] {args.dtype} {time.time() - t0:.1f}s current RSS {el:.2f} GB "
          f"(z_dim={vae.config.z_dim} out_channels={vae.config.out_channels} "
          f"scale={vae.config.scale_factor_spatial})")
    base = el

    rows = []
    for px in args.sizes:
        assert px % (SCALE * 2) == 0, f"{px} 必须能被 {SCALE * 2} 整除（pipeline 的要求）"
        lh = px // SCALE
        z = torch.randn(1, vae.config.z_dim, 1, lh, lh,
                        dtype=torch.float16 if args.dtype != "fp32" else torch.float32)
        t0 = time.time()
        with RssSampler() as s:
            out = vae.decode(z, return_dict=False)[0]
            bytes_out = out.numel() * out.element_size()
            del out
        row = {"px": px, "latent": lh, "secs": round(time.time() - t0, 1),
               "peak_gb": round(s.max_gb, 2), "delta_gb": round(s.max_gb - base, 2),
               "out_bytes": bytes_out, **ledger(vae.config, px)}
        rows.append(row)
        print(f"[decode] {row}")
        del z
        base = max(base, cur_rss_gb())

    if args.json:
        Path(args.json).write_text(json.dumps({"dtype": args.dtype, "base_gb": round(base, 2),
                                               "rows": rows}, indent=1))
        print(f"[json] {args.json}")


if __name__ == "__main__":
    main()
