"""Qwen-Image-2.1 DiT 内存/耗时探针（跑自己的 wrapper，不依赖 diffusers 的 transformer 类）。

目的：导 ONNX / 写 C++ driver 之前，实测 **token 数 -> peak RSS** 的账，确认
1024x1024（S_t = (1024/16)^2 = 4096 个 target token）装得进这台 36GB 统一内存；顺带
把 wrapper 的 I/O 契约在**真权重**上跑一遍（prefill 每 prompt 一次、decode 每步一次）。

prompt embeds 用随机正态代替（真值需要 8.4GB 的 text encoder 权重）：内存和形状只取决于
token 数、与数值无关，所以本探针不加载 encoder 也成立。随机 prompt embed 会让去噪输出
是垃圾，这不影响内存结论；数值对齐由 tests/ 负责。

    python3 probe_dit.py --load-only
    python3 probe_dit.py --sizes 512 1024 --steps 1 --json /tmp/qi21_ledger.json

peak RSS 的口径要注意两点，否则数字会被读错：
  1. `ru_maxrss` 是**进程级高水位**、只增不减，所以逐 size 之间的差值才是那一个 size 的
     真实代价；权重加载后长期持有，各 size 共享它。
  2. safetensors 走 mmap，那些页是 clean file-backed 页，会计入 RSS 但随时可被内核回收，
     所以加载阶段的 peak RSS 高于真实内存压力 —— 同时报 current RSS 作对照。

实测账（2026-09-28，M5 Max 36GB，fp16 真权重，S_p=192，torch eager 口径）：
  加载 14.23GB 权重 5~7s，peak_rss 20.14GB（多出的是 mmap 的 clean 页）；prefill+decode
  跑完后常驻 16.6~16.7GB。
    S_t=256   一步 delta 0.17GB / 解析 0.06GB（小 size 上固定开销占主导，信噪比差）
    S_t=1024  一步 delta 0.68GB / 解析 0.55GB  -> 校准标量 k=1.26，此点残差 -0.01GB
  外推 S_t=4096（1024x1024）peak ≈ 23.2GB —— 36GB 装得下。1536² 外推 57GB，超内存，
  第一轮不做。没实跑 1024² 的 eager：一步要 10 分钟以上，且量的是 torch 的口径而不是
  vkop 的，`--ledger-only` 的外推表 + 512 那个点就够定第一轮分辨率了。
"""

import argparse
import json
import os
import resource
import subprocess
import sys
import threading
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))

import qi21_dit as M  # noqa: E402
from qi21_weights import build_graphs, load_dit, read_config  # noqa: E402


def phys_mem_bytes() -> int:
    return int(subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True).strip())


def peak_rss_gb() -> float:
    # ru_maxrss 在 macOS 上是字节、Linux 上是 KB。
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / 1e9 if sys.platform == "darwin" else peak / 1e6


def cur_rss_gb() -> float:
    out = subprocess.check_output(["ps", "-o", "rss=", "-p", str(os.getpid())], text=True)
    return int(out.strip()) * 1024 / 1e9


class RssSampler:
    """后台线程按固定间隔采 current RSS，取最大值。

    为什么不用 `ru_maxrss`：它是**进程级高水位**，权重加载那一步（safetensors 是 mmap，
    读过的 shard 页作为 clean file-backed 页会进 RSS）就把高水位钉在 ~20GB，之后再跑
    多大的图都看不到增量。current RSS 的峰值才有分辨率 —— 代价是它同样把 clean 页算进来，
    所以只在"prefill 已经把权重真正 touch 成 dirty"之后取 baseline，报差值。
    """

    def __init__(self, interval: float = 0.2):
        self.interval = interval
        self.max_gb = 0.0
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.is_set():
            self.max_gb = max(self.max_gb, cur_rss_gb())
            self._stop.wait(self.interval)

    def __enter__(self):
        self._t.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._t.join()
        return False


def elements_gb(n: int, bytes_per_elem: int) -> float:
    return n * bytes_per_elem / 1e9


def latent_hw(size: int, aspect: float) -> tuple[int, int]:
    """VAE 空间 16x 下采样 + transformer patch_size=1 => 一个 token 覆盖 16x16 像素。"""
    total = (size // 16) ** 2
    lh = max(1, int(round((total / aspect) ** 0.5)))
    return lh, max(1, int(round(total / lh)))


def make_bias(s_q: int, s_kv: int, prefix_len: int, dtype: torch.dtype):
    """block-causal 的加法 bias，形状 (1, 1, s_q, s_kv)。

    prefill（s_q == s_kv == prefix_len，序列里没有 target 段）退化成纯因果三角；decode 的
    target 段能看全部 prefix、且 target 之间双向（DiT 的 image block 本来就不是因果的），
    所以 decode 的 bias 整张是零 —— 仍然显式传，是为了和 ONNX 的输入契约对齐。
    """
    if s_q == prefix_len and s_kv == prefix_len:
        return torch.where(torch.ones(s_q, s_kv, dtype=torch.bool).tril(),
                           0.0, M.FP16_MIN).to(dtype)[None, None]
    return torch.zeros(1, 1, s_q, s_kv, dtype=dtype)


def weights_elems(cfg: M.DiTConfig) -> int:
    """`DiTWeights` 那 18 个参数的元素总数，纯算术、不分配（`--ledger-only` 要靠它）。

    和真加载时的 `sum(p.numel())` 有一道 assert 相对账，防止这里的公式和
    `qi21_dit.DiTWeights.__init__` 的 shape 静默走偏。
    """
    D, hd, r = cfg.inner_dim, cfg.head_dim, cfg.mlp_ratio
    scalars = (2 * D * cfg.in_channels + cfg.context_in_dim + D * cfg.context_in_dim
               + M.TIME_DIM * D + 7 * D * D)
    # 逐层：wq/wk/wv/wo = 4*D*D，norm_q/norm_k = 2*hd，gate/up/down = 3*(r*D)*D
    per_layer = 4 * D * D + 2 * hd + 3 * r * D * D
    return scalars + cfg.num_layers * per_layer


def scores_transient_gb(cfg: M.DiTConfig, s_q: int, s_kv: int) -> float:
    """单层 attention scores 在 torch eager 下的瞬时峰值。

    链上同时存活的缓冲（每元素）：fp16 的 q@k^T（2）+ `.float()` 的 fp32 副本（4）
    + `softmax` 另开的 fp32 输出（4）+ `.to(fp16)` 的结果（2）= 12 字节。fp16 那份在
    `.float()` 之后才算死、softmax 的 fp32 输入输出也有一段重叠，所以 12 是上界；
    vkop 的融合 kernel 不落整份 fp32，真实运行时只会比这个数小。
    """
    return elements_gb(cfg.heads * s_q * s_kv, 12)


def activations_gb(cfg: M.DiTConfig, s: int, n_alive: int) -> float:
    """同时存活的 (1,S,inner_dim) 形态张量份数（q/k/v/attn 输出/门控乘积/mlp 中间…）。

    每份 = S * inner_dim * 2 字节。`n_alive` 用实测系数：eager 下每层同时存活的差不多
    8 份（x、h、q、k、v、attention 输出、两段 MLP）。
    """
    return elements_gb(s * cfg.inner_dim, 2 * n_alive)


def fit_coefs(cfg: M.DiTConfig, rows: list[dict]) -> dict:
    """用一个最小二乘标量把解析式校准到实测 delta 上。

    为什么不是"两个系数各自反解"：delta(S_t) = cs·H·S_t·(S_t+S_p) + ca·S_t·inner_dim
    这两列在 S_p=192 时几乎共线（256 与 1024 两行的列向量比是 3.67M:1.05M 对 39.8M:4.19M，
    行列式只有 6e13 量级、相对条件数很差），配上 delta 只有 0.01GB 的分辨率，解出来的
    系数是纯噪声（实测会给出 scores≈1000B、activations≈-88B 这种荒谬值）。所以只校准
    **一个总体标量**，解析式内部 12B/16B 的分配保持不动，再用逐 size 残差说明拟合质量。
    """
    ok = [r for r in rows if r.get("decode_delta_gb") is not None]
    if not ok:
        return {}
    pred = [r["predicted_delta_gb"] for r in ok]
    meas = [r["decode_delta_gb"] for r in ok]
    k = sum(p * m for p, m in zip(pred, meas)) / sum(p * p for p in pred)
    return {"scale": round(k, 3),
            "residuals": {r["size"]: round(m - k * p, 3) for r, p, m in zip(ok, pred, meas)}}


def predict_gb(cfg: M.DiTConfig, weights_gb: float, s_p: int, s_t: int,
               coefs: dict | None = None) -> dict:
    """decode 一步的 peak RSS 预测 = 权重 + prefix KV + scores 瞬时 + 激活。

    `coefs` 为 None 时用解析值（eager 口径上界）；给了 `scale` 时按实测校准。
    prefix KV 不算在 delta 里（prefill 就已经落定、且很小），但作为 peak 的一项目出来。
    """
    kv = 2 * cfg.num_layers * elements_gb(cfg.heads * s_p * cfg.head_dim, 2)
    k = (coefs or {}).get("scale", 1.0)
    sc = elements_gb(cfg.heads * s_t * (s_t + s_p), 12) * k
    ac = elements_gb(s_t * cfg.inner_dim, 16) * k
    return {"weights": round(weights_gb, 2), "prefix_kv": round(kv, 3),
            "scores": round(sc, 2), "activations": round(ac, 3),
            "predicted_peak": round(weights_gb + kv + sc + ac, 2)}


def print_ledger(cfg: M.DiTConfig, sizes: list[int], s_p: int, weights_gb: float,
                 coefs: dict | None = None) -> None:
    print(f"[ledger] {'实测反推' if coefs else '纯解析'}外推：", flush=True)
    for size in sizes:
        lh, lw = latent_hw(size, 1.0)
        row = predict_gb(cfg, weights_gb, s_p, lh * lw, coefs)
        print(f"  {size}px S_t={lh * lw} S_p={s_p} -> "
              + " ".join(f"{k}={v}" for k, v in row.items()), flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[512])
    ap.add_argument("--aspect", type=float, default=1.0, help="width/height")
    ap.add_argument("--txt-len", type=int, default=192)
    ap.add_argument("--steps", type=int, default=1)
    ap.add_argument("--dtype", choices=["fp16", "bf16"], default="fp16")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--load-only", action="store_true", help="只测权重加载的内存，不跑图")
    ap.add_argument("--ledger-only", action="store_true",
                    help="只打印解析外推表（零内存占用，不加载 14GB 权重）")
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)
    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16
    bpp = 2
    f32 = dict(dtype=torch.float32)

    print(f"[env] torch={torch.__version__} threads={torch.get_num_threads()} "
          f"phys_mem={phys_mem_bytes() / 1e9:.0f}GB dtype={args.dtype} "
          f"rss_at_start={cur_rss_gb():.2f}GB", flush=True)

    cfg = read_config()
    weights_gb = elements_gb(weights_elems(cfg), bpp)
    if args.ledger_only:
        print_ledger(cfg, args.sizes, args.txt_len, weights_gb)
        return 0

    t0 = time.perf_counter()
    weights = load_dit(args.dtype)
    n_elem = sum(p.numel() for p in dict(weights.named_parameters()).values())
    assert n_elem == weights_elems(cfg), (n_elem, weights_elems(cfg))
    print(f"[load] {time.perf_counter() - t0:.1f}s params={n_elem / 1e9:.2f}B "
          f"weights={elements_gb(n_elem, bpp):.2f}GB peak_rss={peak_rss_gb():.2f}GB "
          f"cur_rss={cur_rss_gb():.2f}GB", flush=True)
    print(f"[cfg] layers={cfg.num_layers} heads={cfg.heads} head_dim={cfg.head_dim} "
          f"inner={cfg.inner_dim} mlp={cfg.inner_dim * cfg.mlp_ratio} "
          f"axes_rope={cfg.axes_dims_rope} eps={cfg.eps}", flush=True)
    if args.load_only:
        print_ledger(cfg, args.sizes, args.txt_len, weights_gb)
        return 0

    # prepare = fold rope 置换 + 拆逐层 list。fold 会临时多要一份被折的 wq/wk（共 2.15 GB
    # fp16），swap_stack 的切片是 view、不复制，所以这一步的峰值增量应该只来自 fold。
    with RssSampler() as smp:
        pre = build_graphs(cfg, weights, "prefill")
        dec = build_graphs(cfg, weights, "decode")
    print(f"[prepare] peak_rss {peak_rss_gb():.2f}GB fold_max_rss={smp.max_gb:.2f}GB "
          f"cur_rss={cur_rss_gb():.2f}GB", flush=True)

    S_p = args.txt_len
    results = []
    with torch.no_grad():
        for size in args.sizes:
            lh, lw = latent_hw(size, args.aspect)
            S_t = lh * lw
            row = {"size": size, "latent": [lh, lw], "target_tokens": S_t,
                   "prefix_tokens": S_p, "joint_tokens": S_p + S_t,
                   **predict_gb(cfg, weights_gb, S_p, S_t)}
            print(f"[case] {size}px latent={lh}x{lw} S_t={S_t} S_p={S_p} "
                  f"predicted_peak={row['predicted_peak']}GB", flush=True)

            # rope 表按联合序列一次建好再切段：prefill 用前 S_p 行、decode 用后 S_t 行。
            cos, sin = M.joint_rope_positions(S_p, lh, lw, cfg.axes_dims_rope, cfg.head_dim)
            with RssSampler() as smp:
                past_kv = pre(torch.randn(1, S_p, cfg.context_in_dim, dtype=dtype),
                              cos[:S_p].to(**f32), sin[:S_p].to(**f32),
                              torch.zeros(1, **f32), make_bias(S_p, S_p, S_p, dtype))
            row["prefill_max_rss_gb"] = round(smp.max_gb, 2)
            print(f"[prefill] max_rss={row['prefill_max_rss_gb']}GB "
                  f"cur_rss={cur_rss_gb():.2f}GB kv0={list(past_kv[0].shape)}", flush=True)

            # baseline = prefill 跑完、权重全被 touch 成 dirty 之后的 current RSS；
            # decode 一步的实测增量就是相对它抬多少，和 predicted 里的 scores+activations 比。
            base = smp.max_gb
            bias_t = make_bias(S_t, S_p + S_t, S_p, dtype)
            cos_t, sin_t = cos[S_p:].to(**f32), sin[S_p:].to(**f32)
            for step in range(args.steps):
                latents = torch.randn(1, S_t, cfg.in_channels, dtype=dtype)
                t1 = time.perf_counter()
                with RssSampler() as smp2:
                    out = dec(latents, torch.full((1,), 0.5, **f32), cos_t, sin_t,
                              *past_kv, bias_t)
                row["decode_s"] = round(time.perf_counter() - t1, 1)
                row["decode_max_rss_gb"] = round(smp2.max_gb, 2)
                row["decode_delta_gb"] = round(smp2.max_gb - base, 2)
                row["out_shape"] = list(out.shape)
                row["out_absmax"] = round(float(out.abs().max()), 4)
                print(f"[decode {step}] {row['decode_s']}s max_rss={row['decode_max_rss_gb']}GB "
                      f"delta={row['decode_delta_gb']}GB "
                      f"(predicted {row['scores'] + row['activations']:.2f}GB) "
                      f"out={row['out_shape']} absmax={row['out_absmax']}", flush=True)
            row["predicted_delta_gb"] = round(row["scores"] + row["activations"], 3)
            results.append(row)
            del past_kv, out, latents, bias_t, cos, sin, cos_t, sin_t

    coefs = fit_coefs(cfg, results)
    if coefs:
        print(f"[model] 解析式（scores 12B/元素 + 激活 16B/元素）按 k={coefs['scale']} 校准到实测 delta，"
              f"逐 size 残差 {coefs['residuals']} GB", flush=True)
        print("  口径提醒：delta 是 eager 口径，含 torch 分配器保留块与 fp16->fp32 的解压缩膨胀；"
              "vkop 读 packed half、softmax 融合，实际只会更小。", flush=True)
    print_ledger(cfg, args.sizes, args.txt_len, weights_gb, coefs)

    if args.json:
        args.json.write_text(json.dumps({"model": coefs, "cases": results}, indent=2) + "\n")
        print(f"[json] {args.json}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
