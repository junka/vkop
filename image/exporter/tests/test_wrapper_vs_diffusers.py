"""用 diffusers 上游实现本身当参考，逐张量比对 wrapper 的 prefill+decode 两段。

参考实现是 `reference/qi21_transformer.py` —— 上游 `transformer_qwenimage21.py` 的原样副本，
只把包内相对导入改写成绝对导入（`sed` 由 `reference/make_reference.py` 做），所以它不是我
对它的复述，改错了 forward 也会被发现。

不需要 checkpoint：`QwenImage21Transformer2DModel` 用小 config（2 层 / 2 head / 真 head_dim
128 / 真 axes_dims_rope）实例化，权重由 `qi21_dit.DiTWeights` 灌进去，CPU fp32 就能跑完
整等价性检查 —— rope 布局、共享 modulation 的四段切分、block-causal bias、KV 拼接、
norm_out/proj_out 一次全覆盖。

    /Users/doudou/qi21-env/bin/python tests/test_wrapper_vs_diffusers.py [--fp16]
"""

import argparse
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / "reference"))

import qi21_dit as M  # noqa: E402
import qi21_transformer as R  # noqa: E402

CFG = dict(num_layers=2, heads=2, head_dim=128, mlp_ratio=3,
           axes_dims_rope=(16, 56, 56), eps=1e-6, context_in_dim=96, in_channels=16)


def build(dtype):
    cfg = M.DiTConfig(**CFG)
    w = M.DiTWeights(cfg).eval()
    torch.manual_seed(0)
    with torch.no_grad():
        for _, p in w.named_parameters():
            # norm 权重取 1 附近、其余取小方差，免得随机深度 2 层就数值爆炸。
            if p.ndim == 1 and p.numel() == cfg.head_dim:
                p.normal_(1.0, 0.05)
            elif p.ndim == 1:
                p.normal_(0.0, 0.05)
            else:
                p.normal_(0.0, 0.02)
    ref = R.QwenImage21Transformer2DModel(
        patch_size=1, in_channels=cfg.in_channels, out_channels=cfg.in_channels,
        num_layers=cfg.num_layers, attention_head_dim=cfg.head_dim,
        num_attention_heads=cfg.heads, context_in_dim=cfg.context_in_dim,
        mlp_ratio=cfg.mlp_ratio, axes_dims_rope=list(cfg.axes_dims_rope),
        eps=cfg.eps, causal_condition=True)
    ref.eval()
    src = dict(w.named_parameters())
    n = cfg.num_layers
    put = {
        "img_in.weight": src["img_in_w"], "txt_in.text_norm.weight": src["txt_norm_w"],
        "txt_in.in_layer.weight": src["txt_in_w"], "txt_in.out_layer.weight": src["txt_out_w"],
        "time_text_embed.timestep_embedder.linear_1.weight": src["tl1_w"],
        "time_text_embed.timestep_embedder.linear_2.weight": src["tl2_w"],
        "modulation.1.weight": src["mod_w"], "norm_out.linear.weight": src["out_lin_w"],
        "proj_out.weight": src["proj_out_w"],
    }
    for i in range(n):
        p = f"transformer_blocks.{i}."
        for tail, attr in (("attn.to_q.weight", "wq"), ("attn.to_k.weight", "wk"),
                           ("attn.to_v.weight", "wv"), ("attn.to_out.0.weight", "wo"),
                           ("attn.norm_q.weight", "nq"), ("attn.norm_k.weight", "nk"),
                           ("img_mlp.gate_layer.weight", "gate"), ("img_mlp.proj.weight", "up"),
                           ("img_mlp.out.weight", "down")):
            put[p + tail] = src[attr][i]
    ref.load_state_dict({k: v for k, v in put.items()}, strict=False)
    # strict=False 是因为参考模型还留着 checkpoint 里没有的东西；反过来必须确认要的都灌上了。
    missing = [k for k in put if k not in dict(ref.named_parameters())]
    assert not missing, missing
    # 两边从同一份 fp32 权重出发，再各自降到目标精度
    return cfg, w.to(dtype), ref.to(dtype)


def report(name, mine, ref_t, budget):
    a, b = mine.detach().float(), ref_t.detach().float()
    d = (a - b).abs()
    rel = float(d.max()) / max(float(b.abs().max()), 1e-9)
    ok = float(d.mean()) < budget
    print(f"{'OK ' if ok else 'BAD'} {name:<26} max={float(d.max()):.3e} "
          f"mean={float(d.mean()):.3e} rel={rel:.3e}")
    assert ok, name
    return float(d.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fp16", action="store_true")
    args = ap.parse_args()
    dtype = torch.float16 if args.fp16 else torch.float32
    torch.set_grad_enabled(False)

    cfg, w, ref = build(dtype)
    S_p, lh, lw = 33, 4, 5
    S_t = lh * lw
    # 无条件图的纯 t2i：img_mask = 文本段全 False + 目标图 S_t/4 个 slot 全 True
    assert S_t % M.IMG_TOKENS_PER_SLOT == 0
    img_mask = torch.cat([torch.zeros(S_p, dtype=torch.bool),
                          torch.ones(S_t // M.IMG_TOKENS_PER_SLOT, dtype=torch.bool)])[None]
    pe = torch.randn(1, S_p, cfg.context_in_dim).to(dtype)
    lat = torch.randn(1, S_t, cfg.in_channels).to(dtype)
    t = torch.tensor([0.37], dtype=torch.float32)
    img_shapes = [[(1, lh, lw)]]

    # 参考：整条联合序列一次跑完（不开 kv cache），这就是"正确答案"。
    # `norm_out`/`proj_out` 是作用在整段联合序列上的，所以输出有 S_p+S_t 个 token，
    # 前缀那部分是文本流的去噪值、pipeline 会丢掉，只比最后 S_t 个目标 token。
    ref_sample = ref(hidden_states=lat, encoder_hidden_states=pe, timestep=t,
                     img_shapes=img_shapes, img_mask=img_mask,
                     return_dict=False)[0][:, S_p:]

    # wrapper：prefill(prefix) + decode(target) 两段拼起来
    cos_p, sin_p = (c.to(dtype) for c in M.joint_rope_positions(S_p, 1, 1, cfg.axes_dims_rope,
                                                                cfg.head_dim))
    cos_t, sin_t = M.joint_rope_positions(S_p, lh, lw, cfg.axes_dims_rope, cfg.head_dim)
    # joint_rope_positions 的前 S_p 行是文本段、后 S_t 行是目标图块
    cos_p, sin_p = cos_p[:S_p].contiguous(), sin_p[:S_p].contiguous()
    cos_t, sin_t = cos_t[S_p:].contiguous(), sin_t[S_p:].contiguous()
    bias_p = torch.zeros(1, 1, S_p, S_p, dtype=dtype).masked_fill(
        torch.triu(torch.ones(S_p, S_p, dtype=torch.bool), 1), M.FP16_MIN)
    # 目标图块对 prefix 全可见（因果位置都比它小），块内双向 => 只有 padding 才需要 mask
    bias_t = torch.zeros(1, 1, S_t, S_p + S_t, dtype=dtype)

    w.fold_rope_permutation()
    w.swap_stack()
    kv = M.PrefillGraph(cfg, w)(pe, cos_p, sin_p, torch.zeros(1), bias_p)
    mine = M.DecodeGraph(cfg, w)(lat, t, cos_t, sin_t, *kv, bias_t).float()

    budget = 2e-2 if args.fp16 else 1e-5
    report("sample (prefill+decode vs 上游整跑)", mine, ref_sample.float(), budget)

    # 中间量：逐层 prefix K/V。参考用上游自带的 kv_cache（extract 模式）取。
    # K 的比对口径要注意：`fold_rope_permutation` 把 head_dim 轴的一个固定置换折进了
    # wq/wk/nq/nk，所以 present_k 落在**置换后的基**里。q·k 对 q/k 的同一置换是不变的
    # （上面的 sample 已经全对），但裸 K 必须按同一个 perm 折一遍才能比；V 不过 rope，
    # 没有这个置换，直接比。
    perm = M.rope_fold_perm(cfg.head_dim)
    cache = R.QwenImage21KVCache(cfg.num_layers)
    ref(hidden_states=lat, encoder_hidden_states=pe, timestep=t, img_shapes=img_shapes,
        img_mask=img_mask, kv_cache=cache, kv_cache_mode="extract", return_dict=False)
    for i in range(cfg.num_layers):
        layer = cache.get_layer(i)
        k_raw = layer.k.transpose(1, 2)  # (B,S,H,D) -> (B,H,S,D)
        v_ref = layer.v.transpose(1, 2)
        report(f"layer{i} present_k", kv[i][:, 0], k_raw[..., perm], budget)
        report(f"layer{i} present_v", kv[i][:, 1], v_ref, budget)
        # 反向兜底：置换必须真的动了 K，否则上面那条 OK 也可能是 perm 退化成恒等骗出来的。
        assert float((kv[i][:, 0] - k_raw).abs().max()) > 0.1 * float(k_raw.abs().max())
    print("[verdict] PASS")


if __name__ == "__main__":
    main()
