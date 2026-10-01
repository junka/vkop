"""ORT 参考逐张量对齐：wrapper(torch) 的输出 vs 导出的 ONNX 在 onnxruntime 里跑的结果。

这是 llm/exporter/tests/test_numeric.py 那一层的检查，区别只在参考物：
`tests/test_wrapper_vs_diffusers.py` 证明的是"**wrapper 写的对**"（对上游 diffusers
逐张量），本文件证明的是"**导出的对**"（torch trace -> ONNX -> ORT 这条链没有把语义改
掉：dtype 提升、bias 加法、KV 拼接、rope 表广播、死代码剪枝后的输出顺序）。

小 config（2 层）自己导出到临时目录，不依赖磁盘上的 `dit_*_tiny.onnx` 是不是新鲜的；
`tiny_weights()` 固定 seed，所以三方（wrapper / 图初始值 / ORT 输入）用的是同一份权重。

    /Users/doudou/qi21-env/bin/python tests/test_ort_vs_wrapper.py
"""

import sys
import tempfile
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import qi21_dit as M  # noqa: E402
from qi21_export_onnx import export_pair, tiny_weights  # noqa: E402

S_P, S_T = 128, 256
BUDGET = 2e-2  # 与验收口径一致（fp16 图 + fp16 权重，均值界）


def report(name, mine, ort_out):
    a = np.asarray(mine.detach().cpu().numpy(), dtype=np.float32)
    b = np.asarray(ort_out, dtype=np.float32)
    assert a.shape == b.shape, (name, a.shape, b.shape)
    d = np.abs(a - b)
    rel = d.max() / max(np.abs(b).max(), 1e-9)
    ok = d.mean() < BUDGET
    print(f"{'OK ' if ok else 'BAD'} {name:<24} max={d.max():.3e} "
          f"mean={d.mean():.3e} rel={rel:.3e}")
    assert ok, name


def main():
    torch.set_grad_enabled(False)
    cfg, w = tiny_weights(torch.float16)
    n, hd = cfg.num_layers, cfg.head_dim
    dt = w.wq.dtype
    perm = M.rope_fold_perm(hd)

    # 与导出时同一份权重：fold 折进 wq/wk/nq/nk，swap_stack 换成逐层 list。
    w.fold_rope_permutation()
    w.swap_stack()

    lh = lw = int(S_T ** 0.5)
    cos, sin = M.joint_rope_positions(S_P, lh, lw, cfg.axes_dims_rope, hd)
    cos_p, sin_p = (t[:S_P].contiguous().to(dt) for t in (cos, sin))
    cos_t, sin_t = (t[S_P:].contiguous().to(dt) for t in (cos, sin))
    bias_p = torch.zeros(1, 1, S_P, S_P, dtype=dt).masked_fill(
        torch.triu(torch.ones(S_P, S_P, dtype=torch.bool), 1), M.FP16_MIN)
    bias_t = torch.zeros(1, 1, S_T, S_P + S_T, dtype=dt)

    pe = torch.randn(1, S_P, cfg.context_in_dim, dtype=dt)
    lat = torch.randn(1, S_T, cfg.in_channels, dtype=dt)
    timestep = torch.tensor([0.5], dtype=torch.float32)

    # ---- 参考：wrapper 自己跑 ----
    kv = M.PrefillGraph(cfg, w)(pe, cos_p, sin_p, torch.zeros(1), bias_p)
    sample = M.DecodeGraph(cfg, w)(lat, timestep, cos_t, sin_t, *kv, bias_t)

    # ---- 被测：导出的 ONNX 交给 ORT ----
    with tempfile.TemporaryDirectory() as td:
        made = export_pair(cfg, w, Path(td), S_P, S_T)
        paths = {mode: path for mode, path, _loc, _m in made}
        # 导出走的是 fold+swap 之后的同一份权重；临时目录里的图就是被测对象。
        so = ort.InferenceSession(str(paths["prefill"]), providers=["CPUExecutionProvider"])
        got_kv = so.run(None, {
            "prompt_embeds": pe.numpy(), "cos": cos_p.numpy(), "sin": sin_p.numpy(),
            "timestep_zero": np.zeros(1, dtype=np.float32),
            "attention_bias": bias_p.numpy()})
        report("prefill present_kv_0", kv[0], got_kv[0])
        report("prefill present_kv_1", kv[1], got_kv[1])
        # K 走的是折进权重的那个 head_dim 置换；ORT 与 wrapper 用同一份折过的权重，
        # 所以这里是逐位比，不需要再还原 perm（还原只在和 diffusers 对时才需要）。
        for i in range(n):
            assert got_kv[i].shape == kv[i].shape, (i, got_kv[i].shape)

        sd = ort.InferenceSession(str(paths["decode"]), providers=["CPUExecutionProvider"])
        feed = {"target_latents": lat.numpy(), "timestep": timestep.numpy(),
                "cos": cos_t.numpy(), "sin": sin_t.numpy(), "attention_bias": bias_t.numpy()}
        feed.update({f"past_kv_{i}": kv[i].numpy() for i in range(n)})
        got = sd.run(None, feed)[0]
        report("decode sample", sample, got)

        # 绑定契约：驱动侧按这些名字喂/取，改名就会在这里先响。
        contract = {
            "prefill": (["prompt_embeds", "cos", "sin", "timestep_zero", "attention_bias"],
                        [f"present_kv_{i}" for i in range(n)], "prefix_len"),
            "decode": (["target_latents", "timestep", "cos", "sin"] +
                       [f"past_kv_{i}" for i in range(n)] + ["attention_bias"],
                       ["sample"], "target_len"),
        }
        for sess, (want_in, want_out, seq_sym) in (
                (so, contract["prefill"]), (sd, contract["decode"])):
            got_in = [i.name for i in sess.get_inputs()]
            got_out = [o.name for o in sess.get_outputs()]
            assert got_in == want_in, (got_in, want_in)
            assert got_out == want_out, (got_out, want_out)
            # rope 表的第 0 轴必须是**符号**序列长度（不是 trace 时的样例 128/256），
            # 否则换分辨率就得重导一次图。
            dims = {i.name: i.shape for i in sess.get_inputs()}
            assert dims["cos"] == [seq_sym, cfg.head_dim], dims["cos"]
        print("[ok] 输入/输出名与动态符号轴符合驱动契约")
    print("[verdict] PASS")


if __name__ == "__main__":
    main()
