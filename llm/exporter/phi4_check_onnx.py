#!/usr/bin/env python3
"""Phi-4-mini 导出的 llm.onnx 与 HF 的数值对齐检查（CPU onnxruntime，fp16）。

三项对比，缺一不可：
  1. prefill logits       —— 端到端数值路径
  2. 每层 present_key_values —— 逐层定位（kv 依赖前面所有层，逐层比对等于逐层
     hidden 对齐，不需要给图加中间输出）
  3. decode 单步 logits    —— 带 past_kv 的第二条 shape 分支（q_len=1）

HF 参考用 model.model(...) + lm_head，喂 2D attention_mask；ONNX 侧喂显式的加法
causal bias。两者只在 sliding_window 不起作用时等价 —— Phi-4-mini 的
sliding_window=262144 > 任何上下文，所以窗口不裁剪，纯 causal。

用法:
    python3 phi4_check_onnx.py                    # 默认 L=32
    CHECK_LEN=64 python3 phi4_check_onnx.py
"""
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.environ.get("MODEL_PATH") or os.path.join(
    os.path.expanduser("~"), ".cache/modelscope/models/"
    "LLM-Research--Phi-4-mini-instruct/snapshots/master")
ONNX_PATH = os.environ.get("ONNX_PATH") or os.path.join(HERE, "text_phi4", "llm.onnx")
CHECK_LEN = int(os.environ.get("CHECK_LEN", "32"))

# fp16 + CPU ort 的累积舍入：量级 ~10 的 logits 用 mean<2e-2 判逻辑性偏差，
# kv 量级 ~1 收紧到 5e-3。逻辑错时 mean 通常大一个数量级。
LOGITS_MEAN_TOL = 2e-2
KV_MEAN_TOL = 5e-3


def report(name, a, b, mean_tol):
    """a=ONNX, b=HF。返回是否通过。"""
    a = a.float()
    b = b.float()
    if a.shape != b.shape:
        print(f"[FAIL] {name}: shape {tuple(a.shape)} vs {tuple(b.shape)}")
        return False
    diff = (a - b).abs()
    mean, mx = diff.mean().item(), diff.max().item()
    denom = b.abs().mean().item() + 1e-9
    ok = mean < mean_tol
    print(f"[{'ok ' if ok else 'FAIL'}] {name:28s} shape={tuple(a.shape)} "
          f"mean={mean:.3e} max={mx:.3e} rel={mean / denom:.3e}")
    return ok


def hf_cache_to_kv(cache, nlayers):
    out = []
    for i in range(nlayers):
        layer = cache.layers[i]
        out.append(torch.stack([layer.keys, layer.values], dim=1))
    return out


def ort_prefill(sess, embeds, pos, bias, kv_shape, nlayers):
    feed = {
        "inputs_embeds": embeds.numpy().astype(np.float16),
        "position_ids": pos.numpy().astype(np.int64),
        "attention_bias": bias.numpy().astype(np.float16),
    }
    empty = np.zeros(kv_shape, dtype=np.float16)
    for i in range(nlayers):
        feed[f"past_key_values_{i}"] = empty
    res = sess.run(None, feed)
    return torch.tensor(res[0]), [torch.tensor(res[i + 1]) for i in range(nlayers)]


def main():
    import onnxruntime as ort  # noqa: E402
    from transformers.models.phi3.modeling_phi3 import Phi3ForCausalLM  # noqa: E402

    print(f"[load] HF {MODEL_PATH}")
    model = Phi3ForCausalLM.from_pretrained(
        MODEL_PATH, attn_implementation="eager", torch_dtype=torch.float16,
        local_files_only=True)
    model.eval()
    # 关掉梯度：embeds / kv 要直接 .numpy() 喂 ORT
    model.requires_grad_(False)
    cfg = model.config
    nl, nkv, hd = cfg.num_hidden_layers, cfg.num_key_value_heads, (
        getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads)

    print(f"[load] ORT {ONNX_PATH}")
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(ONNX_PATH, so, providers=["CPUExecutionProvider"])
    print("[ort] inputs=" + ", ".join(i.name for i in sess.get_inputs()[:3]) +
          f" + past_key_values_0..{nl - 1}")

    L = CHECK_LEN
    gen = torch.Generator().manual_seed(1234)
    input_ids = torch.randint(0, cfg.vocab_size, (1, L), generator=gen)
    pos = torch.arange(L, dtype=torch.long).unsqueeze(0)
    emb = model.model.embed_tokens(input_ids)
    causal = torch.triu(
        torch.full((L, L), torch.finfo(torch.float16).min, dtype=torch.float16),
        diagonal=1).unsqueeze(0).unsqueeze(0)

    # ---------------- prefill ----------------
    with torch.no_grad():
        hf = model.model(inputs_embeds=emb, position_ids=pos,
                         attention_mask=torch.ones(1, L, dtype=torch.long),
                         past_key_values=None, use_cache=True)
        hf_logits = model.lm_head(hf.last_hidden_state)
    hf_kv = hf_cache_to_kv(hf.past_key_values, nl)
    onnx_logits, onnx_kv = ort_prefill(
        sess, emb, pos, causal, (1, 2, nkv, 0, hd), nl)

    print(f"\n[prefill] L={L}")
    ok = [report("logits", onnx_logits, hf_logits, LOGITS_MEAN_TOL)]
    arg_ok = torch.argmax(onnx_logits, -1).eq(torch.argmax(hf_logits, -1)).float().mean().item()
    print(f"[{'ok ' if arg_ok > 0.95 else 'FAIL'}] logits argmax 一致率 = {arg_ok:.3f}")
    ok.append(arg_ok > 0.95)

    # 逐层 kv：只打印最差的两层，全部数值参与判定
    worst = []
    for i in range(nl):
        d = (onnx_kv[i].float() - hf_kv[i].float()).abs().mean().item()
        worst.append((d, i))
    worst.sort(reverse=True)
    for d, i in worst[:2]:
        print(f"       layer {i:02d} present_kv mean={d:.3e}")
    kv_bad = [(i, d) for d, i in worst if d >= KV_MEAN_TOL]
    if kv_bad:
        print(f"[FAIL] {len(kv_bad)} 层 kv 均值差超阈值: {kv_bad[:5]}")
        ok.append(False)
    else:
        print(f"[ok ] 全部 {nl} 层 present_key_values 均值差 < {KV_MEAN_TOL}")
        ok.append(True)

    # ---------------- decode 单步 ----------------
    dl = L + 1
    dec_ids = torch.randint(0, cfg.vocab_size, (1, 1), generator=torch.Generator().manual_seed(7))
    dec_pos = torch.full((1, 1), L, dtype=torch.long)
    dec_emb = model.model.embed_tokens(dec_ids)
    dec_bias = torch.zeros(1, 1, 1, dl, dtype=torch.float16)
    with torch.no_grad():
        hf_dec = model.model(inputs_embeds=dec_emb, position_ids=dec_pos,
                             attention_mask=torch.ones(1, dl, dtype=torch.long),
                             past_key_values=hf.past_key_values, use_cache=True)
        hf_dec_logits = model.lm_head(hf_dec.last_hidden_state)
    feed = {
        "inputs_embeds": dec_emb.numpy().astype(np.float16),
        "position_ids": dec_pos.numpy().astype(np.int64),
        "attention_bias": dec_bias.numpy().astype(np.float16),
    }
    for i in range(nl):
        feed[f"past_key_values_{i}"] = hf_kv[i].numpy().astype(np.float16)
    res = sess.run(None, feed)
    onnx_dec = torch.tensor(res[0])
    print(f"\n[decode] q_len=1, kv_len={dl}")
    ok.append(report("logits", onnx_dec, hf_dec_logits, LOGITS_MEAN_TOL))
    same = torch.argmax(onnx_dec, -1).item() == torch.argmax(hf_dec_logits, -1).item()
    print(f"[{'ok ' if same else 'FAIL'}] decode argmax: onnx={torch.argmax(onnx_dec, -1).item()} "
          f"hf={torch.argmax(hf_dec_logits, -1).item()}")
    ok.append(same)
    ok.append(report("present_kv[0]", torch.tensor(res[1]),
                     hf_cache_to_kv(hf_dec.past_key_values, nl)[0], KV_MEAN_TOL))

    print("\n" + ("[✓] 数值对齐通过" if all(ok) else "[✗] 存在未对齐项"))
    return 0 if all(ok) else 1


if __name__ == "__main__":
    sys.exit(main())
