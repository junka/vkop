#!/usr/bin/env python3
"""Phi-4-mini 的 ORT greedy 解码参考：复刻 llm_chat 的输入构造（chat template →
encode → embed 查表 → 绝对位置 causal bias → KV 逐步拼接），逐步打印 argmax id，
用于和 vkop GPU 的输出逐 token 对齐。

用法: python3 llm/exporter/phi4_greedy_ref.py [prompt]
"""
import os
import sys

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
MP = os.environ.get("MODEL_PATH") or os.path.join(
    os.path.expanduser("~"), ".cache/modelscope/models/"
    "LLM-Research--Phi-4-mini-instruct/snapshots/master")
ONNX = os.path.join(HERE, "text_phi4", "llm.onnx")
EMB = os.path.join(HERE, "text_phi4", "embed_tokens.bin")
N = int(os.environ.get("GREEDY_STEPS", "12"))
HIDDEN, NL, NKV, HD = 3072, 32, 8, 128

prompt = sys.argv[1] if len(sys.argv) > 1 else "用一个词回答：天空是什么颜色？"

tok = AutoTokenizer.from_pretrained(MP, trust_remote_code=False)
text = tok.apply_chat_template([{"role": "user", "content": prompt}],
                               tokenize=False, add_generation_prompt=True)
ids = tok(text, add_special_tokens=False)["input_ids"]
print(f"[prompt] {len(ids)} tokens: {ids}")

emb_table = np.fromfile(EMB, dtype=np.float16).reshape(-1, HIDDEN)
past = [np.zeros((1, 2, NKV, 0, HD), dtype=np.float16) for _ in range(NL)]
sess = ort.InferenceSession(ONNX, providers=["CPUExecutionProvider"])


def run(step_ids, pos_start):
    global past
    L = len(step_ids)
    kv = past[0].shape[3] + L
    # causal bias 按「绝对位置」判定：第 i 行可见 j <= pos_start+i 的列。按 chunk 内
    # 相对索引做 triu 会把 decode 步整行遮掉。
    bias = np.zeros((L, kv), dtype=np.float16)
    for i in range(L):
        bias[i, pos_start + i + 1:] = np.finfo(np.float16).min
    feed = {
        "inputs_embeds": np.ascontiguousarray(emb_table[step_ids][None]),
        "position_ids": np.arange(pos_start, pos_start + L, dtype=np.int64)[None],
        "attention_bias": bias[None, None],
    }
    for i in range(NL):
        feed[f"past_key_values_{i}"] = past[i]
    out = sess.run(None, feed)
    past = out[1:]
    return out[0]


logits = run(ids, 0)
gen = [int(np.argmax(logits[0, -1].astype(np.float32)))]
for step in range(N - 1):
    logits = run([gen[-1]], len(ids) + step)
    gen.append(int(np.argmax(logits[0, -1].astype(np.float32))))

print("generated ids:", gen)
print("decoded:", repr(tok.decode(gen, skip_special_tokens=False)))
