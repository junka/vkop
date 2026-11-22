#!/usr/bin/env python3
"""CPU ORT 的 greedy 解码参考：复刻 llm_chat 的输入构造（chat template → encode →
embed 查表 → 绝对位置 causal bias → KV 逐步拼接），逐步打印 argmax id，用于和
vkop GPU 的输出逐 token 对齐。

架构参数全部从 ONNX 图的输入形状与模型目录读，不写死某个模型：
  层数 = past_key_values_i 的个数，nkv/head_dim = past 输入的静态维，
  hidden = inputs_embeds 的最后一维，停止符 = generation_config.eos_token_id。

用法:
    python3 check/greedy_ref.py "用一个词回答：天空是什么颜色？"
    ONNX=text_glm_edge/llm.onnx EMBED=text_glm_edge/embed_tokens.bin \
    MODEL_PATH=~/.cache/modelscope/models/ZhipuAI--glm-edge-1.5b-chat/snapshots/master \
      python3 check/greedy_ref.py "你好"

HF=1 改用 HF transformers 的 model.generate(do_sample=False) 出参考（同一套 chat
template / 分词 / 步数），用来区分「导出图错」还是「vkop 错」：
    HF=1 MODEL_PATH=~/.cache/modelscope/hub/models/Qwen/Qwen3-VL-2B-Instruct \
      python3 check/greedy_ref.py "hi"
"""
import os
import sys

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.environ.get("MODEL_PATH") or os.path.join(
    os.path.expanduser("~"), ".cache/modelscope/models/"
    "LLM-Research--Phi-4-mini-instruct/snapshots/master")
ONNX = os.environ.get("ONNX") or os.path.join(HERE, "text_phi4", "llm.onnx")
EMBED = os.environ.get("EMBED") or os.path.join(HERE, "text_phi4", "embed_tokens.bin")
N = int(os.environ.get("GREEDY_STEPS", "12"))

prompt = sys.argv[1] if len(sys.argv) > 1 else "用一个词回答：天空是什么颜色？"

tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=False)
text = tok.apply_chat_template([{"role": "user", "content": prompt}],
                               tokenize=False, add_generation_prompt=True)
ids = tok(text, add_special_tokens=False)["input_ids"]
print(f"[prompt] {len(ids)} tokens: {ids}")

if os.environ.get("HF") == "1":
    import torch
    import transformers
    for cls in ("AutoModelForCausalLM", "AutoModelForImageTextToText"):
        try:
            model = getattr(transformers, cls).from_pretrained(
                MODEL_PATH, torch_dtype=torch.float16)
            break
        except (ValueError, KeyError):
            continue
    else:
        sys.exit(f"[!] 不认识的模型类型: {MODEL_PATH}")
    model.eval()
    out = model.generate(input_ids=torch.tensor([ids]), max_new_tokens=N,
                         do_sample=False)
    gen = out[0][len(ids):].tolist()
    print("generated ids:", gen)
    print("decoded:", repr(tok.decode(gen, skip_special_tokens=False)))
    sys.exit(0)

# ---- 架构参数：从图输入形状读 ----
sess = ort.InferenceSession(ONNX, providers=["CPUExecutionProvider"])
by_name = {i.name: i for i in sess.get_inputs()}
past_names = sorted((n for n in by_name if n.startswith("past_key_values_")),
                    key=lambda n: int(n.rsplit("_", 1)[1]))
if not past_names:
    sys.exit("[!] 图里没有 past_key_values_i 输入")
nl = len(past_names)
# past 形状 (batch, 2, nkv, past_len, head_dim)；动态维是 str/None，静态维是 int
pk0 = [d for d in by_name[past_names[0]].shape]
nkv, head_dim = pk0[2], pk0[4]
hidden = by_name["inputs_embeds"].shape[2]
if not all(isinstance(d, int) for d in (nkv, head_dim, hidden)):
    sys.exit(f"[!] 形状里还有动态维: past={pk0} embeds={by_name['inputs_embeds'].shape}")
print(f"[arch] layers={nl} nkv={nkv} head_dim={head_dim} hidden={hidden}")

import json
with open(os.path.join(MODEL_PATH, "generation_config.json")) as f:
    eos = json.load(f).get("eos_token_id")
stop = set(eos if isinstance(eos, list) else [eos])
print(f"[stop] eos_token_id={sorted(stop)}")

emb_table = np.fromfile(EMBED, dtype=np.float16).reshape(-1, hidden)
past = [np.zeros((1, 2, nkv, 0, head_dim), dtype=np.float16) for _ in range(nl)]


def run(step_ids, pos_start):
    global past
    L = len(step_ids)
    kv = past[0].shape[3] + L
    # causal bias 按「绝对位置」判定：第 i 行可见 j <= pos_start+i 的列。按 chunk 内
    # 相对索引做 triu 会把 decode 步整行遮掉。
    bias = np.zeros((L, kv), dtype=np.float16)
    for i in range(L):
        bias[i, pos_start + i + 1:] = np.finfo(np.float16).min
    # position_ids: 纯文本 Qwen3/Phi-4/GLM 是 2D (B, q)；Qwen3-VL 是 3D mrope
    # (3, B, q)，文本 token 三轴位置相等，三行重复即可退化为标准 RoPE。
    pos = np.arange(pos_start, pos_start + L, dtype=np.int64)
    pid_shape = by_name["position_ids"].shape
    pos_feed = (np.stack([pos] * 3)[:, None, :] if len(pid_shape) == 3
                else pos[None])
    feed = {
        "inputs_embeds": np.ascontiguousarray(emb_table[step_ids][None]),
        "position_ids": np.ascontiguousarray(pos_feed),
        "attention_bias": bias[None, None],
    }
    for i in range(nl):
        feed[f"past_key_values_{i}"] = past[i]
    # Qwen3-VL 的多模态输入，纯文本时喂 llm_chat 同款的「空」值：deepstack 一行
    # 全零 (1, hidden)，image_pad_mask 全 false (B, L) —— 没有位置被 scatter，
    # 图就退化成纯文本路径。纯文本模型没有这些输入，这里什么也不加。
    for name in by_name:
        if name.startswith("deepstack_embeds_"):
            feed[name] = np.zeros((1, hidden), dtype=np.float16)
        elif name == "image_pad_mask":
            feed[name] = np.zeros((1, L), dtype=bool)
    out = sess.run(None, feed)
    past = out[1:]
    return out[0]


logits = run(ids, 0)
top5 = np.argsort(logits[0, -1].astype(np.float32))[::-1][:5]
print("[prefill top5]", [(int(i), round(float(logits[0, -1, i]), 2)) for i in top5])
gen = [int(np.argmax(logits[0, -1].astype(np.float32)))]
for step in range(N - 1):
    if gen[-1] in stop:
        break                      # 与 llm_chat 同一个停止口径，前缀才可比
    logits = run([gen[-1]], len(ids) + step)
    gen.append(int(np.argmax(logits[0, -1].astype(np.float32))))

print("generated ids:", gen)
print("stopped at stop token:", gen[-1] in stop)
print("decoded:", repr(tok.decode(gen, skip_special_tokens=False)))
