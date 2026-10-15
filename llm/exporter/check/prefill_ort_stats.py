#!/usr/bin/env python3
"""ORT prefill 中间张量统计，与 llm_chat 的 VKOP_DUMP_TENSORS="*" 输出同格式。

dump_ort_intermediates.py 的精简重写：
  - 不依赖 transformers 的模型类（Qwen3VLForConditionalGeneration 导入链因
    torchvision::nms 损坏不可用），输入由 AutoTokenizer + embed_tokens.bin
    构造，与 check/greedy_ref.py 同一套（ids/因果 bias/mrope position_ids
    均已与 C++ 驱动核对一致）。
  - 增强图（所有节点输出追加为 graph outputs）不做 3.4GB 权重拷贝：
    load_external_data=False 保留外部引用，aug.onnx 落在 llm/ 目录原位解析
    llm.weights.bin，用完即删。

用法:
    python3 check/prefill_ort_stats.py [prompt] > /tmp/ort_prefill_stats.txt
"""
import os
import sys

import numpy as np
import onnx
import onnxruntime as ort
from transformers import AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
LLM_DIR = os.path.normpath(os.path.join(HERE, "..", ".."))
ONNX = os.path.join(LLM_DIR, "llm.onnx")
EMBED = os.path.join(LLM_DIR, "exporter", "embed_tokens.bin")
MODEL_PATH = os.path.expanduser(
    "~/.cache/modelscope/hub/models/Qwen/Qwen3-VL-2B-Instruct")

prompt = sys.argv[1] if len(sys.argv) > 1 else "hi"

# ---- 输入构造（与 greedy_ref.py 完全一致） ----
tok = AutoTokenizer.from_pretrained(MODEL_PATH)
text = tok.apply_chat_template([{"role": "user", "content": prompt}],
                               tokenize=False, add_generation_prompt=True)
ids = tok(text, add_special_tokens=False)["input_ids"]
print(f"[prompt] {len(ids)} tokens", file=sys.stderr)

emb_table = np.fromfile(EMBED, dtype=np.float16).reshape(-1, 2048)
sess_in = {
    "inputs_embeds": np.ascontiguousarray(emb_table[ids][None]),
    "position_ids": np.ascontiguousarray(
        np.stack([np.arange(len(ids), dtype=np.int64)] * 3)[:, None, :]),
}
L = len(ids)
sess_in["attention_bias"] = np.ascontiguousarray(np.triu(
    np.full((L, L), np.finfo(np.float16).min, dtype=np.float16),
    k=1)[None, None])

# ---- 增强图：全部节点输出追加为 outputs，外部权重零拷贝 ----
m = onnx.load(ONNX, load_external_data=False)
want = sorted({o for n in m.graph.node for o in n.output if o})
existing = {o.name for o in m.graph.output}
for name in want:
    if name not in existing:
        m.graph.output.add().name = name
aug = os.path.join(LLM_DIR, "aug_prefill.onnx")
onnx.save(m, aug)   # 外部引用相对 llm/ 解析 -> llm.weights.bin，不复制权重
print(f"[aug] {len(m.graph.output)} outputs", file=sys.stderr)

so = ort.SessionOptions()
sess = ort.InferenceSession(aug, sess_options=so,
                            providers=["CPUExecutionProvider"])
in_set = {i.name for i in sess.get_inputs()}
feed = {k: v for k, v in sess_in.items() if k in in_set}
for i in range(28):
    n = f"past_key_values_{i}"
    if n in in_set:
        feed[n] = np.zeros((1, 2, 8, 0, 128), dtype=np.float16)
out_names = [o.name for o in sess.get_outputs()]
res = sess.run(out_names, feed)
os.remove(aug)

# ---- 统计行：与 C++ dump 同格式，按名字排序保证可 join ----
lines = []
for name, arr in zip(out_names, res):
    if arr.dtype != np.float16:
        continue
    f = arr.astype(np.float32)
    finite = f[np.isfinite(f)]
    mn = float(finite.min()) if finite.size else 0.0
    mx = float(finite.max()) if finite.size else 0.0
    first = ",".join(f"{x:04x}" for x in arr.reshape(-1)[:4].astype(np.uint16))
    lines.append((name, f"[{name}] ne={f.size} nan={int(np.isnan(f).sum())} "
                  f"inf={int(np.isinf(f).sum())} zero={int((f == 0).sum())} "
                  f"min={mn:.4g} max={mx:.4g} first=[{first}]"))
for _, line in sorted(lines):
    print(line)
