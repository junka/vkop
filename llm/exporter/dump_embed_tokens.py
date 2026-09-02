#!/usr/bin/env python3
"""单独导出 embed_tokens 权重表为 embed_tokens.bin（裸 fp16 张量）。

llm.onnx 的输入是 inputs_embeds（已 embed），embed_tokens 不在图里。C++ 的
generate loop (llm_chat) 需要这张 [vocab, hidden] fp16 表做 token_id→embed 查表。
本脚本只抽 embed_tokens，不重跑 ONNX 导出，避免重做耗时的 trace/export。

用法:
    python3 llm/exporter/dump_embed_tokens.py [输出路径]
默认输出 llm/exporter/embed_tokens.bin。
"""
import os
import sys
import torch
from transformers import Qwen3VLForConditionalGeneration

MODEL_PATH = os.path.expanduser(
    "~/.cache/modelscope/hub/models/Qwen/Qwen3-VL-2B-Instruct")
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    os.path.dirname(__file__), "embed_tokens.bin")

print(f"[load] {MODEL_PATH}")
model = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_PATH, attn_implementation="eager", torch_dtype=torch.float16).eval()
lm = model.model.language_model
text_config = model.config.text_config
vocab = text_config.vocab_size
hidden = text_config.hidden_size

w = lm.embed_tokens.weight  # (vocab, hidden) fp16
assert w.dtype == torch.float16, f"expected fp16, got {w.dtype}"
assert tuple(w.shape) == (vocab, hidden), f"{w.shape} != ({vocab},{hidden})"
w_cpu = w.detach().cpu().contiguous()
with open(OUT, "wb") as f:
    f.write(w_cpu.numpy().tobytes())
print(f"[✓] {OUT}  shape={tuple(w_cpu.shape)} dtype=fp16  "
      f"{os.path.getsize(OUT)/1e6:.1f}MB  vocab={vocab} hidden={hidden}")
