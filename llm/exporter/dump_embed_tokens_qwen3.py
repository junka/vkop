#!/usr/bin/env python3
"""单独导出纯文本 Qwen3 系列 (Qwen3ForCausalLM) 的 embed_tokens 权重表。

llm.onnx 的输入是 inputs_embeds（已 embed），embed_tokens 不在图里。C++ 的
generate loop (llm_chat) 需要这张 [vocab, hidden] fp16 表做 token_id→embed 查表。
本脚本只抽 embed_tokens，不重跑 ONNX 导出。

自动识别架构参数 (hidden_size / vocab_size) —— 从模型 config 读，不硬编码。
和 llm_chat.cpp 配合：它从 embed_tokens.bin 文件大小反推 hidden 和 vocab。

用法:
    python3 dump_embed_tokens_qwen3.py                    # 默认 Qwen3-8B
    MODEL_PATH=Qwen/Qwen3-4B-Instruct \
    python3 dump_embed_tokens_qwen3.py ./embed_tokens.bin
"""
import os
import sys
import torch

DEFAULT_PATH = "Qwen/Qwen3-8B-Instruct"
MODEL_PATH = os.environ.get("MODEL_PATH") or DEFAULT_PATH
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    os.path.dirname(__file__), "embed_tokens.bin")

print(f"[load] model = {MODEL_PATH}")

# 优先 Qwen3ForCausalLM；fallback 到 AutoModelForCausalLM (兼容老版本 transformers)
try:
    from transformers import Qwen3ForCausalLM
    model = Qwen3ForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16,
        local_files_only=os.path.isdir(os.path.expanduser(MODEL_PATH)),
    )
except ImportError:
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        local_files_only=os.path.isdir(os.path.expanduser(MODEL_PATH)),
    )

model.eval()

config = model.config
hidden = config.hidden_size
vocab = config.vocab_size

embed_tokens = model.model.embed_tokens   # Qwen3: model.model.embed_tokens

w = embed_tokens.weight   # (vocab, hidden) fp16
assert w.dtype == torch.float16, f"expected fp16, got {w.dtype}"
assert tuple(w.shape) == (vocab, hidden), f"{w.shape} != ({vocab},{hidden})"

w_cpu = w.detach().cpu().contiguous()
with open(OUT, "wb") as f:
    f.write(w_cpu.numpy().tobytes())
print(f"[✓] {OUT}  shape={tuple(w_cpu.shape)} dtype=fp16  "
      f"{os.path.getsize(OUT)/1e6:.1f}MB  vocab={vocab} hidden={hidden}")
