#!/usr/bin/env python3
"""单独导出 GLM-Edge-1.5B-Chat 的 embed_tokens 权重表（[vocab, hidden] fp16）。

llm.onnx 的图输入是 inputs_embeds（已 embed），embed_tokens 不在图里；C++ 的
generate loop (llm_chat) 需要这张表做 token_id→embed 查表。

GLM-Edge 是 tie_word_embeddings=true（导出脚本已断言 lm_head 与 embed_tokens 逐位
相等），取 embed 那份即可。hidden / vocab 从 config 读，不硬编码。

用法:
    python3 dump_embed_tokens_glm_edge.py [out.bin]
    MODEL_PATH=/path/to/glm-edge-1.5b-chat python3 dump_embed_tokens_glm_edge.py
"""
import os
import sys
import torch

MODEL_PATH = os.environ.get("MODEL_PATH") or os.path.join(
    os.path.expanduser("~"), ".cache/modelscope/models/"
    "ZhipuAI--glm-edge-1.5b-chat/snapshots/master")
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "text_glm_edge", "embed_tokens.bin")

print(f"[load] model = {MODEL_PATH}")

from transformers.models.glm.modeling_glm import GlmForCausalLM  # noqa: E402

model = GlmForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,
    local_files_only=os.path.isdir(os.path.expanduser(MODEL_PATH)),
)
model.eval()

hidden = model.config.hidden_size
vocab = model.config.vocab_size
w = model.model.embed_tokens.weight

assert w.dtype == torch.float16, f"expected fp16, got {w.dtype}"
assert tuple(w.shape) == (vocab, hidden), f"{w.shape} != ({vocab},{hidden})"
assert torch.equal(w, model.lm_head.weight), "tied 假设不成立，embed 表要单独取"

os.makedirs(os.path.dirname(os.path.abspath(OUT)), exist_ok=True)
with open(OUT, "wb") as f:
    f.write(w.detach().cpu().contiguous().numpy().tobytes())
print(f"[✓] {OUT}  shape={tuple(w.shape)} dtype=fp16  "
      f"{os.path.getsize(OUT)/1e6:.1f}MB  vocab={vocab} hidden={hidden}")
