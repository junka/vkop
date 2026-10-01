#!/usr/bin/env python3
"""单独导出 Phi-4-mini-instruct 的 embed_tokens 权重表。

llm.onnx 的图输入是 inputs_embeds（已 embed），embed_tokens 不在图里；C++ 的
generate loop (llm_chat) 需要这张 [vocab, hidden] fp16 表做 token_id→embed 查表。

Phi-4-mini 是 tie_word_embeddings=true，lm_head.weight 与 embed_tokens.weight
是同一个张量，这里取 embed 那份即可。hidden / vocab 从模型 config 读，不硬编码
（llm_chat 也会从文件大小反推，两者必须一致）。

用法:
    python3 dump_embed_tokens_phi4.py [out.bin]
    MODEL_PATH=/path/to/Phi-4-mini-instruct python3 dump_embed_tokens_phi4.py
"""
import os
import sys
import torch

MODEL_PATH = os.environ.get("MODEL_PATH") or os.path.join(
    os.path.expanduser("~"), ".cache/modelscope/models/"
    "LLM-Research--Phi-4-mini-instruct/snapshots/master")
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "text_phi4", "embed_tokens.bin")

print(f"[load] model = {MODEL_PATH}")

from transformers.models.phi3.modeling_phi3 import Phi3ForCausalLM  # noqa: E402

model = Phi3ForCausalLM.from_pretrained(
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

os.makedirs(os.path.dirname(os.path.abspath(OUT)), exist_ok=True)
with open(OUT, "wb") as f:
    f.write(w.detach().cpu().contiguous().numpy().tobytes())
print(f"[✓] {OUT}  shape={tuple(w.shape)} dtype=fp16  "
      f"{os.path.getsize(OUT)/1e6:.1f}MB  vocab={vocab} hidden={hidden}")
