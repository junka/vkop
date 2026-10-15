"""One-shot HF greedy reference for vkop llm_chat alignment (text-only).

Usage:
    python3 check/hf_greedy_ref.py "hi" [steps]
"""
import os, sys
import torch
from transformers import AutoTokenizer, Qwen3VLForConditionalGeneration

MODEL = os.path.expanduser(
    "~/.cache/modelscope/hub/models/Qwen/Qwen3-VL-2B-Instruct")
PROMPT = sys.argv[1] if len(sys.argv) > 1 else "hi"
STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 8

tok = AutoTokenizer.from_pretrained(MODEL)
text = tok.apply_chat_template(
    [{"role": "user", "content": PROMPT}], tokenize=False,
    add_generation_prompt=True)
ids = tok(text, add_special_tokens=False)["input_ids"]
print("[prompt ids]", ids)

model = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL, torch_dtype=torch.float16)
model.eval()

out = model.generate(
    input_ids=torch.tensor([ids]),
    max_new_tokens=STEPS,
    do_sample=False,
)
new = out[0][len(ids):].tolist()
print("[greedy new ids]", new)
print("[greedy text]", repr(tok.decode(new)))
