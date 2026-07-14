#!/usr/bin/env python3
# 生成 HF tokenizers 的 encode() id 序列作为 ground truth，供 C++ 测试逐条对比。
from tokenizers import Tokenizer
import json

TOK_JSON = "/home/gwm/.cache/modelscope/hub/models/Qwen/Qwen3-VL-2B-Instruct/tokenizer.json"
OUT = "tests/hf_ground_truth.json"

cases = [
    "", "a", "A", " ", "  ", "   ", "\n", "\n\n", "\n\n\n", " \n", "\n ", "a\n", "a \n", "\n a",
    "Hello   world", "Hello   world   ", "Hello world", "Hello world\n", "Hello world \n",
    "a  b", "a  b  c", "  leading spaces", "x   y   z   ", "end   ", "a   b\n", "a   \nb",
    "text\n\n\nmore", "foo \n bar", "foo\rbar", "foo\r\nbar", "a\rB", "\r\n\r", "a\r\r\nb",
    " \r \n b", "x   \n   b", "a\tb", "a\tb\t", "\t\t", " \t \n xyz", "a\x0bB",
    "1+2", "+-*/", "!!!", "a.b.c", "123abc", "abc123", "  123", "a  123", "  +", "a  +",
    "中文", "Café", "naïve", "你好\n世界", "don't", "It's", "Ć", "é", "é",
    "Hello, Qwen3-VL! 你好，世界。", "def main():\n    print(\"Hi\") # 测试",
    "<|im_start|>", "a<|im_end|>b", "<|vision_start|><|image_pad|><|vision_end|>",
    "a b c d e", "multiple   spaces   between   words",
]

tok = Tokenizer.from_file(TOK_JSON)
out = [{"s": c, "ids": tok.encode(c).ids} for c in cases]
with open(OUT, "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=1)
print(f"wrote {len(out)} cases to {OUT}")
