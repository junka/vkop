#!/usr/bin/env python3
# 生成 HF tokenizers 的 encode() id 序列作为 ground truth，供 C++ 测试逐条对比。
#
# 用例表是共用的，一个 tokenizer.json 出一份 ground truth：
#   TOK_JSON=~/.cache/modelscope/hub/models/Qwen/Qwen3-VL-2B-Instruct/tokenizer.json #     python3 gen_ground_truth.py                      # 默认 hf_ground_truth.json
#   TOK_JSON=~/.cache/modelscope/models/LLM-Research--Phi-4-mini-instruct/snapshots/master/tokenizer.json \
#   OUT=phi_ground_truth.json python3 gen_ground_truth.py
# 同一批字符串在各自分词器下的 id，所以每个模型的 bin 都要跟自己的那份比。
from tokenizers import Tokenizer
import json
import os

TOK_JSON = os.environ.get("TOK_JSON") or os.path.expanduser(
    "~/.cache/modelscope/hub/models/Qwen/Qwen3-VL-2B-Instruct/tokenizer.json")
# 输出到脚本所在目录，便于从任意 cwd 运行。
OUT = os.environ.get("OUT") or os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "hf_ground_truth.json")

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
    # Phi-4 的 pre_tokenizer 与 GPT-2 的三处差异，各给一组用例（Qwen 侧同时验证
    # GPT-2 分支不受影响）：
    #  1) 字母按大小写「形状」切分：HelloWorld -> Hello + World，ABc 却是一整段。
    "HelloWorld", "McDonald", "ABc", "ABC", "iPhone 15 Pro", "JSON", "OAuth2token",
    #  2) 数字 \p{N}{1,3}：连续数字最多 3 个一切，GPT-2 是逐位切。
    "1234567", "3.14159", "2026-10-02", "v1.2.3", "0", "42", "7 88 999 1000",
    "12345678901234567890", "3/4", "http://a.com/b", "a/b/c",
    #  3) 缩写只能挂在字母段尾巴上（GPT-2 有独立分支，撇号串可以单独成段）。
    "isn't", "we're", "I'll", "'s", "rock'n'roll", "it' s",
    # 同时属于「大写类」和「小写类」的字母（Lm/Lo）加组合记号 M：[UL]*[LL]+ 的
    # 回退路径只有这类连续字符才会触发（Lo 两边都在，得回退一个才能匹配）。
    "\u1401\u1401\u1401", "e\u0301", "\u0e01\u0e34", "\u0915\u093f",
    "a\u0301b", "\u1401A", "A\u1401", "caf\u00e9 CAF\u00c9",
    "Hello   \u4f60\u597d   world", "\u4f60\u597d123abc",
]

tok = Tokenizer.from_file(TOK_JSON)
out = [{"s": c, "ids": tok.encode(c).ids} for c in cases]
with open(OUT, "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=1)
print(f"wrote {len(out)} cases to {OUT}")
