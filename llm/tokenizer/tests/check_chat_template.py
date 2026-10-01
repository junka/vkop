#!/usr/bin/env python3
"""校验 tokenizer.bin 里烘进去的 chat template 与 HF jinja 渲染逐字符一致。

C++ 侧的 tests/main.cpp 只能验证「按角色拼装」这件事（它拿不到 jinja），拼装内容
对不对必须在这里比：写端 tokenizer_to_bin.py 是用 apply_chat_template 的差分探针提取
各角色 prefix/suffix 的，探针口径错了（例如把模板给整段对话补的收尾串算进最后一个角色
的后缀），这里就会立刻对不上。

用法:
    BIN=phi4_mini.bin MODEL_DIR=/path/to/Phi-4-mini-instruct \
      python3 tests/check_chat_template.py
默认比 Qwen3-VL（与 tokenizer_to_bin.py 的默认 MODEL_DIR 一致）。
"""
import json
import os
import struct
import sys

BIN = os.environ.get("BIN") or os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "qwen3_vl.bin")
MODEL_DIR = os.environ.get("MODEL_DIR") or os.path.expanduser(
    "~/.cache/modelscope/hub/models/Qwen/Qwen3-VL-2B-Instruct")


def read_bin_template(path):
    data = open(path, "rb").read()
    assert data[:4] == b"QW3T", "bad magic"
    _, vocab_n, merges_n, special_n, flags = struct.unpack_from("<5I", data, 4)
    off = 24

    def s():  # 长度前缀串
        nonlocal off
        ln = struct.unpack_from("<H", data, off)[0]
        off += 2
        v = data[off:off + ln].decode("utf-8")
        off += ln
        return v

    for _ in range(vocab_n):
        s()
        off += 4
    off += merges_n * 12
    for _ in range(special_n):
        s()
        off += 4
    roles = {}
    role_n = struct.unpack_from("<I", data, off)[0]
    off += 4
    for _ in range(role_n):
        name, prefix, suffix = s(), s(), s()
        roles[name] = (prefix, suffix)
    cts = {}
    ct_n = struct.unpack_from("<I", data, off)[0]
    off += 4
    for _ in range(ct_n):
        k, v = s(), s()
        cts[k] = v
    gen, default_sys = s(), s()
    return {"flags": flags, "roles": roles, "content_types": cts,
            "generation_prompt": gen, "default_system_prompt": default_sys}


def cpp_render(t, msgs, add_generation_prompt):
    """复刻 qwen::Tokenizer::apply_chat_template 的拼装（纯文本内容）。"""
    out = ""
    if msgs and msgs[0][0] == "system":
        p, sfx = t["roles"]["system"]
        out += p + msgs[0][1] + sfx
        rest = msgs[1:]
    elif t["default_system_prompt"]:
        p, sfx = t["roles"]["system"]
        out += p + t["default_system_prompt"] + sfx
        rest = msgs
    else:
        rest = msgs
    for role, content in rest:
        p, sfx = t["roles"][role]
        out += p + content + sfx
    if add_generation_prompt:
        out += t["generation_prompt"]
    return out


def main():
    from transformers import AutoTokenizer
    try:
        tok = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=False)
    except Exception:
        tok = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    t = read_bin_template(BIN)
    print(f"[bin] {BIN}\n[ref] {MODEL_DIR}\n"
          f"[template] roles={ {k: v for k, v in t['roles'].items()} } "
          f"generation_prompt={t['generation_prompt']!r}")

    cases = [
        ([("user", "你好")], True),
        ([("system", "你是助手。"), ("user", "你好")], True),
        ([("system", "你是助手。"), ("user", "你好"), ("assistant", "我是助手。")], True),
        ([("user", "第一问"), ("assistant", "第一答"), ("user", "第二问")], True),
    ]
    # add_generation_prompt=False 单独比「去掉收尾串」：C++ 的渲染器是逐条消息拼
    # prefix+content+suffix，不模拟模板给整段对话补的收尾（驱动总是带引导串渲染，
    # 逐轮拼装也不需要它）。Phi-4 的收尾是 eos_token，Qwen 的是空串。
    tail_cases = [
        ([("system", "你是助手。"), ("user", "你好"), ("assistant", "我是助手。")], False),
    ]
    bad = 0
    for msgs, gp in cases + tail_cases:
        hf = tok.apply_chat_template(
            [{"role": r, "content": c} for r, c in msgs],
            tokenize=False, add_generation_prompt=gp)
        mine = cpp_render(t, msgs, gp)
        if not gp and not hf.startswith(mine):
            bad += 1
            print(f"[FAIL] gp=0 渲染不是 HF 渲染的前缀")
            continue
        if gp and hf != mine:
            bad += 1
            print(f"[FAIL] gp=1 {msgs[0][0]}x{len(msgs)}")
            print("   HF :", repr(hf))
            print("   BIN:", repr(mine))
            continue
        print(f"[{'ok  ' if gp else 'ok   (gp=0, 差异只在收尾串)'}] "
              f"gp={int(gp)} {msgs[0][0]}x{len(msgs)}"
              + ("" if gp else f"  tail={hf[len(mine):]!r}"))
    if t["flags"] & 1:
        print("[note] phi pre_tokenizer 变体；纯文本模板无 image 占位属预期。")
    print("\n" + ("[✓] chat template 与 HF 渲染逐字符一致" if not bad
                  else f"[✗] {bad} 条不一致"))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
