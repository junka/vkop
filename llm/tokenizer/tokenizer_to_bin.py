import json
import struct
import os
import re

# 一条命令一个模型：MODEL_DIR / OUTPUT_BIN 用环境变量覆盖，默认保持 Qwen3-VL 原样。
#   MODEL_DIR=~/.cache/modelscope/models/LLM-Research--Phi-4-mini-instruct/snapshots/master \
#   OUTPUT_BIN=phi4.bin python3 tokenizer_to_bin.py
MODEL_DIR = os.path.expanduser(os.environ.get("MODEL_DIR")
                               or "~/.cache/modelscope/hub/models/Qwen/Qwen3-VL-2B-Instruct")
TOKENIZER_JSON = os.path.join(MODEL_DIR, "tokenizer.json")
TOKENIZER_CONFIG_JSON = os.path.join(MODEL_DIR, "tokenizer_config.json")
CHAT_TEMPLATE_JSON = os.path.join(MODEL_DIR, "chat_template.json")
OUTPUT_BIN = os.environ.get("OUTPUT_BIN") or "qwen3_vl.bin"

def bytes_to_unicode():
    """标准 BBPE byte-to-unicode 映射表"""
    bs = list(range(ord("!"), ord("~")+1)) + list(range(ord("¡"), ord("¬")+1)) + list(range(ord("®"), ord("ÿ")+1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8+n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))

def common_prefix_len(a, b):
    """逐字符公共前缀长度。

    不能用 os.path.commonprefix：它按路径语义在分隔符处截断（对 a/xx/yy 和 a/xx/zz
    只返回 a/xx/），而这里的字符串是聊天模板的渲染结果，特殊 token 的字面量里就带
    那个分隔符，截断点会错。
    """
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def extract_chat_template():
    """用 HF tokenizer 的 apply_chat_template 探针提取角色 prefix/suffix、
    内容占位格式、generation_prompt、default_system_prompt。

    返回结构与 trt_edgellm 的 processed_chat_template.json 一致：
      roles: {system/user/assistant: {prefix, suffix}}
      content_types: {image/video: {format}}
      generation_prompt, default_system_prompt
    """
    from transformers import AutoTokenizer
    try:
        # 优先走本地原生实现：有些仓库（Phi-4-mini）的 auto_map 把 AutoTokenizer
        # 指向外部仓库名，trust_remote_code=True 会去联网取，离线就失败。
        tok = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=False)
    except Exception:
        tok = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)

    # 每个角色两条哨兵：第二条只用来连发同角色轮做长度差分，两条必须等长，否则
    # 「len(two) - len(one)」量出来的一整轮会带上第二条约内容多出来的那几个字符。
    SYS, SYS2 = "__SENTINEL_SYS_a7f3e2b1__", "__SENTINEL_SYS_b8g4f7d9__"
    USR, USR2 = "__SENTINEL_USR_c9d4f6e8__", "__SENTINEL_USR_d0h5g7f0__"
    AST, AST2 = "<placeholder_assistant_tx1>", "<placeholder_assistant_tx2>"

    sys_msg = {"role": "system", "content": SYS}
    usr_msg = {"role": "user", "content": USR}

    def render(ms, gp=False):
        return tok.apply_chat_template(ms, tokenize=False, add_generation_prompt=gp)

    def probe_role(role, sent, sent2, prev):
        """探一个角色的 (prefix, suffix)，顺带确认/更新整段对话的收尾串。

        不能把单轮差分整段当后缀：有的模板（Phi-4-mini）在
        add_generation_prompt=False 时给整段对话补一个 eos
        （`...{% else %}{{ eos_token }}{% endif %}`），那截收尾会被算进最后一条消息的
        suffix；而 C++ 渲染器是逐条消息拼 prefix+content+suffix，等于每轮都注入一个
        eos。收尾串在「同一角色连发两轮」的差分里长度不变，所以
            一整轮长度 turn = len(two) - len(one) = prefix + content + suffix
        再结合哨兵在渲染串里的位置就能把 prefix/suffix 各自解出来。
        哨兵必须在整段渲染里恰好出现一次，否则位置就不可信（模板把 content 渲染了两
        遍，或者把它吃掉/截断了），这种情况直接报错而不是产出错位的前缀。
        """
        nonlocal tail_len, tail_str
        msgs = [{"role": r, "content": s} for r, s in prev]
        # 空历史不用渲染：transformers 5 直接拒绝空 conversation。
        prev_render = render(msgs) if msgs else ""
        one = render(msgs + [{"role": role, "content": sent}])
        two = render(msgs + [{"role": role, "content": sent},
                             {"role": role, "content": sent2}])
        turn = len(two) - len(one)
        q = one.find(sent)
        if q < 0 or one.count(sent) != 1:
            raise RuntimeError(f"chat template probe failed for role {role!r}: "
                               f"sentinel found {one.count(sent)} time(s)")
        if turn < len(sent):
            # 模板把重复的同角色轮丢掉了（Qwen 的 system 只取第一条）→ 量不出整轮
            # 长度。此时把「整段减去已渲染历史」当作一整轮：两处收尾串相减正好抵消。
            turn = len(one) - len(prev_render)
        # 已渲染历史里属于「正文」的那一段长度：prev_render 结尾带着收尾串，要减掉。
        # 空历史没有渲染，正文长度就是 0。
        prev_body = 0 if not msgs else len(prev_render) - tail_len
        if not msgs:
            # 空历史时整段渲染 = 一整轮 + 收尾串，两个长度之差直接给出收尾串长度，
            # 后面几条探针复用它（连发两轮的差分里收尾串长度不变）。
            tail_len = len(one) - turn
            if tail_len < 0:
                raise RuntimeError(f"chat template probe failed for role {role!r}: "
                                   f"turn {turn} is longer than the whole render")
        p_len = q - prev_body
        if p_len < 0 or p_len + len(sent) > turn:
            raise RuntimeError(f"chat template probe failed for role {role!r}: "
                               f"prefix len {p_len} inconsistent with turn len {turn}")
        suffix = one[q + len(sent): prev_body + turn]
        if prev_body + turn + tail_len != len(one):
            raise RuntimeError(f"chat template probe failed for role {role!r}: "
                               f"turn+tail does not cover the render")
        if not msgs:
            tail_str = one[prev_body + turn:]
        return one[q - p_len:q], suffix

    tail_len = 0
    tail_str = ""
    sys_prefix, sys_suffix = probe_role("system", SYS, SYS2, [])
    usr_prefix, usr_suffix = probe_role("user", USR, USR2, [("system", SYS)])
    ast_prefix, ast_suffix = probe_role("assistant", AST, AST2,
                                        [("system", SYS), ("user", USR)])
    print(f"[+] conversation tail (only with add_generation_prompt=False): {tail_str!r}")

    # generation_prompt：同一段历史在 add_generation_prompt 真/假下的差。False 一侧
    # 结尾是模板给整段对话补的收尾串（probe_role 已经量出来），先剥掉再取公共前缀，
    # 否则收尾串和引导串排在同一位置、开头又都是 '<'，公共前缀会在引导串中间停下。
    hist = [{"role": "system", "content": SYS}, {"role": "user", "content": USR}]
    no_gen, with_gen = render(hist, False), render(hist, True)
    if not no_gen.endswith(tail_str):
        raise RuntimeError("chat template probe failed for generation_prompt: "
                           f"conversation tail {tail_str!r} not at the end of the render")
    cut = common_prefix_len(no_gen[:len(no_gen) - len(tail_str)], with_gen)
    generation_prompt = with_gen[cut:]

    # default_system_prompt：仅 user 消息时若模板自动注入系统块则提取其内容。
    usr_only = tok.apply_chat_template([usr_msg], tokenize=False, add_generation_prompt=False)
    default_system_prompt = ""
    s_start = usr_only.find(sys_prefix)
    if s_start != -1:
        c_start = s_start + len(sys_prefix)
        c_end = usr_only.find(sys_suffix, c_start)
        if c_end != -1:
            default_system_prompt = usr_only[c_start:c_end]
            if default_system_prompt == SYS:
                default_system_prompt = ""

    # image/video 内容占位格式：对比「纯文本」与「带图/视频」的差分。
    # 纯文本模板不接受 list 形式的 content（Phi-4 的模板直接做字符串拼接，遇到
    # content 列表会 TypeError），这类模型没有视觉占位，content_types 留空即可。
    content_types = {}
    base_text = "<placeholder_user_text>"
    try:
        base_fmt = tok.apply_chat_template(
            [sys_msg, {"role": "user", "content": [{"type": "text", "text": base_text}]}],
            tokenize=False, add_generation_prompt=False)
    except Exception as e:
        print(f"[+] chat template takes no list-content messages ({e}); "
              f"no image/video content types.")
        base_fmt = None
    if base_fmt is not None:
        for kind, ph in [("image", "<placeholder_image_path>"), ("video", "<placeholder_video_path>")]:
            u = {"role": "user", "content": [{"type": "text", "text": base_text}, {"type": kind, kind: ph}]}
            withc = tok.apply_chat_template([sys_msg, u], tokenize=False, add_generation_prompt=False)
            tp = base_fmt.find(base_text) + len(base_text)
            cp = withc.find(base_text) + len(base_text)
            # 取 base_text 之后到本轮结束的那一段：后面接着的是 user 轮的
            # prefix+sent+suffix，它在两种渲染里完全一样，所以按「与纯文本渲染相同的
            # 尾巴」剥掉即可。
            bsuf = base_fmt[tp:]
            wsuf = withc[cp:]
            if wsuf.endswith(bsuf) and bsuf:
                pat = wsuf[:-len(bsuf)]
            else:
                pat = wsuf
            pat = re.sub(rf"^{kind.capitalize()} \d+:\s*", "", pat)
            if pat:
                content_types[kind] = {"format": pat}

    roles = {
        "system": {"prefix": sys_prefix, "suffix": sys_suffix},
        "user": {"prefix": usr_prefix, "suffix": usr_suffix},
        "assistant": {"prefix": ast_prefix, "suffix": ast_suffix},
    }
    for name in ("system", "user", "assistant"):
        r = roles[name]
        print(f"[+] role {name:9s} prefix={r['prefix']!r} suffix={r['suffix']!r}")
    print(f"[+] generation_prompt={generation_prompt!r} "
          f"default_system_prompt={default_system_prompt!r}")

    return {
        "model_path": MODEL_DIR,
        "roles": roles,
        "content_types": content_types,
        "generation_prompt": generation_prompt,
        "default_system_prompt": default_system_prompt,
    }


# tokenizer.cpp 里两个手写扫描器各自的正则分支表。分支 = 顶层 '|' 切出来的
# alternative，顺序即 Rust regex 的 leftmost-first 优先级，所以比对必须逐分支、
# 按位置比，不能只判「有没有出现过」。
#   GPT-2 家族（Qwen3-VL / GLM-Edge / Llama-3）
GPT2_BRANCHES = [
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)",
    r"[^\r\n\p{L}\p{N}]?\p{L}+",
    None,                                    # 数字段，见下
    r" ?[^\s\p{L}\p{N}]+[\r\n]*",
    r"\s*[\r\n]+",
    r"\s+(?!\S)",
    r"\s+",
]
GPT2_DIGITS = {r"\p{N}": False, r"\p{N}{1,3}": True}   # 值 = bit2（数字段切 1~3 个）
#   Phi-4 / o200k 家族：字母按大小写「形状」切两段，数字固定 \p{N}{1,3}（扫描器
#   内部就切 1~3 个，bit2 对它没有意义），标点尾巴多一个 '/'。
PHI_BRANCHES = [
    r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*"
    r"[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?",
    r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+"
    r"[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?",
    r"\p{N}{1,3}",
    r" ?[^\s\p{L}\p{N}]+[\r\n/]*",
    r"\s*[\r\n]+",
    r"\s+(?!\S)",
    r"\s+",
]
# 每个家族对应的 Split 段 behavior/invert：GPT-2 用 Isolated/false（匹配到的 span
# 保留、分隔符丢弃），Phi 的 o200k 写成 Removed/invert=true，语义上等价但字段不同，
# 认错了会走错扫描器，所以一并进表。
PRE_TOKENIZER_FAMILIES = {
    "gpt2": (GPT2_BRANCHES, ("Isolated", False)),
    "phi": (PHI_BRANCHES, ("Removed", True)),
}


def _split_stage(data):
    r"""取 pre_tokenizer 里的 Split 段，返回 (正则, behavior, invert)。

    Sequence 里其余子段只允许 ByteLevel(add_prefix_space=false)：add_prefix_space
    会真的往首个 span 前插一个空格、改变切分结果，C++ 侧没有实现它。
    """
    pt = data.get("pre_tokenizer") or {}
    stages = pt.get("pretokenizers") if pt.get("type") == "Sequence" else [pt]
    splits = [s for s in stages or [] if s.get("type") == "Split"]
    if len(splits) != 1:
        raise ValueError(f"pre_tokenizer 需要恰好一个 Split 段，实际 {len(splits)} 个")
    for s in stages or []:
        if s.get("type") != "Split" and s.get("add_prefix_space"):
            raise ValueError("ByteLevel add_prefix_space=true 未实现")
    split = splits[0]
    pattern = (split.get("pattern") or {}).get("Regex")
    if not pattern:
        raise ValueError("pre_tokenizer 的 Split 段不是 Regex 模式")
    return pattern, split.get("behavior"), split.get("invert", False)


def _split_alternatives(pattern):
    r"""按顶层 '|' 切正则。括号内的 '|' 属于内层选择（如 (?i:'s|'t|...)），要跳过。"""
    parts, depth, cur = [], 0, []
    for ch in pattern:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if ch == "|" and depth == 0:
            parts.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    parts.append("".join(cur))
    return parts


def derive_flags(data):
    r"""推导 bin header 的 flags（位定义见 tokenizer.cpp 的 kFlag*，0 = Qwen 口径）。

    不靠模型名硬编码，全部从 tokenizer.json 的实际配置读，且逐分支比对已知表：
      bit0  pre_tokenizer 是 Phi-4/o200k 变体（字母按大小写形状切分 + 标点尾 '/'）。
      bit1  没有 normalizer —— C++ 侧跳过 NFC（Phi 的词表按原样码点建，做 NFC 会把
            「é」这类组合序列换成另一个码点，切出不同的 token）。
      bit2  GPT-2 扫描器的数字段写成 \p{N}{1,3}（GLM-Edge、Llama-3）而不是 \p{N}
            （Qwen3-VL）。

    表外的正则形状一律报错，不做「看起来像」的猜测：切错 span 只会静默产出错 token，
    数值链路一路跑完都发现不了，必须在生成 bin 这一步就炸。早先用
    `"\\p{N}{1,3}" in 全文` 判 Phi 就是这么把 GLM-Edge 误判成 Phi 的 —— 它和 Qwen
    只差数字段这一处，字母分支却完全是 GPT-2 口径。
    """
    regex, behavior, invert = _split_stage(data)
    branches = _split_alternatives(regex)
    no_normalizer = data.get("normalizer") is None

    family = None
    for name, (table, want_bi) in PRE_TOKENIZER_FAMILIES.items():
        if len(branches) != len(table):
            continue
        if table is GPT2_BRANCHES:
            ok = (branches[:2] == table[:2] and branches[3:] == table[3:]
                  and branches[2] in GPT2_DIGITS)
        else:
            ok = branches == table
        if ok:
            if (behavior, invert) != want_bi:
                raise ValueError(
                    f"{name} 家族的 pre_tokenizer 要求 behavior/invert="
                    f"{want_bi}，实际 {(behavior, invert)}")
            family = name
            break

    if family is None:
        diff = ""
        if len(branches) == len(GPT2_BRANCHES):
            bad = [i for i in range(len(branches))
                   if not (GPT2_BRANCHES[i] is None
                           and branches[i] in GPT2_DIGITS)
                   and branches[i] != GPT2_BRANCHES[i]]
            diff = ("与 GPT-2 表不同的分支："
                    + ", ".join(f"[{i}] {branches[i]!r} != {GPT2_BRANCHES[i]!r}"
                                for i in bad))
        raise ValueError(
            "无法识别的 pre_tokenizer 正则：既不是已实现的 GPT-2 扫描器，也不是 "
            "Phi/o200k 扫描器。\n"
            f"  behavior={behavior!r} invert={invert!r}\n"
            + "\n".join(f"  [{i}] {b!r}" for i, b in enumerate(branches)) + "\n"
            + (diff + "\n" if diff else "")
            + "只有 GPT-2 家族的数字段量化上限允许差异；其余分支必须逐字符一致。"
              "新形状请先在 tokenizer.cpp 实现对应扫描器，再进本表。")

    digit3 = family == "gpt2" and GPT2_DIGITS[branches[2]]
    flags = (1 if family == "phi" else 0) | (2 if no_normalizer else 0) \
        | (4 if digit3 else 0)
    print(f"[+] pre_tokenizer={family} digits={'1-3' if digit3 else '1'} "
          f"normalizer={'none' if no_normalizer else 'present'} -> flags={flags}")
    return flags


def convert_tokenizer():
    print(f"[*] Loading tokenizer from: {TOKENIZER_JSON}")
    with open(TOKENIZER_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 1. 提取词表 (Vocab)
    vocab_dict = data["model"]["vocab"]
    sorted_vocab = sorted(vocab_dict.items(), key=lambda x: x[1])
    vocab_size = len(sorted_vocab)

    # 2. 提取合并规则 (Merges)
    merges_list = data["model"]["merges"]
    merge_rules_count = len(merges_list)

    # 3. 提取特殊 Token
    added_tokens = data.get("added_tokens", [])
    special_tokens_count = len(added_tokens)

    print(f"[+] Vocab Size: {vocab_size}")
    print(f"[+] Merge Rules: {merge_rules_count}")
    print(f"[+] Special Tokens: {special_tokens_count}")

    # 构建反向映射表 (Unicode Char -> Byte)
    b2u = bytes_to_unicode()
    u2b = {v: k for k, v in b2u.items()}
    flags = derive_flags(data)

    with open(OUTPUT_BIN, "wb") as f:
        # --- 写入 File Header ---
        f.write(b"QW3T")                  # Magic
        f.write(struct.pack("<I", 1))     # Version
        f.write(struct.pack("<I", vocab_size))
        f.write(struct.pack("<I", merge_rules_count))
        f.write(struct.pack("<I", special_tokens_count))
        f.write(struct.pack("<I", flags))  # 原 Reserved 字段，见 derive_flags

        # --- 写入 Vocab Section ---
        for token, token_id in sorted_vocab:
            token_bytes = token.encode("utf-8")
            f.write(struct.pack("<H", len(token_bytes)))
            f.write(token_bytes)
            f.write(struct.pack("<I", token_id))

        # --- 写入 Merge Rules Section ---
        skip_count = 0
        for merge_entry in merges_list:
            # Qwen3-4B+ 的 tokenizer.json 用 list 格式 ["left","right"]，
            # Qwen3-VL 用 string 格式 "left right"。统一处理。
            if isinstance(merge_entry, list):
                parts = merge_entry
            else:
                parts = merge_entry.split(" ", 1)
            if len(parts) != 2:
                skip_count += 1
                f.write(struct.pack("<III", 0, 0, 0))
                continue
                
            left_str, right_str = parts
            
            # 核心修复：直接在 vocab_dict 中查找，而不是尝试 decode('utf-8')
            # 因为 vocab_dict 的 key 就是 BBPE 原始字符串
            left_id = vocab_dict.get(left_str)
            right_id = vocab_dict.get(right_str)
            
            # 合并后的字符串就是 left + right (中间去掉空格)
            merged_str = left_str + right_str
            merged_id = vocab_dict.get(merged_str)
            
            if left_id is not None and right_id is not None and merged_id is not None:
                f.write(struct.pack("<III", left_id, right_id, merged_id))
            else:
                # 理论上不应该发生，如果发生说明 tokenizer.json 本身数据不一致
                skip_count += 1
                f.write(struct.pack("<III", 0, 0, 0))

        # --- 写入 Special Tokens Section ---
        for token_info in added_tokens:
            token = token_info["content"]
            token_id = token_info["id"]
            token_bytes = token.encode("utf-8")
            f.write(struct.pack("<H", len(token_bytes)))
            f.write(token_bytes)
            f.write(struct.pack("<I", token_id))

        # --- 写入 ChatTemplate Section（可选，追加在文件末尾）---
        # 用 HF apply_chat_template 探针提取 system/user/assistant 各角色的
        # prefix/suffix、image/video 内容占位格式、generation_prompt、
        # default_system_prompt，序列化为长度前缀字段，C++ load 顺序读取。
        try:
            chat_data = extract_chat_template()
        except Exception as e:
            print(f"[!] Warning: failed to extract chat template: {e}. Skipping ChatTemplate section.")
            chat_data = None

        if chat_data is not None:
            # roles: u32 count + (u16 name + u16 prefix + u16 suffix) per role
            roles = chat_data["roles"]
            f.write(struct.pack("<I", len(roles)))
            for name in ("system", "user", "assistant"):
                if name not in roles:
                    continue
                for s in (name, roles[name]["prefix"], roles[name]["suffix"]):
                    b = s.encode("utf-8")
                    f.write(struct.pack("<H", len(b)))
                    f.write(b)
            # content_types: u32 count + (u16 type + u16 format) per type
            cts = chat_data["content_types"]
            f.write(struct.pack("<I", len(cts)))
            for tname, tinfo in cts.items():
                for s in (tname, tinfo["format"]):
                    b = s.encode("utf-8")
                    f.write(struct.pack("<H", len(b)))
                    f.write(b)
            # generation_prompt + default_system_prompt
            for s in (chat_data["generation_prompt"], chat_data["default_system_prompt"]):
                b = s.encode("utf-8")
                f.write(struct.pack("<H", len(b)))
                f.write(b)
            print(f"[✓] Chat template embedded: {len(roles)} roles, {len(cts)} content types.")

    print(f"[✓] Successfully converted to: {OUTPUT_BIN}")
    if skip_count > 0:
        print(f"[!] Warning: Skipped {skip_count} invalid merge rules.")
    else:
        print(f"[✓] All {merge_rules_count} merge rules parsed successfully! 0 Skips!")

if __name__ == "__main__":
    convert_tokenizer()