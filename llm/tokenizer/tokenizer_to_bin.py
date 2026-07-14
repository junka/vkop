import json
import struct
import os
import re

MODEL_DIR = os.path.expanduser("~/.cache/modelscope/hub/models/Qwen/Qwen3-VL-2B-Instruct")
TOKENIZER_JSON = os.path.join(MODEL_DIR, "tokenizer.json")
TOKENIZER_CONFIG_JSON = os.path.join(MODEL_DIR, "tokenizer_config.json")
CHAT_TEMPLATE_JSON = os.path.join(MODEL_DIR, "chat_template.json")
OUTPUT_BIN = "qwen3_vl.bin"

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

def extract_chat_template():
    """用 HF tokenizer 的 apply_chat_template 探针提取角色 prefix/suffix、
    内容占位格式、generation_prompt、default_system_prompt。

    返回结构与 trt_edgellm 的 processed_chat_template.json 一致：
      roles: {system/user/assistant: {prefix, suffix}}
      content_types: {image/video: {format}}
      generation_prompt, default_system_prompt
    """
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)

    SYS = "__SENTINEL_SYS_a7f3e2b1__"
    USR = "__SENTINEL_USR_c9d4f6e8__"
    AST = "<placeholder_assistant_text>"

    sys_msg = {"role": "system", "content": SYS}
    usr_msg = {"role": "user", "content": USR}
    ast_msg = {"role": "assistant", "content": AST}

    sys_fmt = tok.apply_chat_template([sys_msg], tokenize=False, add_generation_prompt=False)
    sys_prefix = sys_fmt[:sys_fmt.find(SYS)]
    sys_suffix = sys_fmt[sys_fmt.find(SYS) + len(SYS):]

    usr_fmt = tok.apply_chat_template([sys_msg, usr_msg], tokenize=False, add_generation_prompt=False)
    delta = usr_fmt[len(sys_fmt):]
    usr_prefix = delta[:delta.find(USR)]
    usr_suffix = delta[delta.find(USR) + len(USR):]

    ast_fmt = tok.apply_chat_template([sys_msg, usr_msg, ast_msg], tokenize=False, add_generation_prompt=False)
    delta2 = ast_fmt[len(usr_fmt):]
    ast_prefix = delta2[:delta2.find(AST)]
    ast_suffix = delta2[delta2.find(AST) + len(AST):]

    gen_fmt = tok.apply_chat_template([sys_msg, usr_msg], tokenize=False, add_generation_prompt=True)
    generation_prompt = gen_fmt[len(usr_fmt):]

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
    content_types = {}
    base_text = "<placeholder_user_text>"
    base_fmt = tok.apply_chat_template(
        [sys_msg, {"role": "user", "content": [{"type": "text", "text": base_text}]}],
        tokenize=False, add_generation_prompt=False)
    for kind, ph in [("image", "<placeholder_image_path>"), ("video", "<placeholder_video_path>")]:
        u = {"role": "user", "content": [{"type": "text", "text": base_text}, {"type": kind, kind: ph}]}
        withc = tok.apply_chat_template([sys_msg, u], tokenize=False, add_generation_prompt=False)
        tp = base_fmt.find(base_text) + len(base_text)
        cp = withc.find(base_text) + len(base_text)
        bsuf = base_fmt[tp:]
        wsuf = withc[cp:]
        if wsuf.endswith(bsuf) and bsuf:
            pat = wsuf[:-len(bsuf)]
        else:
            pat = wsuf
        pat = re.sub(rf"^{kind.capitalize()} \d+:\s*", "", pat)
        if pat:
            content_types[kind] = {"format": pat}

    return {
        "model_path": MODEL_DIR,
        "roles": {
            "system": {"prefix": sys_prefix, "suffix": sys_suffix},
            "user": {"prefix": usr_prefix, "suffix": usr_suffix},
            "assistant": {"prefix": ast_prefix, "suffix": ast_suffix},
        },
        "content_types": content_types,
        "generation_prompt": generation_prompt,
        "default_system_prompt": default_system_prompt,
    }


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

    with open(OUTPUT_BIN, "wb") as f:
        # --- 写入 File Header ---
        f.write(b"QW3T")                  # Magic
        f.write(struct.pack("<I", 1))     # Version
        f.write(struct.pack("<I", vocab_size))
        f.write(struct.pack("<I", merge_rules_count))
        f.write(struct.pack("<I", special_tokens_count))
        f.write(struct.pack("<I", 0))     # Reserved

        # --- 写入 Vocab Section ---
        for token, token_id in sorted_vocab:
            token_bytes = token.encode("utf-8")
            f.write(struct.pack("<H", len(token_bytes)))
            f.write(token_bytes)
            f.write(struct.pack("<I", token_id))

        # --- 写入 Merge Rules Section ---
        skip_count = 0
        for merge_str in merges_list:
            parts = merge_str.split(" ", 1)
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