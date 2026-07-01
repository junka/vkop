import json
import struct
import os

MODEL_DIR = "/Users/uqland/.cache/modelscope/hub/models/Qwen/Qwen3-VL-2B-Instruct"
TOKENIZER_JSON = os.path.join(MODEL_DIR, "tokenizer.json")
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

    print(f"[✓] Successfully converted to: {OUTPUT_BIN}")
    if skip_count > 0:
        print(f"[!] Warning: Skipped {skip_count} invalid merge rules.")
    else:
        print(f"[✓] All {merge_rules_count} merge rules parsed successfully! 0 Skips!")

if __name__ == "__main__":
    convert_tokenizer()