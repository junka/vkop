#!/usr/bin/env python3
"""ORT 张量统计，与 C++ 驱动 VKOP_DUMP_TENSORS 的 [name] 行同格式，逐行 diff。

两种输入模式：
  --round DIR   读 dump_llm_decode.py 落盘的 feed（input_names.txt + *.npy），
                prefill / decode 任意一轮都适用
  --prompt TEXT 由 AutoTokenizer + embed_tokens.bin 构造纯文本 prefill feed
                （与 check/greedy_ref.py 同一套构造），默认 "hi"

输出范围：
  默认          图原有 outputs（logits + present_key_values_*）
  --intermediates  全部节点输出追加为 graph outputs，定位第一个发散 op。
                增强图 load_external_data=False 保留外部引用，aug.onnx 落在
                源 ONNX 同目录原位解析权重——零权重拷贝（把 3.4GB 权重重写进
                临时目录在本机磁盘放不下），用完即删。

统计行按名字排序保证可 join。fp16 输出 ne/nan/inf/zero/min/max/first(hex)；
非 fp16 在默认模式打印 dtype/shape，--intermediates 模式跳过（C++ 侧只 dump
fp16，保持可比）。

用法:
    python3 check/ort_stats.py --prompt "你好" > /tmp/ort_prefill_stats.txt
    python3 check/ort_stats.py --round ../decode_dump/round1 --intermediates
"""
import argparse
import os
import sys

import numpy as np
import onnx
import onnxruntime as ort

HERE = os.path.dirname(os.path.abspath(__file__))
LLM_DIR = os.path.normpath(os.path.join(HERE, "..", ".."))
DEFAULT_ONNX = os.path.join(LLM_DIR, "llm.onnx")
DEFAULT_EMBED = os.path.join(LLM_DIR, "exporter", "embed_tokens.bin")
DEFAULT_MODEL = os.path.expanduser(
    "~/.cache/modelscope/hub/models/Qwen/Qwen3-VL-2B-Instruct")


def fp16_stats(arr):
    f = arr.astype(np.float32)
    nan = int(np.isnan(f).sum())
    inf = int(np.isinf(f).sum())
    zero = int((f == 0).sum())
    finite = f[np.isfinite(f)]
    mn = float(finite.min()) if finite.size else 0.0
    mx = float(finite.max()) if finite.size else 0.0
    first = arr.reshape(-1)[:4].astype(np.uint16)
    first_hex = ",".join(f"{x:04x}" for x in first)
    return (f"ne={f.size} nan={nan} inf={inf} zero={zero} "
            f"min={mn:.4g} max={mx:.4g} first=[{first_hex}]")


def static_dim(vi, idx):
    d = vi.shape[idx]
    if not isinstance(d, int):
        sys.exit(f"[!] {vi.name} 第 {idx} 维是动态维 ({d})，无法构造 feed")
    return d


def build_prompt_feed(args, in_set, by_name):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    text = tok.apply_chat_template([{"role": "user", "content": args.prompt}],
                                   tokenize=False, add_generation_prompt=True)
    ids = tok(text, add_special_tokens=False)["input_ids"]
    print(f"[prompt] {len(ids)} tokens", file=sys.stderr)

    hidden = static_dim(by_name["inputs_embeds"], 2)
    emb_table = np.fromfile(args.embed, dtype=np.float16).reshape(-1, hidden)
    L = len(ids)
    pos = np.arange(L, dtype=np.int64)
    # 纯文本 Qwen3/Phi-4/GLM 是 2D (B, q)；Qwen3-VL 是 3D mrope (3, B, q)，
    # 文本 token 三轴位置相等，三行重复即退化为标准 RoPE。
    pos_feed = (np.stack([pos] * 3)[:, None, :]
                if len(by_name["position_ids"].shape) == 3 else pos[None])
    feed = {
        "inputs_embeds": np.ascontiguousarray(emb_table[ids][None]),
        "position_ids": np.ascontiguousarray(pos_feed),
        "attention_bias": np.ascontiguousarray(np.triu(
            np.full((L, L), np.finfo(np.float16).min, dtype=np.float16),
            k=1)[None, None]),
    }
    # VL 图的纯文本退化输入；不在图输入里的会被下面的过滤丢掉
    for i in range(3):
        feed[f"deepstack_embeds_{i}"] = np.zeros((1, hidden), dtype=np.float16)
    feed["image_pad_mask"] = np.zeros((1, L), dtype=bool)
    past_names = sorted((n for n in in_set if n.startswith("past_key_values_")),
                        key=lambda n: int(n.rsplit("_", 1)[1]))
    for n in past_names:
        nkv, hd = static_dim(by_name[n], 2), static_dim(by_name[n], 4)
        feed[n] = np.zeros((1, 2, nkv, 0, hd), dtype=np.float16)
    return {k: v for k, v in feed.items() if k in in_set}


def load_round_feed(round_dir, in_set):
    feed = {}
    with open(os.path.join(round_dir, "input_names.txt")) as f:
        names = [l.strip() for l in f if l.strip()]
    for name in names:
        arr = np.load(os.path.join(round_dir, name + ".npy"))
        if name in in_set:
            feed[name] = arr
    return feed


def augmented(onnx_path):
    m = onnx.load(onnx_path, load_external_data=False)
    want = sorted({o for n in m.graph.node for o in n.output if o})
    existing = {o.name for o in m.graph.output}
    for name in want:
        if name not in existing:
            m.graph.output.add().name = name
    aug = os.path.join(os.path.dirname(os.path.abspath(onnx_path)),
                       "aug_ort_stats.onnx")
    onnx.save(m, aug)   # 外部引用相对源目录解析，不复制权重
    print(f"[aug] {len(m.graph.output)} outputs", file=sys.stderr)
    return aug


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--round", metavar="DIR",
                   help="dump_llm_decode.py 的轮目录（input_names.txt + *.npy）")
    g.add_argument("--prompt", default="hi",
                   help="纯文本 prefill prompt（默认 hi）")
    ap.add_argument("--intermediates", action="store_true",
                    help="统计全部节点输出，而不只是图原有 outputs")
    ap.add_argument("--onnx", default=DEFAULT_ONNX)
    ap.add_argument("--embed", default=DEFAULT_EMBED)
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help="tokenizer 来源目录（--prompt 模式用）")
    args = ap.parse_args()

    aug = None
    try:
        if args.intermediates:
            aug = augmented(args.onnx)
            sess = ort.InferenceSession(aug, providers=["CPUExecutionProvider"])
        else:
            sess = ort.InferenceSession(args.onnx,
                                        providers=["CPUExecutionProvider"])
        in_set = {i.name for i in sess.get_inputs()}
        by_name = {i.name: i for i in sess.get_inputs()}
        if args.round:
            feed = load_round_feed(args.round, in_set)
            print(f"[round] {args.round}: {len(feed)} inputs", file=sys.stderr)
        else:
            feed = build_prompt_feed(args, in_set, by_name)
        out_names = [o.name for o in sess.get_outputs()]
        print(f"[run] {len(feed)} inputs, {len(out_names)} outputs",
              file=sys.stderr)
        res = sess.run(out_names, feed)
    finally:
        if aug and os.path.exists(aug):
            os.remove(aug)

    lines = []
    for name, arr in zip(out_names, res):
        if arr.dtype == np.float16:
            lines.append((name, f"[{name}] {fp16_stats(arr)}"))
        elif not args.intermediates:
            lines.append((name, f"[{name}] dtype={arr.dtype} shape={arr.shape}"))
    for _, line in sorted(lines):
        print(line)


if __name__ == "__main__":
    main()