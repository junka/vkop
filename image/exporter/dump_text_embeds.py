"""从 Qwen-Image-2.1 文本塔 checkpoint 里只抽 embed_tokens（不建模型、不重跑导出）。

文本塔的 ONNX/vkopbin 输入是 `inputs_embeds`（查表已经在图外），所以 C++ 驱动需要
一张 [151936, 4096] 的 fp16 表做 token_id→embed，和 llm 线的 embed_tokens.bin 同形。
同一条路径顺带能按给定 token id 产出 vkop 侧的对拍输入（`--ids-file`），这样"驱动喂
进去的那 78 行"和"torch 喂进去的那 78 行"是同一个文件，而不是两份各自查表的产物。

    # 驱动用的查表（默认 1.24 GB）
    /Users/doudou/qi21-env/bin/python dump_text_embeds.py --out-bin text_encoder_embeds.bin
    # 对拍输入：ref_real 那次的 78 个 token -> (1,78,4096) fp16
    /Users/doudou/qi21-env/bin/python dump_text_embeds.py \
        --ids-file /tmp/ref_real_ids.txt --out-embeds ref_te/te_embeds.raw

只 `safe_open` 一个张量：checkpoint 是 4 个分片共 17.5 GB，而 embed_tokens 只有 1.24 GB，
用 `from_pretrained` 把整个模型拉起来（就像 llm/exporter/dump_embed_tokens.py 那样）在这
台 36 GB 的机器上等于把交换区打开。
"""

import argparse
import sys
from pathlib import Path

import torch
from safetensors import safe_open

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from qi21_export_text_encoder_onnx import CKPT, EMBED, KEY_PREFIX  # noqa: E402

KEY = KEY_PREFIX + EMBED          # model.language_model.embed_tokens.weight
VOCAB, HIDDEN = 151936, 4096


def read_embed_tokens() -> torch.Tensor:
    """按 index.json 找到那张表所在的分片，只读它一个。"""
    import json
    index = json.loads((CKPT / "model.safetensors.index.json").read_text())["weight_map"]
    assert KEY in index, f"{KEY} 不在 checkpoint 里"
    shard = CKPT / index[KEY]
    with safe_open(shard, framework="pt") as f:
        w = f.get_tensor(KEY)
    assert tuple(w.shape) == (VOCAB, HIDDEN), w.shape
    w = w.to(torch.float16).contiguous()
    print(f"[embed] {index[KEY]}: {KEY} {tuple(w.shape)} bf16->fp16 "
          f"absmax={w.abs().max():.4f}")
    return w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-bin", default=None, help="写这张表（裸 fp16 [vocab,hidden]）")
    ap.add_argument("--ids-file", default=None,
                    help="逗号分隔的完整 token id 序列（模板+prompt，含前 drop_idx 个）")
    ap.add_argument("--out-embeds", default=None, help="查表结果写到这个 raw 文件（fp16）")
    args = ap.parse_args()
    if not (args.out_bin or args.ids_file):
        raise SystemExit("至少给 --out-bin 或 --ids-file 之一")

    w = read_embed_tokens()
    if args.out_bin:
        out = Path(args.out_bin)
        out.parent.mkdir(parents=True, exist_ok=True)
        w.numpy().tofile(out)
        print(f"[embed] wrote {out} ({out.stat().st_size:,} B)")

    if args.ids_file:
        ids = torch.tensor([int(x) for x in Path(args.ids_file).read_text().split(",")],
                           dtype=torch.long)
        assert (ids >= 0).all() and (ids < VOCAB).all(), "token id 越界"
        emb = w[ids]                              # (n_token, hidden)
        out = Path(args.out_embeds or "te_embeds.raw")
        out.parent.mkdir(parents=True, exist_ok=True)
        emb.contiguous().numpy().tofile(out)
        print(f"[embed] {tuple(emb.shape)} fp16 -> {out}（{out.stat().st_size:,} B，"
              f"图长 P 必须 == {ids.shape[0]}）")


if __name__ == "__main__":
    main()
