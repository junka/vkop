"""用真实 text encoder 权重（Qwen3VLForConditionalGeneration）算 prompt_embeds。

复刻 diffusers `QwenImage21Pipeline._get_qwen_prompt_embeds` 的纯文生图路径：
raw 模板字符串（不走 apply_chat_template，两者分词不同）、丢掉 system 段的
`drop_idx` 个 token、**绕开文本塔最后的 RMSNorm**（transformers 5.x 会把
`hidden_states[-1]` 绑到已归一化的 last_hidden_state，官方用 forward hook 中和）。

输出写进 --out-dir：
  prompt_embeds.raw  fp16 [1, prefix_len, 4096]（不足右补零，超出报错）
  prompt_valid_len.txt  真实有效 token 数 L（补零位必须在 bias 里屏蔽掉）
  prompt_tokens.txt  逐 token 的 id / surface，便于核对分词

特殊 token 一律从 tokenizer 的属性/表里取，脚本里不写字面量。
"""

import argparse
import json
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
MS_ROOT = Path("/Users/doudou/qi21-env/ms/models/Qwen--Qwen-Image-2.1/snapshots/master")
SYS_PROMPT = "Comprehend and analyze the provided prompt."


def special(tok, key_part):
    """从 tokenizer 的特殊 token 集合里取形如 `<...key_part...>` 的那个，避免硬编码字面量。"""
    cands = set()
    for attr in ("additional_special_tokens", "all_special_tokens"):
        cands |= {t for t in getattr(tok, attr, None) or []}
    for t in json.loads((MS_ROOT / "processor" / "added_tokens.json").read_text()):
        cands.add(t)
    hits = {t for t in cands if key_part in t}
    if len(hits) != 1:
        raise KeyError(f"{key_part}: {sorted(hits)}")
    return hits.pop()


def build_template(tok):
    """官方 pipeline 的纯文生图模板（`{}` 处填 prompt）。**不能**换成
    apply_chat_template —— 两者分词不同。"""
    im_start, im_end = special(tok, "im_start"), special(tok, "im_end")
    return (f"{im_start}system\n{SYS_PROMPT}{im_end}\n"
            f"{im_start}user\n{{}}{im_end}\n"
            f"{im_start}assistant\n")


def drop_idx_of(proc):
    """tokenized system message 的长度（与官方同源，含模板里的 system 段）。"""
    sys_message = [{"role": "system", "content": [{"type": "text", "text": SYS_PROMPT}]}]
    return len(proc.apply_chat_template(sys_message, tokenize=True, return_dict=False)[0])


def tokenize_prompt(proc, prompt):
    """raw 模板串 + 单条序列（batch=1 故 padding 实际不生效），左侧 pad。"""
    return proc(text=[build_template(proc.tokenizer).format(prompt or " ")],
                padding=True, padding_side="left", return_tensors="pt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--prefix-len", type=int, default=64)
    ap.add_argument("--out-dir", default=str(HERE / "ref_real"))
    ap.add_argument("--dump-prefill", action="store_true",
                    help="额外存 prefill 段 rope 表（cos/sin）和层输入，供 ORT 对齐")
    ap.add_argument("--tokens-only", action="store_true",
                    help="只分词、不加载 17.5 GB 权重，用来挑一条刚好 64 token 的 prompt")
    args = ap.parse_args()

    from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    proc = Qwen3VLProcessor.from_pretrained(str(MS_ROOT / "processor"))
    tok = proc.tokenizer
    drop_idx = drop_idx_of(proc)

    model_inputs = tokenize_prompt(proc, args.prompt)
    real_len = int(model_inputs.attention_mask.sum()) - drop_idx
    if args.tokens_only:
        ids = model_inputs.input_ids[0].tolist()[drop_idx:]
        print(f"[tokens] drop_idx={drop_idx} 模板+prompt 有效 token={real_len} "
              f"(prefix_len={args.prefix_len})")
        print(f"[tokens] 尾部 12 个 surface: {[tok.decode([i]) for i in ids[-12:]]}")
        return

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        str(MS_ROOT / "text_encoder"), dtype=torch.bfloat16)
    model.eval()

    text_model = getattr(model.model, "language_model", model.model)
    handle = text_model.norm.register_forward_hook(lambda m, a, o: a[0])
    try:
        with torch.no_grad():
            outputs = model(
                input_ids=model_inputs.input_ids,
                attention_mask=model_inputs.attention_mask,
                output_hidden_states=True,
            )
    finally:
        handle.remove()

    hidden = outputs.hidden_states[-1]
    mask = model_inputs.attention_mask.bool()
    emb = hidden[0][mask[0]][drop_idx:]          # (L, 4096)，去掉 padding 与 system 段
    real_len = emb.shape[0]
    ids = model_inputs.input_ids[0][mask[0]][drop_idx:].tolist()

    if real_len > args.prefix_len:
        raise SystemExit(f"prompt 实际 {real_len} token > prefix_len={args.prefix_len}，"
                         "请缩短 prompt 或重转更大的 vkopbin")
    pad = args.prefix_len - real_len
    if pad:
        emb = torch.cat([emb, emb.new_zeros(pad, emb.shape[1])], dim=0)
        print(f"[embeds] 右补 {pad} 行零；decode 的 attention_bias 必须屏蔽这 {pad} 列")

    emb16 = emb.to(torch.float16).reshape(1, args.prefix_len, -1).contiguous()
    emb16.numpy().tofile(out / "prompt_embeds.raw")
    (out / "prompt_valid_len.txt").write_text(f"{real_len}\n")
    surfaces = [tok.decode([i]) for i in ids]
    (out / "prompt_tokens.txt").write_text(
        "\n".join(f"{k}\t{i}\t{s!r}" for k, (i, s) in enumerate(zip(ids, surfaces))) + "\n")

    print(f"[embeds] prompt={args.prompt!r}")
    print(f"[embeds] drop_idx={drop_idx} total_tokens={int(mask[0].sum())} real_len={real_len} "
          f"-> padded {args.prefix_len}")
    print(f"[embeds] dtype fp16 shape {tuple(emb16.shape)} absmax={emb16.abs().max().item():.4f} "
          f"rms={emb16.float().pow(2).mean().sqrt().item():.4f}")
    print(f"[embeds] first10 tokens: {surfaces[:10]}")
    print(f"[embeds] wrote {out / 'prompt_embeds.raw'}")


if __name__ == "__main__":
    main()
