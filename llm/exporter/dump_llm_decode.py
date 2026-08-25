#!/usr/bin/env python3
"""Dump prefill + N decode rounds for the C++ vkop driver.

Text-only (no image). Mirrors llm/exporter/infer.py::generate but, instead of
looping inside one ORT session, exports each round's full input set + reference
logits as .npy under <out_dir>/round{t}/. The C++ driver loads each round,
uploads it, runs one Run(), and compares logits — verifying the buffer-backend
ops (esp. the 5-D Gather on growing [1,2,8,kv_len,128] KV-cache and the fp16
Expand) hold up across decode rounds where kv_len increments.

Usage:
  python3 dump_llm_decode.py <text> <out_dir> [max_new]

Writes:
  <out_dir>/round0/   prefill (kv_len=0)
  <out_dir>/round1/   first decode step (kv_len=L)
  <out_dir>/round2/   second decode step (kv_len=L+1)
  ...
  <out_dir>/summary.txt   per-round argmax + token text
"""
import os
import sys

import numpy as np
import torch
import onnxruntime as ort
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
LLM_ONNX = os.path.join(_PKG_DIR, "llm.onnx")
MODEL = os.path.expanduser("~/.cache/modelscope/hub/models/Qwen/Qwen3-VL-2B-Instruct")

NLAYERS = 28
HIDDEN = 2048
NKV = 8
HD = 128
IM_END = 151645


def n16(x):
    return x.cpu().numpy().astype(np.float16)


def causal_bias(q_len, kv_len):
    # (1,1,q,kv) additive mask: upper-triangular = finfo.min.
    m = torch.triu(
        torch.full((q_len, kv_len), torch.finfo(torch.float16).min, dtype=torch.float16),
        diagonal=kv_len - q_len + 1)
    return m.unsqueeze(0).unsqueeze(0)


def dump_round(out_dir, feed, logits, ort_inputs_names):
    os.makedirs(out_dir, exist_ok=True)
    for name, arr in feed.items():
        np.save(os.path.join(out_dir, name + ".npy"), arr)
    np.save(os.path.join(out_dir, "reference_logits.npy"), logits)
    with open(os.path.join(out_dir, "input_names.txt"), "w") as f:
        for name in feed:
            f.write(name + "\n")


def main():
    text = sys.argv[1] if len(sys.argv) > 1 else "Hello"
    out_dir = sys.argv[2] if len(sys.argv) > 2 else os.path.join(_PKG_DIR, "decode_dump")
    max_new = int(sys.argv[3]) if len(sys.argv) > 3 else 8

    print("Loading processor + reference model...")
    proc = AutoProcessor.from_pretrained(MODEL)
    ref = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL, attn_implementation="eager", torch_dtype=torch.float16).eval()
    lm = ref.model.language_model  # for embed_tokens + get_rope_index

    inputs = proc(text=[text], images=None, do_resize=False, padding=True, return_tensors="pt")
    input_ids = inputs.input_ids  # (1, L)
    am = inputs.attention_mask
    mtt = getattr(inputs, "mm_token_type_ids", torch.zeros_like(input_ids, dtype=torch.int32))
    grid_thw = getattr(inputs, "image_grid_thw", None)
    L = input_ids.shape[1]
    print(f"prompt: {text!r}  L={L}  max_new={max_new}")

    image_pad_mask = torch.zeros_like(input_ids, dtype=torch.bool)

    with torch.no_grad():
        emb = lm.embed_tokens(input_ids)  # (1, L, H) fp16
    inputs_embeds = emb

    with torch.no_grad():
        pos_ids, rope_delta = ref.model.get_rope_index(
            input_ids=input_ids, mm_token_type_ids=mtt,
            image_grid_thw=grid_thw, attention_mask=am)
    delta = int(rope_delta[0, 0].item()) if rope_delta is not None else 0
    print(f"rope_delta={delta}")

    ds_zero = [np.zeros((1, HIDDEN), dtype=np.float16) for _ in range(3)]

    print("Loading ONNX Runtime session...")
    so = ort.SessionOptions()
    lsess = ort.InferenceSession(LLM_ONNX, sess_options=so, providers=["CPUExecutionProvider"])
    input_name_set = {i.name for i in lsess.get_inputs()}

    # ---- Round 0: prefill (kv_len=0) ----
    ab = causal_bias(L, L)
    past = [np.zeros((1, 2, NKV, 0, HD), dtype=np.float16) for _ in range(NLAYERS)]

    def make_feed(inputs_emb, pos, attn_bias, past_kv, mask):
        feed = {
            "inputs_embeds": n16(inputs_emb),
            "position_ids": pos.cpu().numpy().astype(np.int64),
            "attention_bias": n16(attn_bias),
            "deepstack_embeds_0": ds_zero[0],
            "deepstack_embeds_1": ds_zero[1],
            "deepstack_embeds_2": ds_zero[2],
            "image_pad_mask": mask.cpu().numpy().astype(bool),
        }
        for i in range(NLAYERS):
            feed[f"past_key_values_{i}"] = past_kv[i]
        return {k: v for k, v in feed.items() if k in input_name_set}

    feed0 = make_feed(inputs_embeds, pos_ids, ab, past, image_pad_mask)
    print(f"[round0] prefill kv_len=0  feed={len(feed0)} inputs")
    res0 = lsess.run(None, feed0)
    logits0 = res0[0]  # (1, L, vocab)
    presents = list(res0[1:1 + NLAYERS])
    next_id = int(np.argmax(logits0[0, -1]))
    dump_round(os.path.join(out_dir, "round0"), feed0, logits0, input_name_set)

    out_ids = [next_id]
    past = [p for p in presents]
    past_len = L
    decode_mask = torch.zeros((1, 1), dtype=torch.bool)  # decode: no image pad

    # ---- Decode rounds 1..max_new-1 ----
    for step in range(1, max_new):
        if next_id == IM_END:
            print(f"[round{step}] IM_END reached, stopping")
            break
        cur = torch.tensor([[next_id]], dtype=torch.long)
        with torch.no_grad():
            cur_emb = lm.embed_tokens(cur)  # (1, 1, H)
        cur_pos = torch.full((3, 1, 1), past_len + delta, dtype=torch.long)
        # decode attention_bias: (1,1,1,past_len+1) all-zero (full history visible)
        cur_ab = torch.zeros((1, 1, 1, past_len + 1), dtype=torch.float16)

        feed_t = make_feed(cur_emb, cur_pos, cur_ab, past, decode_mask)
        print(f"[round{step}] decode kv_len={past_len} q_len=1  pos={past_len + delta}")
        res_t = lsess.run(None, feed_t)
        logits_t = res_t[0]  # (1, 1, vocab)
        presents = list(res_t[1:1 + NLAYERS])
        next_id = int(np.argmax(logits_t[0, -1]))
        dump_round(os.path.join(out_dir, f"round{step}"), feed_t, logits_t, input_name_set)

        out_ids.append(next_id)
        past = [p for p in presents]
        past_len += 1

    # ---- summary ----
    with open(os.path.join(out_dir, "summary.txt"), "w") as f:
        f.write(f"prompt: {text}\n")
        f.write(f"generated ids: {out_ids}\n")
        f.write(f"text: {proc.decode(out_ids, skip_special_tokens=True)}\n")
        for i, tid in enumerate(out_ids):
            f.write(f"  round{i}: argmax={tid}\n")
    print(f"\ngenerated ids: {out_ids}")
    print(f"text: {proc.decode(out_ids, skip_special_tokens=True)}")
    print(f"Dumped {len(out_ids)} rounds to {out_dir}")


if __name__ == "__main__":
    main()
