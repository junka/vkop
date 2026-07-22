#!/usr/bin/env python3
"""用导出的 visual.onnx + llm.onnx 做端到端生成，与 HF Qwen3VLInference 对比一致性。

管线：
  processor(text+image) → pixel_values + input_ids + image_grid_thw + mm_token_type_ids
  → [visual.onnx] 得 pooler_output + 3 deepstack_features
  → 拼输入：inputs_embeds = embed(input_ids)，再把 image_pad 位置 scatter 成 pooler_output
  → get_rope_index 算 3D position_ids
  → 预算 causal attention_bias
  → [llm.onnx] prefill → logits + present_kv
  → greedy 循环：取 argmax → 新 input_id → decode（带 past_kv）→ 更新 kv
"""
import os, sys, torch, numpy as np, onnxruntime as ort
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

MODEL = os.path.expanduser("~/.cache/modelscope/hub/models/Qwen/Qwen3-VL-2B-Instruct")
# 导出产物路径：相对本包目录，让从任意 cwd 调用都能找到。
_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
VISUAL_ONNX = os.path.join(_PKG_DIR, "visual.onnx")
LLM_ONNX = os.path.join(_PKG_DIR, "llm.onnx")
D = "cpu"
NLAYERS = 28; HIDDEN = 2048; NH = 16; NKV = 8; HD = 128
IMG_TOK = 151655
IM_END = 151645
MAX_NEW = 16

def n16(x):
    return x.detach().cpu().numpy().astype(np.float16) if isinstance(x, torch.Tensor) else np.asarray(x).astype(np.float16)

class OnnxQwen3VL:
    def __init__(self, model=MODEL, visual_onnx=VISUAL_ONNX, llm_onnx=LLM_ONNX):
        self.proc = AutoProcessor.from_pretrained(model)
        # 只为拿 embed_tokens / get_rope_index（不跑 forward），权重复用即可
        self.ref = Qwen3VLForConditionalGeneration.from_pretrained(model, attn_implementation="eager", torch_dtype=torch.float16).eval()
        self.lm = self.ref.model.language_model
        self.vsess = ort.InferenceSession(visual_onnx, providers=["CPUExecutionProvider"])
        self.lsess = ort.InferenceSession(llm_onnx, providers=["CPUExecutionProvider"])
        self.in_names = [i.name for i in self.lsess.get_inputs()]
        self.out_names = [o.name for o in self.lsess.get_outputs()]

    def make_inputs(self, text, image=None):
        """复用 HF processor 生成 input_ids / pixel_values / grid_thw / mm_token_type_ids。"""
        images = [image] if image is not None else None
        inputs = self.proc(text=[text], images=images, do_resize=False, padding=True, return_tensors="pt")
        return inputs

    def _get(self, inputs, key, default=None):
        try:
            return getattr(inputs, key)
        except AttributeError:
            return default

    def rope_index(self, input_ids, mm_token_type_ids, image_grid_thw, attention_mask):
        """prefill 的 3D position_ids。同时返回 rope_delta，供 decode 用。

        Qwen3-VL 的 MRoPE：图像 token 在 3D 位置空间只占 max(h,w)//merge 个位置
        （而非 token 数），故「文本位置计数」与「token 序列长度」错位。HF 用
        rope_delta = llm_positions.max()+1 - len(input_ids) 记录这个偏移；decode 时
        position_ids = arange(past_len, past_len+1) + rope_delta（每行同值）。
        若忽略 delta 直接用 past_len，会偏到错误位置（如 OCR 多 token 场景）。
        """
        with torch.no_grad():
            pos_ids, rope_delta = self.ref.model.get_rope_index(
                input_ids=input_ids, mm_token_type_ids=mm_token_type_ids,
                image_grid_thw=image_grid_thw, attention_mask=attention_mask)
        return pos_ids, rope_delta  # (3,B,L), (B,1)

    def run_visual(self, pixel_values):
        pv = pixel_values.cpu().numpy().astype(np.float16)
        out = self.vsess.run(None, {"pixel_values": pv})
        pool = out[0]              # (n_img, 2048) fp16
        ds = [out[1], out[2], out[3]]
        return pool, ds

    def make_embeds(self, input_ids, pool, image_pad_mask):
        """embed(input_ids) 后把 image_pad 位置换成 pool。"""
        with torch.no_grad():
            emb = self.lm.embed_tokens(input_ids)  # (B, L, H) fp16
        B, L, H = emb.shape
        flat = emb.reshape(-1, H).clone()
        idx = torch.nonzero(image_pad_mask.reshape(-1), as_tuple=False).squeeze(-1)
        if pool is not None and idx.numel() > 0:
            flat[idx] = pool
        return flat.reshape(B, L, H)

    def causal_bias(self, q_len, kv_len):
        """(1,1,q,kv) 加法 mask：上三角 = finfo.min。"""
        m = torch.triu(torch.full((q_len, kv_len), torch.finfo(torch.float16).min, dtype=torch.float16), diagonal=kv_len - q_len + 1)
        return m.unsqueeze(0).unsqueeze(0)

    def run_llm(self, inputs_embeds, position_ids, attention_bias, ds, image_pad_mask, past=None):
        feed = {
            "inputs_embeds": n16(inputs_embeds),
            "position_ids": position_ids.cpu().numpy(),
            "attention_bias": n16(attention_bias),
            "deepstack_embeds_0": n16(ds[0]) if ds[0] is not None else np.zeros((1, HIDDEN), dtype=np.float16),
            "deepstack_embeds_1": n16(ds[1]) if ds[1] is not None else np.zeros((1, HIDDEN), dtype=np.float16),
            "deepstack_embeds_2": n16(ds[2]) if ds[2] is not None else np.zeros((1, HIDDEN), dtype=np.float16),
            "image_pad_mask": image_pad_mask.cpu().numpy(),
        }
        if past is None:
            past = [np.zeros((1, 2, NKV, 0, HD), dtype=np.float16) for _ in range(NLAYERS)]
        for i in range(NLAYERS):
            feed[f"past_key_values_{i}"] = past[i]
        res = self.lsess.run(None, feed)
        logits = res[0]                # (B, q, vocab)
        presents = res[1:1+NLAYERS]    # list of (B,2,NKV,kv,hd)
        return logits, presents

    def generate(self, text, image=None, max_new=MAX_NEW):
        inputs = self.make_inputs(text, image)
        input_ids = inputs.input_ids                  # (1, L)
        am = inputs.attention_mask                    # (1, L)
        mtt = self._get(inputs, "mm_token_type_ids", torch.zeros_like(input_ids, dtype=torch.int32))
        grid_thw = self._get(inputs, "image_grid_thw", None)
        pixel_values = self._get(inputs, "pixel_values", None)
        # image_pad mask
        n_img = int((input_ids[0] == IMG_TOK).sum().item())
        image_pad_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        image_pad_mask[input_ids == IMG_TOK] = True
        # visual
        if pixel_values is not None and n_img > 0:
            pool, ds = self.run_visual(pixel_values)  # (n_img, H) numpy
            pool_t = torch.tensor(pool)
            ds_t = [torch.tensor(d) for d in ds]
        else:
            pool_t = None
            ds_t = [torch.zeros((1, HIDDEN), dtype=torch.float16) for _ in range(3)]
        # embeds
        inputs_embeds = self.make_embeds(input_ids, pool_t, image_pad_mask)
        # position_ids（prefill）+ rope_delta（decode 用）
        pos_ids, rope_delta = self.rope_index(input_ids, mtt, grid_thw, am)
        L = input_ids.shape[1]
        ab = self.causal_bias(L, L)
        # prefill
        logits, presents = self.run_llm(inputs_embeds, pos_ids, ab, ds_t, image_pad_mask, past=None)
        next_id = int(np.argmax(logits[0, -1]))
        out_ids = [next_id]
        past = [p for p in presents]
        past_len = L
        # decode loop：position_ids = past_len + rope_delta（MRoPE 错位修正）
        delta = int(rope_delta[0, 0].item()) if rope_delta is not None else 0
        for _ in range(max_new - 1):
            if next_id == IM_END:
                break
            cur = torch.tensor([[next_id]], dtype=torch.long)
            cur_emb = self.lm.embed_tokens(cur)
            cur_pos = torch.full((3, 1, 1), past_len + delta, dtype=torch.long)
            cur_mask = torch.zeros((1, 1), dtype=torch.bool)
            cur_ab = torch.zeros((1, 1, 1, past_len + 1), dtype=torch.float16)
            ds_zero = [torch.zeros((1, HIDDEN), dtype=torch.float16) for _ in range(3)]
            logits, presents = self.run_llm(cur_emb, cur_pos, cur_ab, ds_zero, cur_mask, past=past)
            past = [p for p in presents]
            past_len += 1
            next_id = int(np.argmax(logits[0, -1]))
            out_ids.append(next_id)
        return out_ids


def main():
    img = Image.new("RGB", (224, 224), color=(128, 64, 200))
    text = ("<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
            "What color is the image? Answer in one word.<|im_end|>\n<|im_start|>assistant\n")
    print("=== ONNX generate ===")
    onnx_eng = OnnxQwen3VL()
    onnx_ids = onnx_eng.generate(text, image=img, max_new=MAX_NEW)
    print("onnx ids:", onnx_ids)
    print("onnx text:", onnx_eng.proc.decode(onnx_ids, skip_special_tokens=True))

    print("\n=== HF generate ===")
    hf = Qwen3VLForConditionalGeneration.from_pretrained(MODEL, attn_implementation="eager", torch_dtype=torch.float16).eval()
    inputs = onnx_eng.make_inputs(text, image=img)
    with torch.no_grad():
        gen = hf.generate(**inputs, max_new_tokens=MAX_NEW, do_sample=False)
    hf_ids = gen[0][inputs.input_ids.shape[1]:].tolist()
    print("hf ids:", hf_ids)
    print("hf text:", hf.processor if False else onnx_eng.proc.decode(hf_ids, skip_special_tokens=True))

    match = onnx_ids == hf_ids
    print(f"\n=== result: {'MATCH' if match else 'DIFF'} ({sum(a==b for a,b in zip(onnx_ids,hf_ids))}/{min(len(onnx_ids),len(hf_ids))} prefix equal) ===")
    return 0 if match else 1

if __name__ == "__main__":
    sys.exit(main())
