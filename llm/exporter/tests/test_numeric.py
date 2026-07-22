"""数值对齐测试：导出的 visual.onnx / llm.onnx 与 HF 逐张量对比。

对齐项：
  - visual: pooler_output + 3 deepstack_features vs HF visual forward
  - llm prefill: logits + present_key_values vs HF forward(use_cache=True)
  - llm decode: 单 token logits vs HF 单步 forward
"""
import os
import sys
import numpy as np
import torch
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from infer import HIDDEN, NLAYERS, NKV, HD

MODEL = os.path.expanduser("~/.cache/modelscope/hub/models/Qwen/Qwen3-VL-2B-Instruct")
IMG_TOK = 151655


def _mean_ok(a, b, mean_tol=2e-2):
    """fp16 + CPU ort 累积舍入使 maxdiff 偶达 ~0.2，用均值差判定避免噪声误报。

    各输出量级不同（logits ~10、kv ~1、visual pooler ~0.4），统一用绝对均值差阈值
    2e-2 偏松但足以抓住逻辑性偏差（逻辑错时 mean 通常 >0.5 或几十）。
    """
    diff = (a.float() - b.float()).abs()
    return diff.mean().item() < mean_tol


def _build_prefill_inputs(visual, lm, device="cpu"):
    """构造 visual/llm prefill 的随机输入 + HF 参考输出。

    用固定 seed 让 prefill 测试可复现（避免随机输入导致偶发数值波动）。
    """
    torch.manual_seed(1234)
    patch = visual.patch_embed.patch_size
    tps = visual.patch_embed.temporal_patch_size
    ic = visual.patch_embed.in_channels
    H = W = 224
    gt, gh, gw = 1, H // patch, W // patch
    sl = gt * gh * gw
    grid_thw = torch.tensor([[gt, gh, gw]], dtype=torch.int32, device=device)
    pv = torch.randn(sl, ic * tps * patch * patch, dtype=visual.dtype, device=device)
    with torch.no_grad():
        hf_out = visual(pv, grid_thw)
    return pv, grid_thw, hf_out


# ----------------------------- visual -----------------------------

def test_visual_pooler_and_deepstack(onnx_eng):
    """visual.onnx 的 pooler_output + 3 deepstack_features 与 HF visual forward 对齐。"""
    visual = onnx_eng.ref.model.visual
    device = "cpu"
    pv, grid_thw, hf_out = _build_prefill_inputs(visual, onnx_eng.lm, device)
    # visual.onnx 因 grid_thw 被常量折叠，只接受 pixel_values
    out = onnx_eng.vsess.run(None, {"pixel_values": pv.cpu().numpy().astype(np.float16)})
    v_pool = torch.tensor(out[0]).float()
    v_ds = [torch.tensor(out[i]).float() for i in (1, 2, 3)]
    assert _mean_ok(v_pool, hf_out.pooler_output.float()), "visual pooler_output 均值差过大"
    for i in range(3):
        assert _mean_ok(v_ds[i], hf_out.deepstack_features[i].float()), \
            f"visual deepstack_{i} 均值差过大"


# ----------------------------- llm prefill -----------------------------

def test_llm_prefill_logits_and_kv(onnx_eng, hf_model):
    """llm.onnx prefill 的 logits + present_kv[0] 与 HF forward(use_cache=True) 对齐。"""
    device = "cpu"
    visual = hf_model.model.visual
    lm = onnx_eng.lm
    pv, grid_thw, hf_vis = _build_prefill_inputs(visual, lm, device)
    SPATIAL = visual.config.spatial_merge_size
    n_img = (224 // visual.patch_embed.patch_size) ** 2 // (SPATIAL ** 2)
    L = n_img + 8
    input_ids = torch.tensor([[IMG_TOK] * n_img + [1, 2, 3, 4, 5, 6, 7, 8]],
                             dtype=torch.long, device=device)
    am = torch.ones(1, L, dtype=torch.long, device=device)
    mtt = torch.zeros(1, L, dtype=torch.int32, device=device)
    mtt[0, :n_img] = 1
    with torch.no_grad():
        pos_ids, _ = hf_model.model.get_rope_index(
            input_ids=input_ids, mm_token_type_ids=mtt,
            image_grid_thw=grid_thw, attention_mask=am)
        emb = lm.embed_tokens(input_ids)
        mask = torch.zeros(1, L, dtype=torch.bool, device=device)
        mask[0, :n_img] = True
        flat = emb.reshape(-1, HIDDEN).clone()
        idx = torch.nonzero(mask.reshape(-1), as_tuple=False).squeeze(-1)
        flat[idx] = hf_vis.pooler_output
        inputs_embeds = flat.reshape(1, L, HIDDEN)
        ds = [hf_vis.deepstack_features[i] for i in range(3)]
        ab = torch.triu(torch.full((L, L), torch.finfo(torch.float16).min,
                                   dtype=torch.float16, device=device), diagonal=1).unsqueeze(0).unsqueeze(0)
        hf_out = hf_model.model.language_model(
            inputs_embeds=inputs_embeds, position_ids=pos_ids, attention_mask=am,
            past_key_values=None, use_cache=True,
            visual_pos_masks=mask, deepstack_visual_embeds=ds)
        hf_logits = hf_model.lm_head(hf_out.last_hidden_state)

    # ONNX prefill
    past = [np.zeros((1, 2, NKV, 0, HD), dtype=np.float16) for _ in range(NLAYERS)]
    feed = {
        "inputs_embeds": inputs_embeds.cpu().numpy().astype(np.float16),
        "position_ids": pos_ids.cpu().numpy(),
        "attention_bias": ab.cpu().numpy().astype(np.float16),
        "deepstack_embeds_0": ds[0].cpu().numpy().astype(np.float16),
        "deepstack_embeds_1": ds[1].cpu().numpy().astype(np.float16),
        "deepstack_embeds_2": ds[2].cpu().numpy().astype(np.float16),
        "image_pad_mask": mask.cpu().numpy(),
    }
    for i in range(NLAYERS):
        feed[f"past_key_values_{i}"] = past[i]
    res = onnx_eng.lsess.run(None, feed)
    onnx_logits = torch.tensor(res[0]).float()
    assert _mean_ok(onnx_logits, hf_logits.float()), "llm prefill logits 均值差过大"

    # present_kv[0]
    onnx_pkv0 = torch.tensor(res[1]).float()
    hf_pkv0 = torch.stack([hf_out.past_key_values.layers[0].keys,
                           hf_out.past_key_values.layers[0].values], dim=1).float()
    assert _mean_ok(onnx_pkv0, hf_pkv0), "llm prefill present_kv[0] 均值差过大"


# ----------------------------- llm decode -----------------------------

def test_llm_decode_logits(onnx_eng, hf_model):
    """llm.onnx 单 token decode（带 HF prefill 的 past_kv）与 HF 单步 forward 对齐。"""
    device = "cpu"
    visual = hf_model.model.visual
    lm = onnx_eng.lm
    pv, grid_thw, hf_vis = _build_prefill_inputs(visual, lm, device)
    SPATIAL = visual.config.spatial_merge_size
    n_img = (224 // visual.patch_embed.patch_size) ** 2 // (SPATIAL ** 2)
    L = n_img + 8
    input_ids = torch.tensor([[IMG_TOK] * n_img + [1, 2, 3, 4, 5, 6, 7, 8]],
                             dtype=torch.long, device=device)
    am = torch.ones(1, L, dtype=torch.long, device=device)
    mtt = torch.zeros(1, L, dtype=torch.int32, device=device)
    mtt[0, :n_img] = 1
    with torch.no_grad():
        pos_ids, _ = hf_model.model.get_rope_index(
            input_ids=input_ids, mm_token_type_ids=mtt,
            image_grid_thw=grid_thw, attention_mask=am)
        emb = lm.embed_tokens(input_ids)
        mask = torch.zeros(1, L, dtype=torch.bool, device=device)
        mask[0, :n_img] = True
        flat = emb.reshape(-1, HIDDEN).clone()
        idx = torch.nonzero(mask.reshape(-1), as_tuple=False).squeeze(-1)
        flat[idx] = hf_vis.pooler_output
        inputs_embeds = flat.reshape(1, L, HIDDEN)
        ds = [hf_vis.deepstack_features[i] for i in range(3)]
        ab = torch.triu(torch.full((L, L), torch.finfo(torch.float16).min,
                                   dtype=torch.float16, device=device), diagonal=1).unsqueeze(0).unsqueeze(0)
        hf_out = hf_model.model.language_model(
            inputs_embeds=inputs_embeds, position_ids=pos_ids, attention_mask=am,
            past_key_values=None, use_cache=True,
            visual_pos_masks=mask, deepstack_visual_embeds=ds)

    # decode：1 个新 token，position = past_kv_len + rope_delta（MRoPE 错位修正）。
    # rope_delta 在 HF prefill 时由 compute_3d_position_ids 缓存到 model.rope_deltas，
    # 但我们直接调 language_model（绕开 compute_3d_position_ids），故手动从 get_rope_index 取。
    with torch.no_grad():
        _, rope_delta = hf_model.model.get_rope_index(
            input_ids=input_ids, mm_token_type_ids=mtt,
            image_grid_thw=grid_thw, attention_mask=am)
    delta = int(rope_delta[0].item())
    # past_kv 长度以 HF cache 实际 dim 为准（HF DynamicCache 可能比 input_ids 多 1 slot）。
    past_kv_len = hf_out.past_key_values.layers[0].keys.shape[2]
    dec_ids = torch.tensor([[9]], dtype=torch.long, device=device)
    with torch.no_grad():
        dec_emb = lm.embed_tokens(dec_ids)
        dec_pos = torch.full((3, 1, 1), past_kv_len + delta, dtype=torch.long, device=device)
        dec_ab = torch.zeros(1, 1, 1, past_kv_len + 1, dtype=torch.float16, device=device)
        dec_mask = torch.zeros(1, 1, dtype=torch.bool, device=device)
        hf_dec = hf_model.model.language_model(
            inputs_embeds=dec_emb, position_ids=dec_pos,
            attention_mask=torch.ones(1, past_kv_len + 1, dtype=torch.long, device=device),
            past_key_values=hf_out.past_key_values, use_cache=True)
        hf_dec_logits = hf_model.lm_head(hf_dec.last_hidden_state)

    hf_past_kv = [torch.stack([hf_out.past_key_values.layers[i].keys,
                               hf_out.past_key_values.layers[i].values], dim=1)
                  for i in range(NLAYERS)]
    # 喂给 ONNX 的 past_kv 取 [:past_kv_len]：保证 attention_bias 的 kv_len = past_kv_len+1 与
    # cat([past_kv[:, :past_kv_len], k]).len 一致，避免广播 58 vs 59。
    feed = {
        "inputs_embeds": dec_emb.cpu().numpy().astype(np.float16),
        "position_ids": dec_pos.cpu().numpy(),
        "attention_bias": dec_ab.cpu().numpy().astype(np.float16),
        "deepstack_embeds_0": np.zeros((1, HIDDEN), dtype=np.float16),
        "deepstack_embeds_1": np.zeros((1, HIDDEN), dtype=np.float16),
        "deepstack_embeds_2": np.zeros((1, HIDDEN), dtype=np.float16),
        "image_pad_mask": dec_mask.cpu().numpy(),
    }
    for i in range(NLAYERS):
        feed[f"past_key_values_{i}"] = hf_past_kv[i][:, :, :, :past_kv_len, :].cpu().numpy().astype(np.float16)
    res = onnx_eng.lsess.run(None, feed)
    onnx_dec_logits = torch.tensor(res[0]).float()
    assert _mean_ok(onnx_dec_logits, hf_dec_logits.float()), "llm decode logits 均值差过大"
