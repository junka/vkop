"""Llama-family decoder: llama, mistral, qwen2, qwen3 (and look-alikes).

The graph is op-for-op the one llm/exporter/export/qwen3_export_onnx.py
exports: the converter's attention / RMSNorm / SwiGLU fusions were written
against it and llm_chat reads its I/O. q_norm/k_norm are applied when the
checkpoint has them (Qwen3), which is the one place these families differ in
this block; projection biases (Qwen2) ride along in nn.Linear.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..hf_import import GraphSpec, HFImportError
from . import common

TESTED = {"llama", "mistral", "qwen2", "qwen3"}


def _rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


class DecoderLM(nn.Module):
    def __init__(self, model, cfg):
        super().__init__()
        lm = model.model
        self.layers = lm.layers
        self.norm = lm.norm
        self.lm_head = model.lm_head
        self.rope = common.rope_for(model)
        self.num_heads = int(cfg.num_attention_heads)
        self.num_kv = int(getattr(cfg, "num_key_value_heads", None)
                          or self.num_heads)
        self.head_dim = int(getattr(cfg, "head_dim", None)
                            or cfg.hidden_size // self.num_heads)
        self.scaling = float(getattr(lm.layers[0].self_attn, "scaling",
                                     self.head_dim ** -0.5))

    def block(self, layer, hidden, cos, sin, attention_bias, past_kv):
        nh, nkv, hd = self.num_heads, self.num_kv, self.head_dim
        B, q_len, _ = hidden.shape
        residual = hidden
        h = layer.input_layernorm(hidden)
        attn = layer.self_attn
        q = attn.q_proj(h).view(B, q_len, nh, hd)
        if getattr(attn, "q_norm", None) is not None:
            q = attn.q_norm(q)
        q = q.transpose(1, 2)
        k = attn.k_proj(h).view(B, q_len, nkv, hd)
        if getattr(attn, "k_norm", None) is not None:
            k = attn.k_norm(k)
        k = k.transpose(1, 2)
        v = attn.v_proj(h).view(B, q_len, nkv, hd).transpose(1, 2)
        c = cos.unsqueeze(1)
        s = sin.unsqueeze(1)
        q = (q * c) + (_rotate_half(q) * s)
        k = (k * c) + (_rotate_half(k) * s)
        k_new = torch.cat([past_kv[:, 0], k], dim=2)
        v_new = torch.cat([past_kv[:, 1], v], dim=2)
        groups = nh // nkv
        if groups > 1:
            b, n, slen, d = k_new.shape
            k_r = k_new[:, :, None].expand(b, n, groups, slen, d).reshape(
                b, n * groups, slen, d)
            v_r = v_new[:, :, None].expand(b, n, groups, slen, d).reshape(
                b, n * groups, slen, d)
        else:
            k_r, v_r = k_new, v_new
        w = torch.matmul(q, k_r.transpose(2, 3)) * self.scaling
        w = w + attention_bias
        w = F.softmax(w, dim=-1, dtype=torch.float32).to(q.dtype)
        out = torch.matmul(w, v_r)
        out = out.transpose(1, 2).reshape(B, q_len, nh * hd)
        hidden = residual + attn.o_proj(out)
        residual = hidden
        h = layer.post_attention_layernorm(hidden)
        mlp = layer.mlp
        h = mlp.down_proj(mlp.act_fn(mlp.gate_proj(h)) * mlp.up_proj(h))
        hidden = residual + h
        return hidden, torch.stack([k_new, v_new], dim=1)

    def forward(self, inputs_embeds, position_ids, attention_bias, *past_kvs):
        hidden = inputs_embeds
        cos, sin = self.rope(position_ids, hidden.dtype)
        presents = []
        for i, layer in enumerate(self.layers):
            hidden, pk = self.block(layer, hidden, cos, sin, attention_bias,
                                    past_kvs[i])
            presents.append(pk)
        hidden = self.norm(hidden)
        return (self.lm_head(hidden), *presents)


def check(model, cfg) -> None:
    mtype = getattr(cfg, "model_type", "?")
    if "gemma" in mtype:
        raise HFImportError("Gemma's (1 + w) RMSNorm is not this block")
    lm = getattr(model, "model", None)
    need = []
    if lm is None or not hasattr(lm, "layers") or not hasattr(lm, "norm"):
        need.append("model.layers / model.norm")
    if getattr(model, "lm_head", None) is None:
        need.append("lm_head")
    if not need:
        l0 = lm.layers[0]
        for path in ("input_layernorm", "post_attention_layernorm",
                     "self_attn.q_proj", "self_attn.k_proj",
                     "self_attn.v_proj", "self_attn.o_proj", "mlp.gate_proj",
                     "mlp.up_proj", "mlp.down_proj", "mlp.act_fn"):
            obj = l0
            for p in path.split("."):
                obj = getattr(obj, p, None)
            if obj is None:
                need.append(f"layers.N.{path}")
    if need:
        raise HFImportError(
            f"model_type {mtype!r} is not a Llama-style decoder (missing "
            f"{', '.join(need)}); pass --hf-builder")
    if mtype not in TESTED:
        common.warn(f"model_type {mtype!r} has the Llama block structure but "
                    f"is not one of the tested families {sorted(TESTED)}; "
                    f"verify its output against HF before trusting it")


def make_wrapper(model) -> DecoderLM:
    cfg = model.config
    check(model, cfg)
    return DecoderLM(model, cfg).eval()


def build(config, dtype, index) -> GraphSpec:
    from transformers import AutoModelForCausalLM
    cfg = common.text_config(config)
    rope = common.refuse_dynamic_rope(cfg)
    model = common.meta_model(AutoModelForCausalLM, cfg, dtype)
    w = make_wrapper(model)
    return common.llm_spec(
        w, index, nlayers=len(w.layers), hidden=int(cfg.hidden_size),
        num_kv=w.num_kv, head_dim=w.head_dim, dtype=dtype,
        summary={"builder": "llama", "model_type": cfg.model_type,
                 "layers": len(w.layers), "heads": w.num_heads,
                 "kv_heads": w.num_kv, "head_dim": w.head_dim, "rope": rope})
