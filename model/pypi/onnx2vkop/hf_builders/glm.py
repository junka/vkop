"""GLM / GLM-Edge (`GlmForCausalLM`, model_type glm).

Moved from llm/exporter/export/glm_edge_export_onnx.py, which now exports from
`make_wrapper`. Same I/O contract as qwen3_export_onnx.py. Plain pre-norm
decoder: separate q/k/v (no QK-norm, no bias), fused gate_up_proj (like Phi),
SiLU, RMSNorm, pure causal.

Two deliberate rewrites, both proven equivalent by `selfcheck()` before any
graph is produced (it runs on CPU with random tensors, so it needs no weights
and runs on the meta path too):

  * Interleaved RoPE -> half-split RoPE + a fixed column permutation. HF pairs
    dims (2i, 2i+1) and repeat_interleaves cos/sin; the runtime's
    RotaryEmbedding kernel and the converter's fuse_rotary_embedding pattern
    pair (i, i+D/2). Permuting x to [even cols || odd cols], rotating
    half-split and permuting back is bit-identical, and HF's cos = cat(f, f)
    is already the half-split width.
  * GQA without repeat_kv: the g query heads of one KV head are folded into
    the GEMM's m dimension (B, nkv, g*q, hd) instead of expanding K/V g-fold,
    which on the graph is an Expand rewriting g x the history every decode
    step (56 of them on GLM-Edge). attention_bias is tiled g-fold instead.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ..hf_import import GraphSpec, HFImportError
from . import common


def _to_halfsplit_basis(x):
    """[..., D] -> [even cols || odd cols] (permutation P)."""
    *lead, d = x.shape
    return x.reshape(*lead, d // 2, 2).transpose(-1, -2).reshape(*lead, d)


def _from_halfsplit_basis(y):
    """Inverse of _to_halfsplit_basis: [a || b] -> [a0, b0, a1, b1, ...]."""
    *lead, d = y.shape
    return y.reshape(*lead, 2, d // 2).transpose(-1, -2).reshape(*lead, d)


def _rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def rope_halfsplit_equivalent(q, k, cos, sin):
    """Bit-identical to modeling_glm.apply_rotary_pos_emb, in half-split form."""
    c = cos.unsqueeze(1)
    s = sin.unsqueeze(1)
    qp = _to_halfsplit_basis(q)
    kp = _to_halfsplit_basis(k)
    qr = qp * c + _rotate_half(qp) * s
    kr = kp * c + _rotate_half(kp) * s
    return _from_halfsplit_basis(qr), _from_halfsplit_basis(kr)


def attention_gqa(q, k, v, attention_bias, scaling, groups):
    """Attention equal to repeat_kv-then-attend; returns (B, q, nq*hd).

    q (B, nq, q, hd) -> (B, nkv, g*q, hd): head h = kv*g + j, repeat_kv's
    order. Each output reads the same K/V values; only the GEMM's m changes,
    so fp32 accumulation order may differ in the last bits.
    """
    B, nq, q_len, hd = q.shape
    nkv = k.shape[1]
    kv_len = k.shape[2]
    q_g = q.reshape(B, nkv, groups * q_len, hd)
    scores = torch.matmul(q_g, k.transpose(2, 3)) * scaling
    bias = attention_bias.unsqueeze(2).expand(
        B, 1, groups, q_len, kv_len).reshape(B, 1, groups * q_len, kv_len)
    scores = scores + bias
    w = torch.nn.functional.softmax(scores, dim=-1,
                                    dtype=torch.float32).to(q.dtype)
    out = torch.matmul(w, v)
    out = out.reshape(B, nkv, groups, q_len, hd)
    return out.permute(0, 3, 1, 2, 4).reshape(B, q_len, nq * hd)


class GlmLM(nn.Module):
    """Explicit KV tensors + additive attention bias around HF's submodules."""

    def __init__(self, model):
        super().__init__()
        cfg = model.config
        lm = model.model
        self.layers = lm.layers
        self.norm = lm.norm
        self.lm_head = model.lm_head
        self.rope = common.rope_for(model)
        self.num_heads = int(cfg.num_attention_heads)
        self.num_kv = int(cfg.num_key_value_heads)
        self.head_dim = int(getattr(cfg, "head_dim", None)
                            or cfg.hidden_size // self.num_heads)
        self.groups = self.num_heads // self.num_kv
        self.scaling = float(lm.layers[0].self_attn.scaling)
        self.inter = int(cfg.intermediate_size)

    def block(self, layer, hidden, cos, sin, attention_bias, past_kv):
        B, q_len, _ = hidden.shape
        nh, nkv, hd = self.num_heads, self.num_kv, self.head_dim
        residual = hidden
        h = layer.input_layernorm(hidden)
        attn = layer.self_attn
        q = attn.q_proj(h).view(B, q_len, nh, hd).transpose(1, 2)
        k = attn.k_proj(h).view(B, q_len, nkv, hd).transpose(1, 2)
        v = attn.v_proj(h).view(B, q_len, nkv, hd).transpose(1, 2)
        q, k = rope_halfsplit_equivalent(q, k, cos, sin)
        k_new = torch.cat([past_kv[:, 0], k], dim=2)
        v_new = torch.cat([past_kv[:, 1], v], dim=2)
        out = attention_gqa(q, k_new, v_new, attention_bias, self.scaling,
                            self.groups)
        hidden = residual + attn.o_proj(out)
        residual = hidden
        h = layer.post_attention_layernorm(hidden)
        mlp = layer.mlp
        # constant-boundary split, not chunk(2): see phi3.py for the crash
        up = mlp.gate_up_proj(h)
        gate = up[..., :self.inter]
        up = up[..., self.inter:]
        hidden = residual + mlp.down_proj(up * mlp.activation_fn(gate))
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


def check(model) -> None:
    """The block this builder writes, as hard checks on the model/config."""
    cfg = model.config
    rop = getattr(cfg, "rope_parameters", None) or {}
    hd = int(getattr(cfg, "head_dim", None)
             or cfg.hidden_size // cfg.num_attention_heads)
    if rop.get("rope_type", "default") != "default":
        raise HFImportError(f"glm builder expects default RoPE, got {rop}")
    if float(rop.get("partial_rotary_factor", 1.0)) != 1.0:
        raise HFImportError("partial_rotary_factor != 1: the half-split "
                            "permutation assumes rotary_dim == head_dim")
    if hd % 2 or cfg.num_attention_heads % cfg.num_key_value_heads:
        raise HFImportError("odd head_dim or non-integer GQA group")
    if getattr(cfg, "sliding_window", None) or getattr(cfg, "layer_types",
                                                       None):
        raise HFImportError("sliding-window layers: the pure-causal "
                            "attention_bias contract does not hold")
    if getattr(cfg, "attention_bias", False):
        raise HFImportError("projection biases are not in this block")
    if getattr(cfg, "hidden_act", "silu") != "silu":
        raise HFImportError(f"hidden_act={cfg.hidden_act}")
    attn0 = model.model.layers[0].self_attn
    if hasattr(attn0, "q_norm") or hasattr(attn0, "k_norm"):
        raise HFImportError("QK-norm present: not the GLM-Edge block")
    for mod in (attn0.q_proj, attn0.k_proj, attn0.v_proj, attn0.o_proj,
                model.model.layers[0].mlp.gate_up_proj, model.lm_head):
        if mod.bias is not None:
            raise HFImportError("a projection carries a bias")


def selfcheck(wrapper, hidden_size) -> None:
    """Prove both rewrites against HF before exporting (CPU, random data)."""
    from transformers.models.glm.modeling_glm import (
        apply_rotary_pos_emb, repeat_kv)
    torch.manual_seed(0)
    B, S, H, HKV, D = 2, 7, wrapper.num_heads, wrapper.num_kv, \
        wrapper.head_dim
    rope = common.StaticRope(wrapper.rope.value,
                             wrapper.rope.attention_scaling)
    q = torch.randn(B, H, S, D)
    k = torch.randn(B, HKV, S, D)
    pos = torch.arange(S, dtype=torch.long).unsqueeze(0).expand(B, -1)
    cos, sin = rope(pos, torch.float32)
    q_hf, k_hf = apply_rotary_pos_emb(q, k, cos, sin)
    q_alt, k_alt = rope_halfsplit_equivalent(q, k, cos, sin)
    dq = (q_hf - q_alt).abs().max().item()
    dk = (k_hf - k_alt).abs().max().item()
    print(f"[rope-check] fp32 maxdiff q={dq:.3e} k={dk:.3e}")
    if dq >= 1e-6 or dk >= 1e-6:
        raise HFImportError("half-split rewrite != HF interleaved RoPE")
    g = wrapper.groups
    for B, S, KV in ((2, 7, 11), (1, 1, 32)):
        q = torch.randn(B, H, S, D, dtype=torch.float16)
        k = torch.randn(B, HKV, KV, D, dtype=torch.float16)
        v = torch.randn(B, HKV, KV, D, dtype=torch.float16)
        bias = torch.triu(torch.full((S, KV), torch.finfo(torch.float16).min,
                                     dtype=torch.float16), diagonal=1)[None,
                                                                       None]
        a = attention_gqa(q, k, v, bias, wrapper.scaling, g)
        k_r, v_r = repeat_kv(k, g), repeat_kv(v, g)
        sc = torch.matmul(q, k_r.transpose(2, 3)) * wrapper.scaling
        w = torch.nn.functional.softmax(sc + bias, dim=-1,
                                        dtype=torch.float32).to(q.dtype)
        b = torch.matmul(w, v_r).transpose(1, 2).reshape(B, S, H * D)
        rel = ((a.float() - b.float()).abs().max()
               / b.float().abs().max().clamp(min=1e-6)).item()
        print(f"[gqa-check] B={B} q={S} kv={KV} maxrel={rel:.3e}")
        if rel >= 1e-3:
            raise HFImportError("GQA fold != repeat_kv attention")


def make_wrapper(model) -> GlmLM:
    check(model)
    w = GlmLM(model).eval()
    if w.rope.attention_scaling != 1.0 or w.rope.value.numel() != \
            w.head_dim // 2:
        raise HFImportError("unexpected rotary table for default RoPE")
    selfcheck(w, int(model.config.hidden_size))
    return w


def build(config, dtype, index) -> GraphSpec:
    from transformers import GlmForCausalLM
    cfg = common.text_config(config)
    model = common.meta_model(GlmForCausalLM, cfg, dtype)
    w = make_wrapper(model)
    return common.llm_spec(
        w, index, nlayers=len(w.layers), hidden=int(cfg.hidden_size),
        num_kv=w.num_kv, head_dim=w.head_dim, dtype=dtype,
        summary={"builder": "glm", "layers": len(w.layers),
                 "heads": w.num_heads, "kv_heads": w.num_kv,
                 "head_dim": w.head_dim, "rope": "interleaved->half-split"})
