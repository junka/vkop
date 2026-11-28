"""Phi-3 / Phi-4-mini (`Phi3ForCausalLM`, model_type phi3).

Moved from llm/exporter/export/phi4_export_onnx.py, which now exports from
`make_wrapper`. Same I/O contract as qwen3_export_onnx.py. The three
differences from the Llama block, each op-for-op against HF modeling_phi3:

  1. fused projections: `qkv_proj` yields q/k/v and `gate_up_proj` yields
     gate/up in one MatMul. The OUTPUTS are sliced, the weights stay fused.
     gate/up is split at a constant boundary, not with `chunk`: chunk traces
     into an integer Shape->Gather->Add->Div->Mul chain whose tail Div the
     converter fuses into FUSED_ELEMWISE (fp16/fp32 kernels only), after which
     the Slice's `ends` is no longer int64 and the runtime crashed in
     Tensor<long long>::copyToCPU.
  2. partial RoPE: rotary_dim = head_dim * partial_rotary_factor (Phi-4-mini:
     96 of 128). HF's apply_rotary_pos_emb infers rotary_dim from cos and
     passes the rest through; it is reused as is.
  3. longrope: Phi3RotaryEmbedding multiplies `attention_scaling` (1.190238
     on Phi-4-mini) into cos/sin; StaticRope carries it. longrope picks
     short_factor vs long_factor from max(position_ids)+1 against
     original_max_position_embeddings -- a data-dependent branch a static
     graph cannot hold. The table used is the one HF builds at init (the
     short branch), so the graph is valid for positions below
     original_max_position_embeddings only.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ..hf_import import GraphSpec, HFImportError
from . import common


class Phi3LM(nn.Module):
    """Explicit KV tensors + additive attention bias, around HF's submodules
    (bypasses DynamicCache and create_sliding_window_causal_mask, whose
    torch.diff does not trace at opset 17)."""

    def __init__(self, model):
        super().__init__()
        from transformers.models.phi3.modeling_phi3 import (
            apply_rotary_pos_emb, repeat_kv)
        self._apply_rope = apply_rotary_pos_emb
        self._repeat_kv = repeat_kv
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
        # 1/sqrt(head_dim); longrope's attention_scaling is in cos/sin, not
        # here -- merging the two would not be HF's numeric path.
        self.scaling = float(lm.layers[0].self_attn.scaling)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv * self.head_dim
        self.inter = int(cfg.intermediate_size)

    def block(self, layer, hidden, cos, sin, attention_bias, past_kv):
        B, q_len, _ = hidden.shape
        residual = hidden
        h = layer.input_layernorm(hidden)
        attn = layer.self_attn
        qkv = attn.qkv_proj(h)
        q = qkv[..., :self.q_size].view(B, q_len, self.num_heads,
                                        self.head_dim).transpose(1, 2)
        k = qkv[..., self.q_size:self.q_size + self.kv_size]
        k = k.view(B, q_len, self.num_kv, self.head_dim).transpose(1, 2)
        v = qkv[..., self.q_size + self.kv_size:]
        v = v.view(B, q_len, self.num_kv, self.head_dim).transpose(1, 2)
        q, k = self._apply_rope(q, k, cos, sin)
        k_new = torch.cat([past_kv[:, 0], k], dim=2)
        v_new = torch.cat([past_kv[:, 1], v], dim=2)
        k_r = self._repeat_kv(k_new, self.groups)
        v_r = self._repeat_kv(v_new, self.groups)
        w = torch.matmul(q, k_r.transpose(2, 3)) * self.scaling
        w = w + attention_bias
        w = torch.nn.functional.softmax(w, dim=-1,
                                        dtype=torch.float32).to(q.dtype)
        out = torch.matmul(w, v_r)
        out = out.transpose(1, 2).reshape(B, q_len,
                                          self.num_heads * self.head_dim)
        # resid_attn_dropout is the identity in eval
        hidden = residual + attn.o_proj(out)
        residual = hidden
        h = layer.post_attention_layernorm(hidden)
        mlp = layer.mlp
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


def rope_info(cfg) -> dict:
    params = getattr(cfg, "rope_parameters", None) or {}
    hd = int(getattr(cfg, "head_dim", None)
             or cfg.hidden_size // cfg.num_attention_heads)
    partial = float(params.get("partial_rotary_factor",
                               getattr(cfg, "partial_rotary_factor", 1.0)))
    rot = int(hd * partial)
    return {"rope": params.get("rope_type", "default"), "rotary_dim": rot,
            "head_dim": hd,
            "max_ctx": int(params.get(
                "original_max_position_embeddings",
                getattr(cfg, "original_max_position_embeddings", 0)
                or cfg.max_position_embeddings))}


def make_wrapper(model) -> Phi3LM:
    cfg = model.config
    info = rope_info(cfg)
    if info["rotary_dim"] % 2:
        raise HFImportError(f"rotary_dim {info['rotary_dim']} must be even")
    attn0 = model.model.layers[0].self_attn
    if not hasattr(attn0, "qkv_proj") or not hasattr(
            model.model.layers[0].mlp, "gate_up_proj"):
        raise HFImportError("phi3 builder expects fused qkv_proj/gate_up_proj")
    w = Phi3LM(model).eval()
    if w.rope.value.numel() * 2 != info["rotary_dim"]:
        raise HFImportError(f"inv_freq has {w.rope.value.numel()} entries, "
                            f"rotary_dim is {info['rotary_dim']}")
    return w


def build(config, dtype, index) -> GraphSpec:
    from transformers import Phi3ForCausalLM
    cfg = common.text_config(config)
    common.refuse_dynamic_rope(cfg)
    model = common.meta_model(Phi3ForCausalLM, cfg, dtype)
    w = make_wrapper(model)
    info = rope_info(cfg)
    if info["rope"] == "longrope":
        common.warn(f"longrope: the short-factor table is baked in; the graph "
                    f"is valid for positions < {info['max_ctx']}")
    return common.llm_spec(
        w, index, nlayers=len(w.layers), hidden=int(cfg.hidden_size),
        num_kv=w.num_kv, head_dim=w.head_dim, dtype=dtype,
        summary={"builder": "phi3", "layers": len(w.layers),
                 "heads": w.num_heads, "kv_heads": w.num_kv,
                 "head_dim": w.head_dim, "rotary_dim": info["rotary_dim"],
                 "rope": info["rope"],
                 "attention_scaling": round(w.rope.attention_scaling, 6)})
