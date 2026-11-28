"""Qwen3-VL language model (`Qwen3VLForConditionalGeneration`, qwen3_vl).

Moved from llm/exporter/export/qwen3vl_export_onnx.py, which now exports its
llm.onnx from `make_wrapper`. Only the LANGUAGE model is built here; the
vision tower (visual.onnx, dynamic grid) is still exported by that script.

I/O contract (differs from the plain LLM one in position_ids and the
deepstack inputs):

  in : inputs_embeds (B,q,H), position_ids (3,B,q) int64 (MRoPE t/h/w; equal
       rows for text), attention_bias (B,1,q,kv), deepstack_embeds_i
       (n_img,H) for each deepstack level, image_pad_mask (B,q) bool,
       past_key_values_i (B,2,nkv,kv,hd)
  out: logits (B,q,V), present_key_values_i

Deepstack: the vision tower's i-th deepstack feature is added at the
image_pad positions after TEXT layer i (HF `_deepstack_process`,
`layer_idx in range(len(deepstack_visual_embeds))`), via gather+add+scatter
because opset 17's ScatterND has no add reduction.

MRoPE: HF's interleaved mrope traces into a ~61-op int64 dynamic chain. It is
a static per-column row selection, so `StaticInterleavedMrope` computes it as
one matmul pos_t (B,q,3) @ M (3, hd/2), M[r, j] = inv_freq[j] * (r ==
row(j)); `assert_static_mrope_bitwise` proves it bit-identical to HF first.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..hf_import import GraphSpec, HFImportError
from . import common


def _nonzero(flat_mask):
    """torch.nonzero, also on meta (a trace with a shape-only mask).

    Meta has no data-independent nonzero; torch can fake one that assumes
    every element is set, which is all a trace needs (the op is recorded, the
    shape is dynamic in the graph). Scoped to meta so eager runs are exact.
    """
    if flat_mask.device.type != "meta":
        return torch.nonzero(flat_mask, as_tuple=False)
    import torch.fx.experimental._config as fxc
    old = getattr(fxc, "meta_nonzero_assume_all_nonzero", False)
    fxc.meta_nonzero_assume_all_nonzero = True
    try:
        return torch.nonzero(flat_mask, as_tuple=False)
    finally:
        fxc.meta_nonzero_assume_all_nonzero = old


def scatter_add_visual(hidden, embed, mask):
    """hidden[mask, :] += embed, as gather + add + scatter."""
    B, L, H = hidden.shape
    flat_mask = mask.reshape(-1)
    idx = _nonzero(flat_mask).squeeze(-1)
    flat = hidden.reshape(-1, H).clone()
    gathered = flat[idx] + embed
    idx2 = idx.unsqueeze(1).expand(-1, H)
    flat = torch.scatter(flat, 0, idx2, gathered)
    return flat.reshape(B, L, H)


def mrope_matrix(inv_freq: torch.Tensor, section) -> torch.Tensor:
    inv_freq = inv_freq.float().cpu()
    half = inv_freq.shape[0]
    if not (len(section) == 3 and 3 * section[1] <= half
            and 3 * section[2] <= half):
        raise HFImportError(f"unexpected mrope_section {section} for "
                            f"{half} frequencies")
    s1, s2 = section[1], section[2]
    sel = torch.zeros(half, dtype=torch.long)
    for j in range(half):
        if j % 3 == 1 and j < 3 * s1:
            sel[j] = 1
        elif j % 3 == 2 and j < 3 * s2:
            sel[j] = 2
    M = torch.zeros(3, half, dtype=torch.float32)
    M[sel, torch.arange(half)] = inv_freq
    return M


class StaticInterleavedMrope(nn.Module):
    """Static equivalent of Qwen3VLTextRotaryEmbedding (see module doc)."""

    def __init__(self, rotary_emb, device=None):
        super().__init__()
        inv = getattr(rotary_emb, "inv_freq", None)
        if inv is None or inv.device.type == "meta":
            with torch.device("cpu"):
                rotary_emb = type(rotary_emb)(config=rotary_emb.config)
            inv = rotary_emb.inv_freq
        value = mrope_matrix(inv, rotary_emb.mrope_section)
        self.value = value
        self.M = nn.Parameter(value.to(device or "cpu"), requires_grad=False)
        self.attention_scaling = float(rotary_emb.attention_scaling)

    def hf_constants(self):
        return {"M": self.value.numpy()}

    def forward(self, x, position_ids):
        pos = position_ids.to(torch.float32)         # (3,B,q)
        pos_t = pos.permute(1, 2, 0)                 # (B,q,3)
        freqs = torch.matmul(pos_t, self.M)          # (B,q,hd/2) fp32
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos() * self.attention_scaling
        sin = emb.sin() * self.attention_scaling
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def assert_static_mrope_bitwise(rotary_emb, static_mrope):
    """Bit-compare with HF's rotary_emb: text (equal rows) + random 3-D."""
    x = torch.zeros(1, 1, 4, dtype=torch.float16)
    gen = torch.Generator().manual_seed(0)
    cases = [
        torch.arange(4, dtype=torch.int64).reshape(1, 1, 4).expand(3, 1, 4),
        torch.arange(9, dtype=torch.int64).reshape(1, 1, 9).expand(3, 1, 9),
        torch.randint(0, 3000, (3, 1, 7), generator=gen),
    ]
    ref = StaticInterleavedMrope(rotary_emb)   # CPU copy of the table
    ref.M.data = static_mrope.value.clone()
    with torch.no_grad():
        for pid in cases:
            c0, s0 = rotary_emb(x, pid)
            c1, s1 = ref(x, pid)
            if not (c0.dtype == c1.dtype and torch.equal(c0, c1)
                    and torch.equal(s0, s1)):
                raise HFImportError(
                    f"static mrope != HF rotary_emb for {tuple(pid.shape)}: "
                    f"maxdiff={(c0.float() - c1.float()).abs().max()}")
    print("[mrope] bitwise-identical to HF on all probe shapes")


class Qwen3VLLM(nn.Module):
    """HF layers, hand-written loop: explicit KV tensors, additive bias,
    deepstack injection after text layers 0..n-1."""

    def __init__(self, full_model):
        super().__init__()
        from transformers.models.qwen3_vl.modeling_qwen3_vl import (
            apply_rotary_pos_emb, repeat_kv)
        self._apply_rope = apply_rotary_pos_emb
        self._repeat_kv = repeat_kv
        lm = full_model.model.language_model
        tcfg = full_model.config.text_config
        vcfg = full_model.config.vision_config
        self.layers = lm.layers
        self.norm = lm.norm
        self.mrope_static = StaticInterleavedMrope(
            lm.rotary_emb, device="meta" if common.is_meta(lm) else None)
        self.lm_head = full_model.lm_head
        self.num_heads = int(tcfg.num_attention_heads)
        self.num_kv_heads = int(tcfg.num_key_value_heads)
        self.head_dim = int(tcfg.head_dim or tcfg.hidden_size
                            // self.num_heads)
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.hidden = int(tcfg.hidden_size)
        self.scaling = float(lm.layers[0].self_attn.scaling)
        n_ds = len(vcfg.deepstack_visual_indexes)
        self.deepstack_layers = list(range(n_ds))

    def decoder_layer(self, layer, hidden, cos, sin, attention_bias, past_kv):
        B, q_len, _ = hidden.shape
        residual = hidden
        h = layer.input_layernorm(hidden)
        attn = layer.self_attn
        q = attn.q_proj(h).view(B, q_len, self.num_heads, self.head_dim)
        q = attn.q_norm(q).transpose(1, 2)
        kv_shape = (B, q_len, self.num_kv_heads, self.head_dim)
        k = attn.k_norm(attn.k_proj(h).view(kv_shape)).transpose(1, 2)
        v = attn.v_proj(h).view(kv_shape).transpose(1, 2)
        q, k = self._apply_rope(q, k, cos, sin)
        k_new = torch.cat([past_kv[:, 0], k], dim=2)
        v_new = torch.cat([past_kv[:, 1], v], dim=2)
        k_r = self._repeat_kv(k_new, self.num_kv_groups)
        v_r = self._repeat_kv(v_new, self.num_kv_groups)
        attn_w = torch.matmul(q, k_r.transpose(2, 3)) * self.scaling
        attn_w = attn_w + attention_bias
        attn_w = F.softmax(attn_w, dim=-1, dtype=torch.float32).to(q.dtype)
        out = torch.matmul(attn_w, v_r)
        out = out.transpose(1, 2).reshape(B, q_len, self.hidden)
        hidden = residual + attn.o_proj(out)
        residual = hidden
        h = layer.post_attention_layernorm(hidden)
        mlp = layer.mlp
        h = mlp.down_proj(mlp.act_fn(mlp.gate_proj(h)) * mlp.up_proj(h))
        hidden = residual + h
        return hidden, torch.stack([k_new, v_new], dim=1)

    def forward(self, inputs_embeds, position_ids, attention_bias, *rest):
        n_ds = len(self.deepstack_layers)
        deepstacks = rest[:n_ds]
        image_pad_mask = rest[n_ds]
        past_kvs = rest[n_ds + 1:]
        hidden = inputs_embeds
        cos, sin = self.mrope_static(hidden, position_ids)
        presents = []
        for idx, layer in enumerate(self.layers):
            hidden, pk = self.decoder_layer(layer, hidden, cos, sin,
                                            attention_bias, past_kvs[idx])
            presents.append(pk)
            if idx in self.deepstack_layers:
                ds = self.deepstack_layers.index(idx)
                hidden = scatter_add_visual(hidden, deepstacks[ds],
                                            image_pad_mask)
        hidden = self.norm(hidden)
        return (self.lm_head(hidden), *presents)


def io_names(nlayers: int, n_ds: int):
    input_names = (["inputs_embeds", "position_ids", "attention_bias"]
                   + [f"deepstack_embeds_{i}" for i in range(n_ds)]
                   + ["image_pad_mask"]
                   + [f"past_key_values_{i}" for i in range(nlayers)])
    output_names = (["logits"]
                    + [f"present_key_values_{i}" for i in range(nlayers)])
    dyn = {"inputs_embeds": {0: "batch", 1: "seq"},
           "position_ids": {1: "batch", 2: "seq"},
           "attention_bias": {0: "batch", 2: "q_len", 3: "kv_len"},
           "image_pad_mask": {0: "batch", 1: "seq"},
           "logits": {0: "batch", 1: "seq"}}
    for i in range(n_ds):
        dyn[f"deepstack_embeds_{i}"] = {0: "num_image_patches"}
    for i in range(nlayers):
        dyn[f"past_key_values_{i}"] = {0: "batch", 3: "kv_len"}
        dyn[f"present_key_values_{i}"] = {0: "batch", 3: "kv_len"}
    return input_names, output_names, dyn


def mark_causal_bias(proto) -> None:
    """Declare attention_bias a standard causal mask (model-level metadata, so
    onnxoptimizer does not drop it): the converter then sets mask_causal=1 and
    the kernel stops each row's KV scan at its diagonal."""
    if not any(p.key == "vkop_causal_mask:attention_bias"
               for p in proto.metadata_props):
        proto.metadata_props.add(key="vkop_causal_mask:attention_bias",
                                 value="1")


def make_wrapper(model) -> Qwen3VLLM:
    tcfg = model.config.text_config
    if len(model.model.language_model.layers) < len(
            model.config.vision_config.deepstack_visual_indexes):
        raise HFImportError("fewer text layers than deepstack levels")
    common.refuse_dynamic_rope(tcfg)
    return Qwen3VLLM(model).eval()


def build(config, dtype, index) -> GraphSpec:
    from transformers import Qwen3VLForConditionalGeneration
    model = common.meta_model(Qwen3VLForConditionalGeneration, config, dtype)
    w = make_wrapper(model)
    # Prove the static mrope on a CPU copy of HF's rotary for this config.
    rot = model.model.language_model.rotary_emb
    with torch.device("cpu"):
        rot_cpu = type(rot)(config=rot.config)
    assert_static_mrope_bitwise(rot_cpu, w.mrope_static)

    common.buffers_to_params(w)
    nlayers = len(w.layers)
    n_ds = len(w.deepstack_layers)
    S = 8
    m = "meta"
    dummies = (torch.empty(1, S, w.hidden, dtype=dtype, device=m),
               torch.empty(3, 1, S, dtype=torch.long, device=m),
               torch.empty(1, 1, S, S, dtype=dtype, device=m),
               # rows == S: the meta nonzero assumes every mask bit is set
               *[torch.empty(S, w.hidden, dtype=dtype, device=m)
                 for _ in range(n_ds)],
               torch.empty(1, S, dtype=torch.bool, device=m),
               *[torch.empty(1, 2, w.num_kv_heads, 0, w.head_dim, dtype=dtype,
                             device=m) for _ in range(nlayers)])
    ins, outs, dyn = io_names(nlayers, n_ds)
    prefix = common.text_prefix(index)
    return GraphSpec(
        wrapper=w, dummies=dummies, input_names=ins, output_names=outs,
        dynamic_axes=dyn, constants=common.rope_constants(w),
        aliases=common.tied_head_alias(index, prefix), ckpt_prefix=prefix,
        post_splice=mark_causal_bias,
        summary={"builder": "qwen3vl (language model only)",
                 "layers": nlayers, "heads": w.num_heads,
                 "kv_heads": w.num_kv_heads, "head_dim": w.head_dim,
                 "deepstack": n_ds, "ckpt_prefix": prefix})
