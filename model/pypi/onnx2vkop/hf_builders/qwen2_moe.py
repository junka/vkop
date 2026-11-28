"""Qwen1.5-MoE / Qwen2-MoE (`Qwen2MoeForCausalLM`, model_type qwen2_moe).

Moved from llm/exporter/export/qwen2_moe_export_onnx.py (stage 1 of the
expert-paging pipeline), which now exports from `make_wrapper`. Same I/O
contract as qwen3_export_onnx.py; attention is the dense Qwen2 one, the MLP is
replaced by `SparseMoeOnnx`.

What the graph is -- and is not:

  * HF's Qwen2MoeExperts.forward has nonzero()/where()/index_add_() data
    dependent control flow and scatter, which does not export. Each layer's
    `mlp` container is REPLACED (not forward-patched: an instance-level
    `mlp.forward = fn` crashes torch 2.10's tracer in alias_analysis.cpp) by
    `SparseMoeOnnx`: HF's router, shared expert and shared_expert_gate as is,
    plus top_k fixed expert branches.
  * Slot s is wired to expert s (0..top_k-1) and weighted by the s-th top-k
    probability. That is the stage-1 contract: the file loads and runs as a
    "top_k-expert dense model", and stage 2 (llm/exporter/make_moe_variant.py)
    turns the wired weights into paged slots whose bytes the runtime swaps per
    routing decision. The graph therefore equals HF's MoE output only when the
    routing is trivially [0..top_k-1] in order (e.g. num_experts == top_k == 1);
    for anything else it is NOT the HF model until paging is applied.
  * Slots are summed in top-k order, HF sums in expert-id order: equal
    mathematically, not bitwise.

On the `--hf-checkpoint` path the wired weights are named
`layers.L.mlp.wired_{gate,up,down}_s` on the meta wrapper and aliased to the
checkpoint's per-expert tensors `...mlp.experts.s.{gate,up,down}_proj.weight`
(bf16, stored (out, in)), referenced in place in the shards. Stage 2's paging
rewrite is NOT produced on that path: make_moe_variant.py keys on the eager
export's folded/transposed layout and hard-codes Qwen1.5-MoE-A2.7B geometry,
so the hf path yields the stage-1 (dense, wired) graph only.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..hf_import import GraphSpec, HFImportError
from . import common

KINDS = ("gate", "up", "down")
_CKPT_PROJ = {"gate": "gate_proj", "up": "up_proj", "down": "down_proj"}


class SparseMoeOnnx(nn.Module):
    """One sparse-MoE layer in export form; replaces Qwen2MoeSparseMoeBlock.

    `wired` is [(gate, up, down)] * top_k real tensors (the eager script), or
    None for meta placeholders of the checkpoint's stored shapes (the
    `--hf-checkpoint` path, which splices them to the shards).

    HF's router (Qwen2MoeTopKRouter.forward): logits = linear(h, W);
    probs = softmax(fp32); topv, topi = topk(probs, k); with norm_topk_prob
    topv /= topv.sum(-1). The flag is baked into the graph at export.
    """

    def __init__(self, old, layer_idx, cfg, wired=None):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden = int(cfg.hidden_size)
        self.num_experts = int(cfg.num_experts)
        self.top_k = int(cfg.num_experts_per_tok)
        self.norm_topk = bool(cfg.norm_topk_prob)
        inter = int(cfg.moe_intermediate_size)
        self.gate = old.gate
        self.shared_expert = old.shared_expert
        self.shared_expert_gate = old.shared_expert_gate
        if wired is None:
            dt = old.gate.weight.dtype
            shapes = {"gate": (inter, self.hidden), "up": (inter, self.hidden),
                      "down": (self.hidden, inter)}
            wired = [tuple(torch.empty(*shapes[k], dtype=dt, device="meta")
                           for k in KINDS) for _ in range(self.top_k)]
        if len(wired) != self.top_k:
            raise HFImportError(f"layer {layer_idx}: {len(wired)} wired "
                                f"expert(s) for top_k={self.top_k}")
        for slot, tensors in enumerate(wired):
            for kind, t in zip(KINDS, tensors):
                setattr(self, f"wired_{kind}_{slot}",
                        nn.Parameter(t, requires_grad=False))

    def _gate_weights(self, topv):
        # The one place the absolute gate scale reaches the output: changing
        # norm_topk_prob needs the runtime changed too.
        if self.norm_topk:
            return topv / (topv.sum(dim=-1, keepdim=True) + 1e-6)
        return topv

    def forward(self, x):
        B, S, _ = x.shape
        flat = x.reshape(-1, self.hidden)
        if self.gate.num_experts != self.num_experts or \
                self.gate.top_k != self.top_k:
            raise HFImportError(
                f"layer {self.layer_idx} router is {self.gate.num_experts}/"
                f"{self.gate.top_k}, config says {self.num_experts}/"
                f"{self.top_k}")
        # HF's Qwen2MoeTopKRouter.forward, written out with an explicit
        # `.float()` BEFORE the softmax. Calling the module traced
        # `softmax(logits, dtype=float)` as Softmax-in-fp16 then Cast: not
        # HF's eager math (cast first, softmax in fp32), and a Softmax whose
        # fp16 input sits beside an fp32 output, which onnx2vkop's half-domain
        # pass refuses under -q fp16 AND --act fp32 ("would mix FP32 and FP16
        # operands"). Cast -> Softmax(fp32) -> TopK is the boundary it accepts.
        r = self.gate
        logits = F.linear(flat, r.weight)
        probs = F.softmax(logits.float(), dim=-1)
        topv, _topi = torch.topk(probs, self.top_k, dim=-1)
        if getattr(r, "norm_topk_prob", False):
            topv = topv / topv.sum(dim=-1, keepdim=True)
        # Cast back explicitly so no fp32 multiply drags the rest of the
        # block to fp32.
        topv = topv.to(logits.dtype)
        gate_w = self._gate_weights(topv).to(flat.dtype)
        out = None
        for slot in range(self.top_k):
            g = getattr(self, f"wired_gate_{slot}")
            u = getattr(self, f"wired_up_{slot}")
            d = getattr(self, f"wired_down_{slot}")
            w = gate_w[:, slot:slot + 1]
            h = F.silu(F.linear(flat, g)) * F.linear(flat, u)
            y = F.linear(h, d)
            out = y * w if out is None else out + y * w
        se = self.shared_expert
        sh = se.down_proj(F.silu(se.gate_proj(flat)) * se.up_proj(flat))
        sg = torch.sigmoid(self.shared_expert_gate(flat))
        return (out + sh * sg).reshape(B, S, self.hidden)


class Qwen2MoeLM(nn.Module):
    """Decoder wrapper: dense-Qwen2 attention, MLP = SparseMoeOnnx."""

    def __init__(self, model):
        super().__init__()
        from transformers.models.qwen2_moe.modeling_qwen2_moe import (
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
        self.scaling = float(lm.layers[0].self_attn.scaling)

    def block(self, layer, hidden, cos, sin, attention_bias, past_kv):
        B, q_len, _ = hidden.shape
        residual = hidden
        h = layer.input_layernorm(hidden)
        attn = layer.self_attn
        q = attn.q_proj(h).view(B, q_len, self.num_heads,
                                self.head_dim).transpose(1, 2)
        cs = (B, q_len, self.num_kv, self.head_dim)
        k = attn.k_proj(h).view(cs).transpose(1, 2)
        v = attn.v_proj(h).view(cs).transpose(1, 2)
        q, k = self._apply_rope(q, k, cos, sin)
        k_new = torch.cat([past_kv[:, 0], k], dim=2)
        v_new = torch.cat([past_kv[:, 1], v], dim=2)
        k_r = self._repeat_kv(k_new, self.groups)
        v_r = self._repeat_kv(v_new, self.groups)
        w = torch.matmul(q, k_r.transpose(2, 3)) * self.scaling
        w = w + attention_bias
        w = F.softmax(w, dim=-1, dtype=torch.float32).to(q.dtype)
        out = torch.matmul(w, v_r)
        out = out.transpose(1, 2).reshape(B, q_len,
                                          self.num_heads * self.head_dim)
        hidden = residual + attn.o_proj(out)
        residual = hidden
        h = layer.post_attention_layernorm(hidden)
        hidden = residual + layer.mlp(h)
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
    """Structural preconditions; each one is a silent-wrong-model if violated."""
    cfg = model.config
    qm = getattr(cfg, "quantization_config", None)
    if qm is not None:
        raise HFImportError(f"quantized checkpoint ({qm}): expert weights are "
                            f"not fp16/bf16 parameters; dequantize first")
    if cfg.decoder_sparse_step not in (None, 1) or cfg.mlp_only_layers:
        raise HFImportError(
            f"decoder_sparse_step={cfg.decoder_sparse_step}, mlp_only_layers="
            f"{cfg.mlp_only_layers}: not every layer is a sparse MoE block")
    if cfg.use_sliding_window:
        raise HFImportError("use_sliding_window=true: per-layer window "
                            "policies break the single attention_bias")
    if cfg.tie_word_embeddings:
        raise HFImportError("tie_word_embeddings=true is not the I/O this "
                            "exporter was written for")
    e0 = model.model.layers[0].mlp.experts
    want_gu = (cfg.num_experts, 2 * cfg.moe_intermediate_size,
               cfg.hidden_size)
    want_dp = (cfg.num_experts, cfg.hidden_size, cfg.moe_intermediate_size)
    if tuple(e0.gate_up_proj.shape) != want_gu or \
            tuple(e0.down_proj.shape) != want_dp:
        raise HFImportError(
            f"fused expert shapes {tuple(e0.gate_up_proj.shape)}/"
            f"{tuple(e0.down_proj.shape)}, expected {want_gu}/{want_dp}: "
            f"slot order would silently shift")


def make_wrapper(model, load_expert=None) -> Qwen2MoeLM:
    """Replace every layer's mlp with SparseMoeOnnx and wrap the model.

    `load_expert(layer, expert_id) -> (gate, up, down)` supplies the wired
    experts' real bytes (the eager script reads them from the shards); without
    it the wired experts are meta placeholders for the `--hf-checkpoint`
    splice. Mutates `model` (the old fused-expert containers are dropped).
    """
    check(model)
    cfg = model.config
    top_k = int(cfg.num_experts_per_tok)
    for idx, layer in enumerate(model.model.layers):
        wired = None
        if load_expert is not None:
            wired = [load_expert(idx, s) for s in range(top_k)]
        layer.mlp = SparseMoeOnnx(layer.mlp, idx, cfg, wired=wired)
    return Qwen2MoeLM(model).eval()


def annotate(proto, nlayers: int, top_k: int) -> list:
    """Graph edits the trace cannot carry; returns the TopK index output names.

    * `vkop_causal_mask:attention_bias` = 1: attention_bias is the standard
      causal bias (the converter's fused-attention / no-readback path).
    * Each router TopK's INDICES declared int32 [bs, top_k] and made a graph
      output. Undeclared, the converter guesses fp16, while vkop's TopK kernel
      only writes int32 (ops/Topk.hpp hard gate: crash at construction). A
      graph.output survives onnxoptimizer, value_info does not. The VALUES are
      left undeclared on purpose: as a graph output the loader would pre-bind
      them fp16 while the kernel writes fp32. Their width is k, not
      num_experts (declaring [1, num_experts] read the wrong tensor).
    """
    from onnx import TensorProto, helper
    proto.metadata_props.add(key="vkop_causal_mask:attention_bias", value="1")
    g = proto.graph
    names = []
    for n in g.node:
        if n.op_type != "TopK":
            continue
        vi = helper.make_tensor_value_info(n.output[1], TensorProto.INT32,
                                           ["bs", top_k])
        g.value_info.append(vi)
        g.output.append(vi)
        names.append(n.output[1])
    if len(names) != nlayers:
        raise HFImportError(f"{len(names)} TopK node(s), expected {nlayers}")
    return names


def build(config, dtype, index) -> GraphSpec:
    from transformers import Qwen2MoeForCausalLM
    cfg = common.text_config(config)
    common.refuse_dynamic_rope(cfg)
    model = common.meta_model(Qwen2MoeForCausalLM, cfg, dtype)
    w = make_wrapper(model)
    nlayers = len(w.layers)
    top_k = int(cfg.num_experts_per_tok)
    prefix = common.text_prefix(index) or "model."
    aliases = {}
    for L in range(nlayers):
        for s in range(top_k):
            for kind in KINDS:
                aliases[f"layers.{L}.mlp.wired_{kind}_{s}"] = (
                    f"{prefix}layers.{L}.mlp.experts.{s}."
                    f"{_CKPT_PROJ[kind]}.weight")
    missing = [v for v in aliases.values() if v not in index.entries]
    if missing:
        raise HFImportError(
            f"checkpoint lacks per-expert tensors (e.g. {missing[:2]}); a "
            f"fused-expert checkpoint layout is not supported")

    spec = None

    def post_splice(proto):
        names = annotate(proto, nlayers, top_k)
        # Keep the router outputs' row axis symbolic through the shape repair
        # that runs right after this hook (it reads spec.dynamic_axes).
        for n in names:
            spec.dynamic_axes[n] = {0: "bs"}

    spec = common.llm_spec(
        w, index, nlayers=nlayers, hidden=int(cfg.hidden_size),
        num_kv=w.num_kv, head_dim=w.head_dim, dtype=dtype,
        ckpt_prefix=prefix, aliases=aliases, post_splice=post_splice,
        summary={"builder": "qwen2_moe", "layers": nlayers,
                 "experts": int(cfg.num_experts), "top_k": top_k,
                 "wired": "experts 0..top_k-1 (stage-1 graph, unpaged)"})
    if int(cfg.num_experts) != 1:
        common.warn(
            f"qwen2_moe: slots are wired to experts 0..{top_k - 1}; with "
            f"{cfg.num_experts} experts the graph is the stage-1 (unpaged) "
            f"model, not HF's output, until expert paging is applied")
    return spec
