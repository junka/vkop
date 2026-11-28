"""Qwen3.5 text tower (model_type qwen3_5 / qwen3_5_text).

Moved from llm/exporter/export/qwen3_5_export_onnx.py, which now exports from
`make_wrapper`. Differences from the Qwen3 block:

  * hybrid layers: per `layer_types`, most layers are linear attention (gated
    delta net) and every Nth is full attention. Full layers carry a KV cache
    (B,2,NKV,kv,HD); linear layers carry a conv state (1,C_conv,K-1) and a
    recurrent matrix (1,Hv,Dk,Dv), both fp32.
  * q_proj is twice as wide: query then gate (`attn_output_gate`); the
    attention output is multiplied by sigmoid(gate).
  * 3D interleaved MRoPE with partial_rotary_factor (0.25): only the first
    rotary_dim of head_dim rotate. The t/h/w interleave is static, so it is
    one (B,q,3) @ (3,half) matmul against `mrope.M` instead of HF's
    Shape/Gather/Concat/Where chain.
  * zero-centred RMSNorm (`out * (1 + w)`): `1 + w` is folded into the
    parameter and the forward becomes `out * w`, the shape fuse_rms_norm
    knows. On a META model the in-place `+1` writes nowhere, so the fold is
    handed to the splice as `GraphSpec.addend` instead.
  * two custom ops, Conv1d and LinearAttention: both recurrences are strictly
    serial in seq, and unrolling them would pin seq to the export length.
    They are emitted as custom nodes (ops/Conv1d.hpp, ops/LinearAttention.hpp)
    and their output value_info is declared by `annotate_custom_ops`, which
    the legacy exporter cannot infer (`GraphSpec.post_splice`).

I/O (llm.onnx):
  in : inputs_embeds (B,q,H) fp16, position_ids (3,B,q) int64,
       attention_bias (B,1,q,kv) fp16, past_key_values_{i} (full layers),
       conv_state_{i} (1,C_conv,K-1) fp32, recurrent_state_{i} (1,Hv,Dk,Dv)
       fp32 (linear layers)
  out: logits, present_key_values_{i}, conv_state_out_{i},
       recurrent_state_out_{i}
"""
from __future__ import annotations

import dataclasses
import types
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..hf_import import GraphSpec, HFImportError, buffers_to_params
from . import common


@dataclasses.dataclass(frozen=True)
class Dims:
    nlayers: int
    hidden: int
    num_heads: int
    num_kv: int
    head_dim: int
    groups: int
    eps: float
    scaling: float
    layer_types: tuple
    full_layers: tuple
    lin_layers: tuple
    lin_heads_k: int
    lin_heads_v: int
    lin_dk: int
    lin_dv: int
    conv_k: int
    key_dim: int
    value_dim: int
    conv_dim: int
    rotary_dim: int

    @classmethod
    def from_config(cls, tcfg) -> "Dims":
        hd = tcfg.head_dim or tcfg.hidden_size // tcfg.num_attention_heads
        lt = tuple(tcfg.layer_types)
        bad = sorted(set(lt) - {"full_attention", "linear_attention"})
        if bad:
            raise HFImportError(f"qwen3_5: unknown layer types {bad}")
        kd = tcfg.linear_num_key_heads * tcfg.linear_key_head_dim
        vd = tcfg.linear_num_value_heads * tcfg.linear_value_head_dim
        return cls(
            nlayers=tcfg.num_hidden_layers, hidden=tcfg.hidden_size,
            num_heads=tcfg.num_attention_heads,
            num_kv=tcfg.num_key_value_heads, head_dim=hd,
            groups=tcfg.num_attention_heads // tcfg.num_key_value_heads,
            eps=tcfg.rms_norm_eps, scaling=hd ** -0.5, layer_types=lt,
            full_layers=tuple(i for i, t in enumerate(lt)
                              if t == "full_attention"),
            lin_layers=tuple(i for i, t in enumerate(lt)
                             if t == "linear_attention"),
            lin_heads_k=tcfg.linear_num_key_heads,
            lin_heads_v=tcfg.linear_num_value_heads,
            lin_dk=tcfg.linear_key_head_dim, lin_dv=tcfg.linear_value_head_dim,
            conv_k=tcfg.linear_conv_kernel_dim, key_dim=kd, value_dim=vd,
            conv_dim=kd * 2 + vd,
            rotary_dim=int(hd * tcfg.rope_parameters.get(
                "partial_rotary_factor", 1.0)))


# ---------------------------------------------------------------------------
# zero-centred RMSNorm fold
# ---------------------------------------------------------------------------

def fold_zero_centered_norms(module) -> Dict[str, float]:
    """Fold `1 + weight` into each Qwen3_5RMSNorm and drop the +1 from forward.

    Changing the weight alone is WRONG: HF's forward adds 1 again, giving
    `w + 2` (measured: logits 5% off HF). So both happen together. The gated
    norm (RMSNormGated) is untouched: it adds 1 explicitly in
    `_rmsnorm_gated`, inside linear attention only.

    Returns {parameter name: 1.0}, named as the wrapper names them
    (`layers.N....weight`, `norm.weight`). On a meta model the `add_` is a
    no-op, so the splice must apply this map to the payload (GraphSpec.addend)
    or the graph built for `1 + w` is fed the raw `w`.
    """
    folded = {}

    def make_forward(m):
        eps = m.eps

        def fwd(self, x):
            # ORDER MATTERS for the converter: `fuse_rms_norm` matches
            #   Cast -> Pow -> ReduceMean -> Add -> Sqrt -> Div -> Mul
            #        -> Cast -> Mul(weight)
            # (cast back to the activation dtype BEFORE the weight multiply).
            # HF Qwen3_5's multiply-then-cast left 0 of 14 norms fused.
            xf = x.float()
            var = xf.pow(2).mean(-1, keepdim=True)
            xf = xf * torch.rsqrt(var + eps)
            return self.weight * xf.to(x.dtype)  # weight already holds 1 + w

        return types.MethodType(fwd, m)

    def visit(m, path=""):
        for name, child in m.named_children():
            visit(child, f"{path}.{name}" if path else name)
        if type(m).__name__ == "Qwen3_5RMSNorm":
            with torch.no_grad():
                m.weight.data.add_(1.0)
            m.forward = make_forward(m)
            folded[f"{path}.weight"] = 1.0

    visit(module)
    return folded


# ---------------------------------------------------------------------------
# static interleaved MRoPE
# ---------------------------------------------------------------------------

def mrope_table(inv_freq: torch.Tensor, section) -> torch.Tensor:
    """(3, half): column j takes inv_freq[j] from row t/h/w per HF's
    interleave -- a static selection, so the whole HF chain is one matmul."""
    inv = inv_freq.float().cpu()
    half = inv.shape[0]
    section = list(section)
    if len(section) != 3:
        raise HFImportError(f"mrope_section {section}: expected 3 entries")
    s1, s2 = section[1], section[2]
    sel = torch.zeros(half, dtype=torch.long)
    for j in range(half):
        if j % 3 == 1 and j < 3 * s1:
            sel[j] = 1
        elif j % 3 == 2 and j < 3 * s2:
            sel[j] = 2
    M = torch.zeros(3, half, dtype=torch.float32)
    M[sel, torch.arange(half)] = inv
    return M


class StaticInterleavedMrope(nn.Module):
    def __init__(self, rotary, device=None):
        super().__init__()
        if rotary.inv_freq.device.type == "meta":
            # Rebuild the HF rotary on CPU from the same config for its value.
            with torch.device("cpu"):
                rotary = type(rotary)(config=rotary.config)
        M = mrope_table(rotary.inv_freq, rotary.mrope_section)
        self.M_value = M                     # real table, meta build or not
        self.register_buffer("M", M.to(device) if device else M)
        self.attention_scaling = rotary.attention_scaling

    def forward(self, x, position_ids):
        assert position_ids.ndim == 3 and position_ids.shape[0] == 3
        pos_t = position_ids.to(torch.float32).permute(1, 2, 0)
        freqs = torch.matmul(pos_t, self.M)          # (B,q,ROTARY_DIM/2)
        emb = torch.cat((freqs, freqs), dim=-1)      # (B,q,ROTARY_DIM)
        # HF casts back to the activation dtype; without this the fp32 cos/sin
        # promote q/k and the attention math silently runs at 2x width.
        return ((emb.cos() * self.attention_scaling).to(x.dtype),
                (emb.sin() * self.attention_scaling).to(x.dtype))


def apply_partial_rope(q, k, cos, sin, rotary_dim):
    """Rotate the first `rotary_dim` dims, pass the rest through."""
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
    h = rotary_dim // 2
    qr1, qr2 = q_rot[..., :h], q_rot[..., h:]
    kr1, kr2 = k_rot[..., :h], k_rot[..., h:]
    q_rot = q_rot * cos + torch.cat((-qr2, qr1), -1) * sin
    k_rot = k_rot * cos + torch.cat((-kr2, kr1), -1) * sin
    return (torch.cat([q_rot, q_pass], -1),
            torch.cat([k_rot, k_pass], -1))


# ---------------------------------------------------------------------------
# the two custom ops
# ---------------------------------------------------------------------------

def _gated_delta_rule(q, k, v, g, beta, state):
    """HF `torch_recurrent_gated_delta_rule`, one sequence: the reference
    `_LinearAttention.forward` runs and the self-check measures against; the
    runtime side is ops/LinearAttention.hpp."""
    B, S = q.shape[0], q.shape[1]
    q_ = q.transpose(1, 2).float()          # (B,Hk,S,Dk)
    k_ = k.transpose(1, 2).float()
    v_ = v.transpose(1, 2).float()          # (B,Hv,S,Dv)
    beta_ = beta.transpose(1, 2).float()
    g_ = g.transpose(1, 2).float()

    eps = 1e-6
    qn = (q_ * torch.rsqrt((q_ * q_).sum(-1, keepdim=True) + eps)
          / (q_.shape[-1] ** 0.5))
    kn = k_ * torch.rsqrt((k_ * k_).sum(-1, keepdim=True) + eps)

    # Expand to Hv heads here (reference only): the graph does this by index.
    rep = v_.shape[1] // qn.shape[1]
    if rep > 1:
        qn = qn.repeat_interleave(rep, dim=1)
        kn = kn.repeat_interleave(rep, dim=1)

    st = state.float() if state is not None else torch.zeros(
        B, v_.shape[1], qn.shape[-1], v_.shape[-1], dtype=torch.float32)

    outs = []
    for i in range(S):
        q_t, k_t, v_t = qn[:, :, i], kn[:, :, i], v_[:, :, i]
        st = st * torch.exp(g_[:, :, i])[..., None, None]
        kv = (st * k_t.unsqueeze(-1)).sum(-2)
        delta = (v_t - kv) * beta_[:, :, i].unsqueeze(-1)
        st = st + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
        outs.append((st * q_t.unsqueeze(-1)).sum(-2))
    out = torch.stack(outs, dim=2).transpose(1, 2).contiguous().to(q.dtype)
    return out, st


def make_custom_ops(d: Dims):
    """The two autograd Functions with this config's attributes baked in.

    `symbolic` is only called by the legacy TorchScript exporter and replaces
    the serial computation with one custom ONNX node; `forward` is the real
    implementation (HF's formula) used by the self-check. Attributes are
    closure constants rather than extra apply() arguments, so the traced node
    is exactly the one the script always emitted.
    """
    heads_v, dk, dv, conv_k = d.lin_heads_v, d.lin_dk, d.lin_dv, d.conv_k

    class _LinearAttention(torch.autograd.Function):
        @staticmethod
        def forward(ctx, q, k, v, g, beta, state):
            return _gated_delta_rule(q, k, v, g, beta, state)

        @staticmethod
        def symbolic(graph, q, k, v, decay, beta, state):
            # `outputs=` is the TorchScript exporter's multi-output form
            # (`num_outputs` is read as an attribute name and rejected).
            return graph.op("LinearAttention", q, k, v, decay, beta, state,
                            heads_i=heads_v, dk_i=dk, dv_i=dv, outputs=2)

    class _Conv1d(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, weight, state):
            """x [C,S] channel-major, weight [C,K], state [C,K-1].

            Every slice bound is a python constant or a `Tensor.size()` the
            exporter folds: `padded[:, k:k + S]` with `S = x.shape[1]` makes S
            a graph node that dangles once the function becomes one custom op,
            and the legacy exporter dies in peephole ("Argument passed to at()
            was not in the map"). `narrow` keeps the extent a tensor property.
            """
            K = conv_k
            padded = F.pad(x, (K - 1, 0))
            y = torch.zeros_like(x)
            for k in range(K):
                y = y + padded.narrow(1, k, x.size(1)) * weight[:, k].view(
                    -1, 1)
            hist = torch.cat([state, x], dim=1)
            state_out = hist.narrow(1, x.size(1), K - 1)
            return y, state_out

        @staticmethod
        def symbolic(graph, x, weight, state):
            return graph.op("Conv1d", x, weight, state, kernel_i=conv_k,
                            outputs=2)

    return _LinearAttention, _Conv1d


def _rmsnorm_gated(x, gate, weight, eps):
    """HF `Qwen3_5RMSNormGated`: `weight * norm(x)` then `* silu(gate)`.

    The gated norm is NOT zero-centred (its weight initializes to ones; HF
    multiplies by `weight`, not `1 + weight`) -- unlike every Qwen3_5RMSNorm.
    The export script used `1 + weight` until the move here, which doubled
    each linear-attention output: measured on a tiny checkpoint, wrapper vs HF
    logits 2.5e-3 apart at fp32 (S=7), 9.7e-8 with this form. Same op order as
    HF: norm in fp32, cast to the input dtype, weight, then the fp32 gate.
    """
    xf = x.float()
    var = xf.pow(2).mean(-1, keepdim=True)
    xf = xf * torch.rsqrt(var + eps)
    xf = (weight * xf.to(x.dtype)).float()
    xf = xf * F.silu(gate.float())
    return xf.to(x.dtype)


# ---------------------------------------------------------------------------
# layers
# ---------------------------------------------------------------------------

def linear_attention_forward(layer, x, conv_state, rec_state, d: Dims,
                             la_op, conv_op):
    """One gated delta net layer, in HF `Qwen3_5GatedDeltaNet.forward` order."""
    la = layer.linear_attn
    B, S, _ = x.shape

    mixed = la.in_proj_qkv(x)                       # (B,S,CONV_DIM)
    z = la.in_proj_z(x).reshape(B, S, d.lin_heads_v, d.lin_dv)
    b = la.in_proj_b(x)
    a = la.in_proj_a(x)

    # causal conv: HF's F.conv1d(padding=K-1)[:, :, :S] with the state prefix
    mixed_cs = mixed.transpose(1, 2).contiguous()    # (B,C_conv,S)
    w = la.conv1d.weight.squeeze(1)                  # (C_conv,K)
    # the custom Conv1d takes [C,S] (no batch); B == 1 is the contract
    assert B == 1, "Conv1d custom op assumes batch == 1"
    conv_out, conv_state_out = conv_op.apply(
        mixed_cs[0], w, conv_state[0])               # -> (C,S), (C,K-1)
    conv_out = conv_out.unsqueeze(0)
    conv_state_out = conv_state_out.unsqueeze(0)
    mixed = F.silu(conv_out).transpose(1, 2)         # (B,S,CONV_DIM)

    q, k_, v = torch.split(mixed, [d.key_dim, d.key_dim, d.value_dim], dim=-1)
    q = q.reshape(B, S, d.lin_heads_k, d.lin_dk)
    k_ = k_.reshape(B, S, d.lin_heads_k, d.lin_dk)
    v = v.reshape(B, S, d.lin_heads_v, d.lin_dv)

    beta = torch.sigmoid(b)
    # Decay in fp32 (fp16 A_log.exp() overflows to -inf, HF does the same),
    # handed over at the activation dtype: the LinearAttention kernel binds
    # every data tensor through one buffer type.
    g = (-torch.exp(la.A_log.float())
         * F.softplus(a.float() + la.dt_bias.float())).to(b.dtype)

    # Head expansion stays OUT of the graph: repeat_interleave exports as a
    # Tile (no vkop kernel) and would copy keys per token. LinearAttention.hpp
    # indexes value head h to key head h / (Hv/Hk) -- repeat_interleave's layout.
    core, rec_state_out = la_op.apply(q, k_, v, g, beta, rec_state)

    core = core.reshape(-1, d.lin_dv)
    zz = z.reshape(-1, d.lin_dv)
    out = _rmsnorm_gated(core, zz, la.norm.weight, d.eps)
    return (la.out_proj(out.reshape(B, S, d.value_dim)), conv_state_out,
            rec_state_out)


def full_attention_forward(layer, x, cos, sin, attention_bias, past_kv,
                           d: Dims, repeat_kv):
    attn = layer.self_attn
    B, S, _ = x.shape

    qg = attn.q_proj(x).view(B, S, d.num_heads, d.head_dim * 2)
    q, gate = torch.chunk(qg, 2, dim=-1)
    gate = gate.reshape(B, S, d.num_heads * d.head_dim)

    q = attn.q_norm(q).transpose(1, 2)
    kv_shape = (B, S, d.num_kv, d.head_dim)
    k = attn.k_norm(attn.k_proj(x).view(kv_shape)).transpose(1, 2)
    v = attn.v_proj(x).view(kv_shape).transpose(1, 2)

    q, k = apply_partial_rope(q, k, cos, sin, d.rotary_dim)

    k_new = torch.cat([past_kv[:, 0], k], dim=2)
    v_new = torch.cat([past_kv[:, 1], v], dim=2)
    k_r = repeat_kv(k_new, d.groups)
    v_r = repeat_kv(v_new, d.groups)

    w = torch.matmul(q, k_r.transpose(2, 3)) * d.scaling
    w = w + attention_bias
    w = F.softmax(w, dim=-1, dtype=torch.float32).to(q.dtype)
    out = torch.matmul(w, v_r).to(q.dtype)
    out = out.transpose(1, 2).reshape(B, S, d.num_heads * d.head_dim)
    out = out * torch.sigmoid(gate)                  # attn_output_gate
    return attn.o_proj(out), torch.stack([k_new, v_new], dim=1)


class Qwen3_5LM(nn.Module):
    """HF layer weights, hand-written layer loop; every state is explicit I/O."""

    def __init__(self, model):
        super().__init__()
        from transformers.models.qwen3_5.modeling_qwen3_5 import repeat_kv
        self._repeat_kv = repeat_kv
        full_cfg = model.config
        tcfg = getattr(full_cfg, "text_config", full_cfg)
        self.d = Dims.from_config(tcfg)
        lm = getattr(model.model, "language_model", model.model)
        # Fold BEFORE taking references: the forwards it installs are what
        # the layer loop below calls.
        self.folded = fold_zero_centered_norms(lm)
        self.layers = lm.layers
        self.norm = lm.norm
        self.lm_head = model.lm_head
        self.mrope = StaticInterleavedMrope(
            lm.rotary_emb, device="meta" if common.is_meta(model) else None)
        self._la, self._conv = make_custom_ops(self.d)

    def forward(self, inputs_embeds, position_ids, attention_bias, *states):
        d = self.d
        cos, sin = self.mrope(inputs_embeds, position_ids)
        nf, nl = len(d.full_layers), len(d.lin_layers)
        past_kvs = states[:nf]
        conv_states = states[nf:nf + nl]
        rec_states = states[nf + nl:]

        h = inputs_embeds
        presents, conv_outs, rec_outs = [], [], []
        fi = li = 0
        for idx, layer in enumerate(self.layers):
            residual = h
            h = layer.input_layernorm(h)
            if d.layer_types[idx] == "full_attention":
                a, pk = full_attention_forward(
                    layer, h, cos, sin, attention_bias, past_kvs[fi], d,
                    self._repeat_kv)
                fi += 1
                presents.append(pk)
            else:
                a, cs, rs = linear_attention_forward(
                    layer, h, conv_states[li], rec_states[li], d, self._la,
                    self._conv)
                li += 1
                conv_outs.append(cs)
                rec_outs.append(rs)
            h = residual + a

            residual = h
            h = layer.post_attention_layernorm(h)
            mlp = layer.mlp
            h = mlp.down_proj(mlp.act_fn(mlp.gate_proj(h)) * mlp.up_proj(h))
            h = residual + h

        logits = self.lm_head(self.norm(h))
        return (logits, *presents, *conv_outs, *rec_outs)


def make_wrapper(model) -> Qwen3_5LM:
    return Qwen3_5LM(model).eval()


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def llm_io(d: Dims, dtype=torch.float16, *, seq: int = 16, device="cpu"):
    """(inputs, input_names, output_names, dynamic_axes).

    Every dummy goes on the model's own device: under meta a single CPU tensor
    breaks the trace ("Tensor on device cpu is not on the expected device
    meta")."""
    dev = torch.device(device)
    B, S = 1, seq
    if dev.type == "meta":
        inputs_embeds = torch.empty(B, S, d.hidden, dtype=dtype, device=dev)
    else:
        inputs_embeds = torch.randn(B, S, d.hidden, dtype=dtype, device=dev)
    pos = torch.arange(S, dtype=torch.long, device=dev).view(
        1, 1, -1).expand(3, B, -1)
    causal = torch.triu(torch.full((S, S), torch.finfo(dtype).min,
                                   dtype=dtype, device=dev), diagonal=1)
    attention_bias = causal.unsqueeze(0).unsqueeze(0)
    past_kvs = tuple(torch.zeros(B, 2, d.num_kv, 0, d.head_dim, dtype=dtype,
                                 device=dev) for _ in d.full_layers)
    conv_states = tuple(torch.zeros(B, d.conv_dim, d.conv_k - 1,
                                    dtype=torch.float32, device=dev)
                        for _ in d.lin_layers)
    rec_states = tuple(torch.zeros(B, d.lin_heads_v, d.lin_dk, d.lin_dv,
                                   dtype=torch.float32, device=dev)
                       for _ in d.lin_layers)
    inputs = (inputs_embeds, pos, attention_bias,
              *past_kvs, *conv_states, *rec_states)
    input_names = (["inputs_embeds", "position_ids", "attention_bias"]
                   + [f"past_key_values_{i}" for i in d.full_layers]
                   + [f"conv_state_{i}" for i in d.lin_layers]
                   + [f"recurrent_state_{i}" for i in d.lin_layers])
    output_names = (["logits"]
                    + [f"present_key_values_{i}" for i in d.full_layers]
                    + [f"conv_state_out_{i}" for i in d.lin_layers]
                    + [f"recurrent_state_out_{i}" for i in d.lin_layers])
    dynamic_axes = {
        "inputs_embeds": {0: "batch", 1: "seq"},
        "position_ids": {1: "batch", 2: "seq"},
        "attention_bias": {0: "batch", 2: "q_len", 3: "kv_len"},
        "logits": {0: "batch", 1: "seq"},
    }
    for i in d.full_layers:
        dynamic_axes[f"past_key_values_{i}"] = {0: "batch", 3: "kv_len"}
        dynamic_axes[f"present_key_values_{i}"] = {0: "batch", 3: "kv_len"}
    return inputs, input_names, output_names, dynamic_axes


# ---------------------------------------------------------------------------
# custom-op output declarations
# ---------------------------------------------------------------------------

def annotate_custom_ops(m, d: Dims) -> int:
    """Declare value_info for the Conv1d / LinearAttention outputs, in place.

    The legacy exporter infers no shape for a custom node, so its outputs are
    in neither value_info nor the graph outputs; onnx2vkop then sees
    `shape: []` and the dtype propagation, shape inference and fp16 domain
    decisions all miss (measured: Conv1d's input judged fp16, runtime said
    "fp32-only"). The shapes are written from each op's semantics. Returns
    how many entries were added.
    """
    from onnx import TensorProto, helper

    def _vi(name, dtype, dims):
        return helper.make_tensor_value_info(name, dtype, dims)

    g = m.graph
    have = {v.name for v in list(g.value_info) + list(g.input)
            + list(g.output)}
    added = 0
    for n in g.node:
        if n.op_type == "Conv1d":
            # out [C, S] and state_out [C, K-1]; C and S are the input's.
            x_name = n.input[0]
            x_vi = next((v for v in list(g.value_info) + list(g.input)
                         if v.name == x_name), None)
            c_dim = x_vi.type.tensor_type.shape.dim[0] if x_vi else None
            c = (c_dim.dim_value if (c_dim and c_dim.HasField("dim_value"))
                 else None)
            # out0 follows x's dtype; out1, the conv state, is fp32 in every
            # build (conv1d.comp carries it across rounds). Declaring both
            # FLOAT broke the 27B: x is fp16, the op built the fp32 shader and
            # read halves as floats (output max 1.47e10 instead of ~±40).
            x_dtype = (x_vi.type.tensor_type.elem_type if x_vi else
                       TensorProto.FLOAT)
            g.value_info.append(_vi(n.output[0], x_dtype,
                                    [c if c else "C", "seq"]))
            added += 1
            if n.output[1] not in have:
                g.value_info.append(_vi(n.output[1], TensorProto.FLOAT,
                                        [c if c else "C", d.conv_k - 1]))
                added += 1
        elif n.op_type == "LinearAttention":
            # out [Hv, S, Dv] and state_out [Hv, Dk, Dv]
            for out, dims in ((n.output[0], [d.lin_heads_v, "seq", d.lin_dv]),
                              (n.output[1], [d.lin_heads_v, d.lin_dk,
                                             d.lin_dv])):
                if out in have:
                    continue
                g.value_info.append(_vi(out, TensorProto.FLOAT, dims))
                added += 1
    return added


# ---------------------------------------------------------------------------
# self-check (real weights only)
# ---------------------------------------------------------------------------

def self_check(model, wrapper, inputs, tol: float = 0.05) -> float:
    """Wrapper logits against HF's own text model on the same inputs.

    Both sides read the same (folded) weights -- HF's norms were re-pointed
    at the folded forward, so any op-order error shows in the logits."""
    text_model = getattr(model.model, "language_model", model.model)
    emb, pos = inputs[0], inputs[1]
    with torch.no_grad():
        ref = text_model(inputs_embeds=emb, position_ids=pos, use_cache=False)
        ref_logits = model.lm_head(ref.last_hidden_state)
        got = wrapper(*inputs)[0]
    diff = (got.float() - ref_logits.float()).abs()
    print(f"[self-check] logits maxdiff={diff.max().item():.6g} "
          f"mean={diff.mean().item():.6g}")
    if diff.max().item() >= tol:
        raise HFImportError("wrapper diverges from HF")
    return diff.max().item()


# ---------------------------------------------------------------------------
# --hf-checkpoint entry
# ---------------------------------------------------------------------------

def build(config, dtype, index) -> GraphSpec:
    from transformers import AutoModelForCausalLM
    if dtype != torch.float16:
        # The graph's contract is fp16 activations with fp32 states; an fp32
        # trace would Cast every fp16 input (measured 75 Casts vs 43) and the
        # Attention boundary then rejects the bias dtype.
        common.warn("qwen3_5 traces at fp16 regardless of --hf-trace-dtype")
        dtype = torch.float16
    with torch.device("meta"):
        # dtype is REQUIRED: from_config defaults to fp32 (see above).
        try:
            model = AutoModelForCausalLM.from_config(config, dtype=dtype)
        except TypeError:
            model = AutoModelForCausalLM.from_config(config, torch_dtype=dtype)
    model.eval()
    w = make_wrapper(model)
    d = w.d
    converted = buffers_to_params(w)
    if converted != ["mrope.M"]:
        raise HFImportError(f"unexpected buffers {converted}; only mrope.M "
                            f"has a known value")
    prefix = common.text_prefix(index)
    inputs, ins, outs, dyn = llm_io(d, dtype, device="meta")
    return GraphSpec(
        wrapper=w, dummies=inputs, input_names=ins, output_names=outs,
        dynamic_axes=dyn, constants={"mrope.M": w.mrope.M_value.numpy()},
        aliases=common.tied_head_alias(index, prefix), ckpt_prefix=prefix,
        addend=dict(w.folded),
        post_splice=lambda proto: annotate_custom_ops(proto, d),
        summary={"builder": "qwen3_5", "layers": d.nlayers,
                 "full": len(d.full_layers), "linear": len(d.lin_layers),
                 "heads": d.num_heads, "kv_heads": d.num_kv,
                 "head_dim": d.head_dim, "rotary_dim": d.rotary_dim,
                 "ckpt_prefix": prefix, "norm_addends": len(w.folded)})
