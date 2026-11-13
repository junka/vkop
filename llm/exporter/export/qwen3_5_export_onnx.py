"""导出 Qwen3.5 系列（`Qwen3_5ForConditionalGeneration` 的文本塔）到 ONNX。

与 qwen3_export_onnx.py（纯文本 Qwen3）的区别：

  · **混合层**：每 4 层里 3 层是 linear attention（gated delta net），
    1 层是普通 full attention（`layer_types` 决定）。两种状态完全不同：
      - full_attention  -> KV cache  (B, 2, NKV, kv, HD)   每层一个
      - linear_attention-> conv state (C, K-1) + 递推矩阵 (Hv, Dk, Dv)
  · **q_proj 输出 2 倍宽**：前一半 query，后一半 gate（`attn_output_gate`），
    attention 输出要乘 `sigmoid(gate)`。
  · **3D MRoPE**（不是 Qwen3 的 2D RoPE），`partial_rotary_factor=0.25`，
    只有前 1/4 的 head_dim 参与旋转。
  · **零中心化 RMSNorm**：HF 是 `out * (1 + weight)`，本脚本导出前把每个
    RMSNorm 的 weight 预加 1，图里就退化成标准的 `out * w`。
  · **两个自定义算子**：`Conv1d` 与 `LinearAttention`。递推在序列维上是
    严格串行的，展开成普通 ONNX op 会把 seq 钉死在导出时的长度（prefill
    2048 步意味着上万个节点），所以这两位以自定义节点发出，由运行时的
    ops/Conv1d.hpp / ops/LinearAttention.hpp 承接。

I/O 契约（llm.onnx）：

  入：inputs_embeds        (B, q, HIDDEN)              fp16
      position_ids         (3, B, q)                   int64   3D MRoPE
      attention_bias       (B, 1, q, kv)               fp16    full_attention 层用
      past_key_values_{i}  (B, 2, NKV, kv, HD)         fp16    i ∈ full layers
      conv_state_{i}       (1, C_conv, K-1)            fp32    i ∈ linear layers
      recurrent_state_{i}  (1, Hv, Dk, Dv)             fp32    i ∈ linear layers
  出：logits               (B, q, VOCAB)               fp16
      present_key_values_{i} / conv_state_out_{i} / recurrent_state_out_{i}

用法:
  MODEL_PATH=~/models/Qwen3.5-27B python3 export/qwen3_5_export_onnx.py
  EXPORT_PATH=llm3_5.onnx MODEL_PATH=... python3 export/qwen3_5_export_onnx.py
"""

import os

import torch
import torch.nn as nn
import torch.nn.functional as F

MODEL_PATH = os.environ.get("MODEL_PATH") or "Qwen/Qwen3.5-27B"
EXPORT_PATH = os.environ.get("EXPORT_PATH") or "llm.onnx"
OPSET = 17

# Prefill dummy length: the graph's seq axis is dynamic (the two serial ops take
# their extent from the tensor), so this only has to exercise each layer type.
SEQ_LEN = int(os.environ.get("SEQ_LEN", "16"))

print(f"[load] model = {MODEL_PATH}")

from transformers import AutoModelForCausalLM  # noqa: E402
from transformers.models.qwen3_5.modeling_qwen3_5 import (  # noqa: E402
    repeat_kv,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    attn_implementation="eager",
    torch_dtype=torch.float16,
    local_files_only=os.path.isdir(os.path.expanduser(MODEL_PATH)),
    trust_remote_code=True,
)
model.eval()

# ---------------------------------------------------------------------------
# 架构参数：文本塔的 config，全部动态读取
# ---------------------------------------------------------------------------
full_cfg = model.config
tcfg = getattr(full_cfg, "text_config", full_cfg)

NLAYERS = tcfg.num_hidden_layers
HIDDEN = tcfg.hidden_size
NUM_HEADS = tcfg.num_attention_heads
NUM_KV_HEADS = tcfg.num_key_value_heads
HEAD_DIM = tcfg.head_dim or HIDDEN // NUM_HEADS
NUM_KV_GROUPS = NUM_HEADS // NUM_KV_HEADS
VOCAB = tcfg.vocab_size
EPS = tcfg.rms_norm_eps
SCALING = HEAD_DIM ** -0.5

LAYER_TYPES = list(tcfg.layer_types)
FULL_LAYERS = [i for i, t in enumerate(LAYER_TYPES) if t == "full_attention"]
LIN_LAYERS = [i for i, t in enumerate(LAYER_TYPES) if t == "linear_attention"]

LIN_HEADS_K = tcfg.linear_num_key_heads
LIN_HEADS_V = tcfg.linear_num_value_heads
LIN_DK = tcfg.linear_key_head_dim
LIN_DV = tcfg.linear_value_head_dim
CONV_K = tcfg.linear_conv_kernel_dim
KEY_DIM = LIN_HEADS_K * LIN_DK
VALUE_DIM = LIN_HEADS_V * LIN_DV
CONV_DIM = KEY_DIM * 2 + VALUE_DIM

ROTARY_DIM = int(HEAD_DIM * tcfg.rope_parameters.get("partial_rotary_factor", 1.0))
HEAD_REP = LIN_HEADS_V // LIN_HEADS_K

print(f"[config] layers={NLAYERS} (full={len(FULL_LAYERS)}, "
      f"linear={len(LIN_LAYERS)}) hidden={HIDDEN} heads={NUM_HEADS} "
      f"kv_heads={NUM_KV_HEADS} head_dim={HEAD_DIM} rotary_dim={ROTARY_DIM} "
      f"vocab={VOCAB}")
print(f"[config] linear: k_heads={LIN_HEADS_K} v_heads={LIN_HEADS_V} "
      f"dk={LIN_DK} dv={LIN_DV} conv_k={CONV_K} conv_dim={CONV_DIM} "
      f"head_rep={HEAD_REP}")

lm = getattr(model.model, "language_model", model.model)
layers = lm.layers
norm = lm.norm
rotary_emb = lm.rotary_emb
lm_head = model.lm_head


# ---------------------------------------------------------------------------
# 零中心化 RMSNorm 预折叠：HF 的 forward 是 `out * (1 + weight)`。
# 把 `1 + weight` 写回参数，再让 forward 只做 `out * weight`，图里就只剩一个
# 标准 RMSNorm 乘法 —— 否则每层都会多出一个带常量 initializer 的 Mul，
# converter 的 fuse_rms_norm 看到的是「9 op + 额外 Mul」而不是它认识的形状。
# ---------------------------------------------------------------------------
def fold_zero_centered_norms(module):
    """把 `1 + weight` 折进参数，并把模块的 forward 换成不再加 1 的版本。

    只改 weight 不改 forward 是**错的**：HF 的 forward 自己还会再加一次 1，
    于是变成 `w + 2`（实测 logits 与 HF 差 5%）。所以这里成对地做：参数
    变成 `1 + w`，forward 变成 `out * w`。gated 版(RMSNormGated)不走这条
    路径 —— 它在 `_rmsnorm_gated` 里显式加 1，而且只作用在 linear attention
    内部，不在 HF 自己的层循环里被调用。
    """
    import types

    folded = 0

    def make_forward(m):
        eps = m.eps

        def fwd(self, x):
            # ORDER MATTERS for the converter. `fuse_rms_norm` matches
            #   Cast -> Pow -> ReduceMean -> Add -> Sqrt -> Div -> Mul
            #        -> Cast -> Mul(weight)
            # i.e. the normalization casts back to the activation dtype BEFORE
            # the weight multiply (HF Qwen3RMSNorm's
            # `self.weight * hidden_states.to(input_dtype)`). HF's Qwen3_5
            # multiply-then-cast order produces `Mul -> Cast` and the pattern
            # never matches -- measured: 0 of 14 fused, which left every
            # RMSNorm as a 9-op fp32 island and starved the fp16 domain.
            xf = x.float()
            var = xf.pow(2).mean(-1, keepdim=True)
            xf = xf * torch.rsqrt(var + eps)
            return self.weight * xf.to(x.dtype)  # weight already holds 1 + w

        return types.MethodType(fwd, m)

    def visit(m):
        nonlocal folded
        for child in m.children():
            visit(child)
        if type(m).__name__ == "Qwen3_5RMSNorm":
            with torch.no_grad():
                m.weight.data.add_(1.0)
            m.forward = make_forward(m)
            folded += 1

    visit(module)
    return folded


_folded = fold_zero_centered_norms(lm)
print(f"[norm] pre-folded {_folded} zero-centered RMSNorm weight(s) to (1+w)")


# ---------------------------------------------------------------------------
# 静态 interleaved MRoPE —— 与 qwen3vl_export_onnx.py 同一套做法。
# HF 的 recomposition_frequencies 把 t/h/w 三行按列交错，导出时会变成一条
# Shape/Gather/Concat/Where 的 int64 动态链；但「第 j 列来自哪一行」纯静态，
# 整条链等价于一次 (B,q,3) @ (3,half) 的 matmul。
# ---------------------------------------------------------------------------
class StaticInterleavedMrope(nn.Module):
    def __init__(self, rotary):
        super().__init__()
        inv_freq = rotary.inv_freq.float()
        half = inv_freq.shape[0]
        section = list(rotary.mrope_section)
        assert len(section) == 3, section
        s1, s2 = section[1], section[2]
        sel = torch.zeros(half, dtype=torch.long)
        for j in range(half):
            if j % 3 == 1 and j < 3 * s1:
                sel[j] = 1
            elif j % 3 == 2 and j < 3 * s2:
                sel[j] = 2
            else:
                sel[j] = 0
        M = torch.zeros(3, half, dtype=torch.float32)
        M[sel, torch.arange(half)] = inv_freq
        self.register_buffer("M", M)
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
    """前 `rotary_dim` 维旋转，其余原样透传（partial rotary）。"""
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
# 两个自定义算子的发射器。
#
# `symbolic` 只在导出时被 legacy TorchScript exporter 调用，把这一整段串行
# 计算替换成一个自定义 ONNX 节点。`forward` 是真的实现（与 HF 同一公式），
# 导出前用它做数值自证；导出后运行时的 ops/*.hpp 接的是同一份语义。
# ---------------------------------------------------------------------------
class _LinearAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, g, beta, state):
        return _gated_delta_rule(q, k, v, g, beta, state)

    @staticmethod
    def symbolic(graph, q, k, v, decay, beta, state):
        # `num_outputs` is not a valid attribute spelling in this exporter
        # (it is read as an ONNX attribute name and rejected); `outputs=` is
        # the TorchScript exporter's own multi-output form.
        return graph.op("LinearAttention", q, k, v, decay, beta, state,
                        heads_i=LIN_HEADS_V, dk_i=LIN_DK, dv_i=LIN_DV,
                        outputs=2)


class _Conv1d(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, state):
        """x [C,S] channel-major, weight [C,K], state [C,K-1]. -> (y, state_out).

        Every slice bound is either a python constant or a `Tensor.size()` call
        the exporter can fold. An earlier version wrote `padded[:, k:k + S]`
        with `S = x.shape[1]`, which turns S into a graph NODE; replacing the
        whole function with one custom op then leaves that node dangling and
        the legacy exporter dies in peephole with "Argument passed to at() was
        not in the map". `narrow` keeps the extent a tensor property instead.
        """
        C = x.size(0)
        K = CONV_K
        padded = F.pad(x, (K - 1, 0))
        y = torch.zeros_like(x)
        for k in range(K):
            y = y + padded.narrow(1, k, x.size(1)) * weight[:, k].view(-1, 1)
        hist = torch.cat([state, x], dim=1)
        state_out = hist.narrow(1, x.size(1), K - 1)
        return y, state_out

    @staticmethod
    def symbolic(graph, x, weight, state):
        return graph.op("Conv1d", x, weight, state, kernel_i=CONV_K,
                        outputs=2)


def _rmsnorm_gated(x, gate, weight, eps):
    """HF `Qwen3_5RMSNormGated`：norm * silu(gate)，weight 零中心化。"""
    xf = x.float()
    var = xf.pow(2).mean(-1, keepdim=True)
    xf = xf * torch.rsqrt(var + eps)
    xf = xf * (1.0 + weight.float())
    xf = xf * F.silu(gate.float())
    return xf.to(x.dtype)


def _gated_delta_rule(q, k, v, g, beta, state):
    """HF `torch_recurrent_gated_delta_rule`，单序列。

    这是 `_LinearAttention.forward` 的实现，也是导出前数值自证的基准。
    运行时那一侧是等价的 GPU kernel（ops/LinearAttention.hpp）。
    """
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


# ---------------------------------------------------------------------------
# Linear attention 层
# ---------------------------------------------------------------------------
def linear_attention_forward(layer, x, conv_state, rec_state):
    """一层 gated delta net，按 HF `Qwen3_5GatedDeltaNet.forward` 的顺序。"""
    la = layer.linear_attn
    B, S, _ = x.shape

    mixed = la.in_proj_qkv(x)                       # (B,S,CONV_DIM)
    z = la.in_proj_z(x).reshape(B, S, LIN_HEADS_V, LIN_DV)
    b = la.in_proj_b(x)
    a = la.in_proj_a(x)

    # 因果卷积：HF 的 F.conv1d(padding=K-1)[:, :, :S] 与 state 拼接等价。
    mixed_cs = mixed.transpose(1, 2).contiguous()    # (B,C_conv,S)
    w = la.conv1d.weight.squeeze(1)                  # (C_conv,K)
    # 自定义 Conv1d 只吃 [C,S]（无 batch 维）；B=1 是解码/导出的契约。
    assert B == 1, "Conv1d custom op assumes batch == 1"
    conv_out, conv_state_out = _Conv1d.apply(
        mixed_cs[0], w, conv_state[0])               # -> (C,S), (C,K-1)
    conv_out = conv_out.unsqueeze(0)
    conv_state_out = conv_state_out.unsqueeze(0)
    mixed = F.silu(conv_out).transpose(1, 2)         # (B,S,CONV_DIM)

    q, k_, v = torch.split(mixed, [KEY_DIM, KEY_DIM, VALUE_DIM], dim=-1)
    q = q.reshape(B, S, LIN_HEADS_K, LIN_DK)
    k_ = k_.reshape(B, S, LIN_HEADS_K, LIN_DK)
    v = v.reshape(B, S, LIN_HEADS_V, LIN_DV)

    beta = torch.sigmoid(b)
    # The decay is computed in fp32 (HF does the same: in fp16 A_log.exp()
    # overflows to -inf) and then handed over at the ACTIVATION dtype, so every
    # data tensor the LinearAttention op reads has one width. The op's kernel
    # binds all of them through a single buffer type, which cannot be per-tensor.
    g = (-torch.exp(la.A_log.float())
         * F.softplus(a.float() + la.dt_bias.float())).to(b.dtype)

    # Head expansion stays OUT of the graph: repeat_interleave exports as a
    # Tile, which vkop has no kernel for, and materializing it would copy the
    # key tensor per token (96 x ~50 MB at prefill). ops/LinearAttention.hpp
    # indexes value head h to key head h / (Hv/Hk) instead, which is exactly
    # repeat_interleave's layout.

    core, rec_state_out = _LinearAttention.apply(q, k_, v, g, beta, rec_state)

    core = core.reshape(-1, LIN_DV)
    zz = z.reshape(-1, LIN_DV)
    out = _rmsnorm_gated(core, zz, la.norm.weight, EPS)
    return la.out_proj(out.reshape(B, S, VALUE_DIM)), conv_state_out, rec_state_out


# ---------------------------------------------------------------------------
# Full attention 层
# ---------------------------------------------------------------------------
def full_attention_forward(layer, x, cos, sin, attention_bias, past_kv):
    attn = layer.self_attn
    B, S, _ = x.shape

    qg = attn.q_proj(x).view(B, S, NUM_HEADS, HEAD_DIM * 2)
    q, gate = torch.chunk(qg, 2, dim=-1)
    gate = gate.reshape(B, S, NUM_HEADS * HEAD_DIM)

    q = attn.q_norm(q).transpose(1, 2)
    kv_shape = (B, S, NUM_KV_HEADS, HEAD_DIM)
    k = attn.k_norm(attn.k_proj(x).view(kv_shape)).transpose(1, 2)
    v = attn.v_proj(x).view(kv_shape).transpose(1, 2)

    q, k = apply_partial_rope(q, k, cos, sin, ROTARY_DIM)

    k_new = torch.cat([past_kv[:, 0], k], dim=2)
    v_new = torch.cat([past_kv[:, 1], v], dim=2)
    k_r = repeat_kv(k_new, NUM_KV_GROUPS)
    v_r = repeat_kv(v_new, NUM_KV_GROUPS)

    w = torch.matmul(q, k_r.transpose(2, 3)) * SCALING
    w = w + attention_bias
    w = F.softmax(w, dim=-1, dtype=torch.float32).to(q.dtype)
    out = torch.matmul(w, v_r).to(q.dtype)
    out = out.transpose(1, 2).reshape(B, S, NUM_HEADS * HEAD_DIM)
    out = out * torch.sigmoid(gate)                  # attn_output_gate
    return attn.o_proj(out), torch.stack([k_new, v_new], dim=1)


class Qwen3_5LMOnnx(nn.Module):
    """复用 HF 层权重、手写层循环；状态全部是显式张量 I/O。"""

    def __init__(self):
        super().__init__()
        self.layers = layers
        self.norm = norm
        self.lm_head = lm_head
        self.mrope = StaticInterleavedMrope(rotary_emb)

    def forward(self, inputs_embeds, position_ids, attention_bias, *states):
        cos, sin = self.mrope(inputs_embeds, position_ids)
        nf, nl = len(FULL_LAYERS), len(LIN_LAYERS)
        past_kvs = states[:nf]
        conv_states = states[nf:nf + nl]
        rec_states = states[nf + nl:]

        h = inputs_embeds
        presents, conv_outs, rec_outs = [], [], []
        fi = li = 0
        for idx, layer in enumerate(self.layers):
            residual = h
            h = layer.input_layernorm(h)
            if LAYER_TYPES[idx] == "full_attention":
                a, pk = full_attention_forward(
                    layer, h, cos, sin, attention_bias, past_kvs[fi])
                fi += 1
                presents.append(pk)
            else:
                a, cs, rs = linear_attention_forward(
                    layer, h, conv_states[li], rec_states[li])
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


# ---------------------------------------------------------------------------
# dummy inputs
# ---------------------------------------------------------------------------
B, S = 1, SEQ_LEN

inputs_embeds = torch.randn(B, S, HIDDEN, dtype=torch.float16)
pos = torch.arange(S, dtype=torch.long).view(1, 1, -1).expand(3, B, -1)
causal = torch.triu(
    torch.full((S, S), torch.finfo(torch.float16).min, dtype=torch.float16),
    diagonal=1)
attention_bias = causal.unsqueeze(0).unsqueeze(0)

past_kvs = tuple(torch.zeros(B, 2, NUM_KV_HEADS, 0, HEAD_DIM,
                             dtype=torch.float16) for _ in FULL_LAYERS)
conv_states = tuple(torch.zeros(B, CONV_DIM, CONV_K - 1, dtype=torch.float32)
                    for _ in LIN_LAYERS)
rec_states = tuple(torch.zeros(B, LIN_HEADS_V, LIN_DK, LIN_DV,
                               dtype=torch.float32) for _ in LIN_LAYERS)

inputs = (inputs_embeds, pos, attention_bias,
          *past_kvs, *conv_states, *rec_states)

input_names = (["inputs_embeds", "position_ids", "attention_bias"]
               + [f"past_key_values_{i}" for i in FULL_LAYERS]
               + [f"conv_state_{i}" for i in LIN_LAYERS]
               + [f"recurrent_state_{i}" for i in LIN_LAYERS])
output_names = (["logits"]
                + [f"present_key_values_{i}" for i in FULL_LAYERS]
                + [f"conv_state_out_{i}" for i in LIN_LAYERS]
                + [f"recurrent_state_out_{i}" for i in LIN_LAYERS])

dynamic_axes = {
    "inputs_embeds": {0: "batch", 1: "seq"},
    "position_ids": {1: "batch", 2: "seq"},
    "attention_bias": {0: "batch", 2: "q_len", 3: "kv_len"},
    "logits": {0: "batch", 1: "seq"},
}
for i in FULL_LAYERS:
    dynamic_axes[f"past_key_values_{i}"] = {0: "batch", 3: "kv_len"}
    dynamic_axes[f"present_key_values_{i}"] = {0: "batch", 3: "kv_len"}

wrapper = Qwen3_5LMOnnx()
wrapper.eval()


# ---------------------------------------------------------------------------
# 导出前数值自证：整图 forward 对上 HF 原生 forward。
# ---------------------------------------------------------------------------
def self_check():
    """把 wrapper 的输出与 HF `Qwen3_5TextModel` 的逐层输出对比。

    只跑得动小模型，所以默认关闭（SELF_CHECK=1 打开）。它对拍的是两侧都
    读同一份权重，因此任何 op 顺序错误都会直接体现在 logits 上。
    """
    if os.environ.get("SELF_CHECK", "") != "1":
        return
    print("[self-check] running HF reference forward ...")
    text_model = getattr(model.model, "language_model", model.model)
    with torch.no_grad():
        ref = text_model(inputs_embeds=inputs_embeds,
                         position_ids=pos,
                         use_cache=False)
        ref_hidden = ref.last_hidden_state
        ref_logits = model.lm_head(ref_hidden)
        got = wrapper(inputs_embeds, pos, attention_bias,
                      *past_kvs, *conv_states, *rec_states)[0]
    diff = (got.float() - ref_logits.float()).abs()
    print(f"[self-check] logits maxdiff={diff.max().item():.6g} "
          f"mean={diff.mean().item():.6g}")
    assert diff.max().item() < 0.05, "wrapper diverges from HF"


self_check()

print(f"[export] {NLAYERS} layers, {len(input_names)} inputs / "
      f"{len(output_names)} outputs, opset={OPSET}, dummy S={S} ...")
with torch.no_grad():
    torch.onnx.export(
        wrapper, inputs, EXPORT_PATH,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=OPSET,
        do_constant_folding=True,
        # legacy TorchScript exporter：dynamo 对 RoPE 的 cos/sin 数据依赖会
        # trigger GuardOnDataDependentSymNode（qwen3vl 的脚本同样如此）。
        dynamo=False,
    )
print(f"[✓] exported -> {EXPORT_PATH}")


# ---------------------------------------------------------------------------
# 给两个自定义节点的输出补 value_info。
#
# torch 的 legacy exporter 不给自定义节点推导形状，所以它们的输出既不在
# value_info 也不在图输出里。onnx2vkop 读不到形状就退化成 `shape: []`，之后
# 的 dtype 传播、形状推断、fp16 域判定全部落空（实测：Conv1d 的输入被判成
# fp16，运行时报 "fp32-only"）。这里按算子的语义把形状显式写出来。
# ---------------------------------------------------------------------------
import onnx  # noqa: E402
from onnx import helper, TensorProto  # noqa: E402


def _vi(name, dtype, dims):
    return helper.make_tensor_value_info(name, dtype, dims)


def annotate_custom_ops(path):
    m = onnx.load(path)
    g = m.graph
    have = {v.name for v in list(g.value_info) + list(g.input) + list(g.output)}
    added = 0
    for n in g.node:
        if n.op_type == "Conv1d":
            # out [C, S] and state_out [C, K-1]; C and S are the input's.
            x_name = n.input[0]
            x_vi = next((v for v in list(g.value_info) + list(g.input)
                         if v.name == x_name), None)
            c_dim = x_vi.type.tensor_type.shape.dim[0] if x_vi else None
            s_dim = x_vi.type.tensor_type.shape.dim[1] if x_vi else None
            c = c_dim.dim_value if (c_dim and c_dim.HasField("dim_value")) else None
            s = s_dim.dim_value if (s_dim and s_dim.HasField("dim_value")) else None
            for out, dims in ((n.output[0], [c if c else "C", "seq"]),
                              (n.output[1], [c if c else "C", CONV_K - 1])):
                if out in have:
                    continue
                g.value_info.append(_vi(out, TensorProto.FLOAT, dims))
                added += 1
        elif n.op_type == "LinearAttention":
            # out [Hv, S, Dv] and state_out [Hv, Dk, Dv]
            for out, dims in ((n.output[0], [LIN_HEADS_V, "seq", LIN_DV]),
                              (n.output[1], [LIN_HEADS_V, LIN_DK, LIN_DV])):
                if out in have:
                    continue
                g.value_info.append(_vi(out, TensorProto.FLOAT, dims))
                added += 1
    if added:
        onnx.save(m, path)
    print(f"[onnx] annotated {added} custom-op output(s) with value_info")


annotate_custom_ops(EXPORT_PATH)
