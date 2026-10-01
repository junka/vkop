"""Qwen-Image-2.1 DiT 的 ONNX 友好实现（自定义 wrapper，绕开 diffusers 内部）。

和 llm/exporter/qwen3vl_export_onnx.py 里的 `Qwen3VLLMOnnx` 同一个套路：复用原
checkpoint 的权重，但自己写 forward，把所有不可 trace 的东西换成显式张量 I/O。

绕开的四处不可导出实现（都在 diffusers `transformer_qwenimage21.py`）：
  1. `QwenImage21KVCache` 是 python 对象（每层 .store/.get）→ past/present KV 改成
     显式输入输出张量。
  2. `QwenImage21FlexAttnProcessor` 依赖 flex_attention 的 `BlockMask` → 预计算的
     **加法** attention bias（mask 值用 fp16 min ≈ -65504），和 llm.onnx 的
     `attention_bias` 完全同构。官方自带的 `QwenImage21AttnProcessor` 证明这个
     block-causal 结构本来就能拆成稠密 attention，不需要 flex kernel。
  3. `QwenImage21Rope.forward` 用了 `.tolist()` / `list.index()` / python 切片赋值，
     而且返回的是 `torch.polar` 复数表（ONNX 无复数类型）
     → 位置索引在 host 端算好（`joint_rope_positions`），cos/sin 当输入喂进来；
     配对的实数改写见下面"rope 复用半分裂内核"那一条。
  4. `joint_hidden_states[:, image_pad_mask] = hidden_states` 是布尔 scatter 赋值
     → 纯 t2i 的图像 token 恒在联合序列末尾，直接拆成 prefix/target 两段输入，不拼。

**关键结构发现（决定分图方式）**：`causal_condition=True` 让 text/条件图 token 用 t=0
的调制参数，而 block-causal mask 保证 prefix 的 key 不依赖 target latent。两者相加 ⇒
**prefix 的 K/V 与去噪步无关，extract 图在一个 prompt 的 40 步里只需跑一次**（diffusers
每步都跑一遍 extract 是它的实现选择，不是数学要求）。于是：
  - `dit_prefill.onnx`：输入只有 prompt_embeds + 常量 rope，**没有 timestep**，每 prompt 1 次。
  - `dit_decode.onnx`：target latent + timestep + 冻结的 prefix KV，每步 1 次（40 次）。

对 diffusers 的三处等价改写（由 tests/compare_wrapper.py 的逐张量实测兜底）：
  - rope 复用现成的半分裂内核：官方 `apply_rotary_emb_qwen(..., use_real=False)` 按
    head_dim 的**相邻配对** (x[2i], x[2i+1]) 做复乘，而 vkop 的 `RotaryEmbedding`
    (`shaders/buffer/rotary.comp`) 和 onnx2vkop 的 7 节点 rotate_half 融合
    (`optimizer.py:fold_rotary_embedding`) 都按**分半配对** (x[i], x[i+half])。两者只差
    head_dim 轴上一个固定置换，而 `q·k` 对 q/k 的**同一**置换不变，所以
    `DiTWeights.fold_rope_permutation()` 在导出前把这个置换折进 `wq/wk/nq/nk` 的行序，
    运行时图里就只剩一个标准 RE 节点、不用写新 kernel、也不用 slice/stack 摊平。
    配套地 `joint_rope_positions` 返回**前后两半重复**的 (S, head_dim) 表
    （`cos[d] == cos[d+half]`），正是该内核 `cs_idx = (b*seq+s)*head_dim + d` 的寻址方式。
  - 共享 modulation：checkpoint 实测 `modulation.1.weight` 是 (4*inner_dim, inner_dim)
    = (16384, 4096)，即**一次** `Sequential(SiLU, Linear)` 的输出被 32 层共用，block 里
    `modulation.chunk(2,-1)` → `_modulate` 再 `chunk(2,-1)`，切出
    [scale1|gate1|scale2|gate2] 四段，**层号不进列偏移**。官方把 `[t, 0]` 两行拼在一起
    过这条链、再用 `target_token_mask` 逐 token 选行；因为这条链逐行独立，这里让两张图
    各算自己那一行（prefill 只算 t=0 行、decode 只算采样 t 行），`DiTCore._modulate`
    一次算出 4 个 (1,1,D) 张量、32 层共用。
  - block 末尾的 fp16 `clip(-65504, 65504)` 用 Max/Min 表达，避免依赖 ONNX `Clip`。

dtype 约定：归一化一律 fp32 域算，并按 diffusers 原样的时机 cast 回 fp16
（`RMSNorm`：variance 走 fp32、乘 rsqrt 隐式升 fp32、乘 weight 前 cast 回 weight.dtype；
`ZeroCenterRMSNorm`：全 fp32、一次 cast）。cast 点挪动会在 fp16 下放大偏差。
"""

import math

import torch
import torch.nn.functional as F
from torch import nn

IMG_TOKENS_PER_SLOT = 4  # 一个 VLM image slot 站 2x2 个 latent token
LATENT_CHANNELS = 64  # == vae.config.z_dim == transformer.config.in_channels
CTX_IN_DIM = 4096  # text encoder（Qwen3-VL-7B）的 hidden size
ROPE_THETA = 10000
ROPE_POS_TABLE = 8192  # 正位置表长度
ROPE_NEG_TABLE = 1024  # 负位置表长度
FP16_MIN = -65504.0
FP16_MAX = 65504.0
TIME_DIM = 256  # QwenImage21TemporalTimesteps(timestep_dim=256)
TIME_FACTOR = 1000.0
TIME_MAX_PERIOD = 10000


class DiTConfig:
    """transformer/config.json 的展开形式（默认值即该文件里的数字）。"""

    def __init__(self, num_layers=32, heads=32, head_dim=128, mlp_ratio=3,
                 axes_dims_rope=(16, 56, 56), eps=1e-6, context_in_dim=CTX_IN_DIM,
                 in_channels=LATENT_CHANNELS):
        self.num_layers = num_layers
        self.heads = heads
        self.head_dim = head_dim
        self.inner_dim = heads * head_dim
        self.mlp_ratio = mlp_ratio
        self.axes_dims_rope = tuple(axes_dims_rope)
        self.eps = eps
        self.context_in_dim = context_in_dim
        self.in_channels = in_channels


# --------------------------------------------------------------------------
# RoPE（host 端预计算，等价于 QwenImage21Rope.forward）
# --------------------------------------------------------------------------
def joint_rope_positions(txt_len: int, lh: int, lw: int, axes_dims, head_dim: int):
    """纯 t2i（无条件图）联合序列的 rope cos/sin，形状各 (txt_len + lh*lw, head_dim)。

    返回的是 **铺满 head_dim、前后两半重复** 的表（`cos[d] == cos[d + head_dim/2]`），
    这正是 vkop `RotaryEmbedding` kernel 与 onnx2vkop 的 rotate_half 融合模式要的形态，
    详见下面"rope 用半分裂内核"一节。

    轴布局与 diffusers 一致：`cat([freqs[0][frame], freqs[1][height], freqs[2][width]])`，
    axes_dims_rope=[16,56,56] 之和 = 128 = head_dim，所以每轴各 8/28/28 对、共 64 对，
    **不**按 head 重复。

    diffusers 查的是 `torch.polar(1, angle)` 复数表；这里直接算等价的实数角
    `angle = pos * theta^(-2i/dim)`，避开 ONNX/torch 都不友好的复数张量。负位置的表尾
    回绕（`ROPE_POS_TABLE + ROPE_NEG_TABLE + p`）展开后就是同一个负数，所以两者一致。
    位置推进（forward 677-710 行）：文本占 0..txt_len-1（三轴同值），图像块 frame 轴整块
    冻结在 txt_len，height/width 轴取以 0 为中心的网格 -(n-n//2)..n//2-1。
    """
    n_img = lh * lw
    s = txt_len + n_img

    frame = torch.tensor(list(range(txt_len)) + [txt_len] * n_img, dtype=torch.float32)
    h_grid = [h for h in range(-(lh - lh // 2), lh // 2) for _ in range(lw)]
    w_grid = [w for _ in range(lh) for w in range(-(lw - lw // 2), lw // 2)]
    height = frame.clone()
    width = frame.clone()
    height[txt_len:] = torch.tensor(h_grid, dtype=torch.float32)
    width[txt_len:] = torch.tensor(w_grid, dtype=torch.float32)

    def angles(pos: torch.Tensor, dim: int) -> torch.Tensor:
        inv = 1.0 / torch.pow(torch.tensor(float(ROPE_THETA)),
                              torch.arange(0, dim, 2, dtype=torch.float32).div(dim))
        return pos[:, None] * inv[None, :]

    halves = [dim // 2 for dim in axes_dims]
    ang = torch.cat([angles(frame, axes_dims[0])[:, :halves[0]],
                     angles(height, axes_dims[1])[:, :halves[1]],
                     angles(width, axes_dims[2])[:, :halves[2]]], dim=-1)
    assert ang.shape == (s, head_dim // 2), ang.shape
    full = torch.cat([ang, ang], dim=-1)  # (s, head_dim)：cos[d] == cos[d + half]
    return full.cos(), full.sin()


# --------------------------------------------------------------------------
# 归一化 / rotary 的纯实数等价实现
# --------------------------------------------------------------------------
def _rms_norm(x, weight, eps):
    """diffusers.models.normalization.RMSNorm（DiT 的 norm_q / norm_k）。

    `x.to(fp32)` 必须**提出来只写一次**并复用那个变量：写成 `x.to(f32).pow(2)...` 再
    `x * rsqrt(...)` 的话，第二次 `x`（fp16）乘 fp32 会隐式升精度，trace 出来就是**两个
    独立 Cast**，而 onnx2vkop 的 `match_rms_norm`（priority 82）认的是
    `Cast0 -> Pow -> ReduceMean -> Add -> Sqrt -> Div -> Mul(Cast0 的输出) -> Cast -> Mul(weight)`
    这条链、要求 `Mul1` 的一个输入直接就是 `Cast0` 的输出。llm.onnx 里 113 处能全部融掉，
    是因为 HF 的 RMSNorm 源码本来就是先 `hidden_states = hidden_states.to(torch.float32)`
    再复用的写法。数值上两种写法逐位相同（fp16→fp32 是精确扩张），所以这只是图形态的差别。
    """
    xf = x.to(torch.float32)
    variance = xf.pow(2).mean(-1, keepdim=True)
    xf = xf * torch.rsqrt(variance + eps)
    return xf.to(weight.dtype) * weight


def _zero_center_rms_norm(x, weight, eps):
    """QwenImage21ZeroCenterRMSNorm：checkpoint 存 scale-1，全程 fp32、末尾一次 cast。"""
    input_dtype = x.dtype
    x = x.float()
    rrms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return (x * rrms * (weight.float() + 1)).to(input_dtype)


def _rotary(x, cos, sin):
    """`apply_rotary_emb_qwen(x, (cos, sin), use_real=False)` 的半分裂等价形式。

    官方按**相邻配对** (x[2i], x[2i+1]) 做复乘，vkop 的 `RotaryEmbedding` 内核
    （`shaders/buffer/rotary.comp`）和 onnx2vkop 的 7 节点 rotate_half 融合模式按
    **前后两半** (x[i], x[i+half]) 配对。两者只差 head_dim 轴的一个固定置换：把每个
    head 的第 i 对元素放到 (i, i+half) 上，相邻配对就变成分半配对，而
    `q·k`、`softmax`、`·V` 对 q/k 的**同一**置换是不变的 —— 所以这个置换可以在导出时
    折进 `wq/wk/nq/nk` 的行序（`fold_rope_permutation`），运行时零代价复用现成内核。

    折进去之后，本函数收到的 x 已经是置换后的布局，于是可以直接写成内核/融合模式认的
    rotate_half 形态：
        rotate_half(x) = cat([-x[half:], x[:half]], -1)
        out = x * cos + rotate_half(x) * sin
    cos/sin 是 (S, head_dim) 的两半重复表，`[None, None]` 成 (1,1,S,head_dim) 后对
    (B,H,S,head_dim) 在 batch/head 两轴广播；内存里没有 head 轴，正好和内核的
    `cs_idx = (b*seq+s)*head_dim + d` 一致（onnx2vkop 的 `fold_unsqueeze_into_rotary`
    会把这两个 size-1 Unsqueeze 跳过）。
    官方在 fp32 域算这次复乘再 cast 回 fp16；内核同样是 load 成 float 计算、写回 packed
    half，所以这里的 fp16 表不会引入额外偏差。
    """
    half = x.shape[-1] // 2
    c, s = cos[None, None].to(x.dtype), sin[None, None].to(x.dtype)
    rot = torch.cat([-x[..., half:], x[..., :half]], dim=-1)
    return x * c + rot * s


def _clip_fp16(x):
    """block 末尾的 `clip(-65504, 65504)`，用 Min/Max 表达（不依赖 ONNX Clip）。"""
    return torch.max(torch.min(x, torch.tensor(FP16_MAX, dtype=x.dtype, device=x.device)),
                     torch.tensor(FP16_MIN, dtype=x.dtype, device=x.device))


def rope_fold_perm(head_dim: int) -> torch.Tensor:
    """`fold_rope_permutation` 折进 wq/wk/nq/nk 的那个 head_dim 置换：`new = old[..., perm]`。

    单独暴露出来，是为了让逐张量比对的测试能用**同一个** perm 去还原 K 的基，而不是
    自己再抄一遍索引规则（抄错了会把自己骗过去）。
    """
    return torch.cat([torch.arange(0, head_dim, 2), torch.arange(1, head_dim, 2)])


class DiTWeights(nn.Module):
    """全部 DiT 权重，参数名与 checkpoint 的 `model.safetensors.index.json` 逐一对应。

    checkpoint 里没有任何 bias（`QwenImage21Attention.use_bias = False`、
    `sample_proj_bias=False`、modulation/proj_out/img_in 都是 bias=False），这里也不建。
    """

    def __init__(self, cfg: DiTConfig, dtype: torch.dtype = torch.float32):
        # `dtype` 直接决定缓冲区 dtype，**不要**改用 `model.to(dtype)`：7.115B 参数的
        # fp32 占位缓冲是 28.5 GB，先建后转会瞬时踩出这个数（真机上还有 checkpoint
        # 的 14.2 GB 要往里写），36 GB 统一内存直接开始交换。空壳阶段就把 dtype 定下来，
        # 峰值只有目标精度的那一份。
        super().__init__()
        D = cfg.inner_dim
        self.cfg = cfg
        self.img_in_w = nn.Parameter(torch.empty(D, cfg.in_channels, dtype=dtype))
        self.txt_norm_w = nn.Parameter(torch.empty(cfg.context_in_dim, dtype=dtype))
        self.txt_in_w = nn.Parameter(torch.empty(D, cfg.context_in_dim, dtype=dtype))
        self.txt_out_w = nn.Parameter(torch.empty(D, D, dtype=dtype))
        self.tl1_w = nn.Parameter(torch.empty(D, TIME_DIM, dtype=dtype))
        self.tl2_w = nn.Parameter(torch.empty(D, D, dtype=dtype))
        self.mod_w = nn.Parameter(torch.empty(4 * D, D, dtype=dtype))
        self.out_lin_w = nn.Parameter(torch.empty(D, D, dtype=dtype))
        self.proj_out_w = nn.Parameter(torch.empty(cfg.in_channels, D, dtype=dtype))
        # 逐层：attn.to_{q,k,v}, attn.norm_{q,k}, attn.to_out.0, img_mlp.{gate_layer,proj,out}
        self.wq = nn.Parameter(torch.empty(cfg.num_layers, D, D, dtype=dtype))
        self.wk = nn.Parameter(torch.empty(cfg.num_layers, D, D, dtype=dtype))
        self.wv = nn.Parameter(torch.empty(cfg.num_layers, D, D, dtype=dtype))
        self.wo = nn.Parameter(torch.empty(cfg.num_layers, D, D, dtype=dtype))
        self.nq = nn.Parameter(torch.empty(cfg.num_layers, cfg.head_dim, dtype=dtype))
        self.nk = nn.Parameter(torch.empty(cfg.num_layers, cfg.head_dim, dtype=dtype))
        self.gate = nn.Parameter(torch.empty(cfg.num_layers, D * cfg.mlp_ratio, D, dtype=dtype))
        self.up = nn.Parameter(torch.empty(cfg.num_layers, D * cfg.mlp_ratio, D, dtype=dtype))
        self.down = nn.Parameter(torch.empty(cfg.num_layers, D, D * cfg.mlp_ratio, dtype=dtype))

    def prepare_for_export(self):
        """导出/推理前的两步准备，**顺序不能换**（fold 要动堆叠 Parameter 的 .data）：
        折 rope 置换 -> 拆逐层 list。幂等：已经是 list 状态就直接返回。

        收成一个方法是因为漏掉 fold 不会报错、只会让 rope 的配对方式和内核不一致
        （数值静默错误），这类步骤不该散在调用方各写一遍。
        """
        if torch.is_tensor(self.wq):
            self.fold_rope_permutation()
            self.swap_stack()

    def swap_stack(self):
        """导出时把逐层张量换成 python list，避免 trace 时对大 Parameter 做 tensor 索引
        （会整份进图）。load_state_dict 之后、导出之前调一次。

        `detach()` 是必须的：Parameter 的切片即使在全局 `set_grad_enabled(False)` 下仍然
        `requires_grad=True`（实测），legacy ONNX 导出器把它当常量塞进图里就会直接抛
        "Cannot insert a Tensor that requires grad as a constant"。
        """
        names = ["wq", "wk", "wv", "wo", "nq", "nk", "gate", "up", "down"]
        for n in names:
            t = getattr(self, n)
            self._parameters.pop(n, None)
            setattr(self, n, [t[i].detach() for i in range(self.cfg.num_layers)])

    def fold_rope_permutation(self):
        """把 `_rotary` 注释里那个 head_dim 置换折进 `wq/wk/nq/nk` 的行序。

        必须在 `swap_stack()` 之前调（要动堆叠 Parameter 的 `.data`），且只需一次。
        """
        cfg, hd = self.cfg, self.cfg.head_dim
        perm = rope_fold_perm(hd)  # 相邻配对 -> 分半配对
        for attr in ("wq", "wk"):
            w = getattr(self, attr)
            # (L, heads*hd, in) 的行轴前两级就是 (head, hd)，所以按第 2 轴取行即可。
            w4 = w.view(w.shape[0], cfg.heads, hd, -1)
            out = torch.empty_like(w)
            out.view(w.shape[0], cfg.heads, hd, -1)[:] = w4[:, :, perm, :]
            w.data = out
        for attr in ("nq", "nk"):
            t = getattr(self, attr)
            t.data = t.data.view(-1, cfg.heads, hd)[:, :, perm].reshape(t.shape)


class DiTCore(nn.Module):
    """共享的计算主体；子类只定义输入顺序和返回值。

    权重加载后必须调 `weights.swap_stack()`。
    """

    def __init__(self, cfg: DiTConfig, weights: DiTWeights, mode: str):
        super().__init__()
        assert mode in ("prefill", "decode")
        self.cfg = cfg
        self.w = weights
        self.mode = mode

    # ---- 时间调制 ----
    def _timestep_temb(self, timestep):
        """time_proj -> timestep_embedder，逐行独立，返回 (rows, D)。

        官方在 `causal_condition` 下把 `[t, 0]` 两行拼在一起过这条链，但链上每行都是
        独立映射，所以两张图各算自己那一行即可：prefill 传全零、decode 传采样 t。
        """
        w = self.w
        t = timestep.to(torch.float32).reshape(-1)
        half = TIME_DIM // 2
        freqs = torch.exp(-math.log(TIME_MAX_PERIOD)
                          * torch.arange(0, half, dtype=torch.float32, device=t.device) / half)
        args = (TIME_FACTOR * t)[:, None] * freqs[None]
        emb = torch.cat([args.cos(), args.sin()], dim=-1).to(w.tl1_w.dtype)
        return F.linear(F.silu(F.linear(emb, w.tl1_w)), w.tl2_w)  # (rows, D)

    def _modulate(self, temb):
        """共享调制链：`modulation = Sequential(SiLU, Linear(D -> 4D))`，逐层无参数。

        返回 4 个 (1, 1, D) 张量 = [scale1 | gate1 | scale2 | gate2]，官方在 block 里
        先 `modulation.chunk(2,-1)` 再各自 `chunk(2,-1)`，切出来的就是这四段；
        `unsqueeze(1)` 是 `_select_modulation_rows` 的 token 轴，32 层共用同一份。
        """
        D = self.cfg.inner_dim
        mod = F.linear(F.silu(temb.to(self.w.mod_w.dtype)), self.w.mod_w)
        scale1, gate1, scale2, gate2 = mod.split(D, dim=-1)
        return (t.unsqueeze(1) for t in (scale1, gate1, scale2, gate2))

    def _txt_in(self, prompt_embeds):
        w = self.w
        h = _zero_center_rms_norm(prompt_embeds, w.txt_norm_w, self.cfg.eps)
        h = F.gelu(F.linear(h, w.txt_in_w), approximate="tanh")
        return F.linear(h, w.txt_out_w)

    def _qkv(self, i, x, cos, sin):
        """返回 (q, k, v)，各 (B, heads, S, head_dim)。

        顺序是 `投影 -> unflatten -> norm -> transpose(1,2) -> rope`，**norm 要在转置之前**：
        onnx2vkop 的 `fold_transpose_into_rotary` 只认 `Transpose(perm=[0,2,1,3])` 直接喂
        `RotaryEmbedding`（RE 内核反正逐元素重写一遍，把转置折进去是零成本）；norm 夹在中
        间就折不动 —— tiny 图实测 norm 在后是 `Found 0 Transpose->RotaryEmbedding`，改成
        norm 在前之后 Transpose 紧贴 RE。转置放在 rope 之前也正是这个方向；放在 rope 之后
        就是一个独立的搬移节点了。
        norm 归约的是最后一轴（head_dim），`transpose(1,2)` 动的是 head 轴，两者可交换，
        所以这个重排逐位不变（tests/test_wrapper_vs_diffusers.py 前后数字完全相同）。
        v 没有 RE 可折，它的 Transpose 会留下一次实打实的搬移。
        """
        w, cfg = self.w, self.cfg
        heads_shape = (cfg.heads, cfg.head_dim)
        q = F.linear(x, w.wq[i]).unflatten(-1, heads_shape)
        k = F.linear(x, w.wk[i]).unflatten(-1, heads_shape)
        v = F.linear(x, w.wv[i]).unflatten(-1, heads_shape)
        q = _rms_norm(q, w.nq[i], cfg.eps).transpose(1, 2)
        k = _rms_norm(k, w.nk[i], cfg.eps).transpose(1, 2)
        return _rotary(q, cos, sin), _rotary(k, cos, sin), v.transpose(1, 2)

    def _mlp(self, i, x):
        w = self.w
        return F.linear(F.silu(F.linear(x, w.gate[i])) * F.linear(x, w.up[i]), w.down[i])

    def _attn(self, i, x, cos, sin, bias, past=None):
        """一层完整 attention：qkv 投影+norm+rope -> (拼上 prefix KV) -> attention -> 输出投影。

        边界和 diffusers 的 `QwenImage21Attention` 模块对齐（它的 `to_out[0]` 就在
        processor 里面），这样逐层 hook 比对是一一对应的、不会错位。
        顺带返回 rope 后的 (k, v)：prefill 拿它们写 KV cache，decode 传 `past` 进来、
        返回值里的是拼好的一段，不再使用（图里未消费的中间量会被转换器剪掉）。
        """
        q, k, v = self._qkv(i, x, cos, sin)
        if past is not None:
            k = torch.cat([past[:, 0], k], dim=2)
            v = torch.cat([past[:, 1], v], dim=2)
        return F.linear(self._attention(q, k, v, bias), self.w.wo[i]), k, v

    def _attention(self, q, k, v, bias):
        """稠密 attention + 可选加法 bias（等价 QwenImage21AttnProcessor 的单段路径）。

        官方用 `dispatch_attention_fn`（SDPA 后端），那里 softmax 在 fp32 域算；这里显式
        走 matmul->softmax(fp32)->cast->matmul，和 llm.onnx 的做法一致，避免 ONNX 导出把
        SDPA 摊平成不可控的子图。

        入参已是 (B, heads, S, head_dim)（head 转置在 `_qkv` 里就做好了）。两处矩阵乘
        都尽量把 Transpose 放在**第二个**操作数上：`k.transpose(-1,-2)` 是相邻两轴互换，
        onnx2vkop 的 `fuse_transpose_into_matmul` 能折成 transB 属性省掉一次 dispatch；
        `v` 直接就是 (B,H,S_k,D)，不需要转。反过来 `q` 和 softmax 结果走的是第一个操作数，
        折不了。
        """
        cfg = self.cfg
        out = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(cfg.head_dim)
        if bias is not None:
            out = out + bias
        out = torch.softmax(out.float(), dim=-1).to(q.dtype)
        # 回到 (B, S, heads*head_dim) 好接输出投影；这一次 Transpose 落在 MatMul 的
        # 第一个操作数上，现有 fold 帮不上，是每层一次的实打实搬移（转换期实测计数）。
        return torch.matmul(out, v).transpose(1, 2).flatten(-2)


class PrefillGraph(DiTCore):
    """一次算出全部 32 层的 prefix K/V。无 timestep（causal_condition 保证与步无关）。

    输入：prompt_embeds (1, S_p, 4096)、cos/sin (S_p, head_dim)、timestep_zero (1,)
          fp32 全零（显式传进来，免得图里多出 cat/切片的常量节点）、
          bias (1, 1, S_p, S_p) 或 None（单条 prompt 无 padding 时 block-causal 退化成
          纯因果三角，可以直接不传）
    输出：present_kv_0..N-1，各 (1, 2, H, S_p, head_dim)（dim1 的 2 是 k/v）
    """

    def __init__(self, cfg, weights):
        super().__init__(cfg, weights, "prefill")

    def forward(self, prompt_embeds, cos, sin, timestep_zero, bias=None):
        cfg = self.cfg
        D = cfg.inner_dim
        x = self._txt_in(prompt_embeds)  # (1, S_p, D)
        # causal_condition 下 text/条件图 token 一律用 t=0 那一行调制 —— 与去噪步无关。
        scale1, gate1, scale2, gate2 = self._modulate(self._timestep_temb(timestep_zero))
        presents = []
        for i in range(cfg.num_layers):
            h = F.layer_norm(x, (D,), eps=cfg.eps) * (1 + scale1)
            att, k, v = self._attn(i, h, cos, sin, bias)
            x = _clip_fp16(x + torch.tanh(gate1) * att)
            hn = F.layer_norm(x, (D,), eps=cfg.eps) * (1 + scale2)
            x = _clip_fp16(x + torch.tanh(gate2) * self._mlp(i, hn))
            presents.append(torch.stack([k, v], dim=1))
        return tuple(presents)


class DecodeGraph(DiTCore):
    """每个去噪步跑一次：target latent + timestep + 冻结 prefix KV → 去噪结果。

    target token 只用采样时刻那一行调制（`_select_modulation_rows` 里
    `target_token_mask=True` 走 `params[:-1]`），所以逐层的调制张量在本图内是常量。

    输入顺序：target_latents (1, S_t, 64)、timestep (1,)、cos/sin (S_t, head_dim)、
      past_kv_0..N-1 (1, 2, H, S_p, head_dim)、bias (1, 1, S_t, S_p+S_t) 或 None
    输出：sample (1, S_t, 64)
    """

    def __init__(self, cfg, weights):
        super().__init__(cfg, weights, "decode")

    def forward(self, target_latents, timestep, cos, sin, *rest):
        w, cfg = self.w, self.cfg
        D, n = cfg.inner_dim, cfg.num_layers
        past, bias = rest[:n], (rest[n] if len(rest) > n else None)
        x = F.linear(target_latents, w.img_in_w)  # (1, S_t, D)
        # target token 走 `params[:-1]` —— 采样时刻那一行，与 prefix 用的 t=0 行互不相干。
        temb = self._timestep_temb(timestep)
        scale1, gate1, scale2, gate2 = self._modulate(temb)
        for i in range(n):
            h = F.layer_norm(x, (D,), eps=cfg.eps) * (1 + scale1)
            att, _, _ = self._attn(i, h, cos, sin, bias, past=past[i])
            x = _clip_fp16(x + torch.tanh(gate1) * att)
            hn = F.layer_norm(x, (D,), eps=cfg.eps) * (1 + scale2)
            x = _clip_fp16(x + torch.tanh(gate2) * self._mlp(i, hn))
        # norm_out 是 AdaLayerNormContinuous：silu(temb) -> linear(D) -> 只有 scale。
        scale = F.linear(F.silu(temb.to(w.out_lin_w.dtype)), w.out_lin_w).unsqueeze(1)
        return F.linear(F.layer_norm(x, (D,), eps=cfg.eps) * (1 + scale), w.proj_out_w)
