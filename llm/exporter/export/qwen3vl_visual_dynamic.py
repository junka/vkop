"""Qwen3-VL 视觉塔的**动态 grid** 实现（可 ONNX trace，尺寸不再固化）。

HF 的 `Qwen3VLVisionModel` 从 `grid_thw` 用 `.tolist()` 派生出三样东西：

  1. `fast_pos_embed_interpolate`：pos_embed 的双线性插值索引 + 权重
     （逐图 python 循环 + `torch.linspace(0, N-1, h)`）；
  2. `rot_pos_emb`：每个 patch 的 (row, col) 坐标 → freq_table 查表；
  3. attention 的 `cu_seqlens` → `torch.split(lengths.tolist())` 逐图切分。

1 和 3 是数据相关的 python 控制流，trace 不了；2 的长度也是数据相关的。所以老版
导出（`EXPORT_IMG_SIZE`）在 trace 前 eager 算好这些值、当常量 kwargs 传给 HF 的
forward——图上只剩静态数据流，代价是**图的尺寸被永久固化**，换尺寸必须重跑导出，
且 runtime 侧必须按导出尺寸喂图。

本模块把这三件事重写成「只由 `grid_thw` 这个**图输入**驱动的张量运算」，于是同一
张图能吃任意尺寸（只要 grid 的每条边是偶数且 ≤ MAX_GRID）：

  * **pos_embed 插值**：linspace 的精确值查表（见下）；floor 索引同表另一路给；
    四个角点用一次 broadcast `Add` 生成 `(mh, mw, m, m)` 的索引，权重同理
    broadcast `Mul`——顺带省掉了 HF 的 `view/permute/flatten`（因为索引是直接
    按 merge 后的 token 顺序生成的）。四次 `F.embedding`（Gather）+ 加权求和。
  * **rotary**：`rot[t, j] = coord[t, axis(j)] * inv_freq[j]` 展开成两个互补半表
    （`inv_hi = [inv_freq, 0]`、`inv_lo = [0, inv_freq]`）的 broadcast 乘加。
    0 项精确相加，与 HF 的 `freq_table[pos_ids].flatten(1)` 逐位一致。
  * **attention**：`grid_t == 1` 时 HF 的 `torch.split` 只有一个 chunk，是恒等
    操作（但它仍会 emit 24×3 个 Split + 一堆从 cu_seqlens 推长度的 Shape/Div/
    Cast 节点）。单图直接对整条序列做一次 eager attention。

**为什么 linspace 必须查表**：`torch.linspace` 的 fp32 结果没有任何简单公式能逐位
复现——`i*step`、`i*(N-1)/(h-1)`、甚至 fp64 求值再转 fp32，都在约 5% 的元素上差
1 ULP（torch 自己的实现也不是正确舍入的）。而 ViT 在 fp16 下对 pos_embeds 的
1-ULP 扰动是**混沌放大**的：0.1% 的元素差 1 ULP，最终 `image_features` 就差
max 0.45 / mean 0.002（对比输出自身 |x| 均值 0.3）。所以把 `linspace(0, N-1, h)`
的精确值整表存下——floor 与小数部分各一路，按 `h/2` 行索引——图上只做精确的
整数地址计算，不做任何近似公式。

pos_embed 权重表补了 `N+1` 行零向量，这样 ceil 角点不需要任何 clamp：最大索引
是 `(N-1+1)*N + (N-1+1) = N*N+N`，补零后正好在界内；而越界的那些项必然被乘 0
（最后一个插值点的 dh/dw 精确为 0.0），乘积精确为 0，求和与 HF 逐位相同。

实测：38/38 种 grid（2×2 到 64×64，含非方形）与 HF eager **逐位一致**；同尺寸下
ONNX 节点数比老版静态图少 349 个（Split −73、Cast −83、Shape −49、Div −46 …），
增加的是 Range ×4、Sub ×2 与几十个小的 broadcast 索引节点。
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.qwen3_vl.modeling_qwen3_vl import apply_rotary_pos_emb_vision

# HF smart_resize 的 max_pixels = 4096×4096，边长 /patch_size(16) 后 grid 上界 256。
# 表按 MAX_GRID/2+1 行存放（行号 = grid 边长 / 2，行 0 不用）。
MAX_GRID = 256


class VisualDynamic(nn.Module):
    """HF visual tower 的动态 grid 等价实现。输入 pixel_values (S,row) +
    grid_thw (1,3) int64；输出 (pooler_output, *deepstack_features)。

    只在单图（grid_t == 1）下使用——这是 vkop 的驱动路径。
    """

    def __init__(self, visual):
        super().__init__()
        self.visual = visual
        self.num_grid_per_side = visual.num_grid_per_side
        self.merge = visual.spatial_merge_size
        N = self.num_grid_per_side

        # pos_embed 补 N+1 行零（见模块 docstring：越界项乘 0，与 HF 逐位相同）。
        w = visual.pos_embed.weight
        pad = torch.zeros(N * N + N + 1 - w.shape[0], w.shape[1], dtype=w.dtype)
        self.register_buffer("pos_embed_padded", torch.cat([w, pad]))

        # rotary 的两个互补半表。
        inv = visual.rotary_pos_emb.inv_freq.float()      # (rot_dim/2,)
        z = torch.zeros_like(inv)
        self.register_buffer("inv_hi", torch.cat([inv, z]))
        self.register_buffer("inv_lo", torch.cat([z, inv]))

        # linspace(0, N-1, h) 精确值表：row = h/2，col = 元素，1-D 展平以便用
        # 一个动态索引向量一次 Gather。
        dh = np.zeros((MAX_GRID // 2 + 1, MAX_GRID), dtype=np.float32)
        fl = np.zeros((MAX_GRID // 2 + 1, MAX_GRID), dtype=np.int64)
        for h in range(2, MAX_GRID + 1, 2):
            lin = torch.linspace(0, N - 1, h)
            f = lin.int()
            dh[h // 2, :h] = (lin - f.float()).numpy()
            fl[h // 2, :h] = f.numpy()
        self.register_buffer("lin_dh", torch.from_numpy(dh.reshape(-1)))
        self.register_buffer("lin_fl", torch.from_numpy(fl.reshape(-1)))

    def _axis(self, n):
        """linspace(0, N-1, n) 的 (floor int64 (n,), 小数部分 fp32 (n,))。"""
        i = torch.arange(0, n, dtype=torch.int64)
        idx = i + (n // 2) * MAX_GRID
        return self.lin_fl[idx], self.lin_dh[idx]

    def _pos_embeds(self, h, w):
        """(S, hidden) 的插值位置编码，token 顺序与 HF 的 merge 后顺序一致。"""
        m, N = self.merge, self.num_grid_per_side
        hf, dh = self._axis(h)
        wf, dw = self._axis(w)
        # (mh,1,m,1) + (1,mw,1,m) → (mh,mw,m,m)，即 (br,bc,ir,ic) 的 token 序。
        base = (hf * N).reshape(-1, 1, m, 1) + wf.reshape(1, -1, 1, m)
        idx = [base.reshape(-1), (base + 1).reshape(-1),
               (base + N).reshape(-1), (base + N + 1).reshape(-1)]
        dh4 = dh.reshape(-1, 1, m, 1)
        dw4 = dw.reshape(1, -1, 1, m)
        wts = [(1 - dh4) * (1 - dw4), (1 - dh4) * dw4,
               dh4 * (1 - dw4), dh4 * dw4]
        table = self.pos_embed_padded
        acc = None
        for i in range(4):
            pe = F.embedding(idx[i], table) * wts[i].reshape(-1).to(table.dtype).unsqueeze(-1)
            acc = pe if acc is None else acc + pe
        return acc

    def _rope(self, h, w):
        """(cos, sin)，各 (S, head_dim)，与 HF 的 rot_pos_emb + cat + cos/sin 一致。"""
        m = self.merge
        intra = torch.arange(0, m, dtype=torch.int64)
        row = (torch.arange(0, h // m, dtype=torch.int64)[:, None] * m + intra[None, :])
        col = (torch.arange(0, w // m, dtype=torch.int64)[:, None] * m + intra[None, :])
        row = row.reshape(-1).float()                      # 全长 row 索引 (h,)
        col = col.reshape(-1).float()
        r = row.reshape(-1, 1, m, 1, 1) * self.inv_hi      # (mh,1,m,1,rot/2)
        c = col.reshape(1, -1, 1, m, 1) * self.inv_lo      # (1,mw,1,m,rot/2)
        rot = (r + c).reshape(-1, self.inv_hi.shape[0])    # (S, rot/2)
        emb = torch.cat((rot, rot), dim=-1)                # HF 同款 (S, head_dim)
        return emb.cos(), emb.sin()

    @staticmethod
    def _attention(attn, x, cos, sin):
        """单图 eager attention（HF Qwen3VLVisionAttention.forward 的
        `lengths.tolist()` 切分在单图下是恒等，这里直接省掉）。"""
        S = x.shape[0]
        nh, hd = attn.num_heads, attn.head_dim
        q, k, v = attn.qkv(x).reshape(S, 3, nh, hd).permute(1, 0, 2, 3).unbind(0)
        q, k = apply_rotary_pos_emb_vision(q, k, cos, sin)
        q = q.transpose(0, 1).unsqueeze(0)
        k = k.transpose(0, 1).unsqueeze(0)
        v = v.transpose(0, 1).unsqueeze(0)
        w = torch.matmul(q, k.transpose(2, 3)) * attn.scaling
        w = F.softmax(w, dim=-1, dtype=torch.float32).to(q.dtype)
        o = torch.matmul(w, v).transpose(1, 2).reshape(S, -1)
        return attn.proj(o)

    def forward(self, pixel_values, grid_thw):
        v = self.visual
        h, w = grid_thw[0, 1], grid_thw[0, 2]
        hidden = v.patch_embed(pixel_values)
        hidden = hidden + self._pos_embeds(h, w)
        cos, sin = self._rope(h, w)
        deepstack = []
        for layer, block in enumerate(v.blocks):
            hidden = hidden + self._attention(block.attn, block.norm1(hidden), cos, sin)
            hidden = hidden + block.mlp(block.norm2(hidden))
            if layer in v.deepstack_visual_indexes:
                merger = v.deepstack_merger_list[v.deepstack_visual_indexes.index(layer)]
                deepstack.append(merger(hidden))
        return v.merger(hidden), *deepstack
