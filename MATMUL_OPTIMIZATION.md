# MatMul 优化笔记:常见手段 × 本仓库实测

> 面向改这个 kernel 的人。汇总了 MatMul 的常见优化手段、本仓库(vkop)
> 各自做到什么程度、以及**哪些实测有效、哪些实测无效**。无效的部分尽量写清
> 为什么,免得重复踩。
>
> 代码位置:`ops/Matmul.hpp`(host 侧选路 + 派发)、`shaders/buffer/matmul*.comp`
> 与 `shaders/image/matmul*.comp`(kernel)。运行时只关心 buffer(SSBO)后端。

---

## 0. 基线与瓶颈定位

模型 Qwen2.5-0.5B,buffer 后端,Intel ARL,prefill q_len=32 batch=1
(`VKOP_SUBMIT_PROF=1 VKOP_OPPROF=1`)。

**GPU exec 总计 ≈ 238ms,其中 MatMul = 192ms(81%)。** 所以 prefill 的优化
基本等价于 MatMul 的优化。

按 op-type(权威,全部 829 个 op):

| op_type | calls | gpu_ms | 占比 |
|---|---|---|---|
| **MatMul** | 218 | **192.14** | **80.7%** |
| FusedElemwise | 26 | 20.90 | 8.8% |
| Expand | 49 | 9.26 | 3.9% |
| Add | 120 | 4.46 | 1.9% |
| Cast | 1 | 3.76 | 1.6% |
| Concat / Transpose / RMSNorm / Rotary / Reshape / Gather | 363 | 7.41 | ~3% |

MatMul 按类别(近似,单位 ms):down_proj 59.5、attn_bare(QKᵀ/AV)30.9、
gate_proj 28.4、lm_head 28.5、up_proj 21.6、k_proj 13.1、o_proj 10.8、
v_proj 9.0、q_proj 5.7。**FFN(down+gate+up)= 46%,lm_head 单条 = 12%。**

> 结论先行:**MatMul 已经到顶。** 路径选择没得挑(见 §2),tile 形状改不动
> (§4.3 已证伪),融合到顶(§4.6)。剩下的 81% 就是这个数。下面逐条给依据。

---

## 1. Kernel 路径阶梯(host 选路顺序)

`MatMulBuffer::execute` 按下面顺序**第一个命中即返回**,次序本身就是最重要的
优化决策:

```
attn_softmax (Plan B, 只给 attention QKᵀ)
  → coopmat_i8 (W8A8 SINT8, 默认关)
  → coopmat   (fp16 张量核;ARL/NVIDIA 两种 shader)
  → tiled     (共享内存分块)
  → ksplit    (split-K GEMV;含 WIDE 变体)
  → fused fp16 (偶数 N 单趟 / 奇数 N reduce+pack)
  → reduce    (朴素 fp32)
```

各档门槛(`ops/Matmul.hpp`):

| 路径 | 门槛 | 备注 |
|---|---|---|
| coopmat | fp16、非量化权重、`rep==1`、设备支持、`!epi_bias_`、`batch*m>8`(ARL)/`m>=8`(NVIDIA) | 最高优先级 |
| tiled | `k%16==0`、`m>=12`、`!epi_bias_` | 64×64 tile |
| ksplit | `batch*m<=8`、`k>=256`、`n%4==0`、`k%2==0` | decode GEMV 形状 |
| WIDE(fp16) | `n>=2048`(可调) | 8 列/lane |
| WIDE(byte) | `n>=896`(可调) | 16 列/lane |

**实测的路径分布**(prefill,全图 218 次 MatMul):attn_softmax 24 次、
**coopmat 193 次**、reduce 1 次(k=1 的退化 `/rotary_emb/MatMul`)、
tiled/ksplit 0 次。即 **没有一次 MatMul 被错误 gate 掉或走错路径**。

---

## 2. 常见优化手段 × 本仓库对照

| 手段 | vkop | 结论 |
|---|---|---|
| 共享内存分块 (tiling) | ✅ `tiled` 路径 | 有效,m≥12 时 1.2–2.0x;但 prefill 被 coopmat 盖过 |
| 张量核 / cooperative matrix MMA | ✅ `coopmat` | **最有效的一步**,prefill MatMul −44% |
| split-K(沿 K 并行) | ✅ `ksplit` + `WIDE` | decode 有效(2x);**prefill 灾难性变慢** |
| GEMV / skinny-M 特化 | ✅ ksplit 本身 | 就是为 M=1 设计的 |
| 向量化/合并访存 | ✅ half2 word、WIDE 用 `uvec4` | 有效(WIDE 是纯带宽/load-count win) |
| 线程粗化(一线程多列) | ✅ WIDE 8 列(int8 16 列)/lane | 有效 |
| weight-only 量化(int8/fp8/int4/uint4/nf4/nvfp4) | ✅ 全支持 | 省显存有效;**prefill 提速不划算**(见 §4.2) |
| 削减权重带宽 | ❌ 试过(BM=32)| **无效** —— 不是带宽受限(§4.3) |
| epilogue 融合(scale/bias/softmax) | ✅ Plan A/B | 有效,详见 §3 |
| layout 折叠(吃掉 Transpose) | ✅ `transB` 融合 | 有效,28 个 KV 实例 |
| 结构融合(SwiGLU Mul → down_proj) | ⏳ 未做 | 上限 ~4%,偏低不做(§4.6) |
| 按 shape autotune / 多版本 | ✅ 部分 | 路径阶梯 + per-N 阈值 + ksplit_q |
| double buffering / 软件流水 | ❌ 未做 | 见 §5 候选 |
| batching / grouped GEMM | n/a | LLM batch=1,不适用 |

---

## 3. 做过且实测有效

1. **fp16 cooperative-matrix GEMM**(`8371d4d`)。SSBO 读 fp16 A/B,subgroup
   MMA 累加 fp32,写 flat fp32 scratch,再由 pack pass 打包 half2。
   **prefill MatMul −44%**(见 `8371d4d`)。ARL 用 8×8×16(workgroup
   BM=16/BN=32),NVIDIA Ada 用 16×16×16(BM=32/BN=64)。
   *结构代价*:coopmat 走 scratch+pack 两个 dispatch,且 scratch 是 fp32,
   比直接 fp16 出多一趟读写。
2. **Transpose→MatMul B 折叠**(`93983a9`)。把 `Transpose(last-2-swap)` 折进
   MatMul 的 `transB`,K-cache 上 28 个实例,少一批 dispatch。
3. **split-K GEMV + WIDE**(decode)。`e89287e`(int8 平铺 lane + `ksplit_q`
   占用率旋钮,~1.2x);`8160afd`(fp16 WIDE,`n>=2048` 自动开,ARL decode
   M=1 转 ksplit+WIDE,**decode 2x**);`8c2abcd`(int8 WIDE 阈值降到 `n>=896`,
   MatMul −19%,down/q/o −16..−45%)。
4. **Attention epilogue 融合**。
   - Plan A(`epi_scale`/`epi_bias`):把 `MatMul * scale + bias` 折进 QKᵀ
     epilogue,少 28 个 FusedElemwise dispatch。
   - Plan B(`attn_softmax`):把尾随 Softmax 折进 QKᵀ(online softmax),
     少 56 个 Softmax dispatch。**注意:在静态-N prefill 路径上 Plan B 不只
     是省 dispatch,而是正确性必需**(关掉它 logits 全 0)。A/B 无法单独测收益。
5. **QDQ 权重折叠到 int8 MatMul**(`be93aa0`)。`DQ(u8 const)→MatMul` 折成
   逐列 int8 + `scale[N]`,省 494MB 显存。
6. **Reshape alias**。view 类 op 直接 alias 输入 VkBuffer,省一次 1:1 copy。

---

## 4. 做过但**无效 / 已否决**(重点,别重试)

### 4.1 coopmat tile 形状(BM=32 变体)—— 2026-10 证伪
**假说**:lm_head 是 M=32、N=151936、K=896。ARL coopmat 是 BM=16/BN=32,
M=32 被切成 **2 个 M 行**;全局加载 = `M*N*K*(1/BM + 1/BN)`,而 A(激活)工作集
只有 M·K = 57KB 常驻 L2,**只有 B(权重 272MB)真正走 DRAM**,量 = `M*N*K/BM`
→ BM=16 会把整个权重矩阵**流两遍(544MB)**,BM=32 流一遍(272MB)。按 26–32ms
算是 ~17GB/s,远低于 ARL 带宽,看着像有头寸。

**做法**:写了 `matmul_coop32.comp`(BM=32/BN=16)——纯算子对调,同 256 线程、
同 8 subgroup、同 512 输出面积、每 K-tile shared 流量也一样(768 fp16),
subgroup 网格 4(M)×2(N)。逐算子按 `16 < m <= 32` 选路。

**结果:没有可测差异。** 交替 A/B,10 对相邻运行,lm_head:`min` 26.71 vs
24.08ms,逐对胜负 4/10 vs 6/10;正确性完全一致(`max_abs_diff` 都是 0.1582)。

**结论:coopmat GEMM 不是权重 DRAM 带宽受限的**,是延迟/occupancy 受限的。
把 DRAM 权重流量减半,t 不变。**不要再动 BM/BN 这类 tile 形状**。
(代码已撤,备份 `/tmp/matmul_coop32.comp.saved`。)

### 4.2 量化在 prefill 不划算
- **quantized coopmat**(把 int8 权重解码进 fp16 tile):decode 慢 3x,撤销 `623a836`。
- **W8A8 SINT8 coopmat**(`83338b4`):MMA 不省带宽,m=1 浪费 16×32 tile,
  **decode 慢 ~2x**,prefill 也慢 ~1.4x → 默认 gate 关掉(`VKOP_COOP_I8=1` 才开)。
- **NF4**:在 ≤0.5B 上 codebook 与 Qwen2.5-0.5B 权重不匹配,比 int4 差;小模型用 int4。
> 一句话:量化省显存有效,省时间在 prefill 无效(compute 本来也不饱和,
> 省带宽更是无效,见 4.3)。

### 4.3 削减权重带宽 —— 见 4.1,同一件事,已证伪。

### 4.4 ksplit 在 prefill 是灾难
ksplit 是 GEMV 形状,`batch*m>8` 时被 gate 挡掉是有原因的:实测 prefill
**1510ms vs coopmat 253ms**。**不要把 ksplit 的 gate 放宽到 prefill。**

### 4.5 tiled 比 coopmat 慢
用 `VKOP_NO_COOP=1` 逼到 tiled,lm_head **慢 1.3–1.4x**(44–52ms vs 32–39ms)。
coopmat 是对的路径。

### 4.6 结构融合已到顶
上一轮盘点:剩下的折叠每个最多 ~4%(SwiGLU Mul 折进 down_proj ~10ms;
attention_bias Expand 折进 QKᵀ),Concat/Transpose/Reshape 合计只剩 2.5%,
在噪声底附近,风险不值。

### 4.7 其他已否决
- **fp16 WIDE "net-neutral"**:早期读数是被 coopmat 拦截导致的假象,现已纠正。
- **ksplit_q = 32**:相对 q=16 回退(别被非交替 sweep 的 host drift 骗)。
- **NVIDIA 8×8×16**:RTX 4000 Ada 不支持该组合,当前 shader 会静默算错 →
  必须走 16×16×16 变体 + 运行时选组合。
- **level 合并 / `VKOP_BATCH_LEVELS`**:非确定性崩溃,别重试(与 MatMul 相邻但同属提交层)。

---

## 5. 还没做、但值得想的候选

按性价比排序,均**未验证**:

1. **coopmat 的 scratch+pack 开销**:现在每次 coopmat 都要写 fp32 scratch、
   再一趟 pack 成 half2(偶数 N 其实可以在 kernel 内打包)。若把偶数 N 的
   pack 折回 kernel,能省掉每层一次 dispatch + 一趟 fp32 读写。**这是目前
   最具体、还没试过的结构优化。**
2. **软件流水 / double buffering**:K 循环里每个 K-tile 都有两次 `barrier()`,
   K=896 → 56 次迭代 = 112 次 barrier/ workgroup。若每个 barrier 之间做 2 个
   K-tile(2 个 MMA),barrier 数减半。指向 §4.1 结论里的"延迟受限"。
3. **Coalescing**:B 是 `[K,N]` row-major,每行只读 BN 宽的一段(现 BN=32 →
   64B,一条 cache line 128B)。BN 更大理论上更省 line,但 §4.1 说明带宽不是
   瓶颈,所以这条优先级低。

> 注意:§4.1 已经把"带宽"这条路堵死了,所以 §5.2/5.3 都是围绕**延迟/占用率**
> 而不是带宽 —— 方向要对。

---

## 6. 调参/诊断开关

| 变量 | 作用 |
|---|---|
| `VKOP_NO_COOP=1` | 关掉 coopmat,回落 tiled |
| `VKOP_COOP_I8=1` | 开 W8A8 SINT8 coopmat(默认关) |
| `VKOP_KSPLIT_Q=<n>` | 覆盖 ksplit 的列-quad 数(调优用) |
| `VKOP_KSPLIT_WIDE=0/1` | 关/强制 WIDE |
| `VKOP_KSPLIT_WIDE_N=<n>` | 设 fp16+byte 两个 WIDE 阈值 |
| `VKOP_SUBMIT_PROF=1` | 每个 op 的 GPU 时间戳查询(权威) |
| `VKOP_OPPROF=1` | record 侧 wall time |

---

## 7. 测量方法学(重要)

**这台机器的跑间噪声有 2–3x。**同一个配置,lm_head 在不同 run 之间会在
24–83ms 之间跳(iGPU 时钟/热漂移)。后果:

- 单次运行**毫无意义**;
- 只看 **min-of-N 也不可靠** —— 一个"幸运快跑"就能把某个 arm 的 min 拉低
  20ms 以上,给出假结论(§4.1 一开始就被坑过一次);
- 必须用**交替的成对比较**(A,B,A,B,...),看逐对胜负,而不是看聚合 min;
- **低于 ~20% 的效果在这台机器上量不出来**,除非把测量隔离到单独的
  benchmark 里跑很多次。

---

## 8. 一句话总结

prefill 的 81% 在 MatMul,而 MatMul 的**路径选择**(coopmat)、**tile 形状**、
**带宽削减**三条常规杠杆都已到顶或证伪。如果还要动 prefill,方向是
**降低 coopmat 的固定开销(pack pass)** 或 **K 循环的流水/占用率**,不是再调
shape 或带宽。
