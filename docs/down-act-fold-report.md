# fuse_swiglu_down_proj: 修复与性能报告

日期：2026-11-02。模型：Qwen3-VL decoder（28 层，K_out=2048, N_up=6144, N=2048，
fp16）。decode 测量 = `What is the capital of France?`，A B B A × 3 轮交错，
取每步最小值之和；token 逐个对照 REF `[6722,315,9625,374,12095,13,151645]`，
不对不出数。

## 三个被证实的结论

1. **老 kernel（matmul.comp 内联 down_act）数学正确但 M=1 占用率不足**：
   每块完整走 K_out 后单次 silu（"One silu per reassembled gate"），
   decode 72.30 ms/token vs 不折叠 58.67（1.23×）。
2. **K 轴切分 + 逐 slice silu 在数学上不成立**（第一版 3-pass 的错误）：
   `Σ_s silu(g_s)·u_s ≠ silu(Σg_s)·(Σu_s)`，交叉项 g_s·u_t (s≠t) 全部丢失。
   证据链：ks_n>1 用例坏 / ks_n=1 好 → 全范围硬编码后输出精确 8.000×期望
   （机制全对）→ scratch dump 证明表在 GPU 端逐字完好（交付无嫌疑）。
3. **修复**：deposit 存原始 g/u 偏部分（scratch 翻倍，M=15 时 7.05 MiB），
   down 汇总 Σg、Σu 后单次 silu。与不折叠路径数值等价。

## 测量（ARL，fp16）

| 阶段 | fused | unfused (NOSWIGLU) | 比值 |
|---|---|---|---|
| decode ms/token（修复后，down 未分块） | 87.79 | 69.41 | 1.26× |
| prefill min ms | 339.6 | 233.9 | 1.45× |

逐 op 分解（submitprof 逐调用）：融合 op ~2.5 ms/层/步 vs 不折叠
gate+up+down ~1.8 ms。**deposit（50 MB Wg/Wu）与不折叠 GEMV 对效率相同**
（~37 GB/s）；**down pass（25 MB Wd）~22 GB/s，是不折叠 down GEMV
（ksplit+WIDE，62 GB/s）的一半**——12 个 block + 每 lane 512 次串行 FMA 链，
是延迟不是带宽。

## 优化：down pass j 收缩子分块（ds）

down grid 加 z 轴（ds 个子块，走 uPC.tile）：dp 键控 `(row, gx*ds+z, c)`，
reduce 求和 `mind_k*ds` 个部分；空子块仍写零槽。单元测试 9/9 通过后重测。
**结论（同轮 layer%2 交错 A/B，两臂都是完整节点）：ds=1 最优**——
全节点中位数 ds=1: 1.536 ms vs ds=8: 1.654 ms（96 blocks 反而更慢）。
12 个 block 已能让足够的 Wd load 在飞；子分块只增加 h 组装 barrier 和
8× 的 dp 槽。老 ds=8 "单调变差"（224-250 ms/步）的测量是旧 16-row walk
的假象，但修复后的重试仍然输。

## 本轮修复（2026-11-02 下午）：shader 才是瓶颈

1. **教训：.vkopbin 在转换期嵌入 SPIR-V**。shader 改动必须
   `make shader_archive` **并重新转换模型**，否则嵌入的是旧 kernel。
   本轮早期所有"shader 实验无效"的结论都是这么产生的（host 侧 ds 实验
   是真的，shader 侧全是幻觉）。对照 CMakeLists.txt 的嵌入说明。
2. **WIDE deposit（M==1）**：把 ksplit WIDE GEMV 的配方搬进 deposit——
   lane 持有 8 个连续 up 列（octet），每 k 每 matrix 一次 uvec4 load，
   一次发 8 个 k 再消费。同轮配对 pass-split：**0.816 ms 中位（62 GB/s），
   原 scalar 形式 1.145（44 GB/s）**。gate：`M==1 && N_up%8==0 && ks_n<=8
   && 512%ks_n==0`，否则回退 scalar 多 slice 形式。
3. **da_down mt==1 快路径**：通用 16-row walk 用 `acc[DA_MT][4]` 数组
   需要常量 trip count，mt==1 时 15/16 的 FMA 乘的是死行。快路径保留
   相同 Wd load / 协同 / dp 槽，4 个累加器。ds 无关（只读自己组装的
   h 切片）。
4. **WIDE down walk（N%8==0，ds=1）**：256 个列 octet，每 octet 的
   lane 对拆低/高 4 列；两 lane 发同一 uvec4（合并成一次 16B 事务），
   每 lane 4 个自有列累加器（无需跨 lane 合并，dp 布局不变）；
   一次发 8 个 jj。同轮 layer%2 A/B（uPC.wide 临时开关）：
   **全节点中位 1.332 vs scalar-4 1.541 ms（−0.21 ms/层）**；凉机下
   1.164 ms/层。
5. **WIDE-2 重试（j 劈在 lane 轴、8 累加器、无重复 load，tile=2）**：
   与 tile=1 WIDE 打平（1.159 vs 1.164 中位）——walk 已到该形状的
   内存系统极限。未采纳；已从 shader 移除。

## 测量（ARL，fp16，修复后）

| 阶段 | fused | unfused (NOSWIGLU) | 比值 |
|---|---|---|---|
| decode ms/token（step minima 和，热机 FINAL1） | 93.79 | 84.03 | 1.12× |
| decode MatMul 聚合 min-of-min（热机） | 76.30 | 69.65 | 1.095× |
| decode ms/token（凉机 FINAL2） | 67.56 | 62.09 | 1.09× |
| decode MatMul 聚合 min-of-min（凉机） | 54.96 | 50.90 | **1.080×** |
| prefill min ms | 303.3 | 158.2 | 1.92× |

逐层分解（同轮配对，凉机）：fused 节点 ~1.16-1.33 ms/层 vs 不折叠
三件套 ~1.19-1.27。剩余差距全部在 down pass（~0.49 ms/51 GB/s vs
不折叠 GEMV 0.38 ms/66 GB/s）；deposit 已与 GEMV 同效。

## 状态

- [x] 单测 9/9（含真实形状 15/1 × 2048 × 6144 × 2048）
- [x] 数学修复 + 正确性 gate（REF 七 token，12/12 轮转全对）
- [x] decode prof 轮转测量（本报告数字）
- [x] ds 子分块：测过，ds=1 最优（已关闭）
- [x] WIDE deposit + WIDE down walk：已上线
- [ ] 折叠目标：≥ 不折叠（MatMul 聚合 1.080×，节点级接近持平；
      down pass 还差 ~0.11 ms/层，无已知的干净杠杆剩余）

## 微基准：fold vs trio vs 旧树（/tmp/bench2/da_bench.cpp，2026-11-02）

独立微基准（不进模型）：`da_bench` 用 4 输入 fold op 和 2 输入三件套
在同一张量/队列上交替计时（40 轮轮流先手，取 min），每个 op 打印
`route_info()`；m==1 时用三件套自己的 g/u 重建 h=silu(g)·u 做
like-for-like 正确性探针（gate/up 用不同零和模式，maxdiff 0.0156）。
同一二进制分别链新树（c2470508）与旧树（c2470508^，worktree 重编，
补了两个无关的编译修复：`(void)profile`、VMA `.deviceMemory`），
交替跑 3 轮取 min-of-min。形状 k_out=2048 / n_up=6144 / n=2048，
每臂 75.5 MB 权重。

| m | route(fold) | route(trio) | 旧 fold | 新 fold | fold 提速 | 新 fold/trio | 旧 fold/trio |
|---|---|---|---|---|---|---|---|
| 1 | down_act_gemv | ksplit_wide | 2.230 | 1.166 | **1.92×** | **0.91×** | 1.81× |
| 2 | down_act_gemv | ksplit_wide | 3.152 | 2.155 | 1.46× | 0.99× | 1.43× |
| 4 | down_act_gemv | tiled_ks | 5.535 | 2.778 | **1.99×** | **0.76×** | 1.45× |
| 8 | down_act_gemv | tiled_ks | 10.286 | 3.952 | **2.60×** | 1.04× | 2.61× |
| 15 | down_act_gemv | coopmat | 0.346† | 5.831 | (正确性修复) | **2.33×** | 0.14×† |

† 旧树 m=15 的 "fold" 0.35-0.46 ms（178-218 "GB/s"）是路由 bug 的
实锤：coopmat 抢到 folded 节点后把它当普通 down GEMM 跑
（route_info 显示 `k=2048`，Wg/Wu 被静默丢弃）——算得快但 logit
是错的。c2470508 的路由重排（kDownActFold 先于 coopmat）关掉了它，
代价是 m=15 时 fold 走 down_act_gemv 比 coopmat trio 慢 2.3×。

要点：

- **收益点 1（decode，m=1）**：fold 内核 2.230 → 1.166 ms
  （33.9 → 64.7 GB/s），来自 WIDE deposit（M==1 gate）+ mt==1
  fast path + WIDE down walk。微基准里 fold 首次追平并略胜 trio
  （0.91×）；模型内的轮转测量仍是 1.080×（模型里 trio 的
  实际竞品还包括 glue/缓存效应），两数不矛盾。
- **收益点 2（m=2-8）**：三 pass 重排本身（raw g/u 部分和 +
  单 silu + 更好的 Wd walk）在旧内核的 m>1 病态上赚得 1.5-2.6×；
  m=4 处 fold 反超 tiled_ks trio（0.76×）。
- **遗留（prefill 地雷）**：`down_act_applies` 没有 m 门，长 prompt
  prefill 的 28 个 down_proj 仍会走 down_act_gemv，m=15 时比
  coopmat trio 慢 2.3×，m 更大预计更差。短 prompt（m≤8）不受影响。
  干净的修法是 fold op 在大 m 时内部退化为 unfused 三发（它自己
  就有这些 kernel），或 converter 侧按声明形状决定是否折。

## Prefill 大 m 回退的归因（/tmp/bench2/da_msweep，2026-11-03）

问题:m≥12 起 fold 比 trio-coopmat 慢 2.3×。两个候选:并发(块数)不足,
还是 m 适配缺 shape 变换。三个实验定案:

1. **块数翻倍实验**:down grid = (da_gx=12, ceil(m/16), 1),m≤16 恒
   12 块。m=15→16 块数翻倍到 24,时间 6.01→6.30 ms(≈+4%,纯线性于
   行数);m=17(24 块,第二 tile mt=1 走 fast path)6.19 反而低于
   m=16。**块数翻倍不解锁任何东西 → 并发不是瓶颈**(12 块在 m=1
   已能流 60+ GB/s)。
2. **n 收缩差分**(固定 k_out=2048/n_up=6144,deposit 权重流不变,
   down 的 Wd 读 25.2→3.1 MB):m=15 时 5.917(n=2048)→5.544(n=256)
   ms,**只省 0.37 ms**,且 Δ/22MB ≈ 59 GB/s ≈ m=1 的 down 带宽。
   → down pass 在 m=15 仍满带宽(~60 GB/s,占 0.4ms);**~5.5ms
   (93%)在 deposit+reduce**。
3. **shader 结构**:标量 deposit(m>1 路径,m==1 走 WIDE)的 r(行)
   循环包在 k 循环外——每行重走整个 Wg/Wu K 切片,m=15 名义权重流
   50MB×15=750MB(L2 部分吸收后实测 ~5.5ms/138 GB/s 有效);而
   down walk 是每 jj 载入一次、16 行从 shared 摊("row axis rides
   free"),所以 down 不受害。

**结论:m>1 回退是 deposit 的循环形状(GEMV 逐行重读权重),不是
并发,也不是 down walk。**修法:deposit 按 k 外/r 内重排(每个 k 的
wg/wu 载入一次、对 mt 行 FMA),或把 WIDE deposit 推广到 m>1。
预期 m=15 ≈ deposit 0.85-1.3 + down 0.4 ≈ 1.3-1.7 ms,低于 trio
coopmat 的 2.5 ms——折叠在 prefill 也能赢。
