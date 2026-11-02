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
