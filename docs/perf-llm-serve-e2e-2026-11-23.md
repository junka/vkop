# llm_serve 端到端性能（b3 / RTX 4000 Ada，2026-11-23）

日期：2026-11-23 · 机器：b3（NVIDIA RTX 4000 Ada Generation，20.5 GiB，56 核）·
分支 `fix/vulkan-version-gating-and-replay-classification` HEAD `f0b6517` ·
制品：`llm_v2.vkopbin`（3,456,812,616 B，11-23 重转，内嵌 160 个 shader）

测量前必须重转模型：`matmul_pack.comp` 的 swiglu 模式是新加的，而 shader 内嵌在
`.vkopbin` 里。热替换（`tools/patch_model_shaders.py`）不够用——archive 从
14.7 MB 长到 16.7 MB，超过模型预留的 4 MiB slack，`--shrink-dead` 也还差 1.4 MB。

---

## 1. 结论

1. **prefill ≈ 900–1000 tok/s，decode 在 L≈200 时 ≈ 55–60 tok/s、L≈1000 时 ≈ 40 tok/s、
   L≈8000 时 ≈ 17 tok/s。** decode 随上下文线性变慢是 KV rotation 的既知属性
   （占一轮 48–71%，见 `kv-rotation-is-the-decode-bottleneck`），不是本次改动引入的。
2. **本轮三个 commit 在真实 serving 路径上确认生效**：
   - `tile128` 预占 coopmat：prefill 887 vs 682 tok/s（**+30%**，`VKOP_TILE128=0` 对照）
   - 全部 force-coop（旧行为）：prefill 556–580 tok/s、decode 16.7 tok/s；
     新路由 prefill 839–866、decode 37.9–41.0（**prefill +50%，decode +2.3x**）
   - SwiGLU fold 在 coopmat 上生效：profile 里 `/gate_proj_*/MatMul_swiglu` 是折叠后的
     单次 MatMul（M≈1000 时 3.26 ms/层，force-coop 时 7.4 ms/层）。
3. **每个 prefill chunk 有约 20% 白算**：`/lm_head/MatMul` 39 ms + `post_TopK` 27 ms，
   而除了最后一个 chunk，其余 chunk 的 logits 谁都不读。8k prompt = 28 chunk × 66 ms ≈
   1.85 s，占实测 8.1 s 的 23%。`vkop_logits_last_row` 折叠只挂在 **graph 输出**上，
   而 serving 的 chunk 循环不区分「中间 chunk」和「最后 chunk」，所以折叠只能省掉行数
   （q→1），省不掉这一次派发。**这是本轮最重要的可优化项，且不涉及 shader 改动。**

---

## 2. 端到端实测（HTTP，`/v1/messages`，stream，max_tokens=64）

服务端 `--max-kv 16384 --prefill-chunk 256 --max-output 256`，无 `VKOP_SUBMIT_PROF`
（profiler 会明显抬高 wall clock）。数字来自 `[req]` 行（服务端自报的 wall/ttft/tpot）。

| prompt (words) | in tokens | ttft | wall | tpot | 备注 |
|---|---|---|---|---|---|
| 200 | 208 | 0.18–0.65 s | 1.33–1.70 s | 17–19 ms/tok | 首轮冷启动 0.65 s，次轮 KV 复用 0.18 s |
| 1000 | 1008 | 1.08–1.15 s | 2.60–2.71 s | 25 ms/tok | reuse 203 行 |
| 4000 | 4008 | 3.35–3.50 s | 6.58–6.59 s | 50 ms/tok | reuse 1003 行，12 chunk |
| 8000 | 8008 | 7.43 s | 11.11 s | 58 ms/tok | reuse 1003 行，28 chunk |
| 11000（纯 prefill 测） | 11008 | — | — | — | prefill 11.83 s = 930 tok/s |

`tpot` 是「首 token 之后」的每 token 时间，所以它包含 KV 已经很长时的 attention 代价。
小 prompt 的 17 ms/tok 对得上纯 decode 的 13.8–14.0 ms GPU 时间加上 host/HTTP 开销。

## 3. A/B：新路由 vs 旧行为（force-coop）

同一进程参数，交错执行，`--prefill-chunk 256`，prompt=1000 words（1008 tokens）。

| arm | prefill tok/s | decode tok/s | 8k prompt prefill |
|---|---|---|---|
| 新（默认） | 839 / 842 / 866 | 37.9 / 41.0 / 39.7 | 936–1004 tok/s |
| `VKOP_FORCE_COOP=1` | 556 / 560 / 568 | 16.7 / 16.7 / 17.5 | 579–616 tok/s |
| `VKOP_NO_COOP=1` | 828 / 835 | 34.6 / 37.6 | — |
| `VKOP_TILE128=0` | 682 / 683 | — | — |

force-coop 的 decode 掉一半，是因为 decode 的 M=1 GEMV 形状被 `coopmat_gate` 的
`batch*m > 8` 之外的路径抢走时反而更慢——这条是之前会话已测过的既有结论。

## 4. 逐 op 分解（M≈1000 prefill，单 chunk，`VKOP_SUBMIT_PROF`）

| op | calls | gpu_ms | 备注 |
|---|---|---|---|
| MatMul | 170 | 289.02 | 88% 的时间 |
| TopK | 1 | 27.31 | **每 chunk 一次，只有最后有用** |
| Attention | 28 | 6.27 | |
| RMSNorm | 113 | 2.02 | |
| RotaryEmbedding | 56 | 1.38 | |
| Add | 59 | 1.03 | |
| Transpose | 58 | 0.91 | |
| TOTAL | 622 | 328.09 | wall 报 320 ms，一致 |

最慢的单个 op：`/lm_head/MatMul` 38.98 ms、`post_TopK` 27.31 ms、`/gate_proj_*/MatMul_swiglu`
3.26 ms × 28 层（64+ 个 gate/up，每层一个 swiglu fold）。

decode（L≈1000，25.12 ms/step GPU）：Attention 11.98 ms（28 次 × 0.428 ms，KV rotation）、
MatMul 11.34 ms、其余合计 1.8 ms。`/lm_head/MatMul` 1.78 ms 是最大单项。

## 5. chunk 大小的影响（prompt=4000 tokens，纯 prefill）

| chunk | prefill tok/s | prefill ms |
|---|---|---|
| 64 | 390 | 2584 |
| 128 | 609 | 1655 |
| 256 | 828–918 | 4368（prompt=4000, 16 chunk） |
| 512 | 1125 | 3562 |
| 1024 | 1236 | 3242 |

chunk 越大越快（每 chunk 固定开销被摊薄），但 `--prefill-chunk 1024` 在长 prompt 上会
撞 `q × covered-rows` 的 device-lost 墙（约 6k 行）。256 是能跑完 11k agent prompt 的值，
这也是 README 里已经写下的取舍。

## 6. 与我 11-16 记忆基线的差异（重要）

`b3-llm-serve-setup` 记的是「10968-token prefill 1.4 s（43 chunk @256）、decode 13.6 tok/s」。
本次实测 11008-token prefill **11.83 s**，差 8.5 倍。已排除的原因：

- **不是降频**：负载中实测 SM 1620 MHz（最大 3105 MHz）、功耗 60.7 W / 130 W 上限、
  温度 48 °C、无任何 throttle 事件（HW/SW thermal、power brake 全 Not Active）。
- **不是新路由变慢**：force-coop（旧行为）更慢（13.0 s），tiled 与新路由同档。
- **不是制品**：11-16 的旧 `llm.vkopbin` 配同一二进制是 346 tok/s（更慢，因为 shader 旧）。
- **不是 chunk 开销假象**：`--prefill-chunk 512/1024` 也只到 1125/1236 tok/s。

推论：**1.4 s 那条记录要么是别的口径（例如只记 `last_run_ms` 单 chunk，或当时 prompt
被大幅裁剪），要么当时的构建/制品组合与现在不可比。** 我倾向于前者——1.4 s / 43 chunk =
32 ms/chunk，而现在单 chunk 的 GPU 时间就有 320 ms，一个数量级的差。在这条被重新标定
之前，**不要拿 1.4 s 当基线**。

## 7. 已实现：跳过非末 chunk 的 lm_head + TopK

**实施于 `0faf03b`。** `Runtime::set_unused_outputs(names)` 声明「本轮调用方不读这些
输出」，`Run()` 据此把执行截断到一个 level 前缀。`llm_serve` 对每个非末 chunk 声明
`logits` / `post_TopK0` / `post_TopK1`。`VKOP_SKIP_DEAD_LOGITS=0` 是对照臂。

三条安全性前提（每条都是踩过的坑）：

- 存活性从图推导，不靠算子名：以未声明的输出为种子，沿 `node_dependency_indices_`
  反向遍历。
- 结果必须是**前缀**。调度器按 level 序号给 replay 段编号，中间开洞会让后续序号整体
  漂移 —— 那正是 Run() 里记录过的乱码 token 失效模式。
- 死节点若有副作用仍要跑。这里必须用 `did_host_work_last_round()` 而**不是**
  `has_host_side_effects()`：MatMul 无条件把后者覆写成 true，会让所有 MatMul 钉死在
  存活集里，优化静默失效（实测只跳 1 个节点而不是 9 个）。

顺带修了 profiler：`vkGetQueryPoolResults` 原来对全部 op 用 `WAIT_BIT`，被跳过的 op
其 query 永不写入 → 永久阻塞。现在只查已执行区间。

实测（b3 / RTX 4000 Ada，8k prompt，chunk 256，交错 A/B，3 对）：prefill
**7813 → 7467 ms（−4.4%）**，1029 → 1081 tok/s。单 chunk 的 GPU 时间 245 → 200 ms
（跳过确实砍掉 ~18% 的 chunk GPU 时间），但端到端只有 4.5% —— 剩下的时间是 chunk
之间的主机侧 record/upload，tail 本来就不占那部分。**这修正了本文 §1 里「约 20%」
的估算**：op 占 GPU 总量的百分比不等于占一轮 wall time 的百分比。

## 8. 其余待办

1. 重新标定 b3 prefill 基线（已完成，见 §6；数字在 `b3-llm-serve-setup` 记忆里）。
2. `--prefill-chunk` 自适应：短 prompt 用 512–1024，长 prompt 回落到 256。
3. chunk 之间的主机侧 record/upload 现在是 prefill 的主要非 GPU 时间，若要继续压
   prefill，杠杆在那边而不是 op 本身。
