# 4-commit 性能回归与历史对比（HEAD `97ee63e`，Qwen3-VL 2B，AMD Radeon Pro 560X / MoltenVK）

日期：2026-11-03 ・ 机器：AMD Radeon Pro 560X（Vulkan 1.4.357 / MoltenVK）・ 构建：仓库默认 `-O0 -g`（未改）

---

## 1. 结论

1. **有性能回归，且是硬回归。** 同会话交替 A/B（相同 reps、相邻时间窗）：
   - 端到端 decode **min-of-3 平均 68.5 → 91.8 ms/token（+34.0%）**
   - 稳态中位（decode step ≥100）**60.5 → 84.2 ms/token（+39.2%）**
   - MatMul 分解中位 **40.2 → 66.2 ms（+65%）**，142 次节点调用/step 不变
2. **责任 commit = `598e445`**（`feat(matmul): reorder the SwiGLU fold's gate/up weights to lane-major`）。
   它把 Wg/Wu 的权重形状从 2-D `[2048,6144]` 改成 3-D `[2048,768,8]`，但 `optimizer.py::_quantize_down_act_weights`
   仍以 `a.ndim != 2` 提前 `continue`，导致 **28 个 fold 的三份权重（Wd/Wg/Wu）全部留在 fp16**：
   - 转换日志中 `Converted SwiGLU fold ... to int8` 由 **28 次 → 0 次**；
   - 模型体积 **1.727 GB → 2.782 GB**（int8 663.5 MB + fp16 1057 MB），每 token 多读约 1 GB 权重。
   MatMul 耗时增量的绝对值（+26 ms）≈ 整个 decode 增量（+23.3 ms），即回归几乎全部来自 MatMul。
3. **`4a99ed9` 不是性能回归的原因，但它单独破坏了正确性。** 隔离实验（旧权重 + 旧量化 + 仅替换为 `4a99ed9` 的 shader）：
   min-of-3 平均 **68.7 vs 66.9 ms/token（+2.7%，噪声内）**，稳态 55.8 vs 56.7（−1.6%），MatMul 40.5 vs 42.3 —— **性能持平**；
   但输出从第 0 个 token 起即为 `0`（错误）。见 §5。
4. **`e0366cf`、`97ee63e` 无性能影响**（dense decoder 不走 MoE 分页；后者只改编译告警与测试 fixture）。
5. **历史基线复测情况（对应要求 5）：** 本轮复测了旧代码（`84514d1`）基线，int8 稳态 **60.5 ms/token**、MatMul **40 ms**，
   与历史 int8 基线 58.8–61.9 ms/token 落在同一区间（−2.3% ~ +2.9%）；说明**没有 13% 量级的整机漂移**，
   上述 +34%/+39% 不能用环境噪声解释。整机确实偏慢的只是"min-of-3 平均"口径（含首轮时钟爬坡，旧臂 avg 68.5 比历史 61.9 高 10.7%），
   因此**判定以稳态与同轮配对为准**。
6. **HEAD 的正确性已破。** HEAD 的 int8 融合模型（以及 `fp16` 融合）输出全为 `198`（换行）/无意义 token；
   HEAD 的 **非融合 int8 模型正确**。根因叠加两条：
   - `598e445` 让 Wg/Wu 变成 3-D，而 `ops/Matmul.hpp:861-865` 仍用 `shape[rank-2]` 取 `K_out`：
     `[2048,768,8]` 取到 **768**（分组数）而不是 2048，fold 只算前 768 个 K；
   - `4a99ed9` 的 `matmul_down_act.comp` 新 deposit 路径在本模型几何下产生错误数值。
   这两条与"量化丢失"同源（都由 3-D lane-major 触发/伴随），因此 **`598e445` 是性能回归 + 正确性破坏的主责提交**，
   `4a99ed9` 是独立的正确性缺陷（性能中性）。

---

## 2. 改动点清单

commit 顺序：`84514d1`（改动前） → **`4a99ed9` → `e0366cf` → `598e445` → `97ee63e`（HEAD）**

| commit | 标题 | 关键改动 | 触及热点 | 是否可能影响本机 M=1 decode |
|---|---|---|---|---|
| `4a99ed9` | perf(swiglu): load each fold weight once per row-tile, not once per row | `shaders/buffer/matmul_down_act.comp` deposit 改为 k-outer/r-inner（85 行）；`ops/Matmul.hpp` deposit grid-z 镜像 shader wide gate（+14）；`tests/BufferRankTest.cpp` +85；新增 `docs/down-act-fold-report.md` | 是（fold deposit） | **输出被破坏**；性能持平（实测 C 组） |
| `e0366cf` | feat(model): page MoE slot bytes from the checkpoint | `model/load.cpp/.hpp`、`dag.py`、`weight_stream.py`、`converter.py`、flatbuffer schema、MoE 单测 | 否（dense 不分页） | 无影响 |
| `598e445` | feat(matmul): reorder the SwiGLU fold's gate/up weights to lane-major | `optimizer.py` 打标 + 3-D reshape；`weight_stream._write_lane_major`；`dag.py` 放宽 extent 校验 | **是（权重布局 + 量化路径）** | **是**：量化丢失（+1.05 GB fp16 权重）→ +34~39%；3-D 形状触发 `K_out` 误判（768）→ 输出错误 |
| `97ee63e` | fix(build): keep the release build clean and the shader-archive fixture honest | `vulkan/VulkanDevice.cpp` `(void)profile`；`test_embedded_shaders.py` fixture | 否 | 无影响 |

补充事实：
- 本模型 `N_up = 6144`，`N_up % 8 == 0`，`pad == N_up`，`598e445` 的 lane-major 重排本身是**恒等置换**
  （453 个 initializer payload 与旧模型逐字节相同）——坏点不在置换，而在**紧随其后的量化分支不再匹配 3-D 形状**。
- `598e445` 之前 Wg/Wu 也是 2-D，`shape[rank-2]` 恰好得到正确的 2048，所以 `Matmul.hpp` 的隐患是**被 `598e445` 首次暴露**。

---

## 3. 本轮实测数据

测量口径（所有臂一致）：prompt = `List the integers from 1 to 300, separated by commas.`
（prefill 16 token，207 个 decode step，`--max-kv 320`），环境
`VKOP_NO_COOP=1 VKOP_PROFILE=1 VKOP_RAW_PROMPT=1 VKOP_SUBMIT_PROF=1`，每臂 3 轮、重新起进程，
**同一 A/B 块内逐轮交替**，按 min-of-3 取最优。

### 3.1 臂说明

| 臂 | 树 | 模型 | 体积 | fold 量化 |
|---|---|---|---|---|
| `old` | `84514d1` | `int8_fused_old.vkopbin` | 1.727 GB | 28/28 int8 |
| `4a99` | `4a99ed9` | `int8_fused_4a99.vkopbin` | 1.727 GB | 28/28 int8 |
| `pre598` | `e0366cf` | `int8_fused_pre598.vkopbin` | 1.727 GB | 28/28 int8 |
| `new` | `97ee63e` | `int8_fused_new.vkopbin` | **2.782 GB** | **0/28（全 fp16）** |
| `nofuse` | `97ee63e` | `int8_nofuse_new.vkopbin` | 1.727 GB | 不融合 |
| `fp16` | `84514d1` | `fp16_fused_old.vkopbin` | 3.444 GB | 不量化 |

### 3.2 交替块与结果（解码步 ≥1 的逐 step 墙钟）

> 每列取该臂 3 轮中该指标的**最优值**（不保证同一轮），"各轮"列中加粗者为 avg 最优轮。

| 块 | 臂 | 各轮 avg ms/tok | **min-of-3 avg** | 稳态 med(step≥100) | p50 | MatMul 次数/step | MatMul 中位 ms | TOTAL 中位 ms |
|---|---|---|---|---|---|---|---|---|
| B | `old` r4/r5/r6 | 73.7 / **68.5** / 73.4 | **68.5** | 62.3 / **60.5** / 60.5 | 60.3 | 142 | **40.2** | 46.3 |
| B | `new` r4/r5/r6 | 151.9 / **91.8** / 135.8 | **91.8** | 101.9 / **84.2** / 128.9 | 83.9 | 142 | **66.2** | 70.8 |
| A | `old` r1/r2/r3 | 112.4 / 109.3 / **71.5** | 71.5 | 92.2 / 88.2 / **61.9** | 60.1 | 142 | 39.9 | 44.4 |
| A | `nofuse` r1/r2/r3 | 142.1 / **89.9** / 113.0 | 89.9 | 133.2 / **70.4** / 107.3 | 68.4 | **198** | 45.1 | 52.0 |
| C | `old` r7/r8/r9 | **66.9** / 88.5 / 79.1 | 66.9 | **56.7** / 62.7 / 61.6 | 56.8 | 142 | 42.3 | 47.5 |
| C | `4a99` r1/r2/r3 | 101.5 / 77.8 / **68.7** | 68.7 | 89.7 / 56.8 / **55.8** | 55.4 | 142 | 40.5 | 45.0 |
| — | `fp16` r1/r2/r3 | **106.7** / 121.6 / 121.1 | 106.7 | **97.2** / 118.0 / 125.8 | 100.1 | 142 | 67.7 | 74.8 |

同轮配对（相邻时间窗，剥离时钟漂移）：

| 对照 | avg Δ | 稳态 Δ | MatMul Δ | 判定 |
|---|---|---|---|---|
| `new` vs `old`（块 B） | **+34.0%** | **+39.2%** | **+64.7%** | 回归 |
| `4a99` vs `old`（块 C） | +2.7% | −1.6% | −4.3% | 噪声内（持平） |
| `nofuse` vs `old`（块 A） | +25.7% | +13.7% | +13.0% | 预期变慢（142 → 198 次 MatMul） |
| `fp16` vs `old` | +55.8% | +60.7% | +68.4% | fp16 路径本身更慢 |

> 噪声事实：同一臂不同轮可差 40%~100%（如 `new` r5=91.8 与 r4=151.9），首轮普遍受时钟/热爬坡影响；
> `old` r7（66.9）与 r1（112.4）相差 68%。因此**单轮数字不可用**，本报告只用 min-of-3 与同块配对比值。

### 3.3 融合 vs 非融合（本轮重测"18%"）

- 块 A 同轮：`old` 稳态 61.9 vs `nofuse` 70.4 → **融合快 12.1%**；按 avg 71.5 vs 89.9 → **快 20.5%**。
  历史"快 18%"落在本轮 12%~21% 区间内，**结论仍成立**。
- HEAD：`new`（融合）84.2 vs `nofuse`（非融合）70.4 → **融合反而慢 19.6%**，方向翻转，与 §1 的量化丢失一致。

---

## 4. 与历史基线对比

| 项 | 历史值 | 本轮实测 | delta | 判定 |
|---|---|---|---|---|
| int8 decode（旧代码 `84514d1`，稳态） | 58.8–61.9 ms/tok | **60.5 ms/tok**（MatMul 40.2 ms） | **−2.3% ~ +2.9%** | 基线复测通过（无 13% 漂移） |
| int8 decode（HEAD，稳态） | 58.8–61.9 ms/tok | **84.2 ms/tok**（MatMul 66.2 ms） | **+36.0% ~ +43.2%** | **回归** |
| int8 融合相对非融合 | 快 18% | 旧树 快 12–21%；HEAD **慢 19.6%** | — | 旧树保持；HEAD 翻转 |
| fp16 decode（融合，旧树） | ~87.5 ms/tok | **106.7 ms/tok**（稳态 97.2，MatMul 67.7 ms） | **+22.0%**（稳态 +11.1%） | 见局限：历史 fp16 口径为 254 MatMul op，本轮 142 op，不可严格对齐 |
| fp16 MatMul 聚合 | 254 op / 58.2 ms | 142 op / 67.7 ms | 次数不可比；单次 0.477 vs 0.229 ms（**×2.08**） | 仅作环境参考，不判回归 |

误差口径说明：`delta = (本轮 − 历史) / 历史`；历史 int8 给的是区间，故 delta 写成区间。
责任归因只指向"改动能在本模型 M=1 decode 生效"的提交；`4a99ed9` 的 m>1 deposit 改动与 `598e445` 的恒等置换**不解释**本机性能差异，
但 `598e445` 的**量化分支失配**解释了全部性能差异。

---

## 5. 正确性证据

### 5.1 单测（HEAD 构建）

```
./build/tests/all_tests --gtest_filter='BufferRankTest.*'   → 58/58 PASSED (12.7 s)
./build/tests/all_tests --gtest_filter='*MatMul*'           → 22/22 PASSED (15.9 s)
```

含 `BufferRankTest.MatMulDownAct*`（`4a99ed9` 新增 m=15/m=1 用例）全部通过 —— **单测未覆盖"转换期 3-D 形状 + 宿主取 K_out"这条链路**。

### 5.2 e2e 逐 token（同一 prompt，贪心）

| 臂 | 首个 token | 前 10 token | 结果 |
|---|---|---|---|
| `old`（int8 融合，正确参照） | 11 | `11,1477,279,2629,315,279,18509,315,1817,1372` | 连贯文本；4 轮（r6–r9）**逐 token 完全一致** |
| `fp16`（旧树 fp16） | 11 | `11,1477,279,2629,315,279,18509,315,279,5109` | 连贯；与 `old` 第 8 步分叉（1817 vs 279，贪心近平分叉） |
| `nofuse`（HEAD 非融合） | 11 | 与 `fp16` **前 18 token 完全一致**（第 18 步 11 vs 13） | **正确**；2 轮逐 token 一致 |
| `new`（HEAD int8 融合） | **198** | `198,198,198,…`（全换行） | **错误** |
| `4a99`（旧权重 + 新 shader） | **0** | `0,0,0,…` | **错误**；3 轮逐 token 一致 |
| `pre598`（`e0366cf`） | **0** | `0,0,0,…`（输出 `!!!`） | **错误** |

### 5.3 归因隔离（决定性证据）

- `int8_fused_old` / `int8_fused_pre598` / `int8_fused_4a99` 三个模型：
  **453/453 个 initializer payload 逐字节相同**（以 external blob base 为基准的 md5），
  **576/576 个 node 的 `OpType`/`Name`/`Plan` 完全一致**。
- `pre598`、`4a99` 与 `old` 的唯一差异 = **内嵌 SPIR-V archive**（`buffer_matmul_down_act_fp16` 旧 165,060 B → 新 162,896 B）。
  → `4a99ed9` 的 shader 在 M=1 int8 fold 上**数值错误**，与量化/布局改动无关。
- 四个模型逐函数（SPIR-V）opcode 计数一致，说明差异在操作数/寻址而非控制流规模；**具体哪条指令/索引错尚未定位**（局限）。
- HEAD 三份融合模型（int8 / fp16）均错误、非融合正确，且 `Matmul.hpp:861-865` 在 3-D 形状下取 `K_out=768`，
  与 `598e445` 引入的 3-D 布局一一对应。

---

## 6. 复现命令

### 6.1 构建（新树）

```bash
cd /Users/uqland/github/vkop
make -C build vkop            # Matmul.hpp / VulkanDevice.cpp → libvkop.a
make -C build shader_archive  # 4a99ed9 改了 matmul_down_act.comp → build/vkop_shaders.bin
make -C build llm_chat
make -C build all_tests
# 产物校验：无新增 .dSYM（仅历史 bench_matmul.dSYM）；llm_chat 27,714,664 B；all_tests 74,511,192 B
md5 -q build/vkop_shaders.bin   # 48c5c0d0…（改动前备份 /tmp/vkop_shaders_prev.bin md5 另计）
```

### 6.2 旧树（对照臂）

```bash
git worktree add /tmp/vkop_84514d1 84514d1
cmake -S /tmp/vkop_84514d1 -B /tmp/vkop_84514d1/build -DENABLE_LLM_CHAT=OFF -DENABLE_TESTS=OFF
make -C /tmp/vkop_84514d1/build shader_archive
git worktree add /tmp/vkop_e0366cf e0366cf   # 同理构建 archive
git worktree add /tmp/vkop_4a99ed9 4a99ed9   # 同理构建 archive
```

### 6.3 转换（注意必须关流式）

```bash
# 旧臂
cd /tmp/vkop_84514d1/model/pypi
VKOP_STREAM_WEIGHTS=0 VKOP_SWIGLU_DEBUG=1 python3 -m onnx2vkop.cli \
  -i /Users/uqland/github/vkop/llm/llm.onnx -o /tmp/reg/models/int8_fused_old.vkopbin -q int8 \
  --stream-weights off --master-archive /tmp/vkop_84514d1/build/vkop_shaders.bin \
  --target-backend buffer
# HEAD 臂（int8 融合 / int8 非融合）
cd /Users/uqland/github/vkop/model/pypi
python3 -m onnx2vkop.cli -i ../../llm/llm.onnx -o /tmp/reg/models/int8_fused_new.vkopbin -q int8 \
  --stream-weights off --master-archive ../../build/vkop_shaders.bin --target-backend buffer
python3 -m onnx2vkop.cli -i ../../llm/llm.onnx -o /tmp/reg/models/int8_nofuse_new.vkopbin -q int8 \
  --no-fusions=fuse_swiglu_down_proj --stream-weights off \
  --master-archive ../../build/vkop_shaders.bin --target-backend buffer
```
转换后核对：`grep -c "Converted SwiGLU fold" <log>`（旧树 28，HEAD 0）、`stat -f%z <model>`（1.727 GB vs 2.782 GB）。

### 6.4 基准（交替、3 轮、min-of-3）

```bash
cd /Users/uqland/github/vkop
PROMPT='List the integers from 1 to 300, separated by commas.'
for arm in old new; do                # 顺序：old,new,old,new,old,new
  printf '%s' "$PROMPT" | \
    VKOP_NO_COOP=1 VKOP_PROFILE=1 VKOP_RAW_PROMPT=1 VKOP_SUBMIT_PROF=1 \
    ./build/llm_chat /tmp/reg/models/int8_fused_${arm}.vkopbin \
    llm/exporter/embed_tokens.bin llm/tokenizer/qwen3_vl.bin 208 --max-kv 320 \
    > /tmp/reg/logs/long_${arm}_r<N>.log 2>&1
done
```

- 逐 step 行格式：`[rN] <ms>ms  past_len=… → <token_id>`
- MatMul 分解：`[submitprof] MatMul <calls> <ms> <ms/call>`；`[submitprof] TOTAL <calls> <ms>`
- 汇总脚本解析 `^\[r(\d+)\] (\d+\.?\d*)ms` 与 `^\[submitprof\] MatMul`（丢弃首行 prefill），
  取每臂 min-of-3 的 avg，并统计 step≥100 的稳态中位。

### 6.5 正确性

```bash
cd /Users/uqland/github/vkop
./build/tests/all_tests --gtest_filter='BufferRankTest.*'
./build/tests/all_tests --gtest_filter='*MatMul*'
printf 'What is the capital of France?' | VKOP_NO_COOP=1 VKOP_RAW_PROMPT=1 \
  ./build/llm_chat /tmp/reg/models/int8_fused_old.vkopbin llm/exporter/embed_tokens.bin \
  llm/tokenizer/qwen3_vl.bin 8 --max-kv 64
```

---

## 7. 局限与说明

1. **构建配置固定 `-O0 -g`**（`CMakeLists.txt:104` 硬编码，无 `CMAKE_BUILD_TYPE`）。为与历史数字可比**未改优化级别**；
   所有历史基线同样带 DWARF，因此绝对墙钟含调试构建开销，delta 仍可比。
2. **噪声极大**：同一臂相邻轮可差 40%~100%，与 GPU 时钟/热爬坡、后台负载、KV 增长有关。
   本报告不使用单轮值，只用 min-of-3 与同块同轮配对；判定阈值按 spec（>+5% 回归）。
3. **旧臂使用 HEAD 二进制**跑旧模型。依据：`4a99ed9` 的 host 改动只影响 fold deposit grid-z，
   本模型两版都派发 `z=1`；因此同一二进制下"只换模型（含 converter + 内嵌 shader）"正是要隔离的变量。
   未做整树旧 `libvkop` 重编（`-O0` 全量成本高且无信息增量）。
4. **fp16 历史值不可严格对齐**：历史 fp16 记录为 254 MatMul op / 58.2 ms，本轮旧树 fp16 为 142 op / 67.7 ms，
   说明历史 fp16 臂的模型变体（融合/非融合、op 计数口径）与本轮不同，**只能作量级参考**。
   本轮 fp16 臂因磁盘（`/tmp` 可用 11 GB）只在旧树转换并测量。
5. **单一 prompt、单一 max-kv**：prompt 与 207 step 的 KV 增长固定；历史数字未记录同口径的
   prompt/KV 长度，历史 int8 区间与本轮稳态吻合，但不能排除不同上下文长度带来的 ±3% 偏差。
6. **perf 数据来自"错误"的 HEAD 模型**（输出全 `198`）：工作量（MatMul 次数、权重读取）与正确模型一致，
   墙钟仍可反映真实开销；但它同时也**是回归与正确性双重证据**，不能代表"修好后的 HEAD 性能"。
   修复量化（恢复 28/28 int8）后预计回到 ~60 ms/token 量级，须重测确认。
7. **`4a99ed9` 的 shader 缺陷未定位到具体指令**：已确证"仅 shader 不同即输出错误"，且逐函数 opcode 计数一致；
   具体是寻址/索引还是常量错误，需要后续用 `spirv-dis` + 最小复现（m=1、int8、K=6144）继续收敛。
8. **pre598/4a99 的 `K_out=768` 误判**来自宿主 `ops/Matmul.hpp:861-865` 的既有假设，`598e445` 只是首次把它暴露；
   修复方案应在 `Matmul.hpp` 按 rank 取"矩阵维度"而不是 `rank-2`，或在转换期保留 2-D 视图。
