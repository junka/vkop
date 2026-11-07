# 4-commit 合并后的性能回归与历史对比（Qwen3-VL 2B，AMD RX 560X / MoltenVK）

状态：待批准（Spec Mode）。批准前不执行任何构建、转换或基准。

## 1. 目标

对 HEAD `97ee63e`（其下 4 个新 commit：`4a99ed9`、`e0366cf`、`598e445`、`97ee63e`）做
性能回归测试，沿用与历史基线一致的口径：

- 端到端 decode 墙钟 ms/token（含 `VKOP_NO_COOP=1`）；
- MatMul 分解耗时与 dispatch 次数（`VKOP_SUBMIT_PROF=1`）；
- reps ≥ 200、同一 reps 做 A/B、min-of-3 取最优；
- 与历史基线对比并给出 delta 百分比、回归/提升判定与责任 commit；
- 正确性不破：MatMul 单测 + 逐 token e2e 一致性（REF 七 token）。

产出：中文 Markdown 报告 `docs/perf-regression-97ee63e.md`，并在对话中给出结论摘要。

非目标：不改任何 shader / host 代码；不重构；不新增可选构建配置。

## 2. Commit 改动清单（已核实）

| commit | 改动 | 触及热点？ | 预期对本机 M=1 decode |
|---|---|---|---|
| `4a99ed9` perf(swiglu) | `shaders/buffer/matmul_down_act.comp` deposit 由 r-outer/k-inner 改为 k-outer/r-inner（`mt==1` 保留旧直接走法）；`ops/Matmul.hpp` deposit grid-z 镜像 shader 的 wide gate；`tests/BufferRankTest.cpp` 加用例；新增 `docs/down-act-fold-report.md` | 是（仅 m>1 标量 deposit） | 持平（M=1 走 wide 分支，且 grid-z 仍为 1） |
| `598e445` feat(matmul) | Wg/Wu 转换期重排为 `[K][ceil(N_up/8)][8]`（`optimizer.py` 打标、`weight_stream._write_lane_major` 流式重排、`dag.py` 放宽 deferred extent 校验） | 否（N_up%8==0 时 `pad==N_up`，恒等置换） | 持平 |
| `e0366cf` feat(model) | MoE slot 字节按 checkpoint 分页（`model/load.cpp`、`dag.py`、`weight_stream.py`、`converter.py`、flatbuffer schema、新单测） | 否（dense decoder 不分页） | 无影响 |
| `97ee63e` fix(build) | `vulkan/VulkanDevice.cpp` 加 `(void)profile` 消 Werror；`test_embedded_shaders.py` 对齐 archive fixture | 否 | 无影响 |

关键前提需在转换日志中确认：本模型 `N_up % 8 == 0`（`VKOP_SWIGLU_DEBUG=1` 的
`matches=` 与 Wg/Wu dims），若 `N_up % 8 != 0`，`598e445` 会真实改变权重量，
必须把归因扩展到该 commit。

## 3. 构建方案

- `make -C build vkop`（增量重编 `libvkop.a`；`Matmul.hpp` 与 `VulkanDevice.cpp` 变更）
- `make -C build shader_archive`（`4a99ed9` 改了 `matmul_down_act.comp`，必须重编
  `buffer_matmul_down_act_fp16` 并重打 `build/vkop_shaders.bin`）
- `make -C build llm_chat`
- `make -C build tests/all_tests`（或 `make -C build all_tests`）
- 编译配置保持仓库现状：`CMakeLists.txt:104` 硬编码 `-O0 -g -Wall -Wextra -Werror
  -fno-omit-frame-pointer`，无 `CMAKE_BUILD_TYPE`。**不改优化级别**（历史基线同配置，
  改了不可比）。"无无关产物"检查改为：
  - 不产生新的 `.dSYM` 目录（对比 `ls build/*.dSYM`）；
  - `build/vkop_shaders.bin` 时间戳/CRC 更新，`libvkop.a`、`llm_chat` 重链；
  - 产物清单与改动前一致（无新增可执行文件）。
  - 用 `size -m` / `nm` 抽查无新增调试段属预期外产物（`-g` 本身是仓库标准，报告注明
    局限：本仓库所有历史数字都带 DWARF）。

## 4. 模型转换（必须重转，旧 vkopbin 内嵌旧 SPIR-V）

统一用 `onnx2vkop`，源模型 `llm/llm.onnx`（外部权重 `llm/llm.weights.bin`）。

新树（HEAD）：
```bash
mkdir -p /tmp/reg
VKOP_SWIGLU_DEBUG=1 python3 -m onnx2vkop.cli -i llm/llm.onnx \
  -o /tmp/reg/int8_fused_new.vkopbin -q int8 \
  --master-archive build/vkop_shaders.bin --target-backend buffer
python3 -m onnx2vkop.cli -i llm/llm.onnx \
  -o /tmp/reg/int8_nofuse_new.vkopbin -q int8 --no-fusions=fuse_swiglu_down_proj \
  --master-archive build/vkop_shaders.bin --target-backend buffer
python3 -m onnx2vkop.cli -i llm/llm.onnx \
  -o /tmp/reg/fp16_fused_new.vkopbin \
  --master-archive build/vkop_shaders.bin --target-backend buffer
```

旧臂（`84514d1`，改动前）：
```bash
git worktree add /tmp/vkop_84514d1 84514d1
cmake -S /tmp/vkop_84514d1 -B /tmp/vkop_84514d1/build \
  -DENABLE_LLM_CHAT=OFF -DENABLE_TESTS=OFF
make -C /tmp/vkop_84514d1/build shader_archive
cd /tmp/vkop_84514d1 && python3 -m onnx2vkop.cli -i /Users/uqland/github/vkop/llm/llm.onnx \
  -o /tmp/reg/int8_fused_old.vkopbin -q int8 \
  --master-archive /tmp/vkop_84514d1/build/vkop_shaders.bin --target-backend buffer
```
若 worktree 的 shader archive 构建失败，回退用已备份的 `/tmp/vkop_shaders_prev.bin`。

旧臂运行二进制：使用新 `build/llm_chat`。理由（写入报告）：`4a99ed9` 的 host 改动
只影响 deposit grid-z，而本模型 M=1 满足 wide gate（`n_up%8==0`、`ks_n<=8`、
`512%ks_n==0`），两版都派发 `z=1`；因此用同一二进制跑旧模型等价于只换 shader/转换器，
正是要隔离的变量。不做整树旧 libvkop 重编（`-O0` 全量重建成本高且无信息增量）。

磁盘预算（`/tmp` 24 GB 可用）：新 int8 融合 1.7G + 新 int8 非融合 1.7G + 新 fp16 融合
3.4G + 旧 int8 融合 1.7G ≈ 8.5G；旧 fp16 不转。转换后立即记录体积与
`VKOP_SWIGLU_DEBUG matches=`（预期 28）。

## 5. 测量协议（所有臂统一）

环境：`VKOP_NO_COOP=1 VKOP_PROFILE=1 VKOP_RAW_PROMPT=1 VKOP_SUBMIT_PROF=1`；
`VKOP_LOAD_MMAP` 视内存压力固定为同一取值；固定 `--max-kv 320`。

命令模板：
```bash
printf 'What is the capital of France?' | \
  VKOP_NO_COOP=1 VKOP_PROFILE=1 VKOP_RAW_PROMPT=1 VKOP_SUBMIT_PROF=1 \
  ./build/llm_chat <model> llm/exporter/embed_tokens.bin llm/tokenizer/qwen3_vl.bin \
  208 --max-kv 320 2>&1 | tee /tmp/reg/logs/<arm>_<round>.log
```
- reps：`max_new=208` → 约 200 个 decode step（首 token 来自 prefill，报告用
  `[profile] decode: N tokens ... avg X ms/token` 的 N 与 avg）。
- 每臂 3 轮（`round1..3`），每轮重新进程（清时钟爬坡）；取 3 轮 **最小** avg ms/token。
- 同 reps 做 A/B：`new` 与 `old` 交替执行（old,new,old,new,old,new），或至少确保两臂
  在同一时段内交替，报告记录轮转顺序。
- MatMul 分解：从 `VKOP_SUBMIT_PROF=1` 日志中汇总 MatMul 的 GPU 提交时间与调用次数
  （保留原始日志；若输出格式为逐 op-type 行，用 awk 汇总并贴原始片段进报告）。
- 丢弃首轮作为 warmup 仅当 3 轮间相对差 > 5% 时另行说明；默认三轮全用取 min。

顺序：
1. 新 int8 融合（3 轮）
2. 旧 int8 融合（3 轮，交替）
3. 新 int8 非融合（3 轮）
4. 新 fp16 融合（3 轮）

## 6. 对比与判定

历史基线（同一台 560X，历史会话记录）：

| 项 | 历史值 |
|---|---|
| fp16 decode 稳态 | ~87.5 ms/token |
| fp16 MatMul 聚合 | 254 op / 58.2 ms |
| int8 decode（非融合） | 58.8–61.9 ms/token |
| int8 融合相对非融合 | 快 18% |

delta 计算：`(本轮值 - 历史值) / 历史值 × 100%`；同会话 A/B 另算
`(new - old) / old × 100%`，后者用于剥离环境噪声（此前整机曾偏 ~13%）。

判定规则（写入报告）：
- 同会话 A/B |delta| ≤ 3%：持平/噪声内；
- > +5%：回归；< −5%：提升；
- 3%–5% 之间：标注"轻微变化，需更多样本"，不下结论；
- 责任 commit 归因只允许指向"其改动能在本模型 M=1 路径上生效"的提交；`4a99ed9`
  的 m>1 改动与 `598e445` 的恒等置换必须明确写成"不解释本机 decode 差异"。

补充对照：
- int8 融合 vs 非融合在本轮同会话重测，验证"18%"是否仍成立（若明显偏离，说明
  fold 行为或路由变化，需要单独指出）。
- fp16 新融合的 MatMul 次数会低于历史 254（fold 减少了 dispatch），所以 fp16 只能
  比墙钟与同会话可控项，MatMul 次数差异要在报告解释，不当作回归。

## 7. 正确性验证

1. `build/tests/all_tests`（全量或至少 `--gtest_filter=BufferRankTest.*`），要求全过；
   重点 `BufferRankTest.MatMulDownAct*`（含 `4a99ed9` 新增的 m>1 tile 用例）。
2. e2e 逐 token：`int8_fused_new` 与 `fp16_fused_new` 贪心生成前 7 个 token 必须等于
   REF `[6722,315,9625,374,12095,13,151645]`；`int8_nofuse_new` 至少逐 token 与
   `int8_fused_new` 一致到允许的近平分叉（若分叉，记录位置并用单测兜底）。
3. 旧臂 `int8_fused_old` 同样跑 7 token，作为"改动前行为"的参照。

## 8. 交付物

`docs/perf-regression-97ee63e.md`，结构：
1. 结论（有无回归、幅度、责任 commit、是否复测基线）
2. 改动点清单（表格，含"是否可能影响 M=1 decode"）
3. 本轮实测数据表（每臂 3 轮、min、MatMul 次数/耗时）
4. 与历史基线对比表（含 delta % 与判定）
5. 正确性证据（单测、7 token）
6. 复现命令（构建、转换、基准、单测）
7. 局限（`-O0 -g` 固定构建、噪声、旧臂用新二进制、仅 1 条 prompt、仅 decode 稳态）

## 9. 风险与回退

- 转换很慢/磁盘不足：先跑 int8 两臂（结论主路径），fp16 视磁盘与时间追加；
- 旧臂 archive 构建失败：用 `/tmp/vkop_shaders_prev.bin`，并在报告注明该 archive
  的时间戳早于 `4a99ed9`（11/2 22:53）；
- 若同会话 A/B 显示 > 5% 回归但旧臂也偏离历史基线同幅度，则判定为环境漂移而非
  代码回归，并给出该结论的证据。
