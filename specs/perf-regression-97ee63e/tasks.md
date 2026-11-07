# 任务清单：4-commit 性能回归（HEAD 97ee63e）

约定：所有命令在 `/Users/uqland/github/vkop` 下执行；日志落 `/tmp/reg/logs/`。
批准后按顺序执行，完成一项勾一项。

## T1 记录快照与改动点

- [x] `git log -5 --format="%h %ad %s"`、`git status --short`
- [x] 保存 `git show --stat` 四 commit 输出到 `/tmp/reg/logs/commits.txt`
- [x] 记录改动前产物：`ls -l build/libvkop.a build/llm_chat build/vkop_shaders.bin build/*.dSYM`
- [x] 备份旧 archive 已存在：`/tmp/vkop_shaders_prev.bin`（11/2 22:53）

## T2 重编译（新树）

- [x] `make -C build vkop`
- [x] `make -C build shader_archive`
- [x] `make -C build llm_chat`
- [x] `make -C build all_tests`（或 `tests/all_tests`）
- [x] 校验：`build/vkop_shaders.bin` 与 `libvkop.a`/`llm_chat` 时间戳更新；无新增
      `build/*.dSYM` 与新增可执行文件；`nm -a build/llm_chat | grep -c debug_info`
      与改动前一致（`-g` 为仓库标准，记录为局限）

## T3 转换新模型

- [x] `mkdir -p /tmp/reg/logs`
- [x] int8 融合（`VKOP_SWIGLU_DEBUG=1`，记录 `matches=`）
      `python3 -m onnx2vkop.cli -i llm/llm.onnx -o /tmp/reg/int8_fused_new.vkopbin -q int8 --master-archive build/vkop_shaders.bin --target-backend buffer`
- [x] int8 非融合
      `... -o /tmp/reg/int8_nofuse_new.vkopbin -q int8 --no-fusions=fuse_swiglu_down_proj ...`
- [x] fp16 融合（HEAD 侧转换验证；输出错误且占用 3.4 GB，已删除、改在旧树测量）
      `... -o /tmp/reg/fp16_fused_new.vkopbin ...`
- [x] 记录体积与 `matches`；确认 `N_up % 8`（`N_up=6144`，`pad==N_up`，`598e445` 为恒等置换）

## T4 旧臂（84514d1）

- [x] `git worktree add /tmp/vkop_84514d1 84514d1`
- [x] `cmake -S /tmp/vkop_84514d1 -B /tmp/vkop_84514d1/build -DENABLE_LLM_CHAT=OFF -DENABLE_TESTS=OFF`
- [x] `make -C /tmp/vkop_84514d1/build shader_archive`
- [x] 用旧树转换器 + 旧 archive 转 `/tmp/reg/int8_fused_old.vkopbin`
- [x] 追加隔离臂：`e0366cf`、`4a99ed9` 各转一份（权重/图元数据与旧臂逐字节相同，仅 shader archive 不同）
- [x] 失败回退：未使用（旧树 archive 构建成功）

## T5 基准测量（reps=208，min-of-3，A/B 交替）

- [x] 交替三轮：`int8_fused_old` / `int8_fused_new`（块 B，r4–r6，23:32–23:36 逐轮交替）
- [x] `int8_nofuse_new` 三轮（块 A，与 old r1–r3 交替）
- [x] `int8_fused_4a99` 三轮（块 C，与 old r7–r9 交替）
- [x] `fp16_fused_old` 三轮（旧树 fp16，作历史 fp16 量级参考）
- [x] 汇总：每臂 min avg ms/token；MatMul 次数与 GPU 时间；轮间相对差（噪声证据）

## T6 对照与判定

- [x] 计算同会话 `new vs old` delta（剥离环境噪声）：+34.0%（min-of-3 avg）/ +39.2%（稳态）
- [x] 对照历史：fp16 ~87.5 / MatMul 254 op 58.2 ms；int8 58.8–61.9；融合 +18%
- [x] 判定：≤3% 噪声内；>+5% 回归；<−5% 提升；3–5% 标注需更多样本
- [x] 责任 commit 归因：`598e445`（量化丢失 → 权重 +1.05 GB、MatMul +65%）；`4a99ed9` 性能中性但破坏正确性
- [x] int8 融合 vs 非融合本轮重测：旧树 12–21%（历史 18% 成立）；HEAD 融合反慢 19.6%

## T7 正确性

- [x] `./build/tests/all_tests --gtest_filter=BufferRankTest.*` → 58/58 PASSED
- [x] `./build/tests/all_tests --gtest_filter=*MatMul*` → 22/22 PASSED
- [x] 各模型逐 token 比对：`old` 正确且 4 轮一致；`nofuse` 正确（与 fp16 前 18 token 一致）；
      `new`/`4a99`/`pre598` 从第 0 个 token 即错误
- [x] 记录 int8 融合/非融合首个分叉位置：`old` vs `nofuse` 第 8 步（1817 vs 279，贪心近平分叉）；`nofuse` vs `fp16` 第 18 步

## T8 报告

- [x] 写 `docs/perf-regression-97ee63e.md`（结论 → 改动 → 实测表 → 历史对比 → 证据 → 复现命令 → 局限）
- [x] 对话内给中文摘要结论
- [x] 清理 `/tmp/reg` 大文件（保留日志），`git worktree remove`

# Task Dependencies

- T2 依赖 T1
- T3、T4 依赖 T2
- T5 依赖 T3、T4
- T6、T7 依赖 T5
- T8 依赖 T6、T7
