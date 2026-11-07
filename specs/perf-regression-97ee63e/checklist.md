# 验收清单：4-commit 性能回归（HEAD 97ee63e）

## 提交解析

- [x] 报告中列出 `4a99ed9` / `e0366cf` / `598e445` / `97ee63e` 的实际改动与文件，并逐项标注"是否可能影响 M=1 decode"
- [x] 明确 commit 顺序与改动前基线 `84514d1`

## 构建与产物

- [x] `make -C build vkop`、`make -C build shader_archive`、`make -C build llm_chat`、`make -C build all_tests` 均成功
- [x] `build/vkop_shaders.bin` 已重打（含 `matmul_down_act.comp`），`libvkop.a` / `llm_chat` 已重链
- [x] 无新增 `.dSYM`、无新增可执行文件；`-O0 -g` 保持仓库原状并在局限中说明

## 模型转换

- [x] 所有转换使用 `--stream-weights off`（规避既有 external weight 限制）
- [x] 记录旧臂 28 次 `Converted SwiGLU fold ... to int8`、HEAD 0 次
- [x] 记录模型体积（旧 1.727 GB vs HEAD 2.782 GB）

## 测量协议

- [x] reps ≥ 200（实际 207 个 decode step）
- [x] 同 reps A/B 交替：块 A（old/nofuse）、块 B（old/new）、块 C（old/4a99），逐轮交替
- [x] 每臂 3 轮、min-of-3，并给出稳态 med(step≥100) 作为时钟爬坡的对照
- [x] 报告的判定使用同块同轮配对，而非跨时段单轮绝对值

## 结论与历史对比

- [x] 给出本轮实测值、历史基线与 delta 百分比（fp16 / int8 / 融合收益）
- [x] 明确判定"有回归"，幅度 +34%（min-of-3 avg）/ +39%（稳态）
- [x] 责任 commit 指向 `598e445`，并用隔离实验证明 `4a99ed9` 性能中性
- [x] 说明是否复测基线：旧臂稳态 60.5 ms/token 落在历史 58.8–61.9 区间

## 正确性

- [x] `BufferRankTest.*` 全过（58/58）
- [x] `*MatMul*` 全过（22/22）
- [x] 旧模型逐 token 正确且多轮确定性一致
- [x] HEAD 非融合模型正确（与旧树 fp16 前 18 token 一致）
- [x] HEAD 融合模型与 `pre598` / `4a99` 的错误被记录，并用"权重逐字节相同、仅 shader 不同"完成归因

## 交付

- [x] `docs/perf-regression-97ee63e.md` 存在，含结论 → 改动 → 实测表 → 历史对比 → 证据 → 复现命令 → 局限
- [x] 清理 `/tmp/reg` 大模型文件与临时 worktree（保留日志）
