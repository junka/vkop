# Tasks — int8 SwiGLU down_proj 融合

## 实现

- [x] 1. `shaders/buffer/matmul.comp`：DOWN_ACT 段新增 `load_w8_b2` / `load_w8_ba` /
      `down_act_w_w8` 三个字节读取 helper（复用 `w_val`、`unpack_w8`）。
- [x] 2. 同文件新增 `matmul16_ksplit_down_act_w8()`：stage 1 按列 scale 提出 K 循环
      （`Sg = uScale[N + j]`、`Su = uScale[N + K + j]`），stage 2 用 `down_act_w_w8` 并在
      partial 落盘前乘 `Sd = uScale[c0..c0+3]`；行/列 stride、scratch 布局与 fp16 版一致。
      另加 `matmul16_ksplit_down_act_w8_m1()`：decode（M == 1）走单行 deposit，
      不再为 15 个死行保留 64 个累加器（实测每层 3.6 → 2.4ms）。
- [x] 3. 同文件 DOWN_ACT 的 `main()` 增加 `uPC.weight_int8` 分支（M<=1 走 m1）；
      段头注释记录 scale 三段布局 `[Sd(N) | Sg(N_up) | Su(N_up)]`。
- [x] 4. `ops/Matmul.hpp`：量化权重计数检查允许折叠字节形态（5 输入，`inputs[4]` fp32），
      `scale_index = 4`；`weight_byte` 的 "scale length == N" 轴校验对该形态跳过
      （长度由折叠分支自己校验）。
- [x] 5. `ops/Matmul.hpp`：折叠门接受 4（fp16）或 5（int8）输入；设置
      `para_.weight_int8` / `fp8` / `fp8_e5m2`，校验 scale 长度 `(n + 2*n_up)*4`，
      绑 `objs_[4] = inputs[4]`。
- [x] 6. `ops/Matmul.hpp`：更新折叠相关注释（契约、scale 布局），与 shader/optimizer 对齐。
- [x] 7. `model/pypi/onnx2vkop/optimizer.py`：`_match_weight_only` 对带 `down_act` 属性的
      节点分流到新 helper `_quantize_down_act_weights`（三份权重全量化 + 拼接 scale，
      并对 `only` / transB / 形状 / 4 字节对齐做前置判定）；不满足条件则整体保持 fp16；
      通用路径不变。
- [x] 8. optimizer 末尾加 `_validate_down_act_nodes` 守卫：`down_act` 节点三份权重 dtype
      必须一致；int8 必须有且仅有一个追加 fp32 scale，否则抛异常并点名节点。
- [ ] 9. `tests/BufferRankTest.cpp`：`brt_down_act_w8_case` + `MatMulDownActInt8` 已写入
      （含 m=1 / m=4 / m=64 / GX=2 四种形态）；待测试二进制构建完成后跑通确认。

## 验证

- [x] 10. `make -C build shader_archive vkop llm_chat` 通过。
- [x] 11. 转换 `/tmp/swiglu/i8_fused_w8.vkopbin`：28 个折叠节点全部 int8，merged scale
      14336 = 2048 + 2×6144，体积 1.72 GB（与非融合 int8 相同）。
- [x] 12. 融合 / 非融合 int8 同 prompt 跑通，31 个生成 token 逐字一致。
- [x] 13. `VKOP_PROFILE=1` + `VKOP_SUBMIT_PROF=1` 归因完成：见下方结论。
- [ ] 14. fp16 融合模型回归 **受阻**：按当前 master archive 重转的 fp16 模型在 prefill 报
      `/k_proj/MatMul needs the shader variant 'buffer_matmul_tiled_ks_split_fp16'`（模型内嵌
      archive 缺该 key，非本次改动所致；非融合 fp16 模型同样的告警但能跑）。fp16 非融合
      回归不受影响（115.6ms，与历史 116.8ms 一致，输出文本相同）。
- [x] 15. 结论：见下方。

## 结论（AMD 560X / Qwen3-VL-2B / decode 单 token）

| 模型 | p50 decode | MatMul 次数/步 | down_act(或 MLP 三连) 每层 |
| --- | --- | --- | --- |
| int8 非融合 | 60.4 ms | 198 | gate+up+down = 1.24 ms |
| int8 融合（旧 16 行 tile） | 94.4 ms | 142 | 3.6 ms |
| int8 融合（M=1 kernel） | 94.4 ms | 142 | 2.4 ms |
| fp16 非融合（历史） | 116.8 ms | 198 | — |
| fp16 融合（历史） | 87.1 ms | 142 | 2.7 ms |

int8 融合比非融合慢约 56%（94 vs 60ms）：decode 时整条链路是权重带宽受限，而融合并不减少
权重字节（三个权重都要读），能省的只有 dispatch 数与中间激活；只要折叠核达不到普通 int8
GEMV 的带宽效率（30GB/s vs 实测 16GB/s），就是净亏。理论天花板也只是持平（~1.25ms/层）。

stage 1 剩余的 2× 差距来自「每 lane 一次只取一个字节」（25M 次标量 load/层），普通 int8 GEMV
用的是每 lane 16 列、一次 16 字节的 wide 载入。要追平需重做 stage 1 的列向量化，但收益上限
仅等于普通 GEMM —— 结论是 **int8 不建议开 SwiGLU 融合**，`--no-fusions=fuse_swiglu_down_proj`
仍是 int8 的最优路径；fp16 才是融合的收益区间。
