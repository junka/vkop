# int8 SwiGLU down_proj 融合规格（fuse vs 非 fuse A/B）

## 目标

让 int8 权重模型也能走 SwiGLU down_proj 折叠（`down_act`），从而在同一精度下做
「fuse / 非 fuse」的端到端 A/B。成功判据：

- 转换出的 int8 融合模型能正常推理，输出与非融合 int8 模型逐 token 一致；
- 权重体积 ≈ 1.73 GB（与非融合 int8 相同，不再有 2.43 GB 混合精度模型）；
- decode 稳态 < 58.8 ms/token（当前 int8 非融合基线），预期 45–48 ms/token；
- MatMul dispatch 数由 198 降到 ~142（gate/up/down 三次 GEMM → 一次折叠）。

## 现状与根因（已验证）

1. 折叠 pass `fuse_swiglu_down_proj`（`model/pypi/onnx2vkop/optimizer.py` ≈5516,
   priority 62）把 down 节点改写成 4 输入 `[x, Wd, Wg, Wu]` 并打 `down_act=[1]`
   （同文件 ≈5672）。
2. 量化器 `_match_weight_only`（≈8148–8260）只处理 `inputs[1]`：Wd 变 int8 并在末尾
   追加 scale，节点变 5 输入；Wg/Wu 仍是 fp16 → 2.43 GB。
3. 运行期 `ops/Matmul.hpp` ≈499–508 对量化权重强制 `inputs.size() == 3`，5 输入直接
   `throw "weight needs exactly [A, B, scale], got 5 inputs"`。这是崩溃点。
4. 即使放过计数检查也不行：`fp16_` 由 `inputs[0]`（x，int8 模型的激活仍是 fp16）决定，
   所以会进 fp16 折叠分支；而该分支的 `down_act_gj/uj/w`
   （`shaders/buffer/matmul.comp` ≈1875–1894）按 `unpackHalf2x16` 读权重，把 int8 字节
   当 half2 解释 → 结果错误。
5. 折叠门在 `ops/Matmul.hpp` ≈1472：`down_act_ && down_act_pipeline() && fp16_ &&
   !transB_ && inputs.size() == 4`。模块只有 `buffer_matmul_down_act_fp16` 一个变体，
   CMake ≈387–396 用 `-DFP16 -DDOWN_ACT -DKSPLIT512` 构建，shader 内
   `#if !defined(FP16) #error` 要求 FP16。

## 设计决策

### 1. scale 打包：一个拼接的 fp32 张量，绑在 binding 4

模块只有 9 个 descriptor（0–8：out / A / Wd / scratch / scale / global / epi_bias /
Wg / Wu）。新增 binding 9/10 需要改 pipeline 的 descriptor 数量、converter 的输入契约和
archive 过滤，收益为零。因此三个权重的 per-column scale 拼成一个一维 fp32 张量：

```
[ Sd (N) | Sg (N_up) | Su (N_up) ]     长度 N + 2 * N_up
```

- `Sd`：Wd [N_up, N] 的 per-output-column scale；
- `Sg` / `Su`：Wg / Wu [K_out, N_up] 的 per-column（即 per-up-column）scale。

折叠节点的输入契约为：

```
fp16 : [x, Wd, Wg, Wu]            (4 输入，现状不动)
int8 : [x, Wd_i8, Wg_i8, Wu_i8, scale]   (5 输入，weight dtype = int8, scale = fp32)
```

三份权重都是 dense、行主序、transB == 0，与非折叠 int8 权重的字节布局完全一致
（`load_w8(idx) = uInputB.data[idx>>2] 的第 (idx&3) 个字节`）。

### 2. 数学等价性

per-column scale 沿 K 恒定，因此可以提出 K 循环：

```
gate_raw = Σ_k x[k] * Wg[k][j]      gate = gate_raw * Sg[j]
up_raw   = Σ_k x[k] * Wu[k][j]      up   = up_raw   * Su[j]
h        = silu(gate) * up                                   (fp32, 存 shared)
down[c]  = Σ_j h[j] * Wd[j][c]      out[c] = down[c] * Sd[c] (累加完再乘，然后 pack fp16)
```

与非折叠 int8 路径的「fp32 累加 + 一次 per-column scale」同构，差异只在 fp32 累加顺序。

### 3. 复用现有两段式（deposit + fold）

`matmul16_ksplit_fold_down_act()`（≈2017）只对 fp32 partial 求和并 pack fp16，与新
权重格式无关，**完全不改**。只新增 deposit 的 int8 版本。模块仍是一次编译、一个
archive key（`buffer_matmul_down_act_fp16`），`main()` 按 `uPC.weight_int8` 选路。

## 逐文件改动

### A. `shaders/buffer/matmul.comp`（仅 DOWN_ACT 段，≈1831–2172）

1. 新增字节读取 helper（复用已定义的 `w_val`，它在所有 build 都可见）：
   ```glsl
   float load_w8_b2(int idx) { return w_val((uInputB2.data[idx >> 2] >> ((idx & 3) * 8)) & 0xFFu); }
   float load_w8_ba(int idx) { return w_val((uInputAct.data[idx >> 2] >> ((idx & 3) * 8)) & 0xFFu); }
   ```
2. 新增四列 down 权重读取（c0 % 4 == 0，由 host 的 `n % 4 == 0` 门保证）：
   ```glsl
   vec4 down_act_w_w8(int b, int j, int c0) {
       uint wd = uInputB.data[((bmap(b) * uPC.K + j) * uPC.N + c0) >> 2];
       return unpack_w8(wd);
   }
   ```
3. 新增 `void matmul16_ksplit_down_act_w8()`，结构照抄 `matmul16_ksplit_down_act()`
   （≈1906–2011），差异只在读数与 scale：
   - stage 1：8-k 一组的 MLP 循环里，把 `down_act_gj/down_act_uj` 换成
     `load_w8_b2((bb*uPC.K_act+k)*uPC.K+j)` / `load_w8_ba(...)`（8 组地址独立，保持
     现有 fp16 版本的「全 issued 再消费」规则）；循环结束后
     `gr *= uintBitsToFloat(uScale.data[uPC.N + j]);`
     `ur *= uintBitsToFloat(uScale.data[uPC.N + uPC.K + j]);`
     再写 `da_h[lane*DA_MT+r] = gr/(1.0+exp(-gr))*ur;`
   - stage 2：`vec4 w = down_act_w_w8(b_n, jj, c0);`；partial 落盘前乘
     `uScale.data[c0 .. c0+3]`（4 个 fp32），其余（`stride/base/mind_k` 布局）不变。
   - 越界语义沿用：`j >= uPC.K` 的 lane 与 `r >= mt` 的行喂 0，`silu(0)*0 == 0`。
4. `main()` 的 DOWN_ACT 分支（≈2166）改为：
   ```glsl
   void main() {
       if (uPC.ks_part != 0) {
           matmul16_ksplit_fold_down_act();
       } else if (uPC.weight_int8 != 0) {
           matmul16_ksplit_down_act_w8();
       } else {
           matmul16_ksplit_down_act();
       }
   }
   ```
5. 在 DOWN_ACT 段头注释里记下 scale 的三段布局（ABI 文档的一处）。

不改 `CMakeLists.txt`、不改 `shader_map.py` 的 key、不加 SPIR-V 目标。

### B. `ops/Matmul.hpp`

1. 量化权重输入计数（≈499–508）：增加折叠字节形态。在 `if (weight_byte ||
   weight_4bit)` 内先判
   ```cpp
   const bool down_act_byte = down_act_ && weight_byte && inputs.size() == 5;
   if (down_act_byte) {
       if (inputs[4]->elem_kind() != core::ElemKind::kFloat32) throw ...;
       scale_index = 4;
   } else if (inputs.size() != want) { 原有 throw }
   ```
   （长度校验放折叠分支内，此时 `n` / `n_up` 已知。）
2. 折叠门（≈1472）：
   ```cpp
   const bool da_byte = inputs.size() == 5;
   if (down_act_ && down_act_pipeline() != nullptr && fp16_ != 0 &&
       !transB_ && (inputs.size() == 4 || (da_byte && weight_byte)) &&
       rank_a >= 2 && rank_b >= 2) {
   ```
   并在 `da_ok` 里加一条 `!da_byte || (inputs[4]->size() ==
   (size_t)(n + 2 * n_up) * sizeof(float))`（不满足就 throw，转换器 bug 要炸得响）。
3. 参数与绑定：
   ```cpp
   para_.weight_int8 = (da_byte && weight_int8) ? 1 : 0;
   para_.fp8      = (da_byte && weight_fp8) ? 1 : 0;   // 前向兼容；converter 只出 int8
   para_.fp8_e5m2 = (da_byte && bkind == core::ElemKind::kFloat8E5M2) ? 1 : 0;
   ...
   objs_[4] = da_byte ? std::static_pointer_cast<VulkanResource>(inputs[4]->gpu_buffer())
                      : nullptr;
   ```
   `objs_[5]/[6]` 保持 null；`objs_[2]/[7]/[8]` 仍绑 Wd/Wg/Wu（只是变成字节）。
4. 在 `execute()` 顶部把 `para_.weight_int8 = 0;`（与 `tile/ksplit/ks_part` 同一批默认值）
   显式重置，避免非折叠路径继承残留。
5. 更新 ≈1447–1471 与 ≈2252–2258 的注释：写明 int8 折叠的 5 输入契约与 scale 布局。

### C. `model/pypi/onnx2vkop/optimizer.py`

在 `_match_weight_only`（≈8148）入口处按节点属性分流：

- 若 `node.attributes.get("down_act")` 为真 → 走新 helper `_quantize_down_act`，不走通用
  逻辑；通用路径（含 `inputs[1]` 单权重量化）保持逐字节不变。
- `_quantize_down_act` 要求：
  - `inputs[1..3]` 必须是 initializer 且 dtype 为 fp16（或 fp32，按现有支持的 dtype），
    `transB` 缺失或 0，rank == 2；
  - 三份都满足才做全 int8；任一份不满足 → 整体放弃（保留 fp16 + 现有 `down_act` 折叠，
    仍然可跑），**绝不产出混合精度折叠节点**（这正是当前崩溃的来源）。
- 量化：对每份权重按列（沿 rows 轴）取 absmax，`scale = absmax / 127`（`scale == 0`
  的列置 1 且权重全 0），`q = clamp(round(w / scale), -127, 127)` → int8，dtype/编码
  与非折叠 int8 路径产出的权重一致（转换后与 `i8_nofuse.vkopbin` 的同名权重做字节比对
  校验）。
- 打包：新建一维 fp32 initializer `down_act_scale`，长度 `N + 2*N_up`，内容
  `concat(sd, sg, su)`；节点 `inputs = [x, Wd_i8, Wg_i8, Wu_i8, down_act_scale]`，
  `down_act` 属性保留。
- 该节点不再追加单独 scale、不再跳过 `inputs[2]/[3]`。

### D. 守卫（防止再产出崩溃模型）

转换结束时（或在 int8 量化 pass 末尾）校验：任何带 `down_act` 的节点，其
`inputs[1..3]` 的 dtype 必须三者一致（全 fp16 或全 int8）；若 int8，则必须有且仅有
一个追加的 fp32 scale 输入。违反即抛异常，信息里点名节点名与三个权重的 dtype。

## 重建 / 转换 / 验证

```bash
make -C build shader_archive vkop llm_chat
python3 -m onnx2vkop.cli -i llm/llm.onnx -o /tmp/swiglu/i8_fused_w8.vkopbin \
  -q int8 --master-archive build/vkop_shaders.bin --target-backend buffer
```

- 基线：`/tmp/swiglu/i8_nofuse.vkopbin`（`--no-fusions=fuse_swiglu_down_proj`，已存在）。
- 回归：`/tmp/swiglu/fused.vkopbin`（fp16 融合）输出必须与改动前一致。
- 推理命令模板：
  ```bash
  printf '<prompt>' | VKOP_NO_COOP=1 VKOP_PROFILE=1 VKOP_RAW_PROMPT=1 \
    ./build/llm_chat <model.vkopbin> llm/exporter/embed_tokens.bin \
    llm/tokenizer/qwen3_vl.bin <max_new> --max-kv 320
  ```

验证项（按顺序）：

1. `VKOP_SWIGLU_DEBUG=1` 转换日志 `matches=` 仍为 28；模型体积 ≈1.73 GB。
2. 文本/逐 token 比对：融合 vs 非融合 int8 贪心生成结果一致（允许极少数 logits 近
   平局处分叉，此时用 `tests/BufferRankTest.cpp::brt_down_act_case` 加一例 int8 数值
   校验兜底）。
3. `VKOP_PROFILE=1`：稳态 p50/p90 ms/token、MatMul 次数（198 → ~142）、是否出现
   `down_act_gemv` 路由（没出现说明静默退化成了普通 down GEMM）。
4. `VKOP_SUBMIT_PROF=1`：给出 deposit / fold 两个 dispatch 的耗时归因。
5. fp16 融合模型回归无变化。

## 风险

- **scale 布局是隐式 ABI**：shader / Matmul.hpp / optimizer 三处注释必须一致，否则以后
  改一处会静默算错。任务清单里单独列一条。
- **字节数 > 4 的地址计算**：`(b*K + k)*N + j` 用 `>> 2` 取字，要求整张权重从字节 0
  紧密排列且总长 4 对齐；与 `load_w8` 一致，保持 dense。
- **N / N_up 对齐**：host 已有 `n % 4 == 0`、`n <= 2048`、`n_up % 4 == 0` 三道门，int8
  版本不需要额外约束（`c0 = lane*4`、按字读 Wd 正好落在 `ja` 对齐）。
- **数值差异**：int8 量化 + 折叠后 fp32 partial 求和顺序不同，可能带来 token 级分叉；
  判据用「绝大多数 token 一致 + int8 单测」而非逐位相等。

## 不在本次范围

- fp8 / nvfp4 / int4 的折叠权重（host 参数留了前向兼容位，converter 暂不产出）；
- 256-lane deposit 变体、prefill（M > 16）的 tiled 折叠；
- Wg/Wu 与 Wd 共享权重、或 half2 打包的半精度累加优化。
