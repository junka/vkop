# SwiGLU fold 性能归因报告：84514d1 → 方向 A

> 范围：2026-11-02 的 int8 fold 基线到 2026-11-06 的 M==1/M>1 双 key 拆分。
> 目的：回答两件事——「两个真实 bug 修完后为什么还慢 4.6 ms/步」以及「方向 A 有没有把这段回退拿回来」。
> 文中所有毫秒均来自本机实测（MoltenVK / AMD Radeon Pro 560X，`VKOP_NO_COOP=1`，`-O0 -g`）。
> 推算值单独标注「推算」；凡是标「参考」的绝对毫秒都不作判据。

---

## 0. 结论速览

| 问题 | 结论 | 关键证据 |
|---|---|---|
| 两个真实 bug | 3-D lane-major 权重被 `ndim != 2` 判据整体跳过（28 个 fold 全回退 fp16，模型 1.7267 → 2.782 GB）；`shape[rank-2]` 把 `K_out` 读成 768（应为 2048），首 token 起只收缩前 768 个 K 切片 | `8415f5d`、`78b6b80` |
| 修完后残留的 4.6 ms/步 | 全部来自 `buffer_matmul_down_act_fp16` 这一个 archive 条目：整包替换 old archive（swap）与只替换该条目（splice）都回到 old 水平；布局贡献 0（old/fix 的权重字节 sha256 完全相同）；`ea3bdd0` 的 replay pre-pass ≈0 | 四臂比值 5.3489 / 6.3075 / 5.3512 / 5.3472；`fix−splice` = down +4.275 ms |
| 方向 A 是否回收 | 回收。`down_proj/lm_head` p10 比值 **6.2918 → 5.3743**（old 5.3489，仅 +0.48%）；按比值推算回收 ≈ 3.8–3.9 ms/步（splice 口径为 4.2–4.4，见 §3.4） | 3 轮交错 × 208 步 |
| prefill 是否退化 | 未退化。`_multi` 与拆分前模块**指令流完全相同**（10,018 条）；m=10 热态 min-of-3 **59.22 vs 59.42 ms/层** | SPIR-V diff + 双 prompt 热态 |
| 死分支是否真被裁掉 | 是。M==1 模块 **7,355 条指令**（old 7,581、合并版 10,018）；171,692 B 相对 old 的 +6,628 B 全部是 `-g` 内嵌源码文本 | 反汇编计数 + 内嵌源码长度 |
| 正确性 | `BufferRankTest.*` 59/59、`*MatMul*` 23/23；old/拆分前/新臂 e2e token 流 md5 一致（12 token，含 EOS） | 见 §4.4 |

---

## 1. 事件链与时间线

### 1.1 起点

`84514d1`（11-02）：把 SwiGLU down-proj fold 的三个权重（Wg/Wu/Wd）量化到 int8，这是「int8 融合 fold」的基线。

### 1.2 11-03：四个功能/性能 commit（加两个同日收尾）

| commit | 内容 | 与本报告的关系 |
|---|---|---|
| `4a99ed9` | fold deposit 改为「每个 row-tile 只读一次权重，而不是每 row 读一次」 | 引入了多行 tile walk；它的输出全 0 问题不在本次范围，未逐指令定位 |
| `e0366cf` | MoE slot bytes 从 checkpoint 分页 | 与本链路无关（仅入时间线） |
| `598e445` | gate/up 权重重排为 lane-major | **纯 reshape**：`[K_out, N_up]` → `[K_out, N_up/8, 8]`，内存顺序不变（见 §3.3） |
| `97ee63e` | release 构建清理 + shader archive fixture 诚实化 | archive 组成变化的背景 |
| `754e839`（同日） | fold 多行 deposit 做 `DA_TRIP` unroll | 方向 A 要保留的 prefill 收益来源；也是「死代码拖慢 M==1」的源头 |
| `ea3bdd0`（同日） | runtime 结构化 replay gating（`has_host_side_effects`） | 候选 ③；本次实测贡献 ≈0 |

### 1.3 11-04 / 11-05：六项工程改动

`734baea`（fp16 tie-flip 分类）、`803a2cc`（`VKOP_FORCE_COOP` 旁路）、`07fc838`（RenderDoc 下的 coopmat 枚举守卫）、`edc9370`（测试 PCH，随后被 `96d97cc` 撤掉）、`b5baf41`（yolo 相机窗口）、`61b55e6`（SK_LASTBLOCK coopmat split-K）。

> `61b55e6` 是后续构建阻塞的来源：`ops/Matmul.hpp` 引用了 `kCoopmatSkLast` / `kCoopmatInwg` / `MatmulPlan::coop_split`，但枚举与字段从未入库；`CMakeLists.txt` 还注册了一个仓库里不存在的 `shaders/buffer/matmul_coop16_inwg.comp`。HEAD 因此**完全无法构建**，`build/vkop_shaders.bin` 缺失、`build/llm_chat` 停在 11-04。

### 1.4 11-05：两个真实 bug 的修复

| commit | 内容 |
|---|---|
| `8415f5d` | 量化器接受 3-D lane-major fold 权重（bug 1） |
| `78b6b80` | 3-D 权重从首维取 `K_out`（bug 2），并拒绝 `N_up % 8 != 0` |
| `c313eed` | 让 PCH 提交兼容旧 CMake / 旧 torch（构建链） |

### 1.5 修复后的残留归因（本报告 §3）

修完两个 bug 后：59.25 → 63.85 ms/token，MatMul 42.47 → 44.52 ms，仍比 old 慢约 4.6 ms/步（+7.8% 墙钟、+4.8% MatMul）。归因方法：四臂 + archive 单条目 splice，判据换成频率无关的 `down_proj/lm_head` 比值。

### 1.6 方向 A（11-05/06）

| commit | 内容 |
|---|---|
| `7bd14bd` | 修复 `61b55e6` 遗留的构建阻塞（删除不存在的 `coop16_inwg` 变体；补齐枚举/字段/tag），让树重新可构建 |
| `a1132de` | fold 源码构建两次：M==1 用 `-DDA_M1_ONLY` 裁掉多行 walk，M>1 保留；host 按 `m` 路由；`shader_map.py` 同时携带两个 key |

> 备注：这两个 commit 在本会话中最初落为 `781fe91` / `974285d`，后续仓库历史重写后对应当前分支的 `7bd14bd` / `a1132de`，内容一致。

---

## 2. 两个真实 bug

### 2.1 Bug 1：3-D lane-major 权重被量化器一刀切跳过

**现象**：`598e445` 把 fold 的 gate/up 权重改成 lane-major `[K_out, ceil(N_up/8), 8]` 之后，转换器日志里的 `Converted ... to int8` 从 28/28 变成 0/28；模型从 1.7267 GB 涨到 **2.782 GB**（+61%），每一步多读约 1 GB 权重。

**根因**：`optimizer.py::_quantize_down_act_weights` 用 `ndim != 2` 作为「是不是 fold 权重」的判据，3-D 形状直接走「不认识 → 跳过」，于是 28 个 fold 全部保持 fp16，fold 路径整体回退。

**修法（`8415f5d`）**：不再用维数做判据，而是按布局识别：

- Wd 必须是 `[N_up, N]`；
- Wg/Wu 必须彼此匹配，且要么是 `[K_out, N_up]`（2-D，列轴 1），要么是 `[K_out, bands, 8]` 且 `bands * 8 == N_up`（lane-major，列轴最后一维）；
- scale reduce 仍沿 axis 0（`K_out`）；3-D 时把 `(bands, 8)` 按 `band * 8 + lane` 展平，与 payload 对齐；
- 其他情况（转置、`N_up` 不匹配、bands 不足）仍然回退 fp16，避免量化出内核读不懂的布局；
- `N_up % 8 != 0` 的补齐布局在此拒绝：shader 用 `uPC.K = n_up` 做行 stride，补齐后的行宽会与 stride 不符。

**效果**：28/28 重新走字节路径；模型回到 1,726,722,696 B；新增 `model/pypi/tests/test_down_act_lane_major_quant.py`（3/3）。

### 2.2 Bug 2：`shape[rank-2]` 取 `K_out` 得 768 而非 2048

**现象**：修好量化后，首 token 起 logits 就是错的——fold 只收缩了前 768 个 K 切片。

**根因**：host 侧形状映射对 `K_out` 和 `N_up` 都读 `shape[rank - 2]`。对 lane-major `[K_out, 768, 8]`，`rank-2` 是 **band 数 768**，不是收缩长度 2048。

**修法（`78b6b80`）**：

- `K_out` 取 `shape[0]`（两种布局的 Wg/Wu 都把 `K_out` 放在首维）；
- `N_up` 取 Wd 的 `shape[0]`；
- 加 `down_k_out != k` 守卫，未来布局再分叉时**抛错**而不是静默截断；
- `N_up % 8 != 0` 时拒绝 fold（与 2.1 的量化侧拒绝同源：补齐行宽与 shader stride 不一致）。

**测试**：`MatMulDownActLaneMajorW8`（同一份 flat payload 的 2-D / 3-D 声明输出一致）、`MulMatDownActRejectsNUpNotMultipleOf8`（`N_up=12` 抛错）；e2e 12 token 逐 token 相等（含 EOS）。

### 2.3 修复的量化效果

（来自 `78b6b80` 的提交正文，单 prompt、min-of-3、steady = steps ≥ 100 中位）

| 臂 | 墙钟 ms/token | MatMul 中位 ms |
|---|---|---|
| `84514d1` old 2-D int8 | 59.25 | 42.47 |
| broken HEAD | 84.2 | 66.2 |
| 修复后（本树） | 63.85 | 44.52 |
| 同树 no-fold | 65.20 | 44.92 |

34% 的回归被消除；相对同树 no-fold 只 +2% 墙钟 / +0.9% MatMul。**但相对 old 仍有约 4.6 ms/步的残差**，这才进入 §3。

---

## 3. 残留 4.6 ms/步的单变量归因

### 3.1 四个候选与实验设计

| # | 候选 | 变量 | 实验臂 |
|---|---|---|---|
| ① | 权重布局 2-D → 3-D lane-major（`598e445`） | 权重形状声明 | old / fix 权重字节 sha256 对比 |
| ② | shader archive 变化（`4a99ed9`、`754e839`） | 单个 SPIR-V 条目 | **splice**：fix archive 只把 `buffer_matmul_down_act_fp16` 换成 old 条目 |
| ②' | 整包 archive | 整包替换 | **swap**：fix 模型换 old 全量 archive（交叉验证） |
| ③ | replay pre-pass / `SIMULTANEOUS_USE`（`ea3bdd0`） | 默认路径门控 | 四臂差分里看残差 |

### 3.2 splice 臂怎么做到「只换一个 entry」

模型 `.vkopbin` 里嵌着一段 VKSP archive：`magic(4) + version(4) + count(4) + pad(4)`，每条 entry 72 字节（name 48B、offset u64、size u32、crc u32、reserved 8B），blob 区紧随其后、4 字节对齐。模型文件最后 8 字节是权重 blob 起始偏移（LE u64），所以 archive 的区域长度 = `weight_offset − vksp_offset`。

归因脚本（`tools/perf/attribution/archive_splice.py`）做的事：

1. 在权重 blob 之前 `rfind(b"VKSP")`，解析全部 entry；
2. 按 entry 顺序重新打包 blob，只把目标 key 的字节换成 donor 模型的同 key；
3. **保持 archive 区域总长度不变**（多出来的尾部清零），这样权重 blob 不移动、文件末 8 字节的偏移仍然正确；
4. 重新计算 crc32（`zlib.crc32`），虽然 `core/ShaderArchive` 只校验 offset/size/magic，不校验 crc。

这样 splice 模型与 fix 模型的差异严格限定在**一个 entry 的字节**上，权重、其它 shader、flatbuffer 全部不变。四臂（old / fix / swap / splice）各自转换出完整模型后，交错跑同一 prompt。

### 3.3 为什么判据是 `down_proj / lm_head` 比值，而不是绝对毫秒

本机（MoltenVK + AMD 560X）在持续负载下时钟会爬坡，**同一个模型不同轮次的绝对毫秒中位可漂 ±25%**（本会话甚至见过单臂单轮整体 +54%）。但同一 step 内所有 op 共享同一时钟：`down_proj` 与 `lm_head` 的**比值**把频率因子约掉，实测自噪声只有 **±0.3%**（同一臂三轮 p10 分别 6.2916 / 6.2949 / 6.2885）。

因此：

- **判据**：每个稳态 step 的 `sum(down_proj) / lm_head`，取 p10（避开偶发抢占的长尾）与中位；
- **绝对毫秒**只作为「换算成每步多少 ms」的参考，必须与同轮次、同时钟的臂比较，不能跨轮次比单值；
- 所有 A/B 都是**同 host、交错、min-of-3、reps ≥ 200**。单轮墙钟（尤其单值）一律不下结论。

### 3.4 四臂结果

比值（p10，稳态）：

| 臂 | down/lm_head p10 | 说明 |
|---|---|---|
| old（`84514d1`，2-D + old archive） | **5.3489** | 目标水平 |
| fix（修复后 HEAD，3-D + 新 archive） | **6.3075** | +17.9% 回退 |
| swap（fix 权重 + old 全量 archive） | **5.3512** | 整包换回 old 即复现 old |
| splice（fix archive 只换 `down_act` 条目） | **5.3472** | 只换这一个条目就复现 old |

绝对毫秒（同一批 run 的逐 op 中位，参考）：

| 臂 | MatMul 聚合 | down_proj | 每步 GPU 总 |
|---|---|---|---|
| old | 38.970 | 23.318 | — |
| fix | 42.680 | 27.301 | — |
| swap | 38.470 | 23.077 | — |
| splice | 38.240 | 23.026 | — |

`fix − splice` = down **+4.275 ms**、MatMul **+4.440 ms**、每步 GPU 总 **+4.655 ms**。

### 3.5 归因结论

- ① 布局：**0 ms**。`598e445` 的 lane-major 只是把 `[K_out, 6144]` 解释成 `[K_out, 768, 8]`，两个模型的权重字节 **sha256 相同**，内核读的是同一段内存。
- ② shader archive：**全部 4.2–4.4 ms**。swap 与 splice 都精确落回 old；splice 只动了 `buffer_matmul_down_act_fp16` 一个条目的字节。
- ③ `ea3bdd0` 的 replay pre-pass：**≈0**（<0.2 ms，落在残差里）。
- 未解释残差：**≈0.16 ms**。

进一步解释机制：`754e839` 的多行 deposit walk 对 M==1 是**死代码**（wide gate 在上面先 return，单行又走 `mt==1` 标量分支），但把它留在同一个 entry 里，SPIR-V 从 165,064 B 涨到 213,440 B；**执行的指令相同、模块更大，decode fold 反而慢 18%**——这是代码体积/寄存器压力，不是算法变化。这直接催生了方向 A。

---

## 4. 方向 A：把 M>1 拆成独立 key

### 4.1 构建方式与主机路由

同一个 `shaders/buffer/matmul_down_act.comp` 构建两次：

| key | 构建 | 行为 |
|---|---|---|
| `buffer_matmul_down_act_fp16` | `glslc -DDA_M1_ONLY` | 裁掉 `DA_TRIP` 多行 tile walk；`#else` 保留 pre-`4a99ed9` 的 per-row walk（旧 archive 兜底） |
| `buffer_matmul_down_act_multi_fp16` | `vkop_shader_variant`（无宏） | 与拆分前**完全相同**的多行 unroll |

主机侧（`ops/Matmul.hpp`）：

- `down_act_pipeline(bool multi)` 为每个 key 各持有一份 pipeline + `kInflight` 个 descriptor set；选择器阶段就把 `_multi` 也建好，避免在命令录制中建 pipeline；
- dispatch 处 `m > 1` 才切到 `_multi`，否则用 M==1 模块；
- 判定用的 `m` 就是 `da_deposit_wide_byte` 门限读的同一个变量，**没有第二份 m 逻辑**；
- archive 里没有 `_multi` 时（拆分前转的旧模型）回落到 `_fp16`；旧模型的 `_fp16` 本身是合并模块，行为与拆分前一致。

`model/pypi/onnx2vkop/shader_map.py` 同时列出两个 key，保证新模型把两份 SPIR-V 都嵌进去。

### 4.2 archive 字节数与指令条数

| 条目 | old | 拆分前（合并） | 方向 A |
|---|---|---|---|
| `_fp16` | 165,064 B / 7,581 指令 | 213,440 B / 10,018 指令 | **171,692 B / 7,355 指令** |
| `_multi` | — | — | **215,456 B / 10,018 指令** |

关键点：

- **死分支确实被裁掉**：M==1 模块的指令数 **7,355** 比 old 的 7,581 还少，更远少于合并版的 10,018。
- 171,692 B 相对 old 的 **+6,628 B 不是代码**：`-g` 会把 GLSL 源码文本嵌进 SPIR-V，当前源文件 52,762 B vs old 42,329 B（+10,433 B），代码反而小了；净差 ≈ +6.6 KB。
- `_multi` 与拆分前模块的反汇编**指令流完全相同**：仅 `OpSource`/`OpLine` 等调试文本不同（diff 里没有任何非调试指令差异），所以 m>1 的行为、速度、正确性都不变。

### 4.3 decode 效果

208 步、3 轮交错、每轮去掉前 4 步预热、`VKOP_SUBMITPROF_TOPN=0`：

| 臂 | 比值 p10（逐轮） | 合并 p10 | down 中位 | MatMul 中位 | 每步 GPU 总中位 |
|---|---|---|---|---|---|
| 拆分前 | 6.2916 / 6.2949 / 6.2885 | **6.2918** | 26.02 ms | 40.68 ms | 44.80 ms |
| 方向 A | 5.3714 / 5.3735 / 5.3789 | **5.3743** | 22.64 ms | 37.55 ms | 41.82 ms |
| old（§3 日志） | — | 5.3489 | 23.31 ms | 38.96 ms | 43.64 ms |

- 比值回收 **14.6%**，距 old 仅 **+0.48%**。按比值把 fix 的 down 中位 26.02 ms 归一到 old 时钟：应为 22.12 ms，方向 A 实测 22.64 → 归一后 22.23 ms；即回退量 ≈ 3.9 ms/步中回收 **≈ 3.82 ms**，剩 **≈ 0.11 ms**（推算）。splice 口径给的是 4.2–4.4 ms（那一批 run 的 fix 比值 6.3075、down 中位 27.30，时钟更低）。
- 绝对毫秒：方向 A 的某轮因时钟偏低整体 +54%，但比值不变（5.3714），这正是只信比值的理由；同时钟轮次的 down 差为 −2.5～−3.5 ms、MatMul −1.7～−3.3 ms（参考）。

### 4.4 prefill（m>1）与正确性

**热态口径**：同一进程喂两遍相同 prompt（15 词 → 10 token）。第一遍 prefill 要为所有惰性 pipeline 付 MoltenVK 的 SPIR-V→MSL 编译，不能用于比较；第二遍是热态。

| 臂 | 每层 down 三次（ms） | 中位 | min-of-3 |
|---|---|---|---|
| 拆分前（回落 `_fp16` 合并模块，仍走 unroll） | 60.52 / 59.22 / 60.92 | 60.52 | **59.22** |
| 方向 A（`_multi`） | 59.51 / 59.42 / 64.00 | **59.51** | 59.42 |

方向 A 的日志中没有 `needs the shader variant 'buffer_matmul_down_act_multi_fp16'` 警告 → 确实命中 `_multi`。两者指令流相同，实测同水平，**m>1 未退化**。

**正确性**：

- `BufferRankTest.*` 59/59、`*MatMul*` 23/23（m=1 走 M==1 key；m=4/15/64 走 `_multi`，含真实形状 `K_out=2048 / N_up=6144 / N=2048`）；
- `MatMulDownActInt8`、`MatMulDownActRejectsNUpNotMultipleOf8` 通过；
- e2e：old / 拆分前 / 方向 A 三臂 12 token 的 greedy 流 **md5 完全相同**（含 EOS 151645）。

---

## 5. 方法论

1. **同 host、同二进制、交错 A/B**：所有臂在同一棵树的同一 `llm_chat` 上跑，臂之间只差模型文件；轮次按 A/B/B/A 交错，抵消时钟爬坡与热漂移。
2. **单变量**：能通过文件级替换隔离的，就绝不重编整个 archive（splice）；换不动的那一步（量化 bug）才整模重转。
3. **min-of-3 + reps ≥ 200**：decode 每臂 3 轮 × 208 步（每轮去前 4 步预热）；prefill 每臂 3 次，取 min。
4. **频率无关判据**：`down_proj / lm_head` 的 p10 比值。同 step 内共享时钟，比值自噪声 ±0.3%；绝对毫秒在持续负载下跨轮中位可漂 ±25%、单轮可到 ±50%，**禁止**用单轮墙钟或单值下结论。
5. **剔除首用开销**：MoltenVK 在首次 dispatch 时才编译 pipeline，第一块 prefill 会被污染；prefill 一律比较同进程第二次 prefill。
6. **区分「实测」与「推算」**：本报告的比值、中位、字节数、指令数、token 流是实测；「≈3.82 ms/步已回收、剩 ≈0.11 ms」是按比值换算的推算，已标注。

---

## 6. 已知局限与未解项

1. **prefill 绝对量级口径对不上**：任务书目标是 m=15 每层 fold ~3.1 ms（不得退回 5.9），但本机 int8 实图在 m=10 时两臂都是 ~60 ms/层（28 层合计 ~1.7 s），比 3.1/5.9 大一个量级；`754e839` 的提交正文里也没有该量级的实测记录，推测 3.1/5.9 来自 fp16 或另一台机器的 microbench。方向 A 前后**代码完全相同、实测同水平**，该量级问题与本次拆分无关，但仍未解释。
2. **int8 缺 m=15 用例**：int8 fold 单测是 m=1/4/64；m=15 只覆盖 fp16 的真实形状用例。两个 key 的覆盖由 m=1（M==1 key）与 m=4/15/64（`_multi`）保证，但 int8 没有单独跑 m=15。
3. **`_fp16` 的 `#else` 回退路径无模型实测**：它只能由「新 M==1 模块 + 没有 `_multi` 的 archive」触发；旧 archive 的 `_fp16` 本身是合并模块，不会进 `#else`，现有转换流程也不产生这种组合，因此未构造模型验证。
4. **`4a99ed9` 输出全 0 未逐指令定位**：从任务范围里排除，只记录现象。
5. **`ea3bdd0` 的性能风险未量化**：在本次 decode 图中贡献 ≈0（<0.2 ms），但其默认路径的 pre-pass / `SIMULTANEOUS_USE` 对其它图型是否有风险，未做通用测量。
6. **单轮绝对毫秒噪声大**：方向 A 有一轮整体 +54%（时钟低），比值仍稳定；报告中的绝对毫秒只在同时钟轮次间比较，跨轮次差值仅作参考。

---

## 7. 复现命令

```bash
# 1) 构建（-O0 -g 不变；先出 archive，再出可执行文件）
make -C build shader_archive vkop llm_chat -j6
#    产物：build/vkop_shaders.bin（含两个 fold key）、build/llm_chat

# 2) 重转 int8 融合模型（shader 嵌在 .vkopbin 里，必须用新 archive 重转）
PYTHONPATH=model/pypi python3 -m onnx2vkop.cli -i llm/llm.onnx \
  -o /tmp/reg_analysis/models/int8_fused_a.vkopbin -q int8 \
  --stream-weights off --master-archive build/vkop_shaders.bin

# 3) 单测
VKOP_NO_COOP=1 ./build/tests/all_tests --gtest_filter='BufferRankTest.*'
VKOP_NO_COOP=1 ./build/tests/all_tests --gtest_filter='*MatMul*'

# 4) decode A/B（3 轮交错、208 步、每步全量 MatMul）
tools/perf/attribution/run_ab3.sh \
  /tmp/reg_analysis/models/int8_fused_fix.vkopbin \
  /tmp/reg_analysis/models/int8_fused_a.vkopbin 208
python3 tools/perf/attribution/parse_blocks.py decode tools/perf/logs/ab3_*.log

# 5) warm prefill（两遍同 prompt，比较第二遍）
tools/perf/attribution/run_pre2.sh \
  /tmp/reg_analysis/models/int8_fused_fix.vkopbin \
  /tmp/reg_analysis/models/int8_fused_a.vkopbin 3
python3 tools/perf/attribution/parse_blocks.py blocks tools/perf/logs/pre2_*.log

# 6) archive 单变量（§3 的 splice 臂）
python3 tools/perf/attribution/archive_splice.py list \
  /tmp/reg_analysis/models/int8_fused_fix.vkopbin
python3 tools/perf/attribution/archive_splice.py splice \
  /tmp/reg_analysis/models/int8_fused_fix.vkopbin \
  /tmp/reg_analysis/models/int8_fused_old.vkopbin \
  buffer_matmul_down_act_fp16 \
  /tmp/reg_analysis/models/int8_fused_splice.vkopbin

# 7) SPIR-V 归一化比对（忽略 -g 调试文本）
spirv-dis build/shaders/buffer_matmul_down_act_fp16.spv -o /tmp/m1.spvasm
spirv-dis build/shaders/buffer_matmul_down_act_multi_fp16.spv -o /tmp/multi.spvasm
diff <(grep -vE 'OpLine|^;|^ *//' /tmp/m1.spvasm) \
     <(grep -vE 'OpLine|^;|^ *//' /tmp/multi.spvasm)
```

工具与环境：

- `VKOP_SUBMIT_PROF=1`、`VKOP_SUBMITPROF_TOPN=0`、`VKOP_PROFILE=1`；`VKOP_RAW_PROMPT=1` 用于多 prompt。
- 本报告的所有 run 都带 `VKOP_NO_COOP=1`（MoltenVK 上 coopmat 不可用/不稳）。
- 模型与日志都放在 `/tmp/reg_analysis/`（会随系统清理丢失）；持久化的是 §7.1 的脚本。

### 7.1 脚本清单（已在仓库内）

| 文件 | 用途 |
|---|---|
| `tools/perf/attribution/run_ab3.sh` | decode A/B：3 轮交错、208 步、`TOPN=0` |
| `tools/perf/attribution/run_pre2.sh` | warm prefill：同进程两遍 prompt，比较第二遍 |
| `tools/perf/attribution/parse_blocks.py` | 解析 submitprof 日志：比值 p10/中位、绝对中位、逐 block prefill |
| `tools/perf/attribution/archive_splice.py` | 解析/提取/就地替换 VKSP archive 的单个 entry（§3 的 splice 臂） |

> `run_ab3.sh` 与 `run_pre2.sh` 是会话中实际执行过的版本；`parse_blocks.py` 与 `archive_splice.py` 按实际逻辑重写（原 `/tmp/reg_analysis/` 副本已随系统清理丢失），未在本次重新跑过，引用前建议先 `--help`/小样本自检。

---

## 附录 A：关键数字汇总

| 指标 | old / 拆分前 | 方向 A | 备注 |
|---|---|---|---|
| 模型体积（量化 bug 期） | 2.782 GB | — | 28/28 fold 回退 fp16 |
| 模型体积（修复后） | 1.7267 GB | — | 28/28 走 int8 |
| `_fp16` archive 条目 | 165,064 / 213,440 B | 171,692 B | 7,581 / 10,018 → 7,355 指令 |
| `_multi` archive 条目 | — | 215,456 B | 10,018 指令，与拆分前指令流相同 |
| decode `down/lm_head` p10 | 5.3489 / 6.3075 | 5.3743 | 差 old +0.48% |
| decode down 中位 | 23.32 / 26.02 ms | 22.64 ms | 参考（时钟相关） |
| decode MatMul 中位 | 38.97 / 40.68 ms | 37.55 ms | 参考 |
| prefill m=10 每层 fold（热态 min） | — / 59.22 ms | 59.42 ms | 未退化 |

## 附录 B：判据自噪声

| 指标 | 同一臂三轮 | 波动 |
|---|---|---|
| 拆分前 `down/lm_head` p10 | 6.2916 / 6.2949 / 6.2885 | ±0.05% |
| 方向 A `down/lm_head` p10 | 5.3714 / 5.3735 / 5.3789 | ±0.07% |
| 绝对毫秒（同一臂跨轮） | 例：down 22.27 / 22.64 / 34.39 ms | **±50%** |

这张表就是「只信比值、不信墙钟单值」的直接依据。
