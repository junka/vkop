# vkop 长上下文正确性缺陷（待处理）

日期：2026-11-13
状态：**未修复**。HTTP 服务已绕过一部分（见「已做的缓解」），但根因仍在引擎里。

## 现象

`llm_chat` 用 raw 模式（`VKOP_RAW_PROMPT=1`，不套 chat template、每行独立序列）跑
纯文本，输出在 prompt 跨过某个长度后整体崩坏。可复现、确定性、与 HTTP 层和 chat
template 都无关。

用「The history of computing spans centuries, ...」重复拼接做长度扫描，
`llm_chat llm.vkopbin embed_tokens.bin qwen3_vl.bin 8 --max-kv 16384`：

| prompt tokens | 输出 |
|---|---|
| 301 | `The history of computing spans centuries, from mechanical calcul`（正常续写） |
| 601 | `1000 times.`（合理） |
| 901 | `The history...`（正常） |
| 1201 | `The history...`（正常） |
| 1501 | 以 `!` 开头后跟一段正常文字（**开始退化**） |
| 3001 | `!:.failonsotheloseth`（崩坏） |
| 6001 | 空（崩坏） |
| 12001 | 空（崩坏） |

1501 那条的输出首字符是 `!`，与已知的 NVIDIA coopmat 崩坏症状（`!!!!`）形状相同，
但那台是另一块卡；这台是 Intel ARL，且已用 `VKOP_NO_COOP=1` 排除 coopmat。

同一现象从 HTTP 服务侧也能复现（5416 / 3259 token 的 prompt 输出 `! Art def! Art def!`
与 `!?!?!?!?`），说明问题在 `llm_serve` 和 `llm_chat` 共用的引擎路径上。

## 已排除

- **不是 HTTP 层**：`llm_chat` 的 raw 模式直接喂 stdin，同样复现。
- **不是 chat template / 渲染**：raw 模式不过模板。
- **不是 coopmat**：全程 `VKOP_NO_COOP=1`。
- **不是 KV cache 显存**：28 层 × 16384 行 ≈ 1.88 GB，可正常分配；且崩坏发生在
  1500–3000 token，远未达到预分配上限。

## 已找到的相邻限制（可能不是根因，但都是硬墙）

1. **staging buffer 池上限 1 GB**
   `vulkan/VulkanStagingBufferPool.cpp`:
   ```
   constexpr int kMaxStagingBufferSize = 1024 * 1024 * 1024; // 1GB max
   ```
   超过时 `allocate()` 打印 `size too large <n>` 并返回 `nullopt`。
   实测：`--prefill-chunk 8192` 时 logits 缓冲 `8192 × 151936 × 2 ≈ 2.5 GB`，
   日志出现 `[ERROR] size too large 1645770752`（≈1.53 GB，是 1GB 上限被倍增越过后
   的判定点），随后输出变成乱码。这正是服务端把 `--prefill-chunk` 默认值压到 2048
   的原因：`2048 × 151936 × 2 ≈ 620 MB`，刚好在上限内。
   **这条限制决定了长 prompt 必须分块 prefill，而每块都会完整重算一次 lm_head。**

2. **logits 输出是 `(1, q, vocab)`**
   图里没有「只算最后一行」的路径，所以 q 越大越贵，且中间那块 logits 全部浪费。

## 与用户判断的对应

用户指出「vkop 长上下文支持有问题，看看 kvcache 支持的 resize 长度」。按目前证据，
KV cache 本身（`KVCache::reserve_from` / `preallocate_buffer`）的长度上限不是这个现象的
原因：
- 预分配是 `2*nkv*max_kv*hd` 个元素，与 `max_kv` 线性，16384 行只占 1.88 GB；
- `resume_from()` 只在 `prefix_len > max_kv` 时抛异常，而崩坏发生在远低于上限处；
- 崩坏是**输出质量**问题而不是异常/越界。

但以下位置仍值得单独核查（都没查完，因为本次任务转去提交）：
- `attention_bias` 的 fp16 缓冲在长 `kv` 维度上是否被 shader 正确读取（它是
  `(1,1,q,q+kv)`，1.5k 时约 4.5M 元素，规模不大，但如果 kernel 有分块/索引假设就可能出错）；
- `Attention` kernel 在 `kv_len` 超过某个 tile/块边界时的索引（例如按 1024 或 2048 分段的
  累加是否覆盖尾段）；
- `RotaryEmbedding` / `position_ids` 在长序列上的位置是否仍然正确（raw 模式下是
  2D `arange`，应当不会错）；
- matmul 的 K-split / tiled 路线是否在 m>1 的长 prefill 上出现与长度相关的路由切换
  （`llm_chat` 日志里 `[ksel] CHOSE ...` 会打印路由，可对比 1201 与 3001 两档的选择）。

## 快速复现命令

```bash
# 生成不同长度的同一段文本
python3 -c "
para='The history of computing spans centuries, from mechanical calculators to modern processors. '
print(para*100)" > /tmp/long.txt          # 1501 tokens

VKOP_NO_COOP=1 VKOP_RAW_PROMPT=1 ./build/llm_chat \
  llm/exporter/llm.vkopbin llm/exporter/embed_tokens.bin \
  llm/tokenizer/qwen3_vl.bin 8 --max-kv 16384 < /tmp/long.txt
```

## 对 HTTP 服务的影响（当前已缓解的部分）

- `--prefill-chunk` 默认 2048（受 1 GB staging 池约束，不能再用 8192）。
- 分块 prefill 会让 11k 的系统提示变成 6 次派发，每次重算 lm_head，prefill 约 13-18 秒。
- 输出上限 `--max-output` 默认 1024（实测约 7.9 tok/s）。
- **长上下文一旦触发上面的崩坏，服务会给出语法上合法但语义上是垃圾的回复**，不会报错。
  这是最危险的形态：客户端无法从响应看出后端已经退化。
