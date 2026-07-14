# Qwen Tokenizer (C++)

Qwen3-VL 等基于 GPT-2 ByteLevel BPE 的 tokenizer 的 C++ 实现。从
HuggingFace `tokenizer.json` 离线编译成紧凑二进制（`.bin`），运行时 mmap
零拷贝加载，encode/decode 与 `transformers` 官方 tokenizer **逐 id 对齐**。

```
tokenizer.json  ──[tokenizer_to_bin.py]──▶  qwen3_vl.bin
                                                │
                              mmap + parse      ▼
                                         ┌─────────────┐
                                         │  Tokenizer  │  (tokenizer.cpp/hpp)
                                         └─────────────┘
                                          encode │ decode
                                          ┌──────┴──────┐
                                       ids ◀─┘         └─▶ text
```

## 组件

| 文件 | 作用 |
|---|---|
| [tokenizer_to_bin.py](tokenizer_to_bin.py) | 把 HF `tokenizer.json` 转成紧凑 `.bin`（含 chat template 探针提取）。 |
| [tokenizer.hpp](tokenizer.hpp) / [tokenizer.cpp](tokenizer.cpp) | C++ 加载器 + encode/decode/流式decode/chat template。 |
| [utf8.h](utf8.h) / [utf8.cpp](utf8.cpp) | UTF-8 流式 sanitizer（移植自 trt_edgellm），流式 decode 用。 |
| [tests/main.cpp](tests/main.cpp) | round-trip + HF 一致性 + 流式 + chat template + 性能测试。 |
| [tests/gen_ground_truth.py](tests/gen_ground_truth.py) | 用 HF `tokenizers` 生成 `hf_ground_truth.json` 逐 id 对比基准。 |
| [CMakeLists.txt](CMakeLists.txt) | 构建配置，依赖 `libutf8proc-dev`（pre-tokenizer 改手写扫描器，不再依赖 re2）。 |

## 构建

```bash
apt install libutf8proc-dev   # NFC 规范化 + \p{L}/\p{N}/\s 类别判定
cd llm/tokenizer && cmake -B build && cmake --build build -j
ctest --test-dir build --output-on-failure
```

## 生成 .bin

```bash
python3 tokenizer_to_bin.py
# 默认读 ~/.cache/modelscope/hub/models/Qwen/Qwen3-VL-2B-Instruct/tokenizer.json
# 产出 qwen3_vl.bin。改路径请编辑脚本顶部 MODEL_DIR / OUTPUT_BIN。
```

> 脚本里的 `~` 不会被 `open()` 自动展开，要么用绝对路径，要么 `os.path.expanduser` 包一下。

---

## .bin 文件格式

magic `QW3T` + 版本 + 三段平铺数组，全部小端序：

```
[Header]
  "QW3T"            4B   magic
  u32               version (=1)
  u32               vocab_size
  u32               merge_count
  u32               special_count
  u32               reserved

[Vocab Section]     × vocab_size
  u16 len           token 字节数
  bytes             token（UTF-8）
  u32               token_id

[Merge Rules Section]  × merge_count
  u32 left_id       合并左 token id
  u32 right_id      合并右 token id
  u32 merged_id     合并后 token id
  // 无效规则占 (0,0,0)，加载时跳过

[Special Tokens Section]  × special_count
  u16 len           token 字节数
  bytes             token（如 "<|im_start|>"）
  u32               token_id

[ChatTemplate Section]   （可选，追加在文件末尾；旧 bin 无此段则 chat template 禁用）
  u32 role_count
  × role_count:
    u16 len  role_name    "system"/"user"/"assistant"
    bytes    role_name
    u16 len  prefix       如 "<|im_start|>system\n"
    bytes    prefix
    u16 len  suffix       如 "<|im_end|>\n"
    bytes    suffix
  u32 content_type_count
  × content_type_count:
    u16 len  type_name    "image"/"video"
    bytes    type_name
    u16 len  format       如 "<|vision_start|><|image_pad|><|vision_end|>"
    bytes    format
  u16 len  generation_prompt
  bytes    generation_prompt
  u16 len  default_system_prompt
  bytes    default_system_prompt
```

ChatTemplate Section 由 `tokenizer_to_bin.py` 用 HF `apply_chat_template`
探针提取（roles prefix/suffix、image/video 占位格式、generation_prompt、
default_system_prompt），序列化为长度前缀字段，C++ `load` 顺带解析，零 JSON 依赖。

格式刻意保持简单平铺：均匀定长记录数组，不需要 schema/代码生成。加载端
直接 `reinterpret_cast` 顺序读，mmap 后的内存只是中转，parse 完就建成哈希表。

---

## HF pipeline 各 stage 的作用与本实现的处理

HuggingFace `tokenizers` 的完整 pipeline 是：

```
encode:  text → [Normalizer] → [PreTokenizer] → [Model: BPE] → [PostProcessor] → ids
decode:  ids → [Model: decode] → [Decoder] → text
```

Qwen3-VL 的 `tokenizer.json` 里四个 stage 配置如下，逐个说明：

### 1. Normalizer: `NFC`
Unicode 规范化（组合字符归一，如 `é` 与 `é` 统一）。**本实现未做**——
多数实际输入已是 NFC，影响极小。若需要可引入 `utf8proc`（轻量，自带 NFC）。

### 2. PreTokenizer: `Sequence[Split(Regex), ByteLevel]`
这是和 HF 对齐的**关键**。两步：

- **Split(Regex)**：用 GPT-2 正则把文本切成 spans（字母串 / 数字 / 标点 /
  换行等），每段**独立**做 BPE，不跨段合并。正则（来自 tokenizer.json）：
  ```
  (?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
  ```
  **手写确定性扫描器**实现（不再用 re2）：按 alt1..alt7 顺序在 codepoint 流上
  匹配，精确复刻 HF 行为，**含 re2 不支持的 `\s+(?!\S)` 负向前瞻语义**——通过
  回溯规则实现：trailing 空白串末尾若紧跟非空白，留最后一个空白给后续 alt 吸附。
  `\p{L}`/`\p{N}`/`\s` 用 `utf8proc_category` 判定。已用 HF `tokenizers` 逐 id
  验证（66 条边界 + 210 条随机语料全等）。
- **ByteLevel**：每段内字节→BBPE 字符串映射（见下「字节映射」）。`use_regex=false`
  表示正则交给上一步，本步只做字节映射。

**为什么关键**：不分段、整段塞一个序列做 BPE，会跨词边界合并（如 `dog.`
可能合并出 `g.`），导致 token 序列和官方不一致，影响推理质量与 token 数对齐。

### 3. Model: BPE
vocab（token↔id）+ merges（合并优先级表）。本实现：
- encode：每段字节先逐个映射成 BBPE token id，再在该段内做标准 BPE 合并
  （每轮选 rank 最小的相邻 pair 合并，rank = merge 在文件中的顺序）。
- decode：token 字符串拼起来，按 codepoint 还原回字节。

### 4. PostProcessor: `ByteLevel(add_prefix_space=false, trim_offsets=false, use_regex=false)`
对 **token id 序列是 no-op**。HF 里 ByteLevel post-processor 只做两件事：给
首个 token 加前导空格（`add_prefix_space=true` 时）、计算 offset 时 trim
（`trim_offsets=true` 时）。这里两个 flag 都 false → 不改 id 序列。
**C++ 端无需实现。**

### 5. Decoder: `ByteLevel(...)`
反向 ByteLevel：把 BBPE 字符串的 codepoint 还原回原始字节。**本实现已做**
（`decode()` 里的 `cp_to_byte` 映射，等价于 `bytes_to_unicode` 的逆）。

### Special Tokens（`added_tokens`，如 `<|im_start|>`）
不在 `model.vocab`，id 紧接 vocab 区间（Qwen3-VL: vocab 0..151642，special
151643..151668）。处理顺序：**在 pre_tokenizer 之前**做最长字面量匹配——
命中即吐单个 id，该段不进 BPE；其余文本才走正则切分。这模仿 HF 行为：特殊
token 作为原子单元，不被拆成 `<` `|` `im` ... 普通字节。

---

## 字节映射（ByteLevel 核心）

GPT-2 把 256 个字节映射到可见 Unicode 码点，避免 vocab 里出现控制字符：

- 188 个可见字节（`!`~`~`、`¡`~`¬`、`®`~`ÿ`）映射到自身码点（单字节 UTF-8）。
- 68 个不可见/控制字节映射到 U+0100..U+017F（UTF-8 为 `0xC4 0x80`~`0xC4 0xBF`，两字节）。

因此 vocab 里的 token 字符串是这些 codepoint 的 UTF-8 编码。例如空格
(0x20) → `Ġ` (U+0120, UTF-8 `0xC4 0xA0`)。`tokenizer.cpp` 里的
`ByteUnicodeMap` 一次性建好双向表：`byte_to_str`（encode 用）、`cp_to_byte`
（decode 用）。


---

## 流式 decode（增量输出）

服务端逐 token 生成时，单 token 的 piece 字节可能不是合法 UTF-8（多字节
codepoint 跨 token 边界、或模型吐出孤立字节）。`emit_delta` 增量解码：

```cpp
qwen::StreamState st;                 // 每个生成 slot 一个
std::vector<uint32_t> ids = ...;      // 逐 token 追加
std::string delta = tok.emit_delta(st, ids, /*skip_special=*/true);
// delta 恒为 well-formed UTF-8；尾部不完整 codepoint 暂存于 st.pending_bytes
// 下一轮自动补齐。流结束时 tok.emit_delta_flush(st) 把残留字节转成 U+FFFD。
```

实现：`id_to_piece` 取原始 piece 字节 → `utf8::sanitizeUtf8Streaming`
（[utf8.cpp](utf8.cpp)）扫描，非法序列→U+FFFD，末尾不完整 codepoint→存
`pending_bytes`。移植自 trt_edgellm 的 `common/utf8`。

---

## Chat template

`apply_chat_template` 把多轮对话渲染成 Qwen3-VL 的 ChatML prompt：

```cpp
std::vector<qwen::ChatMessage> msgs = {
    {"system", {{"text", "你是助手。"}}},
    {"user",   {{"text", "你好"}}},
};
std::string prompt = tok.apply_chat_template(msgs, /*add_generation_prompt=*/true);
// = "<|im_start|>system\n你是助手。<|im_end|>\n<|im_start|>user\n你好<|im_end|>\n<|im_start|>assistant\n"
auto ids = tok.encode(prompt);   // 再编码成 token
```

模板数据（roles 的 prefix/suffix、image/video 占位格式、generation_prompt、
default_system_prompt）由 `tokenizer_to_bin.py` 用 HF `apply_chat_template`
探针提取，内嵌进 `qwen3_vl.bin` 的 ChatTemplate Section，`load` 顺带解析，
零 JSON 依赖。多模态内容项 `{"image",""}` / `{"video",""}` 会展开成
`<|vision_start|><|image_pad|><|vision_end|>` 等占位串。
