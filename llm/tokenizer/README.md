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
| [tokenizer_to_bin.py](tokenizer_to_bin.py) | 把 HF `tokenizer.json` 转成紧凑 `.bin`。 |
| [tokenizer.hpp](tokenizer.hpp) / [tokenizer.cpp](tokenizer.cpp) | C++ 加载器 + encode/decode。 |
| [tests/main.cpp](tests/main.cpp) | round-trip + 性能测试。 |
| [CMakeLists.txt](CMakeLists.txt) | 构建配置，依赖 `libre2-dev`。 |

## 构建

```bash
apt install libre2-dev          # re2 用于 pre_tokenizer 正则切分
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
```

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
  C++ 用 [re2](https://github.com/google/re2) 编译（支持 `\p{L}`/`\p{N}`
  Unicode 属性类）。re2 不支持负向前瞻 `(?!...)`，故去掉 `\s+(?!\S)` 子句、
  保留兜底 `\s+`——对常规输入切分结果与 HF 一致（已用官方 tokenizer 逐 id 验证）。
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

