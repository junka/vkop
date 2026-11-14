# Anthropic-compatible HTTP server for vkop

让本机的 `claude` CLI 把 vkop 上的模型当作后端：

```bash
ANTHROPIC_BASE_URL=http://127.0.0.1:8787 claude
```

两层，边界是刻意划的：

| 层 | 文件 | 负责 |
|---|---|---|
| Python | `llm/server/anthropic_server.py` | HTTP 路由、Anthropic schema、SSE 帧、tool_use 解析、历史裁剪 |
| C++ | `llm/exporter/driver/llm_serve.cpp` | 加载 vkop 图、prefill/decode、KV cache、token 流 |

Python 通过两条继承的 fd 和 `llm_serve` 说一套行协议（见 `llm_serve.cpp` 顶部的注释）：请求是
`role` / base64 文本对，响应是 `DELTA` / `DONE`。选这套而不是在 C++ 里解 JSON，是因为
Anthropic 的 schema、SSE 和工具解析本来就要跟着上游变，改 Python 不用重编；而 base64
避开了所有转义与编码问题。

## 启动

```bash
make llm_serve                    # 一次
python3 llm/server/anthropic_server.py --port 8787
```

首次使用需要 `llm/exporter/embed_tokens.bin`（图里不含词嵌入表）：

```bash
python3 llm/server/embed_table.py     # 从 checkpoint 的 safetensors 抽出来，约 3 秒
```

模型文件本身要用当前构建的 shader 归档重转 —— 换了 kernel 就必须重转，运行期没有替代：

```bash
python3 -m onnx2vkop.cli -i llm/exporter/llm.onnx -o llm/exporter/llm.vkopbin \
  --extra-shaders TopK --target-backend buffer --master-archive build/vkop_shaders.bin
```

## 客户端怎么指过来

`~/.claude/settings.json` 里的 `env` 会覆盖 shell 里的同名变量，所以直接导出
`ANTHROPIC_BASE_URL` 往往不生效。用一个独立配置目录最稳：

```bash
mkdir -p /tmp/cccfg
cat > /tmp/cccfg/settings.json <<'EOF'
{"env": {"ANTHROPIC_BASE_URL": "http://127.0.0.1:8787",
         "ANTHROPIC_AUTH_TOKEN": "dummy",
         "ANTHROPIC_API_KEY": "dummy",
         "ANTHROPIC_MODEL": "vkop-qwen3-vl-2b"}}
EOF
CLAUDE_CONFIG_DIR=/tmp/cccfg claude
```

Claude Code 会提示 `doesn't recognize the model`，并假设 1m 上下文窗口 —— 那是它自己的
模型表里没有这个名字，不影响使用；真实的窗口由服务端的 `--max-kv` 决定（见下）。

## 参数

| 开关 | 默认 | 说明 |
|---|---|---|
| `--host` / `--port` | `127.0.0.1` / `8787` | |
| `--max-kv` | 16384 | KV 窗口行数。prompt + 生成长度必须装得下。Claude Code 的系统提示约 11k token，`llm_chat` 的 8192 不够。每行的显存代价是 `2*nkv*hd*2` 字节/层（fp16 缓存），fp8 缓存是它的一半，见「KV cache 精度」 |
| `--prefill-chunk` | 256 | prefill 每次派发的 token 数。两道墙：图里的 logits 是 `(1,q,vocab)`，8192 在 152k 词表上要 2.5 GB；以及这台驱动上单次派发占用 GPU 的工作量 —— `q=1024` 在覆盖行数超过约 6k 后会把 GPU 打掉（`VK_ERROR_DEVICE_LOST`，见「已知问题」）。256 是能跑完 Claude Code 那 11k token 提示的值；更大更快（q=1024 实测 46 tok/s vs 256 的 28），但只适合短 prompt |
| `--max-output` | 1024 | 单次回复的生成上限。客户端要的 `max_tokens` 更小时按客户端的来；更大时按这个截断 —— Claude Code 要 32000，在这块 GPU 上是 1.8 小时 |
| `--kv-kind` | auto | **只做校验，不选精度**。缓存的数据类型是制品的属性（转换器按图里的 past/present 声明选 attention kernel），所以这个参数改不了它。请求值和制品声明不一致时 `llm_serve` 直接报错退出 —— 以前它会把错尺寸的缓冲交给 kernel，静默读错缓存，4 bit 值配 fp16 图还会在建立 plan 时段错误。要 fp8 KV 就导出 fp8 制品（见下） |
| `--kv-cache` | 空 | 把 KV 前缀落到文件里，跨进程复用：新进程启动读它，首轮跑完写它。见「KV 前缀落盘」 |
| `--coop` | 关 | 打开 coopmat 选路（当前构建下它有已知的数值错误，见「已知问题」） |
| `--repetition-penalty` | 1.0 | 对本轮**已生成**的 token 做 HF 式重复惩罚（1.0＝关）。不碰提示 token。结构化输出别超过 1.1，见「协议细节」 |
| `--seed` | 0 | 采样 RNG 种子，每轮请求重置，保证同 prompt+同参数可复现。贪心时无作用 |

客户端的输出上限要和 `--max-output` 对齐，否则它会把「被截断」当成错误：

```bash
CLAUDE_CODE_MAX_OUTPUT_TOKENS=256 CLAUDE_CONFIG_DIR=/tmp/cccfg claude
```

窗口不够时服务端会先把 `max_tokens` 截到剩余空间，再按轮次从最旧的历史开始裁剪；
真正装不下的 prompt 才报 `context_overflow`。

## KV cache 精度（fp8 e4m3）

权重已经是 int8（1.72 GB），能再压的只有缓存：fp16 缓存每行
`2*nkv*hd*2*nlayers = 114688` 字节，16384 行就是 1.75 GiB；e4m3 每元素 1 字节，
同样的窗口只要 0.875 GiB。

```bash
# 1. 图改写：在 past/present 两侧插入 Quantize/DequantizeLinear，纯图操作，不碰权重
python3 llm/exporter/make_kv_variant.py -i llm/llm.onnx -o llm/llm_fp8.onnx --fp8
# 2. 照常转换（权重仍是 -q int8）
PYTHONPATH=model/pypi python3 -m onnx2vkop.cli -i llm/llm_fp8.onnx \
  -o llm/exporter/llm_fp8_kv.vkopbin -q int8 --extra-shaders TopK \
  --target-backend buffer --master-archive build/vkop_shaders.bin
# 3. 跑起来（--kv-kind 留 auto，制品自己声明 float8e4m3fn）
python3 llm/server/anthropic_server.py --model llm/exporter/llm_fp8_kv.vkopbin \
  --max-kv 16384 --prefill-chunk 256
```

不需要 `VK_EXT_shader_float8`：`shaders/buffer/attention.comp` 里的 `e4m3_encode` /
`e4m3_decode` 是手写位运算，本机（MoltenVK / Radeon Pro 560X，没有 fp8 硬件）就能跑。
代价是整个缓存共用**一个** per-tensor scale（`make_kv_variant.py --scale`，默认 0.1），
28 层 × K,V 都是它，可表示范围约 ±44.8，超出会被夹住。

实测（4 GB 卡，同一批探针）：fp16 制品在 `--max-kv 24576` 下每个请求立刻
`Failed to submit sem command buffer!`（缓存要 2.62 GiB），fp8 制品同一个窗口正常服务；
4k token 的长文检索、3 轮前缀复用会话、真实 11335 token 的 Claude Code 请求，
两边输出等价（后者都是 17 token、`end_turn`）。它不改变宽 prefill 的那堵墙 ——
墙是每派发的工作量，见「已知问题」。

## KV 前缀落盘（`--kv-cache`）

Claude Code 第一条消息里约 11k token 是系统提示 + 固定工具 schema，每轮都原样重发。
进程内已经能复用它们（见「协议细节」的 KV 前缀复用），但**重启就没了**：重新 prefill
那 11k 要 407 秒 —— 256 的 chunk 把 1.72 GB 权重整份重读 44 次，缓存字节不是瓶颈。
`--kv-cache <文件>` 把那段的 KV 存成文件，新进程几秒读回来：

```bash
python3 llm/server/anthropic_server.py --model llm/exporter/llm_fp8_kv.vkopbin \
  --max-kv 16384 --prefill-chunk 256 --kv-cache ~/vkop/kv_prefix_fp8.bin
```

一个文件两头用：启动时读（`llm_serve --kv-warm`），首轮跑完写（`--kv-dump`），退出时
如果这段前缀比文件里的更长再补写一次。之所以是「首轮 + 退出」而不是每轮：轮内复用本来
就走 `covered_ids_`，落盘只服务**未来**的进程，每轮搬 620 MB 只为多存几百行不值。
后端因 `VK_ERROR_DEVICE_LOST` 退出被 Python 重新拉起时，argv 里带着这两个参数，所以
重建进程直接从文件继续，不再付那 407 秒。

文件里存的是**渲染后的 token id 流**加它的 KV。id 流就是复用凭据：下一轮渲染出的 ids
与文件里的逐位比到第一个不同处，之前的行全部白拿，之后的重算（因果注意力 + 绝对位置
让「公共前缀之前的行」位置不变，所以这样合法）。因此一个过期的缓存只会让首轮变慢，
不会让答案出错 —— 前缀比对失败就整段重算。

文件头还带 `(制品路径, 制品字节数)`。id 只证明**文本**对得上，只有制品身份能证明这些
字节出自这张图，所以重转换或换制品后旧文件会在启动时被拒，stderr 一条
`--kv-warm ...: belongs to ...; re-dump it`，然后照常冷启动：宁可慢，不读另一张模型的
缓存。4-bit 缓存（int4/nf4）没接：一个字节两个值，行字节数不是「值数 × 1」。

**行距（pitch）**：缓存行在 buffer 里不是紧凑摆的。attention kernel 的行距取自 buffer
的**容量**（`ops/Attention.hpp` 的 `row_capacity`），也就是 `--max-kv`，这正是「追加行
只要 O(S)」的原因 —— `past_key_values_i` 的逻辑形状是 `[1,2,nkv,kv_len,hd]`，但它前面的
字节按 `[...,max_kv,...]` 铺。所以落盘要把 `2*nkv` 个 K/V 块按容量行距逐块收集
（`pack_layer_`），读回时按同样行距填回去（`unpack_layer_`）。第一版直接拷逻辑长度的
前缀，warm 输出立刻变成噪声（`esselėėė…`）—— 那是 head 0 的头几行，别的东西都在后面。

写盘走 `.tmp` + fsync + rename：中途崩溃（或写入/打包任何一步抛错，`.tmp` 由作用域守卫
删掉）不会留下一个看着完整、其实半截为 0 的文件被下次启动当真；旧缓存在这一步之前一直
还在原位。

实测（本机 560X，fp8 制品，`--max-kv 16384 --prefill-chunk 256`，8459 token 的合成
「稳定前缀 + 一个问题」）：

| | 冷启动 | 带 `--kv-cache` |
|---|---|---|
| 首条请求 | 235.7 秒（prefill 34 个派发） | **4.9 秒**（复用 8458 行，1 个派发） |
| 读文件 | | 463 MB / 9.4 秒，进程启动时 |
| 写文件 | 463 MB / 10.5 秒，首轮之后一次 | |

（上一轮同尺寸实测是 244 秒 / 6.9 秒 / 读 13.0 秒 / 写 15.0 秒；读写时间差在页缓存和盘
状态，量级没变。）

两点不漂亮但可接受：读回/上传按行距铺满整个 `[.., max_kv, ..]` 容器，所以真正的字节之外
还要多搬「还没算的那段窗口」的零（这里 940 MB 而非 463 MB）—— 把 `--max-kv` 收到贴近
前缀长度就基本没有零可搬；以及 fp8 缓存下 cold 和 warm 可能差一个首字母大小写。

大小写那件事的实测边界（239 token 的合成 prompt，同一问题、同一制品）：

- fp16 缓存：cold 与 warm 逐字符相同（另加三次连跑都一样）。
- fp8 缓存：这个尺寸上 warm 稳定给小写，连跑三次；同一 prompt 的 cold 无论
  `--prefill-chunk` 取 1 / 128 / 192 / 256 都给大写。**派发形状不是原因**（这条之前
  写错了，已改）。换成 8459 token 的 prompt，方向反过来（cold 小写、warm 大写）——
  所以它是个会往任一边翻的 argmax 近平手，不是 warm 的固定偏差。
- 复用后的发散场景仍然精确：C（cold 问另一个问题）和 D（从同一文件 warm 再问）输出
  逐字符相同，埋在 6 段文字里的「塔台 supervisor 是 Oona Halvorsen」也答对。

也就是说 warm 的注意力读到的历史行**语义上**和 cold 一致，但把这个前缀再落一次盘、拿
两次文件逐字节对比，只有每层 block 0（K 的 head 0）完全相同，head 1..7 和所有 V block
约 98% 的字节对不上。这半格差别从哪来还没定位，是这条路线上目前唯一的未解项；在此之前
不要把「文件逐位无损」当成前提，需要严格可复现就用 `--kv-kind fp16`。

## 协议细节

- **流式**：`stream: true` 走 chunked SSE，事件顺序是
  `message_start` → `content_block_start/delta/stop` → `message_delta` → `message_stop`。
  文本增量在 `text_delta`，工具参数在 `input_json_delta`。
- **工具调用**：模型输出 `<tool_call>{...}</tool_call>`。这两个标记在词表里是**特殊 token**
  （151657/151658），所以 `llm_serve` 解码时必须 `skip_special=false`，否则标记被过滤掉、
  只剩下一个裸 JSON，客户端就会当成正文。服务端把标记之外的内容转成 `text` block、
  里面的 JSON 转成 `tool_use` block，并把 `stop_reason` 置为 `tool_use`。
- **工具结果**：`tool_result` block 渲染成 `<tool_response>...</tool_response>`，连续几个
  合并进同一个 user 轮 —— 和 Qwen 的 chat template 一致。
- **工具定义**：转成 Qwen 的 `# Tools` 系统前缀，schema 按 OpenAI function 形式重排
  （Anthropic 发的是 `{name, description, input_schema}`）。
- **采样参数**：默认仍是纯 argmax，且与实现采样前逐位相同。客户端**显式**传了
  `temperature` / `top_k` / `top_p` 才生效（没传＝负值 sentinel＝用后端默认）；
  `temperature: 0` 等同贪心。全部在主机侧算：`argmax_last_token` 本来就把整行
  logits `copyToCPU` 回来扫描，所以不需要新 shader、不碰 GPU 路径。
  `top_k=1` 或 `top_p→0` 在任何 temperature 下都逐 token 等于贪心（最强的正确性判据）。
  RNG 每轮请求按 `--seed` 重置，所以同 prompt + 同参数必定同输出。
- **重复惩罚**：`--repetition-penalty`（默认 1.0＝关），HF 约定（logit>0 除以它，
  否则乘）。**只作用于本轮已生成的 token，绝不碰提示**——agent 的系统提示每轮都
  合法地重复工具名和 JSON 标点，惩罚它们会把工具调用压坏。开着它解码仍然是确定性的。
  实测边界：1.1 下工具调用的路径仍然正确；**1.5 会把 JSON 里反复出现的 `"` `:` `/`
  压歪，路径被幻觉成别的字符串**。结构化输出场景不要超过 1.1。
- **图片**：不接受（视觉塔是另一张图）。图片 block 会被替换成一行说明文字。
- **KV 前缀复用**：每轮请求都把整段历史重发。`llm_serve` 记住上一轮渲染出的 turn 列表和
  **真实 id 流**（prompt + 生成并喂回的 token + 模板的轮末串），下一轮只要这些 turn 原样
  还在，就把 id 流接上去，只编码新增的轮次。
  复用点取**最长公共前缀**，不是「整段 id 流是不是新请求的严格前缀」：新会话通常只共享
  那段稳定的开头（系统提示 + 工具 schema），第一个 user 轮就分叉 —— 因果注意力下分叉点
  之前的行位置不变，照样白拿，分叉点之后重算。跨进程的版本在「KV 前缀落盘」。
  为什么不能简单地重新分词：BBPE 的 decode→encode 不是恒等（词首空格标记在 decode 时
  丢失），重新编码 assistant 回复会得到与模型实际写下的不同的 id —— 逐 token 语义没错，
  但前缀比对每轮都会失败，于是每轮都要重算整个 11k token 的系统提示。`llm_chat` 用
  `Conversation` 存 token id 规避了这一点；无状态的 HTTP 服务只能自己存。

## 已知问题

- **宽 prefill 会把 GPU 打掉**（`VK_ERROR_DEVICE_LOST`）。同一个 fp8 制品、同一个
  16384 行窗口、同一个 11335 token 的 prompt：`--prefill-chunk 256` 用 407 秒跑完，
  `1024` 在第 113 秒让 `vkWaitSemaphores` 返回 -4。长度阶梯（q=1024、窗口 24576）
  把边界卡了出来：631 / 2395 / 5927 token 都正常（46–69 tok/s），约 9000 token 在
  第 115 秒挂掉。所以决定因素是**单次派发压在 GPU 上的工作量**（约 `q × 已覆盖行数`，
  实测 6.1M 通过、9.2M 失败），不是缓存字节数 —— fp8 把 KV 省掉一半，这堵墙一毫米
  也没挪。具体是哪个分配越界/过大还没查：`34fdca2` 修过同一形状的**静默**变体
  （SwiGLU down-proj fold 的 scratch 按行数线性增长、28 层同时持有 = 17 GB，加上
  分配失败被丢弃 → 读出全零 logits），这次驱动直接终止了 GPU。
  现在的行为：`llm_serve` 认出 -4 就带退出码 4 退出（一条错误比之后每个请求都 500
  诚实），`anthropic_server` 在下一次请求发现子进程已死会重新拉起并重载模型
  （热缓存约 15 秒，冷启动约 90 秒），不需要人工 `pkill`。
- **coopmat 在当前构建里算错**（`MatMulTest.MatMulGroupedBatchBuffer` 不经过模型也失败，
  prefill 一旦宽到会选中 coopmat 路由，模型输出就退化）。服务端因此默认设
  `VKOP_NO_COOP=1`；根因不在本服务里，修好后再用 `--coop` 打开。
- 速度（本机 Radeon Pro 560X、int8 权重）：decode 约 1 s/token，prefill 约 28 tok/s
  （chunk 256）。此前记的 200 ms/token 是 ARL iGPU 上的数。用 `--max-kv` 调窗口时注意：
  更大的窗口让 attention_bias 和 KV 都更长，decode 会更慢。

## 验证

```bash
python3 llm/server/smoke_rpc.py --max-new 8      # 只测行协议，不起 HTTP
curl -s localhost:8787/v1/messages -H 'content-type: application/json' -d '{
  "model":"vkop","max_tokens":16,
  "messages":[{"role":"user","content":"What is the capital of France?"}]}'
```
