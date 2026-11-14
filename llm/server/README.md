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
| `--coop` | 关 | 打开 coopmat 选路（当前构建下它有已知的数值错误，见「已知问题」） |

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
- **采样参数**：`temperature` / `top_p` / `top_k` 被忽略，解码是纯 argmax。要采样得先在
  `llm_serve` 里实现。
- **图片**：不接受（视觉塔是另一张图）。图片 block 会被替换成一行说明文字。
- **KV 前缀复用**：每轮请求都把整段历史重发。`llm_serve` 记住上一轮渲染出的 turn 列表和
  **真实 id 流**（prompt + 生成并喂回的 token + 模板的轮末串），下一轮只要这些 turn 原样
  还在，就把 id 流接上去，只编码新增的轮次。
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
