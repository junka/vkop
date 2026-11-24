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

### 用配置文件代替一串导出变量

一套模型配置原本是五六个 `VKOP_*` 环境变量加命令行开关。`--config` 把它收进一个
JSON，`--preset` 在同一份文件里选一套：

```json
{"presets": {
   "b3": {"model": "data/llm.vkopbin", "embed": "data/embed_tokens.bin",
          "tokenizer": "data/qwen3_vl.bin", "bin": "src/build/llm_serve",
          "max-kv": 16384, "prefill-chunk": 256, "max-output": 512,
          "kv-cache": "kv_prefix.bin", "port": 8787}},
 "default": "b3"}
```

```bash
python3 llm/server/anthropic_server.py --config ~/vkop_serve/serve.json
```

仓库里 `llm/server/presets.json` 是可直接用的例子（2B 的 fp16 / fp8-KV / 短 prompt 三份）。

- 键名就是长选项名（`max-kv`，下划线也接受）。相对路径**按配置文件所在目录解析**，
  不是 cwd —— 配置跟着制品走，搬目录不会失效。
- 文件里没有 `presets` 键时，整份文件当作单一预设（一个文件一套模型时不用编名字）。
- 拼错的键直接报错并列出已知键，不会静默忽略；`--coop` 这类布尔值必须是 JSON 真
  `true`/`false`（`bool("false")` 是 True，会悄悄打开 coopmat 那条有已知数值错误的
  路由，所以字符串只认明确的白名单）。
- 优先级是**命令行 > preset > 环境变量 > 内置默认**，所以命令行上写的仍然赢。preset
  压掉了一个已导出的环境变量时会打印 `[serve] preset overrode env: ...`。
- JSON 没有注释语法，所以 `_comment`（以及 `_comment_1` 等）是被忽略的键，例子里就用它
  逐条说明。规则只认这一种拼法：`_max-kv` 这种带下划线的拼错照样报错 —— 静默丢一个
  拼错的键正是上面那条检查要防的事。
- 制品其实不存在时报 `missing: <path>` 并退出，所以预设可以先写好、制品后转；
  例子里 `qwen3-vl-2b-fp8kv` 就是这样，它要求额外的 fp8-KV 制品。

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

Claude Code 不认识这个模型名，会假设 1m 上下文窗口，于是永远不主动压缩历史，超窗只能靠
服务端裁剪。服务端启动时会打印真实窗口和对应的客户端变量：

```
[serve] --max-kv auto: 57344 rows (model context 262144, budget fits 58161, 6.58 GB of cache)
[serve] KV window 57344 tokens -- start the client with CLAUDE_CODE_MAX_CONTEXT_TOKENS=57344
```

把它放进同一个 `settings.json` 的 `env`（`"CLAUDE_CODE_MAX_CONTEXT_TOKENS": "57344"`），
Claude Code 就按这个窗口做自动压缩。

## 参数

| 开关 | 默认 | 说明 |
|---|---|---|
| `--host` / `--port` | `127.0.0.1` / `8787` | |
| `--max-kv` | auto | KV 窗口行数，prompt + 生成长度必须装得下。`auto`＝模型原生上下文（`--context-length`）与显存预算取小，按 1024 行取整，`llm_serve` 加载完模型后算出来，经 `READY 3 <行数>` 报回 HTTP 层做裁剪。显式给的行数超过 `--context-length` 会被夹到它 —— 再往后的位置模型没训练过。每行的显存代价是 `2*nkv*hd*2` 字节/层（fp16 缓存），fp8 缓存是它的一半，见「KV cache 精度」 |
| `--context-length` | 0 | 模型原生上下文，即 HF config 的 `max_position_embeddings`（Qwen3-VL-2B 是 262144，Phi-4-mini 131072，GLM-Edge 8192）。0＝未知：`auto` 只按显存定，也不夹显式值 |
| `--kv-budget-mb` | 0 | `--max-kv auto` 时 KV 缓存可占的显存。0＝加载模型后设备堆剩余预算（`VK_EXT_memory_budget`）的 25%，没有该扩展时取堆大小的 30%。本机（M5 Max，36 GB 统一内存）得 6.6 GB / 57344 行；取 40% 时是 10.6 GB / 92160 行，系统空闲内存掉到 19%，这台机器曾在这个水位 OOM 重启过，所以默认取得保守 |
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

读回/上传现在只搬活跃行：每层 `2*nkv` 段各一次 ranged `vkCmdCopyBuffer`，经一块映射的
staging 直接读写文件（`transfer_rows_`）。此前是按行距把整个 `[.., max_kv, ..]` 容器读回 /
上传，代价随 `--max-kv` 而不是前缀长度增长 —— 22 行的前缀写盘要 4 秒，窗口改成按模型上下文
定（几万行）以后会到每次 ~19 GB。实测（M5 Max，fp16，`--max-kv auto`＝57344 行）：

| | 之前（16384 行窗口） | 现在（57344 行窗口） |
|---|---|---|
| 写 22 行 | 4.0 秒 | — |
| 写 ~12k 行 | 13594 行 / 1487 MB / 7.1 秒 | 11852 行 / 1296 MB / 1.0 秒 |
| 写 32.9k 行 | 装不下 | 3598 MB / 2.1 秒 |
| 读 32.9k 行 | 装不下 | 3598 MB / 5.7 秒，首请求 TTFT 31.5 秒 → 0.74 秒，答案相同 |

另外 fp8 缓存下 cold 和 warm 可能差一个首字母大小写。

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

## 指标：token 数、延迟、GPU 时间

四层统计，各自回答不同的问题。**不要拿错层**：

| 你要问的 | 看哪里 | 精度 |
|---|---|---|
| 这个请求多少 token、多慢、首字多久 | `[req]` 行（stderr） | 最准；token 数是后端 tokenizer 的真实计数 |
| 上面这些的聚合与历史，可由程序查询 | `GET /metrics` | 与 `[req]` 同源，进程内保留最近 256 条 |
| prompt 单独多少 token | `POST /v1/messages/count_tokens` | 真实计数（不是估算） |
| 客户端视角 | 响应里的 `usage` | `output_tokens` 含未消费的 stop token |
| prefill / decode 分相耗时 | `[serve] prefill ...` / `VKOP_SERVE_VERBOSE=1` 的 `r<N>` 行 | 见下面两个坑 |
| GPU 按算子 | `VKOP_SUBMIT_PROF=1` | 每轮打印一次 op-type 表 + 最慢 op |

### `GET /metrics`：TTFT / TPOT 的查询接口

`[req]` 是给人看的日志，`/metrics` 是给程序和终端查询用的同一批数据。默认端口上：

```bash
curl -s localhost:8787/metrics | python3 -m json.tool
curl -s 'localhost:8787/metrics?limit=5'      # recent 只看最近 5 条；limit=0 = 整个窗口
```

返回（节选）：

```json
{"uptime_s": 16.8, "completed_since_start": 2, "window": 2,
 "all":    {"requests": 2, "errors": 0, "input_tokens": 27, "output_tokens": 20,
            "ttft_s": {"n": 2, "min": 0.2713, "p50": 0.2713, "p90": 0.3109,
                       "max": 0.3109, "mean": 0.2911},
            "tpot_s": {"n": 2, "min": 0.0823, ..., "mean": 0.0837},
            "wall_s": {"n": 2, ...},
            "decode_tokens": 18, "decode_tokens_per_s": 11.9},
 "recent": { ... 同结构，只覆盖最近 limit 条 ... },
 "requests": [{"id": "msg_...", "stream": false, "input_tokens": 15,
               "output_tokens": 8, "wall_s": 0.8477, "ttft_s": 0.2713,
               "tpot_s": 0.0823, "decode_tokens": 7, "stop_reason": "end_turn",
               "trimmed": false, "at": 1794816567.8}, ...]}
```

**TTFT 和 TPOT 是两个不能混的量：**

| | 定义 | 变动因素 |
|---|---|---|
| `ttft_s` | 请求到达到第一个 token | 整个 prefill：prompt 长度、chunk 数、KV 前缀是否命中 |
| `tpot_s` | **首个 token 之后**每个 token 的平均耗时 | decode 速度：模型大小、带宽、batch |

首个 token 是 prefill 的产物，所以从 TPOT 的分母里扣掉（`decode_tokens = out - 1`）。
把它算进 TPOT，TPOT 就变成「第二个 TTFT」，长 prompt 会显得 decode 也慢。
`out=1` 时没有 decode 步可平均，`tpot_s` 为 `null` 而不是 0。失败请求也占一条，
计入 `errors`，但不进 `ttft_s`/`tpot_s` 的统计（`ttft_s` 为 `null`）。

`decode_tokens_per_s` 是**按 token 加权**的总速率（Σtoken / Σdecode 秒），不是各请求
TPOT 的平均——一条 100 token 的回复和一条 1 token 的回复不该等权。

窗口是进程内最近 256 条，进程重启即清零；`completed_since_start` 是累计值。要跨重启
留存就把 `[req]` 行落到文件里另存。

`[req]` 是每个请求结束（或失败）时写的一行，直接回答「这个请求花了多少」：

```
[req] msg_1584a3db... sync   in=15 out=8  wall=0.933s ttft=0.353s decode=82.9ms/tok stop=end_turn trim=no
[req] msg_cd1e0ecb... stream in=12 out=12 wall=1.185s ttft=0.263s decode=83.8ms/tok stop=end_turn trim=no
[req!] msg_3c238e34... sync  in=4009 out=0 wall=0.020s ttft=0.000s stop=error trim=no error=context_overflow: ...
```

（前两行是 560X 本机、`--max-kv 1024 --prefill-chunk 128` 实测；末行是被窗口拒掉的
请求，`in=` 仍报出了真实计数。）

- `wall` 是 HTTP 层从收到请求到回复写完；`ttft` 到第一个 `delta` 为止（prefill + 首个
  采样）；`decode` 是 `(n_out-1)` 个 decode step 的平均值（第一个 token 算 prefill 的
  产物，所以从分母里扣掉），`out<=1` 时不打印。
- 失败请求打 `[req!]` 并带 `error=`，同样占一行，不会被静默吞掉。

### 两个坑

**`[serve] prefill ...` 两个毫秒数，别拿错。** 这行现在是：

```
[serve] prefill L=10518 (reuse 0, 42 chunk(s)) 87936ms total, last chunk 600ms -> 15011
```

`total` 是从进 prefill 到第一个 token 采完的整段墙钟（含所有 chunk 的 host 侧
embed/bias 填充与上传），`last chunk` 是最后一个 chunk 的 `Run()`。两者差得很开：
上面这条实测 146 倍（560X 本机，42 个 chunk）。此前这行只打 `last chunk`，读日志的
人会以为 11k prompt 的 prefill 只要 0.6 秒。

跨 chunk 的成本随上下文增长：`last chunk` 变大是 attention 变长，`total` 大是 chunk
数量多。想快就加大 `--prefill-chunk`（受另外两道墙限制，见参数表），或者用
`--kv-cache` 把稳定前缀留下来。

**`[ksel]` 只在 `VKOP_KERNEL_DEBUG=1` 时输出。** 它是 kernel 选路的逐规则 trace，
每个 MatMul 每次派发都会走一遍；142 个 MatMul × ~20 条规则 = 单个 decode step
两千多行。曾经无条件打印，`/tmp/serve.log` 涨到 200 万行（64 个请求），真正的
`[serve]`/`[http]` 行全被埋掉。要看选路就显式开：

```bash
VKOP_KERNEL_DEBUG=1 python3 llm/server/anthropic_server.py ... 2>&1 | grep '\[ksel\]'
```

### GPU 侧

`VKOP_SUBMIT_PROF=1` 给每轮一张按 op-type 汇总的 GPU 时间表（`--prefill-chunk` 的每个
chunk 是一轮，decode 每步一轮），`VKOP_SUBMITPROF_TOPN=N` 控制最慢算子列表长度
（默认 15；`0` = 全部，截断会少报总量）：

```
[submitprof] op-type       calls   gpu_ms   avg_ms
[submitprof] MatMul            142    290.19    2.044     ← prefill chunk
[submitprof] Attention          28      1.26    0.045
[submitprof] TOTAL             593    298.08
```

同一进程里 decode 一步的 MatMul 合计约 11.9ms、TOTAL 约 12.6ms。它和 `[req]` 的
`decode=21ms/tok` 之差就是 host 侧开销（record + logits 回读做 argmax），不是 GPU 空转。
`VKOP_OPPROF=1` 是 host 侧的对应物，另外还打 `[rbprof]` 的同步回读计数。

## 协议细节

- **流式**：`stream: true` 走 chunked SSE，事件顺序是
  `message_start` → `content_block_start/delta/stop` → `message_delta` → `message_stop`。
  文本增量在 `text_delta`，工具参数在 `input_json_delta`。
- **工具调用**：模型输出 `<tool_call>{...}</tool_call>`。这两个标记在词表里是**特殊 token**
  （151657/151658），所以 `llm_serve` 解码时必须 `skip_special=false`，否则标记被过滤掉、
  只剩下一个裸 JSON，客户端就会当成正文。服务端把标记之外的内容转成 `text` block、
  里面的 JSON 转成 `tool_use` block，并把 `stop_reason` 置为 `tool_use`。
- **其它特殊 token**：`skip_special=false` 也让所有控制 token 原样进了正文 —— 实测 2B 模型
  在 Claude Code 会话里写出过一个内容为 `<|im_start|>user\n<|im_start|>assistant\n` 的
  text block。现在两条规则：轮边界（`<|im_start|>`、`<|endoftext|>`、`<tool_response>`、
  Phi 的 `<|user|>`/`<|system|>`）和模板自带的停止符一样结束本轮 —— 写新轮头或自己编工具
  结果，都说明这一轮已经说完了；其余特殊 token 只有 `<tool_call>`、`</tool_call>`、`<think>`、
  `</think>` 进正文，别的一律丢掉（`InferenceServer::emits_text`）。
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
  （chunk 256）。此前记的 200 ms/token 是 ARL iGPU 上的数。
- 速度（M5 Max，fp16 权重，`--max-kv auto`＝57344 行）：短上下文 decode 14 ms/token，
  12k 上下文 20 ms/token（Claude Code 的会话），33k 上下文 27 ms/token；prefill 约
  1750 tok/s（11.7k）/ 1050 tok/s（32.9k）。decode 时间跟着**已用**行数走，不跟窗口大小走：
  attention 只扫已用的行（见下一条）。
- **decode attention 按 KV 切片**（`attention.comp -DKV_SPLIT`，`AttentionBuffer::split_route`）。
  原来的 kernel 一个查询行一个 workgroup，decode 一步只有 16 个（GQA 16 头），每个串行扫完
  整段历史：12k 上下文时 Attention 占一步 41 ms 里的 29 ms（1.05 ms/层，约为 KV 带宽下限的
  10 倍）。现在每行的 KV 切成若干段整 tile，一个 workgroup 一段（目标 512 个 workgroup，
  `VKOP_ATTN_SPLIT_WGS` 可调），写出 (m, l, o) 局部量，第二次派发按行合并；cache 是原地
  更新时，拷贝那一半也只派发新行（原来每层要发 ~116k 个立刻返回的空 workgroup）。14.5k
  上下文 decode 49.9 → 20.7 ms/token，短上下文 14.1 → 14.2（不变），输出逐 token 相同。
  `VKOP_ATTN_SPLIT=0` 退回原 kernel 做 A/B；旧制品里没有这两个 shader key，会打印一行提示
  并沿用原 kernel，要重转才能用上。

## 验证

```bash
python3 llm/server/smoke_rpc.py --max-new 8      # 只测行协议，不起 HTTP
curl -s localhost:8787/v1/messages -H 'content-type: application/json' -d '{
  "model":"vkop","max_tokens":16,
  "messages":[{"role":"user","content":"What is the capital of France?"}]}'
```
