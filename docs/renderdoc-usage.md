# RenderDoc 使用手册（vkop 项目）

> 适用环境：Intel ARL iGPU + Ubuntu 24.04 + RenderDoc 1.47（2026-11-04 安装）。
> 所有结论均在 1.47 stable 与 nightly 2026_11_03(282b28b3) 上实测/源码验证。

## 1. 安装与目录

RenderDoc 无 apt/snap/flatpak 包，用官方 tarball 装在用户目录：

```
/home/gwm/opt/renderdoc/
├── bin/qrenderdoc        # GUI
├── bin/renderdoccmd      # CLI（capture / replay 控制端）
├── lib/librenderdoc.so   # 捕获层本体
└── share/renderdoc/      # pylibs（只有标准库，命令行无 renderdoc 模块）
```

### 本机特有坑：`/etc/ld.so.preload`

系统 preload 了 `libAppProtection.so`（26.4.0.5），它会拦截 dlsym/dlopen 并无限递归
（gdb 见 26 万帧），把 `renderdoccmd`、`qrenderdoc --help` 打崩。跑任何 RenderDoc
可执行文件都带：

```bash
LD_PRELOAD=/home/gwm/opt/renderdoc/lib/librenderdoc.so
```

## 2. Vulkan 层配置（隐式注入）

用户级 implicit layer manifest（官方包自带的 library_path 是坏的，必须重写）：

```
~/.local/share/vulkan/implicit_layer.d/renderdoc_capture.json
```

```json
{
  "file_format_version": "1.0.0",
  "layer": {
    "name": "VK_LAYER_RENDERDOC_Capture",
    "type": "GLOBAL",
    "library_path": "/home/gwm/opt/renderdoc/lib/librenderdoc.so",
    "api_version": "1.3.274",
    "implementation_version": "1",
    "description": "RenderDoc capture layer",
    "pre_instance_functions": ["vkGetInstanceProcAddr"]
  }
}
```

- **启用采集**：`ENABLE_VULKAN_RENDERDOC_CAPTURE=1`
- **`VK_LAYER_PATH` 不等于注入**：它只做显式层发现，`pre_instance_functions` 被忽略、
  层实际不进链。测试时曾因此得出假阴性结论——确认注入的标志是 vulkaninfo 里出现
  RenderDoc 的 "multiple simultaneous logical devices" 错误，或 `/tmp/RenderDoc/` 产生捕获。
- 产物：`/tmp/RenderDoc/<app>_<日期>_<时间>.rdc`（llm_driver 6 轮 decode ≈ 3.4 GB）

## 3. vkop 侧开关

| 环境变量 | 作用 |
|---|---|
| `VKOP_RDOC=1` | 打开应用内捕获 + `VKOP_DEBUG_LABELS`（debug label 注入） |
| `VKOP_RDOC_RUN=<n>` | 只捕获第 n 轮推理（0 基）；配合 `Runtime::Run()` 里的 `RenderdocRunCapture` |
| `VKOP_DEBUG_LABELS=1` | 只开 label（`vkCmdInsertDebugUtilsLabelEXT` / begin/end region）不采集 |
| `VKOP_RDOC_COOPMAT_ENUM=1` | 强制 coopmat 枚举（测新版层是否已修复时用） |
| `VKOP_NO_HOST_IMAGE_COPY=1` | 绕开 `vkCopyMemoryToImageEXT`（host 裸指针无法被捕获快照） |

### coopmat 崩溃与防护（上游未修）

**症状**：层下设备创建即 `SIGSEGV #0 0x0`。

**根因**（源码定位，1.47 + nightly 均如此）：

- `extension_support.md:304` — `VK_KHR_cooperative_matrix` 在 Unsupported 列表；
  `vk_hookset_defs.h` 里 `CooperativeMatrix` 0 引用 → 层没为
  `vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR` 安装设备入口点；
- 该函数在白名单里但没实现，`vk_layer.cpp:630` 的兜底逻辑按 spec 直通下一层原始指针，
  绕过了 loader trampoline 的合法性检查 → 分发表空槽 → 跳 0x0。
- NV 家族（`vk_hookset_defs.h` 中 0 引用）与 KHR 结构体序列化
  （`vk_serialise.cpp:2209` 起 `PNEXT_UNSUPPORTED`）同样缺失。
- SPIR-V 半支持：能 capture/反汇编 coopmat shader，但 debugger 的 capability 检查
  （`spirv_debug_setup.cpp:874`）把 `CooperativeMatrixKHR` 归入 `supported = false`，不能单步。

**vkop 防护**：`vulkan/VulkanDevice.cpp` 的 `getProperties()` 里，检测到
`ENABLE_VULKAN_RENDERDOC_CAPTURE` 且未设 `VKOP_RDOC_COOPMAT_ENUM` 时跳过 coopmat
combo 枚举。降级安全：`coopmatProps_` 只喂 `supports_coopmat_sint8_tile()` /
`supports_coopmat_fp16_tile()`，跳过即整体退回 buffer kernel 路由，只是慢、不会错。
`VKOP_DISABLE_COOPMAT_RDOC_GUARD` 可编译期去掉该防护。

## 4. 采集流程（LLM 推理示例）

```bash
cd /home/gwm/code/github/vkop/build
ENABLE_VULKAN_RENDERDOC_CAPTURE=1 VKOP_RDOC=1 VKOP_RDOC_RUN=2 \
LD_PRELOAD=/home/gwm/opt/renderdoc/lib/librenderdoc.so \
  ./llm_driver /tmp/audit_llm.vkopbin -p ... 2>&1 | tee /tmp/rd_run.log
```

- 不设 `VKOP_RDOC_RUN` 时抓第一轮；`VKOP_RDOC_RUN=2` 抓 round 2（跳过 warmup）。
- shader 改过之后必须重转 `.vkopbin`/shader archive，否则层下跑的是旧 kernel。
- 捕获是整轮 submit 级别的，包含 GPU 真实执行时间戳。

## 5. UI 分析要点

### Event Browser

- **Duration (µs) 列**：replay 时插入 `vkCmdWriteTimestamp` 测得的 GPU 执行时间，
  真实非零（q_proj/MatMul 216 µs 量级）。点列头可排序 → top kernel 排行。
- **时间过滤语法**（藏在过滤框里，源码 `EventBrowser.cpp:2555`）：

  ```
  duration > 100us
  ```

  运算符 `< > <= >=`，单位 `s/ms/us/ns`。一轮 decode 过滤后剩下的基本全是 MatMul 大头。
- 行名 `q_proj/MatMul` 来自 vkop 的 insert-label（见 §8 的 fake-marker 说明）。

### Timeline

- 横轴是 **EID（事件序号），不是时间**。源码 `TimelineBar.cpp:290`：
  `m_eidWidth = virtualSize / (maxEID + 1)`，每个事件均分宽度，无时间缩放模式，也没有
  切换选项。条短是缩放效应，空隙是 barrier/label/bind 这类无 GPU dispatch 的事件。
- 用途是导航（按事件顺序定位），不是性能展示。

### Performance Counter Viewer

- `GPUCounter.EventGPUDuration`（时间戳）+ `CSInvocations` 等通用 counter 在 Intel 上
  可用（日志 "Intel GPU detected - no counters available" 指的是 IHV 专属 counter）。
- GPU Duration + CS Invocations 组合可做 per-dispatch 吞吐分析。

### 时间数字的正确性验证

ARL `timestampPeriod = 52.08 ns`。抽查：216.14583 µs ÷ 0.05208 = 4150 ticks，
153.125 ÷ 0.05208 = 2940 —— 全是整数，说明 Duration 确为原始 timestamp 换算。

### 解读时的两个坑

1. **eid ≠ 时间**：Timeline 条的宽度与 Duration 无关。
2. **replay 时间 ≠ 线上时间**：同一份 rdc 反复测，总时间 93 → 137 → 176 ms，
   同一 EID 三次 9.0/9.4/14.8 ms。RenderDoc 逐-op 测量是隔离重放（无流水线重叠、
   GPU 时钟不受控），绝对值不可与 vkop 自测时间戳对账；相对排名稳定。
   与 vkop 自己的 min-of-interleaved A/B 方法论互为印证。

## 6. 查看 tensor 内容与 CS 调试

vkop 的 guard 降级让捕获里全是 buffer kernel，所以**每个中间 tensor 都是一个普通
SSBO，可看可调**——§3 的 coopmat 限制不影响这条路径。

### UI：Buffer Viewer 看中间 tensor

1. Event Browser 选中目标 dispatch（如 `lm_head/MatMul` 的 EID）
2. Pipeline State Viewer → Compute pipeline 面板，列出该 dispatch 绑定的
   descriptor set（vkop 每 op 一组绑定，全为 storage buffer）；双击对应行打开
   Buffer Viewer（真实 descriptor set 走 `ViewDescriptorStore`，descriptor buffer 走
   `ViewBuffer`，见 `VulkanPipelineStateViewer.cpp:3363`）
3. Buffer Viewer 按 shader reflection 的声明布局显示；要换视角用顶部 format 框，
   语法（`BufferFormatter.cpp:673`）：

   | 写法 | 含义 |
   |---|---|
   | `half` / `float16_t` | fp16 逐元素（vkop packed fp16 SSBO 用这个直接读值） |
   | `uint` / `xint` | 32 位原样 / 十六进制（packed word 视角） |
   | `float`、`int`、`int8_t`… | fp32 / int32 / int8 |
   | 后缀 `2`-`9` | 向量宽度，如 `half2`、`uint4` |
   | `[N]` | 数组维度；支持 `row_major`/struct 声明 |

   注意 vkop packed fp16 是 2 half/uint word，声明类型是 uint 数组——format 改 `half`
   后按元素逐个列出，与 host 端 `(total+1)/2` words 的解码一致。
4. Buffer Viewer 的 Export 菜单可存 **CSV** 或 **Raw bytes**（`BufferViewer.cpp:2505`）

选中事件后 Buffer Viewer 显示的是**该 dispatch 执行前的状态**（replay 精确重放到
此 EID）。要看 op 的输出，选中下一个 dispatch 看同一 buffer。

### CS 单线程调试

- Pipeline State Viewer → Compute shader 面板 → **Debug** 按钮
  （`computeDebugSelector`，`VulkanPipelineStateViewer.cpp:5558`）→ 选 workgroup +
  thread → 内部调 `DebugThread(groupid, threadid)`（`renderdoc_replay.h:1022`）→
  ShaderViewer 单步 SPIR-V，每步的 SSBO 读取/变量可见
- **coopmat kernel 调不了**：`spirv_debug_setup.cpp:874` 把 `CooperativeMatrixKHR`
  capability 归入不支持；trace 为 NULL 静默失败。调 MatMul 要用 buffer kernel 路由
  （即默认的 guard 降级状态）

### Headless dump（python，嵌在 §8 的脚本框架里）

```python
controller.SetFrameEvent(eid, True)          # replay 到该 dispatch
for b in controller.GetBuffers():            # BufferDescription: .name/.length/.resourceId
    ...
data = controller.GetBufferData(target, 0, 0)   # bytes；len=0 取到末尾

import array                                  # 嵌入式 python 没有 numpy，只有 stdlib
h = array.array('H'); h.frombytes(bytes(data))  # fp16: 原始 half 位型
# 或者：struct.unpack('<%dH' % (len(data)//2), data) 后用 numpy 在外面转 float
```

vkop 目前**不给 buffer 命名**（`VulkanDebug::set_object_name` 没有调用点，只有
dispatch 的 insert-label），所以 capture 里 buffer 是匿名的，按
`BufferDescription.length` / 描述符顺序定位。想按名字找 tensor，先在
`ops/Operator.cpp` 的 label 逻辑旁边加 `set_object_name`（`VKOP_DEBUG_LABELS` 已
有开关保护，零开销路径不变）。

## 7. 截图辅助（无 import/convert 的机器）

```bash
DISPLAY=:1 xwd -id <window_id> -silent > /tmp/shot.xwd
ffmpeg -y -loglevel error -i /tmp/shot.xwd /tmp/shot.png
```

窗口 id 用 `xwininfo -tree -root` 查。`xdotool`/`wmctrl` 本机未装。

## 8. Python 脚本分析（无头聚合，推荐）

### 机制（源码验证）

- `qrenderdoc --python <script.py>`：QApplication 创建后、UI 打开前执行脚本；
  脚本末尾 `raise SystemExit(0)` 可直接退出、跳过 UI（`qrenderdoc.cpp:700`）。
- 嵌入式 python **有** `renderdoc`/`qrenderdoc` 模块（命令行 pylibs 没有是另一回事）。
- Qt 日志走 `RENDERDOC_LogMessage` → 落到 `/tmp/RenderDoc/RenderDoc_*.log`，
  不进 stdout；python 异常 traceback 也只在这里（还有弹窗）。
- `print()` 的输出经 textOutput → qInfo → 同一个 log 文件，所以**把结果写文件最稳**。

### API 链路

```python
pyrenderdoc.LoadCapture(RDC, renderdoc.ReplayOptions(), RDC, False, True)

def cb(controller):                      # IReplayController
    sf = controller.GetStructuredFile()
    res = controller.FetchCounters([renderdoc.GPUCounter.EventGPUDuration,
                                    renderdoc.GPUCounter.CSInvocations])
    # r.eventId / r.counter / r.value.d(秒) / r.value.u64
    for a in controller.GetRootActions():   # 注意是复数 GetRootActions()
        walk(a)                              # a.flags / a.customName / a.children
                                            # a.GetName(sf) / a.dispatchDimension
pyrenderdoc.Replay().BlockInvoke(cb)
```

### action 树与 op 名（关键）

vkop 的 op 名来自 `vkCmdInsertDebugUtilsLabelEXT`（`vulkan/VulkanDebug.hpp`），
在 action 树里是 **`SetMarker` 叶子，和它后面的 dispatch 同级**；UI 里显示成父行
（"q_proj/MatMul" 下挂 "vkCmdDispatch(32,1,1)"）是 Event Browser 的 fake-marker 逻辑
（`EventBrowser.cpp` 的 `UpdateDurationColumn`）。脚本里要把 SetMarker 的 customName
**传给下一个 action**（用 `pend = ['']` 可变容器，函数参数回传不了）：

```python
if a.flags & renderdoc.ActionFlags.SetMarker:
    pend[0] = str(a.customName)     # label 只命名紧随的下一个 action
    return
```

多 dispatch op（split-K fold 等）用 `begin_region` → 正常的 `PushMarker` 父节点。
"Compute Pass #N" / "Copy/Clear Pass #N" 是 replay 生成的假容器
（`replay_controller.cpp:406-416`）。

### 现成脚本

`/tmp/rdoc_dl/agg.py` — 改一行 `RDC` 路径即可用于任何 rdc；输出按 op 聚合的
GPU 时间表 + 单事件 top20，写 `/tmp/rdoc_dl/agg_out.txt`。运行约 1 分钟
（加载 30 s + counter 重放）。

```bash
LD_PRELOAD=/home/gwm/opt/renderdoc/lib/librenderdoc.so DISPLAY=:1 \
  timeout 570 /home/gwm/opt/renderdoc/bin/qrenderdoc --python /tmp/rdoc_dl/agg.py
```

示例结果（llm_driver round 2 decode，469 dispatch / 176 ms）：
`down_proj_N/MatMul` ×28 层 ~74%，`lm_head/MatMul` 8.4%（(75,1,1) WIDE ksplit），
MatMul 类合计 98% —— 与 vkop 自测"MatMul 占 GPU 81%"同数量级。

## 9. 上游参考（clone 在 ~/code/github/renderdoc）

- `docs/python_api/` — 官方示例（`iter_actions.py` 的树遍历最常用）
- `qrenderdoc/Windows/TimelineBar.cpp` — timeline 无时间轴的证据
- `qrenderdoc/Windows/EventBrowser.cpp` — duration 过滤/fake marker/Duration 列
- `renderdoc/driver/vulkan/extension_support.md` — 扩展支持分类表
- 可选后续：给上游提 issue/PR —— 把 KHR coopmat 加入 hookset，或让
  `layerGetPhysicalDeviceProcAddr` 对已知未实现函数返回 NULL（而不是 spec 兜底直通）；
  另 NV 机器上 `vkGetPhysicalDeviceCooperativeMatrixPropertiesNV` 会同样崩，vkop 侧
  防护目前只盖 KHR 入口点。
