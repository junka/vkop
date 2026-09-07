# Plan: GPU-in-place KV cache (eliminate the 58 sync points / decode round)

## 现状（为什么每轮 1.7s）

每轮 decode 的 ~1.7s **不是 GPU 算得慢**，是 58 个 GPU 同步点（submit+wait）的开销。来源：

- `ReadResult()` (core/runtime.cpp:1535): 1 个 `wait_all_done` + 29 个 `copyToCPU` sync（28 层 present_kv + 1 logits）= **30 syncs**
- `feedback_kv()` (llm_chat.cpp:188): 28 层 × (`ResizeInput` 重新分配 VkBuffer + `copyToGPU` sync) = **28 syncs**

每层的 CPU↔GPU 往返：`present VkBuffer(device) →[copyToCPU+wait]→ present data_(host) →[memcpy]→ past data_(host) →[copyToGPU+wait]→ past VkBuffer(device)`。present 和 past 是**两个不同的 device buffer**。

**KV cache 在这里只保证正确性，没有性能收益**——反而每轮把 28 层 KV 全量在 CPU↔GPU 之间搬一遍。

## 目标

把每轮 decode 从 ~1.7s 降到接近纯 GPU 计算时间（预计 ~0.3-0.5s）。核心：**KV cache 在 GPU 上原地滚动，零 CPU 往返**。

## 架构事实（已 verify）

1. `past_key_values_i`（input）和 `present_key_values_i`（output）是**两个不同的 Tensor 对象**，各自有自己的 `vkobj_`（VkBuffer）。都被 `set_ref_cnt_forever()` pin 住，不进 recycle pool。
2. `present = Concat(past, new_kv, axis=3)`。Concat 写入 output tensor 自己的 VkBuffer（`bind_ssbo` → `as_storage_buffer` → `make_vkbuff`，size 匹配时复用）。
3. `ResizeInput` (runtime.cpp:615) → `recreate_storage_buffer` (Tensor.hpp:481) **无条件 drop 旧 VkBuffer + 重新分配**。这是每轮 28 次重新分配的来源。
4. `make_vkbuff` (Tensor.hpp:846) size 匹配时**复用**旧 buffer（不重新分配）。但 `recreate_storage_buffer` 先 `vkobj_.reset()` 了，所以总是重新分配。
5. `copyToGPU`/`copyToCPU` (Tensor.hpp:633/648) 各自 submit+wait = 1 sync point。
6. Tensor **没有**暴露 `vkobj_` 的 setter，没有 attach external buffer 的 API。
7. `VulkanBuffer::copyBufferToStageBuffer` / `copyStageBufferToBuffer` (VulkanBuffer.cpp:204/225) 是现成的 `vkCmdCopyBuffer` 封装（device↔staging）。**没有** device→device 的直接封装，但底层 `vkCmdCopyBuffer` 可用。
8. kv_len 每轮 +1。所以 present 和 past 的 buffer 每轮都变大 1 个 token（`2*NKV*128*2 = 4096 bytes/层/轮`）。
9. `outshape_tensor_map` recycle pool 只在 build-time 共享**中间** tensor，graph I/O 被 `set_ref_cnt_forever` 排除——所以让 present/past 共享 buffer 不会和 recycle 冲突。
10. 非输入项每轮也重新 upload（inputs_embeds/position_ids/attention_bias/deepstack×3/image_pad_mask = 6 个），但它们小，sync 开销主要在 wait 本身不在数据量。

## 方案：双 buffer 滚动 + device→device copy

**核心思路**：past 和 present 用**两个固定大小的 buffer**（预分配到 max_kv_len），每轮 Concat 把新 KV append 进 present buffer 的下一格，然后 past 和 present **交换身份**（下一轮的 past = 这一轮的 present）。完全在 GPU 上，零 CPU 往返。

但 vkop 的 Concat 是「写整个 out_shape」，不是「append 一格」。而且 past/present 是两个独立 Tensor，没法 swap 身份（descriptor binding 是按名字查的）。

**所以实际可行的最小改动**是：**device→device copy 替代 CPU 往返**。每轮 still 28 层 copy，但 copy 在 GPU 上做、**批量提交、单次 wait**，且不重新分配 buffer。

### 方案分两步落地：

---

### Step 1: 消除 buffer 重新分配（最小改动，立竿见影）

**问题**：`ResizeInput` 每轮 `recreate_storage_buffer` 把 past 的 VkBuffer drop 重分配。

**改法**：past 和 present 的 buffer **预分配到 max_kv_len**，之后 `make_vkbuff` size 匹配就一直复用。需要：

1. **Tensor 加一个 `resize_keep_buffer` / 或让 `make_vkbuff` 在 size >= needed 时复用**（当前是 size == 才复用）。最小侵入：给 past input 用一个「预分配大 buffer」路径。
2. **llm_chat 预分配**：LoadModel 后，把每个 `past_key_values_i` 和 `present_key_values_i` 的 buffer 预分配到 `{1,2,NKV,MAX_KV,HD}`（MAX_KV = max_new + L_prefill，比如 4096）。之后每轮只改逻辑 dims（reshape_view），不动 VkBuffer。
3. **`feedback_kv` 不再 `ResizeInput`**：past 的逻辑 shape 用 `reshape_view` 改（metadata only，不动 buffer）。Concat 的 output shape 也用 reshape_view，让 `make_vkbuff` size 匹配复用。

**收益**：消除 28 次 `recreate_storage_buffer` 重新分配。但 copyToCPU/copyToGPU 的 sync 还在。

---

### Step 2: device→device copy + 批量提交（消除 sync points）

**问题**：每层 present→CPU→past→GPU = 2 syncs × 28 = 56 syncs。

**改法**：用一个 GPU command buffer 批量 record 28 层的 `vkCmdCopyBuffer`（present → past，device→device），**单次 submit + 单次 wait**。

需要：

1. **VulkanBuffer 加 `copyDeviceToDevice(cmd, src, dst_offset, size)`**：直接 `vkCmdCopyBuffer(src, dst)`，带 barrier。或复用 `copyStageBufferToBuffer`（它其实就是 `vkCmdCopyBuffer(srcbuffer, buffer, ...)`，src 是 staging，但 `vkCmdCopyBuffer` 不在乎 src 是不是 staging——只要都是 VkBuffer 就行）。**实际可直接用 `copyStageBufferToBuffer`**：`past->copyStageBufferToBuffer(cmd, present->getBuffer(), 0, size, 0)`。验证一下 barrier 方向对不对。

2. **Tensor 加 `copyFromTensor(cmdpool, src_tensor)`**：封装「拿 src 的 VkBuffer，device→device copy 进自己的 VkBuffer」。或者直接在 llm_chat 里拿两个 `as_storage_buffer` 的 VkBuffer 调 `vkCmdCopyBuffer`。

3. **llm_chat 的 `feedback_kv` 重写**：
   ```cpp
   void feedback_kv(rt, cmdpool) {
       auto dev = cmdpool->getVulkanDevice();
       auto cmd = std::make_shared<VulkanCommandBuffer>(dev, ...);  // 单个 cmd buf
       cmd->begin();
       for (int i = 0; i < NLAYERS; ++i) {
           auto pres = as_tensor<uint16_t>(rt->GetOutput(pres_name));
           auto past = as_tensor<uint16_t>(rt->GetInput(past_name));
           // 逻辑 shape 改成新 kv_len（reshape_view，不动 buffer）
           past->reshape_view(pres_shape);
           // device→device copy，record 进同一个 cmd buf
           auto src = pres->as_storage_buffer(dev);  // 复用（size 匹配）
           auto dst = past->as_storage_buffer(dev, cmd);  // readBarrier
           // vkCmdCopyBuffer(src, dst, size)
           vkCmdCopyBuffer(cmd->get(), src->getBuffer(), dst->getBuffer(), 1, &region);
       }
       cmd->end();
       cmd->submit(dev->getComputeQueue());
       cmd->wait();  // 单次 wait，28 层全完成
   }
   ```

**收益**：56 syncs → 1 sync。加上 Step 1 的不重新分配，feedback_kv 从 28×(realloc+2sync) 降到 1×(batch copy + 1 sync)。

---

### Step 3: ReadResult 只读 logits（跳过 present_kv 的 copyToCPU）

**问题**：`ReadResult()` 对 28 个 present_kv 各做一次 `copyToCPU`（28 syncs），但 feedback_kv 改成 device→device 后**根本不需要 present 在 CPU 上**。

**改法**：`ReadResult` 只对 `logits` 做 copyToCPU，跳过 present_kv。两种实现：
- (a) llm_chat 不调 `rt->ReadResult()`，自己只 `copyToCPU` logits（`rt->GetOutput("logits")`）。
- (b) Runtime 加一个 `ReadResult(const std::vector<std::string>& only)` 重载。

**收益**：29 syncs → 1 sync（只 logits）。**这一步收益最大**（28 个 present_kv 的 copyToCPU 全砍）。

---

### Step 4（可选）: 非输入项不重复 upload

decode 每轮重新 upload 6 个非 KV 输入，其中 deepstack×3 和 image_pad_mask 内容不变。但因为 `fill_fp16_input`→`ResizeInput`→`recreate_storage_buffer` 把 buffer drop 了，`copyToGPU` 必须真传。

如果 Step 1 的「预分配 + reshape_view」做好，这些不变的输入可以只在 prefill 后 upload 一次，之后 `is_on_GPU()` 短路。attention_bias 每轮变（kv_len+1），仍需上传（但很小）。

**收益**：6 syncs → 1-2 syncs（只 attention_bias）。优先级低于 Step 1-3。

## 实施顺序与预期收益

| 步骤 | 改动 | syncs 消除 | 预期收益 |
|---|---|---|---|
| Step 3 | ReadResult 只读 logits | 28→0 | **最大**，~28 syncs × 几ms |
| Step 2 | feedback_kv device→device batch | 28→1 | **大**，~27 syncs + 消除 CPU memcpy |
| Step 1 | 预分配 buffer，不 recreate | 0 syncs 但消除 28 次 VkBuffer 重分配 | 中，driver 开销 |
| Step 4 | 非输入不重复 upload | 5→0 | 小 |

**建议先做 Step 3 + Step 2**（收益最大，且 Step 2 依赖 Step 1 的不重新分配——否则 device→device copy 写进一个每轮重分配的 buffer 没意义）。

所以实际顺序：**Step 1（预分配）→ Step 3（只读 logits）→ Step 2（device→device batch）**。Step 4 可选。

---

## 实施结果（2026-09-07）

**Step 1-3 全部完成并验证正确**（文本 "2 + 2 = 4."、多模态 " green" `[6176,151645]` rope_delta=-42 都逐 token 一致）。

详细计时（`std::chrono` 拆分 Run/argmax/feedback）：

```
[r1] submit=1725 gpu=1 argmax=3   feedback=1
[r2] submit=1808 gpu=1 argmax=2   feedback=1
```

**KV feedback 优化成功**：从 ~28 syncs 降到 **1ms**（device→device batch copy + 单次 wait）。这是真实且正确的改进。

**但每轮仍 ~1.8s**，原因**不是** sync / KV 搬运（那部分已降到 1ms），而是：

- `Run()` 本身耗时 ~1.7s，拆分后 `submit=1725ms, gpu=1ms`
- 即 **CPU 侧 command buffer recording + descriptor writes** 耗 1.7s，GPU 实际计算只 ~1ms（与 submit 重叠）
- 根因：图有 **3270 operations / 1593 levels**，每个 op 的 `onExecute`（Operator.hpp:137）每轮都重做 `m_cmd_->begin()` → `execute()`（`fillWriteDescriptorSets` + `vkUpdateDescriptorSets` + `bind` + `vkCmdDispatch`）→ `end()`。3270 次 descriptor 写入 + 命令记录/轮。

### 剩余瓶颈：onExecute 每轮重记录（未做）

**问题**：3270 ops × 每轮重记录 + 重写 descriptor = ~1.7s/轮 CPU 开销。

**理想方案**：record once + replay（第一轮记录 command buffer + descriptor set，后续轮直接 submit）。

**难点（已 verify）**：
1. push constants 每轮变：KV-cache Concat 的 `ConcatPC.outDims`/`offset` 随 kv_len 增长而变；dispatch count（`UP_DIV(nthreads,256)`）也变。replay 会用旧 push constant → 错误 dispatch size。
2. 解法：要么 (a) 预分配到 MAX_KV + 用 MAX_KV 作 dispatch size（shader 里越界线程 no-op，需改 shader），要么 (b) 只对**不变**的 op（initializer/权重 Gemm，占绝大多数）做 record cache，动态 op（Concat/Shape-meta）仍每轮记录。
3. 还需预分配所有变长输入（attention_bias 每轮 kv_len+1 增长）到 MAX_KV，使 VkBuffer 句柄稳定，descriptor 才能复用。

**风险**：record cache 是 runtime 架构级改动，push constant/dispatch size/reshape_view 每轮都变，盲 replay 会出错。需要更细粒度的「哪些 op 可 cache」分析。

**当前状态**：Step 1-3 已提交（KV 搬运开销消除）。record cache 作为后续优化，需单独评估。

## 风险点

1. **Concat 的 output shape 与 buffer size**：预分配大 buffer 后，Concat 写入时 `output->num_elements()` 是逻辑 shape（kv_len+1），但 buffer 是 MAX_KV size。`make_vkbuff` 用 `num_elements()` 算 aligned size，会看到 size 不匹配 → 重新分配。**必须让 Concat 也用预分配的 buffer**：要么 Concat 的 `bind_ssbo` 路径识别「buffer 已存在且 size >= needed 就复用」，要么把 past/present 的逻辑 shape 和物理 size 解耦。这是 Step 1 的关键难点。
2. **reshape_view 的 element-count 守卫** (Tensor.hpp:84)：`reshape_view` 要求 `ne_new == ne_old`，但 kv_len 每轮变 → num_elements 变 → reshape_view 拒绝。需要放宽这个守卫，或用 `resize`（不动 vkobj_）改 dims。
3. **descriptor 每轮重写**：op 的 `execute()` 里 `fillWriteDescriptorSet` 每次重写 descriptor 指向新 buffer。如果 buffer 不变（预分配复用），descriptor 内容不变但仍重写——无 sync 开销，可忽略。
4. **barrier 方向**：device→device copy 需要 present 先 transferReadBarrier、past 先 transferWriteBarrier。`copyStageBufferToBuffer` 内部已处理 past 的 write barrier，但 present 的 read barrier 要单独加（或确认 Concat 写完后 present 的 access flag 状态）。
5. **prefill 后第一次 feedback**：prefill 的 past 是 kv_len=0（空），present 是 kv_len=L。第一次 feedback 要 copy L 个 token。之后每轮 +1。预分配 MAX_KV 要覆盖 L + max_new。

## 关键文件

- `core/Tensor.hpp`: 加 resize-keep-buffer / 放宽 reshape_view / 暴露 device→device copy
- `core/runtime.cpp`: ReadResult 加 only-logits 路径；ResizeInput 加「不 recreate」选项
- `vulkan/VulkanBuffer.hpp/.cpp`: 确认 device→device copy 可用（copyStageBufferToBuffer 即可）
- `llm/exporter/llm_chat.cpp`: feedback_kv 重写（batch device→device）+ 预分配 + ReadResult 只读 logits
- `ops/Concat.hpp`: 确认预分配 buffer 下不重分配（Step 1 难点）

## 验证

1. 文本-only 跑通：`printf 'What is 2+2?' | ./build/llm_chat ... 8` → 仍出 "2 + 2 = 4."
2. 多模态跑通：带 `--image` → 仍出 " green" `[6176, 151645]`
3. 多轮 decode：logits/输出与改前逐 token 一致（KV 内容不变，只是搬运方式变了）
4. 计时：每轮 decode 从 ~1.7s 降到 < 0.6s（目标）
5. ASan：`ENABLE_ASAN=ON` 重 build 跑，确认无 heap 越界（预分配 buffer 容易写超）
