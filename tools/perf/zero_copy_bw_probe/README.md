# zero_copy_bw_probe

dual_gpu_zero_copy.md 方案二 vs staging 中转的带宽/时延实测探针。

四种路径，同一 read-sum kernel（真实读满每个字节，校验和验证）：

- `vram`：数据预先上传到 private VRAM，kernel 直读（基线）
- `staging`：每轮 CPU memcpy → 持久映射 staging(type1) → vkCmdCopyBuffer → kernel 读（引擎现状）
- `host-import`：VK_EXT_external_memory_host 导入页对齐主机内存到 type1，kernel 直读（方案二）
- `managed`：type2 managed 常驻映射，每轮 vkFlushMappedMemoryRanges → kernel 读（方案三）

尺寸 4KB/64KB/1MB/16MB/256MB；每轮内四模式交错（A/B 同块），3 轮取 min。
计时为墙钟（record+submit+wait）：本机 MoltenVK/AMD 栈上 timestamp query 永不
available（vkGetQueryPoolResults 恒 VK_NOT_READY），故不用 GPU 时间戳。
小块结果主要反映提交固定开销，不是带宽。

注意：managed 的 flush 在重复轮中数据未变化，didModifyRange 可能被驱动短路，
flush 成本是下界；staging 的 cpu us/op 是主机 memcpy 成本。

## 构建与运行

```bash
cd tools/perf/zero_copy_bw_probe
glslc --target-env=vulkan1.1 -O read_sum.comp -o read_sum.spv
xxd -i read_sum.spv > read_sum.spv.h
sed -i '' 's/unsigned char read_sum_spv\[\]/static const unsigned char read_sum_spv[]/; s/unsigned int read_sum_spv_len/static const unsigned int read_sum_spv_len/' read_sum.spv.h
clang++ -std=c++17 -O2 -I/usr/local/include probe.cpp \
  -L/usr/local/lib -Wl,-rpath,/usr/local/lib -lvulkan -o probe
./probe --rounds 3
```

需要 MoltenVK 的 portability enumeration（代码已处理）。
2026-11-07 结果：`../logs/zero_copy_bw_probe_20261107.log`。