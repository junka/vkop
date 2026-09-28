---
name: project-new-operators-implemented
description: 2026-09-28 完成 ReduceMean/Min/Max/Mod 4个缺失算子实现，Qwen-Image-2.1 DiT 算子覆盖率从 94.4% 提升到 100%
metadata:
  type: project
---

## 实现的算子

**ReduceMean** - 归约求均值（复用 Reduce shader，reduce_op=MEAN）  
**Min** - 逐元素最小值（buffer/min.comp + fp16）  
**Max** - 逐元素最大值（buffer/max.comp + fp16）  
**Mod** - 逐元素取模（buffer/mod.comp，用 `a - b * floor(a/b)` 实现）

## 关键文件

- `ops/ReduceMean.hpp`, `ops/Min.hpp`, `ops/Max.hpp`, `ops/Mod.hpp` - C++ 算子类
- `shaders/buffer/min.comp`, `max.comp`, `mod.comp` - GLSL shader（各含 fp16 变体）
- `ops/Ops.hpp` - 添加枚举 REDUCEMEAN(54), MIN(55), MAX(56), MOD(57)
- `ops/OperatorFactory.hpp` - 工厂注册
- `ops/BufferBinaryFactory.hpp` - int64 CPU 路径支持 Min/Max/Mod
- `CMakeLists.txt` - DUAL_FP16_SHADERS 列表添加新 shader

## 验证结果

✅ **编译通过** - make -j8 无错误  
✅ **ONNX 转换** - dit_prefill_tiny.onnx (Mod:5, Min:2, Max:2) → vkopbin 成功  
✅ **ONNX 转换** - dit_decode_tiny.onnx (Mod:6, Min:4, Max:4) → vkopbin 成功  
✅ **模型加载** - Prefill: 93 nodes/45 levels, Decode: 130 nodes/68 levels  
✅ **算子调度** - 新算子分配到独立执行层级（Level 28/29/36/37/43/56/63/64）  
⚠️ **GPU 推理** - decode_rt->Run() 因缺少 past_kv 输入崩溃（KV cache 集成未完成）

## 运行命令

```bash
cd /Users/doudou/wjj/vkop
export DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib

# ONNX 转换测试
python3 -c "import sys; sys.path.insert(0,'model/pypi'); from onnx2vkop.cli import main; sys.argv=['onnx2vkop','-i','image/exporter/dit_prefill_tiny.onnx','-o','image/exporter/dit_prefill_tiny.vkopbin']; main()"

# 模型加载测试
./build/image_gen image/exporter/dit_prefill_tiny.vkopbin image/exporter/dit_decode_tiny.vkopbin "test" 2 42 --size 64
```

## 下一步

完整 DiT decode 推理需要：
1. 构造 past_kv_0/past_kv_1 输入（零初始化或从 prefill 获取）
2. 正确的 RoPE cos/sin  embeddings
3. Attention bias（causal mask）
4. KV cache 更新逻辑（present → past）

详见 `image/exporter/NEW_OPERATORS_SUMMARY.md`
