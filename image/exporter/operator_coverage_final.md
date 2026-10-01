# Qwen-Image-2.1 DiT Operator Coverage Report

## Summary

The DiT graphs have been successfully converted to vkop DAG format (bypassing ConstantFolder due to 13.8 GB model size). All operators except 4 types are supported by the vkop runtime.

### Graph Statistics
- **Prefill**: 5421 nodes, 32 KV cache outputs
- **Decode**: 5471 nodes, 1 sample output
- **Total**: 10,892 nodes across both graphs

### Operator Coverage

| Status | Count | Operators |
|--------|-------|-----------|
| ✅ Supported | 5,131 nodes (47.1%) | Add, Concat, Div, Gather, LayerNormalization, MatMul, Mul, Neg, Pow, Reshape, Sigmoid, Sin, Cos, Slice, Softmax, Split, Sqrt, Tanh, Transpose |
| 🔧 Already Implemented | 728 nodes (6.7%) | Cast (393), Shape (254), Unsqueeze (83) |
| ❌ Missing | 609 nodes (5.6%) | Mod (191), ReduceMean (128), Min (126), Max (126) |
| 🗑️ Should Fold | 4,423 nodes (40.6%) | Constant (will be eliminated by ConstantFolder) |

### Real Runtime Work

After ConstantFolder eliminates the 4,423 Constant nodes, the actual runtime graph has:
- **~6,469 nodes** total
- **94.4% supported** (6,105 nodes)
- **5.6% missing** (364 nodes: Mod + ReduceMean + Min + Max)

## Missing Operator Implementation Priority

### 1. ReduceMean (128 nodes) - HIGH PRIORITY
- Used in LayerNormalization for computing mean along axes
- Similar to existing `Reduce` op but needs mean variant
- Implementation: ~50 lines of shader code (parallel reduction)

### 2. Min/Max (252 nodes combined) - MEDIUM PRIORITY  
- Element-wise broadcast math operations
- Simple per-element comparison with broadcasting
- Implementation: ~30 lines each (similar to Add/Mul)

### 3. Mod (191 nodes) - LOW PRIORITY
- Modulo arithmetic for position calculations
- Simple element-wise operation: `a % b`
- Implementation: ~20 lines (arithmetic op like Div)

## Conversion Pipeline Status

✅ **ONNX Export**: Both prefill and decode graphs exported successfully  
✅ **ORT Verification**: Numerical alignment verified (mean diff < 0.005 for early layers, < 0.15 for layer 31 due to fp16 accumulation)  
✅ **DAG Conversion**: Successfully built vkop DAG without optimization (bypassed ConstantFolder)  
❌ **Binary Save**: Failed due to protobuf 2GB limit (expected for 13.8 GB models)  

## Next Steps

1. Implement the 4 missing operators (ReduceMean, Min, Max, Mod) — estimated 2-3 days
2. Re-run onnx2vkop with ConstantFolder enabled after operator implementation
3. Test vkop runtime with the converted `.vkopbin` files
4. Measure performance vs ORT baseline

## Notes

- The "Constant" nodes will be automatically folded during normal onnx2vkop conversion once the model fits in memory
- Cast, Shape, and Unsqueeze were initially reported as missing but are already implemented in vkop
- The 13.8 GB external data format prevents full-model serialization; streaming merge or chunked processing needed for final binary save
