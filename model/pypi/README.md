# ONNX2VKOP

Convert and optimize ONNX models to VKOP (Vulkan Compute) format for efficient inference.

## Features

- **Model Optimization**: Automatically optimize ONNX models with various fusion techniques
- **Quantization Support**: FP16 and INT8 weight-only quantization
- **DAG-based Representation**: Efficient graph representation with parallel execution support
- **Format Conversion**: NCHW to RGBA conversion for Vulkan compute shaders
- **Memory Optimization**: Unified initializer memory blocks

## Installation

```bash
pip install onnx2vkop

pip install -e .
```

## Usage
quantize with fp16

```
onnx2vkop -i model.onnx -q fp16
```

`-q fp16` produces an FP16 artifact. `onnx2vkop/fp16_graph.py` computes the
graph's FP16 domain and rewrites the activation declarations and the weight
bytes together, because the runtime picks each op's shader from the dtype of
the tensors it is handed -- an FP16 weight behind an FP32 Conv reads a 384-byte
table as a 768-byte operand, which is a wrong model, not a smaller one. Only an
op whose build follows the tensor dtype AND that has a real dual FP16 buffer
build can enter the domain; an op that cannot keeps itself and every tensor it
touches at FP32 (that verdict propagates through the graph, and Cast is the only
node allowed to sit between the two widths), and the run names the op and the
reason. An export that already declares FP16 activations but ships FP32 weights
is handled by the older per-weight pass afterwards.

Hold one layer's weights at their source precision, to localize the accuracy a
byte format cost (`onnx2vkop/exclusion.py`):

```
onnx2vkop -i model.onnx -q int4 --exclude-scope model.layers.0
onnx2vkop -i model.onnx -q int8 --exclude-node '*/attn/v_proj/MatMul' \
    --exclude-downstream-of /embed
```

Convert twice -- once with the flag, once without -- and run
`tools/vkop_vs_ort.py` on both artifacts to read which number moved; the two
runs share the input tensor and the shaders, so the delta belongs to the layer
the flag named. See the accuracy-loop section of the repository README.

### A per-node ONNX Runtime reference (`--emit-ort-onnx`)

The fused graph speaks vkop's own vocabulary (RMSNorm, FusedElemwise, the
rotate-half RotaryEmbedding, the vkop Attention, MatMul `swiglu`/`down_act`/
`epi_*`, Conv `activation`, ...), which no other runtime can execute, and its
fused nodes have names the source graph never had. `--emit-ort-onnx` writes
that graph lowered back to standard ONNX (`onnx2vkop/ort_lower.py`): each vkop
node becomes a plain subgraph whose last node writes the vkop node's own output
tensor, every such tensor is a graph output, and the `vkop_node_outputs`
metadata maps node names to them.

```
onnx2vkop -i model.onnx -o model.vkopbin --master-archive build/vkop_shaders.bin \
    --emit-ort-onnx model.ort.onnx
python3 tools/vkop_vs_ort.py model.vkopbin --onnx model.ort.onnx --random-fill --node-dump
```

`--node-dump` runs vkop-bench once with `VKOP_NODE_DUMP` and compares every
dumped node output with its ORT tensor in execution order, marking the first
divergence (report-only; the exit code still follows the graph outputs).
Comparing the source ONNX with `model.ort.onnx` in ORT, on the tensor names both
keep, checks the fusion passes themselves without a GPU (tests/test_ort_lower.py
does this per pass).

The cut is taken right after fusion: before the FP16 domain, weight
quantization and the weight re-layouts, so the reference is the export's own
math. Floats are computed in fp32 unless `--ort-keep-dtype` is given. The flag
forces `--stream-weights off` (the lowering needs the weight bytes), so on a
large LLM it costs the full model in RAM twice over -- prefer a trimmed or tiny
model there. Not lowered yet: LinearAttention/GatedDeltaNet, the Qwen3.5
Conv1d, EmbeddingForward, Nms, and the ring / fp8 KV cache; these raise an
error naming the node.
