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
