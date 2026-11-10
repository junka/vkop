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

`-q fp16` downcasts the weights of ops that ALREADY run fp16 -- an fp16 export,
where a few leftovers are still fp32. It does not turn an fp32 graph into an
fp16 one: the runtime picks each op's shader from its data input, so behind an
fp32 activation an fp16 weight is read as fp32 bytes (a wrong model, not a
smaller one). On a convnet exported in fp32 that means it converts nothing, and
the run now says so instead of printing a bare `Preserved 42 tensors`.

Hold one layer's weights at their source precision, to localize the accuracy a
byte format cost (`onnx2vkop/exclusion.py`):

```
onnx2vkop -i model.onnx -q int4 --exclude-scope model.layers.0
onnx2vkop -i model.onnx -q int8 --exclude-node '*/attn/v_proj/MatMul' \
    --exclude-downstream-of /embed
```
