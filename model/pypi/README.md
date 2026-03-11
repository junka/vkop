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
onnx2vkop -i model.onnx -q --fp16
```