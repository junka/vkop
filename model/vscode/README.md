NEW_FILE_CODE
# VKOP Model Viewer

[![Version](https://img.shields.io/visual-studio-marketplace/v/Wan-Junjie.vkop-model-viewer)](https://marketplace.visualstudio.com/items?itemName=Wan-Junjie.vkop-model-viewer)
[![Installs](https://img.shields.io/visual-studio-marketplace/i/Wan-Junjie.vkop-model-viewer)](https://marketplace.visualstudio.com/items?itemName=Wan-Junjie.vkop-model-viewer)
[![Rating](https://img.shields.io/visual-studio-marketplace/r/Wan-Junjie.vkop-model-viewer)](https://marketplace.visualstudio.com/items?itemName=Wan-Junjie.vkop-model-viewer)

A powerful Visual Studio Code extension for viewing and analyzing VKOP binary model files. This extension provides an interactive, hierarchical visualization of neural network models similar to Netron.

## ✨ Features

### 🎯 Core Features

- **Hierarchical Layer Display**: Automatically arranges model layers in a clear, top-down flow structure
- **Interactive Visualization**: Click on nodes to view detailed information about inputs, outputs, and attributes
- **Color-Coded Operations**: Different operation types are color-coded for easy identification
- **Zoom & Pan**: Navigate through large models with smooth zoom and pan controls
- **Shape Information**: Display tensor shapes for all inputs and outputs

### 🔍 Supported Operations

The viewer supports visualization of various neural network operations including:

- **Convolution Layers**: Conv, Conv2d
- **Matrix Operations**: Gemm, MatMul
- **Activation Functions**: Relu, Sigmoid, Softmax, Softplus
- **Normalization**: BatchNormalization, LayerNorm
- **Pooling**: AveragePool, MaxPool, GlobalAveragePool
- **Element-wise Operations**: Add, Sub, Mul, Div, Pow
- **Data Manipulation**: Reshape, Transpose, Concat, Split, Slice
- **Other Operations**: Erf, Floor, PRelu, Resize, GridSample, Topk, NMS

### 🎨 Visual Features

- **Fixed Hierarchical Layout**: Unlike force-directed graphs, uses a stable top-down layout
- **Minimized Edge Crossing**: Intelligent layout algorithm reduces visual clutter
- **Custom Node Styling**: 
  - Box-shaped nodes for operations
  - Ellipse-shaped nodes for inputs (green) and outputs (red)
  - Color-coded by operation type
- **Smooth Bezier Edges**: Clean, professional-looking connections between nodes

## 📋 Requirements

- Visual Studio Code 1.60.0 or higher
- VKOP binary model files (.vkop or .vkopbin)

## 🚀 Getting Started

### Installation

1. Open Visual Studio Code
2. Go to Extensions (Ctrl+Shift+X)
3. Search for "VKOP Model Viewer"
4. Click Install

Alternatively, install from the command line:
```bash
code --install-extension Wan-Junjie.vkop-model-viewer