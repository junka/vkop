"""Qwen3-VL ONNX 导出器包。

  exporter.py        — 导出逻辑（VisualExport / Qwen3VLLMOnnx / export()）。
  infer.py           — ONNX 端到端推理驱动（OnnxQwen3VL）。
  cases.py           — 一致性测试用例构造（合成图、prompt 模板、CASES 列表）。
  tests/             — pytest 测试（数值对齐 + 端到端 token 一致性）。
  visual.onnx / llm.onnx / llm.weights.bin — 导出产物。
  qwen3vl_infer.py   — HF 原生推理封装（对比基准）。
"""
