"""Qwen3-VL ONNX 导出器包。

  export/  — 导出与中间量 dump 脚本（models → visual.onnx / llm.onnx / *.bin / *_dump/）。
  check/   — 参考与验证脚本（ONNX/HF 推理、greedy 对拍、数据流与覆盖率分析）。
  driver/  — C++ 推理驱动与探针（llm_chat / llm_driver / visual_probe / rope_probe）。
  cases.py — 一致性测试用例构造（合成图、prompt 模板、CASES 列表）。
  tests/   — pytest 测试（数值对齐 + 端到端 token 一致性）。
  visual.onnx / llm.onnx / llm.weights.bin — 导出产物。
"""
