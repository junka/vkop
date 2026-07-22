# tests/

pytest 测试套件，分两类：

## 文件

| 文件 | 作用 |
|---|---|
| [conftest.py](conftest.py) | session 级 fixture：加载 `OnnxQwen3VL`（含 ONNX sessions + HF ref model）与 HF processor，供所有用例复用，避免每用例重新加载 2B 权重。 |
| [test_numeric.py](test_numeric.py) | 数值对齐：visual pooler_output/deepstack、llm prefill logits + present_kv、llm decode logits，逐张量 vs HF。 |
| [test_consistency.py](test_consistency.py) | 端到端 token 一致性：ONNX generate vs HF generate(do_sample=False)，10 个业界通用场景用例（来自 `cases.py`）。 |

## 运行

```bash
# 从 exporter/ 目录
cd llm/exporter
pytest                          # 跑全部
pytest tests/test_numeric.py    # 仅数值对齐
pytest tests/test_consistency.py -k ocr   # 仅 OCR 用例
pytest -v                       # 详细
```

## 用例判定

- **test_numeric**：fp16 + CPU ort 累积舍入使深层输出 maxdiff 偶达 ~0.2，
  故用「均值差 < 1e-2」+ `torch.allclose` 双重判定，避免 fp16 噪声误报。
  单层 forward 与 deepstack 注入单测 maxdiff=0.0（逻辑完全正确）。
- **test_consistency**：greedy 下 ONNX 生成的 token id 序列与 HF 完全一致 = PASS。
  不验证「答案是否正确」（CPU + 低分辨率，小模型 OCR/计数能力有限，
  但只要 ONNX 与 HF 一致即通过）。
