"""端到端 token 一致性测试：ONNX generate vs HF generate(do_sample=False)。

判定：greedy 下 ONNX 生成的 token id 序列与 HF 完全一致 = PASS。
不验证「答案是否正确」（CPU + 低分辨率，小模型能力有限，但只要两边一致即通过）。
用例来自 cases.py（业界通用场景：纯文本 / 图像理解 / OCR / 边界尺寸）。
"""
import os
import sys
import numpy as np
import torch
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cases import CASES, case_prompt

MAX_NEW = 12


@pytest.mark.parametrize("case", CASES, ids=[c[0] for c in CASES])
def test_onnx_vs_hf_token_consistency(onnx_eng, hf_model, case):
    """每个用例：ONNX greedy 生成 vs HF greedy 生成，token id 序列应一致。"""
    text, image = case_prompt(case)
    onnx_ids = onnx_eng.generate(text, image=image, max_new=MAX_NEW)
    inputs = onnx_eng.make_inputs(text, image=image)
    with torch.no_grad():
        gen = hf_model.generate(**inputs, max_new_tokens=MAX_NEW, do_sample=False)
    hf_ids = gen[0][inputs.input_ids.shape[1]:].tolist()
    dec = onnx_eng.proc.decode
    assert onnx_ids == hf_ids, (
        f"ONNX vs HF token 不一致\n  onnx: {onnx_ids} -> {dec(onnx_ids, skip_special_tokens=True)!r}"
        f"\n  hf  : {hf_ids} -> {dec(hf_ids, skip_special_tokens=True)!r}"
    )
