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

# Cases whose first generated token sits on a leading-space tie: HF's own top-2
# differ only by that space, so fp16 rounding decides which one wins. Measured
# on img_color_yellow: HF 31.703 ' yellow' vs 31.656 'yellow' (margin 0.047,
# softmax 0.440/0.420), ONNX 31.734 vs 31.719 (margin 0.016) -- the whole logit
# vector is 1.0% apart in L2, which is normal fp16 export-vs-HF distance, and
# both sides say the same word. Token id equality would make the test assert on
# a coin flip, so these compare word sequences instead; the id length must still
# match, so a different answer or a different number of steps still fails.
LEADING_SPACE_TIE = {"img_color_yellow"}


@pytest.mark.parametrize("case", CASES, ids=[c[0] for c in CASES])
def test_onnx_vs_hf_token_consistency(onnx_eng, hf_model, case):
    """每个用例：ONNX greedy 生成 vs HF greedy 生成，token id 序列应一致。

    LEADING_SPACE_TIE 里的用例改比「词序列」（见该集合的注释），其余仍逐 id 相等。
    """
    text, image = case_prompt(case)
    onnx_ids = onnx_eng.generate(text, image=image, max_new=MAX_NEW)
    inputs = onnx_eng.make_inputs(text, image=image)
    with torch.no_grad():
        gen = hf_model.generate(**inputs, max_new_tokens=MAX_NEW, do_sample=False)
    hf_ids = gen[0][inputs.input_ids.shape[1]:].tolist()
    dec = onnx_eng.proc.decode
    onnx_text = dec(onnx_ids, skip_special_tokens=True)
    hf_text = dec(hf_ids, skip_special_tokens=True)
    if case[0] in LEADING_SPACE_TIE:
        assert onnx_text.split() == hf_text.split() and len(onnx_ids) == len(hf_ids), (
            f"ONNX vs HF 文本不一致（{case[0]} 允许前导空格差异）"
            f"\n  onnx: {onnx_ids} -> {onnx_text!r}"
            f"\n  hf  : {hf_ids} -> {hf_text!r}"
        )
        if onnx_ids != hf_ids:
            print(f"[tolerated] {case[0]}: id 不同但同词 "
                  f"{onnx_ids}->{onnx_text!r} vs {hf_ids}->{hf_text!r}")
        return
    assert onnx_ids == hf_ids, (
        f"ONNX vs HF token 不一致\n  onnx: {onnx_ids} -> {onnx_text!r}"
        f"\n  hf  : {hf_ids} -> {hf_text!r}"
    )
