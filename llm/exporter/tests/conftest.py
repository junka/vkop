"""共享 pytest fixtures。

session 级加载 OnnxQwen3VL（含 ONNX sessions + HF ref model + processor），
供 test_numeric / test_consistency 复用，避免每个用例重新加载 2B 权重。
"""
import os
import sys
import pytest
import torch

# 让 tests/ 子目录能 import 上级 exporter 包（onnx_infer / cases）。
_EXPORTER_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _EXPORTER_DIR not in sys.path:
    sys.path.insert(0, _EXPORTER_DIR)

from infer import OnnxQwen3VL


@pytest.fixture(scope="session")
def onnx_eng():
    """加载 ONNX sessions + HF ref model（一次性，全 session 复用）。"""
    return OnnxQwen3VL()


@pytest.fixture(scope="session")
def hf_model(onnx_eng):
    """HF 参考模型，复用 onnx_eng.ref 避免二次加载。"""
    return onnx_eng.ref


@pytest.fixture(scope="session")
def proc(onnx_eng):
    return onnx_eng.proc
