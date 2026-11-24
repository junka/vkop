"""The H3 text tower must be traced in FP16, not torch's FP32 default.

The mask is the reason. `Qwen3VLTextOnnx.causal_bias` makes `attention_bias`
fp16 (matching HF's `create_causal_mask`), and it is a graph INPUT. On an FP32
trace every other tensor is fp32, so HF's `attn_weights.float()` softmax path
puts a `Cast(to=fp32)` between that fp16 mask and the attention. That Cast seeds
an FP32 island; `GraphFp16` then refuses the graph (250 nodes mixing widths) and
`-q fp16` cannot produce a usable artifact -- the runtime rejects the resulting
Attention with "the attention bias must have Q's element type (Q is float16, the
bias is float32)".

Measured on the real 64-layer config, counting `Cast(attention_bias)`:

    default fp32 build -> 2 casts
    bfloat16 build     -> 2 casts   (does not help; the mask is fp16 regardless)
    float16 build      -> 0 casts   (activations are fp16, no upcast needed)

The reference LLM exporter already builds its tower in fp16
(`from_pretrained(..., torch_dtype=torch.float16)`), which is why that path
works. These tests pin the same choice here, and the reason, so a future edit
does not quietly drop it (the model still traces and converts without it -- it
just converts into something the runtime will not execute).
"""
import inspect
import pathlib

import pytest

HERE = pathlib.Path(__file__).resolve()
# tests/ -> pypi/ -> model/ -> repo root
EXPORTER = (HERE.parents[3] / "model" / "exporter" / "minimax_h3"
            / "h3_export_text_encoder.py")


def source():
    if not EXPORTER.is_file():
        pytest.skip(f"exporter not in this checkout: {EXPORTER}")
    return EXPORTER.read_text()


def test_build_text_model_puts_the_tower_in_fp16():
    src = source()
    start = src.index("def build_text_model")
    end = src.index("\ndef ", start + 1)
    body = src[start:end]
    assert "model.to(torch.float16)" in body or ".half()" in body, (
        "build_text_model no longer moves the tower to fp16; the FP32 trace "
        "re-introduces the fp32 Cast island on attention_bias and -q fp16 "
        "refuses the graph")


def test_the_fp16_build_is_documented_with_its_reason():
    """A reason-less cast gets deleted by the next reader as a 'speed knob'."""
    src = source()
    start = src.index("def build_text_model")
    end = src.index("\ndef ", start + 1)
    body = src[start:end]
    lower = body.lower()
    assert "attention_bias" in lower and ("cast" in lower or "island" in lower), (
        "the fp16 build's comment must name the mask Cast / fp32 island it "
        "prevents, or the cast reads like an unexplained precision choice")
