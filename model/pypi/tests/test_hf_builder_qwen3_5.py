#!/usr/bin/env python3
"""hf_builders/qwen3_5: the Qwen3.5 text tower through `--hf-checkpoint`.

ONNX Runtime cannot run the graph (Conv1d / LinearAttention are vkop custom
ops), so correctness is split:

  * the wrapper itself vs HF's own text model, in torch, on real weights
    (the export script's SELF_CHECK, now `qwen3_5.self_check`);
  * the meta-traced graph, structurally: the zero-centred norm fold reaches
    the payload as an addend (side file holds bf16 w + 1), `mrope.M` is the
    REAL table (the old STREAM_EXPORT path wrote zeros: no RoPE at all in the
    full-attention layers), the custom-op outputs are declared, the `mtp.*`
    namesakes are not what got referenced;
  * the CLI converting the checkpoint end to end.

The tiny checkpoint mirrors the real layout: Qwen3_5ForConditionalGeneration
(`model.language_model.*`, a vision tower), bf16, sharded, plus an `mtp.*`
shard repeating text-tower leaf names and shapes.
"""
import os
import subprocess
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
onnx = pytest.importorskip("onnx")
if not hasattr(transformers, "Qwen3_5ForConditionalGeneration"):
    pytest.skip("transformers has no Qwen3.5", allow_module_level=True)

from onnx2vkop import hf_import  # noqa: E402
from onnx2vkop.hf_builders import qwen3_5 as q35  # noqa: E402

_PYPI = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(scope="module")
def ckpt(tmp_path_factory):
    from safetensors.torch import save_file
    out = str(tmp_path_factory.mktemp("q35") / "ckpt")
    torch.manual_seed(0)
    text = dict(num_hidden_layers=4, hidden_size=64, intermediate_size=128,
                num_attention_heads=4, num_key_value_heads=2, head_dim=32,
                linear_num_key_heads=2, linear_num_value_heads=4,
                linear_key_head_dim=16, linear_value_head_dim=16,
                vocab_size=256, max_position_embeddings=256,
                tie_word_embeddings=False,
                rope_parameters={"rope_type": "default", "rope_theta": 1e4,
                                 "partial_rotary_factor": 0.25,
                                 "mrope_section": [2, 1, 1],
                                 "mrope_interleaved": True})
    cfg = transformers.Qwen3_5Config(
        text_config=text, vision_config=dict(
            depth=1, hidden_size=32, num_heads=2, intermediate_size=64,
            out_hidden_size=64))
    m = transformers.Qwen3_5ForConditionalGeneration(cfg).eval()
    with torch.no_grad():
        for n, p in m.named_parameters():
            if "norm" in n:
                p.add_(torch.randn_like(p) * 0.3)
            if "A_log" in n or "dt_bias" in n:
                p.copy_(torch.randn_like(p) * 0.5)
    m.to(torch.bfloat16).save_pretrained(out, safe_serialization=True,
                                         max_shard_size="200KB")
    sd = m.state_dict()
    save_file({"mtp.layers.0.mlp.gate_proj.weight": torch.randn_like(
                   sd["model.language_model.layers.0.mlp.gate_proj.weight"]),
               "mtp.norm.weight": torch.randn_like(
                   sd["model.language_model.norm.weight"])},
              os.path.join(out, "mtp.safetensors"))
    return out


def test_wrapper_matches_hf(ckpt):
    model = transformers.AutoModelForCausalLM.from_pretrained(
        ckpt, dtype=torch.float32).eval()
    w = q35.make_wrapper(model)
    assert len(w.folded) == 11          # 4x2 layer norms + q/k norm + final
    inputs, *_ = q35.llm_io(w.d, torch.float32, seq=7)
    # exact up to fp32 rounding: the gated norm is weight*norm, not (1+w)
    assert q35.self_check(model, w, inputs, tol=1e-5) < 1e-5


def test_graph_only_export(ckpt, tmp_path):
    from onnx import numpy_helper
    path = str(tmp_path / "g.onnx")
    hf_import.export_graph_onnx(ckpt, path)
    m = onnx.load(path, load_external_data=False)
    g = m.graph
    ops = [n.op_type for n in g.node]
    assert ops.count("LinearAttention") == 3 and ops.count("Conv1d") == 3
    la = next(n for n in g.node if n.op_type == "LinearAttention")
    attrs = {a.name: a.i for a in la.attribute}
    assert attrs == {"heads": 4, "dk": 16, "dv": 16}
    assert [i.name for i in g.input][:3] == ["inputs_embeds", "position_ids",
                                            "attention_bias"]
    assert "recurrent_state_out_0" in [o.name for o in g.output]
    declared = {v.name for v in g.value_info}
    for n in g.node:
        if n.op_type in ("Conv1d", "LinearAttention"):
            assert set(n.output) <= declared | {o.name for o in g.output}

    inits = {t.name: t for t in g.initializer}
    M = numpy_helper.to_array(inits["mrope.M"])
    inv = 1.0 / (1e4 ** (np.arange(0, 8, 2, dtype=np.float32) / 8))
    np.testing.assert_allclose(M, q35.mrope_table(torch.from_numpy(inv),
                                                  [2, 1, 1]).numpy())
    assert np.count_nonzero(M) == 4

    def ext(t):
        return {kv.key: kv.value for kv in t.external_data}
    side = [t for t in g.initializer
            if t.data_location == onnx.TensorProto.EXTERNAL
            and ext(t)["location"] == "g.side.bin"]
    assert len(side) == 11 and all(
        t.data_type == onnx.TensorProto.FLOAT16 for t in side)
    # the folded payload is the checkpoint's bf16 w plus one
    from safetensors import safe_open
    from onnx2vkop.stream_export import SafetensorsIndex
    idx = SafetensorsIndex(ckpt)
    t = inits["norm.weight"]
    e = ext(t)
    with open(tmp_path / "g.side.bin", "rb") as f:
        f.seek(int(e["offset"]))
        got = np.frombuffer(f.read(int(e["length"])), np.float16)
    src = idx.get("model.language_model.norm.weight")
    with safe_open(src["shard"], "pt") as f:
        w = f.get_tensor("model.language_model.norm.weight").float().numpy()
    np.testing.assert_allclose(got.astype(np.float32), w + 1.0, atol=2e-3)
    # every in-place reference is into the text tower, never mtp.safetensors
    locs = {ext(t)["location"] for t in g.initializer
            if t.data_location == onnx.TensorProto.EXTERNAL}
    assert not any("mtp" in loc for loc in locs)


def test_cli_converts(ckpt, tmp_path):
    out = tmp_path / "q35.vkopbin"
    cmd = [sys.executable, "-m", "onnx2vkop.cli", "--hf-checkpoint", ckpt,
           "-o", str(out), "-q", "fp16", "--stream-min-mib", "0",
           "--external-blob-mb", "0"]
    p = subprocess.run(cmd, cwd=_PYPI, capture_output=True, text=True)
    assert p.returncode == 0, (p.stdout + p.stderr)[-3000:]
    assert "builder=qwen3_5" in p.stdout
    assert out.stat().st_size > 0
