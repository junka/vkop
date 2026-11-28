#!/usr/bin/env python3
"""`--hf-checkpoint`: graph-only ONNX that references the safetensors shards.

Builds tiny random Hugging Face checkpoints (sharded, so references cross
files and carry non-zero offsets), exports them through hf_import, and checks:

  * no weight byte is copied: every in-place initializer's external_data names
    a shard and that shard's exact (offset, length) from its header;
  * the graph computes what HF computes: ONNX Runtime on the graph-only ONNX
    against `AutoModelForCausalLM` logits, prefill AND one decode step through
    the explicit KV I/O (Qwen3: tied head, GQA, q/k norm; Llama: llama3 RoPE
    scaling, untied head);
  * dtype policy: an fp32 checkpoint under an fp16 trace goes to the side file;
    a bf16 checkpoint under an fp16 trace stays in the shards as BFLOAT16 (the
    converter normalizes it);
  * the CLI converts a bf16 checkpoint end to end with weights streamed from
    the shards (no GPU needed for that part).

ORT refuses external paths that leave the model's directory, so the ORT
checks write the ONNX inside the checkpoint directory.
"""
import os
import subprocess
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
onnx = pytest.importorskip("onnx")

from onnx2vkop import hf_import  # noqa: E402
from onnx2vkop.stream_export import SafetensorsIndex  # noqa: E402

_PYPI = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _tiny(kind, dtype, out):
    torch.manual_seed(0)
    if kind == "qwen3":
        cfg = transformers.Qwen3Config(
            vocab_size=256, hidden_size=64, intermediate_size=128,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
            head_dim=16, max_position_embeddings=128, tie_word_embeddings=True)
        m = transformers.Qwen3ForCausalLM(cfg)
    else:
        cfg = transformers.LlamaConfig(
            vocab_size=256, hidden_size=64, intermediate_size=128,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
            max_position_embeddings=128, tie_word_embeddings=False,
            rope_scaling={"rope_type": "llama3", "factor": 8.0,
                          "low_freq_factor": 1.0, "high_freq_factor": 4.0,
                          "original_max_position_embeddings": 32})
        m = transformers.LlamaForCausalLM(cfg)
    with torch.no_grad():
        for n, p in m.named_parameters():
            if "norm" in n:          # a wrong norm wiring must show
                p.add_(torch.randn_like(p) * 0.2)
    m.to(dtype).save_pretrained(out, safe_serialization=True,
                                max_shard_size="100KB")
    return out


def _ort_vs_hf(ckpt, onnx_path, trace_dtype):
    ort = pytest.importorskip("onnxruntime")
    model = transformers.AutoModelForCausalLM.from_pretrained(
        ckpt, dtype=torch.float32, attn_implementation="eager").eval()
    cfg = model.config
    S = 5
    ids = torch.randint(0, cfg.vocab_size, (1, S + 1),
                        generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        ref = model(input_ids=ids, use_cache=False).logits.numpy()
        emb = model.model.embed_tokens(ids).numpy()
    np_dt = np.float16 if trace_dtype == "fp16" else np.float32
    nkv = cfg.num_key_value_heads
    hd = getattr(cfg, "head_dim", None) or (cfg.hidden_size
                                            // cfg.num_attention_heads)
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    feeds = {"inputs_embeds": emb[:, :S].astype(np_dt),
             "position_ids": np.arange(S, dtype=np.int64)[None],
             "attention_bias": np.triu(np.full((S, S), np.finfo(np_dt).min,
                                               np_dt), 1)[None, None]}
    for i in range(cfg.num_hidden_layers):
        feeds[f"past_key_values_{i}"] = np.zeros((1, 2, nkv, 0, hd), np_dt)
    outs = sess.run(None, feeds)
    tol = 2e-3 if trace_dtype == "fp16" else 1e-5
    np.testing.assert_allclose(outs[0].astype(np.float32), ref[:, :S],
                               atol=tol, rtol=0)
    step = {"inputs_embeds": emb[:, S:].astype(np_dt),
            "position_ids": np.array([[S]], np.int64),
            "attention_bias": np.zeros((1, 1, 1, S + 1), np_dt)}
    for i in range(cfg.num_hidden_layers):
        step[f"past_key_values_{i}"] = outs[1 + i]
    got = sess.run(None, step)[0].astype(np.float32)
    np.testing.assert_allclose(got, ref[:, S:], atol=tol, rtol=0)


def _ext(init):
    return {kv.key: kv.value for kv in init.external_data}


@pytest.mark.parametrize("kind,dtype,trace,where", [
    ("qwen3", torch.float32, "fp32", "shard"),
    ("qwen3", torch.float32, "fp16", "side"),
    ("llama", torch.float16, "fp16", "shard"),
])
def test_graph_matches_hf(tmp_path, kind, dtype, trace, where):
    ckpt = _tiny(kind, dtype, str(tmp_path / "ckpt"))
    onnx_path = os.path.join(ckpt, "graph.onnx")
    hf_import.export_graph_onnx(ckpt, onnx_path, trace_dtype=trace)

    m = onnx.load(onnx_path, load_external_data=False)
    index = SafetensorsIndex(ckpt)
    ext = [i for i in m.graph.initializer
           if i.data_location == onnx.TensorProto.EXTERNAL]
    assert ext, "no external initializers"
    for init in ext:
        kv = _ext(init)
        if where == "shard":
            assert kv["location"].endswith(".safetensors"), kv
            path = os.path.join(ckpt, kv["location"])
            hits = [e for e in index.entries.values()
                    if os.path.samefile(e["shard"], path)
                    and e["offset"] == int(kv["offset"])]
            assert len(hits) == 1 and hits[0]["nbytes"] == int(kv["length"])
        else:
            assert kv["location"] == "graph.side.bin", kv
    assert not [f for f in os.listdir(ckpt) if f.endswith(".weights.bin")]
    _ort_vs_hf(ckpt, onnx_path, trace)


def test_bf16_stays_in_shards(tmp_path):
    ckpt = _tiny("qwen3", torch.bfloat16, str(tmp_path / "ckpt"))
    onnx_path = str(tmp_path / "out" / "g.onnx")
    hf_import.export_graph_onnx(ckpt, onnx_path, trace_dtype="auto")
    m = onnx.load(onnx_path, load_external_data=False)
    ext = [i for i in m.graph.initializer
           if i.data_location == onnx.TensorProto.EXTERNAL]
    assert ext and all(i.data_type == onnx.TensorProto.BFLOAT16 for i in ext)
    assert all(_ext(i)["location"].startswith("../ckpt/") for i in ext)
    # --act bf16 traces fp32 (the bf16 domain pass only rewrites FLOAT).
    assert hf_import.pick_trace_dtype("auto", "bf16",
                                      SafetensorsIndex(ckpt)) == "fp32"


def test_cli_converts_bf16_checkpoint_streamed(tmp_path):
    ckpt = _tiny("qwen3", torch.bfloat16, str(tmp_path / "ckpt"))
    out = tmp_path / "m.vkopbin"
    cmd = [sys.executable, "-m", "onnx2vkop.cli", "--hf-checkpoint", ckpt,
           "-o", str(out), "-q", "fp16", "--stream-min-mib", "0",
           "--external-blob-mb", "0"]
    p = subprocess.run(cmd, cwd=_PYPI, capture_output=True, text=True)
    assert p.returncode == 0, (p.stdout + p.stderr)[-3000:]
    assert out.stat().st_size > 0
    assert (tmp_path / "m.graph.onnx").exists()


def test_missing_weight_fails_closed(tmp_path):
    ckpt = _tiny("qwen3", torch.float32, str(tmp_path / "ckpt"))
    # Drop q_proj of layer 1 from the index the splice sees.
    index = SafetensorsIndex(ckpt)
    gone = [k for k in index.entries if "layers.1.self_attn.q_proj" in k]
    for k in gone:
        del index.entries[k]
    config = transformers.AutoConfig.from_pretrained(ckpt)
    spec = hf_import.build_llama_decoder(config, torch.float32, index)
    proto, shapes, dtypes = hf_import.trace_weightless(spec)
    with pytest.raises(hf_import.HFImportError, match="not in the checkpoint"):
        hf_import.splice_to_shards(proto, shapes, dtypes, index, spec,
                                   str(tmp_path), "s.bin")
