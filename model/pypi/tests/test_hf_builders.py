#!/usr/bin/env python3
"""Per-architecture builders (onnx2vkop/hf_builders) against HF itself.

Tiny random sharded checkpoints; the graph-only ONNX that `--hf-checkpoint`
writes is run in ONNX Runtime and compared with the HF model's logits, prefill
and one decode step through the explicit KV I/O. ORT refuses external paths
that leave the model directory, so the ONNX is written inside the checkpoint.

  * phi3: fused qkv/gate_up, partial RoPE (0.75), longrope attention_scaling
  * glm : interleaved RoPE rewritten half-split, repeat_kv-free GQA
  * qwen3_vl: MRoPE as a static matmul, deepstack inputs (text-only here,
    zero image rows), the causal-mask metadata the converter reads

The export scripts under llm/exporter/export/ build from the same
`make_wrapper`, so these also cover what they export.
"""
import os

import numpy as np
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
onnx = pytest.importorskip("onnx")
ort = pytest.importorskip("onnxruntime")

from onnx2vkop import hf_builders, hf_import  # noqa: E402


def _save(m, out):
    with torch.no_grad():
        for n, p in m.named_parameters():
            if "norm" in n:
                p.add_(torch.randn_like(p) * 0.2)
    m.to(torch.float16).save_pretrained(out, safe_serialization=True,
                                        max_shard_size="200KB")
    return out


def _phi3(out):
    cfg = transformers.Phi3Config(
        vocab_size=256, hidden_size=128, intermediate_size=256,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=1024, original_max_position_embeddings=256,
        partial_rotary_factor=0.75, tie_word_embeddings=True, pad_token_id=0,
        rope_scaling={"type": "longrope", "short_factor": [1.0] * 12,
                      "long_factor": [2.0] * 12})
    return _save(transformers.Phi3ForCausalLM(cfg), out)


def _glm(out):
    cfg = transformers.GlmConfig(
        vocab_size=256, hidden_size=128, intermediate_size=256,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        head_dim=32, max_position_embeddings=256, tie_word_embeddings=True,
        attention_bias=False, pad_token_id=0, partial_rotary_factor=1.0)
    return _save(transformers.GlmForCausalLM(cfg), out)


def _qwen3vl(out):
    cfg = transformers.Qwen3VLConfig(
        text_config=dict(
            vocab_size=1024, hidden_size=64, intermediate_size=128,
            num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
            head_dim=16, max_position_embeddings=4096,
            rope_parameters={"rope_type": "default", "rope_theta": 500000.0,
                             "mrope_section": [4, 2, 2],
                             "mrope_interleaved": True}),
        vision_config=dict(
            depth=4, hidden_size=32, intermediate_size=64, num_heads=2,
            patch_size=16, temporal_patch_size=2, spatial_merge_size=2,
            out_hidden_size=64, deepstack_visual_indexes=[1, 2, 3],
            num_position_embeddings=64, in_channels=3),
        image_token_id=1000, video_token_id=1001, vision_start_token_id=1002,
        tie_word_embeddings=True)
    torch.manual_seed(0)
    return _save(transformers.Qwen3VLForConditionalGeneration(cfg), out)


def _decoder_vs_hf(ckpt, onnx_path, embed, S=5):
    model = transformers.AutoModelForCausalLM.from_pretrained(
        ckpt, dtype=torch.float32, attn_implementation="eager").eval()
    cfg = model.config
    ids = torch.randint(0, cfg.vocab_size, (1, S + 1),
                        generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        ref = model(input_ids=ids, use_cache=False).logits.numpy()
        emb = embed(model)(ids).numpy()
    nkv = cfg.num_key_value_heads
    hd = getattr(cfg, "head_dim", None) or (cfg.hidden_size
                                            // cfg.num_attention_heads)
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    f = {"inputs_embeds": emb[:, :S].astype(np.float16),
         "position_ids": np.arange(S, dtype=np.int64)[None],
         "attention_bias": np.triu(np.full((S, S), -65504, np.float16),
                                   1)[None, None]}
    for i in range(cfg.num_hidden_layers):
        f[f"past_key_values_{i}"] = np.zeros((1, 2, nkv, 0, hd), np.float16)
    out = sess.run(None, f)
    np.testing.assert_allclose(out[0].astype(np.float32), ref[:, :S],
                               atol=2e-3, rtol=0)
    g = {"inputs_embeds": emb[:, S:].astype(np.float16),
         "position_ids": np.array([[S]], np.int64),
         "attention_bias": np.zeros((1, 1, 1, S + 1), np.float16)}
    for i in range(cfg.num_hidden_layers):
        g[f"past_key_values_{i}"] = out[1 + i]
    np.testing.assert_allclose(sess.run(None, g)[0].astype(np.float32),
                               ref[:, S:], atol=2e-3, rtol=0)


@pytest.mark.parametrize("make,builder", [(_phi3, "phi3"), (_glm, "glm")])
def test_decoder_builders_match_hf(tmp_path, make, builder):
    torch.manual_seed(0)
    ckpt = make(str(tmp_path / "ckpt"))
    cfg = transformers.AutoConfig.from_pretrained(ckpt)
    assert hf_builders.for_config(cfg).__module__.endswith(builder)
    onnx_path = os.path.join(ckpt, "graph.onnx")
    hf_import.export_graph_onnx(ckpt, onnx_path, trace_dtype="fp16")
    m = onnx.load(onnx_path, load_external_data=False)
    ext = {kv.value for i in m.graph.initializer for kv in i.external_data
           if kv.key == "location"}
    assert ext and all(e.endswith(".safetensors") for e in ext), ext
    _decoder_vs_hf(ckpt, onnx_path, lambda mod: mod.model.embed_tokens)


def test_qwen3vl_language_model_matches_hf(tmp_path):
    ckpt = _qwen3vl(str(tmp_path / "ckpt"))
    onnx_path = os.path.join(ckpt, "graph.onnx")
    hf_import.export_graph_onnx(ckpt, onnx_path, trace_dtype="fp16")
    m = onnx.load(onnx_path, load_external_data=False)
    assert {(p.key, p.value) for p in m.metadata_props} >= {
        ("vkop_causal_mask:attention_bias", "1")}
    assert [i.name for i in m.graph.input][:7] == [
        "inputs_embeds", "position_ids", "attention_bias",
        "deepstack_embeds_0", "deepstack_embeds_1", "deepstack_embeds_2",
        "image_pad_mask"]

    model = transformers.Qwen3VLForConditionalGeneration.from_pretrained(
        ckpt, dtype=torch.float32, attn_implementation="eager").eval()
    tc = model.config.text_config
    L = 7
    ids = torch.randint(0, 900, (1, L),
                        generator=torch.Generator().manual_seed(2))
    with torch.no_grad():
        ref = model(input_ids=ids, use_cache=False).logits.numpy()
        emb = model.model.language_model.embed_tokens(ids).numpy()
    f = {"inputs_embeds": emb.astype(np.float16),
         "position_ids": np.arange(L, dtype=np.int64)[None, None].repeat(3, 0),
         "attention_bias": np.triu(np.full((L, L), -65504, np.float16),
                                   1)[None, None],
         "image_pad_mask": np.zeros((1, L), bool)}
    for i in range(3):
        f[f"deepstack_embeds_{i}"] = np.zeros((0, tc.hidden_size), np.float16)
    for i in range(tc.num_hidden_layers):
        f[f"past_key_values_{i}"] = np.zeros(
            (1, 2, tc.num_key_value_heads, 0, tc.head_dim), np.float16)
    got = ort.InferenceSession(
        onnx_path, providers=["CPUExecutionProvider"]).run(None, f)[0]
    np.testing.assert_allclose(got.astype(np.float32), ref, atol=2e-3, rtol=0)


def test_registry_routes_by_model_type():
    names = {"llama": "llama", "qwen3": "llama", "phi3": "phi3",
             "glm": "glm", "qwen3_vl": "qwen3vl"}
    for mtype, mod in names.items():
        assert hf_builders.REGISTRY[mtype] == mod
    with pytest.raises(hf_import.HFImportError):
        hf_builders.by_name("nope")
