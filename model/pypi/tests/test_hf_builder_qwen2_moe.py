#!/usr/bin/env python3
"""`--hf-checkpoint` for Qwen2-MoE (hf_builders/qwen2_moe.py).

The builder's graph is the stage-1 expert-paging graph: slot s wired to expert
s, weighted by the s-th top-k probability. So:

  * with ONE expert (top-1) the routing is trivial and the graph must equal
    HF's logits exactly (fp32, ORT), prefill and a decode step;
  * with several experts it must equal the SAME wrapper run eagerly on the
    real model (make_wrapper with the wired experts read from HF's fused
    parameters) -- which proves the per-expert checkpoint aliasing
    (`mlp.experts.s.*_proj.weight`) picks HF's expert s, gate/up in order;
  * post_splice must declare the router TopK indices as int32 graph outputs
    and set the causal-mask metadata the converter reads.

ORT's TopK emits int64, so the int32 declarations are stripped from the copy
ORT runs (they are for the vkop converter). ORT also refuses external paths
outside the model dir, so the ONNX goes inside the checkpoint.
"""
import os

import numpy as np
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
onnx = pytest.importorskip("onnx")
ort = pytest.importorskip("onnxruntime")

from onnx2vkop import hf_import  # noqa: E402
from onnx2vkop.hf_builders import qwen2_moe  # noqa: E402


def _tiny(out, experts, top_k):
    torch.manual_seed(0)
    cfg = transformers.Qwen2MoeConfig(
        vocab_size=256, hidden_size=64, intermediate_size=128,
        moe_intermediate_size=32, shared_expert_intermediate_size=48,
        num_experts=experts, num_experts_per_tok=top_k, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=128, tie_word_embeddings=False,
        norm_topk_prob=False)
    m = transformers.Qwen2MoeForCausalLM(cfg)
    with torch.no_grad():
        for n, p in m.named_parameters():
            if "norm" in n:
                p.add_(torch.randn_like(p) * 0.2)
            if n.endswith(".bias"):
                p.normal_(0, 0.02)
    m.to(torch.float32).save_pretrained(out, safe_serialization=True,
                                        max_shard_size="100KB")
    return out


def _export(ckpt):
    path = os.path.join(ckpt, "graph.onnx")
    hf_import.export_graph_onnx(ckpt, path, trace_dtype="fp32")
    m = onnx.load(path, load_external_data=False)
    topk = {n.output[1] for n in m.graph.node if n.op_type == "TopK"}
    return path, m, topk


def _ort_session(ckpt, m, topk):
    keep = [o for o in m.graph.output if o.name not in topk]
    del m.graph.output[:]
    m.graph.output.extend(keep)
    keep = [v for v in m.graph.value_info if v.name not in topk]
    del m.graph.value_info[:]
    m.graph.value_info.extend(keep)
    path = os.path.join(ckpt, "ort.onnx")
    with open(path, "wb") as f:
        f.write(m.SerializeToString())
    return ort.InferenceSession(path, providers=["CPUExecutionProvider"])


def _feeds(cfg, emb, S, past=None):
    nkv = cfg.num_key_value_heads
    hd = cfg.hidden_size // cfg.num_attention_heads
    f = {"inputs_embeds": emb.astype(np.float32)}
    if past is None:
        f["position_ids"] = np.arange(S, dtype=np.int64)[None]
        f["attention_bias"] = np.triu(np.full((S, S), np.finfo(np.float32).min,
                                              np.float32), 1)[None, None]
        for i in range(cfg.num_hidden_layers):
            f[f"past_key_values_{i}"] = np.zeros((1, 2, nkv, 0, hd),
                                                 np.float32)
    else:
        f["position_ids"] = np.array([[S]], np.int64)
        f["attention_bias"] = np.zeros((1, 1, 1, S + 1), np.float32)
        for i in range(cfg.num_hidden_layers):
            f[f"past_key_values_{i}"] = past[i]
    return f


def test_one_expert_matches_hf(tmp_path):
    ckpt = _tiny(str(tmp_path / "ckpt"), experts=1, top_k=1)
    _path, m, topk = _export(ckpt)
    # post_splice: router indices as int32 graph outputs + mask metadata
    assert len(topk) == 2
    outs = {o.name: o for o in m.graph.output}
    for n in topk:
        assert outs[n].type.tensor_type.elem_type == onnx.TensorProto.INT32
    assert ("vkop_causal_mask:attention_bias", "1") in [
        (p.key, p.value) for p in m.metadata_props]
    # wired experts referenced in place in the per-expert shard tensors
    ext = {i.name: {kv.key: kv.value for kv in i.external_data}
           for i in m.graph.initializer
           if i.data_location == onnx.TensorProto.EXTERNAL}
    assert ext["layers.1.mlp.wired_down_0"]["location"].endswith(
        ".safetensors")

    sess = _ort_session(ckpt, m, topk)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        ckpt, dtype=torch.float32, attn_implementation="eager").eval()
    cfg = model.config
    S = 5
    ids = torch.randint(0, cfg.vocab_size, (1, S + 1),
                        generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        ref = model(input_ids=ids, use_cache=False).logits.numpy()
        emb = model.model.embed_tokens(ids).numpy()
    o = sess.run(None, _feeds(cfg, emb[:, :S], S))
    np.testing.assert_allclose(o[0], ref[:, :S], atol=1e-5, rtol=0)
    o2 = sess.run(None, _feeds(cfg, emb[:, S:], S, past=o[1:]))
    np.testing.assert_allclose(o2[0], ref[:, S:], atol=1e-5, rtol=0)


def test_wired_experts_match_eager_wrapper(tmp_path):
    ckpt = _tiny(str(tmp_path / "ckpt"), experts=4, top_k=2)
    _path, m, topk = _export(ckpt)
    sess = _ort_session(ckpt, m, topk)

    model = transformers.AutoModelForCausalLM.from_pretrained(
        ckpt, dtype=torch.float32, attn_implementation="eager").eval()
    cfg = model.config
    inter = cfg.moe_intermediate_size
    fused = [(l.mlp.experts.gate_up_proj.detach().clone(),
              l.mlp.experts.down_proj.detach().clone())
             for l in model.model.layers]

    def load_expert(layer, e):
        gu, dp = fused[layer]
        return gu[e, :inter], gu[e, inter:], dp[e]

    with torch.no_grad():
        S = 5
        ids = torch.randint(0, cfg.vocab_size, (1, S),
                            generator=torch.Generator().manual_seed(2))
        emb = model.model.embed_tokens(ids)
        w = qwen2_moe.make_wrapper(model, load_expert=load_expert)
        f = _feeds(cfg, emb.numpy(), S)
        args = [torch.from_numpy(f["inputs_embeds"]),
                torch.from_numpy(f["position_ids"]),
                torch.from_numpy(f["attention_bias"])]
        args += [torch.from_numpy(f[f"past_key_values_{i}"])
                 for i in range(cfg.num_hidden_layers)]
        eager = w(*args)[0].numpy()
    got = sess.run(None, f)[0]
    np.testing.assert_allclose(got, eager, atol=1e-5, rtol=0)


def test_fused_expert_checkpoint_refused(tmp_path):
    """A checkpoint without per-expert tensors cannot be aliased slot-wise."""
    ckpt = _tiny(str(tmp_path / "ckpt"), experts=2, top_k=1)
    from onnx2vkop.stream_export import SafetensorsIndex
    index = SafetensorsIndex(ckpt)
    for k in [k for k in index.entries if ".mlp.experts." in k]:
        del index.entries[k]
    config = transformers.AutoConfig.from_pretrained(ckpt)
    with pytest.raises(hf_import.HFImportError, match="per-expert"):
        qwen2_moe.build(config, torch.float32, index)
