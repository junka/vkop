"""Pieces every LLM builder shares, real-weight export and meta trace alike.

A builder module (llama.py, phi3.py, glm.py, ...) owns the architecture: a
`make_wrapper(model)` that turns an HF model into the module the graph is
traced from. That function must work on BOTH

  * a real model (`from_pretrained`), which the in-tree export scripts under
    llm/exporter/export/ feed to `torch.onnx.export` as before, and
  * a meta model (`from_config` under `torch.device("meta")`), which
    `onnx2vkop --hf-checkpoint` traces with no weight bytes and splices to the
    checkpoint's shards (hf_import.py).

One implementation, two drivers: the export script and the CLI cannot drift.

The only thing that needed changing to make one wrapper serve both is RoPE.
HF's rotary modules keep `inv_freq` as a non-persistent buffer computed in
`__init__` (so a meta build has no value for it), move it to the input's
device, and run under autocast -- none of which a meta trace survives.
`StaticRope` does the same arithmetic (`freqs = inv_freq x pos`,
`cat(freqs, freqs)`, cos/sin times `attention_scaling`) from an `inv_freq`
taken from the HF rotary module of the same config, built for real on CPU.
"""
from __future__ import annotations

import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from ..hf_import import GraphSpec, HFImportError, buffers_to_params


# ---------------------------------------------------------------------------
# model construction
# ---------------------------------------------------------------------------

def text_config(config):
    return config.get_text_config() if hasattr(config, "get_text_config") \
        else config


def meta_model(model_cls, config, dtype):
    """`model_cls.from_config(config)` with no storage, in eager attention."""
    try:
        config._attn_implementation = "eager"
    except Exception:
        pass
    # Auto classes spell it from_config, concrete ones _from_config.
    make = getattr(model_cls, "from_config", None) or model_cls._from_config
    with torch.device("meta"):
        try:
            model = make(config, dtype=dtype)
        except TypeError:
            model = make(config, torch_dtype=dtype)
    return model.to(dtype).eval()


def is_meta(module) -> bool:
    for p in module.parameters():
        return p.device.type == "meta"
    return False


# ---------------------------------------------------------------------------
# RoPE
# ---------------------------------------------------------------------------

class StaticRope(nn.Module):
    """cos/sin from position_ids, the HF default-rotary arithmetic.

    `inv_freq` is a Parameter, not a buffer, so a meta build traces it as a
    named graph input that `GraphSpec.constants` fills (see `llm_spec`); on a
    real model it is an ordinary initializer. `value` keeps the real table
    either way.
    """

    def __init__(self, inv_freq: torch.Tensor, attention_scaling: float,
                 device=None):
        super().__init__()
        self.value = inv_freq.detach().float().cpu().clone()
        self.inv_freq = nn.Parameter(self.value.to(device or "cpu"),
                                     requires_grad=False)
        self.attention_scaling = float(attention_scaling)

    @classmethod
    def from_hf(cls, rotary: nn.Module, device=None) -> "StaticRope":
        """Take inv_freq/attention_scaling from an HF rotary module.

        A meta one has no value, so its class is instantiated again on CPU
        from the same config -- cheap, and it applies the config's rope type
        (llama3, yarn, longrope's short factor, ...) exactly as HF does.
        """
        inv = getattr(rotary, "inv_freq", None)
        if inv is None or inv.device.type == "meta":
            cfg = getattr(rotary, "config", None)
            if cfg is None:
                raise HFImportError(f"{type(rotary).__name__} keeps no config "
                                    f"to rebuild inv_freq from")
            with torch.device("cpu"):
                rotary = type(rotary)(config=cfg)
            inv = rotary.inv_freq
        scale = float(getattr(rotary, "attention_scaling", 1.0))
        if device is None:
            device = inv.device
        return cls(inv, scale, device=device)

    def hf_constants(self):
        return {"inv_freq": self.value.numpy()}

    def forward(self, position_ids, dtype):
        inv = self.inv_freq[None, :, None].float()
        inv = inv.expand(position_ids.shape[0], -1, 1)
        freqs = (inv @ position_ids[:, None, :].float()).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos() * self.attention_scaling
        sin = emb.sin() * self.attention_scaling
        return cos.to(dtype), sin.to(dtype)


def rope_for(model, rotary_path: str = "model.rotary_emb") -> StaticRope:
    obj = model
    for p in rotary_path.split("."):
        obj = getattr(obj, p)
    return StaticRope.from_hf(obj, device="meta" if is_meta(model) else None)


def refuse_dynamic_rope(cfg) -> str:
    params = (getattr(cfg, "rope_parameters", None)
              or getattr(cfg, "rope_scaling", None) or {})
    rtype = (params.get("rope_type") or params.get("type") or "default") \
        if isinstance(params, dict) else "default"
    if rtype == "dynamic":
        raise HFImportError(
            "dynamic-NTK RoPE recomputes inv_freq from the sequence length at "
            "run time; the static graph cannot carry that")
    return rtype


# ---------------------------------------------------------------------------
# the in-tree LLM I/O contract
# ---------------------------------------------------------------------------

def llm_io(nlayers: int, hidden: int, num_kv: int, head_dim: int, dtype,
           *, seq: int = 8, device="cpu", random: bool = True):
    """Dummies + names + dynamic axes of qwen3_export_onnx.py's contract.

      in : inputs_embeds (B,q,H), position_ids (B,q) int64,
           attention_bias (B,1,q,kv), past_key_values_i (B,2,nkv,kv,hd)
      out: logits (B,q,V), present_key_values_i (B,2,nkv,kv,hd)
    """
    B = 1
    if device == "meta":
        emb = torch.empty(B, seq, hidden, dtype=dtype, device="meta")
        pos = torch.empty(B, seq, dtype=torch.long, device="meta")
        bias = torch.empty(B, 1, seq, seq, dtype=dtype, device="meta")
        past = [torch.empty(B, 2, num_kv, 0, head_dim, dtype=dtype,
                            device="meta") for _ in range(nlayers)]
    else:
        emb = (torch.randn if random else torch.zeros)(B, seq, hidden,
                                                       dtype=dtype)
        pos = torch.arange(seq, dtype=torch.long).unsqueeze(0).expand(B, -1)
        causal = torch.triu(torch.full((seq, seq), torch.finfo(dtype).min,
                                       dtype=dtype), diagonal=1)
        bias = causal.unsqueeze(0).unsqueeze(0)
        past = [torch.zeros(B, 2, num_kv, 0, head_dim, dtype=dtype)
                for _ in range(nlayers)]
    dummies = (emb, pos, bias, *past)
    input_names = (["inputs_embeds", "position_ids", "attention_bias"]
                   + [f"past_key_values_{i}" for i in range(nlayers)])
    output_names = (["logits"]
                    + [f"present_key_values_{i}" for i in range(nlayers)])
    dyn = {"inputs_embeds": {0: "batch", 1: "seq"},
           "position_ids": {0: "batch", 1: "seq"},
           "attention_bias": {0: "batch", 2: "q_len", 3: "kv_len"},
           "logits": {0: "batch", 1: "seq"}}
    for i in range(nlayers):
        dyn[f"past_key_values_{i}"] = {0: "batch", 3: "kv_len"}
        dyn[f"present_key_values_{i}"] = {0: "batch", 3: "kv_len"}
    return dummies, input_names, output_names, dyn


# ---------------------------------------------------------------------------
# checkpoint naming
# ---------------------------------------------------------------------------

def text_prefix(index, tail: str = "layers.0.input_layernorm.weight"
                ) -> Optional[str]:
    """Checkpoint prefix of the text tower (`model.`, `model.language_model.`).

    Found from a first-layer tensor; a namesake under a vision tower or an MTP
    head is ignored, since those are what the prefix exists to exclude.
    """
    cands = [k[: -len(tail)] for k in index.entries if k.endswith(tail)]
    skip = ("visual", "vision", "mtp", "audio", "speech")
    cands = [c for c in cands if not any(s in c for s in skip)] or cands
    if not cands:
        return None
    for pref in ("model.language_model.", "model.", ""):
        if pref in cands:
            return pref
    return sorted(cands, key=len)[0]


def tied_head_alias(index, prefix: Optional[str]) -> Dict[str, str]:
    """`lm_head.weight` -> `<prefix>embed_tokens.weight` when the checkpoint
    ties them (stores only the embedding)."""
    if "lm_head.weight" in index.entries or any(
            k.endswith(".lm_head.weight") for k in index.entries):
        return {}
    tied = (prefix or "model.") + "embed_tokens.weight"
    if tied in index.entries:
        return {"lm_head.weight": tied}
    raise HFImportError(f"checkpoint has no lm_head.weight and no {tied} to "
                        f"tie it to")


def rope_constants(wrapper) -> Dict[str, np.ndarray]:
    """{state-dict name: value} for every computed table in `wrapper`.

    A module declares its computed parameters with `hf_constants()` ->
    {local parameter name: ndarray} (StaticRope's inv_freq, Qwen3-VL's mrope
    M): on meta they have no bytes and no checkpoint entry, so the splice
    writes them as inline initializers instead.
    """
    out = {}
    for name, mod in wrapper.named_modules():
        fn = getattr(mod, "hf_constants", None)
        if callable(fn):
            for local, value in fn().items():
                out[(name + "." if name else "") + local] = value
    return out


def llm_spec(wrapper, index, *, nlayers, hidden, num_kv, head_dim, dtype,
             summary=None, ckpt_prefix=None, aliases=None, constants=None,
             addend=None, post_splice=None) -> GraphSpec:
    """GraphSpec for a meta wrapper that speaks the `llm_io` contract."""
    converted = buffers_to_params(wrapper)
    if converted:
        raise HFImportError(f"unexpected buffer(s) {converted[:4]} in the "
                            f"wrapper; their values are unknown on meta")
    prefix = ckpt_prefix if ckpt_prefix is not None else text_prefix(index)
    al = dict(tied_head_alias(index, prefix))
    al.update(aliases or {})
    consts = rope_constants(wrapper)
    consts.update(constants or {})
    dummies, ins, outs, dyn = llm_io(nlayers, hidden, num_kv, head_dim,
                                     dtype, device="meta")
    s = dict(summary or {})
    s.update(ckpt_prefix=prefix, tied_lm_head="lm_head.weight" in al)
    return GraphSpec(wrapper=wrapper, dummies=dummies, input_names=ins,
                     output_names=outs, dynamic_axes=dyn, constants=consts,
                     aliases=al, ckpt_prefix=prefix, addend=dict(addend or {}),
                     post_splice=post_splice, summary=s)


def warn(msg: str) -> None:
    print(f"[hf] {msg}", file=sys.stderr)
