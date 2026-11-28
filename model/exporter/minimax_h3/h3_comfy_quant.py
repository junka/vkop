"""ComfyUI repack pre-quantized checkpoints (`comfy_quant` entries) for the
MiniMax-H3 exporters.

The repack's `*_int8_convrot` files store each quantized Linear as

    <name>.weight        I8  [N, K]   codes of W @ blockdiag(H) (ConvRot)
    <name>.weight_scale  F32 [N, 1]   absmax/127 per output row
    <name>.comfy_quant   U8  json     {"format": "int8_tensorwise",
                                       "convrot": true, "convrot_groupsize": G}

with H the group-G regular Hadamard (`h3_export_fl2va.hadamard`), so
x @ W^T == (x @ H) @ (W @ H)^T. Import as-is: the graph rotates the input of
each such Linear online (`install_input_rotation`, one shared `convrot_h`
initializer) and the weight streams as q * scale (`int8_convrot_transforms`),
from which onnx2vkop's `-q int8` (per-output-channel absmax/127) gets the
repack's codes back exactly.

Exactness when the weight streams as a 16-bit float: the converter's scale is
amax(stream(q*s))/127 = stream(127 s)/127 =: s~, not s. Streaming
stream(q * s~) instead keeps every value within half an ulp (<= 2^-8 relative
in bf16) of q * s~, which for |q| <= 127 is < 0.5 code (127 * 2^-8 = 0.496), so
the codes round back exactly (checked on every streamed tensor); streaming
q * s directly flips ~1e-4 of the codes by one in bf16 (exact in fp16). The
cost is each row's scale moving by <= 2^-8 (bf16) / 2^-11 (fp16) relative.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from h3_export_fl2va import hadamard, rotate_in  # noqa: E402,F401


def ckpt_quant(path: str, prefix: str = "") -> dict:
    """{weight name: comfy_quant dict} for every `<prefix>...comfy_quant`."""
    from safetensors import safe_open
    out = {}
    with safe_open(path, "np") as f:
        for k in f.keys():
            if k.startswith(prefix) and k.endswith(".comfy_quant"):
                out[k[:-len("comfy_quant")] + "weight"] = json.loads(
                    bytes(f.get_tensor(k)).decode())
    return out


def int8_convrot_group(quant: dict) -> int:
    """The ConvRot group when `quant` is an int8_convrot file, 0 when no entry
    is ConvRot (a plain or differently quantized file); mixed is an error."""
    rot = {n: q for n, q in quant.items() if q.get("convrot")}
    if not rot:
        return 0
    bad = [n for n, q in quant.items()
           if q.get("format") != "int8_tensorwise" or not q.get("convrot")]
    groups = {q.get("convrot_groupsize") for q in rot.values()}
    if bad or len(groups) != 1:
        raise SystemExit(f"[int8_convrot] mixed layout: {bad[:3]}, groups {groups}")
    return int(groups.pop())


def _round_to(x: np.ndarray, dtype: str) -> np.ndarray:
    if dtype == "BF16":
        return torch.from_numpy(np.ascontiguousarray(x, np.float32)).to(
            torch.bfloat16).float().numpy()
    if dtype == "F16":
        return x.astype(np.float16).astype(np.float32)
    return x.astype(np.float32)


def stream_scale(s: np.ndarray, dtype: str) -> np.ndarray:
    """The per-row scale the converter will measure on a `dtype` stream whose
    row absmax is |q| = 127 (see the module docstring)."""
    s = np.asarray(s, np.float32)
    return (_round_to(np.float32(127) * s, dtype) / np.float32(127)).astype(np.float32)


def int8_convrot_transforms(path: str, names, dtype: str = "BF16") -> dict:
    """Checkpoint weight name -> fp32 rewrite of its I8 payload: q * s~, so a
    `dtype` stream re-quantizes to exactly q (`export_streaming_real`'s
    `transforms`). Every row must reach |q| = 127."""
    from safetensors import safe_open
    f = safe_open(path, "np")

    def fn(q, n):
        s = f.get_tensor(n + "_scale").astype(np.float32).reshape(-1, 1)
        if q.shape[0] != s.shape[0]:
            raise SystemExit(f"[int8_convrot] {n}: scale {s.shape} vs {q.shape}")
        if not (np.abs(q).max(1) == 127).all():
            raise SystemExit(f"[int8_convrot] {n}: a row does not reach |q|=127")
        return q * stream_scale(s, dtype)

    return {n: (lambda q, n=n: fn(q, n)) for n in names}


def dequant(f, name: str) -> torch.Tensor:
    """fp32 q * scale (the file's own scale) from an open safe_open(..., "np")."""
    q = f.get_tensor(name).astype(np.float32)
    return torch.from_numpy(q * f.get_tensor(name + "_scale").astype(np.float32))


def install_input_rotation(model: torch.nn.Module, g: int, names, dtype,
                           device) -> int:
    """Register `convrot_h` [g, g] on `model` and a forward pre-hook on every
    submodule whose `<qualname>.weight` is in `names`: its input x becomes
    reshape(x, [-1, g]) @ H. Consecutive calls on the same input tensor (q/k/v,
    gate/up) reuse one rotation. The hook reads `model.convrot_h` when called,
    so a later load_state_dict(assign=True) is honored."""
    model.register_buffer("convrot_h", torch.empty(g, g, dtype=dtype,
                                                   device=device))
    names = set(names)
    last = {}

    def hook(mod, args):
        x = args[0]
        if last.get("x") is not x:
            h = model.convrot_h
            last["x"], last["y"] = x, torch.matmul(
                x.reshape(-1, g), h).reshape(x.shape)
        return (last["y"],) + tuple(args[1:])

    n = 0
    for qual, mod in model.named_modules():
        if qual + ".weight" in names:
            if mod.weight.shape[-1] % g:
                raise SystemExit(f"[int8_convrot] {qual}: K {mod.weight.shape} % {g}")
            mod.register_forward_pre_hook(hook)
            n += 1
    if n != len(names):
        raise SystemExit(f"[int8_convrot] hooked {n} of {len(names)} linears")
    return n


def hadamard_initializer(g: int, dtype: str = "BF16", name: str = "convrot_h"):
    """H as an ONNX initializer (+-1/sqrt(g) is exact in fp16 and bf16)."""
    import onnx
    h = hadamard(g)
    if dtype == "F16":
        return onnx.numpy_helper.from_array(h.astype(np.float16), name)
    if dtype == "BF16":
        t = onnx.helper.make_tensor(
            name, onnx.TensorProto.BFLOAT16, list(h.shape),
            (h.view(np.uint32) >> 16).astype(np.uint16).tobytes(), raw=True)
        return t
    return onnx.numpy_helper.from_array(h.astype(np.float32), name)


class Int8ConvRot:
    """One int8_convrot checkpoint, for an exporter that builds its model from
    the plain (unquantized) module tree:

        cr = Int8ConvRot.detect(ckpt, prefix)        # None unless int8_convrot
        kw = cr.attach(model, dtype, device)          # hooks + export kwargs
        ... sd[name] = cr.reference(name) if cr.owns(name) ...   (fp32 ref)
        export_streaming_real(..., **kw)
    """

    def __init__(self, path: str, quant: dict, g: int):
        self.path, self.quant, self.g = path, quant, g
        self.names = []
        self._f = None

    @classmethod
    def detect(cls, path: str, prefix: str = ""):
        quant = ckpt_quant(path, prefix)
        g = int8_convrot_group(quant)
        if not g:
            return None
        print(f"[int8_convrot] {len(quant)} pre-quantized linears, group {g}")
        return cls(path, quant, g)

    def attach(self, model, dtype, device, stream_dtype: str = "BF16") -> dict:
        """Rotate the inputs of `model`'s quantized linears; return the
        `export_streaming_real` kwargs (q * s~ transforms, H, stream dtype)."""
        have = set(model.state_dict())
        self.names = sorted(n for n in self.quant if n in have)
        install_input_rotation(model, self.g, self.names, dtype, device)
        return dict(target_dtype=stream_dtype,
                    transforms=int8_convrot_transforms(self.path, self.names,
                                                       stream_dtype),
                    computed={"convrot_h": hadamard_initializer(self.g,
                                                                stream_dtype)})

    def owns(self, name: str) -> bool:
        return name == "convrot_h" or name in self.quant

    def reference(self, name: str) -> torch.Tensor:
        """fp32 value for the reference model: H, or q * scale (file scale)."""
        if name == "convrot_h":
            return torch.from_numpy(hadamard(self.g))
        if self._f is None:
            from safetensors import safe_open
            self._f = safe_open(self.path, "np")
        return dequant(self._f, name)
