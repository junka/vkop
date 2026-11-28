#!/usr/bin/env python3
"""MiniMax-H3 FL2VA text tower (ComfyUI repack) -> ONNX.

`text_encoders/qwen3vl_32b_minimax_h3_bf16.safetensors` is Qwen3-VL-32B's
language model truncated to its first 50 layers (plus the vision tower, unused
for t2va). ComfyUI (comfy/text_encoders/minimax.py) conditions the DiT on the
UNNORMALIZED hidden state after layer 50: no final norm, no lm_head. For a
text-only prompt Qwen3-VL's interleaved MRoPE has equal (t, h, w) positions and
reduces to plain rotate-half RoPE (theta 5e6, head_dim 128).

  graph inputs
    inputs_embeds   [L, 5120] fp32 embed_tokens rows of the raw prompt tokens
                                   (no template, no specials). The 1.5 GB
                                   table is looked up on the host from
                                   embed_tokens.bf16 (raw [151936, 5120] bf16,
                                   written next to the graph), as image_gen
                                   does: a Gather over it would be hydrated
                                   into the protobuf and overflow 2 GB.
    attention_bias  [L, L]  fp32   causal additive mask (0 / large negative)
    rope_cos        [L, 128] fp32  cos(cat(f, f)), f = pos * inv_freq
    rope_sin        [L, 128] fp32
    deepstack_i     [L, 5120] fp32 (--deepstack only; i = 0, 1, 2, for each of
                                   text layers 0..2 inside the exported slice)
                                   Qwen3-VL DeepStack: added to the hidden
                                   state AFTER text layer i. Zero except the
                                   vision rows (Ref2VA); all zero = t2va.
  graph output
    hidden          [L, 5120]      layer-50 hidden state

With reference images (Ref2VA, FL2VA.md §3f) the host also replaces the
vision rows of inputs_embeds by the vision tower's image_features and builds
rope_cos/sin from the 3-D MRoPE positions (h3_ref2va_prep.py).

Usage: python h3_export_fl2va_te.py --ckpt .../qwen3vl_32b_minimax_h3_bf16.safetensors \
           --out ~/models/h3_fl2va_te/te.onnx [--num-layers 2 --reference ref.npz]
           [--deepstack [--ref2va-inputs DIR --vision-ref vref.npz]]
"""
from __future__ import annotations

import argparse
import math
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

CFG = dict(vocab=151936, hidden=5120, heads=64, kv_heads=8, head_dim=128,
           ffn=25600, layers=50, eps=1e-6, theta=5.0e6)


def rms_norm(x, w, eps):
    xf = x.float()
    y = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return y.to(x.dtype) * w.to(x.dtype)


class RMSNorm(nn.Module):
    def __init__(self, dim, eps, dtype, device):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.empty(dim, dtype=dtype, device=device),
                                   requires_grad=False)

    def forward(self, x):
        return rms_norm(x, self.weight, self.eps)


# --nvfp4: the projections whose AWQ smoothing was not folded into a norm and
# so carry `pre_quant_scale` (multiplies the input). Filled from the checkpoint.
NVFP4 = {"on": False, "pqs": set()}


class QLinear(nn.Module):
    """An NVFP4 Linear as traced: [x * pre_quant_scale] @ W^T. After the trace
    the W input is replaced by the two-level DequantizeLinear chain
    (nvfp4_splice), which onnx2vkop's PreQuantizedWeights imports."""

    def __init__(self, i, o, dtype, device, pqs):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(o, i, dtype=dtype, device=device),
                                   requires_grad=False)
        if pqs:
            self.pre_quant_scale = nn.Parameter(
                torch.empty(i, dtype=dtype, device=device), requires_grad=False)

    def forward(self, x):
        if hasattr(self, "pre_quant_scale"):
            x = x * self.pre_quant_scale.to(x.dtype)
        return torch.matmul(x, self.weight.to(x.dtype).t())


def lin(i, o, dtype, device, name=""):
    if NVFP4["on"]:
        return QLinear(i, o, dtype, device, name in NVFP4["pqs"])
    m = nn.Linear(i, o, bias=False, dtype=dtype, device=device)
    m.requires_grad_(False)
    return m


def rotate_half(x):
    h = x.shape[-1] // 2
    return torch.cat((-x[..., h:], x[..., :h]), dim=-1)


class Attention(nn.Module):
    def __init__(self, c, dtype, device):
        super().__init__()
        self.h, self.kv, self.d = c["heads"], c["kv_heads"], c["head_dim"]
        self.q_proj = lin(c["hidden"], self.h * self.d, dtype, device, "q_proj")
        self.k_proj = lin(c["hidden"], self.kv * self.d, dtype, device, "k_proj")
        self.v_proj = lin(c["hidden"], self.kv * self.d, dtype, device, "v_proj")
        self.o_proj = lin(self.h * self.d, c["hidden"], dtype, device, "o_proj")
        self.q_norm = RMSNorm(self.d, c["eps"], dtype, device)
        self.k_norm = RMSNorm(self.d, c["eps"], dtype, device)

    def forward(self, x, bias, cos, sin):
        L = x.shape[0]
        q = self.q_norm(self.q_proj(x).reshape(L, self.h, self.d))
        k = self.k_norm(self.k_proj(x).reshape(L, self.kv, self.d))
        v = self.v_proj(x).reshape(L, self.kv, self.d)
        q = q * cos + rotate_half(q) * sin
        k = k * cos + rotate_half(k) * sin
        rep = self.h // self.kv
        # HF repeat_kv: query head j reads kv head j // rep
        k = k.unsqueeze(2).expand(L, self.kv, rep, self.d).reshape(L, self.h, self.d)
        v = v.unsqueeze(2).expand(L, self.kv, rep, self.d).reshape(L, self.h, self.d)
        q, k, v = (t.transpose(0, 1).unsqueeze(0) for t in (q, k, v))  # [1,H,L,D]
        s = torch.matmul(q, k.transpose(-1, -2)) * (1.0 / math.sqrt(self.d)) + bias
        p = torch.softmax(s, dim=-1)
        o = torch.matmul(p, v).squeeze(0).transpose(0, 1).reshape(L, self.h * self.d)
        return self.o_proj(o)


class MLP(nn.Module):
    def __init__(self, c, dtype, device):
        super().__init__()
        self.gate_proj = lin(c["hidden"], c["ffn"], dtype, device, "gate_proj")
        self.up_proj = lin(c["hidden"], c["ffn"], dtype, device, "up_proj")
        self.down_proj = lin(c["ffn"], c["hidden"], dtype, device, "down_proj")

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Layer(nn.Module):
    def __init__(self, c, dtype, device):
        super().__init__()
        self.input_layernorm = RMSNorm(c["hidden"], c["eps"], dtype, device)
        self.post_attention_layernorm = RMSNorm(c["hidden"], c["eps"], dtype, device)
        self.self_attn = Attention(c, dtype, device)
        self.mlp = MLP(c, dtype, device)

    def forward(self, x, bias, cos, sin):
        x = x + self.self_attn(self.input_layernorm(x), bias, cos, sin)
        return x + self.mlp(self.post_attention_layernorm(x))


class Tower(nn.Module):
    def __init__(self, c, dtype, device, layer_ids):
        super().__init__()
        # Keyed by the checkpoint's layer index, so a slice (layers 25..49)
        # keeps its tensor names (model.layers.25.*) for the splice.
        self.layers = nn.ModuleDict({str(i): Layer(c, dtype, device)
                                     for i in layer_ids})


# Qwen3-VL DeepStack: the vision tower's 3 extra feature sets are added to the
# hidden state after text layers 0, 1, 2 (comfy llama.py Llama2_.forward,
# HF Qwen3VLTextModel._deepstack_process), at the vision rows only.
DEEPSTACK_LAYERS = (0, 1, 2)


class TextTower(nn.Module):
    def __init__(self, c=CFG, dtype=torch.bfloat16, device="meta",
                 layer_ids=None, deepstack=False):
        super().__init__()
        self.dtype = dtype
        layer_ids = list(layer_ids if layer_ids is not None else range(c["layers"]))
        self.model = Tower(c, dtype, device, layer_ids)
        # the deepstack inputs this slice takes, by text layer index
        self.ds_layers = [i for i in DEEPSTACK_LAYERS if i in layer_ids] if deepstack else []

    def forward(self, inputs_embeds, attention_bias, rope_cos, rope_sin, *deepstack):
        dt = self.dtype
        x = inputs_embeds.to(dt)
        bias = attention_bias.to(dt).unsqueeze(0).unsqueeze(0)       # [1,1,L,L]
        cos = rope_cos.to(dt).unsqueeze(1)                          # [L,1,D]
        sin = rope_sin.to(dt).unsqueeze(1)
        for idx, layer in self.model.layers.items():
            x = layer(x, bias, cos, sin)
            if int(idx) in self.ds_layers:
                x = x + deepstack[self.ds_layers.index(int(idx))].to(dt)
        return x


INPUT_NAMES = ["inputs_embeds", "attention_bias", "rope_cos", "rope_sin"]
OUTPUT_NAMES = ["hidden"]
DYNAMIC_AXES = {"inputs_embeds": {0: "L"}, "attention_bias": {0: "L", 1: "L2"},
                "rope_cos": {0: "L"}, "rope_sin": {0: "L"}, "hidden": {0: "L"}}


def io_spec(tower):
    """(input names, dynamic axes) of a TextTower's graph."""
    ds = ["deepstack_%d" % i for i in tower.ds_layers]
    return INPUT_NAMES + ds, dict(DYNAMIC_AXES, **{n: {0: "L"} for n in ds})


def host_inputs(ids, table, c=CFG):
    """Every graph input for one prompt (the C++ driver builds the same).
    `table` is the bf16 embedding table as uint16 [vocab, hidden]."""
    L = len(ids)
    rows = (table[np.asarray(ids)].astype(np.uint32) << 16).view(np.float32)
    inv = 1.0 / (c["theta"] ** (np.arange(0, c["head_dim"], 2, dtype=np.float32)
                                / c["head_dim"]))
    f = np.arange(L, dtype=np.float32)[:, None] * inv[None, :].astype(np.float32)
    emb = np.concatenate([f, f], -1)
    bias = np.triu(np.full((L, L), -3.0e38, np.float32), 1)
    return {"inputs_embeds": np.ascontiguousarray(rows),
            "attention_bias": bias,
            "rope_cos": np.cos(emb).astype(np.float32),
            "rope_sin": np.sin(emb).astype(np.float32)}


# ---------------------------------------------------------------- NVFP4 import
# The ComfyUI repack's `qwen3vl_32b_minimax_h3_nvfp4_awq` (comfy-kitchen
# layout): weight U8 [N, K/2] with the EVEN element in the HIGH nibble
# (hi_first), weight_scale e4m3 [N, K/16] in cuBLAS's blocked/swizzled tile
# order, weight_scale_2 fp32 scalar; value = e2m1 * block_scale * scale_2.
# ONNX FLOAT4E2M1 puts element 0 in the LOW nibble, so the bytes are swapped,
# and the scales are un-swizzled (comfy_kitchen.float_utils.from_blocked).

E2M1 = np.array([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6],
                np.float32)


def st_header(path):
    import json, struct
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n)), 8 + n


def st_bytes(path, header, base, name):
    e = header[name]
    a, b = e["data_offsets"]
    return np.memmap(path, np.uint8, "r", offset=base + a, shape=(b - a,))


def from_blocked(b, rows, cols):
    nrb, ncb = -(-rows // 128), -(-cols // 4)
    x = np.asarray(b).reshape(-1, 32, 4, 4).transpose(0, 2, 1, 3)
    x = x.reshape(nrb, ncb, 4, 32, 4).reshape(nrb, ncb, 128, 4).transpose(0, 2, 1, 3)
    return np.ascontiguousarray(x.reshape(nrb * 128, ncb * 4)[:rows, :cols])


def nvfp4_parts(path, header, base, name):
    """(low-nibble-first payload [N, K/2] u8, e4m3 bytes [N, K/16], g, N, K)."""
    q = st_bytes(path, header, base, name + ".weight")
    N, K2 = header[name + ".weight"]["shape"]
    K = 2 * K2
    q = np.asarray(q).reshape(N, K2)
    lo_first = ((q << 4) | (q >> 4)).astype(np.uint8)
    s = st_bytes(path, header, base, name + ".weight_scale")
    s = from_blocked(np.asarray(s), N, K // 16)
    g = np.frombuffer(bytes(st_bytes(path, header, base, name + ".weight_scale_2")),
                      np.float32)[0]
    return lo_first, s, float(g), N, K


def nvfp4_dequant(lo_first, s, g, N, K):
    codes = np.stack([lo_first & 15, lo_first >> 4], -1).reshape(N, K)
    s32 = torch.from_numpy(s.copy()).view(torch.float8_e4m3fn).float().numpy()
    return (E2M1[codes].reshape(N, K // 16, 16) * s32[:, :, None] * g).reshape(N, K)


def nvfp4_splice(proto, path, qnames, weights_path, weights_file):
    """Replace each traced W input by DequantizeLinear(W e2m1 [N, K],
    DequantizeLinear(s e4m3 [N, K/16], g), axis=1, block_size=16), the bytes
    appended to `weights_path` one tensor at a time."""
    import onnx
    from onnx import helper, TensorProto
    header, base = st_header(path)
    keep = [vi for vi in proto.graph.input if vi.name not in qnames]
    del proto.graph.input[:]
    proto.graph.input.extend(keep)
    new_nodes = []
    with open(weights_path, "ab") as out:
        for name in qnames:
            mod = name[: -len(".weight")]
            q, s, g, N, K = nvfp4_parts(path, header, base, mod)
            inits = []
            for suffix, arr, dtype, dims in ((".nvfp4_q", q, TensorProto.FLOAT4E2M1, [N, K]),
                                             (".nvfp4_s", s, TensorProto.FLOAT8E4M3FN,
                                              [N, K // 16])):
                off = out.tell()
                out.write(arr.tobytes())
                t = proto.graph.initializer.add()
                t.name, t.data_type = name + suffix, dtype
                t.dims.extend(dims)
                t.data_location = TensorProto.EXTERNAL
                for k, v in (("location", weights_file), ("offset", str(off)),
                             ("length", str(arr.nbytes))):
                    kv = t.external_data.add()
                    kv.key, kv.value = k, v
            gi = proto.graph.initializer.add()
            gi.CopyFrom(onnx.numpy_helper.from_array(np.array(g, np.float32),
                                                     name + ".nvfp4_g"))
            new_nodes.append(helper.make_node(
                "DequantizeLinear", [name + ".nvfp4_s", name + ".nvfp4_g"],
                [name + ".nvfp4_sf32"], name=name + ".nvfp4_dq_scale"))
            new_nodes.append(helper.make_node(
                "DequantizeLinear", [name + ".nvfp4_q", name + ".nvfp4_sf32"],
                [name], name=name + ".nvfp4_dq", axis=1, block_size=16))
    nodes = new_nodes + list(proto.graph.node)
    del proto.graph.node[:]
    proto.graph.node.extend(nodes)
    for op in proto.opset_import:
        if op.domain in ("", "ai.onnx"):
            op.version = max(op.version, 23)   # FLOAT4E2M1 / block_size


def export_nvfp4(wrapper, dummies, ckpt, out_path):
    """trace -> splice norms/pre_quant_scale -> splice NVFP4 chains -> save."""
    import onnx
    import h3_export_common as common
    from onnx2vkop import stream_export as SE
    out_dir = os.path.dirname(os.path.abspath(out_path))
    input_names, dynamic_axes = io_spec(wrapper)
    proto, shapes = common.trace_weightless(wrapper, dummies, input_names,
                                            OUTPUT_NAMES, dynamic_axes)
    qnames = [n for n in shapes if n.endswith("_proj.weight")]
    plain = {n: v for n, v in shapes.items() if n not in qnames}
    ck_dir = os.path.join(out_dir, "ckpt", "te_nvfp4")
    os.makedirs(ck_dir, exist_ok=True)
    link = os.path.join(ck_dir, os.path.basename(ckpt))
    if not os.path.lexists(link):
        os.symlink(os.path.abspath(ckpt), link)
    index = SE.SafetensorsIndex(ck_dir)
    weights_file = os.path.splitext(os.path.basename(out_path))[0] + ".weights.bin"
    wpath = os.path.join(out_dir, weights_file)
    resolved = SE.splice_initializers(proto, plain, index, weights_file=weights_file)
    if len(resolved) != len(plain):
        raise SystemExit(f"[nvfp4] spliced {len(resolved)} of {len(plain)} plain tensors")
    nbytes, offsets = SE.copy_weights(resolved, wpath)
    SE.patch_offsets(proto, offsets)
    nvfp4_splice(proto, ckpt, qnames, wpath, weights_file)
    SE.resolve_shape_dims(proto, dynamic_axes)
    onnx.save(proto, out_path)
    print(f"[nvfp4] wrote {out_path}: {len(qnames)} NVFP4 Linear(s), "
          f"{len(resolved)} plain tensor(s); {os.path.getsize(wpath) / 2**30:.2f} GiB")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--num-layers", type=int, default=CFG["layers"])
    ap.add_argument("--layers", default=None,
                    help="START,END: export only layers [START, END) -- the "
                         "50-layer tower is 24 GB at int8, more than a 36 GB "
                         "box can load beside the OS; two halves run in "
                         "sequence (inputs_embeds of the second = hidden of "
                         "the first)")
    ap.add_argument("--reference", default=None,
                    help="run the truncated tower on CPU (fp32) and save "
                         "inputs/output to this .npz")
    ap.add_argument("--text", default="A red fox runs through a snowy forest at dawn.")
    ap.add_argument("--nvfp4", action="store_true",
                    help="--ckpt is qwen3vl_32b_minimax_h3_nvfp4_awq: import its "
                         "NVFP4 weights as-is (two-level DequantizeLinear chains)")
    ap.add_argument("--embed", default=None,
                    help="bf16 embed_tokens.bf16 to look rows up in (default: "
                         "written from --ckpt, which needs a bf16 table)")
    ap.add_argument("--deepstack", action="store_true",
                    help="add the deepstack_i inputs (Ref2VA reference images, "
                         "FL2VA.md 3f); all-zero deepstack = the plain graph")
    ap.add_argument("--ref2va-inputs", default=None,
                    help="with --deepstack: an h3_ref2va_prep.py qwen dir; its "
                         "token list, MRoPE tables and vision rows replace "
                         "--text (needs --vision-ref)")
    ap.add_argument("--vision-ref", default=None,
                    help="h3_export_ref2va_vision.py --reference npz for "
                         "--ref2va-inputs: image_features / deepstack rows")
    args = ap.parse_args(argv)
    c = dict(CFG, layers=args.num_layers)
    layer_ids = range(c["layers"])
    if args.layers:
        lo, hi = (int(v) for v in args.layers.split(","))
        layer_ids = range(lo, hi)
    out_dir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(out_dir, exist_ok=True)

    if args.ref2va_inputs:
        import h3_ref2va_prep as prep
        if not (args.deepstack and args.vision_ref):
            raise SystemExit("--ref2va-inputs needs --deepstack and --vision-ref")
        r2v = prep.read_qwen_inputs(args.ref2va_inputs)
        ids = r2v["ids"]
    else:
        from transformers import Qwen2Tokenizer
        tok_dir = os.path.join(out_dir, "tokenizer")
        if not os.path.isdir(tok_dir):
            tok_dir = os.path.join(os.path.dirname(out_dir), "h3_fl2va_te", "tokenizer")
        tok = Qwen2Tokenizer.from_pretrained(tok_dir)
        ids = tok(args.text, add_special_tokens=False)["input_ids"]

    if args.nvfp4:
        header, _ = st_header(args.ckpt)
        NVFP4["on"] = True
        NVFP4["pqs"] = {k.split(".")[-2] for k in header
                        if k.startswith("model.layers.0.") and
                        k.endswith(".pre_quant_scale")}
        print(f"[nvfp4] pre_quant_scale on {sorted(NVFP4['pqs'])}")
    # The repack's int8_convrot file, auto-detected from its comfy_quant
    # entries and imported as-is (h3_comfy_quant; FL2VA.md §3g).
    import h3_comfy_quant
    i8cr = None if args.nvfp4 else h3_comfy_quant.Int8ConvRot.detect(
        args.ckpt, "model.layers.")
    if args.embed:
        # An existing bf16 table (the nvfp4 file stores embed_tokens as int8;
        # the bf16 checkpoint's table is the same weights unquantized -- AWQ
        # folds its smoothing into norms and weights, not into embeddings).
        table = np.memmap(args.embed, np.uint16, "r",
                          shape=(c["vocab"], c["hidden"]))
        inputs = host_inputs(ids, table, c)
    # The embedding table as a raw bf16 file the driver mmaps.
    import json, struct
    with open(args.ckpt, "rb") as f:
        hlen = struct.unpack("<Q", f.read(8))[0]
        e = json.loads(f.read(hlen))["model.embed_tokens.weight"]
    if args.embed:
        e = None
    else:
        assert e["dtype"] == "BF16" and e["shape"] == [c["vocab"], c["hidden"]], e
    table = None if args.embed else np.memmap(args.ckpt, np.uint16, "r", offset=8 + hlen + e["data_offsets"][0],
                      shape=(c["vocab"], c["hidden"]))
    emb_path = os.path.join(out_dir, "embed_tokens.bf16")
    if table is not None and (not os.path.exists(emb_path)
                              or os.path.getsize(emb_path) != table.nbytes):
        with open(emb_path, "wb") as dst:
            for r0 in range(0, c["vocab"], 8192):
                dst.write(np.ascontiguousarray(table[r0:r0 + 8192]).tobytes())
        print(f"[embed] wrote {emb_path} ({table.nbytes / 2**30:.2f} GiB)")
    if table is not None:
        inputs = host_inputs(ids, table, c)
    if args.ref2va_inputs:
        z = np.load(args.vision_ref)
        feats = [(z[f"b{i}_image_features"],
                  [z[f"b{i}_deepstack_features_{k}"] for k in range(3)])
                 for i in range(len(r2v["meta"]["blocks"]))]
        rows = table if table is not None else np.memmap(
            args.embed, np.uint16, "r", shape=(c["vocab"], c["hidden"]))
        inputs = prep.assemble_text_inputs(r2v, rows, feats)
    elif args.deepstack:
        for i in DEEPSTACK_LAYERS:
            inputs["deepstack_%d" % i] = np.zeros_like(inputs["inputs_embeds"])

    if args.reference:
        # fp32 on CPU, ONE layer resident at a time (a 32B layer is 1.8 GB fp32)
        from safetensors import safe_open
        if args.nvfp4:
            header, base = st_header(args.ckpt)
        h = torch.from_numpy(inputs["inputs_embeds"])
        rest = [torch.from_numpy(inputs[n]) for n in INPUT_NAMES[1:]]
        with safe_open(args.ckpt, "pt") as f:
            for li in layer_ids:
                # built on meta: load_state_dict(assign=True) supplies every
                # tensor, so no empty fp32 copy is allocated first
                model = TextTower(c, dtype=torch.float32, device="meta",
                                  layer_ids=[li], deepstack=args.deepstack).eval()
                if i8cr:
                    i8cr.attach(model, torch.float32, "cpu")
                sd = {}
                for name in model.state_dict():
                    if i8cr and i8cr.owns(name):
                        sd[name] = i8cr.reference(name)
                        continue
                    if args.nvfp4 and name.endswith("_proj.weight"):
                        sd[name] = torch.from_numpy(nvfp4_dequant(*nvfp4_parts(
                            args.ckpt, header, base, name[: -len(".weight")])))
                    else:
                        sd[name] = f.get_tensor(name).float()
                model.load_state_dict(sd, assign=True)
                ds = [torch.from_numpy(inputs["deepstack_%d" % i]) for i in model.ds_layers]
                with torch.no_grad():
                    h = model(h, *rest, *ds)
                del model, sd
        np.savez(args.reference, **inputs, hidden=h.numpy())
        print(f"[ref] {args.reference}: L={len(ids)} hidden absmax "
              f"{h.abs().max():.1f} rms {h.pow(2).mean().sqrt():.3f}")

    if args.nvfp4:
        wrapper = TextTower(c, device="meta", layer_ids=layer_ids,
                            deepstack=args.deepstack).eval()
        input_names, _ = io_spec(wrapper)
        dummies = tuple(torch.from_numpy(inputs[n]).to("meta") for n in input_names)
        with torch.no_grad():
            export_nvfp4(wrapper, dummies, args.ckpt, args.out)
        return 0

    import h3_export_common as common
    ck_dir = os.path.join(out_dir, "ckpt", "te")
    os.makedirs(ck_dir, exist_ok=True)
    link = os.path.join(ck_dir, os.path.basename(args.ckpt))
    for f in os.listdir(ck_dir):
        if os.path.join(ck_dir, f) != link:
            os.unlink(os.path.join(ck_dir, f))
    if not os.path.lexists(link):
        os.symlink(os.path.abspath(args.ckpt), link)
    wrapper = TextTower(c, device="meta", layer_ids=layer_ids,
                        deepstack=args.deepstack).eval()
    input_names, dynamic_axes = io_spec(wrapper)
    dummies = tuple(torch.from_numpy(inputs[n]).to("meta") for n in input_names)
    kw = i8cr.attach(wrapper, torch.bfloat16, "meta") if i8cr else {}
    with torch.no_grad():
        common.export_streaming_real(
            wrapper, dummies, input_names, OUTPUT_NAMES, dynamic_axes,
            model_dir=os.path.join(out_dir, "ckpt"), component="te",
            out_path=args.out, **kw)
    return 0


if __name__ == "__main__":
    sys.exit(main())
