#!/usr/bin/env python3
"""MiniMax-H3 Ref2VA: the Qwen3-VL-32B vision tower (ComfyUI repack) -> ONNX.

The repack's text-encoder files (`qwen3vl_32b_minimax_h3_{nvfp4_awq,bf16}`)
both carry the full bf16 `visual.*` tower: 27 ViT blocks (hidden 1152, 16
heads, MLP 4304 gelu-tanh), patch 16 x temporal 2, 2x2 spatial merge, a
48x48 learned position grid, the main merger and three DeepStack mergers
(after blocks 8, 16, 24), out dim 5120. 595M parameters, 1.19 GB.

The graph is llm/exporter/export/qwen3vl_visual_dynamic.py's VisualDynamic
around transformers' Qwen3VLVisionModel: everything that depends on the image
size is computed in-graph from `grid_thw`, so one graph serves every
reference size (grid sides even, <= 256 patches, i.e. <= 4096 px).

  graph inputs
    pixel_values   [S, 1536] fp32   patches of ONE vision block (S = gh*gw),
                                    h3_ref2va_prep.qwen_image_patches
    grid_thw       [1, 3]    int64  [1, gh, gw]
  graph outputs (S/4 rows each, 2x2 merge-block order = token order)
    image_features        [S/4, 5120]   replaces the block's pad rows in inputs_embeds
    deepstack_features_0  [S/4, 5120]   added after text layer 0 at those rows
    deepstack_features_1  [S/4, 5120]   ... after layer 1
    deepstack_features_2  [S/4, 5120]   ... after layer 2

The vision tower attends within one block only (cu_seqlens per image), so a
prompt with several images / video blocks runs the graph once per block.

Usage:
  python h3_export_ref2va_vision.py --ckpt .../qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors \
      --out ~/models/h3_ref2va/vision/vision.onnx \
      [--reference ref.npz --inputs <h3_ref2va_prep qwen dir>]
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..", "..", "..", "llm",
                                                 "exporter", "export")))

VISION_CFG = dict(depth=27, hidden_size=1152, intermediate_size=4304, num_heads=16,
                  in_channels=3, patch_size=16, temporal_patch_size=2,
                  spatial_merge_size=2, out_hidden_size=5120,
                  num_position_embeddings=2304, deepstack_visual_indexes=[8, 16, 24],
                  hidden_act="gelu_pytorch_tanh")
OUTPUT_NAMES = ["image_features", "deepstack_features_0", "deepstack_features_1",
                "deepstack_features_2"]
INPUT_NAMES = ["pixel_values", "grid_thw"]


def load_visual(ckpt, dtype):
    """transformers' Qwen3VLVisionModel with the checkpoint's `visual.*`."""
    from safetensors import safe_open
    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLVisionConfig
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel
    cfg = Qwen3VLVisionConfig(**VISION_CFG)
    cfg._attn_implementation = "eager"
    with torch.device("meta"):
        visual = Qwen3VLVisionModel(cfg)
    sd = {}
    with safe_open(ckpt, "pt") as f:
        names = [k for k in f.keys() if k.startswith("visual.")]
        for k in names:
            sd[k[len("visual."):]] = f.get_tensor(k).to(dtype)
    missing, unexpected = visual.load_state_dict(sd, strict=False, assign=True)
    if missing or unexpected:
        raise SystemExit(f"visual weights: missing {missing} unexpected {unexpected}")
    # Non-persistent buffers were built on meta; rebuild them (theta 1e4, dim 72/2).
    rot = visual.rotary_pos_emb
    inv, _ = rot.compute_axial_rope_parameters(cfg)
    rot.inv_freq = inv.float()
    rot.original_inv_freq = inv.float().clone()
    return visual.eval()


class VisionGraph(nn.Module):
    """fp32 in / fp32 out around VisualDynamic (which runs at the weights'
    dtype)."""

    def __init__(self, visual):
        super().__init__()
        from qwen3vl_visual_dynamic import VisualDynamic
        self.vd = VisualDynamic(visual)

    def forward(self, pixel_values, grid_thw):
        return tuple(o.float() for o in self.vd(pixel_values, grid_thw))


def hf_reference(visual, patches, grid):
    """transformers' own forward (fp32): (image_features, [deepstack x3])."""
    out = visual(torch.from_numpy(patches), torch.tensor([grid], dtype=torch.int64))
    return out.pooler_output.float(), [d.float() for d in out.deepstack_features]


def widen_to_fp32(src, dst):
    """Rewrite a bf16-traced graph as fp32: bf16 initializers widened (exact),
    Cast-to-bf16 retargeted to fp32, value types widened. The weights go to
    `<dst stem>.weights.bin` one tensor at a time. Equivalent to tracing at
    fp32, without holding the 2.4 GB fp32 model (the exporter peak stays at
    the bf16 model's ~3.7 GB)."""
    import onnx
    from onnx import TensorProto
    BF16, F32 = TensorProto.BFLOAT16, TensorProto.FLOAT
    m = onnx.load(src)
    wfile = os.path.splitext(os.path.basename(dst))[0] + ".weights.bin"
    n_init = 0
    with open(os.path.join(os.path.dirname(os.path.abspath(dst)), wfile), "wb") as out:
        for t in m.graph.initializer:
            if t.data_type == BF16:
                a = onnx.numpy_helper.to_array(t).astype(np.float32)
                t.ClearField("raw_data")
                t.ClearField("int32_data")
                t.data_type = F32
                raw = a.tobytes()
            elif t.data_type == F32 and t.raw_data and len(t.raw_data) >= 4096:
                raw = t.raw_data
                t.ClearField("raw_data")
            else:
                continue
            off = out.tell()
            out.write(raw)
            t.data_location = TensorProto.EXTERNAL
            del t.external_data[:]
            for k, v in (("location", wfile), ("offset", str(off)), ("length", str(len(raw)))):
                kv = t.external_data.add()
                kv.key, kv.value = k, v
            n_init += 1
    for node in m.graph.node:
        for att in node.attribute:
            if node.op_type == "Cast" and att.name == "to" and att.i == BF16:
                att.i = F32
            if att.type == onnx.AttributeProto.TENSOR and att.t.data_type == BF16:
                att.t.CopyFrom(onnx.numpy_helper.from_array(
                    onnx.numpy_helper.to_array(att.t).astype(np.float32), att.t.name))
    for vi in list(m.graph.input) + list(m.graph.output) + list(m.graph.value_info):
        if vi.type.tensor_type.elem_type == BF16:
            vi.type.tensor_type.elem_type = F32
    onnx.save(m, dst)
    print(f"[widen] {dst}: {n_init} initializer(s) -> {wfile}")


def cos(a, b):
    a, b = np.asarray(a, np.float64).ravel(), np.asarray(b, np.float64).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default=None, help="vision.onnx to write")
    ap.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16",
                    help="graph width: bf16 (traced as is) or fp32 (the bf16 trace widened, external weights; what --act fp16|fp32 conversions need)")
    ap.add_argument("--reference", default=None,
                    help="fp32 transformers reference for every block of --inputs, "
                         "saved as <block>_{pixel_values,grid_thw,image_features,"
                         "deepstack_features_i}; also checks the VisualDynamic port")
    ap.add_argument("--inputs", default=None,
                    help="an h3_ref2va_prep.py qwen output dir")
    args = ap.parse_args(argv)

    import h3_ref2va_prep as prep
    if args.reference:
        if not args.inputs:
            raise SystemExit("--reference needs --inputs")
        d = prep.read_qwen_inputs(args.inputs)
        visual = load_visual(args.ckpt, torch.float32)
        port = VisionGraph(visual).eval()
        ref = {}
        with torch.no_grad():
            for i, (b, p) in enumerate(zip(d["meta"]["blocks"], d["patches"])):
                grid = b["grid"]
                feat, deep = hf_reference(visual, p, grid)
                mine = port(torch.from_numpy(p), torch.tensor([grid], dtype=torch.int64))
                for name, x, y in zip(OUTPUT_NAMES, [feat] + deep, mine):
                    ref[f"b{i}_{name}"] = x.numpy()
                    print(f"[ref] block {i} grid {grid} {name} {tuple(x.shape)} "
                          f"rms {x.pow(2).mean().sqrt():.4f} absmax {x.abs().max():.2f} "
                          f"port cos {cos(x, y):.8f} maxdiff {(x - y).abs().max():.2e}")
                ref[f"b{i}_pixel_values"] = p
                ref[f"b{i}_grid_thw"] = np.asarray([grid], np.int64)
        np.savez(args.reference, **ref)
        print(f"[ref] wrote {args.reference}")
        del visual, port

    if args.out:
        # Traced at bf16 (1.19 GB inline, under protobuf's 2 GB); --dtype fp32
        # then widens it (see widen_to_fp32).
        visual = load_visual(args.ckpt, torch.bfloat16)
        g = VisionGraph(visual).eval()
        gh, gw = 18, 32
        pv = torch.randn(gh * gw, 1536)
        grid = torch.tensor([[1, gh, gw]], dtype=torch.int64)
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        traced = args.out if args.dtype == "bf16" else args.out + ".bf16.tmp"
        with torch.no_grad():
            torch.onnx.export(
                g, (pv, grid), traced, input_names=INPUT_NAMES,
                output_names=OUTPUT_NAMES,
                dynamic_axes={"pixel_values": {0: "S"},
                              **{n: {0: "N"} for n in OUTPUT_NAMES}},
                opset_version=17, dynamo=False)
        del g, visual
        if args.dtype == "fp32":
            widen_to_fp32(traced, args.out)
            os.unlink(traced)
        print(f"[export] wrote {args.out} ({os.path.getsize(args.out) / 2**20:.0f} MiB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
