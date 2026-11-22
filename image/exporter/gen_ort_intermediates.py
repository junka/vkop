#!/usr/bin/env python3
"""Run an ONNX graph on ORT with intermediates exposed, save them as .npz.

Pair with compare_node_dump.py to find the first node where a vkop
VKOP_NODE_DUMP run diverges. Works on any graph: inputs are raw little-endian
files whose dtype comes from the graph input (shape too, unless it is dynamic
and given as NAME=PATH:1x64x1x32x32).

The augmented graph is written next to the source with load_external_data off,
so it resolves the original weight file in place (no multi-GB copy) and is
deleted afterwards.

    # VAE: the stage boundaries BASELINE.md reports (--filter is a regex)
    python gen_ort_intermediates.py vae_decoder_512_clean.onnx vae_ref_intermediates.npz \
        --input latent=vae_latent.raw \
        --filter 'conv_in/Conv_output_0|mid_block/(resnets\.\d|attentions\.0)/Add_output_0' \
        --filter 'up_blocks\.\d/(Add|resnets\.2/Add|upsampler/.*Resize)_output_0' \
        --filter '(norm_out/Add|nonlinearity/Mul|conv_out/Conv)_output_0'

    # DiT prefill, every node output, fed from gen_dit_ref.py's ref/
    python gen_ort_intermediates.py dit_prefill_tiny_si.onnx dit_prefill_ref.npz \
        --input prompt_embeds=ref/prompt_embeds.raw --input cos=ref/cos_prefill.raw \
        --input sin=ref/sin_prefill.raw --input attention_bias=ref/bias_prefill.raw
"""
import argparse
import re
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper

NP = {TensorProto.FLOAT: np.float32, TensorProto.FLOAT16: np.float16,
      TensorProto.INT64: np.int64, TensorProto.INT32: np.int32,
      TensorProto.BOOL: np.bool_, TensorProto.UINT8: np.uint8, TensorProto.INT8: np.int8}


def load_feeds(graph_inputs, specs):
    feeds = {}
    for spec in specs:
        name, path = spec.split("=", 1)
        shape = None
        if ":" in path:
            path, s = path.rsplit(":", 1)
            shape = [int(d) for d in s.split("x")]
        vi = graph_inputs.get(name)
        if vi is None:
            raise SystemExit(f"no graph input named {name!r}; have {sorted(graph_inputs)}")
        tt = vi.type.tensor_type
        if shape is None:
            shape = [d.dim_value if d.HasField("dim_value") else -1 for d in tt.shape.dim]
        feeds[name] = np.fromfile(path, dtype=NP[tt.elem_type]).reshape(shape)
    return feeds


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("model")
    ap.add_argument("out", nargs="?", help="default: <model>_intermediates.npz")
    ap.add_argument("--input", action="append", default=[], metavar="NAME=PATH[:SHAPE]")
    ap.add_argument("--filter", action="append", default=[],
                    help="regex over tensor names to expose (default: every node output)")
    args = ap.parse_args()
    out = args.out or str(Path(args.model).with_suffix("")) + "_intermediates.npz"

    m = onnx.load(args.model, load_external_data=False)
    try:
        m = onnx.shape_inference.infer_shapes(m, strict_mode=False)
    except Exception as e:  # noqa: BLE001 - >2GB protos etc.; dtype falls back to float
        print(f"[warn] shape inference skipped: {e}")
    g = m.graph
    etype = {v.name: v.type.tensor_type.elem_type for v in list(g.value_info) + list(g.output)}
    existing = {o.name for o in g.output}
    rx = [re.compile(f) for f in args.filter]
    picked = [o for n in g.node for o in n.output
              if o and (not rx or any(r.search(o) for r in rx))]
    for name in picked:
        if name not in existing:
            g.output.append(helper.make_tensor_value_info(
                name, etype.get(name) or TensorProto.FLOAT, None))
    print(f"[info] exposing {len(picked)} intermediates")

    feeds = load_feeds({i.name: i for i in g.input}, args.input)
    tmp = Path(args.model).with_suffix(".allout.onnx")
    onnx.save(m, str(tmp))
    try:
        so = ort.SessionOptions()
        so.log_severity_level = 3
        sess = ort.InferenceSession(str(tmp), so, providers=["CPUExecutionProvider"])
        outs = sess.run(None, feeds)
    finally:
        tmp.unlink(missing_ok=True)

    ref = {}
    for o, v in zip(sess.get_outputs(), outs):
        v = np.asarray(v)
        ref[o.name] = v.astype(np.float32).ravel()
        ref[o.name + "@shape"] = np.asarray(v.shape)
    np.savez(out, **ref)
    print(f"[done] wrote {out} ({len(outs)} tensors)")


if __name__ == "__main__":
    main()
