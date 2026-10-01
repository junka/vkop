#!/usr/bin/env python3
"""Export the tiny DiT pair with fully STATIC shapes (no dynamic_axes).

Rationale: with dynamic axes marked, torch keeps Shape/Slice/Concat shape-meta
subgraphs in the graph; vkop's runtime shape recomputation mis-schedules those
chains (present_kv_1 collapses to [2,128], corrupts memory). With static axes
the exporter's constant folding bakes every reshape spec into a Constant and
the dynamic-shape machinery disappears.

Writes dit_{prefill,decode}_tiny_static.onnx (+ .weights.bin) next to this
script. Then convert with:
    python convert_dit_to_vkop_static.py   (or the patched no-optimizer path)
"""

import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import qi21_export_onnx as E  # noqa: E402

_real_export = torch.onnx.export


def _static_export(*args, **kwargs):
    kwargs["dynamic_axes"] = {}
    return _real_export(*args, **kwargs)


def main():
    torch.set_grad_enabled(False)
    cfg, w = E.tiny_weights(torch.float16)
    torch.onnx.export = _static_export
    try:
        E.export_pair(cfg, w, HERE, 8, 1024, suffix="_tiny_static")
    finally:
        torch.onnx.export = _real_export

    import onnx
    for mode in ("prefill", "decode"):
        p = HERE / f"dit_{mode}_tiny_static.onnx"
        m = onnx.load(str(p), load_external_data=False)
        m = onnx.shape_inference.infer_shapes(m)
        onnx.save(m, str(p))
        dyn = [d.dim_param for i in list(m.graph.input) + list(m.graph.output)
               for tt in [i.type.tensor_type] for d in tt.shape.dim
               if d.HasField("dim_param")]
        print(f"[static] {p.name}: symbolic dims left = {dyn}")


if __name__ == "__main__":
    main()
