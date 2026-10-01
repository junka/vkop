#!/usr/bin/env python3
"""Back-fill value_info into the DiT ONNX graphs so onnx2vkop records real shapes.

Without shape inference the converter bakes an empty dims list for every
intermediate, so the runtime shape chain (/Shape -> /Gather -> /Unsqueeze ->
/Reshape) loses ranks: /Unsqueeze sees a rank-0 input with axes=[1], normalizes
the axis against out_rank=1, and walks off the end of the input shape vector.
The tiny graph needs exactly this pass (dit_decode_tiny_si.onnx aligns with
ORT; dit_decode_tiny.onnx does not), so the 7.12B pair gets it too.

External data is never loaded — only the proto metadata is rewritten, and the
weights stay in dit_{mode}.weights.bin.

    python shape_infer_dit.py prefill decode
"""

import sys
from pathlib import Path

import onnx

HERE = Path(__file__).resolve().parent


def infer(suffix):
    src = HERE / f"dit_{suffix}.onnx"
    dst = HERE / f"dit_{suffix}_si.onnx"
    m = onnx.load(str(src), load_external_data=False)
    m = onnx.shape_inference.infer_shapes(m)
    onnx.save(m, str(dst))

    n_sym = sum(1 for i in list(m.graph.input) + list(m.graph.output)
                for d in i.type.tensor_type.shape.dim if d.HasField("dim_param"))
    print(f"[{suffix}] {dst.name}: value_info={len(m.graph.value_info)} "
          f"symbolic io dims={n_sym}")

    # The shape-chain ops are where unknown ranks used to collapse to rank 0.
    for n in m.graph.node:
        if n.op_type in ("Unsqueeze", "Squeeze") and n.name in (
                "/Unsqueeze", "/Unsqueeze_1", "/Unsqueeze_2"):
            print(f"    {n.op_type} {n.name} inputs={list(n.input)}")


if __name__ == "__main__":
    for arg in sys.argv[1:] or ["prefill", "decode"]:
        infer(arg)
