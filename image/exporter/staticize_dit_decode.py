#!/usr/bin/env python3
"""Staticize the DiT decode graph by replacing symbolic dims with constants.

ONNX shape inference alone can't resolve intermediates built on dynamic ops
(Shape/Gather/Slice over symbolic axes like target_len, kv_len). This script
sets concrete input shapes and re-runs infer_shapes to fill value_info.

    python staticize_dit_decode.py --target_len 1024 --prefix_len 8

Writes dit_decode_static.onnx next to dit_decode_si.onnx.
"""

import argparse
from pathlib import Path

import numpy as np
import onnx

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target_len", type=int, default=1024)
    ap.add_argument("--prefix_len", type=int, default=8)
    args = ap.parse_args()

    src = HERE / "dit_decode_si.onnx"
    dst = HERE / "dit_decode_static.onnx"
    m = onnx.load(str(src), load_external_data=False)

    target_len = args.target_len
    prefix_len = args.prefix_len
    kv_len = prefix_len + target_len

    # Set concrete shapes for all graph inputs.
    input_shapes = {
        "target_latents": [1, target_len, 64],
        "timestep": [1],
        "cos": [target_len, 128],
        "sin": [target_len, 128],
        "attention_bias": [1, 1, target_len, kv_len],
    }
    # Add past_kv_* inputs (32 layers).
    for i in range(32):
        input_shapes[f"past_kv_{i}"] = [1, 2, 32, prefix_len, 128]

    for inp in m.graph.input:
        if inp.name in input_shapes:
            shape = input_shapes[inp.name]
            # Clear existing shape and build new one with concrete dims.
            new_shape = onnx.TensorShapeProto()
            for d in shape:
                dim = new_shape.dim.add()
                dim.dim_value = d
            inp.type.tensor_type.ClearField("shape")
            inp.type.tensor_type.shape.CopyFrom(new_shape)
            print(f"  Input {inp.name}: {shape}")

    # Re-infer shapes now that all inputs are concrete.
    m = onnx.shape_inference.infer_shapes(m)

    # Count remaining unknowns.
    unknown = sum(
        1 for v in m.graph.value_info if not v.type.tensor_type.shape.dim
    )
    print(f"[static] {dst.name}: value_info={len(m.graph.value_info)} "
          f"unknown-shape intermediates={unknown}")

    onnx.save(m, str(dst))
    print(f"[static] Saved {dst.name} ({dst.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
