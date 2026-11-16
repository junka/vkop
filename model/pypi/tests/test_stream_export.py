#!/usr/bin/env python3
"""The streaming exporter must produce the same ONNX as the eager one.

Builds a small model twice -- once normally (weights resident) and once through
stream_export -- and compares the graphs node by node plus the weight bytes
tensor by tensor. Any difference is a splice bug, not a design trade-off.

This is the correctness gate for `onnx2vkop/stream_export.py`. Its model is a
plain nn.Module on purpose: qwen3_5 has a separate, still-open problem with
symbolic dims on intermediate tensors (see the KNOWN LIMITATION note at the end
of stream_export.py), so it must not be the thing this test measures.
"""
import os
import sys
import tempfile

import numpy as np
import onnx
import torch
import torch.nn as nn
from onnx import numpy_helper

from onnx2vkop import stream_export  # noqa: E402


class Tiny(nn.Module):
    """Two linear layers and a bias add -- enough for >=2 initializers."""

    def __init__(self):
        super().__init__()
        self.a = nn.Linear(16, 8, bias=True)
        self.b = nn.Linear(8, 4, bias=True)

    def forward(self, x):
        return self.b(torch.relu(self.a(x)))


def save_safetensors(model: nn.Module, out_dir: str) -> None:
    """Write the parameters as one safetensors shard, as an exporter would."""
    from safetensors.torch import save_file
    tensors = {k: v.detach().contiguous() for k, v in model.state_dict().items()}
    save_file(tensors, os.path.join(out_dir, "model.safetensors"))


def run_eager(model: nn.Module, out_dir: str, x: torch.Tensor) -> onnx.ModelProto:
    path = os.path.join(out_dir, "eager.onnx")
    torch.onnx.export(model, (x,), path, input_names=["x"], output_names=["y"],
                      opset_version=17, dynamo=False)
    return onnx.load(path)


def run_streaming(model: nn.Module, src_dir: str, out_dir: str,
                  x: torch.Tensor) -> onnx.ModelProto:
    # The streaming path traces a meta model; the same module's weights are
    # only present in the safetensors shard.
    meta = Tiny().to("meta")

    def build():
        return meta, (x.to("meta"),), ["x"], ["y"]

    mpath, wpath = stream_export.export_streaming(
        src_dir, build, out_dir, model_file="stream.onnx",
        weights_file="stream.weights.bin")
    # load_external_data=False: onnx.load() otherwise reads the weights back
    # and CLEARS each initializer's external_data entries, which is exactly what
    # this test needs to inspect.
    return onnx.load(mpath, load_external_data=False), wpath


def main():
    torch.manual_seed(0)
    model = Tiny().eval()
    x = torch.randn(2, 16)

    with tempfile.TemporaryDirectory() as tmp:
        src = os.path.join(tmp, "ckpt")
        os.makedirs(src)
        save_safetensors(model, src)

        eager = run_eager(model, tmp, x)
        streamed, wpath = run_streaming(model, src, tmp, x)

        # --- graph shape ---------------------------------------------------
        e_ops = [n.op_type for n in eager.graph.node]
        s_ops = [n.op_type for n in streamed.graph.node]
        print(f"[cmp] eager nodes   = {e_ops}")
        print(f"[cmp] streamed nodes= {s_ops}")
        assert e_ops == s_ops, "node sequences differ"

        # --- initializers ---------------------------------------------------
        e_init = {i.name: i for i in eager.graph.initializer}
        s_init = {i.name: i for i in streamed.graph.initializer}
        print(f"[cmp] eager inits   = {sorted(e_init)}")
        print(f"[cmp] streamed inits= {sorted(s_init)}")
        assert set(e_init) == set(s_init), "initializer sets differ"

        # --- weights, byte for byte -----------------------------------------
        blob = open(wpath, "rb").read()
        for name, ei in e_init.items():
            si = s_init[name]
            assert list(ei.dims) == list(si.dims), f"{name}: dims differ"
            assert ei.data_type == si.data_type, f"{name}: dtype differs"
            want = numpy_helper.to_array(ei).tobytes()
            off = next(int(kv.value) for kv in si.external_data
                       if kv.key == "offset")
            got = blob[off:off + len(want)]
            assert got == want, f"{name}: bytes differ"
            print(f"[cmp] {name:24s} {len(want):8d} B  identical")

        # --- and the whole model runs the same ------------------------------
        import onnxruntime as ort

        def run(m, weights_dir):
            opts = ort.SessionOptions()
            sess = ort.InferenceSession(
                m.SerializeToString() if weights_dir is None else
                os.path.join(weights_dir, "stream.onnx"),
                opts, providers=["CPUExecutionProvider"])
            return sess.run(None, {"x": x.numpy()})[0]

        a = run(eager, None)
        # onnxruntime resolves external data relative to the model's directory
        b = run(streamed, tmp)
        diff = np.abs(a - b).max()
        print(f"[cmp] onnxruntime output maxdiff = {diff:.3e}")
        assert diff == 0.0, "outputs differ"

    print("\nRESULT: streaming export is byte-identical to the eager export")


if __name__ == "__main__":
    main()
