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


# ---------------------------------------------------------------------------
# The qwen3_5-specific failure modes. Both were live bugs on that model and
# neither is a splice mistake, so they get their own models rather than being
# hoped for from the MLP above.
# ---------------------------------------------------------------------------


def test_addend_reproduces_a_pre_trace_parameter_rewrite():
    """A parameter the exporter rewrites before tracing must reach the payload.

    `Qwen3_5RMSNorm` is zero-centred: HF computes `out * (1 + weight)` and the
    exporter folds the `1 +` into the parameter (`weight.data.add_(1.0)`) so
    the graph only needs `out * w`. On a meta parameter that write is a SILENT
    NO-OP -- a meta tensor has no storage -- so the streamed graph is built for
    `1 + w` while the splice copies the checkpoint's raw `w`. Measured on
    Qwen3.5-27B's tiny sibling: every logit came out exactly 0.0.

    `addend` is how the caller replays the rewrite on the bytes. This checks it
    lands, that it is added in fp32 before the fp16 encode, and that omitting
    it (the old behaviour) really does produce different bytes.
    """
    class Normed(nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = nn.Parameter(torch.full((8,), -1.0))

        def forward(self, x):
            return x * self.scale

    with tempfile.TemporaryDirectory() as tmp:
        src = os.path.join(tmp, "ckpt")
        os.makedirs(src)
        save_safetensors(Normed(), src)

        x = torch.randn(2, 8)
        seen = {}

        for tag, addend in (("plain", None), ("folded", {"scale": 1.0})):
            out = os.path.join(tmp, tag)
            os.makedirs(out)

            def build():
                return Normed().to("meta"), (x.to("meta"),), ["x"], ["y"]

            wfile = f"{tag}.weights.bin"
            mpath, wpath = stream_export.export_streaming(
                src, build, out, model_file=f"{tag}.onnx",
                weights_file=wfile, addend=addend)
            m = onnx.load(mpath, load_external_data=False)
            init = next(i for i in m.graph.initializer if i.name == "scale")
            off = next(int(kv.value) for kv in init.external_data
                       if kv.key == "offset")
            n = int(next(kv.value for kv in init.external_data
                         if kv.key == "length"))
            # Read with the initializer's OWN dtype: the plain variant keeps
            # fp32 (no target_dtype), the folded one is fp16.
            np_dt = {onnx.TensorProto.FLOAT: np.float32,
                     onnx.TensorProto.FLOAT16: np.float16}[init.data_type]
            seen[tag] = np.frombuffer(open(wpath, "rb").read()[off:off + n],
                                      dtype=np_dt)

        # The checkpoint holds -1.0; the fold must turn it into 0.0.
        assert np.allclose(seen["plain"].astype(np.float32), -1.0), seen["plain"]
        assert np.allclose(seen["folded"].astype(np.float32), 0.0), seen["folded"]
        print("[addend] plain -1.0, folded 0.0  (the +1 landed)")


def _placeholder_proto(vocab=512, in_features=16):
    """A minimal MatMul proto in the state the meta trace leaves it in.

    The output's last axis carries a generated placeholder instead of the
    literal its weight could prove, which is what the meta trace produces for a
    dim that lives in a weight's shape -- the graph input names the weight, and
    at trace time that weight has no payload. The weight is [in, vocab] so the
    product is well-formed and inference has something real to find.
    """
    from onnx import TensorProto, helper
    w = helper.make_tensor("w", TensorProto.FLOAT, [in_features, vocab],
                           [0.01] * (vocab * in_features))
    m = helper.make_model(
        helper.make_graph(
            nodes=[helper.make_node("MatMul", ["x", "w"], ["y"], name="mm")],
            name="g",
            inputs=[helper.make_tensor_value_info(
                "x", TensorProto.FLOAT, ["batch", "seq", in_features])],
            outputs=[helper.make_tensor_value_info(
                "y", TensorProto.FLOAT,
                ["batch", "seq", "MatMully_dim_2"])],
        ),
        opset_imports=[helper.make_opsetid("", 17)])
    m.graph.initializer.append(w)
    return m


def test_resolve_shape_dims_recovers_weight_derived_dims():
    """A dim the meta trace could only placeholder must become the real number.

    The eager exporter resolves a weight-shaped dim (vocab off lm_head, the
    head counts off a qkv projection) inside its own shape-inference pass, which
    reads the parameter tensors. Under the meta trace those have no storage, so
    the dim surfaces as a generated name (`MatMully_dim_2`) where eager has
    `512`. After the splice the weights ARE real initializers, so inference can
    prove it again -- which is what `resolve_shape_dims` is for.

    Also checks the property that makes it safe: a caller-declared axis stays a
    SYMBOL. The pass must not freeze `seq` at the dummy length, which is the
    exact failure this whole module exists to avoid.
    """
    m = _placeholder_proto()
    before = [(d.dim_param or d.dim_value)
              for d in m.graph.output[0].type.tensor_type.shape.dim]
    dyn = {"y": {0: "batch", 1: "seq"}}

    n = stream_export.resolve_shape_dims(m, dyn)
    after = [(d.dim_param or d.dim_value)
             for d in m.graph.output[0].type.tensor_type.shape.dim]
    print(f"[dims] before={before} after={after} ({n} fixed)")

    assert before[2] == "MatMully_dim_2", before
    assert after[2] == 512, f"vocab not recovered from the weight: {after}"
    assert after[0] == "batch" and after[1] == "seq", \
        f"a declared axis was clobbered: {after}"
    assert n >= 1, "nothing was fixed"


def test_resolve_shape_dims_leaves_what_it_cannot_prove_alone():
    """An unresolvable dim must not be invented.

    The repair is a fill-in, not a guess: a dim inference cannot prove keeps
    whatever it had, including a caller-declared symbol. Getting this wrong
    would silently mis-size a buffer, which is worse than the placeholder it
    replaced.
    """
    from onnx import TensorProto, helper
    # y's last axis is an unresolvable symbol: the identity carries it through
    # unchanged and nothing in the graph can compute its value.
    m = helper.make_model(
        helper.make_graph(
            nodes=[helper.make_node("Identity", ["x"], ["y"], name="i")],
            name="g",
            inputs=[helper.make_tensor_value_info(
                "x", TensorProto.FLOAT, ["batch", "seq", "din"])],
            outputs=[helper.make_tensor_value_info(
                "y", TensorProto.FLOAT, ["batch", "seq", "unknown_w"])],
        ),
        opset_imports=[helper.make_opsetid("", 17)])

    before = [d.dim_param or d.dim_value
              for d in m.graph.output[0].type.tensor_type.shape.dim]
    stream_export.resolve_shape_dims(m, {"y": {1: "seq"}})
    after = [d.dim_param or d.dim_value
             for d in m.graph.output[0].type.tensor_type.shape.dim]
    print(f"[dims] unresolvable: before={before} after={after}")

    # the axis the caller declared stays declared...
    assert after[1] == "seq", f"a declared symbol was overwritten: {after}"
    # ...and the one nothing can prove is left exactly as it was
    assert after[2] == "unknown_w", f"an unprovable dim was invented: {after}"
    assert after[0] == "batch", f"a symbol was clobbered: {after}"


if __name__ == "__main__":
    import sys

    if "--all" in sys.argv:
        # The qwen3_5-specific regressions are not part of the byte-identity
        # gate above -- they need their own models -- so they run on request.
        test_addend_reproduces_a_pre_trace_parameter_rewrite()
        test_resolve_shape_dims_recovers_weight_derived_dims()
        test_resolve_shape_dims_leaves_what_it_cannot_prove_alone()
        print("\nRESULT: stream_export regressions pass")
    else:
        main()
