"""Streaming large pass-through weights past the converter without holding them.

A 13.9 GB text encoder used to cost 33 GiB of RSS in `onnx2vkop`, and on a 36 GiB
machine that is a Jetsam kill, not a slow build. Two full copies do it: hydration
puts every weight in a TensorProto's `raw_data`, and protobuf hands out a fresh
copy per `raw_data` read, so `raw_map` alone doubles the set before the first
pass runs. A float16 weight that only MatMul/Gemm/Conv ever reads cannot be
folded or content-compared by any pass, so it stays in its external data file and
is copied from there straight into the .vkopbin trailer.

These cases pin the two properties that make that safe rather than lucky:

  * the artifact is the same model -- byte-for-byte against the hydrated path
    when both take the same view branch, and per-tensor payload-equal otherwise;
  * anything that would need a streamed tensor's bytes says so loudly instead of
    reading zero bytes and shipping a hole.

Measured on this repo's own models: llm.onnx (197 of 310 initializers streamed,
3,442,978,248-byte output identical both ways) peaks at 0.169 GiB instead of
10.332 GiB.
"""

import numpy
import onnx
import pytest
from onnx import TensorProto, external_data_helper, helper, numpy_helper

from onnx2vkop import weight_stream as ws
from onnx2vkop.converter import ModelConverter
from onnx2vkop.dag import DAGBasedModel, Node
from onnx2vkop.optimizer import ONNXOptimizer

ELEMS = 1 << 18  # 512 KiB float16, and three of them clear the 1 MiB blob gate
N_WEIGHTS = 3


def _opts(quant="fp16", unify=False, rgba=False):
    from types import SimpleNamespace
    return SimpleNamespace(quant=quant, unify=unify, rgba=rgba,
                           group_size=64, target_backend="buffer")


def _weights(seed=0):
    """N distinct float16 [64, 2^16] weights -- distinct so a merge is visible."""
    rng = numpy.random.RandomState(seed)
    return [rng.rand(64, ELEMS // 64).astype(numpy.float16) for _ in range(N_WEIGHTS)]


def _model_with_external_weights(tmp_path, location="w.bin"):
    """MatMul chain whose weights live in `location`, as an exporter writes it."""
    x = helper.make_tensor_value_info("X", TensorProto.FLOAT16, [1, 64])
    y = helper.make_tensor_value_info("Y", TensorProto.FLOAT16, [1, 64])
    nodes, inits, prev = [], [], "X"
    for i, w in enumerate(_weights()):
        name = "W%d" % i
        inits.append(helper.make_tensor(name, TensorProto.FLOAT16,
                                        list(w.shape), w.tobytes(), raw=True))
        out = "M%d" % i if i < N_WEIGHTS - 1 else "Y"
        nodes.append(helper.make_node("MatMul", [prev, name], [out], name="mm%d" % i))
        prev = out
    graph = helper.make_graph(nodes, "g", [x], [y], inits)
    model = helper.make_model(graph, producer_name="test")
    model.opset_import[0].version = 17
    model.ir_version = 8
    external_data_helper.convert_model_to_external_data(
        model, all_tensors_to_one_file=True, location=location, size_threshold=0)
    onnx.save_model(model, str(tmp_path / "m.onnx"))
    return tmp_path / "m.onnx"


def _convert(tmp_path, out_name, *, stream, monkeypatch, min_mib=0,
             quant="fp16"):
    """Run the pipeline the CLI runs, and return the written bytes."""
    src = _model_with_external_weights(tmp_path)
    monkeypatch.setenv("VKOP_EXTERNAL_BLOB_MB", "1")
    monkeypatch.setenv("VKOP_STREAM_WEIGHTS", "1" if stream else "0")
    monkeypatch.setenv("VKOP_STREAM_MIN_MIB", str(min_mib))
    converter = ModelConverter()
    dag = converter.parse_onnx_model(str(src), 1, quant=quant)
    converter.apply_optimizations(dag, _opts(quant=quant))
    out = tmp_path / out_name
    dag.save_to_binary(str(out))
    return out.read_bytes(), dag


# --------------------------------------------------------------------------
# the streaming contract: bytes stay in the file, the model does not change
# --------------------------------------------------------------------------

def test_streamed_weights_never_enter_memory(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("VKOP_STREAM_MIN_MIB", "0")
    src = _model_with_external_weights(tmp_path)
    model = onnx.load(str(src), load_external_data=False)
    plan = ws.prepare(model, str(tmp_path), quant="fp16")
    assert sorted(plan) == ["W0", "W1", "W2"], capsys.readouterr().out
    for init in model.graph.initializer:
        if init.name in plan:
            assert ws.is_deferred(init), init.name
            assert len(init.raw_data) == 0
    assert "streaming 3 initializer(s)" in capsys.readouterr().out


def test_streamed_conversion_is_the_same_bytes_as_hydrated(tmp_path, monkeypatch):
    """The strongest claim this feature can make: the artifact is identical.

    Both runs must take the same view branch, or the blob layout differs for a
    reason that has nothing to do with streaming -- the hydrated path only
    demotes once 1.5 GiB of inline bytes are present, so the gate is lowered for
    both.
    """
    import onnx2vkop.optimizer as opt
    monkeypatch.setattr(opt, "_VIEW_MIN_INIT_BYTES", 1 << 20)
    streamed, _ = _convert(tmp_path, "s.vkopbin", stream=True,
                           monkeypatch=monkeypatch)
    hydrated, _ = _convert(tmp_path, "h.vkopbin", stream=False,
                           monkeypatch=monkeypatch)
    assert streamed == hydrated, "streamed artifact differs from the hydrated one"
    assert len(streamed) > N_WEIGHTS * ELEMS * 2, "weights missing from the blob"


def test_payloads_land_at_the_offsets_the_side_table_claims(tmp_path, monkeypatch):
    """Read the artifact back: each side-table row must point at its own bytes.

    The runtime only ever sees (name, dtype, dims, offset, size), so that is the
    contract worth checking -- not the writer's internal ordering.
    """
    from onnx2vkop.inspect import ModelView

    src = _model_with_external_weights(tmp_path)
    monkeypatch.setenv("VKOP_EXTERNAL_BLOB_MB", "1")
    monkeypatch.setenv("VKOP_STREAM_MIN_MIB", "0")
    monkeypatch.setenv("VKOP_STREAM_WEIGHTS", "1")
    converter = ModelConverter()
    dag = converter.parse_onnx_model(str(src), 1, quant="fp16")
    converter.apply_optimizations(dag, _opts())
    out = tmp_path / "p.vkopbin"
    dag.save_to_binary(str(out))

    assert dag.deferred_weights, "the model must actually have streamed a weight"
    view = ModelView(str(out))
    blob = view.external_blob_trailer
    assert blob, "expected the external trailer"
    rows = {i["name"]: i for i in view.initializers()}
    with open(out, "rb") as got, open(tmp_path / "w.bin", "rb") as source:
        for name, entry in dag.deferred_weights.items():
            row = rows[name]
            source.seek(entry["offset"])
            want = source.read(entry["length"])
            assert row["size"] == entry["length"] == len(want), name
            got.seek(blob + row["offset"])
            assert got.read(row["size"]) == want, name


def test_weight_a_pass_could_read_is_hydrated_not_streamed(tmp_path, capsys):
    """A big float16 tensor feeding Gather is a value the folder may want."""
    w = numpy.ones((64, ELEMS // 64), dtype=numpy.float16)
    inits = [helper.make_tensor("W", TensorProto.FLOAT16, list(w.shape),
                                w.tobytes(), raw=True)]
    idx = helper.make_tensor("ix", TensorProto.INT64, [1], [0])
    inits.append(idx)
    nodes = [helper.make_node("Gather", ["W", "ix"], ["Y"], name="gather")]
    graph = helper.make_graph(
        nodes, "g", [helper.make_tensor_value_info("Y", TensorProto.FLOAT16, [1])],
        [], inits)
    model = helper.make_model(graph, producer_name="test")
    model.opset_import[0].version = 17
    external_data_helper.convert_model_to_external_data(
        model, all_tensors_to_one_file=True, location="g.bin", size_threshold=0)
    onnx.save_model(model, str(tmp_path / "g.onnx"))
    loaded = onnx.load(str(tmp_path / "g.onnx"), load_external_data=False)
    plan = ws.prepare(loaded, str(tmp_path), quant="fp16")
    assert plan == {}, "Gather's input is not a pass-through weight"
    assert len(loaded.graph.initializer[0].raw_data) == ELEMS * 2


def test_deferral_is_refused_when_a_payload_would_be_rewritten(monkeypatch):
    monkeypatch.delenv("VKOP_STREAM_WEIGHTS", raising=False)
    # 4-bit and AWQ need the weight's magnitudes in hand to pick their scale
    # (a group and a channel both span K), so their payloads cannot be built
    # from a streamed band and the deferral stays refused.
    for rewritten in ("int4", "nf4", "nvfp4", "awq"):
        assert ws.deferral_refusal(rewritten, False, False) is not None, rewritten
    assert ws.deferral_refusal("fp16", True, False) is not None
    assert ws.deferral_refusal("fp16", False, True) is not None
    assert ws.deferral_refusal("fp16", False, False) is None
    monkeypatch.setenv("VKOP_STREAM_WEIGHTS", "0")
    assert "VKOP_STREAM_WEIGHTS=0" in ws.deferral_refusal("fp16", False, False)


def test_a_byte_format_may_stream_and_quantize_on_the_way_out(monkeypatch):
    """int8 / fp8 keep their scale in one pass over the file.

    Their scale is the weight's per-column absmax and their payload overwrites
    the values, so a deferred float32 weight can be quantized as it is copied
    into the trailer instead of being inflated into an array first. That is why
    these three are the only quantizations deferral still allows.
    """
    monkeypatch.delenv("VKOP_STREAM_WEIGHTS", raising=False)
    for value_free in ("int8", "fp8e4m3", "fp8e5m2"):
        assert ws.deferral_refusal(value_free, False, False) is None
        assert ws.exports_weight_values(value_free, False, False) is False
    assert ws.exports_weight_values("int8", True, False) is True
    assert ws.exports_weight_values("int8", False, True) is True
    for needs_values in ("int4", "nf4", "nvfp4", "awq", "fp16", None):
        assert ws.exports_weight_values(needs_values, False, False) is True


def _float32_weights(seed=0):
    rng = numpy.random.RandomState(seed)
    return [rng.randn(64, ELEMS // 64).astype(numpy.float32) for _ in range(N_WEIGHTS)]


def _model_with_external_float32_weights(tmp_path):
    """Same chain, but float32 weights -- the dtype a byte format may defer."""
    x = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 64])
    y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 64])
    nodes, inits, prev = [], [], "X"
    for i, w in enumerate(_float32_weights()):
        name = "W%d" % i
        inits.append(helper.make_tensor(name, TensorProto.FLOAT,
                                        list(w.shape), w.tobytes(), raw=True))
        out = "M%d" % i if i < N_WEIGHTS - 1 else "Y"
        nodes.append(helper.make_node("MatMul", [prev, name], [out], name="mm%d" % i))
        prev = out
    graph = helper.make_graph(nodes, "g", [x], [y], inits)
    model = helper.make_model(graph, producer_name="test")
    model.opset_import[0].version = 17
    model.ir_version = 8
    external_data_helper.convert_model_to_external_data(
        model, all_tensors_to_one_file=True, location="w32.bin", size_threshold=0)
    onnx.save_model(model, str(tmp_path / "m32.onnx"))
    return tmp_path / "m32.onnx", _float32_weights()


def test_int8_streams_a_float32_weight_and_writes_the_scale(tmp_path, monkeypatch):
    """A float32 weight under int8 is quantized on its way into the trailer.

    The bytes must be exactly what the in-memory quantizer would write (per
    column amax / 127, two's-complement byte), the side table must call them
    INT8 with one byte per element, and the MatMul must carry the scale as its
    third input -- all without the weight ever becoming an array in RAM.
    """
    from onnx2vkop.inspect import ModelView

    src, weights = _model_with_external_float32_weights(tmp_path)
    monkeypatch.setenv("VKOP_EXTERNAL_BLOB_MB", "1")
    monkeypatch.setenv("VKOP_STREAM_MIN_MIB", "0")
    monkeypatch.setenv("VKOP_STREAM_WEIGHTS", "1")
    converter = ModelConverter()
    dag = converter.parse_onnx_model(str(src), 1, quant="int8")
    converter.apply_optimizations(dag, _opts(quant="int8"))
    out = tmp_path / "s8.vkopbin"
    dag.save_to_binary(str(out))

    streamed = {n: e for n, e in dag.deferred_weights.items() if e.get("quant")}
    assert sorted(streamed) == ["W0", "W1", "W2"], sorted(dag.deferred_weights)
    for name, entry in streamed.items():
        assert entry["data_type"] == TensorProto.INT8, name
        assert entry["length"] == 64 * (ELEMS // 64) * 4, name
        assert entry["scale"] == name + "_scale", name

    view = ModelView(str(out))
    rows = {i["name"]: i for i in view.initializers()}
    blob = view.external_blob_trailer
    wanted = {n: w for n, w in zip(("W0", "W1", "W2"), weights)}
    with open(out, "rb") as f:
        for name, entry in streamed.items():
            w = wanted[name]
            scale = numpy.where(numpy.max(numpy.abs(w), axis=0) == 0, 1.0,
                                numpy.max(numpy.abs(w), axis=0) / 127.0)
            q = numpy.round(w / scale[None, :]).astype(numpy.int8)
            row = rows[name]
            # The side table spells the payload's own names ("int8"), not the
            # ONNX enum: the C++ reader keys on these strings, and vkop's
            # neutral spelling is what every other quantized row carries.
            assert row["name"] == name, row
            assert row["size"] == q.size, row
            assert tuple(entry["dims"]) == tuple(row["dims"]), row
            f.seek(blob + row["offset"])
            assert f.read(row["size"]) == q.tobytes(), name

            scale_row = rows[name + "_scale"]
            assert scale_row["name"] == name + "_scale", scale_row
            f.seek(blob + scale_row["offset"])
            got = numpy.frombuffer(f.read(scale_row["size"]), dtype=numpy.float32)
            numpy.testing.assert_allclose(got, scale.astype(numpy.float32),
                                          rtol=1e-6)


# --------------------------------------------------------------------------
# the failure modes, each of which has to be loud
# --------------------------------------------------------------------------

def test_content_dedup_cannot_merge_weightless_initializers():
    """The bug that made this necessary.

    With no bytes, two different 96 MB weights are equal to
    `eliminate_duplicate_initializer`. On text_encoder_tiny it collapsed 14
    distinct layer weights into 4. optimize_model now demotes external-payload
    initializers out of the graph for the passes, so nothing merges them and the
    dedup of the genuinely inline constants still happens.
    """
    def build(distinct):
        inits = []
        for i, name in enumerate(["Wa", "Wb"]):
            w = numpy.full((1024, 1024), distinct[i], dtype=numpy.float16)
            inits.append(helper.make_tensor(name, TensorProto.FLOAT16,
                                            list(w.shape), w.tobytes(), raw=True))
        nodes = [helper.make_node("MatMul", ["X", "Wa"], ["Ya"], name="ma"),
                 helper.make_node("MatMul", ["Ya", "Wb"], ["Y"], name="mb")]
        g = helper.make_graph(
            nodes, "g",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT16, [1, 1024])],
            [helper.make_tensor_value_info("Y", TensorProto.FLOAT16, [1, 1024])],
            inits)
        m = helper.make_model(g, producer_name="test")
        m.opset_import[0].version = 17
        m.ir_version = 8
        return m

    model = build([1.0, 2.0])
    # Spell the weights as external references with no bytes at all, which is
    # exactly what a pass sees after weight_stream deferred them.
    for init in model.graph.initializer:
        init.data_location = TensorProto.EXTERNAL
        e = init.external_data.add()
        e.key, e.value = "location", "nowhere.bin"
        e = init.external_data.add()
        e.key, e.value = "offset", "0"
        e = init.external_data.add()
        e.key, e.value = "length", str(1024 * 1024 * 2)
        init.raw_data = b""

    out = ONNXOptimizer.optimize_model(model, batch_size=1)
    names = sorted(i.name for i in out.graph.initializer)
    assert names == ["Wa", "Wb"], "distinct weights were merged: %s" % names


def test_to_array_guard_refuses_a_streamed_tensor():
    ws.install_guard()
    init = TensorProto()
    init.name = "Wg"
    init.data_type = TensorProto.FLOAT16
    init.dims.extend([4, 4])
    entry = init.external_data.add()
    entry.key, entry.value = "location", "w.bin"
    init.data_location = TensorProto.EXTERNAL
    with pytest.raises(ws.DeferredWeightError, match="Wg"):
        numpy_helper.to_array(init)
    # A tensor that is not deferred must keep decoding normally, guard or not.
    inline = helper.make_tensor("Wi", TensorProto.FLOAT16, [2],
                                numpy.array([1.5, 2.5], numpy.float16).tobytes(),
                                raw=True)
    assert numpy_helper.to_array(inline).tolist() == [1.5, 2.5]


def _deferred_init(dag, name, dims=(64, 16)):
    """An initializer in exactly the state weight_stream leaves a deferred one in."""
    init = TensorProto()
    init.name = name
    init.data_type = TensorProto.FLOAT16
    init.dims.extend(dims)
    length = dims[0] * dims[1] * 2
    for key, value in (("location", "w.bin"), ("offset", "0"),
                       ("length", str(length))):
        e = init.external_data.add()
        e.key, e.value = key, value
    init.data_location = TensorProto.EXTERNAL
    dag.initializers[name] = init
    dag.deferred_weights[name] = {
        "name": name, "location": "w.bin", "offset": 0, "length": length,
        "data_type": TensorProto.FLOAT16, "dims": tuple(dims), "consumers": 1,
    }
    return length


def _dag_with(deferred_names, referenced_by):
    dag = DAGBasedModel()
    for name in deferred_names:
        _deferred_init(dag, name)
    for node_name, inputs in referenced_by.items():
        n = Node("MatMul", node_name, {},
                 [{"name": i} for i in inputs], [{"name": node_name + "_out"}])
        dag.nodes[n.name] = n
    return dag


def test_verify_deferred_flags_a_weight_a_live_node_still_names():
    dag = _dag_with(["W0"], {"mm0": ["X", "GONE"]})
    del dag.initializers["W0"]
    del dag.deferred_weights["W0"]
    dag.deferred_weights["GONE"] = {
        "name": "GONE", "location": "w.bin", "offset": 0, "length": ELEMS * 2,
        "data_type": TensorProto.FLOAT16, "dims": (64, 16), "consumers": 1}
    with pytest.raises(ws.DeferredWeightError, match="GONE"):
        ws.verify_deferred(dag.deferred_weights, dag)


def test_verify_deferred_lets_a_pruned_weight_go():
    """A fusion that stopped reading a weight is a byte saving, not a bug."""
    dag = _dag_with(["W0"], {})
    ws.verify_deferred(dag.deferred_weights, dag)  # must not raise


def test_verify_deferred_flags_a_changed_tensor():
    dag = _dag_with(["W0"], {"mm0": ["X", "W0"]})
    dag.initializers["W0"].dims[1] = 8  # a pass rewrote the shape it promised
    with pytest.raises(ws.DeferredWeightError, match="changed while deferred"):
        ws.verify_deferred(dag.deferred_weights, dag)


def test_inline_blob_cannot_carry_a_streamed_weight(tmp_path, monkeypatch):
    monkeypatch.setenv("VKOP_EXTERNAL_BLOB_MB", "100000")
    monkeypatch.setenv("VKOP_STREAM_WEIGHTS", "1")
    monkeypatch.setenv("VKOP_STREAM_MIN_MIB", "0")
    src = _model_with_external_weights(tmp_path)
    converter = ModelConverter()
    dag = converter.parse_onnx_model(str(src), 1, quant="fp16")
    converter.apply_optimizations(dag, _opts())
    assert dag.deferred_weights, "the test model must exercise the deferred path"
    with pytest.raises(ws.DeferredWeightError, match="external-blob-mb"):
        dag.save_to_binary(str(tmp_path / "x.vkopbin"))


def test_write_payload_refuses_a_short_source_file(tmp_path):
    entry = {"name": "W", "location": "w.bin", "offset": 0, "length": 4096,
             "data_type": TensorProto.FLOAT16, "dims": (64, 32), "consumers": 1}
    (tmp_path / "w.bin").write_bytes(b"\x00" * 1024)
    with pytest.raises(ws.DeferredWeightError, match="runs out"):
        ws.write_payload(entry, open(tmp_path / "out.bin", "wb"), str(tmp_path))


def test_payload_copy_is_byte_faithful(tmp_path):
    """A payload at a non-zero offset must land whole, with no neighbours."""
    payload = numpy.arange(1 << 16, dtype=numpy.float16).tobytes()
    (tmp_path / "w.bin").write_bytes(b"pad!" + payload)
    entry = {"name": "W", "location": "w.bin", "offset": 4,
             "length": len(payload), "data_type": TensorProto.FLOAT16,
             "dims": (1, len(payload)), "consumers": 1}
    out = tmp_path / "out.bin"
    with open(out, "wb") as f:
        assert ws.write_payload(entry, f, str(tmp_path)) == len(payload)
    assert out.read_bytes() == payload


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
