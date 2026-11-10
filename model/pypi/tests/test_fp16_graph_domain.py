"""The FP16 domain pass (fp16_graph.py) is what makes -q fp16 an FP16 artifact.

The runtime takes an op's shader build from the dtype of the tensors it is
handed, so precision is a property of a whole node: one FP16 weight behind an
FP32 Conv walks a 384-byte table as a 768-byte operand, and one FP32 weight in
front of an FP16 MatMul does the same in the other direction. Flipping weights
alone can therefore never produce a smaller model that is also right, which is
what the per-weight guard in test_fp16_weight_dtype_guard.py defends and why
`-q fp16` on an fp32-exported convnet used to convert nothing at all.

The domain pass answers the graph-level question instead: which ops CAN follow
a declared dtype (core/runtime.cpp's `op_shader_fp16` case list, and an op with
no dual FP16 build cannot, because the `_fp16` lookup would resolve to the fp32
key), and, transitively, which tensors may therefore be halves. Everything
below is about that fixed point -- what it flips, what it refuses to flip, and
the one node type allowed to sit between the two widths.
"""
import contextlib
import io

import numpy
import onnx
from onnx import TensorProto, helper, numpy_helper
from onnx.helper import make_tensor_value_info as vi

from onnx2vkop import numeric_audit
from onnx2vkop.converter import ModelConverter
from onnx2vkop.dag import DAGBasedModel, Node
from onnx2vkop.fp16_graph import GraphFp16
from onnx2vkop.optimizer import Quantizer

F, F16 = TensorProto.FLOAT, TensorProto.FLOAT16


def ref(name, dtype=F):
    """One ShapeRef, as the parser writes it: name + shape + elem_type int."""
    return {"name": name, "shape": [-1], "dtype": dtype}


def build(nodes, inputs=(), outputs=(), initializers=None):
    m = DAGBasedModel()
    for n in nodes:
        m.nodes[n.name] = n
    m.inputs = [{"name": name, "shape": [1, 3, 4, 4], "dtype": "float32"}
               for name in inputs]
    m.outputs = [{"name": name, "shape": [1, 10], "dtype": "float32"}
                for name in outputs]
    for name, arr in (initializers or {}).items():
        m.initializers[name] = numpy_helper.from_array(arr, name)
    return m


def node(op, name, ins, outs, **attrs):
    return Node(op, name, attrs, [ref(n) if isinstance(n, str) else n
                                 for n in ins],
                [ref(n) if isinstance(n, str) else n for n in outs])


def apply(m, backend="buffer"):
    with contextlib.redirect_stdout(io.StringIO()):
        return GraphFp16.apply(m, backend)


def dtype_of(m, name):
    """The elem_type the file will declare for a tensor, from any node entry."""
    for n in m.nodes.values():
        for e in list(n.inputs) + list(n.outputs):
            if e["name"] == name:
                return e.get("dtype")
    return None


# ------------------------------------------------------------------ flipping --


def test_a_conv_chain_flips_every_declaration_and_the_weight_bytes():
    x = node("Conv", "conv", ["x", "W"], ["c"])
    m = build([x], inputs=["x"], outputs=["c"],
              initializers={"W": numpy.ones((4, 4, 1, 1), dtype=numpy.float32)})
    summary = apply(m)

    assert not summary["refused"] and not summary["conflicts"]
    # All four places a dtype is spelled, rewritten by name.
    assert m.inputs[0]["dtype"] == "float16"
    assert m.outputs[0]["dtype"] == "float16"
    assert [e["dtype"] for e in x.inputs] == [F16, F16]
    assert x.outputs[0]["dtype"] == F16
    # And the payload matches the declaration: 16 halves, not 16 floats.
    assert m.initializers["W"].data_type == F16
    assert len(m.initializers["W"].raw_data) == 16 * 2
    assert summary["converted"] == 1
    assert summary["bytes_before"] == 64 and summary["bytes_after"] == 32


def test_the_domain_comes_from_the_op_list_not_from_a_whitelist_of_names():
    """MaxPool/GlobalAveragePool/Add/Relu/Softmax flip because the runtime
    dispatches them from the tensor dtype; the resnet18 op set is exactly
    these, which is why the model that used to convert nothing now does."""
    ops = ["Relu", "MaxPool", "GlobalAveragePool", "Add", "Softmax", "MatMul",
           "Transpose", "Concat", "Reshape", "Conv"]
    n = node(ops[0], "n0", ["a", "W"], ["b"])
    nodes = [n]
    for i, op in enumerate(ops[1:], start=1):
        nodes.append(node(op, f"n{i}", [f"b{i-1}"], [f"b{i}"]))
    m = build(nodes, inputs=["a"], outputs=[f"b{len(ops) - 1}"],
              initializers={"W": numpy.ones((2, 2), dtype=numpy.float32)})
    summary = apply(m)
    assert summary["blockers"] == {}
    assert summary["converted"] == 1
    assert all(dtype_of(m, f"b{i}") == F16 for i in range(len(ops)))


def test_the_weight_pass_after_it_stays_quiet():
    """The zero-conversion warning is about an artifact that stayed FP32.

    After the domain pass every weight already holds halves, so the per-weight
    pass converts nothing and must not claim the model is the FP32 one under a
    different name.
    """
    m = build([node("Conv", "conv", ["x", "W"], ["c"])], inputs=["x"],
              outputs=["c"],
              initializers={"W": numpy.ones((4, 4), dtype=numpy.float32)})
    apply(m)
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        Quantizer.quantize_to_fp16_selective(m, "buffer")
    assert "[fp16] WARNING" not in out.getvalue()


def test_plan_computes_without_writing():
    m = build([node("Conv", "conv", ["x", "W"], ["c"])], inputs=["x"],
              outputs=["c"],
              initializers={"W": numpy.ones((4, 4), dtype=numpy.float32)})
    plan = GraphFp16.plan(m)
    assert plan["domain"] == {"x", "W", "c"}
    assert m.initializers["W"].data_type == F
    assert dtype_of(m, "c") == F


# ------------------------------------------------------------------ islands ---


def test_an_op_that_cannot_follow_the_dtype_keeps_its_whole_island_fp32():
    """Gemm is dispatched from --precision, not from the tensor, and has no
    dual FP16 buffer build. The tensor it shares with the Conv above it has to
    stay FP32, which drags the Conv with it -- an FP16 Conv writing halves that
    an FP32 Gemm then reads is the same garbage in the other direction."""
    conv = node("Conv", "conv", ["x", "W"], ["c"])
    gemm = node("Gemm", "head", ["c", "H"], ["y"])
    m = build([conv, gemm], inputs=["x"], outputs=["y"],
              initializers={"W": numpy.ones((4, 4), dtype=numpy.float32),
                            "H": numpy.ones((4, 4), dtype=numpy.float32)})
    summary = apply(m)

    assert summary["domain"] == set()
    assert "head" in summary["blockers"]
    # The Conv is not blocked by nature; it is blocked by what it touches.
    assert "conv" in summary["blockers"]
    assert summary["converted"] == 0
    assert m.initializers["W"].data_type == F
    assert m.inputs[0]["dtype"] == "float32"


def test_a_shared_weight_forces_every_node_that_reads_it_fp32():
    """The exclusion flags cannot be honored under -q fp16 for this reason."""
    mm = node("MatMul", "mm", [ref("a16", TensorProto.FLOAT16), "W"], ["o"])
    bn = node("BatchNormalization", "bn", ["x", "W"], ["b"])
    m = build([mm, bn], inputs=["x"], outputs=["o"],
              initializers={"W": numpy.ones((4, 4), dtype=numpy.float32)})
    m.inputs.append({"name": "a16", "shape": [1, 4], "dtype": "float16"})
    summary = apply(m)

    # W must stay wide for the BatchNorm and the MatMul already reads a FP16
    # activation: the node would mix the two widths, so nothing is rewritten.
    assert summary["refused"]
    assert summary["conflicts"] and "MatMul mm" in summary["conflicts"][0]
    assert m.initializers["W"].data_type == F


def test_a_cast_is_the_only_boundary_and_the_island_stops_at_it():
    """The fp32-domain chain an fp16 export wraps in Cast(to=1)...Cast(to=10).

    Everything before the first Cast and after the second is FP16; the ops
    inside the cast pair keep FP32. This is the shape the per-weight guard was
    written for, and the domain pass has to reproduce its verdict from the
    graph instead of from an op-type list.
    """
    conv = node("Conv", "conv", ["x", "W"], ["c"])
    up = node("Cast", "up", ["c"], ["cf"], to=1)
    pow_ = node("Pow", "pow", ["cf", "two"], ["p"])
    down = node("Cast", "down", ["p"], ["p16"], to=10)
    relu = node("Relu", "relu", ["p16"], ["r16"])
    for e in relu.inputs + relu.outputs + down.outputs:
        e["dtype"] = F16
    m = build([conv, up, pow_, down, relu], inputs=["x"], outputs=["r16"],
              initializers={"W": numpy.ones((4, 4), dtype=numpy.float32),
                            "two": numpy.array(2.0, dtype=numpy.float32)})
    summary = apply(m)

    assert not summary["refused"]
    assert {"x", "W", "c"} <= summary["domain"]
    assert {"cf", "p", "two"} <= summary["forced"]
    # The Cast's INPUT was not forced with its output: that would drag the
    # whole graph back to FP32 at the first Cast(to=1).
    assert conv.outputs[0]["dtype"] == F16
    assert dtype_of(m, "cf") == F and dtype_of(m, "p") == F
    assert m.initializers["W"].data_type == F16
    assert m.initializers["two"].data_type == F
    assert "up" not in summary["blockers"]


def test_host_only_ops_neither_block_nor_decide():
    """Squeeze/Unsqueeze/Identity/Shape run without a shader, so their output
    dtype is their input's by construction. They may sit inside either domain
    and must not be read as a blocker."""
    chain = [node("Conv", "conv", ["x", "W"], ["c"]),
             node("Squeeze", "sq", ["c"], ["s"]),
             node("Add", "add", ["s", "k"], ["a"]),
             node("Unsqueeze", "unsq", ["a"], ["u"])]
    m = build(chain, inputs=["x"], outputs=["u"],
              initializers={"W": numpy.ones((4, 4), dtype=numpy.float32),
                            "k": numpy.ones((4,), dtype=numpy.float32)})
    summary = apply(m)
    assert summary["blockers"] == {}
    assert {"W", "k", "c", "s", "a", "u"} <= summary["domain"]
    assert m.initializers["k"].data_type == F16


# ----------------------------------------------------------------- not dtype --


def test_integer_tensors_are_never_touched():
    conv = node("Conv", "conv", ["x", "W"], ["c"])
    gather = node("Gather", "g", [ref("c"), ref("idx", TensorProto.INT64)],
                  ["gc"])
    m = build([conv, gather], inputs=["x"], outputs=["gc"],
              initializers={"W": numpy.ones((4, 4, 1, 1), dtype=numpy.float32),
                            "idx": numpy.arange(4, dtype=numpy.int64)})
    summary = apply(m)
    assert {"x", "W", "c", "gc"} == summary["domain"]
    assert m.initializers["W"].data_type == F16
    # The index operand is not a float tensor in any domain: neither flipped
    # nor forced, in the declaration and not in the payload.
    assert "idx" not in summary["domain"] and "idx" not in summary["forced"]
    assert gather.inputs[1]["dtype"] == TensorProto.INT64
    assert m.initializers["idx"].data_type == TensorProto.INT64


def test_a_blocked_op_collapses_the_domain_upstream_of_it():
    """Equal is dispatched from --precision rather than from the tensor, so its
    operand cannot be halves -- and that operand is a Gather's output, which
    cannot be halves either, and so on back to the graph input.

    Over-conservative for a run that would pass --precision fp16, and that is
    the point: an artifact must not need a flag to be the precision its own
    file declares. The pass leaves the chain FP32 and names the op that
    decided it.
    """
    conv = node("Conv", "conv", ["x", "W"], ["c"])
    gather = node("Gather", "g", [ref("c"), ref("idx", TensorProto.INT64)],
                  ["gc"])
    eq = node("Equal", "eq", [ref("gc"), ref("one", TensorProto.INT64)],
              [ref("mask", TensorProto.BOOL)])
    m = build([conv, gather, eq], inputs=["x"], outputs=["gc"],
              initializers={"W": numpy.ones((4, 4, 1, 1), dtype=numpy.float32),
                            "idx": numpy.arange(4, dtype=numpy.int64),
                            "one": numpy.ones((4, 4), dtype=numpy.int64)})
    summary = apply(m)
    assert summary["domain"] == set()
    assert "eq" in summary["blockers"]
    assert m.initializers["W"].data_type == F
    assert m.inputs[0]["dtype"] == "float32"


def test_an_orphan_weight_stays_wide():
    """A tensor no node references has no consumer to satisfy and no
    declaration to be consistent with; rewriting its bytes is not what a
    precision pass is for."""
    m = build([node("Conv", "conv", ["x", "W"], ["c"])], inputs=["x"],
              outputs=["c"],
              initializers={"W": numpy.ones((4, 4), dtype=numpy.float32),
                            "unused": numpy.ones((4, 4), dtype=numpy.float32)})
    summary = apply(m)
    assert "unused" not in summary["domain"]
    assert m.initializers["unused"].data_type == F
    assert summary["converted"] == 1


def test_an_already_fp16_export_is_left_alone_without_being_refused():
    """No FLOAT candidate exists, so the domain pass has nothing to say and the
    weight pass afterwards owns the leftovers."""
    n = node("MatMul", "mm", ["a", "W"], ["o"])
    for e in n.inputs + n.outputs:
        e["dtype"] = F16
    m = build([n], inputs=["a"], outputs=["o"])
    m.inputs[0]["dtype"] = "float16"
    m.outputs[0]["dtype"] = "float16"
    m.initializers["W"] = numpy_helper.from_array(
        numpy.ones((4, 4), dtype=numpy.float16), "W")
    summary = apply(m)
    assert summary["domain"] == set() and summary["converted"] == 0
    assert not summary["refused"] and not summary["conflicts"]


def test_the_image_backend_has_nothing_to_declare():
    """It builds every graph input as fp16 regardless of the file, so the
    declaration is not what the run reads and rewriting it changes nothing."""
    m = build([node("Conv", "conv", ["x", "W"], ["c"])], inputs=["x"],
              outputs=["c"],
              initializers={"W": numpy.ones((4, 4), dtype=numpy.float32)})
    summary = apply(m, backend="image")
    assert summary["skipped_backend"] and summary["converted"] == 0
    assert m.inputs[0]["dtype"] == "float32"
    assert m.initializers["W"].data_type == F


def test_a_magnitude_past_fp16_is_reported_not_silently_inf():
    big = numpy.array([[70000.0, 1.0]], dtype=numpy.float32)
    m = build([node("Conv", "conv", ["x", "W"], ["c"])], inputs=["x"],
              outputs=["c"], initializers={"W": big})
    summary = apply(m)
    assert summary["out_of_range"] == [("W", 70000.0)]
    halves = numpy.frombuffer(m.initializers["W"].raw_data, dtype=numpy.float16)
    assert numpy.isinf(halves[0]) and halves[1] == 1.0


def test_the_overflowing_weight_also_reaches_the_numeric_audit():
    """The domain pass owns the downcast, so it owns the hand-off: the same
    weight the summary reports as out_of_range is the row the convert-time
    numeric gate counts as overflow (numeric_audit.py)."""
    big = numpy.array([[70000.0, 1.0]], dtype=numpy.float32)
    m = build([node("Conv", "conv", ["x", "W"], ["c"])], inputs=["x"],
              outputs=["c"], initializers={"W": big})
    with contextlib.redirect_stdout(io.StringIO()), \
            numeric_audit.collecting() as audit:
        apply(m)
    row, = [r for r in audit.rows if r.tensor == "W"]
    assert (row.fmt, row.overflow) == ("fp16", 1)
    assert row.flagged


def test_a_declared_input_the_parser_could_not_type_is_transparent():
    """An entry with no dtype is a tensor shape inference could not type; the
    runtime reads it from its producer. Inventing FP16 here would be a
    declaration nothing verified."""
    x_in = {"name": "x", "shape": []}
    conv = Node("Conv", "conv", {}, [x_in, ref("W")], [ref("c")])
    m = build([conv], outputs=["c"],
              initializers={"W": numpy.ones((4, 4), dtype=numpy.float32)})
    summary = apply(m)
    assert "x" not in summary["domain"] and "x" not in summary["forced"]
    assert "x" not in m.nodes["conv"].inputs[0]
    assert {"W", "c"} == summary["domain"]


# ------------------------------------------------------- the real parse path --


def test_the_parser_records_what_the_pass_reads():
    """The hand-built DAGs above must match what parse_onnx_model produces.

    Every test here keys on node input/output entries carrying an ONNX
    elem_type INT and graph I/O carrying a dtype NAME. If the parser ever
    changes one of those, the domain pass would silently read a graph that is
    not there -- so read the real parser once, end to end.
    """
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        path = f"{d}/c.onnx"
        graph = helper.make_graph(
            [helper.make_node("Conv", ["x", "W", "b"], ["c"]),
             helper.make_node("Relu", ["c"], ["y"])],
            "c",
            inputs=[vi("x", F, [1, 2, 2, 2])],
            outputs=[vi("y", F, [1, 2, 2, 2])],
            initializer=[
                numpy_helper.from_array(
                    numpy.ones((2, 2, 1, 1), dtype=numpy.float32), "W"),
                numpy_helper.from_array(numpy.ones((2,), dtype=numpy.float32),
                                        "b"),
            ],
        )
        onnx.save(helper.make_model(
            graph, opset_imports=[helper.make_opsetid("", 17)]), path)
        m = ModelConverter().parse_onnx_model(path, 1)

        conv = next(n for n in m.nodes.values() if n.op_type == "Conv")
        assert conv.inputs[0]["dtype"] == F, "the parser must type activations"
        assert m.inputs[0]["dtype"] == "float32", "graph I/O is a NAME"

        summary = apply(m)
        assert summary["converted"] == 2
        assert m.initializers["W"].data_type == F16
        assert m.initializers["b"].data_type == F16
        assert m.inputs[0]["dtype"] == "float16"
