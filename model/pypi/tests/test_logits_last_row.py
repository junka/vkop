"""Tests for the opt-in "keep only the last logits position" pass.

The pass folds

    MatMul(activation[1, L, H], lm_head_weight[H, vocab]) -> logits[1, L, vocab]
                                                                  ^ graph output

into

    Slice(activation, starts=[-1], axes=[1]) -> activation__lastrow[1, 1, H]
    MatMul(activation__lastrow, weight)      -> logits[1, 1, vocab]

so the LM head runs with M=1. Three properties are being pinned:

  * it is **opt-in** — with no ``VKOP_LOGITS_LAST_ROW`` name set, the matcher
    returns nothing and the graph stays exactly as exported. The flag changes
    what a graph *output* means, and a model user who wants per-position
    logits must keep getting them;
  * it is **idempotent** — the optimizer runs every pass in a loop until the
    graph stops changing, and a second Slice would cut a row out of a tensor
    that is already one row;
  * it **declines** anything that is not an LM head: only a MatMul whose
    single output is one of the named graph outputs, of rank >= 3, is a
    candidate. There is no other node in the graph that could be meant.

The runtime side (a Slice the buffer backend resolves against the live seq
dim, and an M=1 MatMul) is the same machinery every other Slice/MatMul in the
graph already uses; BufferRankTest covers both ops.
"""
import mmap
import os
import sys

import numpy as np
import pytest
from onnx import numpy_helper

from onnx2vkop.dag import DAGBasedModel, Node
from onnx2vkop.optimizer import FusionOptimizer


def tdict(name, dims=None, dtype=10):
    return {"name": name, "shape": list(dims or []), "dtype": dtype}


def build(declared=("*",), a_dims=(1, 1860, 2048), out_dims=(1, 1860, 151936),
          node_out_dims=None, out_is_graph_output=True, b_dims=(2048, 151936),
          op_type="MatMul"):
    """An LM-head-shaped graph: one MatMul producing a graph output.

    ``a_dims``/``out_dims`` are what the *exporter* recorded. ``a_dims=None``
    stands for the real Qwen3-VL graph, where the activation is an
    intermediate with no value_info at all and the parser writes shape [].
    """
    m = DAGBasedModel()
    if declared:
        m.logits_last_row = set(declared)
    mm = Node(op_type, "LM_HEAD",
              {},
              [tdict("act", a_dims), tdict("w", b_dims)],
              [tdict("logits", node_out_dims if node_out_dims is not None
                     else out_dims)])
    m.nodes["LM_HEAD"] = mm
    if out_is_graph_output:
        m.outputs = [{"name": "logits", "shape": list(out_dims),
                      "dtype": "float16"}]
    return m


def matches_of(dag):
    return FusionOptimizer.match_logits_last_row(dag)


def folded(declared=("*",), **kw):
    dag = build(declared=declared, **kw)
    ms = matches_of(dag)
    assert ms, "the matcher declined a plain LM head"
    assert FusionOptimizer.fold_logits_last_row(dag, ms[0])
    return dag


def init_of(dag, name):
    return numpy_helper.to_array(dag.initializers[name])


def slice_node(dag):
    slices = [n for n in dag.nodes.values() if n.op_type == "Slice"]
    assert len(slices) == 1, f"expected one Slice, got {len(slices)}"
    return slices[0]


# ------------------------------------------------------------------ positives

def test_last_row_slice_is_inserted():
    """The whole point: the activation reaching the MatMul is one row."""
    dag = folded()
    sl = slice_node(dag)
    assert sl.inputs[0]["name"] == "act"
    starts, ends, axes = (init_of(dag, sl.inputs[i]["name"]) for i in (1, 2, 3))
    assert list(starts) == [-1]           # ONNX's last element
    assert list(axes) == [1]              # the seq axis of [b, L, H]
    assert list(ends) == [np.iinfo(np.int64).max]
    for i in (1, 2, 3):
        assert dag.initializers[sl.inputs[i]["name"]].data_type == 7  # int64


def test_matmul_is_rewired_and_output_shrinks():
    dag = folded()
    mm = dag.nodes["LM_HEAD"]
    assert mm.inputs[0]["name"] == slice_node(dag).outputs[0]["name"]
    assert mm.inputs[1]["name"] == "w"    # the weight is untouched
    assert [o for o in dag.outputs if o["name"] == "logits"][0]["shape"] == \
        [1, 1, 151936]
    assert mm.outputs[0]["shape"] == [1, 1, 151936]


def test_a_dims_missing_is_still_matched():
    """The real graph: /norm/Mul_1_output_0 has no value_info, so the parser
    records an empty shape. The axis comes from the graph output instead."""
    dag = folded(a_dims=None)
    assert list(init_of(dag, slice_node(dag).inputs[3]["name"])) == [1]


def test_named_output_is_sliced():
    dag = folded(declared=("logits",))
    assert dag.nodes["LM_HEAD"].inputs[0]["name"] != "act"


def test_batch_greater_than_one():
    """Only the seq axis collapses; a batch stays."""
    dag = folded(a_dims=(2, 1860, 2048), out_dims=(2, 1860, 151936))
    assert slice_node(dag).outputs[0]["shape"] == [2, 1, 2048]
    assert dag.nodes["LM_HEAD"].inputs[0]["name"] != "act"


def test_declared_name_does_not_leak_into_other_outputs():
    """'*' targets every MatMul-produced graph output; a name targets one."""
    m = DAGBasedModel()
    m.logits_last_row = {"logits"}
    for tag, out in (("A", "logits"), ("B", "other")):
        n = Node("MatMul", f"MM_{tag}", {},
                 [tdict(f"act_{tag}", (1, 8, 4)), tdict(f"w_{tag}", (4, 16))],
                 [tdict(out, (1, 8, 16))])
        m.nodes[n.name] = n
        m.outputs.append({"name": out, "shape": [1, 8, 16], "dtype": "float16"})
    assert [x["node"].name for x in matches_of(m)] == ["MM_A"]


# ------------------------------------------------------------------- declines

def test_silence_means_the_old_graph():
    """The default. A model converted without the flag must fold nothing —
    this is the property that keeps the 6/6 gate's reference valid."""
    assert matches_of(build(declared=())) == []
    assert matches_of(DAGBasedModel()) == []


def test_named_output_that_is_not_a_graph_output_declined():
    assert matches_of(build(out_is_graph_output=False)) == []


def test_rank2_output_declined():
    """A plain 2-D MatMul has no seq axis to collapse."""
    m = DAGBasedModel()
    m.logits_last_row = {"*"}
    m.nodes["MM"] = Node("MatMul", "MM", {},
                         [tdict("a", (8, 4)), tdict("w", (4, 16))],
                         [tdict("logits", (8, 16))])
    m.outputs = [{"name": "logits", "shape": [8, 16], "dtype": "float16"}]
    assert matches_of(m) == []


def test_3d_weight_declined():
    """Batched matmul (rank-3 B) is not an LM head."""
    assert matches_of(build(b_dims=(1, 2048, 151936))) == []


def test_non_matmul_producer_declined():
    assert matches_of(build(op_type="Add")) == []


def test_multi_output_matmul_declined():
    dag = build()
    dag.nodes["LM_HEAD"].outputs.append(tdict("second", (1, 1860, 151936)))
    assert matches_of(dag) == []


def test_already_sliced_is_declined():
    """Idempotence: the optimizer re-runs every matcher each round."""
    dag = folded()
    assert matches_of(dag) == []


def test_foreign_slice_is_respected():
    """A graph that already slices its LM head input must not be sliced
    again — the pass has no way to know what that Slice was for."""
    dag = build()
    dag.nodes["PRE"] = Node("Slice", "PRE", {},
                            [tdict("raw", (1, 1860, 2048)),
                             tdict("s", (1,), 7), tdict("e", (1,), 7),
                             tdict("a", (1,), 7)],
                            [tdict("act", (1, 1, 2048))])
    assert matches_of(dag) == []


def test_slice_does_not_shadow_an_existing_tensor():
    """The generated names must not collide with a real tensor."""
    dag = build()
    dag.nodes["OTHER"] = Node("Identity", "OTHER", {},
                              [tdict("x", (1, 4))],
                              [tdict("LM_HEAD__lastrow", (1, 4))])
    folded_dag = folded()
    assert "LM_HEAD__lastrow" not in folded_dag.initializers
    # the collision-free run keeps its plain name
    names = {n.outputs[0]["name"] for n in folded_dag.nodes.values()}
    assert "LM_HEAD__lastrow" in names


def test_colliding_name_gets_a_suffix():
    dag = build()
    dag.nodes["OTHER"] = Node("Identity", "OTHER", {},
                              [tdict("x", (1, 4))],
                              [tdict("LM_HEAD__lastrow", (1, 4))])
    ms = matches_of(dag)
    assert FusionOptimizer.fold_logits_last_row(dag, ms[0])
    assert slice_node(dag).outputs[0]["name"] == "LM_HEAD__lastrow_1"


# ------------------------------------------------- the optimizer plumbing

def test_pass_is_registered_and_runs_to_fixpoint():
    """Registered at priority 1, above prune_and_materialize (0) so the three
    initializers this pass inserts are still materialized. It must not report
    a change on the second round — the multi-round loop would spin."""
    optimizer = FusionOptimizer.create_default_optimizer()
    names = [p.name for p in optimizer.passes]
    assert "logits_last_row" in names
    by_name = {p.name: p for p in optimizer.passes}
    assert by_name["logits_last_row"].priority == 1
    assert by_name["logits_last_row"].priority > by_name[
        "prune_and_materialize"].priority
    dag = build()
    assert by_name["logits_last_row"].apply(dag) == 1
    assert by_name["logits_last_row"].apply(dag) == 0


def test_env_names_the_output(tmp_path, monkeypatch):
    """The conversion-time channel, mirroring VKOP_CAUSAL_MASK."""
    import onnx
    from onnx import TensorProto, helper

    from onnx2vkop.converter import ModelConverter

    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4])
    graph = helper.make_graph(
        [helper.make_node("Identity", ["x"], ["y"], name="ID")], "g", [x], [y])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    path = str(tmp_path / "tiny.onnx")
    onnx.save_model(model, path)

    monkeypatch.setenv("VKOP_LOGITS_LAST_ROW", "logits, logits2")
    dag = ModelConverter().parse_onnx_model(path, 1)
    assert dag.logits_last_row == {"logits", "logits2"}

    monkeypatch.setenv("VKOP_LOGITS_LAST_ROW", "")
    dag = ModelConverter().parse_onnx_model(path, 1)
    assert dag.logits_last_row == set()


def test_written_artifact_carries_the_slice():
    """The runtime reads the graph off the .vkopbin, not off the DAG: the
    Slice node, its three int64 initializers and the shrunken graph output all
    have to survive serialization or the model silently computes L rows
    again."""
    _gen = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "onnx2vkop", "generated")
    if _gen not in sys.path:
        sys.path.insert(0, _gen)
    from vkop.model.Model import Model

    dag = folded()
    dag.build_dependencies()
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        out = os.path.join(d, "m.vkopbin")
        dag.save_to_binary(out)
        with open(out, "rb") as fh:
            mm = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
            model = Model.GetRootAsModel(mm, 0)
            out_dims = None
            for i in range(model.OutputsLength()):
                o = model.Outputs(i)
                if o.Name() == b"logits":
                    out_dims = [o.Dims(j) for j in range(o.DimsLength())]
            op_types = [model.Nodes(i).OpType()
                        for i in range(model.NodesLength())]
            init_names = [model.Initializers(i).Name()
                          for i in range(model.InitializersLength())]
            mm.close()
    assert out_dims == [1, 1, 151936], out_dims
    assert b"Slice" in op_types, op_types
    assert b"MatMul" in op_types, op_types
    assert b"LM_HEAD__lastrow_starts" in init_names, init_names
