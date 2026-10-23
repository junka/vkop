"""Tests for the "keep only the last logits position" pass.

The pass folds

    MatMul(activation[1, L, H], lm_head_weight[H, vocab]) -> logits[1, L, vocab]
                                                                  ^ graph output

into

    Slice(activation, starts=[-1], axes=[1]) -> activation__lastrow[1, 1, H]
    MatMul(activation__lastrow, weight)      -> logits[1, 1, vocab]

so the LM head runs with M=1. Four properties are being pinned:

  * it is **declared, not guessed** — an LM head that says nothing keeps
    computing all L rows, because the graph itself cannot show which rows a
    caller reads (``logits`` has no in-graph consumer). Two channels say it:
    the model-level metadata prop ``vkop_logits_last_row:<output>`` (the
    carrier that travels with the artifact) and the conversion-time env var
    ``VKOP_LOGITS_LAST_ROW`` (the bridge for an .onnx that cannot be
    re-exported);
  * it is **self-described** — a graph that already contains the last-row
    ``Slice`` needs neither declaration. ``Slice`` commutes with a ``MatMul``
    (rows are independent), so the pass pushes it in front, which is a
    peephole and fires on its own;
  * it is **idempotent** — the optimizer runs every pass in a loop until the
    graph stops changing, and a second Slice would cut a row out of a tensor
    that is already one row;
  * it **declines** anything that is not an LM head: only a MatMul whose
    (sliced or direct) output is a graph output, of rank >= 3, is a
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


def _i64(name, value):
    return numpy_helper.from_array(np.array([value], dtype=np.int64), name)


def build_embedded(a_dims=(1, 1860, 2048), out_dims=(1, 1, 151936),
                   axes=1, starts=-1, ends=np.iinfo(np.int64).max,
                   slice_is_graph_output=True, slice_out_dims=None,
                   extra_consumer=False, mm_out_also_graph_output=False,
                   declared=(), op_type="MatMul"):
    """A graph that already slices its own LM head — the exporter said which
    rows it wants, so nothing has to be declared:

        MatMul(act, w) -> mm_raw ; Slice(mm_raw, last, axes=[1]) -> logits
    """
    m = DAGBasedModel()
    if declared:
        m.logits_last_row = set(declared)
    m.nodes["LM_HEAD"] = Node(
        op_type, "LM_HEAD", {},
        [tdict("act", a_dims), tdict("w", (2048, 151936))],
        [tdict("mm_raw", (1, 1860, 151936))])
    m.initializers.update({n: _i64(n, v) for n, v in
                           (("s", starts), ("e", ends), ("a", axes))})
    sl_out = tdict("logits",
                   slice_out_dims if slice_out_dims is not None else out_dims)
    m.nodes["SLICE"] = Node(
        "Slice", "SLICE", {},
        [tdict("mm_raw", (1, 1860, 151936)),
         tdict("s", (1,), 7), tdict("e", (1,), 7), tdict("a", (1,), 7)],
        [sl_out])
    if slice_is_graph_output:
        m.outputs.append({"name": "logits", "shape": list(out_dims),
                          "dtype": "float16"})
    if mm_out_also_graph_output:
        m.outputs.append({"name": "mm_raw", "shape": [1, 1860, 151936],
                          "dtype": "float16"})
    if extra_consumer:
        m.nodes["EXTRA"] = Node(
            "Identity", "EXTRA", {},
            [tdict("mm_raw", (1, 1860, 151936))],
            [tdict("mm_copy", (1, 1860, 151936))])
    return m


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


# ------------------------------------------------- the self-described form

def test_graph_embedded_slice_folds_with_no_declaration():
    """The exporter already wrote the Slice, so the graph states the premise
    itself. No metadata, no env var: the fold is a peephole."""
    dag = build_embedded()
    assert dag.logits_last_row == set()
    ms = matches_of(dag)
    assert [x["kind"] for x in ms] == ["self"]
    assert FusionOptimizer.fold_logits_last_row(dag, ms[0])
    # the old Slice node moved in front of the MatMul; its output tensor went
    # away with it, and the MatMul now writes the name downstream was reading
    assert "SLICE" not in dag.nodes
    produced = {o["name"] for n in dag.nodes.values() for o in n.outputs}
    assert "mm_raw" not in produced
    sl = slice_node(dag)
    assert sl.inputs[0]["name"] == "act"
    mm = dag.nodes["LM_HEAD"]
    assert mm.inputs[0]["name"] == sl.outputs[0]["name"]
    assert mm.inputs[1]["name"] == "w"
    assert mm.outputs[0]["name"] == "logits"
    assert mm.outputs[0]["shape"] == [1, 1, 151936]
    assert [o for o in dag.outputs if o["name"] == "logits"][0]["shape"] == \
        [1, 1, 151936]


def test_embedded_bounds_are_reused_not_rebuilt():
    """The Slice's own starts/ends/axes are already exactly right; the fold
    must not leave a second set of them behind."""
    dag = build_embedded()
    FusionOptimizer.fold_logits_last_row(dag, matches_of(dag)[0])
    sl = slice_node(dag)
    assert [sl.inputs[i]["name"] for i in (1, 2, 3)] == ["s", "e", "a"]
    assert set(dag.initializers) == {"s", "e", "a"}
    assert list(init_of(dag, "a")) == [1]


def test_embedded_fold_is_idempotent():
    dag = build_embedded()
    FusionOptimizer.fold_logits_last_row(dag, matches_of(dag)[0])
    assert matches_of(dag) == []


def test_embedded_needs_no_registration_to_see_it():
    """The whole point of the auto form: a plain optimizer run does it."""
    dag = build_embedded()
    assert FusionOptimizer.optimize(dag) is dag
    assert dag.nodes["LM_HEAD"].inputs[0]["name"] != "act"
    assert "SLICE" not in dag.nodes


def test_embedded_and_declared_produce_the_same_graph():
    a = build_embedded()
    FusionOptimizer.fold_logits_last_row(a, matches_of(a)[0])
    b = folded()
    assert a.nodes["LM_HEAD"].inputs[0]["name"] == \
        b.nodes["LM_HEAD"].inputs[0]["name"]
    assert a.nodes["LM_HEAD"].outputs[0]["name"] == \
        b.nodes["LM_HEAD"].outputs[0]["name"]
    assert a.nodes["LM_HEAD"].outputs[0]["shape"] == \
        b.nodes["LM_HEAD"].outputs[0]["shape"]


def test_embedded_with_a_declaration_also_present_matches_once():
    """Both channels naming the same output must not produce two folds."""
    dag = build_embedded(declared=("logits",))
    ms = matches_of(dag)
    assert len(ms) == 1, ms
    assert ms[0]["kind"] == "self"


def test_embedded_non_last_row_slice_declined():
    assert matches_of(build_embedded(starts=0)) == []


def test_embedded_real_length_end_declined():
    """ends=[1860] is a concrete length, not "to the end" — the pass has no
    proof it is the last row of whatever L the runtime sees."""
    assert matches_of(build_embedded(ends=1860)) == []


def test_embedded_wrong_axis_declined():
    """Slicing the feature axis (v) does not commute with the MatMul: it is a
    slice of the weight's columns, not of a row."""
    assert matches_of(build_embedded(axes=2)) == []


def test_embedded_negative_axis_is_normalized():
    """axes=[-2] on a rank-3 output is ONNX's spelling of axes=[1], so it is
    the same fold; axes=[-1] would be the feature axis and is not."""
    assert [x["kind"] for x in matches_of(build_embedded(axes=-2))] == ["self"]
    assert matches_of(build_embedded(axes=-1)) == []


def test_embedded_omitted_steps_still_folds():
    """A blank optional input is ONNX's "not supplied"; an omitted step is 1."""
    dag = build_embedded()
    dag.nodes["SLICE"].inputs.append(tdict("", (), 7))
    assert [x["kind"] for x in matches_of(dag)] == ["self"]


def test_embedded_nonunit_step_declined():
    dag = build_embedded()
    dag.initializers["st"] = _i64("st", 2)
    dag.nodes["SLICE"].inputs.append(tdict("st", (1,), 7))
    assert matches_of(dag) == []


def test_embedded_slice_output_not_a_graph_output_declined():
    """Only a graph output's tail is taken, so the slice can never start
    walking backwards through whatever produced the activation."""
    assert matches_of(build_embedded(slice_is_graph_output=False)) == []


def test_embedded_shared_matmul_output_declined():
    assert matches_of(build_embedded(extra_consumer=True)) == []


def test_embedded_matmul_output_also_a_graph_output_declined():
    assert matches_of(build_embedded(mm_out_also_graph_output=True)) == []


def test_embedded_non_matmul_producer_declined():
    assert matches_of(build_embedded(op_type="Add")) == []


def test_embedded_slice_on_a_non_matmul_tensor_declined():
    """A plain Slice(a) -> logits is not this pattern; the MatMul is what
    makes the pushed-back Slice free."""
    m = DAGBasedModel()
    m.initializers.update({n: _i64(n, v) for n, v in
                           (("s", -1), ("e", np.iinfo(np.int64).max), ("a", 1))})
    m.nodes["SLICE"] = Node(
        "Slice", "SLICE", {},
        [tdict("act", (1, 8, 4)), tdict("s", (1,), 7), tdict("e", (1,), 7),
         tdict("a", (1,), 7)],
        [tdict("logits", (1, 1, 4))])
    m.outputs = [{"name": "logits", "shape": [1, 1, 4], "dtype": "float16"}]
    assert matches_of(m) == []


# ------------------------------------- the declaration's converter plumbing

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
    """The bridge half, mirroring VKOP_CAUSAL_MASK: a model exported before
    the declaration existed can still be reconverted with it."""
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


def _tiny_onnx(tmp_path, metadata=(), monkeypatch=None):
    """A convertible 1-node ONNX, small enough that the real onnxoptimizer +
    checker pass runs on it."""
    import onnx
    from onnx import TensorProto, helper

    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4])
    graph = helper.make_graph(
        [helper.make_node("Identity", ["x"], ["y"], name="ID")], "g", [x], [y])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    for key, val in metadata:
        model.metadata_props.add(key=key, value=val)
    path = str(tmp_path / "tiny.onnx")
    onnx.save_model(model, path)
    if monkeypatch is not None:
        monkeypatch.delenv("VKOP_LOGITS_LAST_ROW", raising=False)
    return path


def test_model_metadata_declares_last_row(tmp_path, monkeypatch):
    """The carrier half: the declaration travels with the artifact instead of
    with the invocation, so a model converted once is converted the same way
    forever. Model-level (not ValueInfoProto) because onnxoptimizer.optimize()
    rebuilds the graph and drops the per-value-info props — the same reason
    vkop_io_dtype and vkop_causal_mask are model-level."""
    from onnx2vkop.converter import ModelConverter

    path = _tiny_onnx(tmp_path,
                      metadata=[("vkop_logits_last_row:logits", "1")],
                      monkeypatch=monkeypatch)
    dag = ModelConverter().parse_onnx_model(path, 1)
    assert dag.logits_last_row == {"logits"}


def test_metadata_zero_declares_nothing(tmp_path, monkeypatch):
    """Same off-spelling the causal declaration accepts: a "0" value is a
    checked-in template's way of saying "not this model"."""
    from onnx2vkop.converter import ModelConverter

    path = _tiny_onnx(tmp_path,
                      metadata=[("vkop_logits_last_row:logits", "0")],
                      monkeypatch=monkeypatch)
    dag = ModelConverter().parse_onnx_model(path, 1)
    assert dag.logits_last_row == set()


def _lm_onnx(tmp_path, embed_slice):
    """A real (parseable, checkable) 1-vector LM head, optionally with the
    last-row Slice already written into it. Every value_info has to match what
    the graph actually computes or onnxoptimizer's shape pass walks off it."""
    import onnx
    from onnx import TensorProto, helper

    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 8, 4])
    inits = [numpy_helper.from_array(np.zeros((4, 16), dtype=np.float32), "w")]
    if embed_slice:
        for n, v in (("s", -1), ("e", np.iinfo(np.int64).max), ("a", 1)):
            inits.append(numpy_helper.from_array(
                np.array([v], dtype=np.int64), n))
        nodes = [helper.make_node("MatMul", ["x", "w"], ["raw"], name="LM_HEAD"),
                 helper.make_node("Slice", ["raw", "s", "e", "a"], ["logits"],
                                  name="SLICE")]
        y = helper.make_tensor_value_info("logits", TensorProto.FLOAT, [1, 1, 16])
        raw = helper.make_tensor_value_info("raw", TensorProto.FLOAT, [1, 8, 16])
        graph = helper.make_graph(nodes, "g", [x], [y], inits, value_info=[raw])
    else:
        nodes = [helper.make_node("MatMul", ["x", "w"], ["logits"], name="LM_HEAD")]
        y = helper.make_tensor_value_info("logits", TensorProto.FLOAT, [1, 8, 16])
        graph = helper.make_graph(nodes, "g", [x], [y], inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    path = str(tmp_path / "lm.onnx")
    onnx.save_model(model, path)
    return path


def test_metadata_declaration_reaches_the_fold(tmp_path, monkeypatch):
    """End to end through the declaration channel: metadata → DAG → optimizer
    → the Slice sits in front of the MatMul, with no env var set anywhere."""
    import onnx
    from onnx2vkop.converter import ModelConverter

    monkeypatch.delenv("VKOP_LOGITS_LAST_ROW", raising=False)
    path = _lm_onnx(tmp_path, embed_slice=False)
    model = onnx.load(path)
    model.metadata_props.add(key="vkop_logits_last_row:logits", value="1")
    onnx.save_model(model, path)

    dag = ModelConverter().parse_onnx_model(path, 1)
    assert dag.logits_last_row == {"logits"}
    FusionOptimizer.optimize(dag)
    assert any(n.op_type == "Slice" for n in dag.nodes.values())
    mm = next(n for n in dag.nodes.values() if n.op_type == "MatMul")
    assert mm.inputs[0]["name"] != "x"
    assert [o for o in dag.outputs if o["name"] == "logits"][0]["shape"] == [1, 1, 16]


def test_embedded_slice_folds_end_to_end(tmp_path, monkeypatch):
    """The same for the self-described form, parsed off a real ONNX: no
    declaration of any kind is needed."""
    from onnx2vkop.converter import ModelConverter

    monkeypatch.delenv("VKOP_LOGITS_LAST_ROW", raising=False)
    path = _lm_onnx(tmp_path, embed_slice=True)
    dag = ModelConverter().parse_onnx_model(path, 1)
    assert dag.logits_last_row == set()
    FusionOptimizer.optimize(dag)
    mm = next(n for n in dag.nodes.values() if n.op_type == "MatMul")
    assert mm.inputs[0]["name"] != "x"
    assert mm.outputs[0]["name"] == "logits"
    assert [o for o in dag.outputs if o["name"] == "logits"][0]["shape"] == [1, 1, 16]


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
