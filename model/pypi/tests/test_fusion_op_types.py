"""A fusion pass must not be able to emit an op the runtime cannot build.

Four passes shipped emitting SkipLayerNormalization / BiasGelu / MatMulGelu /
EmbeddingLayerNormalization / BertEncoder — none of which exist in ops/Ops.hpp.
The runtime resolves those names to OpType::UNKNOWN and skips the node outright
(core/runtime.cpp:702), so the fold deleted real computation and every consumer
read an unwritten buffer: silent wrong answers, not a crash, and no conversion
error to notice. Nothing in the pipeline compared the names the passes invent
against the names the runtime knows, so nothing said a word.

These tests pin the two halves of the fix:
  * RUNTIME_OPS is the runtime's own list, re-derived from ops/Ops.hpp, so the
    mirror cannot rot without failing here;
  * PatternBasedFusionPass.apply rejects an unknown op_type immediately, and
    MultiRoundOptimizer lets that error out instead of logging it and carrying
    on with a graph the failing pass already rewrote.
"""
from pathlib import Path

import pytest

from onnx2vkop.dag import DAGBasedModel, Node
from onnx2vkop.optimizer import (
    RUNTIME_OPS,
    MultiRoundOptimizer,
    OptimizationPass,
    PatternBasedFusionPass,
    UnknownFusedOpError,
    FusionOptimizer,
)

_OPS_HPP = Path(__file__).resolve().parents[3] / "ops" / "Ops.hpp"


def td(name):
    return {"name": name, "shape": [2, 4], "dtype": 1}


def test_runtime_ops_matches_convert_opstring_to_enum():
    """RUNTIME_OPS must equal the names convert_opstring_to_enum accepts."""
    src = _OPS_HPP.read_text()
    body = src[src.index("convert_opstring_to_enum"):]
    body = body[:body.index("\n}")]
    accepted = set()
    for line in body.splitlines():
        if "name ==" not in line:
            continue
        for part in line.split("name ==")[1:]:
            accepted.add(part.split('"')[1])
    assert accepted, "parsed no names out of ops/Ops.hpp — parser needs updating"
    assert accepted == set(RUNTIME_OPS), (
        f"mirror is stale: only in Ops.hpp {sorted(accepted - RUNTIME_OPS)}, "
        f"only in RUNTIME_OPS {sorted(set(RUNTIME_OPS) - accepted)}")


def test_invented_op_type_is_a_hard_error():
    """The exact shape of the bug: a folder emits a name the runtime lacks."""
    m = DAGBasedModel()
    m.inputs, m.outputs = [td("x")], [td("out")]
    m.nodes["src"] = Node("Identity", "src", {}, [td("x")], [td("y")])

    def matcher(dag_model):
        return [{"node": dag_model.nodes["src"]}]

    def folder(dag_model, match):
        del dag_model.nodes["src"]
        dag_model.nodes["fused"] = Node("SkipLayerNormalization", "fused", {},
                                        [td("x")], [td("out")])
        return True

    opt = MultiRoundOptimizer(max_rounds=3)
    opt.register_pass(PatternBasedFusionPass("fuse_invented", matcher, folder))
    with pytest.raises(UnknownFusedOpError) as e:
        opt.optimize(m, verbose=False)
    assert "SkipLayerNormalization" in str(e.value)
    assert "fuse_invented" in str(e.value)


def test_emitting_a_known_op_is_fine():
    m = DAGBasedModel()
    m.inputs, m.outputs = [td("x")], [td("out")]
    m.nodes["src"] = Node("Identity", "src", {}, [td("x")], [td("y")])

    def matcher(dag_model):
        return [{"node": dag_model.nodes["src"]}]

    def folder(dag_model, match):
        del dag_model.nodes["src"]
        dag_model.nodes["fused"] = Node("RMSNorm", "fused", {}, [td("x")], [td("out")])
        return True

    opt = MultiRoundOptimizer(max_rounds=3)
    opt.register_pass(PatternBasedFusionPass("fuse_known", matcher, folder))
    assert opt.optimize(m, verbose=False).patterns_fused == {"fuse_known": 1}


def _residual_graph(shared_residual=False, n_ln_inputs=3):
    """Add(x, skip) -> LayerNormalization(X, scale, bias) -> out.

    With shared_residual the Add's output also feeds a second consumer, which is
    what a pre-norm transformer looks like (the residual stream continues into
    the next sub-layer) and makes the fold invalid.
    """
    m = DAGBasedModel()
    m.inputs = [td("x"), td("skip"), td("scale"), td("bias")]
    m.outputs = [td("out")]
    add = Node("Add", "add", {}, [td("x"), td("skip")], [td("sum")])
    m.nodes["add"] = add
    ln_inputs = [td("sum"), td("scale"), td("bias")][:n_ln_inputs]
    m.nodes["ln"] = Node("LayerNormalization", "ln", {"epsilon": 1e-6, "axis": -1},
                         ln_inputs, [td("out")])
    if shared_residual:
        # A second consumer of the residual stream (the next sub-layer's Add).
        m.nodes["next_add"] = Node("Add", "next_add", {}, [td("sum"), td("x")],
                                   [td("out2")])
        m.outputs.append(td("out2"))
    return m


def test_matching_residual_folds_into_a_four_input_layernorm():
    m = _residual_graph()
    matches = FusionOptimizer.match_skip_layernorm(m)
    assert len(matches) == 1

    assert FusionOptimizer.fold_skip_layernorm(m, matches[0])
    assert "add" not in m.nodes and "ln" not in m.nodes
    fused = [n for n in m.nodes.values() if n.op_type == "LayerNormalization"]
    assert len(fused) == 1
    # X, Scale, Bias carried through unmodified, skip appended as input 3 —
    # the scale/bias the previous matcher dropped.
    assert [i["name"] for i in fused[0].inputs] == ["x", "scale", "bias", "skip"]
    assert [o["name"] for o in fused[0].outputs] == ["out"]
    assert fused[0].attributes == {"epsilon": 1e-6, "axis": -1}
    assert fused[0].op_type in RUNTIME_OPS


def test_shared_residual_is_not_fused():
    """A pre-norm residual keeps a second consumer, so deleting the Add would
    strand it — the case that covered 47 of visual.onnx's 48 Add -> LN pairs."""
    m = _residual_graph(shared_residual=True)
    assert FusionOptimizer.match_skip_layernorm(m) == []


def test_layernorm_without_bias_is_left_alone():
    """A 2-input LayerNormalization would put skip at index 2 (the bias slot)."""
    m = _residual_graph(n_ln_inputs=2)
    assert FusionOptimizer.match_skip_layernorm(m) == []


def test_deleted_passes_are_not_registered():
    names = {p.name for p in FusionOptimizer.create_default_optimizer().passes}
    assert names.isdisjoint({
        "fuse_bias_gelu", "fuse_embedding_layernorm", "fuse_bert_encoder"})
    assert "fuse_skip_layernorm" in names
