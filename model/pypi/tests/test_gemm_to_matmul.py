"""Standalone test for the Gemm -> MatMul(+Add) lowering pass.

ONNX Gemm and MatMul compute the same product; the runtime only carries
high-performance kernels behind MatMul (split-K GEMV, WIDE, coopmat,
tiled). The converter lowers every Gemm it can to MatMul, adding a
separate Add when the Gemm had a bias. Cases that are NOT plain MatMul
semantics -- a real alpha, transA, or a fused activation -- are left
untouched (fail-closed) so they are never silently mis-compiled.
"""
import mmap
import os
import sys
import tempfile

from onnx2vkop.dag import DAGBasedModel, Node
from onnx2vkop.optimizer import FusionOptimizer

# Model.Nodes(j) does a lazy *absolute* import of `vkop.model.Node` (generated
# FlatBuffers code), so any reader has to have onnx2vkop/generated on sys.path —
# same setup tools/ablate.py and the diagnostic scripts use.
_GENERATED = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "onnx2vkop", "generated")
if _GENERATED not in sys.path:
    sys.path.insert(0, _GENERATED)


def tdict(name, dims=None, dtype=1):
    return {"name": name, "shape": list(dims or []), "dtype": dtype}


def gemm_node(name, attrs, inputs, out_name="gemm_out"):
    return Node("Gemm", name, attrs, inputs, [tdict(out_name, [8, 16])])


def count(dag, op_type):
    return sum(1 for n in dag.nodes.values() if n.op_type == op_type)


def matches_of(dag):
    return FusionOptimizer.match_gemm_to_matmul(dag)


def test_plain_gemm_no_bias_lowers_to_matmul():
    """alpha=1, no C: Gemm(A,B) -> MatMul(A,B), same output name preserved."""
    m = DAGBasedModel()
    g = gemm_node("G", {"alpha": 1.0, "beta": 1.0, "transA": 0, "transB": 0},
                  [tdict("a", [8, 32]), tdict("b", [32, 16])])
    m.nodes[g.name] = g
    m.outputs = [g.outputs[0]]
    ms = matches_of(m)
    assert len(ms) == 1, ms
    FusionOptimizer.fold_gemm_to_matmul(m, ms[0])
    assert count(m, "Gemm") == 0
    assert count(m, "MatMul") == 1
    assert count(m, "Add") == 0
    mm = m.nodes["G"]
    assert mm.op_type == "MatMul"
    # output name unchanged -> graph consumers still resolve
    assert mm.outputs[0]["name"] == "gemm_out"
    assert mm.attributes == {}, mm.attributes
    print("  PASS (plain Gemm -> MatMul, output preserved)")


def test_gemm_with_bias_becomes_matmul_plus_add():
    """beta=1, has C: Gemm(A,B,C) -> MatMul(A,B)->mm + Add(mm, C)->orig_out."""
    m = DAGBasedModel()
    g = gemm_node("G", {"alpha": 1.0, "beta": 1.0, "transA": 0, "transB": 0},
                  [tdict("a", [8, 32]), tdict("b", [32, 16]), tdict("c", [16])])
    m.nodes[g.name] = g
    m.outputs = [g.outputs[0]]
    ms = matches_of(m)
    assert len(ms) == 1, ms
    FusionOptimizer.fold_gemm_to_matmul(m, ms[0])
    assert count(m, "Gemm") == 0
    assert count(m, "MatMul") == 1
    assert count(m, "Add") == 1
    mm = m.nodes["G_mm"]
    add = m.nodes["G_bias"]
    assert mm.op_type == "MatMul"
    assert [i["name"] for i in mm.inputs] == ["a", "b"]
    # Add reads the MatMul's fresh output + the bias, writes the Gemm output.
    assert add.inputs[0]["name"] == "gemm_out_mm"
    assert add.inputs[1]["name"] == "c"
    assert add.outputs[0]["name"] == "gemm_out"
    print("  PASS (Gemm+bias -> MatMul + Add, names wired)")


def test_transb_carried_as_attribute():
    """transB=1 survives as a MatMul attribute (runtime reads B transposed)."""
    m = DAGBasedModel()
    g = gemm_node("G", {"alpha": 1.0, "beta": 0.0, "transA": 0, "transB": 1},
                  [tdict("a", [8, 32]), tdict("b", [16, 32])])
    m.nodes[g.name] = g
    ms = matches_of(m)
    assert len(ms) == 1, ms
    FusionOptimizer.fold_gemm_to_matmul(m, ms[0])
    mm = m.nodes["G"]
    assert mm.op_type == "MatMul"
    assert mm.attributes == {"transB": 1}, mm.attributes
    print("  PASS (transB=1 carried onto MatMul)")


def test_alpha_not_one_rejected():
    """A real alpha != 1 is not a plain MatMul; fail closed."""
    m = DAGBasedModel()
    g = gemm_node("G", {"alpha": 2.0, "beta": 0.0, "transA": 0, "transB": 0},
                  [tdict("a", [8, 32]), tdict("b", [32, 16])])
    m.nodes[g.name] = g
    assert matches_of(m) == []
    assert count(m, "Gemm") == 1  # untouched
    print("  PASS (alpha=2.0 rejected)")


def test_transa_rejected():
    """transA=1 transposes A -- MatMul has no such flag; fail closed."""
    m = DAGBasedModel()
    g = gemm_node("G", {"alpha": 1.0, "beta": 0.0, "transA": 1, "transB": 0},
                  [tdict("a", [32, 8]), tdict("b", [32, 16])])
    m.nodes[g.name] = g
    assert matches_of(m) == []
    print("  PASS (transA=1 rejected)")


def test_fused_activation_rejected():
    """A fused activation cannot be expressed as MatMul(+Add); fail closed."""
    m = DAGBasedModel()
    g = gemm_node("G", {"alpha": 1.0, "beta": 0.0, "transA": 0, "transB": 0,
                        "activation": "Relu"},
                  [tdict("a", [8, 32]), tdict("b", [32, 16])])
    m.nodes[g.name] = g
    assert matches_of(m) == []
    print("  PASS (fused Relu rejected)")


def test_beta_zero_no_bias_lowers():
    """beta=0 with no bias is the plain MatMul case; must lower."""
    m = DAGBasedModel()
    g = gemm_node("G", {"alpha": 1.0, "beta": 0.0, "transA": 0, "transB": 0},
                  [tdict("a", [8, 32]), tdict("b", [32, 16])])
    m.nodes[g.name] = g
    ms = matches_of(m)
    assert len(ms) == 1 and ms[0]["beta"] == 0.0
    FusionOptimizer.fold_gemm_to_matmul(m, ms[0])
    assert count(m, "MatMul") == 1 and count(m, "Add") == 0
    print("  PASS (beta=0, no bias -> MatMul)")


def test_default_attributes_lower():
    """A Gemm with NO attributes at all (alpha/beta/transA default) lowers."""
    m = DAGBasedModel()
    g = gemm_node("G", {}, [tdict("a", [8, 32]), tdict("b", [32, 16])])
    m.nodes[g.name] = g
    ms = matches_of(m)
    assert len(ms) == 1, ms
    FusionOptimizer.fold_gemm_to_matmul(m, ms[0])
    assert count(m, "MatMul") == 1
    print("  PASS (attribute-less Gemm lowers on defaults)")


def test_lowered_add_input_carries_shape_metadata():
    """The Add's operand for the MatMul output must be a full tensor ref.

    Every node input in the tree is {name, shape, dtype, value_dynamic} -- the
    parser builds them that way, and writes an explicit "shape": [] for a tensor
    it cannot resolve. The lowering used to wire the Add with a bare
    {"name": mm_out_name}, so visual.onnx (104 such refs) died in
    dag.build_shape_vec with KeyError('shape').
    """
    m = DAGBasedModel()
    g = gemm_node("G", {"alpha": 1.0, "beta": 1.0, "transA": 0, "transB": 0},
                  [tdict("a", [8, 32]), tdict("b", [32, 16]), tdict("c", [16])])
    m.nodes[g.name] = g
    m.outputs = [g.outputs[0]]
    ms = matches_of(m)
    assert len(ms) == 1, ms
    FusionOptimizer.fold_gemm_to_matmul(m, ms[0])

    for name, n in m.nodes.items():
        for i in n.inputs:
            assert "shape" in i, f"input {i.get('name')} of {name} has no shape"
    # The ref describes the MatMul's fresh output, so it carries the Gemm's
    # output dims (a bias add does not change the shape) and its dtype.
    mm_in = m.nodes["G_bias"].inputs[0]
    assert mm_in["name"] == "gemm_out_mm", mm_in
    assert mm_in["shape"] == [8, 16], mm_in
    assert mm_in["dtype"] == 1, mm_in
    print("  PASS (lowered Add input carries shape/dtype)")


def test_lowered_gemm_model_saves_and_records_the_dims():
    """End-to-end: the saved model must carry the MatMul-output dims.

    The lowering's metadata is what the *runtime* consumes -- an input ref with
    no dims is read as "unknown" (core/runtime.cpp:743 pushes nullptr when the
    tensor is also absent from the map, and node_input_shapes_ treats an empty
    recorded shape as rank-0). So assert on the written artifact, not just on
    the in-memory dict: a tolerant writer alone would hide the regression.
    """
    from onnx2vkop.generated.vkop.model.Model import Model

    m = DAGBasedModel()
    g = gemm_node("G", {"alpha": 1.0, "beta": 1.0, "transA": 0, "transB": 0},
                  [tdict("a", [8, 32]), tdict("b", [32, 16]), tdict("c", [16])])
    m.nodes[g.name] = g
    m.inputs = [tdict("a", [8, 32]), tdict("b", [32, 16]), tdict("c", [16])]
    m.outputs = [g.outputs[0]]
    ms = matches_of(m)
    assert len(ms) == 1, ms
    FusionOptimizer.fold_gemm_to_matmul(m, ms[0])
    m.build_dependencies()

    with tempfile.TemporaryDirectory() as d:
        out = os.path.join(d, "gemm_bias.vkopbin")
        m.save_to_binary(out)
        with open(out, "rb") as fh:
            mm = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
            model = Model.GetRootAsModel(mm, 0)
            found = None
            for i in range(model.NodesLength()):
                node = model.Nodes(i)
                if node.OpType() == b"Add":
                    found = node
                    break
            assert found is not None, "lowered Add missing from the artifact"
            src = found.Inputs(0)
            dims = [src.Dims(k) for k in range(src.DimsLength())]
            mm.close()
    assert dims == [8, 16], f"Add's MatMul-input dims not recorded: {dims}"
    print("  PASS (lowered Gemm+bias model saves with the Add's dims recorded)")
