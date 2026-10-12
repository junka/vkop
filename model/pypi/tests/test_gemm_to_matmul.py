"""Standalone test for the Gemm -> MatMul(+Add) lowering pass.

ONNX Gemm and MatMul compute the same product; the runtime only carries
high-performance kernels behind MatMul (split-K GEMV, WIDE, coopmat,
tiled). The converter lowers every Gemm it can to MatMul, adding a
separate Add when the Gemm had a bias. Cases that are NOT plain MatMul
semantics -- a real alpha, transA, or a fused activation -- are left
untouched (fail-closed) so they are never silently mis-compiled.
"""
from onnx2vkop.dag import DAGBasedModel, Node
from onnx2vkop.optimizer import FusionOptimizer


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
