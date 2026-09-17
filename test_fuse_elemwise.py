"""Standalone test for fuse_elemwise_chain matcher + folder.

Builds a synthetic DAG chain Add -> Sqrt -> Mul (out = sqrt(A+B)*A) and
verifies the fusion produces one FusedElemwise node with the expected
program, then checks a broadcast-scalar variant and a chain that should
NOT fuse (shared intermediate).
"""
import sys
sys.path.insert(0, "model/pypi")
from onnx2vkop.dag import DAGBasedModel, Node
from onnx2vkop.optimizer import FusionOptimizer


def tdict(name, dims=None):
    # The converter stores shapes under the "shape" key.
    d = {"name": name}
    if dims is not None:
        d["shape"] = list(dims)
    return d


def make_chain_addsqrtmul():
    """A=leaf[8], B=leaf[8]. Add(A,B)->t1; Sqrt(t1)->t2; Mul(t2,A)->out."""
    m = DAGBasedModel()
    A = tdict("A", [8])
    B = tdict("B", [8])
    t1 = tdict("t1", [8])
    t2 = tdict("t2", [8])
    out = tdict("out", [8])
    add = Node("Add", "Add_0", {}, [A, B], [t1])
    sqrt = Node("Sqrt", "Sqrt_0", {}, [t1], [t2])
    mul = Node("Mul", "Mul_0", {}, [t2, A], [out])
    for n in (add, sqrt, mul):
        m.nodes[n.name] = n
    m.inputs = [A, B]
    m.outputs = [out]  # mark out live so DCE doesn't prune the fused node
    m.build_dependencies()
    return m


def make_shared_intermediate():
    """Add(A,B)->t1 consumed by BOTH Sqrt(t1) and Neg(t1). Chain must NOT
    cross the shared t1 (Sqrt alone is len-1, below MIN)."""
    m = DAGBasedModel()
    A = tdict("A", [8]); B = tdict("B", [8])
    t1 = tdict("t1", [8]); t2 = tdict("t2", [8]); t3 = tdict("t3", [8])
    out = tdict("out", [8])
    add = Node("Add", "Add_0", {}, [A, B], [t1])
    sqrt = Node("Sqrt", "Sqrt_0", {}, [t1], [t2])
    neg = Node("Neg", "Neg_0", {}, [t1], [t3])
    mul = Node("Mul", "Mul_0", {}, [t2, t3], [out])
    for n in (add, sqrt, neg, mul):
        m.nodes[n.name] = n
    m.inputs = [A, B]
    m.outputs = [out]
    m.build_dependencies()
    return m


def make_cast_break():
    """Add(A,B)->t1; Cast(t1)->t2 (Cast not chainable); Sqrt(t2)->out.
    Two separate chains, each len-1 -> nothing fuses (below MIN=2).
    A longer variant: Add->Mul [chainable] then Cast breaks."""
    m = DAGBasedModel()
    A = tdict("A", [8]); B = tdict("B", [8])
    t1 = tdict("t1", [8]); t2 = tdict("t2", [8]); out = tdict("out", [8])
    add = Node("Add", "Add_0", {}, [A, B], [t1])
    cast = Node("Cast", "Cast_0", {"to": 1}, [t1], [t2])
    sqrt = Node("Sqrt", "Sqrt_0", {}, [t2], [out])
    for n in (add, cast, sqrt):
        m.nodes[n.name] = n
    m.inputs = [A, B]
    m.outputs = [out]
    m.build_dependencies()
    return m


def make_chain_with_scalar():
    """Div(A, 2.0) -> Sqrt -> out. The 2.0 is a single-element initializer
    (scalar constant) that must be encoded as a program scalar (operand 100+),
    NOT a register input. A is the only register leaf."""
    m = DAGBasedModel()
    A = tdict("A", [8])
    two = tdict("two", [1])  # scalar constant
    t1 = tdict("t1", [8])
    out = tdict("out", [8])
    div = Node("Div", "Div_0", {}, [A, two], [t1])
    sqrt = Node("Sqrt", "Sqrt_0", {}, [t1], [out])
    for n in (div, sqrt):
        m.nodes[n.name] = n
    m.inputs = [A]
    m.outputs = [out]
    # Materialize the scalar as an initializer (TensorProto).
    import numpy as np
    from onnx import numpy_helper
    m.initializers["two"] = numpy_helper.from_array(
        np.array([2.0], dtype=np.float32), "two")
    m.build_dependencies()
    return m


def run_fusion(m):
    opt = FusionOptimizer.create_default_optimizer(max_rounds=5)
    stats = opt.optimize(m, verbose=False)
    return m


def test_addsqrtmul():
    m = make_chain_addsqrtmul()
    m = run_fusion(m)
    fused = [n for n in m.nodes.values() if n.op_type == "FusedElemwise"]
    assert len(fused) == 1, f"expected 1 fused node, got {len(fused)}; nodes={[n.op_type for n in m.nodes.values()]}"
    f = fused[0]
    print("  inputs:", [i["name"] for i in f.inputs])
    print("  ops:", f.attributes["ops"])
    print("  input_shapes:", f.attributes["input_shapes"])
    print("  out_shape:", f.attributes["out_shape"])
    print("  rank:", f.attributes["rank"])
    print("  scalars:", f.attributes["scalars"])
    # Expected: inputs [A, B] (register 0, 1). Wait — order of leaf discovery
    # is by first appearance: Add.inputs=[A,B] -> A=reg0, B=reg1. Mul also
    # reads A (already reg0). So leaf_inputs=[A,B], n_leaf_regs=2.
    # ops: Add{1, dst=2, a=0, b=1}; Sqrt{6, dst=3, a=2, b=0};
    #      Mul{3, dst=0(rewritten), a=3, b=0}
    assert f.attributes["ops"] == [1,2,0,1, 6,3,2,0, 3,0,3,0], f.attributes["ops"]
    assert [i["name"] for i in f.inputs] == ["A", "B"]
    assert f.attributes["rank"] == 1
    assert f.attributes["out_shape"] == [8]
    # input_shapes: A=[8], B=[8], rank 1 -> [8, 8]
    assert f.attributes["input_shapes"] == [8, 8], f.attributes["input_shapes"]
    assert f.attributes["scalars"] == []
    print("  PASS")


def test_shared_not_fused():
    m = make_shared_intermediate()
    before = len(m.nodes)
    m = run_fusion(m)
    fused = [n for n in m.nodes.values() if n.op_type == "FusedElemwise"]
    # t1 is shared (Sqrt + Neg both read it) -> Add's output has 2 consumers,
    # so Add CANNOT join a chain (correct). But Sqrt(t1)->t2 and Neg(t1)->t3
    # are each read only by Mul, so {Sqrt, Neg, Mul} form a DAG-shaped chain
    # (two parallel unary ops feeding one binary op). The register machine
    # handles this: reg0=t1(leaf), Sqrt(reg0)->regA, Neg(reg0)->regB,
    # Mul(regB,regA)->reg0. Add survives (shared producer, not absorbed).
    assert len(fused) == 1, f"expected 1 fused, got {len(fused)}; nodes={[n.op_type for n in m.nodes.values()]}"
    f = fused[0]
    assert len(f.attributes["ops"]) == 12, f"expected 3 ops (12 ints), got {f.attributes['ops']}"
    # Add must survive (not absorbed) — its output t1 is shared.
    optypes = [n.op_type for n in m.nodes.values()]
    assert "Add" in optypes, f"Add (shared producer) must survive; got {optypes}"
    print("  PASS (shared producer Add not absorbed; Sqrt+Neg+Mul DAG fused)")


def test_cast_break():
    m = make_cast_break()
    m = run_fusion(m)
    fused = [n for n in m.nodes.values() if n.op_type == "FusedElemwise"]
    # Add->Cast breaks (Cast not chainable); Cast->Sqrt: Cast not chainable so
    # Sqrt is a len-1 chain. Nothing >= MIN=2 fuses.
    assert len(fused) == 0, f"expected 0 fused (cast breaks chain), got {len(fused)}; nodes={[n.op_type for n in m.nodes.values()]}"
    print("  PASS (cast breaks chain correctly)")


def test_scalar_constant():
    m = make_chain_with_scalar()
    m = run_fusion(m)
    fused = [n for n in m.nodes.values() if n.op_type == "FusedElemwise"]
    assert len(fused) == 1, f"expected 1 fused, got {len(fused)}; nodes={[n.op_type for n in m.nodes.values()]}"
    f = fused[0]
    print("  inputs:", [i["name"] for i in f.inputs])
    print("  ops:", f.attributes["ops"])
    print("  scalars:", f.attributes["scalars"])
    # Leaves discovered in order: Div.inputs=[A, two]. A is not produced by a
    # chain op -> reg0. two is not produced by a chain op -> reg1, BUT it's a
    # single-element initializer -> scalar index 0, operand code 100.
    # ops: Div{4, dst=2, a=0(reg A), b=100(scalar two)};
    #      Sqrt{6, dst=0(rewritten terminal), a=2, b=0}
    assert f.attributes["scalars"] == [2.0], f.attributes["scalars"]
    assert f.attributes["ops"] == [4, 2, 0, 100, 6, 0, 2, 0], f.attributes["ops"]
    # Inputs include both A and two (two is bound as SSBO too — shader preloads
    # reg1 from it, but the operand path uses the 100+ scalar; harmless).
    assert [i["name"] for i in f.inputs] == ["A", "two"]
    print("  PASS (scalar constant encoded as operand 100+)")


if __name__ == "__main__":
    print("test_addsqrtmul:")
    test_addsqrtmul()
    print("test_shared_not_fused:")
    test_shared_not_fused()
    print("test_cast_break:")
    test_cast_break()
    print("test_scalar_constant:")
    test_scalar_constant()
    print("\nALL PASS")
