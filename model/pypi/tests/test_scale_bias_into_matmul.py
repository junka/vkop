"""Standalone test for fuse_scale_bias_into_matmul.

A FusedElemwise whose register program is exactly
  op0: MUL(reg[MatMul_out], scalar)   -> dst0
  op1: ADD(dst0, reg[bias_tensor])    -> reg0 (output)
is the attention QK^T * scale + mask chain. The pass folds the scalar into a
MatMul `epi_scale` attribute and the bias into an appended input with
`epi_bias=1`, then deletes the FusedElemwise. Non-matching programs (different
op order, non-scalar operand, non-MatMul producer, quantized MatMul) are left
alone.
"""
from onnx2vkop.dag import DAGBasedModel, Node
from onnx2vkop.optimizer import FusionOptimizer


def tdict(name, dims=None, dtype=10):
    return {"name": name, "shape": list(dims or []), "dtype": dtype}


def fe_node(name, ops, scalars, inputs, out_name="fe_out"):
    return Node("FusedElemwise", name,
                {"ops": list(ops), "scalars": list(scalars)},
                inputs, [tdict(out_name, [1, 16, 4, 8])])


def matmul_node(name, inputs, out_name="mm_out"):
    return Node("MatMul", name, {"transB": 1}, inputs, [tdict(out_name, [1, 16, 4, 8])])


def count(dag, op_type):
    return sum(1 for n in dag.nodes.values() if n.op_type == op_type)


def matches_of(dag):
    return FusionOptimizer.match_scale_bias_into_matmul(dag)


# op codes: MUL=3, ADD=1. Operand encoding: <100 = register, >=100 = scalar.
# leaf inputs: reg0 = MatMul out, reg1 = scalar const (becomes scalar[0]),
# reg2 = bias tensor.
def canonical_fe(mm_out, bias_name, scale, fe_out="fe_out"):
    ops = [
        3, 3, 0, 100,        # MUL  dst=3  a=reg0(mm)  b=scalar[0]
        1, 0, 3, 2,          # ADD  dst=0  a=reg3      b=reg2(bias)
    ]
    return fe_node("FE", ops, [scale],
                   [tdict(mm_out), tdict("scalar_const", [], 10),
                    tdict(bias_name, [1, 1, 4, 8], 10)],
                   out_name=fe_out)


def test_canonical_scale_bias_folds_into_matmul():
    m = DAGBasedModel()
    mm = matmul_node("MM", [tdict("q", [1, 16, 4, 128]), tdict("k", [1, 8, 4, 128])],
                     out_name="mm_out")
    fe = canonical_fe("mm_out", "attn_bias", 0.088379)
    m.nodes[mm.name] = mm
    m.nodes[fe.name] = fe
    m.outputs = [fe.outputs[0]]

    ms = matches_of(m)
    assert len(ms) == 1, ms
    FusionOptimizer.fold_scale_bias_into_matmul(m, ms[0])

    assert count(m, "FusedElemwise") == 0
    assert count(m, "MatMul") == 1
    mm2 = m.nodes["MM"]
    assert mm2.attributes["epi_scale"] == 0.088379
    assert mm2.attributes["epi_bias"] == 1
    # bias appended as inputs[2]
    assert len(mm2.inputs) == 3
    assert mm2.inputs[2]["name"] == "attn_bias"
    # MatMul now produces the FE's output (so Softmax downstream still resolves)
    assert mm2.outputs[0]["name"] == "fe_out"
    print("  PASS (canonical MUL+ADD -> MatMul epi_scale/epi_bias)")


def test_mul_then_mul_does_not_match():
    """op1 must be ADD, not another MUL."""
    m = DAGBasedModel()
    mm = matmul_node("MM", [tdict("q"), tdict("k")], out_name="mm_out")
    ops = [3, 3, 0, 100,   # MUL dst=3 a=reg0 b=scalar[0]
           3, 0, 3, 2]     # MUL (not ADD) dst=0 a=reg3 b=reg2
    fe = fe_node("FE", ops, [0.5],
                 [tdict("mm_out"), tdict("s", [], 10), tdict("b", [1, 1, 4, 8], 10)])
    m.nodes[mm.name] = mm
    m.nodes[fe.name] = fe
    assert matches_of(m) == []
    print("  PASS (MUL+MUL rejected)")


def test_non_scalar_first_operand_rejected():
    """op0's b must be a scalar (>=100); a MUL of two tensors is not the epilogue."""
    m = DAGBasedModel()
    mm = matmul_node("MM", [tdict("q"), tdict("k")], out_name="mm_out")
    # b0 = 2 (a register), not a scalar
    ops = [3, 3, 0, 2,
           1, 0, 3, 100]
    fe = fe_node("FE", ops, [0.5],
                 [tdict("mm_out"), tdict("w", [1, 1, 4, 8], 10), tdict("b", [1, 1, 4, 8], 10)])
    m.nodes[mm.name] = mm
    m.nodes[fe.name] = fe
    assert matches_of(m) == []
    print("  PASS (non-scalar scale rejected)")


def test_three_op_chain_rejected():
    """Only the exact 2-op MUL+ADD matches; a 3-op chain is left alone."""
    m = DAGBasedModel()
    mm = matmul_node("MM", [tdict("q"), tdict("k")], out_name="mm_out")
    ops = [3, 3, 0, 100,   # MUL
           1, 4, 3, 2,     # ADD
           3, 0, 4, 100]   # another MUL
    fe = fe_node("FE", ops, [0.5, 0.25],
                 [tdict("mm_out"), tdict("s1", [], 10), tdict("b", [1, 1, 4, 8], 10),
                  tdict("s2", [], 10)])
    m.nodes[mm.name] = mm
    m.nodes[fe.name] = fe
    assert matches_of(m) == []
    print("  PASS (3-op chain rejected)")


def test_quantized_matmul_rejected():
    """A 3-input (quantized) MatMul must not get a 4th epilogue input."""
    m = DAGBasedModel()
    mm = matmul_node("MM", [tdict("q"), tdict("b_int8"), tdict("scale", [8])],
                     out_name="mm_out")
    fe = canonical_fe("mm_out", "attn_bias", 0.5)
    m.nodes[mm.name] = mm
    m.nodes[fe.name] = fe
    assert matches_of(m) == []
    print("  PASS (quantized 3-input MatMul rejected)")


def test_softmax_consumer_folds_into_attn_softmax():
    """When the FE's single consumer is a Softmax, Plan B folds the Softmax
    into the MatMul: attn_softmax=1, the Softmax node is deleted, and the
    MatMul produces the Softmax's output tensor name (so the ·V MatMul that
    read the Softmax now reads the fused MatMul directly)."""
    m = DAGBasedModel()
    mm = matmul_node("MM", [tdict("q", [1, 16, 4, 128]), tdict("k", [1, 8, 4, 128])],
                     out_name="mm_out")
    fe = canonical_fe("mm_out", "attn_bias", 0.088379, fe_out="fe_out")
    # Softmax reads fe_out; its output feeds the value MatMul downstream.
    sm = Node("Softmax", "SM", {"axis": -1},
              [tdict("fe_out", [1, 16, 4, 8])],
              [tdict("attn_probs", [1, 16, 4, 8])])
    m.nodes[mm.name] = mm
    m.nodes[fe.name] = fe
    m.nodes[sm.name] = sm
    m.outputs = [tdict("attn_probs", [1, 16, 4, 8])]

    ms = matches_of(m)
    assert len(ms) == 1, ms
    assert ms[0].get("fold_softmax") is True, ms[0]
    FusionOptimizer.fold_scale_bias_into_matmul(m, ms[0])

    assert count(m, "FusedElemwise") == 0
    assert count(m, "Softmax") == 0
    assert count(m, "MatMul") == 1
    mm2 = m.nodes["MM"]
    assert mm2.attributes["epi_scale"] == 0.088379
    assert mm2.attributes["epi_bias"] == 1
    assert mm2.attributes["attn_softmax"] == 1
    assert len(mm2.inputs) == 3
    assert mm2.inputs[2]["name"] == "attn_bias"
    # MatMul now produces the Softmax's output name, not the FE's.
    assert mm2.outputs[0]["name"] == "attn_probs"
    print("  PASS (Softmax consumer -> attn_softmax=1, Softmax deleted)")


def test_non_softmax_consumer_not_folded():
    """If the FE's single consumer is NOT a Softmax (e.g. an Add), Plan B
    must not fire: attn_softmax absent, the consumer node survives, and the
    MatMul produces the FE's output (Plan A behaviour)."""
    m = DAGBasedModel()
    mm = matmul_node("MM", [tdict("q"), tdict("k")], out_name="mm_out")
    fe = canonical_fe("mm_out", "attn_bias", 0.5, fe_out="fe_out")
    add = Node("Add", "ADD", {}, [tdict("fe_out"), tdict("resid", [1, 16, 4, 8])],
               [tdict("add_out", [1, 16, 4, 8])])
    m.nodes[mm.name] = mm
    m.nodes[fe.name] = fe
    m.nodes[add.name] = add
    m.outputs = [tdict("add_out", [1, 16, 4, 8])]

    ms = matches_of(m)
    assert len(ms) == 1, ms
    assert ms[0].get("fold_softmax") is False, ms[0]
    FusionOptimizer.fold_scale_bias_into_matmul(m, ms[0])
    mm2 = m.nodes["MM"]
    assert "attn_softmax" not in mm2.attributes
    assert count(m, "Add") == 1   # consumer survives
    assert mm2.outputs[0]["name"] == "fe_out"
    print("  PASS (non-Softmax consumer -> Plan A only, attn_softmax absent)")


def test_dynamic_n_softmax_not_folded():
    """Safety gate: the softmax-fusion kernel reduces one row in a single
    workgroup, so it can only fire when N is STATICALLY known and <= 16384.
    Attention QK^T's N is the dynamic kv_len (a -1 / symbolic dim at convert
    time). Folding the Softmax there would delete the runtime fallback when
    the kernel's N gate trips, so the converter must leave the Softmax in
    place and NOT set attn_softmax. The MatMul still gets the scale+bias
    epilogue (Plan A); only Plan B is suppressed."""
    m = DAGBasedModel()
    # FE output shape with a dynamic last dim (-1 = kv_len).
    fe = Node("FusedElemwise", "FE",
              {"ops": [3, 3, 0, 100, 1, 0, 3, 2], "scalars": [0.088379]},
              [tdict("mm_out", [-1, -1, -1, -1]),
               tdict("scalar_const", [], 10),
               tdict("attn_bias", [1, 1, 4, -1], 10)],
              [tdict("fe_out", [1, 16, 4, -1])])
    mm = matmul_node("MM", [tdict("q", [1, 16, 4, 128]),
                            tdict("k", [1, 8, 4, 128])], out_name="mm_out")
    sm = Node("Softmax", "SM", {"axis": -1},
              [tdict("fe_out", [1, 16, 4, -1])],
              [tdict("attn_probs", [1, 16, 4, -1])])
    m.nodes[mm.name] = mm
    m.nodes[fe.name] = fe
    m.nodes[sm.name] = sm
    m.outputs = [tdict("attn_probs", [1, 16, 4, -1])]

    ms = matches_of(m)
    assert len(ms) == 1, ms
    assert ms[0].get("fold_softmax") is False, ms[0]
    FusionOptimizer.fold_scale_bias_into_matmul(m, ms[0])
    mm2 = m.nodes["MM"]
    # Plan A still applied...
    assert mm2.attributes["epi_scale"] == 0.088379
    assert mm2.attributes["epi_bias"] == 1
    # ...but Plan B is NOT (dynamic N), so the Softmax survives.
    assert "attn_softmax" not in mm2.attributes
    assert count(m, "Softmax") == 1
    print("  PASS (dynamic-N Softmax consumer -> Plan A only, Softmax kept)")


if __name__ == "__main__":
    test_canonical_scale_bias_folds_into_matmul()
    test_mul_then_mul_does_not_match()
    test_non_scalar_first_operand_rejected()
    test_three_op_chain_rejected()
    test_quantized_matmul_rejected()
    test_softmax_consumer_folds_into_attn_softmax()
    test_non_softmax_consumer_not_folded()
    test_dynamic_n_softmax_not_folded()
    print("\nAll fuse_scale_bias_into_matmul tests passed.")


