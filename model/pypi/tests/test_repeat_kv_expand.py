"""Standalone test for the repeat_kv Expand elimination pass
(eliminate_repeat_kv_expand).

HF GQA exports each layer's k/v as R = Reshape(Expand(Unsqueeze(X))): X is the
KV cache slice (b, hkv, S, D) and R is the head-repeated (b, hkv*rep, S, D)
matrix that feeds the attention MatMuls' B side. The pass rewires the MatMul's
B side to X itself and deletes the chain; the head replication moves into the
runtime (host grouped-batch detection + shader bmap). These cases pin the
matcher's static proofs — the rep derivation from R/A head counts (the real
graph's E[2] is dynamic and proves nothing), and the rejections that would
change semantics instead of removing a pure copy.
"""
from onnx2vkop.dag import DAGBasedModel, Node
from onnx2vkop.optimizer import FusionOptimizer

F32 = 1  # onnx.TensorProto.FLOAT


def tdict(name, dims=None):
    d = {"name": name, "dtype": F32}
    if dims is not None:
        d["shape"] = list(dims)
    return d


def repeat_kv_dag(hkv=8, e2=-1, rep_r=16, rep_a=16, seq=-1, dim=128):
    """X (b,hkv,S,D) -> U (b,hkv,1,S,D) -> E (b,hkv,e2,S,D)
    -> R (b,rep_r,S,D) -> MatMul(A (b,rep_a,S,D), R)."""
    m = DAGBasedModel()
    x = tdict("x", [2, hkv, seq, dim])
    u = tdict("u_out", [2, hkv, 1, seq, dim])
    e = tdict("e_out", [2, hkv, e2, seq, dim])
    r = tdict("r_out", [2, rep_r, seq, dim])
    a = tdict("a", [2, rep_a, seq, dim])
    y = tdict("y", [2, rep_a, seq, dim])
    un = Node("Unsqueeze", "Unsqueeze_0", {"axes": [2]}, [x], [u])
    ex = Node("Expand", "Expand_0", {}, [u], [e])
    rs = Node("Reshape", "Reshape_0", {}, [e], [r])
    mm = Node("MatMul", "MatMul_0", {}, [a, r], [y])
    for n in (un, ex, rs, mm):
        m.nodes[n.name] = n
    m.inputs = [a, x]
    m.outputs = [y]
    return m, mm, [un, ex, rs]


def matches_of(m):
    return FusionOptimizer.match_repeat_kv_expand(m)


def test_folds_and_rewires_to_x():
    """The real-graph shape: E[2] is dynamic, rep comes from R[1]//hkv and
    A[1]//hkv (both 16//8 = 2). Fold rewires B to X and orphans the chain."""
    m, mm, chain = repeat_kv_dag()
    ms = matches_of(m)
    assert len(ms) == 1, f"repeat_kv chain must match, got {len(ms)}"
    assert FusionOptimizer.fold_repeat_kv_expand(m, ms[0]) is True
    assert mm.inputs[1]["name"] == "x", \
        f"B side must read X, got {mm.inputs[1]['name']}"
    assert mm.inputs[1]["shape"] == [2, 8, -1, 128], \
        f"B side must carry X's shape, got {mm.inputs[1]['shape']}"
    for nd in chain:
        assert nd.name not in m.nodes, f"{nd.name} should be deleted"
    assert m.nodes["MatMul_0"].op_type == "MatMul"
    print("  PASS (fold rewires B to X and deletes the chain)")


def test_rep_from_expand_when_static():
    """A statically annotated E[2] is also accepted and must agree with the
    R/A derivation."""
    m, _, _ = repeat_kv_dag(e2=2)
    ms = matches_of(m)
    assert len(ms) == 1, f"static E[2]=2 must match, got {len(ms)}"
    print("  PASS (static E[2] agrees with the R/A derivation)")


def test_rep_conflict_rejected():
    """A[1]=24 says rep=3, R[1]=16 says rep=2: no rep satisfies both, so the
    shape contract is broken and the chain must stay."""
    m, _, _ = repeat_kv_dag(rep_a=24)
    assert len(matches_of(m)) == 0, "conflicting rep sources must NOT match"
    print("  PASS (conflicting rep derivation rejected)")


def test_rep_one_rejected():
    """A[1] == R[1] == hkv: the Expand copies nothing, so folding would change
    nothing but also prove nothing — rep >= 2 is required."""
    m, _, _ = repeat_kv_dag(rep_r=8, rep_a=8)
    assert len(matches_of(m)) == 0, "rep=1 must NOT match"
    print("  PASS (rep=1 rejected)")


def test_dynamic_hkv_rejected():
    """hkv=-1: the whole rep derivation divides by it — nothing is provable."""
    m, _, _ = repeat_kv_dag(hkv=-1, rep_r=-1, rep_a=-1)
    assert len(matches_of(m)) == 0, "dynamic hkv must NOT match"
    print("  PASS (dynamic hkv rejected)")


def test_shared_chain_consumer_rejected():
    """R feeding a second consumer: rewiring the MatMul would leave the other
    consumer reading a deleted node."""
    m, mm, chain = repeat_kv_dag()
    other_out = tdict("other_out", [2, 16, -1, 128])
    other = Node("Cast", "Cast_0", {"to": 1},
                 [chain[2].outputs[0]], [other_out])  # reads R
    m.nodes[other.name] = other
    m.outputs = [other_out]
    assert len(matches_of(m)) == 0, "shared chain must NOT match"
    print("  PASS (chain with a second consumer rejected)")


def test_chain_output_rejected():
    """R as a graph output: the node is not dead and cannot be deleted."""
    m, _, _ = repeat_kv_dag()
    m.outputs = [m.nodes["Reshape_0"].outputs[0]]
    assert len(matches_of(m)) == 0, "R as graph output must NOT match"
    print("  PASS (R as graph output rejected)")
