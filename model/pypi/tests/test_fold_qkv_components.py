"""Standalone tests for fuse_qkv_components.

The vision tower projects q/k/v in one MatMul, Reshapes the result to
[N, C, H, D] and then Transposes it to [C, N, H, D] for exactly one reason:
the following Split(axis=0) then yields C *contiguous* slices — zero-copy
range views — and each is Squeezed back to [N, H, D] for RE(Q), RE(K) and
Attention(V):

    Reshape [N,C,H,D] -> Transpose([1,0,2,3]) -> Split(axis=0)
      -> C x Squeeze [N,H,D]
      -> RotaryEmbedding(Q), RotaryEmbedding(K), Attention(V)

The Transpose is a real kernel (9.4 ms over the tower's 24 blocks) and the
Split/Squeezes are views, so dropping all five is worth 120 dispatches. It
cannot be folded into the Reshape (a metadata-only view vs a real permutation)
nor into the MatMul (a row-major [M,N] output has the token axis outermost;
[C,N,H,D] needs it in the middle). So the three consumers read their component
out of the pre-Transpose tensor instead: RE gains `input_component`,
Attention gains `v_component`.

The tests below check both halves of that: the fold rewires the *right*
component to the *right* role (taken from the Split's output position, not
from the order the consumers are discovered), and it declines whenever an
operand is not bound to exactly one consumer — a partial fold would leave a
node naming a tensor that no longer exists, or pair a component with the wrong
operand. The runtime half of the contract (a component read must address the
same bytes the Split+Squeeze handed over) is
BufferRankTest.RotaryEmbeddingComponent / AttentionVComponent.
"""
import numpy as np
from onnx import numpy_helper

from onnx2vkop.dag import DAGBasedModel, Node
from onnx2vkop.optimizer import FusionOptimizer

F16, I64 = 10, 7
C, H, D, N = 3, 16, 64, 49


def tdict(name, dims=None, dtype=F16):
    return {"name": name, "shape": list(dims or []), "dtype": dtype}


def count(dag, op_type):
    return sum(1 for n in dag.nodes.values() if n.op_type == op_type)


def build(perm=(1, 0, 2, 3), sizes=None, split_axis=0, roles=(("qk", 0), ("qk", 1), ("v", 2)),
          extra_squeeze_reader=False, attention_has_cache=False,
          re_takes_second_component=False):
    """One vision-tower block's qkv fan-out.

    `roles[i]` says which consumer the Split's i-th output (component i) feeds:
    ("qk", slot) for a RotaryEmbedding or ("v", slot) for the Attention. The
    default is the real export's Q, K, V order. `sizes` is the Split's split
    input; None omits it (ONNX's even split).
    """
    m = DAGBasedModel()
    n_comp = len(roles)

    # The [N, C, H, D] qkv tensor, hung off a Reshape so it has a producer.
    m.nodes["RS"] = Node("Reshape", "RS", {},
                         [tdict("add_out", [N, n_comp * H * D], F16)],
                         [tdict("qkv", [N, n_comp, H, D], F16)])
    m.nodes["TR"] = Node("Transpose", "TR", {"perm": list(perm)},
                         [tdict("qkv", [N, n_comp, H, D], F16)],
                         [tdict("tr", [n_comp, N, H, D], F16)])
    split_inputs = [tdict("tr", [n_comp, N, H, D], F16)]
    if sizes is not None:
        split_inputs.append(tdict("split_sizes", [n_comp], I64))
        m.initializers["split_sizes"] = numpy_helper.from_array(
            np.array(sizes, dtype=np.int64), "split_sizes")
    m.nodes["SP"] = Node("Split", "SP", {"axis": split_axis}, split_inputs,
                         [tdict(f"sl{i}", [1, N, H, D], F16)
                          for i in range(n_comp)])
    # One Squeeze per slice, sharing a single axes initializer (as the export
    # does).
    m.initializers["sq_ax"] = numpy_helper.from_array(
        np.array([0], dtype=np.int64), "sq_ax")
    for i in range(n_comp):
        m.nodes[f"SQ{i}"] = Node("Squeeze", f"SQ{i}", {},
                                 [tdict(f"sl{i}", [1, N, H, D], F16),
                                  tdict("sq_ax", [1], I64)],
                                 [tdict(f"sq{i}", [N, H, D], F16)])

    # The Consumer of each component, by index, and the tensor name it reads.
    att_inputs = [None, None, None]
    v_idx = None
    for i, (kind, slot) in enumerate(roles):
        if kind != "qk":
            v_idx = i
            continue
        m.nodes[f"RE{i}"] = Node("RotaryEmbedding", f"RE{i}", {},
                                 [tdict(f"sq{i}", [N, H, D], F16),
                                  tdict(f"cos{i}", [N, D], 1),
                                  tdict(f"sin{i}", [N, D], 1)],
                                 [tdict(f"re{i}", [N, H, D], F16)])
        att_inputs[slot] = tdict(f"re{i}", [N, H, D], F16)
    assert v_idx is not None
    att_inputs[[s for k, s in roles if k == "v"][0]] = tdict(
        f"sq{v_idx}", [N, H, D], F16)

    att_attrs = {"num_heads": H, "num_kv_heads": H, "causal": 0, "scale": 0.125,
                 "seq_major": 1}
    if attention_has_cache:
        att_attrs["has_cache"] = 1
    # input order is Q, K, V; att_inputs[] was filled by slot, and V's slot is
    # 2 in every real graph.
    att_inputs = [a for a in att_inputs if a is not None]
    m.nodes["ATT"] = Node("Attention", "ATT", att_attrs, att_inputs,
                          [tdict("y", [1, N, H, D], F16)])

    if extra_squeeze_reader:
        m.nodes["SINK"] = Node("Relu", "SINK", {},
                               [tdict(f"sq{v_idx}", [N, H, D], F16)],
                               [tdict("sink", [N, H, D], F16)])
    if re_takes_second_component:
        # One RE reading two components: component 0 as X and component 1 as a
        # second input. The op has a single read point, so nothing can fold it.
        first = [i for i, (k, _) in enumerate(roles) if k == "qk"][0]
        other = [i for i in range(n_comp) if i != first][0]
        m.nodes[f"RE{first}"].inputs.append(tdict(f"sq{other}", [N, H, D], F16))
    return m


def matches_of(dag):
    return FusionOptimizer.match_qkv_components(dag)


def roles_of(dag):
    """{component index: (op_type, attribute)} for the folded graph."""
    out = {}
    for n in dag.nodes.values():
        if n.op_type == "RotaryEmbedding" and "input_component" in n.attributes:
            out[n.attributes["input_component"]] = ("RotaryEmbedding", n.name)
        if n.op_type == "Attention" and "v_component" in n.attributes:
            out[n.attributes["v_component"]] = ("Attention", n.name)
    return out


# ---------------------------------------------------------------- positives

def test_vision_block_folds():
    """The real export shape: one Transpose, one Split, three Squeezes, and
    the three roles get component 0 / 1 / 2 of the pre-Transpose tensor."""
    m = build()
    ms = matches_of(m)
    assert len(ms) == 1, ms
    assert FusionOptimizer.fold_qkv_components(m, ms[0])

    att = m.nodes["ATT"]
    assert att.inputs[0]["name"] == "re0"        # Q: RE0's output, untouched
    assert att.inputs[1]["name"] == "re1"        # K
    assert att.inputs[2]["name"] == "qkv"        # V: the pre-Transpose tensor
    assert att.inputs[2]["shape"] == [N, C, H, D]
    assert att.attributes["v_component"] == 2
    assert m.nodes["RE0"].attributes["input_component"] == 0
    assert m.nodes["RE1"].attributes["input_component"] == 1
    # Each RE's X is the pre-Transpose tensor, with its shape metadata.
    for i in (0, 1):
        assert m.nodes[f"RE{i}"].inputs[0]["name"] == "qkv"
    # Y keeps its tensor, so downstream consumers still resolve.
    assert [o["name"] for o in att.outputs] == ["y"]

    assert "TR" not in m.nodes
    assert "SP" not in m.nodes
    assert count(m, "Squeeze") == 0
    assert count(m, "Transpose") == 0
    assert count(m, "Split") == 0


def test_component_follows_the_split_position():
    """The component comes from the Split's output position, not from the order
    the consumers are discovered: a graph whose V slice is output 0 gets
    v_component=0 while Q/K take 1 and 2."""
    roles = (("v", 2), ("qk", 1), ("qk", 0))
    m = build(roles=roles)
    ms = matches_of(m)
    assert len(ms) == 1, ms
    assert ms[0]["v"]["comp"] == 0
    assert sorted(e["comp"] for e in ms[0]["qk"]) == [1, 2]
    assert FusionOptimizer.fold_qkv_components(m, ms[0])

    got = roles_of(m)
    assert got[0] == ("Attention", "ATT")
    # RE1 read component 1 and fed the Attention's slot 1 (K); RE2 read
    # component 2 and fed slot 0 (Q).
    assert got[1] == ("RotaryEmbedding", "RE1")
    assert got[2] == ("RotaryEmbedding", "RE2")
    assert m.nodes["ATT"].inputs[1]["name"] == "re1"
    assert m.nodes["ATT"].inputs[0]["name"] == "re2"


def test_missing_split_sizes_folds():
    """No split input is ONNX's default even split, which is exactly the
    [1]*C this fold needs."""
    m = build(sizes=None)
    ms = matches_of(m)
    assert len(ms) == 1, ms
    assert FusionOptimizer.fold_qkv_components(m, ms[0])


def test_explicit_even_split_sizes_fold():
    m = build(sizes=[1, 1, 1])
    ms = matches_of(m)
    assert len(ms) == 1, ms
    assert FusionOptimizer.fold_qkv_components(m, ms[0])


# ---------------------------------------------------------------- negatives

def test_uneven_split_declined():
    """A Split that gave its outputs unequal slices is not the component split:
    the fold hands each consumer an *index*, which only describes equal ones."""
    assert matches_of(build(sizes=[2, 1, 0])) == []


def test_wrong_perm_declined():
    """A different permutation is not the [N,C,H,D] -> [C,N,H,D] swap whose
    addressing the component read has to invert."""
    assert matches_of(build(perm=(0, 1, 2, 3))) == []


def test_split_on_another_axis_declined():
    """Only an axis-0 split makes each slice a component of the leading axis;
    on any other axis the slices are strided, not components."""
    assert matches_of(build(split_axis=1)) == []


def test_shared_squeeze_output_declined():
    """A second consumer of a Squeeze's output: the tensor the fold would delete
    is still read by an op it does not rewire."""
    assert matches_of(build(extra_squeeze_reader=True)) == []


def test_re_taking_a_second_component_declined():
    """A RotaryEmbedding that also reads another component leaves that
    component's slice with two consumers, and the fold requires exactly one per
    slice: it deletes the Squeeze whose output one of them still names, and it
    cannot rewire an op that has two different read points anyway."""
    assert matches_of(build(re_takes_second_component=True)) == []


def test_attention_with_cache_declined():
    """The cache form reads its fresh rows at the native [H,S,D] pitch in the
    copy half of the dispatch, which the component addressing would mis-state;
    ops/Attention.hpp rejects the combination, so the fold must not create it."""
    assert matches_of(build(attention_has_cache=True)) == []


def test_bare_transpose_declined():
    """A Transpose with no Split/Squeeze fan-out behind it is not this pattern
    at all (and is someone else's business — e.g. fold_transpose_into_matmul)."""
    m = build()
    for name in ("SP", "SQ0", "SQ1", "SQ2", "RE0", "RE1", "ATT"):
        del m.nodes[name]
    assert matches_of(m) == []
