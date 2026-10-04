"""Standalone test for the generic Unsqueeze-fold pass (fuse_unsqueeze_eliminate).

The pass deletes a single-axis Unsqueeze and rebinds its consumers to the
pre-view tensor, but it cannot fix up the *live* rank the producer reports. A
rank-increasing fold therefore leaves consumers reading a lower rank than the
tensor actually has -- and ops like ScatterND take index_rank from
indices.shape[-1], so the wrong rank becomes a data-dependent out-of-bounds
write into whatever memory neighbours the output buffer. That is how Qwen3-VL's
rope cos/sin ended up scribbled over the fp16 weight blob (decode all-NaN).

The pass is now fail-closed: only op types listed in
FusionOptimizer._UNSQUEEZE_RANK_TRANSPARENT fold unconditionally, Concat folds
only when the inserted axis is strictly later than the concat axis, everything
else keeps its node. These cases pin that decision table -- they call the
matcher directly (the rest of the optimizer would also fold views through the
separate, already-sound alias pass and blur the result).
"""
import sys

sys.path.insert(0, "model/pypi")
from onnx2vkop.dag import DAGBasedModel, Node
from onnx2vkop.optimizer import FusionOptimizer


def tdict(name, dims=None):
    d = {"name": name}
    if dims is not None:
        d["shape"] = list(dims)
    return d


def unsqueeze_case(in_dims, axes, consumer_type, consumer_attrs=None):
    """src -> Unsqueeze(axes) -> single consumer -> out.

    in_dims=None records no shape at all (the "cannot decide" case).
    """
    m = DAGBasedModel()
    src = tdict("src", in_dims)
    view = tdict("view", None if in_dims is None else in_dims + [1])
    out = tdict("out", [1])
    u = Node("Unsqueeze", "Unsqueeze_0", {"axes": list(axes)}, [src], [view])
    c = Node(consumer_type, "Consumer_0", consumer_attrs or {}, [view], [out])
    m.nodes[u.name] = u
    m.nodes[c.name] = c
    m.inputs = [src]
    m.outputs = [out]
    return m


def other_leaf(dims):
    return tdict("other", dims)


def count_unsqueeze(m):
    return sum(1 for n in m.nodes.values() if n.op_type == "Unsqueeze")


def matches_of(m):
    return FusionOptimizer.match_unsqueeze_eliminate(m)


def test_rank_transparent_unary_folds():
    """src[8] -> Unsqueeze(axes=[1]) -> Cast: elementwise unary ignores rank."""
    m = unsqueeze_case([8], [1], "Cast", {"to": 1})
    ms = matches_of(m)
    assert len(ms) == 1, f"Cast must fold (rank-transparent), got {len(ms)}"
    FusionOptimizer.fold_unsqueeze_eliminate(m, ms[0])
    assert count_unsqueeze(m) == 0, "Unsqueeze should be deleted"
    c = m.nodes["Consumer_0"]
    assert [i["name"] for i in c.inputs] == ["src"], \
        f"consumer must read the pre-view tensor, got {[i['name'] for i in c.inputs]}"
    print("  PASS (rank-transparent Cast folds)")


def test_concat_same_axis_rejected():
    """The Qwen3-VL kv-split signature: view inserted at axis 1, Concat on
    axis 1. The axis would point at a different dim after the fold."""
    m = DAGBasedModel()
    s = tdict("src", [-1, -1, -1])
    view = tdict("view", [-1, 1, -1, -1])
    other = other_leaf([-1, 1, -1, -1])
    out = tdict("out", [-1, 2, -1, -1])
    u = Node("Unsqueeze", "Unsqueeze_0", {"axes": [1]}, [s], [view])
    c = Node("Concat", "Concat_0", {"axis": 1}, [view, other], [out])
    m.nodes[u.name] = u
    m.nodes[c.name] = c
    m.inputs = [s, other]
    m.outputs = [out]
    ms = matches_of(m)
    assert len(ms) == 0, \
        f"Concat axis==inserted axis must NOT fold (u_ax=1, a=1), got {len(ms)}"
    print("  PASS (Concat with axis == inserted axis rejected)")


def test_concat_rotary_signature_rejected():
    """/rotary_emb: Unsqueeze(axes=[-1]) on a rank-3 tensor feeding
    Concat(axis=-1). Both normalize to axis 3 of the rank-4 view, so the fold
    would collapse [1,1,20,3] to [1,1,21] and ScatterND would then take
    index_rank=21 for a rank-3 data tensor -- the exact NaN incident."""
    m = DAGBasedModel()
    s = tdict("cos", [-1, -1, -1])
    view = tdict("cos_u", [-1, -1, -1, 1])
    other = other_leaf([-1, -1, -1, 1])
    out = tdict("cat", [-1, -1, -1, 2])
    u = Node("Unsqueeze", "Unsqueeze_0", {"axes": [-1]}, [s], [view])
    c = Node("Concat", "Concat_0", {"axis": -1}, [view, other], [out])
    m.nodes[u.name] = u
    m.nodes[c.name] = c
    m.inputs = [s, other]
    m.outputs = [out]
    ms = matches_of(m)
    assert len(ms) == 0, \
        f"rotary Unsqueeze(-1)->Concat(-1) must NOT fold, got {len(ms)}"
    print("  PASS (rotary Unsqueeze(-1)->Concat(axis=-1) rejected)")


def test_concat_axis_before_inserted_folds():
    """Safe case: src[8,16] -> Unsqueeze(axes=[2]) (view rank 3) feeding
    Concat(axis=1). The concat axis addresses the same dim before and after."""
    m = DAGBasedModel()
    s = tdict("src", [8, 16])
    view = tdict("view", [8, 16, 1])
    other = other_leaf([8, 4, 1])
    out = tdict("cat", [8, 20, 1])
    u = Node("Unsqueeze", "Unsqueeze_0", {"axes": [2]}, [s], [view])
    c = Node("Concat", "Concat_0", {"axis": 1}, [view, other], [out])
    m.nodes[u.name] = u
    m.nodes[c.name] = c
    m.inputs = [s, other]
    m.outputs = [out]
    ms = matches_of(m)
    assert len(ms) == 1, f"Concat axis(1) < u_ax(2) should fold, got {len(ms)}"
    FusionOptimizer.fold_unsqueeze_eliminate(m, ms[0])
    assert count_unsqueeze(m) == 0, "Unsqueeze should be deleted"
    print("  PASS (Concat axis strictly before the inserted axis folds)")


def test_concat_without_recorded_shape_rejected():
    """No shape on the view's producer edge -> rank unknown -> cannot decide
    the axis order, so fail closed."""
    m = unsqueeze_case(None, [1], "Concat", {"axis": 0})
    ms = matches_of(m)
    assert len(ms) == 0, f"unknown rank must NOT fold, got {len(ms)}"
    print("  PASS (Concat with unknown recorded shape rejected)")


def test_rank_sensitive_ops_rejected():
    """Ops not on the whitelist stay put: Expand broadcasts right-aligned (so
    it reads the rank), Gather takes axis from shape[-1], Reshape/Shape/Slice/
    Transpose/Reduce all read the rank. Runtime's SqueezeUnsqueeze handles them
    as a pure GPU alias."""
    for op_type, attrs in (
        ("Expand", {}),
        ("Gather", {"axis": 0}),
        ("Reshape", {}),
        ("Transpose", {"perm": "[1,0]"}),
        ("ReduceSum", {"axes": "[1]"}),
    ):
        m = unsqueeze_case([8], [1], op_type, attrs)
        ms = matches_of(m)
        assert len(ms) == 0, f"{op_type} is not rank-transparent, got {len(ms)}"
    print("  PASS (Expand/Gather/Reshape/Transpose/ReduceSum all rejected)")


def test_mixed_consumers_rejected():
    """Two readers of one view with different op types -> no uniform fold
    logic -> rejected before the whitelist is even consulted."""
    m = DAGBasedModel()
    s = tdict("src", [8])
    view = tdict("view", [8, 1])
    cast_out = tdict("cast_out", [8, 1])
    neg_out = tdict("neg_out", [8, 1])
    u = Node("Unsqueeze", "Unsqueeze_0", {"axes": [1]}, [s], [view])
    cast = Node("Cast", "Cast_0", {"to": 1}, [view], [cast_out])
    neg = Node("Neg", "Neg_0", {}, [view], [neg_out])
    for n in (u, cast, neg):
        m.nodes[n.name] = n
    m.inputs = [s]
    m.outputs = [cast_out, neg_out]
    ms = matches_of(m)
    assert len(ms) == 0, f"mixed consumer op types must NOT fold, got {len(ms)}"
    print("  PASS (mixed-consumer view rejected)")


if __name__ == "__main__":
    for fn in (test_rank_transparent_unary_folds,
               test_concat_same_axis_rejected,
               test_concat_rotary_signature_rejected,
               test_concat_axis_before_inserted_folds,
               test_concat_without_recorded_shape_rejected,
               test_rank_sensitive_ops_rejected,
               test_mixed_consumers_rejected):
        print(fn.__name__ + ":")
        fn()
    print("\nALL PASS")
