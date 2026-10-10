"""Standalone test for the redundant-Shape CSE pass (eliminate_redundant_shape).

Shape(x) is a pure function of x's *runtime* shape, so two Shape nodes reading
the same upstream tensor always produce bit-identical int64 vectors -- the
later one can be dropped and its consumers rebound onto the first. That is the
only merge the pass performs: the key is producer identity, which the graph
proves by itself.

The tempting stronger key -- "these two tensors have the same declared shape,
so Shape(x)==Shape(y)" -- is deliberately NOT used. ONNX shape inference runs
earlier in optimize_model and rewrites the exporter's 4 dim_params into ~991
per-dim junk symbols (unk__N), and the DAG drops symbol names entirely
(dynamic dims collapse to a -1 sentinel). Under that key `seq` and `kv_len`
are indistinguishable, so distinct runtime dims would be silently merged.

These cases pin the pass's decision table by calling the matcher directly, and
one end-to-end case checks the fold actually rebinds + deletes.
"""
import sys

sys.path.insert(0, "model/pypi")
from onnx2vkop.dag import DAGBasedModel, Node
from onnx2vkop.optimizer import FusionOptimizer


def tdict(name, dims=None, dtype=7):
    return {"name": name, "shape": list(dims or []), "dtype": dtype}


def shape_of(src, out_name):
    return Node("Shape", out_name, {}, [src], [tdict(out_name)])


def build(*shapes):
    """src -> Shape nodes feeding one Concat each, so nothing is dead."""
    m = DAGBasedModel()
    src = tdict("src", [2, 896])
    m.inputs = [src]
    consumers = []
    for i, s in enumerate(shapes):
        sn = shape_of(src, "shape_%d" % i)
        m.nodes[sn.name] = sn
        acc = tdict("cat_%d" % i, [1], dtype=7)
        cn = Node("Concat", "Concat_%d" % i, {"axis": 0},
                  [tdict(sn.outputs[0]["name"], None), tdict("one", [1])], [acc])
        m.nodes[cn.name] = cn
        consumers.append(cn)
    m.outputs = [consumers[-1].outputs[0]]
    return m


def count_shape(m):
    return sum(1 for n in m.nodes.values() if n.op_type == "Shape")


def test_same_source_merges():
    m = build(1, 1, 1)
    matches = FusionOptimizer.match_redundant_shape(m)
    assert len(matches) == 1, matches
    assert matches[0]["canonical"].name == "shape_0"
    assert [d.name for d in matches[0]["duplicates"]] == ["shape_1", "shape_2"]
    assert FusionOptimizer.fold_redundant_shape(m, matches[0]) is True
    assert count_shape(m) == 1, count_shape(m)
    # every remaining consumer reads the canonical vector
    for n in m.nodes.values():
        if n.op_type == "Concat":
            assert n.inputs[0]["name"] == "shape_0", n.inputs[0]
    return True


def test_different_source_does_not_merge():
    """Two Shape nodes over *different* tensors must never merge, even when
    their declared shapes are identical -- that is the unk__N trap."""
    m = DAGBasedModel()
    m.inputs = [tdict("a", [2, 896]), tdict("b", [2, 896])]
    for nm in ("a", "b"):
        sn = shape_of(tdict(nm, [2, 896]), "shape_%s" % nm)
        m.nodes[sn.name] = sn
    assert FusionOptimizer.match_redundant_shape(m) == []
    return True


def test_graph_output_shape_not_deleted():
    m = build(1, 1)
    m.outputs = [tdict("shape_1")]
    matches = FusionOptimizer.match_redundant_shape(m)
    # shape_1 is a graph output -> the group may still exist, but shape_1
    # must not appear as a duplicate.
    for mt in matches:
        assert all(d.name != "shape_1" for d in mt["duplicates"])
    return True


def test_non_int64_shape_skipped():
    """A Shape node whose recorded dtype is not int64 means the metadata is
    not what we think it is -- fail closed and leave it alone."""
    m = build(1, 1)
    m.nodes["shape_1"].outputs[0]["dtype"] = 1
    matches = FusionOptimizer.match_redundant_shape(m)
    assert all(d.name != "shape_1" for mt in matches for d in mt["duplicates"])
    return True


def test_fan_out_all_consumers_rebound():
    m = DAGBasedModel()
    src = tdict("src", [2, 896])
    m.inputs = [src]
    s0 = shape_of(src, "shape_0")
    s1 = shape_of(src, "shape_1")
    m.nodes[s0.name] = s0
    m.nodes[s1.name] = s1
    consumers = []
    for i in range(3):
        c = Node("Gather", "G_%d" % i, {"axis": 0},
                 [tdict("shape_1", None), tdict("idx", [1], dtype=7)], [tdict("g_%d" % i, [1])])
        m.nodes[c.name] = c
        consumers.append(c)
    m.outputs = [consumers[-1].outputs[0]]
    matches = FusionOptimizer.match_redundant_shape(m)
    assert len(matches) == 1
    FusionOptimizer.fold_redundant_shape(m, matches[0])
    assert count_shape(m) == 1
    for c in consumers:
        assert c.inputs[0]["name"] == "shape_0", c.inputs[0]
    return True


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print("PASS %s" % t.__name__)
        except AssertionError as e:
            failed += 1
            print("FAIL %s: %s" % (t.__name__, e))
    sys.exit(1 if failed else 0)
