"""Tests for the host-shape-expression extraction pass (Phase 4-r2).

The pass rewrites Reshape(target) nodes whose int64 target-shape chain is
Shape/Gather/Concat over constants into a single-input node carrying the
`vkop_host_shape` attribute — a term list the C++ runtime evaluates per round
from live tensor metadata. Everything that does not resolve cleanly must stay
in its old two-input form (fail closed).
"""
from onnx import helper, numpy_helper

from onnx2vkop.dag import DAGBasedModel, Node
from onnx2vkop.host_shape import HOST_SHAPE_ATTR, extract_host_shape_expressions


def tdict(name, dims=None, dtype=7):
    return {"name": name, "shape": list(dims or []), "dtype": dtype}


def add_node(m, node):
    m.nodes[node.name] = node
    return node


def init(m, name, values):
    arr = __import__("numpy").array(values, dtype=__import__("numpy").int64)
    m.initializers[name] = numpy_helper.from_array(arr, name)


def build_model():
    """Graph inputs x [1,-1,4], kv [1,2,2,-1,8] (past-like).

    Reshape_0 target = Concat(Shape(x)[0], Shape(x)[1], c4)  -> [1, seq, 4]
    Reshape_1 target = Gather(Shape(kv), 3) unsqueezed? no —
        Concat(Shape(kv)[3], c1)                             -> [kv, 1]
    Reshape_2 target = Gather(Shape(kv), -1)                 -> [8]
    Reshape_3 target = dynamic Gather idx (bail)
    """
    m = DAGBasedModel()
    m.inputs = [tdict("x", [1, -1, 4], dtype=1), tdict("kv", [1, 2, 2, -1, 8], dtype=1)]
    init(m, "c4", [4])
    init(m, "c1", [1])
    init(m, "i0", [0])
    init(m, "i3", [3])
    init(m, "ineg", [-1])

    sx = add_node(m, Node("Shape", "Sx", {}, [tdict("x", None)], [tdict("sx", [3])]))
    skv = add_node(m, Node("Shape", "Skv", {}, [tdict("kv", None)], [tdict("skv", [5])]))

    # Reshape_0: data dx, target Concat(sx[0], sx[1], 4)
    g0 = add_node(m, Node("Gather", "G0", {"axis": 0},
                          [tdict("sx", None), tdict("i0", None)], [tdict("g0", [1])]))
    c0 = add_node(m, Node("Concat", "C0", {"axis": 0},
                          [tdict("g0", None), tdict("sx", None), tdict("c4", None)],
                          [tdict("t0", [3])]))
    r0 = add_node(m, Node("Reshape", "R0", {},
                          [tdict("dx", None), tdict("t0", None)],
                          [tdict("r0_out", None, dtype=1)]))
    # sx is x's shape: x feeds Sx (ancestor of R0? x is a graph input -> ok;
    # Sx itself must be an ancestor of R0 — give R0 a data dep on dx produced
    # after Sx... simplest: dx is a graph input, so make Sx a "producer"
    # ancestor is not needed for the REF terms (they reference x, a graph
    # input). The REF source is x — always fine.

    # Reshape_1: target Concat(skv[3], 1)
    g1 = add_node(m, Node("Gather", "G1", {"axis": 0},
                          [tdict("skv", None), tdict("i3", None)], [tdict("g1", [1])]))
    c1 = add_node(m, Node("Concat", "C1", {"axis": 0},
                          [tdict("g1", None), tdict("c1", None)], [tdict("t1", [2])]))
    r1 = add_node(m, Node("Reshape", "R1", {},
                          [tdict("d1", None), tdict("t1", None)],
                          [tdict("r1_out", None, dtype=1)]))

    # Reshape_2: target Gather(skv, -1)
    g2 = add_node(m, Node("Gather", "G2", {"axis": 0},
                          [tdict("skv", None), tdict("ineg", None)], [tdict("t2", [1])]))
    r2 = add_node(m, Node("Reshape", "R2", {},
                          [tdict("d2", None), tdict("t2", None)],
                          [tdict("r2_out", None, dtype=1)]))

    # Reshape_3: target = Gather(sx, dyn_idx) where dyn_idx is a node output
    dyn = add_node(m, Node("Add", "AddIdx", {},
                           [tdict("i0", None), tdict("c1", None)], [tdict("dynidx", [1])]))
    g3 = add_node(m, Node("Gather", "G3", {"axis": 0},
                          [tdict("sx", None), tdict("dynidx", None)], [tdict("t3", [1])]))
    r3 = add_node(m, Node("Reshape", "R3", {},
                          [tdict("d3", None), tdict("t3", None)],
                          [tdict("r3_out", None, dtype=1)]))

    m.outputs = [r0.outputs[0], r1.outputs[0], r2.outputs[0], r3.outputs[0]]
    return m


def test_extracts_and_rewrites():
    m = build_model()
    stats = extract_host_shape_expressions(m)
    assert stats["reshapes"] == 4, stats
    assert stats["converted"] == 3, stats

    r0 = m.nodes["R0"]
    assert len(r0.inputs) == 1, r0.inputs
    # Concat(G0(sx[0]), sx, c4) = [sx0] ++ [x0,x1,x2] ++ [4]
    assert r0.attributes[HOST_SHAPE_ATTR] == "i0:x,i0:x,i1:x,i2:x,c4", r0.attributes

    r1 = m.nodes["R1"]
    assert r1.attributes[HOST_SHAPE_ATTR] == "i3:kv,c1", r1.attributes

    r2 = m.nodes["R2"]
    assert r2.attributes[HOST_SHAPE_ATTR] == "i4:kv", r2.attributes

    # dynamic Gather idx -> fail closed
    r3 = m.nodes["R3"]
    assert len(r3.inputs) == 2, r3.inputs
    assert HOST_SHAPE_ATTR not in r3.attributes
    return True


def test_non_ancestor_ref_bails():
    """A REF source produced by a node unreachable from the Reshape's data
    input must fail closed: after the chain is pruned, that producer could
    execute after the Reshape in the recomputed levels, and the lazy host
    evaluation would read a not-yet-final (stale) shape."""
    m = DAGBasedModel()
    m.inputs = [tdict("d", [2, 3], dtype=1), tdict("z", [1, -1, 4], dtype=1)]
    # y produced by an isolated branch (no path into R's data input)
    add_node(m, Node("Transpose", "Tz", {"perm": [0, 2, 1]},
                     [tdict("z", None)], [tdict("y", [1, 4, 3])]))
    add_node(m, Node("Shape", "Sy", {}, [tdict("y", [1, 4, 3])], [tdict("skv", [3])]))
    add_node(m, Node("Reshape", "R", {},
                     [tdict("d", None), tdict("skv", None)],
                     [tdict("out", None, dtype=1)]))
    m.outputs = [tdict("out")]
    stats = extract_host_shape_expressions(m)
    assert stats["converted"] == 0, stats
    assert len(m.nodes["R"].inputs) == 2
    return True


def test_ancestor_ref_converts():
    """Same shape chain, but now the REF source lies on the data path (the
    classic LLM pattern: Reshape(data=layer_hidden, target=Shape(layer_hidden))
    converted to a host expression) — must convert."""
    m = DAGBasedModel()
    m.inputs = [tdict("z", [1, -1, 4], dtype=1)]
    add_node(m, Node("Transpose", "Tz", {"perm": [0, 2, 1]},
                     [tdict("z", None)], [tdict("y", [1, 4, 3])]))
    add_node(m, Node("Shape", "Sy", {}, [tdict("y", [1, 4, 3])], [tdict("skv", [3])]))
    add_node(m, Node("Reshape", "R", {},
                     [tdict("y", None), tdict("skv", None)],
                     [tdict("out", None, dtype=1)]))
    m.outputs = [tdict("out")]
    stats = extract_host_shape_expressions(m)
    assert stats["converted"] == 1, stats
    assert m.nodes["R"].attributes[HOST_SHAPE_ATTR] == "i0:y,i1:y,i2:y"
    return True


def test_static_initializer_target_converts():
    m = DAGBasedModel()
    m.inputs = [tdict("x", [1, -1, 4], dtype=1)]
    init(m, "tgt", [1, 896])
    add_node(m, Node("Reshape", "R", {},
                     [tdict("d", None), tdict("tgt", None)],
                     [tdict("out", None, dtype=1)]))
    m.outputs = [tdict("out")]
    stats = extract_host_shape_expressions(m)
    assert stats["converted"] == 1, stats
    assert m.nodes["R"].attributes[HOST_SHAPE_ATTR] == "c1,c896"
    return True


def test_identity_cast_in_chain():
    m = DAGBasedModel()
    m.inputs = [tdict("x", [1, -1, 4], dtype=1)]
    sx = add_node(m, Node("Shape", "Sx", {}, [tdict("x", None)], [tdict("sx", [3])]))
    cx = add_node(m, Node("Cast", "Cx", {"to": 7},
                          [tdict("sx", None)], [tdict("cx", [3])]))
    add_node(m, Node("Reshape", "R", {},
                     [tdict("d", None), tdict("cx", None)],
                     [tdict("out", None, dtype=1)]))
    m.outputs = [tdict("out")]
    stats = extract_host_shape_expressions(m)
    assert stats["converted"] == 1, stats
    assert m.nodes["R"].attributes[HOST_SHAPE_ATTR] == "i0:x,i1:x,i2:x"
    return True


def test_consumer_view_reinterpretation_bails():
    """A REF source that some consumer reinterprets via a recorded reshape
    view (Unsqueeze-fold pattern: producer outputs 4-D, a consumer edge
    records 5-D — same element count, so reshape_view applies) must fail
    closed. The deleted Shape op read the producer's shape, but the runtime
    evaluates REF terms from the LIVE dims, which the intervening view would
    corrupt."""
    m = DAGBasedModel()
    m.inputs = [tdict("d", [2, 3], dtype=1)]
    add_node(m, Node("Transpose", "Tz", {"perm": [0, 2, 1]},
                     [tdict("d", None)], [tdict("y", [1, 4, 3])]))
    add_node(m, Node("Shape", "Sy", {}, [tdict("y", [1, 4, 3])], [tdict("skv", [3])]))
    # Fold-style consumer records a reinterpreted 5-D view on its y edge.
    add_node(m, Node("Expand", "Ex", {},
                     [tdict("y", [1, 4, 1, 3]), tdict("one", [1])],
                     [tdict("ex_out", [1, 4, 1, 3])]))
    # y IS the data input, so the reachability check passes and only the
    # view check can bail.
    add_node(m, Node("Reshape", "R", {},
                     [tdict("y", None), tdict("skv", None)],
                     [tdict("out", None, dtype=1)]))
    m.outputs = [tdict("out")]
    stats = extract_host_shape_expressions(m)
    assert stats["converted"] == 0, stats
    assert len(m.nodes["R"].inputs) == 2
    return True


def test_prune_removes_orphaned_chain_after_rewrite():
    """The converter fix for the `_prune_materialize_done` guard: after the
    rewrite orphans the int64 chain, the standard liveness prune (re-run at
    save time) must delete it. Converted Reshapes' chains die; the bailed
    R3's chain (Sx -> G3 -> AddIdx) stays live through its surviving
    consumer."""
    m = build_model()
    stats = extract_host_shape_expressions(m)
    assert stats["converted"] == 3
    m.prune_dead_and_materialize_constants()
    names = set(m.nodes.keys())
    # Orphaned by the rewrite of R0/R1/R2:
    for dead in ("G0", "C0", "G1", "C1", "G2", "Skv"):
        assert dead not in names, (dead, names)
    # Kept alive by the non-converted R3:
    for live in ("Sx", "G3", "AddIdx", "R3"):
        assert live in names, (live, names)
    return True


def test_real_llm_pipeline_conversion_count():
    """End-to-end: the full llm.onnx pipeline must convert (nearly) all
    Reshapes and thereby orphan the whole int64 shape chain."""
    import os
    import sys
    model_path = "/Users/uqland/github/vkop/llm/llm.onnx"
    if not os.path.exists(model_path):
        return True  # model not present in this checkout; skip
    from onnx2vkop.converter import ModelConverter

    class Args:
        quant = None
        unify = False
        rgba = False
        group_size = 64

    conv = ModelConverter()
    dag = conv.parse_onnx_model(model_path, 1)

    def count():
        c = {}
        for n in dag.nodes.values():
            c[n.op_type] = c.get(n.op_type, 0) + 1
        return c

    before = count()
    conv.apply_optimizations(dag, Args())
    # apply_optimizations runs the pass; prune happens in save_to_binary, so
    # count Shape/Gather chains remaining now: all Reshapes with the attr.
    from onnx2vkop.host_shape import HOST_SHAPE_ATTR as A
    n_rs = sum(1 for n in dag.nodes.values() if n.op_type == "Reshape")
    n_conv = sum(1 for n in dag.nodes.values()
                 if n.op_type == "Reshape" and A in n.attributes)
    # The view-consistency check legitimately bails Reshapes whose REF
    # sources carry a fold-recorded reinterpretation view; require the vast
    # majority to still convert (the whole point is orphaning the chain).
    assert n_conv >= n_rs * 0.9, (n_conv, n_rs)
    return True
