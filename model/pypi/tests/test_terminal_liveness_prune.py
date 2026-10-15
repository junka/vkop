"""Standalone tests for DAGBasedModel.terminal_liveness_prune.

The sweep runs at write time because that is the only point after which no
optimization pass can orphan anything: the fusion passes rebind or delete
consumers without touching dag_model.nodes / .initializers, so anything they
strand would otherwise be uploaded by LoadModel for no reader (one staging copy
plus one submit/fence-wait per initializer entry).

The node half is what makes the initializer half complete: an initializer read
only by a dead node is as dead as that node, and a keep set built from ALL node
inputs (live or not) would keep it.
"""
import numpy as np
from onnx import numpy_helper

from onnx2vkop.dag import DAGBasedModel, Node


def tdict(name, dims=None, dtype=7):
    return {"name": name, "shape": list(dims or []), "dtype": dtype}


def init(name, values):
    return numpy_helper.from_array(np.array(values, dtype=np.float32), name)


def linear_model():
    """x -> Mul(k) -> Add(b) -> out, plus a dead chain.

    Dead: Sq -> Square -> Add(dead_sum) -> dead_out. dead_out is not a graph
    output, so the whole chain is unreachable and its private constants
    (dead_const) must go with it.
    """
    m = DAGBasedModel()
    m.inputs = [tdict("x", [2, 2], dtype=1)]
    m.outputs = [tdict("out", [2, 2], dtype=1)]

    k = Node("Mul", "mul", {}, [tdict("x"), tdict("k")], [tdict("mul_out")])
    b = Node("Add", "add", {}, [tdict("mul_out"), tdict("b")], [tdict("out")])
    m.nodes[k.name] = k
    m.nodes[b.name] = b

    sq = Node("Shape", "sq", {}, [tdict("x")], [tdict("sq_out")])
    sqr = Node("Square", "square", {}, [tdict("sq_out")], [tdict("sqr_out")])
    dead = Node("Add", "dead_sum", {},
                [tdict("sqr_out"), tdict("dead_const")], [tdict("dead_out")])
    m.nodes[sq.name] = sq
    m.nodes[sqr.name] = sqr
    m.nodes[dead.name] = dead

    m.initializers = {
        "k": init("k", [1.0]),
        "b": init("b", [2.0]),
        "dead_const": init("dead_const", [3.0]),
    }
    return m


def test_drops_dead_nodes_and_their_private_initializers():
    m = linear_model()
    m.terminal_liveness_prune()
    assert set(m.nodes) == {"mul", "add"}
    assert set(m.initializers) == {"k", "b"}


def test_live_graph_is_left_untouched():
    m = linear_model()
    m.terminal_liveness_prune()
    before = (set(m.nodes), set(m.initializers))
    m.terminal_liveness_prune()  # idempotent
    assert (set(m.nodes), set(m.initializers)) == before


def test_graph_output_named_after_an_initializer_keeps_it():
    m = linear_model()
    m.outputs = [tdict("out"), tdict("k")]
    m.terminal_liveness_prune()
    assert "k" in m.initializers


def test_unified_member_survives_without_any_consumer():
    m = linear_model()
    m.terminal_liveness_prune()  # "dead_const" is gone by now
    m.initializers["unif"] = init("unif", [4.0])
    m.unified_meta = [{"_name": "unif", "offset": 0, "size": 4, "dims": [1],
                       "name_len": 4, "dtype": 1}]
    m.terminal_liveness_prune()
    assert "unif" in m.initializers


def test_dependencies_are_rebuilt_after_a_node_disappears():
    m = linear_model()
    m.build_dependencies()
    dead_before = m.nodes["dead_sum"].dependencies
    assert dead_before
    m.terminal_liveness_prune()
    # Nothing references a removed node any more, so the level build (and
    # topological_sort, which raises on a short sort) stays consistent.
    assert m.nodes["mul"].dependencies == set()
    assert m.nodes["add"].dependencies == {"mul"}
    assert [n.name for n in m.topological_sort()] == ["mul", "add"]
