"""Standalone tests for fuse_kv_cache_append (priority 74).

The pass folds HF's per-layer KV-cache rotation into the Attention op's cache
form:

    past_key_values_L [1,2,Hkv,kv_past,D]
        |             |
     Gather(ax=1,0) Gather(ax=1,1)
        |               |
        |          v_new     k_new            (this round's projected rows)
        |               |         |
     Concat_5(ax=2)  Concat_6(ax=2)           K_full / V_full
        |    \\         |    \\               (both feed the Attention *and*)
        |   Unsqueeze_32  Unsqueeze_33        (the present Concat)
        |        |             |
        |        ---Concat_12(ax=1)---
        |                    |
    Attention(...)  present_key_values_L

becomes ``Attention(Q, k_new, v_new, past[, mask]) -> [Y, present]`` with
has_cache=1, and all 7 rotation nodes are deleted. On Qwen3-VL that is 196
dispatches/round (48-71% of a decode round at kv_len 1024..4096) that no longer
touch the cache at all.

What is being pinned here is the *soundness* half: every node the fold deletes
is checked single-consumer first, the two Gathers must read the same graph
input (that is what makes it a cache and not two unrelated slices), the indices
must be 0/1 (the layout the kernel hardcodes), and `present` must be the graph
output the driver preallocates. Each of those has a decline test below.

The kernel side is covered numerically by BufferRankTest.AttentionFp32AndFp16.
"""
import os
import sys

import numpy as np
import pytest

from onnx2vkop.dag import DAGBasedModel, Node
from onnx2vkop.optimizer import FusionOptimizer

# Model.Nodes(j) does a lazy *absolute* import of `vkop.model.Node` (generated
# FlatBuffers code), so any reader has to have onnx2vkop/generated on sys.path.
_GENERATED = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "onnx2vkop", "generated")
if _GENERATED not in sys.path:
    sys.path.insert(0, _GENERATED)


def tdict(name, dims=None, dtype=10):
    return {"name": name, "shape": list(dims or []), "dtype": dtype}


def count(dag, op_type):
    return sum(1 for n in dag.nodes.values() if n.op_type == op_type)


def _const(value, raw=False):
    """A scalar/axes initializer. The loader folds Constants into TensorProto
    initializers, but a Constant that survived as a node carries an ndarray —
    the matcher has to read both, so `raw` builds the latter."""
    arr = np.array(value, dtype=np.int64)
    if raw:
        return arr
    from onnx import numpy_helper

    return numpy_helper.from_array(arr, "c")


def matches_of(dag):
    return FusionOptimizer.match_kv_cache_append(dag)


# The post-fuse_into_attention shapes of one Qwen3-VL layer: 16 query heads, 8
# KV heads, res=128, and a symbolic [1, 2, 8, kv, 128] cache.
Q_DIMS = [-1, 16, -1, 128]
KV_DIMS = [-1, 8, -1, 128]
PAST_DIMS = [1, 2, 8, 1, 128]
PRESENT_DIMS = [-1, 2, 8, -1, 128]


def build(mask=True, past_is_input=True, present_is_output=True,
          gather_indices=(0, 1), concat_axis=2, present_axis=1,
          unsqueeze_shape=None, extra_k_consumer=False, extra_gather_consumer=False,
          reversed_concat=False, k_gather_other_past=False, raw_constants=False):
    """Assemble the Attention + KV-rotation subgraph and return the DAG."""
    m = DAGBasedModel()

    m.nodes["ATT"] = Node(
        "Attention", "ATT", {"num_heads": 16, "num_kv_heads": 8, "causal": 0,
                             "scale": 0.08837890625},
        [tdict("q", Q_DIMS), tdict("k_full", KV_DIMS), tdict("v_full", KV_DIMS)]
        + ([tdict("attention_bias", [1, 1, 1, 1])] if mask else []),
        [tdict("out", Q_DIMS)])

    m.initializers["c0"] = _const(gather_indices[0])
    m.initializers["c1"] = _const(gather_indices[1])
    m.initializers["axes1"] = _const([1], raw=raw_constants)

    for tag, past, idx in (("K", "past_key_values_0", "c0"),
                           ("V", "past_key_values_0", "c1")):
        if tag == "V" and k_gather_other_past:
            past = "past_key_values_1"
        m.nodes["G_" + tag] = Node(
            "Gather", "G_" + tag, {"axis": 1},
            [tdict(past, PAST_DIMS), tdict(idx, [])],
            [tdict("g_" + tag + "_out", KV_DIMS)])

    if extra_gather_consumer:
        m.nodes["G_EXTRA"] = Node("Cast", "G_EXTRA", {},
                                  [tdict("g_K_out", KV_DIMS)],
                                  [tdict("g_K_cast", KV_DIMS)])

    for tag, new in (("K", "k_new"), ("V", "v_new")):
        concat_in = ([tdict(new, KV_DIMS), tdict("g_" + tag + "_out", KV_DIMS)]
                     if reversed_concat else
                     [tdict("g_" + tag + "_out", KV_DIMS), tdict(new, KV_DIMS)])
        m.nodes["C_" + tag] = Node(
            "Concat", "C_" + tag, {"axis": concat_axis}, concat_in,
            [tdict(tag.lower() + "_full", KV_DIMS)])

    if extra_k_consumer:
        m.nodes["K_EXTRA"] = Node("Cast", "K_EXTRA", {},
                                  [tdict("k_full", KV_DIMS)],
                                  [tdict("k_cast", KV_DIMS)])

    for tag in ("K", "V"):
        m.nodes["U_" + tag] = Node(
            "Unsqueeze", "U_" + tag, {},
            [tdict(tag.lower() + "_full", KV_DIMS), tdict("axes1", [1])],
            [tdict("u_" + tag + "_out",
                   unsqueeze_shape or [-1, 1, 8, -1, 128])])

    m.nodes["C_PRESENT"] = Node(
        "Concat", "C_PRESENT", {"axis": present_axis},
        [tdict("u_K_out", [-1, 1, 8, -1, 128]),
         tdict("u_V_out", [-1, 1, 8, -1, 128])],
        [tdict("present_key_values_0", PRESENT_DIMS)])

    if past_is_input:
        m.inputs.append(tdict("past_key_values_0", PAST_DIMS))
    else:
        m.nodes["PAST_SRC"] = Node("Cast", "PAST_SRC", {},
                                   [tdict("past_src", PAST_DIMS)],
                                   [tdict("past_key_values_0", PAST_DIMS)])
        m.inputs.append(tdict("past_src", PAST_DIMS))
    m.inputs.append(tdict("past_key_values_1", PAST_DIMS))
    # Q/K_new/V_new and the mask come from earlier nodes in the real graph;
    # this fragment declares them as inputs, which is the same statement.
    m.inputs += [tdict("q", Q_DIMS), tdict("k_new", KV_DIMS),
                 tdict("v_new", KV_DIMS), tdict("attention_bias", [1, 1, 1, 1])]
    m.outputs = [tdict("out", Q_DIMS)]
    if present_is_output:
        m.outputs.append(tdict("present_key_values_0", PRESENT_DIMS))
    return m


ROTATION = ["G_K", "G_V", "C_K", "C_V", "U_K", "U_V", "C_PRESENT"]


# ---------------------------------------------------------------- positives

def test_canonical_rotation_folds():
    m = build()
    n0 = len(m.nodes)
    ms = matches_of(m)
    assert len(ms) == 1, ms
    assert FusionOptimizer.fold_kv_cache_append(m, ms[0])

    assert len(m.nodes) == n0 - 7
    for gone in ROTATION:
        assert gone not in m.nodes, gone
    fused = m.nodes["ATT"]
    # past slots in at index 3, ahead of the additive mask.
    assert [i["name"] for i in fused.inputs] == [
        "q", "k_new", "v_new", "past_key_values_0", "attention_bias"]
    assert [o["name"] for o in fused.outputs] == [
        "out", "present_key_values_0"]
    assert fused.attributes["has_cache"] == 1
    # output 0 keeps the MatMul's tensor name so downstream still resolves
    assert fused.outputs[0]["name"] == "out"
    # and every graph output is still produced by something
    produced = {o["name"] for n in m.nodes.values() for o in n.outputs}
    assert {o["name"] for o in m.outputs} <= produced


def test_no_mask_folds():
    m = build(mask=False)
    ms = matches_of(m)
    assert len(ms) == 1, ms
    assert FusionOptimizer.fold_kv_cache_append(m, ms[0])
    assert [i["name"] for i in m.nodes["ATT"].inputs] == [
        "q", "k_new", "v_new", "past_key_values_0"]


def test_folded_constant_index_still_folds():
    """The Gather index may arrive as an ndarray (a Constant node the folder
    left in place) rather than a TensorProto initializer; both spell the same
    scalar and must read the same."""
    m = build(raw_constants=True)
    ms = matches_of(m)
    assert len(ms) == 1, ms
    assert FusionOptimizer.fold_kv_cache_append(m, ms[0])
    assert m.nodes["ATT"].attributes["has_cache"] == 1


# ---------------------------------------------------------------- declines

def test_swapped_gather_indices_declined():
    """K must be dim-1 index 0 and V index 1 — that is the [1,2,...] layout the
    kernel hardcodes (c_base for K, +nkv*stride for V)."""
    assert matches_of(build(gather_indices=(1, 0))) == []


def test_third_consumer_on_k_full_declined():
    """Deleting C_K would strand K_EXTRA on a tensor nobody writes."""
    assert matches_of(build(extra_k_consumer=True)) == []


def test_extra_consumer_on_gather_declined():
    assert matches_of(build(extra_gather_consumer=True)) == []


def test_past_not_a_graph_input_declined():
    """A cache the driver did not allocate — hence did not preallocate to
    max_kv — is not something the op can read at a stable row pitch."""
    assert matches_of(build(past_is_input=False)) == []


def test_present_not_a_graph_output_declined():
    """`present` must be the graph output the driver preallocates; otherwise
    the op's output 1 would be a tensor nothing binds."""
    assert matches_of(build(present_is_output=False)) == []


def test_concat_axis_1_declined():
    """The history must be *prepended along kv*, not stacked."""
    assert matches_of(build(concat_axis=1)) == []


def test_present_axis_0_declined():
    assert matches_of(build(present_axis=0)) == []


def test_reversed_concat_declined():
    """[new, history] would put the fresh rows at positions [0, S) — the kernel
    indexes K_new from kv_past."""
    assert matches_of(build(reversed_concat=True)) == []


def test_unsqueeze_not_at_axis_1_declined():
    assert matches_of(build(unsqueeze_shape=[1, 8, -1, -1, 128])) == []


def test_k_and_v_gathers_from_different_past_declined():
    """Both gathers must read the *same* graph input: two different tensors are
    not one [1,2,...] cache."""
    assert matches_of(build(k_gather_other_past=True)) == []


def test_already_folded_is_idempotent():
    """After the fold the Attention takes k_new directly, so a second round of
    the optimizer finds nothing to do."""
    m = build()
    assert FusionOptimizer.fold_kv_cache_append(m, matches_of(m)[0])
    assert matches_of(m) == []


def test_written_artifact_carries_the_cache_form():
    """The runtime reads the op's surface off the .vkopbin, not off this dict:
    5 positional inputs (past at index 3, the mask pushed to 4), 2 outputs
    (present is what the driver binds its preallocated buffer to), and
    has_cache as a serializable scalar. A writer that dropped any of those
    would leave the op gating on the wrong tensor or on nothing at all."""
    import mmap
    import os
    import tempfile

    from onnx2vkop.generated.vkop.model.AttrType import AttrType
    from onnx2vkop.generated.vkop.model.Model import Model

    m = build()
    assert FusionOptimizer.fold_kv_cache_append(m, matches_of(m)[0])
    m.build_dependencies()

    with tempfile.TemporaryDirectory() as d:
        out = os.path.join(d, "kv_cache.vkopbin")
        m.save_to_binary(out)
        with open(out, "rb") as fh:
            mm = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
            model = Model.GetRootAsModel(mm, 0)
            found = None
            for i in range(model.NodesLength()):
                node = model.Nodes(i)
                if node.OpType() == b"Attention":
                    found = node
                    break
            assert found is not None, "Attention missing from the artifact"
            inputs = [found.Inputs(k).Name().decode()
                      for k in range(found.InputsLength())]
            outputs = [found.Outputs(k).Name().decode()
                       for k in range(found.OutputsLength())]
            attrs = {}
            for k in range(found.AttributesLength()):
                a = found.Attributes(k)
                key = a.Key().decode()
                if a.Type() == AttrType.Int64:
                    attrs[key] = a.Ival()
                elif a.Type() == AttrType.Float32:
                    attrs[key] = a.Fval()
            mm.close()

    assert inputs == ["q", "k_new", "v_new", "past_key_values_0",
                      "attention_bias"], inputs
    assert outputs == ["out", "present_key_values_0"], outputs
    assert attrs["has_cache"] == 1, attrs
    assert attrs["num_heads"] == 16 and attrs["num_kv_heads"] == 8, attrs
    assert abs(attrs["scale"] - 0.08837890625) < 1e-9, attrs
