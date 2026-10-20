"""Const-index Gather folding.

`Gather(data, indices, axis)` with a constant `indices` is common in exported
graphs (grid decomposition, shape-vector element extraction, picking K or V out
of a stacked pair) and the exporter never emits a cheaper equivalent. The pass
rewrites three exactly-equivalent cases:

  * unit-length axis + scalar index  -> Squeeze (which the next round aliases
    into a metadata-only view, i.e. zero dispatches);
  * indices == [0..n-1]              -> identity: the node disappears and
    consumers read `data`;
  * data is a constant table whose ONLY consumer is this Gather -> the selected
    slice is materialized as a new initializer under the Gather's own output
    name and the table is dropped, so the blob can only shrink. This is the
    gap ConstantFolder leaves open: it refuses to materialize any fold result
    above 1 MiB (the protobuf serialization cap), so a big lookup table sliced
    by a constant index stayed in the graph as a weight plus a dispatch.

Everything else is left alone, which is what most of these tests pin down:
a multi-element axis with a scalar index (the ViT's grid[0]/grid[1] extraction)
is NOT a squeeze, a dynamic index is not constant, >1-D indices change the
rank, and a table with another consumer must not be duplicated.
"""
import numpy as np
from onnx import numpy_helper

from onnx2vkop.dag import DAGBasedModel, Node
from onnx2vkop.optimizer import FusionOptimizer

FP16 = 10   # TensorProto.FLOAT16
INT64 = 7


def td(name, dims=None, dtype=FP16):
    d = {"name": name, "dtype": dtype}
    if dims is not None:
        d["shape"] = list(dims)
    return d


def init(m, name, arr):
    m.initializers[name] = numpy_helper.from_array(np.asarray(arr), name)


def build(in_shape=(1, 3), in_dtype=INT64, indices=np.array(0, dtype=np.int64),
          indices_as_input=False, axis=0, table=None, table_name="T",
          table_shape=(8, 4), consumer_shape=None, out_is_graph_output=False,
          extra_table_consumer=False, logits_as_consumer=False):
    """Gather(X | table) -> [Reshape] and nothing else.

    `X` is a graph input; `T` (when given) is an initializer table. The
    consumer is a Reshape so a consumer-shape assertion stays simple, and
    because Reshape is what shape-vector gathers actually feed in the graphs.
    """
    m = DAGBasedModel()
    if table is not None:
        init(m, table_name, table)
        data = td(table_name, table_shape, FP16 if table.dtype == np.float16 else INT64)
        data_shape = list(table_shape)
    else:
        data = td("X", in_shape, in_dtype)
        data_shape = list(in_shape)

    if consumer_shape is None:
        rank = len(data_shape)
        a = axis + rank if axis < 0 else axis
        out_shape = data_shape[:a] + data_shape[a + 1:] if np.ndim(indices) == 0 \
            else data_shape[:a] + [int(np.size(indices))] + data_shape[a + 1:]
    else:
        out_shape = list(consumer_shape)
    out = td("gat_out", out_shape)

    ins = [data]
    if indices_as_input:
        ins.append(td("idx", [1] if np.ndim(indices) else [], INT64))
    else:
        init(m, "idx", indices)
        ins.append(td("idx", list(np.shape(indices)) or [], INT64))
    a = {"axis": axis}
    g = Node("Gather", "g", a, ins, [out])
    m.nodes[g.name] = g

    # Declare the graph inputs: the DAG's liveness pruning seeds from the
    # outputs, so an undeclared tensor would make the whole chain look dead.
    if table is None:
        m.inputs.append(td("X", in_shape, in_dtype))
    if indices_as_input:
        m.inputs.append(td("idx", list(np.shape(indices)) or [], INT64))

    init(m, "rs", np.array([-1], dtype=np.int64))
    n_out = int(np.prod(out_shape)) if out_shape else 1
    # Two dims out, so the Reshape is never "redundant" (input shape == output
    # shape) and remove_redundant_reshape cannot short-circuit the chain.
    rs = Node("Reshape", "rs_node", {},
              [out, {"name": "rs", "shape": [1], "dtype": INT64}],
              [td("y", [1, n_out])])
    m.nodes[rs.name] = rs
    if out_is_graph_output:
        m.outputs = [td("gat_out", out_shape)]
    else:
        m.outputs = [td("y", [1, n_out])]

    if extra_table_consumer:
        # A second Gather over the same table with a different const index.
        other = Node("Gather", "g2", {"axis": 0},
                     [td(table_name, table_shape),
                      {"name": "idx2", "shape": [], "dtype": INT64}],
                     [td("gat2_out", [table_shape[1]])])
        m.nodes[other.name] = other
        init(m, "idx2", np.array(1, dtype=np.int64))
        m.outputs.append(td("gat2_out", [table_shape[1]]))

    if logits_as_consumer:
        mm = Node("MatMul", "mm", {"transB": 1},
                  [out, td("W", [4, out_shape[-1]])],
                  [td("logits", [out_shape[0], 4])])
        m.nodes[mm.name] = mm
        init(m, "W", np.zeros((4, out_shape[-1]), dtype=np.float16))
        m.outputs.append(td("logits", [out_shape[0], 4]))

    m.build_dependencies()
    return m


def run_fusion(m, max_rounds=3):
    opt = FusionOptimizer.create_default_optimizer(max_rounds=max_rounds)
    opt.optimize(m, verbose=False)
    return m


def optypes(m):
    return sorted(n.op_type for n in m.nodes.values())


def consumer_of(m, name):
    """The single node reading `name`, or None."""
    hits = [n for n in m.nodes.values()
            for inp in n.inputs if inp["name"] == name]
    return hits[0] if hits else None


# ---------------------------------------------------------------- case (1)
def test_unit_axis_scalar_index_becomes_squeeze():
    """[1,3] input, axis 0 has extent 1 -> the Gather's direct replacement is a
    Squeeze that keeps the name and the declared output shape."""
    m = build(in_shape=(1, 3), indices=np.array(0, dtype=np.int64))
    run_fusion(m, max_rounds=1)
    assert "Gather" not in optypes(m), optypes(m)
    sq = [n for n in m.nodes.values() if n.op_type == "Squeeze"]
    assert len(sq) == 1, optypes(m)
    assert sq[0].name == "g", sq[0].name          # same node name, same output
    assert sq[0].outputs[0]["name"] == "gat_out", sq[0].outputs
    assert sq[0].outputs[0]["shape"] == [3], sq[0].outputs
    axes = numpy_helper.to_array(m.initializers[sq[0].inputs[1]["name"]])
    assert axes.tolist() == [0], axes
    assert consumer_of(m, "gat_out").op_type == "Reshape"
    print("  PASS (unit axis + scalar index -> Squeeze(axes=[0]), names kept)")


def test_squeeze_is_then_aliased_away():
    """One more round: the aliasable-squeeze pass turns it into a view, so the
    consumer reads the input tensor directly with the squeezed shape."""
    m = build(in_shape=(1, 3), indices=np.array(0, dtype=np.int64))
    run_fusion(m, max_rounds=3)
    assert optypes(m) == ["Reshape"], optypes(m)
    rs = m.nodes["rs_node"]
    assert rs.inputs[0]["name"] == "X", rs.inputs
    assert rs.inputs[0]["shape"] == [3], rs.inputs
    print("  PASS (next round aliases the Squeeze -> 0 dispatches)")


def test_negative_axis_unit_dim_folds():
    m = build(in_shape=(2, 3, 1), indices=np.array(-1, dtype=np.int64), axis=-1)
    run_fusion(m, max_rounds=1)
    sq = [n for n in m.nodes.values() if n.op_type == "Squeeze"]
    assert len(sq) == 1, optypes(m)
    axes = numpy_helper.to_array(m.initializers[sq[0].inputs[1]["name"]])
    assert axes.tolist() == [2], axes
    print("  PASS (axis=-1 normalizes to the same unit-axis Squeeze)")


def test_multi_element_axis_scalar_index_not_folded():
    """The ViT's grid_thw[0] / grid_thw[1]: a scalar index into a 3-vector is
    not a squeeze (dropping the axis would need a real slice)."""
    m = build(in_shape=(3,), indices=np.array(0, dtype=np.int64))
    run_fusion(m)
    assert "Gather" in optypes(m), optypes(m)
    print("  PASS (scalar index into a multi-element axis left alone)")


def test_unknown_axis_extent_not_folded():
    """dim == -1 (dynamic): cannot prove the axis has extent 1."""
    m = build(in_shape=(-1, 3), indices=np.array(0, dtype=np.int64))
    run_fusion(m)
    assert "Gather" in optypes(m), optypes(m)
    print("  PASS (dynamic axis extent left alone)")


def test_2d_indices_not_folded():
    m = build(in_shape=(1, 3), indices=np.zeros((2, 2), dtype=np.int64))
    run_fusion(m)
    assert "Gather" in optypes(m), optypes(m)
    print("  PASS (2-D indices would raise the rank, left alone)")


def test_dynamic_indices_not_folded():
    m = build(in_shape=(1, 3), indices=np.array(0, dtype=np.int64),
              indices_as_input=True)
    run_fusion(m)
    assert "Gather" in optypes(m), optypes(m)
    print("  PASS (non-constant index left alone)")


def test_graph_output_not_folded():
    m = build(in_shape=(1, 3), indices=np.array(0, dtype=np.int64),
              out_is_graph_output=True)
    run_fusion(m)
    assert "Gather" in optypes(m), optypes(m)
    print("  PASS (Gather output is a graph output: left alone)")


# ---------------------------------------------------------------- case (2)
def test_identity_indices_fold_to_input():
    m = build(in_shape=(4, 3), indices=np.arange(3, dtype=np.int64), axis=1)
    run_fusion(m)
    assert "Gather" not in optypes(m), optypes(m)
    rs = m.nodes["rs_node"]
    assert rs.inputs[0]["name"] == "X", rs.inputs
    assert rs.inputs[0]["shape"] == [4, 3], rs.inputs   # rank and shape kept
    print("  PASS (indices 0..n-1 -> consumers read the input, rank kept)")


def test_partial_permutation_not_folded():
    m = build(in_shape=(4, 3), indices=np.array([0, 2, 1], dtype=np.int64),
              axis=1)
    run_fusion(m)
    assert "Gather" in optypes(m), optypes(m)
    print("  PASS (a real permutation is not an identity)")


# ---------------------------------------------------------------- case (3)
def test_const_table_single_consumer_materialized():
    """A table sliced by a constant index: the slice becomes an initializer
    under the Gather's output name and the table is dropped."""
    table = (np.arange(32, dtype=np.float16).reshape(8, 4))
    idx = np.array([2, 5], dtype=np.int64)
    m = build(table=table, table_shape=(8, 4), indices=idx)
    run_fusion(m)
    assert "Gather" not in optypes(m), optypes(m)
    assert "T" not in m.initializers, "table survived"
    got = numpy_helper.to_array(m.initializers["gat_out"])
    assert got.shape == (2, 4), got.shape
    assert np.array_equal(got, table[[2, 5]]), got
    assert consumer_of(m, "gat_out").op_type == "Reshape"
    print("  PASS (const table sliced -> initializer, table dropped)")


def test_const_table_scalar_index_materialized():
    """Scalar index into a rank-2 table drops one axis, exactly like ONNX."""
    table = np.arange(32, dtype=np.float16).reshape(8, 4)
    m = build(table=table, table_shape=(8, 4), indices=np.array(3, dtype=np.int64))
    run_fusion(m)
    assert "Gather" not in optypes(m), optypes(m)
    got = numpy_helper.to_array(m.initializers["gat_out"])
    assert got.shape == (4,), got.shape
    assert np.array_equal(got, table[3]), got
    print("  PASS (scalar index into a rank-2 table -> rank-1 initializer)")


def test_shape_vector_element_left_alone():
    """A scalar index into a rank-1 table would produce a rank-0 result, and a
    0-d initializer loses its data in the runtime's Tensor<T> (size_ == 0), so
    this shape is declined rather than materialized."""
    table = np.array([14, 14, 2], dtype=np.int64)
    m = build(table=table, table_shape=(3,), indices=np.array(2, dtype=np.int64))
    run_fusion(m)
    assert "Gather" in optypes(m), optypes(m)
    assert "T" in m.initializers, "rank-1 table was dropped"
    print("  PASS (scalar index into a rank-1 table left alone: rank-0 result)")


def test_shape_vector_slice_materialized():
    """An int64 shape-vector slice: the dtype has to survive the
    materialization (Range/Div downstream read it as int64)."""
    table = np.array([14, 14, 2], dtype=np.int64)
    m = build(table=table, table_shape=(3,), indices=np.array([0, 2],
              dtype=np.int64))
    run_fusion(m)
    assert "Gather" not in optypes(m), optypes(m)
    got = numpy_helper.to_array(m.initializers["gat_out"])
    assert got.dtype == np.int64 and got.tolist() == [14, 2], got
    print("  PASS (int64 shape-vector slice materialized as int64)")


def test_const_table_non_leading_axis_materialized():
    """axis != 0 exercises the literal np.take axis argument."""
    table = np.arange(32, dtype=np.float16).reshape(4, 8)
    m = build(table=table, table_shape=(4, 8), indices=np.array([1, 3],
              dtype=np.int64), axis=1)
    run_fusion(m)
    assert "Gather" not in optypes(m), optypes(m)
    got = numpy_helper.to_array(m.initializers["gat_out"])
    assert got.shape == (4, 2), got.shape
    assert np.array_equal(got, table[:, [1, 3]]), got
    print("  PASS (axis=1 table slice materialized along the right axis)")


def test_axis_out_of_range_not_folded():
    """A malformed axis must never index past the shape list."""
    m = build(in_shape=(1, 3, 2), indices=np.array(0, dtype=np.int64), axis=5)
    assert FusionOptimizer.match_const_index_gather(m) == []
    run_fusion(m)
    assert "Gather" in optypes(m), optypes(m)
    print("  PASS (out-of-range axis left alone)")


def test_shared_table_not_materialized():
    """Another consumer of the table: duplicating the slice would grow the blob,
    so the fold declines (and none of the other rules apply here)."""
    table = np.arange(32, dtype=np.float16).reshape(8, 4)
    m = build(table=table, table_shape=(8, 4),
              indices=np.array([2, 5], dtype=np.int64),
              extra_table_consumer=True)
    run_fusion(m)
    assert optypes(m).count("Gather") == 2, optypes(m)
    assert "T" in m.initializers, "shared table was dropped"
    assert "gat_out" not in m.initializers, "shared table was duplicated"
    print("  PASS (table with another consumer left alone)")


def test_const_table_graph_input_not_materialized():
    """The table is also a graph input: the driver fills it, so it must stay."""
    table = np.arange(32, dtype=np.float16).reshape(8, 4)
    m = build(table=table, table_shape=(8, 4), indices=np.array([2, 5], dtype=np.int64))
    m.inputs.append(td("T", (8, 4), FP16))
    run_fusion(m)
    assert "Gather" in optypes(m), optypes(m)
    assert "T" in m.initializers, "graph-input table was dropped"
    print("  PASS (table that is also a graph input left alone)")


def test_unit_axis_const_table_prefers_materialization():
    """Both rules apply (const table + unit axis): dropping the table wins over
    swapping in a Squeeze."""
    table = np.arange(4, dtype=np.float16).reshape(1, 4)
    m = build(table=table, table_shape=(1, 4), indices=np.array(0, dtype=np.int64))
    m.inputs.append(td("X", (1, 4)))
    run_fusion(m)
    assert "Gather" not in optypes(m), optypes(m)
    assert "Squeeze" not in optypes(m), optypes(m)
    got = numpy_helper.to_array(m.initializers["gat_out"])
    assert got.shape == (4,), got.shape
    assert np.array_equal(got, table[0]), got
    print("  PASS (const table on a unit axis -> materialized, not Squeezed)")
