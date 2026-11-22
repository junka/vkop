"""drop_broadcast_expand and transpose_unit_axes_to_reshape.

Both remove data movement the VAE decoder spends ~35 ms of GPU time on: an
Expand that materializes a value its only reader would broadcast anyway, and a
Transpose that only moves size-1 axes (a view, which the runtime aliases as a
Reshape).
"""
import numpy as np
from onnx import TensorProto, numpy_helper

from onnx2vkop.dag import DAGBasedModel, Node
from onnx2vkop.optimizer import FusionOptimizer

F = TensorProto.FLOAT


def t(name, shape):
    return {"name": name, "shape": list(shape), "dtype": F}


def run(m, match, fold):
    hits = match(m)
    for h in hits:
        fold(m, h)
    return len(hits)


def expand_graph(other_shape):
    m = DAGBasedModel()
    m.initializers["shp"] = numpy_helper.from_array(
        np.array([1, 8, 1, 4, 4], dtype=np.int64), "shp")
    m.nodes["ex"] = Node("Expand", "ex", {},
                         [t("n", [1, 1, 1, 4, 4]),
                          {"name": "shp", "shape": [5], "dtype": TensorProto.INT64}],
                         [t("ne", [1, 8, 1, 4, 4])])
    m.nodes["div"] = Node("Div", "div", {},
                          [t("x", other_shape), t("ne", [1, 8, 1, 4, 4])],
                          [t("y", [1, 8, 1, 4, 4])])
    return m


def test_expand_into_broadcasting_div_is_dropped():
    m = expand_graph([1, 8, 1, 4, 4])
    assert run(m, FusionOptimizer.match_broadcast_expand,
               FusionOptimizer.fold_broadcast_expand) == 1
    assert "ex" not in m.nodes
    assert m.nodes["div"].inputs[1]["name"] == "n"


def test_expand_kept_when_it_sets_the_output_shape():
    # The other operand is as small as the Expand input: without the Expand
    # the Div would produce [1,1,1,4,4], not [1,8,1,4,4].
    m = expand_graph([1, 1, 1, 4, 4])
    assert run(m, FusionOptimizer.match_broadcast_expand,
               FusionOptimizer.fold_broadcast_expand) == 0


def transpose_graph(shape, perm):
    out = [shape[p] for p in perm]
    m = DAGBasedModel()
    m.nodes["tr"] = Node("Transpose", "tr", {"perm": perm},
                         [t("x", shape)], [t("y", out)])
    return m


def test_unit_axis_transpose_becomes_reshape():
    m = transpose_graph([1, 1, 6, 4, 4], [0, 2, 1, 3, 4])
    assert run(m, FusionOptimizer.match_unit_axis_transpose,
               FusionOptimizer.fold_unit_axis_transpose) == 1
    n = m.nodes["tr"]
    assert n.op_type == "Reshape"
    shape = numpy_helper.to_array(m.initializers[n.inputs[1]["name"]])
    assert list(shape) == [1, 6, 1, 4, 4]
    # Same element order: a transpose of the data equals a reshape of it.
    x = np.arange(96).reshape(1, 1, 6, 4, 4)
    np.testing.assert_array_equal(np.transpose(x, (0, 2, 1, 3, 4)).ravel(),
                                  x.ravel())


def test_real_transpose_is_kept():
    m = transpose_graph([1, 2, 6, 4, 4], [0, 2, 1, 3, 4])
    assert run(m, FusionOptimizer.match_unit_axis_transpose,
               FusionOptimizer.fold_unit_axis_transpose) == 0
