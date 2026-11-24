"""ConstantFolder must not fold `Shape(x)` when x's RANK is unknown.

ONNX distinguishes two states that both look like "no dims":

  * `tensor_type` with no `shape` field  -> rank unknown (nothing is known);
  * `tensor_type.shape` with 0 dims      -> a genuine rank-0 scalar.

`_shape_known` used `all(...)` over the dim list, which is vacuously True for
both. So `Shape(x)` on a tensor whose shape inference produced NO shape field
folded to an EMPTY int64 array instead of a shape vector. The rest of the chain
then folded to empty and the consuming Mul recorded a 0 in its output dims.

At run time that 0 is honored as a genuinely empty axis (the converter's own
comment says "a concrete 0 means genuinely empty"), so the runtime sized a real
tensor to a zero-length dim. The MiniMax-H3 denoiser and video VAE both hit it
in their `Shape(Transpose(q)) -> Slice -> Cast -> Sqrt -> Div -> Sqrt -> Mul`
attention-scale chain, and Softmax then divided by zero (`total / axis_size`,
`ops/Softmax.hpp:170`). ort ran the same ONNX files fine.

These tests pin the gate, not the whole converter: a known-rank operand still
folds, an unknown-rank one does not, and a rank-0 scalar still folds to `[]`.
"""
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from onnx2vkop.optimizer import ConstantFolder


def _model(*, declared_dims=None):
    """``sf = Cast(Shape(x))`` with x's value_info controlled.

    ``declared_dims is None`` leaves x's `shape` field UNSET, which is ONNX's
    "rank unknown". A list (possibly empty) declares that rank concretely.
    """
    nodes = [
        helper.make_node("Shape", ["x"], ["s"]),
        helper.make_node("Cast", ["s"], ["sf"], to=TensorProto.FLOAT),
    ]
    graph = helper.make_graph(
        nodes, "shape_fold",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, None)],
        [helper.make_tensor_value_info("sf", TensorProto.FLOAT, None)],
        [],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    if declared_dims is not None:
        tt = model.graph.input[0].type.tensor_type
        del tt.shape.dim[:]
        for d in declared_dims:
            tt.shape.dim.add().dim_value = d
        tt.shape.SetInParent()
    return model


def _folded_shape_vector(model):
    """Fold and return the shape vector the Shape chain evaluated to.

    The fold materializes ``s`` and then its single consumer ``sf``; deadend
    elimination can drop ``s`` once ``sf`` is a constant, so accept either. A
    chain that was NOT folded leaves neither name as an initializer.
    """
    out = ConstantFolder.fold(model, verbose=False)
    inits = {init.name: numpy_helper.to_array(init) for init in out.graph.initializer}
    for name in ("s", "sf"):
        if name in inits:
            return inits[name]
    return None


def test_shape_of_known_rank_folds_to_the_shape_vector():
    """Control: the gate still folds when every dim is a known constant."""
    arr = _folded_shape_vector(_model(declared_dims=[1, 4, 128]))
    assert arr is not None, "Shape(x) with a fully-known shape did not fold"
    assert arr.tolist() == [1, 4, 128]


def test_shape_of_unknown_rank_does_not_fold_to_an_empty_array():
    """The regression: rank-unknown must not fold to `[]`."""
    arr = _folded_shape_vector(_model(declared_dims=None))
    assert arr is None or arr.size != 0, (
        "Shape(x) with an UNKNOWN RANK folded to an empty array; the downstream "
        "chain folds to empty and the consumer records a 0 dim, which the "
        "runtime treats as a genuinely empty axis"
    )


def test_shape_known_distinguishes_unknown_rank_from_rank_zero():
    unknown = _model(declared_dims=None)
    assert ConstantFolder._shape_known(unknown, "x") is False, (
        "an omitted shape field means rank unknown, not a known shape")

    rank_zero = _model(declared_dims=[])
    assert ConstantFolder._shape_known(rank_zero, "x") is True, (
        "a declared shape with 0 dims is a rank-0 scalar; Shape() of it is []")


def test_shape_of_rank_zero_scalar_still_folds_to_empty():
    """Rank-0 is the ONLY state that may fold to `[]`."""
    arr = _folded_shape_vector(_model(declared_dims=[]))
    assert arr is not None, "Shape() of a declared rank-0 scalar did not fold"
    assert arr.size == 0
