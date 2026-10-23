"""Standalone test for the Conv weight-image layout pass.

A Conv weight is uploaded as an image and the image conv shader addresses it
through two push constants, `transpose` and `pack`. The runtime used to decide
both at load time from a name -> op_type map plus the weight's geometry -- a
plain last-writer-wins map (a shared weight took whichever consumer was walked
last) with an unguarded transpose (a 5-D Conv3d weight got transpose=1 and
Tensor::getGPUShape then invented a batch/chan). ConvWeightLayoutPass moves that
decision into the converter and writes it on the node as weight_transpose /
weight_pack.

The rule is geometry-only: rank 4 -> both 1; any other rank -> both 0. Covered
here: rank-4 1x1 and 3x3, rank-5, a non-constant weight (skipped), a Conv that
another pass folds away first (annotated before the fold, so nothing survives to
check), idempotence under the fixpoint loop, a shared weight (the case the
runtime's map could not express), and the fail-closed validation.

Run with the venv interpreter (it is the one that has flatbuffers):
    /home/gwm/.venv/bin/python -m pytest test_conv_weight_layout.py
"""
import numpy as np
from onnx import numpy_helper

from onnx2vkop.dag import DAGBasedModel, Node
from onnx2vkop.optimizer import ConvWeightLayoutPass, FusionOptimizer

FP16 = 10  # TensorProto.FLOAT16; the DAG stores raw ONNX elem_type ints


def td(name, dims=None, dtype=FP16):
    d = {"name": name, "dtype": dtype}
    if dims is not None:
        d["shape"] = list(dims)
    return d


def build_conv(weight_shape=(16, 3, 3, 3), kernel=(3, 3), in_shape=(1, 3, 8, 8),
               weight_is_initializer=True, share_weight=False, attrs=None):
    """Conv(X, W) -> Reshape -> y, with y a graph output.

    Defaults are a 3x3/stride-1 conv on an 8x8 input -- kernel != stride and
    kernel != input extent, so fold_single_patch_conv leaves it alone. The
    consumer is a Reshape (not an Identity, which eliminate_identity would
    remove, orphaning the graph output the liveness pass roots on).
    """
    m = DAGBasedModel()
    batch = in_shape[0]
    out_shape = [batch, weight_shape[0]] + [
        (in_shape[2 + i] - kernel[i]) + 1 for i in range(len(kernel))
    ]
    a = {"kernel_shape": list(kernel), "strides": [1] * len(kernel),
         "dilations": [1] * len(kernel), "pads": [0] * (2 * len(kernel)),
         "group": 1}
    a.update(attrs or {})
    conv = Node("Conv", "conv", a, [td("X", in_shape), td("W", weight_shape)],
                [td("conv_out", out_shape)])
    m.nodes[conv.name] = conv

    oc = weight_shape[0]
    m.initializers["rs"] = numpy_helper.from_array(
        np.array([-1, oc], dtype=np.int64), "rs")
    rs = Node("Reshape", "rs_node", {},
              [td("conv_out", out_shape), {"name": "rs", "shape": [2], "dtype": 7}],
              [td("y", [-1, oc])])
    m.nodes[rs.name] = rs
    m.outputs = [td("y", [-1, oc])]

    if weight_is_initializer:
        m.initializers["W"] = numpy_helper.from_array(
            np.zeros(weight_shape, dtype=np.float16), "W")
    if share_weight:
        other = Node("Conv", "conv2", dict(a),
                     [td("X2", in_shape), td("W", weight_shape)],
                     [td("conv2_out", out_shape)])
        m.nodes[other.name] = other
        m.outputs.append(td("conv2_out", out_shape))

    m.build_dependencies()
    return m


def run_fusion(m):
    opt = FusionOptimizer.create_default_optimizer(max_rounds=3)
    opt.optimize(m, verbose=False)
    return m


def convs(m):
    return [n for n in m.nodes.values() if n.op_type in ("Conv", "Conv2d")]


def layout(node):
    return (node.attributes.get("weight_transpose"),
            node.attributes.get("weight_pack"))


def test_rank4_3x3():
    m = run_fusion(build_conv())
    cs = convs(m)
    assert len(cs) == 1, [n.op_type for n in m.nodes.values()]
    assert layout(cs[0]) == (1, 1), cs[0].attributes
    print("  PASS (rank-4 3x3 Conv -> transpose=1 pack=1)")


def test_rank4_1x1():
    m = run_fusion(build_conv(weight_shape=(16, 3, 1, 1), kernel=(1, 1)))
    cs = convs(m)
    assert len(cs) == 1, [n.op_type for n in m.nodes.values()]
    assert layout(cs[0]) == (1, 1), cs[0].attributes
    print("  PASS (rank-4 1x1 Conv -> transpose=1 pack=1)")


def test_rank5_conv3d():
    """A 5-D weight has no folded addressing and no correct transpose.

    strides=[1,1,1] keeps it out of the patch fold (kernel != stride) so the
    Conv reaches the layout pass intact.
    """
    m = run_fusion(build_conv(weight_shape=(16, 3, 2, 3, 3), kernel=(2, 3, 3),
                              in_shape=(1, 3, 8, 8, 8),
                              attrs={"strides": [1, 1, 1]}))
    cs = convs(m)
    assert len(cs) == 1, [n.op_type for n in m.nodes.values()]
    assert layout(cs[0]) == (0, 0), cs[0].attributes
    print("  PASS (rank-5 Conv weight -> transpose=0 pack=0)")


def test_non_constant_weight_untouched():
    m = run_fusion(build_conv(weight_is_initializer=False))
    cs = convs(m)
    assert len(cs) == 1, [n.op_type for n in m.nodes.values()]
    assert "weight_transpose" not in cs[0].attributes
    assert "weight_pack" not in cs[0].attributes
    print("  PASS (non-constant weight left unannotated)")


def test_folded_conv_leaves_no_attribute():
    """fold_single_patch_conv (priority 73) runs above the layout pass (71), so
    a patch_embed Conv is gone -- and a MatMul carries no image flag."""
    m = run_fusion(build_conv(weight_shape=(64, 3, 16, 16), kernel=(16, 16),
                              in_shape=(8, 3, 16, 16),
                              attrs={"strides": [16, 16]}))
    assert not convs(m), [n.op_type for n in m.nodes.values()]
    mm = [n for n in m.nodes.values() if n.op_type == "MatMul"]
    assert len(mm) == 1, [n.op_type for n in m.nodes.values()]
    assert "weight_transpose" not in mm[0].attributes
    assert "weight_pack" not in mm[0].attributes
    print("  PASS (patch Conv folded to MatMul carries no image flag)")


def test_idempotent():
    """The fixpoint loop must converge: the second apply changes nothing."""
    m = build_conv()
    p = ConvWeightLayoutPass()
    assert p.apply(m) == 1, "first apply should annotate the one Conv"
    conv = convs(m)[0]
    assert layout(conv) == (1, 1), conv.attributes
    assert p.apply(m) == 0, "second apply should be a no-op"
    assert layout(conv) == (1, 1), conv.attributes
    print("  PASS (apply is idempotent)")


def test_constant_weight_node_materialized_then_annotated():
    """A Conv whose weight is a Constant *node*, not an initializer.

    prune_and_materialize (priority 0) turns the Constant into an initializer,
    but the layout pass runs before it (71), so the rank has to come off the
    Constant node itself. Otherwise the annotation would depend on a later
    round -- and VKOP_MAX_FUSE_ROUNDS=1 would leave the Conv bare for the
    fail-closed validation to reject.
    """
    m = build_conv(weight_is_initializer=False)
    weight = np.zeros((16, 3, 3, 3), dtype=np.float16)
    cst = Node("Constant", "w_const", {"value": weight}, [],
               [td("W", (16, 3, 3, 3))])
    m.nodes[cst.name] = cst
    m.build_dependencies()

    m = run_fusion(m)
    cs = convs(m)
    assert "W" in m.initializers, "Constant weight was not materialized"
    assert len(cs) == 1, [n.op_type for n in m.nodes.values()]
    assert layout(cs[0]) == (1, 1), cs[0].attributes
    print("  PASS (Constant weight materialized, annotated on the next round)")


def test_shared_weight_annotated_identically():
    """The case the runtime's last-writer-wins op_type map could not express."""
    m = run_fusion(build_conv(share_weight=True))
    cs = convs(m)
    assert len(cs) == 2, [n.op_type for n in m.nodes.values()]
    assert {layout(c) for c in cs} == {(1, 1)}, [c.attributes for c in cs]
    print("  PASS (shared weight gets identical values on both Convs)")


def test_validation_rejects_wrong_layout():
    """An attribute contradicting the weight's rank fails the conversion rather
    than shipping a model the image conv cannot address."""
    from onnx2vkop.converter import ModelConverter, UnsupportedOpError

    m = build_conv(weight_shape=(16, 3, 2, 3, 3), kernel=(2, 3, 3),
                   in_shape=(1, 3, 8, 8, 8))
    conv = convs(m)[0]
    conv.attributes["weight_transpose"] = 1
    conv.attributes["weight_pack"] = 1
    try:
        ModelConverter._validate_conv_weight_layout(m)
    except UnsupportedOpError:
        print("  PASS (rank-5 pack=1 rejected)")
        return
    raise AssertionError("rank-5 pack=1 should have failed validation")


def test_validation_rejects_missing_attribute():
    from onnx2vkop.converter import ModelConverter, UnsupportedOpError

    m = build_conv()
    try:
        ModelConverter._validate_conv_weight_layout(m)
    except UnsupportedOpError:
        print("  PASS (unannotated constant-weight Conv rejected)")
        return
    raise AssertionError("a missing attribute should have failed validation")


def test_validation_skips_ablated_pass(monkeypatch):
    """With the pass ablated there is nothing to check -- the runtime fallback
    (rank-gated geometry) is the only layout."""
    from onnx2vkop.converter import ModelConverter

    monkeypatch.setenv("VKOP_NO_FUSIONS", "annotate_conv_weight_layout")
    ModelConverter._validate_conv_weight_layout(build_conv())  # must not raise
    print("  PASS (validation skipped when the pass is ablated)")


if __name__ == "__main__":
    test_rank4_3x3()
    test_rank4_1x1()
    test_rank5_conv3d()
    test_non_constant_weight_untouched()
    test_folded_conv_leaves_no_attribute()
    test_idempotent()
    test_constant_weight_node_materialized_then_annotated()
    test_shared_weight_annotated_identically()
    test_validation_rejects_wrong_layout()
    test_validation_rejects_missing_attribute()
    print("all conv weight-layout tests passed")
