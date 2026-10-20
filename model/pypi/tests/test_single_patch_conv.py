"""Standalone test for the whole-input Conv -> MatMul fold.

A Conv whose kernel equals both its stride and the input's whole spatial extent
(the ViT/CLIP patch_embed) computes one output pixel per batch item with the
entire input as its receptive field, i.e. a plain GEMM. The pass rewrites it as
Reshape + MatMul(transB) (+ Add for the bias) so it runs on the coopmat MatMul
instead of the naive direct convolution (measured 75-127 ms -> 1.7 ms on the
Qwen3-VL vision tower).

Positive: folded with a bias and without one, weight reshaped to [OC, K].
Negative: every shape that would need real im2col (overlap, padding, dilation,
groups, unknown input shape), a non-constant weight, a shared weight, and a
consumer that would read the dropped all-ones output axes.
"""
import numpy as np
from onnx import numpy_helper

from onnx2vkop.dag import DAGBasedModel, Node
from onnx2vkop.optimizer import FusionOptimizer

FP16 = 10  # TensorProto.FLOAT16; the DAG stores raw ONNX elem_type ints


def td(name, dims=None, dtype=FP16):
    d = {"name": name, "dtype": dtype}
    if dims is not None:
        d["shape"] = list(dims)
    return d


def build(attrs=None, in_shape=(2, 3, 2, 16, 16), kernel=(2, 16, 16),
          weight_shape=(1024, 3, 2, 16, 16), bias=True,
          weight_is_initializer=True, share_weight=False,
          consumer="Reshape", out_shape=None, weight_name=None):
    """Conv(X) -> [Reshape] and nothing else. X is a graph input."""
    m = DAGBasedModel()
    weight_name = weight_name or "W"
    X = td("X", in_shape)
    if out_shape is None:
        batch = in_shape[0] if in_shape else 2  # in_shape=[] models "unknown"
        out_shape = [batch, weight_shape[0]] + [1] * len(kernel)
    out = td("conv_out", out_shape)
    ins = [X, td(weight_name, weight_shape)]
    if bias:
        ins.append(td("B", [weight_shape[0]]))
    a = {"kernel_shape": list(kernel), "strides": list(kernel),
         "dilations": [1] * len(kernel), "pads": [0] * (2 * len(kernel)),
         "group": 1}
    a.update(attrs or {})
    conv = Node("Conv", "conv", a, ins, [out])
    m.nodes[conv.name] = conv

    if consumer == "Reshape":
        m.initializers["rs"] = numpy_helper.from_array(
            np.array([-1, weight_shape[0]], dtype=np.int64), "rs")
        rs = Node("Reshape", "rs_node", {},
                  [out, {"name": "rs", "shape": [2], "dtype": 7}],
                  [td("y", [-1, weight_shape[0]])])
        m.nodes[rs.name] = rs
        m.outputs = [td("y", [-1, weight_shape[0]])]
    elif consumer == "Add":  # a rank-sensitive consumer: broadcasts over axes
        rs = Node("Add", "add_node", {},
                  [out, td("other", out_shape)],
                  [td("y", out_shape)])
        m.nodes[rs.name] = rs
        m.outputs = [td("y", out_shape)]

    if weight_is_initializer:
        m.initializers[weight_name] = numpy_helper.from_array(
            np.zeros(weight_shape, dtype=np.float16), weight_name)
    if share_weight:
        other = Node("Conv", "conv2",
                     {"kernel_shape": list(kernel), "strides": list(kernel),
                      "dilations": [1] * len(kernel),
                      "pads": [0] * (2 * len(kernel)), "group": 1},
                     [td("X2", in_shape), td(weight_name, weight_shape)],
                     [td("conv2_out", out_shape)])
        m.nodes[other.name] = other
        # Live, or prune_and_materialize (priority 0) would delete the second
        # Conv before the next round -- with one consumer left the fold is
        # legal and the pass is right to take it.
        m.outputs.append(td("conv2_out", out_shape))

    m.build_dependencies()
    return m


def run_fusion(m):
    opt = FusionOptimizer.create_default_optimizer(max_rounds=3)
    opt.optimize(m, verbose=False)
    return m


def check_folded(m, oc=1024, k=1536, want_bias=True):
    optypes = sorted(n.op_type for n in m.nodes.values())
    assert "Conv" not in optypes, f"Conv survived the fold: {optypes}"
    mm = [n for n in m.nodes.values() if n.op_type == "MatMul"]
    assert len(mm) == 1, f"expected 1 MatMul, got {optypes}"
    mm = mm[0]
    assert mm.attributes.get("transB") == 1, mm.attributes
    wname = mm.inputs[1]["name"]
    assert list(m.initializers[wname].dims) == [oc, k], \
        f"weight not reshaped to [{oc}, {k}]: {m.initializers[wname].dims}"
    assert mm.inputs[1]["shape"] == [oc, k], mm.inputs[1]
    # A goes through a flattening Reshape before the MatMul.
    flat = mm.inputs[0]["name"]
    producer = [n for n in m.nodes.values() if n.outputs[0]["name"] == flat]
    assert producer and producer[0].op_type == "Reshape", f"A producer: {producer}"
    assert producer[0].inputs[0]["name"] == "X", producer[0].inputs
    adds = [n for n in m.nodes.values() if n.op_type == "Add"]
    assert bool(adds) == want_bias, f"bias Add: {optypes}"
    if want_bias:
        assert adds[0].inputs[1]["name"] == "B", adds[0].inputs
    return mm


def test_patch_embed_folds():
    m = run_fusion(build())
    check_folded(m)
    # The tensor the consumer sees is rank 2 now, and its declaration was
    # updated -- including the graph-output declaration.
    rs = [n for n in m.nodes.values() if n.op_type == "Reshape" and n.name == "rs_node"][0]
    assert rs.inputs[0]["shape"] == [2, 1024], rs.inputs[0]
    assert m.outputs[0]["shape"] == [-1, 1024], m.outputs
    print("  PASS (5-D patch_embed fold, transB, weight reshape, bias Add)")


def test_no_bias_folds():
    m = run_fusion(build(bias=False))
    optypes = sorted(n.op_type for n in m.nodes.values())
    assert "Add" not in optypes, optypes
    check_folded(m, want_bias=False)
    print("  PASS (bias-less Conv folds without an Add)")


def test_2d_conv_folds():
    """The same rule on a 2-D conv: kernel == stride == input HxW."""
    m = run_fusion(build(in_shape=(8, 3, 16, 16), kernel=(16, 16),
                         weight_shape=(64, 3, 16, 16), out_shape=[8, 64, 1, 1]))
    check_folded(m, oc=64, k=768)
    print("  PASS (rank-4 patchify Conv folds)")


def test_overlapping_stride_not_folded():
    m = run_fusion(build(attrs={"strides": [1, 1, 1]}))
    assert any(n.op_type == "Conv" for n in m.nodes.values())
    print("  PASS (stride != kernel left alone)")


def test_padding_not_folded():
    m = run_fusion(build(attrs={"pads": [0, 0, 1, 1, 1, 1]}))
    assert any(n.op_type == "Conv" for n in m.nodes.values())
    print("  PASS (padded Conv left alone)")


def test_dilation_not_folded():
    m = run_fusion(build(attrs={"dilations": [1, 2, 2]}))
    assert any(n.op_type == "Conv" for n in m.nodes.values())
    print("  PASS (dilated Conv left alone)")


def test_groups_not_folded():
    m = run_fusion(build(attrs={"group": 3}, weight_shape=(1024, 1, 2, 16, 16)))
    assert any(n.op_type == "Conv" for n in m.nodes.values())
    print("  PASS (grouped Conv left alone)")


def test_multi_window_not_folded():
    """kernel == stride but the input holds several windows -> real im2col."""
    m = run_fusion(build(in_shape=(2, 3, 2, 32, 16), out_shape=[2, 1024, 1, 2, 1]))
    assert any(n.op_type == "Conv" for n in m.nodes.values())
    print("  PASS (input larger than one window left alone)")


def test_unknown_input_shape_not_folded():
    """No shape inference result: cannot prove the window covers the input."""
    m = run_fusion(build(in_shape=[]))
    assert any(n.op_type == "Conv" for n in m.nodes.values())
    print("  PASS (unknown input shape left alone)")


def test_non_constant_weight_not_folded():
    m = run_fusion(build(weight_is_initializer=False))
    assert any(n.op_type == "Conv" for n in m.nodes.values())
    print("  PASS (non-constant weight left alone)")


def test_shared_weight_not_folded():
    m = run_fusion(build(share_weight=True))
    assert sum(1 for n in m.nodes.values() if n.op_type == "Conv") == 2
    print("  PASS (weight shared with another Conv left alone)")


def test_rank_sensitive_consumer_not_folded():
    """Add would broadcast over the all-ones axes the fold drops."""
    m = run_fusion(build(consumer="Add"))
    assert any(n.op_type == "Conv" for n in m.nodes.values())
    print("  PASS (non-Reshape consumer left alone)")
