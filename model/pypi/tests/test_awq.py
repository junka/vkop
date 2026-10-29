"""AWQ: the channel rescale, the 1/s fold, and the gate it shares with -q.

AWQ does not change the quantization grid -- it changes the weight. Every input
channel (the K axis of a MatMul's B) is multiplied by s, and 1/s is folded into
whatever computes that MatMul's A:

    W' = W * diag(s)     A' = diag(1/s) * A     W'*A' = W*A   (exact)

so the product the model computes is unchanged while the quantization error
moves off the channels whose activations are large.

What these cases pin:

  * the scale goes where AWQ says it should -- down for a channel whose weights
    are large relative to the rest, in proportion to w_amax^-alpha, normalised
    so the tensor's geometric mean is 1;
  * the fold is *exact*: scaling a norm's per-channel weight by 1/s and the
    MatMul weight by s leaves the product equal;
  * a weight whose 1/s has nowhere to go is left alone rather than scaled
    silently (the one failure mode that would produce a wrong model);
  * the eligibility gate is the quantizer's own, so AWQ never scales something
    -q would then refuse.
"""
import contextlib
import io

import numpy
import pytest
from onnx import TensorProto, numpy_helper

from onnx2vkop import awq
from onnx2vkop.dag import DAGBasedModel, Node
from onnx2vkop.optimizer import Quantizer


def tref(name, shape=(-1,)):
    return {"name": name, "shape": list(shape)}


def norm_graph(k=8, n=8, norm="RMSNorm", w=None):
    """MatMul(x <- norm(norm_w), W): the shape every transformer block has."""
    m = DAGBasedModel()
    if w is None:
        # A common magnitude across every channel: w_amax is then the same for
        # all of them, so s is uniform and W' == k*W with one scalar k. That
        # makes the fold's contract -- W scaled by s, the norm's weight by 1/s
        # -- checkable as a plain scalar identity instead of as a per-channel
        # comparison, which only pins the arithmetic and nothing else.
        w = numpy.ones((k, n), dtype=numpy.float32)
    m.initializers["W"] = numpy_helper.from_array(
        numpy.asarray(w, dtype=numpy.float32), "W")
    m.initializers["norm_w"] = numpy_helper.from_array(
        numpy.ones(k, dtype=numpy.float32), "norm_w")
    m.nodes["norm"] = Node(norm, "norm", {"eps": 1e-6},
                           [tref("x"), tref("norm_w", (k,))],
                           [tref("h", (1, -1, k))])
    m.nodes["mm"] = Node("MatMul", "mm", {},
                         [tref("h", (1, -1, k)), tref("W", (k, n))],
                         [tref("y", (1, -1, n))])
    m.inputs = [tref("x", (1, -1, k))]
    return m


def run_awq(m, group_size=8, fmt="int4", **kw):
    with contextlib.redirect_stdout(io.StringIO()) as buf:
        scaled = awq.apply(m, group_size, fmt=fmt, **kw)
    return scaled, buf.getvalue()


# --------------------------------------------------------------------------
# the scale
# --------------------------------------------------------------------------

def test_scale_follows_the_weight_magnitudes():
    """s ∝ w_amax^-alpha: a loud input channel is scaled DOWN, quiet ones UP."""
    k, n = 64, 8
    w = numpy.ones((k, n), dtype=numpy.float32)
    w[0] *= 100.0
    s = awq.solve_scales(w, alpha=0.5)

    assert s.shape == (k,)
    assert s[0] < 0.2, s[0]          # 100^-0.5 = 0.1 before normalization
    assert numpy.allclose(s[1:], s[1:][0])
    assert s[1] > 1.0
    # Geometric mean is 1: the transform cannot inflate or deflate the tensor
    # as a whole, only redistribute.
    assert numpy.isclose(numpy.exp(numpy.mean(numpy.log(s))), 1.0, rtol=1e-5)


def test_scale_is_monotone_in_the_weight_magnitude():
    k, n = 32, 4
    w = numpy.ones((k, n), dtype=numpy.float32) * numpy.linspace(
        1.0, 10.0, k)[:, None]
    s = awq.solve_scales(w, alpha=0.5)
    assert numpy.all(numpy.diff(s) < 0)


def test_all_zero_weight_yields_finite_scales():
    """A degenerate row must not produce inf/NaN and poison the whole table."""
    k, n = 16, 8
    w = numpy.zeros((k, n), dtype=numpy.float32)
    w[3] = 1.0
    s = awq.solve_scales(w, alpha=0.5)
    assert numpy.all(numpy.isfinite(s))
    assert numpy.all(s > 0)


def test_a_flat_activation_profile_falls_back_to_the_weight_only_form():
    """Constants carry no relative information, so stats must not be searched."""
    k, n = 16, 4
    w = numpy.random.RandomState(0).rand(k, n).astype(numpy.float32) + 1.0
    flat = numpy.full(k, 3.0, dtype=numpy.float32)
    searched = awq.solve_scales(w, flat, alpha=0.5, n_grid=5)
    proxy = awq.solve_scales(w, None, alpha=0.5)
    numpy.testing.assert_allclose(searched, proxy, rtol=1e-6)


def test_stats_with_the_wrong_length_is_an_error():
    with pytest.raises(ValueError):
        awq.solve_scales(numpy.ones((8, 4), numpy.float32),
                         numpy.ones(7, numpy.float32))


# --------------------------------------------------------------------------
# the fold
# --------------------------------------------------------------------------

def test_the_fold_is_exact_and_lands_on_the_norm_scale():
    """W becomes W*s, the norm's weight becomes norm_w/s, the product is equal."""
    m = norm_graph()
    w0 = numpy_helper.to_array(m.initializers["W"]).copy()
    nw0 = numpy_helper.to_array(m.initializers["norm_w"]).copy()

    scaled, _ = run_awq(m)

    assert scaled == {"W"}, scaled
    w1 = numpy_helper.to_array(m.initializers["W"])
    nw1 = numpy_helper.to_array(m.initializers["norm_w"])

    # With every channel the same magnitude, the solved s is uniform (=
    # normalized w_amax^-alpha), so W' is W times one scalar and the norm's
    # weight is that scalar's reciprocal. Checking the scalar is checking the
    # fold: the norm output is per-channel-normalized and then multiplied by
    # this weight, so scaling it by 1/s scales the MatMul's A by 1/s exactly.
    ratio = w1 / w0
    assert numpy.allclose(ratio, ratio.flat[0], rtol=1e-6), ratio
    k = float(ratio.flat[0])
    numpy.testing.assert_allclose(w1, w0 * k, rtol=1e-6)
    numpy.testing.assert_allclose(nw1, nw0 * (1.0 / k), rtol=1e-6)

    # And the product itself: (x·k)·W' == x·W, which is the MatMul's forward
    # direction (A is [M, K], B is [K, N]).
    x = numpy.random.RandomState(1).rand(3, w0.shape[0]).astype(numpy.float32)
    numpy.testing.assert_allclose((x * k) @ w1, x @ w0,
                                  rtol=1e-5, atol=1e-5)


def test_layernorm_is_folded_the_same_way():
    m = norm_graph(norm="LayerNormalization")
    scaled, _ = run_awq(m)
    assert scaled == {"W"}


def test_a_non_norm_producer_leaves_the_weight_alone():
    """A' = diag(1/s)·A only holds if the producer is per-channel-normalising.

    An Attention (or any other op) output cannot be rescaled that way, so the
    weight must stay exactly as it was -- the alternative is a model whose
    values are wrong in a way nothing downstream can detect.
    """
    m = DAGBasedModel()
    m.initializers["W"] = numpy_helper.from_array(
        numpy.ones((8, 8), dtype=numpy.float32), "W")
    m.initializers["q"] = numpy_helper.from_array(
        numpy.ones((8, 8), dtype=numpy.float32), "q")
    m.nodes["attn"] = Node("Attention", "attn", {},
                           [tref("q", (1, 2, 8)), tref("q", (1, 2, 8)),
                            tref("q", (1, 2, 8))],
                           [tref("h", (1, 2, 8))])
    m.nodes["mm"] = Node("MatMul", "mm", {},
                         [tref("h", (1, 2, 8)), tref("W", (8, 8))],
                         [tref("y")])
    before = numpy_helper.to_array(m.initializers["W"]).copy()

    scaled, out = run_awq(m)

    assert scaled == set()
    numpy.testing.assert_array_equal(
        numpy_helper.to_array(m.initializers["W"]), before)
    assert "'W'" in out and "Attention" in out


def test_a_norm_of_the_wrong_width_is_not_folded():
    """norm_w is 4 long, the MatMul's K is 8: 1/s has no place to land."""
    m = norm_graph(k=8)
    m.initializers["norm_w"] = numpy_helper.from_array(
        numpy.ones(4, dtype=numpy.float32), "norm_w")
    before = numpy_helper.to_array(m.initializers["W"]).copy()

    scaled, out = run_awq(m)

    assert scaled == set()
    numpy.testing.assert_array_equal(
        numpy_helper.to_array(m.initializers["W"]), before)
    assert "channels" in out


def test_a_matmul_on_a_graph_input_gets_a_constant_mul():
    """No norm above it, but the input is the graph's own: a Mul carries 1/s."""
    m = DAGBasedModel()
    m.initializers["W"] = numpy_helper.from_array(
        numpy.ones((8, 8), dtype=numpy.float32), "W")
    m.nodes["mm"] = Node("MatMul", "mm", {},
                         [tref("x", (1, 8)), tref("W", (8, 8))],
                         [tref("y", (1, 8))])

    scaled, out = run_awq(m)

    assert scaled == {"W"}
    assert "graph input 'x'" in out
    mul = m.nodes["x_awq_scale"]
    assert mul.op_type == "Mul"
    assert [i["name"] for i in mul.inputs] == ["x", "x_awq_inv_scale"]
    assert [i["name"] for i in m.nodes["mm"].inputs] == ["x_awq_scaled", "W"]
    inv = numpy_helper.to_array(m.initializers["x_awq_inv_scale"])
    w1 = numpy_helper.to_array(m.initializers["W"])
    numpy.testing.assert_allclose(
        numpy.ones(8, dtype=numpy.float32), w1.mean(axis=1) * inv.reshape(-1),
        rtol=1e-6)


# --------------------------------------------------------------------------
# the gate is the quantizer's
# --------------------------------------------------------------------------

def test_awq_and_the_quantizer_agree_on_every_format():
    """Whatever AWQ scales, the -q pass for that format must be willing to
    encode; a drift between the two is an unscaled s with nowhere to fold."""
    m = norm_graph()
    for fmt in ("int8", "fp8e4m3", "fp8e5m2", "int4", "nf4", "nvfp4"):
        assert awq._eligible(m, "W", [m.nodes["mm"]], fmt, 8)["eligible"], fmt


def test_a_transposed_weight_is_refused_by_both():
    m = DAGBasedModel()
    m.initializers["W"] = numpy_helper.from_array(
        numpy.ones((8, 8), dtype=numpy.float32), "W")
    m.nodes["mm"] = Node("MatMul", "mm", {"transB": 1},
                         [tref("h"), tref("W", (8, 8))], [tref("y")])
    assert not Quantizer._match_weight_only(
        "W", m.initializers["W"], [m.nodes["mm"]], "int4", 8)["eligible"]
    assert awq._eligible(m, "W", [m.nodes["mm"]], "int4", 8)["eligible"] is False


# --------------------------------------------------------------------------
# end to end through the quantizer entry point
# --------------------------------------------------------------------------

def test_awq_then_int4_quantizes_the_scaled_weight():
    m = norm_graph()
    with contextlib.redirect_stdout(io.StringIO()):
        scaled = Quantizer.quantize_to_awq_weight_only(m, 8, "int4")
    assert scaled == {"W"}
    assert m.initializers["W"].data_type == TensorProto.INT4
    assert m.initializers["W_scale"].data_type == TensorProto.FLOAT
    # The MatMul now reads [A, W, scale].
    assert [i["name"] for i in m.nodes["mm"].inputs] == ["h", "W", "W_scale"]
