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


# --------------------------------------------------------------------------
# the fold is exact on the graph's own output, for every layout and K != N
# --------------------------------------------------------------------------
#
# Everything above used K = N = 8 and a uniform s, where scaling the weight's
# rows and scaling its columns are the same thing. The cases below build
# K != N graphs with an s that varies per channel, evaluate the graph in numpy
# before and after `awq.apply`, and require the same output: W' = W*diag(s)
# along the INPUT-channel axis ([K, N] rows for MatMul, [N, K] columns for a
# transB=1 weight) and A' = A*diag(1/s) cancel exactly, in any layout.

def _arr(init):
    if init.data_type == TensorProto.BFLOAT16:
        return awq._bf16_init_to_f32(init)
    return numpy.asarray(numpy_helper.to_array(init), dtype=numpy.float32)


def evaluate(m, feeds):
    """Every tensor of the DAG, in float64, from a topological sweep."""
    env = {k: numpy.asarray(v, dtype=numpy.float64) for k, v in feeds.items()}
    for name, init in m.initializers.items():
        env[name] = _arr(init).astype(numpy.float64)
    pending = list(m.nodes.values())
    while pending:
        progressed = False
        for node in list(pending):
            ins = [i["name"] for i in node.inputs]
            if not all(i in env for i in ins if i):
                continue
            v = [env[i] if i else None for i in ins]
            a = node.attributes
            op = node.op_type
            if op == "RMSNorm":
                x = v[0]
                r = x / numpy.sqrt(numpy.mean(x * x, axis=-1, keepdims=True)
                                   + a.get("eps", 1e-6))
                out = r * v[1]
            elif op in ("LayerNormalization", "LayerNorm"):
                x = v[0]
                mu = x.mean(axis=-1, keepdims=True)
                var = ((x - mu) ** 2).mean(axis=-1, keepdims=True)
                out = (x - mu) / numpy.sqrt(var + a.get("epsilon", 1e-5)) \
                    * v[1]
                if len(v) > 2 and v[2] is not None:
                    out = out + v[2]
            elif op in ("MatMul", "Gemm"):
                x, w = v[0], v[1]
                if int(a.get("transA", 0) or 0):
                    x = numpy.swapaxes(x, -1, -2)
                if int(a.get("transB", 0) or 0):
                    w = w.T
                # errstate: Accelerate-backed numpy raises spurious
                # divide/overflow flags from matmul on finite inputs.
                with numpy.errstate(all="ignore"):
                    out = float(a.get("alpha", 1.0)) * (x @ w)
                assert numpy.all(numpy.isfinite(out)), node.name
                if op == "Gemm" and len(v) > 2 and v[2] is not None:
                    out = out + float(a.get("beta", 1.0)) * v[2]
            elif op == "Mul":
                out = v[0] * v[1]
            elif op == "Add":
                out = v[0] + v[1]
            elif op == "Identity":
                out = v[0]
            else:
                raise NotImplementedError(op)
            env[node.outputs[0]["name"]] = out
            pending.remove(node)
            progressed = True
        assert progressed, [n.name for n in pending]
    return env


def rel_l2(got, want):
    return float(numpy.linalg.norm(got - want) / numpy.linalg.norm(want))


def _skewed(rng, k, n):
    """A weight whose input channels differ a lot in magnitude, so s does."""
    w = rng.standard_normal((k, n)).astype(numpy.float32)
    return w * numpy.exp(rng.uniform(-2.0, 2.0, size=(k, 1))).astype(
        numpy.float32)


def _store(values, dtype, name):
    if dtype == "bf16":
        return awq._f32_to_bf16_init(values, name)
    return numpy_helper.from_array(
        numpy.asarray(values, numpy.float16 if dtype == "fp16"
                      else numpy.float32), name)


def norm_mm_graph(k, n, *, norm="LayerNormalization", op="MatMul", transb=0,
                  w_dtype="fp32", norm_dtype="fp32", bias=True, seed=0):
    """y = MatMul(norm(x; gamma, beta), W), K != N, every value non-uniform."""
    rng = numpy.random.RandomState(seed)
    w_kn = _skewed(rng, k, n)
    w = w_kn.T.copy() if transb else w_kn
    m = DAGBasedModel()
    m.initializers["W"] = _store(w, w_dtype, "W")
    m.initializers["gamma"] = _store(rng.uniform(0.5, 2.0, k), norm_dtype,
                                     "gamma")
    ins = [tref("x", (1, 3, k)), tref("gamma", (k,))]
    if bias and norm != "RMSNorm":
        m.initializers["beta"] = _store(rng.uniform(-1.0, 1.0, k), norm_dtype,
                                        "beta")
        ins.append(tref("beta", (k,)))
    m.nodes["norm"] = Node(norm, "norm", {}, ins, [tref("h", (1, 3, k))])
    attrs = {"transB": 1} if transb else {}
    m.nodes["mm"] = Node(op, "mm", attrs,
                         [tref("h", (1, 3, k)), tref("W", list(w.shape))],
                         [tref("y", (1, 3, n))])
    m.inputs = [tref("x", (1, 3, k))]
    m.outputs = [tref("y", (1, 3, n))]
    return m


def _x(k, seed=7):
    return numpy.random.RandomState(seed).standard_normal((1, 3, k))


def check_output_preserved(m, fmt, group_size, tol, stats=None,
                           expect=frozenset({"W"}), out="y"):
    k = m.inputs[0]["shape"][-1]
    feeds = {"x": _x(k)}
    before = evaluate(m, feeds)[out]
    scaled, log = run_awq(m, group_size=group_size, fmt=fmt, stats=stats)
    assert scaled == set(expect), log
    after = evaluate(m, feeds)[out]
    assert rel_l2(after, before) < tol, (rel_l2(after, before), log)
    return log


@pytest.mark.parametrize("norm", ["LayerNormalization", "RMSNorm"])
@pytest.mark.parametrize("k,n", [(128, 64), (64, 128)])
def test_matmul_kn_fold_preserves_the_output(norm, k, n):
    """[K, N] MatMul weight: rows scaled by s. Crashed for K != N."""
    m = norm_mm_graph(k, n, norm=norm)
    w0 = _arr(m.initializers["W"])
    check_output_preserved(m, "int4", 64, 1e-5)
    # s really is per input channel and really varies: rows, not columns.
    ratio = _arr(m.initializers["W"]) / w0
    numpy.testing.assert_allclose(ratio, ratio[:, :1].repeat(n, 1),
                                  rtol=1e-5)
    assert numpy.ptp(ratio[:, 0]) > 0.5


def test_matmul_fold_with_activation_stats_preserves_the_output():
    """The searched-alpha path scores W*diag(s) too; it crashed for K != N."""
    k, n = 128, 64
    m = norm_mm_graph(k, n)
    act = numpy.exp(numpy.random.RandomState(3).uniform(-2, 2, k))
    check_output_preserved(m, "int4", 64, 1e-5, stats={"h": act})


def test_solve_scales_scores_rows_of_a_wide_weight():
    """[K, N] with K != N under stats: s has K entries and the search runs."""
    rng = numpy.random.RandomState(1)
    w = _skewed(rng, 64, 192)
    act = numpy.exp(rng.uniform(-2, 2, 64)).astype(numpy.float32)
    s = awq.solve_scales(w, act, alpha=0.5, n_grid=5, group_size=32,
                         calib_ratio=0.1)
    assert s.shape == (64,)
    assert numpy.all(numpy.isfinite(s)) and numpy.ptp(s) > 0


@pytest.mark.parametrize("op", ["Gemm", "MatMul"])
def test_transb_weight_fold_preserves_the_output(op):
    """A transB=1 weight is [N, K]: its COLUMNS carry s (byte formats only;
    the 4-bit gate takes [K, N] alone)."""
    k, n = 128, 64
    m = norm_mm_graph(k, n, norm="RMSNorm", op=op, transb=1)
    assert list(m.initializers["W"].dims) == [n, k]
    w0 = _arr(m.initializers["W"])
    check_output_preserved(m, "int8", 64, 1e-5)
    ratio = _arr(m.initializers["W"]) / w0
    numpy.testing.assert_allclose(ratio, ratio[:1, :].repeat(n, 0),
                                  rtol=1e-5)


def test_the_gate_reports_k_on_the_input_axis_of_a_transb_weight():
    m = norm_mm_graph(128, 64, op="Gemm", transb=1)
    gate = Quantizer._match_weight_only(
        "W", m.initializers["W"], [m.nodes["mm"]], "int8")
    assert gate["eligible"] and gate["transb"] == 1
    assert (gate["k"], gate["n"]) == (128, 64)


@pytest.mark.parametrize("w_dtype,norm_dtype,tol", [
    ("fp16", "fp16", 3e-3),
    ("fp32", "fp16", 3e-3),
    ("fp32", "bf16", 1.5e-2),
    ("fp16", "bf16", 1.5e-2),
])
def test_half_storage_fold_preserves_the_output(w_dtype, norm_dtype, tol):
    """The weight and the norm keep their storage width (rounded once), so
    the identity holds up to that width's rounding."""
    k, n = 128, 64
    m = norm_mm_graph(k, n, w_dtype=w_dtype, norm_dtype=norm_dtype)
    want = {"fp16": TensorProto.FLOAT16, "fp32": TensorProto.FLOAT,
            "bf16": TensorProto.BFLOAT16}
    check_output_preserved(m, "int4", 64, tol)
    assert m.initializers["W"].data_type == want[w_dtype]
    assert m.initializers["gamma"].data_type == want[norm_dtype]
    assert m.initializers["beta"].data_type == want[norm_dtype]


def test_layernorm_bias_is_scaled_with_its_gamma():
    """LN(x) = xhat*gamma + beta: 1/s has to reach beta too, or the bias's
    share of A meets W*s unscaled."""
    m = norm_mm_graph(96, 32, bias=True)
    b0 = _arr(m.initializers["beta"]).copy()
    g0 = _arr(m.initializers["gamma"]).copy()
    check_output_preserved(m, "int8", 0, 1e-5)
    numpy.testing.assert_allclose(_arr(m.initializers["beta"]) / b0,
                                  _arr(m.initializers["gamma"]) / g0,
                                  rtol=1e-5)


def test_graph_input_fold_preserves_the_output():
    k, n = 128, 64
    rng = numpy.random.RandomState(0)
    m = DAGBasedModel()
    m.initializers["W"] = numpy_helper.from_array(_skewed(rng, k, n), "W")
    m.nodes["mm"] = Node("MatMul", "mm", {},
                         [tref("x", (1, 3, k)), tref("W", (k, n))],
                         [tref("y", (1, 3, n))])
    m.inputs = [tref("x", (1, 3, k))]
    m.outputs = [tref("y", (1, 3, n))]
    check_output_preserved(m, "int4", 64, 1e-5)
    assert list(m.initializers["x_awq_inv_scale"].dims) == [1, 1, k]


def qkv_graph(k=128, ns=(64, 32, 32), extra_reader=None):
    """One norm feeding three projections of different widths."""
    rng = numpy.random.RandomState(0)
    m = DAGBasedModel()
    m.initializers["gamma"] = numpy_helper.from_array(
        rng.uniform(0.5, 2.0, k).astype(numpy.float32), "gamma")
    m.nodes["norm"] = Node("RMSNorm", "norm", {},
                           [tref("x", (1, 3, k)), tref("gamma", (k,))],
                           [tref("h", (1, 3, k))])
    outs = []
    for i, n in enumerate(ns):
        w = f"W{i}"
        m.initializers[w] = numpy_helper.from_array(_skewed(rng, k, n), w)
        m.nodes[f"mm{i}"] = Node("MatMul", f"mm{i}", {},
                                 [tref("h", (1, 3, k)), tref(w, (k, n))],
                                 [tref(f"y{i}", (1, 3, n))])
        outs.append(tref(f"y{i}", (1, 3, n)))
    if extra_reader == "add":
        m.nodes["res"] = Node("Add", "res", {},
                              [tref("h", (1, 3, k)), tref("x", (1, 3, k))],
                              [tref("r", (1, 3, k))])
        outs.append(tref("r", (1, 3, k)))
    m.inputs = [tref("x", (1, 3, k))]
    m.outputs = outs
    return m


def test_projections_sharing_a_norm_share_one_scale():
    """q/k/v read one norm: one s for all three, folded into gamma once.

    Per-weight solving folded 1/s_q, 1/s_k and 1/s_v all into the same gamma,
    so each projection came out divided by the other two's scales."""
    m = qkv_graph()
    feeds = {"x": _x(128)}
    before = evaluate(m, feeds)
    w0 = {w: _arr(m.initializers[w]) for w in ("W0", "W1", "W2")}
    scaled, log = run_awq(m, group_size=64)
    assert scaled == {"W0", "W1", "W2"}, log
    after = evaluate(m, feeds)
    for y in ("y0", "y1", "y2"):
        assert rel_l2(after[y], before[y]) < 1e-5, (y, log)
    s = [(_arr(m.initializers[w]) / w0[w])[:, 0] for w in ("W0", "W1", "W2")]
    numpy.testing.assert_allclose(s[0], s[1], rtol=1e-5)
    numpy.testing.assert_allclose(s[0], s[2], rtol=1e-5)


def test_a_norm_output_with_another_reader_is_left_alone():
    """A residual Add reads the norm output too: 1/s would reach it unundone."""
    m = qkv_graph(extra_reader="add")
    before = {nm: _arr(t).copy() for nm, t in m.initializers.items()}
    scaled, log = run_awq(m, group_size=64)
    assert scaled == set(), log
    assert "Add 'res'" in log
    for nm, v in before.items():
        numpy.testing.assert_array_equal(_arr(m.initializers[nm]), v)


def test_an_excluded_projection_keeps_its_siblings_unscaled():
    """Excluding one of q/k/v leaves its MatMul on the shared norm, so none of
    the three may move it."""
    m = qkv_graph()
    feeds = {"x": _x(128)}
    before = evaluate(m, feeds)
    with contextlib.redirect_stdout(io.StringIO()):
        scaled = awq.apply(m, 64, fmt="int4", exclude={"W1"})
    assert scaled == set()
    after = evaluate(m, feeds)
    for y in ("y0", "y1", "y2"):
        numpy.testing.assert_array_equal(after[y], before[y])


def _dequantized(m, name, consumer, k, n, fmt, group):
    """The float weight the kernel reads back from -q's encoded payload."""
    scale = numpy_helper.to_array(
        m.initializers[consumer.inputs[2]["name"]]).astype(numpy.float64)
    codes = numpy.asarray(numpy_helper.to_array(m.initializers[name]),
                          numpy.float64)
    if fmt == "int8":
        return codes.reshape(k, n) * scale.reshape(1, n)
    codes = codes.reshape(-1)[:k * n].reshape(k // group, group, n)
    return (codes * scale.reshape(k // group, 1, n)).reshape(k, n)


@pytest.mark.parametrize("fmt,group,tol", [("int4", 64, 0.2),
                                           ("int8", 0, 0.02)])
def test_awq_then_quantize_tracks_the_fp32_graph(fmt, group, tol):
    """Through the real quantizer: the dequantized AWQ graph stays within the
    format's error of the fp32 one (a misplaced s is O(1) off)."""
    k, n = 128, 64
    m = norm_mm_graph(k, n)
    feeds = {"x": _x(k)}
    want = evaluate(m, feeds)["y"]
    with contextlib.redirect_stdout(io.StringIO()):
        scaled = Quantizer.quantize_to_awq_weight_only(m, group or 64, fmt)
    assert scaled == {"W"}
    node = m.nodes["mm"]
    w = _dequantized(m, "W", node, k, n, fmt, group)
    m.initializers["W"] = numpy_helper.from_array(w.astype(numpy.float32), "W")
    node.inputs = node.inputs[:2]
    got = evaluate(m, feeds)["y"]
    assert rel_l2(got, want) < tol, rel_l2(got, want)
