"""The conversion-time weight-layout audit (onnx2vkop.layout_audit).

Every consumer-driven weight layout is a read attribute, so nothing moves a
byte when a kernel adapts to a weight's storage. The audit exists to make that
property checkable on a real model, which means the checks have to be exactly the
ones the kernels act on -- no more (a false hazard on a good model is worse than
no audit) and no less (a hazard that is not reported is the whole bug class).

The boundary is where these cases earn their keep: the cliff is `>= 65536` AND a
power of two, so 65536 and 131072 are hazards while 32768 (below) and 65600 (not
a power of two) are not. An off-by-one on either side silently defeats the check.

Each case builds the DAG directly rather than converting an ONNX file: the audit
is pure over (initializers, nodes), and a hand-built node keeps `transB` and the
consumer's identity under the test's control.
"""

import io
from contextlib import redirect_stdout
from types import SimpleNamespace

import numpy
import pytest
from onnx import numpy_helper

from onnx2vkop import layout_audit as la
from onnx2vkop.converter import ModelConverter, UnsupportedOpError
from onnx2vkop.dag import DAGBasedModel, Node


def tref(name):
    return {"name": name, "shape": [-1]}


def build(weights, consumers):
    """Model with the given initializers and consumer nodes.

    weights: list of (name, dims, data_type), as `bits_of` makes them. The
    audit reads only dims and data_type, so the payload is one zero scalar.
    consumers: list of (op_type, node_name, attributes, input_names).
    """
    m = DAGBasedModel()
    for name, dims, dtype in weights:
        init = numpy_helper.from_array(numpy.zeros((1,), dtype=numpy.float32), name)
        init.data_type = dtype
        init.dims[:] = list(dims)
        m.initializers[name] = init
    for op_type, node_name, attrs, ins in consumers:
        m.nodes[node_name] = Node(
            op_type, node_name, dict(attrs),
            [tref(i) for i in ins], [tref(node_name + "_out")],
        )
    return m


def bits_of(name, dims, dtype):
    return (name, dims, dtype)


FP32, FP16, INT8, INT4 = 1, 10, 3, 22


def audit(weights, consumers):
    return la.audit_layout(build(weights, consumers))


# ---------------------------------------------------------------- transB tells
# k and n swap with transB, and the audit has to read them the way the kernel
# does or every threshold it computes is off by the wrong dimension.


def test_transb0_b_stored_k_by_n():
    """B is [K, N]: reduce axis 0. n is the last dim."""
    a = audit([bits_of("W", (32, 100), FP32)],
              [("MatMul", "mm", {"transB": 0}, ["X", "W"])])
    assert [(r.k, r.n) for r in a.weights] == [(32, 100)]
    assert a.weights[0].transposed is False


def test_transb1_b_stored_n_by_k():
    """B is [N, K]: reduce axis 1. n is the FIRST dim -- a `dims[-1]` read
    would swap the two and move the row stride by 100/32."""
    a = audit([bits_of("W", (100, 32), FP32)],
              [("MatMul", "mm", {"transB": 1}, ["X", "W"])])
    assert [(r.k, r.n) for r in a.weights] == [(32, 100)]
    assert a.weights[0].transposed is True


def test_gemm_defaults_to_transb0():
    a = audit([bits_of("W", (32, 100), FP32)],
              [("Gemm", "g", {}, ["X", "W"])])
    assert (a.weights[0].k, a.weights[0].n) == (32, 100)


# ------------------------------------------------------------------ the cliff
# ops/Matmul.hpp:864: row_bytes >= 65536 && pow2(row_bytes). row_bytes is
# n * elem_bytes under either layout (ops/Matmul.hpp:862).


def test_cliff_boundary_fp16():
    """fp16: n*2. 32768 -> 65536 is the first hazard; 32768 bytes is under."""
    a = audit([bits_of("Over", (16, 32768), FP16)],  # 65536 B
              [("MatMul", "mm", {"transB": 0}, ["X", "Over"])])
    assert a.weights[0].row_bytes == 65536
    assert "pow2_row_stride" in a.weights[0].flags

    b = audit([bits_of("Under", (16, 16384), FP16)],  # 32768 B
              [("MatMul", "mm", {"transB": 0}, ["X", "Under"])])
    assert b.weights[0].row_bytes == 32768
    assert not b.weights[0].flags


def test_cliff_pow2_one_above_and_a_shifted_stride():
    """The two ways to miss the cliff: one step up, or off the power of two."""
    # fp16 n = 65536 -> 131072 B, still a power of two: a hazard.
    a = audit([bits_of("W", (16, 65536), FP16)],
              [("MatMul", "mm", {"transB": 0}, ["X", "W"])])
    assert "pow2_row_stride" in a.weights[0].flags

    # byte n = 65536 -> 65536 B (hazard), n = 65600 is not a power of two.
    b = audit([bits_of("Pow2", (8, 65536), INT8)],
              [("MatMul", "mm", {"transB": 0}, ["X", "Pow2"])])
    assert "pow2_row_stride" in b.weights[0].flags

    c = audit([bits_of("Shift", (8, 65600), INT8)],
              [("MatMul", "mm", {"transB": 0}, ["X", "Shift"])])
    assert "pow2_row_stride" not in c.weights[0].flags


def test_cliff_ignores_pow2_below_the_floor():
    """byte n = 32768 is a 32 KiB stride -- measured unaffected, not a hazard."""
    a = audit([bits_of("W", (8, 32768), INT8)],
              [("MatMul", "mm", {"transB": 0}, ["X", "W"])])
    assert a.weights[0].row_bytes == 32768
    assert not a.weights[0].flags


def test_cliff_is_transb_symmetric():
    """The stride is n * elem_bytes, and n does not move with transB. A [N,K]
    export of the same weight hits the same cliff, and no store order avoids
    it -- which is why the hazard line offers no transpose hint."""
    a = audit([bits_of("W", (32768, 1024), FP16)],
              [("MatMul", "mm", {"transB": 1}, ["X", "W"])])
    r = a.weights[0]
    assert (r.k, r.n, r.row_bytes) == (1024, 32768, 65536)
    assert "pow2_row_stride" in r.flags


def test_fp32_has_no_row_stride_hazard():
    """fp32 never reaches the tiled path, so a pow2 stride is not a hazard."""
    a = audit([bits_of("W", (64, 16384), FP32)],  # 65536 B if it were fp16
              [("MatMul", "mm", {"transB": 0}, ["X", "W"])])
    assert a.weights[0].row_bytes is None
    assert not a.weights[0].flags


# ------------------------------------------------------- fp16 tiled/coopmat k
def test_fp16_k_not_multiple_of_16():
    """ops/Matmul.hpp:867: the tile needs k % 16 == 0 to align every tile word."""
    a = audit([bits_of("Bad", (15, 128), FP16)],
              [("MatMul", "mm", {"transB": 0}, ["X", "Bad"])])
    assert "k_not_multiple_of_16" in a.weights[0].flags

    b = audit([bits_of("Good", (16, 128), FP16)],
              [("MatMul", "mm", {"transB": 0}, ["X", "Good"])])
    assert not b.weights[0].flags


def test_k_check_is_fp16_only():
    """An int8 weight with an odd k has no fp16 tile to lose, so no flag."""
    a = audit([bits_of("W", (15, 128), INT8)],
              [("MatMul", "mm", {"transB": 0}, ["X", "W"])])
    assert "k_not_multiple_of_16" not in a.weights[0].flags


# ---------------------------------------------------------- byte quad loader
def test_byte_quad_needs_n_multiple_of_4():
    """ops/Matmul.hpp:811: the quad loader reads four columns from one word."""
    a = audit([bits_of("Bad", (8, 130), INT8)],
              [("MatMul", "mm", {"transB": 0}, ["X", "Bad"])])
    assert "byte_quad_ineligible" in a.weights[0].flags

    b = audit([bits_of("Good", (8, 128), INT8)],
              [("MatMul", "mm", {"transB": 0}, ["X", "Good"])])
    assert not b.weights[0].flags


def test_byte_quad_rejects_transb1():
    """The loader reads one K row of four columns, which only exists transB=0."""
    a = audit([bits_of("W", (128, 8), INT8)],
              [("MatMul", "mm", {"transB": 1}, ["X", "W"])])
    assert "byte_quad_ineligible" in a.weights[0].flags


def test_quad_check_is_byte_only():
    """fp16 with the same dims keeps its own path (the w8 quad is byte-only)."""
    a = audit([bits_of("W", (512, 130), FP16)],
              [("MatMul", "mm", {"transB": 0}, ["X", "W"])])
    assert "byte_quad_ineligible" not in a.weights[0].flags


# ------------------------------------------------------------------- 4-bit
def test_four_bit_weight_is_row_stride_free():
    """The tile refuses 4-bit (ops/Matmul.hpp:815), so its packed row is never
    walked; a pow2-looking [_, 65536/2] shape must not be flagged."""
    a = audit([bits_of("W", (64, 32768), INT4)],  # 32768 packed bytes/row
              [("MatMul", "mm", {"transB": 0}, ["X", "W"])])
    r = a.weights[0]
    assert la.elem_bits(INT4) == 4
    assert r.row_bytes is None
    assert not r.flags


# ------------------------------------------------------------------ coverage
def test_non_constant_weight_is_counted_not_modelled():
    """A live tensor weight has no layout to audit; it must be tallied so an
    empty report is distinguishable from one that checked nothing."""
    m = DAGBasedModel()
    m.nodes["mm"] = Node("MatMul", "mm", {}, [tref("X"), tref("Y")],
                         [tref("out")])
    a = la.audit_layout(m)
    assert a.weights == []
    assert a.skipped == 1
    assert "non-constant weight" in a.summary_line()


def test_qdq_folded_weight_is_seen():
    """The 2026-10 throwaway scan keyed off initializer-named node inputs and
    missed a Q/DQ graph. PreQuantizedWeights.fold re-points the consumer at the
    payload initializer, which is what the audit walks -- so the weight is seen
    with its POST-FOLD dims and dtype."""
    m = DAGBasedModel()
    payload = numpy_helper.from_array(numpy.zeros((128, 32768), dtype=numpy.int8), "W_q")
    scale = numpy_helper.from_array(numpy.zeros((32768,), dtype=numpy.float32), "W_s")
    m.initializers["W_q"] = payload
    m.initializers["W_s"] = scale
    # As fold leaves it: consumer renamed to the payload, dtype int8.
    m.nodes["mm"] = Node("MatMul", "mm", {"transB": 0},
                         [tref("X"), tref("W_q"), tref("W_s")], [tref("out")])
    a = la.audit_layout(m)
    r = a.weights[0]
    assert r.tensor == "W_q"
    assert r.data_type == INT8
    assert r.row_bytes == 32768  # 32768 * 1 byte
    assert not r.flags  # a 32 KiB stride is under the cliff


def test_conv_weights_are_tallied_not_hazarded():
    """A Conv weight's layout is the node's weight_transpose/weight_pack, which
    is fail-closed elsewhere; the audit only accounts for it in the summary."""
    weights = [bits_of("W4", (16, 3, 3, 3), FP32), bits_of("W5", (16, 3, 2, 3, 3), FP32)]
    a = audit(weights, [
        ("Conv", "c4", {"weight_transpose": 1, "weight_pack": 1}, ["X", "W4"]),
        ("Conv", "c5", {"weight_transpose": 0, "weight_pack": 0}, ["X", "W5"]),
    ])
    assert a.weights == []
    assert (a.conv_total, a.conv_packed, a.conv_other_rank) == (2, 1, 1)
    assert "packed 1" in a.summary_line()


def test_shared_weight_is_reported_once_per_consumer():
    """Two consumers of one weight are two read contracts; the audit reports
    per consumer, which is the thing that can disagree."""
    a = audit([bits_of("W", (16, 32768), FP16)], [
        ("MatMul", "mm0", {"transB": 0}, ["X", "W"]),
        ("MatMul", "mm1", {"transB": 0}, ["Y", "W"]),
    ])
    assert len(a.weights) == 2
    assert {r.node for r in a.weights} == {"mm0", "mm1"}


def test_rank1_weight_is_skipped():
    """The runtime rejects a rank-<2 B operand, so there is no layout to judge."""
    m = DAGBasedModel()
    init = numpy_helper.from_array(numpy.zeros((16,), dtype=numpy.float16), "W")
    m.initializers["W"] = init
    m.nodes["mm"] = Node("MatMul", "mm", {}, [tref("X"), tref("W")], [tref("out")])
    a = la.audit_layout(m)
    assert a.weights == []
    assert a.skipped == 1


# ------------------------------------------------------------------- modes
def test_mode_parsing_and_unknown_is_an_error(monkeypatch):
    assert la.layout_audit_mode() == "summary"  # default
    for raw, want in [("off", "off"), ("0", "off"), ("full", "full"),
                      ("strict", "strict"), ("SUMMARY", "summary")]:
        monkeypatch.setenv("VKOP_LAYOUT_AUDIT", raw)
        assert la.layout_audit_mode() == want
    monkeypatch.setenv("VKOP_LAYOUT_AUDIT", "loud")
    with pytest.raises(ValueError):
        la.layout_audit_mode()


def test_report_modes_and_stdout(monkeypatch, capsys):
    """A clean weight plus a hazard, so each mode's line set is distinguishable:
    the hazard is explained in EVERY mode, and only `full` adds the clean rows.
    """
    m = build([bits_of("Clean", (512, 1000), FP16), bits_of("W", (16, 32768), FP16)], [
        ("MatMul", "mm0", {"transB": 0}, ["X", "Clean"]),
        ("MatMul", "mm1", {"transB": 0}, ["X", "W"]),
    ])

    monkeypatch.setenv("VKOP_LAYOUT_AUDIT", "off")
    la.report_layout_audit(m)
    assert capsys.readouterr().out == ""

    monkeypatch.setenv("VKOP_LAYOUT_AUDIT", "summary")
    a = la.report_layout_audit(m)
    out = capsys.readouterr().out
    assert len(a.hazards) == 1
    assert "WARNING W" in out  # a hazard is explained even in summary
    assert "1 HAZARD" in out
    assert "Clean" not in out  # ...but a clean weight gets no line at all
    assert out.count("\n") == 2  # hazard line + summary line

    monkeypatch.setenv("VKOP_LAYOUT_AUDIT", "full")
    la.report_layout_audit(m)
    out = capsys.readouterr().out
    assert "[layout]   Clean (MatMul): k=512 n=1000 [K,N], stride 2000B" in out
    assert "stride 65536B" not in out  # the hazard row is in the WARNING line
    assert out.count("\n") == 3


def test_clean_model_summary_has_no_hazard_word(monkeypatch, capsys):
    m = build([bits_of("W", (512, 1000), FP16)],
              [("MatMul", "mm", {"transB": 0}, ["X", "W"])])
    monkeypatch.setenv("VKOP_LAYOUT_AUDIT", "summary")
    a = la.report_layout_audit(m)
    out = capsys.readouterr().out
    assert a.hazards == []
    assert "HAZARD" not in out
    assert "0 pow2-stride" in out


# ------------------------------------------------------- the converter wiring
def _opts(quant="fp32"):
    return SimpleNamespace(quant=quant, unify=False, rgba=False, group_size=64,
                           target_backend="buffer")


def _one_matmul_dag(dims, dtype):
    m = DAGBasedModel()
    init = numpy_helper.from_array(
        numpy.zeros(dims, dtype=numpy.float16 if dtype == FP16 else numpy.float32),
        "W")
    init.data_type = dtype
    m.initializers["W"] = init
    m.nodes["mm"] = Node("MatMul", "mm", {}, [tref("X"), tref("W")], [tref("out")])
    return m


def test_strict_mode_fails_the_conversion(monkeypatch):
    """strict is the CI gate: a hazard becomes a conversion failure that names
    the weight and the check."""
    monkeypatch.setenv("VKOP_LAYOUT_AUDIT", "strict")
    m = _one_matmul_dag((16, 32768), FP16)
    with pytest.raises(UnsupportedOpError) as ei:
        ModelConverter._audit_weight_layouts(m)
    assert "W" in str(ei.value)
    assert "pow2_row_stride" in str(ei.value)


def test_summary_mode_does_not_fail(monkeypatch):
    monkeypatch.setenv("VKOP_LAYOUT_AUDIT", "summary")
    m = _one_matmul_dag((16, 32768), FP16)
    buf = io.StringIO()
    with redirect_stdout(buf):
        a = ModelConverter._audit_weight_layouts(m)
    assert len(a.hazards) == 1
    assert "65536" in buf.getvalue()  # the row stride, in the explanation


def test_pipeline_audits_the_post_quantization_weight(monkeypatch):
    """The audit runs after quantization, so it must see the fp16 weight's
    element size (2 bytes), not the 4 bytes it was declared with. n = 32768
    would be 131072 B/row as fp32 -- also a hazard, but a different one, and the
    fp16 k%16 check would not apply at all."""
    monkeypatch.setenv("VKOP_LAYOUT_AUDIT", "summary")
    m = _one_matmul_dag((16, 32768), FP32)  # the audit's counterfactual row
    # The real weight, already half-precision on disk with the same dims: the
    # selective fp16 pass reads this dtype off the graph, so declaring it fp32
    # and hoping the pass rewrites it is not the same test.
    init = numpy_helper.from_array(
        numpy.zeros((16, 32768), dtype=numpy.float16), "W")
    m.initializers["W"] = init
    # The graph must reach its output: the optimizer's last pass is
    # prune_and_materialize, whose liveness is seeded from the graph outputs and
    # the initializer names (dag.py:351-368), so a model with neither is pruned
    # to nothing before the audit runs. A concrete `out` shape is what keeps the
    # MatMul `mm` live -- the initializer seed alone only keeps the weight.
    m.inputs = [tref("X")]
    m.outputs = [{"name": "out", "shape": [16, 32768]}]
    buf = io.StringIO()
    with redirect_stdout(buf):
        ModelConverter().apply_optimizations(m, _opts(quant="fp16"))
    out = buf.getvalue()
    assert "[layout]" in out
    assert "1 GEMM weight(s)" in out
    assert "1 pow2-stride" in out  # 32768 fp16 columns = 65536 B/row
    assert m.initializers["W"].data_type == FP16
