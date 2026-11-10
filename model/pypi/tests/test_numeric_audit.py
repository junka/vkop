"""numeric_audit.py: what a -q format costs each weight, said at convert time.

The audit exists so that "did this format damage the model" is answered without
running the model: for a weight-only format the kernel reads back
`decode(encode(W))`, and the quantizer is holding both arrays at the moment it
writes them. These tests pin the arithmetic (it has to be the same relative-L2
the accuracy loop uses on outputs, or the prediction cannot be checked against
the measurement), the fp16 saturation and subnormal counts, the mode handling,
and -- the property the whole feature is named for -- that none of it changes
the artifact.
"""
import contextlib
import io

import numpy
import pytest
from onnx import numpy_helper

from onnx2vkop import numeric_audit
from onnx2vkop.dag import DAGBasedModel, Node
from onnx2vkop.optimizer import Quantizer


def rec(*args, **kw):
    """record_roundtrip with the row it appended, inside one collection."""
    with numeric_audit.collecting() as audit:
        numeric_audit.record_roundtrip(*args, **kw)
    return audit


@contextlib.contextmanager
def quiet():
    """No VKOP_NUMERIC_AUDIT printing, so a test reads rows instead of stdout."""
    import os
    old = os.environ.get(numeric_audit._ENV)
    os.environ[numeric_audit._ENV] = "off"
    try:
        yield
    finally:
        if old is None:
            del os.environ[numeric_audit._ENV]
        else:
            os.environ[numeric_audit._ENV] = old


# ------------------------------------------------------------- the arithmetic

def test_rel_l2_is_the_l2_of_the_round_trip():
    """||W - restored|| / ||W||, the loop's formula (vkop_vs_ort.py compare)."""
    src = numpy.array([1.0, 2.0, 3.0, 4.0], dtype=numpy.float32)
    got = src.copy()
    got[0] += 0.1
    row = rec("w", "int8", src, got).rows[0]
    assert row.rel_l2 == pytest.approx(0.1 / numpy.linalg.norm(src))
    assert row.max_abs == pytest.approx(0.1)
    assert row.elements == 4
    assert row.overflow == 0 and row.subnormals == 0


def test_a_grouped_view_of_the_same_element_count_is_the_same_row():
    """The 4-bit paths compare in [groups, size, N]; only the element multiset
    matters to an L2 norm, so the shape must not change the number."""
    src = numpy.arange(12, dtype=numpy.float32).reshape(3, 4)
    got = src + 0.5
    flat = rec("w", "int4", src.reshape(-1), got.reshape(-1)).rows[0]
    grouped = rec("w", "int4", src, got).rows[0]
    assert flat.rel_l2 == pytest.approx(grouped.rel_l2)


def test_an_all_zero_weight_costs_nothing_and_is_not_an_error():
    """A padding channel: no relative scale to be wrong against."""
    z = numpy.zeros((4,), dtype=numpy.float32)
    assert rec("w", "int8", z, z).rows[0].rel_l2 == 0.0


def test_a_zero_weight_the_format_moves_is_infinite_relative_error():
    z = numpy.zeros((4,), dtype=numpy.float32)
    got = numpy.array([0.0, 0.25, 0.0, 0.0], dtype=numpy.float32)
    row = rec("w", "int8", z, got).rows[0]
    assert row.rel_l2 == float("inf")
    assert row.max_abs == pytest.approx(0.25)


def test_a_different_element_count_is_a_caller_bug_not_a_row():
    """A view change is fine; the two halves of a round trip are not
    comparable when they are different tensors."""
    with pytest.raises(ValueError, match="restored 6 element"):
        rec("w", "int8", numpy.zeros((4,), numpy.float32),
            numpy.zeros((6,), numpy.float32))


# ------------------------------------------------------------------ fp16 side

def test_fp16_saturation_past_65504_is_counted_as_overflow():
    """fp16_graph.py downcasts anyway (Inf is the defined outcome); this is the
    half that says so next to the number it cost."""
    src = numpy.array([70000.0, 1.0, 2.0, 3.0], dtype=numpy.float32)
    with numpy.errstate(over="ignore"):
        got = src.astype(numpy.float16).astype(numpy.float32)
    row = rec("w", "fp16", src, got).rows[0]
    assert row.overflow == 1
    assert row.flagged


def test_finite_sources_stay_finite_and_report_no_overflow():
    src = numpy.array([65504.0, 1e-6, 0.0, -3.0], dtype=numpy.float32)
    got = src.astype(numpy.float16).astype(numpy.float32)
    row = rec("w", "fp16", src, got).rows[0]
    assert row.overflow == 0


def test_subnormals_count_the_finite_values_under_the_formats_normal():
    """2**-14 is where half precision stops advertising 10 mantissa bits. Exact
    zeros are not subnormal and are not counted."""
    src = numpy.array([1e-5, 2e-5, 0.0, 1.0, -1.0], dtype=numpy.float32)
    row = rec("w", "fp16", src, src).rows[0]
    assert row.subnormals == 2
    assert row.subnormal_share == pytest.approx(0.4)


def test_a_byte_format_has_no_subnormal_column():
    """There is no exponent in int8, so a small magnitude is not a precision
    loss -- reporting one would be a fabricated number."""
    src = numpy.array([1e-9, 1e-9, 1.0], dtype=numpy.float32)
    row = rec("w", "int8", src, src).rows[0]
    assert row.subnormals == 0


# ------------------------------------------------------------- collection life

def test_recording_is_inert_outside_a_conversion():
    """A quantizer used on its own (a test, a library caller) must not raise and
    must not leak a row into the next conversion."""
    src = numpy.ones((3,), dtype=numpy.float32)
    numeric_audit.record_roundtrip("w", "int8", src, src)
    assert numeric_audit._STACK == []
    with quiet(), numeric_audit.collecting() as audit:
        pass
    assert audit.rows == []
    assert numeric_audit._STACK == []


def test_a_nested_collection_closes_itself_without_closing_the_outer():
    """AWQ runs a byte quantizer for its own format; if that ends the outer
    audit the -q verdict arrives as two partial reports."""
    src = numpy.ones((3,), dtype=numpy.float32)
    outer = numeric_audit.begin()
    with quiet(), numeric_audit.collecting():
        numeric_audit.record_roundtrip("inner", "int8", src, src)
    with quiet(), numeric_audit.collecting() as second:
        numeric_audit.record_roundtrip("outer", "int8", src, src)
    numeric_audit.finish()
    assert [r.tensor for r in second.rows] == ["outer"]
    assert [r.tensor for r in outer.rows] == []


def test_dead_scale_slices_reach_the_row():
    """The channel whose absmax was 0 got scale 1.0 from the division-by-zero
    fallback: not an error, but the one place the stored table stops describing
    the weights."""
    src = numpy.array([0.0, 0.0, 1.0], dtype=numpy.float32)
    row = rec("w", "int8", src, src, dead_slices=2).rows[0]
    assert row.dead_slices == 2
    assert "dead-scale 2" in row.line()


def test_consumer_tag_names_the_node_an_exclusion_flag_would_need():
    m = DAGBasedModel()
    m.nodes["/layer1/conv/Conv"] = Node("Conv", "/layer1/conv/Conv", {}, [], [])
    assert numeric_audit.consumer_tag([m.nodes["/layer1/conv/Conv"]]) == \
        "/layer1/conv/Conv"
    assert numeric_audit.consumer_tag([]) == ""


# ------------------------------------------------------------------- thresholds

def test_only_the_weight_above_its_formats_threshold_is_flagged(monkeypatch):
    """Each format pays its own usual cost: int8 lands near 1.3e-02 per weight
    on resnet18, so the gate sits above that and names an outlier rather than
    the format."""
    monkeypatch.setitem(numeric_audit._REL_L2_WARN, "int8", 0.02)
    src = numpy.array([1.0, 1.0, 1.0, 1.0], dtype=numpy.float32)
    ok = src + 0.005
    bad = src + 0.2
    with quiet(), numeric_audit.collecting() as audit:
        numeric_audit.record_roundtrip("normal", "int8", src, ok)
        numeric_audit.record_roundtrip("outlier", "int8", src, bad,
                                       via="/b/Conv")
    assert [r.tensor for r in audit.flagged()] == ["outlier"]
    lines = audit.hazard_lines()
    assert lines[0].startswith("[numeric-audit] WARNING: 1 int8 weight(s)")
    assert "(2e-02)" in lines[0]
    assert "outlier" in lines[1] and "normal" not in lines[1]
    assert "via /b/Conv" in lines[1]


def test_an_unlisted_format_is_never_flagged_on_error_alone(monkeypatch):
    """A format with no measured baseline gets no verdict; inventing one would
    print a number nobody has to act on."""
    monkeypatch.delitem(numeric_audit._REL_L2_WARN, "int8", raising=False)
    src = numpy.ones((4,), dtype=numpy.float32)
    row = rec("w", "int8", src, src * 3).rows[0]
    assert not row.flagged


def test_summary_reports_median_and_worst_per_format(monkeypatch, capsys):
    monkeypatch.setitem(numeric_audit._REL_L2_WARN, "int8", 1.0)
    src = numpy.ones((4,), dtype=numpy.float32)
    with numeric_audit.collecting():
        numeric_audit.record_roundtrip("a", "int8", src, src + 0.01)
        numeric_audit.record_roundtrip("b", "int8", src, src + 0.03)
    out = capsys.readouterr().out
    assert "[numeric-audit] int8: 2 weight(s), median relL2 2.00e-02, " \
        "worst 3.00e-02" in out


def test_summary_says_nothing_when_no_format_was_applied(capsys):
    """A plain conversion has no -q cost to report; a line per model would be
    noise that trains people to ignore the tag."""
    with numeric_audit.collecting():
        pass
    assert capsys.readouterr().out == ""


def test_full_prints_every_weight_and_off_prints_nothing(monkeypatch, capsys):
    monkeypatch.setitem(numeric_audit._REL_L2_WARN, "int8", 1.0)
    src = numpy.ones((4,), dtype=numpy.float32)
    for mode in ("off", "summary", "full"):
        monkeypatch.setenv(numeric_audit._ENV, mode)
        capsys.readouterr()
        with numeric_audit.collecting():
            numeric_audit.record_roundtrip("a", "int8", src, src + 0.01)
            numeric_audit.record_roundtrip("b", "int8", src, src + 0.02)
        out = capsys.readouterr().out
        assert (out == "") if mode == "off" else (out != "")
        assert ("relL2 1.000e-02" in out) == (mode == "full")


def test_an_unknown_mode_is_a_typo_not_a_default(monkeypatch):
    monkeypatch.setenv(numeric_audit._ENV, "verbose")
    with pytest.raises(ValueError, match="not one of off, summary, full"):
        numeric_audit.numeric_audit_mode()


# ------------------------------------------------------------- the thresholds

def test_every_format_the_pipeline_records_has_a_baseline():
    """A format missing from the table fails open: its weights are never named,
    and nothing says the gate is off. Adding a -q format means measuring its
    cost and writing the number down (numeric_audit.py has the resnet18
    measurements that produced these)."""
    recorded = {"fp16", "int8", "int4", "nf4", "fp8e4m3", "fp8e5m2", "nvfp4"}
    assert recorded <= set(numeric_audit._REL_L2_WARN)


def test_a_healthy_weight_is_quiet_and_an_outlier_is_not():
    """The table sits at about twice the worst per-weight error resnet18 shows
    at each format, so a normal model prints its tally and nothing else."""
    def at(fmt, rel_l2):
        src = numpy.array([1.0, 2.0, 3.0, 4.0], dtype=numpy.float32)
        got = src.copy()
        got[0] += rel_l2 * numpy.linalg.norm(src)   # ||diff|| / ||src|| = rel_l2
        return rec("w", fmt, src, got)

    for fmt, healthy in (("fp16", 2.84e-04), ("int8", 1.60e-02),
                         ("int4", 1.66e-01)):
        assert not at(fmt, healthy).rows[0].flagged, fmt
        assert at(fmt, 10 * healthy).rows[0].flagged, fmt


# --------------------------------------------------------- the artifact is not

def int8_model():
    w = numpy.arange(24, dtype=numpy.float32).reshape(6, 4) - 11.5
    m = DAGBasedModel()
    m.initializers["W"] = numpy_helper.from_array(w, "W")
    m.nodes["mm"] = Node("MatMul", "mm", {},
                         [{"name": "act", "shape": [-1]}, {"name": "W"}],
                         [{"name": "y"}])
    return m


def snapshot(m):
    return ({n: (i.data_type, list(i.dims), i.raw_data or bytes(i.int64_data))
             for n, i in sorted(m.initializers.items())},
            {n: [i["name"] for i in node.inputs] for n, node in m.nodes.items()})


def test_auditing_changes_no_byte_of_the_artifact(capsys):
    """This is what makes the feature allowed to run on every conversion: the
    report reads the numbers the quantizer already computed. If a record call
    ever mutates an array it is handed, this fails."""
    with contextlib.redirect_stdout(io.StringIO()):
        plain = int8_model()
        Quantizer.quantize_to_int8_weight_only(plain)
        audited = int8_model()
    # The report leaves the redirect with the `with`, so the tally lands on the
    # captured stdout while the quantizer's own per-tensor print stays hidden.
    with numeric_audit.collecting():
        with contextlib.redirect_stdout(io.StringIO()):
            Quantizer.quantize_to_int8_weight_only(audited)
    assert snapshot(audited) == snapshot(plain)
    tally = capsys.readouterr().out
    assert "[numeric-audit]" in tally and "int8: 1 weight(s)" in tally


def test_the_row_names_the_format_and_its_consumer():
    with contextlib.redirect_stdout(io.StringIO()):
        with numeric_audit.collecting() as audit:
            Quantizer.quantize_to_int8_weight_only(int8_model())
    assert [(r.fmt, r.via) for r in audit.rows] == [("int8", "mm")]
    assert audit.rows[0].rel_l2 < 0.05
