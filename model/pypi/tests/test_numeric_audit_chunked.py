"""The chunked reductions must reproduce the whole-array ones.

The change is supposed to be memory-only: same numbers, bounded temporaries.
That is a claim to check, not to assume -- reordering a sum, dividing by a
per-chunk element count, or letting eps into a different place all move the
last digits. So each metric is recomputed the OLD way (whole arrays, float64,
the code as it stood) and compared against the RoundTrip accumulator.

The tolerance is tight (1e-12 relative) but not zero: the old code summed in
float64 over the whole array while this sums float64 chunks, so the association
order differs and the last bits may move. Everything else -- which elements
count as subnormal, what a zero weight reports, the size check -- has to match
exactly, and those are asserted with ==.

Why it matters: `-q int4` on the 27B peaked at 54 GiB, and 39.9 GiB of that was
these diagnostics materializing whole-weight float64 copies. b3 has 62 GB.
"""
import numpy as np
import pytest

from onnx2vkop import numeric_audit as na


def old_way(src, got, fmt):
    """The pre-change arithmetic, verbatim: whole arrays, float64."""
    s = np.asarray(src, dtype=np.float64).reshape(-1)
    g = np.asarray(got, dtype=np.float64).reshape(-1)
    diff = g - s
    norm = float(np.linalg.norm(s))
    if norm == 0.0:
        rel_l2 = 0.0 if not np.any(diff) else float("inf")
    else:
        rel_l2 = float(np.linalg.norm(diff)) / norm
    rel = np.abs(diff) / (np.abs(s) + 1e-8)
    return (rel_l2, float(np.mean(diff ** 2)), float(np.mean(np.abs(diff))),
            float(np.mean(rel)), float(np.max(rel)))


def rows():
    rng = np.random.RandomState(20261116)
    return [
        ("plain small", rng.randn(1000).astype(np.float32)),
        # The grouped view the 4-bit encoders compare in.
        ("grouped shape", rng.randn(40, 64, 8).astype(np.float32)),
        # Exactly one chunk and one past it: the loop boundary.
        ("one chunk", rng.randn(na._CHUNK_ELEMS).astype(np.float32)),
        ("one chunk + 7", rng.randn(na._CHUNK_ELEMS + 7).astype(np.float32)),
        ("zeros", np.zeros(4096, dtype=np.float32)),
        ("tiny magnitudes", np.array([1e-30, 0.0, -1e-30], dtype=np.float32)),
    ]


@pytest.mark.parametrize("label,src", rows())
def test_chunked_matches_whole_array(label, src):
    """Every metric, over shapes that straddle the chunk boundary."""
    rng = np.random.RandomState(abs(hash(label)) % (2 ** 31))
    got = (src + rng.randn(*src.shape).astype(np.float32) * 1e-3).astype(
        np.float32)
    rt = na.pair_stats(src, got, "int4")
    w_rel_l2, w_mse, w_mae, w_mean, w_max = old_way(src, got, "int4")
    assert rt.rel_l2 == pytest.approx(w_rel_l2, rel=1e-12, abs=0.0), label
    assert rt.mse == pytest.approx(w_mse, rel=1e-12, abs=0.0), label
    assert rt.mae == pytest.approx(w_mae, rel=1e-12, abs=0.0), label
    assert rt.mean_rel == pytest.approx(w_mean, rel=1e-12, abs=0.0), label
    assert rt.max_rel == pytest.approx(w_max, rel=1e-12, abs=0.0), label
    assert rt.elements == np.asarray(src).size, label


def test_chunking_a_few_at_a_time_is_the_same_as_one_shot():
    """The boundary is an implementation detail, not a knob that changes the
    answer -- so a caller free to size its own slices gets the same row."""
    rng = np.random.RandomState(7)
    src = rng.randn(5000).astype(np.float32)
    got = (src + 1e-3).astype(np.float32)
    one = na.pair_stats(src, got, "int8")
    many = na.RoundTrip("int8")
    for i in range(0, src.size, 64):          # tiny slices on purpose
        many.add(src[i:i + 64], got[i:i + 64])
    assert many.rel_l2 == pytest.approx(one.rel_l2, rel=1e-12)
    assert many.elements == one.elements


def test_a_zero_weight_the_format_moves_is_infinite_relative_error():
    """A padding channel: no relative scale to be wrong against."""
    z = np.zeros(512, dtype=np.float32)
    moved = np.zeros(512, dtype=np.float32)
    moved[3] = 0.25
    rt = na.pair_stats(z, moved, "int8")
    assert rt.rel_l2 == float("inf")
    assert rt.max_abs == pytest.approx(0.25)


def test_a_zero_weight_that_stays_zero_costs_nothing():
    z = np.zeros(512, dtype=np.float32)
    rt = na.pair_stats(z, z, "int8")
    assert rt.rel_l2 == 0.0
    assert rt.max_abs == 0.0


def test_empty_weight_reports_nan_not_a_perfect_zero():
    """np.mean over an empty slice is nan (0/0), and that is what the printed
    line said. Reporting 0.0 here would read as a perfect round trip."""
    rt = na.pair_stats(np.zeros(0, np.float32), np.zeros(0, np.float32), "int4")
    assert np.isnan(rt.mse) and np.isnan(rt.mae) and np.isnan(rt.mean_rel)
    assert rt.rel_l2 == 0.0
    assert rt.elements == 0


def test_fp16_overflow_and_subnormals_are_counted():
    """Both 1e-6 and 1e-8 sit below fp16's smallest normal (2^-14 = 6.1e-5)
    and are non-zero, so both count; the 0.0 does not."""
    src = np.array([70000.0, 1.0, 1e-6, 0.0, -1e-8], dtype=np.float32)
    got = np.array([np.inf, 1.0, 1e-6, 0.0, -1e-8], dtype=np.float32)
    rt = na.pair_stats(src, got, "fp16")
    assert rt.overflow == 1
    assert rt.subnormals == 2


def test_a_different_element_count_is_a_caller_bug_not_a_row():
    """A view change is fine; the two halves of a round trip are not
    comparable when they are different tensors."""
    with pytest.raises(ValueError, match="restored 6 element"):
        na.pair_stats(np.zeros(4, np.float32), np.zeros(6, np.float32), "int4")


def test_record_accepts_an_accumulator_without_the_arrays():
    """The streaming path hands over a RoundTrip; the row must be the same one
    record_roundtrip would have made from the whole arrays."""
    src = np.arange(24, dtype=np.float32).reshape(6, 4)
    got = src + 0.5
    with na.collecting() as audit:
        na.record_roundtrip("w", "int4", src, got)
    with na.collecting() as audit2:
        na.record("w", "int4", na.pair_stats(src, got, "int4"))
    a, b = audit.rows[0], audit2.rows[0]
    assert (a.tensor, a.fmt, a.elements, a.rel_l2, a.max_abs) == \
           (b.tensor, b.fmt, b.elements, b.rel_l2, b.max_abs)


# ------------------------------------------- the printed per-weight line
# The 4-bit pass streams its decode through RoundTrip now, so the line it prints
# (MSE / MAE / Mean and Max Rel Error) has to be the same number the old
# whole-array expression produced -- it is what a human reads to decide whether
# to hold a layer wide, and nothing else checks it.

def test_report_qerror_prints_the_whole_array_numbers():
    import contextlib
    import io

    from onnx2vkop.optimizer import Quantizer

    rng = np.random.RandomState(3)
    for n_groups, group, n in ((8, 64, 16), (100, 64, 7), (3, 64, 40)):
        w3 = (rng.randn(n_groups, group, n) * 0.02).astype(np.float32)
        codes, sc = Quantizer._quantize_w3(w3, "int4")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            Quantizer._report_qerror("W", w3, codes, sc, "int4",
                                     n_groups * group, n, n_groups, group)
        printed = buf.getvalue()

        # The expression this replaced, on the whole arrays.
        dq = Quantizer._dequantize_w3(codes, sc, "int4")
        diff = dq - w3
        rel = np.abs(diff) / (np.abs(w3) + 1e-8)
        for want in (f"{np.mean(diff ** 2):.6e}", f"{np.mean(np.abs(diff)):.6e}",
                     f"{np.mean(rel):.2%}", f"{np.max(rel):.2%}"):
            assert want in printed, (n_groups, group, n, want, printed)
