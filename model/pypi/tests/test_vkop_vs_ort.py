"""The accuracy loop's contract, measured without a GPU.

tools/vkop_vs_ort.py promises three things, and each is a way the loop could
lie instead:

  * the runtime and onnxruntime run on ONE input tensor, so a divergence is
    the kernels' and not two preprocessings' -- checked end to end below with a
    stub vkop-bench that reports the array it was handed.
  * the dumped bytes are read as the manifest says, so a truncated or
    mis-shaped dump fails loudly instead of decoding into plausible numbers --
    checked by writing dumps the wrong way on purpose.
  * the gates mean what they claim: an fp16 run's near-zero elements are
    allowed the rounding floor (the measured 11.1% that a per-element band
    flags at cosine 0.999995), an integer output gets no band at all, and the
    exit code follows the verdict rather than the arithmetic working.

The stub replaces vkop-bench itself, so nothing here needs Vulkan; the real
numbers on real models come from running the tool (see the resnet18 and
add_conv fixtures in docs/ and the manual runs it was written against).
"""
import json
import subprocess
import sys
import textwrap
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest

import vkop_vs_ort as loop

DEFAULTS = {
    "vkopbin": None, "onnx": None, "bench": None, "inputs": None,
    "random_fill": False, "seed": 0xC0FFEE, "precision": "fp16",
    "backend": "buffer", "shape": None, "ort_threads": None,
    "cos_min": 0.999, "rtol": 1e-2, "atol": 1e-3, "bad_frac_max": 1e-3,
    "topk": 0, "limit": 0, "work_dir": None, "keep": False, "verbose": False,
}
LABEL = {np.dtype(np.float32): "float32", np.dtype(np.float16): "float16",
         np.dtype(np.int64): "int64", np.dtype(np.int32): "int32",
         np.dtype(np.int8): "int8", np.dtype(np.uint8): "uint8"}


def args(**kw):
    return Namespace(**{**DEFAULTS, **kw})


def spec(name, dtype="float32", dims=(1, 4)):
    return {"name": name, "dtype": dtype, "dims": list(dims),
            "value_dynamic": False}


def dump(tmp_path, *arrays):
    """Write a manifest.tsv plus one .bin per array, as vkop-bench would."""
    out = tmp_path / "out"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for arr, name in arrays:
        arr.tofile(out / f"{name}.bin")
        rows.append({"name": name, "bytes_as": LABEL[arr.dtype],
                     "dims": list(arr.shape), "nbytes": arr.nbytes})
    write_manifest(out, rows)
    return out, rows


def write_manifest(out, rows, header=True):
    lines = ["# vkop-bench outputs: name\tbytes-as\tdims\tnbytes"] \
        if header else []
    lines += [f"{r['name']}\t{r['bytes_as']}\t"
              f"{'x'.join(str(d) for d in r['dims'])}\t{r['nbytes']}"
              for r in rows]
    (out / "manifest.tsv").write_text("\n".join(lines) + "\n")
    return out / "manifest.tsv"


# ------------------------------------------------------------- the manifest


def test_parse_manifest_reads_what_bench_wrote(tmp_path):
    out = write_manifest(tmp_path, [
        {"name": "y", "bytes_as": "float32", "dims": [1, 4], "nbytes": 16},
        {"name": "cache", "bytes_as": "int64", "dims": [2, 3, 4],
         "nbytes": 192}])
    assert loop.parse_manifest(out) == [
        {"name": "y", "bytes_as": "float32", "dims": [1, 4], "nbytes": 16},
        {"name": "cache", "bytes_as": "int64", "dims": [2, 3, 4],
         "nbytes": 192}]


def test_parse_manifest_refuses_an_unnamed_byte_layout(tmp_path):
    # BYTES_AS is the reader's half of the dispatch in vkop_bench.cpp; a row
    # naming anything else means the two halves drifted apart.
    out = write_manifest(tmp_path, [
        {"name": "y", "bytes_as": "float8e4m3", "dims": [4], "nbytes": 4}])
    with pytest.raises(loop.LoopError, match="no numpy view"):
        loop.parse_manifest(out)


def test_parse_manifest_refuses_a_short_row(tmp_path):
    (tmp_path / "manifest.tsv").write_text("y\tfloat32\t16\n")
    with pytest.raises(loop.LoopError, match="malformed row"):
        loop.parse_manifest(tmp_path / "manifest.tsv")


def test_parse_manifest_refuses_a_missing_manifest(tmp_path):
    with pytest.raises(loop.LoopError, match="no dtype or shape"):
        loop.parse_manifest(tmp_path / "manifest.tsv")


def test_parse_manifest_refuses_a_header_only_file(tmp_path):
    write_manifest(tmp_path, [], header=True)
    with pytest.raises(loop.LoopError, match="no output rows"):
        loop.parse_manifest(tmp_path / "manifest.tsv")


# -------------------------------------------------- reading the dumped bytes


def test_read_output_decodes_the_bytes_the_manifest_describes(tmp_path):
    want = np.arange(8, dtype=np.float32).reshape(2, 4)
    out, rows = dump(tmp_path, (want, "y"))
    assert np.array_equal(loop.read_output(out, rows[0]), want)


def test_read_output_catches_a_truncated_write(tmp_path):
    out, rows = dump(tmp_path, (np.zeros(8, dtype=np.float32).reshape(2, 4),
                                "y"))
    (out / "y.bin").write_bytes(b"\x00" * 12)  # 12 bytes, manifest says 32
    with pytest.raises(loop.LoopError, match="manifest says 32 bytes, file "
                                             "has 12"):
        loop.read_output(out, rows[0])


def test_read_output_catches_dims_that_do_not_fill_the_file(tmp_path):
    # The row is self-contradictory: 3 elements cannot be 8 bytes of float32.
    out = tmp_path / "out"
    out.mkdir()
    (out / "y.bin").write_bytes(b"\x00" * 8)
    bad = {"name": "y", "bytes_as": "float32", "dims": [3], "nbytes": 8}
    with pytest.raises(loop.LoopError, match="dims=\\[3\\]"):
        loop.read_output(out, bad)


def test_read_output_catches_a_row_without_its_file(tmp_path):
    with pytest.raises(loop.LoopError, match="not on disk"):
        loop.read_output(tmp_path, {"name": "y", "bytes_as": "float32",
                                    "dims": [1], "nbytes": 4})


def test_read_output_accepts_fp16_storage(tmp_path):
    want = np.array([1.0, -2.5], dtype=np.float16)
    out, rows = dump(tmp_path, (want, "y"))
    got = loop.read_output(out, rows[0])
    assert got.dtype == np.float16 and np.array_equal(got, want)


# ----------------------------------------------------------- float  metrics


def test_identical_outputs_pass_every_gate():
    ref = np.arange(6, dtype=np.float32).reshape(2, 3)
    m = loop.compare(ref, ref.copy(), 1e-2, 1e-3, 0.999, 1e-3)
    assert m["ok"] and m["maxabs"] == 0.0 and m["rel_l2"] == 0.0
    assert m["cos"] == pytest.approx(1.0) and m["reasons"] == []


def test_shape_difference_is_reported_without_metrics():
    m = loop.compare(np.zeros((1, 4), np.float32), np.zeros((4,), np.float32),
                     1e-2, 1e-3, 0.999, 1e-3)
    assert not m["ok"] and m["reasons"] == [
        "reference shape [1, 4] vs runtime [4]"]


def test_nan_in_the_runtime_output_fails_even_when_cos_is_perfect():
    ref = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    got = ref.copy()
    got[1] = np.nan
    m = loop.compare(ref, got, 1e-2, 1e-3, 0.999, 1e-3)
    assert not m["ok"] and "1 NaN(s) in the runtime output" in m["reasons"]


def test_a_more_inf_than_the_reference_fails():
    ref = np.array([1.0, 2.0], dtype=np.float32)
    got = np.array([1.0, np.inf], dtype=np.float32)
    m = loop.compare(ref, got, 1e-2, 1e-3, 0.999, 1e-3)
    assert not m["ok"]
    assert "1 element(s) where one side is Inf and the other is not" \
        in m["reasons"]


def test_inf_in_the_wrong_element_fails_even_with_equal_counts():
    # An fp16 overflow that lands on a different accumulator than the
    # reference's is a different bug, not the same bug: comparing the Inf
    # COUNTS said "fine" and left the row to the outlier gate, which an Inf
    # defeats because |a-b| is NaN there and NaN > band is False.
    ref = np.array([np.inf, 2.0, 3.0], dtype=np.float32)
    got = np.array([1.0, 2.0, np.inf], dtype=np.float32)
    m = loop.compare(ref, got, 1e-2, 1e-3, 0.999, 1e-3)
    assert not m["ok"]
    assert "2 element(s) where one side is Inf and the other is not" \
        in m["reasons"]


def test_a_matching_inf_does_not_fail_on_its_own():
    # Overflow the reference itself has, at the same element: that is the
    # model's behaviour, not the runtime's, so it is not this gate's business.
    ref = np.array([1.0, np.inf], dtype=np.float32)
    got = np.array([1.0, np.inf], dtype=np.float32)
    m = loop.compare(ref, got, 1e-2, 1e-3, 0.999, 1e-3)
    assert m["inf"] == 0 and m["ok"]


def test_an_inf_in_the_reference_does_not_widen_the_band():
    # max|ref| is Inf, so atol + rtol*max|ref| is Inf and every element clears
    # it: the row used to read OK with the finite half off by 49. The band is
    # built from the largest FINITE element, and the agreed Inf is dropped from
    # the difference so maxabs stays a number.
    ref = np.array([1.0, np.inf], dtype=np.float32)
    got = np.array([50.0, np.inf], dtype=np.float32)
    m = loop.compare(ref, got, 1e-2, 1e-3, 0.999, 1e-3)
    assert not m["ok"] and m["inf"] == 0
    assert m["maxabs"] == 49.0 and m["bad_frac"] == pytest.approx(0.5)


def test_a_zero_vector_pair_reports_a_perfect_cosine():
    ref = np.zeros((2, 3), dtype=np.float32)
    got = np.zeros((2, 3), dtype=np.float32)
    m = loop.compare(ref, got, 1e-2, 1e-3, 0.999, 1e-3)
    assert m["ok"], m["reasons"]
    assert m["cos"] == 1.0 and m["rel_l2"] == 0.0


def test_nonfinite_cosine_is_never_a_reason_of_its_own():
    # '-' in the table, not "cos nan < 0.999": the NaN/Inf mismatch already
    # named the actual failure, and a comparison to NaN would bury it.
    ref = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    got = np.array([np.nan, 2.0, 3.0], dtype=np.float32)
    m = loop.compare(ref, got, 1e-2, 1e-3, 0.999, 1e-3)
    assert np.isnan(m["cos"])
    assert not any("cos" in r for r in m["reasons"]), m["reasons"]
    assert m["ok"] is False


def test_the_outlier_band_is_the_tensors_scale_not_each_element():
    # What the add_conv fixture measures: fp16 storage puts ~11% of elements
    # more than 1% out in RELATIVE terms because they sit near zero while the
    # intermediates they came from are ~4.0. Judged against max|ref| their
    # absolute error is rounding, which is why the band is scale-relative.
    ref = np.array([3.98, 0.16, 0.05, 0.01, 0.02, 0.03], dtype=np.float32)
    got = ref + np.array([0.0, 0.0043, 0.0043, 0.0043, 0.0043, 0.0043],
                         dtype=np.float32)
    m = loop.compare(ref, got, 1e-2, 1e-3, 0.999, 1e-3)
    elementwise = (np.abs(got - ref) > 1e-3 + 1e-2 * np.abs(ref)).mean()
    assert elementwise > 0.5  # the band this gate deliberately does NOT use
    assert m["ok"] and m["bad_frac"] == 0.0


def test_one_element_beyond_the_band_is_caught():
    # 0.05 on one of 40 elements, where max|ref| is 4.0: the band is
    # atol + rtol*4 = 0.041, so this is 1/40 = 2.5% beyond it, while the
    # cosine (0.999995) and the mean error both still look clean. That is the
    # shape of bug an aggregate-only gate misses.
    ref = np.linspace(0.2, 4.0, 40, dtype=np.float32)
    got = ref.copy()
    got[20] += 0.05
    m = loop.compare(ref, got, 1e-2, 1e-3, 0.999, 1e-3)
    assert not m["ok"] and len(m["reasons"]) == 1
    assert "beyond atol + rtol*max|ref|" in m["reasons"][0]
    assert m["cos"] > 0.999 and m["bad_frac"] == pytest.approx(0.025)


def test_cos_min_is_the_gate_the_caller_sets():
    # Cosine is scale-invariant, so a uniform gain cannot trip it: this is a
    # two-element perturbation with the outlier gate loosened out of the way,
    # which leaves cos as the only thing deciding the verdict.
    ref = np.linspace(0, 1, 10, dtype=np.float32)
    got = ref.copy()
    got[5] += 0.10
    got[8] -= 0.08
    assert loop.compare(ref, got, 1e-2, 1e-3, 0.99, 1.0)["ok"]
    m = loop.compare(ref, got, 1e-2, 1e-3, 0.999, 1.0)
    assert not m["ok"] and m["cos"] == pytest.approx(0.99766678, abs=1e-6)
    assert "cos 0.997667 < 0.999" in m["reasons"][0]


def test_a_zero_element_output_passes_on_its_shape():
    # A kv_len=0 cache slice: nothing to compare, and mean()/norm() would
    # raise rather than report.
    m = loop.compare(np.zeros((0, 4), np.float32), np.zeros((0, 4), np.float32),
                     1e-2, 1e-3, 0.999, 1e-3)
    assert m["ok"] and m["cos"] == 1.0 and m["maxabs"] == 0.0


# --------------------------------------------------------- integer  metrics


def test_an_equal_integer_output_passes_exactly():
    ref = np.array([7, 0, 512, -3], dtype=np.int64)
    m = loop.compare(ref, ref.copy(), 1e-2, 1e-3, 0.999, 1e-3)
    assert m["ok"] and m["exact_mismatch"] == 0


def test_a_wrong_token_id_gets_no_tolerance_band():
    ref = np.array([7, 0, 512, -3], dtype=np.int64)
    got = ref.copy()
    got[2] = 513  # 1 in 4096 relative terms, which no float gate would flag
    m = loop.compare(ref, got, 1e-2, 1e-3, 0.999, 1e-3)
    assert not m["ok"] and m["exact_mismatch"] == 1
    assert "no tolerance band" in m["reasons"][0]
    assert m["maxabs"] == 1.0


def test_int8_and_uint8_are_compared_as_integers():
    ref = np.array([-128, 3], dtype=np.int8)
    assert not loop.compare(ref, np.array([-127, 3], dtype=np.int8), 1e-2,
                            1e-3, 0.999, 1e-3)["ok"]
    assert loop.compare(np.array([0, 1], np.uint8), np.array([0, 1], np.uint8),
                        1e-2, 1e-3, 0.999, 1e-3)["ok"]


# ---------------------------------------------------------------- top-k


def test_topk_reports_agreement_and_overlap():
    ref = np.array([[0.1, 0.2, 0.7, 0.9], [0.9, 0.1, 0.2, 0.3]])
    got = np.array([[0.15, 0.25, 0.6, 0.95], [0.8, 0.3, 0.2, 0.1]])
    t = loop.topk(ref, got, 2)
    # Both argmaxes survive, so a top-1-only reading would call this fine;
    # row 1's top-2 set moved from {0,3} to {0,1}.
    assert t["k"] == 2 and t["top1"] == 1.0
    assert t["overlap"] == pytest.approx(0.75)


def test_topk_is_silent_for_an_output_with_no_axis_to_rank():
    assert loop.topk(np.array([0.5]), np.array([0.4]), 5) is None
    assert loop.topk(np.zeros((1, 1), np.float32), np.zeros((1, 1),
                                                             np.float32),
                     5) is None


def test_topk_clamps_k_to_the_class_count():
    t = loop.topk(np.array([[0.1, 0.9]]), np.array([[0.2, 0.8]]), 5)
    assert t["k"] == 2 and t["overlap"] == 1.0


# -------------------------------------------------------- input  ownership


def test_inputs_come_from_the_named_npy_files(tmp_path):
    x = np.arange(4, dtype=np.float32).reshape(1, 4)
    np.save(tmp_path / "x.npy", x)
    got = loop.build_inputs([spec("x")], args(inputs=str(tmp_path)))
    assert np.array_equal(got["x"], x)


def test_one_file_covers_a_single_input_whatever_it_is_called(tmp_path):
    # benchmark/resnet.py --dump-input leaves py_input_nchw.npy behind, and the
    # model's input is named `input`; forcing a rename would make the common
    # CNN path look broken on its first try.
    x = np.ones((1, 3, 4, 4), dtype=np.float32)
    np.save(tmp_path / "py_input_nchw.npy", x)
    got = loop.build_inputs([spec("input", dims=(1, 3, 4, 4))],
                            args(inputs=str(tmp_path)))
    assert np.array_equal(got["input"], x)


def test_the_models_declared_dtype_wins_and_says_so(tmp_path, capsys):
    np.save(tmp_path / "x.npy", np.arange(4, dtype=np.float64).reshape(1, 4))
    got = loop.build_inputs([spec("x", "float16")], args(inputs=str(tmp_path)))
    assert got["x"].dtype == np.float16
    assert "cast float64 -> float16" in capsys.readouterr().out


def test_an_oversized_npy_is_reshaped_not_silently_dropped(tmp_path, capsys):
    np.save(tmp_path / "x.npy", np.arange(4, dtype=np.float32))
    got = loop.build_inputs([spec("x")], args(inputs=str(tmp_path)))
    assert got["x"].shape == (1, 4)
    assert "reshape (4,) -> (1, 4)" in capsys.readouterr().out


def test_an_input_with_the_wrong_element_count_fails(tmp_path):
    np.save(tmp_path / "x.npy", np.arange(5, dtype=np.float32))
    with pytest.raises(loop.LoopError, match="5 element\\(s\\) given"):
        loop.build_inputs([spec("x")], args(inputs=str(tmp_path)))


def test_a_missing_input_tensor_is_an_error_not_a_silent_random(tmp_path):
    # Two inputs, one file: the one-file convenience does not apply, so 'x'
    # really is absent and a random fallback must not paper over that.
    np.save(tmp_path / "w.npy", np.zeros((1, 4), dtype=np.float32))
    with pytest.raises(loop.LoopError, match="dump-input"):
        loop.build_inputs([spec("x"), spec("w")], args(inputs=str(tmp_path)))


def test_random_fill_is_reproducible_across_calls():
    a = loop.build_inputs([spec("x")], args(random_fill=True))
    b = loop.build_inputs([spec("x")], args(random_fill=True))
    assert np.array_equal(a["x"], b["x"])
    # The stability test this replaced: name-keyed seeding via hash() is salted
    # per process, so two runs of the script disagreed about the "same" input.
    assert np.array_equal(loop.random_input(spec("x"), 7),
                          loop.random_input(spec("x"), 7))


def test_random_fill_refuses_an_integer_input():
    with pytest.raises(loop.LoopError, match="indexes out of range"):
        loop.random_input(spec("ids", "int64"), 0)


def test_random_fill_refuses_a_dynamic_dim():
    with pytest.raises(loop.LoopError, match="pass --shape"):
        loop.random_input(spec("x", dims=(-1, 4)), 0)


def test_random_fill_is_deterministic_and_dtype_true():
    a = loop.random_input(spec("x", "float16", (2, 2)), 1)
    b = loop.random_input(spec("x", "float16", (2, 2)), 1)
    c = loop.random_input(spec("y", "float16", (2, 2)), 1)
    assert a.dtype == np.float16 and np.array_equal(a, b)
    assert not np.array_equal(a, c)  # the name seeds it, per-input streams


def test_an_input_dtype_the_reader_does_not_know_fails(tmp_path):
    np.save(tmp_path / "x.npy", np.zeros(4, dtype=np.float32))
    with pytest.raises(loop.LoopError, match="no numpy view"):
        loop.build_inputs([spec("x", "bfloat16")], args(inputs=str(tmp_path)))


# ------------------------------------------------------------ vkop-bench run


STUB = textwrap.dedent("""
    import json, os, sys
    from pathlib import Path
    import numpy as np

    argv = sys.argv[1:]
    def after(flag):
        return argv[argv.index(flag) + 1]

    inputs, outputs = Path(after("--load-inputs")), Path(after("--save-outputs"))
    x = np.load(inputs / "x.npy").astype(np.float32)
    y = (float(os.environ.get("VKOP_STUB_SCALE", "2")) * x).astype(np.float32)
    y.tofile(outputs / "y.bin")
    (outputs / "manifest.tsv").write_text(
        "# vkop-bench outputs\\n"
        "y\\tfloat32\\t%s\\t%d\\n" % ("x".join(map(str, y.shape)), y.nbytes))
    (outputs / "seen.json").write_text(json.dumps({
        "argv": argv,
        "x": [str(x.dtype), list(x.shape), float(x.astype(np.float64).sum())]}))
    if os.environ.get("VKOP_STUB_FAIL"):
        sys.stderr.write("shader build failed\\n")
        sys.exit(3)
""")


@pytest.fixture
def bench(tmp_path):
    path = tmp_path / "stub-bench"
    path.write_text(f"#!{sys.executable}\n{STUB}")
    path.chmod(0o755)
    return path


def run_bench(bench, tmp_path, args_, vkopbin="m.vkopbin"):
    in_dir = tmp_path / "in"
    in_dir.mkdir(exist_ok=True)
    np.save(in_dir / "x.npy", np.arange(4, dtype=np.float32).reshape(1, 4))
    return loop.run_vkopbench(bench, tmp_path / vkopbin, in_dir,
                              tmp_path / "out", args_)


def test_the_bench_is_called_once_and_dumps(bench, tmp_path):
    run_bench(bench, tmp_path, args())
    seen = json.loads((tmp_path / "out" / "seen.json").read_text())
    a = seen["argv"]
    assert a[0].endswith("m.vkopbin")
    # One Run() and no warm-up: the dump must be the run the inputs describe.
    assert a[a.index("--iters") + 1] == "1"
    assert a[a.index("--warmup") + 1] == "0"
    assert a[a.index("--precision") + 1] == "fp16"
    assert a[a.index("--backend") + 1] == "buffer"
    assert "--shape" not in a


def test_shape_and_runtime_flags_reach_bench(bench, tmp_path):
    run_bench(bench, tmp_path, args(shape=["x:1x4", "z:2"], precision="fp32",
                                    backend="image"))
    a = json.loads((tmp_path / "out" / "seen.json").read_text())["argv"]
    assert a[a.index("--shape") + 1] == "x:1x4"
    assert a[a.index("--shape") + 3] == "z:2"
    assert a[a.index("--precision") + 1] == "fp32"
    assert a[a.index("--backend") + 1] == "image"


def test_a_failed_run_is_reported_with_its_tail(bench, tmp_path,
                                                monkeypatch):
    monkeypatch.setenv("VKOP_STUB_FAIL", "1")
    with pytest.raises(loop.LoopError, match="exited 3"):
        run_bench(bench, tmp_path, args())


def test_an_explicit_bench_that_is_not_runnable_stops_the_run(bench,
                                                              tmp_path):
    # Falling through to the build tree's binary would measure a different
    # build than the one --bench named.
    with pytest.raises(loop.LoopError, match="not an executable file"):
        loop.find_bench(str(tmp_path / "nowhere"))
    assert loop.find_bench(str(bench)) == bench


def test_the_default_bench_is_the_repo_build(bench):
    found = loop.find_bench(None)
    assert found.name == "vkop-bench" and found.is_file()


# ------------------------------------------------------------ the whole loop


class FakeView:
    """The .vkopbin side: one fp32 input x[1,4], one fp32 output y[1,4]."""

    def __init__(self, path):
        self.path = path

    def shapes(self):
        return [spec("x")]

    def outputs(self):
        return [spec("y")]


class FakeSession:
    def get_inputs(self):
        return [Namespace(name="x", type="tensor(float)")]

    def get_outputs(self):
        return [Namespace(name="y", type="tensor(float)")]


@pytest.fixture
def models(tmp_path, monkeypatch):
    """The files the loop opens, plus the two sides it talks to.

    ORT's half is replaced by y = 2x, which is what the stub bench computes, so
    a green run means the plumbing agrees end to end; VKOP_STUB_SCALE moves the
    runtime side off that contract and has to be caught.
    """
    vkopbin = tmp_path / "m.vkopbin"
    vkopbin.write_bytes(b"")
    onnx = tmp_path / "m.onnx"
    onnx.write_bytes(b"")
    fed = {}

    def fake_ort_reference(sess, arrays):
        fed["x"] = arrays["x"].copy()
        return {"y": (2.0 * arrays["x"]).astype(np.float32)}

    monkeypatch.setattr(loop, "ModelView", FakeView)
    monkeypatch.setattr(loop, "ort_session",
                        lambda p, t: FakeSession())
    monkeypatch.setattr(loop, "ort_reference", fake_ort_reference)
    return Namespace(vkopbin=vkopbin, onnx=onnx, fed=fed)


def loop_args(tmp_path, bench, models, **kw):
    return args(vkopbin=str(models.vkopbin), onnx=str(models.onnx),
                bench=str(bench), inputs=str(tmp_path / "in"),
                work_dir=str(tmp_path / "work"), **kw)


def make_input_dir(tmp_path):
    d = tmp_path / "in"
    d.mkdir(exist_ok=True)
    x = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
    np.save(d / "x.npy", x)
    return x


def test_both_sides_get_the_same_input_tensor(tmp_path, bench, models, capsys):
    x = make_input_dir(tmp_path)
    assert loop.run(loop_args(tmp_path, bench, models)) == 0
    seen = json.loads(
        (tmp_path / "work" / "out" / "seen.json").read_text())["x"]
    assert seen == [str(x.dtype), list(x.shape), float(x.sum())]
    assert np.array_equal(models.fed["x"], x)
    out = capsys.readouterr().out
    assert "1/1 shown output(s) pass" in out
    assert "maxabs" in out and "0.000e+00" in out


def test_the_loop_fails_when_the_runtime_disagrees(tmp_path, bench, models,
                                                   monkeypatch, capsys):
    make_input_dir(tmp_path)
    monkeypatch.setenv("VKOP_STUB_SCALE", "3")  # y = 3x against ORT's 2x
    assert loop.run(loop_args(tmp_path, bench, models)) == 1
    out = capsys.readouterr().out
    assert "FAIL" in out and "cos " in out


def test_an_output_ort_does_not_have_is_unmatched_not_guessed(
        tmp_path, bench, models, monkeypatch, capsys):
    make_input_dir(tmp_path)
    monkeypatch.setattr(loop, "ort_reference",
                        lambda sess, arrays: {"something_else":
                                              np.zeros((1, 4), np.float32)})
    assert loop.run(loop_args(tmp_path, bench, models)) == 1
    out = capsys.readouterr().out
    assert "UNMATCHED" in out
    assert "onnxruntime has no output 'y'" in out


def test_topk_reports_only_when_asked_for(tmp_path, bench, models, capsys):
    make_input_dir(tmp_path)
    loop.run(loop_args(tmp_path, bench, models))
    assert "top-" not in capsys.readouterr().out
    loop.run(loop_args(tmp_path, bench, models, topk=2))
    assert "top-2 set overlap 1.0000" in capsys.readouterr().out


def test_a_model_with_no_input_files_is_exit_two(tmp_path, bench, models):
    (tmp_path / "in").mkdir(exist_ok=True)
    with pytest.raises(loop.LoopError, match="holds no .npy files"):
        loop.run(loop_args(tmp_path, bench, models))


def test_an_input_the_vkopbin_lacks_is_called_out(tmp_path, bench, models,
                                                  monkeypatch):
    make_input_dir(tmp_path)
    np.save(tmp_path / "in" / "q.npy", np.zeros((1, 4), np.float32))
    monkeypatch.setattr(FakeView, "shapes", lambda self: [spec("x"),
                                                           spec("q")])
    with pytest.raises(loop.LoopError,
                       match=r"input 'q' is in the \.vkopbin but not in"):
        loop.run(loop_args(tmp_path, bench, models))


def test_a_named_work_dir_is_kept_and_a_temp_one_is_not(tmp_path, bench,
                                                        models, capsys):
    make_input_dir(tmp_path)
    named = tmp_path / "work_a"
    loop.run(args(vkopbin=str(models.vkopbin), onnx=str(models.onnx),
                  bench=str(bench), inputs=str(tmp_path / "in"),
                  work_dir=str(named)))
    # --work-dir is a promise: the caller asked for those bytes at that path.
    assert (named / "out" / "manifest.tsv").is_file()

    loop.run(args(vkopbin=str(models.vkopbin), onnx=str(models.onnx),
                  bench=str(bench), inputs=str(tmp_path / "in")))
    line = [l for l in capsys.readouterr().out.splitlines()
            if l.startswith("work dir:")][-1]
    staged = Path(line.split("work dir: ")[1].split(" (")[0])
    assert line.endswith("(removed)") and not staged.exists()


def test_main_reports_a_missing_vkopbin_as_exit_two(tmp_path, monkeypatch,
                                                    capsys):
    monkeypatch.setattr(sys, "argv", ["vkop_vs_ort.py",
                                      str(tmp_path / "gone.vkopbin")])
    assert loop.main() == 2
    assert "not found" in capsys.readouterr().err
