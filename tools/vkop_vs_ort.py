#!/usr/bin/env python3
"""End-to-end accuracy loop: a .vkopbin against the ONNX Runtime reference.

    python3 tools/vkop_vs_ort.py model.vkopbin --onnx model.onnx --inputs DIR

A conversion can be mechanically fine -- every op has a kernel, the graph
loads, the run finishes -- and still compute wrong numbers. The question that
catches that is "does what came out of the GPU match what ONNX Runtime gives
for the same input", and answering it honestly needs three things at once.

ONE input tensor, not two look-alikes. The array is built here (from
`--inputs DIR/<name>.npy`, or a seeded random fill for float inputs), fed to
onnxruntime as-is and written to the directory vkop-bench loads with
--load-inputs. Nothing resizes or renormalizes on either side, so a divergence
belongs to the kernels, not to the preprocessing. For a JPEG, dump the tensor
first with `benchmark/resnet.py --dump-input`: PIL's bilinear and the runtime's
stb triangle filter are not the same resize, and a comparison across them
would measure the resize.

An unambiguous reading of what the GPU wrote. vkop-bench dumps raw bytes plus
a manifest.tsv giving the byte layout, the shape the run actually had, and the
byte count. This script asserts prod(dims) x itemsize == nbytes and fails the
output if it disagrees. Re-deriving those from the .vkopbin would let a
mis-shaped or reinterpreted array decode into plausible numbers -- a
difference that reads as a rounding error rather than as the misread it is.
Output NAMES are the pairing contract: the converter keeps the ONNX graph's
names (verified against llm, visual and the CNN fixtures), so a vkop output
with no same-named ORT output is reported as UNMATCHED rather than guessed at.

A threshold that means something per format. fp32-vs-fp32 should clear
cos >= 0.999 with almost no outliers; an int8 weight-only model legitimately
will not, and measuring how far it drifts is the point. The gates are flags
(--cos-min, --rtol, --atol, --bad-frac-max) so every run states the bar it was
judged against instead of inheriting a magic number. --rtol bands the outlier
count against the reference's largest element rather than against each
element, because per-element relative error on near-zero values measures the
storage format's rounding floor instead of the op's correctness (measured, not
taste: see compare()). Integer outputs (token ids, positions, cache indices)
are compared exactly -- a wrong index has no tolerance band, and --topk over
one is meaningless.

This is what makes the per-node precision experiment in onnx2vkop/exclusion.py
usable: convert with `-q int4 --exclude-scope model.layers.0`, run this on both
artifacts, read which number moved. Same input, same shaders, one precision
difference, so the delta is attributable to the layer the flag named.

Defaults worth knowing: --onnx is the .vkopbin's own stem with .onnx, --bench is
build/benchmark/vkop-bench (then $PATH), and when the model has exactly one
input a --inputs directory with exactly one .npy is used for it whatever the
file is called -- that is what benchmark/resnet.py --dump-input leaves behind.

onnxruntime is deliberately not a converter dependency (it is not in the
wheel's requires) and neither is a GPU here; both are checked up front and
reported with the fix rather than as a traceback.

    exit 0  every output passed
    exit 1  at least one output failed its gates
    exit 2  the loop itself could not run (missing binary/model/ORT, crash)
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import zlib
from pathlib import Path

import numpy as np

# The converter package lives in model/pypi and is not on the path of a
# checkout that has not pip-installed it. ModelView is the only thing needed
# from it: the .vkopbin's declared inputs are what make the shared input
# tensor well-defined.
_REPO = Path(__file__).resolve().parents[1]
_PKG_DIR = str(_REPO / "model" / "pypi")
if _PKG_DIR not in sys.path:
    sys.path.append(_PKG_DIR)

try:
    from onnx2vkop.inspect import ModelView
except ImportError:
    raise SystemExit(
        "Error: cannot import onnx2vkop (expected under "
        f"{_PKG_DIR}).")

# manifest.tsv "bytes-as" -> the numpy view of those bytes. These are the
# storage types vkop_bench.cpp's save_outputs writes, NOT the graph's semantic
# format: an fp8 or int4 payload is storage uint8 whatever it means.
BYTES_AS = {
    "float32": np.dtype(np.float32),
    "float16": np.dtype(np.float16),
    "int32": np.dtype(np.int32),
    "int64": np.dtype(np.int64),
    "int8": np.dtype(np.int8),
    "uint8": np.dtype(np.uint8),
}
# onnxruntime's spelling of an element type, for feeding the reference.
ORT_TYPE = {
    "tensor(float)": np.float32, "tensor(float16)": np.float16,
    "tensor(int64)": np.int64, "tensor(int32)": np.int32,
    "tensor(uint8)": np.uint8, "tensor(bool)": np.bool_,
}


class LoopError(RuntimeError):
    """The comparison could not be made at all -- exit 2, not a failed gate."""


# ----------------------------------------------------------------- runtime side


def parse_manifest(path: Path):
    """Rows of {name, bytes_as, dims, nbytes} from a vkop-bench manifest.tsv."""
    if not path.is_file():
        raise LoopError(
            f"{path} missing: without the manifest the dumped .bin files have "
            "no dtype or shape to be read with")
    rows = []
    for line in path.read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) != 4:
            raise LoopError(f"{path}: malformed row {line!r}")
        name, bytes_as, dims, nbytes = parts
        if bytes_as not in BYTES_AS:
            raise LoopError(
                f"{path}: {name}: bytes-as {bytes_as!r} has no numpy view; add "
                "one only together with the branch in vkop_bench.cpp that "
                "writes it")
        rows.append({"name": name, "bytes_as": bytes_as,
                     "dims": [int(d) for d in dims.split("x") if d != ""],
                     "nbytes": int(nbytes)})
    if not rows:
        raise LoopError(f"{path}: no output rows")
    return rows


def read_output(outdir: Path, row):
    """The array one manifest row describes, or a LoopError explaining why not.

    The size assertion is the reason the manifest exists: a .bin whose byte
    count disagrees with dims x dtype is a misread or a truncated write, and
    both would otherwise decode into a wrong-shaped array that then shows up
    as a numerical difference.
    """
    bin_path = outdir / (row["name"] + ".bin")
    if not bin_path.is_file():
        raise LoopError(f"{bin_path} is in the manifest but not on disk")
    raw = bin_path.read_bytes()
    if len(raw) != row["nbytes"]:
        raise LoopError(f"{bin_path.name}: manifest says {row['nbytes']} "
                        f"bytes, file has {len(raw)}")
    dt = BYTES_AS[row["bytes_as"]]
    elems = int(np.prod(row["dims"])) if row["dims"] else 0
    if elems * dt.itemsize != row["nbytes"]:
        raise LoopError(
            f"{row['name']}: dims={row['dims']} x {dt.itemsize}B = "
            f"{elems * dt.itemsize} != the {row['nbytes']} bytes written")
    return np.frombuffer(raw, dtype=dt).reshape(row["dims"])


def run_vkopbench(bench, vkopbin, inputs_dir, out_dir, args):
    """Exactly one Run(), then its outputs dumped. Returns the manifest rows."""
    cmd = [str(bench), str(vkopbin), "--load-inputs", str(inputs_dir),
           "--save-outputs", str(out_dir), "--iters", "1", "--warmup", "0",
           "--precision", args.precision, "--backend", args.backend]
    for shape in args.shape or []:
        cmd += ["--shape", shape]
    out_dir.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if args.verbose:
        sys.stdout.write(proc.stdout)
        sys.stderr.write(proc.stderr)
    if proc.returncode != 0:
        raise LoopError(
            f"vkop-bench exited {proc.returncode}:\n"
            + "\n".join((proc.stdout + "\n" + proc.stderr).splitlines()[-15:]))
    return parse_manifest(out_dir / "manifest.tsv")


# --------------------------------------------------------------------- ORT side


def ort_session(onnx_path: Path, threads):
    try:
        import onnxruntime as ort
    except ImportError:
        raise LoopError("onnxruntime is not installed; it is deliberately not "
                        "a converter dependency. `pip install onnxruntime`.")
    opts = ort.SessionOptions()
    if threads:
        opts.intra_op_num_threads = int(threads)
    try:
        return ort.InferenceSession(str(onnx_path), sess_options=opts,
                                    providers=["CPUExecutionProvider"])
    except Exception as e:
        raise LoopError(f"onnxruntime could not load {onnx_path}: {e}")


def declared_shape(spec):
    """The .vkopbin's dims, or None when one is still a dynamic sentinel."""
    dims = spec["dims"]
    return tuple(dims) if dims and min(dims) >= 0 else None


def random_input(spec, seed: int):
    if spec["dtype"] not in ("float32", "float16"):
        raise LoopError(
            f"input {spec['name']!r} is {spec['dtype']}, and random fill is "
            "only defined for float inputs: noise in an int64 token id or "
            "position indexes out of range, so both sides would be wrong "
            "together and the numbers would mean nothing")
    dims = declared_shape(spec)
    if dims is None:
        raise LoopError(
            f"input {spec['name']!r} has a dynamic dim; pass --shape "
            "NAME:d1xd2 so the fill knows its size (or give it a .npy)")
    # crc32, not hash(): Python salts str hashing per process, which would
    # make the "same input" array differ between two runs of this script.
    rng = np.random.default_rng(seed + zlib.crc32(spec["name"].encode()))
    return rng.standard_normal(dims).astype(spec["dtype"])


def build_inputs(vk_inputs, args):
    """The one array per graph input, shared by ORT and the runtime.

    The .vkopbin's dtype and dims win: an npy that arrives as another float
    width is cast, and a differently-shaped one is reshaped when the element
    count allows, both printed. A cast the two sides cannot agree on is not a
    comparison, so the array is finalized HERE and only then written and fed.
    """
    files = {}
    if args.inputs:
        d = Path(args.inputs)
        if not d.is_dir():
            raise LoopError(f"--inputs {d} is not a directory")
        files = {npy.stem: npy for npy in sorted(d.glob("*.npy"))}
        if not files:
            raise LoopError(f"--inputs {d} holds no .npy files")
        # The checked-in CNN dumps are named py_input_nchw.npy, not after the
        # model's input, so a one-input/one-file pair needs no name agreement.
        if len(vk_inputs) == 1 and len(files) == 1:
            files = {vk_inputs[0]["name"]: next(iter(files.values()))}

    arrays = {}
    for spec in vk_inputs:
        name, dtype = spec["name"], spec["dtype"]
        if name in files:
            arr = np.load(files[name])
        elif args.random_fill:
            arr = random_input(spec, args.seed)
            print(f"[input] {name}: random fill {arr.shape} {dtype}")
        else:
            raise LoopError(
                f"no input tensor for {name!r}: --inputs has no {name}.npy "
                "and --random-fill is off. Dump the real one first "
                "(benchmark/resnet.py --dump-input) or pass --random-fill.")
        np_dt = BYTES_AS.get(dtype)
        if np_dt is None:
            raise LoopError(
                f"input {name!r}: the .vkopbin declares {dtype!r}, which has "
                "no numpy view here")
        if arr.dtype != np_dt:
            print(f"[input] {name}: cast {arr.dtype} -> {dtype}")
            arr = arr.astype(np_dt)
        want = declared_shape(spec)
        if want is not None and arr.shape != want:
            if arr.size != int(np.prod(want)):
                raise LoopError(
                    f"input {name!r}: {arr.size} element(s) given, the model "
                    f"wants {want} ({int(np.prod(want))})")
            print(f"[input] {name}: reshape {arr.shape} -> {want}")
            arr = arr.reshape(want)
        arrays[name] = np.ascontiguousarray(arr)

    return arrays


def ort_reference(sess, arrays):
    feed = {}
    for i in sess.get_inputs():
        arr = arrays[i.name]
        want = ORT_TYPE.get(i.type)
        if want is not None and arr.dtype != np.dtype(want):
            arr = arr.astype(want)
        feed[i.name] = arr
    outs = sess.run(None, feed)
    return {o.name: np.asarray(v) for o, v in zip(sess.get_outputs(), outs)}


# --------------------------------------------------------------------- metrics


def compare(ref, got, rtol, atol, cos_min, bad_frac_max):
    """One output's metrics, with `ok` and the reasons it is not ok."""
    m = {"shape": list(got.shape), "nan": 0, "inf": 0, "cos": float("nan"),
         "maxabs": float("nan"), "meanabs": float("nan"),
         "rel_l2": float("nan"), "bad_frac": float("nan"),
         "exact_mismatch": None, "reasons": []}
    if ref.shape != got.shape:
        m["reasons"].append(f"reference shape {list(ref.shape)} vs runtime "
                            f"{list(got.shape)}")
        m["ok"] = False
        return m

    # np.inexact covers float and complex; anything else is ids/positions/
    # masks, where a single differing element is a wrong answer, not a
    # rounding difference.
    if not np.issubdtype(ref.dtype, np.inexact):
        diff = np.not_equal(ref, got)
        m["exact_mismatch"] = int(diff.sum())
        m["bad_frac"] = float(diff.mean()) if diff.size else 0.0
        m["maxabs"] = (float(np.abs(ref.astype(np.int64)
                                    - got.astype(np.int64)).max())
                       if diff.size else 0.0)
        if m["exact_mismatch"]:
            m["reasons"].append(f"{m['exact_mismatch']} element(s) differ; an "
                                "integer output has no tolerance band")
        m["ok"] = not m["reasons"]
        return m

    a, b = ref.astype(np.float64), got.astype(np.float64)
    m["nan"] = int(np.isnan(b).sum())
    m["inf"] = int(np.isinf(b).sum()) - int(np.isinf(a).sum())
    d = np.abs(a - b)
    if not d.size:
        # A 0-element output (a kv_len=0 cache slice) has no error to report;
        # it passes on the shape equality already checked above.
        m["cos"], m["rel_l2"], m["bad_frac"] = 1.0, 0.0, 0.0
        m["ok"] = True
        return m
    m["maxabs"] = float(d.max())
    m["meanabs"] = float(d.mean())
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    m["cos"] = float(a.ravel() @ b.ravel() / (na * nb)) if na and nb else 0.0
    m["rel_l2"] = float(np.linalg.norm(a - b) / na) if na else 0.0
    # The outlier band scales with the tensor, not with each element: an
    # fp16-storage run on the Add+Conv fixture puts 11.1% of elements outside
    # atol + rtol*|ref| while its cosine is 0.999995, because the flagged ones
    # are the small-magnitude elements (median |ref| 0.16 against 0.73 overall)
    # carrying ordinary accumulation error from the large intermediates they
    # were computed from. Against max|ref| the same run is 0.0000%.
    scale = float(np.abs(a).max())
    band = atol + rtol * scale
    m["bad_frac"] = float((d > band).mean())
    if m["nan"]:
        m["reasons"].append(f"{m['nan']} NaN(s) in the runtime output")
    if m["inf"] > 0:
        m["reasons"].append(f"{m['inf']} more Inf(s) than the reference")
    if not m["cos"] >= cos_min:
        m["reasons"].append(f"cos {m['cos']:.6f} < {cos_min}")
    if not m["bad_frac"] <= bad_frac_max:
        m["reasons"].append(
            f"{m['bad_frac']:.4%} of elements beyond atol + rtol*max|ref| "
            f"= {band:.4g} (limit {bad_frac_max:.4%})")
    m["ok"] = not m["reasons"]
    return m


def topk(ref, got, k):
    """Top-1 agreement and mean top-k index overlap along the last axis."""
    if ref.ndim < 1 or ref.shape[-1] < 2:
        return None
    kk = max(1, min(k, ref.shape[-1]))
    top1 = float((np.argmax(ref, -1) == np.argmax(got, -1)).mean())
    r = np.argsort(ref, -1)[..., -kk:]
    g = np.argsort(got, -1)[..., -kk:]
    overlap = float(np.mean([
        len(set(rv.ravel().tolist()) & set(gv.ravel().tolist())) / kk
        for rv, gv in zip(r.reshape(-1, kk), g.reshape(-1, kk))]))
    return {"k": kk, "top1": top1, "overlap": overlap}


def find_bench(explicit):
    cands = [explicit] if explicit else []
    cands += [str(_REPO / "build" / "benchmark" / "vkop-bench"),
              "build/benchmark/vkop-bench"]
    for c in cands:
        p = Path(c)
        if p.is_file() and os.access(p, os.X_OK):
            return p
    which = shutil.which("vkop-bench")
    if which:
        return Path(which)
    raise LoopError(f"vkop-bench not found (tried {', '.join(cands)}, $PATH); "
                    "build it with `cmake --build build --target vkop_bench` "
                    "or pass --bench")


# ---------------------------------------------------------------------------


def report_row(name, note, m, t):
    shape = "x".join(str(d) for d in m["shape"]) or "scalar"
    if note:
        shape += f" ({note})"
    tk = f"{t['top1']:6.3f}" if t else "     -"
    print(f"{name:<30} {shape:<18} {m['maxabs']:>9.3e} {m['meanabs']:>9.3e} "
          f"{m['rel_l2']:>9.3e} {m['cos']:>10.6f} {100 * m['bad_frac']:>8.4f} "
          f"{tk} {'OK' if m['ok'] else 'FAIL'}")
    for reason in m["reasons"]:
        print(f"{'':<30} ! {reason}")
    if t:
        print(f"{'':<30} top-{t['k']} set overlap {t['overlap']:.4f}")


def main():
    ap = argparse.ArgumentParser(
        description="Compare a converted .vkopbin against ONNX Runtime on one "
                    "shared input tensor, per output, with stated gates.")
    ap.add_argument("vkopbin", help="the converted model to measure")
    ap.add_argument("--onnx", help="its source model (default: same stem)")
    ap.add_argument("--inputs", metavar="DIR",
                    help="DIR/<input name>.npy: the tensor both sides run on")
    ap.add_argument("--random-fill", action="store_true",
                    help="fill float inputs with seeded noise instead of "
                         "failing; real accuracy needs real inputs")
    ap.add_argument("--seed", type=int, default=0xC0FFEE)
    ap.add_argument("--bench", help="path to vkop-bench")
    ap.add_argument("--precision", default="fp16", choices=("fp16", "fp32"))
    ap.add_argument("--backend", default="buffer", choices=("buffer", "image"))
    ap.add_argument("--shape", action="append", metavar="NAME:D1xD2",
                    help="resize a dynamic input (vkop-bench --shape); repeat")
    ap.add_argument("--ort-threads", type=int, default=None)
    ap.add_argument("--cos-min", type=float, default=0.999,
                    help="minimum cosine per output (default 0.999)")
    ap.add_argument("--rtol", type=float, default=1e-2,
                    help="outlier band as a fraction of the reference's "
                         "largest element, NOT per element (see compare())")
    ap.add_argument("--atol", type=float, default=1e-3,
                    help="absolute part of the same band")
    ap.add_argument("--bad-frac-max", type=float, default=1e-3,
                    help="largest fraction of elements allowed beyond "
                         "atol + rtol*max|ref| (default 0.1%%)")
    ap.add_argument("--topk", type=int, default=0, metavar="K",
                    help="also report top-1 and top-K overlap (5 for a "
                         "classifier); needs a logits-shaped output")
    ap.add_argument("--limit", type=int, default=0, metavar="N",
                    help="report only the first N outputs (a 29-tensor KV "
                         "dump is otherwise a wall of rows)")
    ap.add_argument("--work-dir", help="stage/dump directory (default: a temp "
                                       "directory, removed unless --keep)")
    ap.add_argument("--keep", action="store_true")
    ap.add_argument("-v", "--verbose", action="store_true",
                    help="echo vkop-bench's stdout and stderr")
    args = ap.parse_args()

    try:
        return run(args)
    except LoopError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2


def run(args):
    vkopbin = Path(args.vkopbin)
    if not vkopbin.is_file():
        raise LoopError(f"{vkopbin} not found")
    onnx_path = Path(args.onnx) if args.onnx else vkopbin.with_suffix(".onnx")
    if not onnx_path.is_file():
        raise LoopError(f"reference model {onnx_path} not found (pass --onnx)")
    bench = find_bench(args.bench)

    view = ModelView(str(vkopbin))
    vk_inputs = view.shapes()
    sess = ort_session(onnx_path, args.ort_threads)
    arrays = build_inputs(vk_inputs, args)
    ort_names = {i.name for i in sess.get_inputs()}
    for name in arrays:
        if name not in ort_names:
            raise LoopError(
                f"input {name!r} is in the .vkopbin but not in {onnx_path}; "
                "the two files are not the same model")

    work = Path(args.work_dir) if args.work_dir else Path(
        tempfile.mkdtemp(prefix="vkop_vs_ort."))
    work.mkdir(parents=True, exist_ok=True)
    in_dir, out_dir = work / "in", work / "out"
    in_dir.mkdir(parents=True, exist_ok=True)
    keep = args.keep or args.work_dir is not None
    try:
        for name, arr in arrays.items():
            np.save(in_dir / f"{name}.npy", arr)
        rows = run_vkopbench(bench, vkopbin, in_dir, out_dir, args)
        refs = ort_reference(sess, arrays)

        spec = ", ".join(f"{n} {v.dtype}{list(v.shape)}"
                         for n, v in arrays.items())
        print(f"\n{vkopbin.name} vs {onnx_path.name}"
              f"  [backend={args.backend} precision={args.precision}]")
        print(f"inputs: {spec}")
        print(f"gates: cos>={args.cos_min} atol={args.atol} rtol={args.rtol} "
              f"out<={args.bad_frac_max:.4%}"
              + (f" topk={args.topk}" if args.topk else ""))
        print(f"\n{'output':<30} {'shape':<18} {'maxabs':>9} {'meanabs':>9} "
              f"{'relL2':>9} {'cos':>10} {'out%':>8} {'top1':>6} {'ok':>4}")

        shown = rows[:args.limit] if args.limit else rows
        failed = 0
        for row in shown:
            got = read_output(out_dir, row)
            ref = refs.get(row["name"])
            if ref is None:
                print(f"{row['name']:<30} {row['dims']!s:<18} "
                      f"{'':>9} {'':>9} {'':>9} {'':>10} {'':>8} {'-':>6} "
                      f"FAIL")
                print(f"{'':<30} ! UNMATCHED: onnxruntime has no output "
                      f"{row['name']!r} (it has {sorted(refs)[:4]}"
                      f"{'...' if len(refs) > 4 else ''})")
                failed += 1
                continue
            m = compare(ref, got, args.rtol, args.atol, args.cos_min,
                        args.bad_frac_max)
            t = topk(ref, got, args.topk) if args.topk else None
            failed += 0 if m["ok"] else 1
            report_row(row["name"], "", m, t)
        if len(rows) > len(shown):
            print(f"... {len(rows) - len(shown)} more output(s) not shown "
                  f"(--limit {args.limit})")
        print(f"\nRESULT: {len(shown) - failed}/{len(shown)} shown output(s) "
              f"pass" + ("" if not failed else f", {failed} FAIL"))
        print(f"work dir: {work}" + ("" if keep else " (removed)"))
        return 0 if failed == 0 else 1
    finally:
        if not keep:
            shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
