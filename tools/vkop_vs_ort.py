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
import re
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
    # A bool input is an int8_t tensor in the runtime; vkop-bench fills it
    # byte for byte from a 1-byte numpy bool array.
    "bool": np.dtype(np.bool_),
    # An --act bf16 artifact's I/O. numpy has no bfloat16 (and ml_dtypes is
    # deliberately not a dependency), so the WORDS travel as uint16 -- the .npy
    # vkop-bench loads, the .bin it dumps -- and the VALUES are float32 here:
    # bf16 is fp32's upper half, so the widening is exact (see bf16_to_f32).
    "bfloat16": np.dtype(np.uint16),
}


def f32_to_bf16(values) -> np.ndarray:
    """float32 -> bf16 words (uint16), round to nearest even -- the rounding
    the shaders, core::float_to_bf16 and onnx2vkop's bf16 weights use."""
    a = np.ascontiguousarray(values, dtype=np.float32)
    u = a.view(np.uint32)
    with np.errstate(over="ignore"):
        r = (u + (np.uint32(0x7FFF) + ((u >> np.uint32(16)) & np.uint32(1)))
             ) >> np.uint32(16)
    out = r.astype(np.uint16)
    nan = np.isnan(a)
    if nan.any():  # keep NaN a (quiet) NaN: the carry could make it inf
        out[nan] = ((u[nan] >> np.uint32(16)) | np.uint32(0x40)).astype(
            np.uint16)
    return out


def bf16_to_f32(words) -> np.ndarray:
    """bf16 words (uint16) -> float32. Exact."""
    w = np.ascontiguousarray(words, dtype=np.uint16)
    return (w.astype(np.uint32) << np.uint32(16)).view(np.float32)
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
    # vkop-bench writes '/' in an output name as '_' (a path separator).
    bin_path = outdir / (row["name"].replace("/", "_") + ".bin")
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
    arr = np.frombuffer(raw, dtype=dt).reshape(row["dims"])
    if row["bytes_as"] == "bfloat16":
        return bf16_to_f32(arr).reshape(row["dims"])
    return arr


def run_vkopbench(bench, vkopbin, inputs_dir, out_dir, args, node_dump_cwd=None):
    """Exactly one Run(), then its outputs dumped. Returns the manifest rows.

    With `node_dump_cwd`, the run also writes every node's output
    (VKOP_NODE_DUMP) into `<node_dump_cwd>/node_dump/`."""
    cmd = [str(Path(bench).resolve()), str(Path(vkopbin).resolve()),
           "--load-inputs", str(Path(inputs_dir).resolve()),
           "--save-outputs", str(Path(out_dir).resolve()), "--iters", "1",
           "--warmup", "0", "--precision", args.precision,
           "--backend", args.backend]
    for shape in args.shape or []:
        cmd += ["--shape", shape]
    out_dir.mkdir(parents=True, exist_ok=True)
    env = None
    if node_dump_cwd is not None:
        env = dict(os.environ, VKOP_NODE_DUMP="1")
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env,
                          cwd=str(node_dump_cwd) if node_dump_cwd else None)
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
    if spec["dtype"] not in ("float32", "float16", "bfloat16"):
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
    if spec["dtype"] == "bfloat16":
        return rng.standard_normal(dims).astype(np.float32)
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
        if dtype == "bfloat16":
            # The shared array holds the bf16 VALUES as float32 (exact), so
            # ORT is fed exactly what the runtime decodes; write_input turns
            # them into the words. A uint16 .npy is taken as bf16 words.
            if arr.dtype == np.uint16:
                arr = bf16_to_f32(arr).reshape(arr.shape)
            else:
                rounded = bf16_to_f32(f32_to_bf16(arr)).reshape(arr.shape)
                if arr.dtype != np.float32 or not np.array_equal(
                        rounded, arr.astype(np.float32), equal_nan=True):
                    print(f"[input] {name}: {arr.dtype} rounded to bfloat16")
                arr = rounded
            np_dt = np.dtype(np.float32)
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


def write_input(path: Path, arr, dtype):
    """The .npy vkop-bench loads: a bf16 input as its uint16 words (what
    vkop_bench.cpp fill_bf16_from_npy copies verbatim), anything else as is."""
    if dtype == "bfloat16":
        arr = f32_to_bf16(arr).reshape(np.shape(arr))
    np.save(path, arr)


def ort_reference(sess, arrays):
    feed = {}
    for i in sess.get_inputs():
        arr = arrays[i.name]
        if i.type == "tensor(bfloat16)":
            raise LoopError(
                f"onnxruntime input {i.name!r} is bfloat16, which numpy cannot "
                "hold without ml_dtypes; compare against the fp32/fp16 export "
                "the bf16 artifact was converted from")
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
    # Positional, not a count difference: an Inf the reference has in the same
    # element is the model's own overflow (a -Inf attention mask), an Inf in a
    # different element is not.
    m["inf"] = int((np.isinf(a) ^ np.isinf(b)).sum())
    # Same-signed Inf on both sides is agreement. |inf - inf| is NaN, and one
    # NaN would make maxabs/meanabs/rel_l2 NaN for the whole tensor, hiding the
    # error in every element that could actually be compared.
    agree = np.isinf(a) & np.isinf(b) & (np.sign(a) == np.sign(b))
    # Subtracted with a mask rather than as `np.where(agree, 0.0, a - b)`: the
    # eager form computes inf - inf first and raises an invalid-value warning
    # for the pairs this line exists to ignore.
    diff = np.subtract(a, b, out=np.zeros_like(a), where=~agree)
    d = np.abs(diff)
    if not d.size:
        # A 0-element output (a kv_len=0 cache slice) has no error to report;
        # it passes on the shape equality already checked above.
        m["maxabs"] = m["meanabs"] = m["rel_l2"] = m["bad_frac"] = 0.0
        m["cos"] = 1.0
        m["ok"] = True
        return m
    m["maxabs"] = float(d.max())
    m["meanabs"] = float(d.mean())
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na == 0.0 and nb == 0.0:
        # Both exactly zero (a padding mask, a disabled branch): cosine would be
        # 0/0, and there is nothing to disagree about -- maxabs and the outlier
        # fraction already report the absent error.
        m["cos"], m["rel_l2"] = 1.0, 0.0
    else:
        # A norm that is Inf (a tensor that overflows) cannot ground a cosine
        # or a relative error; report neither and let the Inf gate decide.
        if np.isfinite(na) and np.isfinite(nb):
            m["cos"] = float(a.ravel() @ b.ravel() / (na * nb))
        if np.isfinite(na) and na > 0.0:
            m["rel_l2"] = float(np.linalg.norm(diff) / na)
        elif na == 0.0 and nb > 0.0:
            # A zero reference with a nonzero output is an infinite relative
            # error, not a zero one.
            m["rel_l2"] = float("inf")
    # The outlier band scales with the tensor, not with each element: an
    # fp16-storage run on the Add+Conv fixture puts 11.1% of elements outside
    # atol + rtol*|ref| while its cosine is 0.999995, because the flagged ones
    # are the small-magnitude elements (median |ref| 0.16 against 0.73 overall)
    # carrying ordinary accumulation error from the large intermediates they
    # were computed from. Against max|ref| the same run is 0.0000%.
    # The scale is the largest FINITE element: with an Inf in the reference the
    # band would otherwise be Inf, which flags nothing and turns the row green
    # no matter how far the rest of the tensor drifted.
    finite = np.abs(a)[np.isfinite(np.abs(a))]
    scale = float(finite.max()) if finite.size else 0.0
    band = atol + rtol * scale
    m["bad_frac"] = float((d > band).mean())
    if m["nan"]:
        m["reasons"].append(f"{m['nan']} NaN(s) in the runtime output")
    if m["inf"] > 0:
        m["reasons"].append(f"{m['inf']} element(s) where one side is Inf and "
                            "the other is not")
    # Only a computable cosine can fail this gate. A NaN one means an Inf or
    # NaN above already said what is wrong, and "cos nan < 0.999" would bury it.
    if np.isfinite(m["cos"]) and m["cos"] < cos_min:
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
    if explicit:
        p = Path(explicit)
        # A --bench that is not runnable has to stop the run rather than fall
        # through to the build tree's binary: the whole point of naming one is
        # to measure THAT build.
        if not (p.is_file() and os.access(p, os.X_OK)):
            raise LoopError(f"--bench {explicit} is not an executable file")
        return p
    cands = [str(_REPO / "build" / "benchmark" / "vkop-bench"),
             "build/benchmark/vkop-bench"]
    # Multi-config generators (Visual Studio, Xcode) put the binary one level
    # down, and Windows adds .exe, so the flat name is only one build shape.
    name = "vkop-bench.exe" if os.name == "nt" else "vkop-bench"
    roots = [_REPO / "build" / "benchmark", Path("build/benchmark")]
    cands += [str(r / cfg / name)
              for r in roots for cfg in ("Release", "RelWithDebInfo", "Debug")]
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


# ------------------------------------------------------------ per-node dump

NODE_OUTPUTS_KEY = "vkop_node_outputs"   # onnx2vkop.ort_lower.NODE_OUTPUTS_KEY
_DUMP_RE = re.compile(r"^(\d+)_lvl\d+_n\d+_(.+)_out(\d+)\.raw$")


def node_output_map(onnx_path: Path):
    """{vkop node name: [output tensor names]} from a graph written by
    `onnx2vkop --emit-ort-onnx`, or None for any other ONNX file."""
    import json

    import onnx
    m = onnx.load(str(onnx_path), load_external_data=False)
    for p in m.metadata_props:
        if p.key == NODE_OUTPUTS_KEY:
            return json.loads(p.value)
    return None


def read_node_dumps(dump_dir: Path):
    """{(sanitized node name, k): (seq, path)}, first occurrence per key (a
    decode loop dumps every round; the first is the run that was fed)."""
    found = {}
    for f in dump_dir.glob("*.raw"):
        m = _DUMP_RE.match(f.name)
        if m:
            key = (m.group(2), int(m.group(3)))
            seq = int(m.group(1))
            if key not in found or seq < found[key][0]:
                found[key] = (seq, f)
    return found


def decode_dump(path: Path, ref: np.ndarray):
    """The dump's bytes as an array of ref's shape. The runtime writes fp32,
    fp16 or int64 with no header, so the width is read off the byte count."""
    raw = path.read_bytes()
    n = ref.size
    for dt in (np.float32, np.float16, np.int64):
        if n and len(raw) == n * np.dtype(dt).itemsize:
            got = np.frombuffer(raw, dtype=dt).reshape(ref.shape)
            if np.issubdtype(ref.dtype, np.inexact):
                got = got.astype(np.float32)
            return got, np.dtype(dt).name
    return None, f"{len(raw)} bytes for {n} elements"


def report_nodes(node_map, dumps, refs, args):
    """Every dumped node output against its ORT tensor, in execution order.
    Returns the number of failing tensors."""
    rows, missing = [], 0
    for node, outs in node_map.items():
        key = node.replace("/", "_")
        for k, tensor in enumerate(outs):
            hit = dumps.get((key, k))
            ref = refs.get(tensor)
            if hit is None or ref is None:
                missing += hit is not None
                continue
            seq, path = hit
            got, how = decode_dump(path, ref)
            if got is None:
                rows.append((seq, node, tensor, None, how))
                continue
            # numpy on Accelerate raises spurious FP warnings from float64
            # matmul on large-magnitude rows (a -65504 mask); the metrics
            # themselves are finite and checked by compare().
            with np.errstate(all="ignore"):
                m = compare(ref, got, args.rtol, args.atol, args.cos_min,
                            args.bad_frac_max)
            rows.append((seq, node, tensor, m, how))
    rows.sort(key=lambda r: r[0])
    print(f"\nper-node outputs ({len(rows)} dumped tensors matched to ORT)")
    print(f"{'seq':>6} {'tensor':<48} {'dt':<8} {'maxabs':>9} {'relL2':>9} "
          f"{'cos':>10} {'out%':>8}")
    failed, first = 0, None
    for seq, node, tensor, m, how in rows:
        if m is None:
            print(f"{seq:>6} {tensor:<48} ! cannot decode the dump ({how})")
            failed += 1
            first = first or tensor
            continue
        mark = ""
        if not m["ok"]:
            failed += 1
            mark = "  <-- FIRST DIVERGENCE" if first is None else "  FAIL"
            first = first or tensor
        if args.verbose or not m["ok"] or mark:
            print(f"{seq:>6} {tensor[-48:]:<48} {how:<8} "
                  f"{_cell(m['maxabs'], 9, '.3e')} {_cell(m['rel_l2'], 9, '.3e')} "
                  f"{_cell(m['cos'], 10, '.6f')} "
                  f"{_cell(100 * m['bad_frac'], 8, '.4f')}{mark}")
            for reason in m["reasons"] if mark else []:
                print(f"{'':>6} {'':<48} ! {reason}")
    if missing:
        print(f"({missing} dumped tensor(s) have no ORT counterpart)")
    print(f"per-node: {len(rows) - failed}/{len(rows)} pass"
          + (f", first divergence at {first}" if first else "")
          + ("" if args.verbose else " (passing rows hidden; -v lists them)"))
    return failed


# ---------------------------------------------------------------------------


def _cell(value, width, spec):
    """One table cell, with '-' where a metric does not apply.

    An integer output has no cosine and no fp16-style meanabs to report --
    printing NaN there reads as a broken measurement rather than as "this
    output is compared exactly".
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return " " * (width - 1) + "-"
    return format(value, f">{width}{spec}")


def report_row(name, note, m, t):
    shape = "x".join(str(d) for d in m["shape"]) or "scalar"
    if note:
        shape += f" ({note})"
    tk = _cell(t["top1"] if t else None, 6, ".3f")
    print(f"{name:<30} {shape:<18} {_cell(m['maxabs'], 9, '.3e')} "
          f"{_cell(m['meanabs'], 9, '.3e')} {_cell(m['rel_l2'], 9, '.3e')} "
          f"{_cell(m['cos'], 10, '.6f')} {_cell(100 * m['bad_frac'], 8, '.4f')} "
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
    ap.add_argument("--node-dump", action="store_true",
                    help="also compare EVERY node's output: --onnx must be the "
                         "graph `onnx2vkop --emit-ort-onnx` wrote for this "
                         ".vkopbin (it keeps the runtime's tensor names and "
                         "maps node -> tensors); the run sets VKOP_NODE_DUMP "
                         "and the report names the first node that diverges. "
                         "Report-only: the exit code still follows the graph "
                         "outputs.")
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
        kinds = {s["name"]: s["dtype"] for s in vk_inputs}
        for name, arr in arrays.items():
            write_input(in_dir / f"{name}.npy", arr, kinds.get(name))
        node_map = None
        if getattr(args, "node_dump", False):
            node_map = node_output_map(onnx_path)
            if node_map is None:
                raise LoopError(
                    f"--node-dump: {onnx_path} carries no '{NODE_OUTPUTS_KEY}' "
                    "metadata; convert with `onnx2vkop --emit-ort-onnx PATH` "
                    "and pass that PATH as --onnx")
            shutil.rmtree(work / "node_dump", ignore_errors=True)
        rows = run_vkopbench(bench, vkopbin, in_dir, out_dir, args,
                             node_dump_cwd=work if node_map else None)
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
                print(f"{row['name']:<30} {'x'.join(map(str, row['dims'])):<18} "
                      f"{_cell(None, 9, '')} {_cell(None, 9, '')} "
                      f"{_cell(None, 9, '')} {_cell(None, 10, '')} "
                      f"{_cell(None, 8, '')} {_cell(None, 6, '')} FAIL")
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
        # An output the .vkopbin declares but the run never wrote is a FAIL,
        # not an absence: vkop-bench reports a failed write only on stderr,
        # and this loop iterates the manifest, so it used to vanish silently.
        saved = {r["name"] for r in rows}
        unsaved = [o["name"] for o in view.outputs() if o["name"] not in saved]
        for name in unsaved:
            print(f"{name:<30} ! NOT SAVED: declared by the .vkopbin, absent "
                  "from vkop-bench's manifest (rerun with -v for its stderr)")
        failed += len(unsaved)
        shown = shown + [None] * len(unsaved)
        print(f"\nRESULT: {len(shown) - failed}/{len(shown)} shown output(s) "
              f"pass" + ("" if not failed else f", {failed} FAIL"))
        if node_map is not None:
            report_nodes(node_map, read_node_dumps(work / "node_dump"), refs,
                         args)
        print(f"work dir: {work}" + ("" if keep else " (removed)"))
        return 0 if failed == 0 else 1
    finally:
        if not keep:
            shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
