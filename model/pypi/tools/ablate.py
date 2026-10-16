#!/usr/bin/env python
"""Per-model ablation of the converter's optimization passes.

Which pass pays off is a property of the model, not of the pass: a rope fusion
that removes 200 nodes from an LLM removes nothing from a CNN, and an LLM decode
step is weight-bandwidth-bound so a node it drops saves 0 ms while a small
dispatch-bound model saves real time. So measure instead of arguing:

    cd model/pypi
    python tools/ablate.py -i ../../image/exporter/dit_prefill_tiny_static.onnx --oo --fus

For each pass it converts the model once with that pass disabled
(VKOP_NO_PASSES / VKOP_NO_FUSIONS) and diffs the manifest against the baseline:

    variant                        nodes   inits  levels        bytes
    baseline                         1194     694     576  3441820104
    -fus:fuse_rotary_embedding       1250     694     580  3441824120   +56 nodes

A variant with NO delta is a pass that never fires on this model; a variant that
shrinks the graph is doing work. --run takes a shell template to time each
artifact (e.g. the llm_chat decode loop) when node count is not the point.
"""
import argparse
import os
import subprocess
import sys
import tempfile

PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PKG_ROOT)

from onnx2vkop import optimizer as _opt  # noqa: E402
from onnx2vkop.generated.vkop.model.Model import Model as _ModelT  # noqa: E402


def manifest_stats(path):
    with open(path, "rb") as fh:
        data = fh.read()
    m = _ModelT.GetRootAsModel(data, 0)
    return {"nodes": m.NodesLength(), "inits": m.InitializersLength(),
            "levels": m.ConcurrentLevelsLength(), "bytes": len(data)}


def convert(input_model, out_path, env_extra):
    env = dict(os.environ, **env_extra)
    proc = subprocess.run(
        [sys.executable, "-m", "onnx2vkop.cli", "-i", input_model, "-o", out_path],
        cwd=PKG_ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True)
    if proc.returncode != 0:
        raise RuntimeError("convert failed:\n" + proc.stdout[-4000:])
    return manifest_stats(out_path)


def fusion_pass_names():
    reg = _opt.FusionOptimizer.create_default_optimizer()
    return [p.name for p in reg.passes]


def print_row(label, st, base=None):
    delta = ""
    if base and label != "baseline":
        dn = st["nodes"] - base["nodes"]
        dl = st["levels"] - base["levels"]
        db = st["bytes"] - base["bytes"]
        delta = "  %+d nodes  %+d levels  %+d B" % (dn, dl, db)
    print("%-38s %7d %7d %7d %12d%s" % (
        label, st["nodes"], st["inits"], st["levels"], st["bytes"], delta))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True)
    ap.add_argument("--oo", action="store_true", help="ablate onnxoptimizer passes")
    ap.add_argument("--fus", action="store_true", help="ablate DAG fusion passes")
    ap.add_argument("--only", default="", help="comma list restricting the variants")
    ap.add_argument("--out", default="", help="directory to keep artifacts in")
    ap.add_argument("--run", default="", help="shell template to time an artifact, "
                                              "with {vkopbin} in it; its stdout tail is shown")
    args = ap.parse_args()
    if not args.oo and not args.fus:
        ap.error("pick at least one of --oo / --fus")

    keep_dir = args.out or tempfile.mkdtemp(prefix="ablate_")
    os.makedirs(keep_dir, exist_ok=True)
    wanted = {x.strip() for x in args.only.split(",") if x.strip()}

    variants = []
    if args.oo:
        variants += [("-oo", n) for n in _opt.ONNXOPTIMIZER_PASSES
                     if not wanted or n in wanted]
    if args.fus:
        variants += [("-fus", n) for n in fusion_pass_names()
                     if not wanted or n in wanted]

    base = convert(args.input, os.path.join(keep_dir, "baseline.vkopbin"), {})
    print_row("baseline", base)
    rows = []
    for kind, name in variants:
        var = "VKOP_NO_PASSES" if kind == "-oo" else "VKOP_NO_FUSIONS"
        st = convert(args.input, os.path.join(keep_dir, name + ".vkopbin"), {var: name})
        rows.append((name, st))
        print_row(kind + ":" + name, st, base)
    if args.run:
        for name, _ in rows:
            cmd = args.run.format(vkopbin=os.path.join(keep_dir, name + ".vkopbin"))
            r = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True)
            print("=== %s\n%s" % (name, r.stdout.strip().splitlines()[-3:]))
    if not args.out:
        import shutil
        shutil.rmtree(keep_dir, ignore_errors=True)
        print("\n(artifacts were in a temp dir; pass --out DIR to keep them)")


if __name__ == "__main__":
    main()
