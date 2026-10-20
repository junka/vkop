#!/usr/bin/env python3
"""Command-line interface for onnx2vkop."""

import argparse
import os.path as _osp
import sys
from pathlib import Path

import onnx

# filter_shaders.py lives in tools/, outside the wheel. Reach it via a path
# hop so the CLI does not have to subprocess out to Python again.
_TOOLS = _osp.normpath(
    _osp.join(_osp.dirname(__file__), "..", "..", "..", "tools")
)
if _osp.isdir(_TOOLS) and _TOOLS not in sys.path:
    sys.path.append(_TOOLS)

try:
    from .converter import ModelConverter
    from . import shader_map
except ImportError:
    from converter import ModelConverter
    import shader_map

# filter_shaders may or may not be on sys.path yet (the block above appends
# tools/ when it exists); import lazily so a CLI invocation without
# --emit-shader-archive keeps working even if tools/ is missing.
try:
    import filter_shaders
except ImportError:
    filter_shaders = None  # checked below when --emit-shader-archive is used


def build_shader_subset(dag_model, master_path, backend="auto"):
    """Read the master archive and keep the keys this graph can dispatch.

    Returns (subset, missing, master_count): `subset` is {archive key:
    SPIR-V bytes} ready for filter_shaders.serialize()/write_subset(),
    `missing` lists the keys the graph asks for that the master doesn't
    have, and `master_count` is how many entries the master holds (for the
    summary lines). The op-type census comes from `dag_model.nodes`, the
    same set the CLI already prints in "Operator Statistics", mapped through
    shader_map.shader_keys_for_model.

    backend="auto" (the default) takes the union of the buffer and image key
    sets. Which one the driver picks is a runtime choice -- Runtime defaults
    to the image backend and every in-tree driver calls
    set_backend_buffer(true) -- so a self-contained .vkopbin has to be able
    to answer for both, or a miss would silently drop the op back to the
    libvkop-embedded C arrays (and to a stub, on a stripped build).
    shader_map already over-includes on purpose for the same reason; the
    unused half costs bytes in the file but never faults in, since the op
    never asks for it. Pin "buffer"/"image" for the tight set.
    """
    op_types = {node.op_type for node in dag_model.nodes.values()}
    if backend == "auto":
        keys = (shader_map.shader_keys_for_model(op_types, backend="buffer")
                | shader_map.shader_keys_for_model(op_types, backend="image"))
    else:
        keys = shader_map.shader_keys_for_model(op_types, backend=backend)
    master = filter_shaders.read_master(master_path)
    subset = {k: master[k] for k in keys if k in master}
    # Only complain about a key that is absent under BOTH prefixes: the
    # selection helper emits e.g. "image_gather" for an op that has no image
    # variant, meaning "the runtime falls back to buffer_gather", and that
    # phantom can't be in any archive. Both spellings missing is a real gap.
    have = {k.split("_", 1)[1] for k in subset}
    missing = sorted(k for k in keys
                     if k not in master and k.split("_", 1)[1] not in have)
    return subset, missing, len(master)


def report_missing_keys(missing, master_path):
    if not missing:
        return
    for k in missing:
        print(f"[shaders] master has no {k!r}", file=sys.stderr)
    print(
        f"[shaders] {len(missing)} key(s) missing from {master_path}; "
        "an op whose shader is absent falls back to its embedded C array "
        "under the default build and to a stubbed pointer under "
        "VKOP_STRIP_BUILTIN_SHADERS=ON.",
        file=sys.stderr,
    )


def emit_shader_archive(dag_model, master_path, out_path, backend="auto"):
    """Write a per-model vkop_shaders.bin subset at out_path.

    The standalone-file spelling of what --embed-shaders puts in the
    .vkopbin: same key set, same bytes, kept for callers that deploy the
    archive separately (VKOP_SHADER_ARCHIVE).
    """
    subset, missing, master_count = build_shader_subset(
        dag_model, master_path, backend)
    report_missing_keys(missing, master_path)
    filter_shaders.write_subset(subset, out_path)
    out_size = Path(out_path).stat().st_size
    master_size = Path(master_path).stat().st_size
    print(
        f"[shaders] wrote {out_path}: {len(subset)} of {master_count} "
        f"shaders, {out_size:,} / {master_size:,} bytes "
        f"({100.0 * out_size / max(1, master_size):.1f}%)"
    )


def resolve_master_archive(explicit=None):
    """Find the full vkop_shaders.bin to filter the per-model subset from.

    Precedence: --master-archive, then VKOP_MASTER_ARCHIVE, then the two
    build trees next to this checkout (build/, build-strip/). Returns
    (None, None) when nothing is found -- the caller warns and converts
    without an embedded archive rather than failing, since a converter run
    in an environment with no vkop build is legitimate.
    """
    import os
    if explicit:
        return Path(explicit), "--master-archive"
    env = os.environ.get("VKOP_MASTER_ARCHIVE")
    if env:
        return Path(env), "VKOP_MASTER_ARCHIVE"
    root = Path(__file__).resolve().parents[3]  # model/pypi/onnx2vkop -> repo root
    for rel in ("build/vkop_shaders.bin", "build-strip/vkop_shaders.bin"):
        for base in (root, Path.cwd()):
            cand = base / rel
            if cand.is_file():
                return cand, str(cand)
    return None, None


def newest_shader_source(root):
    """mtime of the most recently edited .comp, or None if the tree is absent."""
    best = None
    for p in root.glob("shaders/**/*.comp"):
        m = p.stat().st_mtime
        if best is None or m > best:
            best = m
    return best


def embed_shader_archive(dag_model, master_path, backend="auto"):
    """Build the per-model VKSP subset and return it as bytes to embed.

    The .vkopbin then carries its own shaders: Runtime::LoadModel picks it up
    without VKOP_SHADER_ARCHIVE, and a libvkop built with
    VKOP_STRIP_BUILTIN_SHADERS=ON can still run the model.
    """
    subset, missing, master_count = build_shader_subset(
        dag_model, master_path, backend)
    report_missing_keys(missing, master_path)
    if not subset:
        return None
    # The embedded archive becomes authoritative at load time, so an archive
    # older than the shader sources is worth shouting about: the model would
    # carry kernels from before the edit and silently ignore the rebuild.
    src_mtime = newest_shader_source(Path(__file__).resolve().parents[3])
    if src_mtime is not None and master_path.stat().st_mtime < src_mtime:
        print(
            f"[shaders] WARNING: {master_path} is older than the newest "
            f"shader source in shaders/ -- the embedded SPIR-V is stale. "
            f"Rebuild (cmake --build <dir> --target shader_archive) and "
            f"reconvert.",
            file=sys.stderr,
        )
    blob = filter_shaders.serialize(subset)
    print(
        f"[shaders] embedded {len(subset)} of {master_count} shader(s) from "
        f"{master_path} ({len(blob):,} bytes) into the vkopbin"
    )
    return blob


def build_embedded_archive(args, dag_model, backend):
    """Resolve the master archive and return the subset to embed, or None.

    Every failure path here is a warning plus a None return, never an abort:
    a model whose .vkopbin lacks the archive still runs against the SPIR-V
    linked into libvkop, which is exactly the pre-embedding behaviour. The
    one caller that must not silently proceed (--emit-shader-archive, which
    has nothing to write) checks for None itself.
    """
    if filter_shaders is None:
        print(f"[shaders] no filter_shaders.py under {_TOOLS}; converting "
              f"without an embedded archive", file=sys.stderr)
        return None
    master, how = resolve_master_archive(args.master_archive)
    if master is None:
        print(
            "[shaders] no vkop_shaders.bin found (looked at --master-archive, "
            "$VKOP_MASTER_ARCHIVE, build/vkop_shaders.bin, "
            "build-strip/vkop_shaders.bin); converting without an embedded "
            "archive. Pass --no-embed-shaders to silence this.",
            file=sys.stderr,
        )
        return None
    print(f"[shaders] master archive: {master} (via {how})")
    return embed_shader_archive(dag_model, master, backend=backend)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Convert and optimize ONNX models to VKOP format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic conversion
  onnx2vkop -i model.onnx

  # With quantization
  onnx2vkop -i model.onnx -q fp16
  onnx2vkop -i model.onnx -q int8
  onnx2vkop -i model.onnx -q fp8e4m3
  onnx2vkop -i model.onnx -q int4 --group-size 64
  onnx2vkop -i model.onnx -q nf4
  onnx2vkop -i model.onnx -q nvfp4

  # With all optimizations
  onnx2vkop -i model.onnx -u -r -b 4
        """,
    )

    parser.add_argument("-i", "--input", required=True, help="Input ONNX model file path")
    parser.add_argument(
        "-o", "--output", help="Output VKOP binary file path (default: input_name.vkopbin)"
    )
    parser.add_argument(
        "-q", "--quant", choices=["fp16", "int8", "fp8e4m3", "fp8e5m2",
                                  "int4", "nf4", "nvfp4"],
        help="Quantization type: fp16, int8 (per-column scale), an fp8 "
             "weight-only MatMul payload with a per-column scale (e4m3 or e5m2, "
             "one byte per value), or a 4-bit weight-only MatMul payload with a "
             "per-K-group scale (int4 = signed nibble, nf4 = NF4 codebook index, "
             "nvfp4 = E2M1 nibble with an fp8 e4m3 scale per 16 values plus a "
             "per-tensor fp32 factor)",
    )
    parser.add_argument(
        "--group-size", type=int, default=64,
        help="Values of K sharing one fp32 scale for -q int4/nf4 (default: 64). "
             "-q nvfp4 ignores it: its block is fixed at 16 values.",
    )
    parser.add_argument(
        "-u", "--unify", action="store_true", help="Convert initializers to a single memory block"
    )
    parser.add_argument(
        "-r", "--rgba", action="store_true", help="NCHW to RGBA conversion for initializers"
    )
    parser.add_argument(
        "-b", "--batch", type=int, default=1, help="Batch size for inference (default: 1)"
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__import__('onnx2vkop').__version__}"
    )
    parser.add_argument(
        "--no-embed-shaders", action="store_true",
        help="Do not embed the per-model SPIR-V archive in the .vkopbin "
             "(which is what makes it a single self-contained file). The "
             "model then needs VKOP_SHADER_ARCHIVE at run time, or a libvkop "
             "built without VKOP_STRIP_BUILTIN_SHADERS=ON.",
    )
    parser.add_argument(
        "--emit-shader-archive", metavar="PATH",
        help="Additionally write the same per-model vkop_shaders.bin subset "
             "to PATH as a standalone file, for deployments that hand the "
             "archive to VKOP_SHADER_ARCHIVE instead of relying on the copy "
             "embedded in the .vkopbin.",
    )
    parser.add_argument(
        "--master-archive", metavar="PATH",
        help="Path to the full vkop_shaders.bin the CMake build produced. "
             "Defaults to $VKOP_MASTER_ARCHIVE, then build/vkop_shaders.bin "
             "or build-strip/vkop_shaders.bin next to this checkout.",
    )
    parser.add_argument(
        "--target-backend", choices=["auto", "buffer", "image"], default="auto",
        help="Which backend's shader variant names to embed. The default "
             "(auto) ships both, because the backend is the driver's runtime "
             "choice -- Runtime defaults to image and the in-tree drivers "
             "call set_backend_buffer(true) -- and a wrong guess would make "
             "the op fall back to libvkop's SPIR-V (or to a stub on a "
             "stripped build). Unused entries never fault in, so pinning "
             "buffer/image only shrinks the file.",
    )
    # Build-time knobs that mirror env vars the code already reads. Setting
    # these on the CLI overrides the env for the current invocation only.
    parser.add_argument(
        "--external-blob-mb", type=int, default=None, metavar="N",
        help="Weight blob larger than N MiB switches to the external-data "
             "vkopbin layout (default 800, matching the FlatBuffer 2 GiB "
             "vector cap with safety margin). Set to 1 to force a small "
             "model through the external path for testing.",
    )
    parser.add_argument(
        "--max-fold-rounds", type=int, default=None, metavar="N",
        help="ConstantFolder iteration cap (default 25). Raise when a model "
             "has a deep Shape/Gather/Concat chain that needs more rounds "
             "to reach fixed point.",
    )
    parser.add_argument(
        "--max-fuse-rounds", type=int, default=None, metavar="N",
        help="MultiRoundOptimizer iteration cap (default 10). Each round "
             "re-runs every registered fusion pass until nothing changes.",
    )
    parser.add_argument(
        "--no-passes", metavar="A,B,C", default=None,
        help="Comma-separated ONNXOptimizer pass names to skip "
             "(equivalent to VKOP_NO_PASSES=A,B,C). Use to ablate a suspect "
             "pass on a single model without an env edit.",
    )
    parser.add_argument(
        "--no-fusions", metavar="A,B,C", default=None,
        help="Comma-separated FusionOptimizer pass names to skip "
             "(equivalent to VKOP_NO_FUSIONS=A,B,C). Common pair for "
             "attention A/B: --no-fusions fuse_into_attention.",
    )

    return parser.parse_args()


def _apply_build_knobs(args) -> None:
    """Push the CLI's build-time knobs into the env the pipeline reads.

    Only the ones the user actually set; anything left as None lets the
    code's own default (or a pre-existing env from the shell) win. This
    keeps CLI and env as two spellings of the same knobs rather than two
    independent config sources that can disagree.
    """
    import os
    mapping = {
        "VKOP_EXTERNAL_BLOB_MB": args.external_blob_mb,
        "VKOP_MAX_FOLD_ROUNDS": args.max_fold_rounds,
        "VKOP_MAX_FUSE_ROUNDS": args.max_fuse_rounds,
        "VKOP_NO_PASSES": args.no_passes,
        "VKOP_NO_FUSIONS": args.no_fusions,
    }
    for env_name, val in mapping.items():
        if val is not None:
            os.environ[env_name] = str(val)
            print(f"[config] {env_name}={val}")


def main():
    """Main entry point for the CLI."""
    args = parse_args()
    _apply_build_knobs(args)

    # Validate input file
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file '{args.input}' does not exist")
        sys.exit(1)

    if not input_path.suffix.lower() in [".onnx"]:
        print("Error: Input file must be an ONNX model (.onnx)")
        sys.exit(1)

    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = input_path.with_suffix(".vkopbin")

    print(f"Converting model: {input_path}")
    print(f"Output will be saved to: {output_path}")

    try:
        # Create converter and run conversion
        converter = ModelConverter()
        dag_model = converter.parse_onnx_model(str(input_path), args.batch)

        # Create a simple args object for apply_optimizations
        class Args:
            def __init__(self, quant, unify, rgba, group_size):
                self.quant = quant
                self.unify = unify
                self.rgba = rgba
                self.group_size = group_size

        converter.apply_optimizations(
            dag_model, Args(args.quant, args.unify, args.rgba, args.group_size))

        # Print statistics (before binary save — dag_model may be huge for LLMs)
        op_stats = {}
        for node in dag_model.nodes.values():
            op_type = node.op_type
            op_stats[op_type] = op_stats.get(op_type, 0) + 1

        print("\nOperator Statistics:")
        print(f"{'idx':<5} {'type':<20} {'count':<10}")
        for idx, (op_type, count) in enumerate(op_stats.items(), 1):
            print(f"{idx:<5} {op_type:<20} {count:<10}")

        # Build the model's own shader subset once, then use the same bytes
        # both for the embedded copy and (if asked) the standalone file. The
        # graph has to be final by now -- the key set is derived from the op
        # census, and FusionOptimizer already ran inside apply_optimizations.
        backend = args.target_backend
        shader_blob = None
        if not args.no_embed_shaders or args.emit_shader_archive:
            shader_blob = build_embedded_archive(args, dag_model, backend)

        # Save the converted model
        dag_model.save_to_binary(str(output_path), shader_archive=shader_blob)

        print("\nConversion completed successfully!")
        print(f"Output file size: {output_path.stat().st_size:,} bytes")

        if args.emit_shader_archive:
            if shader_blob is None:
                print(
                    "Error: --emit-shader-archive produced no archive (no "
                    "master vkop_shaders.bin was found, or the graph needs no "
                    "shaders); pass --master-archive PATH.",
                    file=sys.stderr,
                )
                sys.exit(2)
            Path(args.emit_shader_archive).write_bytes(shader_blob)
            print(f"[shaders] wrote {args.emit_shader_archive}: "
                  f"{len(shader_blob):,} bytes")

    except Exception as e:
        print(f"\nError during conversion: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
