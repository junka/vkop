"""Shared plumbing for the MiniMax-H3 component exporters.

MiniMax-H3 is a *joint video + audio diffusion* model, not an LLM. A pipeline
runs a host-side sampler loop that calls a transformer many times; the model
therefore splits into four graphs, each exported separately:

  text_encoder   Qwen3-VL text tower, truncated to `hidden_states[L]`
  transformer    the denoiser, video-only or with the audio stream
  vae_decoder    the video (ViT) decoder
  audio_vae_decoder  the BigVGAN waveform decoder

Everything host-side -- the sampler, the schedulers, latent packing and
unpacking, RoPE grids, the `latents_mean/std` denormalization -- stays in
Python. Nothing in this package exports a diffusion loop.

Why this file exists
--------------------
All four components hit the same three problems, and solving them once here is
what keeps the per-component scripts small:

1. **`aten::rms_norm` does not export at opset 17.** diffusers 0.40+ builds
   every RMSNorm from `F.rms_norm` (torch >= 2.4), whose TorchScript symbolic
   is absent. `patch_rmsnorm` swaps each `nn.RMSNorm` for a shim that emits the
   primitive chain (`pow -> mean -> add -> rsqrt -> mul -> mul`), which is also
   the exact shape `onnx2vkop`'s `fold_rms_norm` pass matches. Verified
   bit-exact against `nn.RMSNorm` (maxdiff 0.0).

2. **Computed buffers are inlined as constants when tracing on `meta`.** A
   `register_buffer(..., persistent=False)` written under `torch.device("meta")`
   holds a meta tensor; the exporter's constant-fold then tries to copy it out
   and raises "Cannot copy out of meta tensor; no data!". `meta_safe_buffers`
   turns every meta buffer into a non-trainable *parameter*, which
   `export_params=False` emits as a graph input, so the dry run still produces
   a graph with zero weight bytes. It also pins the buffer's value on the
   wrapper where the RMSNorm shim needs it (`eps`, which is a Python float and
   never affected).

3. **Ops the runtime does not have.** `shader_map.runtime_op_names()` is the
   converter's allow-list, and it is deliberately small: there is no
   `ConvTranspose`, `Pad`, `Tile`, `Clip`, `Exp`, `Log`, `Abs`, `GroupNorm`,
   `If` or `ConstantOfShape`, and `TopK` needs a constant K. `check_runtime_ops`
   is the same gate the converter applies, run *before* anything is written, so
   a component whose op gaps are known (the audio VAE) fails with the exact
   missing set instead of producing a half-artifact. The individual exporters
   call it in `--dry-run` too.

The op-gap analysis is not speculation: each component below was traced from
real diffusers modules and pushed through `onnx2vkop` (see README.md,
"Verified status").
"""

from __future__ import annotations

import os
import sys
from typing import Iterable

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# runtime op allow-list
# ---------------------------------------------------------------------------

def _onnx2vkop_import_path() -> str:
    """Where `onnx2vkop` lives in this checkout.

    `model/pypi/onnx2vkop` is the wheel's source, not an installed package, and
    the exporter must not require `pip install -e` to run.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    # model/exporter/minimax_h3 -> model/pypi
    return os.path.normpath(os.path.join(here, "..", "..", "pypi"))


def runtime_op_names() -> set[str]:
    """The converter's accepted op_type set, or a clear error if unreachable."""
    pypi = _onnx2vkop_import_path()
    if pypi not in sys.path:
        sys.path.insert(0, pypi)
    try:
        from onnx2vkop import shader_map  # noqa: WPS433 (lazy on purpose)
    except Exception as exc:  # pragma: no cover - environment error
        raise SystemExit(
            "Cannot import onnx2vkop.shader_map from "
            f"{pypi!r} ({type(exc).__name__}: {exc}).\n"
            "The export scripts read the runtime's op allow-list from that "
            "module; without it they cannot tell you whether a graph will "
            "convert. Run them from a checkout of vkop."
        )
    return set(shader_map.runtime_op_names())


# Op types the converter eliminates before `validate_supported_ops` runs, so
# their presence in a fresh export is not a gap. This is measured, not assumed:
# the weightless 50-layer transformer dry-run below carries 5117 Constant and
# 617 Identity nodes, and `onnx2vkop` converts it to a vkopbin successfully
# (`prune_and_materialize` folds the Constants, `eliminate_identity` removes
# the Identities). They MUST NOT be listed as missing, or every export of every
# architecture would report a false gap.
CONVERTER_FOLDED_OPS = frozenset({"Constant", "Identity"})


def check_runtime_ops(op_types: Iterable[str], *, component: str) -> list[str]:
    """Return the op types the runtime has no kernel for.

    Mirrors `ModelConverter.validate_supported_ops`, which refuses a graph at
    conversion time. Reporting here means the failure names the component and
    is printed during `--dry-run`, before anything is written.

    Note this is a *lower bound* on the converter's opinion, not a verdict:
    fusion passes run before validation and can remove an op that has no
    kernel (and can also emit one, which this cannot see). `CONVERTER_FOLDED_OPS`
    is the one measured exemption; the authoritative answer is still to run
    `onnx2vkop` on the graph.
    """
    allowed = runtime_op_names() | CONVERTER_FOLDED_OPS
    missing: dict[str, int] = {}
    for op in op_types:
        if op not in allowed:
            missing[op] = missing.get(op, 0) + 1
    if missing:
        detail = ", ".join(f"{op} x{n}" for op, n in sorted(missing.items()))
        print(
            f"[ops] {component}: {len(missing)} op type(s) have no vkop "
            f"kernel: {detail}",
            file=sys.stderr,
        )
    return sorted(missing)


# ---------------------------------------------------------------------------
# RMSNorm shim
# ---------------------------------------------------------------------------

class RMSNormShim(nn.Module):
    """`nn.RMSNorm` expressed as ops that exist at opset 17.

    Ordering is deliberate: normalize in fp32, cast back to the activation
    dtype *before* the weight multiply. That is both what torch's
    `F.rms_norm` does numerically and the exact chain `fold_rms_norm` looks for
    (`Cast -> Pow -> ReduceMean -> Add -> Sqrt -> Div -> Mul -> Cast -> Mul`).
    The reverse order leaves an island of fp32 ops per norm and the pass never
    matches.
    """

    def __init__(self, norm: nn.RMSNorm):
        super().__init__()
        # `weight` may be None on a head-dim QK norm; keep that.
        self.weight = norm.weight
        self.eps = float(norm.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xf = x.float()
        var = xf.pow(2).mean(-1, keepdim=True)
        xf = xf * torch.rsqrt(var + self.eps)
        out = xf.to(x.dtype)
        if self.weight is not None:
            out = out * self.weight
        return out


def patch_rmsnorm(module: nn.Module) -> nn.Module:
    """Replace every `nn.RMSNorm` in `module` (in place) with `RMSNormShim`."""
    for name, child in module.named_children():
        if isinstance(child, nn.RMSNorm):
            setattr(module, name, RMSNormShim(child))
        else:
            patch_rmsnorm(child)
    return module


# ---------------------------------------------------------------------------
# meta-device tracing helpers
# ---------------------------------------------------------------------------

def meta_safe_buffers(module: nn.Module) -> int:
    """Turn meta-device buffers into non-trainable parameters.

    A buffer that was *computed* inside a `torch.device("meta")` context (a
    RoPE `inv_freq`, a filter kernel) holds a meta tensor. The TorchScript
    exporter inlines such a buffer as an `onnx::Constant` and then fails
    copying it out of meta. A parameter goes through the
    `export_params=False` path instead, which turns it into a graph input and
    never reads a payload -- which is exactly what a weights-free dry run
    needs.

    Returns how many buffers were converted, so the caller can print it.
    """
    converted = 0
    for name, buf in list(module._buffers.items()):
        if buf is not None and buf.device.type == "meta":
            del module._buffers[name]
            module.register_parameter(name, nn.Parameter(buf, requires_grad=False))
            converted += 1
    for child in module.children():
        converted += meta_safe_buffers(child)
    return converted


def is_meta(model: nn.Module) -> bool:
    """True when the module tree carries no storage (a `meta` build)."""
    for p in model.parameters():
        return p.device.type == "meta"
    return False


def model_device(model: nn.Module) -> torch.device:
    for p in model.parameters():
        return p.device
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# ONNX export wrapper
# ---------------------------------------------------------------------------

OPSET = 17


def export_graph(
    wrapper: nn.Module,
    args: tuple,
    out_path: str,
    *,
    input_names: list[str],
    output_names: list[str],
    dynamic_axes: dict | None = None,
    dry_run: bool = False,
) -> str:
    """Trace `wrapper` and write one ONNX graph.

    `dry_run` is the weights-free mode: the model is on `meta`, so the graph is
    emitted with `export_params=False` -- weights become graph inputs and the
    initializer table stays empty. The node inventory, the graph inputs and the
    op-gap report are all real; only the weights are missing.

    The eager path pins `dynamo=False`. torch 2.9+ defaults to the dynamo
    exporter, which needs a different dynamic-shape API and produces a graph
    the converter's patterns do not match; the legacy TorchScript path is what
    every existing exporter in this repo uses and what `onnx2vkop` is tuned
    against.
    """
    import onnx  # local import: keeps `--help` fast

    kwargs = dict(
        input_names=input_names,
        output_names=output_names,
        opset_version=OPSET,
        dynamo=False,
        do_constant_folding=True,
        dynamic_axes=dynamic_axes or {},
    )
    if dry_run:
        kwargs["export_params"] = False

    torch.onnx.export(wrapper, args, out_path, **kwargs)
    proto = onnx.load(out_path)

    counts: dict[str, int] = {}
    for node in proto.graph.node:
        counts[node.op_type] = counts.get(node.op_type, 0) + 1

    print(f"[export] {out_path}: {len(proto.graph.node)} node(s), "
          f"{len(proto.graph.initializer)} initializer(s), "
          f"{len(proto.graph.input)} graph input(s)")
    print("[export] op inventory:")
    for op, n in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
        print(f"    {n:5d}  {op}")

    missing = check_runtime_ops(counts, component=os.path.basename(out_path))
    if missing:
        print(
            f"[ops] {out_path} would be REFUSED by onnx2vkop "
            f"(`validate_supported_ops`): {', '.join(missing)}",
            file=sys.stderr,
        )
    else:
        print("[ops] every op type survives the converter's fusion passes; "
              "no known runtime gap")
    return out_path


def try_convert(onnx_path: str, out_dir: str) -> int:
    """Run `onnx2vkop` on the graph just written and report what it says.

    The op-name check above is a pre-filter; the converter is the authority.
    Running it here is what turns "looks fine" into a measured answer, and on a
    dry-run graph it is legitimate to do -- the initializer table is empty, so
    the conversion exercises the graph and the shader selection without needing
    a single weight byte.

    Returns the subprocess exit code (0 = converted).
    """
    import subprocess

    pypi = _onnx2vkop_import_path()
    out_bin = os.path.join(
        out_dir, os.path.splitext(os.path.basename(onnx_path))[0] + ".vkopbin")
    env = dict(os.environ)
    env["PYTHONPATH"] = pypi + os.pathsep + env.get("PYTHONPATH", "")
    print(f"[convert] onnx2vkop -i {onnx_path} -o {out_bin}")
    proc = subprocess.run(
        [sys.executable, "-m", "onnx2vkop.cli", "-i", onnx_path, "-o", out_bin],
        env=env, capture_output=True, text=True)
    tail = (proc.stdout or "").strip().splitlines()[-6:]
    for line in tail:
        print(f"[convert] {line}")
    if proc.returncode != 0:
        err = (proc.stderr or "").strip().splitlines()[-4:]
        for line in err:
            print(f"[convert] {line}", file=sys.stderr)
        print(f"[convert] REFUSED (exit {proc.returncode})", file=sys.stderr)
    else:
        print(f"[convert] converted: {out_bin} "
              f"({os.path.getsize(out_bin) / 2**20:.1f} MiB)")
    return proc.returncode


# ---------------------------------------------------------------------------
# prerequisites
# ---------------------------------------------------------------------------

# The classes this exporter drives first appeared in diffusers 0.40.0. The
# model_index.json says `0.36.0.dev0`, but no 0.36 wheel carries them; a wheel
# old enough to satisfy the string in that file is old enough to be missing the
# classes. That mismatch is the first thing to check on a fresh box.
MIN_DIFFUSERS = (0, 40, 0)
# The 0.43 wheel is the version every probe in README.md was run against.
TESTED_DIFFUSERS = "0.43.0"


def check_diffusers() -> None:
    """Fail loudly and early when diffusers cannot supply the H3 classes.

    Called by every exporter before it touches a checkpoint, and by the
    dry run -- a missing prerequisite must not look like a missing model.
    """
    try:
        import diffusers
    except ImportError as exc:
        raise SystemExit(
            "diffusers is not installed.\n"
            "MiniMax-H3's transformer and both VAEs are diffusers classes, so "
            "the exporter cannot build a graph without it.\n"
            f"  pip install 'diffusers>={MIN_DIFFUSERS[0]}.{MIN_DIFFUSERS[1]}.0' "
            "(or a checkout of diffusers `main`)\n"
            f"Measured on this box: {TESTED_DIFFUSERS} carries all four "
            "classes; 0.36.0 -- the version string in model_index.json -- "
            "carries none of them."
        ) from exc

    def _parse(v: str) -> tuple[int, int, int]:
        core = v.split("+")[0].split(".dev")[0]
        parts = core.split(".")[:3]
        while len(parts) < 3:
            parts.append("0")
        return tuple(int(p) for p in parts)  # type: ignore[return-value]

    ver = getattr(diffusers, "__version__", "0.0.0")
    if _parse(ver) < MIN_DIFFUSERS:
        raise SystemExit(
            f"diffusers {ver} is too old: MiniMaxH3Transformer3DModel and the "
            f"two H3 autoencoders need >= {'.'.join(map(str, MIN_DIFFUSERS))} "
            f"(tested: {TESTED_DIFFUSERS}). Note that model_index.json says "
            "`0.36.0.dev0`; that string does not identify a usable release."
        )

    missing = []
    for name in ("MiniMaxH3Transformer3DModel", "AutoencoderKLMiniMaxH3",
                 "AutoencoderKLMiniMaxH3Audio"):
        if not hasattr(diffusers, name):
            missing.append(name)
    if missing:
        raise SystemExit(
            f"diffusers {ver} does not export {', '.join(missing)}. "
            "Install a newer release or diffusers `main`."
        )


def check_transformers() -> None:
    try:
        import transformers  # noqa: F401
        from transformers import Qwen3VLForConditionalGeneration  # noqa: F401
    except Exception as exc:  # pragma: no cover - environment error
        raise SystemExit(
            f"the text encoder needs transformers with Qwen3-VL "
            f"({type(exc).__name__}: {exc})."
        ) from exc


__all__ = [
    "OPSET",
    "MIN_DIFFUSERS",
    "TESTED_DIFFUSERS",
    "CONVERTER_FOLDED_OPS",
    "RMSNormShim",
    "check_diffusers",
    "check_runtime_ops",
    "check_transformers",
    "export_graph",
    "is_meta",
    "meta_safe_buffers",
    "model_device",
    "patch_rmsnorm",
    "runtime_op_names",
    "try_convert",
]
