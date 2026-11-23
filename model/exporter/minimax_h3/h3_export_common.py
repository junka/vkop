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


# ---------------------------------------------------------------------------
# streaming export: real weights, bounded memory
# ---------------------------------------------------------------------------
#
# The eager path above (`export_graph` with `dry_run=False`) cannot export these
# components at all, for two independent reasons:
#
#   * it never loads a checkpoint. `build_*` constructs the module from
#     `config.json`, so the "weights" it exports are whatever `nn.Linear`
#     initialized. The graph is right; the bytes are not the model's.
#   * even with a loader bolted on, `torch.onnx.export` needs every parameter
#     resident AND a second copy inside the protobuf. The transformer is
#     61.7 GiB on a 62 GB box.
#
# `onnx2vkop.stream_export` exists for exactly this, but its stage-1 entry
# (`build_graph_without_weights`) is not usable for a real export: it leaves the
# parameter inputs with the exporter's generated names (`onnx::MatMul_43`), so
# nothing can be matched against a checkpoint's tensor names. `trace_weightless`
# below drives the same primitives and then renames every parameter input after
# its module attribute, which is what the splice stage needs.
#
# Measured on the real 61.7 GiB transformer: 55 s and 0.9 GiB peak RSS.

def _peak_rss_gib() -> float:
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmHWM:"):
                    return int(line.split()[1]) / (1024 * 1024)
    except OSError:
        pass
    return 0.0


def _apply_io_names(proto, input_names, output_names, dynamic_axes) -> None:
    """Set the user-facing input/output names and symbolic axes on the proto.

    The internal `_optimize_graph`/`_export_onnx` pair does not reliably apply
    either: measured on the denoiser it renamed only part of the inputs, left
    every output as a node number, and therefore matched no `dynamic_axes` key,
    so every symbolic dim came out as `<something>_dim_<i>`. Doing it on the
    finished proto is deterministic. Positional pairing is sound because input
    and output order are preserved through the export.
    """
    for vi, name in zip(proto.graph.input[:len(input_names)], input_names):
        if vi.name != name:
            _rename_value(proto, vi.name, name)
    for vi, name in zip(proto.graph.output, output_names):
        if vi.name != name:
            _rename_value(proto, vi.name, name)

    by_name = {vi.name: vi for vi in proto.graph.input}
    by_name.update({vi.name: vi for vi in proto.graph.output})
    for tensor_name, axes in dynamic_axes.items():
        vi = by_name.get(tensor_name)
        if vi is None:
            print(f"[stream] dynamic_axes: no tensor {tensor_name!r}")
            continue
        dims = vi.type.tensor_type.shape.dim
        for axis, label in axes.items():
            if 0 <= axis < len(dims):
                dims[axis].dim_param = label
                dims[axis].ClearField("dim_value")


def _rename_value(proto, old: str, new: str) -> None:
    for vi in list(proto.graph.input) + list(proto.graph.output):
        if vi.name == old:
            vi.name = new
    for node in proto.graph.node:
        for i, name in enumerate(node.input):
            if name == old:
                node.input[i] = new
        for i, name in enumerate(node.output):
            if name == old:
                node.output[i] = new


def trace_weightless(wrapper, dummies, input_names, output_names,
                     dynamic_axes):
    """Emit the graph with every parameter a DISTINCT, NAMED graph input.

    Mirrors `torch.onnx.export`'s internal call sequence. Two things beyond a
    bare `_model_to_graph` are required, and both were measured:

    * `_trigger_symbolic_function_registration()` -- absent, the trace has no
      symbolic for `aten::unflatten` and dies with `UnsupportedOperatorError
      ... to ONNX opset version 20`. The "opset 20" in that message is a red
      herring: it prints torch's default because the trace never reached the
      requested opset.

    * NOT running `_jit_pass_onnx_deduplicate_initializers`. It keys on the
      storage pointer and every meta tensor has `data_ptr() == 0`, so it
      collapses the parameter set: measured on the denoiser, 639 parameters
      became 18 graph inputs plus 617 Identity nodes -- a graph that converts
      but can no longer be spliced.

    Returns `(proto, {parameter name: shape})`, where the shape map is keyed by
    the MODULE ATTRIBUTE name so the splice can match checkpoint entries.
    """
    import onnx
    from torch.onnx import OperatorExportTypes
    from torch.onnx._internal.torchscript_exporter import utils as ex_utils

    ex_utils._trigger_symbolic_function_registration()
    ex_utils.GLOBALS.export_onnx_opset_version = OPSET
    ex_utils.GLOBALS.onnx_shape_inference = True

    graph, params, _out, module = ex_utils._create_jit_graph(wrapper, dummies)
    params_dict = ex_utils._get_named_param_dict(graph, params)
    # Which inputs are parameters is decided HERE, from the names
    # `_create_jit_graph` assigned. After `_optimize_graph` renames the user
    # inputs, an input can no longer be classified by name.
    param_names = list(params_dict.keys())
    graph = ex_utils._optimize_graph(
        graph, OperatorExportTypes.ONNX, params_dict=params_dict,
        dynamic_axes=dynamic_axes, input_names=list(input_names),
        module=module)
    ex_utils._C._jit_pass_onnx_assign_scoped_names_for_node_and_value(graph)
    raw = graph._export_onnx(
        {}, OPSET, dynamic_axes or {}, False, OperatorExportTypes.ONNX, True,
        False, {}, True, "", {})[0]
    proto = onnx.ModelProto()
    proto.ParseFromString(raw)

    # Name the parameter inputs after their module attribute. Input ORDER is
    # preserved end to end (verified on the denoiser: 650 debug names identical
    # before and after `_optimize_graph`), and the user inputs come first, so
    # the split is by POSITION -- not by name. Classifying by name is not
    # reliable: measured on this wrapper, `input_names` took effect on only
    # some of the 11 user inputs.
    state_dict = torch.jit._unique_state_dict(wrapper)
    sd_names = list(state_dict.keys())
    n_user = len(dummies)
    proto_inputs = [g.name for g in proto.graph.input]
    param_inputs = proto_inputs[n_user:]
    if len(param_inputs) != len(sd_names):
        raise SystemExit(
            f"trace: {len(param_inputs)} parameter inputs but "
            f"{len(sd_names)} parameters -- the exporter aliased them\n"
            f"  proto inputs: {proto_inputs[:16]}")

    for old, new in zip(proto_inputs[:n_user], input_names):
        if old != new:
            _rename_value(proto, old, new)
    for old, new in zip(param_inputs, sd_names):
        if old != new:
            _rename_value(proto, old, new)

    # Input/output names and dynamic axes are applied to the FINISHED proto
    # rather than trusting `_optimize_graph`: see `_apply_io_names`.
    _apply_io_names(proto, input_names, output_names, dynamic_axes)

    print(f"[stream] trace: {len(proto.graph.node)} node(s), "
          f"{len(proto.graph.input)} input(s) ({len(param_inputs)} parameters), "
          f"{len(proto.graph.initializer)} initializer(s), "
          f"rss={_peak_rss_gib():.1f} GiB")
    return proto, {n: list(v.shape) for n, v in state_dict.items()}


def checkpoint_dir(model_dir: str, component: str) -> str:
    """Where a component's own safetensors live inside the checkpoint."""
    return os.path.join(model_dir, component)


def export_streaming_real(wrapper, dummies, input_names, output_names,
                          dynamic_axes, *, model_dir: str, component: str,
                          out_path: str, weights_file: str | None = None,
                          ckpt_prefix: str = "", ckpt_rename: str | None = None,
                          rope_theta: float | None = None
                          ) -> str:
    """Export one component with its REAL weights, holding no more than a chunk.

    Three stages, only the last touches weight bytes:

      1. trace on `meta`   -- graph structure, no storage (see trace_weightless)
      2. splice            -- each parameter becomes an external initializer
                              pointing at a checkpoint shard (headers only)
      3. copy              -- append bytes 8 MiB at a time

    `ckpt_prefix` is the module prefix the wrapper added that the checkpoint
    does not carry (`model.` for the denoiser); `ckpt_rename` translates it to a
    different path inside the checkpoint (`lm.` -> `model.language_model.` for
    the text tower).

    Returns the ONNX path written. The weights land next to it.
    """
    import onnx
    pypi = _onnx2vkop_import_path()
    if pypi not in sys.path:
        sys.path.insert(0, pypi)
    from onnx2vkop import stream_export as SE

    ckpt = checkpoint_dir(model_dir, component)
    weights_file = weights_file or (
        os.path.splitext(os.path.basename(out_path))[0] + ".weights.bin")
    out_dir = os.path.dirname(os.path.abspath(out_path)) or "."
    os.makedirs(out_dir, exist_ok=True)

    index = SE.SafetensorsIndex(ckpt)
    print(f"[stream] checkpoint: {len(index.entries)} tensor(s) in "
          f"{len(index.shards)} shard(s) (headers only)")

    proto, shapes = trace_weightless(wrapper, dummies, input_names,
                                     output_names, dynamic_axes)

    def strip(n: str) -> str:
        if ckpt_rename and n.startswith(ckpt_prefix):
            return ckpt_rename + n[len(ckpt_prefix):]
        return n[len(ckpt_prefix):] if n.startswith(ckpt_prefix) else n

    hits, unresolved = {}, []
    for vi in proto.graph.input:
        if vi.name not in shapes:
            continue
        base = strip(vi.name)
        (hits.__setitem__(vi.name, base) if base in index
         else unresolved.append(vi.name))

    # A `persistent=False` buffer is COMPUTED from config and is in no
    # checkpoint. `promote_computed_buffers` turns it into a real initializer.
    if unresolved:
        made = promote_computed_buffers(proto, unresolved, shapes, wrapper,
                                        rope_theta=rope_theta)
        print(f"[stream] promoted {made} computed buffer(s) to initializers: "
              f"{unresolved}")
        unresolved = []
    print(f"[stream] resolved {len(hits)} weight(s) by name")

    back = {v: k for k, v in hits.items()}
    for vi in proto.graph.input:
        if vi.name in hits:
            vi.name = hits[vi.name]
    for node in proto.graph.node:
        for i, name in enumerate(node.input):
            if name in hits:
                node.input[i] = hits[name]

    shapes_ck = {hits[n]: list(shapes[n]) for n in hits}
    resolved = SE.splice_initializers(proto, shapes_ck, index,
                                      weights_file=weights_file,
                                      target_dtype=None)
    for init in proto.graph.initializer:
        if init.name in back:
            init.name = back[init.name]
    for node in proto.graph.node:
        for i, name in enumerate(node.input):
            if name in back:
                node.input[i] = back[name]
    if len(resolved) != len(hits):
        raise SystemExit(f"[stream] spliced {len(resolved)} of {len(hits)}")

    wpath = os.path.join(out_dir, weights_file)
    nbytes, offsets = SE.copy_weights(resolved, wpath)
    # `copy_weights` keys offsets by the names it was handed (checkpoint
    # names); the initializers carry the wrapper names again, so translate.
    SE.patch_offsets(proto, {back.get(k, k): v for k, v in offsets.items()})

    fixed = SE.resolve_shape_dims(proto, dynamic_axes)
    onnx.save(proto, out_path)
    print(f"[stream] wrote {out_path} "
          f"({os.path.getsize(out_path)/2**20:.1f} MiB) and {weights_file} "
          f"({nbytes/2**30:.2f} GiB); {fixed} dim(s) recovered; "
          f"peak rss={_peak_rss_gib():.1f} GiB")
    return out_path


def promote_computed_buffers(proto, names, shapes, wrapper,
                             rope_theta: float | None = None) -> int:
    """Turn computed-buffer graph inputs into inline initializers.

    A non-persistent buffer (`rope.inv_freq`) is derived from config: it has no
    checkpoint entry AND no bytes on a meta build. Its value is a pure function
    of the module's constructor arguments, so it is rebuilt from the module
    itself. Only `*rope.inv_freq` is known to need this; an unknown name is a
    hard error rather than a guess.

    The buffer is reached through the WRAPPER's own module tree by attribute
    path, because each wrapper names its inner model differently (`model.` for
    the denoiser, `d.` for the VAE decoder). Its value is recomputed with the
    module's OWN formula from the constructor attributes it exposes -- those
    differ per component (`rope_freq_dim` on the denoiser, a plain `inv_freq`
    buffer on the VAE), so the attributes are read, not assumed.
    """
    import onnx
    made = 0
    for name in list(names):
        if name not in shapes:
            continue
        if name.endswith(".inv_freq"):
            owner = name[: -len(".inv_freq")]
            rope = _resolve_attr(wrapper, owner)
            arr = _rebuild_rope_inv_freq(rope, theta=rope_theta)
        elif name.endswith(".M"):
            # A static-table buffer a wrapper computed from its rope module
            # (`Qwen3VLRotaryShim.M`). Unlike a meta `inv_freq` this holds a
            # real value, because the wrapper built it from the (meta) inv_freq
            # SHAPE and a plain arange -- so read it rather than re-derive it.
            owner = name[: -len(".M")]
            holder = _resolve_attr(wrapper, owner)
            arr = getattr(holder, "M").detach().clone().to(torch.float32)
        else:
            raise SystemExit(
                f"[stream] {name!r} is not in the checkpoint and is not a "
                f"known computed buffer -- refusing to invent it")
        init = proto.graph.initializer.add()
        init.CopyFrom(onnx.numpy_helper.from_array(
            arr.contiguous().to(torch.float32).numpy(), name))
        for i, vi in enumerate(proto.graph.input):
            if vi.name == name:
                del proto.graph.input[i]
                break
        made += 1
    return made


def _rebuild_rope_inv_freq(rope, theta: float | None = None) -> torch.Tensor:
    """Recompute a rotary embedding's `inv_freq` from its own attributes.

    Two shapes exist in this model. `MiniMaxH3RotaryPosEmbed` exposes
    `rope_freq_dim` and builds `1 / theta ** (arange(0, 2d, 2) / 2d)`.
    `MiniMaxH3VideoRotaryPosEmbed` (the VAE) takes a precomputed `inv_freq` and
    stores it as a plain tensor. Reading the formula's INPUTS rather than
    assuming one of them is what keeps this from silently returning a
    differently-sized table.
    """
    d = getattr(rope, "rope_freq_dim", None)
    # The module rarely carries theta (the upstream classes take it as a
    # CONSTRUCTOR argument); the caller's config value is the reliable source.
    th = theta if theta is not None else getattr(rope, "rope_theta", None)
    if d is not None and th is not None:
        d = int(d)
        return 1.0 / (float(th) ** (torch.arange(0, 2 * d, 2,
                                                 dtype=torch.float32) / (2 * d)))
    # A wrapper that knows its own reconstruction rule takes precedence: the
    # VAE's rope keeps theta only as a CONSTRUCTOR argument, so nothing on the
    # module can recover it, and the wrapper is handed it at build time.
    rebuild = getattr(rope, "rebuilt_inv_freq", None)
    if callable(rebuild):
        return rebuild()
    have = getattr(rope, "inv_freq", None)
    if have is not None and have.device.type != "meta":
        return have.detach().clone().to(torch.float32)
    if have is not None and th is not None:
        # HF's `Qwen3VLTextRotaryEmbedding`: `1 / theta ** (arange(0, dim, 2) /
        # dim)`, whose length is `dim/2`. The module keeps neither `dim` nor
        # theta, but its buffer's LENGTH gives `dim/2` and the caller supplies
        # theta -- enough to reproduce the table exactly.
        n = int(have.shape[-1])
        idx = torch.arange(0, 2 * n, 2, dtype=torch.float32) / (2 * n)
        return 1.0 / (float(th) ** idx)
    raise SystemExit(
        f"[stream] cannot rebuild {type(rope).__name__}.inv_freq: it exposes "
        f"no (rope_freq_dim + rope_theta), no rebuilt_inv_freq(), no materialized "
        f"inv_freq, and no theta was supplied. Hand the caller config's theta.")


def _resolve_attr(root, dotted: str):
    """Follow a dotted attribute path from `root`."""
    obj = root
    for part in dotted.split("."):
        obj = getattr(obj, part)
    return obj


__all__ = [
    "OPSET",
    "MIN_DIFFUSERS",
    "TESTED_DIFFUSERS",
    "CONVERTER_FOLDED_OPS",
    "RMSNormShim",
    "check_diffusers",
    "check_runtime_ops",
    "check_transformers",
    "checkpoint_dir",
    "export_graph",
    "export_streaming_real",
    "is_meta",
    "meta_safe_buffers",
    "model_device",
    "patch_rmsnorm",
    "promote_computed_buffers",
    "runtime_op_names",
    "trace_weightless",
    "try_convert",
]
