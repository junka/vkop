"""AWQ: activation-aware weight quantization.

AWQ (arXiv 2306.00978) does not change the quantization grid — it changes the
*weight*. Before quantizing it rescales each input channel (the K axis of a
MatMul's B weight) by s, and folds 1/s into whatever computes that MatMul's A:

    W' = W · diag(s)        A' = diag(1/s) · A        W'·A' = W·A  (exact)

The product is unchanged, but a channel whose activations are large now lands
on a smaller weight grid, and a channel with a large weight now sees a larger
activation — the quantization error moves to where it costs least. s is chosen
from activation and weight magnitudes:

    s_j = act_amax_j^alpha / w_amax_j^(1-alpha),  normalized to mean 1

This module only computes s and rewrites the graph's weights / norm scales
around it. The int4 or nf4 encode itself is the ordinary symmetric per-group
grid (`Quantizer._quantize_w3`), so everything the runtime does afterward — the
[K/group, N] scale table, the packed nibbles, the W4A16 kernel — is unchanged.

Two sources of s:

  * calibration stats (--awq-stats), per-MatMul activation absmax read off a
    few real forwards. This is the paper's recipe at its useful end;
  * a weight-only proxy (`act_amax ≡ 1`), the paper's `w_only` variant. Zero
    setup, weaker signal — it can only see that some input channels carry much
    bigger weights than others.

Where 1/s can be folded (anything else leaves that weight on the plain grid,
reported):

  * a norm weight (RMSNorm / LayerNormalization) reaching the MatMul through
    metadata-only views. Scaling the norm's per-channel weight by 1/s is exact:
    the norm output IS the MatMul's A;
  * a graph input that feeds the MatMul directly: a constant Mul goes in.

Every fold is gated on the exact shape the runtime will hand the kernel. A
fold the graph cannot carry is skipped, never approximated: a weight scaled by
s with no 1/s consumed anywhere is a silent wrong answer, so those weights are
left unscaled instead.
"""
from __future__ import annotations

import os
from collections import defaultdict

import numpy as np
from onnx import TensorProto, numpy_helper

# Views that do not move data, only reshape it (or re-tag its dtype). Walking
# through one means the tensor the fold lands on carries the same channel
# values the MatMul sees, possibly permuted.
_VIEW_OPS = frozenset(
    {"Reshape", "Transpose", "Squeeze", "Unsqueeze", "Identity", "Flatten",
     "Cast"})

_NORM_OPS = frozenset({"RMSNorm", "LayerNormalization", "LayerNorm"})

# How many K axes the walk will cross before giving up. Real LLM graphs are
# two or three (norm -> view -> MatMul); a longer chain is a structural oddity
# worth skipping rather than unbounded walking.
_MAX_DEPTH = 8

_EPS = 1e-12


# --------------------------------------------------------------------------
# scale search
# --------------------------------------------------------------------------

def _normalize(s):
    """Rescale so the geometric mean is 1: the W·A product is invariant to a
    constant multiplying s, so only the *relative* channel weights matter, and
    keeping s near 1 leaves the norm weight and W in their original range."""
    s = np.asarray(s, dtype=np.float64)
    if s.size == 0:
        return s.astype(np.float32)
    s = np.maximum(s, _EPS)
    return (s / np.exp(np.mean(np.log(s)))).astype(np.float32)


def w_amax(w):
    """Per-input-channel (K row) weight absmax of a [K, N] weight."""
    return np.maximum(np.max(np.abs(np.asarray(w, dtype=np.float32)), axis=1),
                      _EPS)


def solve_scales(w, act_amax=None, alpha: float = 0.5,
                 n_grid: int = 20, group_size: int = 64,
                 fmt: str = "int4", calib_ratio: float = 0.0):
    """The per-input-channel scale s (length K) for a [K, N] weight.

    Without `act_amax` this is the paper's `w_only` form: s ∝ w_amax^-alpha,
    i.e. channels that carry large weights are scaled up so the group grid can
    spend its codes where the weight actually is. The objective has no local
    minimum in alpha, so a closed form is enough.

    With `act_amax` (per-channel activation absmax, length K) the alpha grid is
    searched the way the paper does: each candidate is scored by the
    *output-side* group reconstruction error of the scaled weight,

        e(alpha) = || (W·diag(s) - Q(W·diag(s))) · diag(1/s) ||

    — the error the model actually sees, after 1/s has been folded back into
    the activation. `calib_ratio` (0..1) narrows the grid to
    [1-calib_ratio, 1+calib_ratio] x grid around the best exponent, which is
    AWQ's two-round "search twice, best-of" refinement without the memory cost
    of a full second sweep over a wider grid (both rounds use the same
    quantizer, so a finer grid over a narrower window is the same search).

    The quantizer used for scoring is the real symmetric per-group grid from
    `optimizer.Quantizer`, so the exponent picked is the one that minimizes
    the error the shipped kernels will make.
    """
    w = np.asarray(w, dtype=np.float32)
    if w.ndim != 2:
        raise ValueError(f"solve_scales wants a [K, N] weight, got {w.shape}")
    k, n = w.shape
    w_a = w_amax(w)
    if act_amax is None:
        return _normalize(np.power(w_a, -float(alpha)))

    a = np.asarray(act_amax, dtype=np.float32).reshape(-1)
    if a.size != k:
        raise ValueError(
            f"activation stats have {a.size} channels but the weight's K is {k}")
    a = np.maximum(a, _EPS)
    # A constant activation profile carries no relative information: fall back
    # to the weight-only form rather than searching a flat objective.
    if float(np.ptp(a)) <= _EPS * float(np.max(a)):
        return _normalize(np.power(w_a, -float(alpha)))

    from .optimizer import Quantizer  # local: optimizer must not import awq

    def score(alpha_value):
        s = _normalize(a ** alpha_value * w_a ** (-(1.0 - alpha_value)))
        out = _group_error(w * s[None, :], s, group_size, fmt)
        return out, s

    if n_grid <= 1:
        return score(float(alpha))[1]

    grid = np.linspace(0.0, 1.0, int(n_grid))
    errs = [score(float(x))[0] for x in grid]
    best = int(np.argmin(errs))
    if calib_ratio > 0.0:
        lo = max(0.0, grid[best] - calib_ratio)
        hi = min(1.0, grid[best] + calib_ratio)
        if hi > lo:
            fine = np.linspace(lo, hi, max(3, int(n_grid)))
            errs2 = [score(float(x))[0] for x in fine]
            if errs2 and min(errs2) < errs[best]:
                return score(float(fine[int(np.argmin(errs2))]))[1]
    return score(float(grid[best]))[1]


def _group_error(scaled, s, group_size, fmt):
    """Output-side reconstruction error of quantize(scaled) dequantized back
    through diag(1/s): the function the alpha search minimizes."""
    from .optimizer import Quantizer

    k, n = scaled.shape
    g = group_size if k % group_size == 0 else k
    codes, scale = Quantizer._quantize_w3(
        scaled.reshape(k // g, g, n), fmt)
    deq = Quantizer._dequantize_w3(codes, scale, fmt).reshape(k, n)
    err = (deq - scaled) / s[None, :]
    return float(np.mean(err * err))


# --------------------------------------------------------------------------
# activation statistics
# --------------------------------------------------------------------------

def load_stats(path: str):
    """Read a `.npz` of per-tensor activation magnitudes.

    Keys are tensor names. A value may carry any trailing shape (a raw
    [samples, K] dump, or a [K] absmax) — it is reduced to its per-channel
    absmax here, along every axis but the last: the channel axis of these
    graphs is the last one (inputs_embeds / hidden_states are [B, S, H]).
    A value that is already 1-D is taken as-is.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"--awq-stats: {path!r} does not exist")
    data = np.load(path)
    stats = {}
    for key in data.files:
        arr = np.asarray(data[key], dtype=np.float32)
        if arr.ndim > 1:
            arr = np.max(np.abs(arr), axis=tuple(range(arr.ndim - 1)))
        else:
            arr = np.abs(arr).reshape(-1)
        stats[key] = arr.reshape(-1)
    return stats


# --------------------------------------------------------------------------
# graph rewriting
# --------------------------------------------------------------------------

class _Fold:
    """Outcome of trying to place 1/s for one MatMul: either a callable that
    applies it to the graph, or the reason it cannot be applied."""

    def __init__(self, apply_fn=None, reason=None, consumed=None):
        self.apply_fn = apply_fn
        self.reason = reason
        self.consumed = consumed


def _node_by_output(dag_model):
    """tensor name -> the node that produces it (None for graph inputs /
    initializers, i.e. a name nobody in the node map writes)."""
    prod = {}
    for node in dag_model.nodes.values():
        for out in node.outputs:
            prod[out["name"]] = node
    return prod


def _find_fold(dag_model, mm_node, s):
    """Walk up from the MatMul's A to a norm or a graph input.

    Returns a _Fold. The walk crosses only metadata-only views and accepts one
    shape: the folding target's last axis IS the MatMul's K (possibly through
    leading `1` axes that Squeeze/Unsqueeze add and remove). Anything else —
    a Transpose that moves the channel axis, a norm whose scale length is not
    K — is reported, not approximated: a misaligned 1/s is a silently wrong
    model, the one outcome a fold must never produce.
    """
    node = mm_node
    idx = 0
    depth = 0
    while depth <= _MAX_DEPTH:
        depth += 1
        name = node.inputs[idx]["name"]
        if name in dag_model.initializers:
            return _Fold(reason=f"input '{name}' is an initializer")
        producer = _node_by_output(dag_model).get(name)
        if producer is None:
            return _fold_into_graph_input(dag_model, node, idx, name, s)
        if producer.op_type in _NORM_OPS:
            return _fold_into_norm(dag_model, producer, s)
        if producer.op_type not in _VIEW_OPS:
            return _Fold(
                reason=f"input comes from {producer.op_type} '{producer.name}'")
        node = producer
        idx = 0
    return _Fold(reason="view chain too long")


def _fold_into_norm(dag_model, norm, s):
    """1/s into the norm's scale initializer.

    The caller's shape checks (scale length == K, last axis is the channel
    axis) are what make this exact: the norm emits a per-channel-normalized
    value times its scale, so multiplying the scale by 1/s multiplies every
    channel of its output — the MatMul's A — by 1/s.
    """
    if len(norm.inputs) < 2:
        return _Fold(reason=f"{norm.op_type} '{norm.name}' has no scale input")
    w_name = norm.inputs[1]["name"]
    w_init = dag_model.initializers.get(w_name)
    if w_init is None:
        return _Fold(reason=f"{norm.op_type} scale '{w_name}' is not constant")
    if w_init.data_type not in (1, 10):  # FLOAT, FLOAT16
        return _Fold(reason=f"{norm.op_type} scale '{w_name}' is not float")
    wk = np.asarray(numpy_helper.to_array(w_init), dtype=np.float32)
    if wk.ndim != 1:
        return _Fold(reason=f"{norm.op_type} scale '{w_name}' is not 1-D")
    if wk.size != s.size:
        return _Fold(
            reason=f"{norm.op_type} scale has {wk.size} channels, weight K is "
                   f"{s.size}")
    inv = 1.0 / s

    def _apply():
        new = wk * inv
        out = numpy_helper.from_array(
            new.astype(np.float16 if w_init.data_type == 10 else np.float32),
            w_name)
        dag_model.initializers[w_name] = out

    return _Fold(apply_fn=_apply, consumed=f"norm '{norm.name}'")


def _fold_into_graph_input(dag_model, consumer, idx, name, s):
    """1/s into a graph input via a constant Mul.

    The constant is 1 everywhere but the last axis, where it is 1/s. The
    runtime applies an elementwise Mul over the flat tensor with numpy-style
    right-aligned broadcasting, so the const's rank must equal the input's
    rank for the broadcast to line up with the channel axis.
    """
    shape = consumer.inputs[idx].get("shape") or []
    rank = len(shape)
    if rank == 0:
        return _Fold(reason=f"input '{name}' has unknown rank")
    k = s.size
    dims = [d for d in shape]
    if dims and isinstance(dims[-1], int) and dims[-1] > 0 and dims[-1] != k:
        return _Fold(
            reason=f"input '{name}' last dim is {dims[-1]}, weight K is {k}")
    inv = (1.0 / s).reshape([1] * (rank - 1) + [k])

    def _apply():
        from .dag import Node

        const_name = f"{name}_awq_inv_scale"
        # The MLIR-shape declaration for a graph input carries no dtype, and
        # the Mul shader is chosen from its operands: without one, an fp16
        # activation meets an fp32 const and the runtime either downcasts the
        # constant (wrong on an fp16 grid) or refuses the pair.
        out_dtype = consumer.inputs[idx].get("dtype")
        inv_dtype = out_dtype if out_dtype else TensorProto.FLOAT
        inv_np = np.float16 if inv_dtype == TensorProto.FLOAT16 else np.float32
        dag_model.initializers[const_name] = numpy_helper.from_array(
            np.ascontiguousarray(inv, dtype=inv_np), const_name)
        mul = Node(
            op_type="Mul",
            name=f"{name}_awq_scale",
            attributes={},
            inputs=[{"name": name, "shape": list(shape)},
                    {"name": const_name, "shape": list(inv.shape)}],
            outputs=[{"name": f"{name}_awq_scaled", "shape": list(shape),
                      "dtype": consumer.inputs[idx].get("dtype")}],
        )
        dag_model.nodes[mul.name] = mul
        consumer.inputs[idx] = dict(mul.outputs[0])

    return _Fold(apply_fn=_apply, consumed=f"graph input '{name}'")


def _eligible(dag_model, name, consumers, fmt, group_size):
    """Which format's gate applies, delegated to the quantizer's own predicate.

    AWQ scales a weight by s and folds 1/s into its input; that transform is
    format-independent (a real weight stays a real weight), but whether the
    quantizer will then accept the weight is not. Calling the same
    `Quantizer._match_weight_only` the -q passes call means the two sets are
    identical by construction: AWQ never puts s on a weight that would then be
    left float (an unscaled 1/s, a silently wrong model), and never skips one
    the quantizer would have taken.

    4-bit and the byte formats disagree on `group_size`, so it is passed
    through only where the format has a notion of one.
    """
    from .optimizer import Quantizer

    gate_fmt = "int4" if fmt in ("int4", "nf4", "nvfp4") else fmt
    gate_group = group_size if gate_fmt == "int4" else 0
    return Quantizer._match_weight_only(
        name, dag_model.initializers[name], consumers, gate_fmt, gate_group)


def apply(dag_model, group_size: int, fmt: str = "int4",
          stats=None, alpha: float = 0.5, n_grid: int = 20,
          calib_ratio: float = 0.0, names=None):
    """Scale MatMul weights by s and fold 1/s into each one's input.

    Returns the set of weight initializer names that now carry an s factor.
    Only those go through the symmetric per-group 4-bit encode; a weight whose
    1/s cannot be folded is left untouched here and encoded by the plain
    quantizer exactly as `-q int4` would — never scaled without its 1/s.

    `fmt` selects which quantizer's gate decides eligibility (int8, fp8e4m3,
    fp8e5m2, int4, nf4, nvfp4). The scale search itself does not depend on it
    except for the grouping used when scoring candidates.

    `stats` maps a MatMul *input tensor name* to its per-channel absmax; a
    MatMul whose A name is not in it falls back to the weight-only proxy.
    `names`, when given, restricts the pass to those initializers.
    """
    stats = stats or {}
    initializer_consumers = defaultdict(list)
    for node in dag_model.nodes.values():
        for inp in node.inputs:
            initializer_consumers[inp["name"]].append(node)

    scaled_names = set()
    skipped = []
    scaled = 0

    for name in list(dag_model.initializers.keys()):
        if names is not None and name not in names:
            continue
        init = dag_model.initializers[name]
        consumers = initializer_consumers.get(name, [])
        gate = _eligible(dag_model, name, consumers, fmt, group_size)
        if not gate["eligible"]:
            continue
        dims = gate["dims"]
        k = gate["k"]

        for node in consumers:
            a = node.inputs[0]
            a_name = a["name"]
            a_shape = [d for d in (a.get("shape") or [])]
            if a_shape and isinstance(a_shape[-1], int) and a_shape[-1] > 0 \
                    and a_shape[-1] != k:
                # The B weight's K is not the A's last axis: a transposed A,
                # or a batched weight. No fold this pass can place.
                continue

            w = np.asarray(numpy_helper.to_array(init), dtype=np.float32)
            s = solve_scales(w, stats.get(a_name), alpha=alpha, n_grid=n_grid,
                             group_size=group_size, fmt=fmt,
                             calib_ratio=calib_ratio)
            fold = _find_fold(dag_model, node, s)
            if fold.apply_fn is None:
                skipped.append((name, node.name, fold.reason))
                print(f"AWQ: leaving '{name}' (MatMul '{node.name}') on the "
                      f"plain grid: {fold.reason}")
                continue

            # The fold does not touch B, and to_array re-read a fresh copy, so
            # the weight update is safe whether or not another consumer of the
            # same initializer already put 1/s somewhere.
            fold.apply_fn()
            new_w = w * s[None, :]
            if init.data_type == 10:
                # A half weight stays half: the kernels read it at that width,
                # and the fold's (W·s)·(1/s) identity has to hold in the same
                # arithmetic the model runs in.
                new_w = new_w.astype(np.float16)
            dag_model.initializers[name] = numpy_helper.from_array(new_w, name)
            init = dag_model.initializers[name]
            scaled_names.add(name)
            scaled += 1
            print(f"AWQ: scaled '{name}' {list(dims)} (1/s -> {fold.consumed})")

    print(f"AWQ: scaled {scaled} MatMul weight(s), "
          f"{len(skipped)} consumer(s) left on the plain grid")
    return scaled_names
