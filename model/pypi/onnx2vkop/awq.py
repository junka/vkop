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
        # s is per input channel, i.e. per ROW of the [K, N] weight.
        out = _group_error(w * s[:, None], s, group_size, fmt)
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
    err = (deq - scaled) / s[:, None]
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

class _Site:
    """Where one MatMul's 1/s would be folded: a norm, or one edge off a graph
    input. `key` identifies the site, so MatMuls that read the same norm output
    share one; `reason` is set instead when there is nowhere exact to fold."""

    def __init__(self, kind=None, key=None, node=None, idx=0, name=None,
                 reason=None):
        self.kind = kind          # "norm" | "input"
        self.key = key
        self.node = node          # the norm, or the consumer of the input
        self.idx = idx
        self.name = name          # the graph input's name (kind == "input")
        self.reason = reason


def _producers(dag_model):
    """tensor name -> the node that produces it (None for graph inputs /
    initializers, i.e. a name nobody in the node map writes)."""
    prod = {}
    for node in dag_model.nodes.values():
        for out in node.outputs:
            prod[out["name"]] = node
    return prod


def _readers(dag_model):
    """tensor name -> [(node, input index)]."""
    out = defaultdict(list)
    for node in dag_model.nodes.values():
        for i, inp in enumerate(node.inputs):
            out[inp["name"]].append((node, i))
    return out


def _moves_channel_axis(node) -> bool:
    """A Transpose whose last output axis is not the input's last axis puts a
    different axis where K was: the per-channel 1/s would land on the wrong
    values (the length checks miss it when two axes are equally long)."""
    if node.op_type != "Transpose":
        return False
    perm = node.attributes.get("perm")
    if not perm:
        return True  # the default perm reverses every axis
    return int(perm[-1]) != len(perm) - 1


def _find_site(dag_model, mm_node, producers) -> _Site:
    """Walk up from the MatMul's A to a norm or a graph input.

    The walk crosses only metadata-only views that keep the channel axis last.
    Anything else -- an op that is not a view, a Transpose that moves the
    channel axis -- is reported, not approximated: a misaligned 1/s is a
    silently wrong model, the one outcome a fold must never produce.
    """
    node = mm_node
    idx = 0
    depth = 0
    while depth <= _MAX_DEPTH:
        depth += 1
        name = node.inputs[idx]["name"]
        if name in dag_model.initializers:
            return _Site(reason=f"input '{name}' is an initializer")
        producer = producers.get(name)
        if producer is None:
            return _Site(kind="input", key=("input", node.name, idx),
                         node=node, idx=idx, name=name)
        if producer.op_type in _NORM_OPS:
            if producer.outputs[0]["name"] != name:
                # A skip-LayerNorm's residual-sum output is not scaled by the
                # norm weight.
                return _Site(reason=f"input is a secondary output of "
                                    f"{producer.op_type} '{producer.name}'")
            return _Site(kind="norm", key=("norm", producer.name),
                         node=producer)
        if producer.op_type not in _VIEW_OPS:
            return _Site(
                reason=f"input comes from {producer.op_type} '{producer.name}'")
        if _moves_channel_axis(producer):
            return _Site(reason=f"Transpose '{producer.name}' moves the "
                                f"channel axis")
        node = producer
        idx = 0
    return _Site(reason="view chain too long")


def _scaled_readers(dag_model, site, readers):
    """Every MatMul/Gemm whose A carries the site's 1/s, or a reason.

    For a norm that is every reader of its output, through views; for a graph
    input it is the one edge the Mul is inserted on (and, if that edge is a
    view, everything downstream of it). Each has to be a MatMul reading it as
    A whose weight is scaled by the SAME s -- any other reader (a residual
    Add, a graph output, a MatMul whose weight could not be scaled) would see
    1/s with nothing undoing it.
    """
    graph_outputs = {o["name"] for o in (dag_model.outputs or [])}
    if site.kind == "norm":
        first = site.node.outputs[0]["name"]
        if first in graph_outputs:
            return None, f"norm output '{first}' is a graph output"
        edges = list(readers.get(first, []))
    else:
        edges = [(site.node, site.idx)]
    found = {}
    seen = set()
    while edges:
        node, i = edges.pop()
        if (node.name, i) in seen:
            continue
        seen.add((node.name, i))
        if node.op_type in ("MatMul", "Gemm") and i == 0:
            found[node.name] = node
            continue
        if node.op_type in _VIEW_OPS and i == 0 \
                and not _moves_channel_axis(node):
            for out in node.outputs:
                if out["name"] in graph_outputs:
                    return None, f"'{out['name']}' is a graph output"
                edges.extend(readers.get(out["name"], []))
            continue
        return None, (f"{node.op_type} '{node.name}' also reads the scaled "
                      f"activation")
    return found, None


def _bf16_init_to_f32(init) -> np.ndarray:
    """A BFLOAT16 initializer's values as float32 -- exact (bf16 is fp32's
    high half), from raw_data or int32_data, whatever numpy_helper's bf16
    support on the installed onnx."""
    from .bf16_weights import bf16_bits_to_fp32
    if init.raw_data:
        bits = np.frombuffer(init.raw_data, dtype="<u2")
    else:
        bits = np.asarray(init.int32_data, dtype=np.uint32).astype(np.uint16)
    return bf16_bits_to_fp32(bits).reshape(tuple(int(d) for d in init.dims))


def _f32_to_bf16_init(values, name):
    from onnx import helper
    from .bf16_weights import fp32_to_bf16_bits
    bits = fp32_to_bf16_bits(np.asarray(values, dtype=np.float32))
    return helper.make_tensor(name, TensorProto.BFLOAT16, list(bits.shape),
                              np.ascontiguousarray(bits).astype("<u2")
                              .tobytes(), raw=True)


def _float_init_as_f32(init):
    """A FLOAT/FLOAT16/BFLOAT16 initializer's values as float32 (exact), or
    None for any other dtype."""
    if init.data_type == TensorProto.BFLOAT16:
        # BFLOAT16 is what the --act bf16 domain pass (which runs before AWQ)
        # makes of a norm scale, as the fp16 pass makes FLOAT16 of it.
        return _bf16_init_to_f32(init)
    if init.data_type in (TensorProto.FLOAT, TensorProto.FLOAT16):
        return np.asarray(numpy_helper.to_array(init), dtype=np.float32)
    return None


def _same_dtype_init(values, like, name):
    """`values` stored in `like`'s float dtype, rounded once (RNE)."""
    if like.data_type == TensorProto.BFLOAT16:
        return _f32_to_bf16_init(values, name)
    return numpy_helper.from_array(
        np.asarray(values, dtype=np.float16
                   if like.data_type == TensorProto.FLOAT16 else np.float32),
        name)


def _norm_fold(dag_model, norm, k):
    """The norm's per-channel initializers 1/s multiplies, or a reason.

    The norm emits normalize(x) * gamma (+ beta for LayerNorm): its output --
    the MatMul's A -- is scaled by 1/s per channel exactly when gamma AND
    beta are. Scaling gamma alone leaves beta's share of A unscaled.
    """
    if len(norm.inputs) < 2:
        return None, f"{norm.op_type} '{norm.name}' has no scale input"
    names = [norm.inputs[1]["name"]]
    if norm.op_type != "RMSNorm" and len(norm.inputs) > 2 \
            and norm.inputs[2].get("name"):
        names.append(norm.inputs[2]["name"])
    for nm in names:
        init = dag_model.initializers.get(nm)
        if init is None:
            return None, f"{norm.op_type} input '{nm}' is not constant"
        vals = _float_init_as_f32(init)
        if vals is None:
            return None, f"{norm.op_type} input '{nm}' is not float"
        if vals.ndim != 1:
            return None, f"{norm.op_type} input '{nm}' is not 1-D"
        if vals.size != k:
            return None, (f"{norm.op_type} scale has {vals.size} channels, "
                          f"weight K is {k}")
    return names, None


def _apply_norm_fold(dag_model, names, s):
    """Divide each of the norm's per-channel initializers by s, keeping its
    dtype (an fp16 or bf16 scale stays that width, rounded once)."""
    inv = (1.0 / np.asarray(s, dtype=np.float32)).astype(np.float32)
    for nm in names:
        init = dag_model.initializers[nm]
        vals = _float_init_as_f32(init)
        dag_model.initializers[nm] = _same_dtype_init(vals * inv, init, nm)


def _input_fold_refusal(site, k):
    shape = site.node.inputs[site.idx].get("shape") or []
    if len(shape) == 0:
        return f"input '{site.name}' has unknown rank"
    if isinstance(shape[-1], int) and shape[-1] > 0 and shape[-1] != k:
        return f"input '{site.name}' last dim is {shape[-1]}, weight K is {k}"
    return None


def _apply_input_fold(dag_model, site, s):
    """1/s into a graph input via a constant Mul on the one edge.

    The constant is 1 everywhere but the last axis, where it is 1/s. The
    runtime applies an elementwise Mul over the flat tensor with numpy-style
    right-aligned broadcasting, so the const's rank must equal the input's
    rank for the broadcast to line up with the channel axis.
    """
    from .dag import Node

    consumer, idx, name = site.node, site.idx, site.name
    shape = consumer.inputs[idx].get("shape") or []
    rank = len(shape)
    k = s.size
    inv = (1.0 / np.asarray(s, dtype=np.float32)).reshape(
        [1] * (rank - 1) + [k])
    # The MLIR-shape declaration for a graph input carries no dtype, and the
    # Mul shader is chosen from its operands: without one, a half activation
    # meets an fp32 const and the runtime either downcasts the constant (wrong
    # on a half grid) or refuses the pair.
    act_dtype = consumer.inputs[idx].get("dtype")
    mul_name = f"{name}_awq_scale"
    const_name = f"{name}_awq_inv_scale"
    out_name = f"{name}_awq_scaled"
    if mul_name in dag_model.nodes:
        # A second edge off the same graph input: its own Mul and constant.
        suffix = f"_{consumer.name}_{idx}"
        mul_name += suffix
        const_name += suffix
        out_name += suffix
    if act_dtype == TensorProto.BFLOAT16:
        dag_model.initializers[const_name] = _f32_to_bf16_init(inv, const_name)
    else:
        dag_model.initializers[const_name] = numpy_helper.from_array(
            np.ascontiguousarray(
                inv, dtype=np.float16 if act_dtype == TensorProto.FLOAT16
                else np.float32), const_name)
    mul = Node(
        op_type="Mul",
        name=mul_name,
        attributes={},
        inputs=[{"name": name, "shape": list(shape)},
                {"name": const_name, "shape": list(inv.shape)}],
        outputs=[{"name": out_name, "shape": list(shape), "dtype": act_dtype}],
    )
    dag_model.nodes[mul.name] = mul
    consumer.inputs[idx] = dict(mul.outputs[0])


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


def _as_kn(w, transb):
    """A stored B weight in [K, N] orientation (a transB=1 weight is [N, K])."""
    return w.T if transb else w


def scale_weight(w, s, transb=0):
    """W' = W * diag(s) along the INPUT-channel axis of a stored B weight:
    the rows of a [K, N] MatMul weight, the columns of a transB=1 [N, K] one.
    That is the axis A's channels -- which carry 1/s -- are contracted with."""
    w = np.asarray(w, dtype=np.float32)
    s = np.asarray(s, dtype=np.float32).reshape(-1)
    return w * (s[None, :] if transb else s[:, None])


def apply(dag_model, group_size: int, fmt: str = "int4",
          stats=None, alpha: float = 0.5, n_grid: int = 20,
          calib_ratio: float = 0.0, names=None, exclude=None):
    """Scale MatMul weights by s and fold 1/s into each one's input.

    Returns the set of weight initializer names that now carry an s factor.
    Only those go through the quantizer; a weight whose 1/s cannot be folded
    is left untouched here and encoded by the plain quantizer exactly as
    `-q <fmt>` would -- never scaled without its 1/s.

    One s per fold SITE, not per weight: a transformer's q/k/v projections
    (or gate/up) all read the same norm output, and that norm has one gamma.
    Solving each weight on its own and folding each 1/s into the shared gamma
    leaves projection i scaled by s_i but its input by 1/(s_q s_k s_v). So the
    weights behind one site are solved together (their [K, N_i] matrices side
    by side) and share the one s the site carries. A site that any other
    reader also sees -- a residual Add, a MatMul whose weight is excluded or
    ineligible -- is left alone.

    `fmt` selects which quantizer's gate decides eligibility (int8, fp8e4m3,
    fp8e5m2, int4, nf4, nvfp4). The scale search itself does not depend on it
    except for the grouping used when scoring candidates.

    `stats` maps a MatMul *input tensor name* to its per-channel absmax; a
    site none of whose MatMul A names is in it falls back to the weight-only
    proxy. `names`, when given, restricts the pass to those initializers.

    `exclude` drops initializers from the pass entirely. It has to be honoured
    here and not by filtering the return value: a weight already multiplied by
    s and missing the 1/s that went into its input is a wrong model, not a
    smaller one.
    """
    stats = stats or {}
    readers = _readers(dag_model)
    producers = _producers(dag_model)
    consumers_of = {nm: [n for n, _ in rs] for nm, rs in readers.items()}

    skipped = []

    def _skip(name, node, reason):
        skipped.append((name, node, reason))
        print(f"AWQ: leaving '{name}' (MatMul '{node}') on the plain grid: "
              f"{reason}")

    # 1. Every eligible weight, its layout, and the fold site of its consumers.
    cands = {}          # weight name -> {"transb", "k", "dims", "key"}
    sites = {}          # site key -> _Site
    for name in list(dag_model.initializers.keys()):
        if exclude is not None and name in exclude:
            print(f"AWQ: leaving '{name}' alone -- --exclude-* names it")
            continue
        if names is not None and name not in names:
            continue
        consumers = consumers_of.get(name, [])
        gate = _eligible(dag_model, name, consumers, fmt, group_size)
        if not gate["eligible"]:
            continue
        dims = list(gate["dims"])
        if len(dims) != 2 or any(n.op_type not in ("MatMul", "Gemm")
                                 for n in consumers):
            continue  # a byte format's Conv weight: no K axis to scale here
        transb = int(gate.get("transb") or 0)
        k = int(dims[1] if transb else dims[0])
        keys = set()
        reason = None
        for node in consumers:
            if int(node.attributes.get("transA", 0) or 0):
                reason = "transA=1 (A's channel axis is not its last)"
                break
            a_shape = node.inputs[0].get("shape") or []
            if a_shape and isinstance(a_shape[-1], int) and a_shape[-1] > 0 \
                    and a_shape[-1] != k:
                # The B weight's K is not the A's last axis: a transposed A,
                # or a batched weight. No fold this pass can place.
                reason = f"A's last dim {a_shape[-1]} is not K={k}"
                break
            site = _find_site(dag_model, node, producers)
            if site.reason is not None:
                reason = site.reason
                break
            keys.add(site.key)
            sites[site.key] = site
        if reason is None and len(keys) != 1:
            reason = "its consumers read differently-scaled inputs"
        if reason is not None:
            _skip(name, consumers[0].name, reason)
            continue
        cands[name] = {"transb": transb, "k": k, "dims": dims,
                       "key": next(iter(keys))}

    # 2. Group by site. A site is folded only if every MatMul its 1/s reaches
    #    has a candidate weight solved with it.
    groups = defaultdict(list)
    for name in sorted(cands):
        groups[cands[name]["key"]].append(name)

    scaled_names = set()
    for key, members in groups.items():
        site = sites[key]
        ks = {cands[m]["k"] for m in members}
        k = ks.pop() if len(ks) == 1 else None
        reason = None
        norm_names = None
        if k is None:
            reason = "weights behind one input disagree on K"
        else:
            found, reason = _scaled_readers(dag_model, site, readers)
            if reason is None:
                member_nodes = {n.name for m in members
                                for n in consumers_of[m]}
                stray = sorted(set(found) - member_nodes)
                if stray:
                    reason = (f"'{stray[0]}' reads the same input and its "
                              f"weight is not AWQ-scaled")
        if reason is None:
            if site.kind == "norm":
                norm_names, reason = _norm_fold(dag_model, site.node, k)
            else:
                reason = _input_fold_refusal(site, k)
        if reason is not None:
            for m in members:
                _skip(m, consumers_of[m][0].name, reason)
            continue

        weights = {m: np.asarray(numpy_helper.to_array(
            dag_model.initializers[m]), dtype=np.float32) for m in members}
        w_kn = np.concatenate(
            [_as_kn(weights[m], cands[m]["transb"]) for m in members], axis=1)
        act = None
        for m in members:
            for node in consumers_of[m]:
                act = stats.get(node.inputs[0]["name"])
                if act is not None:
                    break
            if act is not None:
                break
        s = solve_scales(w_kn, act, alpha=alpha, n_grid=n_grid,
                         group_size=group_size, fmt=fmt,
                         calib_ratio=calib_ratio)
        del w_kn

        if site.kind == "norm":
            _apply_norm_fold(dag_model, norm_names, s)
            consumed = f"norm '{site.node.name}'"
        else:
            _apply_input_fold(dag_model, site, s)
            consumed = f"graph input '{site.name}'"
        for m in members:
            init = dag_model.initializers[m]
            new_w = scale_weight(weights[m], s, cands[m]["transb"])
            if init.data_type == TensorProto.FLOAT16:
                # A half weight stays half: the kernels read it at that width,
                # and the fold's (W·s)·(1/s) identity has to hold in the same
                # arithmetic the model runs in.
                new_w = new_w.astype(np.float16)
            dag_model.initializers[m] = numpy_helper.from_array(new_w, m)
            scaled_names.add(m)
            print(f"AWQ: scaled '{m}' {cands[m]['dims']} (1/s -> {consumed})")

    print(f"AWQ: scaled {len(scaled_names)} MatMul weight(s), "
          f"{len(skipped)} weight(s) left on the plain grid")
    return scaled_names
