"""Conversion-time audit of what a format costs the weights it is applied to.

The accuracy loop (tools/vkop_vs_ort.py) answers "did this artifact survive its
format" by running it next to ONNX Runtime, which needs a GPU, a reference
model and a minute per variant. Most of the answer is available before any of
that: for a weight-only format the runtime reads back exactly
`decode(encode(W))`, so the error the format introduces is computable from the
two arrays the quantizer is already holding. This module collects those numbers
while -q runs and reports them once, per weight, at the end of the quantization
block -- so the conversion of a model says which of its own weights the chosen
format is bad at, instead of leaving that to be discovered downstream.

Three numbers, per weight:

  * rel_l2 -- `||W - restored||_2 / ||W||_2`, the same formula the loop uses on
    model outputs (tools/vkop_vs_ort.py:361). It is the headline because it is
    the one that tracks end-to-end damage. On resnet18 the two ends of that
    claim line up: per-weight median 2.07e-04 (-q fp16, 42 weights) against
    7.45e-04 measured at the model output, and 1.27e-02 (-q int8, 21 weights)
    against 3.92e-02 -- a factor of 3.5ish in both cases, which is what passing
    through 21 layers does to an error that starts at each weight. The int8
    number is the quantizer's and not the kernels' (dequantizing the shipped
    codes and scale tables and replaying them in onnxruntime matches the GPU
    output to 9.5e-06), so the per-tensor figure here names which weights paid
    for a model-level failure the loop would only report after a run.
  * overflow -- a restored value that is infinite or NaN while the source was
    finite. FP16 saturates past 65504 and the domain pass downcasts anyway
    (fp16_graph.py), because a weight that large only reaches an fp16 kernel
    once the graph is in the domain; naming it here is what turns that defined
    outcome into a decision the user can see before running the model.
  * subnormal share -- magnitudes below the format's smallest normal value,
    which keep fewer mantissa bits than the format's advertised precision. A
    weight tensor whose distribution sits mostly under 2**-14 is not fp16, it is
    fp16-with-a-short-mantissa, and nothing else in the pipeline says so.

REPORT-ONLY ON PURPOSE. A numeric hazard makes a wrong-ish model, not a broken
one, and the alternative to converting is the user re-running with a narrower
format they have not measured; a hard failure would decide their accuracy budget
for them. So there is no `strict` mode here, unlike VKOP_LAYOUT_AUDIT (a weight
in a slow layout is a performance question with a correct answer either way,
which is why that one may be made fatal in CI). Raise it from the caller if a
particular model ever needs that.

Modes: VKOP_NUMERIC_AUDIT / --numeric-audit = off | summary | full. `summary`
(default) prints the tally plus a line per weight above its format's threshold;
`full` prints every weight. Collection itself always happens -- it is arithmetic
on arrays the quantizer already built -- so the rows are available to callers and
tests with the printing switched off.
"""
import contextlib
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

_ENV = "VKOP_NUMERIC_AUDIT"
_MODES = ("off", "summary", "full")

# Smallest NORMAL value per float format (byte-int formats have none, which is
# what `None` means below): 2**-14 for IEEE half, and the E4M3/E5M2 normals.
# The byte formats' own clip against Inf/NaN lives with the encoder
# (optimizer.py's _FP8_FORMATS), which already refuses a bad code outright.
_MIN_NORMAL: Dict[str, Optional[float]] = {
    "fp16": 2.0 ** -14,
    "fp8e4m3": 2.0 ** -6,
    "fp8e5m2": 2.0 ** -14,
}

# Per-format rel-L2 threshold above which a weight is named. These are NOT the
# format's own error -- every one sits at about twice the WORST per-weight error
# measured on resnet18's 11.66M weight elements (21 tensors, each read in its
# GEMM view [input fan-in, output channels], so the 4-bit and fp8 passes see a
# shape they accept; -q fp16/int8 measured through the real pipeline, where
# Conv is in range):
#
#   fp16 2.07e-04 median / 2.84e-04 worst      int8 1.27e-02 / 1.60e-02
#   fp8e4m3 2.61e-02 / 2.64e-02                fp8e5m2 5.19e-02 / 5.25e-02
#   nvfp4 9.32e-02 / 9.61e-02                  nf4 1.02e-01 / 1.16e-01
#   int4 1.37e-01 / 1.66e-01
#
# So a weight that lands here is an outlier inside its own format, not an
# example of what 4 bits cost: a single threshold for every format would drown
# -q int4 in warnings (its usual cost is 7x int8's) and stay silent on an fp16
# weight that is 100x worse than fp16 should be. The spread within a format is
# narrow -- 1.2x from median to worst for fp16 and int8, under 1.1x for the
# rest -- which is what makes "2x the worst" a usable line rather than a coin
# flip.
_REL_L2_WARN: Dict[str, float] = {
    "fp16": 5e-4,
    "int8": 3e-2,
    "fp8e4m3": 5e-2,
    "fp8e5m2": 1e-1,
    "nvfp4": 2e-1,
    "nf4": 2e-1,
    "int4": 3e-1,
}

# The conversions currently being audited, innermost last. A stack rather than
# one global so a nested -q (AWQ running a byte quantizer for its own format)
# cannot end the outer audit by accident.
_STACK: List["NumericAudit"] = []


@dataclass
class WeightRow:
    """One weight, as the format that was applied to it left it."""

    tensor: str
    fmt: str
    elements: int
    rel_l2: float
    max_abs: float
    overflow: int = 0
    subnormals: int = 0
    # The weight's own name is often an exporter's opaque `onnx::Conv_226`,
    # which tells the reader nothing about which layer to hold wide; the
    # consumer's node name is what --exclude-node takes.
    via: str = ""
    # Slices the quantizer had to give up on: a channel/group whose absmax was
    # zero, so its scale fell back to 1.0 (or a block whose scale rounded to
    # zero). Not an error -- a zero channel stays a zero channel -- but it is
    # the one case where the stored scale table stops describing the weights.
    dead_slices: int = 0

    @property
    def subnormal_share(self) -> float:
        return self.subnormals / self.elements if self.elements else 0.0

    @property
    def flagged(self) -> bool:
        return (self.rel_l2 > _REL_L2_WARN.get(self.fmt, float("inf"))
                or self.overflow > 0)

    def line(self) -> str:
        extra = ""
        if self.overflow:
            extra += f" OVERFLOW {self.overflow}"
        if self.subnormals:
            extra += f" subnormal {self.subnormal_share:.1%}"
        if self.dead_slices:
            extra += f" dead-scale {self.dead_slices}"
        mark = "WARN " if self.flagged else "     "
        via = f" via {self.via}" if self.via else ""
        return (f"[numeric-audit] {mark}{self.fmt:8s} {self.tensor}: "
                f"relL2 {self.rel_l2:.3e} maxabs {self.max_abs:.3e}"
                f" ({self.elements} elem{via}){extra}")


@dataclass
class NumericAudit:
    rows: List[WeightRow] = field(default_factory=list)
    skipped: int = 0

    def flagged(self) -> List[WeightRow]:
        return [r for r in self.rows if r.flagged]

    def by_format(self) -> Dict[str, List[WeightRow]]:
        out: Dict[str, List[WeightRow]] = {}
        for r in self.rows:
            out.setdefault(r.fmt, []).append(r)
        return out

    def summary_line(self) -> str:
        if not self.rows:
            return ("[numeric-audit] no weight round-tripped through a "
                    "quantizer on this model")
        parts = []
        for fmt, rows in sorted(self.by_format().items()):
            errs = np.array([r.rel_l2 for r in rows], dtype=np.float64)
            over = sum(r.overflow for r in rows)
            dead = sum(r.dead_slices for r in rows)
            p = (f"{fmt}: {len(rows)} weight(s), median relL2 "
                 f"{np.median(errs):.2e}, worst {errs.max():.2e}")
            if over:
                p += f", {over} overflow"
            if dead:
                p += f", {dead} dead-scale slice(s)"
            parts.append(p)
        n_flag = len(self.flagged())
        tail = f" -- {n_flag} above threshold" if n_flag else ""
        return "[numeric-audit] " + "; ".join(parts) + tail

    def hazard_lines(self) -> List[str]:
        out = []
        for fmt, rows in sorted(self.by_format().items()):
            limit = _REL_L2_WARN.get(fmt)
            bad = [r for r in rows if r.flagged]
            if not bad:
                continue
            out.append(
                f"[numeric-audit] WARNING: {len(bad)} {fmt} weight(s) cost more "
                f"than the format's usual relL2"
                + (f" ({limit:.0e})" if limit else "")
                + ", so the artifact is expected to differ from its fp32 "
                  "source at these tensors first:"
            )
            out += [r.line() for r in sorted(bad, key=lambda r: -r.rel_l2)]
        return out

    def full_lines(self) -> List[str]:
        return [r.line() for r in sorted(self.rows, key=lambda r: -r.rel_l2)]


def numeric_audit_mode() -> str:
    """The active mode, validated. Unknown values are a typo, not a default."""
    raw = os.environ.get(_ENV, "summary").strip().lower()
    if raw in ("", "1", "true", "on", "yes"):
        return "summary"
    if raw in ("0", "false", "no", "none"):
        return "off"
    if raw not in _MODES:
        raise ValueError(f"{_ENV}={raw!r} is not one of {', '.join(_MODES)}")
    return raw


def begin() -> NumericAudit:
    """Start collecting rows for one conversion. Idempotent per nesting level."""
    audit = NumericAudit()
    _STACK.append(audit)
    return audit


def finish(mode: Optional[str] = None) -> NumericAudit:
    """Stop collecting, print per the mode, and return what was gathered."""
    audit = _STACK.pop() if _STACK else NumericAudit()
    report(audit, mode)
    return audit


@contextlib.contextmanager
def collecting():
    """`with numeric_audit.collecting() as audit:` for one quantization block.

    Tests and one-shot callers use this; the pipeline uses begin/finish because
    the block it spans is 60 lines of -q dispatch that a `with` would only
    re-indent.
    """
    audit = begin()
    try:
        yield audit
    finally:
        finish()


def consumer_tag(consumers) -> str:
    """The node name to show beside a weight: the first consumer, which is the
    one an --exclude-node has to name to keep that weight wide."""
    node = consumers[0] if consumers else None
    return getattr(node, "name", "") or ""


def record_roundtrip(tensor: str, fmt: str, source: np.ndarray,
                     restored: np.ndarray, dead_slices: int = 0,
                     via: str = "") -> None:
    """Name what `fmt` did to one weight, from the two arrays it is between.

    `source` is the weight the quantizer read, `restored` is what the kernel
    will read back: the same float the runtime decodes, not the stored codes.
    Every caller already has both (each -q path computes its own diff for the
    per-tensor error print), so this adds two reductions and no copies.

    A no-op when nothing is collecting, so a quantizer used outside the
    pipeline (a test, or a caller that imports Optimizer directly) simply does
    not contribute rows.
    """
    if not _STACK:
        return
    src = np.asarray(source, dtype=np.float64).reshape(-1)
    got = np.asarray(restored, dtype=np.float64).reshape(-1)
    if src.size != got.size:
        # A view change (the grouped formats compare in [groups, size, N]) is
        # fine; a size change means the caller handed over a different tensor.
        raise ValueError(
            f"numeric audit: '{tensor}' restored {got.size} element(s) from "
            f"{src.size}")
    diff = got - src
    norm = float(np.linalg.norm(src))
    if norm == 0.0:
        # A weight of all zeros (a padding channel) has no relative scale to be
        # wrong against, but a format that fails to reproduce zero does.
        rel_l2 = 0.0 if not np.any(diff) else float("inf")
    else:
        rel_l2 = float(np.linalg.norm(diff)) / norm
    finite_in = np.isfinite(got)
    overflow = int(np.count_nonzero(~finite_in & np.isfinite(src)))
    subnormals = 0
    min_normal = _MIN_NORMAL.get(fmt)
    if min_normal is not None:
        subnormals = int(np.count_nonzero(
            (np.abs(got) < min_normal) & (got != 0.0) & finite_in))
    _STACK[-1].rows.append(WeightRow(
        tensor=tensor, fmt=fmt, elements=int(src.size), rel_l2=rel_l2,
        max_abs=float(np.max(np.abs(diff))) if diff.size else 0.0,
        overflow=overflow, subnormals=subnormals, dead_slices=dead_slices,
        via=via))


def report(audit: NumericAudit, mode: Optional[str] = None) -> NumericAudit:
    """Print per VKOP_NUMERIC_AUDIT and return the audit. Never raises."""
    mode = mode or numeric_audit_mode()
    if mode == "off":
        return audit
    if not audit.rows:
        # A conversion with no -q has nothing to say about formats it did not
        # apply, so `summary` stays quiet; `full` still reports that it looked.
        if mode == "full":
            print(audit.summary_line())
        return audit
    for line in audit.hazard_lines():
        print(line)
    if mode == "full":
        for line in audit.full_lines():
            print(line)
    print(audit.summary_line())
    return audit
