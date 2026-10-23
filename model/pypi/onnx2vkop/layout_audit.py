"""Conversion-time audit of the consumer-driven weight layouts.

Every consumer-driven weight layout in this repo is a node attribute a kernel
reads natively -- transB/transA, input_untransposed, seq_major, v_component,
weight_transpose/weight_pack -- and none of them moves a byte. A weight is
stored once, in the layout its exporter chose, and it is the kernel that adapts.
That is the property this module checks, because it is the one property no
single pass owns: it emerges from a dozen decisions spread across the fusion
passes (a folded Transpose sets transB), the weight-only fold (a Q/DQ consumer
is re-pointed at its payload initializer) and quantization (which fixes the
exponent, hence the element size). An audit has to be able to say "on this
model, no weight sits in a layout a kernel cannot read at speed", and to keep
saying it as any of those passes change.

The 2026-10 audit that motivated this was a throwaway script over one model's
blob. It keyed off initializer-named node inputs, so two shapes of graph slipped
past it: a `unified: true` artifact and a Q/DQ graph whose weight feeds a
DequantizeLinear rather than the MatMul directly. This walks the CONSUMERS
instead -- for every MatMul/Gemm, resolve input 1 against
dag_model.initializers -- and runs at the one point in ModelConverter.
apply_optimizations where those consumers have already been re-pointed at their
real weights and quantization has already fixed each weight's element size, but
RGBA (which rewrites a 4-D initializer's dims into its packed form) has not yet.
So it sees exactly the weights the kernels will see, in the form they read them.

Three checks, each one a line of a kernel, and nothing invented:

  * pow2_row_stride -- the shared-memory tiled MatMul stages B 64 columns per
    row (ops/Matmul.hpp:845-866). When the weight's byte row stride lands on a
    power of two of 64 KiB or more the staging reads lose miss-level
    parallelism and fall to 6-13 GB/s where the identical shape shifted by 64
    bytes runs at 35-54 GB/s. The runtime makes a tiled-vs-GEMV choice on it
    (the stride is in b_row_bytes, ops/Matmul.hpp:862) but a weight that keeps
    the shape is a permanent 4-6x on its own kernel.
  * k_not_multiple_of_16 -- k % 16 keeps every tile word aligned on both B
    layouts, so an fp16 weight that misses it never reaches the tiled or
    cooperative-matrix path (ops/Matmul.hpp:867).
  * byte_quad_ineligible -- the byte-quantized tile loader reads four columns of
    one K row out of one weight word, which needs transB == 0 and n % 4 == 0
    (ops/Matmul.hpp:811, 841). A byte weight that misses either loses the quad
    path and its tile stages decoded values per row instead.

Row stride is n * element_bytes because that is what the kernel computes
(ops/Matmul.hpp:862) -- n is the output-column count, and it is n regardless of
transB, so the check is layout-symmetric.

This is a REPORT, not a gate. A performance hazard must never fail a
conversion, so the default prints a one-line summary; VKOP_LAYOUT_AUDIT=full
lists every weight, and =strict is the CI mode that also raises. The structural
half of the contract -- a Conv with a constant weight carrying a self-consistent
image layout -- IS fail-closed, and lives in
ModelConverter._validate_conv_weight_layout. A Conv weight is tallied here only
so the summary accounts for every weight in the model; its image layout is not a
row-stride property, so no hazard class is defined for it.

Nothing here is auto-fixed. Every "fix" for the checks above is a transpose of
the stored payload -- bytes moving, the one thing the read-attribute layouts
exist to avoid -- and the converter has no license to rewrite an exported
weight's layout behind the model's back. The hazard line names the export that
would avoid the hazard instead.
"""
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# The byte-size tables are the writer's (dag.py:117, _init_byte_len). Importing
# the private names rather than copying them is deliberate: the audit is only
# worth anything while it agrees with the kernels about how many bytes a weight
# element is, and a second copy of that table is a second thing to drift.
from .dag import _DTYPE_BYTES, _PACKED_BITS, _PACKED_DATA_TYPES

# Each threshold is the kernel's own constant, cited where it is used.
_CLIFF_MIN_BYTES = 65536  # ops/Matmul.hpp:864
_TILE_K = 16  # ops/Matmul.hpp:867
_QUAD_COLUMNS = 4  # ops/Matmul.hpp:811

# The GEMM family: the ops whose consumer-driven weight layout is B's.
_GEMM_OPS = ("MatMul", "Gemm")

_ENV = "VKOP_LAYOUT_AUDIT"
_MODES = ("off", "summary", "full", "strict")


def elem_bits(data_type: int) -> Optional[int]:
    """Bit width of one weight element, or None for a dtype we do not know.

    None is not 0: an unrecognized element type means the row stride cannot be
    computed, and the caller must skip the row rather than guess a width.
    4-bit types (uint4/int4/float4e2m1fn) report 4 here even though their packed
    row is byte-aligned; the callers exclude them from every check because the
    tiled path refuses 4-bit outright (ops/Matmul.hpp:815).
    """
    if data_type in _PACKED_DATA_TYPES:
        return _PACKED_BITS
    nbytes = _DTYPE_BYTES.get(data_type)
    return nbytes * 8 if nbytes else None


def _pow2(value: int) -> bool:
    return value > 0 and (value & (value - 1)) == 0


@dataclass(frozen=True)
class WeightRow:
    """One constant weight consumed at a MatMul/Gemm B position.

    k is the reduced dim and n the output-column dim, resolved through transB:
    transB=0 stores B as [K, N], transB=1 as [N, K]. row_bytes is None when the
    weight cannot reach the tiled staging path at all (fp32, 4-bit, or an
    unknown dtype), which is also when every check is vacuous.
    """

    tensor: str
    node: str
    op_type: str
    data_type: int
    dims: Tuple[int, ...]
    transposed: bool
    k: int
    n: int
    row_bytes: Optional[int]
    flags: Tuple[str, ...] = ()

    @property
    def hazards(self) -> bool:
        return bool(self.flags)


@dataclass
class LayoutAudit:
    """The audit result: one row per GEMM-family weight, plus a Conv tally."""

    weights: List[WeightRow] = field(default_factory=list)
    # Conv weights are counted, not modelled: their layout is the node's
    # weight_transpose/weight_pack, validated fail-closed elsewhere.
    conv_total: int = 0
    conv_packed: int = 0
    conv_other_rank: int = 0
    # Consumers whose weight input was not a plain initializer (a live tensor, a
    # rank-<2 operand, a Constant never materialized). Counted so a summary that
    # reports nothing is distinguishable from one that checked nothing.
    skipped: int = 0

    @property
    def hazards(self) -> List[WeightRow]:
        return [r for r in self.weights if r.hazards]

    def _tallies(self) -> Dict[str, int]:
        return {
            "pow2_row_stride": sum("pow2_row_stride" in r.flags for r in self.weights),
            "k_not_multiple_of_16": sum(
                "k_not_multiple_of_16" in r.flags for r in self.weights
            ),
            "byte_quad_ineligible": sum(
                "byte_quad_ineligible" in r.flags for r in self.weights
            ),
        }

    def summary_line(self) -> str:
        t = self._tallies()
        parts = [
            f"[layout] {len(self.weights)} GEMM weight(s): "
            f"{t['pow2_row_stride']} pow2-stride, "
            f"{t['k_not_multiple_of_16']} k%{_TILE_K}, "
            f"{t['byte_quad_ineligible']} byte-quad",
        ]
        if self.conv_total:
            parts.append(
                f"{self.conv_total} conv (packed {self.conv_packed}, "
                f"rank!=4 {self.conv_other_rank})"
            )
        if self.skipped:
            parts.append(f"{self.skipped} non-constant weight(s)")
        line = "; ".join(parts) + "."
        if self.hazards:
            line += (
                f" {len(self.hazards)} HAZARD(s) -- see {_ENV}=full."
            )
        return line

    def hazard_lines(self) -> List[str]:
        out = []
        for row in self.hazards:
            out.append(
                f"[layout] WARNING {row.tensor} ({row.op_type} {row.node}): "
                f"{', '.join(row.flags)} -- {_explain(row)}"
            )
        return out

    def full_lines(self) -> List[str]:
        out = []
        for row in self.weights:
            if row.hazards:
                continue  # printed by hazard_lines(), with the explanation
            where = f"k={row.k} n={row.n} {'[N,K]' if row.transposed else '[K,N]'}"
            stride = "stride n/a" if row.row_bytes is None else f"stride {row.row_bytes}B"
            out.append(f"[layout]   {row.tensor} ({row.op_type}): {where}, {stride}")
        return out


def _explain(row: WeightRow) -> str:
    bits = []
    if "pow2_row_stride" in row.flags:
        # No transB hint: the stride is n * elem_bytes under both layouts
        # (ops/Matmul.hpp:862), so transposing B does not move it. The only
        # escape measured is a different N (a 64-byte shift, byte N = 65600 ->
        # back to 37 GB/s), which is a different export, not a read flag.
        bits.append(
            f"byte row stride {row.row_bytes} is a power of two >= "
            f"{_CLIFF_MIN_BYTES}: tiled B staging falls to 6-13 GB/s where the "
            f"same shape shifted 64 bytes runs at 35-54 GB/s "
            f"(ops/Matmul.hpp:845-866)"
        )
    if "k_not_multiple_of_16" in row.flags:
        bits.append(
            f"k={row.k} is not a multiple of {_TILE_K}: no fp16 tiled/coopmat "
            f"path (ops/Matmul.hpp:867)"
        )
    if "byte_quad_ineligible" in row.flags:
        why = "transB=1" if row.transposed else f"n={row.n}"
        bits.append(
            f"{why}: the byte quad loader needs transB=0 and n%{_QUAD_COLUMNS}==0 "
            f"(ops/Matmul.hpp:811, 841)"
        )
    return ". ".join(bits)


def _classify(bits: Optional[int], k: int, n: int, transposed: bool,
              row_bytes: Optional[int]) -> Tuple[str, ...]:
    flags = []
    if row_bytes is not None and row_bytes >= _CLIFF_MIN_BYTES and _pow2(row_bytes):
        flags.append("pow2_row_stride")
    if bits == 16 and k % _TILE_K != 0:
        flags.append("k_not_multiple_of_16")
    if bits == 8 and (transposed or n % _QUAD_COLUMNS != 0):
        flags.append("byte_quad_ineligible")
    return tuple(flags)


def audit_layout(dag_model) -> LayoutAudit:
    """Walk every weight position and check the layout its consumer needs.

    Pure and read-only: it decodes no payload (dims and data_type are enough,
    which matters on a multi-GB weight set) and mutates nothing.
    """
    audit = LayoutAudit()
    for node in dag_model.nodes.values():
        if node.op_type in _GEMM_OPS:
            if len(node.inputs) < 2:
                continue
            init = dag_model.initializers.get(node.inputs[1]["name"])
            if init is None:
                audit.skipped += 1
                continue
            dims = tuple(int(d) for d in init.dims)
            transposed = int((node.attributes or {}).get("transB", 0) or 0) != 0
            if len(dims) < 2:
                # A rank-<2 B operand: the runtime rejects it (ops/Matmul.hpp:
                # 529), so there is no layout to judge here.
                audit.skipped += 1
                continue
            # transB=0 stores B as [K, N], transB=1 as [N, K].
            k, n = (dims[-1], dims[-2]) if transposed else (dims[-2], dims[-1])
            bits = elem_bits(init.data_type)
            # Only the fp16 and byte tiled paths compute a row stride; fp32 has
            # no tile at all, and 4-bit is refused by the tile outright (ops/
            # Matmul.hpp:815), so its row stride is never read row-by-row.
            if bits in (8, 16):
                row_bytes = n * (bits // 8)
            else:
                row_bytes = None
            row = WeightRow(
                tensor=node.inputs[1]["name"],
                node=node.name,
                op_type=node.op_type,
                data_type=int(init.data_type),
                dims=dims,
                transposed=transposed,
                k=k,
                n=n,
                row_bytes=row_bytes,
                flags=_classify(bits, k, n, transposed, row_bytes),
            )
            audit.weights.append(row)
        elif node.op_type in ("Conv", "Conv2d") and len(node.inputs) >= 2:
            init = dag_model.initializers.get(node.inputs[1]["name"])
            if init is None:
                audit.skipped += 1
                continue
            audit.conv_total += 1
            attrs = node.attributes or {}
            rank = len(init.dims)
            if rank == 4 and attrs.get("weight_pack") and attrs.get("weight_transpose"):
                audit.conv_packed += 1
            elif rank != 4:
                audit.conv_other_rank += 1
    return audit


def layout_audit_mode() -> str:
    """The active mode, validated. Unknown values are a typo, not a default."""
    raw = os.environ.get(_ENV, "summary").strip().lower()
    if raw in ("", "1", "true", "on", "yes"):
        return "summary"
    if raw in ("0", "false", "no", "none"):
        return "off"
    if raw not in _MODES:
        raise ValueError(
            f"{_ENV}={raw!r} is not one of {', '.join(_MODES)}"
        )
    return raw


def report_layout_audit(dag_model) -> LayoutAudit:
    """Audit, print per VKOP_LAYOUT_AUDIT, and return the result.

    summary (default) prints any hazard with its explanation plus the one-line
    tally; full prints the tally and then every weight row; off prints nothing.
    strict is full plus the caller's hard failure -- the raise lives in the
    caller (converter._audit_weight_layouts) because this module does not import
    the converter's exception, and a hazard has to be able to be reported
    "full" without also being fatal.
    """
    audit = audit_layout(dag_model)
    mode = layout_audit_mode()
    if mode == "off":
        return audit
    lines = audit.hazard_lines()
    if mode in ("full", "strict"):
        lines += audit.full_lines()
    for line in lines:
        print(line)
    print(audit.summary_line())
    return audit
