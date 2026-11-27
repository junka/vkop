"""bfloat16 weights in the two slots the runtime reads them from.

The converter's default is to normalize every BFLOAT16 tensor away
(converter._downcast_bfloat16): bf16 -> fp16 is exact inside fp16's range, and
until now no kernel could read a bf16 buffer. The runtime now reads a bf16
payload in exactly two places, and only for a CONSTANT initializer:

  * the weight (B) input of MatMul / Gemm   -- input index 1;
  * the table (data) input of Gather        -- input index 0.

The payload is the raw bf16 bits (little-endian uint16) in the same layout and
orientation an fp16 weight would have in that slot -- no scale tensor, bf16 is a
real float format, not a quantization grid. Activations stay in the fp16/fp32
domain the rest of the converter decides; nothing else may be bf16.

Two ways a weight becomes bf16 here, both opt-in:

  --keep-bf16  (keep=True)
      A weight that ARRIVED as bf16 is emitted with its original bits instead of
      the fp16 downcast. The rest of the pipeline is left byte-for-byte as it
      is today: `_downcast_bfloat16` still runs, every fusion and the FP16
      domain pass still see the fp16 (or fp32-widened) tensor. The original
      bits are stashed by name before the downcast (`stash_sources`), and at
      the end each qualifying weight is compared against its stash: only a
      weight whose current values are EXACTLY the downcast of a stashed bf16
      tensor -- as-is, or as its 2-D transpose -- gets the bf16 bits back.
      Anything a pass computed (a scale folded into a weight, the SwiGLU gate/up
      interleave, a concatenation) fails that comparison and keeps the fp16
      payload it has today. That makes the mode lossless by construction:
      either the artifact carries the source bits, or it carries exactly what
      the default conversion would have.

  -q bf16      (quantize=True)
      Every float weight in a qualifying slot is emitted as bf16, rounded
      round-to-nearest-even from its fp32/fp16 value (NaN kept a NaN). A weight
      whose source was bf16 is matched against the stash first, so it is
      bit-identical rather than re-rounded off an fp16 intermediate. The FP16
      domain pass is told to leave these payloads alone (see
      `collect_names`), exactly like a weight-only quantizer's, so an fp32
      source is rounded once, fp32 -> bf16, not fp32 -> fp16 -> bf16.

A deferred (streamed) weight under --keep-bf16 is handled by flipping its
deferred-table row: a plain pass-through row whose source file holds bf16 is
copied verbatim instead of being converted to fp16 chunk-wise. Rows that encode
something (a quantizing row, the SwiGLU interleave, a lane-major reorder, a
transposed payload) keep their existing behaviour.

Range scan (`TileScan`). Every MatMul route accumulates in fp32, and the GEMV
and column-parallel routes decode a bf16 B straight to fp32, but the tile routes
(the SIMT tile, tile-split-K and the Apple coop8 GEMM) stage B as half, which
clamps |w| > 65504. Each emitted bf16 weight -- kept, rounded or streamed -- is
scanned on its exact bits: with x = bits & 0x7FFF, x >= 0x4780 is past fp16's
range (inf and NaN included) and 0 < x < 0x3880 is below fp16's smallest normal.
A weight with any overflow element gets the node attribute `bf16_wide=1` on each
consuming MatMul/Gemm (`BF16_WIDE_ATTR`), which keeps the runtime off the tile
routes for it; nothing is written when the weight is in range, so such an
artifact is byte-identical to one built without the scan. Both counts go into
the numeric audit (WeightRow.tile_overflow / tile_subnormal); the subnormal
count is informational only.

Pure numpy on purpose (no ml_dtypes), like the rest of the bf16 handling.
"""
import os
from collections import defaultdict
from typing import Dict, Optional, Set, Tuple

import numpy as np
import onnx
from onnx import TensorProto, numpy_helper

_BF16 = TensorProto.BFLOAT16
_FLOAT = TensorProto.FLOAT
_FLOAT16 = TensorProto.FLOAT16
_FLOAT_KINDS = (_FLOAT, _FLOAT16)

# Ops whose reading of a weight is a pure re-indexing (see weight_stream's
# _VIEW_OPS). A bf16 initializer reaching MatMul through `Transpose` is the
# shape every exporter with constant folding off emits; the fusion passes fold
# that into transB=1 (same name) or the ConstantFolder materializes it (a new
# name, matched by value below), so the stash has to take such a weight too.
_VIEW_OPS = frozenset({"Transpose", "Reshape", "Squeeze", "Unsqueeze",
                       "Identity", "Flatten"})

# MatMul attributes that mean the node is a FOLD with its own weight ABI
# (Quantizer._match_weight_only / _validate_folded_weights): down_act reads
# three weights, swiglu an interleaved one. v1 of the bf16 contract covers the
# plain MatMul/Gemm B operand only.
_FOLD_ATTRS = ("down_act", "swiglu")

# The node attribute that tells the runtime a bf16 B holds an element past
# fp16's range (ops/Matmul.hpp, core::MatmulFeatures::bf16_wide).
BF16_WIDE_ATTR = "bf16_wide"

# bf16 bit thresholds on x = bits & 0x7FFF (sign dropped).
#   0x4780 = 65536, the first bf16 magnitude past fp16's max 65504 (0x477F is
#            65280, exact in fp16); every inf/NaN pattern (>= 0x7F80) is above.
#   0x3880 = 2**-14, fp16's smallest normal.
_TILE_OVERFLOW_BITS = 0x4780
_FP16_MIN_NORMAL_BITS = 0x3880
# uint16 elements per scan slice: 8.4M elements -> 16 MiB slice plus one
# same-size temporary, whatever the weight's size.
_SCAN_CHUNK = 1 << 23


# ---------------------------------------------------------------------------
# range scan: which bf16 weights a half-staging tile would clamp
# ---------------------------------------------------------------------------

class TileScan:
    """Counts over a bf16 payload's bits, fed in slices of any size.

    `overflow`  -- elements with |w| > 65504, inf or NaN (x >= 0x4780).
    `subnormal` -- nonzero elements below fp16's smallest normal
                   (0 < x < 0x3880); informational.
    A byte stream may be split anywhere (`add_bytes` carries an odd byte).
    """

    __slots__ = ("elements", "overflow", "subnormal", "_carry")

    def __init__(self):
        self.elements = 0
        self.overflow = 0
        self.subnormal = 0
        self._carry = b""

    def add_bits(self, bits) -> "TileScan":
        u = np.asarray(bits)
        if u.dtype != np.uint16:
            u = u.view(np.uint16) if u.dtype.itemsize == 2 else u.astype(
                np.uint16)
        u = u.reshape(-1)
        for i in range(0, u.size, _SCAN_CHUNK):
            x = u[i:i + _SCAN_CHUNK] & np.uint16(0x7FFF)
            self.elements += int(x.size)
            self.overflow += int(np.count_nonzero(
                x >= np.uint16(_TILE_OVERFLOW_BITS)))
            self.subnormal += int(np.count_nonzero(
                (x != 0) & (x < np.uint16(_FP16_MIN_NORMAL_BITS))))
        return self

    def add_bytes(self, block) -> "TileScan":
        if self._carry:
            block = self._carry + bytes(block)
            self._carry = b""
        n = len(block)
        if n & 1:
            self._carry = bytes(block[n - 1:])
            n -= 1
        if n:
            self.add_bits(np.frombuffer(block, dtype="<u2", count=n // 2))
        return self


def scan_bits(bits) -> TileScan:
    """TileScan over an in-memory payload (uint16 bits, any shape)."""
    return TileScan().add_bits(bits)


def scan_file(path: str, offset: int, length: int,
              chunk_bytes: int = _SCAN_CHUNK * 2) -> TileScan:
    """TileScan over `length` bytes of a bf16 payload at `offset` of `path`,
    read in bounded chunks (a streamed weight's source in its external file)."""
    scan = TileScan()
    with open(path, "rb") as f:
        f.seek(offset)
        remaining = int(length)
        while remaining:
            block = f.read(min(chunk_bytes, remaining))
            if not block:
                raise ValueError(
                    f"bf16 scan: {path} ends {remaining} byte(s) before the "
                    f"{length} at offset {offset}")
            scan.add_bytes(block)
            remaining -= len(block)
    return scan


def _flag_wide(slots) -> None:
    """bf16_wide=1 on each consuming MatMul/Gemm (Gather has no tile route)."""
    for node, _i in slots:
        if node.op_type in ("MatMul", "Gemm"):
            node.attributes[BF16_WIDE_ATTR] = [1]


def _record_scan(name, scan, slots, summary) -> None:
    try:
        from . import numeric_audit
    except ImportError:  # flat-module layout (cli run from the dir)
        import numeric_audit
    via = ""
    for node, _i in slots:
        via = getattr(node, "name", "") or ""
        if via:
            break
    numeric_audit.record_tile_scan(name, scan.elements, scan.overflow,
                                   scan.subnormal, via=via)
    if scan.overflow:
        _flag_wide(slots)
        summary["wide"].append(name)
    summary["tile_overflow"] += scan.overflow
    summary["tile_subnormal"] += scan.subnormal


# ---------------------------------------------------------------------------
# bit-level conversions
# ---------------------------------------------------------------------------

def fp32_to_bf16_bits(values) -> np.ndarray:
    """float32 (or anything numpy widens to it exactly) -> bf16 bits as uint16.

    Round to nearest, ties to even, on the 16 dropped mantissa bits -- the
    rounding PyTorch's `.to(torch.bfloat16)` uses. An overflow past bf16's
    largest finite value rounds to +-inf, as IEEE RNE does. NaN stays NaN: the
    rounding add could carry a NaN whose payload lives only in the low half
    into the exponent (0x7F800001 would become +inf), so a NaN keeps its sign
    and high payload bits and is forced quiet instead.
    """
    a = np.ascontiguousarray(values, dtype=np.float32)
    u = a.view(np.uint32)
    # u + 0x7FFF + lsb never overflows uint32 for a non-NaN input: the largest
    # non-NaN pattern is -inf (0xFF800000). NaNs are overwritten below.
    with np.errstate(over="ignore"):
        rounded = (u + (np.uint32(0x7FFF) + ((u >> np.uint32(16))
                                             & np.uint32(1)))) >> np.uint32(16)
    out = rounded.astype(np.uint16)
    nan = np.isnan(a)
    if nan.any():
        out[nan] = ((u[nan] >> np.uint32(16)) | np.uint32(0x0040)).astype(
            np.uint16)
    return out


def bf16_bits_to_fp32(bits) -> np.ndarray:
    """bf16 bits (uint16) -> float32. Exact: bf16 is fp32's high half."""
    u16 = np.ascontiguousarray(bits, dtype=np.uint16)
    return (u16.astype(np.uint32) << np.uint32(16)).view(np.float32)


def _tensor_as_fp32(init) -> np.ndarray:
    """A FLOAT/FLOAT16 initializer's values widened (exactly) to float32."""
    return numpy_helper.to_array(init).astype(np.float32, copy=False)


def _make_bf16(name: str, dims, bits: np.ndarray):
    t = onnx.helper.make_tensor(name, _BF16, list(dims),
                                np.ascontiguousarray(bits, dtype=np.uint16)
                                .tobytes(), raw=True)
    return t


# ---------------------------------------------------------------------------
# the stash: original bf16 bits, captured before _downcast_bfloat16
# ---------------------------------------------------------------------------

def stash_sources(model) -> Dict[str, Tuple[Tuple[int, ...], bytes]]:
    """name -> (dims, raw bf16 bytes) for every bf16 initializer that could end
    up in a qualifying slot.

    Called on the proto BEFORE `_downcast_bfloat16` rewrites the bytes. The
    filter is deliberately loose -- every consumer is a MatMul/Gemm B, a Gather
    table, or a pure view -- because which slot a weight finally lands in is
    only known after fusion; `apply` decides on the finished DAG. A tensor that
    is still external (a streamed weight: no raw_data in the proto) is not
    stashed; its bytes are still in the source file and its deferred row is
    what `apply` flips.
    """
    uses = defaultdict(list)
    for node in model.graph.node:
        for i, name in enumerate(node.input):
            if name:
                uses[name].append((node.op_type, i))
    out = {}
    for init in model.graph.initializer:
        if init.data_type != _BF16 or not init.raw_data:
            continue
        consumers = uses.get(init.name)
        if not consumers:
            continue
        ok = True
        for op, i in consumers:
            if (op in ("MatMul", "Gemm") and i == 1) or (op == "Gather"
                                                         and i == 0):
                continue
            if op in _VIEW_OPS and i == 0:
                continue
            ok = False
            break
        if ok:
            out[init.name] = (tuple(int(d) for d in init.dims),
                              bytes(init.raw_data))
    return out


# ---------------------------------------------------------------------------
# which DAG initializers sit in a qualifying slot
# ---------------------------------------------------------------------------

def _slot_ok(node, index: int, name: str) -> bool:
    op = node.op_type
    if op in ("MatMul", "Gemm"):
        if index != 1:
            return False
        attrs = node.attributes or {}
        if any(attrs.get(k) for k in _FOLD_ATTRS):
            return False
        return True
    if op == "Gather":
        return index == 0
    return False


def qualifying_initializers(dag_model, *, moe_names=()) -> Dict[str, list]:
    """name -> [(node, input_index)] for each initializer whose EVERY use is a
    qualifying slot (MatMul/Gemm input 1, Gather input 0).

    A tensor shared with any other reader -- a bias slot, a Mul, a second
    MatMul reading it as A -- is not returned: the bf16 contract covers only
    those two slots, and one initializer has one dtype. The weight also has to
    be a plain 2-D matrix (the layout both slots already read for fp16) and a
    FLOAT/FLOAT16 tensor (anything else was already quantized or is not a
    float). MoE paging slots are excluded: the runtime pages them from the
    checkpoint with their own source-dtype bookkeeping.
    """
    uses = defaultdict(list)
    for node in dag_model.nodes.values():
        for i, inp in enumerate(node.inputs):
            name = inp["name"] if isinstance(inp, dict) else inp
            if name:
                uses[name].append((node, i))
    moe = set(moe_names or ())
    out = {}
    for name, init in dag_model.initializers.items():
        if name in moe:
            continue
        if int(init.data_type) not in _FLOAT_KINDS:
            continue
        if len(init.dims) != 2:
            continue
        slots = uses.get(name)
        if not slots:
            continue
        if all(_slot_ok(node, i, name) for node, i in slots):
            out[name] = slots
    return out


def _moe_names(dag_model):
    return set(getattr(dag_model, "moe_keep_alive", ()) or ())


def collect_names(dag_model, exclude=None) -> Set[str]:
    """Initializers `-q bf16` will replace -- the FP16 domain's skip set.

    Mirrors `apply(quantize=True)`: qualifying, in memory, not excluded. The
    domain pass still flips the declaration (the op's activations run fp16);
    only the bytes are left for the bf16 pass to round once from the source.
    """
    deferred = getattr(dag_model, "deferred_weights", {}) or {}
    names = set()
    for name in qualifying_initializers(dag_model,
                                        moe_names=_moe_names(dag_model)):
        if name in deferred:
            continue
        if exclude and name in exclude:
            continue
        names.add(name)
    return names


# ---------------------------------------------------------------------------
# matching a DAG weight back to its bf16 source
# ---------------------------------------------------------------------------

class _SourceIndex:
    """Find the stashed bf16 tensor a DAG weight is an unmodified copy of."""

    def __init__(self, sources):
        self.sources = sources or {}
        self.by_numel = defaultdict(list)
        for name, (dims, _raw) in self.sources.items():
            n = 1
            for d in dims:
                n *= d
            self.by_numel[n].append(name)

    def _bits(self, name) -> np.ndarray:
        dims, raw = self.sources[name]
        return np.frombuffer(raw, dtype=np.uint16).reshape(dims)

    @staticmethod
    def _same(cur: np.ndarray, src_bits: np.ndarray) -> bool:
        """Is `cur` exactly what the default pipeline makes of `src_bits`?

        `cur` is either fp16 (the downcast / the FP16 domain's rounding of the
        fp32 widening -- both a single fp32 -> fp16 round of the same exact
        value) or fp32 (the widening for an fp32 consumer, or -q fp32). Both
        are compared in `cur`'s own dtype, NaN equal to NaN.
        """
        ref = bf16_bits_to_fp32(src_bits)
        if cur.dtype == np.float16:
            with np.errstate(over="ignore"):
                ref = ref.astype(np.float16)
        elif cur.dtype != np.float32:
            return False
        return bool(np.array_equal(cur, ref, equal_nan=True))

    def match(self, name: str, init) -> Optional[np.ndarray]:
        """The bf16 bits to emit for `init` (already in its layout), or None."""
        dims = tuple(int(d) for d in init.dims)
        n = 1
        for d in dims:
            n *= d
        cands = self.by_numel.get(n)
        if not cands:
            return None
        cur = numpy_helper.to_array(init)
        # The weight's own name first: the common case (direct, or a
        # Transpose folded into transB=1, which keeps the name).
        order = ([name] if name in self.sources else []) + [
            c for c in cands if c != name]
        for src in order:
            sdims = self.sources[src][0]
            if sdims == dims:
                bits = self._bits(src)
                if self._same(cur, bits):
                    return bits
            elif len(dims) == 2 and sdims == (dims[1], dims[0]):
                # A materialized 2-D transpose (ConstantFolder on
                # Transpose(W), which names the result after the node).
                # Cheap prefilter on one row before the full compare: an LLM
                # has dozens of same-shape weights.
                bits = self._bits(src)
                if not self._same(cur[:1], bits[:, :1].T):
                    continue
                t = np.ascontiguousarray(bits.T)
                if self._same(cur, t):
                    return t
        return None


# ---------------------------------------------------------------------------
# the pass
# ---------------------------------------------------------------------------

def _retag(slots, dtype=_BF16):
    """The consuming nodes' input declaration follows the payload's dtype."""
    for node, i in slots:
        inp = node.inputs[i]
        if isinstance(inp, dict):
            inp["dtype"] = dtype


def _flip_deferred(entry: dict, init) -> bool:
    """Keep a streamed bf16 source as bf16 instead of converting chunk-wise.

    Only a plain pass-through row qualifies: one that copies the file's bytes
    as they are. Any row that encodes (quant), reorders (lane_major,
    interleave) or transposes the payload keeps today's behaviour.
    """
    if entry.get("src_dtype") != _BF16:
        return False
    for key in ("quant", "lane_major", "interleave", "transpose_payload",
                "src_dims"):
        if entry.get(key) is not None:
            return False
    if tuple(int(d) for d in init.dims) != tuple(entry.get("dims", ())):
        return False
    entry["data_type"] = _BF16
    entry["keep_bf16"] = True
    init.data_type = _BF16
    return True


def apply(dag_model, *, keep: bool = False, quantize: bool = False,
          sources=None, exclude=None, only=None, verbose: bool = True) -> dict:
    """Emit qualifying weights as bf16. Returns a summary dict.

    `keep`      -- restore stashed bf16 sources (lossless; see module doc).
    `quantize`  -- -q bf16: round every qualifying float weight to bf16.
    `sources`   -- the stash from `stash_sources` (defaults to
                   dag_model.bf16_sources).
    `exclude`   -- --exclude-* names (honoured by `quantize` only; a restored
                   source is the weight the user already had).
    `only`      -- restrict `quantize` to these names (the FP16 domain's skip
                   set, so every payload the domain pass left wide is
                   rewritten and nothing else is).
    """
    summary = {"kept": [], "rounded": [], "deferred": [], "fallback": [],
               "wide": [], "tile_overflow": 0, "tile_subnormal": 0}
    if not (keep or quantize):
        return summary
    if sources is None:
        sources = getattr(dag_model, "bf16_sources", {}) or {}
    index = _SourceIndex(sources)
    deferred = getattr(dag_model, "deferred_weights", {}) or {}
    quals = qualifying_initializers(dag_model,
                                    moe_names=_moe_names(dag_model))

    for name, slots in sorted(quals.items()):
        init = dag_model.initializers[name]
        if name in deferred:
            if keep or quantize:
                entry = deferred[name]
                flipped = _flip_deferred(entry, init)
                if (not flipped and quantize
                        and not (exclude and name in exclude)
                        and entry.get("src_dtype", entry.get("data_type"))
                        == _FLOAT16
                        and entry.get("interleave") is None):
                    # -q bf16 on a streamed fp16 source: rounded once at
                    # write time, as `quantize` does to an in-memory one.
                    flipped = _ws().mark_bf16_payload(deferred, name, init)
                if flipped:
                    _retag(slots)
                    summary["deferred"].append(name)
                    # The payload is written after the graph, so the scan
                    # reads the source bytes here, while the attribute can
                    # still be emitted; write_payload re-checks them as they
                    # pass (weight_stream.py).
                    scan = _scan_deferred(entry)
                    entry["bf16_tile_overflow"] = scan.overflow
                    _record_scan(name, scan, slots, summary)
            continue
        bits = index.match(name, init) if sources else None
        if bits is not None:
            dag_model.initializers[name] = _make_bf16(name, init.dims, bits)
            _retag(slots)
            summary["kept"].append(name)
            _record_scan(name, scan_bits(bits), slots, summary)
            continue
        if not quantize:
            if name in sources:
                summary["fallback"].append(name)
            continue
        if exclude and name in exclude:
            continue
        if only is not None and name not in only:
            continue
        src = _tensor_as_fp32(init)
        bits = fp32_to_bf16_bits(src)
        dag_model.initializers[name] = _make_bf16(name, init.dims, bits)
        _retag(slots)
        summary["rounded"].append(name)
        try:
            from . import numeric_audit
        except ImportError:  # flat-module layout (cli run from the dir)
            import numeric_audit
        numeric_audit.record_roundtrip(name, "bf16", src,
                                       bf16_bits_to_fp32(bits))
        _record_scan(name, scan_bits(bits), slots, summary)

    # Freed: the stash is the second copy of every in-memory bf16 weight.
    if hasattr(dag_model, "bf16_sources"):
        dag_model.bf16_sources = {}

    if verbose:
        print(f"[bf16] {len(summary['kept'])} weight(s) keep their bf16 source "
              f"bits, {len(summary['rounded'])} rounded to bf16 (RNE), "
              f"{len(summary['deferred'])} streamed as bf16"
              + (f"; {len(summary['fallback'])} bf16 source(s) were rewritten "
                 f"by a pass and stay fp16/fp32: "
                 + ", ".join(summary["fallback"][:5])
                 if summary["fallback"] else ""))
        n_bf16 = (len(summary["kept"]) + len(summary["rounded"])
                  + len(summary["deferred"]))
        if n_bf16:
            print(tile_scan_line(summary))
    return summary


def tile_scan_line(summary: dict) -> str:
    """The one-line summary of the range scan over every emitted bf16 weight."""
    wide = summary["wide"]
    line = (f"[bf16] range scan: {len(wide)} weight(s) hold |w| > 65504 / "
            f"inf / NaN ({summary['tile_overflow']} element(s))")
    if wide:
        line += (f" -> {BF16_WIDE_ATTR}=1 on their MatMul/Gemm, kept off the "
                 f"half-staging tile routes: " + ", ".join(wide[:5])
                 + (" ..." if len(wide) > 5 else ""))
    line += (f"; {summary['tile_subnormal']} element(s) below fp16's "
             f"smallest normal (informational)")
    return line


def _ws():
    try:
        from . import weight_stream
    except ImportError:  # flat-module layout (cli run from the dir)
        import weight_stream
    return weight_stream


def _scan_deferred(entry: dict) -> TileScan:
    path = entry["location"]
    if not os.path.isabs(path):
        path = os.path.join(entry.get("base_dir") or "", path)
    if entry.get("src_dtype") != _FLOAT16:
        return scan_file(path, int(entry["offset"]), int(entry["length"]))
    # An fp16 source becomes bf16 only at write time: scan what will be
    # written (the RNE rounding can carry 65504 up to 65536), not the file.
    scan = TileScan()
    convert = _ws()._fp16_block_to_bf16
    with open(path, "rb") as f:
        f.seek(int(entry["offset"]))
        remaining = int(entry["length"])
        while remaining:
            block = f.read(min(_SCAN_CHUNK * 2, remaining))
            if not block:
                raise ValueError(
                    f"bf16 scan: {path} ends {remaining} byte(s) before the "
                    f"{entry['length']} at offset {entry['offset']}")
            scan.add_bytes(convert(block))
            remaining -= len(block)
    return scan
