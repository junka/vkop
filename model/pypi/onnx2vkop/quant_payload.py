"""Finalize the deferred-table entries for a byte-format or int4 quantization.

The scale of an int8 / fp8 weight is its per-column absmax and its payload
overwrites the values, so the conversion neither needs nor wants the weight in
RAM. weight_stream defers such a weight (see exports_weight_values) and records
what the payload will be; this module computes the one thing the writer cannot
derive on its own -- the scale -- and puts it into the graph as an ordinary
initializer, appended to each consumer the way every other weight-only format
appends its scale.

int4 is deferrable for a different reason. Its scale is grouped along K -- one
entry per (`group_size` values of K, output column) -- which sounds like it
needs the whole matrix. It does not: each group's scale is an absmax over its
OWN `group_size` rows and nothing else, so a band-at-a-time sweep can accumulate
`amax[n_groups, N]` exactly (`_grouped_amax`). The band boundary and the group
boundary need not coincide; the reduceat slice below is computed from
`offset % group_size` rather than assumed. That is what makes `-q int4`
convertible on a machine that cannot hold the model. nf4 streams the same way
(its codebook lookup is per-element, its scale the group absmax), and so does
nvfp4: its per-tensor factor is the max of the 16-value block absmax table the
same sweep builds (`nvfp4_tables`), and it stores two scale initializers
instead of one.

It runs on the **onnx proto**, right after `weight_stream.prepare`, because no
DAGBasedModel exists at that point. Adding the scale here (rather than later)
means every pass downstream sees the same three-input `MatMul(A, W, s)` shape it
sees for `-q int8` on an in-memory model.

The values are read once, at the end: `write_payload` streams the source again
band by band, so nothing here ever holds more than a band.
"""
from __future__ import annotations

import os

import numpy as np
from onnx import TensorProto, helper, numpy_helper

# Rows per read pass. Bounded so a 151936-row weight never materializes more
# than a few hundred MiB at a time.
ROW_BLOCK = 2048

# ...and values per read pass, because a row count alone does not bound a band:
# 2048 rows of GLM-Edge's 59264-column lm_head is the WHOLE 121M-value matrix,
# i.e. 485 MiB of fp32 plus as much again for np.abs in the scale sweep -- the
# 1.7 GiB that set a streamed -q nf4 conversion's peak. The default band is the
# smaller of the two, kept even (the transposed 4-bit writer packs column
# pairs and needs every band to start on an even column).
BAND_VALUES = 1 << 22


def _np_dtype_of(dtype):
    """The numpy dtype that decodes this source file's element type.

    BFLOAT16 has no numpy dtype, so it is read as its 2-byte storage and
    widened in `WeightReader.bands` -- the same `u16 << 16` the converter's
    `_bf16_bytes_to_fp16` and `write_payload` use, which is exact (bf16 is fp32
    with the low mantissa half dropped, so shifting the storage into the high
    half IS the value).
    """
    if dtype == TensorProto.FLOAT16:
        return np.dtype(np.float16)
    if dtype == TensorProto.FLOAT:
        return np.dtype(np.float32)
    if dtype == TensorProto.BFLOAT16:
        return np.dtype(np.uint16)
    raise ValueError(f"deferred weight dtype {dtype} is not a float")


def _bf16_band_to_f32(band: np.ndarray) -> np.ndarray:
    """Widen a uint16 view of bf16 storage to float32 (exact, bit-shift only)."""
    return (band.astype(np.uint32) << 16).view(np.float32)


class WeightReader:
    """Streams a 2-D deferred weight, one band of rows at a time."""

    def __init__(self, entry, name, path_for=None):
        self.entry = entry
        self.name = name
        # Resolves the source path. weight_stream passes its source_read_path so
        # an in-place reconversion (source and output are one file) reads the
        # guard's copy instead of the truncated destination; a caller that does
        # not care gets the plain location resolution.
        self._path_for = path_for
        # `dims` is the LOGICAL shape the side table records, which is what the
        # kernel indexes: [N, K] for a transB=1 weight, and [K, N] once
        # KnWeightLayout has flipped it. `src_dims` is how the SOURCE FILE is
        # laid out, and it is what this reader seeks by; the two differ exactly
        # when the payload is transposed on the way out. Defaulting to `dims`
        # keeps every entry that was never flipped reading what it always did.
        dims = list(entry.get("src_dims") or entry["dims"])
        if len(dims) != 2:
            raise ValueError(
                f"'{name}': a byte-format weight must be 2-D, got {dims}")
        self.k, self.n = int(dims[0]), int(dims[1])
        src = entry.get("src_dtype", entry["data_type"])
        self.is_bf16 = src == TensorProto.BFLOAT16
        self.dtype = _np_dtype_of(src)

    def bands(self, block=None):
        """Yield (row_offset, float32 [rows, N]) for the whole weight.

        The source is the file the entry points at; `length` is its extent in
        source bytes, which is `numel * itemsize(src_dtype)`. A short read is
        a hard error for the same reason it is in write_payload: the entry
        already promised those bytes.

        `block` is rows per band; the default is ROW_BLOCK rows or
        BAND_VALUES values, whichever is fewer (and even).
        """
        if block is None:
            block = default_block(self.n)
        path = (self._path_for(self.entry) if self._path_for is not None
                else self.entry["location"])
        if not os.path.isabs(path):
            path = os.path.join(self.entry["base_dir"], path)
        row_bytes = self.n * self.dtype.itemsize
        with open(path, "rb") as f:
            f.seek(int(self.entry["offset"]))
            remaining = int(self.entry["length"])
            offset = 0
            while remaining:
                want = min(row_bytes * block, remaining)
                raw = f.read(want)
                if len(raw) != want:
                    raise ValueError(
                        f"'{self.name}': source ended at row {offset} of "
                        f"{self.k} (wanted {want} bytes, got {len(raw)})")
                rows = want // row_bytes
                if rows:
                    band = np.frombuffer(raw[:rows * row_bytes],
                                         dtype=self.dtype).reshape(rows, self.n)
                    if self.is_bf16:
                        yield offset, _bf16_band_to_f32(band)
                    else:
                        yield offset, band.astype(np.float32)
                offset += rows
                remaining -= rows * row_bytes


def default_block(n):
    """Rows per band for an n-column source: see BAND_VALUES."""
    rows = min(ROW_BLOCK, BAND_VALUES // max(1, int(n)))
    return max(2, rows & ~1)


def _amax(reader, transposed: bool):
    """Per-output-column absmax, along whichever STORED axis holds the columns.

    A scale always has one entry per output column (the kernel indexes it by
    output channel / N), but which axis of the stored bytes that is depends on
    the consumer chain:

        direct    MatMul(X, W[K, N])            columns are axis 1 -> reduce axis 0
        Transpose MatMul(X, Transpose(W[N, K])) columns are axis 0 -> reduce axis 1

    `transposed` comes from weight_stream.is_transposed_weight, which reads it
    off the graph while the Transpose is still there. Reducing the wrong axis
    does not fail loudly -- the payload keeps its size and the artifact loads --
    it just decodes every weight with a scale from the wrong direction.

    Note the two cases differ in more than the axis: reducing axis 0 is
    position-independent (every row of a band contributes to the same N
    columns), while reducing axis 1 gives one value PER BAND ROW, which has to
    land at that row's offset. So the transposed branch places its result
    rather than broadcasting it.
    """
    if transposed:
        # Stored [N, K] (reader.k == N == dims[0]): one scale per stored row,
        # so band row `offset + i` is output column `offset + i`.
        amax = np.zeros(reader.k, dtype=np.float32)
        for offset, band in reader.bands():
            rows = band.shape[0]
            np.maximum(amax[offset:offset + rows],
                       np.max(np.abs(band), axis=1),
                       out=amax[offset:offset + rows])
        return amax
    amax = np.zeros(reader.n, dtype=np.float32)
    for _, band in reader.bands():
        np.maximum(amax, np.max(np.abs(band), axis=0), out=amax)
    return amax


def scale_for(fmt, reader, transposed: bool = False, group: int = 0):
    """The fp32 scale a streamed payload is decoded with.

    Two shapes, because the formats ask for two different things:

      * int8 / fp8 -- one entry per output column, `[N]`;
      * int4       -- one entry per (`group` values of K, output column),
                      `[n_groups, N]`, row-major, the layout the nibble
                      kernels read with `scale4(g * uPC.N + ja)`.

    The 4-bit branch is `Quantizer._quantize_w3`'s arithmetic on a table this
    module accumulated band by band, not on a materialized weight: same absmax
    per group, same `/7`, same fp32 rounding, so the payload and the table are
    bit-identical to the in-memory quantizer's.
    """
    from .optimizer import Quantizer

    if fmt == "int4":
        amax = grouped_amax(reader, group, transposed)
        return np.where(amax == 0, 1.0, amax / 7.0).astype(np.float32)
    if fmt == "nf4":
        # nf4 decodes as codebook[q] * scale with the codebook spanning
        # [-1, 1], so the scale IS the group absmax (no /7) -- `_quantize_w3`'s
        # unsigned branch, same zero guard.
        amax = grouped_amax(reader, group, transposed)
        return np.where(amax == 0, 1.0, amax).astype(np.float32)
    if fmt == "nvfp4":
        # The fp32 factor per block the kernel applies (e4m3 block scale *
        # global), not what the graph stores -- finalize stores the two levels
        # (nvfp4_tables). This is the writer's fallback when no graph is at
        # hand, and it is the same number either way.
        return nvfp4_tables(reader, transposed, reader.name)[2]
    amax = _amax(reader, transposed)
    if fmt == "int8":
        return np.where(amax == 0, 1.0, amax / 127.0).astype(np.float32)
    max_finite = Quantizer._FP8_FORMATS[fmt][0]
    return np.where(amax == 0, 1.0, amax / max_finite).astype(np.float32)


def grouped_amax(reader, group: int, transposed: bool = False):
    """Per-(K group, output column) absmax, accumulated one band at a time.

    This is the function that makes int4 streamable, and the reason it is not
    obvious is worth spelling out. A 4-bit weight's scale table is `[K/g, N]`,
    which looks like "the scale spans K, so the whole matrix must be in hand".
    But each entry is an absmax over its OWN `group` values of K and nothing
    else -- a group's scale never sees another group's rows. So a band of rows
    can reduce into the groups it overlaps and later bands add to the same
    accumulator, exactly as `_amax` does for the per-column formats.

    The two stored layouts put the group axis in different places, and neither
    is allowed to assume the band size is a multiple of `group`:

      * stored [K, N] (a direct MatMul weight): groups are ROWS. A band of
        ROW_BLOCK rows usually contains whole groups, but a smaller band or a
        larger `group` straddles boundaries -- handled by `np.maximum.reduceat`
        on the group boundaries that fall INSIDE the band, so the first and
        last segments are partial and still exact.
      * stored [N, K] (a Transpose-fed weight, KnWeightLayout's [K, N] source):
        groups are COLUMNS of the stored row, i.e. one stored row is one whole
        output column's K axis. A band is then complete per group by
        construction, and `amax[g, column]` is just the band's row reshaped to
        [`n_groups`, `group`] and reduced.

    Returns a `[n_groups, n_columns]` float32 table, which is what the writer
    emits and what the kernel indexes.
    """
    if group <= 0 or group % 2:
        raise ValueError(
            f"'{reader.name}': a 4-bit group must be a positive even number of "
            f"K values, got group={group}")
    if transposed:
        # Stored [N, K]: reader.n is K, and every stored row is one output
        # column whose whole K axis sits inside the band it arrives in.
        if reader.n % group:
            raise ValueError(
                f"'{reader.name}': stored K={reader.n} is not a multiple of "
                f"group={group}, so a group would straddle an output column")
        n_groups = reader.n // group
        amax = np.zeros((n_groups, reader.k), dtype=np.float32)
        for offset, band in reader.bands():
            rows = band.shape[0]
            # [rows, K] -> [rows, n_groups, group] -> [rows, n_groups]
            per_group = np.abs(band).reshape(rows, n_groups, group).max(axis=2)
            np.maximum(amax[:, offset:offset + rows], per_group.T,
                       out=amax[:, offset:offset + rows])
        return amax

    if reader.k % group:
        raise ValueError(
            f"'{reader.name}': K={reader.k} is not a multiple of group={group}, "
            f"so the last group would be partial")
    n_groups = reader.k // group
    amax = np.zeros((n_groups, reader.n), dtype=np.float32)
    for offset, band in reader.bands():
        rows = band.shape[0]
        first = offset // group
        last = (offset + rows - 1) // group
        # Segment starts INSIDE the band: 0 and every group boundary after it.
        # The first and last segments are partial whenever a band does not
        # begin/end on a group boundary, which is exactly the case a "ROW_BLOCK
        # is a multiple of group_size" assumption would get wrong.
        bounds = [0] + [g * group - offset for g in range(first + 1, last + 1)]
        seg = np.maximum.reduceat(np.abs(band), bounds, axis=0)
        np.maximum(amax[first:last + 1], seg, out=amax[first:last + 1])
    return amax


def nvfp4_tables(reader, transposed: bool = False, name="<weight>"):
    """NVFP4's two scale levels from one band-at-a-time sweep.

    The per-tensor factor looks like it needs the whole matrix -- it is
    amax_tensor / (448 * 6) and every block scale is divided by it -- but
    amax_tensor is just the max of the per-block absmax table, and that table is
    `grouped_amax` with a 16-value group: each block's absmax is over its own 16
    values of K. So the same single sweep int4 does yields both levels, and the
    table is reduced to them by the in-memory pass's own arithmetic
    (`Quantizer._nvfp4_scales`), which is what keeps the bytes identical.

    Returns (scale_codes uint8 [K/16, N], global_scale np.float32,
    dequant_scale float32 [K/16, N], dead).
    """
    from .optimizer import Quantizer

    amax = grouped_amax(reader, Quantizer._NVFP4_BLOCK, transposed)
    return Quantizer._nvfp4_scales(amax, name)


# The grouped 4-bit formats: one scale per (group of K, output column), two
# codes per byte. int4 is a signed nibble (scale = absmax / 7), nf4 an index
# into Quantizer._NF4_CODEBOOK (scale = absmax), nvfp4 an E2M1 code against a
# 16-value block's e4m3 scale times one fp32 factor per tensor (the writer and
# the encoder see the product, an fp32 [K/16, N] table like the other two).
# All three stream the same way; only the per-element encode differs.
FOUR_BIT_FORMATS = ("int4", "nf4", "nvfp4")


def _encode_4bit(fmt):
    """(normalized fp32 values) -> int8 codes, the in-memory quantizer's rule."""
    if fmt == "int4":
        return lambda x: np.clip(np.round(x), -8, 7).astype(np.int8)
    from .optimizer import Quantizer
    return Quantizer._nf4_encode


def code_nibble(fmt, reader, scale, group: int, transposed: bool = False):
    """A function (row_offset, band) -> int8 codes, band-shaped.

    int4 codes are in [-8, 7]; nf4 codes are codebook indices in [0, 15];
    nvfp4 codes are E2M1 nibbles (sign in bit 3), already uint8.

    `scale` is the table `scale_for` produced: `[n_groups, N]`. The grouping is
    applied by indexing it with the K index's group, so a band that starts
    mid-group still divides every row by the scale of the group that row
    belongs to.

    The division is the in-memory quantizer's, element for element:
    `Quantizer._quantize_w3` divides the [n_groups, group, N] operand by a
    [n_groups, 1, N] table, which for a row inside group g is exactly
    `band[row] / scale[g]`. Same operands, same fp32 rounding, and the same
    encoder after it (nf4 calls `Quantizer._nf4_encode` itself), so the codes
    are bit-identical -- the packed bytes only ever differ in how they are laid
    down, never in what they say.
    """
    if fmt not in FOUR_BIT_FORMATS:
        raise ValueError(f"code_nibble is the 4-bit encoder; got {fmt!r}")
    scale = np.asarray(scale, dtype=np.float32)
    if fmt == "nvfp4":
        # The in-memory pass's encode (Quantizer._nvfp4_encode): divide by the
        # block's dequant factor (a zero factor divides as 1), clip to +-6,
        # round to E2M1. Same elementwise fp32 division, same bytes.
        from .optimizer import Quantizer

        if transposed:
            def codes(offset, band):
                rows = band.shape[0]
                s = np.repeat(scale[:, offset:offset + rows].T, group, axis=1)
                return Quantizer._nvfp4_encode(band, s)
            return codes

        def codes(offset, band):
            rows = band.shape[0]
            idx = np.arange(offset, offset + rows) // group
            return Quantizer._nvfp4_encode(band, scale[idx])
        return codes
    encode = _encode_4bit(fmt)

    if transposed:
        def codes(offset, band):
            rows = band.shape[0]
            # [rows, n_groups] -> [rows, K], one scale per K value of the row.
            s = np.repeat(scale[:, offset:offset + rows].T, group, axis=1)
            return encode(band / s)
        return codes

    def codes(offset, band):
        rows = band.shape[0]
        idx = np.arange(offset, offset + rows) // group
        return encode(band / scale[idx])
    return codes


def dequantize_4bit(fmt, codes, scale):
    """What the kernel reads back for `codes` under a broadcastable `scale`.

    `Quantizer._dequantize_w3`'s arithmetic (int4: code * scale; nf4:
    codebook[code] * scale), on whatever layout the caller holds.
    """
    from .optimizer import Quantizer
    if fmt == "nvfp4":
        return Quantizer._e2m1_decode(codes) * np.asarray(scale, np.float32)
    return Quantizer._dequantize_w3(codes, scale, fmt)


def audit_4bit(fmt, reader, scale, group: int, transposed: bool = False):
    """The numeric-audit RoundTrip of a streamed 4-bit weight, one band at a time.

    The in-memory pass reports every weight it encodes (`_report_qerror`: the
    "Quantized '<name>'" lines and the audit row); a streamed weight is encoded
    by the writer, after the audit has been printed, so without this it would
    silently drop out of both. This sweeps the source once more -- one band in
    RAM -- encodes it with the very codes the writer will emit, decodes it the
    way the kernel does and folds the pair into a RoundTrip.

    The sweep is bounded by AUDIT_VALUES values per band (~4 MiB of fp32,
    ~40 MiB of float64 temporaries in RoundTrip.add) rather than by
    numeric_audit's own 8M-value chunk: at that size the float64 working set
    of the audit, not the weight, was the conversion's peak (+0.8 GiB on
    GLM-Edge). The cost is the summation order -- the in-memory pass sums its
    8M-value slices, this sums smaller ones -- so the float64 totals agree to
    rounding (relative ~1e-16), far below the printed digits; the maxima and
    the counts are order-independent and identical.

    Returns (RoundTrip, dead_slices) -- dead_slices is `_report_qerror`'s
    count of zero entries in the scale table.
    """
    from . import numeric_audit

    scale = np.asarray(scale, dtype=np.float32)
    stats = numeric_audit.RoundTrip(fmt)
    codes = code_nibble(fmt, reader, scale, group, transposed)
    block = max(2, (AUDIT_VALUES // max(1, reader.n)) & ~1)
    for offset, band in reader.bands(block=block):
        rows = band.shape[0]
        if transposed:
            # [rows, n_groups] -> [rows, K], one scale per K value of the row.
            s = np.repeat(scale[:, offset:offset + rows].T, group, axis=1)
        else:
            s = scale[np.arange(offset, offset + rows) // group]
        stats.add(band, dequantize_4bit(fmt, codes(offset, band), s))
    return stats, int(np.count_nonzero(scale == 0))


# Values per band of the audit sweep (see audit_4bit).
AUDIT_VALUES = 1 << 20


class InterleavedReader:
    """The SwiGLU gate/up fold's [K, 2N] weight, read from its two sources.

    `FusionOptimizer.fold_swiglu_gate_up` replaces Wg and Wu ([K, N] each) with
    one [K, 2N] weight whose columns interleave four at a time (g0..g3 u0..u3
    g4 ..). A quantizing deferred entry for that weight keeps both sources
    (`entry["interleave"]`), and this reader yields bands of the interleaved
    matrix -- the same float32 values, in the same places, the in-memory fold
    builds with np.concatenate -- so the encoder, the writer and the audit see
    the fused weight without either source ever being whole in RAM.
    """

    def __init__(self, entry, name, path_for=None):
        spec = entry["interleave"]
        self.name = name
        self.g = WeightReader(spec["g"], name + " (gate)", path_for=path_for)
        self.u = WeightReader(spec["u"], name + " (up)", path_for=path_for)
        if (self.g.k, self.g.n) != (self.u.k, self.u.n) or self.g.n % 4:
            raise ValueError(
                f"'{name}': gate {self.g.k}x{self.g.n} and up "
                f"{self.u.k}x{self.u.n} cannot interleave four columns at a time")
        self.k, self.n = self.g.k, 2 * self.g.n

    def bands(self, block=None):
        half = self.g.n
        if block is None:
            block = default_block(self.n)
        for (og, bg), (ou, bu) in zip(self.g.bands(block), self.u.bands(block)):
            if og != ou or bg.shape != bu.shape:
                raise ValueError(
                    f"'{self.name}': gate and up bands disagree at row {og}/{ou}")
            rows = bg.shape[0]
            yield og, np.concatenate(
                [bg.reshape(rows, half // 4, 4), bu.reshape(rows, half // 4, 4)],
                axis=2).reshape(rows, 2 * half)


def reader_for(entry, name, path_for=None):
    """The band reader for a deferred entry: plain, or the SwiGLU interleave."""
    if entry.get("interleave") is not None:
        return InterleavedReader(entry, name, path_for=path_for)
    return WeightReader(entry, name, path_for=path_for)


def code_byte(fmt, reader, scale, transposed: bool = False):
    """A function (row_offset, band) -> one uint8 per value, the byte the kernel
    decodes.

    int8's code is a signed value; the payload stores it as two's complement,
    so `& 0xFF` is spelled out rather than left to numpy's int8 storage. fp8's
    `_fp8_encode` already returns the finished byte (sign in bit 7).

    `row_offset` is the band's first stored row, and the transposed case needs
    it: there the rows ARE the output columns, so the scale for a band is the
    slice that band covers, not the whole table. Passing the whole table would
    broadcast against the wrong rows and decode every band but the first with
    the wrong scales.
    """
    from .optimizer import Quantizer

    scale = np.asarray(scale, dtype=np.float32)

    def slice_for(row_offset, band):
        rows = band.shape[0]
        if transposed:
            # One scale per stored row -> per band row, at that row's offset.
            return scale[row_offset:row_offset + rows].reshape(-1, 1)
        # One scale per column -> same for every band.
        return scale.reshape(1, -1)

    if fmt == "int8":
        def codes(row_offset, band):
            q = np.round(band / slice_for(row_offset, band)).astype(np.int8)
            return q.astype(np.uint8)

        return codes

    max_finite, exp_bits, mant_bits = Quantizer._FP8_FORMATS[fmt]

    def codes(row_offset, band):
        clipped = np.clip(band / slice_for(row_offset, band),
                          -max_finite, max_finite)
        return Quantizer._fp8_encode(clipped, exp_bits, mant_bits)

    return codes


def _has_bias(node) -> bool:
    """Does this weight consumer already use the slot a scale would take?

    Gemm's third input is C, Conv's is B. MatMul has neither, so slot 2 there is
    free for vkop's `[A, B, scale]` extension. Gemm and Conv with a bias take
    the scale one slot further along, which is the shape `ops/Gemm.hpp` and
    `shaders/buffer/conv2d.comp` document:

        MatMul, no bias : [A, B, scale]
        Gemm,   w/ bias : [A, B, C, scale]
        Conv,   w/ bias : [X, W, B, scale]

    These are vkop extensions, not ONNX -- `onnx.checker` rejects them (a
    4-input Gemm is "not in range [min=2, max=3]") -- but the converter never
    runs the checker on a model this large, and onnxoptimizer and shape
    inference both tolerate the extra input. What must NOT happen is treating
    slot 2 as the scale when it is really a bias: that is what corrupted the
    MiniMax-H3 int4 conversion (see `scale_slot`).
    """
    return node.op_type in ("Gemm", "Conv") and len(node.input) >= 3


def scale_slot(node) -> int | None:
    """The input index a weight-only scale occupies, or will occupy, on `node`.

    A bias, when present, sits between the weight and the scale -- the kernels
    read `[A, B, C, scale]` and key the scale off `has_bias`, precisely so the
    two cannot be confused. So the answer is 3 for a biased Gemm/Conv and 2
    otherwise. MatMul never has a bias.
    """
    if node.op_type == "Gemm":
        return 3 if _has_bias(node) else 2
    if node.op_type == "Conv":
        return 3 if _has_bias(node) else 2
    if node.op_type == "MatMul":
        return 2
    return None


def finalize(model, deferred, verbose=True) -> int:
    """Attach a scale to every quantizing deferred entry, on the proto.

    For each entry carrying `quant`, computes `[name]_scale`, adds it as a
    FLOAT initializer and appends it to the node that CONSUMES the weight --
    the same third input `-q int8` appends for an in-memory weight. The tensor
    itself keeps its FLOAT type and its external reference; `write_payload`
    turns the bytes into the payload the entry names.

    The scale goes on the TERMINAL consumer, reached through the weight's pure
    view chain, not on whatever node happens to read the tensor. With constant
    folding off a linear arrives as `Transpose(w) -> MatMul(B)`, so the reader
    is a Transpose -- and Transpose is unary. Appending there makes a 2-input
    Transpose, which is not a legal ONNX node, and onnxoptimizer's IR asserts
    on it:

        ir.h:554: input: Assertion `inputs_.size() == 1' failed.

    That exception makes `optimize_model` skip its entire structural pass set,
    leaving only ConstantFolder -- and measured on the 27B that is enough to
    change which Shape/Slice/Gather chain folds, leaving Q's head axis symbolic
    (`[-1,-1,-1,256]` instead of `[-1,24,-1,256]`). The runtime then derived
    num_heads = -1 and rejected the artifact with the misleading
    `Attention: head_dim must be even and <= 256`.

    The terminal consumer is also where the in-memory path puts it, and
    `optimizer.fold_transpose_into_matmul` carries an extra operand from either
    side, so the two paths still meet at the same `MatMul(A, w, s, transB=1)`.
    """
    quantized = {n: e for n, e in deferred.items() if e.get("quant")}
    if not quantized:
        return 0
    # Imported here, not at module scope: weight_stream imports this module
    # lazily inside _write_quantized, and a top-level import would close the
    # cycle.
    from .weight_stream import view_chain_to_weight

    graph = model.graph
    uses = {}
    for node in graph.node:
        for idx, name in enumerate(node.input):
            uses.setdefault(name, []).append((node, idx))

    existing = {init.name for init in graph.initializer}
    appended = {}
    for name, entry in quantized.items():
        chain = view_chain_to_weight(uses, name)
        if chain is None:
            # prepare's deferral gate only lets a weight out of RAM when a pure
            # view chain reaches a weight position, so this should be
            # unreachable -- and it is a hard error rather than a guess because
            # the alternative is appending a scale to a node that computes on
            # the weight.
            raise ValueError(
                f"'{name}' is quantized on the way out but does not reach a "
                f"weight consumer through a pure view chain, so no node can "
                f"take its scale; the runtime would bind no scale and read raw "
                f"codes")
        carrier, _operand = chain[-1]
        slot = scale_slot(carrier)
        if slot is None:
            raise ValueError(
                f"'{name}' is quantized on the way out but its consumer is a "
                f"{carrier.op_type}, which has no weight-only scale contract; "
                f"the runtime would bind no scale and read raw codes")
        if len(carrier.input) > slot:
            # Already carries a scale: the quantizer's contract is a fixed
            # input list, and this pass must not manufacture a duplicate.
            # Record what is there so the missing check below does not mistake
            # "already done" for "could not".
            entry["scale"] = carrier.input[slot]
            if entry["quant"] == "nvfp4":
                if len(carrier.input) <= slot + 1:
                    raise ValueError(
                        f"'{name}' is nvfp4 but its consumer carries a block "
                        f"scale and no global scale; the kernel needs both")
                entry["scale_global"] = carrier.input[slot + 1]
            appended[name] = carrier.input[slot]
            continue
        scale_name = entry.get("scale") or f"{name}_scale"
        if entry["quant"] == "nvfp4":
            # Two levels, two initializers, two inputs: [A, W, s_e4m3, g], the
            # list ops/Matmul.hpp requires for a float4e2m1 weight and the
            # in-memory pass builds. Named as that pass names them.
            global_name = entry.get("scale_global") or f"{name}_scale_global"
            if scale_name not in existing:
                codes_t, global_scale, _deq, dead = nvfp4_tables(
                    WeightReader(entry, name),
                    bool(entry.get("transposed")), name)
                del _deq
                n_blocks, n_cols = codes_t.shape
                graph.initializer.append(helper.make_tensor(
                    scale_name, TensorProto.FLOAT8E4M3FN, [n_blocks, n_cols],
                    codes_t.reshape(-1).tobytes(), raw=True))
                graph.initializer.append(numpy_helper.from_array(
                    np.array([global_scale], dtype=np.float32), global_name))
                existing.update((scale_name, global_name))
                entry["nvfp4_dead"] = int(dead)
                del codes_t
            entry["scale"] = scale_name
            entry["scale_global"] = global_name
            carrier.input.append(scale_name)
            carrier.input.append(global_name)
            appended[name] = scale_name
            continue
        if scale_name not in existing:
            reader = WeightReader(entry, name)
            scale = scale_for(entry["quant"], reader,
                              bool(entry.get("transposed")),
                              int(entry.get("group") or 0))
            init = numpy_helper.from_array(scale, scale_name)
            init.data_type = TensorProto.FLOAT
            graph.initializer.append(init)
            existing.add(scale_name)
            # The values are now in the proto, and the writer reads them back
            # from there (weight_stream.scale_values_for). Keeping a second
            # numpy copy on the entry was the peak of a `-q int4` conversion:
            # the group tables are 1.5 GiB for the 27B, so both copies together
            # are 3 GiB of RSS that nothing needs at the same time -- measured
            # on a 10-layer slice, finalize alone accounted for 2.1 GiB of a
            # 5.2 GiB peak.
            del scale
        entry["scale"] = scale_name
        # A 4-bit scale is a [n_groups, N] TABLE, so the shape the consumer
        # records has to be the table's, not a per-column vector: the runtime
        # allocates the scale input from what the node declares and a [N]
        # declaration would truncate it to a quarter of its entries.
        carrier.input.append(scale_name)
        appended[name] = scale_name
    missing = sorted(set(quantized) - set(appended))
    if missing:
        raise ValueError(
            f"{len(missing)} deferred weight(s) are quantized on the way out "
            f"but no consumer could take their scale (e.g. {missing[:3]}); the "
            f"runtime would bind no scale and read raw codes")
    if verbose:
        print(f"[quant] {len(appended)} streaming weight(s) will be quantized "
              f"on the way into the trailer")
    return len(appended)
