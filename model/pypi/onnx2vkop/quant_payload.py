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
convertible on a machine that cannot hold the model, and it is the only 4-bit
format here: nf4's codebook lookup is per-element but its *scale* is the group
absmax, so it would stream too -- it is refused for a different reason (the
nibble grid it writes is not what this module was validated against).

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
from onnx import TensorProto, numpy_helper

# Rows per read pass. Bounded so a 151936-row weight never materializes more
# than a few hundred MiB at a time.
ROW_BLOCK = 2048


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

    def bands(self, block=ROW_BLOCK):
        """Yield (row_offset, float32 [rows, N]) for the whole weight.

        The source is the file the entry points at; `length` is its extent in
        source bytes, which is `numel * itemsize(src_dtype)`. A short read is
        a hard error for the same reason it is in write_payload: the entry
        already promised those bytes.
        """
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


def code_nibble(fmt, reader, scale, group: int, transposed: bool = False):
    """A function (row_offset, band) -> int8 codes in [-8, 7], band-shaped.

    `scale` is the table `scale_for` produced: `[n_groups, N]` for int4. The
    grouping is applied by indexing it with the K index's group, so a band that
    starts mid-group still divides every row by the scale of the group that row
    belongs to.

    The division is the in-memory quantizer's, element for element:
    `Quantizer._quantize_w3` divides the [n_groups, group, N] operand by a
    [n_groups, 1, N] table, which for a row inside group g is exactly
    `band[row] / scale[g]`. Same operands, same fp32 rounding, so the codes are
    bit-identical -- the packed bytes only ever differ in how they are laid
    down, never in what they say.
    """
    if fmt != "int4":
        raise ValueError(f"code_nibble is the 4-bit encoder; got {fmt!r}")
    scale = np.asarray(scale, dtype=np.float32)

    if transposed:
        def codes(offset, band):
            rows = band.shape[0]
            # [rows, n_groups] -> [rows, K], one scale per K value of the row.
            s = np.repeat(scale[:, offset:offset + rows].T, group, axis=1)
            return np.clip(np.round(band / s), -8, 7).astype(np.int8)
        return codes

    def codes(offset, band):
        rows = band.shape[0]
        idx = np.arange(offset, offset + rows) // group
        return np.clip(np.round(band / scale[idx]), -8, 7).astype(np.int8)
    return codes


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
        if len(carrier.input) > 2:
            # Already carries a scale (or something else past the weight): the
            # quantizer's contract is a three-input MatMul, and this pass must
            # not manufacture a fourth. Record what is there so the missing
            # check below does not mistake "already done" for "could not".
            entry["scale"] = carrier.input[2]
            appended[name] = carrier.input[2]
            continue
        scale_name = entry.get("scale") or f"{name}_scale"
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
