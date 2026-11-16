"""Finalize the deferred-table entries for a byte-format quantization.

The scale of an int8 / fp8 weight is its per-column absmax and its payload
overwrites the values, so the conversion neither needs nor wants the weight in
RAM. weight_stream defers such a weight (see exports_weight_values) and records
what the payload will be; this module computes the one thing the writer cannot
derive on its own -- the scale -- and puts it into the graph as an ordinary
initializer, appended to each consumer the way every other weight-only format
appends its scale.

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
    if dtype == TensorProto.FLOAT16:
        return np.dtype(np.float16)
    if dtype == TensorProto.FLOAT:
        return np.dtype(np.float32)
    raise ValueError(f"deferred weight dtype {dtype} is not a float")


class WeightReader:
    """Streams a 2-D deferred weight, one band of rows at a time."""

    def __init__(self, entry, name):
        self.entry = entry
        self.name = name
        dims = list(entry["dims"])
        if len(dims) != 2:
            raise ValueError(
                f"'{name}': a byte-format weight must be 2-D, got {dims}")
        self.k, self.n = int(dims[0]), int(dims[1])
        self.dtype = _np_dtype_of(entry.get("src_dtype", entry["data_type"]))

    def bands(self, block=ROW_BLOCK):
        """Yield (row_offset, float32 [rows, N]) for the whole weight.

        The source is the file the entry points at; `length` is its extent in
        source bytes, which is `numel * itemsize(src_dtype)`. A short read is
        a hard error for the same reason it is in write_payload: the entry
        already promised those bytes.
        """
        path = self.entry["location"]
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


def scale_for(fmt, reader, transposed: bool = False):
    """The per-output-column fp32 scale an int8 / fp8 payload is decoded with."""
    from .optimizer import Quantizer

    amax = _amax(reader, transposed)
    if fmt == "int8":
        return np.where(amax == 0, 1.0, amax / 127.0).astype(np.float32)
    max_finite = Quantizer._FP8_FORMATS[fmt][0]
    return np.where(amax == 0, 1.0, amax / max_finite).astype(np.float32)


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
    FLOAT initializer and appends it to every node that reads the weight --
    the same third input `-q int8` appends for an in-memory weight. The tensor
    itself keeps its FLOAT type and its external reference; `write_payload`
    turns the bytes into the payload the entry names.
    """
    quantized = {n: e for n, e in deferred.items() if e.get("quant")}
    if not quantized:
        return 0
    graph = model.graph
    existing = {init.name for init in graph.initializer}
    appended = {}
    for node in graph.node:
        for idx in range(len(node.input)):
            name = node.input[idx]
            entry = quantized.get(name)
            if entry is None or len(node.input) > 2:
                # Already carries a scale (or something else in slot 2): the
                # quantizer's own contract is a three-input MatMul, and this
                # pass must not manufacture a fourth.
                continue
            scale_name = entry.get("scale") or f"{name}_scale"
            if scale_name not in existing:
                reader = WeightReader(entry, name)
                scale = scale_for(entry["quant"], reader,
                                  bool(entry.get("transposed")))
                init = numpy_helper.from_array(scale, scale_name)
                init.data_type = TensorProto.FLOAT
                graph.initializer.append(init)
                existing.add(scale_name)
                # The writer decodes with the same numbers; keeping them on the
                # entry means it does not have to sweep the source a second time.
                entry["scale_values"] = scale
            entry["scale"] = scale_name
            node.input.append(scale_name)
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
