#!/usr/bin/env python3
"""Report-only probe: checksum the initializer blob embedded in a .vkopbin.

Parses the Model table by raw flatbuffers offsets (no generated code) and
per-initializer nonzero counts, so a silently-zeroed weight in a freshly
converted binary is caught without running the GPU.

    /Users/doudou/qi21-env/bin/python probe_vkopbin_weights.py dit_prefill_tiny_si.vkopbin ...
"""

import mmap
import struct
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent

# Model table field ids (declaration order in vkop_model.fbs)
F_INPUTS, F_OUTPUTS = 2, 3
F_NODES, F_INITIALZERS, F_BLOB = 4, 5, 6
F_UNIFIED, F_UNIFIED_META, F_UNIFIED_NAMES = 7, 9, 10
F_UNIFIED_BLOB_OFFSET = 13
# Node field ids
N_OPTYPE, N_NAME, N_INPUTS, N_OUTPUTS, N_DEPS = 0, 1, 3, 4, 5
# ShapeRef field ids
S_NAME, S_DIMS, S_VALUE_DYNAMIC = 0, 1, 4
# InitializerEntry field ids
E_NAME, E_DTYPE, E_DIMS, E_OFFSET, E_SIZE = 0, 1, 2, 3, 4

# InitializerEntry.dtype -> numpy view. `bfloat16` and the quantized payloads
# (fp8 one byte/value, 4-bit two values/byte, per dag.py _PACKED_DATA_TYPES) are
# shown as bytes: the probe reports what is stored, not what it decodes to.
_NP_VIEW = {
    "float16": np.float16, "float32": np.float32, "float64": np.float64,
    "bool": np.uint8, "string": np.uint8, "bfloat16": np.uint8,
    "uint8": np.uint8, "int8": np.int8,
    "uint16": np.uint16, "int16": np.int16,
    "uint32": np.uint32, "int32": np.int32,
    "uint64": np.uint64, "int64": np.int64,
    "float8e4m3fn": np.uint8, "float8e5m2": np.uint8,
    "uint4": np.uint8, "int4": np.uint8, "float4e2m1fn": np.uint8,
}
# Two values per byte: the element count is 2 * the byte count, not size/itemsize.
_PACKED = frozenset(("uint4", "int4", "float4e2m1fn"))

# UnifiedMeta.dtype stores the ONNX TensorProto element type verbatim
# (optimizer.py `dtype_map = {1: 1, 10: 10, 3: 3}`), so decode it with the same
# names dag.py `_DATA_TYPE_MAP` writes into InitializerEntry.dtype.
_ONNX_DTYPE_NAME = {1: "float32", 2: "uint8", 3: "int8", 6: "int32", 7: "int64",
                    9: "bool", 10: "float16", 11: "float64"}


def u32(b, p):
    return struct.unpack_from("<I", b, p)[0]


def field_pos(b, table_pos, field_id):
    soffset = struct.unpack_from("<i", b, table_pos)[0]
    vtable = table_pos - soffset
    vt_len = struct.unpack_from("<H", b, vtable)[0]
    off = 4 + field_id * 2
    if off >= vt_len:
        return None
    fo = struct.unpack_from("<H", b, vtable + off)[0]
    return table_pos + fo if fo else None


def field_u64(b, table_pos, field_id):
    """Scalar ulong field. A flatbuffers writer omits a field that equals its
    schema default, and `offset: ulong` (vkop_model.fbs) defaults to 0 -- so an
    absent field is 0, not a parse failure. The first initializer of every blob
    sits at offset 0 for exactly this reason."""
    pos = field_pos(b, table_pos, field_id)
    return 0 if pos is None else struct.unpack_from("<Q", b, pos)[0]


def read_string(b, pos):
    p = pos + u32(b, pos)
    n = u32(b, p)
    return b[p + 4:p + 4 + n].decode()


def read_uintvec(b, pos):
    if pos is None:  # absent vector == empty
        return []
    p = pos + u32(b, pos)
    n = u32(b, p)
    return list(struct.unpack_from(f"<{n}I", b, p + 4))


def read_intvec(b, pos):
    p = pos + u32(b, pos)
    n = u32(b, p)
    return list(struct.unpack_from(f"<{n}i", b, p + 4))


def read_blob(b, pos):
    p = pos + u32(b, pos)
    n = u32(b, p)
    return memoryview(b)[p + 4:p + 4 + n]


def probe_unified(b, root, blob, init_names):
    """Walk the `-u` unified region, which is where the weights actually live once
    Unifier has merged them.

    `unified_meta` is a vector of inline 32-byte `struct UnifiedMeta` (8 x 4-byte
    words: dtype, name_len, offset, size, dims[4]) and `unified_names` is ONE
    concatenated byte string consumed sequentially by name_len in meta order --
    the same two facts the unified loop in core/runtime.cpp reads. `offset` is
    RELATIVE to the region: the runtime basing its pointer at
    unified_blob_offset is what makes it absolute, and that is also the only way
    a multi-GiB weight region fits a 32-bit field at all. The InitializerEntry
    rows for the same names carry the absolute offset.

    offset/size are UNSIGNED (model/vkop_model.fbs): an fp16 LLM's region runs
    past 2^31, so reading them as signed would report the tail members as huge
    negative numbers.
    """
    mpos = field_pos(b, root, F_UNIFIED_META)
    if mpos is None:
        return
    p = mpos + u32(b, mpos)
    n = u32(b, p)
    npos = field_pos(b, root, F_UNIFIED_NAMES)
    if npos is None:
        print("[unified] meta present but no unified_names -- cannot resolve names")
        return
    np_ = npos + u32(b, npos)
    names = b[np_ + 4:np_ + 4 + u32(b, np_)]
    base = field_u64(b, root, F_UNIFIED_BLOB_OFFSET)
    print(f"[unified] {n} sub-tensors at blob+{base}, names table {len(names)} B, "
          f"{len(init_names)} InitializerEntry rows")
    cur = missing = 0
    for i in range(n):
        ep = p + 4 + i * 32
        dtype_i, name_len = struct.unpack_from("<2i", b, ep)
        off, size = struct.unpack_from("<2I", b, ep + 8)
        dims = list(struct.unpack_from("<4i", b, ep + 16))
        while dims and dims[-1] == 0:  # fixed-4 slot, zero padded
            dims.pop()
        name = names[cur:cur + name_len].decode(errors="replace")
        cur += name_len
        if name not in init_names:
            missing += 1
        dt = _ONNX_DTYPE_NAME.get(dtype_i, f"onnx{dtype_i}")
        abs_off = base + off
        view = np.frombuffer(blob[abs_off:abs_off + size], dtype=np.dtype(_NP_VIEW.get(dt, np.uint8)))
        nz = int(np.count_nonzero(view))
        print(f"    {name[:36]:36s} {dt:14s} {str(dims)[:22]:22s} "
              f"off={off}(+{base}={abs_off}) size={size:<8d} nz={nz}/{len(view)}")
    if missing:
        print(f"    !! {missing}/{n} unified names have no InitializerEntry row, "
              f"but the runtime looks each one up there (runtime.cpp:418)")


def probe(vkopbin):
    path = Path(vkopbin) if Path(vkopbin).is_absolute() else HERE / vkopbin
    b = path.read_bytes()
    if b[4:8] != b"VKOP":
        print(f"[{vkopbin}] bad magic {b[:8]!r}")
        return
    # flatbuffers GetRoot: root table at (0 + u32@0); file id follows at +4
    root = u32(b, 0)
    ipos = field_pos(b, root, F_INITIALZERS)
    bpos = field_pos(b, root, F_BLOB)
    if ipos is None or bpos is None:
        print(f"[{vkopbin}] missing initializers/blob field")
        return
    blob = read_blob(b, bpos)
    npos = field_pos(b, root, F_NODES)
    ops = {}
    if npos is not None:
        p = npos + u32(b, npos)
        for i in range(u32(b, p)):
            epos = p + 4 + i * 4
            tpos = epos + u32(b, epos)
            f = field_pos(b, tpos, N_OPTYPE)
            op = read_string(b, f) if f else "?"
            ops[op] = ops.get(op, 0) + 1
    print(f"[{vkopbin}] nodes={sum(ops.values())} ops={dict(sorted(ops.items()))}")
    p = ipos + u32(b, ipos)
    n = u32(b, p)
    init_names = set()
    print(f"[{vkopbin}] {n} initializers, blob={len(blob)} B "
          f"blob_nz={int(np.count_nonzero(np.frombuffer(blob, dtype=np.uint8)))}")
    for i in range(n):
        epos = p + 4 + i * 4
        tpos = epos + u32(b, epos)
        name = read_string(b, field_pos(b, tpos, E_NAME))
        init_names.add(name)
        dtype = read_string(b, field_pos(b, tpos, E_DTYPE))
        dims = read_uintvec(b, field_pos(b, tpos, E_DIMS))
        off = field_u64(b, tpos, E_OFFSET)
        size = field_u64(b, tpos, E_SIZE)
        np_dt = np.dtype(_NP_VIEW.get(dtype, np.uint8))
        view = np.frombuffer(blob[off:off + size], dtype=np_dt)
        nel = 2 * size if dtype in _PACKED else len(view)
        nz = int(np.count_nonzero(view))
        fin = view[np.isfinite(view)] if view.dtype.kind == "f" else view
        print(f"    {name[:36]:36s} {dtype:14s} {str(dims)[:22]:22s} "
              f"off={off:<9d} size={size:<8d} nz={nz}/{nel} "
              f"min={float(fin.min()) if fin.size else 0:.4g} max={float(fin.max()) if fin.size else 0:.4g} "
              f"mean={float(fin.mean()) if fin.size else 0:.4g}")
    probe_unified(b, root, blob, init_names)


def read_vec_strs(b, pos):
    p = pos + u32(b, pos)
    out = []
    for i in range(u32(b, p)):
        ep = p + 4 + i * 4
        sp = ep + u32(b, ep)  # sp is the string struct: u32 len + bytes
        n = u32(b, sp)
        out.append(b[sp + 4:sp + 4 + n].decode(errors="replace"))
    return out


def dump_nodes(vkopbin, pattern):
    path = Path(vkopbin) if Path(vkopbin).is_absolute() else HERE / vkopbin
    # mmap, not read_bytes: the 7.12B graphs are 13 GB and only the header is
    # touched here.
    with open(path, "rb") as fh:
        b = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
        try:
            _dump_nodes(b, vkopbin, pattern)
        finally:
            b.close()


def _dump_nodes(b, vkopbin, pattern):
    root = u32(b, 0)
    npos = field_pos(b, root, F_NODES)
    p = npos + u32(b, npos)
    print(f"[{vkopbin}] node dump (filter={pattern})")
    for i in range(u32(b, p)):
        epos = p + 4 + i * 4
        tpos = epos + u32(b, epos)
        f = field_pos(b, tpos, N_NAME)
        name = read_string(b, f) if f else ""
        if pattern and pattern not in name:
            continue
        fo = field_pos(b, tpos, N_OPTYPE)
        op = read_string(b, fo) if fo else "?"
        ins_pos = field_pos(b, tpos, N_INPUTS)
        ins = []
        if ins_pos:
            ip = ins_pos + u32(b, ins_pos)
            for k in range(u32(b, ip)):
                sp = ip + 4 + k * 4
                sp = sp + u32(b, sp)
                nf = field_pos(b, sp, 0)
                df = field_pos(b, sp, S_DIMS)
                ins.append((read_string(b, nf) if nf else "",
                            read_intvec(b, df) if df else []))
        outs_pos = field_pos(b, tpos, N_OUTPUTS)
        outs = []
        if outs_pos:
            op_ = outs_pos + u32(b, outs_pos)
            for k in range(u32(b, op_)):
                sp = op_ + 4 + k * 4
                sp = sp + u32(b, sp)
                nf = field_pos(b, sp, 0)
                df = field_pos(b, sp, S_DIMS)
                vd = field_pos(b, sp, S_VALUE_DYNAMIC)
                vdyn = b[vd] if vd else 0
                outs.append((read_string(b, nf) if nf else "",
                            read_intvec(b, df) if df else [], vdyn))
        dpos = field_pos(b, tpos, N_DEPS)
        deps = read_vec_strs(b, dpos) if dpos else []
        print(f"  #{i} {op} {name}")
        for tname, dims in ins:
            print(f"      in  {tname:34s} {dims}")
        for tname, dims, vdyn in outs:
            print(f"      out {tname:34s} {dims} dyn={vdyn}")
        print(f"      deps {deps}")


def dump_shapes(vkopbin):
    """Print the graph inputs/outputs baked into a .vkopbin (mmap: no 13GB read)."""
    path = Path(vkopbin) if Path(vkopbin).is_absolute() else HERE / vkopbin
    # The fd must stay open for the mmap's lifetime: closing it first makes the
    # mapping read garbage.
    with open(path, "rb") as fh:
        b = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
        try:
            root = u32(b, 0)
            for field, label in ((F_INPUTS, "in "), (F_OUTPUTS, "out")):
                pos = field_pos(b, root, field)
                if pos is None:
                    print(f"[{vkopbin}] no {label.strip()}put list")
                    continue
                p = pos + u32(b, pos)
                n = u32(b, p)
                print(f"[{vkopbin}] {label} {n} tensors")
                limit = min(n, 6) if label == "out" else n
                for i in range(limit):
                    sp = p + 4 + i * 4
                    sp = sp + u32(b, sp)
                    nf = field_pos(b, sp, S_NAME)
                    df = field_pos(b, sp, S_DIMS)
                    dims = read_intvec(b, df) if df else []
                    print(f"    {read_string(b, nf)[:36]:36s} {dims}")
                if n > limit:
                    print(f"    ... {n - limit} more")
        finally:
            b.close()


if __name__ == "__main__":
    args = sys.argv[1:]
    if args and args[0] == "--shapes":
        for arg in args[1:]:
            dump_shapes(arg)
        sys.exit(0)
    if args and args[0] == "--nodes":
        dump_nodes(args[1], args[2] if len(args) > 2 else "")
        sys.exit(0)
    for arg in args or ["dit_prefill_tiny.vkopbin",
                        "dit_prefill_tiny_si.vkopbin",
                        "dit_prefill_tiny_static.vkopbin"]:
        probe(arg)
