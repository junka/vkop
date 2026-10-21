#!/usr/bin/env python3
"""Filter a full vkop_shaders.bin archive down to a per-model subset.

Reads the master archive produced by `spv_to_archive.py` at build time and
writes a new archive that contains only the entries named on stdin (one per
line). Blobs are copied verbatim, so a shader selected here is byte-identical
to the one a full-archive run would use; a pipeline cache built against one
archive hits on the other.

serialize() returns that archive as bytes instead of writing it, which is what
onnx2vkop uses to embed the subset in the .vkopbin (Model.shader_archive).

Layout of the input and output archives is the same VKSP format (see
spv_to_archive.py for the field-by-field definition). Output is deterministic:
entries are re-sorted by name, so two invocations with the same set of keys
produce a byte-equal file regardless of the order they were fed in.

Usage:
    filter_shaders.py --master build/vkop_shaders.bin \
        --manifest /tmp/keys.txt \
        --output build/llm.vkopbin.shaders.bin

Missing keys are reported on stderr but do NOT abort the run — but a miss is a
hard failure at op build time, since libvkop embeds no SPIR-V of its own, so the
key set onnx2vkop feeds in has to be complete: the per-model subset is
filtered from the graph's kernel plan, which is the same set the runtime looks
up, and cli.build_shader_subset reports every planned key the master lacks.
"""
import argparse
import struct
import sys

MAGIC = b"VKSP"
VERSION = 1
HEADER_SIZE = 16
ENTRY_SIZE = 72
NAME_MAX = 48
BLOB_ALIGN = 4


def _align_up(n, a):
    return (n + a - 1) // a * a


def read_master(path):
    with open(path, "rb") as f:
        data = f.read()
    if len(data) < HEADER_SIZE or data[:4] != MAGIC:
        raise SystemExit(f"{path}: not a VKSP archive")
    version, count = struct.unpack_from("<II", data, 4)
    if version != VERSION:
        raise SystemExit(f"{path}: unsupported archive version {version}")
    out = {}
    for i in range(count):
        row = data[HEADER_SIZE + i * ENTRY_SIZE : HEADER_SIZE + (i + 1) * ENTRY_SIZE]
        name = row[:NAME_MAX].rstrip(b"\x00").decode("utf-8")
        off, size, crc = struct.unpack_from("<QII", row, 48)
        if off + size > len(data):
            raise SystemExit(f"{path}: entry {name} runs off the end")
        out[name] = data[off : off + size]
    return out


def write_subset(entries, path):
    """entries is a dict {name: bytes} to serialise into a new archive."""
    with open(path, "wb") as f:
        f.write(serialize(entries))


def serialize(entries):
    """Serialise {name: bytes} into the VKSP layout, returning the bytes.

    Same layout write_subset() puts on disk; onnx2vkop embeds the result in
    the .vkopbin instead of writing it next to one. Deterministic: entries
    are re-sorted by name, so the same key set always yields the same bytes
    (the FlatBuffers writer's determinism test relies on this).
    """
    ordered = sorted(entries.items(), key=lambda kv: kv[0].encode("utf-8"))
    count = len(ordered)
    blob_start = HEADER_SIZE + ENTRY_SIZE * count
    placed = []
    cursor = blob_start
    for name, data in ordered:
        crc = _crc32(data)
        placed.append((name, cursor, len(data), crc))
        cursor = _align_up(cursor + len(data), BLOB_ALIGN)

    buf = bytearray()
    buf += MAGIC
    buf += struct.pack("<II", VERSION, count)
    buf += b"\x00" * 4
    for name, off, size, crc in placed:
        row = bytearray(ENTRY_SIZE)
        nb = name.encode("utf-8")
        if len(nb) > NAME_MAX:
            raise SystemExit(f"entry name too long: {name!r}")
        row[0 : len(nb)] = nb
        row[48:56] = struct.pack("<Q", off)
        row[56:60] = struct.pack("<I", size)
        row[60:64] = struct.pack("<I", crc)
        buf += row
    for (name, off, size, crc), (_, data) in zip(placed, ordered):
        assert len(buf) == off, f"placement drift at {name}"
        buf += data
        pad = _align_up(len(buf), BLOB_ALIGN) - len(buf)
        buf += b"\x00" * pad

    return bytes(buf)


def _crc32(data):
    import zlib
    return zlib.crc32(data) & 0xFFFFFFFF


def parse_args(argv):
    p = argparse.ArgumentParser(prog="filter_shaders.py")
    p.add_argument("--master", required=True,
                   help="path to the full vkop_shaders.bin")
    p.add_argument("--manifest", required=True,
                   help="path to a text file listing one archive key per line")
    p.add_argument("--output", required=True,
                   help="path to write the filtered archive")
    p.add_argument("--strict", action="store_true",
                   help="abort if a manifest key is missing from the master")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv if argv is not None else sys.argv[1:])
    master = read_master(args.master)
    with open(args.manifest) as f:
        wanted = [line.strip() for line in f if line.strip() and not line.startswith("#")]
    missing = [k for k in wanted if k not in master]
    if missing:
        for k in missing:
            print(f"filter_shaders: master has no {k!r}", file=sys.stderr)
        if args.strict:
            raise SystemExit("strict mode: aborting on missing keys")
    subset = {k: master[k] for k in wanted if k in master}
    write_subset(subset, args.output)
    print(
        f"filter_shaders: {len(subset)}/{len(wanted)} keys kept, "
        f"{len(master)} available in master → {args.output}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
