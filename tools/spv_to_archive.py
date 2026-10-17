#!/usr/bin/env python3
"""Pack a set of compiled SPIR-V blobs into a single vkop_shaders.bin archive.

The archive mirrors the folder_name convention CMakeLists.txt already uses
when embedding each shader as a C array (`shaders/buffer/add.comp ->
buffer_add_spv[]`). Entry names are therefore the same strings an op header
would reference through its `extern unsigned char X_spv[]` block, so a runtime
loader can look up "buffer_add" or "image_add_fp16" and get the byte range
that was previously baked into libvkop.

Layout (little-endian):
    offset  size   field
    0       4      magic "VKSP"
    4       4      version (u32, currently 1)
    8       4      entry count (u32)
    12      4      header pad (zero)
    16      72N    entry table, one fixed 72-byte row per entry:
                     0..48    name, ASCII, zero-padded
                     48..56   offset (u64) from file start to the blob
                     56..60   size (u32)
                     60..64   crc32 (u32) of the blob
                     64..72   reserved (zero)
    16+72N  ...    blob area, each blob padded to 4-byte alignment so a
                   SPIR-V word stream can be read directly via a uint32_t
                   cast into the mapped file.

Entries are sorted by name so the writer produces a deterministic file
regardless of the input order (the writer's non-determinism that showed up
in the FlatBuffers path would repeat here otherwise).

Usage:
    spv_to_archive.py --output build/vkop_shaders.bin \
        --spv buffer_add:/abs/path/build/shaders/buffer_add.spv \
        --spv buffer_add_fp16:/abs/path/build/shaders/buffer_add_fp16.spv \
        ...
"""
import argparse
import struct
import sys
import zlib

MAGIC = b"VKSP"
VERSION = 1
HEADER_SIZE = 16
ENTRY_SIZE = 72  # 48 name + 8 offset + 4 size + 4 crc + 8 reserved
NAME_MAX = 48
BLOB_ALIGN = 4  # SPIR-V is uint32-word-addressable


def _align_up(n, a):
    return (n + a - 1) // a * a


def parse_args(argv):
    p = argparse.ArgumentParser(prog="spv_to_archive.py")
    p.add_argument("--output", required=True, help="path to the .bin to write")
    p.add_argument(
        "--spv",
        action="append",
        required=True,
        metavar="NAME:PATH",
        help="entry NAME (e.g. buffer_add) mapped to a source .spv path; "
        "may be repeated",
    )
    return p.parse_args(argv)


def collect(entries):
    """Turn (NAME:PATH) strings into (NAME, bytes) tuples, checking sizes."""
    out = []
    for spec in entries:
        if ":" not in spec:
            raise SystemExit(f"spv_to_archive: malformed --spv {spec!r}")
        name, path = spec.split(":", 1)
        if not name:
            raise SystemExit(f"spv_to_archive: empty name in {spec!r}")
        if len(name.encode("utf-8")) > NAME_MAX:
            raise SystemExit(
                f"spv_to_archive: name {name!r} exceeds {NAME_MAX} bytes"
            )
        with open(path, "rb") as f:
            data = f.read()
        if len(data) < 4 or data[:4] != b"\x03\x02\x23\x07":
            raise SystemExit(
                f"spv_to_archive: {path} is not a SPIR-V module "
                f"(magic 0x07230203 expected)"
            )
        out.append((name, data))
    return out


def pack(entries):
    """Serialise entries into a byte string; entries sorted for determinism."""
    entries = sorted(entries, key=lambda e: e[0].encode("utf-8"))
    count = len(entries)

    blob_start = HEADER_SIZE + ENTRY_SIZE * count
    offsets = []
    cursor = blob_start
    for name, data in entries:
        crc = zlib.crc32(data) & 0xFFFFFFFF
        offsets.append((name, cursor, len(data), crc))
        cursor = _align_up(cursor + len(data), BLOB_ALIGN)

    buf = bytearray()
    buf += MAGIC
    buf += struct.pack("<II", VERSION, count)
    buf += b"\x00" * 4  # header pad

    for name, off, size, crc in offsets:
        row = bytearray(ENTRY_SIZE)
        nb = name.encode("utf-8")
        row[0:len(nb)] = nb
        row[48:56] = struct.pack("<Q", off)
        row[56:60] = struct.pack("<I", size)
        row[60:64] = struct.pack("<I", crc)
        # bytes 64..72 remain zero (reserved)
        buf += row

    for (_, _, _, _), (name, data) in zip(offsets, entries):
        assert len(buf) == offsets[[n for n, _ in entries].index(name)][1], (
            f"blob placement drifted for {name}"
        )
        buf += data
        pad = _align_up(len(buf), BLOB_ALIGN) - len(buf)
        buf += b"\x00" * pad

    return bytes(buf)


def main(argv=None):
    args = parse_args(argv if argv is not None else sys.argv[1:])
    entries = collect(args.spv)
    image = pack(entries)
    with open(args.output, "wb") as f:
        f.write(image)
    print(
        f"spv_to_archive: wrote {args.output} "
        f"({len(image)} bytes, {len(entries)} shaders)",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
