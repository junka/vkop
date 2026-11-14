#!/usr/bin/env python3
"""Pack a set of compiled SPIR-V blobs into a single vkop_shaders.bin archive.

The entry names are the folder-derived keys CMakeLists.txt compiles each
shader under (`shaders/buffer/add.comp -> buffer_add`), which is exactly what
core::shader_key_for() returns for an op. So a runtime loader can look up
"buffer_add" or "image_add_fp16" and get the byte range for that kernel --
nothing about the shader is linked into libvkop.

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
    p = argparse.ArgumentParser(prog="spv_to_archive.py", fromfile_prefix_chars="@")
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
        try:
            with open(path, "rb") as f:
                data = f.read()
        except FileNotFoundError:
            raise SystemExit(
                f"spv_to_archive: source .spv missing for key {name!r}: {path}"
            )
        if len(data) == 0:
            raise SystemExit(
                f"spv_to_archive: source .spv is EMPTY for key {name!r}: {path}"
            )
        if len(data) < 4 or data[:4] != b"\x03\x02\x23\x07":
            raise SystemExit(
                f"spv_to_archive: {path} is not a SPIR-V module "
                f"(magic 0x07230203 expected, got {data[:4].hex()})"
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

    for i, ((_, expected_off, _, _), (name, data)) in enumerate(zip(offsets, entries)):
        if len(buf) != expected_off:
            raise AssertionError(
                f"blob placement drifted for {name}: "
                f"buf len is {len(buf)}, expected {expected_off} "
                f"(index {i}, {len(offsets)} total offsets)"
            )
        buf += data
        pad = _align_up(len(buf), BLOB_ALIGN) - len(buf)
        buf += b"\x00" * pad

    return bytes(buf)


def validate_written(path, expected_names):
    """Re-read the written archive and hard-fail on any structural problem.

    Checks:
      * magic + version
      * entry count matches expected
      * every named entry is present and has size > 0
      * no extra entries beyond expected_names
      * every blob starts with SPIR-V magic 0x07230203
      * blob bytes are exactly what we fed in (crc32 round-trip)
    """
    with open(path, "rb") as f:
        blob = f.read()

    errs = []
    if len(blob) < HEADER_SIZE or blob[:4] != MAGIC:
        raise SystemExit(
            f"spv_to_archive: written archive {path} has bad magic "
            f"(got {blob[:4].hex()})"
        )

    version, count = struct.unpack_from("<II", blob, 4)
    if version != VERSION:
        errs.append(f"archive version {version} != expected {VERSION}")

    expected = set(expected_names)
    seen = {}  # name -> (offset, size, crc)

    off = HEADER_SIZE
    for i in range(count):
        row = blob[off:off + ENTRY_SIZE]
        if len(row) < ENTRY_SIZE:
            errs.append(
                f"entry table truncated at index {i} "
                f"(offset={off}, file_size={len(blob)})"
            )
            break
        name = row[0:NAME_MAX].rstrip(b"\x00").decode("utf-8", "replace")
        bo, sz, crc = struct.unpack_from("<QII", row, 48)

        if not name:
            errs.append(f"entry {i} has empty name")
        elif name in seen:
            errs.append(f"duplicate name {name!r} at entry {i}")
        else:
            seen[name] = (bo, sz, crc)

        if sz == 0:
            errs.append(f"entry {i} ({name!r}) has size=0")
            off += ENTRY_SIZE
            continue

        if bo + sz > len(blob):
            errs.append(
                f"entry {i} ({name!r}) blob offset {bo} + size {sz} "
                f"exceeds file size {len(blob)}"
            )
        else:
            actual_magic = blob[bo:bo + 4]
            if actual_magic != b"\x03\x02\x23\x07":
                errs.append(
                    f"entry {i} ({name!r}) blob at offset {bo} "
                    f"has bad SPIR-V magic {actual_magic.hex()}"
                )

        off += ENTRY_SIZE

    # Count vs expected
    if count != len(expected):
        errs.append(
            f"entry count {count} != expected {len(expected)}"
        )

    missing = expected - set(seen.keys())
    extra = set(seen.keys()) - expected
    if missing:
        errs.append(
            f"archive is missing {len(missing)} expected keys: "
            f"{sorted(missing)[:10]}{' ...' if len(missing) > 10 else ''}"
        )
    if extra:
        errs.append(
            f"archive has {len(extra)} unexpected keys: "
            f"{sorted(extra)[:10]}{' ...' if len(extra) > 10 else ''}"
        )

    if errs:
        print("spv_to_archive: VALIDATION FAILED on written archive:",
              file=sys.stderr)
        for e in errs:
            print(f"  - {e}", file=sys.stderr)
        raise SystemExit(1)

    # All good — print summary
    total_bytes = sum(sz for _, sz, _ in seen.values())
    print(
        f"spv_to_archive: archive self-check OK — "
        f"{count} entries, {total_bytes} bytes of SPIR-V, "
        f"{len(blob)} bytes on disk",
        file=sys.stderr,
    )


def main(argv=None):
    args = parse_args(argv if argv is not None else sys.argv[1:])
    entries = collect(args.spv)
    expected_names = sorted(n for n, _ in entries)
    image = pack(entries)
    with open(args.output, "wb") as f:
        f.write(image)
    validate_written(args.output, expected_names)
    print(
        f"spv_to_archive: wrote {args.output} "
        f"({len(image)} bytes, {len(entries)} shaders)",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
