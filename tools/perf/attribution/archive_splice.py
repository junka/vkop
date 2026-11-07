#!/usr/bin/env python3
"""Extract / splice the VKSP shader archive embedded in a .vkopbin.

The single-variable attribution arm needs a model whose archive differs from
the "fix" model in exactly one entry (buffer_matmul_down_act_fp16), with every
other byte -- weights included -- identical. Rebuilding the model is not
possible (the converter would re-embed a different archive), so this script
rewrites the archive region in place:

  * the archive is the VKSP block that ends where the weight blob starts; the
    last 8 bytes of the model file hold that weight offset (LE u64), so the
    region length is weight_off - vksp_off;
  * blobs are repacked in entry order, 4-byte aligned, the target key's bytes
    taken from the donor model; the region keeps its original length and the
    leftover tail is zeroed, so nothing after the archive moves;
  * crc32 is recomputed (zlib.crc32) to match tools/spv_to_archive.py, even
    though core/ShaderArchive only validates offset/size/magic.

Usage:
  archive_splice.py list <model.vkopbin>
  archive_splice.py extract <model.vkopbin> <key> <out.spv>
  archive_splice.py splice <base.vkopbin> <donor.vkopbin> <key> <out.vkopbin>
"""
import struct, sys, zlib

HDR, ENTRY, ALIGN = 16, 72, 4


def align(n, a=ALIGN):
    return (n + a - 1) // a * a


def parse(path):
    d = open(path, 'rb').read()
    woff = struct.unpack_from('<Q', d, len(d) - 8)[0]
    if woff <= 0 or woff > len(d):
        sys.exit('bad weight-offset trailer')
    off = d.rfind(b'VKSP', 0, woff)
    if off < 0:
        sys.exit('no VKSP before the weight blob')
    ver, count = struct.unpack_from('<II', d, off + 4)
    entries = []
    for i in range(count):
        row = d[off + HDR + i * ENTRY: off + HDR + (i + 1) * ENTRY]
        name = row[:48].rstrip(b'\0').decode()
        bo, sz, crc = struct.unpack_from('<QII', row, 48)
        entries.append((name, d[bo:bo + sz]))
    return d, woff, off, entries


def pack(entries, region):
    rows, blobs = [], []
    cur = align(HDR + ENTRY * len(entries))
    for name, data in entries:
        rows.append((name, cur, len(data), zlib.crc32(data) & 0xFFFFFFFF))
        blobs.append((cur, data))
        cur = align(cur + len(data))
    total = cur
    if total > region:
        sys.exit('spliced archive (%d B) does not fit the region (%d B)'
                 % (total, region))
    buf = bytearray(region)
    buf[0:4] = b'VKSP'
    struct.pack_into('<II', buf, 4, 1, len(entries))
    for i, (name, off, sz, crc) in enumerate(rows):
        base = HDR + i * ENTRY
        nb = name.encode()[:48]
        buf[base:base + len(nb)] = nb
        struct.pack_into('<QII', buf, base + 48, off, sz, crc)
    for off, data in blobs:
        buf[off:off + len(data)] = data
    return bytes(buf)


def main():
    cmd = sys.argv[1]
    if cmd == 'list':
        _, _, off, entries = parse(sys.argv[2])
        for name, data in entries:
            print('%-52s %8d' % (name, len(data)))
        return
    if cmd == 'extract':
        _, _, _, entries = parse(sys.argv[2])
        key = sys.argv[3]
        for name, data in entries:
            if name == key:
                open(sys.argv[4], 'wb').write(data)
                return
        sys.exit('no such key: ' + key)
    if cmd == 'splice':
        src, donor, key, out = sys.argv[2:6]
        d, woff, off, entries = parse(src)
        _, _, _, dentries = parse(donor)
        hit = [e for e in dentries if e[0] == key]
        if not hit:
            sys.exit('donor has no such key: ' + key)
        new = [(n, hit[0][1] if n == key else b) for n, b in entries]
        region = woff - off
        blob = pack(new, region)
        d = bytearray(d)
        d[off:off + region] = blob
        open(out, 'wb').write(d)
        # The weight offset stays where it was: the region length did not
        # change, so the trailer is still correct.
        print('spliced %s from %s into %s (%d -> %d bytes, region %d)'
              % (key, donor, out, len(hit[0][1]), region, region))
        return
    sys.exit('usage: archive_splice.py list|extract|splice ...')


if __name__ == '__main__':
    main()
