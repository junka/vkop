#!/usr/bin/env python3
"""Patch the embedded VKSP shader archive inside a .vkopbin, in place.

Re-converting a model just because one .comp file changed costs minutes on a
multi-GB artifact. The shaders live in a FlatBuffer byte-vector near the START
of the file, while the weights live in a blob AFTER the FlatBuffer -- so if the
replacement archive fits in the bytes the old one occupied, the model can be
patched without touching the weights at all.

What it does:
  1. Parse the file's root FlatBuffer table and find the shader_archive
     vector (its length and the offset of its first byte).
  2. Parse the master archive (build/vkop_shaders.bin) and rebuild the SUBSET
     the model carries, taking each entry's bytes from the master.
  3. Require: same key SET (no additions or removals -- the KernelPlan names
     keys and tests/KernelPlanTest.cpp asserts every planned key resolves),
     and new blob size <= the space the old vector occupied.
  4. Write the new VKSP bytes over the old ones, zero-fill the remainder, and
     leave the vector's declared length at the OLD value so every offset after
     it stays put. The archive parser stops at its own entry table, so the
     zero tail is never read as a shader.

Safety: refuses unless --force when the key sets differ. The file is mmapped,
never read whole: in place, only the archive range is written, after saving
the old archive bytes to `<model>.bak_shaders@<offset>`; with --out the file
is copied first and the copy is patched, then renamed into place. The result
is re-parsed and compared entry by entry before the tool reports success.

Requires the model to carry slack. `onnx2vkop` writes
`_SHADER_ARCHIVE_SLACK` (4 MiB) of trailing zeros inside the shader_archive
vector for exactly this; ShaderArchive::parse stops at its own entry table, so
the zeros never reach the runtime. For a model written before that, use
--shrink-dead to reclaim the UINT8 coop_i8 family (unreachable from a SINT8
weight) instead.

Usage:
  patch_model_shaders.py <model.vkopbin> <master vkop_shaders.bin>
                        [--out NEW.vkopbin] [--dry-run] [--force]
                        [--shrink-dead]
"""
import argparse
import mmap
import os
import shutil
import struct
import sys

KSP_MAGIC = b"VKSP"
KSP_HEADER = 16
KSP_ENTRY = 72
KSP_NAME_MAX = 48
SPIRV_MAGIC = b"\x03\x02\x23\x07"

# Keys whose kernel a SINT8 model can never dispatch (see --shrink-dead).
# The UINT8 coop_i8 family is selected only when VKOP_FORCE_I8_UINT8 is set;
# an ordinary -q int8 graph takes the SINT8 sibling.
import re
DEAD_KEY_RE = re.compile(r"^buffer_matmul_coop_u8")

# A minimal well-formed SPIR-V module: magic + version + generator + bound +
# schema (5 header words, no instructions). ShaderArchive::parse only checks
# the magic and that the range is in bounds, and nothing dispatches these.
DEAD_STUB = SPIRV_MAGIC + struct.pack("<IIII", 0x00010000, 0, 1, 0)
DEAD_STUB_SIZE = len(DEAD_STUB)


def parse_vksp(data, base=0):
    """Return {name: (abs_offset, size)} for a VKSP archive starting at `base`.

    Entry offsets are relative to `base` (the archive's own start), NOT to the
    file -- a model's archive sits a few MB into the .vkopbin.
    """
    if len(data) < base + KSP_HEADER or data[base:base + 4] != KSP_MAGIC:
        raise SystemExit("not a VKSP archive at offset %d" % base)
    version, count = struct.unpack_from("<II", data, base + 4)
    ents = {}
    for i in range(count):
        row = base + KSP_HEADER + i * KSP_ENTRY
        name = data[row:row + KSP_NAME_MAX].split(b"\x00", 1)[0].decode()
        rel, size = struct.unpack_from("<QI", data, row + 48)
        off = base + rel
        if off + size > len(data):
            continue
        if data[off:off + 4] != SPIRV_MAGIC:
            continue
        ents[name] = (off, size)
    return ents, version, count


def fb_root(data):
    """Byte offset of the root table (standard FlatBuffers header)."""
    if data[4:8] != b"VKOP":
        raise SystemExit("missing VKOP file identifier; not a .vkopbin")
    root_off = struct.unpack_from("<I", data, 0)[0]
    return root_off


def fb_field(data, table_off, field_index):
    """Locate a table field -> (offset_of_value, size_of_value_or_None).

    Handles the two shapes we need: a vector (returns (data_offset, length))
    and a scalar. Returns None when the field is absent.
    """
    vtable_rel = struct.unpack_from("<i", data, table_off)[0]
    vtable = table_off - vtable_rel
    vt_size = struct.unpack_from("<H", data, vtable)[0]
    slot = 4 + field_index * 2
    if slot >= vt_size:
        return None
    rel = struct.unpack_from("<H", data, vtable + slot)[0]
    if rel == 0:
        return None
    return table_off + rel


def patch(model_path, master_path, out_path, dry_run, force,
          shrink_dead=False):
    # mmap, not read(): a DiT/text-tower model is 14 GB and this machine has
    # 36 GB, so a whole-file bytearray (plus the old whole-file backup) could
    # push it into swap or an OOM reboot. Only the archive range is touched.
    # --out copies the file first and patches the copy.
    if out_path and not dry_run:
        tmp = out_path + ".tmp"
        shutil.copyfile(model_path, tmp)
        target = tmp
    else:
        target = model_path
    fh = open(target, "rb" if dry_run else "r+b")
    model = mmap.mmap(fh.fileno(), 0,
                      access=mmap.ACCESS_READ if dry_run else mmap.ACCESS_WRITE)
    root = fb_root(model)

    # shader_archive is field #15 (0-based) in `table Model` (magic, version,
    # inputs, outputs, nodes, initializers, initializer_blob, unified, rgba,
    # unified_meta, unified_names, rgba_meta, rgba_names, unified_blob_offset,
    # concurrent_levels, shader_archive). Count them in vkop_model.fbs if that
    # list ever grows -- a wrong index here points at a different vector.
    fld = fb_field(model, root, 15)
    if fld is None:
        raise SystemExit("model has no shader_archive field")
    vec_off = fld + struct.unpack_from("<I", model, fld)[0]
    vec_len = struct.unpack_from("<I", model, vec_off)[0]
    vec_data = vec_off + 4
    if vec_len == 0:
        raise SystemExit("model's shader_archive vector is empty")
    if bytes(model[vec_data:vec_data + 4]) != KSP_MAGIC:
        raise SystemExit("shader_archive field does not start with VKSP")

    old_ents, old_ver, old_count = parse_vksp(model, vec_data)
    master = open(master_path, "rb").read()
    new_all, new_ver, _ = parse_vksp(master, 0)

    if new_ver != old_ver:
        raise SystemExit(f"VKSP version mismatch: model v{old_ver}, "
                         f"master v{new_ver}")

    missing = sorted(set(old_ents) - set(new_all))
    added = sorted(set(new_all) - set(old_ents))
    if missing:
        raise SystemExit(
            "master archive lacks %d key(s) the model carries: %s"
            % (len(missing), ", ".join(missing[:5])))
    if added and not force:
        print(f"[patch] note: master has {len(added)} key(s) the model does "
              f"not carry; they are NOT added (a plan may only name keys the "
              f"model has). Pass --force to proceed anyway.")
        added = []

    # Rebuild the model's subset with the master's bytes.
    keys = sorted(old_ents)
    blobs = [(k, master[new_all[k][0]:new_all[k][0] + new_all[k][1]])
             for k in keys]
    n = len(blobs)
    hdr = KSP_HEADER + n * KSP_ENTRY
    body_len = 0
    for _, b in blobs:
        body_len += (len(b) + 3) & ~3
    new_size = hdr + body_len

    # The vector's declared length includes the writer's slack (see
    # _SHADER_ARCHIVE_SLACK in onnx2vkop/dag.py), so a bigger archive usually
    # still fits. The trailing zeros are invisible to ShaderArchive::parse.
    if new_size > vec_len:
        # An older model written without slack: the vector is exactly full, and
        # a recompiled shader
        # is usually a little larger (adding a branch adds SPIR-V). There is
        # no slack to write into, so the patcher has to free some first.
        #
        # `--shrink-dead` reclaims space from keys whose kernel the graph can
        # never dispatch: the UINT8 coop_i8 family is only reachable through
        # VKOP_FORCE_I8_UINT8, and a -q int8 model always takes the SINT8
        # sibling. Their entries keep their NAME (a KernelPlan may name them,
        # and tests/KernelPlanTest.cpp asserts every planned key resolves) but
        # point at a minimal valid SPIR-V module.
        if not shrink_dead:
            raise SystemExit(
                f"replacement archive is {new_size} bytes but the model "
                f"reserved {vec_len} ({new_size - vec_len} over). Re-run with "
                f"--shrink-dead to reclaim the UINT8 coop_i8 family "
                f"(unreachable for a SINT8 weight), or reconvert.")
        reclaimed = 0
        for i, (name, blob) in enumerate(blobs):
            if not DEAD_KEY_RE.search(name):
                continue
            if len(blob) > DEAD_STUB_SIZE:
                reclaimed += len(blob) - DEAD_STUB_SIZE
                blobs[i] = (name, DEAD_STUB)
        body_len = sum((len(b) + 3) & ~3 for _, b in blobs)
        new_size = hdr + body_len
        print(f"[patch] shrank {len(DEAD_KEY_RE.pattern)}-matching keys: "
              f"reclaimed ~{reclaimed} bytes, archive now {new_size}")
        if new_size > vec_len:
            raise SystemExit(
                f"still {new_size - vec_len} bytes over after shrinking; "
                f"reconvert instead.")

    out = bytearray(struct.pack("<4sIII", KSP_MAGIC, new_ver, n, 0))
    table = bytearray()
    body = bytearray()
    cursor = hdr
    for name, blob in blobs:
        nm = name.encode()
        if len(nm) > KSP_NAME_MAX:
            raise SystemExit("key too long for the entry table: " + name)
        table += nm.ljust(KSP_NAME_MAX, b"\x00")
        table += struct.pack("<QI", cursor, len(blob))
        table += struct.pack("<I", 0)          # crc (parser ignores)
        table += b"\x00" * 8                   # reserved
        body += blob
        pad = len(blob) % 4
        if pad:
            body += b"\x00" * (4 - pad)
            cursor += len(blob) + (4 - pad)
        else:
            cursor += len(blob)
    assert len(table) == n * KSP_ENTRY
    archive = bytes(out + table + body)
    assert len(archive) == new_size

    if dry_run:
        print(f"[patch] dry run: {n} key(s), archive {len(model[vec_data:vec_data + vec_len])}"
              f" -> {new_size} bytes; would fit ({vec_len} available)")
        changed = []
        for name, blob in blobs:
            o_off, o_size = old_ents[name]
            if model[o_off:o_off + o_size] != blob:
                changed.append(name)
        print(f"[patch] {len(changed)} shader(s) differ: "
              f"{', '.join(changed[:8])}{'...' if len(changed) > 8 else ''}")
        return 0

    # Write in place, keeping the OLD declared length so nothing after the
    # vector moves. The parser uses the entry table, not the vector's tail.
    # Back up only the bytes about to change (the archive range), with the
    # offset in the name so it can be written back.
    if not out_path:
        backup = f"{model_path}.bak_shaders@{vec_data}"
        with open(backup, "wb") as f:
            f.write(model[vec_data:vec_data + vec_len])
        print(f"[patch] old archive bytes saved to {backup}")
    model[vec_data:vec_data + vec_len] = archive + b"\x00" * (vec_len - len(archive))

    # Verify.
    check, _, _ = parse_vksp(model, vec_data)
    ok = set(check) == set(old_ents)
    for name, blob in blobs:
        o_off, o_size = check.get(name, (0, -1))
        if o_size != len(blob) or model[o_off:o_off + o_size] != blob:
            ok = False
    if not ok:
        if not out_path:
            model[vec_data:vec_data + vec_len] = open(backup, "rb").read()
        model.flush()
        raise SystemExit("patched archive does not round-trip; restored")
    model.flush()
    model.close()
    fh.close()

    dest = out_path or model_path
    if out_path:
        os.replace(target, out_path)
    print(f"[patch] OK: {n} shader(s) patched into {dest} "
          f"({new_size}/{vec_len} bytes used)")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("master")
    ap.add_argument("--out")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--shrink-dead", action="store_true",
                    help="free space by stubbing keys this graph can "
                         "never dispatch (UINT8 coop_i8 family)")
    a = ap.parse_args()
    return patch(a.model, a.master, a.out, a.dry_run, a.force,
                 a.shrink_dead)


if __name__ == "__main__":
    sys.exit(main())
