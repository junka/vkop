#!/usr/bin/env python3
"""MiniMax-H3 Ref2VA host preparation (ComfyUI `MiniMaxH3ReferenceToVideo` port).

One section per consumer of every reference image. This file currently holds
the VAE side: the ref resize policy (shared with the Qwen side) and the
video VAE encoder's input and spatial tiling (h3_export_fl2va_vae_encoder.py
encodes one tile). numpy + Pillow, no model weights.

    python h3_ref2va_prep.py vae IMAGE --width 512 --height 288 [--length 56] \
        [--ref-image-size match|max] --out DIR/ref0
writes DIR/ref0.f32 ([3, th, tw] float32 planar RGB in [0, 1]) and
DIR/ref0.json (th, tw, latent_h/w, the tile plan, the generation geometry).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np

# ============================================================== VAE side
# The ref image both consumers see is comfy's `resized` (MiniMaxH3ReferenceToVideo
# .execute): LoadImage (EXIF transpose, RGB, /255) -> aspect-preserving size,
# multiples of 32 -> comfy.utils.lanczos. Then MiniMaxH3VideoVAE.encode on a
# one-frame clip, 256-px tiles (h3_export_fl2va_vae_encoder.py encodes one tile).
CANVAS_MULTIPLE = 32
REF_IMAGE_SHORT_EDGE = 2048
FPS = 24
AUDIO_LATENT_FPS = 40
VAE_RATIO = 16
TILE, TILE_OVERLAP_MIN = 256, 64


def align_frame_count(n):
    while n % 17 != 5:
        n += 1
    return n


def temporal_shape(length):
    """-> (frame_count, video latent_t, audio_t), comfy's temporal_shape."""
    fc = align_frame_count(max(5, length))
    lt = 2 if fc <= 5 else ((fc - 5) // 17) * 5 + 2
    return fc, lt, round(fc / FPS * AUDIO_LATENT_FPS)


def load_image(path):
    """-> [H, W, 3] float32 in [0, 1] (comfy LoadImage, first frame)."""
    from PIL import Image, ImageOps
    im = ImageOps.exif_transpose(Image.open(path))
    if im.mode == "I":
        im = im.point(lambda i: i * (1 / 255))
    return np.asarray(im.convert("RGB")).astype(np.float32) / 255.0


def ref_image_target(w, h, width, height, mode="match"):
    """Source (w, h) -> (tw, th): "match" scales (down only, aspect kept) to the
    generation's pixel area, "max" to a <= 2048 short edge (never up); each side
    rounded to a multiple of 32 (min 32)."""
    if mode == "match":
        scale = min(1.0, math.sqrt((width * height) / (w * h)))
    elif mode == "max":
        scale = min(1.0, REF_IMAGE_SHORT_EDGE / min(w, h))
    else:
        raise ValueError(f"ref_image_size must be match or max, got {mode!r}")
    tw = max(CANVAS_MULTIPLE, round(w * scale / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
    th = max(CANVAS_MULTIPLE, round(h * scale / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
    return tw, th


def lanczos_resize(img, tw, th):
    """[H, W, 3] float32 [0, 1] -> [th, tw, 3] float32, comfy.utils.lanczos:
    x255 -> clip -> uint8 by truncation -> PIL LANCZOS -> /255. (For an 8-bit
    source the float round trip is exact, and PIL returns a copy at equal size,
    so this equals resizing the decoded PIL image directly.)"""
    from PIL import Image
    u8 = np.clip(np.float32(255.0) * img, 0, 255).astype(np.uint8)
    out = Image.fromarray(u8).resize((tw, th), resample=Image.Resampling.LANCZOS)
    return np.asarray(out).astype(np.float32) / 255.0


def load_ref_image(path, width, height, mode="match"):
    """-> float32 [th, tw, 3] in [0, 1]: comfy's `resized` ref (both consumers)."""
    img = load_image(path)
    tw, th = ref_image_target(img.shape[1], img.shape[0], width, height, mode)
    return lanczos_resize(img, tw, th)


def vae_ref_pixels(resized):
    """resized [th, tw, 3] -> [3, th, tw] float32 planar RGB in [0, 1]: the VAE
    encoder graph's pixels (the [-1, 1] map and imagenet norm are in the graph)."""
    return np.ascontiguousarray(resized[..., :3].transpose(2, 0, 1), dtype=np.float32)


def tile_plan(n):
    """One side of n pixels -> (starts, lengths, overlaps) in pixels
    (comfy MiniMaxH3VideoVAE.split_tiles)."""
    if TILE >= n:
        return [0], [n], []
    N = math.ceil(n / TILE)
    while True:
        ov = [TILE_OVERLAP_MIN] * (N - 1)
        rem = TILE * N - sum(ov) - n
        if rem < 0:
            N += 1
        else:
            break
    for i in range(rem // VAE_RATIO):
        ov[i % (N - 1)] += VAE_RATIO
    starts = [0]
    for i in range(N - 1):
        starts.append(starts[-1] + TILE - ov[i])
    return starts, [TILE] * N, ov


def _blend(a, b, ext, axis):
    """comfy blend: b's head = a's tail * (1 - i/ext) + b's head * (i/ext)."""
    ext = min(a.shape[axis], b.shape[axis], ext)
    wb = np.arange(ext, dtype=np.float32) / np.float32(ext)
    shape = [1] * a.ndim
    shape[axis] = ext
    wb = wb.reshape(shape)
    ta = np.take(a, np.arange(a.shape[axis] - ext, a.shape[axis]), axis)
    hb = np.take(b, np.arange(ext), axis)
    rest = np.take(b, np.arange(ext, b.shape[axis]), axis)
    return np.concatenate([ta * (1 - wb) + hb * wb, rest], axis)


def tiled_encode(encode_tile, px):
    """px [3, H, W]; encode_tile(px[:, y:y+h, x:x+w]) -> [24, h/16, w/16].
    -> [24, H/16, W/16] as comfy's tiled_encode: each tile blends against the
    UNBLENDED tile above, then the UNBLENDED tile to its left (latent overlap =
    pixel overlap / 16), then drops its trailing overlap rows/cols."""
    H, W = px.shape[1:]
    ys, yl, yo = tile_plan(H)
    xs, xl, xo = tile_plan(W)
    rows = [[np.asarray(encode_tile(px[:, y:y + h, x:x + w]), np.float32)
             for x, w in zip(xs, xl)] for y, h in zip(ys, yl)]
    lyo = [o // VAE_RATIO for o in yo]
    lxo = [o // VAE_RATIO for o in xo]
    out_rows = []
    for i, row in enumerate(rows):
        out = []
        for j, t in enumerate(row):
            if i > 0:
                t = _blend(rows[i - 1][j], t, lyo[i - 1], -2)
            if j > 0:
                t = _blend(row[j - 1], t, lxo[j - 1], -1)
            if i < len(rows) - 1:
                t = t[..., :-lyo[i], :]
            if j < len(row) - 1:
                t = t[..., :, :-lxo[j]]
            out.append(t)
        out_rows.append(np.concatenate(out, -1))
    return np.concatenate(out_rows, -2)


def write_vae_inputs(prefix, resized, width=0, height=0, length=0, mode="match"):
    """PREFIX.f32 = vae_ref_pixels [3, th, tw]; PREFIX.json = th, tw, latent
    h/w, the tile plan (and the generation geometry when given)."""
    px = vae_ref_pixels(resized)
    th, tw = px.shape[1:]
    os.makedirs(os.path.dirname(os.path.abspath(prefix)) or ".", exist_ok=True)
    px.tofile(prefix + ".f32")
    ys, _, yo = tile_plan(th)
    xs, _, xo = tile_plan(tw)
    meta = {"ref_image_size": mode, "tw": tw, "th": th,
            "latent_h": th // VAE_RATIO, "latent_w": tw // VAE_RATIO,
            "pixels": os.path.basename(prefix) + ".f32",
            "vae_tiles": {"tile_h": min(TILE, th), "tile_w": min(TILE, tw),
                          "y": ys, "x": xs, "y_overlap": yo, "x_overlap": xo}}
    if length:
        fc, lt, at = temporal_shape(length)
        meta.update(width=width, height=height, length=length,
                    frame_count=fc, latent_t=lt, audio_t=at)
    with open(prefix + ".json", "w") as f:
        json.dump(meta, f, indent=1)
    return meta


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("vae", help="ref pixels for the video VAE encoder graph (PREFIX.f32 + .json)")
    v.add_argument("image")
    v.add_argument("--width", type=int, required=True, help="generation width")
    v.add_argument("--height", type=int, required=True, help="generation height")
    v.add_argument("--length", type=int, default=0, help="frames (recorded in the json)")
    v.add_argument("--ref-image-size", choices=["match", "max"], default="match")
    v.add_argument("--out", required=True, help="prefix: writes PREFIX.f32, PREFIX.json")
    args = ap.parse_args(argv)
    resized = load_ref_image(args.image, args.width, args.height, args.ref_image_size)
    m = write_vae_inputs(args.out, resized, args.width, args.height, args.length,
                         args.ref_image_size)
    print(f"[vae] {args.image} -> {m['tw']}x{m['th']} ({args.ref_image_size}), "
          f"{len(m['vae_tiles']['y']) * len(m['vae_tiles']['x'])} tile(s) -> {args.out}.f32")
    return 0


if __name__ == "__main__":
    sys.exit(main())
