#!/usr/bin/env python3
"""MiniMax-H3 Ref2VA host preparation (ComfyUI `MiniMaxH3ReferenceToVideo` port).

One section per consumer of every reference image. numpy + Pillow, no model
weights.

* VAE side: the ref resize policy (shared with the Qwen side) and the video
  VAE encoder's input and spatial tiling (h3_export_fl2va_vae_encoder.py
  encodes one tile).

    python h3_ref2va_prep.py vae IMAGE --width 512 --height 288 [--length 56] \
        [--ref-image-size match|max] --out DIR/ref0
  writes DIR/ref0.f32 ([3, th, tw] float32 planar RGB in [0, 1]) and
  DIR/ref0.json (th, tw, latent_h/w, the tile plan, the generation geometry).

* Qwen side: the token presentation, pixel patches for the vision tower
  (h3_export_ref2va_vision.py), MRoPE rope tables, DeepStack rows and the DiT
  token tags for the text tower (h3_export_fl2va_te.py --deepstack).

    python h3_ref2va_prep.py qwen --prompt "..." --image A.png [--image B.png] \
        [--audio-refs N] --width 512 --height 288 [--ref-image-size match|max] --out DIR

The Qwen side follows comfy/text_encoders/minimax.py (MiniMaxH3Tokenizer,
token_tags_from_embeds_info), qwen3vl.py (Qwen3VL.preprocess_embed /
build_image_inputs), qwen_vl.py (smart_resize, process_qwen2vl_images,
qwen2vl_patchify, qwen2vl_mrope_position_ids) and llama.py
(precompute_freqs_cis with interleaved MRoPE, DeepStack after layers 0..2).

Presentation (no chat template, no special tokens except the vision brackets):

    per ref, in request order (images, then videos, then standalone audio;
    1-based ordinals per kind):
      image -> "<Picture i>: " <|vision_start|> <n vision rows> <|vision_end|>
      audio -> "<Audio j>: "                   (audio never enters Qwen)
      video -> "<Video k>: " then per 2-frame block
               "<T.T seconds>" <|vision_start|> <rows> <|vision_end|>
    then the prompt. Each text piece is tokenized on its own.

Files written by `qwen` (all little-endian, row-major):

    ids.i64          [L]          token ids; vision rows hold 151655 (<|image_pad|>),
                                  their embedding is replaced by the vision tower
    tags.i64         [L]          DiT adaLN token tag: 0 for every vision block
                                  INCLUDING its <|vision_start|>/<|vision_end|>, 1 else
    rope_cos.f32     [L, 128]     text tower RoPE from the (t, h, w) MRoPE positions
    rope_sin.f32     [L, 128]
    pos3.i64         [3, L]       the MRoPE positions themselves (for checks)
    vision_<b>.f32   [S_b, 1536]  pixel patches of vision block b (grid_t = 1)
    ref2va_qwen.json              L, blocks: [{kind, ref, grid [1, gh, gw], start,
                                  rows, file}], deepstack layers, label pieces

The driver then runs, per block, vision graph(vision_<b>.f32, grid) ->
image_features [rows, 5120] + deepstack_features_0..2 [rows, 5120]; writes
image_features into inputs_embeds rows [start, start+rows) (the table rows of
all other ids come from embed_tokens.bf16); builds deepstack_i [L, 5120], zero
except those rows; and runs the text tower with all of them.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np

# ============================================================== Qwen side
VISION_START = 151652
VISION_END = 151653
IMAGE_PAD = 151655  # placeholder id in ids.i64 for rows the vision tower fills
# FL2VA/Ref2VA tokenizer extensions (ids fixed by the released tokenizer)
MINIMAX_EXTRA_TOKENS = {"<d>": 151669, "</d>": 151670, "<|cutoff|>": 151671,
                        "<|lyrics_start|>": 151672, "<|lyrics_end|>": 151673,
                        "<|caption_start|>": 151674, "<|caption_end|>": 151675}
QWEN_MEAN = 0.5
QWEN_STD = 0.5
PATCH, TEMPORAL, MERGE = 16, 2, 2
MIN_PIXELS, MAX_PIXELS = 3136, 12845056
# text tower (Qwen3-VL-32B): head_dim 128, theta 5e6, interleaved MRoPE [24, 20, 20]
HEAD_DIM = 128
ROPE_THETA = 5.0e6
MROPE_SECTION = (24, 20, 20)
DEEPSTACK_TEXT_LAYERS = (0, 1, 2)
FPS = 24


def qwen_tokenizer(tok_dir):
    """ComfyUI's Qwen2 tokenizer files plus MINIMAX_EXTRA_TOKENS."""
    from transformers import Qwen2Tokenizer
    tok = Qwen2Tokenizer.from_pretrained(tok_dir)
    tok.add_special_tokens({"additional_special_tokens": list(MINIMAX_EXTRA_TOKENS)})
    for t, i in MINIMAX_EXTRA_TOKENS.items():
        got = tok.convert_tokens_to_ids(t)
        if got != i:
            raise SystemExit(f"tokenizer maps {t} to {got}, want {i}")
    return tok


def text_ids(tok, s):
    """One text piece as comfy's SDTokenizer emits it: no specials, no padding."""
    return list(tok(s, add_special_tokens=False)["input_ids"]) if s else []


def smart_resize(height, width, factor=PATCH * MERGE, min_pixels=MIN_PIXELS,
                 max_pixels=MAX_PIXELS):
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


def qwen_patchify(frames):
    """frames float32 [T(=2), 3, H, W], normalized -> ([gh*gw, 1536], (1, gh, gw)).
    qwen2vl_patchify for grid_t = 1: rows in 2x2 merge-block order, each row
    (c, t, py, px)."""
    t, c, h, w = frames.shape
    assert t == TEMPORAL
    gh, gw = h // PATCH, w // PATCH
    p = frames.reshape(1, TEMPORAL, c, gh // MERGE, MERGE, PATCH, gw // MERGE, MERGE, PATCH)
    p = p.transpose(0, 3, 6, 4, 7, 2, 1, 5, 8)
    return (np.ascontiguousarray(p.reshape(gh * gw, c * TEMPORAL * PATCH * PATCH),
                                 dtype=np.float32), (1, gh, gw))


def _bilinear_to(frames, h, w):
    """F.interpolate(bilinear, align_corners=False) as comfy runs it; the
    identity when the size already matches (the ref2va case: the resized ref
    is a multiple of 32)."""
    if frames.shape[-2:] == (h, w):
        return frames
    import torch
    import torch.nn.functional as F
    return F.interpolate(torch.from_numpy(frames), size=(h, w), mode="bilinear",
                         align_corners=False).numpy()


def qwen_image_patches(image):
    """process_qwen2vl_images(patch 16, mean/std 0.5): `image` float32
    [H, W, 3] in [0, 1] (the same resized picture the VAE encodes) -> patches,
    grid. The frame is repeated to fill the temporal patch."""
    img = np.ascontiguousarray(image[..., :3].transpose(2, 0, 1))[None].astype(np.float32)
    h, w = smart_resize(img.shape[2], img.shape[3])
    img = _bilinear_to(img, h, w)
    img = (img - QWEN_MEAN) / QWEN_STD
    return qwen_patchify(np.repeat(img, TEMPORAL, axis=0))


def qwen_video_block_patches(frames):
    """process_video_block: two frames [2, H, W, 3] in [0, 1] fill the
    temporal patch (grid_t = 1)."""
    f = np.ascontiguousarray(frames[..., :3].transpose(0, 3, 1, 2)).astype(np.float32)
    h, w = smart_resize(f.shape[2], f.shape[3])
    f = _bilinear_to(f, h, w)
    return qwen_patchify((f - QWEN_MEAN) / QWEN_STD)


def ref2va_presentation(tok, prompt, refs):
    """Token list for MiniMaxH3Tokenizer.tokenize_with_weights(minimax_ref_items).

    `refs`: list of {"type": "image", "image": [H,W,3]} |
            {"type": "audio"} |
            {"type": "video", "frames": [T,H,W,3] (2 fps), "timestamps": opt}
    in comfy's ref_items order. Returns (ids, blocks, pieces): ids with
    IMAGE_PAD on vision rows, blocks [{kind, ref, start, rows, grid, patches}],
    pieces = the text pieces (for the json)."""
    ids, blocks, pieces = [], [], []
    counters = {"image": 0, "audio": 0, "video": 0}

    def add_text(s):
        pieces.append(s)
        ids.extend(text_ids(tok, s))

    def add_vision(kind, ref, patches, grid):
        ids.append(VISION_START)
        rows = patches.shape[0] // (MERGE * MERGE)
        blocks.append({"kind": kind, "ref": ref, "start": len(ids), "rows": rows,
                       "grid": list(grid), "patches": patches})
        ids.extend([IMAGE_PAD] * rows)
        ids.append(VISION_END)

    for r in refs:
        kind = r["type"]
        counters[kind] += 1
        if kind == "image":
            add_text("<Picture %d>: " % counters["image"])
            p, g = qwen_image_patches(r["image"])
            add_vision("image", counters["image"], p, g)
        elif kind == "audio":
            add_text("<Audio %d>: " % counters["audio"])
        elif kind == "video":
            frames = r["frames"]
            ts = r.get("timestamps")
            if ts is None:
                ts = [i / 2.0 for i in range(frames.shape[0])]
            if frames.shape[0] % 2 == 1:
                frames = np.concatenate([frames, frames[-1:]], 0)
                ts = list(ts) + [ts[-1]]
            add_text("<Video %d>: " % counters["video"])
            for i in range(0, frames.shape[0], 2):
                add_text("<%.1f seconds>" % ((ts[i] + ts[i + 1]) / 2.0))
                p, g = qwen_video_block_patches(frames[i:i + 2])
                add_vision("video", counters["video"], p, g)
        else:
            raise ValueError(kind)
    add_text(prompt)
    if not ids:
        ids.append(151643)
    return np.asarray(ids, np.int64), blocks, pieces


def mrope_positions(L, blocks):
    """qwen2vl_mrope_position_ids: [3, L] (t, h, w). Text runs sequentially;
    a vision block of merged grid (gh/2, gw/2) takes t = start, h/w = start +
    row/col, and the text after it resumes at start + max(gh, gw)/2."""
    pos = np.tile(np.arange(L, dtype=np.int64), (3, 1))
    offset = 0
    for b in blocks:
        start, end = b["start"], b["start"] + b["rows"]
        _, gh, gw = b["grid"]
        len_max = max(b["grid"]) // 2
        mh, mw = gh // 2, gw // 2
        nxt = start + len_max + offset
        pos[:, end:] = np.arange(nxt, nxt + (L - end))
        pos[0, start:end] = start + offset
        pos[1, start:end] = start + offset + np.repeat(np.arange(mh), mw)[:end - start]
        pos[2, start:end] = start + offset + np.tile(np.arange(mw), mh)[:end - start]
        offset += len_max - (end - start)
    return pos


def mrope_tables(pos3):
    """precompute_freqs_cis(interleaved_mrope=True, rope_dims [24, 20, 20]):
    T frequencies by default, H on dims 1, 4, .., 58 and W on 2, 5, .., 59 of
    the 64; then cos/sin of cat(f, f) -> [L, 128] each (rotate-half layout,
    what h3_export_fl2va_te.py consumes). Equal (t, h, w) gives plain RoPE."""
    inv = (1.0 / (ROPE_THETA ** (np.arange(0, HEAD_DIM, 2, dtype=np.float32)
                                 / HEAD_DIM))).astype(np.float32)
    f = pos3.astype(np.float32)[:, :, None] * inv[None, None, :]   # [3, L, 64]
    fi = f[0].copy()
    for axis, off in ((1, 1), (2, 2)):
        sl = slice(off, MROPE_SECTION[axis] * 3, 3)
        fi[:, sl] = f[axis][:, sl]
    emb = np.concatenate([fi, fi], -1)
    return np.cos(emb).astype(np.float32), np.sin(emb).astype(np.float32)


def token_tags(L, blocks):
    """token_tags_from_embeds_info: 0 over each vision block widened by one
    on each side (its <|vision_start|> and <|vision_end|>), 1 elsewhere."""
    tags = np.ones(L, np.int64)
    for b in blocks:
        tags[max(0, b["start"] - 1):b["start"] + b["rows"] + 1] = 0
    return tags


def causal_bias(L):
    return np.triu(np.full((L, L), -3.0e38, np.float32), 1)


def write_qwen_inputs(out_dir, ids, blocks, pieces):
    """Write the Qwen-side files (see the module docstring)."""
    os.makedirs(out_dir, exist_ok=True)
    L = len(ids)
    pos3 = mrope_positions(L, blocks)
    cos, sin = mrope_tables(pos3)
    ids.astype(np.int64).tofile(os.path.join(out_dir, "ids.i64"))
    token_tags(L, blocks).tofile(os.path.join(out_dir, "tags.i64"))
    pos3.tofile(os.path.join(out_dir, "pos3.i64"))
    cos.tofile(os.path.join(out_dir, "rope_cos.f32"))
    sin.tofile(os.path.join(out_dir, "rope_sin.f32"))
    meta = {"L": L, "hidden": 5120, "head_dim": HEAD_DIM,
            "deepstack_text_layers": list(DEEPSTACK_TEXT_LAYERS),
            "image_pad_id": IMAGE_PAD, "pieces": pieces, "blocks": []}
    for i, b in enumerate(blocks):
        fn = "vision_%d.f32" % i
        b["patches"].astype(np.float32).tofile(os.path.join(out_dir, fn))
        meta["blocks"].append({k: b[k] for k in ("kind", "ref", "start", "rows", "grid")}
                              | {"file": fn, "patch_rows": int(b["patches"].shape[0]),
                                 "patch_dim": int(b["patches"].shape[1])})
    with open(os.path.join(out_dir, "ref2va_qwen.json"), "w") as f:
        json.dump(meta, f, indent=1)
    return meta


def read_qwen_inputs(out_dir):
    """Inverse of write_qwen_inputs (patches loaded per block)."""
    with open(os.path.join(out_dir, "ref2va_qwen.json")) as f:
        meta = json.load(f)
    L = meta["L"]
    d = {"meta": meta,
         "ids": np.fromfile(os.path.join(out_dir, "ids.i64"), np.int64),
         "tags": np.fromfile(os.path.join(out_dir, "tags.i64"), np.int64),
         "rope_cos": np.fromfile(os.path.join(out_dir, "rope_cos.f32"), np.float32).reshape(L, HEAD_DIM),
         "rope_sin": np.fromfile(os.path.join(out_dir, "rope_sin.f32"), np.float32).reshape(L, HEAD_DIM)}
    d["patches"] = [np.fromfile(os.path.join(out_dir, b["file"]), np.float32)
                    .reshape(b["patch_rows"], b["patch_dim"]) for b in meta["blocks"]]
    return d


def assemble_text_inputs(d, table, features):
    """Every text tower input for one prompt, as the C++ driver builds them.
    `table`: embed_tokens.bf16 as uint16 [vocab, 5120]; `features`: per block
    (image_features [rows, 5120], [deepstack_0, _1, _2] each [rows, 5120])."""
    ids, meta = d["ids"], d["meta"]
    L, H = len(ids), meta["hidden"]
    emb = (table[ids].astype(np.uint32) << 16).view(np.float32).copy()
    ds = np.zeros((len(DEEPSTACK_TEXT_LAYERS), L, H), np.float32)
    for b, (feat, deep) in zip(meta["blocks"], features):
        s, n = b["start"], b["rows"]
        emb[s:s + n] = feat
        for i, x in enumerate(deep):
            ds[i, s:s + n] = x
    out = {"inputs_embeds": emb, "attention_bias": causal_bias(L),
           "rope_cos": d["rope_cos"], "rope_sin": d["rope_sin"]}
    for i in range(len(DEEPSTACK_TEXT_LAYERS)):
        out["deepstack_%d" % i] = ds[i]
    return out


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
    q = sub.add_parser("qwen", help="token presentation + vision patches + rope + tags")
    q.add_argument("--tokenizer", default=os.path.expanduser("~/models/h3_fl2va_te/tokenizer"))
    q.add_argument("--prompt", required=True, help="prompt text, or @file")
    q.add_argument("--image", action="append", default=[], help="reference image (repeatable)")
    q.add_argument("--audio-refs", type=int, default=0,
                   help="number of standalone reference audios (labels only)")
    q.add_argument("--width", type=int, required=True, help="generation width")
    q.add_argument("--height", type=int, required=True, help="generation height")
    q.add_argument("--ref-image-size", choices=["match", "max"], default="match")
    q.add_argument("--out", required=True)
    v = sub.add_parser("vae", help="ref pixels for the video VAE encoder graph (PREFIX.f32 + .json)")
    v.add_argument("image")
    v.add_argument("--width", type=int, required=True, help="generation width")
    v.add_argument("--height", type=int, required=True, help="generation height")
    v.add_argument("--length", type=int, default=0, help="frames (recorded in the json)")
    v.add_argument("--ref-image-size", choices=["match", "max"], default="match")
    v.add_argument("--out", required=True, help="prefix: writes PREFIX.f32, PREFIX.json")
    args = ap.parse_args(argv)
    if args.cmd == "vae":
        resized = load_ref_image(args.image, args.width, args.height, args.ref_image_size)
        m = write_vae_inputs(args.out, resized, args.width, args.height, args.length,
                             args.ref_image_size)
        print(f"[vae] {args.image} -> {m['tw']}x{m['th']} ({args.ref_image_size}), "
              f"{len(m['vae_tiles']['y']) * len(m['vae_tiles']['x'])} tile(s) -> {args.out}.f32")
        return 0
    prompt = args.prompt
    if prompt.startswith("@"):
        with open(prompt[1:]) as f:
            prompt = f.read().strip()
    tok = qwen_tokenizer(args.tokenizer)
    refs = [{"type": "image", "image": load_ref_image(p, args.width, args.height,
                                                      args.ref_image_size)}
            for p in args.image]
    refs += [{"type": "audio"}] * args.audio_refs
    ids, blocks, pieces = ref2va_presentation(tok, prompt, refs)
    meta = write_qwen_inputs(args.out, ids, blocks, pieces)
    print(f"[qwen] L={meta['L']} blocks=" +
          ", ".join(f"{b['kind']}{b['ref']} grid {b['grid']} rows {b['start']}+{b['rows']}"
                    for b in meta["blocks"]) + f" -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
