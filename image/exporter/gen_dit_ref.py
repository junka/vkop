#!/usr/bin/env python3
"""Generate ORT reference inputs/outputs for a DiT vkop alignment check.

Everything the C++ driver needs is written to ref/ as raw little-endian files
(dtype given in ref/manifest.txt). The same files feed ORT here, so both
runtimes see bit-identical inputs and their velocities are directly comparable.
Model widths (ctx_dim, latent_c) are read from the ONNX graph, so the same
script serves the tiny and the 7.12B model.

    /Users/doudou/qi21-env/bin/python gen_dit_ref.py --steps 1
    /Users/doudou/qi21-env/bin/python gen_dit_ref.py --suffix "" --steps 1 --refdir ref_full
"""

import argparse
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort

HERE = Path(__file__).resolve().parent
REF = HERE / "ref"


def save(name, arr):
    arr = np.ascontiguousarray(arr)
    (REF / name).write_bytes(arr.tobytes())
    dims = "x".join(str(d) for d in arr.shape)
    with (REF / "shapes.txt").open("a") as f:
        f.write(f"{name} {dims} {arr.dtype.itemsize}\n")
    print(f"[ref] {name}: {arr.shape} {arr.dtype} -> {(REF/name).stat().st_size} B")
    return arr


def last_dim(onnx_path, input_name):
    """Trailing (feature) dim of a graph input, read from the proto only."""
    m = onnx.load(str(onnx_path), load_external_data=False)
    for i in m.graph.input:
        if i.name == input_name:
            return i.type.tensor_type.shape.dim[-1].dim_value
    raise KeyError(f"{input_name} not an input of {onnx_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix-len", type=int, default=8)
    ap.add_argument("--target-len", type=int, default=1024)
    ap.add_argument("--size", type=int, default=512, help="image size, target_len=(size/16)^2")
    ap.add_argument("--steps", type=int, default=1, help="decode steps to reference")
    ap.add_argument("--suffix", default="_tiny", help="onnx model suffix, e.g. _tiny_static")
    ap.add_argument("--refdir", default="ref", help="output ref directory name")
    args = ap.parse_args()

    global REF
    REF = HERE / args.refdir

    latent_h = latent_w = args.size // 16
    target_len = latent_h * latent_w
    assert target_len == args.target_len

    prefix_len = args.prefix_len
    prefill_path = HERE / f"dit_prefill{args.suffix}.onnx"
    decode_path = HERE / f"dit_decode{args.suffix}.onnx"
    ctx_dim = last_dim(prefill_path, "prompt_embeds")
    latent_c = last_dim(decode_path, "target_latents")
    hd = 128
    print(f"[model] ctx_dim={ctx_dim} latent_c={latent_c} (from {args.suffix or 'full'})")

    REF.mkdir(exist_ok=True)
    (REF / "shapes.txt").write_text("")
    rng = np.random.RandomState(1234)

    # ---- shared inputs (bit-identical for ORT and vkop) ----
    prompt_embeds = (rng.randn(1, prefix_len, ctx_dim) * 0.02).astype(np.float16)
    latent_init = (rng.randn(1, target_len, latent_c)).astype(np.float32)
    latent_init_fp16 = latent_init.astype(np.float16)

    # RoPE with the same simplified formula the C++ driver uses
    freqs = (np.arange(prefix_len + target_len)[:, None] * 0.01
             + np.arange(hd)[None, :] * 0.001)
    cos_full = np.cos(freqs).astype(np.float16)
    sin_full = np.sin(freqs).astype(np.float16)

    bias_p = np.zeros((1, 1, prefix_len, prefix_len), dtype=np.float16)
    triu = np.triu(np.ones((prefix_len, prefix_len), dtype=bool), 1)
    bias_p[0, 0][triu] = -65504.0

    bias_t = np.zeros((1, 1, target_len, prefix_len + target_len), dtype=np.float16)

    save("prompt_embeds.raw", prompt_embeds)
    save("latent_init.raw", latent_init)  # fp32; C++ converts to fp16 itself
    save("cos_prefill.raw", cos_full[:prefix_len])
    save("sin_prefill.raw", sin_full[:prefix_len])
    save("cos_decode.raw", cos_full[prefix_len:])
    save("sin_decode.raw", sin_full[prefix_len:])
    save("bias_prefill.raw", bias_p)
    save("bias_decode.raw", bias_t)

    # ---- ORT prefill ----
    prefill_sess = ort.InferenceSession(
        str(prefill_path), providers=["CPUExecutionProvider"])
    kv_names = [o.name for o in prefill_sess.get_outputs()]
    outs = prefill_sess.run(None, {
        "prompt_embeds": prompt_embeds,
        "cos": cos_full[:prefix_len],
        "sin": sin_full[:prefix_len],
        "timestep_zero": np.zeros(1, dtype=np.float32),
        "attention_bias": bias_p,
    })
    ort_kv = {}
    for name, val in zip(kv_names, outs):
        save(f"ort_{name}.raw", val)
        ort_kv[name.replace("present_", "past_")] = val
    del prefill_sess

    # ---- ORT decode, steps worth ----
    decode_sess = ort.InferenceSession(
        str(decode_path), providers=["CPUExecutionProvider"])
    timesteps = np.linspace(1.0, 0.0, args.steps, endpoint=False)
    latent = latent_init_fp16.copy()
    for step in range(args.steps):
        t = float(timesteps[step])
        sigma = 1.0 - t
        vel = decode_sess.run(None, {
            "target_latents": latent,
            "timestep": np.array([t], dtype=np.float32),
            "cos": cos_full[prefix_len:],
            "sin": sin_full[prefix_len:],
            "attention_bias": bias_t,
            **ort_kv,
        })[0].astype(np.float16)
        save(f"ort_velocity_step{step}.raw", vel)
        latent = (latent - sigma * vel).astype(np.float16)
        save(f"ort_latent_after_step{step}.raw", latent)

    manifest = {
        "prefix_len": prefix_len, "target_len": target_len, "ctx_dim": ctx_dim,
        "latent_c": latent_c, "hd": hd, "steps": args.steps,
        "num_kv_layers": len(kv_names),
        "dtypes": {"latent_init": "fp32", "default": "fp16"},
    }
    (REF / "manifest.txt").write_text(json.dumps(manifest, indent=2))
    print("[ref] done ->", REF)


if __name__ == "__main__":
    main()
