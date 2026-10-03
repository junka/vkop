#!/usr/bin/env python3
"""Generate ORT reference inputs/outputs for a DiT vkop alignment check.

Everything the C++ driver needs is written to ref/ as raw little-endian files
(dtype given in ref/manifest.txt). The same files feed ORT here, so both
runtimes see bit-identical inputs and their velocities are directly comparable.
Model widths (ctx_dim, latent_c) are read from the ONNX graph, so the same
script serves the tiny and the 7.12B model.

    /Users/doudou/qi21-env/bin/python gen_dit_ref.py --steps 1
    /Users/doudou/qi21-env/bin/python gen_dit_ref.py --suffix "" --steps 1 --refdir ref_full

真实文本 + 官方调度的完整参考（对齐口径，只跑前 2 步 ORT，省 13.8 GB 图上的 38 次 decode）：

    encode_prompt_real.py --prompt "..." --prefix-len 64 --out-dir ref_real
    gen_dit_ref.py --suffix _static --prefix-len 64 --size 512 --steps 40 --ort-steps 2 \
        --real-rope --schedule diffusers --embeds ref_real/prompt_embeds.raw --refdir ref_text

只产输入张量、完全不碰 ORT（纯 vkop 出图，换 prompt 秒级）：

    gen_dit_ref.py --suffix _static --prefix-len 64 --size 512 --steps 40 \
        --real-rope --schedule diffusers --no-ort --embeds ... --refdir ref_text
    然后 image_gen ... --ref ref_text，env VKOP_REF_FREE=1（不回锚 ORT latent）
         + VKOP_REF_VK_KV=1（用 vkop 自己的 prefill KV）
"""

import argparse
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort

HERE = Path(__file__).resolve().parent
REF = HERE / "ref"
SCHED_DIR = Path("/Users/doudou/qi21-env/ms/models/Qwen--Qwen-Image-2.1"
                 "/snapshots/master/scheduler")


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
    ap.add_argument("--ort-steps", type=int, default=None,
                    help="只跑前 K 步 ORT 参考（调度 sigmas.raw 仍按 --steps 生成）；"
                         "省掉 13.8 GB 图上没必要的 39 次 decode")
    ap.add_argument("--suffix", default="_tiny", help="onnx model suffix, e.g. _tiny_static")
    ap.add_argument("--refdir", default="ref", help="output ref directory name")
    ap.add_argument("--embeds", default=None,
                    help="真实 prompt_embeds（fp16 raw [1,P,ctx_dim]，encode_prompt_real.py 产出）；"
                         "缺省用随机数")
    ap.add_argument("--latent", default=None,
                    help="初始 latent fp32 raw [1,target_len,latent_c]；缺省用随机数")
    ap.add_argument("--real-rope", action="store_true",
                    help="用 joint_rope_positions 的真 rope 取代简化公式")
    ap.add_argument("--no-ort", action="store_true",
                    help="只产输入张量（prompt_embeds/rope/bias/sigmas/manifest），"
                         "不跑 ORT；纯 vkop 真实文生图换 prompt 时用")
    ap.add_argument("--schedule", default="naive", choices=["naive", "diffusers"],
                    help="diffusers=FlowMatchEulerDiscrete 的动态位移 sigma（含 shift_terminal），"
                         "并把 sigmas.raw 写进 ref 目录供 C++ 用同一条调度")
    args = ap.parse_args()

    global REF
    REF = HERE / args.refdir
    args.ort_steps = args.steps if args.ort_steps is None else min(args.ort_steps, args.steps)

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
    valid_len = prefix_len
    if args.embeds:
        ep = Path(args.embeds)
        blob = ep.read_bytes()
        prompt_embeds = np.frombuffer(blob, dtype=np.float16)
        assert len(blob) % (2 * ctx_dim) == 0, "embeds 文件长度不是 ctx_dim 的整数倍"
        prompt_embeds = prompt_embeds.reshape(1, len(blob) // 2 // ctx_dim, ctx_dim)
        prefix_len = prompt_embeds.shape[1]
        vlen = ep.parent / "prompt_valid_len.txt"
        if vlen.exists():
            valid_len = int(vlen.read_text().strip())
        print(f"[ref] real prompt_embeds: prefix_len={prefix_len} valid_len={valid_len}")
    else:
        prompt_embeds = (rng.randn(1, prefix_len, ctx_dim) * 0.02).astype(np.float16)
    if args.latent:
        latent_init = np.fromfile(args.latent, dtype=np.float32).reshape(1, target_len, latent_c)
    else:
        latent_init = (rng.randn(1, target_len, latent_c)).astype(np.float32)
    latent_init_fp16 = latent_init.astype(np.float16)

    # latent_h / latent_w 已由 --size 定好（58 行），real rope 直接复用
    if args.real_rope:
        from qi21_dit import DiTConfig, joint_rope_positions
        axes = DiTConfig().axes_dims_rope
        # 补零行只在 prefill 里存在，decode 的 bias 会把它们屏蔽掉；图像段的 frame
        # 位置必须按**真实** token 数 valid_len 冻结，才和官方（无 padding）逐位等价。
        cos_p, sin_p = joint_rope_positions(prefix_len, latent_h, latent_w, axes, hd)
        cos_i, sin_i = joint_rope_positions(valid_len, latent_h, latent_w, axes, hd)
        cos_full = np.concatenate([cos_p[:prefix_len], cos_i[valid_len:]]).astype(np.float16)
        sin_full = np.concatenate([sin_p[:prefix_len], sin_i[valid_len:]]).astype(np.float16)
        print(f"[ref] real rope: axes={axes} txt_len={valid_len} grid={latent_h}x{latent_w}")
    else:
        # RoPE with the same simplified formula the C++ driver uses
        freqs = (np.arange(prefix_len + target_len)[:, None] * 0.01
                 + np.arange(hd)[None, :] * 0.001)
        cos_full = np.cos(freqs).astype(np.float16)
        sin_full = np.sin(freqs).astype(np.float16)

    bias_p = np.zeros((1, 1, prefix_len, prefix_len), dtype=np.float16)
    triu = np.triu(np.ones((prefix_len, prefix_len), dtype=bool), 1)
    bias_p[0, 0][triu] = -65504.0

    bias_t = np.zeros((1, 1, target_len, prefix_len + target_len), dtype=np.float16)
    if valid_len < prefix_len:
        bias_t[0, 0, :, valid_len:prefix_len] = -65504.0
        print(f"[ref] bias_decode: 屏蔽 {prefix_len - valid_len} 个 padding 列")

    save("prompt_embeds.raw", prompt_embeds)
    save("latent_init.raw", latent_init)  # fp32; C++ converts to fp16 itself
    save("cos_prefill.raw", cos_full[:prefix_len])
    save("sin_prefill.raw", sin_full[:prefix_len])
    save("cos_decode.raw", cos_full[prefix_len:])
    save("sin_decode.raw", sin_full[prefix_len:])
    save("bias_prefill.raw", bias_p)
    save("bias_decode.raw", bias_t)

    sigmas = None
    if args.schedule == "diffusers":
        # 官方调度：sigmas = linspace(1, 1/N, N) 经 FlowMatchEulerDiscrete 的动态位移
        # （mu 由 image seq len 算出，含 shift_terminal 收尾）。模型 timestep 用的就是
        # 位移后的 sigma 本身（pipeline 传的是 t/1000，而 t = sigma*1000）。
        # C++ 侧读同一份 sigmas.raw，两边逐位对齐。
        from diffusers import FlowMatchEulerDiscreteScheduler
        from diffusers.pipelines.qwenimage21.pipeline_qwenimage21 import calculate_shift

        sched = FlowMatchEulerDiscreteScheduler.from_pretrained(str(SCHED_DIR))
        mu = calculate_shift(target_len, sched.config.base_image_seq_len,
                             sched.config.max_image_seq_len, sched.config.base_shift,
                             sched.config.max_shift)
        sched.set_timesteps(sigmas=np.linspace(1.0, 1.0 / args.steps, args.steps),
                            mu=mu, device="cpu")
        sigmas = sched.sigmas.numpy().astype(np.float32)
        print(f"[ref] schedule: mu={mu:.4f} sigmas[:4]={sigmas[:4]} "
              f"sigmas[-2:]={sigmas[-2:]} len={len(sigmas)}")
        assert len(sigmas) == args.steps + 1, f"期望 steps+1 个 sigma，得到 {len(sigmas)}"
        save("sigmas.raw", sigmas)

    if args.no_ort:
        # 只产输入张量：vkop 纯 GPU 真实文生图用（配 VKOP_REF_VK_KV=1 + VKOP_REF_FREE=1），
        # 不必为每次换 prompt 载入 13.8 GB 的 ORT 图。
        kv_count = len(onnx.load(str(prefill_path), load_external_data=False).graph.output)
        manifest = {
            "prefix_len": prefix_len, "target_len": target_len, "ctx_dim": ctx_dim,
            "latent_c": latent_c, "hd": hd, "steps": args.steps,
            "valid_len": valid_len, "schedule": args.schedule, "real_rope": args.real_rope,
            "num_kv_layers": kv_count,
            "dtypes": {"latent_init": "fp32", "sigmas": "fp32", "default": "fp16"},
        }
        (REF / "manifest.txt").write_text(json.dumps(manifest, indent=2))
        print("[ref] inputs only (--no-ort) ->", REF)
        return

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
    if args.schedule == "diffusers":
        latent = latent_init.astype(np.float32)  # host 端 fp32 累加，喂图时才转 fp16
        for step in range(args.ort_steps):
            sigma, sigma_next = float(sigmas[step]), float(sigmas[step + 1])
            vel = decode_sess.run(None, {
                "target_latents": latent.astype(np.float16),
                "timestep": np.array([sigma], dtype=np.float32),
                "cos": cos_full[prefix_len:],
                "sin": sin_full[prefix_len:],
                "attention_bias": bias_t,
                **ort_kv,
            })[0].astype(np.float16)
            save(f"ort_velocity_step{step}.raw", vel)
            latent = latent + (sigma_next - sigma) * vel.astype(np.float32)
            save(f"ort_latent_after_step{step}.raw", latent.astype(np.float16))
    else:
        timesteps = np.linspace(1.0, 0.0, args.steps, endpoint=False)
        latent = latent_init_fp16.copy()
        for step in range(args.ort_steps):
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
        "valid_len": valid_len, "schedule": args.schedule, "real_rope": args.real_rope,
        "num_kv_layers": len(kv_names),
        "dtypes": {"latent_init": "fp32", "sigmas": "fp32", "default": "fp16"},
    }
    (REF / "manifest.txt").write_text(json.dumps(manifest, indent=2))
    print("[ref] done ->", REF)


if __name__ == "__main__":
    main()
