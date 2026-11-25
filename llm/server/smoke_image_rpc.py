#!/usr/bin/env python3
"""Smoke-test the image_serve line protocol (no HTTP involved).

Runs: COUNT, an over-long prompt (rejected without touching the GPU), a fresh
prompt with n=2, the same prompt again (prompt-KV cache hit, decode graph still
resident), a STOP after two steps, UNLOAD, BYE. Prints every protocol line and
the per-request metrics.

  python3 llm/server/smoke_image_rpc.py --out-dir /tmp/img --steps 8
"""
import argparse
import base64
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
IMG = os.path.join(ROOT, "image", "exporter")


def b64(s):
    return base64.b64encode(s.encode()).decode()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", default=os.path.join(ROOT, "build", "image_serve"))
    ap.add_argument("--prefill", default=os.path.join(IMG, "dit_prefill_static.vkopbin"))
    ap.add_argument("--decode", default=os.path.join(IMG, "dit_decode_static.vkopbin"))
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--prompt", default="A red fox sitting on a wooden bench in a sunlit park")
    ap.add_argument("--log", default=None, help="backend log file (default: discard)")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    r_in, w_in = os.pipe()
    r_out, w_out = os.pipe()
    env = dict(os.environ)
    env.setdefault("DYLD_FALLBACK_LIBRARY_PATH", "/opt/homebrew/lib")
    env.setdefault("VK_ICD_FILENAMES", "/opt/homebrew/etc/vulkan/icd.d/MoltenVK_icd.json")
    log = open(a.log, "wb") if a.log else subprocess.DEVNULL
    p = subprocess.Popen(
        [a.bin, a.prefill, a.decode,
         "--text-encoder", os.path.join(IMG, "text_encoder.vkopbin"),
         "--embeds", os.path.join(IMG, "text_encoder_embeds.bin"),
         "--tokenizer", os.path.join(ROOT, "llm", "tokenizer", "qwen3_vl.bin"),
         "--vae", os.path.join(IMG, "vae_decoder_512.vkopbin"),
         "--out-dir", a.out_dir,
         "--rpc-in", str(r_in), "--rpc-out", str(w_out)],
        stdout=log, stderr=subprocess.STDOUT, pass_fds=(r_in, w_out), env=env)
    os.close(r_in)
    os.close(w_out)
    wf = os.fdopen(w_in, "wb", buffering=0)
    rf = os.fdopen(r_out, "rb")
    t0 = time.time()

    def send(line):
        wf.write((line + "\n").encode())

    def recv():
        line = rf.readline()
        if not line:
            raise SystemExit(f"server closed the pipe (rc={p.wait()})")
        return line.decode().rstrip("\n")

    first = recv()
    print(f"[{time.time() - t0:6.2f}s] {first}")
    if not first.startswith("READY"):
        raise SystemExit(1)
    max_tokens = int(first.split()[4])

    def run(req_id, stop_after=None):
        """Drain one GEN; send STOP once `stop_after` steps are reported."""
        t = time.time()
        images = []
        while True:
            line = recv()
            parts = line.split(" ")
            kind = parts[0]
            if kind == "STEP":
                if stop_after and int(parts[3]) == stop_after:
                    send(f"STOP {req_id}")
                if int(parts[3]) not in (1, int(parts[4])):
                    continue  # keep the log short
            if kind == "IMAGE":
                images.append(base64.b64decode(parts[4]).decode())
                line = " ".join(parts[:4]) + " " + images[-1]
            if kind == "DONE":
                m = json.loads(base64.b64decode(parts[4]))
                print(f"[{time.time() - t0:6.2f}s] DONE {req_id} {parts[2]} n={parts[3]} "
                      f"wall={time.time() - t:.2f}s metrics={m}")
                return parts[2], images, m
            if kind == "ERROR":
                msg = base64.b64decode(parts[2]).decode()
                print(f"[{time.time() - t0:6.2f}s] ERROR {req_id}: {msg}")
                return "error", images, None
            print(f"[{time.time() - t0:6.2f}s] {line}")

    send(f"COUNT c1 {b64(a.prompt)}")
    print(f"[{time.time() - t0:6.2f}s] {recv()}")

    send(f"GEN long {a.steps} 1 1 {b64('a cat ' * max_tokens)}")
    reason, _, _ = run("long")
    assert reason == "error", "over-long prompt must be rejected"

    send(f"GEN a {a.steps} {a.seed} 2 {b64(a.prompt)}")
    reason, imgs_a, m_a = run("a")
    assert reason == "done" and len(imgs_a) == 2 and not m_a["prompt_cache_hit"]

    send(f"GEN b {a.steps} {a.seed + 10} 1 {b64(a.prompt)}")
    reason, imgs_b, m_b = run("b")
    assert reason == "done" and m_b["prompt_cache_hit"] and m_b["load_s"] == 0

    send(f"GEN c {a.steps} 3 2 {b64(a.prompt)}")
    reason, imgs_c, m_c = run("c", stop_after=2)
    assert reason == "stop" and not imgs_c and m_c["steps_run"] <= 3

    send("UNLOAD")
    print(f"[{time.time() - t0:6.2f}s] {recv()}")
    send("BYE")
    print(f"[{time.time() - t0:6.2f}s] {recv()}")
    print("rc", p.wait())
    print("images:", imgs_a + imgs_b)


if __name__ == "__main__":
    main()
