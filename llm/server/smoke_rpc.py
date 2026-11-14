#!/usr/bin/env python3
"""Smoke-test the llm_serve line protocol (no HTTP involved)."""
import base64
import json
import os
import subprocess
import sys

if os.name == "nt":
    import msvcrt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
BIN = os.path.join(ROOT, "build",
                   "llm_serve.exe" if os.name == "nt" else "llm_serve")
MODEL = os.path.join(ROOT, "llm", "exporter", "llm.vkopbin")
EMBED = os.path.join(ROOT, "llm", "exporter", "embed_tokens.bin")
TOK = os.path.join(ROOT, "llm", "tokenizer", "qwen3_vl.bin")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-new", type=int, default=16)
    ap.add_argument("--max-kv", type=int, default=1024)
    ap.add_argument("--bin", default=BIN)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--prompt", default="What is the capital of France?")
    ap.add_argument("--verbose", action="store_true",
                    help="pass the backend's own log through to stderr")
    a = ap.parse_args()

    r_in, w_in = os.pipe()
    r_out, w_out = os.pipe()
    if os.name == "nt":
        # pass_fds is POSIX-only; llm_serve wraps these HANDLEs with
        # _open_osfhandle, and close_fds=False is what inherits them.
        os.set_inheritable(r_in, True)
        os.set_inheritable(w_out, True)
        rpc = ["--rpc-in", str(msvcrt.get_osfhandle(r_in)),
               "--rpc-out", str(msvcrt.get_osfhandle(w_out))]
        extra = {"close_fds": False}
    else:
        rpc = ["--rpc-in", str(r_in), "--rpc-out", str(w_out)]
        extra = {"pass_fds": (r_in, w_out)}
    p = subprocess.Popen(
        [a.bin, a.model, EMBED, TOK, "--max-kv", str(a.max_kv)] + rpc,
        stdout=sys.stderr if a.verbose else subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
        **extra,
    )
    os.close(r_in)
    os.close(w_out)
    wf = os.fdopen(w_in, "wb", buffering=0)
    rf = os.fdopen(r_out, "rb")

    def send(line):
        wf.write((line + "\n").encode())

    def recv():
        line = rf.readline()
        if not line:
            raise SystemExit("server closed the pipe")
        return line.decode().rstrip("\n")

    first = recv()
    print("hello:", first)
    if not first.startswith("READY"):
        p.kill()
        raise SystemExit(1)

    def send_msgs(msgs):
        for role, text in msgs:
            send(role)
            send(base64.b64encode(text.encode()).decode())

    send(f"COUNT 1 1")
    send_msgs([("user", a.prompt)])
    print("count:", recv())

    send(f"GEN 2 {a.max_new} 1")
    send_msgs([("user", a.prompt)])
    out = []
    while True:
        line = recv()
        kind = line.split(" ", 1)[0]
        if kind == "DELTA":
            _, rid, b = line.split(" ", 2)
            out.append(base64.b64decode(b).decode())
        elif kind in ("DONE", "ERROR"):
            print(kind.lower() + ":", line)
            break
        else:
            print("frame:", line)
    print("TEXT:", "".join(out))

    # Second request with history: exercises the KV prefix reuse path.
    send(f"GEN 3 {a.max_new} 3")
    send_msgs([
        ("user", a.prompt),
        ("assistant", "".join(out)),
        ("user", "And of Italy?"),
    ])
    out2 = []
    while True:
        line = recv()
        kind = line.split(" ", 1)[0]
        if kind == "DELTA":
            out2.append(base64.b64decode(line.split(" ", 2)[2]).decode())
        elif kind in ("DONE", "ERROR"):
            print(kind.lower() + ":", line)
            break
        else:
            print("frame:", line)
    print("TEXT2:", "".join(out2))

    send("BYE")
    print("bye:", recv())
    p.wait(timeout=30)


if __name__ == "__main__":
    main()
