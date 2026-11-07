#!/usr/bin/env python3
"""P0-1/P0-2 四臂 decode A/B：base/replay/batch/batch+replay × 3 轮交错。

臂（二进制 × VKOP_SEGMENT_REPLAY）：
  A = base  binary, replay off     B = base  binary, replay on
  C = batch binary, replay off     D = batch binary, replay on
每轮顺序 A B C D / D C B A / A B C D 抵消时钟漂移。
判据：全程墙钟/token（min-of-3）、rt->Run 逐步中位、token 流 md5 一致性。
"""
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
LOGDIR = ROOT / "tools" / "perf" / "logs"
MODEL = ROOT / "build" / "p0_models" / "int8_fused_p0.vkopbin"
EMBED = ROOT / "llm" / "exporter" / "embed_tokens.bin"
TOK = ROOT / "llm" / "tokenizer" / "qwen3_vl.bin"
PROMPT = "List the integers from 1 to 300, separated by commas."
STEPS = 208
CYCLES = 3

ARMS = {
    "A_base": (ROOT / "build" / "llm_chat_base", False),
    "B_base_replay": (ROOT / "build" / "llm_chat_base", True),
    "C_batch": (ROOT / "build" / "llm_chat", False),
    "D_batch_replay": (ROOT / "build" / "llm_chat", True),
}
ORDERS = [
    ["A_base", "B_base_replay", "C_batch", "D_batch_replay"],
    ["D_batch_replay", "C_batch", "B_base_replay", "A_base"],
    ["A_base", "B_base_replay", "C_batch", "D_batch_replay"],
]

STEP_RE = re.compile(r"\[r(\d+)\] ([\d.]+)ms")
TOK_RE = re.compile(r"→ (\d+)")


def run_arm(name, binary, replay, cycle):
    env = {
        "PATH": "/usr/local/bin:/usr/bin:/bin",
        "VKOP_NO_COOP": "1",
        "VKOP_PROFILE": "1",
        "VKOP_RAW_PROMPT": "1",
        "VKOP_SUBMIT_PROF": "1",
        "VKOP_SUBMITPROF_TOPN": "0",
    }
    if replay:
        env["VKOP_SEGMENT_REPLAY"] = "1"
    log_path = LOGDIR / f"p0_{name}_c{cycle}.log"
    t0 = time.perf_counter()
    with open(log_path, "w") as log:
        proc = subprocess.run(
            [str(binary), str(MODEL), str(EMBED), str(TOK), str(STEPS),
             "--max-kv", "320"],
            input=PROMPT, capture_output=True, text=True, env=env, cwd=ROOT,
            timeout=600)
        log.write(proc.stdout)
        log.write(proc.stderr)
    wall_s = time.perf_counter() - t0
    steps = [(int(m.group(1)), float(m.group(2)))
             for m in STEP_RE.finditer(proc.stdout)]
    toks = TOK_RE.findall(proc.stdout)
    if proc.returncode != 0 or len(steps) < STEPS * 0.9:
        print(f"  {name} c{cycle}: FAILED rc={proc.returncode} "
              f"steps={len(steps)} see {log_path}", flush=True)
        return None
    run_ms = [ms for _, ms in steps]
    return {
        "wall_ms_per_token": wall_s * 1000.0 / len(steps),
        "run_median": statistics.median(run_ms),
        "tok_md5": __import__("hashlib").md5(
            " ".join(toks).encode()).hexdigest(),
        "ntok": len(toks),
        "log": str(log_path),
    }


def main():
    LOGDIR.mkdir(parents=True, exist_ok=True)
    results = {}
    for cycle, order in enumerate(ORDERS, 1):
        for name in order:
            binary, replay = ARMS[name]
            print(f"cycle {cycle}: {name} ...", flush=True)
            r = run_arm(name, binary, replay, cycle)
            if r:
                results.setdefault(name, []).append(r)
                print(f"  wall/token={r['wall_ms_per_token']:.2f}ms "
                      f"run_med={r['run_median']:.2f}ms "
                      f"ntok={r['ntok']}", flush=True)
    print("\n==== min-of-3 汇总 ====")
    print(f"{'arm':<18} {'wall ms/tok':>12} {'run med':>10} {'tok md5':>10}")
    md5s = {}
    for name, rs in results.items():
        wall = min(r["wall_ms_per_token"] for r in rs)
        med = min(r["run_median"] for r in rs)
        md5s[name] = {r["tok_md5"] for r in rs}
        print(f"{name:<18} {wall:>12.2f} {med:>10.2f} "
              f"{('#' + sorted(md5s[name])[0][:8]):>10}")
    uniq = {m for s in md5s.values() for m in s}
    print(f"\ntoken 流一致性: {'一致 (' + str(len(md5s)) + ' 臂同 md5)' if len(uniq) == 1 else '不一致!! ' + str(uniq)}")


if __name__ == "__main__":
    sys.exit(main())
