"""Where a tool call ends, and when the reply is over.

The model sometimes writes a complete call and then skips the closing marker
and rambles. Requiring the marker turns a usable call into one raw blob the
client retries forever; accepting the balanced JSON object instead recovers it.
The streaming and non-streaming paths must agree, so both are checked against
the same inputs at several chunkings.

Run: python llm/server/tests/test_tool_parser.py
"""
import importlib.util
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location(
    "anthropic_server", os.path.join(HERE, os.pardir, "anthropic_server.py"))
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

OPEN, CLOSE = m.TOOL_OPEN, m.TOOL_CLOSE
CALL = '{"name":"Read","arguments":{"file_path":"D:/a.json"}}'
ARGS = {"file_path": "D:/a.json"}
RAMBLE = ('\n' + OPEN + '\n{"name":"Skill","arguments":{}}\n'
          '{"name":"ScheduleWakeup","arguments":{"delay":1}}\n' * 3)

fails = []


def check(label, ok, detail=""):
    print("%s  %s%s" % ("ok  " if ok else "FAIL", label,
                        "" if ok else "  " + detail))
    if not ok:
        fails.append(label)


def stream(text, chunks=None, stop=()):
    """Feed `text` to StreamWriter; return (blocks, stop_reason, over_calls)."""
    over = []
    w = m.StreamWriter(None, "test-model", 256, list(stop), lambda b: None,
                       on_over=lambda: over.append(1))
    for c in (chunks if chunks is not None else [text]):
        w.feed(c)
    w.finish("length", 99)
    return w.blocks, w.stop_reason, len(over)


def cuts(s, n, rng):
    pts = sorted(rng.sample(range(1, len(s)), min(n, len(s) - 1)))
    out, prev = [], 0
    for p in pts + [len(s)]:
        out.append(s[prev:p])
        prev = p
    return out


def kinds(blocks):
    return [b["type"] for b in blocks]


def args_of(blocks):
    return [b["input"] for b in blocks if b["type"] == "tool_use"]


def names_of(blocks):
    return [b["name"] for b in blocks if b["type"] == "tool_use"]


def same(a, b):
    """Block equality ignoring the random tool_use ids."""
    strip = lambda bs: [{k: v for k, v in x.items() if k != "id"} for x in bs]
    return strip(a) == strip(b)


# ---------------------------------------------------------------- verdicts --
for tail, want in [
    ("", "wait"), ("\n \t", "wait"), ("\n</tool", "wait"), ("\n<tool", "wait"),
    ("<", "wait"), ("</", "wait"), ("\n" + CLOSE, "close"),
    (CLOSE + "trailing", "close"), ("\n" + OPEN, "other"),
    ("rambling", "other"), ("<x", "other"), ("</tool_callx", "other"),
]:
    got = m._after_json(tail)
    check("_after_json %-22r -> %s" % (tail, want), got == want, "got " + got)

# ------------------------------------------------------------ parse_output --
CASES = [
    ("well formed + trailing text", "hello\n" + OPEN + "\n" + CALL + "\n" + CLOSE + "\nbye",
     ["text", "tool_use", "text"], [ARGS], ["Read"], False),
    ("balanced, marker skipped, then rambling", OPEN + "\n" + CALL + RAMBLE,
     ["tool_use"], [ARGS], ["Read"], True),
    ("balanced, marker skipped, EOF", OPEN + "\n" + CALL,
     ["tool_use"], [ARGS], ["Read"], False),
    ("balanced, marker skipped, trailing whitespace", OPEN + "\n" + CALL + "\n \n",
     ["tool_use"], [ARGS], ["Read"], False),
    ("two well formed calls", OPEN + CALL + CLOSE + "\n" + OPEN + CALL + CLOSE,
     ["tool_use", "text", "tool_use"], [ARGS, ARGS], ["Read", "Read"], False),
    ("missing final brace, marker present",
     OPEN + '{"name":"Read","arguments":{"file_path":"D:/a.json"' + CLOSE,
     ["tool_use"], [ARGS], ["Read"], False),
    ("cut mid-string, marker present",
     OPEN + '{"name":"Read","arguments":{"file_path":"D:/a' + CLOSE,
     ["tool_use"], None, ["Read"], False),
    ("brace inside a string argument",
     OPEN + '{"name":"Grep","arguments":{"pattern":"a{b}c"}}' + CLOSE,
     ["tool_use"], [{"pattern": "a{b}c"}], ["Grep"], False),
    ("escaped quote inside a string argument",
     OPEN + '{"name":"R","arguments":{"s":"a\\"b{c"}}' + CLOSE,
     ["tool_use"], [{"s": 'a"b{c'}], ["R"], False),
    ("text before an unterminated call", "sure.\n" + OPEN + CALL + RAMBLE,
     ["text", "tool_use"], [ARGS], ["Read"], True),
]

rng = random.Random(20261115)
for label, text, want_kinds, want_args, want_names, want_over in CASES:
    blocks = m.parse_output(text)
    ok = kinds(blocks) == want_kinds and names_of(blocks) == want_names
    if ok and want_args is not None:
        ok = args_of(blocks) == want_args
    check("parse_output: " + label, ok,
          "got %s %s" % (kinds(blocks), json.dumps(args_of(blocks))[:120]))

    # The streaming path must produce the same blocks at any chunking, and must
    # ask the backend to stop exactly when the closing marker was skipped.
    chunkings = [("one shot", [text]), ("char by char", list(text))]
    chunkings += [("random %d cuts" % n, cuts(text, n, rng)) for n in (2, 5, 13)]
    for cl, ch in chunkings:
        sb, reason, over = stream(text, ch)
        ok = same(sb, blocks) and over == (1 if want_over else 0)
        if want_over:
            ok = ok and reason == "tool_use"
        check("stream %-14s %s" % (cl, label), ok,
              "blocks=%s over=%d reason=%s" % (kinds(sb), over, reason))

# A well formed call must never trigger the abort, even when the marker arrives
# one character at a time -- that is the false positive that would truncate
# every normal reply.
_, reason, over = stream("x" + OPEN + CALL + CLOSE + "y", list("x" + OPEN + CALL + CLOSE + "y"))
check("no abort on a well formed call", over == 0, "over=%d" % over)

# Stop sequences still cut text, and still do not fire the tool abort.
sb, reason, over = stream("alpha STOPHERE beta", ["alpha STOP", "HERE beta"], stop=["STOPHERE"])
check("stop sequence honoured", reason == "stop_sequence" and over == 0
      and "".join(b.get("text", "") for b in sb) == "alpha ",
      "reason=%s blocks=%s" % (reason, sb))

# The real degeneration: a correct first call followed by hallucinated ones.
real = (OPEN + '\n{"name": "Read", "arguments": {"file_path": '
        '"D:/third_party/claude-config/settings.json"}}\n' + RAMBLE)
blocks = m.parse_output(real)
check("real degenerate shape recovers the first call",
      kinds(blocks) == ["tool_use"] and blocks[0]["name"] == "Read"
      and blocks[0]["input"] == {"file_path": "D:/third_party/claude-config/settings.json"},
      json.dumps(blocks)[:200])

# A call cut off by the output cap must not be delivered as a tool_use.
# _parse_tool_call closes the JSON, so the client would receive an unusable
# argument blob, report "Invalid tool parameters", and the model would
# regenerate into the same cap -- an endless loop. Report max_tokens instead.
def finish_cut(text, reason, stopped=False):
    w = m.StreamWriter(None, "test-model", 16, [], lambda b: None)
    w.feed(text)
    if stopped:
        w.stopped = True
    w.finish(reason, 16)
    return w.blocks, w.stop_reason


cut_mid = OPEN + '\n{"name": "Bash", "arguments": {"command": "ec'
bl, rs = finish_cut(cut_mid, "length")
check("truncated call reports max_tokens", rs == "max_tokens", rs)
check("truncated call emits no tool_use", kinds(bl) == [], bl)

# A call that completed before the cut is still usable: the client can run it,
# and the truncation applies to whatever followed it.
bl, rs = finish_cut(OPEN + "\n" + CALL + "\n" + CLOSE + "then more text", "length")
check("completed call survives a later cut",
      kinds(bl) == ["tool_use", "text"] and rs == "tool_use",
      "kinds=%s reason=%s" % (kinds(bl), rs))

# An unclosed call that ended for a non-length reason (client STOP) is kept:
# that is the "stop decoding once a tool call is usable" path, not a cap.
bl, rs = finish_cut(OPEN + "\n" + CALL, "stop")
check("unclosed call kept when not truncated",
      kinds(bl) == ["tool_use"] and rs == "tool_use",
      "kinds=%s reason=%s" % (kinds(bl), rs))

print("\n%d failure(s)" % len(fails))
sys.exit(1 if fails else 0)