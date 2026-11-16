"""The system prompt must not change when the client's build changes.

Claude Code sends its build identity as the first system block:

    x-anthropic-billing-header: cc_version=2.1.325.a12; cc_entrypoint=sdk-cli;

It tokenizes to ~22 tokens and the whole 11k-token agent prompt follows it.
The KV prefix cache keys on the rendered id stream, so a build bump rewrites
token 22 and every row after it stops matching -- measured as a full 45 s cold
prefill for a 5-character difference the model never answers from.

_stabilise_telemetry replaces those values with a placeholder. The tests below
pin both halves: the values must disappear, and nothing else may.

Run: python llm/server/tests/test_telemetry.py
"""
import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location(
    "anthropic_server", os.path.join(HERE, os.pardir, "anthropic_server.py"))
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

fails = []


def check(name, cond, detail=""):
    if cond:
        print("ok   ", name)
    else:
        print("FAIL ", name, detail)
        fails.append(name)


A12 = "x-anthropic-billing-header: cc_version=2.1.325.a12; cc_entrypoint=sdk-cli;"
AEE = "x-anthropic-billing-header: cc_version=2.1.325.aee; cc_entrypoint=sdk-cli;"


def test_build_bump_normalises_to_the_same_text():
    """The whole point: two builds must render byte-identically."""
    a, ca = m._stabilise_telemetry(A12)
    b, cb = m._stabilise_telemetry(AEE)
    check("a12 was changed", ca is True, ca)
    check("aee was changed", cb is True, cb)
    check("both builds render identically", a == b, (a, b))
    check("no version string survives", "2.1.325" not in a, a)


def test_placeholder_keeps_the_line_shape():
    """Same keys, same order, same separators -- only the values are gone.
    A rewrite that dropped the whole line would change the token count and
    still invalidate the prefix."""
    out, _ = m._stabilise_telemetry(A12)
    check("header name kept",
          out.startswith("x-anthropic-billing-header:"), out)
    check("cc_version key kept", "cc_version=" in out, out)
    check("cc_entrypoint key kept", "cc_entrypoint=" in out, out)
    check("separators kept", out.count(";") == A12.count(";"), out)


def test_ordinary_text_is_untouched():
    prose = "You are a Claude agent, built on Anthropic's Claude Agent SDK."
    out, changed = m._stabilise_telemetry(prose)
    check("prose untouched", changed is False and out == prose, out)


def test_a_mention_of_the_header_in_prose_is_untouched():
    """The line must start the line. Prose that happens to quote the header --
    a model asked about its own telemetry, say -- is content, not metadata."""
    prose = ("The line x-anthropic-billing-header: cc_version=1.0 appears "
             "in the log.")
    out, changed = m._stabilise_telemetry(prose)
    check("quoted header untouched", changed is False and out == prose, out)


def test_unknown_x_anthropic_line_is_untouched():
    """Only the known volatile keys are rewritten. A future x-anthropic- line
    carrying real instructions must not be silently mangled."""
    line = "x-anthropic-something: other=1;"
    out, changed = m._stabilise_telemetry(line)
    check("unknown line untouched", changed is False and out == line, out)


def test_multiline_text_keeps_its_other_lines():
    text = "first line\n" + A12 + "\nlast line"
    out, changed = m._stabilise_telemetry(text)
    lines = out.splitlines()
    check("changed", changed is True, changed)
    check("line count preserved", len(lines) == 3, lines)
    check("first line intact", lines[0] == "first line", lines[0])
    check("last line intact", lines[2] == "last line", lines[2])


def test_trailing_newline_survives():
    """splitlines(keepends) round-trips the ending; losing it would move every
    following token."""
    for ending in ("\n", "\r\n", ""):
        text = A12 + ending
        out, _ = m._stabilise_telemetry(text)
        check(f"ending {ending!r} preserved", out.endswith(ending), repr(out[-4:]))


def test_no_header_means_no_work():
    """The fast path: most requests have no such line, and the function must
    not rebuild the string for them."""
    text = "hello\nworld\n"
    out, changed = m._stabilise_telemetry(text)
    check("unchanged", changed is False, changed)
    check("identical object", out == text, out)


for fn in [test_build_bump_normalises_to_the_same_text,
           test_placeholder_keeps_the_line_shape,
           test_ordinary_text_is_untouched,
           test_a_mention_of_the_header_in_prose_is_untouched,
           test_unknown_x_anthropic_line_is_untouched,
           test_multiline_text_keeps_its_other_lines,
           test_trailing_newline_survives,
           test_no_header_means_no_work]:
    fn()

print(f"\n{len(fails)} failure(s)")
sys.exit(1 if fails else 0)
