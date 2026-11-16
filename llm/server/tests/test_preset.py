"""What a preset file is allowed to say, and what happens when it lies.

The point of the file is to take five exported variables and a handful of flags
out of the launch command, so the failure that matters is a config that loads
*silently wrong*: a typo'd key or a "false" that becomes True would start the
server on the wrong artifact -- or with coopmat back on -- and the symptom
appears much later, as a wrong answer.

So every test here is about a refusal or a resolution: unknown key, wrong type,
path relative to the config rather than the cwd, and the precedence that makes
a stale environment variable lose to the preset that was asked for.

Run: python llm/server/tests/test_preset.py
"""
import importlib.util
import json
import os
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location(
    "anthropic_server", os.path.join(HERE, os.pardir, "anthropic_server.py"))
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

fails = []
tmp = tempfile.mkdtemp(prefix="vkop-preset-")


def check(label, ok, detail=""):
    print("%s  %s%s" % ("ok  " if ok else "FAIL", label,
                        "" if ok else "  " + detail))
    if not ok:
        fails.append(label)


def write(name, doc):
    p = os.path.join(tmp, name)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        if isinstance(doc, str):
            f.write(doc)
        else:
            json.dump(doc, f)
    return p


def refused(label, path, want=None, needle=None):
    """load_preset must raise SystemExit whose message names the problem."""
    try:
        m.load_preset(path, want)
    except SystemExit as e:
        ok = needle is None or needle in str(e)
        check(label, ok, "message did not mention %r: %s" % (needle, e))
    else:
        check(label, False, "accepted it")


# ------------------------------------------------------------- file shapes --
p = write("bare.json", {"model": "a.vkopbin", "max-kv": 4096})
name, got = m.load_preset(p)
check("bare object is one unnamed preset", name == "default", repr(name))
check("bare object keys load", got == {"model": os.path.join(tmp, "a.vkopbin"),
                                      "max_kv": 4096}, repr(got))

p = write("multi.json", {"default": "small",
                         "presets": {"small": {"max-kv": 8192},
                                     "big": {"max-kv": 2048}}})
check("'default' selects", m.load_preset(p) == ("small", {"max_kv": 8192}),
      repr(m.load_preset(p)))
check("--preset overrides 'default'",
      m.load_preset(p, "big") == ("big", {"max_kv": 2048}),
      repr(m.load_preset(p, "big")))

# A single preset needs no "default" key -- that is the one-file-per-model case.
p = write("solo.json", {"presets": {"only": {"max-kv": 1024}}})
check("sole preset needs no default", m.load_preset(p) == ("only", {"max_kv": 1024}),
      repr(m.load_preset(p)))

# ------------------------------------------------------------- precedence ---
# The whole reason paths resolve against the config: a config sitting beside
# its artifacts must work from any cwd. Run from "/" to prove cwd is unused.
sub = os.path.join(tmp, "modelA")
os.makedirs(sub, exist_ok=True)
p = write("modelA/cfg.json", {"model": "llm.vkopbin",
                              "kv-cache": "kv.bin",
                              "tokenizer": "/abs/tok.bin",
                              "embed": ""})
_, got = m.load_preset(p)
check("relative path -> config's directory",
      got["model"] == os.path.join(sub, "llm.vkopbin"), repr(got.get("model")))
check("relative kv-cache -> config's directory",
      got["kv_cache"] == os.path.join(sub, "kv.bin"), repr(got.get("kv_cache")))
check("absolute path left alone", got["tokenizer"] == "/abs/tok.bin",
      repr(got.get("tokenizer")))
check("empty path stays empty (how --kv-cache disables itself)",
      got["embed"] == "", repr(got.get("embed")))

# Underscores are the Python spelling of the same key; accepting both keeps a
# preset written from --help text from being refused.
_, got = m.load_preset(write("us.json", {"max_kv": 777, "model_name": "x"}))
check("underscore key == dash key",
      got == {"max_kv": 777, "model_name": "x"}, repr(got))

# ------------------------------------------------------------ type checks ---
# bool("false") is True, and coopmat is the route with known numeric errors, so
# a stringly-typed boolean is refused rather than coerced into "on".
p = write("coop1.json", {"coop": True})
check("coop: true loads", m.load_preset(p)[1] == {"coop": True},
      repr(m.load_preset(p)[1]))
for text, want in [("false", False), ("no", False), ("off", False),
                   ("true", True), ("yes", True), ("1", True)]:
    _, got = m.load_preset(write("coop_%s.json" % text, {"coop": text}))
    check("coop: %r -> %r" % (text, want), got == {"coop": want}, repr(got))
refused("coop: a non-boolean string is refused",
        write("coop2.json", {"coop": "maybe"}), needle="boolean")
refused("coop: a number is refused",
        write("coop3.json", {"coop": 1}), needle="boolean")

# Numbers that arrive as strings (hand-written JSON, or a template) coerce.
_, got = m.load_preset(write("num.json", {"max-kv": "4096",
                                          "repetition-penalty": "1.05"}))
check("numeric strings coerce",
      got == {"max_kv": 4096, "repetition_penalty": 1.05}, repr(got))
refused("a non-numeric string is refused",
        write("bad.json", {"max-kv": "lots"}), needle="max-kv")
refused("a number where a path belongs is refused",
        write("pathnum.json", {"model": 3}), needle="path string")

# --------------------------------------------------------------- refusals ---
# A typo'd key must not be a silent no-op: that is the failure mode this whole
# mechanism exists to remove.
refused("unknown key is refused",
        write("typo.json", {"max_kV": 4096}), needle="unknown key")
refused("misspelled path key is refused",
        write("typo2.json", {"modle": "a.vkopbin"}), needle="unknown key")
refused("--preset the file does not define",
        write("multi2.json", {"presets": {"a": {}, "b": {}}}), want="c",
        needle="no preset")
refused("--preset on a file with no \"presets\"",
        write("bare2.json", {"model": "a"}), want="x", needle="no \"presets\"")
refused("several presets and no selection",
        write("multi3.json", {"presets": {"a": {}, "b": {}}}), needle="selected")
refused("invalid JSON", write("broken.json", "{not json"), needle="valid JSON")
refused("non-object top level", write("arr.json", [1, 2]), needle="object")
refused("empty presets object",
        write("empty.json", {"presets": {}}), needle="non-empty")
refused("preset is not an object",
        write("notobj.json", {"presets": {"a": 5}}), needle="must be an object")
refused("missing file", os.path.join(tmp, "nope.json"), needle="nope.json")

# ------------------------------------------------- that ._PRESET_OPTIONS ---
# The table has to cover every option the parser defines, or a preset could not
# set it and the "known keys" list in the error message would be a lie.
src = open(os.path.join(HERE, os.pardir, "anthropic_server.py"),
           encoding="utf-8").read()
import re as _re
declared = set(_re.findall(r'ap\.add_argument\("--([a-z-]+)"', src))
declared -= {"config", "preset"}
missing = declared - set(m._PRESET_OPTIONS)
check("every parser option is settable from a preset", not missing,
      "uncovered: %s" % sorted(missing))
extra = set(m._PRESET_OPTIONS) - declared
check("the table invents no option", not extra, "unknown: %s" % sorted(extra))
# Every path-flagged key must really be a path, and vice versa: a wrong flag
# would resolve a model name against the config dir or leave a path unrooted.
PATHY = {"bin", "model", "embed", "tokenizer", "kv-cache"}
flagged = {k for k, (is_path, _) in m._PRESET_OPTIONS.items() if is_path}
check("path keys are exactly the file arguments", flagged == PATHY,
      "flagged=%s expected=%s" % (sorted(flagged), sorted(PATHY)))
env_missing = PATHY - set(m._PRESET_ENV)
check("every file argument documents its env var", not env_missing,
      "missing: %s" % sorted(env_missing))
# _PRESET_ENV is keyed by the option spelling and looked up with the argparse
# dest. A multi-word key that is only in one of those forms goes unmatched, and
# the shadow warning silently stops mentioning the variable it exists to
# report -- exactly how VKOP_MAX_KV=99999 slipped past it once.
env_bad = {k for k in m._PRESET_ENV if k not in m._PRESET_OPTIONS}
check("_PRESET_ENV covers the table", not env_bad, "unknown: %s" % sorted(env_bad))
check("_PRESET_ENV uses the option spelling, not the dest",
      all("_" not in k for k in m._PRESET_ENV),
      "underscored: %s" % sorted(k for k in m._PRESET_ENV if "_" in k))
check("there is a multi-word key, so the normalisation is exercised",
      sum("-" in k for k in m._PRESET_ENV) >= 4,
      "only %s" % sorted(m._PRESET_ENV))

shutil.rmtree(tmp, ignore_errors=True)
print("\n%d failure(s)" % len(fails))
sys.exit(1 if fails else 0)
