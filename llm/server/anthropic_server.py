#!/usr/bin/env python3
"""Anthropic Messages API server backed by vkop's llm_serve.

Claude Code points at this with ANTHROPIC_BASE_URL; the request travels
  HTTP (this file)  ->  line protocol  ->  llm_serve (vkop on Vulkan)
and only the HTTP/JSON/SSE half lives here.

What this layer owns
-------------------
* the Anthropic request/response schema and SSE event framing,
* flattening Anthropic content blocks into the ChatML text the model was
  trained to see (tool_use -> <tool_call>, tool_result -> <tool_response>,
  tool definitions -> the Qwen "# Tools" preamble),
* parsing <tool_call> out of the generated stream back into tool_use blocks
  with stop_reason="tool_use".

What llm_serve owns
-------------------
Running the graph. It takes role/text pairs, renders them through the
tokenizer's chat template, and streams token deltas back.

Limits that are deliberate
--------------------------
* Greedy by default. temperature/top_k/top_p are forwarded when the client
  sends them; llm_serve samples from the GPU's top-64 logits and reads the
  full row only on the steps the candidates cannot settle exactly
  (Sampler::pick_topk in llm/exporter/driver/llm_common.hpp).
* Images are not accepted (the vision tower is a separate graph); an image
  block is reported as text so the model at least knows one was dropped.
* One request at a time: there is one GPU context and one KV cache.
"""
import argparse
import base64
import collections
import itertools
import json
import os
import re
import secrets
import subprocess
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

if os.name == "nt":
    import msvcrt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))

TOOL_OPEN = "<tool_call>"
TOOL_CLOSE = "</tool_call>"

# Client telemetry lines that ride inside the system prompt. Claude Code sends
# its build identity as the FIRST system block:
#
#   x-anthropic-billing-header: cc_version=2.1.325.a12; cc_entrypoint=sdk-cli;
#
# which tokenizes to about 22 tokens -- and then the whole 11k-token agent
# prompt follows it. The KV prefix cache is keyed on the rendered id stream, so
# a build change rewrites token 22 and every row after it stops matching: the
# client waits a full cold prefill (~45 s here) for a 5-character difference
# that the model is not supposed to answer from.
#
# Rewriting the value to a fixed placeholder keeps the strings and their
# position identical while removing the variation, so a build bump no longer
# invalidates the prefix. Only the VALUE changes -- the line still occupies the
# same slot, so a prompt whose telemetry differs is bit-identical to the cached
# one. This is a deliberate, narrow exception to "send the client's bytes
# verbatim": it must stay limited to lines that carry no instruction.
_TELEMETRY_LINE = re.compile(
    r"^(x-anthropic-[a-z-]+:)\s*(.*)$", re.IGNORECASE)
# The fields inside such a line that identify the client build rather than say
# anything to the model.
_TELEMETRY_VOLATILE = re.compile(
    r"\b(cc_version|cc_entrypoint)=([^;\s]*)", re.IGNORECASE)


def _stabilise_telemetry(text):
    """Replace the varying values in client telemetry lines with a placeholder.

    Returns (text, changed). A line is only touched when it both starts with
    x-anthropic- (so it is client metadata, not prose that happens to mention
    a version) and carries one of the known volatile keys.
    """
    if not text or "x-anthropic-" not in text.lower():
        return text, False
    changed = False
    out = []
    for line in text.splitlines(keepends=True):
        stripped = line.rstrip("\r\n")
        ending = line[len(stripped):]
        if _TELEMETRY_LINE.match(stripped) and _TELEMETRY_VOLATILE.search(stripped):
            new = _TELEMETRY_VOLATILE.sub(lambda m: m.group(1) + "=<v>", stripped)
            if new != stripped:
                changed = True
            out.append(new + ending)
        else:
            out.append(line)
    return ("".join(out), changed)

# VKOP_SERVE_DUMP_DIR writes every request body verbatim. A client's real
# prompt is the only trustworthy input for reproducing a tool-loop failure:
# a hand-built approximation of it decodes differently.
DUMP_DIR = os.environ.get("VKOP_SERVE_DUMP_DIR", "")
_dump_seq = itertools.count(1)


def _dump_request(path, raw):
    if not DUMP_DIR:
        return
    name = "%03d_%s.json" % (next(_dump_seq),
                             path.strip("/").replace("/", "_") or "root")
    try:
        os.makedirs(DUMP_DIR, exist_ok=True)
        with open(os.path.join(DUMP_DIR, name), "wb") as f:
            f.write(raw)
    except OSError as e:
        sys.stderr.write(f"[serve] request dump failed: {e}\n")


# --------------------------------------------------------------------------
# Backend: one llm_serve process, one request at a time.
# --------------------------------------------------------------------------
class Backend:
    def __init__(self, argv, env=None, max_kv=8192):
        self.max_kv = max_kv
        self.argv = argv
        self.env = env
        self.abi = 1
        self._lock = threading.Lock()
        self._spawn()

    def _spawn(self):
        r_in, w_in = os.pipe()
        r_out, w_out = os.pipe()
        child_env = dict(os.environ)
        if self.env:
            child_env.update(self.env)
        if os.name == "nt":
            # pass_fds is POSIX-only. Hand over the HANDLE values instead and
            # let llm_serve wrap them with _open_osfhandle; close_fds=False is
            # what makes Windows inherit the handles marked below at all.
            os.set_inheritable(r_in, True)
            os.set_inheritable(w_out, True)
            rpc = ["--rpc-in", str(msvcrt.get_osfhandle(r_in)),
                   "--rpc-out", str(msvcrt.get_osfhandle(w_out))]
            extra = {"close_fds": False}
        else:
            rpc = ["--rpc-in", str(r_in), "--rpc-out", str(w_out)]
            extra = {"pass_fds": (r_in, w_out)}
        self.proc = subprocess.Popen(
            self.argv + rpc,
            env=child_env,
            stdout=sys.stderr,
            stderr=sys.stderr,
            **extra,
        )
        os.close(r_in)
        os.close(w_out)
        self._w = os.fdopen(w_in, "wb", buffering=0)
        self._r = os.fdopen(r_out, "rb")

    def _ensure_alive(self):
        """Relaunch a backend that exited on its own.

        llm_serve quits when the driver kills the GPU mid-request
        (VK_ERROR_DEVICE_LOST): that VkDevice cannot be reused, so a dead
        child is the honest outcome -- but the server must not stay dead.
        Called under the lock, so exactly one request pays the reload.
        """
        if self.proc.poll() is None:
            return
        code = self.proc.returncode
        for stream in (self._w, self._r):
            try:
                stream.close()
            except OSError:
                pass
        print(f"[serve] backend exited with {code}; relaunching and "
              "reloading the model", file=sys.stderr)
        self._spawn()
        self.wait_ready()

    def _send(self, line):
        self._w.write((line + "\n").encode())

    def _recv(self):
        raw = self._r.readline()
        if not raw:
            raise RuntimeError("llm_serve closed the pipe")
        return raw.decode("utf-8", "replace").rstrip("\n")

    def wait_ready(self, timeout=180.0):
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(f"llm_serve exited with {self.proc.returncode}")
            line = self._recv()
            kind, _, rest = line.partition(" ")
            if kind == "READY":
                self.abi = int(rest or 1)
                return
            if kind == "FATAL":
                raise RuntimeError(base64.b64decode(rest).decode("utf-8", "replace"))
        raise RuntimeError("llm_serve did not become ready")

    @staticmethod
    def _write_msgs(send, messages):
        for role, text in messages:
            send(role)
            send(base64.b64encode(text.encode()).decode())

    def count(self, messages, rid="0"):
        with self._lock:
            self._ensure_alive()
            self._send(f"COUNT {rid} {len(messages)}")
            self._write_msgs(self._send, messages)
            while True:
                line = self._recv()
                kind, _, rest = line.partition(" ")
                if kind == "COUNTED":
                    # COUNTED <id> <n>
                    return int(rest.split()[-1])
                if kind == "ERROR":
                    raise RuntimeError(_decode_err(rest))

    def generate(self, messages, max_new, rid="0", sample=None):
        """Yield ('tokens_in', n), ('delta', text), ('done', reason, n_out)."""
        with self._lock:
            self._ensure_alive()
            head = f"GEN {rid} {max_new} {len(messages)}"
            if sample is not None:
                head += " %g %d %g" % sample
            self._send(head)
            self._write_msgs(self._send, messages)
            while True:
                line = self._recv()
                kind, _, rest = line.partition(" ")
                if kind == "DELTA":
                    _, b64 = rest.split(" ", 1)
                    yield ("delta", base64.b64decode(b64).decode("utf-8", "replace"))
                elif kind == "TOKENS_IN":
                    yield ("tokens_in", int(rest.split()[-1]))
                elif kind == "DONE":
                    parts = rest.split()
                    yield ("done", parts[1], int(parts[2]))
                    return
                elif kind == "ERROR":
                    raise RuntimeError(_decode_err(rest))

    def stop(self, rid="0"):
        """End the round in flight.

        Deliberately takes no lock: the caller is the thread already inside
        generate(), suspended at a yield. The backend reads this between decode
        steps, so the reply stops within one token instead of running to the
        output cap. An abi-2 backend has no STOP: it would read the line as its
        next command and desynchronise, so there the round just runs to its cap.
        """
        if self.abi >= 3:
            self._send(f"STOP {rid}")

    def close(self):
        try:
            self._send("BYE")
        except OSError:
            pass
        try:
            self.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.proc.kill()


def _decode_err(rest):
    parts = rest.split(" ", 1)
    payload = parts[1] if len(parts) > 1 else parts[0]
    try:
        return base64.b64decode(payload).decode("utf-8", "replace")
    except Exception:
        return payload


# --------------------------------------------------------------------------
# Anthropic request -> ChatML role/text pairs
# --------------------------------------------------------------------------
def _text_of(content):
    """Flatten Anthropic `content` (string or block list) to plain text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    out = []
    for block in content:
        if not isinstance(block, dict):
            continue
        t = block.get("type")
        if t == "text":
            out.append(block.get("text", ""))
        elif t == "image":
            out.append("[image omitted: this backend has no vision tower]")
        elif t == "tool_result":
            out.append(_tool_result_text(block))
    return "".join(out)


def _tool_result_text(block):
    inner = block.get("content")
    if inner is None:
        body = ""
    elif isinstance(inner, str):
        body = inner
    else:
        body = "".join(b.get("text", "") for b in inner
                       if isinstance(b, dict) and b.get("type") == "text")
    return f"<tool_response>\n{body}\n</tool_response>"


def _assistant_text(message):
    """Render an assistant turn the way the Qwen template would: the text
    blocks concatenated, then one <tool_call> per tool_use, newline-separated
    exactly as the ChatML template does."""
    content = message.get("content")
    text = ""
    tool_calls = []
    if isinstance(content, str):
        text = content
    else:
        for block in content or []:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "text":
                text += block.get("text", "")
            elif block.get("type") == "tool_use":
                tool_calls.append(block)
    parts = [text] if text else []
    for i, tc in enumerate(tool_calls):
        if (i == 0 and text) or i > 0:
            parts.append("\n")
        args = tc.get("input", {})
        if not isinstance(args, str):
            args = json.dumps(args, ensure_ascii=False)
        parts.append(f'{TOOL_OPEN}\n{{"name": "{tc.get("name", "")}", '
                     f'"arguments": {args}}}\n{TOOL_CLOSE}')
    return "".join(parts)


def _tools_preamble(tools):
    """Qwen3-VL's chat template turns `tools` into this system preamble. The
    schemas are re-shaped to the OpenAI function form the model was trained on
    (Anthropic sends {name, description, input_schema})."""
    if not tools:
        return ""
    rendered = []
    for t in tools:
        rendered.append({
            "type": "function",
            "function": {
                "name": t.get("name", ""),
                "description": t.get("description", ""),
                "parameters": t.get("input_schema", {"type": "object"}),
            },
        })
    body = "\n".join(json.dumps(r, ensure_ascii=False) for r in rendered)
    return (
        "# Tools\n\nYou may call one or more functions to assist with the user "
        "query.\n\nYou are provided with function signatures within "
        "<tools></tools> XML tags:\n<tools>\n"
        + body
        + "\n</tools>\n\nFor each function call, return a json object with "
        "function name and arguments within <tool_call></tool_call> XML tags:\n"
        "<tool_call>\n{\"name\": <function-name>, \"arguments\": "
        "<args-json-object>}\n</tool_call>"
    )


def _sampling(req):
    """Anthropic sampling params -> the GEN line's three trailing fields.

    -1 means the client never sent the parameter, so the backend default
    (greedy) stands. An absent temperature must not silently enable sampling.
    """
    def num(key, cast):
        v = req.get(key)
        if v is None:
            return -1
        try:
            return cast(v)
        except (TypeError, ValueError):
            return -1
    return (num("temperature", float), num("top_k", int), num("top_p", float))


def to_backend_messages(req):
    """Anthropic request -> [(role, text)] with the tool markup folded in.

    Consecutive tool_result blocks become one user turn, matching the template
    (it opens <|im_start|>user for the first of a run and closes it after the
    last)."""
    msgs_in = req.get("messages", [])
    # Claude Code's SDK sends most of its instructions as a `system` message
    # *inside* messages, after the user turn. ChatML has one system turn at the
    # top, so it has to be hoisted: left in place it renders as a second user
    # turn and the environment/slash-command wall lands between the question
    # and the answer. The tool preamble stays last, where the template puts it.
    hoisted = [_text_of(m.get("content")) for m in msgs_in
               if m.get("role") == "system"]
    parts = [_text_of(req.get("system"))] + hoisted
    parts.append(_tools_preamble(req.get("tools")))
    # Client build telemetry rides in the system text and changes on every
    # Claude Code upgrade; left as-is it invalidates the KV prefix ~22 tokens
    # in and costs a full cold prefill. See _stabilise_telemetry.
    parts = [p for p in parts if p]
    for i, p in enumerate(parts):
        parts[i], _ = _stabilise_telemetry(p)
    system = "\n\n".join(parts)

    out = []
    if system:
        out.append(("system", system))

    pending_tool_results = []

    def flush_tool_results():
        if pending_tool_results:
            out.append(("user", "\n".join(pending_tool_results)))
            pending_tool_results.clear()

    for msg in msgs_in:
        role = msg.get("role")
        if role == "system":
            continue
        if role == "tool":
            # Not an Anthropic role; tolerate OpenAI-shaped histories.
            content = msg.get("content")
            pending_tool_results.append(
                f"<tool_response>\n{_text_of(content)}\n</tool_response>")
            continue
        if role == "assistant":
            flush_tool_results()
            out.append(("assistant", _assistant_text(msg)))
            continue
        # user (and anything unrecognised)
        content = msg.get("content")
        if isinstance(content, list) and content and all(
                isinstance(b, dict) and b.get("type") == "tool_result"
                for b in content):
            for block in content:
                pending_tool_results.append(_tool_result_text(block))
            continue
        flush_tool_results()
        out.append(("user", _text_of(content)))
    flush_tool_results()
    return out


def trim_to_window(msgs, max_tokens, budget, count=None):
    """Drop the oldest conversation rounds until the whole thing fits.

    The system message (which carries the tool preamble) is never dropped; a
    `user` turn and the `assistant` turn after it are dropped as a pair, the
    same rule llm_chat uses, so the history always starts on a user turn.

    `count` is the backend's exact tokenizer count. With it the loop is exact;
    without it a character heuristic (~3 chars/token) is the only estimate
    available, and the backend's own clamp catches the rest.

    Returns (msgs, trimmed).
    """
    if budget <= 0 or len(msgs) <= 2:
        return msgs, False
    head = msgs[0] if msgs and msgs[0][0] == "system" else None
    body = msgs[1:] if head else msgs
    trimmed = False

    def fits():
        cur = ([head] if head else []) + body
        if count is not None:
            return count(cur) + max_tokens <= budget
        return sum(len(t) for _, t in body) // 3 + max_tokens <= budget

    while len(body) > 1 and not fits():
        if body[0][0] == "user" and len(body) > 1 and body[1][0] == "assistant":
            body = body[2:]
        else:
            body = body[1:]
        trimmed = True
    return ([head] if head else []) + body, trimmed


# --------------------------------------------------------------------------
# Model output -> Anthropic content blocks
# --------------------------------------------------------------------------
def _tool_block(inner):
    name, args = _parse_tool_call(inner)
    return {
        "type": "tool_use",
        "id": "toolu_" + secrets.token_hex(12),
        "name": name,
        "input": args,
    }


def parse_output(text):
    """Split generated text into text / tool_use blocks.

    A call is delimited by its balanced JSON object or by the closing marker,
    whichever comes first: the model sometimes skips the marker and rambles on,
    and requiring it would turn a usable call into one unusable raw blob.
    """
    blocks = []
    rest = text
    while True:
        i = rest.find(TOOL_OPEN)
        if i < 0:
            if rest:
                blocks.append({"type": "text", "text": rest})
            break
        if i > 0:
            blocks.append({"type": "text", "text": rest[:i]})
        body = rest[i + len(TOOL_OPEN):]
        end = _json_end(body)
        j = body.find(TOOL_CLOSE)
        over = True
        if end is not None and (j < 0 or end <= j):
            inner = body[:end]
            tail = body[end:]
            if _after_json(tail) == "close":
                rest = tail.lstrip()[len(TOOL_CLOSE):]
                over = False
            else:
                rest = ""
        elif j >= 0:
            inner = body[:j]
            rest = body[j + len(TOOL_CLOSE):]
            over = False
        else:
            inner = body
            rest = ""
        blocks.append(_tool_block(inner))
        if over:
            break
    return blocks


def _loads_dict(text):
    if not text:
        return None
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def _scan_json(text):
    """Walk the first JSON value in `text` with string/escape state.

    Returns (end, closers, in_string): `end` is the index just past a balanced
    value or None if it never closes, `closers` the delimiters still missing,
    `in_string` whether the text ends inside a string literal. Braces inside a
    string never count -- a tool argument is routinely a path or a regex.
    """
    stack = []
    in_str = esc = False
    for k, ch in enumerate(text):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        elif ch == '"':
            in_str = True
        elif ch in "{[":
            stack.append("}" if ch == "{" else "]")
        elif ch in "}]":
            if stack and stack[-1] == ch:
                stack.pop()
                if not stack:
                    return k + 1, [], False
    return None, stack, in_str


def _json_end(text):
    """Index just past the first balanced JSON object in `text`, else None."""
    return _scan_json(text)[0]


def _after_json(tail):
    """What follows a balanced tool-call object: 'close', 'wait' or 'other'.

    'close' means the closing marker is there and the call is well formed.
    'wait' means only whitespace so far, or a prefix of either marker -- more
    tokens are needed to tell. 'other' means the model skipped its closing
    marker: whatever it generates next is degeneration, not reply content.
    """
    s = tail.lstrip()
    if s.startswith(TOOL_CLOSE):
        return "close"
    if s.startswith(TOOL_OPEN):
        return "other"
    if not s or any(m.startswith(s) for m in (TOOL_CLOSE, TOOL_OPEN)):
        return "wait"
    return "other"


def _close_json(text):
    """Append the closing delimiters a structurally truncated JSON value lacks.

    Qwen3-VL drops the final brace on some tool calls deterministically, and the
    raw fallback below then hands the client an argument object no tool schema
    accepts -- which the client retries forever. A value cut mid-string is left
    alone: closing it would invent content the model never wrote.
    """
    _, stack, in_str = _scan_json(text)
    if in_str or not stack:
        return None
    return text + "".join(reversed(stack))


def _parse_tool_call(inner):
    inner = inner.strip()
    obj = _loads_dict(inner)
    if obj is None:
        obj = _loads_dict(_close_json(inner))
    if obj is not None:
        name = obj.get("name", "")
        args = obj.get("arguments", {})
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {"raw": args}
        if not isinstance(args, dict):
            args = {"value": args}
        return name, args
    # Tolerate the tool name sitting outside the JSON, e.g. when the model
    # emits `{"name": "Bash", "arguments": {...}}` badly or a bare function
    # call. Recover the name if we can, and keep the rest as a raw argument so
    # the client sees a tool_use rather than silent text.
    m = re.search(r'"name"\s*:\s*"([^"]+)"', inner)
    name = m.group(1) if m else ""
    return name, {"raw": inner}


# --------------------------------------------------------------------------
# A streaming parser that emits SSE events as the model generates.
# --------------------------------------------------------------------------
class StreamWriter:
    """Consumes token deltas and writes Anthropic SSE events.

    Holds back the tail of the text buffer so a `<tool_call>` split across
    tokens is never emitted as text first, and honours stop sequences the same
    way. `blocks` accumulates the same content as Anthropic blocks, which is
    what the non-streaming path returns -- one parser, so the two paths cannot
    disagree about where a tool call ends.
    """

    def __init__(self, wfile, model, max_tokens, stop_sequences, write,
                 on_over=None):
        self.wfile = wfile
        self.model = model
        self.max_tokens = max_tokens
        self.stop_sequences = [s for s in (stop_sequences or []) if s]
        self.holdback = max([len(TOOL_OPEN) - 1] +
                            [max(0, len(s) - 1) for s in self.stop_sequences])
        self.buf = ""
        self.in_tool = False
        self.after_tool = False
        self.text_buf = ""
        self.index = 0
        self.text_open = False
        self.stopped = False
        self.stop_reason = None
        self.stop_sequence = None
        self.text_emitted = 0
        self.saw_tool = False
        self.blocks = []
        self._write = write
        self._on_over = on_over

    # -- SSE ---------------------------------------------------------------
    def event(self, name, payload):
        data = json.dumps(payload, ensure_ascii=False)
        self._write(f"event: {name}\ndata: {data}\n\n".encode())

    def message_start(self, input_tokens, msg_id):
        self.event("message_start", {
            "type": "message_start",
            "message": {
                "id": msg_id,
                "type": "message",
                "role": "assistant",
                "model": self.model,
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": input_tokens, "output_tokens": 0},
            },
        })

    # -- content blocks ----------------------------------------------------
    def _open_text(self):
        if not self.text_open:
            self.text_open = True
            self.event("content_block_start", {
                "type": "content_block_start",
                "index": self.index,
                "content_block": {"type": "text", "text": ""},
            })

    def _close_text(self):
        if self.text_open:
            self.event("content_block_stop",
                       {"type": "content_block_stop", "index": self.index})
            self.index += 1
            self.text_open = False

    def _emit_text(self, chunk):
        if not chunk or self.stopped:
            return
        self._open_text()
        self.text_emitted += len(chunk)
        if self.blocks and self.blocks[-1]["type"] == "text":
            self.blocks[-1]["text"] += chunk
        else:
            self.blocks.append({"type": "text", "text": chunk})
        self.event("content_block_delta", {
            "type": "content_block_delta",
            "index": self.index,
            "delta": {"type": "text_delta", "text": chunk},
        })

    def _emit_tool(self, inner):
        block = _tool_block(inner)
        self._close_text()
        idx = self.index
        self.blocks.append(block)
        self.event("content_block_start", {
            "type": "content_block_start",
            "index": idx,
            "content_block": {
                "type": "tool_use",
                "id": block["id"],
                "name": block["name"],
                "input": {},
            },
        })
        self.event("content_block_delta", {
            "type": "content_block_delta",
            "index": idx,
            "delta": {"type": "input_json_delta",
                      "partial_json": json.dumps(block["input"],
                                                 ensure_ascii=False)},
        })
        self.event("content_block_stop",
                   {"type": "content_block_stop", "index": idx})
        self.index += 1
        self.saw_tool = True

    # -- feeding -----------------------------------------------------------
    def feed(self, delta):
        if self.stopped:
            return
        self.buf += delta
        if self.after_tool:
            self._resolve_after_tool()
        else:
            self._pump()

    def _pump(self):
        while not self.stopped:
            if not self.in_tool:
                i = self.buf.find(TOOL_OPEN)
                if i < 0:
                    # Nothing to open yet: emit everything that cannot start a
                    # marker or a stop sequence. The sequence has to be searched
                    # for in the whole buffer -- cutting a fixed tail off and
                    # then looking only at the emitted part splits a sequence
                    # across the boundary, and neither half ever matches again.
                    k, hit = -1, None
                    for s in self.stop_sequences:
                        p = self.buf.find(s)
                        if p >= 0 and (k < 0 or p < k):
                            k, hit = p, s
                    if k >= 0:
                        self._emit_text(self.buf[:k])
                        self.stopped = True
                        self.stop_reason = "stop_sequence"
                        self.stop_sequence = hit
                        self.buf = ""
                        return
                    keep = self.holdback
                    if len(self.buf) > keep:
                        self._emit_text(self.buf[:len(self.buf) - keep])
                        self.buf = self.buf[len(self.buf) - keep:]
                    return
                self._emit_safe(self.buf[:i])
                self._close_text()
                self.buf = self.buf[i + len(TOOL_OPEN):]
                self.in_tool = True
            # Inside a call the only delimiters are its own balanced JSON and
            # the closing marker. Searching for another opening marker here
            # would emit this call's JSON as prose when the model skips the
            # closer and starts a second one.
            end = _json_end(self.buf)
            j = self.buf.find(TOOL_CLOSE)
            if end is not None and (j < 0 or end <= j):
                inner, self.buf = self.buf[:end], self.buf[end:]
                self.in_tool = False
                self.after_tool = True
                self._emit_tool(inner)
                self._resolve_after_tool()
                return
            if j >= 0:
                inner, self.buf = self.buf[:j], self.buf[j + len(TOOL_CLOSE):]
                self.in_tool = False
                self._emit_tool(inner)
                continue
            return

    def _resolve_after_tool(self):
        verdict = _after_json(self.buf)
        if verdict == "wait":
            return
        self.after_tool = False
        if verdict == "close":
            self.buf = self.buf.lstrip()[len(TOOL_CLOSE):]
            self._pump()
        else:
            self._end_reply()

    def _end_reply(self):
        """A tool call finished without its closing marker. The reply is over:
        tell the backend to stop instead of paying for the degeneration that
        follows, which runs to the output cap at hundreds of ms per token."""
        self.stopped = True
        self.stop_reason = "tool_use"
        self.buf = ""
        if self._on_over is not None:
            self._on_over()

    def _emit_safe(self, chunk):
        """Emit text, honouring stop sequences inside it."""
        if not chunk or self.stopped:
            return
        for s in self.stop_sequences:
            k = chunk.find(s)
            if k >= 0:
                self._emit_text(chunk[:k])
                self.stopped = True
                self.stop_reason = "stop_sequence"
                self.stop_sequence = s
                return
        self._emit_text(chunk)

    def finish(self, reason, output_tokens):
        # A tool call cut off by the output cap is not usable. _parse_tool_call
        # would close its JSON and hand the client an argument blob like
        # {"raw": "{\"name\": \"Bash\", \"comm..."} -- the client then reports
        # "Invalid tool parameters", feeds that back, and the model regenerates
        # into the same cap: a loop that never terminates on its own. Report
        # max_tokens instead, which tells the client why and lets it retry the
        # turn with more room (or not at all).
        truncated_in_call = self.in_tool and reason == "length"
        if not self.stopped:
            tail = self.buf
            self.buf = ""
            if self.in_tool:
                if not truncated_in_call:
                    # Generation ended inside a tool call for some other
                    # reason (client STOP, a stop token): the call is still
                    # complete enough to be worth delivering.
                    self._emit_tool(tail)
            elif not self.after_tool:
                self._emit_safe(tail)
        self._close_text()
        if self.stop_reason is None:
            if truncated_in_call:
                self.stop_reason = "max_tokens"
            elif getattr(self, "saw_tool", False):
                self.stop_reason = "tool_use"
            elif reason == "length":
                self.stop_reason = "max_tokens"
            else:
                self.stop_reason = "end_turn"
        self.event("message_delta", {
            "type": "message_delta",
            "delta": {"stop_reason": self.stop_reason,
                      "stop_sequence": self.stop_sequence},
            "usage": {"output_tokens": output_tokens},
        })
        self.event("message_stop", {"type": "message_stop"})
        return self.stop_reason


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
class Metrics:
    """Per-request latency and token accounting, queryable over HTTP.

    The [req] log line already carries these numbers, but a log is not a
    query surface: from a terminal you cannot ask "what was the median TTFT
    over the last 20 requests" without parsing it, and a client cannot see it
    at all. This keeps the same measurements in memory, in a bounded deque so
    a long-lived server cannot grow without limit.

    Two rates, and the distinction matters for a serving comparison:

    * TTFT -- request arrival to the first token. Contains the whole prefill,
      so it is the number that moves when the prompt gets longer or the KV
      prefix fails to hit.
    * TPOT -- time per output token, over the tokens AFTER the first one.
      The first token is prefill's product; including it would make TPOT a
      function of prompt length, which is exactly what TTFT already reports.
      This is the decode rate people mean when they compare hardware.

    Threading: requests are serialised by the backend lock, but the deque is
    touched from the handler thread and read from another, so it gets a lock
    of its own. Cheap -- a few appends per request.
    """

    def __init__(self, capacity=256):
        self._lock = threading.Lock()
        self._rows = collections.deque(maxlen=capacity)
        self._started = time.time()
        self._total = 0

    def record(self, msg_id, stream, t0, t_first, input_tokens, out_tokens,
               stop_reason, trimmed, error=None, t_end=None):
        """t_end is for tests: production passes None and the clock is read
        here, so callers cannot accidentally measure the wrong span."""
        t_end = time.time() if t_end is None else t_end
        wall = t_end - t0
        ttft = (t_first - t0) if t_first is not None else None
        n_out = out_tokens or 0
        # Tokens after the first: what TPOT averages over.
        n_dec = max(0, n_out - 1)
        dec_s = 0.0
        if ttft is not None and n_dec > 0:
            # Decode span is wall minus the prefill+first-token part. It can
            # round to <= 0 on a timer-granularity edge, hence the clamp.
            dec_s = max(0.0, wall - ttft)
        row = {
            "id": msg_id,
            "at": t_end,
            "stream": bool(stream),
            "input_tokens": input_tokens,
            "output_tokens": n_out,
            "wall_s": round(wall, 4),
            "ttft_s": None if ttft is None else round(ttft, 4),
            "tpot_s": round(dec_s / n_dec, 4) if n_dec > 0 else None,
            "decode_tokens": n_dec,
            # Unrounded, so the aggregate rate below is not built from
            # already-rounded per-request values.
            "_decode_s": dec_s,
            "stop_reason": stop_reason,
            "trimmed": bool(trimmed),
        }
        if error is not None:
            row["error"] = error
        with self._lock:
            self._rows.append(row)
            self._total += 1
        return row

    @staticmethod
    def _percentile(sorted_vals, q):
        """Nearest-rank percentile on an already-sorted list. q in [0, 100].

        Nearest-rank rather than interpolated: with a handful of samples a
        server operator wants an actual observed value, not a synthetic point
        between two of them, and it never reports a number better than the
        best or worse than the worst that really happened.
        """
        if not sorted_vals:
            return None
        k = max(0, min(len(sorted_vals) - 1,
                       int(round((q / 100.0) * (len(sorted_vals) - 1)))))
        return sorted_vals[k]

    def _summary(self, rows):
        def stats(key):
            vals = sorted(r[key] for r in rows if r[key] is not None)
            if not vals:
                return {"n": 0, "min": None, "p50": None, "p90": None,
                        "max": None, "mean": None}
            return {"n": len(vals),
                    "min": round(vals[0], 4),
                    "p50": round(self._percentile(vals, 50), 4),
                    "p90": round(self._percentile(vals, 90), 4),
                    "max": round(vals[-1], 4),
                    "mean": round(sum(vals) / len(vals), 4)}

        ok = [r for r in rows if "error" not in r]
        out = {
            "requests": len(rows),
            "errors": len(rows) - len(ok),
            "input_tokens": sum(r["input_tokens"] for r in rows),
            "output_tokens": sum(r["output_tokens"] for r in rows),
            "ttft_s": stats("ttft_s"),
            "tpot_s": stats("tpot_s"),
            "wall_s": stats("wall_s"),
        }
        # Decode tokens/second is what people quote against other engines.
        # Derived from summed tokens over summed decode time, so a long reply
        # counts more than a one-token one (per-request TPOTs would not).
        dec_tokens = sum(r["decode_tokens"] for r in ok)
        dec_s = sum(r["_decode_s"] for r in ok)
        out["decode_tokens"] = dec_tokens
        out["decode_tokens_per_s"] = (
            round(dec_tokens / dec_s, 2) if dec_s > 0 else None)
        return out

    def snapshot(self, limit=20):
        with self._lock:
            rows = list(self._rows)
            total = self._total
        recent = rows[-limit:] if limit and limit > 0 else rows
        # _decode_s is an aggregate input, not part of the public row shape.
        public_recent = [{k: v for k, v in r.items() if not k.startswith("_")}
                         for r in reversed(recent)]
        return {
            "uptime_s": round(time.time() - self._started, 1),
            "completed_since_start": total,
            "window": len(rows),
            "all": self._summary(rows),
            "recent": self._summary(recent),
            "recent_n": len(recent),
            "requests": public_recent,
        }


# --------------------------------------------------------------------------
# HTTP
# --------------------------------------------------------------------------
class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "vkop-anthropic/1"

    backend: Backend = None
    metrics: Metrics = None
    default_model = "vkop-qwen3-vl-2b"
    max_output = 1024

    def log_message(self, fmt, *args):
        sys.stderr.write("[http] %s\n" % (fmt % args))

    # -- helpers -----------------------------------------------------------
    def _json(self, code, payload):
        body = json.dumps(payload, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    # -- SSE ---------------------------------------------------------------
    # BaseHTTPRequestHandler speaks HTTP/1.1 here, and an event stream has no
    # Content-Length, so the reply must be chunked. Without it the client keeps
    # the connection open waiting for a body that never ends -- the stream
    # arrives but the request never completes.
    def _sse_start(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.send_header("Transfer-Encoding", "chunked")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()

    def _sse_write(self, data: bytes):
        self.wfile.write(b"%X\r\n" % len(data))
        self.wfile.write(data)
        self.wfile.write(b"\r\n")
        self.wfile.flush()

    def _sse_close(self):
        self.wfile.write(b"0\r\n\r\n")
        self.wfile.flush()

    def _error(self, code, err_type, message):
        self._json(code, {"type": "error",
                          "error": {"type": err_type, "message": message}})

    def _read_json(self):
        length = int(self.headers.get("Content-Length", "0") or 0)
        if length <= 0:
            return {}
        raw = self.rfile.read(length)
        _dump_request(self._path_only(), raw)
        return json.loads(raw.decode("utf-8"))

    # -- routes ------------------------------------------------------------
    # Claude Code appends query parameters (e.g. ?beta=true) to the path, and
    # probes /api/hello with HEAD before the first request. Both are routing
    # concerns, so normalise them here rather than in each handler.
    def _path_only(self):
        return self.path.split("?", 1)[0].rstrip("/") or "/"

    def do_HEAD(self):
        code = 200 if self._path_only() in ("/health", "/healthz", "/") else 404
        self.send_response(code)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self):
        path = self._path_only()
        if path in ("/health", "/healthz"):
            self._json(200, {"status": "ok"})
        elif path in ("/metrics", "/v1/metrics"):
            # GET /metrics?limit=N -- N most recent requests in "recent" and
            # in the per-request list; "all" always covers the whole window.
            # ?limit=0 makes "recent" the whole window too.
            limit = 20
            qs = self.path.split("?", 1)
            if len(qs) == 2:
                for part in qs[1].split("&"):
                    k, _, v = part.partition("=")
                    if k == "limit":
                        try:
                            limit = int(v)
                        except ValueError:
                            self._error(400, "invalid_request_error",
                                        f"limit must be an integer, got {v!r}")
                            return
            if self.metrics is None:
                self._error(503, "api_error", "metrics not initialised")
                return
            self._json(200, self.metrics.snapshot(limit))
        elif path.startswith("/v1/models"):
            self._json(200, {"data": [{"id": self.default_model,
                                       "type": "model"}],
                             "has_more": False})
        elif path == "/api/hello":
            # Not an Anthropic endpoint; probe traffic just needs "reachable".
            self._json(200, {"ok": True})
        else:
            self._error(404, "not_found_error", self.path)

    def do_POST(self):
        try:
            req = self._read_json()
        except (ValueError, UnicodeDecodeError) as e:
            self._error(400, "invalid_request_error", f"bad JSON body: {e}")
            return

        path = self._path_only()
        if path == "/v1/messages/count_tokens":
            self._count_tokens(req)
        elif path == "/v1/messages":
            self._messages(req)
        else:
            self._error(404, "not_found_error", self.path)

    def _prepare(self, req, max_tokens):
        """Request -> backend message list that fits the KV window.

        Returns (msgs, max_tokens, trimmed). The backend clamps max_tokens to
        the room it has, but a prompt that fills the window on its own has to
        be shortened here: unlike llm_chat there is no ring window, so the
        whole conversation has to fit at once.
        """
        msgs = to_backend_messages(req)
        # A client's max_tokens is a ceiling, not a target. Claude Code asks
        # for 32000 and treats a truncated reply as an error ("exceeded the
        # 32000 output token maximum"), so the safe shape is: honour a small
        # request exactly, but never let the window decide the answer -- cap at
        # --max-output, which is what this hardware can actually finish.
        if max_tokens > self.max_output:
            max_tokens = self.max_output
        budget = self.backend.max_kv - max_tokens
        trimmed = False
        if budget > 0:
            # The heuristic pass first: cheap, and it removes the obviously
            # oversized cases. Then an exact tokenizer count decides whether
            # anything is still over -- CJK, code, and special markers run
            # closer to 1.3 bytes/token, so 3 chars/token under-estimates
            # badly and would let an over-long prompt through.
            msgs, trimmed = trim_to_window(msgs, max_tokens, budget)
            try:
                if self.backend.count(msgs) + max_tokens > budget:
                    msgs, trimmed2 = trim_to_window(
                        msgs, max_tokens, budget,
                        count=lambda m: self.backend.count(m))
                    trimmed = trimmed or trimmed2
            except Exception as e:
                sys.stderr.write(f"[serve] exact recount failed ({e}); "
                                 "using the character estimate\n")
        return msgs, max_tokens, trimmed

    def _count_tokens(self, req):
        try:
            msgs = to_backend_messages(req)
            n = self.backend.count(msgs)
        except Exception as e:
            self._error(400, "invalid_request_error", str(e))
            return
        self._json(200, {"input_tokens": n})

    def _log_req(self, msg_id, stream, t0, t_first, input_tokens, out_tokens,
                 stop_reason, trimmed, error=None):
        """One line per request: the numbers a client actually feels.

        The backend's own log answers a different question and cannot answer
        this one: its "prefill ... ms" is the LAST chunk's Run(), so a
        multi-chunk prompt is under-reported there (see llm_serve.cpp), and it
        never sees the HTTP round trip or the token counts together. Wall time
        and time-to-first-token only exist at this layer.
        """
        wall = time.time() - t0
        ttft = (t_first - t0) if t_first is not None else 0.0
        n_out = out_tokens or 0
        # The first token is prefill's product, not a decode step, so the
        # per-token rate is over the tokens after it. With 0 or 1 out there is
        # no decode span to average and the field is left out.
        dec = ""
        n_dec = n_out - 1
        if n_dec > 0:
            span = max(1e-6, wall - ttft)
            dec = f" tpot={1000.0 * span / n_dec:.1f}ms/tok"
        tag = "req" if error is None else "req!"
        sys.stderr.write(
            f"[{tag}] {msg_id} {'stream' if stream else 'sync'} "
            f"in={input_tokens} out={n_out} wall={wall:.3f}s "
            f"ttft={ttft:.3f}s{dec} stop={stop_reason or 'error'}"
            f" trim={'yes' if trimmed else 'no'}"
            + (f" error={error}" if error is not None else "")
            + "\n")
        sys.stderr.flush()
        # Same measurements, kept for GET /metrics. Recorded even on failure
        # so an error appears in the counters rather than vanishing from them.
        if self.metrics is not None:
            self.metrics.record(msg_id, stream, t0, t_first, input_tokens,
                                out_tokens, stop_reason, trimmed, error)

    def _messages(self, req):
        model = req.get("model") or self.default_model
        max_tokens = int(req.get("max_tokens") or 1024)
        stream = bool(req.get("stream"))
        stop_sequences = req.get("stop_sequences") or []
        msg_id = "msg_" + uuid.uuid4().hex[:24]
        t0 = time.time()
        try:
            msgs, max_tokens, trimmed = self._prepare(req, max_tokens)
        except Exception as e:
            self._log_req(msg_id, stream, t0, None, 0, 0, None, False,
                          error=str(e))
            self._error(400, "invalid_request_error", str(e))
            return
        if trimmed:
            sys.stderr.write("[serve] history trimmed to fit the KV window\n")

        if not stream:
            self._messages_sync(req, msgs, model, max_tokens, stop_sequences,
                                msg_id, t0, trimmed)
            return

        self._sse_start()

        writer = StreamWriter(self.wfile, model, max_tokens, stop_sequences,
                              self._sse_write,
                              on_over=lambda: self.backend.stop(msg_id))
        out_tokens = 0
        input_tokens = 0
        t_first = None
        started = False
        done_reason = "stop"
        err = None
        try:
            gen = self.backend.generate(msgs, max_tokens, rid=msg_id,
                                        sample=_sampling(req))
            for frame in gen:
                if frame[0] == "tokens_in":
                    # message_start carries the prompt's token count, so it
                    # waits for the backend's first frame (the count) instead
                    # of announcing a zero.
                    input_tokens = frame[1]
                    writer.message_start(input_tokens, msg_id)
                    started = True
                elif frame[0] == "delta":
                    if t_first is None:
                        t_first = time.time()
                    if not started:
                        writer.message_start(0, msg_id)
                        started = True
                    writer.feed(frame[1])
                elif frame[0] == "done":
                    done_reason, out_tokens = frame[1], frame[2]
            if not started:
                writer.message_start(0, msg_id)
            writer.finish(done_reason, out_tokens)
        except BrokenPipeError:
            err = "client disconnected"
            return
        except Exception as e:
            err = str(e)
            try:
                writer.event("error", {"type": "error",
                                       "error": {"type": "api_error",
                                                 "message": str(e)}})
            except OSError:
                pass
        finally:
            self._log_req(msg_id, stream, t0, t_first, input_tokens,
                          out_tokens, writer.stop_reason, trimmed, err)
            try:
                self._sse_close()
            except OSError:
                pass

    def _messages_sync(self, req, msgs, model, max_tokens, stop_sequences,
                       msg_id, t0, trimmed):
        # The streaming writer with the SSE sink discarded: one parser decides
        # where a tool call ends, so a reply cannot be a usable tool_use over
        # SSE and an unusable raw blob here.
        writer = StreamWriter(None, model, max_tokens, stop_sequences,
                              lambda data: None,
                              on_over=lambda: self.backend.stop(msg_id))
        reason = "stop"
        out_tokens = 0
        input_tokens = 0
        t_first = None
        try:
            for frame in self.backend.generate(msgs, max_tokens, rid=msg_id,
                                               sample=_sampling(req)):
                if frame[0] == "tokens_in":
                    input_tokens = frame[1]
                elif frame[0] == "delta":
                    if t_first is None:
                        t_first = time.time()
                    writer.feed(frame[1])
                elif frame[0] == "done":
                    reason, out_tokens = frame[1], frame[2]
        except Exception as e:
            self._log_req(msg_id, False, t0, t_first, input_tokens, out_tokens,
                          None, trimmed, error=str(e))
            self._error(500, "api_error", str(e))
            return

        writer.finish(reason, out_tokens)
        self._log_req(msg_id, False, t0, t_first, input_tokens, out_tokens,
                      writer.stop_reason, trimmed)
        self._json(200, {
            "id": msg_id,
            "type": "message",
            "role": "assistant",
            "model": model,
            "content": writer.blocks,
            "stop_reason": writer.stop_reason,
            "stop_sequence": writer.stop_sequence,
            "usage": {"input_tokens": input_tokens,
                      "output_tokens": out_tokens},
        })


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def build_backend(args):
    # The model path is part of the KV cache fingerprint, so normalise it here:
    # a relative --model would make the file record (and later demand) a path
    # that only resolves from whatever cwd the server happened to start in.
    model = os.path.abspath(os.path.expanduser(args.model))
    argv = [args.bin, model, args.embed, args.tokenizer,
            "--max-kv", str(args.max_kv), "--kv-kind", args.kv_kind,
            "--prefill-chunk", str(args.prefill_chunk)]
    if args.kv_cache:
        # One path, both directions. Expand it here: the backend is exec'd
        # without a shell, so a ~/... the user typed would stay literal.
        cache = os.path.abspath(os.path.expanduser(args.kv_cache))
        argv += ["--kv-warm", cache, "--kv-dump", cache]
    if args.repetition_penalty != 1.0:
        argv += ["--repetition-penalty", repr(args.repetition_penalty)]
    if args.seed != 0:
        argv += ["--seed", str(args.seed)]
    env = {}
    # The coopmat routes compute wrong values on this GPU in the current
    # build (MatMulTest/MatMulGroupedBatchBuffer fail without any model, and
    # the model's answers degrade as soon as prefill is wide enough to select
    # them). Until that is fixed the server pins the known-correct routes;
    # --coop restores the default selection for benchmarking.
    if not args.coop:
        env["VKOP_NO_COOP"] = "1"
    return Backend(argv, env, max_kv=args.max_kv)


# Every option a preset may set: long name -> (does it name a file, converter).
# One table drives the parser defaults, the preset loader and the unknown-key
# check, so those three cannot drift apart. A key the table does not hold is an
# error rather than a silent no-op: a typo'd "max_kV" that the server ignores is
# the failure this whole file exists to make visible.
_PRESET_OPTIONS = {
    "host": (False, str),
    "port": (False, int),
    "bin": (True, str),
    "model": (True, str),
    "embed": (True, str),
    "tokenizer": (True, str),
    "model-name": (False, str),
    "max-kv": (False, int),
    "kv-kind": (False, str),
    "prefill-chunk": (False, int),
    "kv-cache": (True, str),
    "coop": (False, bool),
    "repetition-penalty": (False, float),
    "seed": (False, int),
    "max-output": (False, int),
}

# The environment variable each option already honoured, so a preset that takes
# precedence over one can say which. Absent means the option has no env var
# (--coop, --model-name) and nothing can be shadowed.
_PRESET_ENV = {
    "host": "VKOP_SERVE_HOST", "port": "VKOP_SERVE_PORT",
    "bin": "VKOP_SERVE_BIN", "model": "VKOP_MODEL", "embed": "VKOP_EMBED",
    "tokenizer": "VKOP_TOKENIZER", "max-kv": "VKOP_MAX_KV",
    "kv-kind": "VKOP_KV_KIND", "prefill-chunk": "VKOP_SERVE_CHUNK",
    "kv-cache": "VKOP_KV_CACHE",
    "repetition-penalty": "VKOP_REPETITION_PENALTY",
    "seed": "VKOP_SEED", "max-output": "VKOP_MAX_OUTPUT",
}


def _preset_bool(value, path, name, key):
    """JSON has real booleans, so anything else here is a typo.

    `bool("false")` is True, which would turn --coop on for a config that asked
    for it off -- silently, and coopmat is the route with known numeric errors.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.strip().lower() in (
            "true", "false", "yes", "no", "on", "off", "1", "0"):
        return value.strip().lower() in ("true", "yes", "on", "1")
    raise SystemExit(f"--config {path}: preset {name!r} key {key!r} wants a "
                     f"boolean, got {value!r}")


def load_preset(path, want=None):
    """Read one preset out of the config at `path`.

    Schema is ``{"presets": {name: {...}}, "default": name}``. A file with no
    ``presets`` key is taken as a single unnamed preset: the common case is one
    file per model setup, and making that file invent a name is friction.

    Relative paths resolve against the config's own directory, not the cwd, so
    a config that sits beside its artifacts keeps working when it is moved or
    checked out elsewhere -- which is most of the reason to have the file. (The
    rest is that the paths stop living in five exported variables.)

    Returns ``(name, {argparse_dest: value})``.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            doc = json.load(f)
    except OSError as e:
        raise SystemExit(f"--config {path}: {e}")
    except json.JSONDecodeError as e:
        raise SystemExit(f"--config {path}: not valid JSON: {e}")
    if not isinstance(doc, dict):
        raise SystemExit(f"--config {path}: top level must be an object")

    if "presets" in doc:
        presets = doc["presets"]
        if not isinstance(presets, dict) or not presets:
            raise SystemExit(f'--config {path}: "presets" must be a non-empty '
                             f"object")
        name = want or doc.get("default")
        if name is None:
            if len(presets) == 1:
                name = next(iter(presets))
            else:
                raise SystemExit(
                    f"--config {path}: {len(presets)} presets and none "
                    f"selected; pass --preset "
                    f"{{{','.join(sorted(presets))}}} or set \"default\"")
        if name not in presets:
            raise SystemExit(f"--config {path}: no preset {name!r} "
                             f"(have {', '.join(sorted(presets))})")
        raw = presets[name]
        if not isinstance(raw, dict):
            raise SystemExit(f"--config {path}: preset {name!r} must be an "
                             f"object")
    else:
        if want is not None:
            raise SystemExit(f"--config {path}: --preset {want!r} given but the "
                             f'file defines no "presets"')
        name, raw = "default", doc

    base = os.path.dirname(os.path.abspath(path))
    out: dict = {}
    for key, value in raw.items():
        if key == "_comment" or key.startswith("_comment_"):
            # JSON has no comment syntax, and the shipped example earns its keep
            # by explaining each preset in place. Exactly `_comment`/`_comment_*`
            # and nothing else: a blanket "any leading underscore" rule would
            # swallow `_max-kv`, which is a typo, and silently dropping a typo is
            # the failure the check below exists to prevent.
            continue
        norm = key.replace("_", "-")
        if norm not in _PRESET_OPTIONS:
            raise SystemExit(
                f"--config {path}: preset {name!r} has unknown key {key!r} "
                f"(known: {', '.join(sorted(_PRESET_OPTIONS))})")
        is_path, coerce = _PRESET_OPTIONS[norm]
        if coerce is bool:
            value = _preset_bool(value, path, name, key)
        elif is_path and value == "":
            # An empty path is how --kv-cache says "disabled"; it must not be
            # joined into the config's directory.
            value = ""
        elif is_path:
            if not isinstance(value, str):
                raise SystemExit(
                    f"--config {path}: preset {name!r} key {key!r} wants a "
                    f"path string, got {value!r}")
            value = os.path.normpath(
                os.path.join(base, os.path.expanduser(value)))
        else:
            try:
                value = coerce(value)
            except (TypeError, ValueError):
                raise SystemExit(
                    f"--config {path}: preset {name!r} key {key!r} wants "
                    f"{coerce.__name__}, got {value!r}")
        out[norm.replace("-", "_")] = value
    return name, out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default=os.environ.get("VKOP_SERVE_HOST", "127.0.0.1"))
    ap.add_argument("--port", type=int,
                    default=int(os.environ.get("VKOP_SERVE_PORT", "8787")))
    ap.add_argument("--bin", default=os.environ.get(
        "VKOP_SERVE_BIN", os.path.join(ROOT, "build",
                                       "llm_serve.exe" if os.name == "nt"
                                       else "llm_serve")))
    ap.add_argument("--model", default=os.environ.get(
        "VKOP_MODEL", os.path.join(ROOT, "llm", "exporter", "llm.vkopbin")))
    ap.add_argument("--embed", default=os.environ.get(
        "VKOP_EMBED", os.path.join(ROOT, "llm", "exporter",
                                   "embed_tokens.bin")))
    ap.add_argument("--tokenizer", default=os.environ.get(
        "VKOP_TOKENIZER", os.path.join(ROOT, "llm", "tokenizer",
                                       "qwen3_vl.bin")))
    ap.add_argument("--max-kv", type=int,
                    default=int(os.environ.get("VKOP_MAX_KV", "16384")),
                    help="KV window in rows (prompt + generation must fit). "
                         "Claude Code's own system prompt is ~11k tokens, so "
                         "the 8192 default of llm_chat is too tight. Each row "
                         "costs 2*nkv*hd*2 bytes per layer.")
    ap.add_argument("--kv-kind", default=os.environ.get("VKOP_KV_KIND", "auto"))
    ap.add_argument("--prefill-chunk", type=int,
                    default=int(os.environ.get("VKOP_SERVE_CHUNK", "256")),
                    help="prompt tokens per prefill dispatch. Two walls bound "
                         "it: the graph's (1, q, vocab) logits buffer (~2.5 GB "
                         "at 8192 with a 152k vocab) and, on this driver, the "
                         "work one dispatch keeps the GPU busy with -- a wide q "
                         "over a long context ends in VK_ERROR_DEVICE_LOST "
                         "measured at q=1024 above ~6k covered rows. 256 is the "
                         "value that serves Claude Code's ~11k token prompt; "
                         "larger is ~2x faster but only for shorter prompts")
    ap.add_argument("--kv-cache", default=os.environ.get("VKOP_KV_CACHE", ""),
                    metavar="FILE",
                    help="persist the KV prefix across processes in this file. "
                         "One path is both the load (--kv-warm) and the write "
                         "(--kv-dump): the first finished round dumps the "
                         "agent's stable system prompt + tool schemas, and a "
                         "later process resumes there instead of prefilling it "
                         "(~630 MB of fp8 rows read in seconds, against ~400 s "
                         "of prefill). Needs the same disk footprint free as "
                         "the model, and a restart after the artifact changes "
                         "-- the file's fingerprint then refuses it and the "
                         "server runs cold. Empty disables both.")
    ap.add_argument("--coop", action="store_true",
                    help="leave the coopmat routes enabled (default: pinned off)")
    ap.add_argument("--repetition-penalty", type=float,
                    default=float(os.environ.get("VKOP_REPETITION_PENALTY", "1.0")),
                    help="HF-style penalty on tokens this reply has already "
                         "generated (1.0 = off). Prompt tokens are never "
                         "penalised: an agent's system prompt legitimately "
                         "repeats tool names and JSON punctuation every turn. "
                         "1.05-1.1 is the usual range for breaking a "
                         "repetition loop; decoding stays deterministic")
    ap.add_argument("--seed", type=int,
                    default=int(os.environ.get("VKOP_SEED", "0")),
                    help="RNG seed for sampling, so a run is reproducible. "
                         "Ignored while decoding is greedy")
    ap.add_argument("--max-output", type=int,
                    default=int(os.environ.get("VKOP_MAX_OUTPUT", "1024")),
                    help="ceiling on generated tokens per reply (the model "
                         "runs at ~5 tok/s on this class of GPU)")
    ap.add_argument("--model-name", default="vkop-qwen3-vl-2b")
    ap.add_argument("--config", metavar="FILE",
                    default=os.environ.get("VKOP_SERVE_CONFIG", ""),
                    help="JSON preset file, so a model setup stops being five "
                         "exported variables. Schema is {\"presets\": {name: "
                         "{...}}, \"default\": name}, or a bare object taken as "
                         "a single preset. Keys are the long option names with "
                         "- or _ (model, max-kv, prefill-chunk, coop, ...); "
                         "file paths in the preset resolve against the config's "
                         "own directory. An unknown key is refused, not ignored")
    ap.add_argument("--preset", metavar="NAME",
                    default=os.environ.get("VKOP_SERVE_PRESET", "") or None,
                    help="which preset in --config to use (default: the file's "
                         '"default", or the only one it defines)')
    # Resolve the config BEFORE the real parse and feed it in as argparse
    # defaults: that is what makes the precedence CLI > preset > env > built-in
    # fall out for free, without a sentinel on all fifteen options. parse_args
    # then still lets anything typed on the command line override.
    pre, _ = ap.parse_known_args()
    preset_name, preset = None, {}
    shadowed = []
    if pre.config:
        preset_name, preset = load_preset(pre.config, pre.preset)
        for key in preset:
            # `key` is the argparse dest (underscores); the tables are keyed by
            # the option spelling (dashes).
            env = _PRESET_ENV.get(key.replace("_", "-"))
            if env and os.environ.get(env):
                shadowed.append(f"{env}={os.environ[env]}")
        ap.set_defaults(**preset)
    elif pre.preset:
        # Naming a preset with no file to read it from would otherwise be a
        # silent no-op, which is the shape of mistake this file is meant to stop.
        raise SystemExit("--preset needs --config (or VKOP_SERVE_CONFIG)")
    args = ap.parse_args()

    for path in (args.bin, args.model, args.embed, args.tokenizer):
        if not os.path.exists(path):
            raise SystemExit(f"missing: {path}")
    if preset_name:
        src = f"--config {args.config}" + (f" --preset {preset_name}"
                                           if pre.preset else "")
        print(f"[serve] preset {preset_name!r} from {src}: "
              f"{len(preset)} key(s)", file=sys.stderr)
        for key in sorted(preset):
            print(f"[serve]   {key.replace('_', '-'):20s} "
                  f"{preset[key]!r}", file=sys.stderr)
        if shadowed:
            # A stale export silently losing to a preset is exactly the kind of
            # thing that costs an afternoon; say it out loud.
            print(f"[serve] preset overrode env: {', '.join(shadowed)}",
                  file=sys.stderr)

    backend = build_backend(args)
    print("[serve] waiting for the model to load...", file=sys.stderr)
    backend.wait_ready()
    print(f"[serve] ready: http://{args.host}:{args.port}", file=sys.stderr)

    Handler.backend = backend
    Handler.metrics = Metrics()
    Handler.default_model = args.model_name
    Handler.max_output = args.max_output
    print(f"[serve] metrics: http://{args.host}:{args.port}/metrics",
          file=sys.stderr)
    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    httpd.daemon_threads = True
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
        backend.close()


if __name__ == "__main__":
    main()
