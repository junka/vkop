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
* Greedy decoding only: sampling params (temperature/top_p/top_k) are ignored
  because llm_serve's argmax is the only decode rule it implements. A client
  that expects sampling gets determinism instead of an error.
* Images are not accepted (the vision tower is a separate graph); an image
  block is reported as text so the model at least knows one was dropped.
* One request at a time: there is one GPU context and one KV cache.
"""
import argparse
import base64
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


# --------------------------------------------------------------------------
# Backend: one llm_serve process, one request at a time.
# --------------------------------------------------------------------------
class Backend:
    def __init__(self, argv, env=None, max_kv=8192):
        self.max_kv = max_kv
        self.argv = argv
        self.env = env
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
    system = _text_of(req.get("system"))
    preamble = _tools_preamble(req.get("tools"))
    if preamble:
        system = (system + "\n\n" + preamble) if system else preamble

    out = []
    if system:
        out.append(("system", system))

    pending_tool_results = []

    def flush_tool_results():
        if pending_tool_results:
            out.append(("user", "\n".join(pending_tool_results)))
            pending_tool_results.clear()

    for msg in req.get("messages", []):
        role = msg.get("role")
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
def parse_output(text):
    """Split generated text into text / tool_use blocks."""
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
        j = rest.find(TOOL_CLOSE, i + len(TOOL_OPEN))
        if j < 0:
            inner = rest[i + len(TOOL_OPEN):]
            rest = ""
        else:
            inner = rest[i + len(TOOL_OPEN):j]
            rest = rest[j + len(TOOL_CLOSE):]
        name, args = _parse_tool_call(inner)
        blocks.append({
            "type": "tool_use",
            "id": "toolu_" + secrets.token_hex(12),
            "name": name,
            "input": args,
        })
    return blocks


def _loads_dict(text):
    if not text:
        return None
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def _close_json(text):
    """Append the closing delimiters a structurally truncated JSON value lacks.

    Qwen3-VL drops the final brace on some tool calls deterministically, and the
    raw fallback below then hands the client an argument object no tool schema
    accepts -- which the client retries forever. Braces inside strings do not
    count, and a value cut mid-string is left alone: closing it would invent
    content the model never wrote.
    """
    stack = []
    in_str = esc = False
    for ch in text:
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
    way.
    """

    def __init__(self, wfile, model, max_tokens, stop_sequences, write):
        self.wfile = wfile
        self.model = model
        self.max_tokens = max_tokens
        self.stop_sequences = [s for s in (stop_sequences or []) if s]
        self.holdback = max([len(TOOL_OPEN) - 1] +
                            [max(0, len(s) - 1) for s in self.stop_sequences])
        self.buf = ""
        self.in_tool = False
        self.text_buf = ""
        self.index = 0
        self.text_open = False
        self.stopped = False
        self.stop_reason = None
        self.stop_sequence = None
        self.text_emitted = 0
        self.saw_tool = False
        self._write = write

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
        self.event("content_block_delta", {
            "type": "content_block_delta",
            "index": self.index,
            "delta": {"type": "text_delta", "text": chunk},
        })

    def _emit_tool(self, inner):
        name, args = _parse_tool_call(inner)
        self._close_text()
        idx = self.index
        self.event("content_block_start", {
            "type": "content_block_start",
            "index": idx,
            "content_block": {
                "type": "tool_use",
                "id": "toolu_" + secrets.token_hex(12),
                "name": name,
                "input": {},
            },
        })
        self.event("content_block_delta", {
            "type": "content_block_delta",
            "index": idx,
            "delta": {"type": "input_json_delta",
                      "partial_json": json.dumps(args, ensure_ascii=False)},
        })
        self.event("content_block_stop",
                   {"type": "content_block_stop", "index": idx})
        self.index += 1
        self.saw_tool = True

    # -- feeding -----------------------------------------------------------
    def feed(self, delta):
        if self.stopped:
            return
        if self.in_tool:
            self.buf += delta
            j = self.buf.find(TOOL_CLOSE)
            if j >= 0:
                self._emit_tool(self.buf[:j])
                self.buf = self.buf[j + len(TOOL_CLOSE):]
                self.in_tool = False
                self._pump()
            return
        self.buf += delta
        self._pump()

    def _pump(self):
        while not self.stopped:
            i = self.buf.find(TOOL_OPEN)
            if i >= 0:
                self._emit_safe(self.buf[:i])
                self._close_text()
                self.buf = self.buf[i + len(TOOL_OPEN):]
                self.in_tool = True
                j = self.buf.find(TOOL_CLOSE)
                if j < 0:
                    return
                self._emit_tool(self.buf[:j])
                self.buf = self.buf[j + len(TOOL_CLOSE):]
                self.in_tool = False
                continue
            # Nothing to open yet: emit everything that cannot start a marker
            # or a stop sequence.
            keep = self.holdback
            if len(self.buf) > keep:
                self._emit_safe(self.buf[:len(self.buf) - keep])
                self.buf = self.buf[len(self.buf) - keep:]
            return

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
        if not self.stopped:
            tail = self.buf
            self.buf = ""
            if self.in_tool:
                # Generation ended inside a tool call: emit what we have as a
                # tool_use rather than dropping it.
                self._emit_tool(tail)
            else:
                self._emit_safe(tail)
        self._close_text()
        if self.stop_reason is None:
            if getattr(self, "saw_tool", False):
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
# HTTP
# --------------------------------------------------------------------------
class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "vkop-anthropic/1"

    backend: Backend = None
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
        return json.loads(self.rfile.read(length).decode("utf-8"))

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

    def _messages(self, req):
        model = req.get("model") or self.default_model
        max_tokens = int(req.get("max_tokens") or 1024)
        stream = bool(req.get("stream"))
        stop_sequences = req.get("stop_sequences") or []
        msg_id = "msg_" + uuid.uuid4().hex[:24]
        try:
            msgs, max_tokens, trimmed = self._prepare(req, max_tokens)
        except Exception as e:
            self._error(400, "invalid_request_error", str(e))
            return
        if trimmed:
            sys.stderr.write("[serve] history trimmed to fit the KV window\n")

        if not stream:
            self._messages_sync(req, msgs, model, max_tokens, stop_sequences,
                                msg_id)
            return

        self._sse_start()

        writer = StreamWriter(self.wfile, model, max_tokens, stop_sequences,
                              self._sse_write)
        out_tokens = 0
        started = False
        done_reason = "stop"
        try:
            gen = self.backend.generate(msgs, max_tokens, rid=msg_id,
                                        sample=_sampling(req))
            for frame in gen:
                if frame[0] == "tokens_in":
                    # message_start carries the prompt's token count, so it
                    # waits for the backend's first frame (the count) instead
                    # of announcing a zero.
                    writer.message_start(frame[1], msg_id)
                    started = True
                elif frame[0] == "delta":
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
            return
        except Exception as e:
            try:
                writer.event("error", {"type": "error",
                                       "error": {"type": "api_error",
                                                 "message": str(e)}})
            except OSError:
                pass
        finally:
            try:
                self._sse_close()
            except OSError:
                pass

    def _messages_sync(self, req, msgs, model, max_tokens, stop_sequences,
                       msg_id):
        text_parts = []
        reason = "stop"
        out_tokens = 0
        input_tokens = 0
        try:
            for frame in self.backend.generate(msgs, max_tokens, rid=msg_id,
                                               sample=_sampling(req)):
                if frame[0] == "tokens_in":
                    input_tokens = frame[1]
                elif frame[0] == "delta":
                    text_parts.append(frame[1])
                elif frame[0] == "done":
                    reason, out_tokens = frame[1], frame[2]
        except Exception as e:
            self._error(500, "api_error", str(e))
            return

        text = "".join(text_parts)
        # Stop sequences first: they cut the visible text.
        stop_sequence = None
        for s in stop_sequences:
            k = text.find(s)
            if k >= 0:
                text = text[:k]
                reason = "stop"
                stop_sequence = s
                break

        blocks = parse_output(text)
        if any(b["type"] == "tool_use" for b in blocks):
            stop_reason = "tool_use"
        elif stop_sequence is not None:
            stop_reason = "stop_sequence"
        elif reason == "length":
            stop_reason = "max_tokens"
        else:
            stop_reason = "end_turn"

        self._json(200, {
            "id": msg_id,
            "type": "message",
            "role": "assistant",
            "model": model,
            "content": blocks,
            "stop_reason": stop_reason,
            "stop_sequence": stop_sequence,
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
    args = ap.parse_args()

    for path in (args.bin, args.model, args.embed, args.tokenizer):
        if not os.path.exists(path):
            raise SystemExit(f"missing: {path}")

    backend = build_backend(args)
    print("[serve] waiting for the model to load...", file=sys.stderr)
    backend.wait_ready()
    print(f"[serve] ready: http://{args.host}:{args.port}", file=sys.stderr)

    Handler.backend = backend
    Handler.default_model = args.model_name
    Handler.max_output = args.max_output
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
