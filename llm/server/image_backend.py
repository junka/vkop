"""image_serve behind the gateway: process, protocol and the OpenAI images route.

  POST /v1/images/generations   (OpenAI Images API shape)
    {"prompt": "...", "n": 1, "size": "512x512",
     "response_format": "b64_json",           # the only format: nothing hosts files
     "steps": 40, "seed": 7,                  # vkop extensions, both optional
     "enhance": false,                        # vkop: let the LLM rewrite it first
     "stream": false}                         # true: SSE progress, see below
  -> {"created": 1732500000,
      "data": [{"b64_json": "...", "seed": 7,
                "revised_prompt": "..."}],    # only with "enhance": true
      "usage": {"input_tokens": 21},
      "vkop": {<per-stage timings from image_serve>}}

With "stream": true the reply is an event stream:
  event: progress   data: {"stage": "enhance", "revised_prompt": "..."}
  event: progress   data: {"stage": "denoise", "image": 0, "step": 3, "steps": 40}
  event: image      data: {"index": 0, "seed": 7, "b64_json": "..."}
  event: done       data: {"stop_reason": "done", "vkop": {...}}
  event: error      data: {"error": {...}}
A client that disconnects mid-stream stops the generation within one step.

"enhance": true runs the prompt through the gateway's LLM first (see
enhance_prompt): a short idea becomes a detailed scene description, and a long
one is condensed, so the original may exceed the text tower's 64-token limit.
The rewrite is greedy and cached, so the same request gets the same revised
prompt -- and with it image_serve's prompt cache -- without asking the LLM
again.
"""
import base64
import collections
import json
import os
import re
import secrets
import shutil
import sys
import tempfile
import threading
import time

from anthropic_server import Backend, _decode_err


class ImageBackend(Backend):
    """One image_serve process. Reuses Backend's spawn/relaunch plumbing."""

    def __init__(self, argv, env=None, mem_gb=22.0):
        self.mem_gb = float(mem_gb)
        self.width = self.height = 0
        self.max_prompt_tokens = 0
        self._loaded = False  # graphs resident in the process (not just alive)
        self.out_dir = tempfile.mkdtemp(prefix="vkop-img-")
        super().__init__(argv + ["--out-dir", self.out_dir], env, max_kv=0)

    # -- ModelManager worker interface -------------------------------------
    @property
    def resident(self):
        return self._loaded and self.proc.poll() is None

    def evict(self):
        """Release the graphs; the process and its prompt cache stay."""
        with self._lock:
            if self.proc.poll() is not None:
                self._loaded = False
                return
            self._send("UNLOAD")
            while True:
                kind, _, rest = self._recv().partition(" ")
                if kind == "OK":
                    break
                if kind == "ERROR":
                    raise RuntimeError(_decode_err(rest))
            self._loaded = False

    # -- protocol ----------------------------------------------------------
    def wait_ready(self, timeout=60.0):
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(f"image_serve exited with {self.proc.returncode}")
            kind, _, rest = self._recv().partition(" ")
            if kind == "READY":
                # READY <abi> <width> <height> <max_prompt_tokens>
                f = rest.split()
                self.abi = int(f[0])
                self.width, self.height, self.max_prompt_tokens = map(int, f[1:4])
                self._loaded = False  # a fresh process holds no graphs
                return
            if kind == "FATAL":
                raise RuntimeError(base64.b64decode(rest).decode("utf-8", "replace"))
        raise RuntimeError("image_serve did not become ready")

    def count(self, prompt, rid="c"):
        with self._lock:
            self._ensure_alive()
            self._send(f"COUNT {rid} {base64.b64encode(prompt.encode()).decode()}")
            while True:
                kind, _, rest = self._recv().partition(" ")
                if kind == "COUNTED":
                    return int(rest.split()[1])
                if kind == "ERROR":
                    raise RuntimeError(_decode_err(rest))

    def generate(self, prompt, steps, seed, n, rid):
        """Yield ('stage', name), ('step', image, step, steps),
        ('image', index, seed, png_bytes), ('done', reason, metrics)."""
        with self._lock:
            self._ensure_alive()
            self._send(f"GEN {rid} {steps} {seed} {n} "
                       f"{base64.b64encode(prompt.encode()).decode()}")
            while True:
                kind, _, rest = self._recv().partition(" ")
                f = rest.split()
                if kind == "STAGE":
                    if f[1] in ("load", "denoise"):
                        self._loaded = True
                    yield ("stage", f[1])
                elif kind == "STEP":
                    yield ("step", int(f[1]), int(f[2]), int(f[3]))
                elif kind == "IMAGE":
                    path = base64.b64decode(f[3]).decode()
                    with open(path, "rb") as fh:
                        png = fh.read()
                    os.unlink(path)
                    yield ("image", int(f[1]), int(f[2]), png)
                elif kind == "DONE":
                    metrics = json.loads(base64.b64decode(f[3]))
                    yield ("done", f[1], metrics)
                    return
                elif kind == "ERROR":
                    raise RuntimeError(_decode_err(rest))

    def stop(self, rid="0"):
        self._send(f"STOP {rid}")

    def close(self):
        super().close()
        shutil.rmtree(self.out_dir, ignore_errors=True)


# --------------------------------------------------------------------------
# HTTP route (mixed into anthropic_server.Handler)
# --------------------------------------------------------------------------
class BadImageRequest(ValueError):
    pass


def parse_request(req, backend, default_steps):
    prompt = req.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise BadImageRequest("prompt must be a non-empty string")
    n = req.get("n", 1)
    if not isinstance(n, int) or not 1 <= n <= 16:
        raise BadImageRequest("n must be an integer in 1..16")
    size = req.get("size") or f"{backend.width}x{backend.height}"
    if size != f"{backend.width}x{backend.height}":
        raise BadImageRequest(
            f"size {size!r} is not available: the graphs were exported for "
            f"{backend.width}x{backend.height}")
    fmt = req.get("response_format") or "b64_json"
    if fmt != "b64_json":
        raise BadImageRequest("response_format must be b64_json (no file hosting)")
    steps = req.get("steps", default_steps)
    if not isinstance(steps, int) or not 1 <= steps <= 200:
        raise BadImageRequest("steps must be an integer in 1..200")
    seed = req.get("seed")
    if seed is None:
        seed = secrets.randbelow(2**31)
    if not isinstance(seed, int) or not 0 <= seed < 2**32 - 16:
        raise BadImageRequest("seed must be a non-negative 32-bit integer")
    enhance = req.get("enhance", False)
    if not isinstance(enhance, bool):
        raise BadImageRequest("enhance must be a boolean")
    if enhance and len(prompt) > MAX_ENHANCE_CHARS:
        raise BadImageRequest(
            f"prompt is {len(prompt)} characters; enhance takes at most "
            f"{MAX_ENHANCE_CHARS}")
    return prompt, n, steps, seed, bool(req.get("stream")), enhance

# --------------------------------------------------------------------------
# Prompt enhancement: the LLM rewrites the prompt before the image model runs
# --------------------------------------------------------------------------
MAX_ENHANCE_CHARS = 4000
# Small instruct models ignore "at most N words" (the 2B LLM wrote ~110
# tokens for a 40-word request), so the length is also capped where it is
# cheap to cap: at generation. The LLM and the text tower share the Qwen
# tokenizer, so a few tokens over the tower's limit leaves room to cut back
# to a sentence end in fit_tokens.
ENHANCE_SLACK_TOKENS = 16

ENHANCE_SYSTEM = (
    "You write prompts for a text-to-image model. Rewrite the user's request "
    "as one vivid, concrete description of the image: subject, setting, "
    "composition, lighting, colours and style. Keep every detail the user "
    "asked for and invent nothing that contradicts it. Write in the user's "
    "language. One or two sentences, under 35 words (or 50 Chinese "
    "characters). No lists, no quotes, no preamble -- output only the "
    "description.")

_THINK = re.compile(r"<think>.*?(</think>|$)", re.S)
_LABEL = re.compile(r"^\s*(prompt|description|image prompt|提示词|描述)\s*[:：]\s*",
                    re.I)


def clean_rewrite(text):
    """The LLM's reply -> one line of prompt text."""
    text = _THINK.sub("", text)
    text = " ".join(text.split())
    text = _LABEL.sub("", text)
    return text.strip().strip('"\'“”「」`').strip()


def fit_tokens(text, limit, count):
    """Longest prefix of `text` that `count` puts at <= `limit` tokens, cut
    at a word or clause boundary when there is one. The LLM is asked for a
    short reply, not trusted to give one; the text tower rejects anything
    longer outright."""
    if count(text) <= limit:
        return text
    lo, hi = 0, len(text)  # count(text[:lo]) <= limit < count(text[:hi])
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if count(text[:mid]) <= limit:
            lo = mid
        else:
            hi = mid
    cut = text[:lo]
    for seps in (".!?。！？", ",;，；、", " "):
        i = max(cut.rfind(c) for c in seps)
        if i >= len(cut) // 2:
            keep = cut[:i + 1] if seps == ".!?。！？" else cut[:i]
            return keep.rstrip(" ,;，；、")
    return cut.rstrip(" ,;，；、")


class _RewriteCache:
    def __init__(self, capacity=64):
        self.capacity = capacity
        self._d = collections.OrderedDict()
        self._lock = threading.Lock()

    def get(self, key):
        with self._lock:
            if key in self._d:
                self._d.move_to_end(key)
                return self._d[key]
        return None

    def put(self, key, value):
        with self._lock:
            self._d[key] = value
            self._d.move_to_end(key)
            while len(self._d) > self.capacity:
                self._d.popitem(last=False)


_rewrites = _RewriteCache()


def enhance_prompt(handler, prompt, image):
    """Rewrite `prompt` with the gateway's LLM; return (revised, info).

    Takes the GPU for the LLM only for the rewrite. With both models over the
    memory budget this costs a swap each way, which is why rewrites are
    cached: a repeated request goes straight to the image model.
    """
    t0 = time.time()
    revised = _rewrites.get(prompt)
    if revised is not None:
        return revised, {"enhance_cached": True, "enhance_s": 0.0}
    msgs = [("system", ENHANCE_SYSTEM), ("user", prompt)]
    text, out_tokens = [], 0
    with handler.manager.use(handler.llm_model) as llm:
        for fr in llm.generate(msgs, image.max_prompt_tokens + ENHANCE_SLACK_TOKENS,
                               rid="enh_" + secrets.token_hex(6)):
            if fr[0] == "delta":
                text.append(fr[1])
            elif fr[0] == "done":
                out_tokens = fr[2]
    revised = clean_rewrite("".join(text))
    if not revised:
        raise RuntimeError("the LLM returned an empty rewrite")
    revised = fit_tokens(revised, image.max_prompt_tokens, image.count)
    _rewrites.put(prompt, revised)
    return revised, {"enhance_cached": False,
                     "enhance_s": round(time.time() - t0, 3),
                     "enhance_tokens": out_tokens}


class _ClientGone(Exception):
    pass


def _try_event(event, kind, payload):
    try:
        event(kind, payload)
        return True
    except OSError:
        return False


def handle_generation(handler, req):
    """POST /v1/images/generations on an anthropic_server.Handler."""
    mm, name = handler.manager, handler.image_model
    if mm is None or name is None:
        handler._error(404, "not_found_error", "no image model configured")
        return
    backend = mm.workers[name]
    tokens = None
    try:
        prompt, n, steps, seed, stream, enhance = parse_request(
            req, backend, handler.image_steps)
        if not enhance:
            # Tokenizer only, no GPU, so it runs before taking the GPU lock:
            # an over-long prompt is a 400 at once rather than after the queue.
            tokens = backend.count(prompt)
            if tokens > backend.max_prompt_tokens:
                raise BadImageRequest(
                    f"prompt is {tokens} tokens; the text tower takes at most "
                    f"{backend.max_prompt_tokens} (or send \"enhance\": true "
                    f"to have it condensed)")
    except BadImageRequest as e:
        handler._error(400, "invalid_request_error", str(e))
        return

    rid = "img_" + secrets.token_hex(8)
    t0 = time.time()
    images, metrics, reason, err = [], {}, None, None
    revised, enh_info = None, {}
    if stream:
        handler._sse_start()

    def event(kind, payload):
        data = f"event: {kind}\ndata: {json.dumps(payload)}\n\n".encode()
        handler._sse_write(data)

    try:
        if enhance:
            gone = False
            if stream:
                gone = not _try_event(event, "progress", {"stage": "enhance"})
            if not gone:
                revised, enh_info = enhance_prompt(handler, prompt, backend)
                prompt = revised
            if stream and not gone:
                gone = not _try_event(event, "progress", {
                    "stage": "enhance", "revised_prompt": revised, **enh_info})
            if gone:
                # Nobody is reading: do not load the image model for them.
                raise _ClientGone()
        if tokens is None:
            tokens = backend.count(prompt)
        with mm.use(name) as be:
            gen = be.generate(prompt, steps, seed, n, rid)
            stopping = False
            for fr in gen:
                if not stream or stopping:
                    if fr[0] == "image":
                        images.append(fr)
                    elif fr[0] == "done":
                        reason, metrics = fr[1], fr[2]
                    continue
                try:
                    if fr[0] == "stage":
                        event("progress", {"stage": fr[1]})
                    elif fr[0] == "step":
                        event("progress", {"stage": "denoise", "image": fr[1],
                                           "step": fr[2], "steps": fr[3]})
                    elif fr[0] == "image":
                        images.append(fr)
                        event("image", {"index": fr[1], "seed": fr[2],
                                        "b64_json": base64.b64encode(fr[3]).decode()})
                    elif fr[0] == "done":
                        reason, metrics = fr[1], fr[2]
                        event("done", {"stop_reason": reason, "vkop": metrics})
                except OSError:
                    # The client went away: stop at the next step boundary,
                    # then keep draining so the protocol stays in sync.
                    err = "client disconnected"
                    stopping = True
                    be.stop(rid)
    except _ClientGone:
        err = "client disconnected"
    except Exception as e:  # backend error: report, and log below
        err = str(e)
        if stream:
            try:
                event("error", {"error": {"type": "api_error", "message": err}})
            except OSError:
                pass
        else:
            handler._error(500, "api_error", err)
    finally:
        sys.stderr.write(
            f"[img] {rid} {'stream' if stream else 'sync'} in={tokens} n={n} "
            f"steps={steps} seed={seed} out={len(images)} "
            f"wall={time.time() - t0:.2f}s stop={reason or 'error'} "
            f"cache_hit={metrics.get('prompt_cache_hit')} "
            f"s/step={metrics.get('s_per_step')}"
            + (f" enhance={enh_info.get('enhance_s')}s"
               f"{' cached' if enh_info.get('enhance_cached') else ''}"
               if enhance else "")
            + (f" error={err}" if err else "") + "\n")
        if stream:
            try:
                handler._sse_close()
            except OSError:
                pass
    if stream or reason is None:
        return
    data = [{"b64_json": base64.b64encode(png).decode(), "seed": s}
            for _, _, s, png in images]
    if revised is not None:
        for d in data:
            d["revised_prompt"] = revised
    handler._json(200, {
        "created": int(t0),
        "data": data,
        "usage": {"input_tokens": tokens},
        "vkop": {**metrics, **enh_info},
    })
