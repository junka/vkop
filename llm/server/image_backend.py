"""image_serve behind the gateway: process, protocol and the OpenAI images route.

  POST /v1/images/generations   (OpenAI Images API shape)
    {"prompt": "...", "n": 1, "size": "512x512",
     "response_format": "b64_json",           # the only format: nothing hosts files
     "steps": 40, "seed": 7,                  # vkop extensions, both optional
     "stream": false}                         # true: SSE progress, see below
  -> {"created": 1732500000,
      "data": [{"b64_json": "...", "seed": 7}],
      "usage": {"input_tokens": 21},
      "vkop": {<per-stage timings from image_serve>}}

With "stream": true the reply is an event stream:
  event: progress   data: {"stage": "denoise", "image": 0, "step": 3, "steps": 40}
  event: image      data: {"index": 0, "seed": 7, "b64_json": "..."}
  event: done       data: {"stop_reason": "done", "vkop": {...}}
  event: error      data: {"error": {...}}
A client that disconnects mid-stream stops the generation within one step.
"""
import base64
import json
import os
import secrets
import shutil
import sys
import tempfile
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
    return prompt, n, steps, seed, bool(req.get("stream"))


def handle_generation(handler, req):
    """POST /v1/images/generations on an anthropic_server.Handler."""
    mm, name = handler.manager, handler.image_model
    if mm is None or name is None:
        handler._error(404, "not_found_error", "no image model configured")
        return
    backend = mm.workers[name]
    try:
        prompt, n, steps, seed, stream = parse_request(req, backend,
                                                       handler.image_steps)
        # Tokenizer only, no GPU, so it runs before taking the GPU lock: an
        # over-long prompt is a 400 at once rather than after the queue.
        tokens = backend.count(prompt)
        if tokens > backend.max_prompt_tokens:
            raise BadImageRequest(
                f"prompt is {tokens} tokens; the text tower takes at most "
                f"{backend.max_prompt_tokens}")
    except BadImageRequest as e:
        handler._error(400, "invalid_request_error", str(e))
        return

    rid = "img_" + secrets.token_hex(8)
    t0 = time.time()
    images, metrics, reason, err = [], {}, None, None
    if stream:
        handler._sse_start()

    def event(kind, payload):
        data = f"event: {kind}\ndata: {json.dumps(payload)}\n\n".encode()
        handler._sse_write(data)

    try:
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
            + (f" error={err}" if err else "") + "\n")
        if stream:
            try:
                handler._sse_close()
            except OSError:
                pass
    if stream or reason is None:
        return
    handler._json(200, {
        "created": int(t0),
        "data": [{"b64_json": base64.b64encode(png).decode(), "seed": s}
                 for _, _, s, png in images],
        "usage": {"input_tokens": tokens},
        "vkop": metrics,
    })
