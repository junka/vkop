"""One GPU, several resident models: who runs, and who has to make room.

The gateway fronts more than one backend process (llm_serve, image_serve, ...)
but there is one GPU and one pool of unified memory. Two rules follow:

* GPU work is serialised. Two graphs running at once do not finish sooner --
  they share the same ALUs -- and every model's peak memory is sized assuming
  it has the device to itself.
* Resident models must fit a memory budget. On the 36 GB machine this was
  built on, Qwen-Image alone peaks at ~22 GB, so the LLM and the image graphs
  cannot both stay loaded. Before a model runs, the least recently used other
  models are evicted until the sum of resident estimates fits.

What "evict" means is the worker's business: llm_serve has no way to drop its
weights and keep running, so it exits (BYE also writes its KV prefix file, so
the respawn resumes warm); image_serve stays up and releases its graphs on
UNLOAD, keeping its per-prompt cache in host memory.

Workers are duck-typed: ``mem_gb`` (float), ``resident`` (bool property) and
``evict()``. Nothing here touches a process directly, which is what keeps this
testable without a GPU (tests/test_model_manager.py).
"""
import contextlib
import sys
import threading
import time


class ModelManager:
    def __init__(self, budget_gb, log=None):
        self.budget_gb = float(budget_gb)
        self.workers = {}
        self._last_used = {}
        self._gpu = threading.Lock()
        self._log = log or (lambda msg: print(msg, file=sys.stderr))

    def add(self, name, worker):
        self.workers[name] = worker
        self._last_used[name] = 0.0 if not worker.resident else time.monotonic()

    def resident_gb(self, exclude=None):
        return sum(w.mem_gb for n, w in self.workers.items()
                   if n != exclude and w.resident)

    def _make_room(self, name):
        need = self.workers[name].mem_gb
        # Oldest first: the model idle longest is the least likely to be asked
        # for next, and it is the one whose reload the user notices least.
        others = sorted((n for n in self.workers if n != name),
                        key=lambda n: self._last_used[n])
        for other in others:
            if self.resident_gb(exclude=name) + need <= self.budget_gb:
                return
            w = self.workers[other]
            if not w.resident:
                continue
            t = time.monotonic()
            self._log(f"[mm] evicting {other} ({w.mem_gb:g} GB) to make room "
                      f"for {name} ({need:g} GB, budget {self.budget_gb:g} GB)")
            w.evict()
            self._log(f"[mm] {other} evicted in {time.monotonic() - t:.2f}s")
        if self.resident_gb(exclude=name) + need > self.budget_gb:
            # Only reachable when `name` alone is over budget: run it anyway --
            # refusing would make the model unusable -- but say so.
            self._log(f"[mm] {name} needs {need:g} GB, over the "
                      f"{self.budget_gb:g} GB budget on its own")

    @contextlib.contextmanager
    def use(self, name):
        """Hold the GPU for one request to `name`, with room made for it.

        The lock spans the whole request, including a streamed reply: the
        backend is busy until its DONE line, whether or not the HTTP client is
        still reading.
        """
        if name not in self.workers:
            raise KeyError(f"no model {name!r} (have {', '.join(self.workers)})")
        with self._gpu:
            self._make_room(name)
            self._last_used[name] = time.monotonic()
            try:
                yield self.workers[name]
            finally:
                self._last_used[name] = time.monotonic()

    def status(self):
        return {n: {"resident": w.resident, "mem_gb": w.mem_gb}
                for n, w in self.workers.items()}
