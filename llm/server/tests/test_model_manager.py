"""Who the gateway unloads when a model needs room, and what it leaves alone.

The budget is the whole point: on a 36 GB machine Qwen-Image (~22 GB) and the
LLM (~10 GB) must not be resident together, and getting that wrong is an OOM
reboot rather than an error message. These tests drive ModelManager with fake
workers, so they need neither a GPU nor a backend binary.

Also covers the image route's request validation (parse_request), which
decides what reaches the GPU queue at all.

Run: python llm/server/tests/test_model_manager.py
"""
import os
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, os.pardir))
from model_manager import ModelManager  # noqa: E402
import image_backend  # noqa: E402

fails = []


def check(name, cond, detail=""):
    if cond:
        print("ok   ", name)
    else:
        print("FAIL ", name, detail)
        fails.append(name)


class Fake:
    def __init__(self, mem_gb, resident=False):
        self.mem_gb = mem_gb
        self.resident = resident
        self.evictions = 0

    def evict(self):
        self.evictions += 1
        self.resident = False


def mm_with(budget, **workers):
    mm = ModelManager(budget, log=lambda msg: None)
    for name, w in workers.items():
        mm.add(name, w)
    return mm


def test_too_big_together_evicts_the_other():
    llm, img = Fake(10, resident=True), Fake(22)
    mm = mm_with(25, llm=llm, image=img)
    with mm.use("image") as w:
        img.resident = True  # what a real backend does once its graphs load
        check("yields the worker", w is img)
    check("llm evicted for the image", llm.evictions == 1 and not llm.resident)
    with mm.use("llm"):
        llm.resident = True
    check("image evicted for the llm", img.evictions == 1 and not img.resident)


def test_fits_together_evicts_nothing():
    llm, img = Fake(10, resident=True), Fake(22)
    mm = mm_with(40, llm=llm, image=img)
    with mm.use("image"):
        img.resident = True
    with mm.use("llm"):
        pass
    check("nothing evicted under a large budget",
          llm.evictions == 0 and img.evictions == 0)


def test_lru_order():
    a, b, c = Fake(8, True), Fake(8, True), Fake(8)
    mm = mm_with(17, a=a, b=b, c=c)
    with mm.use("a"):
        pass
    time.sleep(0.01)
    with mm.use("b"):
        pass
    with mm.use("c"):
        c.resident = True
    check("least recently used goes first", a.evictions == 1 and b.evictions == 0,
          (a.evictions, b.evictions))


def test_only_as_many_as_needed():
    a, b, c = Fake(5, True), Fake(5, True), Fake(10)
    mm = mm_with(20, a=a, b=b, c=c)
    with mm.use("c"):
        pass
    check("no eviction when the sum fits", a.evictions == b.evictions == 0)


def test_non_resident_is_never_evicted():
    a, b = Fake(30, resident=False), Fake(10)
    mm = mm_with(20, a=a, b=b)
    with mm.use("b"):
        pass
    check("a non-resident worker is skipped", a.evictions == 0)


def test_over_budget_alone_still_runs():
    a = Fake(50)
    mm = mm_with(20, a=a)
    ran = False
    with mm.use("a"):
        ran = True
    check("a model bigger than the budget still runs", ran)


def test_unknown_model():
    mm = mm_with(20, a=Fake(1))
    try:
        with mm.use("nope"):
            pass
        check("unknown model raises", False)
    except KeyError:
        check("unknown model raises", True)


def test_requests_are_serialised():
    mm = mm_with(100, a=Fake(1), b=Fake(1))
    inside, peak = [0], [0]
    lock = threading.Lock()

    def run(name):
        with mm.use(name):
            with lock:
                inside[0] += 1
                peak[0] = max(peak[0], inside[0])
            time.sleep(0.02)
            with lock:
                inside[0] -= 1

    ts = [threading.Thread(target=run, args=(n,)) for n in "abab"]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    check("one request on the GPU at a time", peak[0] == 1, peak[0])


class _Img:
    width = height = 512
    max_prompt_tokens = 64


def _bad(req):
    try:
        image_backend.parse_request(req, _Img, 40)
        return None
    except image_backend.BadImageRequest as e:
        return str(e)


def test_image_request_validation():
    p, n, steps, seed, stream, enhance = image_backend.parse_request(
        {"prompt": "a fox"}, _Img, 40)
    check("defaults: n=1, server steps, random seed, sync, no enhance",
          (p, n, steps, stream, enhance) == ("a fox", 1, 40, False, False)
          and 0 <= seed < 2**31)
    check("explicit steps/seed kept",
          image_backend.parse_request({"prompt": "x", "steps": 8, "seed": 7},
                                      _Img, 40)[2:4] == (8, 7))
    check("empty prompt refused", _bad({"prompt": " "}) is not None)
    check("other size refused", "512x512" in (_bad({"prompt": "x",
                                                    "size": "1024x1024"}) or ""))
    check("url format refused", _bad({"prompt": "x", "response_format": "url"})
          is not None)
    check("n out of range refused", _bad({"prompt": "x", "n": 0}) is not None
          and _bad({"prompt": "x", "n": 17}) is not None)
    check("steps out of range refused", _bad({"prompt": "x", "steps": 0}) is not None)
    check("negative seed refused", _bad({"prompt": "x", "seed": -1}) is not None)
    check("non-int n refused", _bad({"prompt": "x", "n": "2"}) is not None)
    check("non-bool enhance refused", _bad({"prompt": "x", "enhance": 1}) is not None)
    check("huge prompt refused for enhance",
          _bad({"prompt": "x" * 5000, "enhance": True}) is not None)


def test_clean_rewrite():
    c = image_backend.clean_rewrite
    check("think block and label dropped",
          c("<think>hmm\nok</think>\n\nPrompt: A fox, golden light.")
          == "A fox, golden light.", c("<think>hmm</think>Prompt: A fox."))
    check("quotes and newlines", c('"A fox\non a bench"') == "A fox on a bench")
    check("unterminated think is empty", c("<think>still going") == "")
    check("chinese label", c("提示词：一只狐狸") == "一只狐狸")


def test_fit_tokens():
    words = lambda t: len(t.split())  # noqa: E731
    fit = image_backend.fit_tokens
    check("short text untouched", fit("a b c", 5, words) == "a b c")
    out = fit("A fox sits. It is on a bench in a park at dusk", 6, words)
    check("cut at the sentence end", out == "A fox sits.", out)
    out = fit("one two three four five six seven eight", 5, words)
    check("cut at a word boundary", out == "one two three four five", out)
    chars = len
    out = fit("一只红色的狐狸，坐在公园的长椅上，阳光明媚", 12, chars)
    check("cjk cut at a clause, separator dropped", out == "一只红色的狐狸", out)


class _Llm:
    mem_gb, resident = 1, True

    def __init__(self, reply):
        self.reply, self.calls = reply, 0

    def evict(self):
        pass

    def generate(self, msgs, max_new, rid="0", sample=None):
        self.calls += 1
        assert msgs[0][0] == "system" and msgs[1] == ("user", "fox")
        yield ("tokens_in", 50)
        for w in self.reply:
            yield ("delta", w)
        yield ("done", "stop", len(self.reply))


def test_enhance_prompt():
    llm = _Llm(["A red fox ", "on a bench, ", "golden hour light, ",
                "shallow depth of field, ", "photorealistic."])
    mm = mm_with(100, llm=llm)

    class H:
        manager, llm_model = mm, "llm"

    class Img:
        max_prompt_tokens = 6

        @staticmethod
        def count(t):
            return len(t.split())

    image_backend._rewrites = image_backend._RewriteCache()
    out, info = image_backend.enhance_prompt(H, "fox", Img)
    check("rewrite fitted to the token limit",
          out == "A red fox on a bench" and not info["enhance_cached"], out)
    out2, info2 = image_backend.enhance_prompt(H, "fox", Img)
    check("second call is cached, LLM not asked again",
          out2 == out and info2["enhance_cached"] and llm.calls == 1)


for fn in [test_too_big_together_evicts_the_other,
           test_fits_together_evicts_nothing,
           test_lru_order,
           test_only_as_many_as_needed,
           test_non_resident_is_never_evicted,
           test_over_budget_alone_still_runs,
           test_unknown_model,
           test_requests_are_serialised,
           test_image_request_validation,
           test_clean_rewrite,
           test_fit_tokens,
           test_enhance_prompt]:
    fn()

print(f"\n{len(fails)} failure(s)")
sys.exit(1 if fails else 0)
