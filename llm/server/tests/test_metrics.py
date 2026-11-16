"""What /metrics reports, and the two rates it must not confuse.

TTFT and TPOT answer different questions and a serving comparison is only
honest if they are computed separately:

  TTFT = request arrival -> first token, i.e. the whole prefill. It grows with
         prompt length and collapses when the KV prefix hits.
  TPOT = per-token cost AFTER the first one. Including the first token would
         fold prefill into it, making TPOT a second, worse TTFT -- and a long
         prompt would look like slow decode.

Every test here pins one of those boundaries: the first token is excluded from
TPOT, a failed request is counted as an error instead of vanishing, and the
deque stays bounded so a long-lived server cannot grow.

Run: python llm/server/tests/test_metrics.py
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


def _rec(met, t0, ttft, wall, n_in, n_out, err=None):
    """A request that started at t0, first token at t0+ttft, ended at t0+wall.

    t_end is passed explicitly so the test pins the span rather than racing the
    wall clock; production leaves it None and record() reads the clock itself.
    """
    return met.record("msg", True, t0, t0 + ttft if ttft is not None else None,
                      n_in, n_out, "end_turn", False, err, t_end=t0 + wall)


def test_ttft_is_arrival_to_first_token():
    met = m.Metrics()
    row = _rec(met, 100.0, 2.5, 4.0, 10, 5)
    check("ttft measured from arrival", abs(row["ttft_s"] - 2.5) < 1e-9,
          row["ttft_s"])
    check("wall spans the whole request", abs(row["wall_s"] - 4.0) < 1e-9,
          row["wall_s"])


def test_tpot_excludes_the_first_token():
    # 4.0 s wall, 2.5 s to the first token, 5 out => 4 decode steps in 1.5 s.
    met = m.Metrics()
    row = _rec(met, 100.0, 2.5, 4.0, 10, 5)
    check("decode_tokens is out-1", row["decode_tokens"] == 4,
          row["decode_tokens"])
    check("tpot = decode span / decode tokens",
          abs(row["tpot_s"] - 1.5 / 4) < 1e-6, row["tpot_s"])


def test_single_token_has_no_tpot():
    """One output token is entirely prefill; there is no decode step to
    average, and reporting 0.0 would read as an infinitely fast decode."""
    met = m.Metrics()
    row = _rec(met, 100.0, 1.0, 1.2, 10, 1)
    check("n_out=1 -> tpot None", row["tpot_s"] is None, row["tpot_s"])
    check("n_out=1 -> decode_tokens 0", row["decode_tokens"] == 0,
          row["decode_tokens"])


def test_error_request_is_counted_not_dropped():
    met = m.Metrics()
    _rec(met, 100.0, 0.5, 1.0, 10, 3)
    _rec(met, 200.0, None, 0.01, 4009, 0, err="context_overflow")
    snap = met.snapshot()
    check("errors counted", snap["all"]["errors"] == 1, snap["all"]["errors"])
    check("requests still counted", snap["all"]["requests"] == 2,
          snap["all"]["requests"])
    # A failed request has no ttft; it must not enter the latency stats as 0.
    check("failed request absent from ttft stats", snap["all"]["ttft_s"]["n"] == 1,
          snap["all"]["ttft_s"])


def test_aggregate_rate_weights_by_tokens():
    """A 1-token reply and a 100-token reply must not average their per-request
    TPOTs: the decode work is in the long one."""
    met = m.Metrics()
    # 2 decode tokens in 1.0 s -> 2 tok/s over 2 tokens
    _rec(met, 0.0, 1.0, 2.0, 10, 3)
    # 99 decode tokens in 0.99 s -> 100 tok/s over 99 tokens
    _rec(met, 100.0, 1.0, 1.99, 10, 100)
    snap = met.snapshot()
    # 101 tokens over 1.99 s ~= 50.75 tok/s; a mean of per-request rates would
    # be ~51 too here, so check the token-weighted identity directly.
    want = 101 / (1.0 + 0.99)
    got = snap["all"]["decode_tokens_per_s"]
    check("rate = sum(tokens)/sum(span)", abs(got - want) < 0.05, (got, want))


def test_window_is_bounded():
    met = m.Metrics(capacity=3)
    for i in range(10):
        _rec(met, float(i), 0.1, 0.2, 1, 2)
    snap = met.snapshot()
    check("deque keeps only capacity", snap["window"] == 3, snap["window"])
    check("total still counts every request",
          snap["completed_since_start"] == 10,
          snap["completed_since_start"])


def test_limit_selects_the_recent_tail():
    met = m.Metrics()
    for i in range(6):
        _rec(met, float(i), 0.1 * (i + 1), 0.2, 1, 2)
    one = met.snapshot(1)
    allr = met.snapshot(0)
    check("limit=1 keeps one recent", one["recent_n"] == 1, one["recent_n"])
    check("limit=0 means whole window", allr["recent_n"] == 6, allr["recent_n"])
    check("'all' is unaffected by limit", one["all"]["requests"] == 6,
          one["all"]["requests"])


def test_private_fields_stay_private():
    """_decode_s is an aggregate input; it must not leak into the JSON rows."""
    met = m.Metrics()
    _rec(met, 0.0, 1.0, 2.0, 10, 3)
    row = met.snapshot(5)["requests"][0]
    check("no underscore keys in output",
          not any(k.startswith("_") for k in row), sorted(row))


def test_percentile_is_nearest_rank():
    vals = [1.0, 2.0, 3.0, 4.0, 5.0]
    check("p50 picks an observed value", m.Metrics._percentile(vals, 50) == 3.0,
          m.Metrics._percentile(vals, 50))
    check("p100 stays in range", m.Metrics._percentile(vals, 100) == 5.0,
          m.Metrics._percentile(vals, 100))
    check("empty -> None", m.Metrics._percentile([], 50) is None,
          m.Metrics._percentile([], 50))


for fn in [test_ttft_is_arrival_to_first_token,
           test_tpot_excludes_the_first_token,
           test_single_token_has_no_tpot,
           test_error_request_is_counted_not_dropped,
           test_aggregate_rate_weights_by_tokens,
           test_window_is_bounded,
           test_limit_selects_the_recent_tail,
           test_private_fields_stay_private,
           test_percentile_is_nearest_rank]:
    fn()

print(f"\n{len(fails)} failure(s)")
sys.exit(1 if fails else 0)
