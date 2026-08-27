"""
Runs the dispatch benchmark and regression-gates the results against the
most recent entry in benchmarks/results/history.jsonl, if one exists.
Skips the regression assertion gracefully on the very first run (no
history to compare against yet) -- the benchmark still runs and writes
a new history entry either way.

This is a timing test, not a correctness test -- see
test_dispatch_equivalence.py for that. Marked slow since it does real
forward+backward passes across every registered strategy.
"""
import json
import os

import pytest

from benchmarks.benchmark_dispatch import main as run_benchmark

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "benchmarks", "results")
HISTORY_PATH = os.path.join(RESULTS_DIR, "history.jsonl")

# How much slower (as a fraction) a strategy is allowed to get before
# this test fails a regression. 1.5 = 50% slower than last recorded run.
REGRESSION_TOLERANCE = 1.5


def _last_history_entry():
    if not os.path.exists(HISTORY_PATH):
        return None
    with open(HISTORY_PATH, "r") as f:
        lines = [line for line in f if line.strip()]
    if not lines:
        return None
    return json.loads(lines[-1])


@pytest.mark.slow
def test_no_dispatch_regression():
    previous = _last_history_entry()

    results = run_benchmark(T=1024, n_embd=64, hidden=256, num_experts=8, top_k=2, iters=5, synthetic=True)

    if previous is None:
        pytest.skip("no prior benchmarks/results/history.jsonl entry to compare against; this run establishes the baseline")

    prev_results = previous.get("results_ms", {})
    regressions = []
    for name, ms in results.items():
        prev_ms = prev_results.get(name)
        if prev_ms is None:
            continue  # strategy is new since the last recorded run, nothing to compare
        if ms > prev_ms * REGRESSION_TOLERANCE:
            regressions.append(f"{name}: {prev_ms:.3f}ms -> {ms:.3f}ms (>{REGRESSION_TOLERANCE}x slower)")

    assert not regressions, "dispatch regression(s) detected:\n" + "\n".join(regressions)
