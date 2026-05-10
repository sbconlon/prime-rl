"""Analyze PROF log lines emitted by the Advantage Server\'s profiling
instrumentation (src/prime_rl/advantage_server/_prof.py).

Input: a log file containing lines of the form
    PROF rid=<id> stage=<name> duration_ms=<X.X> [k=v ...]

Output: a table per stage with count, mean, p50, p95, max, total time across
all observations, sorted by total time descending. Plus a "per-request"
breakdown showing average time spent in each stage per request, computed
relative to the TOTAL stage.

Usage:
    uv run python scripts/canary/parse_prof_log.py /tmp/adv_prof_run/server.log
"""

from __future__ import annotations

import argparse
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

LINE_RE = re.compile(
    r"PROF\s+rid=(?P<rid>\S+)\s+stage=(?P<stage>\S+)\s+duration_ms=(?P<dur>[0-9.]+)(?P<extras>.*)$"
)


def parse(path: Path) -> dict[str, list[float]]:
    """Return {stage_name: [duration_ms, ...]}."""
    by_stage: dict[str, list[float]] = defaultdict(list)
    n_lines = 0
    n_matched = 0
    request_ids: set[str] = set()
    with open(path) as f:
        for line in f:
            n_lines += 1
            m = LINE_RE.search(line)
            if not m:
                continue
            n_matched += 1
            stage = m.group("stage")
            dur = float(m.group("dur"))
            rid = m.group("rid")
            by_stage[stage].append(dur)
            request_ids.add(rid)
    print(f"Parsed {n_matched} PROF lines from {n_lines} total lines, {len(request_ids)} distinct requests")
    print()
    return by_stage


def percentile(sorted_values: list[float], pct: float) -> float:
    if not sorted_values:
        return 0.0
    k = max(0, min(len(sorted_values) - 1, int(pct * (len(sorted_values) - 1))))
    return sorted_values[k]


def render(by_stage: dict[str, list[float]]) -> None:
    if not by_stage:
        print("No PROF lines found. Was PRIME_RL_ADV_PROF=1 set when the server ran?")
        return

    # Build table rows.
    rows = []
    for stage, durs in by_stage.items():
        sorted_durs = sorted(durs)
        rows.append({
            "stage": stage,
            "count": len(durs),
            "mean": statistics.mean(durs),
            "p50": percentile(sorted_durs, 0.50),
            "p95": percentile(sorted_durs, 0.95),
            "max": max(durs),
            "total_s": sum(durs) / 1000.0,
        })
    rows.sort(key=lambda r: r["total_s"], reverse=True)

    # Headers.
    cols = [
        ("stage", 56, "{:<56}"),
        ("count", 7, "{:>7}"),
        ("mean_ms", 10, "{:>10.1f}"),
        ("p50_ms", 10, "{:>10.1f}"),
        ("p95_ms", 10, "{:>10.1f}"),
        ("max_ms", 10, "{:>10.1f}"),
        ("total_s", 10, "{:>10.2f}"),
    ]
    header_fmt = "  ".join("{{:>{w}}}".format(w=w) if name != "stage" else "{{:<{w}}}".format(w=w) for name, w, _ in cols)
    print(header_fmt.format(*[c[0] for c in cols]))
    print(header_fmt.format(*["-" * c[1] for c in cols]))
    for r in rows:
        out = []
        for name, _, fmt in cols:
            key = name.replace("_ms", "").replace("_s", "" if name != "total_s" else "_s")
            # Map column name back to the row dict key.
            mapping = {
                "stage": "stage", "count": "count",
                "mean_ms": "mean", "p50_ms": "p50", "p95_ms": "p95", "max_ms": "max",
                "total_s": "total_s",
            }
            out.append(fmt.format(r[mapping[name]]))
        print("  ".join(out))

    # Per-request attribution: for each non-TOTAL stage, what fraction of
    # the overall TOTAL stage time it accounts for.
    total_keys = [s for s in by_stage if s.endswith("TOTAL")]
    if total_keys:
        print()
        print("=== per-request attribution (fraction of TOTAL) ===")
        # Use the most-encompassing TOTAL stage (longest mean).
        total_key = max(total_keys, key=lambda k: statistics.mean(by_stage[k]))
        total_sum = sum(by_stage[total_key])
        n_requests = len(by_stage[total_key])
        print(f"  reference stage: {total_key} (n={n_requests}, total {total_sum/1000:.2f}s, mean {statistics.mean(by_stage[total_key]):.1f}ms/req)")
        print()
        print(f"  {{:<56}}  {{:>10}}  {{:>10}}  {{:>10}}".format(
            "stage", "%_of_TOTAL", "ms/req", "calls/req"
        ))
        print("  " + "-" * 95)
        for r in rows:
            if r["stage"] == total_key:
                continue
            pct = 100.0 * (r["total_s"] * 1000.0) / total_sum if total_sum > 0 else 0.0
            ms_per_req = r["total_s"] * 1000.0 / n_requests if n_requests else 0.0
            calls_per_req = r["count"] / n_requests if n_requests else 0.0
            print(f"  {{:<56}}  {{:>10.1f}}  {{:>10.1f}}  {{:>10.2f}}".format(
                r["stage"], pct, ms_per_req, calls_per_req,
            ))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("log_path", type=Path)
    args = ap.parse_args()
    if not args.log_path.exists():
        print(f"ERROR: {args.log_path} does not exist", file=sys.stderr)
        sys.exit(1)

    by_stage = parse(args.log_path)
    render(by_stage)


if __name__ == "__main__":
    main()
