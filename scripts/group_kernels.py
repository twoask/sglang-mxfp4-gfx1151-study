"""Group GPU kernel time in a torch-profiler chrome trace by regex rules, in ms per decode step."""

import collections
import gzip
import json
import re
import sys


def _load(path):
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as f:
        return json.load(f)


def group_trace(trace_path: str, rules: list[tuple[str, str]], n_steps: int | None = None) -> dict[str, float]:
    events = _load(trace_path)["traceEvents"]
    compiled = [(g, re.compile(rx)) for g, rx in rules]
    if n_steps is None:
        n_steps = sum(1 for e in events if re.match(r"ProfilerStep#\d+", str(e.get("name", ""))))
    if not n_steps:
        raise ValueError("no ProfilerStep markers in the trace; pass n_steps")
    totals = collections.Counter()
    for e in events:
        if e.get("ph") != "X" or e.get("cat") != "kernel":
            continue
        group = next((g for g, rx in compiled if rx.search(e["name"])), "unmatched")
        totals[group] += e["dur"]
    return {g: round(us / n_steps / 1000, 6) for g, us in totals.items()}


def load_rules(path: str) -> list[tuple[str, str]]:
    rules = []
    for line in open(path):
        line = line.strip()
        if line and not line.startswith("#"):
            group, rx = line.split(None, 1)
            rules.append((group, rx))
    return rules


if __name__ == "__main__":
    got = group_trace(sys.argv[1], load_rules(sys.argv[2]), int(sys.argv[3]) if len(sys.argv) > 3 else None)
    total = sum(got.values())
    for g, ms in sorted(got.items(), key=lambda kv: -kv[1]):
        print(f"{g:16s} {ms:8.3f} ms/step  {ms / total:6.1%}")
    print(f"{'total GPU':16s} {total:8.3f} ms/step")
