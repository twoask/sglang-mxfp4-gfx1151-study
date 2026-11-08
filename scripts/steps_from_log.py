"""Steady-state decode ms/step per batch size from an sglang server log ('Decode batch' lines, queue empty)."""

import collections
import re
import statistics
import sys

_LINE = re.compile(r"#running-req: (\d+),.*gen throughput \(token/s\): ([\d.]+), #queue-req: 0\b")


def parse_steps(log_path: str, skip_first: int = 1) -> dict[int, float]:
    by_bs = collections.defaultdict(list)
    with open(log_path) as f:
        for line in f:
            m = _LINE.search(line)
            if m:
                by_bs[int(m[1])].append(float(m[2]))
    out = {}
    for bs, thr in by_bs.items():
        thr = thr[skip_first:]
        if len(thr) >= 3:
            out[bs] = round(bs / statistics.median(thr) * 1e3, 2)
    return dict(sorted(out.items()))


if __name__ == "__main__":
    for bs, ms in parse_steps(sys.argv[1]).items():
        print(f"bs={bs}: {ms} ms/step")
