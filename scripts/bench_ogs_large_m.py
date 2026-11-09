"""Prefill-sized (large M) timing of the MXFP4 MoE matmul: default opt flags vs a few fixed tiles.

Same shapes and routing as bench_ogs_tiles.py, at num_tokens up to 8192 (rows = tokens x top-8).
    PYTHONPATH=$SGLANG_SRC/python python scripts/bench_ogs_large_m.py
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))
from bench_ogs_tiles import SHAPES, make, routing, run, time_us  # noqa: E402

CONFIGS = [
    {},
    {"block_n": 64, "num_warps": 4},
    {"block_n": 128, "num_warps": 4},
    {"block_n": 64, "num_warps": 8},
    {"block_n": 128, "num_warps": 8},
]


def main():
    torch.manual_seed(0)
    for name, (N, K) in SHAPES.items():
        tw, pc = make(N, K)
        for tokens in (16, 64, 256, 1024, 2048, 4096, 8192):
            combine, ragged, _ = routing(tokens)
            x = torch.randn(tokens, K, dtype=torch.bfloat16, device="cuda")
            ref = run(x, tw, pc, combine, ragged, {})
            cells = []
            for c in CONFIGS:
                out = run(x, tw, pc, combine, ragged, c)
                ok = torch.allclose(out.float(), ref.float(), rtol=2e-2, atol=2e-2 * ref.abs().max().item())
                us = time_us(lambda: run(x, tw, pc, combine, ragged, c), iters=50 if tokens >= 1024 else 200)
                cells.append((us, ok))
            base = cells[0][0]
            row = "  ".join(f"{us:9.1f}{'' if ok else '!'} ({base / us:4.2f}x)" for us, ok in cells)
            print(f"{name:7s} tokens {tokens:5d} rows {tokens * 8:6d}: {row}", flush=True)
    print("columns:", ["default" if not c else c for c in CONFIGS])


if __name__ == "__main__":
    main()
