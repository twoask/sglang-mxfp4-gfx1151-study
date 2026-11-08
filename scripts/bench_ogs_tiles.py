"""Time triton_kernels' MXFP4 MoE matmul at Qwen3.5-35B-A3B decode shapes under different opt-flag constraints.

Shapes per layer: gate_up N=1024 K=2048, down N=2048 K=512, 256 experts, top-8. Each config's output is checked
against the default config's. Constraints go through scoped_opt_flags_constraints, the same knob sglang sets on NVIDIA.
    PYTHONPATH=$SGLANG_SRC/python python bench_ogs_tiles.py
"""

import itertools

import torch

from sglang.srt.layers.quantization.mxfp4 import _swizzle_mxfp4
from triton_kernels.matmul import FlexCtx, PrecisionConfig, matmul
from triton_kernels.matmul_details.opt_flags import scoped_opt_flags_constraints
from triton_kernels.tensor import make_ragged_tensor_metadata

E, TOPK = 256, 8
SHAPES = {"gate_up": (1024, 2048), "down": (2048, 512)}


def make(N, K, dev="cuda"):
    w = torch.randint(0, 256, (E, N, K // 2), dtype=torch.uint8, device=dev)
    scale = torch.randint(122, 130, (E, N, K // 32), dtype=torch.uint8, device=dev)
    tw, flex, tscale = _swizzle_mxfp4(w, scale, 4)
    return tw, PrecisionConfig(b_mx_scale=tscale, b_microblock_size=32, flex_ctx=FlexCtx(rhs_data=flex))


def routing(batch, dev="cuda"):
    ids = torch.stack([torch.randperm(E, device=dev)[:TOPK] for _ in range(batch)]).flatten()
    # gather_indx indexes rows of x (tokens), in expert-sorted slot order
    combine = (torch.argsort(ids, stable=True) // TOPK).to(torch.int32)
    counts = torch.bincount(ids, minlength=E).to(torch.int32)
    return combine, make_ragged_tensor_metadata(counts, ids.numel()), ids


def run(x, tw, pc, combine, ragged, constraints):
    with scoped_opt_flags_constraints(constraints):
        return matmul(x, tw, None, a_ragged_metadata=ragged, gather_indx=combine, precision_config=pc)


def time_us(fn, iters=200):
    for _ in range(10):
        fn()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) * 1000 / iters


def main():
    torch.manual_seed(0)
    configs = [{}] + [
        {k: v for k, v in zip(("block_n", "num_warps", "split_k"), c) if v is not None}
        for c in itertools.product((32, 64, 128, None), (2, 4, 8, None), (1, 2, 4, None))
        if any(v is not None for v in c)
    ]
    for name, (N, K) in SHAPES.items():
        tw, pc = make(N, K)
        mib = E * N * K * (0.5 + 1 / 32) / 2**20 / E  # per expert
        for batch in (1, 4, 16):
            combine, ragged, ids = routing(batch)
            x = torch.randn(batch, K, dtype=torch.bfloat16, device="cuda")
            touched = ids.unique().numel() * mib
            ref = run(x, tw, pc, combine, ragged, {})
            rows = []
            for c in configs:
                try:
                    out = run(x, tw, pc, combine, ragged, c)
                    ok = torch.allclose(out.float(), ref.float(), rtol=2e-2, atol=2e-2 * ref.abs().max().item())
                    us = time_us(lambda: run(x, tw, pc, combine, ragged, c))
                    rows.append((us, c, ok))
                except Exception as e:  # invalid combination for this shape
                    rows.append((float("inf"), c, f"{type(e).__name__}"))
            rows.sort(key=lambda r: r[0])
            base = next(r[0] for r in rows if r[1] == {})
            print(f"\n{name} batch {batch}: default {base:.1f} us ({touched / base * 1e6 / 1024:.1f} GB/s)")
            for us, c, ok in rows[:6]:
                print(f"  {us:8.1f} us  {base / us:5.2f}x  {touched / us * 1e6 / 1024:6.1f} GB/s  ok={ok}  {c}")


if __name__ == "__main__":
    main()
