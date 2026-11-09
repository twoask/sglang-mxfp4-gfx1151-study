"""Run check_triton_kernels.main under the default opt flags and under block_n 64 / 4 warps, at both MoE shapes.
    PYTHONPATH=$SGLANG_SRC/python python scripts/check_bn64w4_reference.py scripts
"""
import sys, torch
sys.path.insert(0, sys.argv[1])
import check_triton_kernels as c
from triton_kernels.matmul_details.opt_flags import scoped_opt_flags_constraints
from triton_kernels import matmul as mm_mod
orig = mm_mod.matmul
for cons in ({}, {"block_n": 64, "num_warps": 4}):
    for (N, K) in ((1024, 2048), (2048, 512)):
        for rows in (8, 32, 128):
            def wrapped(*a, **k):
                with scoped_opt_flags_constraints(cons):
                    return orig(*a, **k)
            c.matmul = wrapped
            print(cons, N, K, rows, end=": ")
            c.main(E=16, N=N, K=K, tokens=rows)
