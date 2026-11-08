import sys, torch
sys.path.insert(0, __import__("os").path.dirname(__file__))
from bench_ogs_tiles import make, routing, run, time_us, SHAPES
torch.manual_seed(0)
CFG = {"ours": {"block_n": 64, "num_warps": 4},
       "10810": {"block_m": 16, "block_n": 64, "block_k": 512, "num_warps": 4},
       "10810_bk128": {"block_m": 16, "block_n": 64, "num_warps": 4}}
for name, (N, K) in SHAPES.items():
    tw, pc = make(N, K)
    for b in (1, 4, 16, 32):
        combine, ragged, ids = routing(b); x = torch.randn(b, K, dtype=torch.bfloat16, device="cuda")
        ref = run(x, tw, pc, combine, ragged, {}).float(); line = f"{name:7s} b{b:<3d} default {time_us(lambda: run(x, tw, pc, combine, ragged, {})):7.1f}"
        for k, c in CFG.items():
            try:
                out = run(x, tw, pc, combine, ragged, c).float()
                ok = ((out - ref).abs().max() / ref.abs().max()).item() < 2e-2
                line += f" | {k} {time_us(lambda: run(x, tw, pc, combine, ragged, c)):7.1f}{'' if ok else ' BAD'}"
            except Exception as e:
                line += f" | {k} {type(e).__name__}"
        print(line)
