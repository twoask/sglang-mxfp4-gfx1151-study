# Qwen3.5-35B-A3B MXFP4 decode on gfx1151 (Strix Halo)

Why the Quark MXFP4 W4A16 MoE path from [sglang#41389](https://github.com/sgl-project/sglang/pull/41389) is slower than bf16 at batch 1 on gfx1151, and a tile setting that closes the gap in testing. This repo holds the measurements only; no fix is proposed here as code.

## Summary

- At batch 1, the MXFP4 checkpoint decodes slower than bf16: 74.2 vs 59.7 ms/step on current main (#41389 reported 118 vs 103 ms).
- The whole difference is in the MoE matmul (`triton_kernels` `matmul`). At batch 1 its gate/up call launches **32 workgroups of 2 warps** and reads its ~9 MB of expert weights at ~11 GB/s.
- Cause: the AMD branch of `triton_kernels`' opt-flag heuristic, in the triton-kernels build sglang pins (`1.0.0+amd.rocm7.2.0.git89002410`). It picks `block_n = 256`, `num_warps = 2 if m <= 16 else 8` (m = tokens x top-k, so 8 at batch 1), and split-k never triggers here. sglang sets no opt-flag constraints on AMD.
- Upstream Triton already tuned this for RDNA in [triton-lang/triton#10810](https://github.com/triton-lang/triton/pull/10810) (July 2026), after the pinned build.
- Forcing `{"block_n": 64, "num_warps": 4}` through `opt_flags.update_opt_flags_constraints` (the mechanism sglang already uses on NVIDIA) gives:

| batch | now (ms/step) | block_n 64, 4 warps | speedup | bf16 |
|---|---|---|---|---|
| 1 | 74.2 | **37.9** | 1.96x | 59.7 |
| 4 | 71.1 | 57.4 | 1.24x | 113.3 |
| 16 | 137.1 | 100.3 | 1.37x | 234.4 |
| 32 | 195.1 | 139.4 | 1.40x | n/a (capped at 16) |

With 16 concurrent requests, output throughput goes from 96 to 124 tok/s (bf16: 61). GSM8K is unchanged (95.22% vs 94.84%), and the outputs match a dequantized fp32 reference.

## Setup

| | |
|---|---|
| GPU | Radeon 8060S (gfx1151), 94.5 GB GPU-visible, `multi_processor_count` = 20 |
| OS | Ubuntu 24.04.5, kernel 6.17.0-1033-oem |
| torch | 2.10.0+rocm7.13.0a20260513 (HIP 7.13) |
| Triton | 3.6.0 |
| triton-kernels | 1.0.0+amd.rocm7.2.0.git89002410 (the sglang ROCm docker pin), plus two shims for Triton 3.6, below |
| sglang | main @ `3bcde0c80e`, `--attention-backend triton --context-length 4096 --mem-fraction-static 0.85 --skip-server-warmup` |
| Models | `amd/Qwen3.5-35B-A3B-MXFP4` (2e19c65), `Qwen/Qwen3.5-35B-A3B` (59d61f3) |

Notes on this environment:
- triton-kernels git89002410 targets Triton 3.7. On Triton 3.6 it needed two local shims: a fallback `is_hip_gfx1250` in `target_info.py`, and a `hasattr` guard around `def_file_line_number` in `specialize.py` (debug metadata only). Neither touches the matmul or opt flags.
- At this sglang commit the quark schemes import `aiter` unconditionally on HIP (#46914 and #47585 address that), so an import stub stood in for it. No aiter code runs on this path.
- sglang's VL warmup fails here (no FFmpeg for torchcodec), hence `--skip-server-warmup`; the scripts send a text warmup.

## Method

Timing: one server per session, no other traffic, `sglang.bench_serving` random dataset, 1024 in / 256 out, concurrency 1/4/16/32. ms/step comes from the server's `Decode batch` log lines with no queued requests (`scripts/steps_from_log.py`); the client's median ITL agrees within ~1%. CUDA graphs on. Arms alternate (base r1, tile r1, base r2, tile r2).

Accuracy: GSM8K, all 1319 questions, temperature 0, max 1024 tokens, thinking off, with a step-by-step prompt (`scripts/gsm8k_cot.yaml`). sgl-eval's default math prompt asks for "the answer (and only answer)", and with thinking off this model then answers with a bare `\boxed{N}`, which scored 48.8% for reasons unrelated to quantization.

Profiling: torch profiler on the decode stage, 30 steps, CUDA graphs off, kernels grouped by regex (`scripts/rules_qwen35.txt`, 0.0% unmatched).

## Results

### 1. The inversion reproduces (`results/phase1/`)

| batch | MXFP4 r1 / r2 | bf16 r1 / r2 |
|---|---|---|
| 1 | 74.27 / 74.13 | 59.70 / 59.61 |
| 4 | 70.72 / 71.07 | 113.31 / 113.25 |
| 16 | 137.04 / 136.68 | 234.47 / 234.36 |
| 32 | 195.13 / 194.61 | n/a |

bf16 is capped at 16 running requests by the mamba state cache (12.7 GB left after 67 GB of weights), so its c=32 run ran at 16.

Two things stand out. MXFP4 reads 1.48 GB less per token than bf16 (`scripts/floor.py`), so it should be ~6 ms faster at batch 1, not 14.5 ms slower. And MXFP4 batch 4 is faster than batch 1, though it reads up to 4x the expert bytes, so batch 1 is not bandwidth-bound.

GSM8K: MXFP4 94.84%, bf16 94.62%.

### 2. Where the time goes (`results/phase2/breakdown_*.txt`)

GPU kernel time per decode step:

| group | MXFP4 b1 | MXFP4 b4 | bf16 b1 | MXFP4 b16 | bf16 b16 |
|---|---|---|---|---|---|
| MoE matmul | 41.7 | 39.5 | 33.7 | 92.0 | 197.6 |
| dense GEMMs (attn/GDN proj, shared expert, lm_head) | 21.7 | 20.4 | 19.8 | 24.8 | 25.0 |
| everything else | 3.8 | 5.9 | 2.1 | 12.5 | 11.3 |
| total | 67.1 | 65.8 | 55.7 | 129.3 | 233.8 |

Everything except the MoE matmul is the same in both formats. Per MoE matmul call:

| | grid x threads | median time |
|---|---|---|
| MXFP4 gate/up, batch 1 | 32 x 64 | 820 us |
| MXFP4 gate/up, batch 4 | 128 x 256 | 776 us |
| MXFP4 down, batch 1 | 64 x 64 | 241 us |
| bf16 fused_moe, batch 1 | 512 x 128 | 381 us |

Batch 4 moves ~4x the bytes in less time than batch 1 because it crosses `m > 16` and gets 8 warps. At batch 16 the MXFP4 MoE matmul reaches an estimated ~74 GB/s against bf16 fused_moe's ~130 GB/s (bytes from ~102 distinct experts per layer, assuming uniform routing), so it is under-occupied there too.

### 3. Tile sweep (`results/phase2/ogs_tiles.txt`, `scripts/bench_ogs_tiles.py`)

The real swizzled weights and `triton_kernels.matmul` at this model's shapes (256 experts, top-8; gate/up N=1024 K=2048, down N=2048 K=512), sweeping `block_n`, `num_warps` and `split_k` through `scoped_opt_flags_constraints`. Every config's output is compared with the default config's (all match). Separately, the default, `block_n 32` and `block_n 128` were checked against a hand-dequantized fp32 reference at batch 1 and 4: max relative error 0.24%, identical across configs (`scripts/check_triton_kernels.py` does the same check on its own). `block_n 64, 4 warps` was also checked directly against the fp32 reference at both shapes and 8 / 32 / 128 rows: all pass, with the same error as the default config (`results/phase2/check_bn64w4_reference.txt`); at batch 1 its output is bitwise identical to the default's.

| shape | default | block_n 64, 4 warps |
|---|---|---|
| gate/up b1 | 684 us | 177 us (3.86x) |
| gate/up b4 | 471 | 451 (1.04x) |
| gate/up b16 | 1800 | 1038 (1.73x) |
| down b1 | 248 | 149 (1.67x) |
| down b4 | 299 | 238 (1.26x) |
| down b16 | 595 | 449 (1.33x) |

`block_n 32, 2 warps` is slightly faster at batch 1 (147 us) but not at larger batches. `block_n 64, 4 warps` is near the top at every batch.

#10810's RDNA small-M config (`block_m 16, block_n 64, block_k 512, num_warps 4`) is 2x slower here on gate/up at batch >= 4 (`results/phase2/ogs_10810.txt`). It pairs with `num_stages = 1`, which the constraints API can't set. With the default `block_k`, `block_m 16 + block_n 64 + 4 warps` is the best or tied in the sweep; served, it is within 1-2% of `block_n 64, 4 warps` (inside its own run-to-run spread; `results/phase3/`).

### 4. Served A/B (`results/phase3/`)

| batch | base r1 / r2 | block_n 64, 4 warps r1 / r2 |
|---|---|---|
| 1 | 74.24 / 74.18 | 37.91 / 37.80 |
| 4 | 71.04 / 71.07 | 57.51 / 57.27 |
| 16 | 137.04 / 137.07 | 100.41 / 100.24 |
| 32 | 194.97 / 195.29 | 139.57 / 139.28 |

Output tok/s at 16 concurrent: 96.6 / 96.1 -> 123.3 / 124.3. GSM8K with the change: 95.22% (base 94.84%). No HIP errors or memory faults in any log.

## Possible fixes

`patches/0001-study-env-driven-opt-flag-constraints.patch` is only the test switch used for the A/B: it reads constraints from the `MXFP4_STUDY_OGS` environment variable in `_swizzle_mxfp4`, so one checkout serves both arms. It is not a fix and not meant for upstream. A fix in sglang could be either:

1. a small RDNA-gated `opt_flags.update_opt_flags_constraints({"block_n": 64, "num_warps": 4})` in `_swizzle_mxfp4`, next to the existing NVIDIA branches, or
2. moving sglang's ROCm triton-kernels pin to a build that includes #10810 (then re-checking, since #10810's values were slower here with the constraints API).

## Reproduce

```bash
export SGLANG_SRC=/path/to/sglang          # at 3bcde0c80e, with patches/0001-*.patch applied
export VENV=/path/to/venv EXTRA_PYTHONPATH=/path/to/aiter-stub   # if needed
M=amd/Qwen3.5-35B-A3B-MXFP4

# served A/B (writes runs/phase3/summary.txt)
for r in 1 2; do
  PHASE=phase3 bash scripts/serve_ab.sh mxfp4 $M base $r
  PHASE=phase3 MXFP4_STUDY_OGS='{"block_n":64,"num_warps":4}' bash scripts/serve_ab.sh mxfp4 $M tile64w4 $r
done

# kernel sweep and reference check (no server)
PYTHONPATH=$SGLANG_SRC/python python scripts/check_triton_kernels.py
PYTHONPATH=$SGLANG_SRC/python python scripts/bench_ogs_tiles.py

# decode profile + grouping
bash scripts/profile_session.sh mxfp4 $M 1
python scripts/group_kernels.py runs/phase2/prof_mxfp4_c1/*.trace.json.gz scripts/rules_qwen35.txt 30
```

The raw profiler traces (85-125 MB each at batch 16) are too big for the repo, but I'm happy to share them if they'd help.
