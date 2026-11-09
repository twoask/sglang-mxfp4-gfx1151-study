#!/bin/bash
# Phase 4: PR branch vs its main base. Timing (alternating, 2 reps), then sgl-eval GSM8K per arm.
S=$(dirname "$0"); M=amd/Qwen3.5-35B-A3B-MXFP4
export PHASE=phase4 EXTRA_PYTHONPATH=${EXTRA_PYTHONPATH:-$HOME/dummy_tv}
BASE=${BASE_SRC:?} PR=${PR_SRC:?}
for r in 1 2; do
  SGLANG_SRC=$BASE bash $S/serve_prefill_ab.sh mxfp4 $M base $r
  SGLANG_SRC=$PR bash $S/serve_prefill_ab.sh mxfp4 $M pr $r
done
SGLANG_SRC=$BASE bash $S/eval_session.sh base $M
SGLANG_SRC=$PR bash $S/eval_session.sh pr $M
