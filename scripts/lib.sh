# Shared helpers for MXFP4 study runs: serve <log> <model> [extra args]; stop; bench <out> <in> <out_len> <prompts> <conc>
# Env: VENV (python env with torch/triton/triton-kernels), SGLANG_SRC (sglang checkout), EXTRA_PYTHONPATH (e.g. an aiter stub dir).
# Outputs go to <repo>/runs/<phase>/.
D=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
source ${VENV:-$HOME/sgl-env}/bin/activate
export PYTHONPATH=${SGLANG_SRC:?set SGLANG_SRC to the sglang checkout}/python${EXTRA_PYTHONPATH:+:$EXTRA_PYTHONPATH}
export SGLANG_DISABLE_LEAN_ATTENTION=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
serve() {
  local LOG=$1 MODEL=$2; shift 2
  (cd ~ && exec python -m sglang.launch_server --model-path $MODEL --attention-backend triton --context-length 4096 \
    --mem-fraction-static ${MEMF:-0.85} --port 30000 --host 127.0.0.1 --skip-server-warmup "$@") > $LOG 2>&1 &
  SP=$!
  for _ in $(seq 1 2400); do curl -sf 127.0.0.1:30000/health >/dev/null 2>&1 && return 0; kill -0 $SP 2>/dev/null || return 1; sleep 1; done
  return 1
}
stop() { kill $SP 2>/dev/null; wait $SP 2>/dev/null; pkill -f "[s]glang\.launch_server|[s]glang::" 2>/dev/null; sleep 5; }
bench() {
  (cd ~ && python -m sglang.bench_serving --backend sglang --host 127.0.0.1 --port 30000 --dataset-name random \
    --random-input-len $2 --random-output-len $3 --random-range-ratio 1 --num-prompts $4 --max-concurrency $5 --seed 1) > $1 2>&1
}
faults() { grep -c "illegal memory\|HIP error\|Memory access fault" $1; }
