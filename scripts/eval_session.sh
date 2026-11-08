#!/bin/bash
# Accuracy session (separate from timing): eval_session.sh <label> <model> -> smoke answer + sgl-eval GSM8K (1319)
source "$(dirname "$0")/lib.sh"
L=$1 M=$2; O=$D/runs/${PHASE:-phase1}; mkdir -p $O; LOG=$O/srv_eval_$L.log
serve $LOG $M ${EXTRA_ARGS:-} || { echo "$L eval: serve failed (see $LOG)" | tee -a $O/summary.txt; stop; exit 1; }
grep -m1 "W4A16" $LOG > $O/path_$L.txt
curl -s 127.0.0.1:30000/v1/chat/completions -H 'Content-Type: application/json' -d '{"model":"x","messages":[{"role":"user","content":"What is 17*23? Answer with the number only."}],"max_tokens":64,"temperature":0,"chat_template_kwargs":{"enable_thinking":false}}' > $O/smoke_$L.json
(cd ~ && ${SGL_EVAL:-sgl-eval} run gsm8k --base-url http://127.0.0.1:30000/v1 --temperature 0 --max-tokens 1024 \
  --num-threads 64 --chat-template-kwarg enable_thinking=false --prompt $D/scripts/gsm8k_cot.yaml --out-dir $O/gsm8k_$L) > $O/gsm8k_$L.out 2>&1
F=$(faults $LOG); stop
echo "$L eval faults=$F smoke=$(python -c "import json;print(repr(json.load(open('$O/smoke_$L.json'))['choices'][0]['message']['content'][:40]))" 2>/dev/null) $(grep -h 'score' $O/gsm8k_$L.out | tail -1)" | tee -a $O/summary.txt
