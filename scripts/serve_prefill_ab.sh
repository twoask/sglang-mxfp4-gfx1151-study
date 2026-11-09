#!/bin/bash
# Decode + prefill session: serve_prefill_ab.sh <label> <model> <arm-tag> <rep>
# Decode: c=1/4/16/32, 1024 in / 256 out (ms/step from the server log). Prefill: radix cache off so every prompt
# is computed; 8 x 2k and 4 x 8k one at a time (TTFT), then 16 x 2k at once (large-M chunks).
source "$(dirname "$0")/lib.sh"
L=$1 M=$2 A=$3 RP=$4; O=$D/runs/${PHASE:-phase4}; mkdir -p $O
LOG=$O/srv_${L}_${A}_r$RP.log
serve $LOG $M --context-length 10240 --disable-radix-cache ${EXTRA_ARGS:-} || { echo "$L $A r$RP: serve failed (see $LOG)" | tee -a $O/summary.txt; stop; exit 1; }
bench $O/warm_${L}_${A}_r$RP.txt 1024 16 2 1
for c in 1 4 16 32; do bench $O/bench_${L}_${A}_r${RP}_c$c.txt 1024 256 $(( c < 2 ? 2 : c )) $c; done
bench $O/prefill_${L}_${A}_r${RP}_2k_c1.txt 2048 1 8 1
bench $O/prefill_${L}_${A}_r${RP}_8k_c1.txt 8192 1 4 1
bench $O/prefill_${L}_${A}_r${RP}_2k_c16.txt 2048 1 16 16
F=$(faults $LOG); stop
P=$(for f in 2k_c1 8k_c1 2k_c16; do printf '%s ttft_med=%s ' $f "$(grep -m1 'Median TTFT' $O/prefill_${L}_${A}_r${RP}_$f.txt | awk '{print $NF}')"; done)
echo "$L $A r$RP faults=$F $(python $D/scripts/steps_from_log.py $LOG | tr '\n' ' ') $P" | tee -a $O/summary.txt
