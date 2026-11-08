#!/bin/bash
# Timing session: serve_ab.sh <label> <model> <arm-tag> <rep> [conc list] -> runs/<phase>/srv_*.log, bench_*_c*.txt
# No other traffic in this session; ms/step comes from the server log (steps_from_log.py).
source "$(dirname "$0")/lib.sh"
L=$1 M=$2 A=$3 RP=$4; CONCS=${5:-"1 4 16 32"}; O=$D/runs/${PHASE:-phase1}; mkdir -p $O
LOG=$O/srv_${L}_${A}_r$RP.log
serve $LOG $M ${EXTRA_ARGS:-} || { echo "$L $A r$RP: serve failed (see $LOG)" | tee -a $O/summary.txt; stop; exit 1; }
bench $O/warm_${L}_${A}_r$RP.txt 1024 16 2 1
for c in $CONCS; do bench $O/bench_${L}_${A}_r${RP}_c$c.txt 1024 256 $(( c < 2 ? 2 : c )) $c; done
F=$(faults $LOG); stop
echo "$L $A r$RP faults=$F $(python $D/scripts/steps_from_log.py $LOG | tr '\n' ' ')" | tee -a $O/summary.txt
