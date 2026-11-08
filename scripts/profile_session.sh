#!/bin/bash
# Profile session: profile_session.sh <label> <model> <conc> -> runs/phase2/prof_<label>_c<conc>/ (30 decode steps, graphs off)
source "$(dirname "$0")/lib.sh"
L=$1 M=$2 C=$3; O=$D/runs/phase2; mkdir -p $O; P=$O/prof_${L}_c$C; LOG=$O/srv_prof_${L}_c$C.log
serve $LOG $M --disable-cuda-graph ${EXTRA_ARGS:-} || { echo "$L prof: serve failed" | tee -a $O/summary.txt; stop; exit 1; }
bench $O/warm_prof_${L}_c$C.txt 1024 16 $(( C < 2 ? 2 : C )) $C
( bench $O/bench_prof_${L}_c$C.txt 1024 1024 $C $C ) & BP=$!
sleep 20
curl -s 127.0.0.1:30000/start_profile -H 'Content-Type: application/json' \
  -d "{\"output_dir\":\"$P\",\"num_steps\":30,\"profile_by_stage\":true,\"profile_stages\":[\"decode\"],\"activities\":[\"CPU\",\"GPU\"]}" > $O/startprof_${L}_c$C.txt
wait $BP
F=$(faults $LOG); stop
echo "$L prof c$C faults=$F traces=$(ls $P 2>/dev/null | wc -l)" | tee -a $O/summary.txt
