#!/usr/bin/env bash
# Usage: final_ab.sh  — same-session A/B of the original build vs the final build on spark-5 :8112.
# Runs exp.sh (random 16K TTFT x4, real-text prefill x3, decode c=4 and c=1 in tokens/s) and lp_check.py for
# each arm, original first, then leaves the final build running.
set -u
D=~/.cache/vllm/nsys-flashnext; AB=~/ncu-side/ab; S=~/ncu-side/start.sh
stop_engine() {
  docker exec vllm_qwen38_next pkill -INT -f "bin/vllm serve" 2>/dev/null
  for i in $(seq 1 120); do
    sleep 5
    p=$(docker exec vllm_qwen38_next pgrep -f "bin/vllm serve|EngineCore" | head -1)
    avail=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)
    [ -z "$p" ] && [ "$avail" -gt 100 ] && { echo "stopped $(date -u +%T) UTC, MemAvailable ${avail} GiB"; return 0; }
  done
  echo "engine did not stop"; exit 4
}
cp $D/extra.env $D/extra.env.final
echo "=== arm 1: original build (no switches, plain attention) $(date -u +%T) UTC"
: > $D/extra.env
bash $S plain || exit 2
bash $AB/exp.sh final-session-original
python3 $AB/lp_check.py final-session-original
stop_engine
echo "=== arm 2: final build (patched attention + 4 switches) $(date -u +%T) UTC"
cp $D/extra.env.final $D/extra.env
bash $S patched || exit 3
bash $AB/exp.sh final-session-all
python3 $AB/lp_check.py final-session-all
echo "=== done $(date -u +%T) UTC; final build left running"
