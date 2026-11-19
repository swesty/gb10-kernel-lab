#!/usr/bin/env bash
# Usage: capture_current.sh <name> — Nsight Systems capture of one random 16K prefill on the current build
# (patched attention + all extra.env switches), exported to sqlite; then restores the normal launch.
set -u
C=vllm_qwen38_next; D=~/.cache/vllm/nsys-flashnext; N=$1
stop_engine() {
  docker exec $C pkill -INT -f "bin/vllm serve" 2>/dev/null
  for i in $(seq 1 360); do sleep 5
    [ -z "$(docker exec $C pgrep -f 'bin/vllm serve|EngineCore|nsys profile' | head -1)" ] && \
    [ "$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)" -gt 100 ] && return 0
  done; echo "engine did not stop"; exit 4
}
stop_engine
cp $D/extra.env $D/extra.env.keep
printf "B12X_SELECTED_PIPELINE=1\nB12X_SELECTED_SPLITQK=1\nB12X_SELECTED_SWZ333=1\n" >> $D/extra.env
bash ~/ncu-side/start.sh nsys || { cp $D/extra.env.keep $D/extra.env; exit 2; }
bash ~/ncu-side/prefill-window.sh
echo "window done $(date -u +%T); stopping engine so nsys writes the report"
stop_engine
cp $D/extra.env.keep $D/extra.env
docker exec $C bash -c "cd /root/.cache/vllm/nsys-flashnext && ls -la flashnext*.nsys-rep && mv flashnext.nsys-rep $N.nsys-rep && /opt/nsys/target-linux-sbsa-armv8/nsys export --type sqlite --force-overwrite true -o $N.sqlite $N.nsys-rep >/dev/null && ls -la $N.*"
bash ~/ncu-side/start.sh patched
