#!/usr/bin/env bash
# Usage: ple_disk_sweep.sh "<label>:<ENV=VAL ENV=VAL>" ...  — for each arm: add the env lines to extra.env,
# restart the current build, run exp.sh <label>; restores extra.env afterwards and leaves the engine running.
set -u
C=vllm_qwen38_next; D=~/.cache/vllm/nsys-flashnext; AB=~/ncu-side/ab
stop_engine() {
  docker exec $C pkill -INT -f "bin/vllm serve" 2>/dev/null
  for i in $(seq 1 120); do sleep 5
    [ -z "$(docker exec $C pgrep -f 'bin/vllm serve|EngineCore' | head -1)" ] && \
    [ "$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)" -gt 100 ] && return 0
  done; echo "engine did not stop"; exit 4
}
cp $D/extra.env $D/extra.env.base
docker cp ~/ncu-side/ple_qd_patch.py $C:/tmp/ple_qd_patch.py && docker exec $C python3 /tmp/ple_qd_patch.py || exit 5
for arm in "$@"; do
  label=${arm%%:*}; envs=${arm#*:}
  stop_engine
  cp $D/extra.env.base $D/extra.env
  for kv in $envs; do echo "$kv" >> $D/extra.env; done
  echo "=== arm $label: $envs"
  bash ~/ncu-side/start.sh patched || { echo "arm $label failed to start"; docker logs --tail 30 $C 2>&1 | grep -i "error\|disk\|gds\|cufile" | tail -8; continue; }
  bash $AB/exp.sh $label | grep -E "Mean TTFT|real-text|decode c=1"
done
stop_engine
cp $D/extra.env.base $D/extra.env
bash ~/ncu-side/start.sh patched
