#!/usr/bin/env bash
# Usage: arm_live.sh <label> "<ENV=VAL ...>" — add env lines to extra.env (kept), restart the current build,
# run exp.sh and lp_check.py as <label>. Leaves the engine running with the change.
set -u
C=vllm_qwen38_next; D=~/.cache/vllm/nsys-flashnext; AB=~/ncu-side/ab; L=$1; shift
docker exec $C pkill -INT -f "bin/vllm serve" 2>/dev/null
for i in $(seq 1 120); do sleep 5
  [ -z "$(docker exec $C pgrep -f 'bin/vllm serve|EngineCore' | head -1)" ] && \
  [ "$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)" -gt 100 ] && break
done
for kv in $@; do k=${kv%%=*}; grep -v "^$k=" $D/extra.env > $D/extra.env.tmp; mv $D/extra.env.tmp $D/extra.env; echo "$kv" >> $D/extra.env; done
cat $D/extra.env
bash ~/ncu-side/start.sh patched || { docker logs --tail 60 $C 2>&1 | grep -iE "error|Traceback" | tail -20; exit 2; }
bash $AB/exp.sh $L
python3 $AB/lp_check.py $L
