#!/usr/bin/env bash
# Usage: ple_prefetch_live.sh — apply the PLE prefetch patch, enable VLLM_PLE_PREFETCH=1 on the current build,
# restart, run exp.sh and lp_check.py as arm "ple-prefetch". Leaves the engine running with it.
set -u
C=vllm_qwen38_next; D=~/.cache/vllm/nsys-flashnext; AB=~/ncu-side/ab
docker exec $C pkill -INT -f "bin/vllm serve" 2>/dev/null
for i in $(seq 1 120); do sleep 5
  [ -z "$(docker exec $C pgrep -f 'bin/vllm serve|EngineCore' | head -1)" ] && \
  [ "$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)" -gt 100 ] && break
done
docker cp ~/ncu-side/ple_prefetch_patch.py $C:/tmp/ple_prefetch_patch.py
docker exec $C python3 /tmp/ple_prefetch_patch.py || exit 5
grep -q VLLM_PLE_PREFETCH $D/extra.env || echo VLLM_PLE_PREFETCH=1 >> $D/extra.env
cat $D/extra.env
bash ~/ncu-side/start.sh patched || { docker logs --tail 60 $C 2>&1 | grep -iE "error|Traceback|prefetch" | tail -20; exit 2; }
bash $AB/exp.sh ple-prefetch
python3 $AB/lp_check.py ple-prefetch
docker logs --since 10m $C 2>&1 | grep -iE "Traceback|prefetch|Error" | grep -v "non-default" | tail -10
