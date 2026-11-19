#!/usr/bin/env bash
# guarded_start.sh — start the optimised Flash Next engine (launch.py patched + extra.env); abort if b12x starts
# autotuning (cache miss) or MemAvailable falls below 5 GiB. Prints KV cache size and the result.
set -u
C=vllm_qwen38_next
SINCE=$(date -u +%Y-%m-%dT%H:%M:%S)
bash ~/ncu-side/start.sh patched > /tmp/guarded_start.log 2>&1 &
SP=$!
while kill -0 $SP 2>/dev/null; do
  sleep 10
  avail=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)
  if docker logs --since $SINCE $C 2>&1 | grep -q "^b12x autotuning"; then
    echo "ABORT: b12x autotuning started (cache miss)"; docker logs --since $SINCE $C 2>&1 | grep "^b12x autotuning" | tail -1 | cut -c1-140
    docker exec $C pkill -INT -f "bin/vllm serve"; wait $SP; exit 3
  fi
  if [ "$avail" -lt 5 ]; then echo "ABORT: MemAvailable ${avail} GiB"; docker exec $C pkill -INT -f "bin/vllm serve"; wait $SP; exit 4; fi
done
cat /tmp/guarded_start.log | tail -2
docker logs --since $SINCE $C 2>&1 | grep -E "Available KV cache memory|GPU KV cache size" | tail -2 | cut -c1-160
curl -sf localhost:8112/health >/dev/null && echo "8112 healthy"
