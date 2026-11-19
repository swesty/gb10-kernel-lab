#!/usr/bin/env bash
# Usage: start.sh <launch mode>  — start Flash Next on :8112 via launch.py and wait for /health
vpid() { docker exec vllm_qwen38_next pgrep -f "bin/vllm serve" | head -1; }
[ -n "$(vpid)" ] && { echo "engine already running (pid $(vpid))"; exit 1; }
docker exec -d vllm_qwen38_next sh -c "python3 /root/.cache/vllm/nsys-flashnext/launch.py $1 > /proc/1/fd/1 2>&1"
T=$(date +%s); echo "launched $1 $(date -u +%T) UTC"
for i in $(seq 1 720); do
  curl -sf localhost:8112/health >/dev/null && { echo "HEALTHY $(date -u +%T) UTC ($(( $(date +%s) - T )) s)"; exit 0; }
  if [ $(( $(date +%s) - T )) -gt 90 ] && [ -z "$(vpid)" ]; then echo "DIED $(date -u +%T) UTC"; docker logs --tail 40 vllm_qwen38_next 2>&1; exit 2; fi
  sleep 5
done
echo TIMEOUT; exit 3
