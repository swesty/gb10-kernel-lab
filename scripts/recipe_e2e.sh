#!/usr/bin/env bash
# recipe_e2e.sh — end-to-end test of recipe/ as a user would run it: pause the vllm_qwen38_next engine, launch the
# gb10fast recipe through spark-vllm-docker (container vllm_node, port 8113), wait for health, run bench.sh,
# then remove vllm_node and restart the normal engine.
set -u
SVD=~/spark-vllm-docker; T=~/ncu-side/recipe-test/recipe; PORT=8113
docker exec vllm_qwen38_next pkill -INT -f "bin/vllm serve" 2>/dev/null
for i in $(seq 1 120); do sleep 5
  [ -z "$(docker exec vllm_qwen38_next pgrep -f 'bin/vllm serve|EngineCore' | head -1)" ] && \
  [ "$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)" -gt 100 ] && break
done
cp $T/recipes/qwen3.8-flash-next-nvfp4-solo-gb10fast.yaml $SVD/recipes/
rm -rf $SVD/mods/flashnext-gb10-fast && cp -r $T/mods/flashnext-gb10-fast $SVD/mods/
cd $SVD && ./run-recipe.sh qwen3.8-flash-next-nvfp4-solo-gb10fast --solo -d --port $PORT 2>&1 | tail -5
T0=$(date +%s)
for i in $(seq 1 120); do
  curl -sf localhost:$PORT/health >/dev/null && { echo "HEALTHY after $(( $(date +%s) - T0 )) s"; break; }
  docker ps --format '{{.Names}}' | grep -q '^vllm_node$' || { echo "vllm_node exited"; break; }
  sleep 5
done
docker logs vllm_node 2>&1 | grep -E "flashnext-gb10-fast|Traceback|Error" | head -8
curl -sf localhost:$PORT/health >/dev/null && CONTAINER=vllm_node bash $T/bench/bench.sh $PORT
docker stop vllm_node >/dev/null 2>&1; sleep 10
rm -f $SVD/recipes/qwen3.8-flash-next-nvfp4-solo-gb10fast.yaml; rm -rf $SVD/mods/flashnext-gb10-fast
for i in $(seq 1 60); do [ "$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)" -gt 100 ] && break; sleep 5; done
bash ~/ncu-side/start.sh patched
