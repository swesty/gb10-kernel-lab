#!/usr/bin/env bash
# recipe_e2e.sh — end-to-end test of recipe/ exactly as a user runs it: pause the vllm_qwen38_next engine,
# launch the gb10fast recipe via spark-vllm-docker (container vllm_node, port 8113), wait for health (aborting if
# host memory drops below 5 GiB), run bench.sh, remove vllm_node, wait for memory to be released, then restart
# the normal optimised engine with guarded_start.sh. Full log: ~/ncu-side/ab/recipe-e2e.log
set -u
exec > ~/ncu-side/ab/recipe-e2e.log 2>&1
SVD=~/spark-vllm-docker; T=~/ncu-side/recipe-test/recipe; PORT=8113
mem() { awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo; }
echo "=== $(date -u +%T) pause vllm_qwen38_next engine"
docker exec vllm_qwen38_next pkill -INT -f "bin/vllm serve" 2>/dev/null
for i in $(seq 1 120); do sleep 5
  [ -z "$(docker exec vllm_qwen38_next pgrep -f 'bin/vllm serve|EngineCore' | head -1)" ] && [ "$(mem)" -gt 110 ] && break
done
echo "MemAvailable $(mem) GiB"
cp $T/recipes/qwen3.8-flash-next-nvfp4-solo-gb10fast.yaml $SVD/recipes/
rm -rf $SVD/mods/flashnext-gb10-fast && cp -r $T/mods/flashnext-gb10-fast $SVD/mods/
echo "=== $(date -u +%T) run-recipe"
cd $SVD && ./run-recipe.sh qwen3.8-flash-next-nvfp4-solo-gb10fast --solo -d --port $PORT 2>&1 | tail -8
T0=$(date +%s); ok=0
for i in $(seq 1 480); do
  curl -sf localhost:$PORT/health >/dev/null && { echo "HEALTHY after $(( $(date +%s) - T0 )) s"; ok=1; break; }
  docker ps --format '{{.Names}}' | grep -q '^vllm_node$' || { echo "vllm_node exited after $(( $(date +%s) - T0 )) s"; break; }
  [ "$(mem)" -lt 5 ] && { echo "ABORT: MemAvailable $(mem) GiB"; break; }
  sleep 5
done
echo "--- vllm_node log highlights"
docker logs vllm_node 2>&1 | grep -E "flashnext-gb10-fast|KV cache memory|kv_cache_memory_bytes|GPU KV cache size|^b12x autotuning|Traceback|Error" | sed -E 's/\r//' | awk '!seen[substr($0,1,60)]++' | head -12
echo "autotune lines: $(docker logs vllm_node 2>&1 | grep -c '^b12x autotuning')"
if [ $ok = 1 ]; then
  ( minm=999; for i in $(seq 1 400); do m=$(mem); [ $m -lt $minm ] && minm=$m; echo $minm > /tmp/e2e-minmem; sleep 2; done ) & MW=$!
  CONTAINER=vllm_node bash $T/bench/bench.sh $PORT
  echo "--- concurrent load: 4-12K in / 256 out, 16 requests, c=8"
  docker exec -e HF_HUB_OFFLINE=1 vllm_node vllm bench serve --backend vllm --base-url http://127.0.0.1:$PORT --model local-inference-lab/Qwen3.8-Flash-Next-NVFP4 --dataset-name random --ignore-eos --random-input-len 8192 --random-range-ratio 0.5 --random-output-len 256 --num-prompts 16 --max-concurrency 8 2>&1 | grep -E "Failed|Mean TTFT|Mean TPOT"
  kill $MW 2>/dev/null; echo "lowest MemAvailable during load: $(cat /tmp/e2e-minmem) GiB"
fi
echo "=== $(date -u +%T) teardown"
docker stop vllm_node >/dev/null 2>&1
rm -f $SVD/recipes/qwen3.8-flash-next-nvfp4-solo-gb10fast.yaml; rm -rf $SVD/mods/flashnext-gb10-fast
for i in $(seq 1 120); do [ "$(mem)" -gt 110 ] && break; sleep 5; done
echo "MemAvailable $(mem) GiB; restarting optimised engine"
bash ~/ncu-side/guarded_start.sh
echo "=== $(date -u +%T) done"
