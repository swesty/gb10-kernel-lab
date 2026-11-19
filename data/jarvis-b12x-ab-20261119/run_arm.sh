#!/bin/bash
# run_arm.sh ARM IMAGE [extra vllm flags...] — start one arm on :8001 with production args, bench it, stop it.
set -uo pipefail
ARM=$1; IMAGE=$2; shift 2
PORT=8001; OUT=~/ab-b12x/results/$ARM; mkdir -p "$OUT"
NAME=ab-$ARM
log() { echo "[$(date -u +%H:%M:%S)] $ARM: $*" | tee -a ~/ab-b12x/results/driver.log; }

docker rm -f "$NAME" >/dev/null 2>&1
log "start image=$IMAGE extra=$*"
docker run -d --name "$NAME" ${GATE:+-e VLLM_SERVER_DEV_MODE=1} --gpus all --network host --shm-size 4g -v /home/jarvis/data/models:/models \
  "$IMAGE" --port $PORT --model /models/qwen38-dflash2-20260824 --gpu-memory-utilization 0.65 \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder --max-model-len 262144 \
  --max-num-batched-tokens 8192 --max-num-seqs 16 --prefix-cache-retention-interval 6592 \
  --reasoning-parser qwen3 \
  --speculative-config '{"method":"dflash","model":"/models/qwen38-dflash2-20260824/draft","num_speculative_tokens":7}' \
  --async-scheduling --enable-prefix-caching "$@" >/dev/null

t0=$(date +%s)
until curl -sf http://127.0.0.1:$PORT/v1/models >/dev/null; do
  if [ "$(docker inspect -f '{{.State.Running}}' "$NAME" 2>/dev/null)" != "true" ]; then
    log "FAILED to start"; docker logs "$NAME" > "$OUT/startup.log" 2>&1; docker rm -f "$NAME" >/dev/null 2>&1
    grep -E "Error|error|raise|Traceback" "$OUT/startup.log" | tail -8 | tee -a ~/ab-b12x/results/driver.log; exit 1
  fi
  [ $(( $(date +%s) - t0 )) -gt 1200 ] && { log "TIMEOUT"; docker logs "$NAME" > "$OUT/startup.log" 2>&1; docker rm -f "$NAME"; exit 1; }
  sleep 10
done
log "ready after $(( $(date +%s) - t0 )) s"
sleep 5; nvidia-smi --query-gpu=memory.used --format=csv,noheader > "$OUT/coldstart_mem.txt"
docker logs "$NAME" > "$OUT/startup.log" 2>&1
grep -E "Selected|Using .*(Kernel|backend|GDN)|b12x|B12x|B12X|KV cache|Maximum concurrency" "$OUT/startup.log" \
  | grep -v -i deprecat | cut -c1-220 > "$OUT/kernels.txt"

bench() {
  docker exec -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 "$NAME" vllm bench serve --backend vllm \
    --base-url http://127.0.0.1:$PORT --model /models/qwen38-dflash2-20260824 --dataset-name random \
    --ignore-eos "$@" 2>&1 | grep -E "Failed requests|Mean TTFT|Median TTFT|Mean TPOT|Output token throughput"
}
bench --random-input-len 1024 --random-output-len 16 --num-prompts 2 --max-concurrency 1 >/dev/null
{ echo "--- prefill 16K x4 c=1"; bench --random-input-len 16384 --random-output-len 8 --num-prompts 4 --max-concurrency 1
  echo "--- prefill 4K x8 c=1"; bench --random-input-len 4096 --random-output-len 8 --num-prompts 8 --max-concurrency 1
  echo "--- prefill 16K x8 c=4"; bench --random-input-len 16384 --random-output-len 8 --num-prompts 8 --max-concurrency 4
} > "$OUT/prefill.txt"
python3 ~/ab-b12x/ab_bench.py "$ARM" $PORT "$OUT" > "$OUT/bench.txt" 2>&1 || log "ab_bench failed: $(tail -3 $OUT/bench.txt)"
if [ -n "${GATE:-}" ]; then
  log "gate checks"
  python3 ~/ab-b12x/gate_check.py "$ARM" $PORT "$OUT" > "$OUT/gate.txt" 2>&1 || log "gate_check failed: $(tail -3 $OUT/gate.txt)"
  ( while true; do nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits >> "$OUT/mem_samples.csv"; sleep 1; done ) & SP=$!
  { echo "--- c=16 8K in / 512 out x48"; bench --random-input-len 8192 --random-output-len 512 --num-prompts 48 --max-concurrency 16
    echo "--- c=16 2K in / 1024 out x32"; bench --random-input-len 2048 --random-output-len 1024 --num-prompts 32 --max-concurrency 16
  } > "$OUT/concurrency.txt"
  kill $SP; sort -t, -k1 -n "$OUT/mem_samples.csv" | tail -1 > "$OUT/peak_mem.txt"
  docker logs "$NAME" 2>&1 | grep -i -E "Xid|illegal|CUDA error|out of memory|Traceback" | head -5 > "$OUT/errors.txt"
fi
docker rm -f "$NAME" >/dev/null 2>&1
for i in $(seq 1 30); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits); [ "$u" -lt 2000 ] && break; sleep 2; done
log "done; gpu mem now ${u} MiB"
